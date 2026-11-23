#!/usr/bin/env python3

"""Filter repetition segments using the original PEERING and ASdb E&R rules."""

from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


"""Repository-relative input, output, and classification directories."""
REPO_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_DIR / "data" / "prefix.aspth.parquet" / "poisoning"
INPUT_DIR = BASE_DIR / "segments"
OUTPUT_DIR = BASE_DIR / "segments.filtered"
ASDB_DIR = REPO_DIR / "data" / "asdb"

PEERING_ASN = 47065
EDUCATION_CATEGORY = "Education and Research"
ASDB_SNAPSHOTS = {
    "2021-05", "2021-11", "2022-05",
    "2022-12", "2023-05", "2024-01",
}
ASN_RE = re.compile(r"\d+")
FILE_RE = re.compile(r"segments\.(\d{4}-\d{2}-\d{2})\.parquet")


def parse_asn(value):
    if pd.isna(value):
        return None
    match = ASN_RE.search(str(value))
    return int(match.group()) if match else None


def parse_segment_asns(value):
    if pd.isna(value):
        return []
    return [int(asn) for asn in ASN_RE.findall(str(value))]


def load_education_asns(year_month):
    """Use every 'Category N - Layer 1' column, as in the original notebook."""
    path = ASDB_DIR / f"{year_month}_categorized_ases.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ASdb snapshot: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "ASN" not in df.columns:
        raise ValueError(f"Missing ASN column: {path}")

    category_columns = [
        col for col in df.columns
        if re.fullmatch(r"Category \d+ - Layer 1", col)
    ]
    if not category_columns:
        raise ValueError(f"Missing Category N - Layer 1 columns: {path}")

    asns = df["ASN"].apply(parse_asn)
    mask = (
        df[category_columns]
        .fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip().eq(EDUCATION_CATEGORY))
        .any(axis=1)
    )
    education_asns = set(asns[mask & asns.notna()].astype(int))
    if not education_asns:
        raise ValueError(f"No Education and Research ASNs found: {path}")
    return education_asns


def classify_exclusion(repeated_raw, segment_raw, education_asns):
    """Exclude repeated AS47065 in all months; exclude all-E&R RSs only in ASdb months."""
    is_peering = parse_asn(repeated_raw) == PEERING_ASN
    is_all_er = False

    if education_asns is not None:
        segment_asns = parse_segment_asns(segment_raw)
        if len(segment_asns) >= 3:
            participating_asns = set(segment_asns)
            is_all_er = bool(participating_asns) and participating_asns.issubset(
                education_asns
            )

    if is_peering and is_all_er:
        return "PEERING+ALL_E&R"
    if is_peering:
        return "PEERING"
    if is_all_er:
        return "ALL_E&R"
    return None


def main():
    """Write filtered daily Parquet files without modifying the original segment files."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Missing input directory: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("segments.*.parquet"))
    if not files:
        raise FileNotFoundError(f"No segment Parquet files in: {INPUT_DIR}")

    dated_files = []
    for path in files:
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected filename: {path.name}")
        dated_files.append((path, match.group(1)[:7]))

    needed_months = {month for _, month in dated_files} & ASDB_SNAPSHOTS
    asdb_lookup = {
        month: load_education_asns(month) for month in sorted(needed_months)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, month in tqdm(dated_files, desc="Filtering segments", unit="file"):
        output_path = OUTPUT_DIR / path.name
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite: {output_path}")

        table = pq.read_table(path)
        for required in ("repeated_asn", "poisoned_segment"):
            if required not in table.column_names:
                raise ValueError(f"Missing {required} column in {path}")

        education_asns = asdb_lookup.get(month)
        reasons = [
            classify_exclusion(repeated, segment, education_asns)
            for repeated, segment in zip(
                table.column("repeated_asn").to_pylist(),
                table.column("poisoned_segment").to_pylist(),
            )
        ]
        keep = pa.array([reason is None for reason in reasons], type=pa.bool_())
        filtered = table.filter(keep)

        if table.num_rows != filtered.num_rows + sum(reason is not None for reason in reasons):
            raise RuntimeError(f"Row conservation failed: {path.name}")
        if not filtered.schema.equals(table.schema, check_metadata=True):
            raise RuntimeError(f"Schema changed: {path.name}")

        pq.write_table(filtered, output_path, compression="zstd")
        print(f"{path.name}: {table.num_rows:,} input, {filtered.num_rows:,} retained")


if __name__ == "__main__":
    main()
