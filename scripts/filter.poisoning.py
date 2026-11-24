#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_DIR / "data/prefix.aspth.parquet/poisoning"
OUTPUT_DIR = INPUT_DIR / "poisoning.filtered"
ASDB_DIR = REPO_DIR / "data/asdb"
PEERING_ASN = 47065
EDUCATION_CATEGORY = "Education and Research"
EXPECTED_FILE_COUNT = 396
ASDB_MONTHS = {
    "2021-05": 5842,
    "2021-11": 7677,
    "2022-05": 6631,
    "2022-12": 6717,
    "2023-05": 6784,
    "2024-01": 6812,
}
POISON_FILE_RE = re.compile(r"^poisoning\.(\d{4}-\d{2}-\d{2})\.parquet$")
ASN_RE = re.compile(r"\d+")

def parse_asn(value):
    if pd.isna(value):
        return None

    match = ASN_RE.search(str(value))

    if match is None:
        return None

    return int(match.group())


def parse_path(value):
    if value is None:
        return []

    return [
        int(x)
        for x in ASN_RE.findall(str(value))
    ]


def parse_repeated_asns(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [
            int(x)
            for x in value
            if x is not None
        ]

    if hasattr(value, "tolist"):
        converted = value.tolist()

        if isinstance(converted, list):
            return [
                int(x)
                for x in converted
                if x is not None
            ]

    return [
        int(x)
        for x in ASN_RE.findall(str(value))
    ]

def get_input_files():
    """
    Return ONLY the original daily poisoning snapshot files.

    Accepted filename:

        poisoning.YYYY-MM-DD.parquet

    Other parquet files in the poisoning directory are ignored.
    """

    files = []

    for path in INPUT_DIR.iterdir():

        if not path.is_file():
            continue

        match = POISON_FILE_RE.fullmatch(
            path.name
        )

        if match is None:
            continue

        date = match.group(1)

        year = int(
            date[:4]
        )

        if 2014 <= year <= 2024:
            files.append(path)

    files.sort(
        key=lambda path: path.name
    )

    if len(files) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FILE_COUNT} daily poisoning "
            f"files, but found {len(files)}."
        )

    dates = [
        POISON_FILE_RE.fullmatch(
            path.name
        ).group(1)
        for path in files
    ]

    if len(dates) != len(set(dates)):
        raise RuntimeError(
            "Duplicate poisoning snapshot dates detected."
        )
    return files


def load_education_asns(month):
    """
    Load the Education and Research ASNs for one ASdb snapshot.

    Definition is exactly the one used in the successful
    segment filtering:

    ASN is E&R if ANY
    'Category N - Layer 1'
    equals
    'Education and Research'.
    """

    path = (
        ASDB_DIR
        / f"{month}_categorized_ases.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ASdb file: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "ASN" not in df.columns:
        raise RuntimeError(
            f"Missing ASN column in {path}"
        )

    category_columns = [
        column
        for column in df.columns
        if re.fullmatch(
            r"Category \d+ - Layer 1",
            column,
        )
    ]

    if not category_columns:
        raise RuntimeError(
            f"No Layer-1 category columns in {path}"
        )

    education_mask = (
        df[category_columns]
        .fillna("")
        .astype(str)
        .apply(
            lambda column:
            column.str.strip().eq(
                EDUCATION_CATEGORY
            )
        )
        .any(axis=1)
    )

    parsed_asns = (
        df["ASN"]
        .apply(parse_asn)
    )

    education_asns = set(
        parsed_asns[
            education_mask
            & parsed_asns.notna()
        ].astype(int)
    )

    expected = ASDB_MONTHS[month]

    if len(education_asns) != expected:
        raise RuntimeError(
            f"{month}: loaded "
            f"{len(education_asns):,} E&R ASNs; "
            f"expected {expected:,}."
        )

    return education_asns


def load_all_asdb():
    lookup = {}

    for month in sorted(ASDB_MONTHS):

        lookup[month] = (
            load_education_asns(
                month
            )
        )
    return lookup

def extract_rs(
    path_asns,
    repeated_asn,
):
    """
    Reconstruct one repetition section.

    The RS starts at the first occurrence of the repeated ASN
    and ends at its last non-consecutive occurrence.

    Example:

        28792 6939 37662 9498 36926 327708 36926

    becomes:

        36926 327708 36926
    """

    positions = [
        index
        for index, asn
        in enumerate(path_asns)
        if asn == repeated_asn
    ]

    if len(positions) < 2:
        return None

    first = positions[0]
    last = positions[-1]

    # Consecutive repetition is not a non-consecutive RS.
    if last - first < 2:
        return None

    rs = tuple(
        path_asns[
            first:last + 1
        ]
    )
    # Confirm that something different occurs inside the RS.
    if not any(
        asn != repeated_asn
        for asn in rs[1:-1]
    ):
        return None
    return rs

def classify_announcement(
    cleaned_aspath,
    repeated_asns_raw,
    education_asns,
):
    """
    Classify one complete poisoning announcement.

    The complete announcement is excluded when ANY RS is:

      PEERING
          repeated ASN == 47065

      ALL_E&R
          every unique ASN participating in the RS belongs
          to the contemporaneous E&R ASN set

    E&R classification is disabled when education_asns is None.
    """

    path_asns = parse_path(
        cleaned_aspath
    )

    repeated_asns = sorted(
        set(
            parse_repeated_asns(
                repeated_asns_raw
            )
        )
    )

    peering_matches = []
    er_matches = []

    for repeated_asn in repeated_asns:

        rs = extract_rs(
            path_asns,
            repeated_asn,
        )

        if rs is None:
            raise RuntimeError(
                "Could not reconstruct RS for repeated ASN "
                f"{repeated_asn} from path: "
                f"{cleaned_aspath}"
            )

        if repeated_asn == PEERING_ASN:
            peering_matches.append(
                rs
            )

        if education_asns is not None:

            participating_asns = set(
                rs
            )

            if (
                participating_asns
                and participating_asns.issubset(
                    education_asns
                )
            ):
                er_matches.append(
                    rs
                )

    has_peering = bool(
        peering_matches
    )

    has_er = bool(
        er_matches
    )

    if has_peering and has_er:

        reason = (
            "PEERING+ALL_E&R"
        )

    elif has_peering:

        reason = (
            "PEERING"
        )

    elif has_er:

        reason = (
            "ALL_E&R"
        )

    else:

        reason = None

    matched = []

    for rs in peering_matches:

        matched.append(
            "PEERING:"
            + " ".join(
                map(str, rs)
            )
        )

    for rs in er_matches:

        matched.append(
            "ALL_E&R:"
            + " ".join(
                map(str, rs)
            )
        )

    matched_rs = (
        " | ".join(matched)
    )

    return reason, matched_rs


def main():
    files = get_input_files()
    education_lookup = load_all_asdb()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT_DIR}")
    for path in files:
        missing = {"cleaned_aspath", "repeated_asns"} - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_input = total_kept = 0
    for path in tqdm(files, desc="Filtering announcements", unit="file"):
        date = POISON_FILE_RE.fullmatch(path.name).group(1)
        source_schema = pq.read_schema(path)
        df = pd.read_parquet(path)
        education_asns = education_lookup.get(date[:7])
        keep = [
            classify_announcement(row.cleaned_aspath, row.repeated_asns, education_asns)[0] is None
            for row in df.itertuples(index=False)
        ]
        filtered = pa.Table.from_pandas(df.loc[keep], schema=source_schema, preserve_index=False, safe=True)
        if not filtered.schema.equals(source_schema, check_metadata=False):
            raise RuntimeError(f"Schema mismatch: {path.name}")
        pq.write_table(filtered, OUTPUT_DIR / path.name, compression="zstd")
        total_input += len(df)
        total_kept += filtered.num_rows
    print(f"Files: {len(files):,}")
    print(f"Input announcements: {total_input:,}")
    print(f"Filtered announcements: {total_kept:,}")
    print(f"Removed announcements: {total_input - total_kept:,}")


if __name__ == "__main__":
    main()
