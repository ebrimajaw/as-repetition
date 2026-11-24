#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/data/prefix.aspth/prefix.aspth.parquet"
)

ANNOTATED_FILE = (
    BASE_DIR
    / "rs.analysis"
    / "asRel"
    / "data"
    / "rs_pair_records.caida_latest.parquet"
)

ASDB_DIR = (
    BASE_DIR
    / "poisoning"
    / "segment_pairs"
    / "asdb"
)

OUT_DIR = (
    BASE_DIR
    / "poisoning"
    / "resubmission_analysis"
    / "04_classification"
    / "data"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_FILE = (
    OUT_DIR
    / "rs_pair_records.caida_latest.excluded_peering_er.parquet"
)

AUDIT_FILE = (
    OUT_DIR
    / "rs_pair_records.caida_latest.excluded_peering_er.audit.csv"
)


# ============================================================
# CONFIG
# ============================================================

PEERING_ASN = 47065

ASDB_MONTHS = [
    "2021-05",
    "2021-11",
    "2022-05",
    "2022-12",
    "2023-05",
    "2024-01",
]


# ============================================================
# HELPERS
# ============================================================

def parse_segment_asns(segment):
    """
    Parse all unique numeric ASNs participating in an RS.

    Example:
        "47065 72 47065" -> {47065, 72}
    """
    if pd.isna(segment):
        return set()

    out = set()

    for token in str(segment).split():
        try:
            out.add(int(token))
        except (TypeError, ValueError):
            pass

    return out


def load_education_asns(csv_file):
    """
    Reproduce the established ASdb Education & Research rule:

    An ASN is E&R if ANY 'Category N - Layer 1' column equals
    'Education and Research'.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    if "ASN" not in df.columns:
        raise ValueError(
            f"Missing ASN column in {csv_file}"
        )

    category_columns = [
        col
        for col in df.columns
        if col.startswith("Category ")
        and col.endswith(" - Layer 1")
    ]

    if not category_columns:
        raise ValueError(
            f"No 'Category N - Layer 1' columns found in {csv_file}"
        )

    education_mask = pd.Series(False, index=df.index)

    for col in category_columns:
        education_mask |= (
            df[col]
            .astype(str)
            .str.strip()
            .eq("Education and Research")
        )

    education_asns = set(
        pd.to_numeric(
            df.loc[education_mask, "ASN"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    return education_asns


# ============================================================
# LOAD ASDB E&R SETS
# ============================================================

education_by_month = {}

print("=" * 100)
print("LOADING ASDB EDUCATION & RESEARCH ASNs")
print("=" * 100)

for month in ASDB_MONTHS:

    asdb_file = (
        ASDB_DIR
        / f"{month}_categorized_ases.csv"
    )

    if not asdb_file.exists():
        raise FileNotFoundError(asdb_file)

    education_asns = load_education_asns(asdb_file)

    education_by_month[month] = education_asns

    print(
        f"{month}: "
        f"{len(education_asns):,} E&R ASNs"
    )


# ============================================================
# OPEN INPUT PARQUET
# ============================================================

if not ANNOTATED_FILE.exists():
    raise FileNotFoundError(ANNOTATED_FILE)

pf = pq.ParquetFile(ANNOTATED_FILE)

original_schema = pf.schema_arrow

print("\n" + "=" * 100)
print("INPUT")
print("=" * 100)

print(f"File       : {ANNOTATED_FILE}")
print(f"Rows       : {pf.metadata.num_rows:,}")
print(f"Row groups : {pf.metadata.num_row_groups:,}")
print(f"Columns    : {pf.metadata.num_columns:,}")


# ============================================================
# FILTER ROW GROUP BY ROW GROUP
# ============================================================

writer = None

audit_rows = []

total_input = 0
total_retained = 0

total_peering_only = 0
total_er_only = 0
total_peering_and_er = 0

try:

    for rg in range(pf.metadata.num_row_groups):

        table = pf.read_row_group(rg)
        df = table.to_pandas()

        input_rows = len(df)
        total_input += input_rows

        # ----------------------------------------------------
        # DATE / MONTH
        # ----------------------------------------------------

        date = pd.to_datetime(
            df["date"],
            errors="raise",
        )

        month = date.dt.strftime("%Y-%m")

        # ----------------------------------------------------
        # PEERING
        #
        # Established rule:
        # introducing/repeated ASN == AS47065
        # ----------------------------------------------------

        is_peering = (
            df["introducing_asn"]
            .astype("int64")
            .eq(PEERING_ASN)
        )

        # ----------------------------------------------------
        # ALL-E&R
        #
        # Only evaluated for the six ASdb-aligned months.
        #
        # Every unique ASN participating in the complete RS
        # must be classified as Education and Research in the
        # corresponding month's ASdb snapshot.
        # ----------------------------------------------------

        is_all_er = pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

        for aligned_month, education_asns in education_by_month.items():

            month_mask = month.eq(aligned_month)

            if not month_mask.any():
                continue

            segments = df.loc[
                month_mask,
                "segment",
            ]

            er_result = segments.map(
                lambda segment: (
                    bool(parse_segment_asns(segment))
                    and
                    parse_segment_asns(segment).issubset(
                        education_asns
                    )
                )
            )

            is_all_er.loc[month_mask] = (
                er_result.astype(bool)
            )

        # ----------------------------------------------------
        # MUTUALLY EXCLUSIVE AUDIT CATEGORIES
        # ----------------------------------------------------

        peering_only = is_peering & ~is_all_er
        er_only = ~is_peering & is_all_er
        peering_and_er = is_peering & is_all_er

        # ----------------------------------------------------
        # FINAL EXCLUSION
        # ----------------------------------------------------

        exclude = is_peering | is_all_er

        retained = df.loc[~exclude].copy()

        # ----------------------------------------------------
        # COUNTS
        # ----------------------------------------------------

        n_peering_only = int(peering_only.sum())
        n_er_only = int(er_only.sum())
        n_peering_and_er = int(peering_and_er.sum())

        removed_rows = int(exclude.sum())
        retained_rows = len(retained)

        total_peering_only += n_peering_only
        total_er_only += n_er_only
        total_peering_and_er += n_peering_and_er

        total_retained += retained_rows

        # ----------------------------------------------------
        # RECONCILIATION
        # ----------------------------------------------------

        if (
            n_peering_only
            + n_er_only
            + n_peering_and_er
            != removed_rows
        ):
            raise RuntimeError(
                f"Removal-category reconciliation failed "
                f"for row group {rg}"
            )

        if retained_rows + removed_rows != input_rows:
            raise RuntimeError(
                f"Input/output reconciliation failed "
                f"for row group {rg}"
            )

        # ----------------------------------------------------
        # WRITE FILTERED DATA
        #
        # Preserve exactly the original 17 columns.
        # ----------------------------------------------------

        retained_table = pa.Table.from_pandas(
            retained,
            schema=original_schema,
            preserve_index=False,
        )

        if writer is None:
            writer = pq.ParquetWriter(
                FILTERED_FILE,
                original_schema,
                compression="snappy",
            )

        writer.write_table(retained_table)

        # ----------------------------------------------------
        # AUDIT
        # ----------------------------------------------------

        audit_rows.append(
            {
                "row_group": rg,
                "input_rows": input_rows,
                "peering_only_removed": n_peering_only,
                "other_all_er_removed": n_er_only,
                "peering_and_all_er_removed": n_peering_and_er,
                "total_removed": removed_rows,
                "retained_rows": retained_rows,
            }
        )

        print(
            f"Row group {rg + 1:3d}/{pf.metadata.num_row_groups}: "
            f"input={input_rows:,} | "
            f"removed={removed_rows:,} | "
            f"retained={retained_rows:,}"
        )

finally:

    if writer is not None:
        writer.close()


# ============================================================
# WRITE AUDIT
# ============================================================

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    AUDIT_FILE,
    index=False,
)


# ============================================================
# FINAL ACCOUNTING
# ============================================================

total_removed = (
    total_peering_only
    + total_er_only
    + total_peering_and_er
)

inclusive_peering = (
    total_peering_only
    + total_peering_and_er
)

inclusive_all_er = (
    total_er_only
    + total_peering_and_er
)

removed_pct = (
    total_removed / total_input * 100.0
    if total_input
    else 0.0
)


# ============================================================
# VALIDATION
# ============================================================

assert total_input == pf.metadata.num_rows

assert (
    total_retained
    + total_removed
    == total_input
)

assert (
    total_peering_only
    + total_er_only
    + total_peering_and_er
    == total_removed
)


# ============================================================
# VALIDATE OUTPUT SCHEMA / ROW COUNT
# ============================================================

out_pf = pq.ParquetFile(FILTERED_FILE)

if out_pf.schema_arrow != original_schema:
    raise RuntimeError(
        "Filtered output schema does not match input schema."
    )

if out_pf.metadata.num_rows != total_retained:
    raise RuntimeError(
        "Filtered output row count does not match "
        "the computed retained count."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("CAIDA RELATIONSHIP RECORD EXCLUSION SUMMARY")
print("=" * 100)

print(f"Input rows              : {total_input:,}")
print(f"Retained rows           : {total_retained:,}")
print(f"Removed rows            : {total_removed:,}")
print(f"Removed share           : {removed_pct:.6f}%")

print("\nMutually exclusive removed categories:")
print(f"  PEERING only          : {total_peering_only:,}")
print(f"  Other all-E&R only    : {total_er_only:,}")
print(f"  PEERING + all-E&R     : {total_peering_and_er:,}")

print("\nInclusive counts:")
print(f"  All PEERING           : {inclusive_peering:,}")
print(f"  All all-E&R           : {inclusive_all_er:,}")

print("\nValidation:")
print(
    f"  Input = retained + removed : "
    f"{total_input:,} = "
    f"{total_retained:,} + "
    f"{total_removed:,}"
)
print("  Accounting             : PASS")
print("  Schema preservation    : PASS")
print("  Output row count       : PASS")

print("\nOutputs:")
print(f"  Filtered parquet       : {FILTERED_FILE}")
print(f"  Audit CSV              : {AUDIT_FILE}")

print("=" * 100)
