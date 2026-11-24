#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/data/prefix.aspth/prefix.aspth.parquet"
)

POISONING_DIR = BASE_DIR / "poisoning"


# Original pattern table used by the original computation.
PATTERN_FILE = (
    POISONING_DIR
    / "repetition_patterns_with_listed_poisoned_asns.parquet"
)


# Original combined statistics.
#
# We reuse ONLY the raw_unique_ases denominator from this file.
ORIGINAL_STATS_FILE = (
    POISONING_DIR
    / "recomputed.daily_rs_participation_ratios.from_patterns.parquet"
)


# ASdb files already used and validated in the successful
# PEERING/E&R filtering.
ASDB_DIR = (
    POISONING_DIR
    / "segment_pairs"
    / "asdb"
)


# New outputs.
#
# Do NOT overwrite the original statistics.
OUTPUT_DIR = (
    POISONING_DIR
    / "resubmission_analysis"
    / "03_temporal_analysis"
    / "as_participation"
    / "data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUT_ROLE_STATS = (
    OUTPUT_DIR
    / "excluded_peering_er.daily_rs_participation_ases.from_patterns.parquet"
)

OUT_COMBINED_STATS = (
    OUTPUT_DIR
    / "excluded_peering_er.daily_rs_participation_ratios.from_patterns.parquet"
)

OUT_AUDIT = (
    OUTPUT_DIR
    / "excluded_peering_er.daily_rs_participation_audit.csv"
)


# ============================================================
# FILTER CONFIG
# ============================================================

PEERING_ASN = 47065

EDUCATION_CATEGORY = "Education and Research"

EXPECTED_SNAPSHOTS = 396


# Exact ASdb snapshots and expected E&R ASN counts from the
# already validated filtering analysis.
ASDB_MONTHS = {
    "2021-05": 5842,
    "2021-11": 7677,
    "2022-05": 6631,
    "2022-12": 6717,
    "2023-05": 6784,
    "2024-01": 6812,
}


# ============================================================
# PROCESSING CONFIG
# ============================================================

START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2025-01-01")

BATCH_SIZE = 1_000_000

ASN_RE = re.compile(r"\d+")

POISONING_SOURCE_RE = re.compile(
    r"poisoning\.(\d{4}-\d{2}-\d{2})\.parquet"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def parse_asn(value: Any) -> int | None:
    """
    Parse one ASN value.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    match = ASN_RE.search(
        str(value)
    )

    if match is None:
        return None

    return int(
        match.group()
    )


def parse_middle_segment(
    value: Any,
) -> set[int]:
    """
    Preserve the original AS-role computation:

    every ASN represented in middle_segment is an appearing AS.
    """

    if value is None:
        return set()

    return {
        int(x)
        for x in ASN_RE.findall(
            str(value)
        )
    }


def extract_pattern_date(
    source_file: Any,
) -> pd.Timestamp | None:
    """
    Extract YYYY-MM-DD from the original poisoning source
    filename stored in the pattern table.
    """

    match = POISONING_SOURCE_RE.search(
        str(source_file)
    )

    if match is None:
        return None

    return pd.to_datetime(
        match.group(1)
    )


def safe_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    result = (
        numerator.astype("float64")
        / denominator.astype("float64")
    )

    result = result.where(
        denominator > 0
    )

    return result * 100.0


# ============================================================
# ASDB
# ============================================================

def load_education_asns(
    month: str,
) -> set[int]:
    """
    Load Education and Research ASNs using exactly the same
    definition as the successful announcement/segment filters.

    An ASN is E&R if ANY:

        Category N - Layer 1

    equals:

        Education and Research
    """

    path = (
        ASDB_DIR
        / f"{month}_categorized_ases.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ASdb file:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "ASN" not in df.columns:
        raise RuntimeError(
            f"Missing ASN column in:\n{path}"
        )

    category_columns = [
        column
        for column in df.columns
        if re.fullmatch(
            r"Category \d+ - Layer 1",
            str(column),
        )
    ]

    if not category_columns:
        raise RuntimeError(
            f"No Layer-1 category columns in:\n{path}"
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

    expected = ASDB_MONTHS[
        month
    ]

    if len(education_asns) != expected:
        raise RuntimeError(
            f"{month}: loaded "
            f"{len(education_asns):,} E&R ASNs; "
            f"expected {expected:,}."
        )

    return education_asns


def load_all_asdb() -> dict[str, set[int]]:
    """
    Load and validate all six ASdb snapshots.
    """

    lookup = {}

    print()
    print("=" * 72)
    print("LOAD ASDB")
    print("-" * 72)

    for month in sorted(
        ASDB_MONTHS
    ):

        education_asns = (
            load_education_asns(
                month
            )
        )

        lookup[
            month
        ] = education_asns

        print(
            f"{month}: "
            f"{len(education_asns):,} "
            "E&R ASNs PASS"
        )

    print("=" * 72)

    return lookup


# ============================================================
# RAW DENOMINATOR
# ============================================================

def load_raw_denominator() -> pd.DataFrame:
    """
    Reuse the raw_unique_ases denominator from the original
    computation.

    This ensures that the denominator is IDENTICAL between the
    original and revised analyses.
    """

    if not ORIGINAL_STATS_FILE.exists():
        raise FileNotFoundError(
            "Original statistics file not found:\n"
            f"{ORIGINAL_STATS_FILE}"
        )

    df = pd.read_parquet(
        ORIGINAL_STATS_FILE
    )

    required = {
        "date",
        "raw_unique_ases",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Original statistics file is missing "
            f"required columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df[
        (df["date"] >= START_DATE)
        & (df["date"] < END_DATE)
    ].copy()

    df = (
        df[
            [
                "date",
                "raw_unique_ases",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates in original raw-AS denominator."
        )

    if len(df) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            "Unexpected number of denominator snapshots.\n"
            f"Expected: {EXPECTED_SNAPSHOTS:,}\n"
            f"Found   : {len(df):,}"
        )

    return df


# ============================================================
# FILTERED RS-ROLE NUMERATORS
# ============================================================

def build_role_stats(
    education_lookup: dict[str, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the original pattern-based AS-role computation,
    but exclude:

      1. PEERING RS patterns:
            poisoning_asn == 47065

         across the complete 2014--2024 period.

      2. Other all-E&R RS patterns:

         only during the six ASdb-aligned months, when every
         participating ASN belongs to the corresponding E&R set.

    Unit of filtering:
        one RS pattern row.

    Introducing AS:
        poisoning_asn

    Appearing ASes:
        ASNs in middle_segment
    """

    if not PATTERN_FILE.exists():
        raise FileNotFoundError(
            f"Missing pattern file:\n{PATTERN_FILE}"
        )

    pf = pq.ParquetFile(
        PATTERN_FILE
    )

    required = {
        "source_file",
        "poisoning_asn",
        "middle_segment",
    }

    missing = (
        required
        - set(pf.schema_arrow.names)
    )

    if missing:
        raise RuntimeError(
            "Pattern file is missing required columns:\n"
            f"{sorted(missing)}"
        )


    # ========================================================
    # DAILY UNIQUE ROLE SETS
    # ========================================================

    daily_intro: dict[
        pd.Timestamp,
        set[int],
    ] = {}

    daily_appear: dict[
        pd.Timestamp,
        set[int],
    ] = {}


    # ========================================================
    # PER-SNAPSHOT AUDIT
    # ========================================================

    audit: dict[
        pd.Timestamp,
        dict,
    ] = {}


    # ========================================================
    # GLOBAL COUNTERS
    # ========================================================

    rows_seen = 0

    rows_in_window = 0

    rows_retained = 0

    rows_peering_removed = 0

    rows_er_removed = 0

    rows_bad_date = 0

    rows_bad_intro = 0


    # ========================================================
    # PROCESS ORIGINAL PATTERN FILE
    # ========================================================

    for batch in tqdm(
        pf.iter_batches(
            columns=[
                "source_file",
                "poisoning_asn",
                "middle_segment",
            ],
            batch_size=BATCH_SIZE,
        ),
        total=pf.metadata.num_row_groups,
        desc="Computing revised daily RS-role ASNs",
        unit="batch",
    ):

        batch_df = (
            batch.to_pandas()
        )

        for row in batch_df.itertuples(
            index=False
        ):

            rows_seen += 1


            # =================================================
            # DATE
            # =================================================

            date = extract_pattern_date(
                row.source_file
            )

            if date is None:
                rows_bad_date += 1
                continue

            if not (
                START_DATE
                <= date
                < END_DATE
            ):
                continue

            rows_in_window += 1


            # =================================================
            # INTRODUCING ASN
            #
            # Preserve original definition exactly.
            # =================================================

            intro_asn = parse_asn(
                row.poisoning_asn
            )

            if intro_asn is None:
                rows_bad_intro += 1
                continue


            # =================================================
            # APPEARING ASNs
            #
            # Preserve original definition exactly.
            # =================================================

            appearing_ases = (
                parse_middle_segment(
                    row.middle_segment
                )
            )


            # =================================================
            # AUDIT INITIALIZATION
            # =================================================

            if date not in audit:

                audit[
                    date
                ] = {
                    "date": date,
                    "pattern_rows_seen": 0,
                    "peering_removed": 0,
                    "all_er_removed": 0,
                    "pattern_rows_retained": 0,
                }

            audit[
                date
            ][
                "pattern_rows_seen"
            ] += 1


            # =================================================
            # EXCLUSION 1:
            # PEERING
            #
            # The introducing/repeated boundary ASN is AS47065.
            #
            # Applied across the complete 2014--2024 period.
            # =================================================

            if intro_asn == PEERING_ASN:

                audit[
                    date
                ][
                    "peering_removed"
                ] += 1

                rows_peering_removed += 1

                continue


            # =================================================
            # EXCLUSION 2:
            # OTHER ALL-E&R
            #
            # Only evaluated when an ASdb snapshot exists for
            # the corresponding month.
            #
            # Participating ASNs =
            #
            #     introducing ASN
            #       UNION
            #     appearing ASNs
            #
            # Every participating ASN must be E&R.
            #
            # PEERING has already been excluded above, so this
            # category is explicitly "other all-E&R".
            # =================================================

            month = (
                date.strftime(
                    "%Y-%m"
                )
            )

            education_asns = (
                education_lookup.get(
                    month
                )
            )

            if education_asns is not None:

                participating_ases = (
                    {intro_asn}
                    | appearing_ases
                )

                is_all_er = (
                    bool(
                        participating_ases
                    )
                    and
                    participating_ases.issubset(
                        education_asns
                    )
                )

                if is_all_er:

                    audit[
                        date
                    ][
                        "all_er_removed"
                    ] += 1

                    rows_er_removed += 1

                    continue


            # =================================================
            # RETAINED PATTERN
            #
            # Only retained patterns enter the daily role sets.
            # =================================================

            daily_intro.setdefault(
                date,
                set(),
            ).add(
                intro_asn
            )

            daily_appear.setdefault(
                date,
                set(),
            ).update(
                appearing_ases
            )

            audit[
                date
            ][
                "pattern_rows_retained"
            ] += 1

            rows_retained += 1


    # ========================================================
    # BUILD DAILY ROLE COUNTS
    # ========================================================

    dates = sorted(
        set(daily_intro)
        | set(daily_appear)
    )

    role_rows = []

    for date in dates:

        intro = daily_intro.get(
            date,
            set(),
        )

        appear = daily_appear.get(
            date,
            set(),
        )

        combined = (
            intro
            | appear
        )

        overlap = (
            intro
            & appear
        )

        role_rows.append({
            "date":
                date,

            "introducing_ases":
                len(intro),

            "appearing_ases":
                len(appear),

            "combined_ases":
                len(combined),

            "overlap_ases":
                len(overlap),
        })


    role_df = (
        pd.DataFrame(
            role_rows
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # BUILD AUDIT
    # ========================================================

    audit_df = (
        pd.DataFrame(
            list(
                audit.values()
            )
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    audit_df[
        "total_removed"
    ] = (
        audit_df[
            "peering_removed"
        ]
        +
        audit_df[
            "all_er_removed"
        ]
    )

    audit_df[
        "reconciled"
    ] = (
        audit_df[
            "pattern_rows_seen"
        ]
        ==
        (
            audit_df[
                "pattern_rows_retained"
            ]
            +
            audit_df[
                "total_removed"
            ]
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if len(audit_df) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            "Unexpected number of pattern snapshots.\n"
            f"Expected: {EXPECTED_SNAPSHOTS:,}\n"
            f"Found   : {len(audit_df):,}"
        )

    if not audit_df[
        "reconciled"
    ].all():

        bad = audit_df.loc[
            ~audit_df[
                "reconciled"
            ]
        ]

        raise RuntimeError(
            "Pattern-row reconciliation failed:\n"
            f"{bad}"
        )

    if len(role_df) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            "Unexpected number of role-stat snapshots.\n"
            f"Expected: {EXPECTED_SNAPSHOTS:,}\n"
            f"Found   : {len(role_df):,}"
        )


    # ========================================================
    # GLOBAL RECONCILIATION
    # ========================================================

    audit_seen = int(
        audit_df[
            "pattern_rows_seen"
        ].sum()
    )

    audit_retained = int(
        audit_df[
            "pattern_rows_retained"
        ].sum()
    )

    audit_peering = int(
        audit_df[
            "peering_removed"
        ].sum()
    )

    audit_er = int(
        audit_df[
            "all_er_removed"
        ].sum()
    )

    if audit_seen != (
        audit_retained
        + audit_peering
        + audit_er
    ):
        raise RuntimeError(
            "Global pattern-row reconciliation failed."
        )

    if audit_seen != rows_in_window:
        raise RuntimeError(
            "Global in-window row accounting mismatch.\n"
            f"Loop count : {rows_in_window:,}\n"
            f"Audit count: {audit_seen:,}"
        )


    # ========================================================
    # SAVE
    # ========================================================

    role_df.to_parquet(
        OUT_ROLE_STATS,
        index=False,
    )

    audit_df.to_csv(
        OUT_AUDIT,
        index=False,
    )


    # ========================================================
    # PRINT AUDIT
    # ========================================================

    removed_total = (
        audit_peering
        + audit_er
    )

    removal_pct = (
        100.0
        * removed_total
        / audit_seen
        if audit_seen
        else 0.0
    )

    print()
    print("=" * 72)
    print("REVISED PATTERN-LEVEL RS-AS ROLE AUDIT")
    print("-" * 72)

    print(
        f"Rows seen in pattern file    : "
        f"{rows_seen:,}"
    )

    print(
        f"Rows in study window         : "
        f"{audit_seen:,}"
    )

    print(
        f"PEERING rows removed         : "
        f"{audit_peering:,}"
    )

    print(
        f"Other all-E&R rows removed   : "
        f"{audit_er:,}"
    )

    print(
        f"Total rows removed           : "
        f"{removed_total:,}"
    )

    print(
        f"Removed share                : "
        f"{removal_pct:.6f}%"
    )

    print(
        f"Rows retained                : "
        f"{audit_retained:,}"
    )

    print(
        f"Bad date rows                : "
        f"{rows_bad_date:,}"
    )

    print(
        f"Bad introducing-AS rows      : "
        f"{rows_bad_intro:,}"
    )

    print(
        f"Snapshots                    : "
        f"{len(audit_df):,}"
    )

    print("-" * 72)

    print(
        f"Role output                  : "
        f"{OUT_ROLE_STATS}"
    )

    print(
        f"Audit output                 : "
        f"{OUT_AUDIT}"
    )

    print("=" * 72)

    return (
        role_df,
        audit_df,
    )


# ============================================================
# COMBINE WITH ORIGINAL RAW DENOMINATOR
# ============================================================

def build_combined_stats(
    role_df: pd.DataFrame,
) -> pd.DataFrame:

    raw_df = (
        load_raw_denominator()
    )


    # ========================================================
    # DATE CONSISTENCY
    # ========================================================

    raw_dates = set(
        raw_df[
            "date"
        ]
    )

    role_dates = set(
        role_df[
            "date"
        ]
    )

    missing_dates = sorted(
        raw_dates
        - role_dates
    )

    extra_dates = sorted(
        role_dates
        - raw_dates
    )

    if missing_dates:
        raise RuntimeError(
            "Role statistics are missing dates:\n"
            f"{missing_dates}"
        )

    if extra_dates:
        raise RuntimeError(
            "Role statistics contain unexpected dates:\n"
            f"{extra_dates}"
        )


    # ========================================================
    # MERGE
    # ========================================================

    df = (
        raw_df
        .merge(
            role_df,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    if len(df) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            "Unexpected number of combined snapshots.\n"
            f"Expected: {EXPECTED_SNAPSHOTS:,}\n"
            f"Found   : {len(df):,}"
        )


    # ========================================================
    # ORIGINAL RATIO DEFINITIONS
    # ========================================================

    df[
        "introducing_ratio"
    ] = safe_rate(
        df[
            "introducing_ases"
        ],
        df[
            "raw_unique_ases"
        ],
    )

    df[
        "appearing_ratio"
    ] = safe_rate(
        df[
            "appearing_ases"
        ],
        df[
            "raw_unique_ases"
        ],
    )

    df[
       "combined_ratio"
    ] = safe_rate(
        df[
            "combined_ases"
        ],
        df[
            "raw_unique_ases"
        ],
    )


    # ========================================================
    # ORIGINAL ACCOUNTING FIELDS
    # ========================================================

    df[
        "sum_role_ases"
    ] = (
        df[
            "introducing_ases"
        ]
        +
        df[
            "appearing_ases"
        ]
    )

    df[
        "double_count_excess"
    ] = (
        df[
            "sum_role_ases"
        ]
        -
        df[
            "combined_ases"
        ]
    )


    # ========================================================
    # SANITY CHECKS
    # ========================================================

    expected_overlap = (
        df[
            "introducing_ases"
        ]
        +
        df[
            "appearing_ases"
        ]
        -
        df[
            "combined_ases"
        ]
    )

    if not (
        expected_overlap
        ==
        df[
            "overlap_ases"
        ]
    ).all():

        raise RuntimeError(
            "Role-set union/overlap reconciliation failed."
        )

    if (
        df[
            "combined_ases"
        ]
        >
        df[
            "raw_unique_ases"
        ]
    ).any():

        raise RuntimeError(
            "Combined RS-role AS count exceeds "
            "the raw-AS denominator."
        )

    if (
        df[
            "combined_ratio"
        ]
        > 100.0
    ).any():

        raise RuntimeError(
            "Combined ratio exceeds 100%."
        )


    # ========================================================
    # SAVE
    # ========================================================

    df.to_parquet(
        OUT_COMBINED_STATS,
        index=False,
    )

    return df


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print("REVISED DAILY RS-AS PARTICIPATION METRICS")
    print("-" * 72)

    print(
        f"Pattern input                : "
        f"{PATTERN_FILE}"
    )

    print(
        f"Original denominator source  : "
        f"{ORIGINAL_STATS_FILE}"
    )

    print(
        f"Role stats output            : "
        f"{OUT_ROLE_STATS}"
    )

    print(
        f"Combined stats output        : "
        f"{OUT_COMBINED_STATS}"
    )

    print(
        f"Audit output                 : "
        f"{OUT_AUDIT}"
    )

    print("-" * 72)

    print(
        f"Snapshots                    : "
        f"{len(df):,}"
    )

    print(
        f"Date range                   : "
        f"{df['date'].min().date()} to "
        f"{df['date'].max().date()}"
    )

    print("-" * 72)

    print(
        f"Max raw unique ASes          : "
        f"{df['raw_unique_ases'].max():,}"
    )

    print(
        f"Max introducing ASes         : "
        f"{df['introducing_ases'].max():,}"
    )

    print(
        f"Max appearing ASes           : "
        f"{df['appearing_ases'].max():,}"
    )

    print(
        f"Max combined ASes            : "
        f"{df['combined_ases'].max():,}"
    )

    print(
        f"Max overlap ASes             : "
        f"{df['overlap_ases'].max():,}"
    )

    print("-" * 72)

    print(
        f"Mean introducing ratio       : "
        f"{df['introducing_ratio'].mean():.4f}%"
    )

    print(
        f"Mean appearing ratio         : "
        f"{df['appearing_ratio'].mean():.4f}%"
    )

    print(
        f"Mean combined ratio          : "
        f"{df['combined_ratio'].mean():.4f}%"
    )

    print(
        f"Median combined ratio        : "
        f"{df['combined_ratio'].median():.4f}%"
    )

    print(
        f"Max combined ratio           : "
        f"{df['combined_ratio'].max():.4f}%"
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    education_lookup = (
        load_all_asdb()
    )

    role_df, audit_df = (
        build_role_stats(
            education_lookup
        )
    )

    combined_df = (
        build_combined_stats(
            role_df
        )
    )

    print_summary(
        combined_df
    )


if __name__ == "__main__":
    main()
