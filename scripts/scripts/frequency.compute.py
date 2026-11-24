#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/data/prefix.aspth/prefix.aspth.parquet/poisoning"
)

# ------------------------------------------------------------
# INPUT
# Newly filtered RS dataset.
# ------------------------------------------------------------

SEGMENTS_DIR = BASE_DIR / "segments.filtered"


# ------------------------------------------------------------
# RESUBMISSION ANALYSIS ROOT
# ------------------------------------------------------------

RESUBMISSION_DIR = BASE_DIR / "resubmission_analysis"


# ------------------------------------------------------------
# Organize the analyses we expect to perform for the
# resubmission.
# ------------------------------------------------------------

ANALYSIS_DIRS = [
    "01_asn_participation",
    "02_collector_visibility",
    "03_temporal_analysis",
    "04_classification",
    "05_case_studies",
]

for analysis_name in ANALYSIS_DIRS:

    analysis_dir = (
        RESUBMISSION_DIR
        / analysis_name
    )

    for subdir in [
        "data",
        "figures",
        "tables",
    ]:
        (
            analysis_dir
            / subdir
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ------------------------------------------------------------
# Current analysis
# ------------------------------------------------------------

ASN_ANALYSIS_DIR = (
    RESUBMISSION_DIR
    / "01_asn_participation"
)

DATA_DIR = (
    ASN_ANALYSIS_DIR
    / "data"
)

TABLE_DIR = (
    ASN_ANALYSIS_DIR
    / "tables"
)

FIGURE_DIR = (
    ASN_ANALYSIS_DIR
    / "figures"
)


# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUT_APPEARING = (
    DATA_DIR
    / "appearing_asn_ranking.csv"
)

OUT_INTRODUCING = (
    DATA_DIR
    / "introducing_asn_ranking.csv"
)

OUT_TOP10 = (
    TABLE_DIR
    / "top10_asn_participation.csv"
)

OUT_LATEX = (
    TABLE_DIR
    / "top10_asn_participation.tex"
)

OUT_SUMMARY = (
    DATA_DIR
    / "asn_participation_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

ASN_RE = re.compile(r"\d+")


def parse_asn(value):
    """
    Parse one ASN from a value.
    """

    if pd.isna(value):
        return None

    match = ASN_RE.search(
        str(value)
    )

    if match is None:
        return None

    return int(
        match.group()
    )


def parse_segment_asns(value):
    """
    Extract all ASNs from a repetition segment.

    Example:

        47065 6939 47065

    becomes:

        [47065, 6939, 47065]
    """

    if pd.isna(value):
        return []

    return [
        int(x)
        for x in ASN_RE.findall(
            str(value)
        )
    ]


def format_count(value):
    """
    Compact count formatting used in the paper table.

    908     -> 908
    1300    -> 1.3K
    55900   -> 55.9K
    """

    value = int(value)

    if value >= 1_000_000:
        return (
            f"{value / 1_000_000:.1f}M"
        )

    if value >= 1_000:
        return (
            f"{value / 1_000:.1f}K"
        )

    return str(value)


def format_denominator(value):
    """
    Compact formatting for denominators in the table note.
    """

    value = int(value)

    if value >= 1_000_000:
        return (
            f"{value / 1_000_000:.2f}M"
        )

    if value >= 1_000:
        return (
            f"{value / 1_000:.2f}K"
        )

    return str(value)


# ============================================================
# VALIDATE INPUT
# ============================================================

if not SEGMENTS_DIR.exists():
    raise RuntimeError(
        "Filtered segment directory does not exist:\n"
        f"{SEGMENTS_DIR}"
    )


files = sorted(
    SEGMENTS_DIR.glob(
        "segments.*.parquet"
    )
)


if not files:
    raise RuntimeError(
        "No filtered segment files found in:\n"
        f"{SEGMENTS_DIR}"
    )


print(
    "============================================================"
)
print(
    "ASN PARTICIPATION ANALYSIS — FILTERED DATA"
)
print(
    "============================================================"
)

print(
    f"Input directory : {SEGMENTS_DIR}"
)

print(
    f"Output directory: {ASN_ANALYSIS_DIR}"
)

print(
    f"Snapshots       : {len(files):,}"
)

print()


if len(files) != 396:
    print(
        "[WARNING] Expected 396 snapshots, "
        f"but found {len(files):,}."
    )
else:
    print(
        "[PASS] Found all 396 filtered snapshots."
    )


# ============================================================
# GLOBAL DATA STRUCTURES
# ============================================================

# ------------------------------------------------------------
# LEFT SIDE OF TABLE
#
# appearing ASN ->
#     distinct introducing ASNs
# ------------------------------------------------------------

appearing_to_introducing = defaultdict(
    set
)


# ------------------------------------------------------------
# RIGHT SIDE OF TABLE
#
# introducing ASN ->
#     distinct appearing ASNs
# ------------------------------------------------------------

introducing_to_appearing = defaultdict(
    set
)


# ------------------------------------------------------------
# Prefix coverage
# ------------------------------------------------------------

appearing_to_prefixes = defaultdict(
    set
)

introducing_to_prefixes = defaultdict(
    set
)


# ------------------------------------------------------------
# Global denominator sets
# ------------------------------------------------------------

all_introducing_asns = set()

all_appearing_asns = set()

all_prefixes = set()


# ------------------------------------------------------------
# Processing counters
# ------------------------------------------------------------

total_rows = 0
usable_rows = 0
invalid_rows = 0


# ============================================================
# PROCESS FILTERED SNAPSHOTS
# ============================================================

print()
print(
    "============================================================"
)
print(
    "PROCESSING FILTERED SNAPSHOTS"
)
print(
    "============================================================"
)
print()


for path in tqdm(
    files,
    total=len(files),
    desc="Processing snapshots",
    unit="file",
    dynamic_ncols=True,
):

    df = pd.read_parquet(
        path,
        columns=[
            "prefix",
            "repeated_asn",
            "poisoned_segment",
        ],
        engine="pyarrow",
    )

    total_rows += len(df)

    for (
        prefix,
        repeated_raw,
        segment_raw,
    ) in zip(
        df["prefix"],
        df["repeated_asn"],
        df["poisoned_segment"],
    ):

        introducing_asn = parse_asn(
            repeated_raw
        )

        segment_asns = parse_segment_asns(
            segment_raw
        )

        # ----------------------------------------------------
        # Valid RS:
        #
        # repeated ASN + at least one enclosed ASN +
        # repeated ASN.
        # ----------------------------------------------------

        if (
            introducing_asn is None
            or len(segment_asns) < 3
        ):
            invalid_rows += 1
            continue


        # ----------------------------------------------------
        # Appearing ASes are the unique ASNs enclosed by
        # the two occurrences of the introducing ASN.
        #
        # Example:
        #
        #   47065 6939 3356 47065
        #
        # introducing = 47065
        # appearing   = {6939, 3356}
        #
        # If the repeated ASN occurs again internally,
        # remove it from the appearing-AS set.
        # ----------------------------------------------------

        appearing_asns = set(
            segment_asns[1:-1]
        )

        appearing_asns.discard(
            introducing_asn
        )


        if not appearing_asns:
            invalid_rows += 1
            continue


        usable_rows += 1

        prefix = str(prefix)


        # ====================================================
        # GLOBAL DENOMINATORS
        # ====================================================

        all_introducing_asns.add(
            introducing_asn
        )

        all_appearing_asns.update(
            appearing_asns
        )

        all_prefixes.add(
            prefix
        )


        # ====================================================
        # RIGHT SIDE
        #
        # introducing ASN -> distinct appearing ASNs
        # ====================================================

        introducing_to_appearing[
            introducing_asn
        ].update(
            appearing_asns
        )

        introducing_to_prefixes[
            introducing_asn
        ].add(
            prefix
        )


        # ====================================================
        # LEFT SIDE
        #
        # appearing ASN -> distinct introducing ASNs
        # ====================================================

        for appearing_asn in appearing_asns:

            appearing_to_introducing[
                appearing_asn
            ].add(
                introducing_asn
            )

            appearing_to_prefixes[
                appearing_asn
            ].add(
                prefix
            )


# ============================================================
# GLOBAL DENOMINATORS
# ============================================================

n_introducing = len(
    all_introducing_asns
)

n_appearing = len(
    all_appearing_asns
)

n_prefixes = len(
    all_prefixes
)


print()
print(
    "============================================================"
)
print(
    "GLOBAL DENOMINATORS — FILTERED DATA"
)
print(
    "============================================================"
)

print(
    f"Total rows               : {total_rows:,}"
)

print(
    f"Usable RS rows           : {usable_rows:,}"
)

print(
    f"Invalid/skipped rows      : {invalid_rows:,}"
)

print(
    f"Distinct introducing ASes : {n_introducing:,}"
)

print(
    f"Distinct appearing ASes    : {n_appearing:,}"
)

print(
    f"Unique prefixes            : {n_prefixes:,}"
)


# ============================================================
# BASIC CONSERVATION CHECK
# ============================================================

if total_rows != (
    usable_rows
    + invalid_rows
):
    raise RuntimeError(
        "Row accounting failed."
    )

print()
print(
    "[PASS] Row accounting reconciles."
)


# ============================================================
# PEERING SANITY CHECK
# ============================================================

print()
print(
    "============================================================"
)
print(
    "PEERING SANITY CHECK"
)
print(
    "============================================================"
)


if 47065 in all_introducing_asns:

    print(
        "[FAIL] AS47065 still occurs as an "
        "introducing ASN."
    )

else:

    print(
        "[PASS] AS47065 does not occur as an "
        "introducing ASN."
    )


if 47065 in all_appearing_asns:

    print(
        "[INFO] AS47065 occurs as an appearing ASN "
        "inside RSs introduced by another ASN."
    )

else:

    print(
        "[INFO] AS47065 does not occur as an "
        "appearing ASN."
    )


# ============================================================
# BUILD APPEARING-AS RANKING
# ============================================================

appearing_rows = []


for (
    asn,
    introducing_set,
) in appearing_to_introducing.items():

    distinct_introducing = len(
        introducing_set
    )

    unique_prefixes = len(
        appearing_to_prefixes[
            asn
        ]
    )

    introducing_pct = (
        100.0
        * distinct_introducing
        / n_introducing
        if n_introducing
        else 0.0
    )

    prefix_pct = (
        100.0
        * unique_prefixes
        / n_prefixes
        if n_prefixes
        else 0.0
    )

    appearing_rows.append(
        {
            "asn": asn,

            "distinct_introducing_asns":
                distinct_introducing,

            "distinct_introducing_pct":
                introducing_pct,

            "unique_prefixes":
                unique_prefixes,

            "unique_prefix_pct":
                prefix_pct,
        }
    )


appearing_df = pd.DataFrame(
    appearing_rows
)


appearing_df = appearing_df.sort_values(
    [
        "distinct_introducing_asns",
        "asn",
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(
    drop=True
)


# ============================================================
# BUILD INTRODUCING-AS RANKING
# ============================================================

introducing_rows = []


for (
    asn,
    appearing_set,
) in introducing_to_appearing.items():

    distinct_appearing = len(
        appearing_set
    )

    unique_prefixes = len(
        introducing_to_prefixes[
            asn
        ]
    )

    appearing_pct = (
        100.0
        * distinct_appearing
        / n_appearing
        if n_appearing
        else 0.0
    )

    prefix_pct = (
        100.0
        * unique_prefixes
        / n_prefixes
        if n_prefixes
        else 0.0
    )

    introducing_rows.append(
        {
            "asn": asn,

            "distinct_appearing_asns":
                distinct_appearing,

            "distinct_appearing_pct":
                appearing_pct,

            "unique_prefixes":
                unique_prefixes,

            "unique_prefix_pct":
                prefix_pct,
        }
    )


introducing_df = pd.DataFrame(
    introducing_rows
)


introducing_df = introducing_df.sort_values(
    [
        "distinct_appearing_asns",
        "asn",
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(
    drop=True
)


# ============================================================
# SAVE COMPLETE RANKINGS
# ============================================================

appearing_df.to_csv(
    OUT_APPEARING,
    index=False,
)

introducing_df.to_csv(
    OUT_INTRODUCING,
    index=False,
)


# ============================================================
# TOP 10
# ============================================================

top_appearing = (
    appearing_df
    .head(10)
    .copy()
)

top_introducing = (
    introducing_df
    .head(10)
    .copy()
)


print()
print(
    "============================================================"
)
print(
    "TOP 10 — ASes APPEARING IN RSs"
)
print(
    "============================================================"
)

print(
    top_appearing.to_string(
        index=False,
        formatters={
            "distinct_introducing_pct":
                lambda x: f"{x:.2f}",

            "unique_prefix_pct":
                lambda x: f"{x:.2f}",
        },
    )
)


print()
print(
    "============================================================"
)
print(
    "TOP 10 — ASes INTRODUCING RSs"
)
print(
    "============================================================"
)

print(
    top_introducing.to_string(
        index=False,
        formatters={
            "distinct_appearing_pct":
                lambda x: f"{x:.2f}",

            "unique_prefix_pct":
                lambda x: f"{x:.2f}",
        },
    )
)


# ============================================================
# COMBINED TOP-10 CSV
# ============================================================

combined_rows = []


for i in range(
    min(
        10,
        len(top_appearing),
        len(top_introducing),
    )
):

    left = top_appearing.iloc[i]
    right = top_introducing.iloc[i]

    combined_rows.append(
        {
            "appearing_asn":
                int(left["asn"]),

            "distinct_introducing_asns":
                int(
                    left[
                        "distinct_introducing_asns"
                    ]
                ),

            "distinct_introducing_pct":
                float(
                    left[
                        "distinct_introducing_pct"
                    ]
                ),

            "appearing_unique_prefixes":
                int(
                    left[
                        "unique_prefixes"
                    ]
                ),

            "appearing_unique_prefix_pct":
                float(
                    left[
                        "unique_prefix_pct"
                    ]
                ),

            "introducing_asn":
                int(right["asn"]),

            "distinct_appearing_asns":
                int(
                    right[
                        "distinct_appearing_asns"
                    ]
                ),

            "distinct_appearing_pct":
                float(
                    right[
                        "distinct_appearing_pct"
                    ]
                ),

            "introducing_unique_prefixes":
                int(
                    right[
                        "unique_prefixes"
                    ]
                ),

            "introducing_unique_prefix_pct":
                float(
                    right[
                        "unique_prefix_pct"
                    ]
                ),
        }
    )


combined_df = pd.DataFrame(
    combined_rows
)

combined_df.to_csv(
    OUT_TOP10,
    index=False,
)


# ============================================================
# BUILD LATEX TABLE
# ============================================================

latex_rows = []


for i in range(
    min(
        10,
        len(top_appearing),
        len(top_introducing),
    )
):

    left = top_appearing.iloc[i]
    right = top_introducing.iloc[i]


    left_asn = int(
        left["asn"]
    )

    left_count = int(
        left[
            "distinct_introducing_asns"
        ]
    )

    left_pct = float(
        left[
            "distinct_introducing_pct"
        ]
    )

    left_prefixes = format_count(
        left[
            "unique_prefixes"
        ]
    )

    left_prefix_pct = float(
        left[
            "unique_prefix_pct"
        ]
    )


    right_asn = int(
        right["asn"]
    )

    right_count = int(
        right[
            "distinct_appearing_asns"
        ]
    )

    right_pct = float(
        right[
            "distinct_appearing_pct"
        ]
    )

    right_prefixes = format_count(
        right[
            "unique_prefixes"
        ]
    )

    right_prefix_pct = float(
        right[
            "unique_prefix_pct"
        ]
    )


    row = (
        f"{left_asn} "
        f"& {left_count} ({left_pct:.2f}) "
        f"& {left_prefixes} ({left_prefix_pct:.2f}) "
        f"& {right_asn} "
        f"& {right_count} ({right_pct:.2f}) "
        f"& {right_prefixes} ({right_prefix_pct:.2f}) "
        "\\\\"
    )

    latex_rows.append(
        row
    )


latex_note = (
    "\\textbf{Notes:} "
    "Counts denote the number of distinct introducing ASes "
    "(left) and appearing ASes (right). "
    f"Percentages are computed over "
    f"{format_denominator(n_introducing)} introducing ASes "
    f"and {format_denominator(n_appearing)} appearing ASes, "
    "respectively. "
    "Unique-prefix percentages are computed over "
    f"{format_denominator(n_prefixes)} total unique prefixes."
)


latex_table = (
    "\\begin{table}[!h]\n"
    "\\centering\n"
    "\\caption{Most frequent ASes appearing within RSs and "
    "introducing RSs, with unique-prefix coverage.}\n"
    "\\label{tab:top_asn_prefixes_compact}\n"
    "\n"
    "{\n"
    "\\begin{tabular}{lcc@{\\hspace{6pt}}|"
    "@{\\hspace{6pt}}lcc}\n"
    "\\toprule\n"
    "\\multicolumn{3}{c}{ASes appearing in RSs} &\n"
    "\\multicolumn{3}{c}{ASes introducing RSs} \\\\\n"
    "\\cmidrule(lr){1-3} \\cmidrule(lr){4-6}\n"
    "\n"
    "ASes\n"
    "& \\shortstack{Distinct\\\\ introducing\\\\ ASes (\\%)}\n"
    "& \\shortstack{Unique\\\\ prefixes\\\\ (\\%)}\n"
    "& ASes\n"
    "& \\shortstack{Distinct\\\\ appearing\\\\ ASes (\\%)}\n"
    "& \\shortstack{Unique\\\\ prefixes\\\\ (\\%)} \\\\\n"
    "\n"
    "\\midrule\n"
    + "\n".join(
        latex_rows
    )
    + "\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "}\n"
    "\\vspace{2pt}\n"
    "\\begin{minipage}{\\columnwidth}\n"
    "\\footnotesize\n"
    "\\raggedright\n"
    + latex_note
    + "\n"
    "\\end{minipage}\n"
    "\n"
    "\\end{table}\n"
)


with open(
    OUT_LATEX,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        latex_table
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_lines = [
    "ASN PARTICIPATION ANALYSIS — FILTERED DATA",
    "=" * 60,
    "",
    f"Input directory: {SEGMENTS_DIR}",
    f"Snapshots: {len(files):,}",
    "",
    f"Total rows: {total_rows:,}",
    f"Usable RS rows: {usable_rows:,}",
    f"Invalid/skipped rows: {invalid_rows:,}",
    "",
    f"Distinct introducing ASes: {n_introducing:,}",
    f"Distinct appearing ASes: {n_appearing:,}",
    f"Unique prefixes: {n_prefixes:,}",
    "",
    (
        "AS47065 introducing: "
        f"{47065 in all_introducing_asns}"
    ),
    (
        "AS47065 appearing: "
        f"{47065 in all_appearing_asns}"
    ),
]


with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# PRINT LATEX ROWS
# ============================================================

print()
print(
    "============================================================"
)
print(
    "LATEX TABLE ROWS"
)
print(
    "============================================================"
)

for row in latex_rows:
    print(
        row
    )


print()
print(
    "============================================================"
)
print(
    "LATEX NOTE"
)
print(
    "============================================================"
)

print(
    latex_note
)


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

print()
print(
    "============================================================"
)
print(
    "OUTPUTS"
)
print(
    "============================================================"
)

print(
    f"Full appearing-AS ranking:\n"
    f"  {OUT_APPEARING}"
)

print(
    f"\nFull introducing-AS ranking:\n"
    f"  {OUT_INTRODUCING}"
)

print(
    f"\nCombined top-10 table data:\n"
    f"  {OUT_TOP10}"
)

print(
    f"\nReady-to-use LaTeX table:\n"
    f"  {OUT_LATEX}"
)

print(
    f"\nAnalysis summary:\n"
    f"  {OUT_SUMMARY}"
)

print()
print(
    "[DONE] ASN participation analysis completed."
)
