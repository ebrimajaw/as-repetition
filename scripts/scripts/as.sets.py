from pathlib import Path
from datetime import datetime
import csv

import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(
    "/data/prefix.aspth/prefix.aspth.parquet"
)

INPUT_PATTERN = "prefix.aspath.*.parquet"

START_DATE = datetime.strptime(
    "2014-01-01",
    "%Y-%m-%d",
).date()

END_DATE = datetime.strptime(
    "2024-12-31",
    "%Y-%m-%d",
).date()

OUTPUT_CSV = (
    INPUT_DIR
    / "as_set_scan_summary.2014-2024.csv"
)


# ============================================================
# SELECT ONLY STUDY-PERIOD FILES
# ============================================================

files = []

for path in INPUT_DIR.glob(INPUT_PATTERN):

    date_str = (
        path.name
        .removeprefix("prefix.aspath.")
        .removesuffix(".parquet")
    )

    try:
        file_date = datetime.strptime(
            date_str,
            "%Y-%m-%d",
        ).date()

    except ValueError:
        # Ignore files matching the broad pattern whose
        # filename does not contain a valid YYYY-MM-DD date.
        continue

    if START_DATE <= file_date <= END_DATE:
        files.append(path)


files = sorted(files)


# ============================================================
# SANITY CHECK
# ============================================================

print("=" * 60)
print("AS-SET SCAN")
print("=" * 60)
print(f"Study period : {START_DATE} to {END_DATE}")
print(f"Matched files: {len(files):,}")

if not files:
    raise RuntimeError(
        f"No files between {START_DATE} and {END_DATE} "
        f"matched {INPUT_PATTERN} in {INPUT_DIR}"
    )

print(f"First file   : {files[0].name}")
print(f"Last file    : {files[-1].name}")

print("\nFirst 10 matched files:")

for file in files[:10]:
    print(f"  {file.name}")

if len(files) > 10:
    print(
        f"  ... and {len(files) - 10:,} more"
    )

print("=" * 60)


# ============================================================
# SCAN AS PATHS
# ============================================================

total_paths = 0
paths_with_as_set = 0

for file in tqdm(
    files,
    desc="Scanning files",
    unit="file",
):

    pf = pq.ParquetFile(file)

    for rg in range(
        pf.metadata.num_row_groups
    ):

        table = pf.read_row_group(
            rg,
            columns=["as_path"],
        )

        for as_path in (
            table.column("as_path").to_pylist()
        ):

            total_paths += 1

            if as_path is None:
                continue

            # AS sets are represented using {...}.
            if (
                "{" in as_path
                or "}" in as_path
            ):
                paths_with_as_set += 1


# ============================================================
# RESULTS
# ============================================================

pct = (
    100
    * paths_with_as_set
    / total_paths
    if total_paths
    else 0.0
)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Study period        : "
    f"{START_DATE} to {END_DATE}"
)

print(
    f"Files scanned       : "
    f"{len(files):,}"
)

print(
    f"Total AS paths      : "
    f"{total_paths:,}"
)

print(
    f"Paths with AS sets  : "
    f"{paths_with_as_set:,}"
)

print(
    f"AS-set percentage   : "
    f"{pct:.8f}%"
)

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
) as fh:

    writer = csv.writer(fh)

    writer.writerow([
        "start_date",
        "end_date",
        "files_scanned",
        "total_as_paths",
        "paths_with_as_set",
        "as_set_percentage",
    ])

    writer.writerow([
        START_DATE.isoformat(),
        END_DATE.isoformat(),
        len(files),
        total_paths,
        paths_with_as_set,
        f"{pct:.8f}",
    ])


print(
    f"\nSaved results to: "
    f"{OUTPUT_CSV}"
)
