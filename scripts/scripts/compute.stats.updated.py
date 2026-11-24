#!/usr/bin/env python3

"""
Efficient monthly statistics for RIPE RIS visibility datasets.

This script replaces the previous TXT-bucket implementation.

Analytical definitions preserved
--------------------------------
PP:
    Deduplicated route = (prefix, AS path)

POP:
    Deduplicated route = (prefix, origin, AS path)

RIB AS path:
    as_path

Non-cons AS path:
    cleaned_aspath

Observation points:
    collector = distinct collector
    peer ASN  = distinct peer_asn
    session   = distinct (collector, peer_ip)

RS-specific statistics:
    distinct rs_path
    distinct introducing_as
    distinct ASNs contained in appearing_ases

Important implementation properties
-----------------------------------
1. No TXT intermediate files.
2. No Python row-by-row loop over the RIB.
3. Exact counts -- no approximate distinct counting.
4. Reads Parquet directly with DuckDB.
5. Processes February and October automatically.
6. Automatically chooses a conservative DuckDB memory limit.
7. Uses a month-specific binary DuckDB temporary directory.
8. Removes temporary state after each month, including after failure.
9. Writes final CSV outputs only after validation succeeds.
"""

from pathlib import Path
import csv
import json
import os
import shutil
import time

import duckdb


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_ROOT = Path(
    "/data/jaw/rs.visibility.analysis/parquet"
)

MONTHS = [
    "2024.02",
    "2024.10",
]

RIB_PATTERN = "rrc*.parquet"
POISON_PATTERN = "rrc*.poisoning.parquet"

OUTPUT_ROOT = (
    BASE_ROOT
    / "dataset.stats"
)

# Leave this as None for automatic selection.
#
# If set manually, examples:
#     MEMORY_LIMIT = "32GB"
#     MEMORY_LIMIT = "64GB"
#
MEMORY_LIMIT = None

# Leave this as None to use a conservative automatic thread count.
THREADS = None

# Fraction of currently available RAM DuckDB may use.
AUTO_MEMORY_FRACTION = 0.60

# Never automatically request more than this much RAM from DuckDB.
AUTO_MEMORY_CAP_GB = 64

# Keep at least this much filesystem space free.
#
# This is a safety guard against filling /data.
MIN_FREE_DISK_GB = 100


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def percentage(numerator, denominator):
    if not denominator:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def fmt(value):
    return f"{int(value):,}"


def fmt_noncons(value, denominator):
    return (
        f"{int(value):,} "
        f"({percentage(value, denominator):.2f}%)"
    )


def elapsed_string(seconds):
    seconds = int(seconds)

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours:d}h "
            f"{minutes:02d}m "
            f"{seconds:02d}s"
        )

    if minutes:
        return (
            f"{minutes:d}m "
            f"{seconds:02d}s"
        )

    return f"{seconds:d}s"


def sql_quote(value):
    return (
        "'"
        + str(value).replace(
            "'",
            "''",
        )
        + "'"
    )


def parquet_list_sql(files):
    if not files:
        raise ValueError(
            "No Parquet files supplied."
        )

    return (
        "["
        + ", ".join(
            sql_quote(path)
            for path in files
        )
        + "]"
    )


def print_header(title):
    print(
        "\n"
        + "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )


# =============================================================================
# RESOURCE MANAGEMENT
# =============================================================================

def available_memory_bytes():
    """
    Return currently available physical memory using Linux /proc/meminfo.

    MemAvailable is preferable to MemFree because Linux deliberately uses
    otherwise-unused RAM for filesystem cache.
    """

    meminfo = Path(
        "/proc/meminfo"
    )

    if not meminfo.exists():
        return None

    values = {}

    with meminfo.open() as fh:
        for line in fh:
            parts = line.split()

            if len(parts) < 2:
                continue

            key = parts[0].rstrip(":")
            value_kb = int(parts[1])

            values[key] = (
                value_kb
                * 1024
            )

    return values.get(
        "MemAvailable"
    )


def choose_memory_limit():
    """
    Choose a conservative DuckDB memory limit automatically.
    """

    if MEMORY_LIMIT is not None:
        return MEMORY_LIMIT

    available = (
        available_memory_bytes()
    )

    if available is None:
        # Conservative fallback.
        return "16GB"

    available_gb = (
        available
        / (1024 ** 3)
    )

    selected_gb = int(
        available_gb
        * AUTO_MEMORY_FRACTION
    )

    selected_gb = min(
        selected_gb,
        AUTO_MEMORY_CAP_GB,
    )

    # Avoid absurdly small values on a busy machine.
    selected_gb = max(
        selected_gb,
        4,
    )

    return (
        f"{selected_gb}GB"
    )


def choose_threads():
    if THREADS is not None:
        return int(THREADS)

    cpu_count = (
        os.cpu_count()
        or 1
    )

    # Avoid excessive parallelism creating unnecessary memory pressure.
    return max(
        1,
        min(
            cpu_count,
            4,
        ),
    )


def disk_free_bytes(path):
    usage = shutil.disk_usage(
        path
    )

    return usage.free


def disk_free_gb(path):
    return (
        disk_free_bytes(path)
        / (1024 ** 3)
    )


def check_disk_space(path):
    free_gb = disk_free_gb(
        path
    )

    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Insufficient free disk space before starting query. "
            f"Available: {free_gb:.2f} GB; "
            f"required safety reserve: "
            f"{MIN_FREE_DISK_GB} GB."
        )

    return free_gb


# =============================================================================
# DUCKDB
# =============================================================================

def make_connection(
    temp_dir,
    memory_limit,
    threads,
):
    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect(
        database=":memory:"
    )

    con.execute(
        f"SET threads = {threads}"
    )

    con.execute(
        "SET memory_limit = "
        f"{sql_quote(memory_limit)}"
    )

    con.execute(
        "SET temp_directory = "
        f"{sql_quote(temp_dir)}"
    )

    con.execute(
        "SET preserve_insertion_order = false"
    )

    return con


def execute_query(
    con,
    sql,
    label,
):
    """
    Execute a query and report elapsed time.
    """

    print_header(label)

    start = time.time()

    result = (
        con.execute(sql)
        .fetchone()
    )

    elapsed = (
        time.time()
        - start
    )

    print(
        f"Completed in "
        f"{elapsed_string(elapsed)}"
    )

    return result


# =============================================================================
# INPUT DISCOVERY / VALIDATION
# =============================================================================

def discover_files(month):
    base_dir = (
        BASE_ROOT
        / month
    )

    poison_dir = (
        base_dir
        / "poisoning"
    )

    rib_files = sorted(
        str(path)
        for path in base_dir.glob(
            RIB_PATTERN
        )
        if path.is_file()
    )

    poison_files = sorted(
        str(path)
        for path in poison_dir.glob(
            POISON_PATTERN
        )
        if path.is_file()
    )

    if not rib_files:
        raise RuntimeError(
            f"{month}: no RIB files found."
        )

    if not poison_files:
        raise RuntimeError(
            f"{month}: no non-cons files found."
        )

    if len(rib_files) != len(poison_files):
        raise RuntimeError(
            f"{month}: RIB/non-cons file count mismatch: "
            f"{len(rib_files):,} vs "
            f"{len(poison_files):,}."
        )

    return (
        base_dir,
        rib_files,
        poison_files,
    )


def get_schema_names(
    con,
    sample_file,
):
    rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            {sql_quote(sample_file)}
        )
        """
    ).fetchall()

    return {
        row[0]
        for row in rows
    }


def validate_schema(
    con,
    sample_file,
    required_columns,
    dataset_name,
):
    available = get_schema_names(
        con,
        sample_file,
    )

    missing = (
        set(required_columns)
        - available
    )

    if missing:
        raise RuntimeError(
            f"{dataset_name}: missing columns "
            f"{sorted(missing)}. "
            f"Available columns: "
            f"{sorted(available)}"
        )

    print(
        f"{dataset_name} schema: PASS"
    )


def validate_view_schema(con, view_name, required_columns, dataset_name):
    """Validate the schema of the combined month view, not only one file."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM {view_name}"
    ).fetchall()
    available = {row[0] for row in rows}
    missing = set(required_columns) - available
    if missing:
        raise RuntimeError(
            f"{dataset_name}: combined view missing columns "
            f"{sorted(missing)}. Available columns: {sorted(available)}"
        )
    print(f"{dataset_name} combined schema: PASS")


# =============================================================================
# TEMPORARY MONTH VIEWS
# =============================================================================

def create_month_views(
    con,
    rib_files,
    poison_files,
):
    """
    Create logical views only.

    No source data is copied.
    """

    rib_sql = parquet_list_sql(
        rib_files
    )

    poison_sql = parquet_list_sql(
        poison_files
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW rib AS
        SELECT
            prefix,
            as_path,
            origin,
            collector,
            peer_asn,
            peer_ip
        FROM read_parquet(
            {rib_sql},
            union_by_name = true
        )
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE VIEW noncons AS
        SELECT
            prefix,
            origin,
            cleaned_aspath,
            collector,
            peer_asn,
            peer_ip,
            rs_path,
            introducing_as,
            appearing_ases
        FROM read_parquet(
            {poison_sql},
            union_by_name = true
        )
        """
    )


# =============================================================================
# INPUT / OBSERVATION STATISTICS
# =============================================================================

def compute_basic_stats(
    con,
    view_name,
    label,
):
    """
    Compute row count and observation-point statistics together.

    This avoids the old implementation's TXT spill for:
        collectors
        peer ASNs
        sessions
    """

    sql = f"""
        SELECT
            COUNT(*) AS rows,

            COUNT(DISTINCT collector)
                FILTER (
                    WHERE collector IS NOT NULL
                )
                AS collectors,

            COUNT(DISTINCT peer_asn)
                FILTER (
                    WHERE peer_asn IS NOT NULL
                )
                AS peer_asns,

            COUNT(
                DISTINCT
                struct_pack(
                    collector := collector,
                    peer_ip := peer_ip
                )
            )
                FILTER (
                    WHERE
                        collector IS NOT NULL
                        AND peer_ip IS NOT NULL
                )
                AS sessions

        FROM {view_name}
    """

    row = execute_query(
        con,
        sql,
        f"{label.upper()} ROW / OBSERVATION-POINT STATISTICS",
    )

    return {
        "rows": int(row[0]),
        "collectors": int(row[1]),
        "peer_asns": int(row[2]),
        "sessions": int(row[3]),
    }


# =============================================================================
# LARGE-RIB EXACT DEDUPLICATION
# =============================================================================

RIB_HASH_PARTITIONS = 256
RIB_STAGE_VERSION = 2


def all_staged_parquet_files(stage_root):
    """Return all locally deduplicated RIB staging files."""
    return sorted(
        str(path)
        for path in stage_root.glob("source_*.parquet")
        if path.is_file()
    )


def rib_stage_manifest(rib_files):
    """Build an exact manifest used to decide whether a stage is reusable."""
    sources = []

    for source_file in rib_files:
        path = Path(source_file)
        stat = path.stat()
        sources.append({
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })

    return {
        "stage_version": RIB_STAGE_VERSION,
        "hash_partitions": RIB_HASH_PARTITIONS,
        "sources": sources,
    }


def load_json(path):
    if not path.exists():
        return None

    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception:
        return None


def write_json_atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
    tmp.replace(path)


def stage_rib_for_exact_dedup(con, rib_files, work_dir):
    """
    Build or resume a persistent, disk-backed, locally deduplicated RIB stage.

    Each source RIB becomes one compact Parquet file containing distinct
    (prefix, path, origin) rows plus four deterministic hash buckets:

        route_bucket  = hash(prefix, path)
        path_bucket   = hash(path)
        prefix_bucket = hash(prefix)
        origin_bucket = hash(origin)

    route_bucket preserves the exact PP/POP composite-route calculation.
    The single-column buckets allow exact monthly distinct counts to be summed
    over disjoint buckets without ever building one month-wide DISTINCT table.

    The stage is persistent and resumable. It is reused only when its manifest
    exactly matches the current source files and staging definition.
    """
    stage_root = work_dir / "rib_dedup_stage"
    manifest_file = stage_root / "manifest.json"
    expected_manifest = rib_stage_manifest(rib_files)

    existing_manifest = load_json(manifest_file)

    if existing_manifest != expected_manifest:
        if stage_root.exists():
            print_header("RESET STALE RIB DEDUP STAGE")
            print(
                "Existing stage does not match the current source files or "
                "staging definition; rebuilding it."
            )
            shutil.rmtree(stage_root, ignore_errors=True)

        stage_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_file, expected_manifest)
    else:
        stage_root.mkdir(parents=True, exist_ok=True)

    print_header("BUILD / RESUME DISK-BACKED RIB DEDUP STAGE")
    start = time.time()
    reused = 0
    built = 0

    for idx, source_file in enumerate(rib_files, start=1):
        output_file = stage_root / f"source_{idx:04d}.parquet"

        if output_file.exists() and output_file.stat().st_size > 0:
            reused += 1
        else:
            tmp_output = stage_root / f"source_{idx:04d}.parquet.tmp"
            tmp_output.unlink(missing_ok=True)

            sql = f"""
                COPY (
                    SELECT *
                    FROM (
                        SELECT DISTINCT
                            prefix,
                            as_path AS path,
                            origin,
                            CASE
                                WHEN strpos(CAST(prefix AS VARCHAR), ':') = 0
                                    THEN 'ipv4'
                                ELSE 'ipv6'
                            END AS af,
                            CAST(
                                hash(prefix, as_path)
                                % {RIB_HASH_PARTITIONS}
                                AS INTEGER
                            ) AS route_bucket,
                            CAST(
                                hash(as_path)
                                % {RIB_HASH_PARTITIONS}
                                AS INTEGER
                            ) AS path_bucket,
                            CAST(
                                hash(prefix)
                                % {RIB_HASH_PARTITIONS}
                                AS INTEGER
                            ) AS prefix_bucket,
                            CASE
                                WHEN origin IS NOT NULL THEN CAST(
                                    hash(origin)
                                    % {RIB_HASH_PARTITIONS}
                                    AS INTEGER
                                )
                                ELSE NULL
                            END AS origin_bucket
                        FROM read_parquet(
                            {sql_quote(source_file)}
                        )
                        WHERE
                            prefix IS NOT NULL
                            AND as_path IS NOT NULL
                    )
                    ORDER BY af, route_bucket
                )
                TO {sql_quote(tmp_output)}
                (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE 122880
                )
            """

            con.execute(sql)
            tmp_output.replace(output_file)
            built += 1

        free_gb = check_disk_space(BASE_ROOT)
        if idx == 1 or idx % 25 == 0 or idx == len(rib_files):
            print(
                f"Stage {idx:,}/{len(rib_files):,}: "
                f"built={built:,}, reused={reused:,}, "
                f"free disk={free_gb:.2f} GB"
            )

    staged_files = all_staged_parquet_files(stage_root)
    if len(staged_files) != len(rib_files):
        raise RuntimeError(
            "RIB stage is incomplete after staging: "
            f"expected {len(rib_files):,} files, found {len(staged_files):,}."
        )

    print(
        "Stage ready in "
        f"{elapsed_string(time.time() - start)} "
        f"(built={built:,}, reused={reused:,})"
    )

    return stage_root


def load_bucket_checkpoint(path):
    """Load completed exact bucket counts from a small CSV checkpoint."""
    values = {}

    if not path.exists():
        return values

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            values[int(row["bucket"])] = int(row["count"])

    return values


def save_bucket_checkpoint(path, values):
    """Atomically save exact bucket counts so a failed run can resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bucket", "count"])
        for bucket in sorted(values):
            writer.writerow([bucket, values[bucket]])

    tmp.replace(path)


def exact_bucket_distinct(
    con,
    files_sql,
    af,
    column,
    bucket_column,
    require_origin,
    metric_label,
    checkpoint_dir,
):
    """
    Count one column exactly by summing distinct counts over disjoint buckets.

    Correctness condition:
        bucket_column is a deterministic hash of `column` itself. Therefore one
        distinct value can occur in exactly one bucket, making the sum exact.
    """
    checkpoint_file = (
        checkpoint_dir
        / f"{af}.{metric_label.lower().replace(' ', '_')}.csv"
    )
    counts = load_bucket_checkpoint(checkpoint_file)

    if counts:
        print(
            f"Resuming {metric_label}: "
            f"{len(counts):,}/{RIB_HASH_PARTITIONS:,} buckets already complete."
        )

    origin_clause = (
        " AND origin IS NOT NULL"
        if require_origin
        else ""
    )

    start = time.time()

    for bucket in range(RIB_HASH_PARTITIONS):
        if bucket in counts:
            continue

        row = con.execute(
            f"""
            SELECT COUNT(DISTINCT {column})
            FROM read_parquet(
                {files_sql},
                union_by_name = true
            )
            WHERE af = {sql_quote(af)}
              AND {column} IS NOT NULL
              AND {bucket_column} = {bucket}
              {origin_clause}
            """
        ).fetchone()

        counts[bucket] = int(row[0])
        save_bucket_checkpoint(checkpoint_file, counts)

        if (
            bucket == 0
            or (bucket + 1) % 32 == 0
            or bucket == RIB_HASH_PARTITIONS - 1
        ):
            print(
                f"Bucket {bucket + 1:3d}/{RIB_HASH_PARTITIONS}: "
                f"distinct={sum(counts.values()):,}, "
                f"free disk={disk_free_gb(BASE_ROOT):.2f} GB"
            )

    if len(counts) != RIB_HASH_PARTITIONS:
        raise RuntimeError(
            f"Incomplete checkpoint for {metric_label}: "
            f"{len(counts):,}/{RIB_HASH_PARTITIONS:,} buckets."
        )

    total = sum(counts.values())
    print(
        f"Completed in {elapsed_string(time.time() - start)}; "
        f"exact distinct={total:,}"
    )
    return total


def compute_rib_pp_pop_stats_from_stage(con, stage_root, checkpoint_dir):
    """
    Compute exact RIB PP and POP statistics from the persistent disk stage.

    No month-wide DISTINCT is used for the large RIB metrics. Composite route
    counts and single-column cardinalities are all computed over exact disjoint
    hash buckets and summed. Small CSV checkpoints make every bucketed stage
    resumable after failure.
    """
    pp = {}
    pop = {}

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    files = all_staged_parquet_files(stage_root)
    if not files:
        raise RuntimeError("No staged RIB files found.")
    files_sql = parquet_list_sql(files)

    for af in ("ipv4", "ipv6"):
        print_header(f"RIB PP/POP {af.upper()} — EXACT HASH BUCKETS")
        af_start = time.time()

        pp_checkpoint = checkpoint_dir / f"{af}.pp_routes.csv"
        pop_checkpoint = checkpoint_dir / f"{af}.pop_routes.csv"
        pp_bucket_counts = load_bucket_checkpoint(pp_checkpoint)
        pop_bucket_counts = load_bucket_checkpoint(pop_checkpoint)

        for bucket in range(RIB_HASH_PARTITIONS):
            if bucket not in pp_bucket_counts:
                row = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT
                            prefix,
                            path
                        FROM read_parquet(
                            {files_sql},
                            union_by_name = true
                        )
                        WHERE af = {sql_quote(af)}
                          AND route_bucket = {bucket}
                    )
                    """
                ).fetchone()
                pp_bucket_counts[bucket] = int(row[0])
                save_bucket_checkpoint(pp_checkpoint, pp_bucket_counts)

            if bucket not in pop_bucket_counts:
                row = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT
                            prefix,
                            origin,
                            path
                        FROM read_parquet(
                            {files_sql},
                            union_by_name = true
                        )
                        WHERE af = {sql_quote(af)}
                          AND route_bucket = {bucket}
                          AND origin IS NOT NULL
                    )
                    """
                ).fetchone()
                pop_bucket_counts[bucket] = int(row[0])
                save_bucket_checkpoint(pop_checkpoint, pop_bucket_counts)

            if (
                bucket == 0
                or (bucket + 1) % 32 == 0
                or bucket == RIB_HASH_PARTITIONS - 1
            ):
                free_gb = check_disk_space(BASE_ROOT)
                print(
                    f"Bucket {bucket + 1:3d}/{RIB_HASH_PARTITIONS}: "
                    f"PP routes={sum(pp_bucket_counts.values()):,}, "
                    f"POP routes={sum(pop_bucket_counts.values()):,}, "
                    f"free disk={free_gb:.2f} GB"
                )

        if (
            len(pp_bucket_counts) != RIB_HASH_PARTITIONS
            or len(pop_bucket_counts) != RIB_HASH_PARTITIONS
        ):
            raise RuntimeError(f"Incomplete route bucket checkpoints for {af}.")

        pp_routes = sum(pp_bucket_counts.values())
        pop_routes = sum(pop_bucket_counts.values())

        print_header(f"RIB {af.upper()} PP UNIQUE PATHS — EXACT HASH BUCKETS")
        pp_unique_paths = exact_bucket_distinct(
            con, files_sql, af,
            "path", "path_bucket", False,
            "PP UNIQUE PATHS", checkpoint_dir,
        )

        print_header(f"RIB {af.upper()} PP UNIQUE PREFIXES — EXACT HASH BUCKETS")
        pp_unique_prefixes = exact_bucket_distinct(
            con, files_sql, af,
            "prefix", "prefix_bucket", False,
            "PP UNIQUE PREFIXES", checkpoint_dir,
        )

        # PP origins intentionally preserve the original source-row semantics:
        # prefix and origin must be non-null, but path may be null. The origin
        # cardinality is small, so this exact source-view DISTINCT is retained.
        af_condition = (
            "strpos(CAST(prefix AS VARCHAR), ':') = 0"
            if af == "ipv4"
            else "strpos(CAST(prefix AS VARCHAR), ':') > 0"
        )
        row = execute_query(
            con,
            f"""
            SELECT COUNT(DISTINCT origin)
            FROM rib
            WHERE prefix IS NOT NULL
              AND origin IS NOT NULL
              AND {af_condition}
            """,
            f"RIB {af.upper()} PP UNIQUE ORIGINS",
        )
        pp_unique_origins = int(row[0])

        print_header(f"RIB {af.upper()} POP UNIQUE PATHS — EXACT HASH BUCKETS")
        pop_unique_paths = exact_bucket_distinct(
            con, files_sql, af,
            "path", "path_bucket", True,
            "POP UNIQUE PATHS", checkpoint_dir,
        )

        print_header(f"RIB {af.upper()} POP UNIQUE PREFIXES — EXACT HASH BUCKETS")
        pop_unique_prefixes = exact_bucket_distinct(
            con, files_sql, af,
            "prefix", "prefix_bucket", True,
            "POP UNIQUE PREFIXES", checkpoint_dir,
        )

        print_header(f"RIB {af.upper()} POP UNIQUE ORIGINS — EXACT HASH BUCKETS")
        pop_unique_origins = exact_bucket_distinct(
            con, files_sql, af,
            "origin", "origin_bucket", True,
            "POP UNIQUE ORIGINS", checkpoint_dir,
        )

        pp[f"{af}_routes"] = pp_routes
        pp[f"{af}_paths"] = pp_unique_paths
        pp[f"{af}_prefixes"] = pp_unique_prefixes
        pp[f"{af}_origins"] = pp_unique_origins

        pop[f"{af}_routes"] = pop_routes
        pop[f"{af}_paths"] = pop_unique_paths
        pop[f"{af}_prefixes"] = pop_unique_prefixes
        pop[f"{af}_origins"] = pop_unique_origins

        print(
            f"Completed {af.upper()} PP/POP in "
            f"{elapsed_string(time.time() - af_start)}"
        )

    return pp, pop


# =============================================================================
# PP STATISTICS
# =============================================================================

def compute_pp_stats(
    con,
    view_name,
    path_column,
    label,
):
    """
    PP = unique (prefix, AS path).

    For each address family:
        - deduplicated routes
        - unique paths
        - unique prefixes

    PP unique origins preserve the old code's semantics:
        distinct origin values among rows with a non-null prefix and origin
        for the corresponding address family.
    """

    results = {}

    for af, af_condition in [
        (
            "ipv4",
            "strpos(CAST(prefix AS VARCHAR), ':') = 0",
        ),
        (
            "ipv6",
            "strpos(CAST(prefix AS VARCHAR), ':') > 0",
        ),
    ]:

        sql = f"""
            WITH dedup AS MATERIALIZED (
                SELECT DISTINCT
                    prefix,
                    {path_column} AS path
                FROM {view_name}
                WHERE
                    prefix IS NOT NULL
                    AND {path_column} IS NOT NULL
                    AND {af_condition}
            )

            SELECT
                COUNT(*) AS deduplicated_routes,
                COUNT(DISTINCT path) AS unique_paths,
                COUNT(DISTINCT prefix) AS unique_prefixes

            FROM dedup
        """

        row = execute_query(
            con,
            sql,
            (
                f"{label.upper()} PP "
                f"{af.upper()}"
            ),
        )

        results[
            f"{af}_routes"
        ] = int(row[0])

        results[
            f"{af}_paths"
        ] = int(row[1])

        results[
            f"{af}_prefixes"
        ] = int(row[2])

    # -------------------------------------------------------------
    # PP origins for both AFs in one scan.
    # -------------------------------------------------------------

    sql = f"""
        SELECT
            COUNT(DISTINCT origin)
                FILTER (
                    WHERE
                        prefix IS NOT NULL
                        AND origin IS NOT NULL
                        AND strpos(
                            CAST(prefix AS VARCHAR),
                            ':'
                        ) = 0
                )
                AS ipv4_origins,

            COUNT(DISTINCT origin)
                FILTER (
                    WHERE
                        prefix IS NOT NULL
                        AND origin IS NOT NULL
                        AND strpos(
                            CAST(prefix AS VARCHAR),
                            ':'
                        ) > 0
                )
                AS ipv6_origins

        FROM {view_name}
    """

    row = execute_query(
        con,
        sql,
        (
            f"{label.upper()} PP "
            "UNIQUE ORIGINS"
        ),
    )

    results["ipv4_origins"] = int(
        row[0]
    )

    results["ipv6_origins"] = int(
        row[1]
    )

    return results


# =============================================================================
# POP STATISTICS
# =============================================================================

def compute_pop_stats(
    con,
    view_name,
    path_column,
    label,
):
    """
    POP = unique (prefix, origin, AS path).

    The old implementation generated POP keys only when origin was
    non-null. This implementation preserves that condition.
    """

    results = {}

    for af, af_condition in [
        (
            "ipv4",
            "strpos(CAST(prefix AS VARCHAR), ':') = 0",
        ),
        (
            "ipv6",
            "strpos(CAST(prefix AS VARCHAR), ':') > 0",
        ),
    ]:

        sql = f"""
            WITH dedup AS MATERIALIZED (
                SELECT DISTINCT
                    prefix,
                    origin,
                    {path_column} AS path
                FROM {view_name}
                WHERE
                    prefix IS NOT NULL
                    AND origin IS NOT NULL
                    AND {path_column} IS NOT NULL
                    AND {af_condition}
            )

            SELECT
                COUNT(*) AS deduplicated_routes,
                COUNT(DISTINCT path) AS unique_paths,
                COUNT(DISTINCT prefix) AS unique_prefixes,
                COUNT(DISTINCT origin) AS unique_origins

            FROM dedup
        """

        row = execute_query(
            con,
            sql,
            (
                f"{label.upper()} POP "
                f"{af.upper()}"
            ),
        )

        results[
            f"{af}_routes"
        ] = int(row[0])

        results[
            f"{af}_paths"
        ] = int(row[1])

        results[
            f"{af}_prefixes"
        ] = int(row[2])

        results[
            f"{af}_origins"
        ] = int(row[3])

    return results


# =============================================================================
# RS-SPECIFIC STATISTICS
# =============================================================================

def compute_rs_specific_stats(
    con,
):
    """
    Compute exact monthly unique:
        RS paths
        introducing ASes
        appearing ASes
    """

    sql = """
        SELECT
            COUNT(DISTINCT rs_path)
                FILTER (
                    WHERE rs_path IS NOT NULL
                )
                AS rs_paths,

            COUNT(DISTINCT introducing_as)
                FILTER (
                    WHERE introducing_as IS NOT NULL
                )
                AS introducing_ases

        FROM noncons
    """

    row = execute_query(
        con,
        sql,
        "RS-SPECIFIC COUNTS",
    )

    rs_paths = int(
        row[0]
    )

    introducing_ases = int(
        row[1]
    )

    sql = """
        SELECT
            COUNT(DISTINCT appearing_asn)
        FROM (
            SELECT
                UNNEST(appearing_ases)
                    AS appearing_asn
            FROM noncons
            WHERE
                appearing_ases IS NOT NULL
        )
        WHERE
            appearing_asn IS NOT NULL
    """

    row = execute_query(
        con,
        sql,
        "UNIQUE APPEARING ASes",
    )

    appearing_ases = int(
        row[0]
    )

    return {
        "rs_paths": rs_paths,
        "introducing_ases":
            introducing_ases,
        "appearing_ases":
            appearing_ases,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(
    rib_basic,
    rs_basic,
    pp_rib,
    pp_rs,
    pop_rib,
    pop_rs,
    rs_specific,
):
    print_header(
        "VALIDATION"
    )

    checks = []

    # -------------------------------------------------------------
    # Basic positivity.
    # -------------------------------------------------------------

    checks.append((
        "RIB rows > 0",
        rib_basic["rows"] > 0,
    ))

    checks.append((
        "Non-cons rows > 0",
        rs_basic["rows"] > 0,
    ))

    # -------------------------------------------------------------
    # Observation-point subset checks.
    #
    # The non-cons files are derived from the RIB observations,
    # so these populations should not exceed the RIB populations.
    # -------------------------------------------------------------

    checks.append((
        "Non-cons collectors <= RIB collectors",
        (
            rs_basic["collectors"]
            <= rib_basic["collectors"]
        ),
    ))

    checks.append((
        "Non-cons peer ASNs <= RIB peer ASNs",
        (
            rs_basic["peer_asns"]
            <= rib_basic["peer_asns"]
        ),
    ))

    checks.append((
        "Non-cons sessions <= RIB sessions",
        (
            rs_basic["sessions"]
            <= rib_basic["sessions"]
        ),
    ))

    # -------------------------------------------------------------
    # Route-level internal consistency.
    # -------------------------------------------------------------

    for dataset_label, stats, rows in [
        (
            "RIB PP",
            pp_rib,
            rib_basic["rows"],
        ),
        (
            "RIB POP",
            pop_rib,
            rib_basic["rows"],
        ),
        (
            "Non-cons PP",
            pp_rs,
            rs_basic["rows"],
        ),
        (
            "Non-cons POP",
            pop_rs,
            rs_basic["rows"],
        ),
    ]:

        for af in [
            "ipv4",
            "ipv6",
        ]:

            routes = stats[
                f"{af}_routes"
            ]

            paths = stats[
                f"{af}_paths"
            ]

            prefixes = stats[
                f"{af}_prefixes"
            ]

            origins = stats[
                f"{af}_origins"
            ]

            checks.append((
                (
                    f"{dataset_label} "
                    f"{af} routes <= rows"
                ),
                routes <= rows,
            ))

            checks.append((
                (
                    f"{dataset_label} "
                    f"{af} paths <= routes"
                ),
                paths <= routes,
            ))

            checks.append((
                (
                    f"{dataset_label} "
                    f"{af} prefixes <= routes"
                ),
                prefixes <= routes,
            ))

            # For POP, origins are derived directly from the
            # deduplicated POP relation and therefore must be <= routes.
            #
            # For PP, origins preserve the original script's independent
            # source-row semantics. They should still be far below the
            # route count, but we avoid relying on that as a definition.
            if "POP" in dataset_label:
                checks.append((
                    (
                        f"{dataset_label} "
                        f"{af} origins <= routes"
                    ),
                    origins <= routes,
                ))

    # -------------------------------------------------------------
    # POP cannot exceed PP for the same source rows?
    #
    # NO assertion is made here.
    #
    # Adding origin to the key can split one PP route into multiple POP
    # routes if the same prefix/path is represented with different origins.
    # -------------------------------------------------------------

    checks.append((
        "Unique RS paths > 0",
        rs_specific["rs_paths"] > 0,
    ))

    checks.append((
        "Introducing ASes > 0",
        (
            rs_specific[
                "introducing_ases"
            ]
            > 0
        ),
    ))

    checks.append((
        "Appearing ASes > 0",
        (
            rs_specific[
                "appearing_ases"
            ]
            > 0
        ),
    ))

    failed = []

    for description, passed in checks:

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status:<5} "
            f"{description}"
        )

        if not passed:
            failed.append(
                description
            )

    if failed:
        raise RuntimeError(
            "Validation failed:\n  - "
            + "\n  - ".join(
                failed
            )
        )

    print(
        "-" * 72
    )

    print(
        f"All {len(checks):,} "
        "validation checks PASS."
    )


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_observation_stats(
    rib_basic,
    rs_basic,
):
    print(
        "\n"
        + "=" * 80
    )

    print(
        "OBSERVATION-POINT STATISTICS"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Metric':<24}"
        f"{'All RIB':>18}"
        f"{'Non-cons':>18}"
        f"{'Non-cons/RIB':>18}"
    )

    print(
        "-" * 80
    )

    for label, key in [
        (
            "Rows",
            "rows",
        ),
        (
            "Unique collectors",
            "collectors",
        ),
        (
            "Unique peer ASNs",
            "peer_asns",
        ),
        (
            "Unique sessions",
            "sessions",
        ),
    ]:

        rib_value = (
            rib_basic[key]
        )

        rs_value = (
            rs_basic[key]
        )

        print(
            f"{label:<24}"
            f"{rib_value:>18,}"
            f"{rs_value:>18,}"
            f"{percentage(rs_value, rib_value):>17.2f}%"
        )

    print(
        "=" * 80
    )


def print_dedup_table(
    title,
    rib_stats,
    rs_stats,
    rs_specific,
):
    print(
        "\n"
        + "=" * 104
    )

    print(title)

    print(
        "=" * 104
    )

    print(
        f"{'Metric':<24}"
        f"{'AS Paths IPv4':>18}"
        f"{'AS Paths IPv6':>18}"
        f"{'Non-Cons IPv4':>22}"
        f"{'Non-Cons IPv6':>22}"
    )

    print(
        "-" * 104
    )

    print(
        f"{'Deduplicated routes':<24}"
        f"{fmt(rib_stats['ipv4_routes']):>18}"
        f"{fmt(rib_stats['ipv6_routes']):>18}"
        f"{fmt(rs_stats['ipv4_routes']):>22}"
        f"{fmt(rs_stats['ipv6_routes']):>22}"
    )

    print(
        f"{'Unique paths':<24}"
        f"{fmt(rib_stats['ipv4_paths']):>18}"
        f"{fmt(rib_stats['ipv6_paths']):>18}"
        f"{fmt_noncons(rs_stats['ipv4_paths'], rib_stats['ipv4_paths']):>22}"
        f"{fmt_noncons(rs_stats['ipv6_paths'], rib_stats['ipv6_paths']):>22}"
    )

    print(
        f"{'Unique prefixes':<24}"
        f"{fmt(rib_stats['ipv4_prefixes']):>18}"
        f"{fmt(rib_stats['ipv6_prefixes']):>18}"
        f"{fmt_noncons(rs_stats['ipv4_prefixes'], rib_stats['ipv4_prefixes']):>22}"
        f"{fmt_noncons(rs_stats['ipv6_prefixes'], rib_stats['ipv6_prefixes']):>22}"
    )

    print(
        f"{'Unique origins':<24}"
        f"{fmt(rib_stats['ipv4_origins']):>18}"
        f"{fmt(rib_stats['ipv6_origins']):>18}"
        f"{fmt_noncons(rs_stats['ipv4_origins'], rib_stats['ipv4_origins']):>22}"
        f"{fmt_noncons(rs_stats['ipv6_origins'], rib_stats['ipv6_origins']):>22}"
    )

    print(
        "-" * 104
    )

    print(
        "Unique repetition segments : "
        f"{fmt(rs_specific['rs_paths'])}"
    )

    print(
        "Introducing ASes           : "
        f"{fmt(rs_specific['introducing_ases'])}"
    )

    print(
        "Appearing ASes             : "
        f"{fmt(rs_specific['appearing_ases'])}"
    )

    print(
        "=" * 104
    )


# =============================================================================
# OUTPUT CSVs
# =============================================================================

def save_observation_csv(
    month,
    out_dir,
    rib_basic,
    rs_basic,
):
    output_file = (
        out_dir
        / "observation_point_stats.csv"
    )

    with output_file.open(
        "w",
        newline="",
    ) as fh:

        writer = csv.writer(
            fh
        )

        writer.writerow([
            "month",
            "metric",
            "all_rib",
            "noncons",
            "noncons_to_rib_pct",
        ])

        for key in [
            "rows",
            "collectors",
            "peer_asns",
            "sessions",
        ]:

            writer.writerow([
                month,
                key,
                rib_basic[key],
                rs_basic[key],
                (
                    f"{percentage(rs_basic[key], rib_basic[key]):.6f}"
                ),
            ])

    print(
        f"Saved: {output_file}"
    )


def save_comparison_csv(
    month,
    out_dir,
    pp_rib,
    pp_rs,
    pop_rib,
    pop_rs,
    rs_specific,
):
    output_file = (
        out_dir
        / "dedup_comparison.csv"
    )

    with output_file.open(
        "w",
        newline="",
    ) as fh:

        writer = csv.writer(
           fh
        )

        writer.writerow([
            "month",
            "deduplication",
            "metric",
            "as_paths_ipv4",
            "as_paths_ipv6",
            "noncons_ipv4",
            "noncons_ipv6",
        ])

        for (
            deduplication,
            rib_stats,
            rs_stats,
        ) in [
            (
                "prefix_aspath",
                pp_rib,
                pp_rs,
            ),
            (
                "prefix_origin_aspath",
                pop_rib,
                pop_rs,
            ),
        ]:

            for (
                metric_name,
                metric_key,
            ) in [
                (
                    "deduplicated_routes",
                    "routes",
                ),
                (
                    "unique_paths",
                    "paths",
                ),
                (
                    "unique_prefixes",
                    "prefixes",
                ),
                (
                    "unique_origins",
                    "origins",
                ),
            ]:

                writer.writerow([
                    month,
                    deduplication,
                    metric_name,
                    rib_stats[
                        f"ipv4_{metric_key}"
                    ],
                    rib_stats[
                        f"ipv6_{metric_key}"
                    ],
                    rs_stats[
                        f"ipv4_{metric_key}"
                    ],
                    rs_stats[
                        f"ipv6_{metric_key}"
                    ],
                ])

        writer.writerow([
            month,
            "rs_specific",
            "unique_repetition_segments",
            "",
            "",
            rs_specific["rs_paths"],
            "",
        ])

        writer.writerow([
            month,
            "rs_specific",
            "introducing_ases",
            "",
            "",
            rs_specific[
                "introducing_ases"
            ],
            "",
        ])

        writer.writerow([
            month,
            "rs_specific",
            "appearing_ases",
            "",
            "",
            rs_specific[
                "appearing_ases"
            ],
            "",
        ])

    print(
        f"Saved: {output_file}"
    )


def save_summary_csv(
    month,
    out_dir,
    rib_basic,
    rs_basic,
    pp_rib,
    pp_rs,
    pop_rib,
    pop_rs,
    rs_specific,
):
    output_file = (
        out_dir
        / "monthly_summary.csv"
    )

    rows = []

    def add(
        section,
        metric,
        rib_value,
        rs_value,
    ):
        rows.append([
            month,
            section,
            metric,
            rib_value,
            rs_value,
            (
                f"{percentage(rs_value, rib_value):.6f}"
                if rib_value is not None
                and rs_value is not None
                else ""
            ),
        ])

    add(
        "input",
        "rows",
        rib_basic["rows"],
        rs_basic["rows"],
    )

    for (
        deduplication,
        rib_stats,
        rs_stats,
    ) in [
        (
            "prefix_aspath",
            pp_rib,
            pp_rs,
        ),
        (
            "prefix_origin_aspath",
            pop_rib,
            pop_rs,
        ),
    ]:

        for af in [
            "ipv4",
            "ipv6",
        ]:

            for metric in [
                "routes",
                "paths",
                "prefixes",
                "origins",
            ]:

                add(
                    (
                        f"{deduplication}_"
                        f"{af}"
                    ),
                    metric,
                    rib_stats[
                        f"{af}_{metric}"
                    ],
                    rs_stats[
                        f"{af}_{metric}"
                    ],
                )

    for metric in [
        "collectors",
        "peer_asns",
        "sessions",
    ]:

        add(
            "observation_points",
            metric,
            rib_basic[metric],
            rs_basic[metric],
        )

    rows.extend([
        [
            month,
            "rs_specific",
            "unique_repetition_segments",
            "",
            rs_specific[
                "rs_paths"
            ],
            "",
        ],
        [
            month,
            "rs_specific",
            "introducing_ases",
            "",
            rs_specific[
                "introducing_ases"
            ],
            "",
        ],
        [
            month,
            "rs_specific",
            "appearing_ases",
            "",
            rs_specific[
                "appearing_ases"
            ],
            "",
        ],
    ])

    with output_file.open(
        "w",
        newline="",
    ) as fh:

        writer = csv.writer(
            fh
        )

        writer.writerow([
            "month",
            "section",
            "metric",
            "all_rib",
            "noncons",
            "noncons_to_rib_pct",
        ])

        writer.writerows(
            rows
        )

    print(
        f"Saved: {output_file}"
    )


# =============================================================================
# PROCESS ONE MONTH
# =============================================================================

def process_month(
    month,
    memory_limit,
    threads,
):
    month_start = time.time()

    (
        base_dir,
        rib_files,
        poison_files,
    ) = discover_files(
        month
    )

    out_dir = (
        OUTPUT_ROOT
        / month
    )

    temp_dir = (
        out_dir
        / "duckdb_tmp"
    )

    # Persistent work state for the expensive exact RIB computation.
    # Unlike DuckDB spill files, this is intentionally retained across runs so
    # staging and completed hash buckets can be resumed after a failure.
    work_dir = (
        out_dir
        / "exact_rib_work"
    )

    checkpoint_dir = (
        work_dir
        / "checkpoints"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only stale temporary state created by this new script.
    #
    # We deliberately do NOT touch the old "tmp" directory here because
    # deleting old data automatically is a separate decision from computing
    # the new results.
    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    free_before = check_disk_space(
        BASE_ROOT
    )

    print(
        "\n"
        + "#" * 72
    )

    print(
        f"MONTH: {month}"
    )

    print(
        "#" * 72
    )

    print(
        f"Input directory : {base_dir}"
    )

    print(
        f"Output directory: {out_dir}"
    )

    print(
        f"RIB files       : "
        f"{len(rib_files):,}"
    )

    print(
        f"Non-cons files  : "
        f"{len(poison_files):,}"
    )

    print(
        f"DuckDB memory   : "
        f"{memory_limit}"
    )

    print(
        f"DuckDB threads  : "
        f"{threads}"
    )

    print(
        f"Free disk       : "
        f"{free_before:.2f} GB"
    )

    con = None

    try:
        con = make_connection(
            temp_dir,
            memory_limit,
            threads,
        )

        # -------------------------------------------------------------
        # Validate before starting expensive computation.
        # -------------------------------------------------------------

        validate_schema(
            con,
            rib_files[0],
            {
                "prefix",
                "as_path",
                "origin",
                "collector",
                "peer_asn",
                "peer_ip",
            },
            "RIB",
        )

        validate_schema(
            con,
            poison_files[0],
            {
                "prefix",
                "origin",
                "cleaned_aspath",
                "collector",
                "peer_asn",
                "peer_ip",
                "rs_path",
                "introducing_as",
                "appearing_ases",
            },
            "Non-cons",
        )

        create_month_views(
            con,
            rib_files,
            poison_files,
        )

        validate_view_schema(
            con,
            "rib",
            {
                "prefix", "as_path", "origin",
                "collector", "peer_asn", "peer_ip",
            },
            "RIB",
        )
        validate_view_schema(
            con,
            "noncons",
            {
                "prefix", "origin", "cleaned_aspath",
                "collector", "peer_asn", "peer_ip",
                "rs_path", "introducing_as", "appearing_ases",
            },
            "Non-cons",
        )

        # -------------------------------------------------------------
        # Observation populations + row counts.
        # -------------------------------------------------------------

        rib_basic = (
            compute_basic_stats(
                con,
                "rib",
                "RIB",
            )
        )

        rs_basic = (
            compute_basic_stats(
                con,
                "noncons",
                "Non-cons",
            )
        )

        print(
            "\nINPUT COUNTS"
        )

        print(
            "-" * 72
        )

        print(
            f"RIB rows      : "
            f"{rib_basic['rows']:,}"
        )

        print(
            f"Non-cons rows : "
            f"{rs_basic['rows']:,}"
        )

        # -------------------------------------------------------------
        # Prefix + AS path.
        # -------------------------------------------------------------

        # The RIB contains billions of observations. Do not run a
        # month-wide composite DISTINCT in memory. Build one bounded,
        # disk-backed staging relation and compute exact PP/POP route
        # counts bucket by bucket.
        rib_stage = stage_rib_for_exact_dedup(
            con,
            rib_files,
            work_dir,
        )

        pp_rib, pop_rib = (
            compute_rib_pp_pop_stats_from_stage(
                con,
                rib_stage,
                checkpoint_dir,
            )
        )

        # Non-cons is several orders of magnitude smaller and is safe to
        # compute directly with the original exact definitions.
        pp_rs = (
            compute_pp_stats(
                con,
                "noncons",
                "cleaned_aspath",
                "Non-cons",
            )
        )

        # -------------------------------------------------------------
        # Prefix + origin + AS path.
        # -------------------------------------------------------------

        pop_rs = (
            compute_pop_stats(
                con,
                "noncons",
                "cleaned_aspath",
                "Non-cons",
            )
        )

        # -------------------------------------------------------------
        # RS-specific.
        # -------------------------------------------------------------

        rs_specific = (
            compute_rs_specific_stats(
                con
            )
        )

        # -------------------------------------------------------------
        # Validate before writing final results.
        # -------------------------------------------------------------

        validate_results(
            rib_basic,
            rs_basic,
            pp_rib,
            pp_rs,
            pop_rib,
            pop_rs,
            rs_specific,
        )

        # -------------------------------------------------------------
        # Print.
        # -------------------------------------------------------------

        print_observation_stats(
            rib_basic,
            rs_basic,
        )

        print_dedup_table(
            (
                "DEDUPLICATION: "
                "PREFIX + AS PATH"
            ),
            pp_rib,
            pp_rs,
            rs_specific,
        )

        print_dedup_table(
            (
                "DEDUPLICATION: "
                "PREFIX + ORIGIN + AS PATH"
            ),
            pop_rib,
            pop_rs,
            rs_specific,
        )

        # -------------------------------------------------------------
        # Save.
        # -------------------------------------------------------------

        save_observation_csv(
            month,
            out_dir,
            rib_basic,
            rs_basic,
        )

        save_comparison_csv(
            month,
            out_dir,
            pp_rib,
            pp_rs,
            pop_rib,
            pop_rs,
            rs_specific,
        )

        save_summary_csv(
            month,
            out_dir,
            rib_basic,
            rs_basic,
            pp_rib,
            pp_rs,
            pop_rib,
            pop_rs,
            rs_specific,
        )

    finally:
        # -------------------------------------------------------------
        # Guaranteed cleanup of DuckDB spill state only.
        # exact_rib_work is deliberately retained for crash-safe resume.
        # -------------------------------------------------------------

        if con is not None:
            try:
                con.close()
            except Exception:
                pass

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    elapsed = (
        time.time()
        - month_start
    )

    free_after = disk_free_gb(
        BASE_ROOT
    )

    print(
        "\n"
        + "#" * 72
    )

    print(
        f"COMPLETED {month}"
    )

    print(
        f"Runtime   : "
        f"{elapsed_string(elapsed)}"
    )

    print(
        f"Free disk : "
        f"{free_after:.2f} GB"
    )

    print(
        "#" * 72
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    overall_start = time.time()

    memory_limit = (
        choose_memory_limit()
    )

    threads = (
        choose_threads()
    )

    print(
        "=" * 72
    )

    print(
        "MONTHLY RIB / NON-CONS STATISTICS"
    )

    print(
        "=" * 72
    )

    print(
        "Implementation : DuckDB direct Parquet"
    )

    print(
        "TXT files      : NONE"
    )

    print(
        "Distinct counts: EXACT"
    )

    print(
        "Months         : "
        + ", ".join(MONTHS)
    )

    print(
        f"Memory limit   : "
        f"{memory_limit}"
    )

    print(
        f"Threads        : "
        f"{threads}"
    )

    print(
        f"Disk reserve   : "
        f"{MIN_FREE_DISK_GB} GB"
    )

    print(
        "=" * 72
    )

    completed = []

    for month in MONTHS:
        process_month(
            month,
            memory_limit,
            threads,
        )

        completed.append(
            month
        )

    elapsed = (
        time.time()
        - overall_start
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL MONTHS COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        "Completed : "
        + ", ".join(completed)
    )

    print(
        f"Runtime   : "
        f"{elapsed_string(elapsed)}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
