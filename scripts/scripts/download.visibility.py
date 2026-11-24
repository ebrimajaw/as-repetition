#!/usr/bin/env python3

from pathlib import Path
import calendar
import csv
import logging
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

MONTHS = [
    (2024, 2),   # baseline
    (2024, 10),  # spike
]

RIB_TIME = "1600"
RRC_IDS = range(27)

BASE_URL = "https://data.ris.ripe.net"
BASE_DIR = Path("/data/jaw/rs.visibility.analysis")
MRT_DIR = BASE_DIR / "mrt"
PARQUET_DIR = BASE_DIR / "parquet"

BATCH_SIZE = 500_000
ZSTD_LEVEL = 10

MRT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parquet schema
# ============================================================

ARROW_SCHEMA = pa.schema([
    pa.field("date", pa.string()),
    pa.field("rib_timestamp", pa.int64()),
    pa.field("collector", pa.string()),
    pa.field("peer_asn", pa.int64()),
    pa.field("peer_ip", pa.string()),
    pa.field("prefix", pa.string()),
    pa.field("next_hop", pa.string()),
    pa.field("as_path", pa.string()),
    pa.field("origin", pa.string()),
    pa.field("communities", pa.string()),
])


# ============================================================
# Helpers
# ============================================================

def rrc_name(rrc_id):
    return f"rrc{rrc_id:02d}"


def build_url(rrc, date):
    year_month = date[:7].replace("-", ".")
    compact_date = date.replace("-", "")
    filename = f"bview.{compact_date}.{RIB_TIME}.gz"
    url = f"{BASE_URL}/{rrc}/{year_month}/{filename}"
    return url, filename


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f} sec"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} hr"


def format_size(num_bytes):
    mib = num_bytes / (1024 ** 2)
    if mib < 1024:
        return f"{mib:,.1f} MiB"
    return f"{num_bytes / (1024 ** 3):,.2f} GiB"


# ============================================================
# Logging
# ============================================================

def setup_logger(month_dir):
    log_file = month_dir / "_ingestion.log"

    logger = logging.getLogger(f"ris-{month_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)

    return logger, log_file


# ============================================================
# CSV statistics
# ============================================================

STATS_FIELDS = [
    "date",
    "rib_time",
    "collector",
    "status",
    "routes",
    "row_groups",
    "mrt_mib",
    "parquet_mib",
    "download_seconds",
    "download_rate_mib_s",
    "parse_seconds",
    "skipped_lines",
    "error",
]


def initialize_stats_csv(stats_file):
    if stats_file.exists():
        return

    with open(stats_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=STATS_FIELDS,
        )
        writer.writeheader()


def append_stats(
    stats_file,
    date,
    rrc,
    status,
    routes=0,
    row_groups=0,
    mrt_bytes=0,
    parquet_bytes=0,
    download_seconds=0.0,
    parse_seconds=0.0,
    skipped_lines=0,
    error="",
):
    if download_seconds > 0 and mrt_bytes > 0:
        download_rate = (
            mrt_bytes
            / (1024 ** 2)
            / download_seconds
        )
    else:
        download_rate = 0.0

    row = {
        "date": date,
        "rib_time": RIB_TIME,
        "collector": rrc,
        "status": status,
        "routes": routes,
        "row_groups": row_groups,
        "mrt_mib": round(mrt_bytes / (1024 ** 2), 2),
        "parquet_mib": round(parquet_bytes / (1024 ** 2), 2),
        "download_seconds": round(download_seconds, 2),
        "download_rate_mib_s": round(download_rate, 2),
        "parse_seconds": round(parse_seconds, 2),
        "skipped_lines": skipped_lines,
        "error": error,
    }

    with open(stats_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=STATS_FIELDS,
        )
        writer.writerow(row)
        handle.flush()


# ============================================================
# Download
# ============================================================

def download_file(url, destination):
    """
    Download one RIB with wget.

    Returns:
        status: success | unavailable | failed
        elapsed_seconds
        size_bytes
    """

    start = time.perf_counter()

    cmd = [
        "wget",
        "--progress=bar:force",
        "--timeout=60",
        "--tries=3",
        "-O",
        str(destination),
        url,
    ]

    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        if not destination.exists():
            return "failed", elapsed, 0

        size_bytes = destination.stat().st_size

        if size_bytes == 0:
            destination.unlink(missing_ok=True)
            return "failed", elapsed, 0

        return "success", elapsed, size_bytes

    # wget exit code 8 = server returned an error response.
    # For the deterministic RIPE archive URL, classify it
    # as unavailable.
    if result.returncode == 8:
        destination.unlink(missing_ok=True)
        return "unavailable", elapsed, 0

    destination.unlink(missing_ok=True)
    return "failed", elapsed, 0


# ============================================================
# Parquet batching
# ============================================================

def new_batch():
    return {
        "date": [],
        "rib_timestamp": [],
        "collector": [],
        "peer_asn": [],
        "peer_ip": [],
        "prefix": [],
        "next_hop": [],
        "as_path": [],
        "origin": [],
        "communities": [],
    }


def flush_batch(writer, batch):
    if not batch["date"]:
        return

    table = pa.Table.from_pydict(
        batch,
        schema=ARROW_SCHEMA,
    )

    writer.write_table(table)

    for values in batch.values():
        values.clear()


# ============================================================
# Parse RIB
# ============================================================

def parse_rib_to_parquet(date, rrc, mrt_file, output_file):
    """
    Stream BGPReader output into a Zstd-compressed Parquet file.

    BGPReader 2.3.0 fields used:
        0  record type
        1  element type
        2  timestamp
        7  peer ASN
        8  peer IP
        9  prefix
        10 next hop
        11 AS path
        12 origin
        13 communities
    """

    temp_file = Path(str(output_file) + ".tmp")
    stderr_file = Path(
        str(output_file) + ".bgpreader.stderr.tmp"
    )

    temp_file.unlink(missing_ok=True)
    stderr_file.unlink(missing_ok=True)

    cmd = [
        "bgpreader",
        "-d",
        "singlefile",
        "-o",
        f"rib-file={mrt_file}",
    ]

    route_count = 0
    skipped_lines = 0
    pending_progress = 0

    batch = new_batch()
    writer = None
    process = None

    parse_start = time.perf_counter()

    progress = tqdm(
        desc=f"Parse {date} {rrc}",
        unit="routes",
        unit_scale=True,
        dynamic_ncols=True,
    )

    try:
        with open(
            stderr_file,
            "w",
            encoding="utf-8",
        ) as stderr_handle:

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout is None:
                raise RuntimeError(
                    "Could not read bgpreader stdout."
                )

            for line in process.stdout:
                fields = line.rstrip("\n").split("|")

                if len(fields) < 14:
                    skipped_lines += 1
                    continue

                # Metadata is R|B|...
                # Actual RIB route elements are R|R|...
                if fields[0] != "R" or fields[1] != "R":
                    continue

                try:
                    rib_timestamp = int(float(fields[2]))
                    peer_asn = int(fields[7])
                except (ValueError, TypeError):
                    skipped_lines += 1
                    continue

                batch["date"].append(date)
                batch["rib_timestamp"].append(rib_timestamp)
                batch["collector"].append(rrc)
                batch["peer_asn"].append(peer_asn)
                batch["peer_ip"].append(fields[8])
                batch["prefix"].append(fields[9])
                batch["next_hop"].append(fields[10])
                batch["as_path"].append(fields[11])
                batch["origin"].append(fields[12])
                batch["communities"].append(fields[13])

                route_count += 1
                pending_progress += 1

                if len(batch["date"]) >= BATCH_SIZE:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_file,
                            ARROW_SCHEMA,
                            compression="zstd",
                            compression_level=ZSTD_LEVEL,
                            use_dictionary=["date", "collector"],
                            write_statistics=True,
                        )

                    flush_batch(writer, batch)

                    progress.update(pending_progress)
                    pending_progress = 0

            # Final incomplete batch.
            if batch["date"]:
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_file,
                        ARROW_SCHEMA,
                        compression="zstd",
                        compression_level=ZSTD_LEVEL,
                        use_dictionary=["date", "collector"],
                        write_statistics=True,
                    )

                flush_batch(writer, batch)

            if pending_progress:
                progress.update(pending_progress)

            return_code = process.wait()

    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise

    finally:
        progress.close()

        if writer is not None:
            writer.close()

    parse_seconds = time.perf_counter() - parse_start

    stderr_text = ""

    if stderr_file.exists():
        stderr_text = stderr_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

    stderr_file.unlink(missing_ok=True)

    if return_code != 0:
        temp_file.unlink(missing_ok=True)

        raise RuntimeError(
            f"bgpreader failed for {rrc} "
            f"with return code {return_code}:\n"
            f"{stderr_text}"
        )

    if route_count == 0:
        temp_file.unlink(missing_ok=True)

        raise RuntimeError(
            f"No RIB route elements were parsed for {rrc}."
        )

    # Check temporary Parquet before exposing it as complete.
    parquet = pq.ParquetFile(temp_file)

    if parquet.metadata.num_rows != route_count:
        parquet_rows = parquet.metadata.num_rows
        temp_file.unlink(missing_ok=True)

        raise RuntimeError(
            f"Row-count mismatch for {rrc}: "
            f"parsed {route_count:,}, "
            f"Parquet contains {parquet_rows:,}."
        )

    temp_file.replace(output_file)

    return route_count, parse_seconds, skipped_lines


# ============================================================
# Validate Parquet
# ============================================================

def validate_parquet(
    output_file,
    expected_rrc,
    expected_date,
    expected_routes=None,
):
    """
    Lightweight validation:
      - file opens as Parquet
      - row count is non-zero
      - schema matches
      - first and last row groups contain expected collector
      - first and last row groups contain expected date
    """

    parquet = pq.ParquetFile(output_file)

    routes = parquet.metadata.num_rows
    row_groups = parquet.metadata.num_row_groups

    if routes <= 0:
        raise RuntimeError(
            "Parquet contains zero routes."
        )

    if row_groups <= 0:
        raise RuntimeError(
            "Parquet contains zero row groups."
        )

    if expected_routes is not None and routes != expected_routes:
        raise RuntimeError(
            f"Expected {expected_routes:,} rows, "
            f"found {routes:,}."
        )

    if parquet.schema_arrow != ARROW_SCHEMA:
        raise RuntimeError(
            f"Unexpected Parquet schema:\n"
            f"{parquet.schema_arrow}"
        )

    check_groups = {0, row_groups - 1}

    for group_index in check_groups:
        table = parquet.read_row_group(
            group_index,
            columns=["date", "collector"],
        )

        collectors = set(
            table.column("collector").to_pylist()
        )

        dates = set(
            table.column("date").to_pylist()
        )

        if collectors != {expected_rrc}:
            raise RuntimeError(
                f"Unexpected collector values in row group "
                f"{group_index}: {collectors}"
            )

        if dates != {expected_date}:
            raise RuntimeError(
                f"Unexpected date values in row group "
                f"{group_index}: {dates}"
            )

    return {
        "routes": routes,
        "row_groups": row_groups,
    }


# ============================================================
# Process one month
# ============================================================

def process_month(year, month):
    month_label = f"{year:04d}.{month:02d}"
    month_dir = PARQUET_DIR / month_label

    month_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger, log_file = setup_logger(month_dir)

    stats_file = (
        month_dir
        / "_ingestion_stats.csv"
    )

    initialize_stats_csv(stats_file)

    days_in_month = calendar.monthrange(
        year,
        month,
    )[1]

    planned = days_in_month * len(RRC_IDS)

    successful = 0
    skipped = 0
    unavailable = 0
    failed = 0

    total_download_bytes = 0
    total_download_seconds = 0.0
    total_parse_seconds = 0.0

    month_start = time.perf_counter()

    print("\n" + "=" * 80)
    print(f"RIPE RIS VISIBILITY - {month_label}")
    print("=" * 80)
    print(f"RIB snapshot     : {RIB_TIME} UTC")
    print("Collectors       : rrc00-rrc26")
    print(f"Days             : {days_in_month}")
    print(f"Collector-days   : {planned}")
    print(f"Output directory : {month_dir}")
    print(f"Statistics       : {stats_file}")
    print(f"Log              : {log_file}")
    print("=" * 80)

    logger.info(
        f"MONTH START | {month_label} | "
        f"RIB={RIB_TIME} UTC | "
        f"days={days_in_month} | "
        f"collector-days={planned}"
    )

    for day in range(1, days_in_month + 1):
        date = f"{year:04d}-{month:02d}-{day:02d}"
        file_date = date.replace("-", ".")

        print("\n" + "=" * 80)
        print(f"DATE: {date}")
        print("=" * 80)

        logger.info(f"DATE START | {date}")

        for rrc_id in RRC_IDS:
            rrc = rrc_name(rrc_id)
            url, filename = build_url(rrc, date)

            mrt_file = MRT_DIR / f"{rrc}.{filename}"

            output_file = (
                month_dir
                / f"{rrc}.{file_date}.parquet"
            )

            print("\n" + "-" * 80)
            print(f"[{date} | {rrc}]")
            print(f"    URL: {url}")

            download_seconds = 0.0
            parse_seconds = 0.0
            mrt_size_bytes = 0
            skipped_lines = 0

            # ------------------------------------------------
            # Restart-safe existing Parquet
            # ------------------------------------------------

            if output_file.exists():
                print("    Existing Parquet found. Validating...")

                try:
                    stats = validate_parquet(
                        output_file,
                        expected_rrc=rrc,
                        expected_date=date,
                    )

                    parquet_size_bytes = (
                        output_file.stat().st_size
                    )

                    print(
                        "    Valid existing Parquet. Skipping."
                    )
                    print(
                        f"    Routes        : "
                        f"{stats['routes']:,}"
                    )
                    print(
                        f"    Parquet size  : "
                        f"{format_size(parquet_size_bytes)}"
                    )

                    logger.info(
                        f"{date} | {rrc} | "
                        f"SKIPPED_EXISTING | "
                        f"routes={stats['routes']} | "
                        f"parquet="
                        f"{format_size(parquet_size_bytes)}"
                    )

                    append_stats(
                        stats_file,
                        date,
                        rrc,
                        "skipped_existing",
                        routes=stats["routes"],
                        row_groups=stats["row_groups"],
                        parquet_bytes=parquet_size_bytes,
                    )

                    # If a stale MRT exists but the final
                    # Parquet is valid, it is unnecessary.
                    mrt_file.unlink(missing_ok=True)

                    skipped += 1
                    continue

                except Exception as exc:
                    print(
                        f"    Existing Parquet invalid: {exc}"
                    )
                    print("    Removing and rebuilding.")

                    logger.warning(
                        f"{date} | {rrc} | "
                        f"INVALID_EXISTING_PARQUET | {exc}"
                    )

                    output_file.unlink(missing_ok=True)

            # ------------------------------------------------
            # Download or reuse MRT
            # ------------------------------------------------

            if mrt_file.exists():
                mrt_size_bytes = mrt_file.stat().st_size

                if mrt_size_bytes > 0:
                    print(
                        f"    Existing MRT  : "
                        f"{format_size(mrt_size_bytes)}"
                    )

                    logger.info(
                        f"{date} | {rrc} | "
                        f"REUSE_MRT | "
                        f"size={format_size(mrt_size_bytes)}"
                    )

                else:
                    mrt_file.unlink(missing_ok=True)

            if not mrt_file.exists():
                print("    Downloading MRT with wget...")

                logger.info(
                    f"{date} | {rrc} | DOWNLOAD_START"
                )

                (
                    status,
                    download_seconds,
                    mrt_size_bytes,
                ) = download_file(
                    url,
                    mrt_file,
                )

                if status == "unavailable":
                    print("    RIB unavailable.")

                    logger.warning(
                        f"{date} | {rrc} | "
                        f"UNAVAILABLE | "
                        f"download_time={download_seconds:.2f}s"
                    )

                    append_stats(
                        stats_file,
                        date,
                        rrc,
                        "unavailable",
                        download_seconds=download_seconds,
                        error="RIB unavailable",
                    )

                    unavailable += 1
                    continue

                if status == "failed":
                    print("    Download failed.")

                    logger.error(
                        f"{date} | {rrc} | "
                        f"DOWNLOAD_FAILED | "
                        f"download_time={download_seconds:.2f}s"
                    )

                    append_stats(
                        stats_file,
                        date,
                        rrc,
                        "download_failed",
                        download_seconds=download_seconds,
                        error="wget failed",
                    )

                    failed += 1
                    continue

                total_download_seconds += download_seconds
                total_download_bytes += mrt_size_bytes

                print(
                    f"    Download size : "
                    f"{format_size(mrt_size_bytes)}"
                )

                print(
                    f"    Download time : "
                    f"{format_duration(download_seconds)}"
                )

                if download_seconds > 0:
                    rate = (
                        mrt_size_bytes
                        / (1024 ** 2)
                        / download_seconds
                    )

                    print(
                        f"    Average rate  : "
                        f"{rate:,.2f} MiB/s"
                    )

                logger.info(
                    f"{date} | {rrc} | "
                    f"DOWNLOAD_SUCCESS | "
                    f"size={format_size(mrt_size_bytes)} | "
                    f"time={download_seconds:.2f}s"
                )

            # ------------------------------------------------
            # Parse
            # ------------------------------------------------

            print("    Parsing with bgpreader...")

            logger.info(
                f"{date} | {rrc} | PARSE_START"
            )

            parse_start = time.perf_counter()

            try:
                (
                    route_count,
                    parse_seconds,
                    skipped_lines,
                ) = parse_rib_to_parquet(
                    date,
                    rrc,
                    mrt_file,
                    output_file,
                )

            except Exception as exc:
                parse_seconds = (
                    time.perf_counter()
                    - parse_start
                )

                print(f"    Parsing failed: {exc}")

                logger.error(
                    f"{date} | {rrc} | "
                    f"PARSE_FAILED | "
                    f"time={parse_seconds:.2f}s | "
                    f"error={exc}"
                )

                append_stats(
                    stats_file,
                    date,
                    rrc,
                    "parse_failed",
                    mrt_bytes=mrt_size_bytes,
                    download_seconds=download_seconds,
                    parse_seconds=parse_seconds,
                    error=str(exc),
                )

                # Keep MRT for retry.
                failed += 1
                continue

            total_parse_seconds += parse_seconds

            print(
                f"    Parsed routes : {route_count:,}"
            )
            print(
                f"    Parse time    : "
                f"{format_duration(parse_seconds)}"
            )
            print(
                f"    Skipped lines : {skipped_lines:,}"
            )

            logger.info(
                f"{date} | {rrc} | "
                f"PARSE_SUCCESS | "
                f"routes={route_count} | "
                f"skipped_lines={skipped_lines} | "
                f"time={parse_seconds:.2f}s"
            )

            # ------------------------------------------------
            # Validte
            # ------------------------------------------------

            print("    Validating Parquet...")

            try:
                stats = validate_parquet(
                    output_file,
                    expected_rrc=rrc,
                    expected_date=date,
                    expected_routes=route_count,
                )

            except Exception as exc:
                print(f"    Validation failed: {exc}")

                logger.error(
                    f"{date} | {rrc} | "
                    f"VALIDATION_FAILED | error={exc}"
                )

                output_file.unlink(missing_ok=True)

                append_stats(
                    stats_file,
                    date,
                    rrc,
                    "validation_failed",
                    routes=route_count,
                    mrt_bytes=mrt_size_bytes,
                    download_seconds=download_seconds,
                    parse_seconds=parse_seconds,
                    skipped_lines=skipped_lines,
                    error=str(exc),
                )

                # Keep MRT for retry.
                failed += 1
                continue

            parquet_size_bytes = (
                output_file.stat().st_size
            )

            print(
                f"    Routes        : "
                f"{stats['routes']:,}"
            )
            print(
                f"    Row groups    : "
                f"{stats['row_groups']:,}"
            )
            print(
                f"    Parquet size  : "
                f"{format_size(parquet_size_bytes)}"
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            mrt_file.unlink(missing_ok=True)

            print(
                "    MRT removed after successful "
                "Parquet validation."
            )

            logger.info(
                f"{date} | {rrc} | SUCCESS | "
                f"routes={stats['routes']} | "
                f"row_groups={stats['row_groups']} | "
                f"mrt={format_size(mrt_size_bytes)} | "
                f"parquet={format_size(parquet_size_bytes)} | "
                f"download={download_seconds:.2f}s | "
                f"parse={parse_seconds:.2f}s"
            )

            append_stats(
                stats_file,
                date,
                rrc,
                "success",
                routes=stats["routes"],
                row_groups=stats["row_groups"],
                mrt_bytes=mrt_size_bytes,
                parquet_bytes=parquet_size_bytes,
                download_seconds=download_seconds,
                parse_seconds=parse_seconds,
                skipped_lines=skipped_lines,
            )

            successful += 1

        logger.info(f"DATE COMPLETE | {date}")

    # ========================================================
    # Monthly summary from files actually on disk
    # ========================================================

    parquet_files = sorted(
        month_dir.glob(
            "rrc*.????.??.??.parquet"
        )
    )

    total_routes = 0
    total_parquet_bytes = 0

    for path in parquet_files:
        try:
            parquet = pq.ParquetFile(path)
            total_routes += parquet.metadata.num_rows
            total_parquet_bytes += path.stat().st_size

        except Exception as exc:
            logger.error(
                f"MONTH_END_CHECK_FAILED | "
                f"{path.name} | {exc}"
            )

    month_seconds = (
        time.perf_counter()
        - month_start
    )

    print("\n" + "=" * 80)
    print(f"MONTH COMPLETE - {month_label}")
    print("=" * 80)
    print(f"Collector-days planned : {planned}")
    print(f"Successful this run    : {successful}")
    print(f"Skipped existing       : {skipped}")
    print(f"Unavailable            : {unavailable}")
    print(f"Failed                 : {failed}")
    print(f"Parquet files on disk  : {len(parquet_files)}")
    print(f"Total route rows       : {total_routes:,}")
    print(
        f"Total Parquet size     : "
        f"{format_size(total_parquet_bytes)}"
    )

    print("\nTIMING")
    print(
        f"Download time          : "
        f"{format_duration(total_download_seconds)}"
    )
    print(
        f"Parsing time           : "
        f"{format_duration(total_parse_seconds)}"
    )
    print(
        f"Total runtime          : "
        f"{format_duration(month_seconds)}"
    )
    print(
        f"Downloaded this run    : "
        f"{format_size(total_download_bytes)}"
    )

    if total_download_seconds > 0:
        rate = (
            total_download_bytes
            / (1024 ** 2)
            / total_download_seconds
        )

        print(
            f"Average download rate  : "
            f"{rate:,.2f} MiB/s"
        )

    print(f"\nStatistics: {stats_file}")
    print(f"Log       : {log_file}")
    print("=" * 80)

    logger.info(
        f"MONTH COMPLETE | {month_label} | "
        f"planned={planned} | "
        f"success={successful} | "
        f"skipped={skipped} | "
        f"unavailable={unavailable} | "
        f"failed={failed} | "
        f"parquets={len(parquet_files)} | "
        f"routes={total_routes} | "
        f"parquet_size={format_size(total_parquet_bytes)} | "
        f"download_time={format_duration(total_download_seconds)} | "
        f"parse_time={format_duration(total_parse_seconds)} | "
        f"runtime={format_duration(month_seconds)}"
    )

    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


# ============================================================
# Main
# ============================================================

def main():
    if shutil.which("wget") is None:
        raise SystemExit(
            "ERROR: wget was not found in PATH."
        )

    if shutil.which("bgpreader") is None:
        raise SystemExit(
            "ERROR: bgpreader was not found in PATH."
        )

    total_start = time.perf_counter()

    print("\n" + "=" * 80)
    print("RIPE RIS VISIBILITY - PRODUCTION DOWNLOAD")
    print("=" * 80)
    print("Months           : 2024.02, 2024.10")
    print(f"RIB snapshot     : {RIB_TIME} UTC")
    print("Collectors       : rrc00-rrc26")
    print(f"Batch size       : {BATCH_SIZE:,} routes")
    print(f"MRT directory    : {MRT_DIR}")
    print(f"Parquet directory: {PARQUET_DIR}")
    print("=" * 80)

    for year, month in MONTHS:
        process_month(year, month)

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 80)
    print("ALL DOWNLOADS COMPLETE")
    print("=" * 80)
    print(
        f"Total runtime: "
        f"{format_duration(total_seconds)}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
