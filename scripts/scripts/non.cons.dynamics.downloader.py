#!/usr/bin/env python3

from pathlib import Path
from datetime import date, timedelta
import csv
import logging
import shutil
import subprocess
import time
import traceback

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

START_DATE = date(2021, 12, 31)
END_DATE = date(2022, 1, 2)

# 23 active RIPE RIS collectors for this period.
RRC_IDS = [
    0, 1,
    3, 4, 5, 6, 7,
    10, 11, 12, 13, 14, 15, 16,
    18, 19, 20, 21, 22, 23, 24, 25, 26,
]

BASE_URL = "https://data.ris.ripe.net"

BASE_DIR = Path(
    "/data/jaw/pam2026/updates/2021-12-31_2022-01-02"
)

MRT_DIR = BASE_DIR / "mrt"
PARQUET_DIR = BASE_DIR / "parquet"

BATCH_SIZE = 500_000
ZSTD_LEVEL = 10

# RIPE RIS UPDATE files are generated every 5 minutes.
UPDATE_INTERVAL_MINUTES = 5

MRT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARQUET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parquet schema
# ============================================================

ARROW_SCHEMA = pa.schema([
    pa.field(
        "timestamp",
        pa.int64(),
    ),

    pa.field(
        "collector",
        pa.string(),
    ),

    pa.field(
        "peer_asn",
        pa.int64(),
    ),

    pa.field(
        "peer_ip",
        pa.string(),
    ),

    pa.field(
        "element_type",
        pa.string(),
    ),

    pa.field(
        "prefix",
        pa.string(),
    ),

    pa.field(
        "next_hop",
        pa.string(),
    ),

    pa.field(
        "as_path",
        pa.string(),
    ),

    pa.field(
        "origin",
        pa.string(),
    ),

    pa.field(
        "communities",
        pa.string(),
    ),
])


# ============================================================
# Helpers
# ============================================================

def rrc_name(rrc_id):

    return f"rrc{rrc_id:02d}"


def iter_dates(start_date, end_date):

    current = start_date

    while current <= end_date:

        yield current

        current += timedelta(
            days=1
        )


def update_times():

    """
    Return all 288 five-minute UPDATE timestamps
    within one UTC day:

        0000, 0005, 0010, ..., 2355
    """

    times = []

    for hour in range(24):

        for minute in range(
            0,
            60,
            UPDATE_INTERVAL_MINUTES,
        ):

            times.append(
                f"{hour:02d}{minute:02d}"
            )

    return times


UPDATE_TIMES = update_times()


def build_url(
    rrc,
    current_date,
    update_time,
):

    year_month = (
        current_date
        .strftime("%Y.%m")
    )

    compact_date = (
        current_date
        .strftime("%Y%m%d")
    )

    filename = (
        f"updates."
        f"{compact_date}."
        f"{update_time}.gz"
    )

    url = (
        f"{BASE_URL}/"
        f"{rrc}/"
        f"{year_month}/"
        f"{filename}"
    )

    return url, filename


def format_duration(seconds):

    if seconds < 60:

        return (
            f"{seconds:.1f} sec"
        )

    if seconds < 3600:

        return (
            f"{seconds / 60:.2f} min"
        )

    return (
        f"{seconds / 3600:.2f} hr"
    )


def format_size(num_bytes):

    mib = (
        num_bytes
        / (1024 ** 2)
    )

    if mib < 1024:

        return (
            f"{mib:,.1f} MiB"
        )

    return (
        f"{num_bytes / (1024 ** 3):,.2f} GiB"
    )


# ============================================================
# Logging
# ============================================================

def setup_logger():

    log_file = (
        BASE_DIR
        / "_ingestion.log"
    )

    logger = logging.getLogger(
        "ris-update-ingestion"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    for handler in logger.handlers[:]:

        handler.close()

        logger.removeHandler(
            handler
        )

    handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s",
            datefmt=(
                "%Y-%m-%d "
                "%H:%M:%S"
            ),
        )
    )

    logger.addHandler(
        handler
    )

    return logger, log_file


# ============================================================
# Statistics
# ============================================================

STATS_FIELDS = [
    "date",
    "collector",
    "status",

    "expected_update_files",
    "downloaded_update_files",
    "reused_update_files",
    "unavailable_update_files",

    "elements",
    "announcements",
    "withdrawals",

    "row_groups",

    "mrt_mib",
    "parquet_mib",

    "download_seconds",
    "download_rate_mib_s",
    "parse_seconds",

    "skipped_lines",
    "error",
]


def initialize_stats_csv(
    stats_file,
):

    if stats_file.exists():

        return

    with open(
        stats_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=STATS_FIELDS,
        )

        writer.writeheader()


def append_stats(
    stats_file,
    current_date,
    rrc,
    status,

    expected_update_files=0,
    downloaded_update_files=0,
    reused_update_files=0,
    unavailable_update_files=0,

    elements=0,
    announcements=0,
    withdrawals=0,

    row_groups=0,

    mrt_bytes=0,
    parquet_bytes=0,

    download_seconds=0.0,
    parse_seconds=0.0,

    skipped_lines=0,
    error="",
):

    if (
        download_seconds > 0
        and mrt_bytes > 0
    ):

        download_rate = (
            mrt_bytes
            / (1024 ** 2)
            / download_seconds
        )

    else:

        download_rate = 0.0

    row = {
        "date":
            current_date,

        "collector":
            rrc,

        "status":
            status,

        "expected_update_files":
            expected_update_files,

        "downloaded_update_files":
            downloaded_update_files,

        "reused_update_files":
            reused_update_files,

        "unavailable_update_files":
            unavailable_update_files,

        "elements":
            elements,

        "announcements":
            announcements,

        "withdrawals":
            withdrawals,

        "row_groups":
            row_groups,

        "mrt_mib":
            round(
                mrt_bytes
                / (1024 ** 2),
                2,
            ),

        "parquet_mib":
            round(
                parquet_bytes
                / (1024 ** 2),
                2,
            ),

        "download_seconds":
            round(
                download_seconds,
                2,
            ),

        "download_rate_mib_s":
            round(
                download_rate,
                2,
            ),

        "parse_seconds":
            round(
                parse_seconds,
                2,
            ),

        "skipped_lines":
            skipped_lines,

        "error":
            error,
    }

    with open(
        stats_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=STATS_FIELDS,
        )

        writer.writerow(
            row
        )

        handle.flush()


# ============================================================
# Download
# ============================================================

def download_file(
    url,
    destination,
):

    """
    Download one RIPE RIS UPDATE MRT file.

    Returns:

        status:
            success
            unavailable
            failed

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

    result = subprocess.run(
        cmd
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if result.returncode == 0:

        if not destination.exists():

            return (
                "failed",
                elapsed,
                0,
            )

        size_bytes = (
            destination
            .stat()
            .st_size
        )

        if size_bytes == 0:

            destination.unlink(
                missing_ok=True
            )

            return (
                "failed",
                elapsed,
                0,
            )

        return (
            "success",
            elapsed,
            size_bytes,
        )

    # wget exit code 8 means that the server
    # returned an error response.
    if result.returncode == 8:

        destination.unlink(
            missing_ok=True
        )

        return (
            "unavailable",
            elapsed,
            0,
        )

    destination.unlink(
        missing_ok=True
    )

    return (
        "failed",
        elapsed,
        0,
    )


# ============================================================
# Parquet batching
# ============================================================

def new_batch():

    return {
        "timestamp": [],
        "collector": [],
        "peer_asn": [],
        "peer_ip": [],
        "element_type": [],
        "prefix": [],
        "next_hop": [],
        "as_path": [],
        "origin": [],
        "communities": [],
    }


def flush_batch(
    writer,
    batch,
):

    if not batch["timestamp"]:

        return

    table = pa.Table.from_pydict(
        batch,
        schema=ARROW_SCHEMA,
    )

    writer.write_table(
        table
    )

    for values in batch.values():

        values.clear()


# ============================================================
# Parse one UPDATE MRT into an existing Parquet writer
# ============================================================

def parse_update_file(
    rrc,
    mrt_file,
    writer,
    batch,
    progress,
):

    """
    Parse one RIPE RIS UPDATE MRT file.

    BGPReader 2.3.0 fields:

        0   record type
        1   element type
        2   timestamp
        7   peer ASN
        8   peer IP
        9   prefix
        10  next hop
        11  AS path
        12  origin
        13  communities

    UPDATE element types retained:

        A = announcement
        W = withdrawal

    Returns:

        element_count
        announcement_count
        withdrawal_count
        skipped_lines
    """

    stderr_file = Path(
        str(mrt_file)
        + ".bgpreader.stderr.tmp"
    )

    stderr_file.unlink(
        missing_ok=True
    )

    cmd = [
        "bgpreader",
        "-d",
        "singlefile",
        "-o",
        f"upd-file={mrt_file}",
    ]

    element_count = 0
    announcement_count = 0
    withdrawal_count = 0
    skipped_lines = 0

    pending_progress = 0

    process = None

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
                    "Could not read "
                    "bgpreader stdout."
                )

            for line in process.stdout:

                fields = (
                    line
                    .rstrip("\n")
                    .split("|")
                )

                if len(fields) < 14:

                    skipped_lines += 1

                    continue

                element_type = (
                    fields[1]
                )

                # Keep only actual BGP announcement
                # and withdrawal elements.
                if element_type not in {
                    "A",
                    "W",
                }:

                    continue

                try:

                    timestamp = int(
                        float(
                            fields[2]
                        )
                    )

                    peer_asn = int(
                        fields[7]
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    skipped_lines += 1

                    continue

                prefix = (
                    fields[9]
                )

                if not prefix:

                    skipped_lines += 1

                    continue

                # ----------------------------------------
                # Announcement
                # ----------------------------------------

                if element_type == "A":

                    next_hop = (
                        fields[10]
                    )

                    as_path = (
                        fields[11]
                    )

                    origin = (
                        fields[12]
                    )

                    communities = (
                        fields[13]
                    )

                    announcement_count += 1

                # ----------------------------------------
                # Withdrawal
                # ----------------------------------------

                else:

                    # Withdrawals do not carry the
                    # announcement path attributes.
                    next_hop = ""
                    as_path = ""
                    origin = ""
                    communities = ""

                    withdrawal_count += 1

                batch[
                    "timestamp"
                ].append(
                    timestamp
                )

                batch[
                    "collector"
                ].append(
                    rrc
                )

                batch[
                    "peer_asn"
                ].append(
                    peer_asn
                )

                batch[
                    "peer_ip"
                ].append(
                    fields[8]
                )

                batch[
                    "element_type"
                ].append(
                    element_type
                )

                batch[
                    "prefix"
                ].append(
                    prefix
                )

                batch[
                    "next_hop"
                ].append(
                    next_hop
                )

                batch[
                    "as_path"
                ].append(
                    as_path
                )

                batch[
                    "origin"
                ].append(
                    origin
                )

                batch[
                    "communities"
                ].append(
                    communities
                )

                element_count += 1
                pending_progress += 1

                if (
                    len(
                        batch[
                            "timestamp"
                        ]
                    )
                    >= BATCH_SIZE
                ):

                    flush_batch(
                        writer,
                        batch,
                    )

                    progress.update(
                        pending_progress
                    )

                    pending_progress = 0

            return_code = (
                process.wait()
            )

        if pending_progress:

            progress.update(
                pending_progress
            )

    except Exception:

        if (
            process is not None
            and process.poll() is None
        ):

            process.kill()
            process.wait()

        raise

    stderr_text = ""

    if stderr_file.exists():

        stderr_text = (
            stderr_file
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

    stderr_file.unlink(
        missing_ok=True
    )

    if return_code != 0:

        raise RuntimeError(
            f"bgpreader failed for "
            f"{mrt_file.name} "
            f"with return code "
            f"{return_code}:\n"
            f"{stderr_text}"
        )

    return (
        element_count,
        announcement_count,
        withdrawal_count,
        skipped_lines,
    )


# ============================================================
# Validate collector-day Parquet
# ============================================================

def validate_parquet(
    output_file,
    expected_rrc,
    expected_elements=None,
):

    """
    Lightweight validation:

      - file opens as Parquet
      - row count is non-zero
      - schema matches
      - row groups exist
      - first and last row groups contain expected collector
      - element_type contains only A/W in checked groups
    """

    parquet = pq.ParquetFile(
        output_file
    )

    elements = (
        parquet
        .metadata
        .num_rows
    )

    row_groups = (
        parquet
        .metadata
        .num_row_groups
    )

    if elements <= 0:

        raise RuntimeError(
            "Parquet contains "
            "zero UPDATE elements."
        )

    if row_groups <= 0:

        raise RuntimeError(
            "Parquet contains "
            "zero row groups."
        )

    if (
        expected_elements is not None
        and elements
        != expected_elements
    ):

        raise RuntimeError(
            f"Expected "
            f"{expected_elements:,} rows, "
            f"found "
            f"{elements:,}."
        )

    if (
        parquet.schema_arrow
        != ARROW_SCHEMA
    ):

        raise RuntimeError(
            "Unexpected Parquet "
            f"schema:\n"
            f"{parquet.schema_arrow}"
        )

    check_groups = {
        0,
        row_groups - 1,
    }

    for group_index in check_groups:

        table = (
            parquet
            .read_row_group(
                group_index,
                columns=[
                    "collector",
                    "element_type",
                ],
            )
        )

        collectors = set(
            table
            .column(
                "collector"
            )
            .to_pylist()
        )

        if collectors != {
            expected_rrc
        }:

            raise RuntimeError(
                "Unexpected collector "
                "values in row group "
                f"{group_index}: "
                f"{collectors}"
            )

        element_types = set(
            table
            .column(
                "element_type"
            )
            .to_pylist()
        )

        invalid_types = (
            element_types
            - {"A", "W"}
        )

        if invalid_types:

            raise RuntimeError(
                "Unexpected element "
                "types in row group "
                f"{group_index}: "
                f"{invalid_types}"
            )

    return {
        "elements":
            elements,

        "row_groups":
            row_groups,
    }


# ============================================================
# Process one collector-day
# ============================================================

def process_collector_day(
    current_date,
    rrc,
    logger,
    stats_file,
):

    date_string = (
        current_date
        .strftime("%Y-%m-%d")
    )

    file_date = (
        current_date
        .strftime("%Y.%m.%d")
    )

    day_mrt_dir = (
        MRT_DIR
        / date_string
        / rrc
    )

    day_mrt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    day_parquet_dir = (
        PARQUET_DIR
        / date_string
    )

    day_parquet_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        day_parquet_dir
        / (
            f"{rrc}."
            f"{file_date}."
            f"parquet"
        )
    )

    temp_file = Path(
        str(output_file)
        + ".tmp"
    )

    print(
        "\n"
        + "-"
        * 80
    )

    print(
        f"[{date_string} | {rrc}]"
    )

    # --------------------------------------------------------
    # Restart-safe existing Parquet
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "    Existing Parquet "
            "found. Validating..."
        )

        try:

            stats = (
                validate_parquet(
                    output_file,
                    expected_rrc=rrc,
                )
            )

            parquet_bytes = (
                output_file
                .stat()
                .st_size
            )

            print(
                "    Valid existing "
                "Parquet. Skipping."
            )

            print(
                f"    Elements      : "
                f"{stats['elements']:,}"
            )

            print(
                f"    Parquet size  : "
                f"{format_size(parquet_bytes)}"
            )

            logger.info(
                f"{date_string} | "
                f"{rrc} | "
                f"SKIPPED_EXISTING | "
                f"elements="
                f"{stats['elements']} | "
                f"parquet="
                f"{format_size(parquet_bytes)}"
            )

            append_stats(
                stats_file,
                date_string,
                rrc,
                "skipped_existing",
                expected_update_files=(
                    len(
                        UPDATE_TIMES
                    )
                ),
                elements=(
                    stats[
                        "elements"
                    ]
                ),
                row_groups=(
                    stats[
                        "row_groups"
                    ]
                ),
                parquet_bytes=(
                    parquet_bytes
                ),
            )

            return {
                "status":
                    "skipped",

                "elements":
                    stats[
                        "elements"
                    ],

                "parquet_bytes":
                    parquet_bytes,

                "download_bytes":
                    0,

                "download_seconds":
                    0.0,

                "parse_seconds":
                    0.0,
            }

        except Exception as exc:

            print(
                "    Existing Parquet "
                f"invalid: {exc}"
            )

            print(
                "    Removing and "
                "rebuilding."
            )

            logger.warning(
                f"{date_string} | "
                f"{rrc} | "
                f"INVALID_EXISTING_PARQUET | "
                f"{exc}"
            )

            output_file.unlink(
                missing_ok=True
            )

    temp_file.unlink(
        missing_ok=True
    )

    # --------------------------------------------------------
    # Download all 288 UPDATE MRTs
    # --------------------------------------------------------

    expected_files = (
        len(
            UPDATE_TIMES
        )
    )

    downloaded_files = 0
    reused_files = 0
    unavailable_files = 0

    total_mrt_bytes = 0
    total_download_seconds = 0.0

    mrt_files = []

    print(
        f"    UPDATE files  : "
        f"{expected_files}"
    )

    logger.info(
        f"{date_string} | "
        f"{rrc} | "
        f"DOWNLOAD_START | "
        f"expected_files="
        f"{expected_files}"
    )

    for index, update_time in enumerate(
        UPDATE_TIMES,
        start=1,
    ):

        url, filename = build_url(
            rrc,
            current_date,
            update_time,
        )

        mrt_file = (
            day_mrt_dir
            / filename
        )

        # ----------------------------------------------------
        # Reuse previously downloaded MRT
        # ----------------------------------------------------

        if mrt_file.exists():

            size_bytes = (
                mrt_file
                .stat()
                .st_size
            )

            if size_bytes > 0:

                reused_files += 1

                total_mrt_bytes += (
                    size_bytes
                )

                mrt_files.append(
                    mrt_file
                )

                continue

            mrt_file.unlink(
                missing_ok=True
            )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        print(
            f"\r"
            f"    Downloading   : "
            f"{index:3d}/"
            f"{expected_files} "
            f"({update_time} UTC)",
            end="",
            flush=True,
        )

        (
            status,
            download_seconds,
            size_bytes,
        ) = download_file(
            url,
            mrt_file,
        )

        total_download_seconds += (
            download_seconds
        )

        if status == "success":

            downloaded_files += 1

            total_mrt_bytes += (
                size_bytes
            )

            mrt_files.append(
                mrt_file
            )

        elif status == "unavailable":

            unavailable_files += 1

            logger.warning(
                f"{date_string} | "
                f"{rrc} | "
                f"UPDATE_UNAVAILABLE | "
                f"time={update_time} | "
                f"url={url}"
            )

        else:

            print()

            logger.error(
                f"{date_string} | "
                f"{rrc} | "
                f"DOWNLOAD_FAILED | "
                f"time={update_time} | "
                f"url={url}"
            )

            append_stats(
                stats_file,
                date_string,
                rrc,
                "download_failed",
                expected_update_files=(
                    expected_files
                ),
                downloaded_update_files=(
                    downloaded_files
                ),
                reused_update_files=(
                    reused_files
                ),
                unavailable_update_files=(
                    unavailable_files
                ),
                mrt_bytes=(
                    total_mrt_bytes
                ),
                download_seconds=(
                    total_download_seconds
                ),
                error=(
                    f"wget failed for "
                    f"{filename}"
                ),
            )

            return {
                "status":
                    "failed",

                "elements":
                    0,

                "parquet_bytes":
                    0,

                "download_bytes":
                    total_mrt_bytes,

                "download_seconds":
                    total_download_seconds,

                "parse_seconds":
                    0.0,
            }

    print()

    available_files = (
        len(
            mrt_files
        )
    )

    print(
        f"    Available MRTs: "
        f"{available_files:,}/"
        f"{expected_files:,}"
    )

    print(
        f"    Downloaded    : "
        f"{downloaded_files:,}"
    )

    print(
        f"    Reused        : "
        f"{reused_files:,}"
    )

    print(
        f"    Unavailable   : "
        f"{unavailable_files:,}"
    )

    print(
        f"    MRT size      : "
        f"{format_size(total_mrt_bytes)}"
    )

    print(
        f"    Download time : "
        f"{format_duration(total_download_seconds)}"
    )

    if available_files == 0:

        logger.error(
            f"{date_string} | "
            f"{rrc} | "
            f"NO_UPDATE_FILES"
        )

        append_stats(
            stats_file,
            date_string,
            rrc,
            "unavailable",
            expected_update_files=(
                expected_files
            ),
            downloaded_update_files=(
                downloaded_files
            ),
            reused_update_files=(
                reused_files
            ),
            unavailable_update_files=(
                unavailable_files
            ),
            download_seconds=(
                total_download_seconds
            ),
            error=(
                "No UPDATE files "
                "available"
            ),
        )

        return {
            "status":
                "unavailable",

            "elements":
                0,

            "parquet_bytes":
                0,

            "download_bytes":
                total_mrt_bytes,

            "download_seconds":
                total_download_seconds,

            "parse_seconds":
                0.0,
        }

    # --------------------------------------------------------
    # Parse all MRTs into ONE collector-day Parquet
    # --------------------------------------------------------

    print(
        "    Parsing UPDATE MRTs "
        "with bgpreader..."
    )

    logger.info(
        f"{date_string} | "
        f"{rrc} | "
        f"PARSE_START | "
        f"mrt_files="
        f"{available_files}"
    )

    parse_start = (
        time.perf_counter()
    )

    writer = None
    batch = new_batch()

    total_elements = 0
    total_announcements = 0
    total_withdrawals = 0
    total_skipped_lines = 0

    progress = tqdm(
        desc=(
            f"Parse "
            f"{date_string} "
            f"{rrc}"
        ),
        unit="elements",
        unit_scale=True,
        dynamic_ncols=True,
    )

    try:

        writer = pq.ParquetWriter(
            temp_file,
            ARROW_SCHEMA,
            compression="zstd",
            compression_level=(
                ZSTD_LEVEL
            ),
            use_dictionary=[
                "collector",
                "element_type",
            ],
            write_statistics=True,
        )

        for mrt_index, mrt_file in enumerate(
            mrt_files,
            start=1,
        ):

            (
                element_count,
                announcement_count,
                withdrawal_count,
                skipped_lines,
            ) = parse_update_file(
                rrc,
                mrt_file,
                writer,
                batch,
                progress,
            )

            total_elements += (
                element_count
            )

            total_announcements += (
                announcement_count
            )

            total_withdrawals += (
                withdrawal_count
            )

            total_skipped_lines += (
                skipped_lines
            )

        # Final incomplete batch spanning the
        # last UPDATE MRT.
        if batch["timestamp"]:

            flush_batch(
                writer,
                batch,
            )

    except Exception as exc:

        parse_seconds = (
            time.perf_counter()
            - parse_start
        )

        progress.close()

        if writer is not None:

            writer.close()

        temp_file.unlink(
            missing_ok=True
        )

        print(
            f"    Parsing failed: "
            f"{exc}"
        )

        logger.error(
            f"{date_string} | "
            f"{rrc} | "
            f"PARSE_FAILED | "
            f"time="
            f"{parse_seconds:.2f}s | "
            f"error={exc}"
        )

        logger.error(
            traceback.format_exc()
        )

        append_stats(
            stats_file,
            date_string,
            rrc,
            "parse_failed",

            expected_update_files=(
                expected_files
            ),

            downloaded_update_files=(
                downloaded_files
            ),

            reused_update_files=(
                reused_files
            ),

            unavailable_update_files=(
                unavailable_files
            ),

            mrt_bytes=(
                total_mrt_bytes
            ),

            download_seconds=(
                total_download_seconds
            ),

            parse_seconds=(
                parse_seconds
            ),

            skipped_lines=(
                total_skipped_lines
            ),

            error=str(exc),
        )

        # Keep MRTs for restart.
        return {
            "status":
                "failed",

            "elements":
                0,

            "parquet_bytes":
                0,

            "download_bytes":
                total_mrt_bytes,

            "download_seconds":
                total_download_seconds,

            "parse_seconds":
                parse_seconds,
        }

    finally:

        if not progress.disable:

            progress.close()

    if writer is not None:

        writer.close()

    parse_seconds = (
        time.perf_counter()
        - parse_start
    )

    # --------------------------------------------------------
    # Basic count checks
    # --------------------------------------------------------

    if total_elements == 0:

        temp_file.unlink(
            missing_ok=True
        )

        error = (
            "No announcement or "
            "withdrawal elements parsed."
        )

        logger.error(
            f"{date_string} | "
            f"{rrc} | "
            f"PARSE_EMPTY"
        )

        append_stats(
            stats_file,
            date_string,
            rrc,
            "parse_failed",

            expected_update_files=(
                expected_files
            ),

            downloaded_update_files=(
                downloaded_files
            ),

            reused_update_files=(
                reused_files
            ),

            unavailable_update_files=(
                unavailable_files
            ),

            mrt_bytes=(
                total_mrt_bytes
            ),

            download_seconds=(
                total_download_seconds
            ),

            parse_seconds=(
                parse_seconds
            ),

            skipped_lines=(
                total_skipped_lines
            ),

            error=error,
        )

        return {
            "status":
                "failed",

            "elements":
                0,

            "parquet_bytes":
                0,

            "download_bytes":
                total_mrt_bytes,

            "download_seconds":
                total_download_seconds,

            "parse_seconds":
                parse_seconds,
        }

    if (
        total_elements
        != (
            total_announcements
            + total_withdrawals
        )
    ):

        temp_file.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Element accounting mismatch: "
            f"elements={total_elements:,}, "
            f"announcements="
            f"{total_announcements:,}, "
            f"withdrawals="
            f"{total_withdrawals:,}."
        )

    # --------------------------------------------------------
    # Validate temporary Parquet before exposing it
    # --------------------------------------------------------

    parquet = pq.ParquetFile(
        temp_file
    )

    parquet_rows = (
        parquet
        .metadata
        .num_rows
    )

    if (
        parquet_rows
        != total_elements
    ):

        temp_file.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Row-count mismatch for "
            f"{date_string} {rrc}: "
            f"parsed "
            f"{total_elements:,}, "
            f"Parquet contains "
            f"{parquet_rows:,}."
        )

    temp_file.replace(
        output_file
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        "    Validating Parquet..."
    )

    try:

        stats = (
            validate_parquet(
                output_file,
                expected_rrc=rrc,
                expected_elements=(
                    total_elements
                ),
            )
        )

    except Exception as exc:

        print(
            f"    Validation failed: "
            f"{exc}"
        )

        logger.error(
            f"{date_string} | "
            f"{rrc} | "
            f"VALIDATION_FAILED | "
            f"error={exc}"
        )

        output_file.unlink(
            missing_ok=True
        )

        append_stats(
            stats_file,
            date_string,
            rrc,
            "validation_failed",

            expected_update_files=(
                expected_files
            ),

            downloaded_update_files=(
                downloaded_files
            ),

            reused_update_files=(
                reused_files
            ),

            unavailable_update_files=(
                unavailable_files
            ),

            elements=(
                total_elements
            ),

            announcements=(
                total_announcements
            ),

            withdrawals=(
                total_withdrawals
            ),

            mrt_bytes=(
                total_mrt_bytes
            ),

            download_seconds=(
                total_download_seconds
            ),

            parse_seconds=(
                parse_seconds
            ),

            skipped_lines=(
                total_skipped_lines
            ),

            error=str(exc),
        )

        # Keep MRTs for retry.
        return {
            "status":
                "failed",

            "elements":
                0,

            "parquet_bytes":
                0,

            "download_bytes":
                total_mrt_bytes,

            "download_seconds":
                total_download_seconds,

            "parse_seconds":
                parse_seconds,
        }

    parquet_bytes = (
        output_file
        .stat()
        .st_size
    )

    print(
        f"    Elements      : "
        f"{total_elements:,}"
    )

    print(
        f"      Announcements: "
        f"{total_announcements:,}"
    )

    print(
        f"      Withdrawals  : "
        f"{total_withdrawals:,}"
    )

    print(
        f"    Row groups    : "
        f"{stats['row_groups']:,}"
    )

    print(
        f"    Skipped lines : "
        f"{total_skipped_lines:,}"
    )

    print(
        f"    Parse time    : "
        f"{format_duration(parse_seconds)}"
    )

    print(
        f"    Parquet size  : "
        f"{format_size(parquet_bytes)}"
    )

    # --------------------------------------------------------
    # Delete MRTs only after successful validation
    # --------------------------------------------------------

    removed_mrts = 0

    for mrt_file in mrt_files:

        if mrt_file.exists():

            mrt_file.unlink()

            removed_mrts += 1

    # Remove empty collector MRT directory.
    try:

        day_mrt_dir.rmdir()

    except OSError:

        pass

    print(
        f"    MRTs removed  : "
        f"{removed_mrts:,}"
    )

    logger.info(
        f"{date_string} | "
        f"{rrc} | "
        f"SUCCESS | "
        f"update_files="
        f"{available_files} | "
        f"elements="
        f"{total_elements} | "
        f"announcements="
        f"{total_announcements} | "
        f"withdrawals="
        f"{total_withdrawals} | "
        f"parquet="
        f"{format_size(parquet_bytes)} | "
        f"download="
        f"{total_download_seconds:.2f}s | "
        f"parse="
        f"{parse_seconds:.2f}s"
    )

    append_stats(
        stats_file,
        date_string,
        rrc,
        "success",

        expected_update_files=(
            expected_files
        ),

        downloaded_update_files=(
            downloaded_files
        ),

        reused_update_files=(
            reused_files
        ),

        unavailable_update_files=(
            unavailable_files
        ),

        elements=(
            total_elements
        ),

        announcements=(
            total_announcements
        ),

        withdrawals=(
            total_withdrawals
        ),

        row_groups=(
            stats[
                "row_groups"
            ]
        ),

        mrt_bytes=(
            total_mrt_bytes
        ),

        parquet_bytes=(
            parquet_bytes
        ),

        download_seconds=(
            total_download_seconds
        ),

        parse_seconds=(
            parse_seconds
        ),

        skipped_lines=(
            total_skipped_lines
        ),
    )

    return {
        "status":
            "success",

        "elements":
            total_elements,

        "announcements":
            total_announcements,

        "withdrawals":
            total_withdrawals,

        "parquet_bytes":
            parquet_bytes,

        "download_bytes":
            total_mrt_bytes,

        "download_seconds":
            total_download_seconds,

        "parse_seconds":
            parse_seconds,
    }


# ============================================================
# Main
# ============================================================

def main():

    if shutil.which(
        "wget"
    ) is None:

        raise SystemExit(
            "ERROR: wget was not "
            "found in PATH."
        )

    if shutil.which(
        "bgpreader"
    ) is None:

        raise SystemExit(
            "ERROR: bgpreader was not "
            "found in PATH."
        )

    logger, log_file = (
        setup_logger()
    )

    stats_file = (
        BASE_DIR
        / "_ingestion_stats.csv"
    )

    initialize_stats_csv(
        stats_file
    )

    dates = list(
        iter_dates(
            START_DATE,
            END_DATE,
        )
    )

    collector_days = (
        len(dates)
        * len(RRC_IDS)
    )

    expected_mrt_files = (
        collector_days
        * len(UPDATE_TIMES)
    )

    total_start = (
        time.perf_counter()
    )

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "RIPE RIS UPDATE TEST DATASET"
    )

    print(
        "="
        * 80
    )

    print(
        f"Start date       : "
        f"{START_DATE}"
    )

    print(
        f"End date         : "
        f"{END_DATE}"
    )

    print(
        f"Days             : "
        f"{len(dates)}"
    )

    print(
        f"Collectors       : "
        f"{len(RRC_IDS)}"
    )

    print(
        f"Collector-days   : "
        f"{collector_days}"
    )

    print(
        f"Files/day/collector: "
        f"{len(UPDATE_TIMES)}"
    )

    print(
        f"Expected MRT files: "
        f"{expected_mrt_files:,}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE:,}"
    )

    print(
        f"MRT directory    : "
        f"{MRT_DIR}"
    )

    print(
        f"Parquet directory: "
        f"{PARQUET_DIR}"
    )

    print(
        f"Statistics       : "
        f"{stats_file}"
    )

    print(
        f"Log              : "
        f"{log_file}"
    )

    print(
        "="
        * 80
    )

    logger.info(
        "DATASET START | "
        f"start={START_DATE} | "
        f"end={END_DATE} | "
        f"days={len(dates)} | "
        f"collectors="
        f"{len(RRC_IDS)} | "
        f"collector_days="
        f"{collector_days} | "
        f"expected_mrt_files="
        f"{expected_mrt_files}"
    )

    successful = 0
    skipped = 0
    unavailable = 0
    failed = 0

    total_elements = 0
    total_announcements = 0
    total_withdrawals = 0

    total_download_bytes = 0
    total_parquet_bytes = 0

    total_download_seconds = 0.0
    total_parse_seconds = 0.0

    for current_date in dates:

        date_string = (
            current_date
            .strftime("%Y-%m-%d")
        )

        print(
            "\n"
            + "="
            * 80
        )

        print(
            f"DATE: "
            f"{date_string}"
        )

        print(
            "="
            * 80
        )

        logger.info(
            f"DATE START | "
            f"{date_string}"
        )

        for rrc_id in RRC_IDS:

            rrc = rrc_name(
                rrc_id
            )

            try:

                result = (
                    process_collector_day(
                        current_date,
                        rrc,
                        logger,
                        stats_file,
                    )
                )

            except Exception as exc:

                failed += 1

                print(
                    f"\n[{date_string} | "
                    f"{rrc}] "
                    f"UNHANDLED FAILURE: "
                    f"{exc}"
                )

                logger.error(
                    f"{date_string} | "
                    f"{rrc} | "
                    f"UNHANDLED_FAILURE | "
                    f"{exc}"
                )

                logger.error(
                    traceback.format_exc()
                )

                continue

            status = (
                result[
                    "status"
                ]
            )

            if status == "success":

                successful += 1

            elif status == "skipped":

                skipped += 1

            elif status == "unavailable":

                unavailable += 1

            else:

                failed += 1

            total_elements += (
                result.get(
                    "elements",
                    0,
                )
            )

            total_announcements += (
                result.get(
                    "announcements",
                    0,
                )
            )

            total_withdrawals += (
                result.get(
                    "withdrawals",
                    0,
                )
            )

            total_download_bytes += (
                result.get(
                    "download_bytes",
                    0,
                )
            )

            total_parquet_bytes += (
                result.get(
                    "parquet_bytes",
                    0,
                )
            )

            total_download_seconds += (
                result.get(
                    "download_seconds",
                    0.0,
                )
            )

            total_parse_seconds += (
                result.get(
                    "parse_seconds",
                    0.0,
                )
            )

        logger.info(
            f"DATE COMPLETE | "
            f"{date_string}"
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # Final files-on-disk summary
    # ========================================================

    parquet_files = sorted(
        PARQUET_DIR.glob(
            "*/*.parquet"
        )
    )

    disk_rows = 0
    disk_bytes = 0

    for parquet_file in parquet_files:

        try:

            parquet = pq.ParquetFile(
                parquet_file
            )

            disk_rows += (
                parquet
                .metadata
                .num_rows
            )

            disk_bytes += (
                parquet_file
                .stat()
                .st_size
            )

        except Exception as exc:

            logger.error(
                "FINAL_CHECK_FAILED | "
                f"{parquet_file} | "
                f"{exc}"
            )

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "UPDATE DATASET COMPLETE"
    )

    print(
        "="
        * 80
    )

    print(
        f"Collector-days planned : "
        f"{collector_days}"
    )

    print(
        f"Successful this run    : "
        f"{successful}"
    )

    print(
        f"Skipped existing       : "
        f"{skipped}"
    )

    print(
        f"Unavailable            : "
        f"{unavailable}"
    )

    print(
        f"Failed                 : "
        f"{failed}"
    )

    print(
        f"Parquet files on disk  : "
        f"{len(parquet_files)}"
    )

    print(
        f"Total rows on disk     : "
        f"{disk_rows:,}"
    )

    print(
        f"Total Parquet size     : "
        f"{format_size(disk_bytes)}"
    )

    print(
        "\nTHIS RUN"
    )

    print(
        f"Elements processed     : "
        f"{total_elements:,}"
    )

    print(
        f"Announcements          : "
        f"{total_announcements:,}"
    )

    print(
        f"Withdrawals            : "
        f"{total_withdrawals:,}"
    )

    print(
        f"Downloaded             : "
        f"{format_size(total_download_bytes)}"
    )

    print(
        "\nTIMING"
    )

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
        f"{format_duration(total_seconds)}"
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

    print(
        f"\nStatistics: "
        f"{stats_file}"
    )

    print(
        f"Log       : "
        f"{log_file}"
    )

    print(
        "="
        * 80
    )

    logger.info(
        "DATASET COMPLETE | "
        f"planned={collector_days} | "
        f"success={successful} | "
        f"skipped={skipped} | "
        f"unavailable={unavailable} | "
        f"failed={failed} | "
        f"parquets="
        f"{len(parquet_files)} | "
        f"rows={disk_rows} | "
        f"parquet_size="
        f"{format_size(disk_bytes)} | "
        f"runtime="
        f"{format_duration(total_seconds)}"
    )

    for handler in logger.handlers[:]:

        handler.flush()
        handler.close()

        logger.removeHandler(
            handler
        )


if __name__ == "__main__":

    main()
