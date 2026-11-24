#!/usr/bin/env python3

from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


# ============================================================
# Configuration
# ============================================================

RIB_DAY_DIR = Path(
    "/data/jaw/pam2026/2022-01-01"
)

POISONING_DIR = (
    RIB_DAY_DIR
    / "poisoning"
)

UPDATE_DIR = Path(
    "/data/jaw/pam2026/updates/"
    "2021-12-31_2022-01-02/parquet"
)

OUTPUT_DIR = Path(
    "/data/jaw/pam2026/"
    "2022-01-01/path_dynamics"
)

EVENTS_FILE = (
    OUTPUT_DIR
    / "noncons_events.parquet"
)

FILTERED_UPDATES_FILE = (
    OUTPUT_DIR
    / "affected_prefix_updates.parquet"
)

COMBINED_RESULTS_FILE = (
    OUTPUT_DIR
    / "path_dynamics.parquet"
)

RS_SUMMARY_FILE = (
    OUTPUT_DIR
    / "rs_summary.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "analysis_summary.txt"
)


# ============================================================
# Analysis windows
# ============================================================

WINDOW_MINUTES = (
    5,
    15,
)

MAX_WINDOW_MINUTES = max(
    WINDOW_MINUTES
)

MAX_WINDOW_SECONDS = (
    MAX_WINDOW_MINUTES
    * 60
)


BATCH_SIZE = 500_000

ZSTD_LEVEL = 10


# ============================================================
# Helpers
# ============================================================

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


def validate_directories():

    if not POISONING_DIR.exists():

        raise RuntimeError(
            "\nABORT: poisoning directory "
            "does not exist:\n"
            f"{POISONING_DIR}"
        )

    if not UPDATE_DIR.exists():

        raise RuntimeError(
            "\nABORT: UPDATE directory "
            "does not exist:\n"
            f"{UPDATE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_poisoning_files():

    files = sorted(
        POISONING_DIR.glob(
            "*.poisoning.parquet"
        )
    )

    if not files:

        raise RuntimeError(
            "No poisoning Parquet "
            "files found."
        )

    return files


def get_update_files():

    files = sorted(
        UPDATE_DIR.glob(
            "*/*.parquet"
        )
    )

    if not files:

        raise RuntimeError(
            "No UPDATE Parquet "
            "files found."
        )

    return files


# ============================================================
# Phase 1
# Construct non-cons event table
# ============================================================

def build_event_table(
    poisoning_files,
):

    print()
    print("=" * 72)
    print("PHASE 1: NON-CONS EVENTS")
    print("=" * 72)

    frames = []

    required_columns = [
        "rib_timestamp",
        "collector",
        "peer_asn",
        "peer_ip",
        "prefix",
        "cleaned_aspath",
        "introducing_as",
        "rs_path",
    ]

    total_rows = 0

    for index, input_file in enumerate(
        poisoning_files,
        start=1,
    ):

        print(
            f"\r[{index}/"
            f"{len(poisoning_files)}] "
            f"{input_file.name}",
            end="",
            flush=True,
        )

        df = pl.read_parquet(
            input_file,
            columns=required_columns,
        )

        total_rows += (
            df.height
        )

        frames.append(
            df
        )

    print()

    events = pl.concat(
        frames,
        how="vertical",
    )

    del frames

    # --------------------------------------------------------
    # One event per:
    #
    #   RIB timestamp
    #   collector-peer session
    #   prefix
    #   RS
    #
    # --------------------------------------------------------

    events = (
        events
        .unique(
            subset=[
                "rib_timestamp",
                "collector",
                "peer_ip",
                "prefix",
                "rs_path",
            ],
            maintain_order=False,
        )
        .sort(
            [
                "rib_timestamp",
                "prefix",
                "collector",
                "peer_ip",
            ]
        )
    )

    # --------------------------------------------------------
    # The UPDATE scan only needs the largest window.
    #
    # Both ±5 and ±15 minute measurements will later be
    # computed from this same reduced UPDATE dataset.
    # --------------------------------------------------------

    events = (
        events
        .with_columns([
            (
                pl.col(
                    "rib_timestamp"
                )
                - MAX_WINDOW_SECONDS
            )
            .alias(
                "window_start"
            ),

            (
                pl.col(
                    "rib_timestamp"
                )
                + MAX_WINDOW_SECONDS
            )
            .alias(
                "window_end"
            ),
        ])
    )

    events.write_parquet(
        EVENTS_FILE,
        compression="zstd",
        compression_level=(
            ZSTD_LEVEL
        ),
    )

    print(
        f"Input RS rows       : "
        f"{total_rows:,}"
    )

    print(
        f"Distinct events     : "
        f"{events.height:,}"
    )

    print(
        f"Affected prefixes   : "
        f"{events['prefix'].n_unique():,}"
    )

    print(
        f"Distinct RSs        : "
        f"{events['rs_path'].n_unique():,}"
    )

    print(
        f"Collectors          : "
        f"{events['collector'].n_unique():,}"
    )

    print(
        f"Analysis windows    : "
        f"±5 and ±15 minutes"
    )

    print(
        f"UPDATE scan envelope: "
        f"±{MAX_WINDOW_MINUTES} minutes"
    )

    print(
        f"Output              : "
        f"{EVENTS_FILE}"
    )

    return events


# ============================================================
# Phase 2
# Filter UPDATEs
# ============================================================

FILTERED_UPDATE_SCHEMA = pa.schema([
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
        "as_path",
        pa.string(),
    ),
])


def build_prefix_windows(
    events,
):

    prefix_windows = (
        events
        .group_by(
            "prefix"
        )
        .agg([
            pl.col(
                "window_start"
            )
            .min()
            .alias(
                "min_timestamp"
            ),

            pl.col(
                "window_end"
            )
            .max()
            .alias(
                "max_timestamp"
            ),
        ])
    )

    return prefix_windows


def filter_updates(
    events,
    update_files,
):

    print()
    print("=" * 72)
    print("PHASE 2: FILTER UPDATE DATASET")
    print("=" * 72)

    prefix_windows = (
        build_prefix_windows(
            events
        )
    )

    affected_prefixes = set(
        prefix_windows[
            "prefix"
        ].to_list()
    )

    window_lookup = {
        prefix_value: (
            start,
            end,
        )
        for (
            prefix_value,
            start,
            end,
        )
        in prefix_windows.iter_rows()
    }

    print(
        f"Affected prefixes   : "
        f"{len(affected_prefixes):,}"
    )

    print(
        f"UPDATE files        : "
        f"{len(update_files):,}"
    )

    print(
        f"Filter envelope     : "
        f"±{MAX_WINDOW_MINUTES} minutes"
    )

    temp_file = Path(
        str(
            FILTERED_UPDATES_FILE
        )
        + ".tmp"
    )

    temp_file.unlink(
        missing_ok=True
    )

    FILTERED_UPDATES_FILE.unlink(
        missing_ok=True
    )

    writer = pq.ParquetWriter(
        temp_file,
        FILTERED_UPDATE_SCHEMA,
        compression="zstd",
        compression_level=(
            ZSTD_LEVEL
        ),
    )

    scanned_rows = 0
    retained_rows = 0

    start_time = (
        time.perf_counter()
    )

    try:

        for file_index, update_file in enumerate(
            update_files,
            start=1,
        ):

            parquet = pq.ParquetFile(
                update_file
            )

            print(
                f"\r[{file_index}/"
                f"{len(update_files)}] "
                f"{update_file.name} "
                f"| scanned="
                f"{scanned_rows:,} "
                f"| kept="
                f"{retained_rows:,}",
                end="",
                flush=True,
            )

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=[
                    "timestamp",
                    "collector",
                    "peer_asn",
                    "peer_ip",
                    "element_type",
                    "prefix",
                    "as_path",
                ],
            ):

                df = pl.from_arrow(
                    batch
                )

                scanned_rows += (
                    df.height
                )

                # --------------------------------------------
                # First filter:
                # affected prefixes only.
                # --------------------------------------------

                df = df.filter(
                    pl.col(
                        "prefix"
                    )
                    .is_in(
                        affected_prefixes
                    )
                )

                if df.height == 0:

                    continue

                # --------------------------------------------
                # Then restrict to the overall ±15-minute
                # event envelope for each affected prefix.
                # --------------------------------------------

                rows = []

                for row in df.iter_rows(
                    named=True
                ):

                    (
                        min_timestamp,
                        max_timestamp,
                    ) = window_lookup[
                        row["prefix"]
                    ]

                    timestamp = (
                        row[
                            "timestamp"
                        ]
                    )

                    if (
                        timestamp
                        >= min_timestamp
                        and timestamp
                        <= max_timestamp
                    ):

                        rows.append(
                            row
                        )

                if not rows:

                    continue

                filtered = pl.DataFrame(
                    rows,
                    schema={
                        "timestamp":
                            pl.Int64,

                        "collector":
                            pl.Utf8,

                        "peer_asn":
                            pl.Int64,

                        "peer_ip":
                            pl.Utf8,

                        "element_type":
                            pl.Utf8,

                        "prefix":
                            pl.Utf8,

                        "as_path":
                            pl.Utf8,
                    },
                )

                retained_rows += (
                    filtered.height
                )

                writer.write_table(
                    filtered
                    .to_arrow()
                    .cast(
                        FILTERED_UPDATE_SCHEMA
                    )
                )

        writer.close()

    except Exception:

        writer.close()

        temp_file.unlink(
            missing_ok=True
        )

        raise

    print()

    if retained_rows == 0:

        temp_file.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "No affected-prefix UPDATEs "
            "were retained."
        )

    temp_file.replace(
        FILTERED_UPDATES_FILE
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Rows scanned        : "
        f"{scanned_rows:,}"
    )

    print(
        f"Rows retained       : "
        f"{retained_rows:,}"
    )

    print(
        f"Retention           : "
        f"{retained_rows / scanned_rows * 100:.6f}%"
    )

    print(
        f"Runtime             : "
        f"{format_duration(elapsed)}"
    )

    print(
        f"Output              : "
        f"{FILTERED_UPDATES_FILE}"
    )


# ============================================================
# Phase 3
# Load filtered UPDATEs
# ============================================================

def load_filtered_updates():

    print()
    print("=" * 72)
    print("PHASE 3: LOAD FILTERED UPDATEs")
    print("=" * 72)

    updates = pl.read_parquet(
        FILTERED_UPDATES_FILE
    )

    announcements = (
        updates
        .filter(
            pl.col(
                "element_type"
            )
            == "A"
        )
        .height
    )

    withdrawals = (
        updates
        .filter(
            pl.col(
                "element_type"
            )
            == "W"
        )
        .height
    )

    print(
        f"Filtered UPDATEs    : "
        f"{updates.height:,}"
    )

    print(
        f"Announcements       : "
        f"{announcements:,}"
    )

    print(
        f"Withdrawals         : "
        f"{withdrawals:,}"
    )

    return updates


# ============================================================
# Phase 4
# Event-level path dynamics
# ============================================================

def get_path_set(
    df,
):

    if df.height == 0:

        return set()

    paths = set(
        df[
            "as_path"
        ]
        .drop_nulls()
        .to_list()
    )

    paths.discard(
        ""
    )

    return paths


def analyze_one_window(
    prefix_updates,
    event_time,
    collector,
    peer_ip,
    window_minutes,
):

    window_seconds = (
        window_minutes
        * 60
    )

    if prefix_updates is None:

        return {
            "ris_paths_before": 0,
            "ris_paths_after": 0,
            "ris_new_paths_after": 0,
            "ris_disappeared_paths_after": 0,
            "ris_has_new_path_after": False,
            "ris_path_delta": 0,
            "ris_withdrawals_before": 0,
            "ris_withdrawals_after": 0,

            "session_paths_before": 0,
            "session_paths_after": 0,
            "session_new_paths_after": 0,
            "session_disappeared_paths_after": 0,
            "session_has_new_path_after": False,
            "session_path_delta": 0,
            "session_withdrawals_before": 0,
            "session_withdrawals_after": 0,
        }

    # --------------------------------------------------------
    # Announcements
    # --------------------------------------------------------

    announcements = (
        prefix_updates
        .filter(
            pl.col(
                "element_type"
            )
            == "A"
        )
    )

    before_ris = (
        announcements
        .filter(
            (
                pl.col(
                    "timestamp"
                )
                >= (
                    event_time
                    - window_seconds
                )
            )
            &
            (
                pl.col(
                    "timestamp"
                )
                < event_time
            )
        )
    )

    after_ris = (
        announcements
        .filter(
            (
                pl.col(
                    "timestamp"
                )
                > event_time
            )
            &
            (
                pl.col(
                    "timestamp"
                )
                <= (
                    event_time
                    + window_seconds
                )
            )
        )
    )

    before_ris_paths = (
        get_path_set(
            before_ris
        )
    )

    after_ris_paths = (
        get_path_set(
            after_ris
        )
    )

    new_ris_paths = (
        after_ris_paths
        - before_ris_paths
    )

    disappeared_ris_paths = (
        before_ris_paths
        - after_ris_paths
    )

    # --------------------------------------------------------
    # Same collector-peer session
    # --------------------------------------------------------

    before_session = (
        before_ris
        .filter(
            (
                pl.col(
                    "collector"
                )
                == collector
            )
            &
            (
                pl.col(
                    "peer_ip"
                )
                == peer_ip
            )
        )
    )

    after_session = (
        after_ris
        .filter(
            (
                pl.col(
                    "collector"
                )
                == collector
            )
            &
            (
                pl.col(
                    "peer_ip"
                )
                == peer_ip
            )
        )
    )

    before_session_paths = (
        get_path_set(
            before_session
        )
    )

    after_session_paths = (
        get_path_set(
            after_session
        )
    )

    new_session_paths = (
        after_session_paths
        - before_session_paths
    )

    disappeared_session_paths = (
        before_session_paths
        - after_session_paths
    )

    # --------------------------------------------------------
    # Withdrawals
    # --------------------------------------------------------

    withdrawals = (
        prefix_updates
        .filter(
            pl.col(
                "element_type"
            )
            == "W"
        )
    )

    withdrawals_before = (
        withdrawals
        .filter(
            (
                pl.col(
                    "timestamp"
                )
                >= (
                    event_time
                    - window_seconds
                )
            )
            &
            (
                pl.col(
                    "timestamp"
                )
                < event_time
            )
        )
    )

    withdrawals_after = (
        withdrawals
        .filter(
            (
                pl.col(
                    "timestamp"
                )
                > event_time
            )
            &
            (
                pl.col(
                    "timestamp"
                )
                <= (
                    event_time
                    + window_seconds
                )
            )
        )
    )

    session_withdrawals_before = (
        withdrawals_before
        .filter(
            (
                pl.col(
                    "collector"
                )
                == collector
            )
            &
            (
                pl.col(
                    "peer_ip"
                )
                == peer_ip
            )
        )
        .height
    )

    session_withdrawals_after = (
        withdrawals_after
        .filter(
            (
                pl.col(
                    "collector"
                )
                == collector
            )
            &
            (
                pl.col(
                    "peer_ip"
                )
                == peer_ip
            )
        )
        .height
    )

    return {
        # RIS-wide
        "ris_paths_before":
            len(
                before_ris_paths
            ),

        "ris_paths_after":
            len(
                after_ris_paths
            ),

        "ris_new_paths_after":
            len(
                new_ris_paths
            ),

        "ris_disappeared_paths_after":
            len(
                disappeared_ris_paths
            ),

        "ris_has_new_path_after":
            len(
                new_ris_paths
            )
            > 0,

        "ris_path_delta":
            (
                len(
                    after_ris_paths
                )
                - len(
                    before_ris_paths
                )
            ),

        "ris_withdrawals_before":
            withdrawals_before.height,

        "ris_withdrawals_after":
            withdrawals_after.height,

        # Same session
        "session_paths_before":
            len(
                before_session_paths
            ),

        "session_paths_after":
            len(
                after_session_paths
            ),

        "session_new_paths_after":
            len(
                new_session_paths
            ),

        "session_disappeared_paths_after":
            len(
                disappeared_session_paths
            ),

        "session_has_new_path_after":
            len(
                new_session_paths
            )
            > 0,

        "session_path_delta":
            (
                len(
                    after_session_paths
                )
                - len(
                    before_session_paths
                )
            ),

        "session_withdrawals_before":
            session_withdrawals_before,

        "session_withdrawals_after":
            session_withdrawals_after,
    }


def analyze_events(
    events,
    updates,
):

    print()
    print("=" * 72)
    print("PHASE 4: EVENT-LEVEL PATH DYNAMICS")
    print("=" * 72)

    results = []

    total_events = (
        events.height
    )

    start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Group UPDATEs once by prefix.
    # --------------------------------------------------------

    updates_by_prefix = {
        prefix_value: group
        for (
            prefix_value,
            group,
        )
        in updates.group_by(
            "prefix",
            maintain_order=False,
        )
    }

    for event_index, event in enumerate(
        events.iter_rows(
            named=True
        ),
        start=1,
    ):

        if (
            event_index == 1
            or event_index % 1000 == 0
            or event_index == total_events
        ):

            print(
                f"\rEvents: "
                f"{event_index:,}/"
                f"{total_events:,}",
                end="",
                flush=True,
            )

        prefix_value = (
            event[
                "prefix"
            ]
        )

        event_time = (
            event[
                "rib_timestamp"
            ]
        )

        collector = (
            event[
                "collector"
            ]
        )

        peer_ip = (
            event[
                "peer_ip"
            ]
        )

        prefix_updates = (
            updates_by_prefix.get(
                prefix_value
            )
        )

        result = {
            "rib_timestamp":
                event_time,

            "collector":
                collector,

            "peer_asn":
                event[
                    "peer_asn"
                ],

            "peer_ip":
                peer_ip,

            "prefix":
                prefix_value,

            "introducing_as":
                event[
                    "introducing_as"
                ],

            "rs_path":
                event[
                    "rs_path"
                ],

            "noncons_as_path":
                event[
                   "cleaned_aspath"
                ],
        }

        # ----------------------------------------------------
        # Compute exactly the same metrics independently
        # for ±5 and ±15 minutes.
        # ----------------------------------------------------

        for window_minutes in WINDOW_MINUTES:

            metrics = (
                analyze_one_window(
                    prefix_updates,
                    event_time,
                    collector,
                    peer_ip,
                    window_minutes,
                )
            )

            suffix = (
                f"{window_minutes}m"
            )

            for (
                metric_name,
                metric_value,
            ) in metrics.items():

                result[
                    f"{metric_name}_{suffix}"
                ] = metric_value

        results.append(
            result
        )

    print()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    result_df = pl.DataFrame(
        results
    )

    result_df.write_parquet(
        COMBINED_RESULTS_FILE,
        compression="zstd",
        compression_level=(
            ZSTD_LEVEL
        ),
    )

    print(
        f"Events analyzed     : "
        f"{result_df.height:,}"
    )

    print(
        f"Windows             : "
        f"±5 and ±15 minutes"
    )

    print(
        f"Runtime             : "
        f"{format_duration(elapsed)}"
    )

    print(
        f"Output              : "
        f"{COMBINED_RESULTS_FILE}"
    )

    return result_df


# ============================================================
# Phase 5
# RS summary
# ============================================================

def build_rs_summary(
    results,
):

    print()
    print("=" * 72)
    print("PHASE 5: RS SUMMARY")
    print("=" * 72)

    aggregations = [
        pl.count()
        .alias(
            "events"
        ),

        pl.col(
            "prefix"
        )
        .n_unique()
        .alias(
            "unique_prefixes"
        ),

        pl.col(
            "introducing_as"
        )
        .n_unique()
        .alias(
            "introducing_ases"
        ),
    ]

    for window_minutes in WINDOW_MINUTES:

        suffix = (
            f"{window_minutes}m"
        )

        aggregations.extend([
            pl.col(
                f"ris_has_new_path_after_{suffix}"
            )
            .sum()
            .alias(
                f"events_with_new_ris_path_{suffix}"
            ),

            pl.col(
                f"session_has_new_path_after_{suffix}"
            )
            .sum()
            .alias(
                f"events_with_new_session_path_{suffix}"
            ),

            pl.col(
                f"ris_new_paths_after_{suffix}"
            )
            .sum()
            .alias(
                f"total_new_ris_paths_{suffix}"
            ),

            pl.col(
                f"session_new_paths_after_{suffix}"
            )
            .sum()
            .alias(
                f"total_new_session_paths_{suffix}"
            ),

            pl.col(
                f"ris_new_paths_after_{suffix}"
            )
            .median()
            .alias(
                f"median_new_ris_paths_{suffix}"
            ),

            pl.col(
                f"session_new_paths_after_{suffix}"
            )
            .median()
            .alias(
                f"median_new_session_paths_{suffix}"
            ),
        ])

    rs_summary = (
        results
        .group_by(
            "rs_path"
        )
        .agg(
            aggregations
        )
    )

    percentage_columns = []

    for window_minutes in WINDOW_MINUTES:

        suffix = (
            f"{window_minutes}m"
        )

        percentage_columns.extend([
            (
                pl.col(
                    f"events_with_new_ris_path_{suffix}"
                )
                / pl.col(
                    "events"
                )
                * 100
            )
            .alias(
                f"pct_events_with_new_ris_path_{suffix}"
            ),

            (
                pl.col(
                    f"events_with_new_session_path_{suffix}"
                )
                / pl.col(
                    "events"
                )
                * 100
            )
            .alias(
                f"pct_events_with_new_session_path_{suffix}"
            ),
        ])

    rs_summary = (
        rs_summary
        .with_columns(
            percentage_columns
        )
        .sort(
            [
                "events",
                "total_new_ris_paths_15m",
            ],
            descending=True,
        )
    )

    rs_summary.write_csv(
        RS_SUMMARY_FILE
    )

    print(
        f"Distinct RSs        : "
        f"{rs_summary.height:,}"
    )

    print(
        f"Output              : "
        f"{RS_SUMMARY_FILE}"
    )

    return rs_summary


# ============================================================
# Phase 6
# Final comparison
# ============================================================

def count_true(
    results,
    column,
):

    return (
        results
        .filter(
            pl.col(
                column
            )
        )
        .height
    )


def count_positive(
    results,
    column,
):

    return (
        results
        .filter(
            pl.col(
                column
            )
            > 0
        )
        .height
    )


def median_value(
    results,
    column,
):

    value = (
        results[
            column
        ]
        .median()
    )

    if value is None:

        return 0.0

    return float(
        value
    )


def summarize_window(
    results,
    window_minutes,
):

    suffix = (
        f"{window_minutes}m"
    )

    return {
        "ris_new":
            count_true(
                results,
                f"ris_has_new_path_after_{suffix}",
            ),

        "ris_increase":
            count_positive(
                results,
                f"ris_path_delta_{suffix}",
            ),

        "ris_withdrawal":
            count_positive(
                results,
                f"ris_withdrawals_after_{suffix}",
            ),

        "ris_before":
            median_value(
                results,
                f"ris_paths_before_{suffix}",
            ),

        "ris_after":
            median_value(
                results,
                f"ris_paths_after_{suffix}",
            ),

        "ris_new_median":
            median_value(
                results,
                f"ris_new_paths_after_{suffix}",
            ),

        "session_new":
            count_true(
                results,
                f"session_has_new_path_after_{suffix}",
            ),

        "session_increase":
            count_positive(
                results,
                f"session_path_delta_{suffix}",
            ),

        "session_withdrawal":
            count_positive(
                results,
                f"session_withdrawals_after_{suffix}",
            ),

        "session_before":
            median_value(
                results,
                f"session_paths_before_{suffix}",
            ),

        "session_after":
            median_value(
                results,
                f"session_paths_after_{suffix}",
            ),

        "session_new_median":
            median_value(
                results,
                f"session_new_paths_after_{suffix}",
            ),
    }


def format_count_pct(
    count,
    total,
):

    if total == 0:

        return (
            f"{count:,} (0.00%)"
        )

    return (
        f"{count:,} "
        f"({count / total * 100:.2f}%)"
    )


def summarize(
    events,
    results,
    rs_summary,
):

    print()
    print("=" * 72)
    print("PATH-DYNAMICS SUMMARY")
    print("=" * 72)

    total_events = (
        results.height
    )

    s5 = (
        summarize_window(
            results,
            5,
        )
    )

    s15 = (
        summarize_window(
            results,
            15,
        )
    )

    summary = f"""
NON-CONS PATH-DYNAMICS TEST
========================================================================

Windows
-------
Immediate window                : ±5 minutes
Longer window                   : ±15 minutes

Dataset
-------
Non-cons events                 : {total_events:,}
Affected prefixes               : {events['prefix'].n_unique():,}
Distinct RSs                    : {events['rs_path'].n_unique():,}

RIS-WIDE PATH DYNAMICS
------------------------------------------------------------------------
Metric                                      ±5 min              ±15 min
Events with >=1 new path after     {format_count_pct(s5['ris_new'], total_events):>18}   {format_count_pct(s15['ris_new'], total_events):>18}
Events with path-count increase    {format_count_pct(s5['ris_increase'], total_events):>18}   {format_count_pct(s15['ris_increase'], total_events):>18}
Events with withdrawal after       {format_count_pct(s5['ris_withdrawal'], total_events):>18}   {format_count_pct(s15['ris_withdrawal'], total_events):>18}
Median paths before                {s5['ris_before']:>18.2f}   {s15['ris_before']:>18.2f}
Median paths after                 {s5['ris_after']:>18.2f}   {s15['ris_after']:>18.2f}
Median new paths after             {s5['ris_new_median']:>18.2f}   {s15['ris_new_median']:>18.2f}

SAME COLLECTOR-PEER SESSION
------------------------------------------------------------------------
Metric                                      ±5 min              ±15 min
Events with >=1 new path after     {format_count_pct(s5['session_new'], total_events):>18}   {format_count_pct(s15['session_new'], total_events):>18}
Events with path-count increase    {format_count_pct(s5['session_increase'], total_events):>18}   {format_count_pct(s15['session_increase'], total_events):>18}
Events with withdrawal after       {format_count_pct(s5['session_withdrawal'], total_events):>18}   {format_count_pct(s15['session_withdrawal'], total_events):>18}
Median paths before                {s5['session_before']:>18.2f}   {s15['session_before']:>18.2f}
Median paths after                 {s5['session_after']:>18.2f}   {s15['session_after']:>18.2f}
Median new paths after             {s5['session_new_median']:>18.2f}   {s15['session_new_median']:>18.2f}

RS CONCENTRATION
----------------
Distinct RSs                    : {rs_summary.height:,}

========================================================================
""".strip()

    print(
        summary
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            summary
        )

        handle.write(
            "\n"
        )

    print()

    print(
        f"Summary             : "
        f"{SUMMARY_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    print()
    print("=" * 72)
    print("NON-CONS PATH-DYNAMICS ANALYSIS")
    print("=" * 72)

    print(
        f"Poisoning directory : "
        f"{POISONING_DIR}"
    )

    print(
        f"UPDATE directory    : "
        f"{UPDATE_DIR}"
    )

    print(
        f"Output directory    : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Analysis windows    : "
        f"±5 and ±15 minutes"
    )

    print(
        f"UPDATE scan envelope: "
        f"±{MAX_WINDOW_MINUTES} minutes"
    )

    validate_directories()

    poisoning_files = (
        get_poisoning_files()
    )

    update_files = (
        get_update_files()
    )

    print(
        f"Poisoning files     : "
        f"{len(poisoning_files)}"
    )

    print(
        f"UPDATE files        : "
        f"{len(update_files)}"
    )

    # ========================================================
    # Phase 1
    # ========================================================

    events = build_event_table(
        poisoning_files
    )

    # ========================================================
    # Phase 2
    #
    # Scan the 2.1B UPDATE rows only once.
    # The retained dataset covers ±15 minutes and therefore
    # contains everything needed for both measurements.
    # ========================================================

    if (
        FILTERED_UPDATES_FILE
        .exists()
    ):

        print()

        print(
            "Filtered UPDATE dataset "
            "already exists:"
        )

        print(
            FILTERED_UPDATES_FILE
        )

        print(
            "Skipping UPDATE scan."
        )

    else:

        filter_updates(
            events,
            update_files,
        )

    # ========================================================
    # Phase 3
    # ========================================================

    updates = (
        load_filtered_updates()
    )

    # ========================================================
    # Phase 4
    # ========================================================

    results = (
        analyze_events(
            events,
            updates,
        )
    )

    # ========================================================
    # Phase 5
    # ========================================================

    rs_summary = (
        build_rs_summary(
            results
        )
    )

    # ========================================================
    # Phase 6
    # ========================================================

    summarize(
        events,
        results,
        rs_summary,
    )

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 72)

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        f"Total runtime       : "
        f"{format_duration(total_runtime)}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()
