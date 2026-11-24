# COMPUTE STATS FOR ALL DATA FINAL
#!/usr/bin/env python3

from pathlib import Path
import csv
import shutil
import zlib
from collections import defaultdict

import pyarrow.dataset as ds
from tqdm import tqdm


BASE_DIR = Path(
    "/data/jaw/pam2026/2022-01-01"
)

POISON_DIR = BASE_DIR / "poisoning"

OUT_DIR = BASE_DIR / "dataset.stats.dedup.test"
TMP_DIR = OUT_DIR / "tmp"

RIB_PATTERN = "rrc*.parquet"
POISON_PATTERN = "rrc*.poisoning.parquet"

BATCH_SIZE = 200_000
FLUSH_EVERY_ROWS = 200_000

ROUTE_BUCKETS = 2048
PATH_BUCKETS = 1024
PREFIX_BUCKETS = 512
ASN_BUCKETS = 256
RS_BUCKETS = 512
OBS_BUCKETS = 256

SEP = "\x1f"


def is_ipv6(prefix):
    return ":" in str(prefix)


def bucket_id(value, n_buckets):
    return (
        zlib.crc32(
            str(value).encode("utf-8")
        )
        % n_buckets
    )


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def append_values(path, values):
    if not values:
        return

    with path.open("a") as fh:
        fh.write(
            "\n".join(values) + "\n"
        )


def make_bucket_paths(name, count):

    directory = TMP_DIR / name

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return [
        directory / f"{name}.{i:04d}.txt"
        for i in range(count)
    ]


def count_unique(paths, desc):

    total = 0

    for path in tqdm(
        paths,
        desc=desc,
        leave=False,
    ):

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            continue

        with path.open() as fh:
            values = {
                line.rstrip("\n")
                for line in fh
                if line.rstrip("\n")
            }

        total += len(values)

    return total


def new_buffers():
    return defaultdict(
        lambda: defaultdict(list)
    )


def add_to_buffer(
    buffers,
    key,
    value,
    n_buckets,
):

    if value is None:
        return

    value = str(value)

    if not value:
        return

    bid = bucket_id(
        value,
        n_buckets,
    )

    buffers[key][bid].append(
        value
    )


def flush_buffers(
    buffers,
    bucket_paths,
):

    for key, mapping in buffers.items():

        for bid, values in mapping.items():

            append_values(
                bucket_paths[key][bid],
                values,
            )

        mapping.clear()

def make_prefix_path_key(
    prefix,
    as_path,
):
    return (
        f"{prefix}{SEP}"
        f"{as_path}"
    )


def make_prefix_origin_path_key(
    prefix,
    origin,
    as_path,
):
    return (
        f"{prefix}{SEP}"
        f"{origin}{SEP}"
        f"{as_path}"
    )


def prepare_route_buckets():

    return {

        "rib_pp_ipv4":
            make_bucket_paths(
                "rib_pp_ipv4",
                ROUTE_BUCKETS,
            ),

        "rib_pp_ipv6":
            make_bucket_paths(
                "rib_pp_ipv6",
                ROUTE_BUCKETS,
            ),

        "rib_pop_ipv4":
            make_bucket_paths(
                "rib_pop_ipv4",
                ROUTE_BUCKETS,
            ),

        "rib_pop_ipv6":
            make_bucket_paths(
                "rib_pop_ipv6",
                ROUTE_BUCKETS,
            ),

        "rs_pp_ipv4":
            make_bucket_paths(
                "rs_pp_ipv4",
                ROUTE_BUCKETS,
            ),

        "rs_pp_ipv6":
            make_bucket_paths(
                "rs_pp_ipv6",
                ROUTE_BUCKETS,
            ),

        "rs_pop_ipv4":
            make_bucket_paths(
                "rs_pop_ipv4",
                ROUTE_BUCKETS,
            ),

        "rs_pop_ipv6":
            make_bucket_paths(
                "rs_pop_ipv6",
                ROUTE_BUCKETS,
            ),
    }

def spill_rib_route_keys(
    files,
    bucket_paths,
):

    print("\n" + "=" * 72)
    print("BUILDING RIB DEDUPLICATION KEYS")
    print("=" * 72)
    print(f"Files: {len(files):,}")

    dataset = ds.dataset(
        [str(p) for p in files],
        format="parquet",
    )

    buffers = new_buffers()

    total_rows = 0
    usable_rows = 0

    columns = [
        "prefix",
        "as_path",
        "origin",
    ]

    with tqdm(
        desc="RIB batches",
        unit="batch",
    ) as pbar:

        for batch in dataset.to_batches(
            columns=columns,
            batch_size=BATCH_SIZE,
        ):

            data = batch.to_pydict()

            for (
                prefix,
                as_path,
                origin,
            ) in zip(
                data["prefix"],
                data["as_path"],
                data["origin"],
            ):

                total_rows += 1

                if (
                    prefix is None
                    or as_path is None
                ):
                    continue

                prefix = str(prefix)
                as_path = str(as_path)

                pp_key = make_prefix_path_key(
                    prefix,
                    as_path,
                )

                if is_ipv6(prefix):
                    pp_name = "rib_pp_ipv6"
                    pop_name = "rib_pop_ipv6"
                else:
                    pp_name = "rib_pp_ipv4"
                    pop_name = "rib_pop_ipv4"

                add_to_buffer(
                    buffers,
                    pp_name,
                    pp_key,
                    ROUTE_BUCKETS,
                )
                if origin is not None:

                    pop_key = (
                        make_prefix_origin_path_key(
                            prefix,
                            origin,
                            as_path,
                        )
                    )

                    add_to_buffer(
                        buffers,
                        pop_name,
                        pop_key,
                        ROUTE_BUCKETS,
                    )

                usable_rows += 1

                if (
                    usable_rows
                    % FLUSH_EVERY_ROWS
                    == 0
                ):
                    flush_buffers(
                        buffers,
                        bucket_paths,
                    )

            flush_buffers(
                buffers,
                bucket_paths,
            )

            pbar.update(1)

    print(
        f"RIB rows processed : "
        f"{total_rows:,}"
    )

    print(
        f"RIB usable rows    : "
        f"{usable_rows:,}"
    )

def spill_rs_route_keys(
    files,
    bucket_paths,
):

    print("\n" + "=" * 72)
    print("BUILDING NON-CONS DEDUPLICATION KEYS")
    print("=" * 72)
    print(f"Files: {len(files):,}")

    dataset = ds.dataset(
        [str(p) for p in files],
        format="parquet",
    )

    buffers = new_buffers()

    total_rows = 0
    usable_rows = 0

    columns = [
        "prefix",
        "origin",
        "cleaned_aspath",
    ]

    with tqdm(
        desc="Non-cons batches",
        unit="batch",
    ) as pbar:

        for batch in dataset.to_batches(
            columns=columns,
            batch_size=BATCH_SIZE,
        ):

            data = batch.to_pydict()

            for (
                prefix,
                origin,
                cleaned_aspath,
            ) in zip(
                data["prefix"],
                data["origin"],
                data["cleaned_aspath"],
            ):

                total_rows += 1

                if (
                    prefix is None
                    or cleaned_aspath is None
                ):
                    continue

                prefix = str(prefix)
                cleaned_aspath = str(
                    cleaned_aspath
                )

                pp_key = make_prefix_path_key(
                    prefix,
                    cleaned_aspath,
                )

                if is_ipv6(prefix):
                    pp_name = "rs_pp_ipv6"
                    pop_name = "rs_pop_ipv6"
                else:
                    pp_name = "rs_pp_ipv4"
                    pop_name = "rs_pop_ipv4"

                add_to_buffer(
                    buffers,
                    pp_name,
                    pp_key,
                    ROUTE_BUCKETS,
                )

                if origin is not None:

                    pop_key = (
                        make_prefix_origin_path_key(
                            prefix,
                            origin,
                            cleaned_aspath,
                        )
                    )

                    add_to_buffer(
                        buffers,
                        pop_name,
                        pop_key,
                        ROUTE_BUCKETS,
                    )

                usable_rows += 1

                if (
                    usable_rows
                    % FLUSH_EVERY_ROWS
                    == 0
                ):
                    flush_buffers(
                        buffers,
                        bucket_paths,
                    )

            flush_buffers(
                buffers,
                bucket_paths,
            )

            pbar.update(1)

    print(
        f"Non-cons rows processed : "
        f"{total_rows:,}"
    )

    print(
        f"Non-cons usable rows    : "
        f"{usable_rows:,}"
    )

def count_route_keys(bucket_paths):

    print("\n" + "=" * 72)
    print("COUNTING DEDUPLICATED ROUTES")
    print("=" * 72)

    counts = {}

    for key, paths in bucket_paths.items():

        counts[key] = count_unique(
            paths,
            desc=f"Dedup {key}",
        )

    return counts


def iter_unique_bucket_values(paths):

    for path in paths:

        if (
            not path.exists()
            or path.stat().st_size == 0
        ):
            continue

        with path.open() as fh:

            values = {
                line.rstrip("\n")
                for line in fh
                if line.rstrip("\n")
            }

        for value in values:
            yield value

def prepare_property_buckets(view_name):

    return {
        "rib_ipv4_paths":
            make_bucket_paths(
                f"{view_name}_rib_ipv4_paths",
                PATH_BUCKETS,
            ),

        "rib_ipv6_paths":
            make_bucket_paths(
                f"{view_name}_rib_ipv6_paths",
                PATH_BUCKETS,
            ),

        "rib_ipv4_prefixes":
            make_bucket_paths(
                f"{view_name}_rib_ipv4_prefixes",
                PREFIX_BUCKETS,
            ),

        "rib_ipv6_prefixes":
            make_bucket_paths(
                f"{view_name}_rib_ipv6_prefixes",
                PREFIX_BUCKETS,
            ),

        "rib_ipv4_origins":
            make_bucket_paths(
                f"{view_name}_rib_ipv4_origins",
                ASN_BUCKETS,
            ),

        "rib_ipv6_origins":
            make_bucket_paths(
                f"{view_name}_rib_ipv6_origins",
                ASN_BUCKETS,
            ),

        "rs_ipv4_paths":
            make_bucket_paths(
                f"{view_name}_rs_ipv4_paths",
                PATH_BUCKETS,
            ),

        "rs_ipv6_paths":
            make_bucket_paths(
                f"{view_name}_rs_ipv6_paths",
                PATH_BUCKETS,
            ),

        "rs_ipv4_prefixes":
            make_bucket_paths(
                f"{view_name}_rs_ipv4_prefixes",
                PREFIX_BUCKETS,
            ),

        "rs_ipv6_prefixes":
            make_bucket_paths(
                f"{view_name}_rs_ipv6_prefixes",
                PREFIX_BUCKETS,
            ),

        "rs_ipv4_origins":
            make_bucket_paths(
                f"{view_name}_rs_ipv4_origins",
                ASN_BUCKETS,
            ),

        "rs_ipv6_origins":
            make_bucket_paths(
                f"{view_name}_rs_ipv6_origins",
                ASN_BUCKETS,
            ),
    }

def properties_from_pp(
    route_paths,
    property_paths,
    dataset_name,
    af,
):

    buffers = new_buffers()

    path_key = (
        f"{dataset_name}_{af}_paths"
    )

    prefix_key = (
        f"{dataset_name}_{af}_prefixes"
    )

    rows = 0

    for value in tqdm(
        iter_unique_bucket_values(
            route_paths
        ),
        desc=f"{dataset_name} {af} PP properties",
        leave=False,
    ):

        parts = value.split(
            SEP,
            1,
        )

        if len(parts) != 2:
            continue

        prefix, as_path = parts

        add_to_buffer(
            buffers,
            path_key,
            as_path,
            PATH_BUCKETS,
        )

        add_to_buffer(
            buffers,
            prefix_key,
            prefix,
            PREFIX_BUCKETS,
        )

        rows += 1

        if rows % FLUSH_EVERY_ROWS == 0:
            flush_buffers(
                buffers,
                property_paths,
            )

    flush_buffers(
        buffers,
        property_paths,
    )

def properties_from_pop(
    route_paths,
    property_paths,
    dataset_name,
    af,
):

    buffers = new_buffers()

    path_key = (
        f"{dataset_name}_{af}_paths"
    )

    prefix_key = (
        f"{dataset_name}_{af}_prefixes"
    )

    origin_key = (
        f"{dataset_name}_{af}_origins"
    )

    rows = 0

    for value in tqdm(
        iter_unique_bucket_values(
            route_paths
        ),
        desc=f"{dataset_name} {af} POP properties",
        leave=False,
    ):

        parts = value.split(
            SEP,
            2,
        )

        if len(parts) != 3:
            continue

        prefix, origin, as_path = parts

        add_to_buffer(
            buffers,
            path_key,
            as_path,
            PATH_BUCKETS,
        )

        add_to_buffer(
            buffers,
            prefix_key,
            prefix,
            PREFIX_BUCKETS,
        )

        add_to_buffer(
            buffers,
            origin_key,
            origin,
            ASN_BUCKETS,
        )

        rows += 1

        if rows % FLUSH_EVERY_ROWS == 0:
            flush_buffers(
                buffers,
                property_paths,
            )

    flush_buffers(
        buffers,
        property_paths,
    )

def add_pp_origins(
    files,
    property_paths,
    dataset_name,
):

    dataset = ds.dataset(
        [str(p) for p in files],
        format="parquet",
    )

    buffers = new_buffers()

    columns = [
        "prefix",
        "origin",
    ]

    rows = 0

    for batch in dataset.to_batches(
        columns=columns,
        batch_size=BATCH_SIZE,
    ):

        data = batch.to_pydict()

        for prefix, origin in zip(
            data["prefix"],
            data["origin"],
        ):

            if (
                prefix is None
                or origin is None
            ):
                continue

            prefix = str(prefix)

            if is_ipv6(prefix):
                key = (
                    f"{dataset_name}_ipv6_origins"
                )
            else:
                key = (
                    f"{dataset_name}_ipv4_origins"
                )

            add_to_buffer(
                buffers,
                key,
                origin,
                ASN_BUCKETS,
            )

            rows += 1

            if rows % FLUSH_EVERY_ROWS == 0:
                flush_buffers(
                    buffers,
                    property_paths,
                )

        flush_buffers(
            buffers,
            property_paths,
        )

def compute_property_stats(
    property_paths,
):

    stats = {}

    for key, paths in property_paths.items():

        stats[key] = count_unique(
            paths,
            desc=f"Dedup {key}",
        )

    return stats

def compute_rs_specific_stats(
    poison_files,
):

    print("\n" + "=" * 72)
    print("COMPUTING RS-SPECIFIC PROPERTIES")
    print("=" * 72)

    rs_paths = make_bucket_paths(
        "specific_rs_paths",
        RS_BUCKETS,
    )

    introducing = make_bucket_paths(
        "specific_introducing_ases",
        ASN_BUCKETS,
    )

    appearing = make_bucket_paths(
        "specific_appearing_ases",
        ASN_BUCKETS,
    )

    bucket_paths = {
        "rs_paths": rs_paths,
        "introducing": introducing,
        "appearing": appearing,
    }

    buffers = new_buffers()

    dataset = ds.dataset(
        [str(p) for p in poison_files],
        format="parquet",
    )

    columns = [
        "rs_path",
        "introducing_as",
        "appearing_ases",
    ]

    rows = 0

    for batch in tqdm(
        dataset.to_batches(
            columns=columns,
            batch_size=BATCH_SIZE,
        ),
        desc="RS-specific batches",
        unit="batch",
    ):

        data = batch.to_pydict()

        for (
            rs_path,
            introducing_as,
            appearing_ases,
        ) in zip(
            data["rs_path"],
            data["introducing_as"],
            data["appearing_ases"],
        ):

            if rs_path is not None:
                add_to_buffer(
                    buffers,
                    "rs_paths",
                    rs_path,
                    RS_BUCKETS,
                )

            if introducing_as is not None:
                add_to_buffer(
                    buffers,
                    "introducing",
                    introducing_as,
                    ASN_BUCKETS,
                )

            if appearing_ases:
                for asn in appearing_ases:
                    add_to_buffer(
                        buffers,
                        "appearing",
                        asn,
                        ASN_BUCKETS,
                    )

            rows += 1

            if rows % FLUSH_EVERY_ROWS == 0:
                flush_buffers(
                    buffers,
                    bucket_paths,
                )

        flush_buffers(
            buffers,
            bucket_paths,
        )

    return {
        "rs_paths":
            count_unique(
                rs_paths,
                "Unique RS",
            ),

        "introducing_ases":
            count_unique(
                introducing,
                "Introducing ASes",
            ),

        "appearing_ases":
            count_unique(
                appearing,
                "Appearing ASes",
            ),
    }


def compute_observation_stats(files, dataset_name):
    """
    Count unique collectors, peer ASNs, and collector-peer sessions.
    Session identity is (collector, peer_ip), matching the visibility analysis.
    Counts are unique across the complete month.
    """

    print("\n" + "=" * 72)
    print(f"COMPUTING {dataset_name.upper()} OBSERVATION-POINT STATS")
    print("=" * 72)

    collectors = make_bucket_paths(
        f"{dataset_name}_collectors",
        OBS_BUCKETS,
    )
    peer_asns = make_bucket_paths(
        f"{dataset_name}_peer_asns",
        OBS_BUCKETS,
    )
    sessions = make_bucket_paths(
        f"{dataset_name}_sessions",
        OBS_BUCKETS,
    )

    bucket_paths = {
        "collectors": collectors,
        "peer_asns": peer_asns,
        "sessions": sessions,
    }

    dataset = ds.dataset(
        [str(p) for p in files],
        format="parquet",
    )

    required = {"collector", "peer_asn", "peer_ip"}
    available = set(dataset.schema.names)
    missing = required - available

    if missing:
        raise RuntimeError(
            f"{dataset_name}: missing required columns "
            f"{sorted(missing)}. Available columns: "
            f"{sorted(available)}"
        )

    buffers = new_buffers()
    rows = 0

    for batch in tqdm(
        dataset.to_batches(
            columns=["collector", "peer_asn", "peer_ip"],
            batch_size=BATCH_SIZE,
        ),
        desc=f"{dataset_name} observation batches",
        unit="batch",
    ):
        data = batch.to_pydict()

        for collector, peer_asn, peer_ip in zip(
            data["collector"],
            data["peer_asn"],
            data["peer_ip"],
        ):
            rows += 1

            if collector is not None:
                add_to_buffer(
                    buffers,
                    "collectors",
                    collector,
                    OBS_BUCKETS,
                )

            if peer_asn is not None:
                add_to_buffer(
                    buffers,
                    "peer_asns",
                    peer_asn,
                    OBS_BUCKETS,
                )

            if collector is not None and peer_ip is not None:
                add_to_buffer(
                    buffers,
                    "sessions",
                    f"{collector}{SEP}{peer_ip}",
                    OBS_BUCKETS,
                )

            if rows % FLUSH_EVERY_ROWS == 0:
                flush_buffers(
                    buffers,
                    bucket_paths,
                )

        flush_buffers(
            buffers,
            bucket_paths,
        )

    return {
        "rows": rows,
        "collectors": count_unique(
            collectors,
            f"{dataset_name} collectors",
        ),
        "peer_asns": count_unique(
            peer_asns,
            f"{dataset_name} peer ASNs",
        ),
        "sessions": count_unique(
            sessions,
            f"{dataset_name} sessions",
        ),
    }


def print_observation_stats(rib_obs, rs_obs):
    print("\n" + "=" * 80)
    print("OBSERVATION-POINT STATISTICS")
    print("=" * 80)
    print(
        f"{'Metric':<24}"
        f"{'All RIB':>18}"
        f"{'Non-cons':>18}"
        f"{'Non-cons/RIB':>18}"
    )
    print("-" * 80)

    for label, key in [
        ("Rows", "rows"),
        ("Unique collectors", "collectors"),
        ("Unique peer ASNs", "peer_asns"),
        ("Unique sessions", "sessions"),
    ]:
        rib_value = rib_obs[key]
        rs_value = rs_obs[key]

        print(
            f"{label:<24}"
            f"{rib_value:>18,}"
            f"{rs_value:>18,}"
            f"{percentage(rs_value, rib_value):>17.2f}%"
        )

    print("=" * 80)


def save_observation_csv(rib_obs, rs_obs):
    output_file = OUT_DIR / "observation_point_stats.csv"

    with output_file.open("w", newline="") as fh:
        writer = csv.writer(fh)

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
                BASE_DIR.name,
                key,
                rib_obs[key],
                rs_obs[key],
                f"{percentage(rs_obs[key], rib_obs[key]):.6f}",
            ])

    print(f"Saved: {output_file}")


def percentage(
    numerator,
    denominator,
):

    if not denominator:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


def fmt(value):
    return f"{value:,}"


def fmt_noncons(
    value,
    denominator,
):

    return (
        f"{value:,} "
        f"({percentage(value, denominator):.2f}%)"
    )

def print_table(
    title,
    stats,
    route_counts,
    route_prefix,
    rs_specific,
):

    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)

    print(
        f"{'Metric':<24}"
        f"{'AS Paths IPv4':>18}"
        f"{'AS Paths IPv6':>18}"
        f"{'Non-Cons IPv4':>22}"
        f"{'Non-Cons IPv6':>22}"
    )

    print("-" * 104)

    print(
        f"{'Deduplicated routes':<24}"
        f"{fmt(route_counts[f'rib_{route_prefix}_ipv4']):>18}"
        f"{fmt(route_counts[f'rib_{route_prefix}_ipv6']):>18}"
        f"{fmt(route_counts[f'rs_{route_prefix}_ipv4']):>22}"
        f"{fmt(route_counts[f'rs_{route_prefix}_ipv6']):>22}"
    )

    print(
        f"{'Unique paths':<24}"
        f"{fmt(stats['rib_ipv4_paths']):>18}"
        f"{fmt(stats['rib_ipv6_paths']):>18}"
        f"{fmt_noncons(stats['rs_ipv4_paths'], stats['rib_ipv4_paths']):>22}"
        f"{fmt_noncons(stats['rs_ipv6_paths'], stats['rib_ipv6_paths']):>22}"
    )

    print(
        f"{'Unique prefixes':<24}"
        f"{fmt(stats['rib_ipv4_prefixes']):>18}"
        f"{fmt(stats['rib_ipv6_prefixes']):>18}"
        f"{fmt_noncons(stats['rs_ipv4_prefixes'], stats['rib_ipv4_prefixes']):>22}"
        f"{fmt_noncons(stats['rs_ipv6_prefixes'], stats['rib_ipv6_prefixes']):>22}"
    )

    print(
        f"{'Unique origins':<24}"
        f"{fmt(stats['rib_ipv4_origins']):>18}"
        f"{fmt(stats['rib_ipv6_origins']):>18}"
        f"{fmt_noncons(stats['rs_ipv4_origins'], stats['rib_ipv4_origins']):>22}"
        f"{fmt_noncons(stats['rs_ipv6_origins'], stats['rib_ipv6_origins']):>22}"
    )

    print("-" * 104)

    print(
        f"Unique repetition segments : "
        f"{fmt(rs_specific['rs_paths'])}"
    )

    print(
        f"Introducing ASes           : "
        f"{fmt(rs_specific['introducing_ases'])}"
    )

    print(
        f"Appearing ASes             : "
        f"{fmt(rs_specific['appearing_ases'])}"
    )

    print("=" * 104)

def save_comparison_csv(
    pp_stats,
    pop_stats,
    route_counts,
    rs_specific,
):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUT_DIR
        / "dedup_comparison.csv"
    )

    with output_file.open(
        "w",
        newline="",
    ) as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "deduplication",
            "metric",
            "as_paths_ipv4",
            "as_paths_ipv6",
            "noncons_ipv4",
            "noncons_ipv6",
        ])

        for (
            label,
            route_prefix,
            stats,
        ) in [
            (
                "prefix_aspath",
                "pp",
                pp_stats,
            ),
            (
                "prefix_origin_aspath",
                "pop",
                pop_stats,
            ),
        ]:

            writer.writerow([
                label,
                "deduplicated_routes",
                route_counts[
                    f"rib_{route_prefix}_ipv4"
                ],
                route_counts[
                    f"rib_{route_prefix}_ipv6"
                ],
                route_counts[
                    f"rs_{route_prefix}_ipv4"
                ],
                route_counts[
                    f"rs_{route_prefix}_ipv6"
                ],
            ])

            writer.writerow([
                label,
                "unique_paths",
                stats["rib_ipv4_paths"],
                stats["rib_ipv6_paths"],
                stats["rs_ipv4_paths"],
                stats["rs_ipv6_paths"],
            ])

            writer.writerow([
                label,
                "unique_prefixes",
                stats["rib_ipv4_prefixes"],
                stats["rib_ipv6_prefixes"],
                stats["rs_ipv4_prefixes"],
                stats["rs_ipv6_prefixes"],
            ])

            writer.writerow([
                label,
                "unique_origins",
                stats["rib_ipv4_origins"],
                stats["rib_ipv6_origins"],
                stats["rs_ipv4_origins"],
                stats["rs_ipv6_origins"],
            ])

        writer.writerow([
            "rs_specific",
            "unique_repetition_segments",
            "",
            "",
            rs_specific["rs_paths"],
            "",
        ])

        writer.writerow([
            "rs_specific",
            "introducing_ases",
            "",
            "",
            rs_specific["introducing_ases"],
            "",
        ])

        writer.writerow([
            "rs_specific",
            "appearing_ases",
            "",
            "",
            rs_specific["appearing_ases"],
            "",
        ])

    print(
        f"\nSaved: {output_file}"
    )

def process_month(month):
    global BASE_DIR, POISON_DIR, OUT_DIR, TMP_DIR

    BASE_DIR = BASE_ROOT / month
    POISON_DIR = BASE_DIR / "poisoning"
    OUT_DIR = BASE_ROOT / "dataset.stats" / month
    TMP_DIR = OUT_DIR / "tmp"

    print("\n" + "#" * 72)
    print(f"MONTH: {month}")
    print("#" * 72)
    print(f"Input directory : {BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")

    ensure_clean_dir(
        TMP_DIR
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rib_files = sorted(
        p
        for p in BASE_DIR.glob(
            RIB_PATTERN
        )
        if p.is_file()
    )

    poison_files = sorted(
        p
        for p in POISON_DIR.glob(
            POISON_PATTERN
        )
        if p.is_file()
    )

    print("=" * 72)
    print("DEDUPLICATION COMPARISON TEST")
    print("=" * 72)
    print(
        f"RIB files      : "
        f"{len(rib_files):,}"
    )
    print(
        f"Non-cons files : "
        f"{len(poison_files):,}"
    )
    print("=" * 72)

    if not rib_files:
        raise RuntimeError(
            "No RIB files found."
        )

    if not poison_files:
        raise RuntimeError(
            "No non-cons files found."
        )

    if len(rib_files) != len(poison_files):
        raise RuntimeError(
            "RIB/non-cons file count mismatch: "
            f"{len(rib_files)} RIB files vs "
            f"{len(poison_files)} non-cons files."
        )

    route_buckets = (
        prepare_route_buckets()
    )

    spill_rib_route_keys(
        rib_files,
        route_buckets,
    )

    spill_rs_route_keys(
        poison_files,
        route_buckets,
    )

    route_counts = count_route_keys(
        route_buckets
    )

    pp_properties = (
        prepare_property_buckets(
            "pp"
        )
    )

    properties_from_pp(
        route_buckets["rib_pp_ipv4"],
        pp_properties,
        "rib",
        "ipv4",
    )

    properties_from_pp(
        route_buckets["rib_pp_ipv6"],
        pp_properties,
        "rib",
        "ipv6",
    )

    properties_from_pp(
        route_buckets["rs_pp_ipv4"],
        pp_properties,
        "rs",
        "ipv4",
    )

    properties_from_pp(
        route_buckets["rs_pp_ipv6"],
        pp_properties,
        "rs",
        "ipv6",
    )

    add_pp_origins(
        rib_files,
        pp_properties,
        "rib",
    )

    add_pp_origins(
        poison_files,
        pp_properties,
        "rs",
    )

    pp_stats = compute_property_stats(
        pp_properties
    )

    pop_properties = (
        prepare_property_buckets(
            "pop"
        )
    )

    properties_from_pop(
        route_buckets["rib_pop_ipv4"],
        pop_properties,
        "rib",
        "ipv4",
    )

    properties_from_pop(
        route_buckets["rib_pop_ipv6"],
        pop_properties,
        "rib",
        "ipv6",
    )

    properties_from_pop(
        route_buckets["rs_pop_ipv4"],
        pop_properties,
        "rs",
        "ipv4",
    )

    properties_from_pop(
        route_buckets["rs_pop_ipv6"],
        pop_properties,
        "rs",
        "ipv6",
    )

    pop_stats = compute_property_stats(
        pop_properties
    )

    rs_specific = (
        compute_rs_specific_stats(
            poison_files
        )
    )

    rib_obs = compute_observation_stats(
        rib_files,
        "rib",
    )

    rs_obs = compute_observation_stats(
        poison_files,
        "noncons",
    )

    print_observation_stats(
        rib_obs,
        rs_obs,
    )

    save_observation_csv(
        rib_obs,
        rs_obs,
    )

    print_table(
        "DEDUPLICATION: PREFIX + AS PATH",
        pp_stats,
        route_counts,
        "pp",
        rs_specific,
    )

    print_table(
        "DEDUPLICATION: PREFIX + ORIGIN + AS PATH",
        pop_stats,
        route_counts,
        "pop",
        rs_specific,
    )

    save_comparison_csv(
        pp_stats,
        pop_stats,
        route_counts,
        rs_specific,
    )

    shutil.rmtree(
        TMP_DIR,
        ignore_errors=True,
    )


def main():
    for month in MONTHS:
        process_month(month)


if __name__ == "__main__":
    main()
