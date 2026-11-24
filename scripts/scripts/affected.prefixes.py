#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/data/prefix.aspth/prefix.aspth.parquet")

RAW_DIR = BASE_DIR

# Filtered RS observations.
SEGMENTS_DIR = BASE_DIR / "poisoning" / "segments.filtered"

# Completely separate output namespace for the filtered analysis.
OUT_BASE = BASE_DIR / "rs.analysis" / "prefix_ratio.filtered"

DATA_DIR = OUT_BASE / "data"
FIG_DIR = OUT_BASE / "figures"

PER_DAY_AFFECTED_DIR = DATA_DIR / "per_day_rs_prefixes"
PER_DAY_RATIO_DIR = DATA_DIR / "per_day_prefix_ratios"

for directory in [
    DATA_DIR,
    FIG_DIR,
    PER_DAY_AFFECTED_DIR,
    PER_DAY_RATIO_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


OUT_ASN_DAY_IP = (
    DATA_DIR / "rs_prefix_ratio.asn_day_ip.filtered.parquet"
)

OUT_ASN_DAY = (
    DATA_DIR / "rs_prefix_ratio.asn_day.filtered.parquet"
)

OUT_DAILY = (
    DATA_DIR / "rs_prefix_ratio.daily_summary.filtered.parquet"
)

OUT_STATS = (
    DATA_DIR / "rs_prefix_ratio.build_stats.filtered.csv"
)

OUT_AFFECTED_STATS = (
    DATA_DIR / "rs_prefix_ratio.affected_build_stats.filtered.csv"
)


RAW_FILE_RE = re.compile(
    r"^prefix\.aspath\.(\d{4}-\d{2}-\d{2})\.parquet$"
)

SEGMENT_FILE_RE = re.compile(
    r"^segments\.(\d{4}-\d{2}-\d{2})\.parquet$"
)


START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2024-12-31")

EXPECTED_DATES = 396

BATCH_SIZE = 1_000_000
NUM_WORKERS = min(8, max(1, cpu_count() - 1))

# Set True only when intentionally rebuilding existing outputs.
FORCE_REBUILD_AFFECTED = False
FORCE_RECOMPUTE_PER_DAY = False
FORCE_REBUILD_FINAL = True


ASN_RE = re.compile(r"\d+")
AS_SET_TOKEN_RE = re.compile(r"^\{([^{}]+)\}$")


# ============================================================
# BASIC HELPERS
# ============================================================

def classify_ip_version(prefix: str) -> str:
    return "IPv6" if ":" in str(prefix) else "IPv4"


def get_first_existing(
    schema_names: list[str],
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in schema_names:
            return col
    return None


def extract_date(
    path: Path,
    file_re: re.Pattern,
) -> str:
    match = file_re.match(path.name)

    if not match:
        raise ValueError(
            f"Cannot extract date from filename: {path.name}"
        )

    return match.group(1)


def get_files(
    directory: Path,
    file_re: re.Pattern,
) -> dict[str, Path]:

    files = {}

    for path in directory.iterdir():
        if path.is_file() and file_re.match(path.name):
            date = extract_date(path, file_re)
            files[date] = path

    return dict(sorted(files.items()))


def in_study_period(date: str) -> bool:
    ts = pd.Timestamp(date)
    return START_DATE <= ts <= END_DATE


def affected_path(date: str) -> Path:
    return (
        PER_DAY_AFFECTED_DIR
        / f"rs_prefixes.{date}.parquet"
    )


def ratio_path(date: str) -> Path:
    return (
        PER_DAY_RATIO_DIR
        / f"ratio.{date}.parquet"
    )


def stats_path(date: str) -> Path:
    return (
        PER_DAY_RATIO_DIR
        / f"stats.{date}.csv"
    )


# ============================================================
# ASN / RS PARSING
# ============================================================

def parse_asn(value: Any) -> int | None:

    if value is None:
        return None

    s = str(value).strip()

    if not s or s.lower() in {
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return None

    values = ASN_RE.findall(s)

    if len(values) != 1:
        return None

    return int(values[0])


def parse_segment(
    value: Any,
) -> tuple[int, ...] | None:
    """
    Parse and validate one repetition segment.

    Expected form:
        A X A
        A X Y A
        ...

    Requirements:
      * at least three ASNs
      * first ASN == last ASN
      * at least one different ASN between them
      * no consecutive occurrence of the repeated ASN

    The returned tuple retains the complete RS.
    """

    if value is None:
        return None

    raw = str(value).strip()

    if not raw:
        return None

    tokens = raw.split()

    if len(tokens) < 3:
        return None

    try:
        segment = tuple(int(token) for token in tokens)
    except ValueError:
        return None

    if segment[0] != segment[-1]:
        return None

    repeated_as = segment[0]

    middle = segment[1:-1]

    if not any(asn != repeated_as for asn in middle):
        return None

    # Preserve the validation used in the previous RS analyses.
    for i in range(len(segment) - 1):
        if (
            segment[i] == repeated_as
            and segment[i + 1] == repeated_as
        ):
            return None

    return segment


def get_appearing_ases(
    segment: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Return the unique appearing ASes in an RS.

    Appearing ASes are the ASes strictly between the first and
    last occurrence of the introducing/repeated AS.

    Example:
        131477 65511 140096 131477
        -> (65511, 140096)

    If an AS occurs multiple times inside the RS, it is returned
    once because the downstream metric is prefix coverage per AS.
    """

    if len(segment) < 3:
        return tuple()

    introducing_as = segment[0]

    appearing = {
        int(asn)
        for asn in segment[1:-1]
        if int(asn) != introducing_as
    }

    return tuple(sorted(appearing))


# ============================================================
# RAW AS-PATH NORMALIZATION
# ============================================================

def normalize_raw_aspath_origin_single_as_set(
    value: Any,
) -> tuple[list[int], bool, str]:
    """
    Preserve the raw-path handling used by the original analysis.

    Accepted:
        265 765 876 {897}
        -> [265, 765, 876, 897]

    Rejected:
        265 765 {876} 897
        265 765 876 {897 860}
        malformed AS-SET tokens
    """

    if value is None:
        return [], False, "empty"

    if isinstance(value, list):
        tokens = [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]
    else:
        tokens = str(value).strip().split()

    if not tokens:
        return [], False, "empty"

    cleaned = []

    for idx, token in enumerate(tokens):

        is_last = idx == len(tokens) - 1
        match = AS_SET_TOKEN_RE.match(token)

        if match:

            if not is_last:
                return [], False, "non_origin_as_set"

            asns = ASN_RE.findall(match.group(1))

            if len(asns) != 1:
                return [], False, "multi_as_origin_as_set"

            cleaned.append(int(asns[0]))
            continue

        if "{" in token or "}" in token:
            return [], False, "malformed_as_set"

        if not token.isdigit():
            return [], False, "invalid_token"

        cleaned.append(int(token))

    if not cleaned:
        return [], False, "empty"

    return cleaned, True, "ok"


# ============================================================
# STEP 1
# BUILD ROLE-AWARE PREFIX SETS FROM FILTERED RS FILES
# ============================================================

def build_one_affected_file(
    segment_path: Path,
    date: str,
) -> dict:
    """
    Build the numerator candidates for one sampled day.

    For every retained RS:

      Origin role:
          associate the prefix with the origin AS.

      Appearing role:
          associate the prefix with every unique AS strictly
          inside the RS.

    Prefixes are deduplicated for each:
        (date, role, ASN, IP version)

    rs_class is retained for auditing and compatibility with the
    original analysis.
    """

    out_path = affected_path(date)

    if (
        out_path.exists()
        and not FORCE_REBUILD_AFFECTED
    ):
        existing = pd.read_parquet(out_path)

        return {
            "date": date,
            "status": "cached",
            "segment_file": str(segment_path),
            "output_rows": len(existing),
        }

    pf = pq.ParquetFile(segment_path)
    schema = pf.schema.names

    prefix_col = get_first_existing(
        schema,
        ["prefix"],
    )

    segment_col = get_first_existing(
        schema,
        [
            "poisoned_segment",
            "segment",
            "rs",
            "rs_path",
        ],
    )

    repeated_col = get_first_existing(
        schema,
        [
            "repeated_asn",
            "introducing_asn",
            "introducing_as",
        ],
    )

    origin_col = get_first_existing(
        schema,
        [
            "origin_as",
            "origin",
            "origin_asn",
        ],
    )

    aspath_col = get_first_existing(
        schema,
        [
            "aspath",
            "as_path",
            "clean_aspath",
        ],
    )

    if prefix_col is None:
        raise ValueError(
            f"{segment_path} has no prefix column"
        )

    if segment_col is None:
        raise ValueError(
            f"{segment_path} has no RS/segment column. "
            f"Columns: {schema}"
        )

    if origin_col is None and aspath_col is None:
        raise ValueError(
            f"{segment_path} has neither an origin-AS "
            f"column nor an AS-path column"
        )

    columns = [prefix_col, segment_col]

    if repeated_col is not None:
        columns.append(repeated_col)

    if origin_col is not None:
        columns.append(origin_col)

    if (
        aspath_col is not None
        and aspath_col not in columns
    ):
        columns.append(aspath_col)

    records = set()

    stats = {
        "date": date,
        "status": "computed",
        "segment_file": str(segment_path),
        "segment_rows_seen": 0,
        "segment_rows_used": 0,
        "segment_rows_invalid_segment": 0,
        "segment_rows_missing_prefix": 0,
        "segment_rows_missing_origin": 0,
        "segment_rows_repeated_mismatch": 0,
        "origin_role_records_before_dedup": 0,
        "appearing_role_records_before_dedup": 0,
        "rs_with_zero_appearing_ases": 0,
    }

    for batch in pf.iter_batches(
        columns=columns,
        batch_size=BATCH_SIZE,
    ):

        data = batch.to_pydict()
        n = len(data[prefix_col])

        for i in range(n):

            stats["segment_rows_seen"] += 1

            prefix_value = data[prefix_col][i]

            if (
                prefix_value is None
                or not str(prefix_value).strip()
            ):
                stats["segment_rows_missing_prefix"] += 1
                continue

            prefix = str(prefix_value).strip()
            ip_version = classify_ip_version(prefix)

            segment = parse_segment(
                data[segment_col][i]
            )

            if segment is None:
                stats["segment_rows_invalid_segment"] += 1
                continue

            introducing_as = int(segment[0])

            # If the filtered data contain an explicit repeated-AS
            # field, require it to agree with the RS boundaries.
            if repeated_col is not None:

                repeated_as = parse_asn(
                    data[repeated_col][i]
                )

                if (
                    repeated_as is not None
                    and repeated_as != introducing_as
                ):
                    stats[
                        "segment_rows_repeated_mismatch"
                    ] += 1
                    continue

            # Determine the origin AS.
            origin_as = None

            if origin_col is not None:
                origin_as = parse_asn(
                    data[origin_col][i]
                )

            # Fallback to the last ASN in the complete AS path.
            if (
                origin_as is None
                and aspath_col is not None
            ):
                raw_path = data[aspath_col][i]

                if isinstance(raw_path, list):
                    path_tokens = [
                        str(x).strip()
                        for x in raw_path
                        if str(x).strip()
                    ]
                else:
                    path_tokens = (
                        str(raw_path).strip().split()
                        if raw_path is not None
                        else []
                    )

                if path_tokens:
                    origin_as = parse_asn(
                        path_tokens[-1]
                    )

            if origin_as is None:
                stats["segment_rows_missing_origin"] += 1
                continue

            rs_class = (
                "origin_rs"
                if introducing_as == origin_as
                else "internal_rs"
            )

            # ----------------------------------------------------
            # Origin-AS role
            # ----------------------------------------------------

            records.add(
                (
                    rs_class,
                    "origin",
                    int(origin_as),
                    ip_version,
                    prefix,
                )
            )

            stats[
                "origin_role_records_before_dedup"
            ] += 1

            # ----------------------------------------------------
            # Appearing-AS role
            # ----------------------------------------------------

            appearing_ases = get_appearing_ases(
                segment
            )

            if not appearing_ases:
                stats[
                    "rs_with_zero_appearing_ases"
                ] += 1

            for appearing_as in appearing_ases:

                records.add(
                    (
                        rs_class,
                        "appearing",
                        int(appearing_as),
                        ip_version,
                        prefix,
                    )
                )

                stats[
                    "appearing_role_records_before_dedup"
                ] += 1

            stats["segment_rows_used"] += 1

    rows = [
        {
            "date": date,
            "rs_class": rs_class,
            "role": role,
            "asn": asn,
            "ip_version": ip_version,
            "prefix": prefix,
        }
        for (
            rs_class,
            role,
            asn,
            ip_version,
            prefix,
        ) in records
    ]

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            [
                "date",
                "rs_class",
                "role",
                "asn",
                "ip_version",
                "prefix",
            ]
        ).reset_index(drop=True)

    out.to_parquet(out_path, index=False)

    stats["output_rows"] = len(out)

    if not out.empty:
        stats["output_origin_rows"] = int(
            (out["role"] == "origin").sum()
        )
        stats["output_appearing_rows"] = int(
            (out["role"] == "appearing").sum()
        )
        stats["output_origin_asns"] = int(
            out.loc[
                out["role"] == "origin",
                "asn",
            ].nunique()
        )
        stats["output_appearing_asns"] = int(
            out.loc[
                out["role"] == "appearing",
                "asn",
            ].nunique()
        )
    else:
        stats["output_origin_rows"] = 0
        stats["output_appearing_rows"] = 0
        stats["output_origin_asns"] = 0
        stats["output_appearing_asns"] = 0

    return stats


def build_per_day_affected_prefixes(
    segment_files: dict[str, Path],
) -> None:

    all_stats = []

    for date, segment_path in tqdm(
        segment_files.items(),
        total=len(segment_files),
        desc="Building filtered RS prefix sets",
        unit="day",
    ):

        stats = build_one_affected_file(
            segment_path,
            date,
        )

        all_stats.append(stats)

    pd.DataFrame(all_stats).to_csv(
        OUT_AFFECTED_STATS,
        index=False,
    )


# ============================================================
# STEP 2
# RAW DENOMINATOR SCAN
# ============================================================

def process_day(
    job: tuple[str, str],
) -> tuple[str, bool]:

    date, raw_path_str = job

    raw_path = Path(raw_path_str)

    out_ratio = ratio_path(date)
    out_stats = stats_path(date)

    if (
        out_ratio.exists()
        and out_stats.exists()
        and not FORCE_RECOMPUTE_PER_DAY
    ):
        return date, False

    aff_path = affected_path(date)

    if not aff_path.exists():
        raise FileNotFoundError(
            f"Missing affected-prefix file: {aff_path}"
        )

    affected = pd.read_parquet(aff_path)

    if affected.empty:

        pd.DataFrame().to_parquet(
            out_ratio,
            index=False,
        )

        pd.DataFrame(
            [{
                "date": date,
                "status": "empty_affected_file",
                "raw_file": str(raw_path),
            }]
        ).to_csv(
            out_stats,
            index=False,
        )

        return date, True

    affected["asn"] = (
        affected["asn"].astype(int)
    )

    affected["prefix"] = (
        affected["prefix"].astype(str)
    )

    affected_sets = defaultdict(set)

    for row in affected.itertuples(
        index=False
    ):

        key = (
            str(row.rs_class),
            str(row.role),
            int(row.asn),
            str(row.ip_version),
        )

        affected_sets[key].add(
            str(row.prefix)
        )

    # ASes whose denominators actually need to be computed.
    target_origin_by_ip = defaultdict(set)
    target_appearing_by_ip = defaultdict(set)

    for (
        _rs_class,
        role,
        asn,
        ip_version,
    ) in affected_sets:

        if role == "origin":
            target_origin_by_ip[
                ip_version
            ].add(asn)

        elif role == "appearing":
            target_appearing_by_ip[
                ip_version
            ].add(asn)

    raw_origin_prefixes = defaultdict(set)
    raw_appearing_prefixes = defaultdict(set)

    pf = pq.ParquetFile(raw_path)
    schema = pf.schema.names

    prefix_col = "prefix"

    aspath_col = get_first_existing(
        schema,
        [
            "as_path",
            "aspath",
            "clean_aspath",
        ],
    )

    if prefix_col not in schema:
        raise ValueError(
            f"{raw_path} missing prefix column"
        )

    if aspath_col is None:
        raise ValueError(
            f"{raw_path} missing AS-path column"
        )

    stats = {
        "date": date,
        "status": "processed",
        "raw_file": str(raw_path),
        "affected_file": str(aff_path),
        "affected_rows": len(affected),
        "affected_keys": len(affected_sets),
        "target_origin_asns": sum(
            len(x)
            for x in target_origin_by_ip.values()
        ),
        "target_appearing_asns": sum(
            len(x)
            for x in target_appearing_by_ip.values()
        ),
        "raw_rows_seen": 0,
        "raw_rows_used": 0,
        "raw_rows_origin_target": 0,
        "raw_rows_appearing_target": 0,
        "raw_rows_invalid_path": 0,
        "raw_rows_non_origin_as_set_skipped": 0,
        "raw_rows_multi_as_origin_as_set_skipped": 0,
        "raw_rows_malformed_as_set_skipped": 0,
        "raw_rows_single_as_origin_as_set_used": 0,
    }

    for batch in pf.iter_batches(
        columns=[prefix_col, aspath_col],
        batch_size=BATCH_SIZE,
    ):

        data = batch.to_pydict()

        for (
            prefix_value,
            raw_path_value,
        ) in zip(
            data[prefix_col],
            data[aspath_col],
        ):

            stats["raw_rows_seen"] += 1

            if prefix_value is None:
                continue

            prefix = str(prefix_value).strip()

            if not prefix:
                continue

            if isinstance(raw_path_value, list):
                raw_tokens = [
                    str(x).strip()
                    for x in raw_path_value
                    if str(x).strip()
                ]
            else:
                raw_tokens = (
                    str(raw_path_value).strip().split()
                    if raw_path_value is not None
                    else []
                )

            aspath, usable, reason = (
                normalize_raw_aspath_origin_single_as_set(
                    raw_path_value
                )
            )

            if not usable:

                if reason == "non_origin_as_set":
                    stats[
                        "raw_rows_non_origin_as_set_skipped"
                    ] += 1

                elif reason == "multi_as_origin_as_set":
                    stats[
                        "raw_rows_multi_as_origin_as_set_skipped"
                    ] += 1

                elif reason == "malformed_as_set":
                    stats[
                        "raw_rows_malformed_as_set_skipped"
                    ] += 1

                else:
                    stats[
                        "raw_rows_invalid_path"
                    ] += 1

                continue

            if (
                raw_tokens
                and AS_SET_TOKEN_RE.match(
                    raw_tokens[-1]
                )
            ):
                stats[
                    "raw_rows_single_as_origin_as_set_used"
                ] += 1

            stats["raw_rows_used"] += 1

            ip_version = classify_ip_version(
                prefix
            )

            origin_as = int(aspath[-1])

            # ----------------------------------------------------
            # Origin denominator
            # ----------------------------------------------------

            if origin_as in target_origin_by_ip.get(
                ip_version,
                set(),
            ):

                raw_origin_prefixes[
                    (origin_as, ip_version)
                ].add(prefix)

                stats[
                    "raw_rows_origin_target"
                ] += 1

            # ----------------------------------------------------
            # Appearing-AS denominator
            #
            # Preserve original metric:
            # all prefixes whose AS paths contain the AS.
            # ----------------------------------------------------

            appearing_targets = (
                target_appearing_by_ip.get(
                    ip_version,
                    set(),
                )
            )

            if appearing_targets:

                path_asns = set(aspath)

                hits = (
                    path_asns
                    & appearing_targets
                )

                if hits:

                    stats[
                        "raw_rows_appearing_target"
                    ] += 1

                    for asn in hits:

                        raw_appearing_prefixes[
                            (asn, ip_version)
                        ].add(prefix)

    # ========================================================
    # COMPUTE RATIOS
    # ========================================================

    rows = []

    for (
        rs_class,
        role,
        asn,
        ip_version,
    ), affected_set in affected_sets.items():

        if role == "origin":

            raw_set = raw_origin_prefixes.get(
                (asn, ip_version),
                set(),
            )

        elif role == "appearing":

            raw_set = raw_appearing_prefixes.get(
                (asn, ip_version),
                set(),
            )

        else:
            raise ValueError(
                f"Unexpected role: {role}"
            )

        matched = affected_set & raw_set
        missing = affected_set - raw_set

        total_prefixes = len(raw_set)
        rs_prefixes = len(matched)

        ratio = (
            rs_prefixes
            / total_prefixes
            *00.0
            if total_prefixes > 0
            else 0.0
        )

        rows.append(
            {
                "date": date,
                "rs_class": rs_class,
                "role": role,
                "asn": asn,
                "ip_version": ip_version,
                "rs_prefixes": rs_prefixes,
                "total_prefixes": total_prefixes,
                "rs_prefixes_before_raw_match": len(
                    affected_set
                ),
                "rs_prefixes_not_in_raw": len(
                    missing
                ),
                "rs_prefix_ratio_pct": ratio,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:

        out = out.sort_values(
            [
                "date",
                "rs_class",
                "role",
                "asn",
                "ip_version",
            ]
        ).reset_index(drop=True)

        bad = out[
            out["rs_prefix_ratio_pct"]
            > 100.000001
        ]

        if not bad.empty:

            bad_file = (
                PER_DAY_RATIO_DIR
                / f"bad_ratio_gt_100.{date}.csv"
            )

            bad.to_csv(
                bad_file,
                index=False,
            )

            raise RuntimeError(
                f"{date}: found impossible "
                f"ratio >100%. Saved: {bad_file}"
            )

    out.to_parquet(
        out_ratio,
        index=False,
    )

    stats["output_rows"] = len(out)

    if not out.empty:

        stats["output_rs_prefixes"] = int(
            out["rs_prefixes"].sum()
        )

        stats["output_total_prefixes"] = int(
            out["total_prefixes"].sum()
        )

        stats["output_prefixes_not_in_raw"] = int(
            out["rs_prefixes_not_in_raw"].sum()
        )

    else:

        stats["output_rs_prefixes"] = 0
        stats["output_total_prefixes"] = 0
        stats["output_prefixes_not_in_raw"] = 0

    pd.DataFrame([stats]).to_csv(
        out_stats,
        index=False,
    )

    return date, True


# ============================================================
# STEP 3
# BUILD FINAL ASN-DAY TABLES
# ============================================================

def build_final(
    raw_files: dict[str, Path],
    segment_files: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if (
        OUT_ASN_DAY_IP.exists()
        and OUT_ASN_DAY.exists()
        and OUT_DAILY.exists()
        and not FORCE_REBUILD_FINAL
        and not FORCE_RECOMPUTE_PER_DAY
    ):

        print(
            f"[INFO] Loading existing final cache: "
            f"{OUT_ASN_DAY}"
        )

        return (
            pd.read_parquet(OUT_ASN_DAY_IP),
            pd.read_parquet(OUT_ASN_DAY),
        )

    dates = sorted(segment_files)

    jobs = []

    missing_raw = []

    for date in dates:

        raw_path = raw_files.get(date)

        if raw_path is None:
            missing_raw.append(date)
            continue

        jobs.append(
            (date, str(raw_path))
        )

    if missing_raw:
        raise RuntimeError(
            "Filtered RS dates without corresponding raw "
            f"BGP files: {missing_raw[:20]}"
        )

    print(
        f"Scanning raw denominators for "
        f"{len(jobs):,} sampled days "
        f"using {NUM_WORKERS} workers"
    )

    with Pool(
        processes=NUM_WORKERS
    ) as pool:

        for date, recomputed in tqdm(
            pool.imap_unordered(
                process_day,
                jobs,
                chunksize=1,
            ),
            total=len(jobs),
            desc="Computing daily prefix ratios",
            unit="day",
        ):
            pass

    ratio_files = sorted(
        PER_DAY_RATIO_DIR.glob(
            "ratio.*.parquet"
        )
    )

    parts = []

    for path in tqdm(
        ratio_files,
        desc="Combining daily ratios",
        unit="file",
    ):

        df = pd.read_parquet(path)

        if not df.empty:
            parts.append(df)

    if not parts:
        raise RuntimeError(
            "No non-empty daily ratio files found"
        )

    asn_day_ip = pd.concat(
        parts,
        ignore_index=True,
    )

    asn_day_ip["date"] = pd.to_datetime(
        asn_day_ip["date"]
    )

    # Strictly retain the study period.
    asn_day_ip = asn_day_ip[
        (asn_day_ip["date"] >= START_DATE)
        & (asn_day_ip["date"] <= END_DATE)
    ].copy()

    asn_day_ip = asn_day_ip.sort_values(
        [
            "date",
            "rs_class",
            "role",
            "asn",
            "ip_version",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Combine IPv4 + IPv6 exactly as the original analysis did.
    # --------------------------------------------------------

    asn_day = (
        asn_day_ip.groupby(
            [
                "date",
                "rs_class",
                "role",
                "asn",
            ],
            as_index=False,
        )
        .agg(
            rs_prefixes=(
                "rs_prefixes",
                "sum",
            ),
            total_prefixes=(
                "total_prefixes",
                "sum",
            ),
            rs_prefixes_before_raw_match=(
                "rs_prefixes_before_raw_match",
                "sum",
            ),
            rs_prefixes_not_in_raw=(
                "rs_prefixes_not_in_raw",
                "sum",
            ),
        )
        .sort_values(
            [
                "date",
                "rs_class",
                "role",
                "asn",
            ]
        )
        .reset_index(drop=True)
    )

    asn_day["rs_prefix_ratio_pct"] = (
        asn_day["rs_prefixes"]
        / asn_day["total_prefixes"]
        * 100.0
    )

    asn_day.loc[
        asn_day["total_prefixes"] <= 0,
        "rs_prefix_ratio_pct",
    ] = 0.0

    # --------------------------------------------------------
    # Daily audit table
    # --------------------------------------------------------

    daily = (
        asn_day.groupby(
            [
                "date",
                "rs_class",
                "role",
            ],
            as_index=False,
        )
        .agg(
            rs_prefixes=(
                "rs_prefixes",
                "sum",
            ),
            total_prefixes=(
                "total_prefixes",
                "sum",
            ),
            unique_asns=(
                "asn",
                "nunique",
            ),
            rs_prefixes_before_raw_match=(
                "rs_prefixes_before_raw_match",
                "sum",
            ),
            rs_prefixes_not_in_raw=(
                "rs_prefixes_not_in_raw",
                "sum",
            ),
        )
        .sort_values(
            [
                "date",
                "rs_class",
                "role",
            ]
        )
        .reset_index(drop=True)
    )

    daily["rs_prefix_ratio_pct"] = (
        daily["rs_prefixes"]
        / daily["total_prefixes"]
        * 100.0
    )

    daily.loc[
        daily["total_prefixes"] <= 0,
        "rs_prefix_ratio_pct",
    ] = 0.0

    # --------------------------------------------------------
    # Final validations
    # --------------------------------------------------------

    if (
        asn_day["rs_prefix_ratio_pct"]
        > 100.000001
    ).any():

        raise RuntimeError(
            "Final ASN-day table contains "
            "ratio >100%"
        )

    if (
        asn_day["rs_prefixes"]
        > asn_day["total_prefixes"]
    ).any():

        raise RuntimeError(
            "Final ASN-day table contains "
            "numerator > denominator"
        )

    asn_day_ip.to_parquet(
        OUT_ASN_DAY_IP,
        index=False,
    )

    asn_day.to_parquet(
        OUT_ASN_DAY,
        index=False,
    )

    daily.to_parquet(
        OUT_DAILY,
        index=False,
    )

    stats_files = sorted(
        PER_DAY_RATIO_DIR.glob(
            "stats.*.csv"
        )
    )

    if stats_files:

        stats = pd.concat(
            [
                pd.read_csv(path)
                for path in stats_files
            ],
            ignore_index=True,
        )

        stats.to_csv(
            OUT_STATS,
            index=False,
        )

    return asn_day_ip, asn_day


# ============================================================
# AUDIT
# ============================================================

def print_audit(
    raw_files: dict[str, Path],
    segment_files: dict[str, Path],
    asn_day_ip: pd.DataFrame,
    asn_day: pd.DataFrame,
) -> None:

    dates = sorted(segment_files)

    print()
    print("=" * 78)
    print("FILTERED RS PREFIX-RATIO ANALYSIS")
    print("=" * 78)

    print(
        f"Filtered segment files       : "
        f"{len(segment_files):,}"
    )

    print(
        f"Raw files in study period    : "
        f"{len(raw_files):,}"
    )

    if dates:
        print(
            f"First filtered date          : "
            f"{dates[0]}"
        )
        print(
            f"Last filtered date           : "
            f"{dates[-1]}"
        )

    print(
        f"ASN-day-IP rows              : "
        f"{len(asn_day_ip):,}"
    )

    print(
        f"ASN-day rows                 : "
        f"{len(asn_day):,}"
    )

    print()

    summary = (
        asn_day.groupby(
            ["rs_class", "role"]
        )
        .agg(
            rows=(
                "rs_prefix_ratio_pct",
                "size",
            ),
            unique_asns=(
                "asn",
                "nunique",
            ),
            observed_days=(
                "date",
                "nunique",
            ),
            median_ratio=(
                "rs_prefix_ratio_pct",
                "median",
            ),
            mean_ratio=(
                "rs_prefix_ratio_pct",
                "mean",
            ),
            max_ratio=(
                "rs_prefix_ratio_pct",
                "max",
            ),
            rs_prefixes=(
                "rs_prefixes",
                "sum",
            ),
            total_prefixes=(
                "total_prefixes",
                "sum",
            ),
            not_in_raw=(
                "rs_prefixes_not_in_raw",
                "sum",
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print()
    print("Output directory:")
    print(OUT_BASE)
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not SEGMENTS_DIR.exists():
        raise FileNotFoundError(
            SEGMENTS_DIR
        )

    raw_files_all = get_files(
        RAW_DIR,
        RAW_FILE_RE,
    )

    segment_files_all = get_files(
        SEGMENTS_DIR,
        SEGMENT_FILE_RE,
    )

    raw_files = {
        date: path
        for date, path in raw_files_all.items()
        if in_study_period(date)
    }

    segment_files = {
        date: path
        for date, path in segment_files_all.items()
        if in_study_period(date)
    }

    if not segment_files:
        raise RuntimeError(
            "No filtered segment files found "
            "for 2014-2024"
        )

    dates = sorted(segment_files)

    # --------------------------------------------------------
    # Strict study-period validation
    # --------------------------------------------------------

    if len(segment_files) != EXPECTED_DATES:
        raise RuntimeError(
            f"Expected {EXPECTED_DATES} filtered "
            f"segment files, found "
            f"{len(segment_files)}"
        )

    if dates[0] != "2014-01-01":
        raise RuntimeError(
            f"Unexpected first date: {dates[0]}"
        )

    if dates[-1] != "2024-12-31":
        raise RuntimeError(
            f"Unexpected last date: {dates[-1]}"
        )

    missing_raw = sorted(
        set(segment_files)
        - set(raw_files)
    )

    if missing_raw:
        raise RuntimeError(
            f"Missing raw files for "
            f"{len(missing_raw)} dates: "
            f"{missing_raw[:20]}"
        )

    print("=" * 78)
    print("INPUT VALIDATION")
    print("=" * 78)
    print(
        f"Filtered RS files : "
        f"{len(segment_files):,}"
    )
    print(
        f"First date        : "
        f"{dates[0]}"
    )
    print(
        f"Last date         : "
        f"{dates[-1]}"
    )
    print(
        f"Workers           : "
        f"{NUM_WORKERS}"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # Numerators from filtered RS observations
    # --------------------------------------------------------

    build_per_day_affected_prefixes(
        segment_files
    )

    # --------------------------------------------------------
    # Denominators from unchanged raw BGP observations
    # --------------------------------------------------------

    asn_day_ip, asn_day = build_final(
        raw_files,
        segment_files,
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    print_audit(
        raw_files,
        segment_files,
        asn_day_ip,
        asn_day,
    )


if __name__ == "__main__":
    main()
