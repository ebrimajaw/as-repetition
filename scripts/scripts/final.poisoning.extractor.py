# Poisoning extractor final
#!/usr/bin/env python3

from pathlib import Path
import re
import time
import traceback
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(
    "/data/jaw/rs.visibility.analysis/parquet"
)

MONTHS = [
    "2024.02",
    "2024.10",
]

INPUT_PATTERN = "rrc*.parquet"

INTEGER_TOKEN_RE = re.compile(r"^\d+$")

def is_as_set_token(token):
    """
    Return True if the token represents an AS_SET.
    """
    return (
        "{" in token
        or "}" in token
        or "," in token
    )


def parse_single_as_from_set(token):
    """
    Return the ASN from a singleton AS_SET such as {123}.

    Return None for a multi-AS AS_SET such as {123,456}.
    """

    value = (
        token
        .replace("{", "")
        .replace("}", "")
        .strip()
    )

    parts = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    if (
        len(parts) == 1
        and INTEGER_TOKEN_RE.fullmatch(parts[0])
    ):
        return int(parts[0])

    return None


def collapse_consecutive_duplicates(path):
    """
    Collapse consecutive occurrences of the same ASN.

    Example:

        1299 1299 898 765 1299 1299

    becomes:

        1299 898 765 1299

    This removes AS prepending from the path used for
    non-consecutive repetition detection.
    """

    collapsed = []

    for asn in path:
        if not collapsed or collapsed[-1] != asn:
            collapsed.append(asn)

    return collapsed


def clean_as_path(as_path):
    """
    Construct the sequential AS path used for RS detection.

    Rules:
      1. Ordinary ASN tokens are retained.
      2. AS_SET tokens before the origin are removed.
      3. A singleton AS_SET at the origin is converted to
         its ASN and retained.
      4. A multi-AS AS_SET at the origin is removed because
         the origin is ambiguous.
      5. Consecutive duplicate ASNs are collapsed.

    Returns:
        normalized_path
        had_as_set
        as_set_tokens_removed
        origin_from_single_as_set
        ambiguous_origin_as_set
        origin_as_set_raw
    """

    if as_path is None:
        return [], False, 0, False, False, None

    tokens = [
        token.strip()
        for token in str(as_path).strip().split()
        if token.strip()
    ]

    if not tokens:
        return [], False, 0, False, False, None

    sequential_path = []

    had_as_set = False
    as_set_tokens_removed = 0

    origin_from_single_as_set = False
    ambiguous_origin_as_set = False
    origin_as_set_raw = None

    last_index = len(tokens) - 1

    for index, token in enumerate(tokens):

        if is_as_set_token(token):

            had_as_set = True

            # AS_SET at the origin position
            if index == last_index:

                origin_as_set_raw = token

                singleton = parse_single_as_from_set(token)

                if singleton is not None:
                    sequential_path.append(singleton)
                    origin_from_single_as_set = True

                else:
                    ambiguous_origin_as_set = True
                    as_set_tokens_removed += 1

            # AS_SET elsewhere in the path
            else:
                as_set_tokens_removed += 1

            continue

        # Ordinary ASN
        if INTEGER_TOKEN_RE.fullmatch(token):
            sequential_path.append(int(token))

    normalized_path = collapse_consecutive_duplicates(
        sequential_path
    )

    return (
        normalized_path,
        had_as_set,
        as_set_tokens_removed,
        origin_from_single_as_set,
        ambiguous_origin_as_set,
        origin_as_set_raw,
    )

def unique_preserve_order(values):
    """
    Return unique values while preserving first-appearance
    order.
    """

    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def extract_rs(path):
    """
    Extract one RS for each ASN with a non-consecutive
    repetition.

    The input path has already had consecutive duplicates
    collapsed, so prepending cannot create an RS.

    The RS for an introducing ASN spans from its first
    occurrence to its last occurrence.
    """

    positions = {}

    for index, asn in enumerate(path):
        positions.setdefault(asn, []).append(index)

    results = []

    for introducing_as, indices in positions.items():

        if len(indices) < 2:
            continue

        first = indices[0]
        last = indices[-1]

        segment = path[first:last + 1]

        appearing_ases = unique_preserve_order(
            [
                asn
                for asn in segment[1:-1]
                if asn != introducing_as
            ]
        )

        if not appearing_ases:
            continue

        results.append({
            "introducing_as": introducing_as,
            "appearing_ases": appearing_ases,
            "rs_path": " ".join(
                map(str, segment)
            ),
        })

    return results

def write_log(log_file, message):
    """
    Write a timestamped message to both stdout and the
    monthly extraction log.
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"

    print(line, flush=True)

    with log_file.open("a") as f:
        f.write(line + "\n")


def validate_parquet(path):
    """
    Return True if an existing Parquet file can be opened
    and its metadata read.
    """

    if not path.exists():
        return False

    try:
        pf = pq.ParquetFile(path)

        # Reading these values forces metadata access.
        _ = pf.metadata.num_rows
        _ = pf.schema_arrow

        return True

    except Exception:
        return False

def process_file(
    input_file,
    output_file,
    log_file,
):

    start_time = time.time()

    pf = pq.ParquetFile(input_file)

    input_rows = pf.metadata.num_rows

    output_schema = pa.schema(
        list(pf.schema_arrow) + [
            pa.field(
                "cleaned_aspath",
                pa.string(),
            ),
            pa.field(
                "introducing_as",
                pa.int64(),
            ),
            pa.field(
                "appearing_ases",
                pa.list_(pa.int64()),
            ),
            pa.field(
                "rs_path",
                pa.string(),
            ),
            pa.field(
                "num_repeated_asns",
                pa.int64(),
            ),
            pa.field(
                "had_as_set",
                pa.bool_(),
            ),
            pa.field(
                "as_set_tokens_removed",
                pa.int64(),
            ),
            pa.field(
                "origin_from_single_as_set",
                pa.bool_(),
            ),
            pa.field(
                "ambiguous_origin_as_set",
                pa.bool_(),
            ),
            pa.field(
                "origin_as_set_raw",
                pa.string(),
            ),
        ]
    )

    writer = None

    total_rows = 0
    matched_routes = 0
    rs_occurrences = 0
    routes_with_multiple_rs = 0

    routes_with_as_set = 0
    routes_with_singleton_origin_as_set = 0
    routes_with_ambiguous_origin_as_set = 0

    try:

        for rg in tqdm(
            range(pf.metadata.num_row_groups),
            desc=input_file.name,
            unit="rg",
            leave=False,
        ):

            table = pf.read_row_group(rg)

            original_columns = table.column_names
            data = table.to_pydict()

            output = {
                name: []
                for name in original_columns
            }

            output["cleaned_aspath"] = []
            output["introducing_as"] = []
            output["appearing_ases"] = []
            output["rs_path"] = []
            output["num_repeated_asns"] = []

            output["had_as_set"] = []
            output["as_set_tokens_removed"] = []
            output["origin_from_single_as_set"] = []
            output["ambiguous_origin_as_set"] = []
            output["origin_as_set_raw"] = []

            for row_index in range(table.num_rows):

                total_rows += 1

                raw_path = data["as_path"][row_index]

                (
                    cleaned,
                    had_as_set,
                    as_set_tokens_removed,
                    origin_from_single_as_set,
                    ambiguous_origin_as_set,
                    origin_as_set_raw,
                ) = clean_as_path(raw_path)

                if had_as_set:
                    routes_with_as_set += 1

                if origin_from_single_as_set:
                    routes_with_singleton_origin_as_set += 1

                if ambiguous_origin_as_set:
                    routes_with_ambiguous_origin_as_set += 1

                if not cleaned:
                    continue

                repetitions = extract_rs(cleaned)

                if not repetitions:
                    continue

                matched_routes += 1

                if len(repetitions) > 1:
                    routes_with_multiple_rs += 1

                for rs in repetitions:

                    # Preserve every original input field
                    for column in original_columns:
                        output[column].append(
                            data[column][row_index]
                        )

                    # Normalized path
                    output["cleaned_aspath"].append(
                        " ".join(map(str, cleaned))
                    )

                    # RS fields
                    output["introducing_as"].append(
                        rs["introducing_as"]
                    )

                    output["appearing_ases"].append(
                        rs["appearing_ases"]
                    )

                    output["rs_path"].append(
                        rs["rs_path"]
                    )

                    output["num_repeated_asns"].append(
                        len(repetitions)
                    )

                    # AS_SET metadata
                    output["had_as_set"].append(
                        had_as_set
                    )

                    output[
                        "as_set_tokens_removed"
                    ].append(
                        as_set_tokens_removed
                    )

                    output[
                        "origin_from_single_as_set"
                    ].append(
                        origin_from_single_as_set
                    )

                    output[
                        "ambiguous_origin_as_set"
                    ].append(
                        ambiguous_origin_as_set
                    )

                    output[
                        "origin_as_set_raw"
                    ].append(
                        origin_as_set_raw
                    )

                    rs_occurrences += 1

            if not output["introducing_as"]:
                continue

            output_table = pa.Table.from_pydict(
                output,
                schema=output_schema,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    output_schema,
                    compression="zstd",
                    compression_level=10,
                    use_dictionary=True,
                    write_statistics=True,
                )

            writer.write_table(
                output_table
            )

    finally:
        if writer is not None:
            writer.close()

    if writer is None:

        empty_table = pa.Table.from_batches(
            [],
            schema=output_schema,
        )

        pq.write_table(
            empty_table,
            output_file,
            compression="zstd",
            compression_level=10,
        )

    if not validate_parquet(output_file):
        raise RuntimeError(
            f"Output validation failed: {output_file}"
        )

    result = pq.ParquetFile(output_file)

    output_rows = result.metadata.num_rows

    if output_rows != rs_occurrences:
        raise RuntimeError(
            "Output row count does not match extracted "
            f"RS occurrences: {output_rows:,} != "
            f"{rs_occurrences:,}"
        )

    elapsed = time.time() - start_time

    return {
        "input_rows": input_rows,
        "processed_rows": total_rows,
        "matched_routes": matched_routes,
        "rs_occurrences": rs_occurrences,
        "routes_with_multiple_rs": routes_with_multiple_rs,
        "routes_with_as_set": routes_with_as_set,
        "singleton_origin_as_set":
            routes_with_singleton_origin_as_set,
        "ambiguous_origin_as_set":
            routes_with_ambiguous_origin_as_set,
        "output_rows": output_rows,
        "output_mib":
            output_file.stat().st_size / (1024 ** 2),
        "seconds": elapsed,
    }

def process_month(month):

    month_dir = BASE_DIR / month
    output_dir = month_dir / "poisoning"
    log_file = output_dir / "_extraction.log"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = sorted(
        month_dir.glob(INPUT_PATTERN)
    )

    input_files = [
        path
        for path in input_files
        if path.parent == month_dir
    ]

    write_log(
        log_file,
        "=" * 72,
    )

    write_log(
        log_file,
        f"START MONTH {month}",
    )

    write_log(
        log_file,
        f"Input files found: {len(input_files):,}",
    )

    write_log(
        log_file,
        "=" * 72,
    )

    successful = 0
    skipped = 0
    failed = 0

    total_input_rows = 0
    total_matched_routes = 0
    total_rs_occurrences = 0

    month_start = time.time()

    for index, input_file in enumerate(
        input_files,
        start=1,
    ):

        output_file = (
            output_dir
            / f"{input_file.stem}.poisoning.parquet"
        )

        write_log(
            log_file,
            (
                f"[{index}/{len(input_files)}] "
                f"{input_file.name}"
            ),
        )

        if output_file.exists():

            if validate_parquet(output_file):

                result = pq.ParquetFile(
                    output_file
                )

                write_log(
                    log_file,
                    (
                        "SKIP valid existing output | "
                        f"rows={result.metadata.num_rows:,} | "
                        f"{output_file.name}"
                    ),
                )

                skipped += 1
                continue

            write_log(
                log_file,
                (
                    "REMOVE invalid/partial output | "
                    f"{output_file.name}"
                ),
            )

            output_file.unlink()

        try:

            stats = process_file(
                input_file,
                output_file,
                log_file,
            )

            successful += 1

            total_input_rows += (
                stats["input_rows"]
            )

            total_matched_routes += (
                stats["matched_routes"]
            )

            total_rs_occurrences += (
                stats["rs_occurrences"]
            )

            prevalence = (
                100
                * stats["matched_routes"]
                / stats["input_rows"]
                if stats["input_rows"]
                else 0.0
            )

            write_log(
                log_file,
                (
                    "SUCCESS | "
                    f"input_rows="
                    f"{stats['input_rows']:,} | "
                    f"RS_routes="
                    f"{stats['matched_routes']:,} | "
                    f"RS_occurrences="
                    f"{stats['rs_occurrences']:,} | "
                    f"prevalence="
                    f"{prevalence:.6f}% | "
                    f"output="
                    f"{stats['output_mib']:.2f} MiB | "
                    f"time="
                    f"{stats['seconds'] / 60:.2f} min"
                ),
            )

        except Exception as exc:

            failed += 1

            write_log(
                log_file,
                (
                    f"FAILED | "
                    f"{input_file.name} | "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

            # Preserve traceback in the log.
            with log_file.open("a") as f:
                f.write(
                    traceback.format_exc()
                )
                f.write("\n")

            # Remove partial output so restart will retry it.
            if output_file.exists():
                try:
                    output_file.unlink()

                    write_log(
                        log_file,
                        (
                            "REMOVED partial output | "
                            f"{output_file.name}"
                        ),
                    )

                except Exception as cleanup_exc:
                    write_log(
                        log_file,
                        (
                            "WARNING could not remove "
                            f"partial output | "
                            f"{output_file.name} | "
                            f"{cleanup_exc}"
                        ),
                    )

            continue

    month_elapsed = time.time() - month_start

    write_log(
        log_file,
        "=" * 72,
    )

    write_log(
        log_file,
        f"MONTH COMPLETE {month}",
    )

    write_log(
        log_file,
        f"Input files       : {len(input_files):,}",
    )

    write_log(
        log_file,
        f"Successful        : {successful:,}",
    )

    write_log(
        log_file,
        f"Skipped existing  : {skipped:,}",
    )

    write_log(
        log_file,
        f"Failed            : {failed:,}",
    )

    write_log(
        log_file,
        f"Input routes      : {total_input_rows:,}",
    )

    write_log(
        log_file,
        f"RS routes         : {total_matched_routes:,}",
    )

    write_log(
        log_file,
        f"RS occurrences    : {total_rs_occurrences:,}",
    )

    write_log(
        log_file,
        (
            f"Runtime           : "
            f"{month_elapsed / 3600:.2f} hr"
        ),
    )

    write_log(
        log_file,
        "=" * 72,
    )

    return {
        "month": month,
        "input_files": len(input_files),
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "input_rows": total_input_rows,
        "matched_routes": total_matched_routes,
        "rs_occurrences": total_rs_occurrences,
        "seconds": month_elapsed,
    }

def main():

    overall_start = time.time()

    results = []

    print("=" * 72)
    print("NON-CONSECUTIVE REPETITION EXTRACTION")
    print("=" * 72)
    print(f"Base directory: {BASE_DIR}")
    print(f"Months        : {', '.join(MONTHS)}")
    print("=" * 72)

    for month in MONTHS:

        month_dir = BASE_DIR / month

        if not month_dir.exists():
            print(
                f"ERROR: month directory does not exist: "
                f"{month_dir}"
            )
            continue

        result = process_month(month)

        results.append(result)

    overall_elapsed = (
        time.time() - overall_start
    )

    print("\n" + "=" * 72)
    print("ALL EXTRACTIONS COMPLETE")
    print("=" * 72)

    for result in results:

        print(
            f"{result['month']} | "
            f"files={result['input_files']:,} | "
            f"success={result['successful']:,} | "
            f"skipped={result['skipped']:,} | "
            f"failed={result['failed']:,} | "
            f"RS={result['rs_occurrences']:,}"
        )

    print(
        f"Total runtime: "
        f"{overall_elapsed / 3600:.2f} hr"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
