#!/usr/bin/env python3

from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "/data/jaw/pam2026/2022-01-01/rrc00.parquet"
)

OUTPUT_DIR = INPUT_FILE.parent / "poisoning"
OUTPUT_FILE = OUTPUT_DIR / "rrc00.poisoning.parquet"

INTEGER_TOKEN_RE = re.compile(r"^\d+$")


# ============================================================
# AS-path handling
# ============================================================

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

            # --------------------------------------------
            # AS_SET at the origin position
            # --------------------------------------------
            if index == last_index:

                origin_as_set_raw = token

                singleton = parse_single_as_from_set(token)

                if singleton is not None:
                    sequential_path.append(singleton)
                    origin_from_single_as_set = True

                else:
                    ambiguous_origin_as_set = True
                    as_set_tokens_removed += 1

            # --------------------------------------------
            # AS_SET elsewhere in the path
            # --------------------------------------------
            else:
                as_set_tokens_removed += 1

            continue

        # --------------------------------------------
        # Ordinary ASN
        # --------------------------------------------
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


# ============================================================
# RS extraction
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Delete previous/partial rrc00 output
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():
        print(
            f"Removing previous test output: "
            f"{OUTPUT_FILE}"
        )
        OUTPUT_FILE.unlink()

    # --------------------------------------------------------
    # Open input
    # --------------------------------------------------------

    pf = pq.ParquetFile(INPUT_FILE)

    # --------------------------------------------------------
    # Fixed output schema
    #
    # Preserve the complete original input schema and append
    # explicitly typed RS fields. This prevents schema changes
    # between row groups when a nullable field contains only
    # None values in one batch.
    # --------------------------------------------------------

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

    print("=" * 72)
    print("NON-CONSECUTIVE REPETITION TEST")
    print("=" * 72)
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows  : {pf.metadata.num_rows:,}")
    print("=" * 72)

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
            desc="Row groups",
            unit="rg",
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

                    # ----------------------------------------
                    # Preserve every original input field
                    # ----------------------------------------

                    for column in original_columns:
                        output[column].append(
                            data[column][row_index]
                        )

                    # ----------------------------------------
                    # Normalized path
                    # ----------------------------------------

                    output["cleaned_aspath"].append(
                        " ".join(map(str, cleaned))
                    )

                    # ----------------------------------------
                    # RS fields
                    # ----------------------------------------

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

                    # ----------------------------------------
                    # AS_SET metadata
                    # ----------------------------------------

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

            # ------------------------------------------------
            # Write this row group's RS observations
            # ------------------------------------------------

            if not output["introducing_as"]:
                continue

            output_table = pa.Table.from_pydict(
                output,
                schema=output_schema,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
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

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)

    print(
        f"Input routes                    : "
        f"{total_rows:,}"
    )

    print(
        f"Routes containing RS            : "
        f"{matched_routes:,}"
    )

    print(
        f"RS occurrences                  : "
        f"{rs_occurrences:,}"
    )

    print(
        f"Routes with multiple RS         : "
        f"{routes_with_multiple_rs:,}"
    )

    print(
        f"Routes containing AS_SET        : "
        f"{routes_with_as_set:,}"
    )

    print(
        f"Singleton origin AS_SET routes  : "
        f"{routes_with_singleton_origin_as_set:,}"
    )

    print(
        f"Ambiguous origin AS_SET routes  : "
        f"{routes_with_ambiguous_origin_as_set:,}"
    )

    if total_rows:
        print(
            f"RS route prevalence             : "
            f"{100 * matched_routes / total_rows:.6f}%"
        )

    if OUTPUT_FILE.exists():

        result = pq.ParquetFile(
            OUTPUT_FILE
        )

        print(
            f"Output rows                     : "
            f"{result.metadata.num_rows:,}"
        )

        print(
            f"Output size                     : "
            f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):,.2f} MiB"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
