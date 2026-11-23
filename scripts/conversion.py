
#!/usr/bin/env python3

import bz2
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

INPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = INPUT_DIR / "prefix.aspth.parquet"
PATTERN = "prefix.aspath.*.bz2"

CHUNKSIZE = 2_000_000
NUM_WORKERS = min(3, max(1, cpu_count() - 1))
KEEP_COLUMNS = ["timestamp", "prefix", "as_path"]

SCHEMA = pa.schema([
    pa.field("timestamp", pa.int64()),
    pa.field("prefix", pa.string()),
    pa.field("as_path", pa.string()),
])


def convert_one_file(input_file: str):
    input_path = Path(input_file)
    output_path = OUTPUT_DIR / input_path.with_suffix(".parquet").name
    tmp_path = output_path.with_suffix(".parquet.tmp")

    if output_path.exists():
        return input_path.name, "skipped", 0, 0.0

    writer = None
    total_rows = 0
    start = time.perf_counter()

    try:
        with bz2.open(input_path, "rt", encoding="utf-8", errors="replace") as fh:
            reader = pd.read_csv(
                fh,
                sep="|",
                usecols=KEEP_COLUMNS,
                dtype={
                    "timestamp": "Int64",
                    "prefix": "string",
                    "as_path": "string",
                },
                chunksize=CHUNKSIZE,
            )

            for chunk in reader:
                chunk["timestamp"] = chunk["timestamp"].astype("int64")
                table = pa.Table.from_pandas(
                    chunk, schema=SCHEMA, preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path,
                        schema=SCHEMA,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )

                writer.write_table(table)
                total_rows += len(chunk)

        if writer is None:
            raise ValueError("Input file contains no records")

        writer.close()
        writer = None
        tmp_path.replace(output_path)

        return input_path.name, "done", total_rows, time.perf_counter() - start

    except Exception as e:
        if writer is not None:
            writer.close()
        tmp_path.unlink(missing_ok=True)
        return input_path.name, f"error: {e}", total_rows, time.perf_counter() - start


def main():
    files = sorted(str(p) for p in INPUT_DIR.glob(PATTERN) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {PATTERN} in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = skipped = failed = total_rows = 0

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(convert_one_file, files, chunksize=1)

        for name, status, rows, elapsed in tqdm(
            results, total=len(files), desc="Converting files", unit="file"
        ):
            if status == "done":
                completed += 1
                total_rows += rows
                tqdm.write(f"[DONE] {name}: {rows:,} rows ({elapsed:.1f}s)")
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                tqdm.write(f"[ERROR] {name}: {status}")

    print(
        f"Completed: {completed:,} | Skipped: {skipped:,} | "
        f"Failed: {failed:,} | Rows: {total_rows:,}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
