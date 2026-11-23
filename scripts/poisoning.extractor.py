#!/usr/bin/env python3

import json
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

INPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'prefix.aspth.parquet'
OUTPUT_DIR = INPUT_DIR / 'poisoning'
DONE_DIR = OUTPUT_DIR / '.done'
NUM_WORKERS = min(4, max(1, cpu_count() - 1))
INTEGER_TOKEN_RE = re.compile(r'^\d+$')

OUTPUT_SCHEMA = pa.schema([
    ('timestamp', pa.int64()),
    ('prefix', pa.string()),
    ('aspath', pa.string()),
    ('origin', pa.int64()),
    ('repeated_asns', pa.list_(pa.int64())),
    ('num_repeated_asns', pa.int32()),
])


def find_nonconsecutive_repeated_asns(path):
    last = {}
    repeated = []
    seen = set()
    for i, asn in enumerate(path):
        if asn in last and i - last[asn] > 1:
            if asn not in seen:
                repeated.append(asn)
                seen.add(asn)
        last[asn] = i
    return repeated


def process_one_file(filename):
    path = Path(filename)
    date = path.name.removeprefix('prefix.aspath.').removesuffix('.parquet')
    output = OUTPUT_DIR / f'poisoning.{date}.parquet'
    marker = DONE_DIR / f'{date}.done.json'
    temporary = output.with_suffix('.parquet.tmp')

    if marker.exists():
        return 'skipped', date, 0, 0, 0
    if output.exists():
        return 'error', date, 0, 0, 'output exists without a completion marker'

    scanned = excluded = matched = 0
    writer = None
    try:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            data = pf.read_row_group(rg, columns=['timestamp', 'prefix', 'as_path']).to_pydict()
            records = {name: [] for name in OUTPUT_SCHEMA.names}
            for ts, prefix, raw in zip(data['timestamp'], data['prefix'], data['as_path']):
                scanned += 1
                if raw is None:
                    continue
                tokens = str(raw).strip().split()
                if any('{' in tok or '}' in tok or ',' in tok for tok in tokens):
                    excluded += 1
                    continue
                parsed = [int(tok) for tok in tokens if INTEGER_TOKEN_RE.fullmatch(tok)]
                if not parsed:
                    continue
                repeated = find_nonconsecutive_repeated_asns(parsed)
                if not repeated:
                    continue
                matched += 1
                values = (int(ts), '' if prefix is None else str(prefix), str(raw),
                          parsed[-1], repeated, len(repeated))
                for name, value in zip(OUTPUT_SCHEMA.names, values):
                    records[name].append(value)
            if records['timestamp']:
                if writer is None:
                    writer = pq.ParquetWriter(temporary, OUTPUT_SCHEMA, compression='zstd',
                                              use_dictionary=True, write_statistics=True)
                writer.write_table(pa.Table.from_pydict(records, schema=OUTPUT_SCHEMA))

        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=OUTPUT_SCHEMA), temporary,
                           compression='zstd')
        else:
            writer.close()
            writer = None
        temporary.replace(output)
        summary = {'total_rows_scanned': scanned, 'as_set_rows_excluded': excluded,
                   'matched_rows': matched}
        marker_tmp = marker.with_suffix('.json.tmp')
        marker_tmp.write_text(json.dumps(summary, indent=2))
        marker_tmp.replace(marker)
        return 'done', date, scanned, excluded, matched
    except Exception as exc:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        return 'error', date, 0, 0, str(exc)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob('prefix.aspath.*.parquet'))
    if not files:
        raise FileNotFoundError(f'No input files found in {INPUT_DIR}')
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_one_file, map(str, files), chunksize=1),
                           total=len(files), desc='Extracting non-cons AS paths', unit='file'):
            results.append(result)
            if result[0] == 'error':
                tqdm.write(f'[ERROR] {result[1]}: {result[4]}')
    completed = [r for r in results if r[0] == 'done']
    print(f'Completed: {len(completed):,} | Skipped: {sum(r[0] == "skipped" for r in results):,} | '
          f'Errors: {sum(r[0] == "error" for r in results):,}')
    print(f'Rows scanned: {sum(r[2] for r in completed):,} | '
          f'AS-SET excluded: {sum(r[3] for r in completed):,} | '
          f'Non-cons: {sum(r[4] for r in completed):,}')
    if any(r[0] == 'error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
