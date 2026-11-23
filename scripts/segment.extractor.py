#!/usr/bin/env python3

import json
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

INPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'prefix.aspth.parquet' / 'poisoning'
OUTPUT_DIR = INPUT_DIR / 'segments'
DONE_DIR = INPUT_DIR / '.done.segments'
NUM_WORKERS = min(4, max(1, cpu_count() - 1))
INTEGER_TOKEN_RE = re.compile(r'^\d+$')

SEGMENTS_SCHEMA = pa.schema([
    ('timestamp', pa.int64()),
    ('prefix', pa.string()),
    ('aspath', pa.string()),
    ('cleaned_aspath', pa.string()),
    ('origin', pa.int64()),
    ('repeated_asn', pa.int64()),
    ('segment_start_idx', pa.int32()),
    ('segment_end_idx', pa.int32()),
    ('poisoned_segment', pa.string()),
    ('gap', pa.int32()),
    ('segment_len', pa.int32()),
    ('repeat_count_per_path', pa.int32()),
    ('is_origin_repeated', pa.bool_()),
])


def collapse_consecutive_duplicates(path):
    if not path:
        return []
    collapsed = [path[0]]
    for asn in path[1:]:
        if asn != collapsed[-1]:
            collapsed.append(asn)
    return collapsed


def extract_poisoned_segments(path):
    last = {}
    segments = []
    for idx, asn in enumerate(path):
        if asn in last:
            prev_idx = last[asn]
            if idx - prev_idx > 1:
                segments.append((asn, prev_idx, idx, path[prev_idx:idx + 1]))
        last[asn] = idx
    return segments


def process_one_file(filename):
    path = Path(filename)
    date = path.name.removeprefix('poisoning.').removesuffix('.parquet')
    output = OUTPUT_DIR / f'segments.{date}.parquet'
    marker = DONE_DIR / f'{date}.done.json'
    temporary = output.with_suffix('.parquet.tmp')

    if marker.exists():
        return 'skipped', date, 0, 0
    if output.exists():
        return 'error', date, 0, 'output exists without a completion marker'

    total_paths = total_segments = 0
    writer = None
    try:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            data = pf.read_row_group(rg, columns=['timestamp', 'prefix', 'aspath', 'origin']).to_pydict()
            records = {name: [] for name in SEGMENTS_SCHEMA.names}
            for ts, prefix, aspath, origin in zip(data['timestamp'], data['prefix'],
                                                  data['aspath'], data['origin']):
                total_paths += 1
                original = [int(x) for x in str(aspath).split()
                            if INTEGER_TOKEN_RE.fullmatch(x)] if aspath else []
                if not original:
                    continue
                collapsed = collapse_consecutive_duplicates(original)
                segments = extract_poisoned_segments(collapsed)
                if not segments:
                    continue
                collapsed_str = ' '.join(map(str, collapsed))
                repeat_count = len(segments)
                for repeated_asn, start, end, segment in segments:
                    values = (int(ts), '' if prefix is None else str(prefix), str(aspath),
                              collapsed_str, int(origin), int(repeated_asn), int(start),
                              int(end), ' '.join(map(str, segment)), int(end - start - 1),
                              len(segment), repeat_count, repeated_asn == int(origin))
                    for name, value in zip(SEGMENTS_SCHEMA.names, values):
                        records[name].append(value)
                    total_segments += 1
            if records['timestamp']:
                if writer is None:
                    writer = pq.ParquetWriter(temporary, SEGMENTS_SCHEMA, compression='zstd',
                                              use_dictionary=True, write_statistics=True)
                writer.write_table(pa.Table.from_pydict(records, schema=SEGMENTS_SCHEMA))

        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=SEGMENTS_SCHEMA), temporary,
                           compression='zstd')
        else:
            writer.close()
            writer = None
        temporary.replace(output)
        summary = {'total_path_rows_read': total_paths, 'total_segments': total_segments}
        marker_tmp = marker.with_suffix('.json.tmp')
        marker_tmp.write_text(json.dumps(summary, indent=2))
        marker_tmp.replace(marker)
        return 'done', date, total_paths, total_segments
    except Exception as exc:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        return 'error', date, 0, str(exc)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob('poisoning.*.parquet'))
    if not files:
        raise FileNotFoundError(f'No poisoning files found in {INPUT_DIR}')
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_one_file, map(str, files), chunksize=1),
                           total=len(files), desc='Extracting repetition segments', unit='file'):
            results.append(result)
            if result[0] == 'error':
                tqdm.write(f'[ERROR] {result[1]}: {result[3]}')
    completed = [r for r in results if r[0] == 'done']
    print(f'Completed: {len(completed):,} | Skipped: {sum(r[0] == "skipped" for r in results):,} | '
          f'Errors: {sum(r[0] == "error" for r in results):,}')
    print(f'Paths processed: {sum(r[2] for r in completed):,} | '
          f'Segments: {sum(r[3] for r in completed):,}')
    if any(r[0] == 'error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
