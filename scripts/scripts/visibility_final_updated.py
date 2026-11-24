#!/usr/bin/env python3
"""RIPE RIS non-cons visibility analysis.

Analytical definitions are unchanged from the original implementation.
Phase 3 remains collector-by-collector, memory-bounded, and restart-safe.
"""

from pathlib import Path
import time
import shutil
import polars as pl
BASE_DIR = Path('/data/jaw/rs.visibility.analysis')
PARQUET_DIR = BASE_DIR / 'parquet'
OUTPUT_ROOT = BASE_DIR / 'visibility.results'
MONTHS = ['2024.02', '2024.10']
ZSTD_LEVEL = 10
HIGH_VISIBILITY_THRESHOLD = 50.0

def format_duration(seconds):
    if seconds < 60:
        return f'{seconds:.1f} sec'
    if seconds < 3600:
        return f'{seconds / 60:.2f} min'
    return f'{seconds / 3600:.2f} hr'

def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0

def write_single_row_csv(path, values):
    pl.DataFrame([values]).write_csv(path)

def validate_month(month):
    month_dir = PARQUET_DIR / month
    poisoning_dir = month_dir / 'poisoning'
    if not month_dir.exists():
        raise RuntimeError(f'\nABORT: month directory does not exist:\n{month_dir}')
    if not poisoning_dir.exists():
        raise RuntimeError(f'\nABORT: poisoning directory does not exist:\n{poisoning_dir}')
    rib_files = sorted(month_dir.glob('rrc*.parquet'))
    poisoning_files = sorted(poisoning_dir.glob('rrc*.poisoning.parquet'))
    if not rib_files:
        raise RuntimeError(f'No RIB files found for {month}')
    if not poisoning_files:
        raise RuntimeError(f'No poisoning files found for {month}')
    rib_stems = {file.stem for file in rib_files}
    poisoning_stems = {file.name.replace('.poisoning.parquet', '') for file in poisoning_files}
    missing_poisoning = rib_stems - poisoning_stems
    extra_poisoning = poisoning_stems - rib_stems
    if missing_poisoning:
        raise RuntimeError('\nABORT: RIB files without matching poisoning output:\n' + '\n'.join(sorted(missing_poisoning)))
    if extra_poisoning:
        raise RuntimeError('\nABORT: poisoning files without matching RIB input:\n' + '\n'.join(sorted(extra_poisoning)))
    return (month_dir, poisoning_dir, rib_files, poisoning_files)

def prepare_output_dirs(month):
    month_output = OUTPUT_ROOT / month
    localization_dir = month_output / '01_localization'
    direct_peer_dir = month_output / '02_direct_peer'
    multi_collector_dir = month_output / '03_multi_collector'
    high_visibility_dir = month_output / '04_high_visibility'
    month_output.mkdir(parents=True, exist_ok=True)
    for directory in [localization_dir, direct_peer_dir, multi_collector_dir, high_visibility_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    summary_file = month_output / 'analysis_summary.txt'
    if summary_file.exists():
        summary_file.unlink()
    return {'root': month_output, 'localization': localization_dir, 'direct_peer': direct_peer_dir, 'multi_collector': multi_collector_dir, 'high_visibility': high_visibility_dir}

def validate_visibility_file(visibility_file, poisoning_files):
    """Trust a cached final table only when it matches the current poisoning input."""
    if not visibility_file.exists():
        return False

    try:
        required = {
            "prefix", "rs_path", "introducing_as", "num_observations",
            "pattern_collectors", "num_pattern_collectors",
            "pattern_peer_ases", "num_pattern_peer_ases",
            "pattern_sessions", "num_pattern_sessions",
            "introducing_as_is_ris_peer", "num_direct_peer_sessions",
            "prefix_collectors", "num_prefix_collectors",
            "prefix_peer_ases", "num_prefix_peer_ases",
            "prefix_sessions", "num_prefix_sessions",
            "collector_visibility_pct", "peer_visibility_pct",
            "session_visibility_pct",
        }
        schema = pl.read_parquet_schema(visibility_file)
        missing = required - set(schema)
        if missing:
            print("Cached visibility table is missing columns:")
            for column in sorted(missing):
                print(f"  {column}")
            return False

        visibility = pl.read_parquet(visibility_file)
        if visibility.height == 0:
            print("Cached visibility table contains zero rows.")
            return False

        # Exactly one final row must exist for every (prefix, rs_path).
        if visibility.select(["prefix", "rs_path"]).n_unique() != visibility.height:
            print("Cached visibility table contains duplicate (prefix, rs_path) rows.")
            return False

        # Internal consistency of the cached table.
        bad = visibility.filter(
            (pl.col("num_prefix_collectors") <= 0)
            | (pl.col("num_prefix_peer_ases") <= 0)
            | (pl.col("num_prefix_sessions") <= 0)
            | (pl.col("num_pattern_collectors") > pl.col("num_prefix_collectors"))
            | (pl.col("num_pattern_peer_ases") > pl.col("num_prefix_peer_ases"))
            | (pl.col("num_pattern_sessions") > pl.col("num_prefix_sessions"))
            | (pl.col("collector_visibility_pct") < 0)
            | (pl.col("collector_visibility_pct") > 100.000001)
            | (pl.col("peer_visibility_pct") < 0)
            | (pl.col("peer_visibility_pct") > 100.000001)
            | (pl.col("session_visibility_pct") < 0)
            | (pl.col("session_visibility_pct") > 100.000001)
        )
        if bad.height:
            print("Cached visibility table failed denominator/visibility checks.")
            return False

        # Reconstruct the expected pattern population from the current poisoning
        # inputs using only the three columns needed for validation. This does not
        # alter any metric; it only prevents a stale final file from being reused.
        expected = (
            pl.scan_parquet([str(path) for path in poisoning_files])
            .select(["prefix", "rs_path", "introducing_as"])
            .group_by(["prefix", "rs_path"])
            .agg([
                pl.col("introducing_as").n_unique().alias("num_introducing_ases"),
                pl.col("introducing_as").first().alias("introducing_as"),
                pl.count().alias("num_observations"),
            ])
            .collect()
        )

        if expected.filter(pl.col("num_introducing_ases") != 1).height:
            print("Current poisoning input has a pattern mapping to multiple introducing ASes.")
            return False

        expected = expected.drop("num_introducing_ases")
        actual = visibility.select(
            ["prefix", "rs_path", "introducing_as", "num_observations"]
        )

        missing_patterns = expected.join(
            actual, on=["prefix", "rs_path", "introducing_as", "num_observations"], how="anti"
        )
        extra_patterns = actual.join(
            expected, on=["prefix", "rs_path", "introducing_as", "num_observations"], how="anti"
        )

        if missing_patterns.height or extra_patterns.height:
            print("Cached visibility table does not match the current poisoning patterns.")
            print(f"  missing/mismatched: {missing_patterns.height:,}")
            print(f"  extra/mismatched  : {extra_patterns.height:,}")
            return False

        print("Cached visibility table passed semantic validation.")
        return True

    except Exception as exc:
        print("Cached visibility table failed validation:")
        print(str(exc))
        return False

def load_noncons_observations(poisoning_files):
    print()
    print('=' * 72)
    print('PHASE 1: LOAD NON-CONS OBSERVATIONS')
    print('=' * 72)
    required_columns = ['collector', 'peer_asn', 'peer_ip', 'prefix', 'introducing_as', 'rs_path']
    frames = []
    total_rows = 0
    for index, input_file in enumerate(poisoning_files, start=1):
        df = pl.read_parquet(input_file, columns=required_columns)
        total_rows += df.height
        frames.append(df)
        if index == 1 or index % 50 == 0 or index == len(poisoning_files):
            print(f'\r[{index}/{len(poisoning_files)}] rows={total_rows:,}', end='', flush=True)
    print()
    observations = pl.concat(frames, how='vertical')
    del frames
    print(f'Non-cons observations: {observations.height:,}')
    print(f"Poisoning prefixes    : {observations['prefix'].n_unique():,}")
    print(f"Distinct RSs         : {observations['rs_path'].n_unique():,}")
    return observations

def build_pattern_visibility(observations):
    print()
    print('=' * 72)
    print('PHASE 2: BUILD PATTERN VISIBILITY')
    print('=' * 72)
    introducing_check = observations.group_by('rs_path').agg(pl.col('introducing_as').n_unique().alias('num_introducing_ases')).filter(pl.col('num_introducing_ases') != 1)
    if introducing_check.height > 0:
        raise RuntimeError('\nABORT: at least one rs_path maps to multiple introducing ASes.')
    total_patterns = observations.select(['prefix', 'rs_path']).unique().height
    print(f'Distinct patterns    : {total_patterns:,}')
    pattern_rows = []
    grouped = observations.group_by(['prefix', 'rs_path'], maintain_order=False)
    processed = 0
    for key, group in grouped:
        if isinstance(key, tuple):
            prefix_value = key[0]
            rs_path_value = key[1]
        else:
            raise RuntimeError('Unexpected Polars group key format.')
        introducing_values = group['introducing_as'].unique().to_list()
        if len(introducing_values) != 1:
            raise RuntimeError(f'Pattern maps to multiple introducing ASes:\n{prefix_value}\n{rs_path_value}')
        introducing_as = introducing_values[0]
        collectors = sorted(group['collector'].drop_nulls().unique().to_list())
        peer_ases = sorted(group['peer_asn'].drop_nulls().unique().to_list())
        sessions_df = group.select(['collector', 'peer_ip']).unique()
        sessions = sorted([f'{collector}|{peer_ip}' for collector, peer_ip in sessions_df.iter_rows()])
        direct_sessions_df = group.filter(pl.col('peer_asn') == introducing_as).select(['collector', 'peer_ip']).unique()
        introducing_is_peer = direct_sessions_df.height > 0
        pattern_rows.append({'prefix': prefix_value, 'rs_path': rs_path_value, 'introducing_as': introducing_as, 'num_observations': group.height, 'pattern_collectors': collectors, 'num_pattern_collectors': len(collectors), 'pattern_peer_ases': peer_ases, 'num_pattern_peer_ases': len(peer_ases), 'pattern_sessions': sessions, 'num_pattern_sessions': len(sessions), 'introducing_as_is_ris_peer': introducing_is_peer, 'num_direct_peer_sessions': direct_sessions_df.height})
        processed += 1
        if processed == 1 or processed % 5000 == 0 or processed == total_patterns:
            print(f'\rPatterns: {processed:,}/{total_patterns:,}', end='', flush=True)
    print()
    patterns = pl.DataFrame(pattern_rows)
    print(f'Pattern rows         : {patterns.height:,}')
    return patterns

def validate_collector_checkpoint(checkpoint_file, collector):
    if not checkpoint_file.exists():
        return False
    try:
        schema = pl.read_parquet_schema(checkpoint_file)
        required = {'prefix', 'collector', 'collector_peer_ases', 'collector_peer_ips', 'num_collector_sessions'}
        if not required.issubset(set(schema.keys())):
            return False
        check = pl.scan_parquet(checkpoint_file).select([pl.count().alias('rows'), pl.col('collector').n_unique().alias('collectors'), pl.col('collector').first().alias('collector')]).collect()
        if check['rows'][0] <= 0:
            return False
        if check['collectors'][0] != 1:
            return False
        if check['collector'][0] != collector:
            return False
        return True
    except Exception:
        return False

def atomic_write_parquet(df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + '.tmp')
    if temporary_file.exists():
        temporary_file.unlink()
    df.write_parquet(temporary_file, compression='zstd', compression_level=ZSTD_LEVEL)
    temporary_file.replace(output_file)

def build_prefix_denominators(rib_files, poisoning_prefixes, checkpoint_dir):
    print()
    print('=' * 72)
    print('PHASE 3: BUILD PREFIX DENOMINATORS')
    print('=' * 72)
    poisoning_prefixes = list(poisoning_prefixes)
    poisoning_prefix_set = set(poisoning_prefixes)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for temporary_file in checkpoint_dir.glob('*.tmp'):
        temporary_file.unlink()
    by_collector = {}
    for input_file in rib_files:
        collector = input_file.name.split('.')[0]
        by_collector.setdefault(collector, []).append(input_file)
    collector_summaries = []
    scanned_rows = 0
    num_collectors = len(by_collector)
    for collector_index, collector in enumerate(sorted(by_collector), start=1):
        files = sorted(by_collector[collector])
        checkpoint_file = checkpoint_dir / f'{collector}.prefix_visibility.parquet'
        if validate_collector_checkpoint(checkpoint_file, collector):
            collector_df = pl.read_parquet(checkpoint_file)
            collector_summaries.append(collector_df)
            print(f'Collector {collector_index:02d}/{num_collectors:02d} {collector} | checkpoint found | prefixes={collector_df.height:,}')
            continue
        if checkpoint_file.exists():
            print(f'Removing invalid collector checkpoint: {checkpoint_file}')
            checkpoint_file.unlink()
        peer_ases_by_prefix = {}
        peer_ips_by_prefix = {}
        for file_index, input_file in enumerate(files, start=1):
            df = pl.read_parquet(input_file, columns=['peer_asn', 'peer_ip', 'prefix'])
            scanned_rows += df.height
            daily = df.filter(pl.col('prefix').is_in(poisoning_prefix_set)).group_by('prefix').agg([pl.col('peer_asn').drop_nulls().unique().alias('peer_ases'), pl.col('peer_ip').drop_nulls().unique().alias('peer_ips')])
            del df
            for prefix, peer_ases, peer_ips in daily.iter_rows():
                if prefix not in peer_ases_by_prefix:
                    peer_ases_by_prefix[prefix] = set()
                    peer_ips_by_prefix[prefix] = set()
                if peer_ases:
                    peer_ases_by_prefix[prefix].update(peer_ases)
                if peer_ips:
                    peer_ips_by_prefix[prefix].update(peer_ips)
            del daily
            print(f'\rCollector {collector_index:02d}/{num_collectors:02d} {collector} | day {file_index:02d}/{len(files):02d} | scanned={scanned_rows:,} | prefixes={len(peer_ases_by_prefix):,}', end='', flush=True)
        print()
        rows = []
        for prefix in peer_ases_by_prefix:
            peer_ases = sorted(peer_ases_by_prefix[prefix])
            peer_ips = sorted(peer_ips_by_prefix[prefix])
            rows.append({'prefix': prefix, 'collector': collector, 'collector_peer_ases': peer_ases, 'collector_peer_ips': peer_ips, 'num_collector_sessions': len(peer_ips)})
        collector_df = pl.DataFrame(rows)
        atomic_write_parquet(collector_df, checkpoint_file)
        collector_summaries.append(collector_df)
        print(f'  {collector}: {collector_df.height:,} poisoning prefixes retained and checkpointed')
        del rows
        del peer_ases_by_prefix
        del peer_ips_by_prefix
    compact = pl.concat(collector_summaries, how='vertical')
    del collector_summaries
    print()
    print(f'Compact prefix-collector rows: {compact.height:,}')
    compact_checkpoint = checkpoint_dir / 'all_collectors.prefix_visibility.parquet'
    atomic_write_parquet(compact, compact_checkpoint)
    print(f'Combined checkpoint  : {compact_checkpoint}')
    denominator_rows = []
    grouped = compact.group_by('prefix', maintain_order=False)
    total_prefixes = compact['prefix'].n_unique()
    processed = 0
    for key, group in grouped:
        if isinstance(key, tuple):
            prefix_value = key[0]
        else:
            prefix_value = key
        collectors = sorted(group['collector'].drop_nulls().unique().to_list())
        peer_as_set = set()
        sessions = []
        for collector, peer_ases, peer_ips in group.select(['collector', 'collector_peer_ases', 'collector_peer_ips']).iter_rows():
            if peer_ases:
                peer_as_set.update(peer_ases)
            if peer_ips:
                sessions.extend((f'{collector}|{peer_ip}' for peer_ip in peer_ips))
        peer_ases = sorted(peer_as_set)
        sessions = sorted(set(sessions))
        denominator_rows.append({'prefix': prefix_value, 'prefix_collectors': collectors, 'num_prefix_collectors': len(collectors), 'prefix_peer_ases': peer_ases, 'num_prefix_peer_ases': len(peer_ases), 'prefix_sessions': sessions, 'num_prefix_sessions': len(sessions)})
        processed += 1
        if processed == 1 or processed % 5000 == 0 or processed == total_prefixes:
            print(f'\rPrefixes: {processed:,}/{total_prefixes:,}', end='', flush=True)
    print()
    del compact
    denominators = pl.DataFrame(denominator_rows)
    del denominator_rows
    missing_prefixes = len(poisoning_prefix_set) - denominators.height
    if missing_prefixes != 0:
        raise RuntimeError(f'\nABORT: some poisoning prefixes have no ordinary-RIB denominator.\nMissing prefixes: {missing_prefixes:,}')
    print(f'Poisoning prefixes    : {denominators.height:,}')
    if scanned_rows > 0:
        print(f'RIB rows scanned     : {scanned_rows:,}')
    else:
        print('RIB rows scanned     : 0 (all collector checkpoints reused)')
    return denominators

def build_visibility_table(patterns, denominators, output_file):
    print()
    print('=' * 72)
    print('PHASE 4: COMPUTE NORMALIZED VISIBILITY')
    print('=' * 72)
    visibility = patterns.join(denominators, on='prefix', how='left').with_columns([(pl.col('num_pattern_collectors') / pl.col('num_prefix_collectors') * 100.0).alias('collector_visibility_pct'), (pl.col('num_pattern_peer_ases') / pl.col('num_prefix_peer_ases') * 100.0).alias('peer_visibility_pct'), (pl.col('num_pattern_sessions') / pl.col('num_prefix_sessions') * 100.0).alias('session_visibility_pct')]).sort(['prefix', 'rs_path'])
    bad_denominators = visibility.filter((pl.col('num_prefix_collectors') <= 0) | (pl.col('num_prefix_peer_ases') <= 0) | (pl.col('num_prefix_sessions') <= 0))
    if bad_denominators.height > 0:
        raise RuntimeError('Invalid zero visibility denominator.')
    bad_visibility = visibility.filter((pl.col('collector_visibility_pct') > 100.000001) | (pl.col('peer_visibility_pct') > 100.000001) | (pl.col('session_visibility_pct') > 100.000001))
    if bad_visibility.height > 0:
        raise RuntimeError('\nABORT: pattern visibility exceeds prefix visibility.')
    visibility.write_parquet(output_file, compression='zstd', compression_level=ZSTD_LEVEL)
    print(f'Patterns             : {visibility.height:,}')
    print(f'Output               : {output_file}')
    return visibility

def analyze_localization(visibility, output_dir):
    print()
    print('=' * 72)
    print('ANALYSIS 1: LOCALIZATION')
    print('=' * 72)
    total = visibility.height
    summary = {'patterns': total, 'affected_prefixes': visibility['prefix'].n_unique(), 'single_collector_patterns': visibility.filter(pl.col('num_pattern_collectors') == 1).height, 'multiple_collector_patterns': visibility.filter(pl.col('num_pattern_collectors') > 1).height, 'single_peer_patterns': visibility.filter(pl.col('num_pattern_peer_ases') == 1).height, 'multiple_peer_patterns': visibility.filter(pl.col('num_pattern_peer_ases') > 1).height, 'single_session_patterns': visibility.filter(pl.col('num_pattern_sessions') == 1).height, 'multiple_session_patterns': visibility.filter(pl.col('num_pattern_sessions') > 1).height, 'median_pattern_collectors': visibility['num_pattern_collectors'].median(), 'median_prefix_collectors': visibility['num_prefix_collectors'].median(), 'median_collector_visibility_pct': visibility['collector_visibility_pct'].median(), 'median_pattern_peer_ases': visibility['num_pattern_peer_ases'].median(), 'median_prefix_peer_ases': visibility['num_prefix_peer_ases'].median(), 'median_peer_visibility_pct': visibility['peer_visibility_pct'].median(), 'median_pattern_sessions': visibility['num_pattern_sessions'].median(), 'median_prefix_sessions': visibility['num_prefix_sessions'].median(), 'median_session_visibility_pct': visibility['session_visibility_pct'].median()}
    summary['single_collector_pct'] = pct(summary['single_collector_patterns'], total)
    summary['multiple_collector_pct'] = pct(summary['multiple_collector_patterns'], total)
    summary['single_peer_pct'] = pct(summary['single_peer_patterns'], total)
    summary['multiple_peer_pct'] = pct(summary['multiple_peer_patterns'], total)
    summary['single_session_pct'] = pct(summary['single_session_patterns'], total)
    summary['multiple_session_pct'] = pct(summary['multiple_session_patterns'], total)
    write_single_row_csv(output_dir / 'visibility_summary.csv', summary)
    bins = [('0-1%', 0.0, 1.0), ('1-5%', 1.0, 5.0), ('5-10%', 5.0, 10.0), ('10-25%', 10.0, 25.0), ('25-50%', 25.0, 50.0), ('50-75%', 50.0, 75.0), ('75-100%', 75.0, 100.000001)]
    bin_rows = []
    for label, lower, upper in bins:
        if lower == 0.0:
            lower_filter_peer = pl.col('peer_visibility_pct') >= lower
            lower_filter_session = pl.col('session_visibility_pct') >= lower
            lower_filter_collector = pl.col('collector_visibility_pct') >= lower
        else:
            lower_filter_peer = pl.col('peer_visibility_pct') > lower
            lower_filter_session = pl.col('session_visibility_pct') > lower
            lower_filter_collector = pl.col('collector_visibility_pct') > lower
        peer_count = visibility.filter(lower_filter_peer & (pl.col('peer_visibility_pct') <= upper)).height
        session_count = visibility.filter(lower_filter_session & (pl.col('session_visibility_pct') <= upper)).height
        collector_count = visibility.filter(lower_filter_collector & (pl.col('collector_visibility_pct') <= upper)).height
        bin_rows.append({'bin': label, 'collector_patterns': collector_count, 'collector_pct': pct(collector_count, total), 'peer_patterns': peer_count, 'peer_pct': pct(peer_count, total), 'session_patterns': session_count, 'session_pct': pct(session_count, total)})
    pl.DataFrame(bin_rows).write_csv(output_dir / 'visibility_bins.csv')
    visibility.select(['prefix', 'num_prefix_collectors', 'num_prefix_peer_ases', 'num_prefix_sessions']).unique().sort('prefix').write_csv(output_dir / 'prefix_denominators.csv')
    return summary

def analyze_direct_peer(visibility, output_dir):
    print()
    print('=' * 72)
    print('ANALYSIS 2: DIRECT-PEER STRUCTURE')
    print('=' * 72)
    total = visibility.height
    direct = visibility.filter(pl.col('introducing_as_is_ris_peer'))
    non_direct = visibility.filter(~pl.col('introducing_as_is_ris_peer'))
    rows = []
    for label, df in [('direct_peer', direct), ('non_direct', non_direct)]:
        count = df.height
        single_session = df.filter(pl.col('num_pattern_sessions') == 1).height
        rows.append({'group': label, 'patterns': count, 'share_of_all_patterns_pct': pct(count, total), 'single_session_patterns': single_session, 'single_session_pct': pct(single_session, count), 'median_collectors': df['num_pattern_collectors'].median() if count else None, 'median_peer_ases': df['num_pattern_peer_ases'].median() if count else None, 'median_sessions': df['num_pattern_sessions'].median() if count else None, 'median_collector_visibility_pct': df['collector_visibility_pct'].median() if count else None, 'median_peer_visibility_pct': df['peer_visibility_pct'].median() if count else None, 'median_session_visibility_pct': df['session_visibility_pct'].median() if count else None})
    comparison = pl.DataFrame(rows)
    comparison.write_csv(output_dir / 'direct_peer_comparison.csv')
    return comparison

def analyze_multi_collector(visibility, output_dir):
    print()
    print('=' * 72)
    print('ANALYSIS 3: MULTI-COLLECTOR STRUCTURE')
    print('=' * 72)
    multi = visibility.filter(pl.col('num_pattern_collectors') > 1)
    total_multi = multi.height
    one_peer = multi.filter(pl.col('num_pattern_peer_ases') == 1)
    multiple_peers = multi.filter(pl.col('num_pattern_peer_ases') > 1)
    one_peer_is_introducing = one_peer.filter(pl.col('introducing_as_is_ris_peer'))
    summary = {'multi_collector_patterns': total_multi, 'through_one_peer_as': one_peer.height, 'through_one_peer_as_pct': pct(one_peer.height, total_multi), 'through_multiple_peer_ases': multiple_peers.height, 'through_multiple_peer_ases_pct': pct(multiple_peers.height, total_multi), 'single_peer_is_introducing_as': one_peer_is_introducing.height, 'single_peer_is_introducing_as_pct': pct(one_peer_is_introducing.height, one_peer.height)}
    write_single_row_csv(output_dir / 'multi_collector_summary.csv', summary)
    breakdown = one_peer.select(['prefix', 'rs_path', 'introducing_as', 'pattern_collectors', 'num_pattern_collectors', 'pattern_peer_ases', 'num_pattern_sessions', 'introducing_as_is_ris_peer', 'collector_visibility_pct', 'peer_visibility_pct', 'session_visibility_pct']).with_columns([pl.col('pattern_collectors').list.eval(pl.element().cast(pl.Utf8)).list.join('|').alias('pattern_collectors'), pl.col('pattern_peer_ases').list.eval(pl.element().cast(pl.Utf8)).list.join('|').alias('pattern_peer_ases')]).sort(['num_pattern_collectors', 'num_pattern_sessions'], descending=True)
    breakdown.write_csv(output_dir / 'single_peer_breakdown.csv')
    return summary

def concentration_table(df, column, output_file):
    total = df.height
    if total == 0:
        empty = pl.DataFrame({column: [], 'patterns': [], 'share_pct': []})
        empty.write_csv(output_file)
        return empty
    concentration = df.group_by(column).agg(pl.count().alias('patterns')).sort('patterns', descending=True).with_columns((pl.col('patterns') / total * 100.0).alias('share_pct'))
    concentration.write_csv(output_file)
    return concentration

def top_n_share(concentration, n):
    if concentration.height == 0:
        return 0.0
    return float(concentration.head(n)['share_pct'].sum())

def analyze_high_visibility(visibility, output_dir):
    print()
    print('=' * 72)
    print('ANALYSIS 4: HIGH-VISIBILITY TAIL')
    print('=' * 72)
    high = visibility.filter((pl.col('peer_visibility_pct') > HIGH_VISIBILITY_THRESHOLD) | (pl.col('session_visibility_pct') > HIGH_VISIBILITY_THRESHOLD))
    total_patterns = visibility.height
    high_count = high.height
    direct_count = high.filter(pl.col('introducing_as_is_ris_peer')).height
    rs_concentration = concentration_table(high, 'rs_path', output_dir / 'rs_concentration.csv')
    introducing_concentration = concentration_table(high, 'introducing_as', output_dir / 'introducing_as_concentration.csv')
    prefix_concentration = concentration_table(high, 'prefix', output_dir / 'prefix_concentration.csv')
    summary = {'threshold_pct': HIGH_VISIBILITY_THRESHOLD, 'high_visibility_patterns': high_count, 'share_of_all_patterns_pct': pct(high_count, total_patterns), 'unique_rs': high['rs_path'].n_unique() if high_count else 0, 'unique_prefixes': high['prefix'].n_unique() if high_count else 0, 'unique_introducing_ases': high['introducing_as'].n_unique() if high_count else 0, 'median_peer_ases': high['num_pattern_peer_ases'].median() if high_count else None, 'median_sessions': high['num_pattern_sessions'].median() if high_count else None, 'median_peer_visibility_pct': high['peer_visibility_pct'].median() if high_count else None, 'median_session_visibility_pct': high['session_visibility_pct'].median() if high_count else None, 'direct_peer_patterns': direct_count, 'direct_peer_pct': pct(direct_count, high_count), 'top1_rs_concentration_pct': top_n_share(rs_concentration, 1), 'top5_rs_concentration_pct': top_n_share(rs_concentration, 5), 'top10_rs_concentration_pct': top_n_share(rs_concentration, 10), 'top10_introducing_as_concentration_pct': top_n_share(introducing_concentration, 10), 'top10_prefix_concentration_pct': top_n_share(prefix_concentration, 10)}
    write_single_row_csv(output_dir / 'high_visibility_summary.csv', summary)
    high.write_parquet(output_dir / 'high_visibility_patterns.parquet', compression='zstd', compression_level=ZSTD_LEVEL)
    return summary

def process_month(month):
    month_start = time.perf_counter()
    print()
    print()
    print('#' * 72)
    print(f'VISIBILITY ANALYSIS: {month}')
    print('#' * 72)
    month_dir, poisoning_dir, rib_files, poisoning_files = validate_month(month)
    output_dirs = prepare_output_dirs(month)
    visibility_file = output_dirs['root'] / 'noncons_visibility.parquet'
    print(f'RIB directory       : {month_dir}')
    print(f'Poisoning directory : {poisoning_dir}')
    print(f"Output directory    : {output_dirs['root']}")
    print(f'RIB files           : {len(rib_files)}')
    print(f'Poisoning files     : {len(poisoning_files)}')
    if validate_visibility_file(visibility_file, poisoning_files):
        print()
        print('=' * 72)
        print('EXISTING VISIBILITY TABLE FOUND')
        print('=' * 72)
        print(f'Loading             : {visibility_file}')
        print('Skipping Phases 1-4.')
        visibility = pl.read_parquet(visibility_file)
        print(f'Patterns             : {visibility.height:,}')
        print(f"Poisoning prefixes    : {visibility['prefix'].n_unique():,}")
    else:
        if visibility_file.exists():
            print(f'Removing invalid/partial visibility file: {visibility_file}')
            visibility_file.unlink()
        observations = load_noncons_observations(poisoning_files)
        patterns = build_pattern_visibility(observations)
        poisoning_prefixes = patterns['prefix'].unique().to_list()
        del observations
        phase3_checkpoint_dir = output_dirs['root'] / 'phase3_checkpoints'
        denominators = build_prefix_denominators(rib_files, poisoning_prefixes, phase3_checkpoint_dir)
        visibility = build_visibility_table(patterns, denominators, visibility_file)
        del patterns
        del denominators
    localization = analyze_localization(visibility, output_dirs['localization'])
    direct_peer = analyze_direct_peer(visibility, output_dirs['direct_peer'])
    multi_collector = analyze_multi_collector(visibility, output_dirs['multi_collector'])
    high_visibility = analyze_high_visibility(visibility, output_dirs['high_visibility'])
    elapsed = time.perf_counter() - month_start
    print()
    print('=' * 72)
    print(f'MONTH COMPLETE: {month}')
    print(f'Patterns            : {visibility.height:,}')
    print(f'Runtime             : {format_duration(elapsed)}')
    print(f"Results             : {output_dirs['root']}")
    print('=' * 72)

def main():
    total_start = time.perf_counter()
    print()
    print('=' * 72)
    print('PRODUCTION NON-CONS VISIBILITY ANALYSIS')
    print('=' * 72)
    print(f'Input root          : {PARQUET_DIR}')
    print(f'Output root         : {OUTPUT_ROOT}')
    print(f"Months              : {', '.join(MONTHS)}")
    print('Pattern identity    : (prefix, rs_path)')
    print('Session identity    : (collector, peer_ip)')
    print(f'High-vis threshold  : > {HIGH_VISIBILITY_THRESHOLD:.0f}%')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        process_month(month)
    total_elapsed = time.perf_counter() - total_start
    print()
    print('=' * 72)
    print('ALL VISIBILITY ANALYSES COMPLETE')
    print('=' * 72)
    print(f'Months processed    : {len(MONTHS)}')
    print(f'Total runtime       : {format_duration(total_elapsed)}')
    print(f'Results             : {OUTPUT_ROOT}')
    print('=' * 72)
if __name__ == '__main__':
    main()
