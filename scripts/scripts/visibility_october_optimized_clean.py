#!/usr/bin/env python3
"""Optimized October 2024 non-cons visibility analysis.

This script does NOT read raw RIB files and does NOT modify the original
visibility.results tree. It rebuilds October pattern numerators from the
poisoning parquet files, reads the completed Phase-3 combined checkpoint,
and performs the prefix-denominator aggregation in bulk with Polars.

Measurement definitions are unchanged:
  pattern = (prefix, rs_path)
  session = (collector, peer_ip)
"""

from pathlib import Path
import time
import shutil
import polars as pl

BASE_DIR = Path('/data/jaw/rs.visibility.analysis')
PARQUET_DIR = BASE_DIR / 'parquet'
ORIGINAL_OUTPUT_ROOT = BASE_DIR / 'visibility.results'
OUTPUT_ROOT = BASE_DIR / 'visibility.results.optimized'
MONTH = '2024.10'
ZSTD_LEVEL = 10
HIGH_VISIBILITY_THRESHOLD = 50.0

POISONING_DIR = PARQUET_DIR / MONTH / 'poisoning'
COMBINED_CHECKPOINT = ORIGINAL_OUTPUT_ROOT / MONTH / 'phase3_checkpoints' / 'all_collectors.prefix_visibility.parquet'
MONTH_OUTPUT = OUTPUT_ROOT / MONTH
DENOMINATOR_FILE = MONTH_OUTPUT / 'prefix_denominators.parquet'
PATTERN_FILE = MONTH_OUTPUT / 'pattern_visibility.parquet'
VISIBILITY_FILE = MONTH_OUTPUT / 'noncons_visibility.parquet'


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


def atomic_write_parquet(df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + '.tmp')
    if temporary_file.exists():
        temporary_file.unlink()
    df.write_parquet(temporary_file, compression='zstd', compression_level=ZSTD_LEVEL)
    temporary_file.replace(output_file)


def validate_inputs():
    if not POISONING_DIR.exists():
        raise RuntimeError(f'ABORT: poisoning directory does not exist: {POISONING_DIR}')
    poisoning_files = sorted(POISONING_DIR.glob('rrc*.poisoning.parquet'))
    if not poisoning_files:
        raise RuntimeError(f'ABORT: no poisoning files found: {POISONING_DIR}')
    if not COMBINED_CHECKPOINT.exists():
        raise RuntimeError(f'ABORT: completed combined Phase-3 checkpoint not found: {COMBINED_CHECKPOINT}')
    required = {'prefix', 'collector', 'collector_peer_ases', 'collector_peer_ips', 'num_collector_sessions'}
    schema = pl.read_parquet_schema(COMBINED_CHECKPOINT)
    missing = required - set(schema.keys())
    if missing:
        raise RuntimeError('ABORT: combined checkpoint is missing columns: ' + ', '.join(sorted(missing)))
    return poisoning_files


def prepare_output_dirs():
    # Clean rerun: remove ONLY the independent optimized October output tree.
    # The original Phase-3 checkpoint remains untouched in visibility.results.
    if MONTH_OUTPUT.exists():
        print(f'Cleaning optimized output: {MONTH_OUTPUT}')
        shutil.rmtree(MONTH_OUTPUT)
    MONTH_OUTPUT.mkdir(parents=True, exist_ok=True)
    dirs = {
        'root': MONTH_OUTPUT,
        'localization': MONTH_OUTPUT / '01_localization',
        'direct_peer': MONTH_OUTPUT / '02_direct_peer',
        'multi_collector': MONTH_OUTPUT / '03_multi_collector',
        'high_visibility': MONTH_OUTPUT / '04_high_visibility',
    }
    for key in ['localization', 'direct_peer', 'multi_collector', 'high_visibility']:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def load_poisoning_observations(poisoning_files):
    print('\n' + '=' * 72)
    print('STEP 1: LOAD OCTOBER POISONING OBSERVATIONS')
    print('=' * 72)
    cols = ['collector', 'peer_asn', 'peer_ip', 'prefix', 'introducing_as', 'rs_path']
    frames = []
    total = 0
    for i, path in enumerate(poisoning_files, 1):
        df = pl.read_parquet(path, columns=cols)
        total += df.height
        frames.append(df)
        if i == 1 or i % 50 == 0 or i == len(poisoning_files):
            print(f'\r[{i}/{len(poisoning_files)}] rows={total:,}', end='', flush=True)
    print()
    observations = pl.concat(frames, how='vertical')
    del frames
    print(f'Non-cons observations: {observations.height:,}')
    print(f"Poisoning prefixes    : {observations['prefix'].n_unique():,}")
    print(f"Distinct RSs          : {observations['rs_path'].n_unique():,}")
    return observations


def build_patterns_bulk(observations):
    print('\n' + '=' * 72)
    print('STEP 2: BUILD PATTERN NUMERATORS IN BULK')
    print('=' * 72)

    # Preserve the original invariant: one introducing AS per rs_path.
    introducing_check = (
        observations.group_by('rs_path')
        .agg(pl.col('introducing_as').n_unique().alias('n'))
        .filter(pl.col('n') != 1)
    )
    if introducing_check.height:
        raise RuntimeError('ABORT: at least one rs_path maps to multiple introducing ASes.')

    # Session identity is exactly (collector, peer_ip).
    observations = observations.with_columns(
        (
            pl.col('collector').cast(pl.Utf8)
            + pl.lit('|')
            + pl.col('peer_ip').cast(pl.Utf8)
        ).alias('_session')
    )

    # Main pattern numerators.
    patterns = (
        observations.group_by(['prefix', 'rs_path'])
        .agg([
            pl.col('introducing_as').first().alias('introducing_as'),
            pl.count().alias('num_observations'),
            pl.col('collector').drop_nulls().unique().alias('pattern_collectors'),
            pl.col('collector').drop_nulls().n_unique().alias('num_pattern_collectors'),
            pl.col('peer_asn').drop_nulls().unique().alias('pattern_peer_ases'),
            pl.col('peer_asn').drop_nulls().n_unique().alias('num_pattern_peer_ases'),
            pl.col('_session').drop_nulls().unique().alias('pattern_sessions'),
            pl.col('_session').drop_nulls().n_unique().alias('num_pattern_sessions'),
        ])
    )

    # Direct-peer metrics are computed from observation-level rows BEFORE the
    # collector/peer-IP/peer-AS mapping is discarded. A direct-peer observation
    # is one for which peer_asn == introducing_as.
    direct = (
        observations
        .filter(
            pl.col('peer_asn').is_not_null()
            & pl.col('introducing_as').is_not_null()
            & (pl.col('peer_asn') == pl.col('introducing_as'))
        )
        .group_by(['prefix', 'rs_path'])
        .agg(
            pl.col('_session').drop_nulls().n_unique().alias('num_direct_peer_sessions')
        )
    )

    patterns = (
        patterns
        .join(direct, on=['prefix', 'rs_path'], how='left')
        .with_columns(
            pl.col('num_direct_peer_sessions').fill_null(0).cast(pl.UInt32)
        )
        .with_columns(
            (pl.col('num_direct_peer_sessions') > 0).alias('introducing_as_is_ris_peer')
        )
        .select([
            'prefix', 'rs_path', 'introducing_as', 'num_observations',
            'pattern_collectors', 'num_pattern_collectors',
            'pattern_peer_ases', 'num_pattern_peer_ases',
            'pattern_sessions', 'num_pattern_sessions',
            'introducing_as_is_ris_peer', 'num_direct_peer_sessions',
        ])
        .sort(['prefix', 'rs_path'])
    )

    # Direct-peer fields must be complete Booleans/counts, never null.
    direct_nulls = patterns.select([
        pl.col('introducing_as_is_ris_peer').is_null().sum().alias('bool_nulls'),
        pl.col('num_direct_peer_sessions').is_null().sum().alias('count_nulls'),
    ]).row(0)
    if direct_nulls[0] or direct_nulls[1]:
        raise RuntimeError(
            'ABORT: direct-peer fields contain nulls after construction: '
            f'bool={direct_nulls[0]}, count={direct_nulls[1]}'
        )

    # Cross-check the Boolean against the peer-AS set stored for the pattern.
    # list.contains is only a validation here; the authoritative computation
    # above uses the observation-level peer_asn/session mapping.
    direct_check = patterns.filter(
        pl.col('introducing_as_is_ris_peer')
        != pl.col('pattern_peer_ases').list.contains(pl.col('introducing_as'))
    )
    if direct_check.height:
        raise RuntimeError(
            f'ABORT: {direct_check.height:,} patterns fail the direct-peer peer-AS cross-check.'
        )

    direct_count = patterns.filter(pl.col('introducing_as_is_ris_peer')).height
    non_direct_count = patterns.height - direct_count
    print(f'Pattern rows          : {patterns.height:,}')
    print(f'Direct-peer patterns  : {direct_count:,}')
    print(f'Non-direct patterns   : {non_direct_count:,}')
    print('Direct-peer nulls     : 0')
    atomic_write_parquet(patterns, PATTERN_FILE)
    print(f'Pattern checkpoint    : {PATTERN_FILE}')
    return patterns


def build_denominators_bulk():
    print('\n' + '=' * 72)
    print('STEP 3: BUILD PREFIX DENOMINATORS IN BULK')
    print('=' * 72)
    print(f'Reading checkpoint    : {COMBINED_CHECKPOINT}')
    compact = pl.read_parquet(COMBINED_CHECKPOINT)
    print(f'Compact rows          : {compact.height:,}')
    print(f"Prefixes              : {compact['prefix'].n_unique():,}")

    # Collector denominator: unique collectors observing each prefix.
    collectors = (
        compact.select(['prefix', 'collector'])
        .drop_nulls()
        .unique()
        .group_by('prefix')
        .agg([
            pl.col('collector').unique().alias('prefix_collectors'),
            pl.col('collector').n_unique().alias('num_prefix_collectors'),
        ])
    )
    print('Collector denominators complete.')

    # Peer-AS denominator: union the per-collector peer-AS lists for each prefix.
    peer_ases = (
        compact.select(['prefix', 'collector_peer_ases'])
        .explode('collector_peer_ases')
        .drop_nulls()
        .unique()
        .group_by('prefix')
        .agg([
            pl.col('collector_peer_ases').unique().alias('prefix_peer_ases'),
            pl.col('collector_peer_ases').n_unique().alias('num_prefix_peer_ases'),
        ])
    )
    print('Peer-AS denominators complete.')

    # Session identity remains exactly (collector, peer_ip).
    sessions = (
        compact.select(['prefix', 'collector', 'collector_peer_ips'])
        .explode('collector_peer_ips')
        .drop_nulls()
        .with_columns(
            (pl.col('collector').cast(pl.Utf8) + pl.lit('|') + pl.col('collector_peer_ips').cast(pl.Utf8))
            .alias('_session')
        )
        .select(['prefix', '_session'])
        .unique()
        .group_by('prefix')
        .agg([
            pl.col('_session').unique().alias('prefix_sessions'),
            pl.col('_session').n_unique().alias('num_prefix_sessions'),
        ])
    )
    print('Session denominators complete.')

    denominators = (
        collectors.join(peer_ases, on='prefix', how='inner')
        .join(sessions, on='prefix', how='inner')
        .select([
            'prefix', 'prefix_collectors', 'num_prefix_collectors',
            'prefix_peer_ases', 'num_prefix_peer_ases',
            'prefix_sessions', 'num_prefix_sessions',
        ])
        .sort('prefix')
    )

    expected = compact['prefix'].n_unique()
    if denominators.height != expected:
        raise RuntimeError(
            f'ABORT: denominator prefix count mismatch: expected {expected:,}, got {denominators.height:,}'
        )
    bad = denominators.filter(
        (pl.col('num_prefix_collectors') <= 0)
        | (pl.col('num_prefix_peer_ases') <= 0)
        | (pl.col('num_prefix_sessions') <= 0)
    )
    if bad.height:
        raise RuntimeError(f'ABORT: {bad.height:,} prefixes have zero denominators.')

    atomic_write_parquet(denominators, DENOMINATOR_FILE)
    print(f'Prefix denominators   : {denominators.height:,}')
    print(f'Denominator output    : {DENOMINATOR_FILE}')
    return denominators


def build_visibility(patterns, denominators):
    print('\n' + '=' * 72)
    print('STEP 4: COMPUTE NORMALIZED VISIBILITY')
    print('=' * 72)
    visibility = (
        patterns.join(denominators, on='prefix', how='left')
        .with_columns([
            (pl.col('num_pattern_collectors') / pl.col('num_prefix_collectors') * 100.0).alias('collector_visibility_pct'),
            (pl.col('num_pattern_peer_ases') / pl.col('num_prefix_peer_ases') * 100.0).alias('peer_visibility_pct'),
            (pl.col('num_pattern_sessions') / pl.col('num_prefix_sessions') * 100.0).alias('session_visibility_pct'),
        ])
        .sort(['prefix', 'rs_path'])
    )
    if visibility.height != patterns.height:
        raise RuntimeError('ABORT: pattern count changed during denominator join.')
    bad_den = visibility.filter(
        pl.col('num_prefix_collectors').is_null()
        | pl.col('num_prefix_peer_ases').is_null()
        | pl.col('num_prefix_sessions').is_null()
        | (pl.col('num_prefix_collectors') <= 0)
        | (pl.col('num_prefix_peer_ases') <= 0)
        | (pl.col('num_prefix_sessions') <= 0)
    )
    if bad_den.height:
        raise RuntimeError(f'ABORT: {bad_den.height:,} patterns have missing/zero denominators.')
    bad_vis = visibility.filter(
        (pl.col('collector_visibility_pct') > 100.000001)
        | (pl.col('peer_visibility_pct') > 100.000001)
        | (pl.col('session_visibility_pct') > 100.000001)
    )
    if bad_vis.height:
        raise RuntimeError(f'ABORT: {bad_vis.height:,} patterns exceed 100% visibility.')
    atomic_write_parquet(visibility, VISIBILITY_FILE)
    print(f'Patterns              : {visibility.height:,}')
    print(f'Visibility output     : {VISIBILITY_FILE}')
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


def main():
    start = time.perf_counter()
    print('\n' + '=' * 72)
    print('OPTIMIZED OCTOBER 2024 NON-CONS VISIBILITY ANALYSIS')
    print('=' * 72)
    print(f'Month                 : {MONTH}')
    print(f'Original checkpoint   : {COMBINED_CHECKPOINT}')
    print(f'Independent output    : {MONTH_OUTPUT}')
    print('Raw RIB access        : NONE')
    print('Pattern identity      : (prefix, rs_path)')
    print('Session identity      : (collector, peer_ip)')

    poisoning_files = validate_inputs()
    output_dirs = prepare_output_dirs()

    observations = load_poisoning_observations(poisoning_files)
    patterns = build_patterns_bulk(observations)
    poisoning_prefix_count = patterns['prefix'].n_unique()
    del observations

    denominators = build_denominators_bulk()
    if denominators.height != poisoning_prefix_count:
        raise RuntimeError(
            f'ABORT: poisoning-prefix mismatch: patterns contain {poisoning_prefix_count:,}, '
            f'denominators contain {denominators.height:,}'
        )

    visibility = build_visibility(patterns, denominators)
    del patterns
    del denominators

    analyze_localization(visibility, output_dirs['localization'])
    analyze_direct_peer(visibility, output_dirs['direct_peer'])
    analyze_multi_collector(visibility, output_dirs['multi_collector'])
    analyze_high_visibility(visibility, output_dirs['high_visibility'])

    elapsed = time.perf_counter() - start
    print('\n' + '=' * 72)
    print('OPTIMIZED OCTOBER ANALYSIS COMPLETE')
    print('=' * 72)
    print(f'Patterns              : {visibility.height:,}')
    print(f"Poisoning prefixes    : {visibility['prefix'].n_unique():,}")
    print(f'Runtime               : {format_duration(elapsed)}')
    print(f'Results               : {MONTH_OUTPUT}')
    print('=' * 72)


if __name__ == '__main__':
    main()
