# Non-Consecutive AS Repetition

This repository contains code and processed datasets for studying non-consecutive AS repetition in BGP AS paths.

## Overview

The repository provides scripts for extracting BGP AS paths, identifying non-consecutive AS repetition, extracting repetition segments (RSs), and filtering segments associated with PEERING and Education and Research networks.

It also provides an analysis of the visibility of non-consecutive AS repetition across RIPE RIS collector peers, using February 2024 as a baseline month and October 2024 as a peak month.

The longitudinal study covers January 2014 through December 2024, using three sampling days per month. The visibility analysis uses daily RIPE RIS routing snapshots collected at 16:00 UTC during February and October 2024.

## Repository Structure

```text
as-repetition/
├── README.md
├── LICENSE
├── scripts/
│   ├── aspath.dataset.py
│   ├── conversion.py
│   ├── poisoning.extractor.py
│   ├── segment.extractor.py
│   ├── filter.segment.py
│   └── [visibility analysis script]
├── data/
│   ├── asdb/
│   │   └── YYYY-MM_categorized_ases.csv
│   ├── prefix.aspth.parquet/
│   │   └── poisoning/
│   │       ├── poisoning.YYYY-MM-DD.parquet
│   │       ├── segments/
│   │       │   └── segments.YYYY-MM-DD.parquet
│   │       └── segments.filtered/
│   │           └── segments.YYYY-MM-DD.parquet
│   └── visibility/
│       ├── 2024.02/
│       │   ├── poisoning/
│       │   │   └── rrc*.poisoning.parquet
│       │   └── all_collectors.prefix_visibility.parquet
│       └── 2024.10/
│           ├── poisoning/
│           │   └── rrc*.poisoning.parquet
│           └── all_collectors.prefix_visibility.parquet
└── figures/
    └── visibility/
        ├── visibility_ecdf_concentration_february_2024.pdf
        └── visibility_ecdf_concentration_october_2024.pdf
```

The visibility analysis script entry represents the implementation used to generate the figures; its final filename and location should match the committed repository.

## Artifact Description

### Scripts

- `aspath.dataset.py`: Extracts sampled BGP AS paths from RIPE RIS and Route Views.
- `conversion.py`: Converts extracted AS-path datasets to Parquet.
- `poisoning.extractor.py`: Identifies announcements containing non-consecutive AS repetition.
- `segment.extractor.py`: Extracts repetition segments from identified announcements.
- `filter.segment.py`: Excludes selected PEERING and Education and Research repetition segments.
- Visibility analysis script: Computes normalized visibility across RIPE RIS peers and generates the ECDF figures for February and October 2024.

### Longitudinal Datasets

**`data/prefix.aspth.parquet/poisoning/`**

Contains 396 daily Parquet files identifying announcements with non-consecutive AS repetition.

**`data/prefix.aspth.parquet/poisoning/segments/`**

Contains 396 daily Parquet files with the extracted repetition segments before filtering.

**`data/prefix.aspth.parquet/poisoning/segments.filtered/`**

Contains 396 daily Parquet files after applying the PEERING and Education and Research exclusions.

**`data/asdb/`**

Expected directory for six ASdb classification snapshots used to identify Education and Research networks. These external files are not included in the repository.

### Visibility Datasets

**`data/visibility/2024.02/` and `data/visibility/2024.10/`**

Contain the processed RIPE RIS datasets used for the visibility analysis.

Each monthly directory contains:

- `poisoning/`: Per-collector Parquet files containing announcements with non-consecutive AS repetition.
- `all_collectors.prefix_visibility.parquet`: Prefix-level information about the collectors, peer ASes, and collector-peer sessions observing each prefix.

The February dataset contains 667 poisoning files, while the October dataset contains 713.

These processed datasets are required to reproduce the visibility figures directly. Their inclusion in the Git repository depends on the final data-release configuration.

## Filtering

The filtering script excludes repetition segments introduced by PEERING AS47065 throughout the observation period. It also excludes segments in which every unique AS belongs to the ASdb Education and Research category, using the six corresponding monthly classification snapshots.

The Education and Research exclusion applies only to the six months for which classification snapshots are used.

## Visibility Analysis

The visibility analysis examines how widely prefix–RS patterns are observed across RIPE RIS collector peers.

We define a prefix–RS pattern as a unique combination of a prefix and repetition segment. For each pattern, the analysis computes two normalized visibility metrics:

- **Peer-AS visibility:** The fraction of distinct peer ASes observing the prefix that also observe the prefix with the corresponding RS.
- **Collector-peer session visibility:** The fraction of collector-peer sessions observing the prefix that also observe the prefix with the corresponding RS.

Both metrics are expressed as percentages.

The analysis generates ECDFs for three distributions:

1. All prefix–RS patterns.
2. Prefix–RS patterns excluding the two most prevalent RSs, ranked by the number of unique prefix–RS patterns.
3. Maximum normalized visibility per distinct RS.

The implementation uses DuckDB with disk-backed aggregation to limit memory consumption when processing large datasets. Temporary intermediate files are automatically removed after each monthly computation.

### Reproduced Dataset Statistics

| Metric | February 2024 | October 2024 |
|---|---:|---:|
| Unique prefix–RS patterns | 106,168 | 261,470 |
| Distinct RSs | 1,215 | 1,317 |
| Patterns associated with the two most prevalent RSs | 86,728 | 225,333 |
| Patterns remaining after excluding the top two RSs | 19,440 | 36,137 |
| Invalid visibility records | 0 | 0 |

### Visibility Results

| Distribution | February 2024 | October 2024 |
|---|---:|---:|
| All patterns: peer-AS visibility ≤10% | 97.85% | 99.47% |
| Excluding top two: peer-AS visibility ≤10% | 95.38% | 96.90% |
| All patterns: session visibility ≤10% | 97.87% | 99.48% |
| Excluding top two: session visibility ≤10% | 95.43% | 96.96% |

The results show that most prefix–RS patterns have limited visibility across collector peers. This observation persists after excluding the two most prevalent RSs.

### Generated Figures

The visibility analysis produces:

- `figures/visibility/visibility_ecdf_concentration_february_2024.pdf`
- `figures/visibility/visibility_ecdf_concentration_october_2024.pdf`

## Data Availability

The underlying BGP data are publicly available from:

- RIPE RIS: https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/
- Route Views: https://www.routeviews.org/

The longitudinal processed datasets are organized into 396 daily files per dataset.

The ASdb classification files must be obtained separately.

The visibility analysis requires the monthly processed poisoning files and prefix-visibility checkpoints described above. These files must be available locally to reproduce the figures.

## License

The source code is released under the MIT License. See [LICENSE](LICENSE). Third-party datasets remain subject to their respective terms.