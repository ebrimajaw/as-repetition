# Non-Consecutive AS Repetition

This repository contains code and processed datasets for studying non-consecutive AS repetition in BGP AS paths.

## Overview

The repository provides scripts for extracting BGP AS paths, identifying non-consecutive AS repetition, extracting repetition segments (RSs), and filtering segments associated with PEERING and Education and Research networks.

The datasets originate from publicly available RIPE RIS and Route Views routing data. The study covers January 2014 through December 2024, using three sampling days per month.

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
│   └── filter.segment.py
└── data/
    ├── asdb/
    │   └── YYYY-MM_categorized_ases.csv
    └── prefix.aspth.parquet/
        └── poisoning/
            ├── poisoning.YYYY-MM-DD.parquet
            ├── segments/
            │   └── segments.YYYY-MM-DD.parquet
            └── segments.filtered/
                └── segments.YYYY-MM-DD.parquet
```

## Artifact Description

**scripts/**

- `aspath.dataset.py`: Extracts sampled BGP AS paths from RIPE RIS and Route Views.
- `conversion.py`: Converts extracted AS-path datasets to Parquet.
- `poisoning.extractor.py`: Identifies announcements containing non-consecutive AS repetition.
- `segment.extractor.py`: Extracts repetition segments from identified announcements.
- `filter.segment.py`: Excludes selected PEERING and Education and Research repetition segments.

**data/prefix.aspth.parquet/poisoning/**

Contains 396 daily Parquet files identifying announcements with non-consecutive AS repetition.

**data/prefix.aspth.parquet/poisoning/segments/**

Contains 396 daily Parquet files with the extracted repetition segments before filtering.

**data/prefix.aspth.parquet/poisoning/segments.filtered/**

Contains 396 daily Parquet files after applying the PEERING and Education and Research exclusions.

**data/asdb/**

Expected directory for six ASdb classification snapshots used to identify Education and Research networks. These external files are not included in the repository.

## Filtering

The filtering script excludes repetition segments introduced by PEERING AS47065 throughout the observation period. It also excludes segments in which every unique AS belongs to the ASdb Education and Research category, using the six corresponding monthly classification snapshots.

The Education and Research exclusion applies only to the six months for which classification snapshots are used.

## Data Availability

The underlying BGP data are publicly available from:

- RIPE RIS: https://www.ripe.net/analyse/internet-measurements/routing-information-service-ris/
- Route Views: https://www.routeviews.org/

The processed datasets are organized into 396 daily files per dataset. The ASdb classification files must be obtained separately.

## License

The source code is released under the MIT License. See [LICENSE](LICENSE). Third-party datasets remain subject to their respective terms.
