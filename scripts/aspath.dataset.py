
#!/usr/bin/env python3

import argparse
import bz2
import calendar
import csv
import datetime as dt
import logging
import os
import sqlite3
import tempfile
from pathlib import Path

import pybgpstream
from tqdm import tqdm

COLLECTORS = (
    "rrc00", "rrc01", "rrc19", "rrc25",
    "route-views.amsix", "route-views.linx",
    "route-views.napafrica", "route-views.saopaulo",
)
FIELDS = ("date", "timestamp", "prefix", "as_path", "origin")


def sampled_days(start, end):
    for year in range(start, end + 1):
        for month in range(1, 13):
            last = calendar.monthrange(year, month)[1]
            for day in (1, last // 2, last):
                yield dt.date(year, month, day)


def extract_day(day, output_dir, overwrite=False):
    date = day.isoformat()
    destination = output_dir / f"prefix.aspath.{date}.bz2"
    if destination.exists() and not overwrite:
        return "skipped", 0

    next_date = (day + dt.timedelta(days=1)).isoformat()

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        tmp = Path(tmp)
        db = sqlite3.connect(tmp / "paths.sqlite")
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("""
                CREATE TABLE paths (
                    prefix TEXT NOT NULL,
                    as_path TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    PRIMARY KEY (prefix, as_path, origin)
                ) WITHOUT ROWID
            """)

            for collector in COLLECTORS:
                stream = pybgpstream.BGPStream(
                    from_time=f"{date}T00:00:00",
                    until_time=f"{next_date}T00:00:00",
                    collectors=[collector],
                )
                stream.add_filter("record-type", "ribs")

                total = 0
                for elem in stream:
                    prefix, path = elem.fields.get("prefix"), elem.fields.get("as-path")
                    if not prefix or not path or not path.strip():
                        continue

                    path = path.strip()
                    db.execute(
                        "INSERT OR IGNORE INTO paths VALUES (?, ?, ?, ?)",
                        (prefix, path, path.split()[-1], int(elem.time)),
                    )
                    total += 1

                db.commit()
                logging.info("%s %s: %d usable records", date, collector, total)

            count = db.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
            if not count:
                raise RuntimeError(f"No RIB data extracted for {date}")

            output = tmp / destination.name
            with bz2.open(output, "wt", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(FIELDS)
                for prefix, path, origin, timestamp in db.execute(
                    "SELECT prefix, as_path, origin, timestamp FROM paths"
                ):
                    writer.writerow((date, timestamp, prefix, path, origin))

            os.replace(output, destination)
            return "written", count
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=dt.date.fromisoformat)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.output_dir / "dataset_creation.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    days = [args.date] if args.date else list(sampled_days(args.start_year, args.end_year))

    for day in tqdm(days, desc="RIB extraction", unit="day"):
        try:
            status, count = extract_day(day, args.output_dir, args.overwrite)
            tqdm.write(f"{day}: {status} ({count:,} distinct paths)")
        except Exception:
            logging.exception("Extraction failed: %s", day)
            raise


if __name__ == "__main__":
    main()
