"""Fetch anonymous daily Hazard-request aggregates from NYC Open Data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowdim_games.data import download_daily_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--years", type=int, nargs="+", default=[2019, 2021, 2022])
    parser.add_argument("--refresh", action="store_true", help="Fetch a new snapshot instead of verifying/reusing cache")
    args = parser.parse_args()
    for year in args.years:
        result = download_daily_year(args.data_dir, year, refresh=args.refresh)
        print(json.dumps({"year": year, "days": result["calendar_days"],
                          "requests": result["included_requests"],
                          "excluded_unknown_borough": result["excluded_unknown_borough_requests"],
                          "csv_sha256": result["csv_sha256"]}))


if __name__ == "__main__":
    main()
