"""Public aggregate demand data and an explicitly quantized finite game.

The source data measure registered requests, not inspections, staffing effort,
or unresolved queues.  Profile fitting sees training data only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
DATASET_ID = "mu46-p9is"
SOURCE_URL = "https://data.cityofnewyork.us/Environment/Forestry-Service-Requests/mu46-p9is"
API_URL = "https://data.cityofnewyork.us/resource/mu46-p9is.json"
CATEGORY = "Hazard"
DATE_FIELD = "createddate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DailyDemand:
    dates: tuple[str, ...]
    counts: np.ndarray
    boroughs: tuple[str, ...] = BOROUGHS


@dataclass(frozen=True)
class QuantizedDemand:
    indices: np.ndarray
    mixtures: np.ndarray
    projected: np.ndarray
    errors_l2: np.ndarray
    errors_l1: np.ndarray
    raw_l2_norms: np.ndarray
    raw_l1_totals: np.ndarray

    def summary(self) -> dict:
        return {
            "rounds": int(len(self.indices)),
            "mean_l2": float(np.mean(self.errors_l2)),
            "median_l2": float(np.median(self.errors_l2)),
            "p95_l2": float(np.quantile(self.errors_l2, 0.95)),
            "max_l2": float(np.max(self.errors_l2)),
            "mean_l1": float(np.mean(self.errors_l1)),
            "relative_l2_rms": (float(np.linalg.norm(self.errors_l2) / np.linalg.norm(self.raw_l2_norms))
                                if np.linalg.norm(self.raw_l2_norms) > 0 else None),
            "relative_l1_total": (float(self.errors_l1.sum() / self.raw_l1_totals.sum())
                                  if self.raw_l1_totals.sum() > 0 else None),
            "visited_profiles": int(len(np.unique(self.indices))),
            "realized_simplex_affine_dimension": int(len(np.unique(self.indices)) - 1),
        }


@dataclass(frozen=True)
class NYCBenchmark:
    train: DailyDemand
    test: DailyDemand
    profiles: np.ndarray
    quantized_test: QuantizedDemand


def cache_paths(data_dir: str | Path, year: int) -> tuple[Path, Path]:
    stem = f"nyc_forestry_hazard_daily_{int(year)}"
    directory = Path(data_dir)
    return directory / f"{stem}.csv", directory / f"{stem}.provenance.json"


def _days_in_year(year: int) -> tuple[str, ...]:
    first, stop = date(year, 1, 1), date(year + 1, 1, 1)
    return tuple((first + timedelta(days=i)).isoformat() for i in range((stop - first).days))


def aggregate_query(year: int, *, offset: int = 0, page_size: int = 5000) -> str:
    """The server returns day/borough counts; personal records never leave it."""
    year = int(year)
    if not 2015 <= year <= 2100 or offset < 0 or page_size < 1:
        raise ValueError("Invalid year or pagination parameters")
    params = {
        "$select": f"date_trunc_ymd({DATE_FIELD}) as day, boroughcode as borough, count(*) as requests",
        "$where": (f"{DATE_FIELD} >= '{year}-01-01T00:00:00' AND "
                   f"{DATE_FIELD} < '{year + 1}-01-01T00:00:00' AND srcategory = '{CATEGORY}'"),
        "$group": "day, borough",
        "$order": "day, borough",
        "$limit": str(page_size),
        "$offset": str(offset),
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "AAMAS-reproducibility-data-client/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except (OSError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Unable to fetch public aggregate data: {url}") from last_error


def _aggregate_rows(rows: list[dict], year: int) -> tuple[DailyDemand, dict]:
    days = _days_in_year(year)
    positions = {day: i for i, day in enumerate(days)}
    borough_index = {name.casefold(): i for i, name in enumerate(BOROUGHS)}
    counts = np.zeros((len(days), len(BOROUGHS)), dtype=np.int64)
    excluded_requests, excluded_groups = 0, 0
    seen = set()
    for row in rows:
        day = str(row["day"])[:10]
        if day not in positions:
            raise ValueError("Source returned a day outside the requested year")
        borough = str(row.get("borough", "")).strip().casefold()
        key = (day, borough)
        if key in seen:
            raise ValueError("Duplicate day/borough aggregate; refusing to double count")
        seen.add(key)
        amount = int(row["requests"])
        if amount < 0:
            raise ValueError("Negative request count")
        if borough not in borough_index:
            excluded_requests += amount
            excluded_groups += 1
            continue
        counts[positions[day], borough_index[borough]] = amount
    if int(counts.sum()) == 0:
        raise ValueError("No nonzero known-borough demand; no synthetic fallback is used")
    return DailyDemand(days, counts), {
        "source_aggregate_rows": len(rows),
        "included_requests": int(counts.sum()),
        "excluded_unknown_borough_requests": excluded_requests,
        "excluded_unknown_borough_groups": excluded_groups,
        "calendar_days": len(days),
        "zero_demand_days": int(np.sum(counts.sum(axis=1) == 0)),
    }


def _csv_bytes(series: DailyDemand) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("date",) + BOROUGHS)
    for day, counts in zip(series.dates, series.counts):
        writer.writerow((day,) + tuple(int(value) for value in counts))
    return stream.getvalue().encode("utf-8")


def download_daily_year(data_dir: str | Path, year: int, *, refresh: bool = False,
                        fetch: Callable[[str], bytes] | None = None) -> dict:
    """Download only aggregate counts, hash them, and retain exact query provenance."""
    csv_path, provenance_path = cache_paths(data_dir, year)
    if not refresh and csv_path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != provenance["csv_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {csv_path}")
        return provenance
    if not refresh and (csv_path.exists() or provenance_path.exists()):
        raise ValueError("Incomplete cache: rerun with refresh=True to replace both files")
    fetch = fetch or _http_get
    rows, pages = [], []
    page_size, offset = 5000, 0
    while True:
        url = aggregate_query(year, offset=offset, page_size=page_size)
        body = fetch(url)
        page = json.loads(body)
        if not isinstance(page, list):
            raise ValueError("Expected a JSON list from Socrata")
        pages.append({"url": url, "response_sha256": hashlib.sha256(body).hexdigest(), "rows": len(page)})
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    series, details = _aggregate_rows(rows, int(year))
    content = _csv_bytes(series)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL,
        "api_url": API_URL,
        "category": CATEGORY,
        "timestamp_field": DATE_FIELD,
        "calendar_interpretation": "Published floating local timestamps; date_trunc_ymd, no UTC conversion",
        "year": int(year),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "csv_sha256": hashlib.sha256(content).hexdigest(),
        "stored_fields": ["date"] + list(BOROUGHS),
        "not_stored": "Request IDs, addresses, coordinates, names, notes, or request-level records",
        "interpretation": "Registered Hazard request counts; not staffing effort or counterfactual inspections",
        **details,
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(content)
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return provenance


def load_daily_counts(data_dir: str | Path, years: Iterable[int]) -> DailyDemand:
    years = tuple(sorted(int(year) for year in years))
    if not years or len(set(years)) != len(years):
        raise ValueError("Specify distinct, nonempty years")
    all_dates, all_counts = [], []
    for year in years:
        csv_path, provenance_path = cache_paths(data_dir, year)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        content = csv_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != provenance["csv_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {csv_path}")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        if reader.fieldnames != ["date"] + list(BOROUGHS):
            raise ValueError("Unexpected demand CSV columns")
        records = list(reader)
        expected_dates = _days_in_year(year)
        if tuple(row["date"] for row in records) != expected_dates:
            raise ValueError("CSV must contain every calendar day exactly once in chronological order")
        counts = np.asarray([[int(row[name]) for name in BOROUGHS] for row in records], dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("Negative demand")
        all_dates.extend(expected_dates)
        all_counts.append(counts)
    return DailyDemand(tuple(all_dates), np.vstack(all_counts))


def _validated_counts(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Counts must be a nonempty, finite, nonnegative matrix")
    return values


def fit_profile_library(train_counts: np.ndarray, n_profiles: int = 6, *, seed: int = 20261003,
                        n_init: int = 8, max_iter: int = 100) -> np.ndarray:
    """NumPy k-means++/Lloyd; Euclidean counts, training observations only.

    Centroids represent modeled joint demand, and need not be integer counts.
    No test quantile, capacity, threshold, or test observation enters fitting.
    """
    values = _validated_counts(train_counts)
    if not 1 <= n_profiles <= len(np.unique(values, axis=0)) or n_init < 1 or max_iter < 1:
        raise ValueError("Invalid number of profiles, initializations, or iterations")
    rng = np.random.default_rng(seed)
    best_centers, best_inertia = None, float("inf")
    for _ in range(n_init):
        centers = [values[int(rng.integers(len(values)))].copy()]
        while len(centers) < n_profiles:
            distance2 = np.min(np.sum((values[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
            centers.append(values[int(rng.choice(len(values), p=distance2 / distance2.sum()))].copy())
        centers = np.asarray(centers)
        for _ in range(max_iter):
            distance2 = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distance2, axis=1)
            updated = centers.copy()
            for i in range(n_profiles):
                selected = values[labels == i]
                if len(selected):
                    updated[i] = selected.mean(axis=0)
            if np.max(np.abs(updated - centers)) <= 1e-10:
                centers = updated
                break
            centers = updated
        inertia = float(np.sum(np.min(np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)))
        if inertia < best_inertia:
            best_inertia, best_centers = inertia, centers.copy()
    # Stable public numbering makes the library easy to inspect across runs.
    order = sorted(range(n_profiles), key=lambda i: (float(best_centers[i].sum()), *best_centers[i].tolist()))
    return best_centers[order]


def quantize_demands(counts: np.ndarray, profiles: np.ndarray) -> QuantizedDemand:
    values, centers = _validated_counts(counts), _validated_counts(profiles)
    if values.shape[1] != centers.shape[1]:
        raise ValueError("Demand and profile dimensions differ")
    distance2 = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    indices = np.argmin(distance2, axis=1)
    projected = centers[indices]
    residual = values - projected
    return QuantizedDemand(indices, np.eye(len(centers))[indices], projected,
                           np.linalg.norm(residual, axis=1), np.sum(np.abs(residual), axis=1),
                           np.linalg.norm(values, axis=1), np.sum(values, axis=1))


def build_nyc_benchmark(data_dir: str | Path, *, train_years: Iterable[int] = (2019,),
                        test_years: Iterable[int] = (2021, 2022), n_profiles: int = 6,
                        seed: int = 20261003) -> NYCBenchmark:
    train, test = load_daily_counts(data_dir, train_years), load_daily_counts(data_dir, test_years)
    if train.dates[-1] >= test.dates[0]:
        raise ValueError("Training observations must strictly precede all test observations")
    profiles = fit_profile_library(train.counts, n_profiles, seed=seed)
    return NYCBenchmark(train, test, profiles, quantize_demands(test.counts, profiles))
