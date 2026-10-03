"""Meaningful boundary checks for the public data and finite-game adapter."""
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from lowdim_games.data import (aggregate_query, build_nyc_benchmark, cache_paths,
                              download_daily_year, fit_profile_library,
                              load_daily_counts, quantize_demands)


class DataTests(unittest.TestCase):
    def test_server_aggregation_has_no_personal_fields(self):
        query = parse_qs(urlsplit(aggregate_query(2019)).query)
        self.assertIn("date_trunc_ymd(createddate)", query["$select"][0])
        self.assertEqual(query["$group"], ["day, borough"])
        self.assertNotIn("streetname", query["$select"][0])
        self.assertIn("2020-01-01", query["$where"][0])

    def test_missing_days_unknown_borough_and_cache_integrity(self):
        records = [{"day": "2019-01-02T00:00:00.000", "borough": "Queens", "requests": "7"},
                   {"day": "2019-01-02T00:00:00.000", "requests": "2"}]
        with tempfile.TemporaryDirectory() as directory:
            provenance = download_daily_year(directory, 2019, fetch=lambda url: json.dumps(records).encode())
            series = load_daily_counts(directory, [2019])
            self.assertEqual(series.counts.shape, (365, 5))
            self.assertEqual(series.counts[0].sum(), 0)
            self.assertEqual(series.counts[1, 3], 7)
            self.assertEqual(provenance["excluded_unknown_borough_requests"], 2)
            self.assertEqual(provenance["zero_demand_days"], 364)
            # Reuse must not access the network.
            download_daily_year(directory, 2019, fetch=lambda url: self.fail("Unexpected network request"))
            csv_path, _ = cache_paths(directory, 2019)
            csv_path.write_bytes(csv_path.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_daily_counts(directory, [2019])

    def test_duplicate_aggregate_and_empty_source_fail(self):
        record = {"day": "2019-01-01T00:00:00.000", "borough": "Bronx", "requests": "1"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                download_daily_year(directory, 2019, fetch=lambda url: json.dumps([record, record]).encode())
            with self.assertRaisesRegex(ValueError, "No nonzero"):
                download_daily_year(directory, 2019, fetch=lambda url: b"[]")

    def test_training_only_quantization_and_actual_vertex_dimension(self):
        train = np.array([[0., 0.], [0., 2.], [8., 8.], [8., 10.]])
        profiles = fit_profile_library(train, 2, seed=42)
        self.assertTrue(np.allclose(profiles, [[0., 1.], [8., 9.]]))
        self.assertTrue(np.array_equal(profiles, fit_profile_library(train, 2, seed=42)))
        result = quantize_demands(np.array([[0., 1.], [80., 90.], [0., 0.]]), profiles)
        self.assertTrue(np.allclose(result.mixtures.sum(axis=1), 1))
        self.assertTrue(np.allclose(result.projected, result.mixtures @ profiles))
        self.assertGreater(result.errors_l2[1], 100)
        self.assertEqual(result.summary()["realized_simplex_affine_dimension"], 1)
        self.assertTrue(np.allclose(profiles, [[0., 1.], [8., 9.]]))

    def test_overlapping_years_fail_instead_of_leaking(self):
        record = {"day": "2019-01-01T00:00:00.000", "borough": "Bronx", "requests": "1"}
        with tempfile.TemporaryDirectory() as directory:
            download_daily_year(directory, 2019, fetch=lambda url: json.dumps([record]).encode())
            with self.assertRaisesRegex(ValueError, "strictly precede"):
                build_nyc_benchmark(directory, train_years=[2019], test_years=[2019], n_profiles=1)


if __name__ == "__main__":
    unittest.main()
