"""Scientific integrity of the manuscript presentation inputs."""
from pathlib import Path

import numpy as np
import pytest

from scripts.build_aamas_paper_figures import FLOOR, load_inputs, log_summary


REPORT = Path(__file__).resolve().parents[1] / "results/cage_adaptation"


def test_unchanged_primary_decomposition_and_cumulative_units():
    inputs = load_inputs(REPORT)
    components = inputs["components"]
    differences = components[0] - components[1:]
    np.testing.assert_allclose(differences.sum(axis=1), inputs["costs"][0]-inputs["costs"][1:], atol=1e-12)
    expected_endpoint = 50 * 512 * differences.sum(axis=1)
    np.testing.assert_allclose(inputs["dynamics"]["cumulative_difference"][-1], expected_endpoint, atol=1e-8)
    assert differences.sum(axis=1)[0] > 0  # Window remains better on native cost.
    assert differences.sum(axis=1)[1] < 0  # The Hedge advantage is retained.


def test_floor_censoring_preserves_zero_inputs_and_reports_all_paths():
    values = np.array([[0.0, 2e-4], [FLOOR, 4e-4], [FLOOR/10, 6e-4]])
    unchanged = values.copy()
    summary = log_summary(values, axis=0)
    np.testing.assert_array_equal(values, unchanged)
    assert np.isnan(summary["median"][0])
    assert np.isnan(summary["low"][0])
    assert np.isnan(summary["high"][0])
    np.testing.assert_array_equal(summary["percent_unresolved"], [100, 0])
    assert summary["median"][1] == pytest.approx(4e-4)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_geometry_values_are_rejected(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        log_summary(np.array([[bad], [1e-4]]), axis=0)
