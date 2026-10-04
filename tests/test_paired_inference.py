"""Statistical checks of complete-episode pairing and numerical tail reporting."""
import json
import math

import numpy as np
import pytest
from scipy import special, stats

from lowdim_games.paired_inference import (
    bootstrap_mean_band, holm_adjust, paired_difference_summary,
)


def test_paired_mean_t_test_and_family_interval_match_scipy():
    differences = np.array([-0.11, -0.05, 0.03, -0.08, 0.01, -0.04, 0.02, -0.09])
    record = paired_difference_summary(differences, bootstrap_resamples=299, rng=11)
    expected = stats.ttest_rel(differences, np.zeros(len(differences)))
    interval = expected.confidence_interval(confidence_level=.95)
    np.testing.assert_allclose(record["t_ci_95"], [interval.low, interval.high], rtol=2e-14)
    assert record["mean_difference"] == pytest.approx(differences.mean())
    assert record["sample_standard_deviation"] == pytest.approx(differences.std(ddof=1))
    assert record["standard_error"] == pytest.approx(differences.std(ddof=1) / np.sqrt(len(differences)))
    assert record["t_statistic"] == pytest.approx(expected.statistic)
    assert record["p_value"] == pytest.approx(expected.pvalue)
    assert record["d_z"] == pytest.approx(differences.mean() / differences.std(ddof=1))
    assert record["degrees_of_freedom"] == len(differences) - 1
    family_interval = expected.confidence_interval(confidence_level=1-.05/3)
    np.testing.assert_allclose(record["t_ci_family"], [family_interval.low, family_interval.high], rtol=2e-14)
    assert record["family_interval_confidence_level"] == pytest.approx(1-.05/3)
    assert record["t_ci_family"][0] < record["t_ci_95"][0]
    assert record["t_ci_family"][1] > record["t_ci_95"][1]
    json.dumps(record, allow_nan=False)


def test_symmetric_nonconstant_zero_mean_is_valid_t_test_not_constant():
    record = paired_difference_summary([-2, -1, 0, 1, 2], bootstrap_resamples=99, rng=12)
    assert record["p_value"] == 1
    assert record["t_statistic"] == 0
    assert record["d_z"] == 0
    assert record["inference_status"] == "ok"
    assert not record["degenerate_empirical_intervals"]
    assert record["wins"] == 2
    assert record["ties"] == 1
    assert record["losses"] == 2
    assert record["win_fraction"] + record["tie_fraction"] + record["loss_fraction"] == 1


@pytest.mark.parametrize("constant,status", [(0., "constant_zero"), (-.25, "constant_nonzero")])
def test_constant_differences_do_not_invent_significance_or_infinite_effect(constant, status):
    record = paired_difference_summary(np.full(20, constant), bootstrap_resamples=99, rng=13)
    assert record["p_value"] is None
    assert record["log_p_value"] is None
    assert record["negative_log10_p_value"] is None
    assert record["t_statistic"] is None
    assert record["d_z"] is None
    assert record["standard_error"] == 0
    assert record["inference_status"] == status
    assert record["degenerate_empirical_intervals"]
    assert not record["p_value_underflow"]
    np.testing.assert_allclose(record["t_ci_95"], [constant, constant])
    np.testing.assert_allclose(record["bootstrap_mean_ci_95"], [constant, constant])
    json.dumps(record, allow_nan=False)


def test_extreme_valid_t_tail_is_flagged_underflow_and_retains_finite_log_p():
    differences = -1 + np.arange(200) * 1e-10
    record = paired_difference_summary(differences, bootstrap_resamples=199, rng=14)
    assert record["p_value"] == 0
    assert record["p_value_underflow"]
    assert math.isfinite(record["negative_log10_p_value"])
    assert record["negative_log10_p_value"] > 1000
    # In this tiny-x incomplete-beta tail, its hypergeometric correction differs
    # from one by less than machine precision. This is an independent analytic
    # check on the log-scale result where direct SciPy survival has underflowed.
    df = len(differences) - 1
    t = abs(record["t_statistic"])
    a = df / 2
    log_x = math.log(df) - np.logaddexp(math.log(df), 2 * math.log(t))
    expected_log_p = a * log_x - math.log(a) - special.betaln(a, .5)
    assert record["log_p_value"] == pytest.approx(expected_log_p, abs=1e-10)
    assert record["negative_log10_p_value"] == pytest.approx(-expected_log_p / math.log(10))
    json.dumps(record, allow_nan=False)


def test_scalar_bootstrap_matches_direct_episode_resampling_and_is_reproducible():
    differences = np.array([-5., -1., 2., 4., 9.])
    resamples, seed = 401, 15
    direct_rng = np.random.default_rng(seed)
    direct = differences[direct_rng.integers(0, len(differences), size=(resamples, len(differences)))].mean(axis=1)
    expected = np.quantile(direct, [.025, .975])
    a = paired_difference_summary(differences, bootstrap_resamples=resamples, rng=seed)
    b = paired_difference_summary(differences, bootstrap_resamples=resamples, rng=seed)
    assert a == b
    np.testing.assert_allclose(a["bootstrap_mean_ci_95"], expected, atol=2e-15)
    np.testing.assert_allclose(a["bootstrap_mean_ci_family"],
                               np.quantile(direct, [.05 / 6, 1-.05 / 6]), atol=2e-15)


def test_curve_bootstrap_resamples_whole_episode_rows_with_shared_time_weights():
    episode_value = np.array([-4., -1., 0., 3., 7.])
    # Perfect within-episode dependence: every draw must preserve these affine
    # identities. Independent resampling of columns would break this check.
    values = np.column_stack([episode_value, 2 * episode_value + 5, 8 - episode_value])
    resamples, seed = 403, 16
    indices = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    direct = values[indices].mean(axis=1)
    mean, lower, upper = bootstrap_mean_band(values, n_resamples=resamples, rng=seed)
    np.testing.assert_allclose(mean, values.mean(axis=0))
    np.testing.assert_allclose(lower, np.quantile(direct, .025, axis=0), atol=2e-15)
    np.testing.assert_allclose(upper, np.quantile(direct, .975, axis=0), atol=2e-15)
    assert lower[1] == pytest.approx(2 * lower[0] + 5)
    assert upper[1] == pytest.approx(2 * upper[0] + 5)
    assert lower[2] == pytest.approx(8 - upper[0])
    assert upper[2] == pytest.approx(8 - lower[0])
    again = bootstrap_mean_band(values, n_resamples=resamples, rng=seed)
    for first, repeated in zip((mean, lower, upper), again):
        np.testing.assert_array_equal(first, repeated)


def test_same_rng_keeps_identical_resampling_weights_for_paired_curve_calls():
    first = np.array([[1., 4.], [2., 3.], [7., 1.], [5., 9.]])
    second = 3 * first - 2
    _, low_a, high_a = bootstrap_mean_band(first, n_resamples=301, rng=17)
    _, low_b, high_b = bootstrap_mean_band(second, n_resamples=301, rng=17)
    np.testing.assert_allclose(low_b, 3 * low_a - 2, atol=3e-15)
    np.testing.assert_allclose(high_b, 3 * high_a - 2, atol=3e-15)


def test_holm_order_monotonicity_clipping_ties_and_undefined_tests():
    np.testing.assert_allclose(holm_adjust([.04, .001, .03, .02]), [.06, .004, .06, .06])
    np.testing.assert_allclose(holm_adjust([.01, .01, .9]), [.03, .03, .9])
    np.testing.assert_allclose(holm_adjust([.8, .9]), [1., 1.])
    np.testing.assert_allclose(holm_adjust([None, .01, .04]), [1., .03, .08])
    np.testing.assert_array_equal(holm_adjust([]), np.array([]))
    np.testing.assert_allclose(holm_adjust([0.0, 1.0]), [0.0, 1.0])


@pytest.mark.parametrize("values", [[], [1.], [[1., 2.], [3., 4.]], [0., np.nan], [0., np.inf]])
def test_scalar_rejects_missing_replication_wrong_shape_and_nonfinite_data(values):
    with pytest.raises(ValueError, match="episode"):
        paired_difference_summary(values, bootstrap_resamples=99)


@pytest.mark.parametrize("values", [[], [[1., 2.]], [1., 2.], [[], []], [[1., np.nan], [2., 3.]]])
def test_curve_rejects_missing_replication_wrong_shape_and_empty_grid(values):
    with pytest.raises(ValueError):
        bootstrap_mean_band(values, n_resamples=99)


@pytest.mark.parametrize("kwargs", [{"bootstrap_resamples": 0}, {"bootstrap_resamples": 2.5},
                                    {"bootstrap_resamples": True}, {"family_size": 0},
                                    {"family_size": False}, {"confidence_level": 1.0},
                                    {"confidence_level": np.nan}])
def test_scalar_rejects_invalid_protocol_settings(kwargs):
    with pytest.raises(ValueError):
        paired_difference_summary([-1., 2., 3.], **kwargs)


@pytest.mark.parametrize("pvalues", [[-.01], [1.01], [np.nan], [np.inf], [[.1, .2]]])
def test_holm_rejects_invalid_probabilities(pvalues):
    with pytest.raises(ValueError):
        holm_adjust(pvalues)
