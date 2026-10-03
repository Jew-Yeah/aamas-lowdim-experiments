import numpy as np
import pytest

from lowdim_games.learners import OneSwitchLearner
from lowdim_games.switching_tracking import (
    TrackingLagSafeLearner, controlled_tracking_path, run_tracking_path,
    tracking_response, tracking_safe_budget, tracking_tensor,
)


def test_tracking_game_identity_target_and_normalization():
    rng = np.random.default_rng(21)
    tensor = tracking_tensor(4)
    for _ in range(20):
        p, ell = rng.dirichlet(np.ones(4), size=2)
        payoff = np.einsum("i,ijd,j->d", p, tensor, ell)
        np.testing.assert_allclose(payoff, (p - ell) / np.sqrt(2.0), atol=2e-16)
        np.testing.assert_allclose(np.einsum("i,ijd,j->d", ell, tensor, ell), 0, atol=2e-16)
    assert np.max(np.linalg.norm(tensor, axis=2)) == pytest.approx(1.0)
    assert tracking_safe_budget(0) == 0
    assert [tracking_safe_budget(h) for h in (1, 20, 1000)] == [1, 1, 1]


def test_lag_certificate_telescopes_for_every_prefix_of_mixture_path():
    path = np.random.default_rng(22).dirichlet(np.ones(5), size=81)
    safe = TrackingLagSafeLearner(tracking_tensor(5), tracking_response, len(path))
    total = np.zeros(5)
    uniform = np.full(5, 1 / 5)
    for t, ell in enumerate(path):
        np.testing.assert_allclose(safe.choose(), uniform if t == 0 else path[t - 1])
        record = safe.observe(ell)
        total += record["payoff"]
        np.testing.assert_allclose(total, (uniform - ell) / np.sqrt(2.0), atol=8e-16)
        assert np.linalg.norm(total) <= tracking_safe_budget(t + 1) + 1e-14
        assert record["mode"] == "safe"
        np.testing.assert_array_equal(record["target_witness"], np.zeros(5))


def test_master_crossing_round_is_fast_and_safe_tail_is_fresh():
    path = controlled_tracking_path(seed=23, horizon=512)
    run = run_tracking_path(path)
    master = run["trajectories"]["one_switch"]
    tau = run["summary"]["one_switch"]["switch_round"]
    # Exact E reaches 1 at round 129. Floating projection residuals can make
    # the monitored strict crossing occur before the third mode appears.
    assert 129 <= tau <= 257
    assert master["mode"][tau - 1] == "fast"
    assert master["switch"][tau - 1]
    assert master["E"][tau - 2] <= 1 < master["E"][tau - 1]
    assert master["mode"][tau] == "safe"
    assert master["safe_local_t"][tau] == 1
    np.testing.assert_allclose(master["actions"][tau], np.full(3, 1 / 3))
    np.testing.assert_allclose(master["actions"][tau + 1:], path[tau:-1])
    np.testing.assert_allclose(master["E"][tau:], master["E"][tau - 1])
    np.testing.assert_allclose(master["actions"][:tau],
        run["trajectories"]["shared_past_hull"]["actions"][:tau])
    assert not master["fast_residual_defined"][tau:].any()
    assert run["metadata"]["master_G_T"] == 1
    assert not run["metadata"]["learner_restarts"]


def test_no_switch_for_known_constant_path_and_equal_initial_information():
    path = np.repeat([[0.2, 0.5, 0.3]], 17, axis=0)
    run = run_tracking_path(path, window=4)
    assert run["summary"]["one_switch"]["switch_round"] is None
    for trajectory in run["trajectories"].values():
        np.testing.assert_allclose(trajectory["actions"][0], np.full(3, 1 / 3))
        np.testing.assert_allclose(trajectory["distances"],
            np.linalg.norm(trajectory["payoffs"].cumsum(axis=0), axis=1) / np.arange(1, len(path) + 1))
    np.testing.assert_allclose(run["trajectories"]["one_switch"]["actions"],
                               run["trajectories"]["shared_past_hull"]["actions"])


def test_future_changes_do_not_change_common_prefix_actions():
    path_a = controlled_tracking_path(seed=24, horizon=40)
    path_b = path_a.copy()
    path_b[15:] = np.random.default_rng(25).dirichlet(np.ones(3), size=25)
    a, b = run_tracking_path(path_a), run_tracking_path(path_b)
    for name in a["trajectories"]:
        # Action at round 16 precedes observation of the first changed ell.
        np.testing.assert_allclose(a["trajectories"][name]["actions"][:16],
                                   b["trajectories"][name]["actions"][:16], atol=1e-12)


def test_factory_rejects_unsupported_game_response_and_bad_paths():
    tensor = tracking_tensor()
    damaged = tensor.copy()
    damaged[0, 0, 0] += 1e-4
    with pytest.raises(ValueError, match="fixed tracking tensor"):
        TrackingLagSafeLearner(damaged, tracking_response, 4)
    with pytest.raises(ValueError, match="tracking_response"):
        TrackingLagSafeLearner(tensor, lambda ell: np.array([1., 0., 0.]), 4)
    with pytest.raises(ValueError, match="simplex"):
        run_tracking_path([[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="nonnegative integer"):
        tracking_safe_budget(-1)
    # It remains an actual original OneSwitchLearner, with a certified factory.
    learner = OneSwitchLearner(tensor, tracking_response, 4,
        safe_factory=TrackingLagSafeLearner, safe_budget=tracking_safe_budget)
    assert learner.G_T == 1
