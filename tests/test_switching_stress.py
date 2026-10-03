"""Independent oracle and lifecycle checks for the resource-balance diagnostic."""

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from lowdim_games.learners import SafeBlockLearner, paper_safe_budget
from lowdim_games.switching_stress import (
    MODE_FAST, MODE_SAFE, ResourceBalanceFastLearner,
    resource_response, run_default_budget_stress,
)


def _vertex(label, dimension=5):
    point = np.zeros(dimension)
    if label:
        point[label - 1] = 1.0
    return point


def _independent_projection(query, vertices):
    """Generic dense QP, independent of the production closed-form oracle."""
    vertices = np.asarray(vertices)
    result = minimize(
        lambda w: np.sum((w @ vertices - query) ** 2),
        np.full(len(vertices), 1.0 / len(vertices)),
        jac=lambda w: 2.0 * vertices @ (w @ vertices - query),
        bounds=[(0.0, 1.0)] * len(vertices),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones(len(vertices))},
        method="SLSQP", options={"ftol": 1e-14, "maxiter": 1000},
    )
    assert result.success
    return result.x @ vertices


@pytest.mark.parametrize("labels", [[0, 0, 1, 2, 1, 3, 0], [1, 2, 0, 3, 2, 0]])
def test_closed_form_fast_matches_independent_dense_lp_and_qp(labels):
    learner = ResourceBalanceFastLearner(len(labels))
    history = []
    cumulative = 0.0
    expected_direction = 0.0
    for t, label in enumerate(labels, 1):
        p = learner.choose()
        before = learner.direction
        query = _vertex(label)
        if history:
            projected = _independent_projection(query, history)
            demands = np.asarray(history).sum(axis=1)
            matrix = before * np.stack((-demands, 1.0 - demands))
            # Independent general minimax LP over the original two actions.
            primal = linprog(
                [0.0, 0.0, 1.0], A_ub=np.c_[matrix.T, -np.ones(len(history))],
                b_ub=np.zeros(len(history)), A_eq=[[1.0, 1.0, 0.0]], b_eq=[1.0],
                bounds=[(0.0, None), (0.0, None), (None, None)], method="highs",
            )
            assert primal.success
            actual_upper = np.max(np.array([1.0 - p, p]) @ matrix)
            assert actual_upper == pytest.approx(primal.fun, abs=1e-10)
            a = p - projected.sum()  # Full response payoff is identically zero.
            expected_direction = before + a / (2.0 * np.sqrt(t))
            expected_direction /= max(1.0, abs(expected_direction))
            residual = abs(query.sum() - projected.sum())
            cumulative += residual
        record = learner.observe(label)
        if history:
            assert record["projected_demand"] == pytest.approx(projected.sum(), abs=1e-8)
            assert record["h_t"] == pytest.approx(np.linalg.norm(query - projected), abs=1e-8)
            assert record["residual"] == pytest.approx(residual, abs=1e-8)
            assert learner.direction == pytest.approx(expected_direction, abs=1e-8)
        assert record["cumulative_residual"] == pytest.approx(cumulative, abs=1e-8)
        assert record["projection_gap"] == 0
        assert record["saddle_gap"] <= 1e-12
        assert record["target_witness"] == 0
        history.append(query)


def test_fast_choose_is_causal_and_requires_prior_choose():
    first, second = ResourceBalanceFastLearner(8), ResourceBalanceFastLearner(8)
    with pytest.raises(RuntimeError, match="choose"):
        first.observe(0)
    for label in [0, 0, 1]:
        assert first.choose() == second.choose()
        first.observe(label)
        second.observe(label)
    assert first.choose() == second.choose()
    # Distinct next observations cannot change already selected actions.
    assert first.choose() == 0
    first.observe(2)
    second.observe(0)
    assert first.cumulative_residual == 2
    assert second.cumulative_residual == 1


def test_compressed_safe_response_balances_every_unseen_mixture():
    tensor = np.array([[[0.0], [-1.0]], [[1.0], [0.0]]])
    for r in [0.0, 0.1, 0.5, 0.77, 1.0]:
        ell = np.array([1.0 - r, r])
        p_star = resource_response(ell)
        assert np.einsum("a,ajd,j->d", p_star, tensor, ell)[0] == pytest.approx(0, abs=1e-16)
    with pytest.raises(ValueError):
        resource_response(np.array([1.0, 1.0]))


def test_default_master_preserves_crossing_round_and_uses_a_fresh_exact_safe_tail():
    run = run_default_budget_stress(T=4096, quiet_rounds=32)
    master = run["trajectories"]["one_switch"]
    fast = run["trajectories"]["shared_past_hull"]
    tau = run["summary"]["one_switch"]["switch_round"]
    assert run["metadata"]["switch_threshold"] == paper_safe_budget(4096, 1) == 3072
    assert tau == 3105
    assert master["cumulative_residual"][tau - 2] == 3072
    assert master["cumulative_residual"][tau - 1] == 3073
    assert master["mode"][tau - 1] == MODE_FAST
    assert master["mode"][tau] == MODE_SAFE
    assert master["switch"].sum() == 1
    np.testing.assert_array_equal(master["capacity"][:tau], fast["capacity"][:tau])
    assert master["safe_local_t"][tau] == 1
    assert master["direction"][tau] == 0
    assert master["capacity"][tau] == 0.5
    np.testing.assert_array_equal(master["cumulative_residual"][tau:], 3073)
    # Recompute the entire tail with the original class, without any production
    # helper or rescaled step/budget. This checks all restarts and remainder rounds.
    tensor = np.array([[[0.0], [-1.0]], [[1.0], [0.0]]])
    independent = SafeBlockLearner(tensor, lambda z: np.array([z[0], z[1]]), 4096 - tau)
    actions, blocks = [], []
    for _ in range(4096 - tau):
        actions.append(independent.choose()[1])
        blocks.append(independent.observe(np.array([0.0, 1.0]))["block_index"])
    np.testing.assert_array_equal(master["capacity"][tau:], actions)
    np.testing.assert_array_equal(master["safe_block_index"][tau:], blocks)
    assert run["summary"]["one_switch"]["safe_tail_budget"] == independent.budget


def test_primary_default_really_crosses_and_reports_stronger_controls():
    run = run_default_budget_stress()
    summary = run["summary"]
    assert summary["one_switch"]["switch_round"] == 8817
    assert summary["one_switch"]["safe_rounds"] == 7567
    assert summary["shared_past_hull"]["final_distance"] == pytest.approx(0.99212646484375)
    assert summary["one_switch"]["final_distance"] == pytest.approx(0.6022961476839039)
    assert summary["block_safe"]["final_distance"] == pytest.approx(0.1243004061577185)
    assert summary["lag_safe"]["final_distance"] == 0.5 / 16384
    assert (summary["lag_safe"]["final_distance"]
            < summary["block_safe"]["final_distance"]
            < summary["one_switch"]["final_distance"]
            < summary["shared_past_hull"]["final_distance"])
    assert summary["one_switch"]["prefix_plus_safe_bound"] < summary["shared_past_hull"]["final_distance"]
    assert run["metadata"]["realized_affine_dimension"] == 16256
    for logs in run["trajectories"].values():
        np.testing.assert_array_equal(logs["distance"], np.abs(np.cumsum(logs["payoff"]) / logs["t"]))
        np.testing.assert_array_equal(logs["full_target_projection"], 0)


def test_short_horizon_no_crossing_and_input_validation():
    run = run_default_budget_stress(T=24, quiet_rounds=3, window=2)
    assert run["summary"]["one_switch"]["switch_round"] is None
    assert run["summary"]["one_switch"]["safe_rounds"] == 0
    np.testing.assert_array_equal(run["trajectories"]["one_switch"]["capacity"],
                                  run["trajectories"]["shared_past_hull"]["capacity"])
    for kwargs in ({"T": 0}, {"T": True}, {"quiet_rounds": 1},
                   {"T": 10, "quiet_rounds": 10}, {"window": 0}):
        with pytest.raises(ValueError):
            run_default_budget_stress(**kwargs)
