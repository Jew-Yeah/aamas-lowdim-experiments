"""Independent numerical checks of algorithm equations and causal protocol."""

import numpy as np
import pytest
from scipy.linalg import helmert

from lowdim_games.learners import (
    FastHullLearner,
    OneSwitchLearner,
    SafeBlockLearner,
    paper_safe_budget,
    project_simplex,
    solve_saddle_lp,
)


def weighted_response(tensor, weights=None):
    weights = np.ones(tensor.shape[2]) if weights is None else weights

    def response(ell):
        losses = np.einsum("ajd,j,d->a", tensor, ell, weights)
        return np.eye(len(losses))[np.argmin(losses)]

    return response


def normalized_tensor(seed=13, k=3, m=4, d=2):
    a = np.random.default_rng(seed).normal(size=(k, m, d))
    return a / np.max(np.linalg.norm(a, axis=2))


def test_saddle_matching_pennies_and_feasible_gap():
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    result = solve_saddle_lp(matrix)
    assert result.success
    np.testing.assert_allclose(result.p, [0.5, 0.5], atol=1e-9)
    np.testing.assert_allclose(result.opponent_weights, [0.5, 0.5], atol=1e-9)
    assert abs(result.value) < 1e-9
    assert result.gap < 1e-9
    # Check best-response inequalities without trusting solver status.
    assert np.max(result.p @ matrix) - np.min(matrix @ result.opponent_weights) < 1e-9


def test_simplex_projection_kkt_conditions():
    vector = np.array([0.2, 1.7, -0.8, 0.4])
    p = project_simplex(vector)
    assert p.sum() == pytest.approx(1)
    assert np.min(p) >= 0
    active = p > 1e-10
    multiplier = (vector - p)[active][0]
    np.testing.assert_allclose((vector - p)[active], multiplier)
    assert np.all(vector[~active] <= multiplier + 1e-10)


def test_fast_is_causal_and_response_witness_uses_past_hull():
    tensor = normalized_tensor()
    response = weighted_response(tensor)
    first, second = [FastHullLearner(tensor, response, 5) for _ in range(2)]
    prefix = np.array([[1, 0, 0, 0], [0.2, 0.8, 0, 0], [0.1, 0.2, 0.7, 0]])
    for ell in prefix:
        np.testing.assert_array_equal(first.choose(), second.choose())
        record = first.observe(ell)
        second.observe(ell)
        if record["t"] > 1:
            assert record["saddle_gap"] <= 1e-8
            # Every response input and projection is a feasible convex
            # combination of observations revealed before this round.
            old = prefix[:record["t"] - 1]
            assert np.max(record["response_point"][np.max(old, axis=0) == 0]) == 0
            assert np.max(record["projected_ell"][np.max(old, axis=0) == 0]) == 0
    # Distinct unrevealed next actions cannot influence the next chosen mixture.
    np.testing.assert_array_equal(first.choose(), second.choose())
    first.observe(np.array([0, 0, 0, 1]))
    second.observe(np.array([1, 0, 0, 0]))


def test_fast_update_and_residual_decomposition():
    tensor = normalized_tensor(seed=17, k=2, m=3)
    learner = FastHullLearner(tensor, weighted_response(tensor), 3)
    path = np.eye(3)
    accumulated = 0.0
    for ell in path:
        learner.choose()
        record = learner.observe(ell)
        if record["t"] == 1:
            np.testing.assert_array_equal(learner.direction, np.zeros(2))
            continue
        projected_payoff = learner.payoff(record["p"], record["projected_ell"])
        vector = projected_payoff - record["target_witness"]
        proposal = record["direction"] + vector / (2 * np.sqrt(record["t"]))
        expected_direction = proposal / max(1.0, np.linalg.norm(proposal))
        np.testing.assert_allclose(learner.direction, expected_direction)
        assert np.dot(record["direction"], vector) <= record["t"]**-0.5 + 1e-8
        residual = np.linalg.norm(record["payoff"] - projected_payoff)
        accumulated += residual
        assert record["residual"] == pytest.approx(residual)
        assert record["cumulative_residual"] == pytest.approx(accumulated)


def test_safe_step_matches_ogd_in_explicit_john_simplex_coordinates():
    tensor = normalized_tensor(seed=29, k=3, m=3)
    learner = SafeBlockLearner(tensor, weighted_response(tensor), 81)
    # First block direction is zero; ending it establishes a nonzero outer
    # direction. The next inner step is checked in separate affine coordinates.
    for _ in range(learner.n_h):
        learner.choose()
        learner.observe(np.array([1.0, 0, 0]))
    assert np.linalg.norm(learner.direction) > 0
    p = learner.choose()
    direction = learner.direction.copy()
    ell = np.array([0, 1.0, 0])
    gradient_p = np.einsum("ajd,j,d->a", tensor, ell, direction)
    basis = helmert(learner.K, full=False).T
    center = np.full(learner.K, 1.0 / learner.K)
    scale = np.sqrt(learner.K * learner.k)
    x = scale * basis.T @ (p - center)
    gradient_x = basis.T @ gradient_p / scale
    trial_x = x - learner.k / np.sqrt(learner.n_h) * gradient_x
    # Projection in orthonormal affine coordinates is equivalent to simplex
    # projection after translating/scaling back.
    p_expected = project_simplex(center + basis @ trial_x / scale)
    learner.observe(ell)
    np.testing.assert_allclose(learner.choose(), p_expected, atol=1e-12)


def test_safe_target_witness_and_nonasymptotic_certificate():
    tensor = normalized_tensor(seed=41, k=4, m=5, d=3)
    learner = SafeBlockLearner(tensor, weighted_response(tensor), 150)
    path = np.random.default_rng(42).dirichlet(np.ones(5), size=150)
    target_sum = np.zeros(3)
    actual_sum = np.zeros(3)
    for ell in path:
        learner.choose()
        record = learner.observe(ell)
        actual_sum += record["payoff"]
        if record["target_witness"] is not None:
            weight = learner.n_h if record["block_index"] >= 0 else 1
            target_sum += weight * record["target_witness"]
    # The constructed witness is in S(Q_T): each block average and every
    # remainder action lies in Q_T. Its distance bounds the actual target error.
    witness_cost = np.linalg.norm(actual_sum - target_sum)
    middle_bound = (
        learner.m_h * learner.k * np.sqrt(learner.n_h)
        + 2 * learner.n_h * np.sqrt(learner.m_h) + 2 * learner.r_h
    )
    assert witness_cost <= min(2 * learner.horizon, middle_bound) + 1e-8
    assert witness_cost <= learner.budget + 1e-8


def test_master_default_budget_and_fast_equivalence_before_switch():
    tensor = normalized_tensor(seed=3, k=2, m=3)
    response = weighted_response(tensor)
    master = OneSwitchLearner(tensor, response, 12)
    fast = FastHullLearner(tensor, response, 12)
    assert master.G_T == pytest.approx(paper_safe_budget(12, 1))
    for ell in np.random.default_rng(5).dirichlet(np.ones(3), size=12):
        np.testing.assert_allclose(master.choose(), fast.choose())
        a, b = master.observe(ell), fast.observe(ell)
        np.testing.assert_allclose(a["payoff"], b["payoff"])
        assert a["cumulative_residual"] == pytest.approx(b["cumulative_residual"])
    assert master.switch_round is None  # Here G_T>2T, so no crossing is possible.


def test_master_switches_after_crossing_round_with_a_valid_zero_error_base():
    # All learner rows are identical. Thus for every causal learner and path,
    # average payoff equals u(p*, average ell) in S(Q); B_0(h)=0 is valid.
    tensor = np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]])
    response = lambda ell: np.array([1.0, 0.0])
    created_horizons = []

    def zero_error_factory(a, resp, horizon):
        created_horizons.append(horizon)
        return SafeBlockLearner(a, resp, horizon)

    master = OneSwitchLearner(tensor, response, 4,
                             safe_factory=zero_error_factory, safe_budget=lambda h: 0.0)
    records = []
    for ell in np.array([[1, 0], [0, 1], [1, 0], [0, 1]]):
        master.choose()
        records.append(master.observe(ell))
    assert [r["mode"] for r in records] == ["fast", "fast", "safe", "safe"]
    assert [r["switch"] for r in records] == [False, True, False, False]
    assert records[1]["residual"] == pytest.approx(2.0)
    assert master.switch_round == 2
    assert created_horizons == [2]
    assert [r.get("safe_local_t") for r in records[2:]] == [1, 2]
    assert all(r["cumulative_residual"] == 2 for r in records[1:])


def test_causal_protocol_and_input_rejection():
    tensor = normalized_tensor()
    learner = FastHullLearner(tensor, weighted_response(tensor), 1)
    with pytest.raises(RuntimeError, match="choose"):
        learner.observe(np.full(4, 0.25))
    np.testing.assert_array_equal(learner.choose(), learner.choose())
    learner.observe(np.full(4, 0.25))
    with pytest.raises(RuntimeError, match="horizon"):
        learner.choose()
    with pytest.raises(ValueError, match="both"):
        OneSwitchLearner(tensor, weighted_response(tensor), 3, safe_budget=lambda h: 0)
