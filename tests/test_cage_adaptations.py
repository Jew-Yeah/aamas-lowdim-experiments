from types import SimpleNamespace

import numpy as np
import pytest

from lowdim_games.cage_adaptations import (
    BlockRestartLearner, ScalarAwareFastHullLearner, ScalarAwareOneSwitchLearner,
)
from lowdim_games.game import FiniteGame
from lowdim_games.learners import paper_safe_budget, solve_saddle_lp


def game():
    return FiniteGame(np.array([[[.1, .05], [.8, .01]],
                               [[.5, .1], [.2, .1]],
                               [[.3, .1], [.3, .2]]]), [1, 1])


def test_zero_slack_selects_minimum_scalar_loss_on_a_saddle_face():
    instance = game()
    learner = ScalarAwareFastHullLearner(instance.tensor, instance.response, 8,
                                         weights=instance.weights, window=4, rho=0)
    np.testing.assert_allclose(learner.choose(), np.ones(3) / 3)
    learner.observe([1, 0])
    # lambda is still zero: every primal mixture is saddle feasible.
    action = learner.choose()
    np.testing.assert_allclose(action, [1, 0, 0])
    assert learner._saddle.gap == 0
    original_dual = solve_saddle_lp(np.zeros((3, 1))).opponent_weights
    np.testing.assert_array_equal(learner._saddle.opponent_weights, original_dual)
    record = learner.observe([0, 1])
    assert record["scalar_forecast_selected_loss"] <= record["scalar_forecast_original_loss"]
    assert record["scalar_oracle_epsilon"] == 0
    assert record["beta_t"] == 0


def test_actual_saddle_gap_and_preserved_dual_are_checked_independently():
    instance = game()
    learner = ScalarAwareFastHullLearner(instance.tensor, instance.response, 24,
                                         weights=instance.weights, window=3, rho=.25)
    path = np.eye(2)[np.arange(24) % 2]
    for ell in path:
        action = learner.choose()
        if learner.round:
            vertices = np.asarray(learner.history)
            matrix = np.einsum("ajd,nj,d->an", instance.tensor, vertices, learner.direction)
            original = solve_saddle_lp(matrix)
            np.testing.assert_allclose(learner._saddle.opponent_weights, original.opponent_weights)
            actual_gap = max(0, np.max(action @ matrix) - np.min(matrix @ original.opponent_weights))
            assert learner._saddle.gap == pytest.approx(actual_gap, abs=1e-14)
            assert actual_gap <= .25 / np.sqrt(learner.round + 1) + 2e-8
            np.testing.assert_allclose(learner._response_point, original.opponent_weights @ vertices)
        record = learner.observe(ell)
        assert record["scalar_oracle_contract_excess"] <= 2e-8 if learner.round > 1 else True
    assert len(learner.history) == 2
    assert len(learner.chronological_history) == 24


@pytest.mark.parametrize("failure", ["solver", "gap", "simplex"])
def test_bad_scalar_candidates_fall_back_to_original_saddle_pair(monkeypatch, failure):
    instance = game()
    learner = ScalarAwareFastHullLearner(instance.tensor, instance.response, 6,
                                         weights=instance.weights, rho=0)
    learner.choose()
    learner.observe([1, 0])
    learner.direction = np.array([1., 0])
    if failure == "solver":
        result = SimpleNamespace(success=False)
    elif failure == "gap":
        result = SimpleNamespace(success=True, x=np.array([0., 1., 0.]))
    else:
        result = SimpleNamespace(success=True, x=np.array([-.1, 1.1, 0.]))
    monkeypatch.setattr(learner, "_solve_scalar_lp", lambda *args: result)
    matrix = np.einsum("ajd,nj,d->an", instance.tensor, np.asarray(learner.history), learner.direction)
    expected = solve_saddle_lp(matrix)
    np.testing.assert_allclose(learner.choose(), expected.p)
    record = learner.observe([0, 1])
    assert record["scalar_oracle_fallback"] is True
    assert record["scalar_oracle_used"] is False
    assert record["scalar_oracle_actual_gap"] == pytest.approx(expected.gap)


def test_scalar_forecast_is_causal_and_master_keeps_original_budget():
    instance = game()
    first = ScalarAwareOneSwitchLearner(instance.tensor, instance.response, 8,
                                        weights=instance.weights, window=2, rho=.5)
    second = ScalarAwareOneSwitchLearner(instance.tensor, instance.response, 8,
                                         weights=instance.weights, window=2, rho=.5)
    assert first.G_T == paper_safe_budget(8, 2)
    paths = [np.eye(2)[[0, 1, 0, 0]], np.eye(2)[[0, 1, 1, 1]]]
    actions = [[], []]
    for learner, path, saved in zip((first, second), paths, actions):
        for ell in path:
            saved.append(learner.choose())
            learner.observe(ell)
    # Round three commitments share the same past, although ell_3 differs.
    np.testing.assert_array_equal(actions[0][:3], actions[1][:3])
    np.testing.assert_allclose(first.fast.records[2]["scalar_forecast"], [.5, .5])


def test_block_restart_resets_uniform_direction_residual_and_short_tail_budget():
    tensor = np.array([[[0.], [1.]], [[0.], [1.]]])
    instance = FiniteGame(tensor, [1])
    learner = BlockRestartLearner(tensor, instance.response, 8, block_length=3)
    records = []
    for ell in np.eye(2)[[0, 1, 0, 0, 1, 0, 1, 0]]:
        action = learner.choose()
        records.append(learner.observe(ell))
        if records[-1]["local_t"] == 1:
            np.testing.assert_allclose(action, [.5, .5])
            np.testing.assert_array_equal(records[-1]["direction"], [0])
            assert records[-1]["cumulative_residual"] == 0
    assert [records[i]["restart"] for i in (0, 3, 6)] == [False, True, True]
    assert records[1]["cumulative_residual"] == pytest.approx(1)
    assert records[4]["cumulative_residual"] == pytest.approx(1)
    assert records[-1]["block_horizon"] == 2
    assert records[-1]["local_switch_threshold"] == paper_safe_budget(2, 1)
    assert learner.switch_rounds == []
    assert learner.active_master is None
    with pytest.raises(RuntimeError, match="exhausted"):
        learner.choose()


@pytest.mark.parametrize("scalar_aware", [False, True])
def test_retained_hull_removes_false_novelty_and_retains_forecast_chronology(scalar_aware):
    tensor = np.array([[[0.], [1.]], [[0.], [1.]]])
    instance = FiniteGame(tensor, [1])
    retained = BlockRestartLearner(tensor, instance.response, 6, block_length=3,
                                   retain_history=True, scalar_aware=scalar_aware,
                                   weights=[1], window=8)
    fresh = BlockRestartLearner(tensor, instance.response, 6, block_length=3,
                                retain_history=False, scalar_aware=scalar_aware,
                                weights=[1], window=8)
    saved = [[], []]
    for learner, records in zip((retained, fresh), saved):
        for ell in np.eye(2)[[0, 1, 0, 0, 1, 0]]:
            learner.choose()
            records.append(learner.observe(ell))
    assert saved[0][4]["residual"] == 0
    assert saved[1][4]["residual"] == pytest.approx(1)
    assert len(retained.global_history) == 6
    if scalar_aware:
        np.testing.assert_allclose(saved[0][4]["scalar_forecast"], [.75, .25])
        np.testing.assert_allclose(saved[1][4]["scalar_forecast"], [1, 0])
        assert saved[0][4]["forecast_history_size"] == 5
        assert saved[1][4]["forecast_history_size"] == 2


def test_invalid_adaptation_parameters_rejected_and_observe_requires_commit():
    instance = game()
    for options in ({"rho": 1.1}, {"rho": -.1}, {"window": 0}, {"window": True}):
        with pytest.raises(ValueError):
            ScalarAwareFastHullLearner(instance.tensor, instance.response, 4, **options)
    with pytest.raises(ValueError):
        BlockRestartLearner(instance.tensor, instance.response, 4, block_length=0)
    with pytest.raises(RuntimeError, match="choose"):
        BlockRestartLearner(instance.tensor, instance.response, 4).observe([1, 0])


def test_retained_restart_collects_safe_tail_observations_and_global_switches(monkeypatch):
    import lowdim_games.cage_adaptations as module
    from lowdim_games.learners import SafeBlockLearner

    # Identical rows imply zero approachability error for every strategy: the
    # average payoff is the response payoff at average ell. Thus B_0(h)=0 is
    # a legitimate abstract safe guarantee here, not a guessed experiment budget.
    tensor = np.array([[[-1.], [0.], [1.]], [[-1.], [0.], [1.]]])
    instance = FiniteGame(tensor, [1])
    original_master = module.OneSwitchLearner
    tail_horizons = []

    def zero_error_factory(a, response, horizon):
        tail_horizons.append(horizon)
        return SafeBlockLearner(a, response, horizon)

    def valid_zero_budget_master(a, response, horizon, **kwargs):
        return original_master(a, response, horizon, **kwargs,
                               safe_factory=zero_error_factory, safe_budget=lambda h: 0.)

    monkeypatch.setattr(module, "OneSwitchLearner", valid_zero_budget_master)
    learner = BlockRestartLearner(tensor, instance.response, 8, block_length=4,
                                  retain_history=True)
    records = []
    for ell in np.eye(3)[[0, 1, 2, 0, 0, 2, 1, 0]]:
        learner.choose()
        records.append(learner.observe(ell))
    assert records[1]["global_switch"] is True
    assert records[1]["global_switch_round"] == 2
    assert records[2]["mode"] == "safe"  # e_2 first appears in the safe tail.
    assert records[4]["restart"] is True
    assert records[4]["cumulative_residual"] == 0
    assert records[5]["residual"] == 0  # Retained hull includes that e_2.
    assert learner.switch_rounds == [2]
    assert learner.switch_round == 2
    assert tail_horizons == [2]
