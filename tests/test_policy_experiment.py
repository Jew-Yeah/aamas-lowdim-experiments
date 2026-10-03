import json

import numpy as np
import pytest

from lowdim_games.game import FiniteGame
from lowdim_games.policy_experiment import (
    AdaptivePolicyOpponent, HedgeLearner, PastWindowResponse,
    curriculum_path, mean_ci95, run_policy_comparison,
)


def calibration():
    tensor = np.array([[[0.2, 0.05], [0.8, 0.05]],
                       [[0.5, 0.1], [0.1, 0.3]]])
    bank = np.array([tensor - 0.01, tensor, tensor + 0.01])
    return {"tensor": tensor, "train_mean": tensor.copy(), "test_mean": bank.mean(axis=0),
            "test_episode_losses": bank, "weights": np.ones(2),
            "blue_names": ("cheap", "strong"), "red_names": ("weak", "targeted"),
            "metric_names": ("damage", "defense_cost"), "normalization_scale": 1.0}


def test_hedge_uses_current_losses_only_after_commit_and_has_declared_bound():
    game = FiniteGame(calibration()["tensor"], [1, 1])
    learner = HedgeLearner(game, 40)
    first = learner.choose()
    assert np.array_equal(first, [0.5, 0.5])
    assert np.array_equal(learner.choose(), first)
    with pytest.raises(RuntimeError):
        HedgeLearner(game, 40).observe([1, 0])
    learner.observe([1, 0])
    expected = np.exp(-learner.learning_rate * game.scalar_costs[:, 0])
    expected /= expected.sum()
    assert np.allclose(learner.choose(), expected)
    total_loss = float(first @ game.scalar_costs[:, 0])
    for index in range(1, 40):
        p = learner.choose()
        ell = np.eye(2)[index % 2]
        total_loss += float(p @ game.scalar_costs @ ell)
        learner.observe(ell)
    fixed_losses = 20 * game.scalar_costs.sum(axis=1)
    assert total_loss - fixed_losses.min() <= learner.regret_bound + 1e-12


def test_window_response_initialization_and_past_only():
    game = FiniteGame(calibration()["tensor"], [1, 1])
    learner = PastWindowResponse(game, 5, [1, 0], window=1)
    assert np.array_equal(learner.choose(), game.response([1, 0]))
    learner.observe([0, 1])
    assert np.array_equal(learner.choose(), game.response([0, 1]))
    learner.observe([1, 0])
    assert np.array_equal(learner.choose(), game.response([1, 0]))


def test_attacker_feedback_is_only_past_defender_and_commit_is_stable():
    tensor = np.array([[[0.0], [1.0]], [[1.0], [0.0]]])
    left = AdaptivePolicyOpponent(tensor, [1], 24, 7, exploration=0)
    right = AdaptivePolicyOpponent(tensor, [1], 24, 7, exploration=0)
    for index in range(18):
        ell_left, ell_right = left.choose(), right.choose()
        assert np.array_equal(left.choose(), ell_left)
        # Current commit cannot depend on the defender supplied after choose.
        assert np.array_equal(ell_left, ell_right)
        left.observe([1, 0])
        right.observe([0, 1])
    assert left.log_weights[1] > left.log_weights[0]
    assert right.log_weights[0] > right.log_weights[1]
    # After the historical observations, the learning phase reacts differently.
    assert np.argmax(left.choose()) == 1
    assert np.argmax(right.choose()) == 0


@pytest.mark.parametrize("scenario", ["curriculum", "interactive"])
def test_heldout_table_cannot_change_learner_or_attacker_actions(tmp_path, scenario):
    first = calibration()
    second = calibration()
    second["test_episode_losses"] = second["test_episode_losses"] * 100 + 20
    second["test_mean"] = second["test_episode_losses"].mean(axis=0)
    methods = ["one_switch", "shared_past_hull", "block_safe", "uniform",
               "historical_best", "last_window", "hedge", "fixed_cheap", "fixed_strong"]
    run_a = run_policy_comparison(first, scenario=scenario, horizon=12, seed=6,
                                  output_dir=tmp_path / "a", name="test", methods=methods)
    run_b = run_policy_comparison(second, scenario=scenario, horizon=12, seed=6,
                                  output_dir=tmp_path / "b", name="test", methods=methods)
    for method in methods:
        a = np.load(tmp_path / "a" / f"test__{method}.npz")
        b = np.load(tmp_path / "b" / f"test__{method}.npz")
        assert np.array_equal(a["actions"], b["actions"])
        assert np.array_equal(a["opponent_actions"], b["opponent_actions"])
        assert np.array_equal(a["train_payoffs"], b["train_payoffs"])
        assert not np.array_equal(a["test_payoffs"], b["test_payoffs"])
    assert run_a["same_opponent_path_across_methods"] == (scenario == "curriculum")
    assert run_a["test_mean_sha256"] != run_b["test_mean_sha256"]
    assert run_a["max_target_projection_gap"] <= 1e-9


def test_seed_intervals_use_whole_table_clusters_and_paired_differences(tmp_path):
    source = calibration()
    result = run_policy_comparison(source, horizon=12, seed=5, output_dir=tmp_path,
                                   name="test", methods=["one_switch", "uniform", "hedge"])
    weights = np.array([0.5, 0.5])
    policy_samples = {}
    for method in result["methods"]:
        arrays = np.load(tmp_path / f"test__{method}.npz")
        assert np.allclose(arrays["actions"].sum(axis=1), 1)
        occupancy = arrays["actions"].T @ arrays["opponent_actions"] / 12
        replicates = np.einsum("ij,eijd->ed", occupancy, source["test_episode_losses"])
        policy_samples[method] = replicates
        summary = next(item for item in result["summaries"] if item["method"] == method)
        assert summary["heldout_metric_ci95"]["seeds"] == 3
        assert np.allclose(summary["heldout_metric_ci95"]["mean"], replicates.mean(axis=0))
        assert np.allclose(list(summary["mean_test_metrics"].values()), replicates.mean(axis=0))
    one_switch = result["summaries"][0]
    expected = mean_ci95((policy_samples["one_switch"] - policy_samples["hedge"]) @ weights)
    actual = one_switch["heldout_paired_weighted_loss_difference_ci95"]["hedge"]
    assert np.allclose(actual["mean"], expected["mean"])
    assert np.allclose(actual["lower"], expected["lower"])
    saved = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))
    assert saved["heldout_feedback_to_learners_or_attacker"] is False
    assert len(saved["summaries"][0]["phases"]) == 3


def test_exogenous_curriculum_is_common_but_interactive_histories_can_differ(tmp_path):
    source = {"tensor": np.array([[[0.0], [1.0]], [[1.0], [0.0]]]),
              "weights": [1], "blue_names": ["left", "right"], "red_names": ["left", "right"],
              "metric_names": ["damage"]}
    source["train_mean"] = source["tensor"]
    source["test_mean"] = source["tensor"]
    methods = ["fixed_left", "fixed_right"]
    common = run_policy_comparison(source, scenario="curriculum", horizon=48, seed=9,
                                    output_dir=tmp_path, name="common", methods=methods)
    reactive = run_policy_comparison(source, scenario="interactive", horizon=48, seed=9,
                                      output_dir=tmp_path, name="reactive", methods=methods)
    assert len({s["opponent_path_sha256"] for s in common["summaries"]}) == 1
    assert len({s["opponent_path_sha256"] for s in reactive["summaries"]}) == 2
    assert reactive["same_opponent_path_across_methods"] is False
    for summary in reactive["summaries"]:
        assert summary["realized_affine_dimension"] <= 1
    path = curriculum_path(source["tensor"], [1], 48, 9)
    saved = np.load(tmp_path / "common__fixed_left.npz")["opponent_actions"]
    assert np.array_equal(path, saved)


def test_inconsistent_heldout_bank_rejected_before_running(tmp_path):
    source = calibration()
    source["test_mean"] = source["test_mean"] + 1
    with pytest.raises(ValueError, match="does not average"):
        run_policy_comparison(source, output_dir=tmp_path)
