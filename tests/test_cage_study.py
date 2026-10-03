import numpy as np
import pytest

from lowdim_games.cage_study import (
    bonferroni_alpha, evaluate_episode_bank, holm_adjust,
    paired_two_way_bootstrap, simulate_policy_selection,
)


def calibration():
    tensor = np.array([[[0.1, 0.03], [0.7, 0.01]],
                       [[0.4, 0.05], [0.2, 0.1]]])
    return {"tensor": tensor, "weights": [1, 1],
            "blue_names": ("cheap", "strong"), "red_names": ("weak", "targeted"),
            "metric_names": ("damage", "cost")}


def test_scalar_study_actions_are_causal_and_occupancies_reconstruct_losses():
    source = calibration()
    path_a = np.eye(2)[[0, 0, 0, 1, 1, 1, 0, 1]]
    path_b = path_a.copy()
    path_b[2] = [0, 1]
    methods = ["one_switch", "shared_past_hull", "block_safe", "hedge", "last_window", "uniform"]
    first = simulate_policy_selection(source, scenario="shift", opponent_path=path_a, methods=methods)
    second = simulate_policy_selection(source, scenario="shift", opponent_path=path_b, methods=methods)
    assert first["target_geometry_evaluated"] is False
    for method in methods:
        left, right = first["trajectories"][method], second["trajectories"][method]
        # Current-round and future red labels cannot affect the first three commitments.
        np.testing.assert_array_equal(left["actions"][:3], right["actions"][:3])
        assert left["occupancy"].sum() == pytest.approx(1)
        expected = np.einsum("ij,ijd->d", left["occupancy"], source["tensor"])
        np.testing.assert_allclose(expected, left["train_payoffs"].mean(axis=0))
        assert np.all(left["actions"] >= 0)
        np.testing.assert_allclose(left["actions"].sum(axis=1), 1)
    np.testing.assert_array_equal(first["trajectories"]["one_switch"]["actions"],
                                  first["trajectories"]["shared_past_hull"]["actions"])


@pytest.mark.parametrize("scenario", ["fixed", "curriculum", "interactive"])
def test_study_does_not_read_heldout_values(scenario):
    source = calibration()
    changed = calibration()
    changed["test_mean"] = np.full((2, 2, 2), np.nan)
    changed["test_episode_losses"] = "This object must never be consumed by simulation."
    first = simulate_policy_selection(source, horizon=12, seed=4, scenario=scenario,
                                      methods=["hedge", "last_window", "fixed_cheap"])
    second = simulate_policy_selection(changed, horizon=12, seed=4, scenario=scenario,
                                       methods=["hedge", "last_window", "fixed_cheap"])
    for name in first["trajectories"]:
        np.testing.assert_array_equal(first["trajectories"][name]["actions"],
                                      second["trajectories"][name]["actions"])
        np.testing.assert_array_equal(first["trajectories"][name]["opponent_actions"],
                                      second["trajectories"][name]["opponent_actions"])


def test_episode_evaluation_preserves_whole_seed_tables_and_native_sum():
    source = calibration()
    run = simulate_policy_selection(source, horizon=8, scenario="fixed", methods=["uniform", "hedge"])
    bank = np.array([source["tensor"], source["tensor"] + 0.2, source["tensor"] + 0.4])
    evaluations = evaluate_episode_bank(run, bank)
    for method, trajectory in run["trajectories"].items():
        direct = np.einsum("ti,eijd,tj->ed", trajectory["actions"], bank,
                           trajectory["opponent_actions"]) / run["horizon"]
        actual = evaluations["methods"][method]
        np.testing.assert_allclose(actual["by_seed_components"], direct)
        np.testing.assert_allclose(actual["by_seed_native_loss"], direct.sum(axis=1))
    assert evaluations["episode_seeds"] == 3


def test_crossed_bootstrap_matches_whole_seed_and_path_count_resampling():
    bank = np.array([[[[1.0], [2.0]], [[0.2], [0.1]]],
                     [[[1.5], [2.5]], [[0.1], [0.3]]],
                     [[[0.8], [2.2]], [[0.3], [0.4]]]])
    occupations = np.array([[[1.0, 0], [0, 0]], [[0, 1.0], [0, 0]]])
    references = np.array([[[0, 0], [1.0, 0]], [[0, 0], [0, 1.0]]])
    result = paired_two_way_bootstrap(occupations, references, bank, samples=200,
                                      seed=12, alpha=0.1, batch_size=17)
    matrix = np.array([[0.8, 1.9], [1.4, 2.2], [0.5, 1.8]])
    np.testing.assert_allclose(result["paired_seed_path_differences"], matrix)
    assert result["estimate"] == pytest.approx(matrix.mean())
    streams = np.random.SeedSequence(12).spawn(2)
    left_rng, right_rng = map(np.random.default_rng, streams)
    episode_counts = left_rng.multinomial(3, np.full(3, 1 / 3), size=200)
    path_counts = right_rng.multinomial(2, [0.5, 0.5], size=200)
    draws = np.sum((episode_counts @ matrix / 3) * path_counts / 2, axis=1)
    np.testing.assert_allclose([result["lower"], result["upper"]], np.quantile(draws, [.05, .95]))
    assert result["bootstrap_std"] == pytest.approx(draws.std(ddof=1))
    centered_pvalue = (1 + np.sum(np.abs(draws - matrix.mean()) >= abs(matrix.mean()))) / 201
    assert result["approximate_two_sided_pvalue"] == pytest.approx(centered_pvalue)
    rerun = paired_two_way_bootstrap(occupations, references, bank, samples=200,
                                     seed=12, alpha=0.1, batch_size=200)
    assert result["lower"] == pytest.approx(rerun["lower"])
    assert result["upper"] == pytest.approx(rerun["upper"])
    assert result["approximate_two_sided_pvalue"] == rerun["approximate_two_sided_pvalue"]


def test_paired_seed_noise_cancels_instead_of_resampling_cells_independently():
    base = np.array([[[[1.0, 0.2], [4.0, 0.2]], [[2.0, 0.2], [1.0, 0.2]]]])
    bank = np.repeat(base, 4, axis=0)
    shifted = bank + np.array([0, 1, 5, 20])[:, None, None, None]
    occupations = np.array([[[1.0, 0], [0, 0]], [[0, 1.0], [0, 0]]])
    references = np.array([[[0, 0], [1.0, 0]], [[0, 0], [0, 1.0]]])
    first = paired_two_way_bootstrap(occupations, references, bank, samples=500, seed=3)
    second = paired_two_way_bootstrap(occupations, references, shifted, samples=500, seed=3)
    np.testing.assert_allclose(first["paired_seed_path_differences"],
                               second["paired_seed_path_differences"], atol=1e-14)
    assert first["bootstrap_std"] == pytest.approx(second["bootstrap_std"])
    assert second["conditional_episode_t_interval"]["upper"] == pytest.approx(second["estimate"])


def test_declared_multiplicity_and_invalid_sampling_units():
    adjusted = holm_adjust({"first": 0.001, "second": 0.01, "third": 0.04, "fourth": 0.3})
    assert adjusted == {"first": 0.004, "second": 0.03, "third": 0.08, "fourth": 0.3}
    assert bonferroni_alpha(6) == pytest.approx(0.05 / 6)
    with pytest.raises(ValueError, match="p-value"):
        holm_adjust({"invalid": np.nan})
    with pytest.raises(ValueError, match="positive integer"):
        bonferroni_alpha(0)
    with pytest.raises(ValueError, match="At least two"):
        paired_two_way_bootstrap(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1, 1, 1)))
    with pytest.raises(ValueError, match="sum to one"):
        paired_two_way_bootstrap(np.zeros((1, 1)), np.ones((1, 1)), np.ones((2, 1, 1, 1)))
