"""Reward accounting, split integrity, and an optional real-simulator regression."""

from collections import namedtuple
import json
from pathlib import Path
import random

import numpy as np
import pytest

from lowdim_games.cage import (BLUE_NAMES, RED_NAMES, METRIC_NAMES, CageCalibration,
                              reconstruct_loss_vector, run_cage_episode,
                              save_cage_calibration, load_cage_calibration,
                              scenario_component_bounds, _validate_seeds)


HostReward = namedtuple("HostReward", "confidentiality availability")


def test_native_reward_decomposition_includes_failed_restore_cost():
    scores = {"User1": HostReward(-0.1, 0), "Defender": HostReward(-0.1, 0),
              "Enterprise0": HostReward(-1, 0), "Op_Server0": HostReward(-1, -10)}
    vector, error = reconstruct_loss_vector(scores, -1, -13.2)
    np.testing.assert_allclose(vector, [0.2, 2, 10, 1])
    assert error < 1e-12


def test_decomposition_rejects_dropped_action_cost_and_invalid_actions():
    with pytest.raises(RuntimeError, match="decomposition mismatch"):
        reconstruct_loss_vector({}, -1, 0)
    with pytest.raises(RuntimeError, match="InvalidAction"):
        reconstruct_loss_vector({}, -0.1, -0.1)


def test_train_test_seeds_are_disjoint_and_uncertainty_has_replicates():
    with pytest.raises(ValueError, match="disjoint"):
        _validate_seeds([1, 2], [2, 3])
    with pytest.raises(ValueError, match="at least two"):
        _validate_seeds([1], [2, 3])


def test_calibration_roundtrip_retains_seed_banks_and_checks_hash(tmp_path):
    rng = np.random.default_rng(9)
    train = rng.uniform(size=(3, len(BLUE_NAMES), len(RED_NAMES), 4))
    test = rng.uniform(size=(4, len(BLUE_NAMES), len(RED_NAMES), 4))
    mean = train.mean(axis=0)
    calibration = CageCalibration(mean / 11, mean, test.mean(axis=0), train, test,
                                  np.array([1, 2, 3]), np.array([4, 5, 6, 7]),
                                  np.ones(4), BLUE_NAMES, RED_NAMES, METRIC_NAMES, 11, {})
    path = save_cage_calibration(calibration, tmp_path)
    restored = load_cage_calibration(path)
    np.testing.assert_array_equal(restored.test_episode_losses, test)
    np.testing.assert_array_equal(restored.tensor, calibration.tensor)
    stats = json.loads((tmp_path / "cell_statistics.json").read_text())
    np.testing.assert_allclose(stats["train"]["standard_error"],
                               train.std(axis=0, ddof=1) / np.sqrt(3))
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum"):
        load_cage_calibration(path)


def _official_source():
    path = Path(__file__).parents[1] / ".external/cage-challenge-2"
    if not path.is_dir():
        pytest.skip("Optional official CAGE checkout is absent.")
    for dependency in ("yaml", "paramiko", "prettytable"):
        pytest.importorskip(dependency)
    return path


def test_real_cage_seeded_before_construction_is_reproducible_and_rng_isolated():
    source = _official_source()
    state = random.getstate()
    first = run_cage_episode(source, "monitor", "b_line", 20261003, 30)
    assert random.getstate() == state
    second = run_cage_episode(source, "monitor", "b_line", 20261003, 30)
    np.testing.assert_array_equal(first.step_losses, second.step_losses)
    np.testing.assert_allclose(first.scalar_rewards, -first.step_losses.sum(axis=1), atol=1e-12)
    assert first.mean_loss.sum() > 0
    assert first.max_reward_reconstruction_error < 1e-8
    np.testing.assert_allclose(scenario_component_bounds(source), [.8, 4, 10, 1])


def test_real_cage_all_frozen_blue_policies_have_valid_actions():
    source = _official_source()
    for blue in BLUE_NAMES:
        result = run_cage_episode(source, blue, "sleep", 20261003, 8)
        assert result.max_reward_reconstruction_error < 1e-8
        assert np.all(result.step_losses[:, :3] == 0)
        if blue == "critical_restore":
            assert result.mean_loss[3] == pytest.approx(3 / 8)
