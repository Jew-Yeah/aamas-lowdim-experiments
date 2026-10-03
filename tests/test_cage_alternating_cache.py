"""Memoization must preserve actual deterministic trajectories and disclose reuse."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts import cache_cage_alternating_paths as cache
from scripts import run_cage_adaptation_study as runner


def test_real_factories_on_training_shaped_game_ignore_alternating_path_seed(tmp_path):
    tensor = np.random.default_rng(12).uniform(0, .05, size=(6, 3, 4))
    selected = {"selected": {"scalar": {"kind": "scalar", "window": 4, "rho": .5},
                             "window": {"kind": "window", "window": 4},
                             "hedge": {"kind": "hedge", "eta_multiplier": .5}}}
    config = {"scenario": "alternating500", "horizon": 8,
              "configurations": runner._selected_configs(selected, 2),
              "initial_red_index": 1, "other_red_index": 2, "attack_block_length": 2}
    outputs = []
    for seed in (43000000, 43000001):
        path = tmp_path / f"seed_{seed}.npz"
        runner._simulate_path((tensor, np.ones(4), config, seed, str(path), "training_only_probe"))
        outputs.append(cache._arrays(path))
    cache._assert_equal_arrays(outputs[0], outputs[1], independent=True)


def test_memoized_slots_change_only_seed_and_preserve_existing_files(tmp_path, monkeypatch):
    tensor = np.random.default_rng(12).uniform(0, .05, size=(6, 3, 4))
    config = {"name": "restarts_alternating500", "scenario": "alternating500", "horizon": 8,
              "configurations": [{"name": "original_one_switch", "kind": "original"}],
              "initial_red_index": 1, "other_red_index": 2, "attack_block_length": 2,
              "path_seeds": [43000000, 43000001, 43000002],
              "protocol_sha256": "training_fixture", "selection_sha256": "training_fixture"}
    signature = runner._signature(config)
    original = SimpleNamespace(tensor=tensor, weights=np.ones(4))
    monkeypatch.setattr(cache, "frozen_configuration", lambda *args: (original, config, signature))
    source = cache.prepare("unused", tmp_path, "unused")
    result = cache.materialize("unused", tmp_path, "unused")
    assert result["helper_actual_computations"] == 1
    assert result["materialized_seed_slots"] == [43000001, 43000002]
    assert result["added_independent_attack_path_variation"] is False
    for seed in config["path_seeds"]:
        clone = source.parent / f"seed_{seed}.npz"
        cache._assert_equal_arrays(cache._arrays(source), cache._arrays(clone))
        assert int(cache._arrays(clone)["seed"]) == seed
    hashes = result["checkpoint_sha256"]
    replay = cache.materialize("unused", tmp_path, "unused")
    assert replay["checkpoint_sha256"] == hashes


def test_cache_refuses_an_active_main_stage(tmp_path):
    runner._atomic_json(tmp_path / "meta.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="already executing"):
        cache._require_inactive(tmp_path)
