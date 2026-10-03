"""The collector must resume fixed observations and keep independent seeds."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

# Scripts are deliberately outside the installable package.  Resolve this
# repository explicitly so both pytest entry points work and spawned workers
# can import the same collector module.
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.collect_cage_bank import (
    _atomic_json, _atomic_npz, _checked_seeds, _read_shard,
    collect_cage_bank, load_episode_bank,
)
from lowdim_games.cage import BLUE_NAMES, RED_NAMES, METRIC_NAMES


def test_bank_seed_validation_rejects_overlap_duplicates_and_float_truncation():
    with pytest.raises(ValueError, match="overlap"):
        _checked_seeds([100, 101], [101])
    with pytest.raises(ValueError, match="distinct"):
        _checked_seeds([100, 100])
    with pytest.raises(ValueError, match="integer"):
        _checked_seeds([100.2])


def test_atomic_checkpoint_checks_configuration_and_native_loss_bounds(tmp_path):
    path = tmp_path / "seed_100.npz"
    shape = (len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES))
    _atomic_npz(path, seed=100, context_signature="fixed", episode_losses=np.zeros(shape),
                reward_errors=np.zeros(shape[:2]), elapsed_seconds=.2)
    losses, _, _ = _read_shard(path, 100, "fixed", [.8, 4, 10, 1])
    assert losses.shape == shape
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="configuration"):
        _read_shard(path, 101, "fixed", [.8, 4, 10, 1])
    _atomic_json(tmp_path / "manifest.json", {"status": "running"})
    with pytest.raises(ValueError, match="incomplete"):
        load_episode_bank(tmp_path)


def test_parallel_real_collection_resumes_without_replacing_shards(tmp_path):
    source = Path(__file__).parents[1] / ".external/cage-challenge-2"
    if not source.is_dir():
        pytest.skip("Optional official CAGE checkout is absent.")
    for dependency in ("yaml", "paramiko", "prettytable"):
        pytest.importorskip(dependency)
    path = collect_cage_bank(source, [9021, 9022], tmp_path, episode_steps=4, workers=2)
    first = load_episode_bank(path)
    shard_times = {file.name: file.stat().st_mtime_ns for file in (tmp_path / "shards").glob("*.npz")}
    resumed_path = collect_cage_bank(source, [9021, 9022], tmp_path, episode_steps=4, workers=1)
    second = load_episode_bank(resumed_path)
    np.testing.assert_array_equal(first["episode_losses"], second["episode_losses"])
    assert second["manifest"]["resumed_seeds"] == 2
    assert shard_times == {file.name: file.stat().st_mtime_ns for file in (tmp_path / "shards").glob("*.npz")}
    assert first["episode_losses"].shape == (2, len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES))
    assert np.max(first["reward_errors"]) < 1e-8
    with pytest.raises(ValueError, match="another source"):
        collect_cage_bank(source, [9021, 9022], tmp_path, episode_steps=5, workers=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["no_recalibration"] is True
