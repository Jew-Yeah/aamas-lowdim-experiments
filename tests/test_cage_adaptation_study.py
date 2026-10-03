"""Split integrity, honest model selection and causal runner regressions."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts import run_cage_adaptation_study as runner


@pytest.fixture
def protocol():
    return json.loads((REPOSITORY / "docs/cage_adaptation_protocol.json").read_text(encoding="utf-8"))


@pytest.fixture
def model(protocol):
    tensor = np.random.default_rng(7).uniform(0, .01, size=(6, 3, 4))
    return SimpleNamespace(
        tensor=tensor, train_mean=tensor * 11, normalization_scale=11,
        weights=np.ones(4), train_seeds=np.arange(1000, 1040),
        test_seeds=np.arange(2000, 2040),
        blue_names=tuple(protocol["fixed"]["defender_policies"]),
        red_names=tuple(protocol["fixed"]["red_policies"]),
        metric_names=("host", "server", "disruption", "restore"),
        provenance={"calibration_npz_sha256": protocol["training"]["calibration_npz_sha256"],
                    "source": {"test_fixture": True}})


def test_validation_runs_entire_grid_and_paths_without_opening_outcome_banks(
        tmp_path, monkeypatch, protocol, model):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runner, "load_cage_calibration", lambda _: model)
    def forbidden_bank(*args, **kwargs):
        pytest.fail("Trajectory generation must not open validation or final outcome banks.")
    monkeypatch.setattr(runner, "_load_bank", forbidden_bank)
    calls = []
    def capture(original, config, output, workers):
        assert original is model
        calls.append(config)
        return {"status": "complete"}
    monkeypatch.setattr(runner, "_run_paths", capture)
    runner.run_stage("unused", "unopened_banks", tmp_path / "output", path,
                     stage="validation", workers=1)
    assert len(calls) == 1
    config = calls[0]
    assert config["path_seeds"] == list(range(40000000, 40000020))
    assert config["horizon"] == 512
    assert len(config["configurations"]) == 30
    assert [item["family"] for item in config["configurations"]].count("scalar") == 20
    assert [item["family"] for item in config["configurations"]].count("window") == 5
    assert [item["family"] for item in config["configurations"]].count("hedge") == 5


def _validation_fixture(tmp_path, protocol):
    grid = runner._grid(protocol)
    count = protocol["validation"]["path_count"]
    occupancy = np.zeros((count, len(grid), 6, 3))
    occupancy[:, :, 0, 0] = 1
    # Candidate one looks best on the first path AND on the first simulator
    # seed. Across all paths/seeds its loss is 3.762. Candidates two/three
    # both have loss 2, testing whole-bank averaging and the declared tie rule.
    for family in ("scalar", "window", "hedge"):
        indices = [index for index, config in enumerate(grid) if config["family"] == family]
        first, second, third = indices[:3]
        occupancy[:, first] = 0
        occupancy[:, first, 2, 0] = 1
        occupancy[0, first] = 0
        occupancy[0, first, 3, 0] = 1
        for index in (second, third):
            occupancy[:, index] = 0
            occupancy[:, index, 1, 0] = 1
    losses = np.zeros((100, 6, 3, 4))
    losses[:, 0, :, 2] = 6
    losses[:, 1, :, 2] = 2
    losses[1:, 2, :, 2] = 4
    validation = tmp_path / "validation"
    validation.mkdir()
    aggregate_path = validation / "occupancies.npz"
    runner._atomic_npz(aggregate_path, occupancies=occupancy)
    metadata = {"path_seeds": runner._seeds(protocol["validation"], "path_seed_start", "path_count"),
                "configurations": grid, "aggregate_sha256": runner._sha256(aggregate_path)}
    bank = {"episode_losses": losses, "seeds": np.arange(38000000, 38000100),
            "manifest": {"bank_npz_sha256": "synthetic_validation_only"}}
    return {"occupancies": occupancy}, metadata, bank


def test_selection_uses_all_validation_seeds_paths_and_exact_grid_ties(
        tmp_path, monkeypatch, protocol, model):
    aggregate, metadata, bank = _validation_fixture(tmp_path, protocol)
    monkeypatch.setattr(runner, "_aggregate", lambda *args: (aggregate, metadata))
    roles = []
    def validation_only(root, role, *args):
        roles.append(role)
        assert role == "validation", "Selection leaked final outcomes."
        return bank
    monkeypatch.setattr(runner, "_load_bank", validation_only)
    selected = runner._select(tmp_path, "unopened_final_bank", protocol, {}, model)
    assert selected["selected"]["scalar"]["name"] == "scalar_w4_r0p25"
    assert selected["selected"]["window"]["name"] == "window_w8"
    assert selected["selected"]["hedge"]["name"] == "hedge_eta0p5"
    assert selected["scores"][0]["mean_native_loss"] == pytest.approx(3.762)
    assert selected["scores"][1]["mean_native_loss"] == pytest.approx(2)
    assert selected["scores"][2]["mean_native_loss"] == pytest.approx(2)
    assert selected["final_bank_loaded"] is False
    locked_bytes = (tmp_path / "selection.json").read_bytes()
    assert b"\r" not in locked_bytes
    # Rerunning selection cannot silently reset its timestamp or locked bytes.
    assert runner._select(tmp_path, "unopened_final_bank", protocol, {}, model) == selected
    assert (tmp_path / "selection.json").read_bytes() == locked_bytes
    assert roles == ["validation", "validation"]


@pytest.mark.parametrize("missing", ["path", "configuration"])
def test_selection_refuses_partial_protocol_before_loading_any_bank(
        tmp_path, monkeypatch, protocol, model, missing):
    aggregate, metadata, _ = _validation_fixture(tmp_path, protocol)
    if missing == "path":
        metadata["path_seeds"] = metadata["path_seeds"][:-1]
    else:
        metadata["configurations"] = metadata["configurations"][:-1]
    monkeypatch.setattr(runner, "_aggregate", lambda *args: (aggregate, metadata))
    monkeypatch.setattr(runner, "_load_bank", lambda *args: pytest.fail("Partial selection opened a bank."))
    with pytest.raises(ValueError, match="every pre-specified"):
        runner._select(tmp_path, "unopened_banks", protocol, {}, model)
    assert not (tmp_path / "selection.json").exists()


def _lock_fixture(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    aggregate = validation / "occupancies.npz"
    aggregate.write_bytes(b"frozen_validation_fixture")
    selection = {"frozen_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                 "validation_aggregate_sha256": runner._sha256(aggregate),
                 "selected": {"scalar": {"kind": "scalar", "window": 4, "rho": .25},
                              "window": {"kind": "window", "window": 8},
                              "hedge": {"kind": "hedge", "eta_multiplier": .5}}}
    runner._atomic_json(tmp_path / "selection.json", selection)
    digest = runner._sha256(tmp_path / "selection.json")
    runner._atomic_json(tmp_path / "selection.sha256.json", {"sha256": digest})
    return selection, digest


def test_report_refuses_edited_selection_before_final_bank_access(tmp_path, monkeypatch, protocol, model):
    selection, _ = _lock_fixture(tmp_path)
    selection["selected"]["scalar"]["rho"] = 1
    runner._atomic_json(tmp_path / "selection.json", selection)
    monkeypatch.setattr(runner, "_load_bank", lambda *args: pytest.fail("Edited selection reached final outcomes."))
    with pytest.raises(ValueError, match="Locked selection was modified"):
        runner._report(tmp_path, "unopened_banks", protocol, {}, model)


def test_report_rejects_changed_selection_even_when_local_checksum_is_refreshed(
        tmp_path, monkeypatch, protocol, model):
    selection, original_digest = _lock_fixture(tmp_path)
    selection["selected"]["scalar"]["window"] = 64
    runner._atomic_json(tmp_path / "selection.json", selection)
    runner._atomic_json(tmp_path / "selection.sha256.json", {
        "sha256": runner._sha256(tmp_path / "selection.json")})
    # Already fixed final trajectories bind the original selection digest.
    monkeypatch.setattr(runner, "_aggregate", lambda *args: ({}, {"selection_sha256": original_digest}))
    monkeypatch.setattr(runner, "_load_bank", lambda *args: pytest.fail("Changed configuration reached outcome analysis."))
    with pytest.raises(ValueError, match="not produced by the locked selection"):
        runner._report(tmp_path, "unopened_banks", protocol, {}, model)


def test_test_stage_rejects_bad_lock_before_trajectory_or_final_outcome_access(
        tmp_path, monkeypatch, protocol, model):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(runner, "load_cage_calibration", lambda *args: model)
    selection, _ = _lock_fixture(tmp_path)
    selection.update(runner._identity(path, model))
    runner._atomic_json(tmp_path / "selection.json", selection)
    # The earlier checksum cannot validate the rewritten identity/configuration.
    monkeypatch.setattr(runner, "_load_bank", lambda *args: pytest.fail("Bad lock opened final outcomes."))
    monkeypatch.setattr(runner, "_run_paths", lambda *args: pytest.fail("Bad lock started final trajectories."))
    with pytest.raises(ValueError, match="Locked selection was modified"):
        runner.run_stage("unused", "unopened_banks", tmp_path, path, stage="test", workers=1)


def test_training_validation_overlap_and_incomplete_seed_bank_are_rejected(monkeypatch, protocol, model):
    monkeypatch.setattr(runner, "load_cage_calibration", lambda *args: model)
    model.train_seeds[0] = protocol["banks"]["validation"]["seed_start"]
    with pytest.raises(ValueError, match="must be disjoint"):
        runner._load_original("unused", protocol)
    monkeypatch.setattr(runner, "load_episode_bank", lambda *args: {
        "seeds": np.asarray(runner._seeds(protocol["banks"]["validation"])[:-1])})
    with pytest.raises(ValueError, match="seed list differs"):
        runner._load_bank("unused", "validation", protocol, model)


def test_runner_actions_cannot_depend_on_current_or_future_opponent_mode(
        tmp_path, monkeypatch, protocol, model):
    class PreviousModeProbe:
        def __init__(self):
            self.history = []
            self.pending = False
        def choose(self):
            assert not self.pending
            self.pending = True
            if not self.history:
                return np.full(6, 1 / 6)
            return np.eye(6)[int(np.argmax(self.history[-1]))]
        def observe(self, ell):
            assert self.pending, "Current opponent was disclosed before choose()."
            self.pending = False
            self.history.append(ell.copy())
            return {"t": len(self.history)}
    monkeypatch.setattr(runner, "_new_learner", lambda *args: PreviousModeProbe())
    config = runner._group_config("causal_probe", protocol["validation"], [{"name": "probe"}],
                                 protocol, {}, model)
    config["horizon"] = 4
    actions = []
    for index, modes in enumerate(([1, 1, 2, 2], [1, 1, 0, 0])):
        path = np.eye(3)[modes]
        monkeypatch.setattr(runner, "curriculum_path", lambda *args, **kwargs: path.copy())
        output = tmp_path / f"probe_{index}.npz"
        runner._simulate_path((model.tensor, model.weights, config, 3, str(output), "synthetic_probe"))
        with np.load(output, allow_pickle=False) as values:
            actions.append(values["method_0_actions"].copy())
    # The two paths differ starting at round 3; action 3 must still agree.
    np.testing.assert_array_equal(actions[0][:3], actions[1][:3])
    assert not np.array_equal(actions[0][3], actions[1][3])
