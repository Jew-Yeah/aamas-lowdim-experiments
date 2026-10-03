"""Meaningful checks for aggregate-only scientific figure reconstruction."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = spec_from_file_location("cage_dynamic_figures", Path(__file__).resolve().parents[1] / "scripts/build_cage_dynamic_figures.py")
module = module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_trailing_visualization_window_has_no_future_or_phase_reset():
    values = np.array([[1., 4.], [2., 3.], [30., 2.], [40., 1.]])
    expected = np.array([[1., 4.], [1.5, 3.5], [16., 2.5], [35., 1.5]])
    np.testing.assert_allclose(module.trailing_mean(values, window=2), expected)
    changed_future = values.copy()
    changed_future[2:] += 1000
    np.testing.assert_array_equal(module.trailing_mean(changed_future, 2)[:2], expected[:2])
    np.testing.assert_allclose(module.trailing_mean(values.T, 2, axis=1), expected.T)


def test_joint_episode_mixtures_keep_path_round_and_method_axes():
    actions = np.array([[[[.25, .75], [1., 0.]], [[.5, .5], [0., 1.]]]])
    opponent = np.array([[[1., 0.], [.2, .8]]])
    joint = module.joint_weights(actions, opponent)
    assert joint.shape == (1, 2, 2, 4)
    np.testing.assert_allclose(joint[0, 0, 0], [.25, 0, .75, 0])
    np.testing.assert_allclose(joint[0, 1, 1], [0, 0, .2, .8])
    np.testing.assert_allclose(joint.sum(axis=-1), 1)
    with pytest.raises(ValueError, match="probability"):
        module.joint_weights(actions * 2, opponent)


def test_crossed_bootstrap_matches_explicit_resampling_of_whole_units():
    rng = np.random.default_rng(15)
    joint = rng.dirichlet(np.ones(4), size=(3, 5, 3))
    bank = rng.uniform(0, 1, size=(4, 4, 2))
    series, components = module.crossed_bootstrap(joint, bank, samples=7, seed=81)
    reference_rng = np.random.default_rng(81)
    path_weights = reference_rng.multinomial(3, np.full(3, 1 / 3), size=7) / 3
    seed_weights = reference_rng.multinomial(4, np.full(4, 1 / 4), size=7) / 4
    for draw in range(7):
        direct = np.einsum("p,e,ptmj,ejc->tmc", path_weights[draw], seed_weights[draw], joint, bank)
        np.testing.assert_allclose(series[draw], direct.sum(axis=-1), rtol=0, atol=2e-15)
        np.testing.assert_allclose(components[draw], direct.mean(axis=0), rtol=0, atol=2e-15)


def test_bootstrap_preserves_correlated_cells_and_method_pairing():
    joint = np.full((3, 4, 3, 2), .5)
    # Each whole table has the same sum, despite strongly varying individual cells.
    bank = np.array([[[0.], [10.]], [[2.], [8.]], [[7.], [3.]], [[10.], [0.]]])
    series, components = module.crossed_bootstrap(joint, bank, samples=100, seed=43, batch_size=17)
    np.testing.assert_allclose(series, 5, rtol=0, atol=1e-15)
    np.testing.assert_allclose(components, 5, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(series[:, :, 0] - series[:, :, 1], 0)


def test_first_action_difference_plateaus_and_cost_units_include_50_steps():
    actions = np.zeros((2, 3, 6, 2))
    actions[..., 0] = 1
    actions[:, 0, 0] = [0, 1]
    opponent = np.ones((2, 6, 1))
    joint = module.joint_weights(actions, opponent)
    bank = np.array([[[1., 2.], [3., 5.]], [[2., 4.], [6., 10.]]])
    data = module.analyze(joint, bank, samples=100, seed=18)
    expected_first_cost = 50 * ((bank[:, 1] - bank[:, 0]).sum(axis=-1).mean())
    np.testing.assert_allclose(data["cumulative_difference"], expected_first_cost)
    np.testing.assert_allclose(np.diff(data["cumulative_difference"], axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(data["mean_components"].sum(axis=-1), data["mean_native_loss"].mean(axis=0))
    np.testing.assert_allclose(data["path_mean_difference"], expected_first_cost / (50 * 6))


def test_validation_grid_rejects_missing_or_duplicate_configurations():
    rows = [{"family": "scalar", "window": window, "rho": rho, "mean_native_loss": window + rho}
            for window in (4, 16) for rho in (0, .25)]
    windows, rhos, matrix = module.validation_matrix({"rows": rows})
    assert windows == [4, 16] and rhos == [0, .25]
    np.testing.assert_allclose(matrix, [[4, 4.25], [16, 16.25]])
    with pytest.raises(ValueError, match="full declared"):
        module.validation_matrix({"rows": rows[:-1]})
    with pytest.raises(ValueError, match="duplicate"):
        module.validation_matrix({"rows": rows[:3] + [rows[0]]})


def make_checked_fixture(tmp_path):
    report = tmp_path / "report"
    inputs = report / "geometry/primary_inputs.npz"
    bank_path = tmp_path / "bank/bank.npz"
    for path in [inputs.parent, bank_path.parent, report / "groups/test"]:
        path.mkdir(parents=True, exist_ok=True)
    actions = np.zeros((3, 3, 5, 2))
    actions[..., 0] = 1
    actions[:, 0, 0] = [.5, .5]
    opponents = np.ones((3, 5, 1))
    bank = np.arange(16, dtype=float).reshape(4, 2, 1, 2) / 10
    np.savez_compressed(inputs, actions=actions, opponent_actions=opponents,
                        methods=np.array(module.METHODS), path_seeds=np.arange(100, 103))
    np.savez_compressed(bank_path, episode_losses=bank, seeds=np.arange(200, 204),
                        metric_names=np.array(["a", "b"]), red_names=np.array(["red"]))
    module.write(bank_path.parent / "manifest.json", {"context": {"episode_steps": 50}})
    occupancies = np.einsum("pmti,ptj->pmij", actions, opponents) / 5
    occupancy_path = report / "groups/test/occupancies.npz"
    np.savez_compressed(occupancy_path, occupancies=occupancies, methods=np.array(module.METHODS))
    hashes = {f"seed_{seed}.npz": str(seed) for seed in range(100, 103)}
    module.write(inputs.parent / "primary_sources.json", {"source_checkpoint_sha256": hashes})
    module.write(report / "groups/test/meta.json", {
        "path_seeds": list(range(100, 103)), "completed_seeds": list(range(100, 103)),
        "status": "complete", "path_sha256": hashes, "aggregate_sha256": module.sha(occupancy_path)})
    module.write(report / "protocol.json", {"primary_test": {"path_seed_start": 100, "path_count": 3, "horizon": 5}})
    module.write(report / "selection.json", {"selected": {"scalar": {"window": 16, "rho": .25}}})
    module.write(report / "validation_grid.json", {"rows": []})
    components = np.einsum("pmij,eijc->mc", occupancies, bank) / 12
    module.write(report / "analysis.json", {
        "selection_sha256": module.sha(report / "selection.json"), "final_bank_sha256": module.sha(bank_path),
        "test_seeds": list(range(200, 204)),
        "primary": {"methods": {name: {"mean_components": components[i].tolist()}
                                for i, name in enumerate(module.METHODS)}}})
    return inputs, report, bank_path


def test_public_reconstruction_checks_all_completed_paths_and_primary_means(tmp_path):
    inputs, report, bank = make_checked_fixture(tmp_path)
    joint, costs, path, metrics, red, source = module.load_checked_inputs(inputs, report, bank)
    assert joint.shape == (3, 5, 3, 2)
    assert costs.shape == (4, 2, 2)
    assert metrics == ["a", "b"] and red == ["red"]
    assert source["path_seeds"] == [100, 101, 102]
    assert len(source["checkpoint_sha256"]) == 3


def test_public_reconstruction_rejects_omitted_or_tampered_path_provenance(tmp_path):
    inputs, report, bank = make_checked_fixture(tmp_path)
    source = module.read(inputs.parent / "primary_sources.json")
    source["source_checkpoint_sha256"].pop("seed_101.npz")
    module.write(inputs.parent / "primary_sources.json", source)
    with pytest.raises(ValueError, match="every original checkpoint"):
        module.load_checked_inputs(inputs, report, bank)


def test_public_reconstruction_rejects_changed_action_occupancies(tmp_path):
    inputs, report, bank = make_checked_fixture(tmp_path)
    with np.load(inputs, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values["actions"][1, 2, 3] = [0, 1]
    np.savez_compressed(inputs, **values)
    with pytest.raises(ValueError, match="original occupancies"):
        module.load_checked_inputs(inputs, report, bank)
