from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_cage_geometry_figures as module
from lowdim_games.game import FiniteGame
from lowdim_games.geometry import ProjectionResult, project_convex_hull


def crossing_game():
    return FiniteGame(np.array([[[1.,0.],[0.,0.]],[[0.,2.],[1.,2.]]]),[1.,0.])


def test_full_target_keeps_interior_mixture_responses_and_duplicate_mode_reduction():
    game = crossing_game()
    path = np.eye(2)[[0,1,0,1]]
    query = np.array([.5,1.])
    result = module.projection_row(game,query,path)
    observed = np.array([game.response_payoff(z) for z in path])
    proxy = project_convex_hull(query,observed)
    assert proxy.distance == pytest.approx(.5)
    assert result[1] < 1e-10
    assert result[0] <= result[1] <= proxy.distance
    assert result[4] == 1
    assert result[7] == 1


def test_singleton_exact_distance_and_failed_certificate_raises(monkeypatch):
    game = crossing_game()
    result = module.projection_row(game,np.array([0.,1.]),np.array([[1.,0.]]))
    assert result[0] == result[1] == 1
    assert result[2] == result[3] == 0
    monkeypatch.setattr(game,"target_projection",lambda *args,**kwargs:
                        ProjectionResult(np.zeros(2),np.ones(1),1.,1.,False))
    with pytest.raises(RuntimeError,match="certificate"):
        module.projection_row(game,np.zeros(2),np.eye(2))


def test_checkpoint_grid_captures_new_modes_and_phase_changes():
    path = np.eye(3)[[1]*16+[0,1,2]+[1]*45]
    grid = module.checkpoints(path)
    assert {1,2,4,8,16,17,19,32,48,49,64}.issubset(set(grid))
    assert len(grid)==len(set(grid))
    assert np.all(np.diff(grid)>0)


def test_no_rate_estimate_for_zero_or_uncertified_positive_distances():
    horizons = np.array([64,128,256])
    upper = np.tile(1/np.sqrt(horizons)[:,None],(1,4))
    lower = upper.copy()
    resolved = module.slope_summary(horizons,lower,upper,1e-10)
    assert resolved["estimated_slope"] == pytest.approx(-.5)
    lower[1] = 0
    assert module.slope_summary(horizons,lower,upper,1e-10)["estimated_slope"] is None
    assert module.slope_summary(horizons,np.zeros_like(lower),np.zeros_like(upper),1e-10)["estimated_slope"] is None


def test_sweep_replays_causally_and_retains_exact_selected_initialization(tmp_path):
    tensor = np.array([[[.1,.05],[.8,.01],[.5,.03]],
                       [[.5,.1],[.2,.1],[.2,.1]],
                       [[.3,.1],[.3,.2],[.4,.1]]])
    fixed = {"initial_red_index":1,"phase_fractions":[.25,.5,.25],
             "attacker_exploration":.1,"protocol_sha256":"fixture"}
    target = tmp_path/"sweep.npz"
    module.sweep_job((tensor,np.ones(2),8,45000000,fixed,str(target)))
    data = module.verify_sweep_checkpoint(target,tensor,"fixture",8,45000000)
    np.testing.assert_allclose(data["actions"][0,0],np.ones(3)/3)
    np.testing.assert_allclose(data["actions"][2,0],np.ones(3)/3)
    assert np.all(data["geometry"][:,4]==1)
    expected = module.curriculum_path(tensor,np.ones(2),8,45000000,
                                     initial_distribution=np.eye(3)[1],
                                     phase_fractions=[.25,.5,.25],exploration=.1)
    np.testing.assert_array_equal(data["opponent_actions"],expected)
    with pytest.raises(ValueError,match="frozen"):
        module.verify_sweep_checkpoint(target,tensor,"tampered",8,45000000)


def test_public_primary_loader_checks_fingerprint_and_replays_without_checkpoints(tmp_path):
    tensor = np.ones((2,3,1))*.1
    path = np.eye(3)[[1,0,2]]
    actions = np.ones((1,3,3,2))/2
    target = tmp_path/"primary_inputs.npz"
    np.savez_compressed(target,tensor=tensor,weights=np.ones(1),methods=np.asarray(module.METHODS),
                        path_seeds=np.asarray([41000000]),opponent_actions=path[None],actions=actions)
    sources = {"source_checkpoint_sha256":{"seed_41000000.npz":"original"},"retained_all_primary_paths":True}
    module.write(tmp_path/"primary_sources.json",sources)
    protocol = {"prefix":{"seeds":[41000000]},"primary_inputs_sha256":module.sha(target),
                "primary_sources_sha256":module.sha(tmp_path/"primary_sources.json"),
                "primary_checkpoint_sha256":sources["source_checkpoint_sha256"]}
    original = SimpleNamespace(tensor=tensor,weights=np.ones(1))
    got_path,got_actions,_ = module.load_primary_inputs(tmp_path,tmp_path/"absent",protocol,original)
    np.testing.assert_array_equal(got_path,path[None])
    np.testing.assert_array_equal(got_actions,actions)
    protocol["primary_inputs_sha256"]="tampered"
    with pytest.raises(ValueError,match="checksum"):
        module.load_primary_inputs(tmp_path,tmp_path/"absent",protocol,original)


def test_repeated_public_prefix_replay_preserves_frozen_crlf_sources_and_geometry(tmp_path,monkeypatch):
    tensor = np.array([[[.1],[.3],[.4]],[[.2],[.2],[.2]]])
    path = np.eye(3)[[1,1,0,1,2,1,0,2]]
    actions = np.ones((1,3,8,2))/2
    np.savez_compressed(tmp_path/"primary_inputs.npz",tensor=tensor,weights=np.ones(1),
                        methods=np.asarray(module.METHODS),path_seeds=np.asarray([41000000]),
                        opponent_actions=path[None],actions=actions)
    source = tmp_path/"primary_sources.json"
    frozen = b'{\r\n  "source_checkpoint_sha256": {"seed_41000000.npz": "original"},\r\n  "retained_all_primary_paths": true\r\n}\r\n'
    source.write_bytes(frozen)
    protocol = {"prefix":{"seeds":[41000000]},"primary_inputs_sha256":module.sha(tmp_path/"primary_inputs.npz"),
                "primary_sources_sha256":module.sha(source),
                "primary_checkpoint_sha256":{"seed_41000000.npz":"original"},
                "tensor_sha256":module.tensor_hash(tensor)}
    module.write(tmp_path/"protocol.json",protocol)
    original = SimpleNamespace(tensor=tensor,weights=np.ones(1),normalization_scale=1.)
    monkeypatch.setattr(module,"verify_protocol",lambda output:protocol)
    monkeypatch.setattr(module,"load_cage_calibration",lambda calibration:original)
    module.prefix_geometry(tmp_path,tmp_path/"absent","ignored")
    first = (tmp_path/"prefix_geometry.npz").read_bytes()
    module.prefix_geometry(tmp_path,tmp_path/"absent","ignored")
    assert (tmp_path/"prefix_geometry.npz").read_bytes() == first
    assert source.read_bytes() == frozen
