"""The public recommended entry point preserves the locked implementation."""

import numpy as np
from pathlib import Path
import hashlib
import json

from lowdim_games import RecommendedOneSwitchLearner
from lowdim_games.cage_adaptations import ScalarAwareOneSwitchLearner
from lowdim_games.cli import main
from lowdim_games.game import FiniteGame


def test_recommended_actions_and_diagnostics_match_locked_configuration():
    game = FiniteGame(np.array([[[.1, .05], [.8, .01]],
                               [[.5, .1], [.2, .1]],
                               [[.3, .1], [.3, .2]]]), [1, 1])
    args = (game.tensor, game.response, 24)
    options = dict(weights=game.weights, initial_prior=[1, 0])
    recommended = RecommendedOneSwitchLearner(*args, **options)
    explicit = ScalarAwareOneSwitchLearner(*args, **options, window=16, rho=.25)
    for ell in np.eye(2)[[0, 1, 1, 0, 0, 1] * 4]:
        actual, expected = recommended.choose(), explicit.choose()
        np.testing.assert_array_equal(actual, expected)
        if recommended.round == 0:
            np.testing.assert_array_equal(actual, np.ones(3) / 3)
        left, right = recommended.observe(ell), explicit.observe(ell)
        for key in ("cumulative_residual", "beta_t", "saddle_gap",
                    "scalar_oracle_actual_gap", "scalar_oracle_fallback"):
            assert left.get(key) == right.get(key)
    assert recommended.G_T == explicit.G_T


def test_selected_cli_routes_only_fixed_parameters(monkeypatch, tmp_path):
    import lowdim_games.recommended as module

    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"native_loss_means": {}}
    monkeypatch.setattr(module, "run_selected_cage", run)
    main(["cage-selected", "--horizon", "16", "--seeds", "41000000",
          "--output", str(tmp_path)])
    assert calls[0]["horizon"] == 16
    assert calls[0]["seeds"] == [41000000]
    assert calls[0]["selection"] == "results/cage_adaptation/selection.json"


def test_cached_bank_endpoint_is_native_sum_not_normalized_objective(tmp_path):
    from lowdim_games.recommended import run_selected_cage

    bank = tmp_path / "bank"
    bank.mkdir()
    manifest = json.loads(Path("data/cage2_adaptation/test50/manifest.json").read_text())
    manifest["requested_seeds"] = [39000000]
    np.savez_compressed(bank / "bank.npz", seeds=np.array([39000000]),
                        context_signature=np.asarray(manifest["context_signature"]),
                        episode_losses=np.ones((1, 6, 3, 4)))
    manifest["bank_npz_sha256"] = hashlib.sha256((bank / "bank.npz").read_bytes()).hexdigest()
    (bank / "manifest.json").write_text(json.dumps(manifest))
    summary = run_selected_cage(bank=bank, horizon=4, seeds=[41000000],
                                output=tmp_path / "run")
    # Each of four native components costs one for every policy pair, so the
    # endpoint must be four regardless of mixtures or normalized LP weights.
    for value in summary["native_loss_means"].values():
        np.testing.assert_allclose(value, 4, rtol=0, atol=1e-14)
