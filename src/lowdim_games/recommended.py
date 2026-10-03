"""Selected CAGE 2 implementation and a focused, fixed-parameter comparison.

W=16 and rho=0.25 were selected on separate validation data. They are a
recommendation for this recorded benchmark, not universal optimal parameters.
The frozen implementation remains unchanged; this module only exposes it.
"""

from pathlib import Path
import hashlib
import json

import numpy as np

from .cage_adaptations import ScalarAwareOneSwitchLearner


class RecommendedOneSwitchLearner(ScalarAwareOneSwitchLearner):
    """Continuous scalar-aware one-switch with the validated CAGE 2 settings."""

    def __init__(self, tensor, response, horizon, *, weights=None,
                 initial_prior=None, oracle_tol=1e-9):
        super().__init__(tensor, response, horizon, weights=weights,
                         window=16, rho=0.25, initial_prior=initial_prior,
                         oracle_tol=oracle_tol)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_selected_cage(*, calibration="data/cage2",
                      bank="data/cage2_adaptation/test50",
                      selection="results/cage_adaptation/selection.json",
                      protocol="docs/cage_adaptation_protocol.json",
                      horizon=512, seeds=range(41000000, 41000050),
                      output="results/runs/cage_selected"):
    """Run three locked methods, then evaluate on the cached simulator bank.

    This reproduces fixed settings; it performs no parameter search or new
    simulator collection. Short runs are smoke checks, not new statistical
    evidence. No held-out bank is opened until all actions are committed.
    """
    from .cage import load_cage_calibration
    from .experiment import software_versions, tensor_hash, write_json
    from .game import FiniteGame
    from .policy_experiment import PastWindowResponse, HedgeLearner, curriculum_path

    seeds = list(seeds)
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 3:
        raise ValueError("horizon must be an integer of at least three.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use a nonempty list of distinct path seeds.")
    horizon = int(horizon)
    selection_path, protocol_path = Path(selection), Path(protocol)
    chosen = json.loads(selection_path.read_text(encoding="utf-8"))
    checksum = json.loads(selection_path.with_name("selection.sha256.json").read_text(encoding="utf-8"))
    if checksum["sha256"] != _sha256(selection_path):
        raise ValueError("Locked selection checksum mismatch.")
    if chosen["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Frozen protocol checksum mismatch.")
    settings = chosen["selected"]
    if (settings["scalar"]["window"], settings["scalar"]["rho"],
            settings["window"]["window"], settings["hedge"]["eta_multiplier"]) != (16, .25, 16, 4):
        raise ValueError("Selection differs from the validated recommended settings.")
    original = load_cage_calibration(calibration)
    if original.provenance["calibration_npz_sha256"] != chosen["calibration_sha256"]:
        raise ValueError("Calibration differs from the locked study.")
    game = FiniteGame(original.tensor, original.weights)
    fixed = json.loads(protocol_path.read_text(encoding="utf-8"))["fixed"]
    initial = np.eye(game.M)[fixed["initial_red_index"]]
    methods = ["selected_scalar_one_switch", "selected_window", "selected_hedge"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    occupancies, diagnostics, hashes = [], [], {}
    for seed in seeds:
        path = curriculum_path(game.tensor, game.weights, horizon, seed,
                               initial_distribution=initial,
                               phase_fractions=fixed["phase_fractions"],
                               exploration=fixed["attacker_exploration"])
        ours = RecommendedOneSwitchLearner(game.tensor, game.response, horizon,
                                          weights=game.weights, initial_prior=initial)
        window = PastWindowResponse(game, horizon, initial, window=16)
        hedge = HedgeLearner(game, horizon)
        hedge.learning_rate *= 4
        values = {"seed": np.asarray(seed), "methods": np.asarray(methods),
                  "opponent_actions": path}
        path_occupancies, path_diagnostics = [], []
        for index, learner in enumerate((ours, window, hedge)):
            actions, records = [], []
            for ell in path:
                actions.append(learner.choose())
                records.append(learner.observe(ell))
            actions = np.asarray(actions)
            occupancy = actions.T @ path / horizon
            values[f"method_{index}_actions"] = actions
            values[f"method_{index}_occupancy"] = occupancy
            path_occupancies.append(occupancy)
            path_diagnostics.append({
                "method": methods[index], "actions_sha256": tensor_hash(actions),
                "switch_round": getattr(learner, "switch_round", None),
                "max_saddle_gap": max((r.get("saddle_gap", 0) or 0) for r in records),
                "max_beta": max((r.get("beta_t", 0) or 0) for r in records),
                "scalar_fallback_count": sum(bool(r.get("scalar_oracle_fallback", False)) for r in records),
            })
        target = output / f"seed_{seed}.npz"
        np.savez_compressed(target, **values)
        hashes[target.name] = _sha256(target)
        occupancies.append(path_occupancies)
        diagnostics.append(path_diagnostics)

    # All learner choices are fixed before loading held-out episode outcomes.
    bank_root = Path(bank)
    manifest = json.loads((bank_root / "manifest.json").read_text(encoding="utf-8"))
    bank_path = bank_root / "bank.npz"
    if manifest.get("status") != "complete" or _sha256(bank_path) != manifest["bank_npz_sha256"]:
        raise ValueError("Incomplete or changed held-out bank.")
    context = manifest["context"]
    if (tuple(context["blue_names"]) != original.blue_names
            or tuple(context["red_names"]) != original.red_names
            or tuple(context["metric_names"]) != original.metric_names
            or context["source"] != original.provenance["source"]
            or context["episode_steps"] != 50):
        raise ValueError("Held-out bank does not match the fixed CAGE model.")
    with np.load(bank_path, allow_pickle=False) as data:
        episode_losses = data["episode_losses"].copy()
        episode_seeds = data["seeds"].copy()
        if (str(data["context_signature"]) != manifest["context_signature"]
                or episode_seeds.tolist() != manifest["requested_seeds"]):
            raise ValueError("Held-out bank layout or seed list changed.")
    if set(episode_seeds.tolist()) & set(original.train_seeds.tolist()):
        raise ValueError("Training and evaluation seeds overlap.")
    # The reported endpoint is the native SUM. FiniteGame normalizes the
    # optimization weights, so using game.weights here would divide by four.
    scalar_tables = episode_losses.sum(axis=-1)
    losses = np.einsum("pmij,eij->mep", np.asarray(occupancies), scalar_tables)
    score_path = output / "scores.npz"
    np.savez_compressed(score_path, methods=np.asarray(methods),
                        path_seeds=np.asarray(seeds), episode_seeds=episode_seeds,
                        by_method_episode_path_loss=losses)
    hashes[score_path.name] = _sha256(score_path)
    summary = {
        "settings": {"window": 16, "rho": .25, "hedge_eta_multiplier": 4,
                     "first_action": "uniform", "restarts": False},
        "horizon": horizon, "path_seeds": seeds, "episode_seed_count": len(episode_seeds),
        "native_loss_means": dict(zip(methods, losses.mean(axis=(1, 2)).tolist())),
        "selection_sha256": _sha256(selection_path), "protocol_sha256": _sha256(protocol_path),
        "calibration_sha256": chosen["calibration_sha256"],
        "bank_sha256": manifest["bank_npz_sha256"], "software": software_versions(),
        "diagnostics": diagnostics, "files_sha256": hashes,
        "interpretation": "Fixed-parameter replay on cached simulator episodes; no new tuning, statistical claims or target-geometry evaluation.",
    }
    write_json(output / "summary.json", summary)
    return summary
