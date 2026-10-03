"""Validation, locked selection and independent tests of CAGE adaptations.

This is a separate experimental extension.  The original calibration remains
fixed.  Validation/select never open the final simulator bank; test trajectories
also need no final outcomes.  The report opens that bank only after verifying
the locked selection, completed trajectories and all declared seed splits.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

from lowdim_games.cage import load_cage_calibration
from lowdim_games.cage_adaptations import ScalarAwareOneSwitchLearner, BlockRestartLearner
from lowdim_games.cage_study import paired_two_way_bootstrap, holm_adjust
from lowdim_games.experiment import tensor_hash
from lowdim_games.game import FiniteGame
from lowdim_games.policy_experiment import curriculum_path, make_policy_learner
import lowdim_games.cage_adaptations as adaptation_module
import lowdim_games.cage_study as study_module
import lowdim_games.learners as learner_module
import lowdim_games.policy_experiment as policy_module
import lowdim_games.game as game_module
import lowdim_games.geometry as geometry_module

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.collect_cage_bank import _atomic_npz, load_episode_bank


def _atomic_json(path, value):
    """Canonical LF bytes keep published selection and metadata hashes valid."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _seeds(section, start="seed_start", count="count"):
    return list(range(section[start], section[start] + section[count]))


def _grid(protocol):
    validation = protocol["validation"]
    grid = []
    def add(family, kind, name, **parameters):
        grid.append({"name": name, "family": family, "kind": kind,
                     "grid_index": len(grid), **parameters})
    for window in validation["scalar_windows"]:
        for rho in validation["scalar_rhos"]:
            label = format(rho, "g").replace(".", "p")
            add("scalar", "scalar", f"scalar_w{window}_r{label}", window=window, rho=rho)
    for window in validation["window_grid"]:
        add("window", "window", f"window_w{window}", window=window)
    for multiplier in validation["hedge_eta_multipliers"]:
        label = format(multiplier, "g").replace(".", "p")
        add("hedge", "hedge", f"hedge_eta{label}", eta_multiplier=multiplier)
    return grid


def _identity(protocol_path, original):
    return {"protocol_sha256": _sha256(protocol_path),
            "calibration_sha256": original.provenance["calibration_npz_sha256"],
            "tensor_sha256": tensor_hash(original.tensor),
            "runner_sha256": _sha256(__file__),
            "implementation_sha256": {Path(module.__file__).name: _sha256(module.__file__)
                                      for module in (adaptation_module, study_module, learner_module,
                                                     policy_module, game_module, geometry_module)}}


def _load_original(calibration_path, protocol):
    original = load_cage_calibration(calibration_path)
    if original.provenance["calibration_npz_sha256"] != protocol["training"]["calibration_npz_sha256"]:
        raise ValueError("Calibration differs from the frozen adaptation protocol.")
    fixed = protocol["fixed"]
    if (original.blue_names != tuple(fixed["defender_policies"])
            or original.red_names != tuple(fixed["red_policies"])
            or not np.array_equal(original.weights, fixed["weights"])):
        raise ValueError("Original frozen policy menu or scalar weights changed.")
    training = set(original.train_seeds.tolist())
    validation = set(_seeds(protocol["banks"]["validation"]))
    testing = set(_seeds(protocol["banks"]["test"]))
    if training & validation or training & testing or validation & testing:
        raise ValueError("Training, validation and final-test seeds must be disjoint.")
    if (set(original.test_seeds.tolist()) & validation
            or set(original.test_seeds.tolist()) & testing):
        raise ValueError("Old pilot test seeds cannot enter the new validation/test banks.")
    if len(training) != protocol["training"]["train_count"]:
        raise ValueError("Wrong number of original training seeds.")
    return original


def _load_bank(bank_root, role, protocol, original):
    # Callers must not use role='test' until locked selection verification.
    bank = load_episode_bank(Path(bank_root) / f"{role}50")
    expected = _seeds(protocol["banks"][role])
    if bank["seeds"].tolist() != expected:
        raise ValueError(f"{role} bank seed list differs from the frozen protocol.")
    context = bank["manifest"]["context"]
    if (context["episode_steps"] != protocol["banks"][role]["episode_steps"]
            or tuple(context["blue_names"]) != original.blue_names
            or tuple(context["red_names"]) != original.red_names
            or tuple(context["metric_names"]) != original.metric_names
            or abs(context["normalization_scale"] - original.normalization_scale) > 1e-12
            or context["source"] != original.provenance["source"]):
        raise ValueError(f"{role} bank changed the frozen simulator, menu, components or scale.")
    return bank


def _new_learner(config, game, horizon, initial):
    kind = config["kind"]
    if kind == "scalar":
        return ScalarAwareOneSwitchLearner(
            game.tensor, game.response, horizon, weights=game.weights,
            window=config["window"], rho=config["rho"], initial_prior=initial)
    if kind == "restart":
        return BlockRestartLearner(
            game.tensor, game.response, horizon, block_length=config["block_length"],
            retain_history=config["retain_history"], scalar_aware=config["scalar_aware"],
            weights=game.weights, window=config.get("window", 16),
            rho=config.get("rho", 1.0), initial_prior=initial)
    if kind == "window":
        return make_policy_learner("last_window", game, horizon, (), initial, config["window"])
    if kind == "hedge":
        learner = make_policy_learner("hedge", game, horizon, (), initial, 16)
        learner.learning_rate *= config.get("eta_multiplier", 1.0)
        return learner
    if kind == "original":
        return make_policy_learner("one_switch", game, horizon, (), initial, 16)
    raise ValueError(f"Unknown adaptation kind {kind}.")


def _simulate_path(job):
    tensor, weights, config, seed, output, signature = job
    started = time.perf_counter()
    game = FiniteGame(tensor, weights)
    initial = np.eye(game.M)[config["initial_red_index"]]
    if config["scenario"] == "curriculum":
        path = curriculum_path(tensor, weights, config["horizon"], seed,
                               initial_distribution=initial,
                               phase_fractions=config["phase_fractions"],
                               exploration=config["attacker_exploration"])
    elif config["scenario"] == "alternating500":
        rounds = np.arange(config["horizon"]) // config["attack_block_length"]
        indices = np.where(rounds % 2 == 0, config["initial_red_index"], config["other_red_index"])
        path = np.eye(game.M)[indices]
    else:
        raise ValueError("Use the declared curriculum or alternating500 scenario.")
    values = {"seed": np.asarray(seed, dtype=np.int64),
              "context_signature": np.asarray(signature),
              "methods": np.asarray([method["name"] for method in config["configurations"]]),
              "opponent_actions": path}
    summaries = []
    for index, method in enumerate(config["configurations"]):
        learner = _new_learner(method, game, config["horizon"], initial)
        actions, payoffs, records = [], [], []
        for ell in path:
            action = learner.choose()
            record = learner.observe(ell)
            actions.append(action)
            payoffs.append(game.payoff(action, ell))
            records.append(record)
        actions, payoffs = np.asarray(actions), np.asarray(payoffs)
        occupancy = actions.T @ path / config["horizon"]
        values[f"method_{index}_actions"] = actions
        values[f"method_{index}_occupancy"] = occupancy
        values[f"method_{index}_train_payoffs"] = payoffs
        values[f"method_{index}_records_json"] = np.asarray(json.dumps(_jsonable(records), allow_nan=False))
        switches = getattr(learner, "switch_rounds", None)
        if switches is None:
            switch = getattr(learner, "switch_round", None)
            switches = [] if switch is None else [switch]
        restart_rounds = [round_index + 1 for round_index, record in enumerate(records)
                          if record.get("restart", False)]
        scalar_records = [record for record in records if "scalar_oracle_epsilon" in record]
        summaries.append({
            "method": method["name"], "actions_sha256": tensor_hash(actions),
            "opponent_path_sha256": tensor_hash(path), "switch_rounds": list(switches),
            "restart_rounds": restart_rounds,
            "max_saddle_gap": max((record.get("saddle_gap", 0) or 0) for record in records),
            "max_projection_gap": max((record.get("projection_gap", 0) or 0) for record in records),
            "scalar_fallback_count": sum(bool(record.get("scalar_oracle_fallback", False)) for record in scalar_records),
            "max_scalar_gap_excess": max([0.0] + [record.get("scalar_oracle_gap_excess", 0) or 0 for record in scalar_records]),
            "max_scalar_contract_excess": max([0.0] + [record.get("scalar_oracle_contract_excess", 0) or 0 for record in scalar_records]),
            "max_beta": max([0.0] + [record.get("beta_t", 0) or 0 for record in records]),
            "max_scalar_oracle_actual_gap": max([0.0] + [record.get("scalar_oracle_actual_gap", 0) or 0 for record in scalar_records]),
            "learning_rate": getattr(learner, "learning_rate", None)})
    values["summary_json"] = np.asarray(json.dumps(_jsonable(summaries), allow_nan=False))
    values["elapsed_seconds"] = np.asarray(time.perf_counter() - started)
    _atomic_npz(output, **values)
    return int(seed)


def _read_path(path, config, seed, signature):
    with np.load(path, allow_pickle=False) as values:
        if int(values["seed"]) != seed or str(values["context_signature"]) != signature:
            raise ValueError(f"Adaptation checkpoint changed seed/configuration: {path}.")
        methods = [method["name"] for method in config["configurations"]]
        if values["methods"].tolist() != methods:
            raise ValueError("Adaptation checkpoint changed its method menu.")
        opponent = values["opponent_actions"]
        if opponent.shape != (config["horizon"], 3):
            raise ValueError("Adaptation checkpoint opponent dimensions mismatch.")
        summaries = json.loads(str(values["summary_json"]))
        occupancies = []
        for index, method in enumerate(methods):
            actions = values[f"method_{index}_actions"]
            occupancy = values[f"method_{index}_occupancy"].copy()
            if actions.shape != (config["horizon"], 6):
                raise ValueError("Adaptation checkpoint action dimensions mismatch.")
            if (not np.isfinite(actions).all() or np.min(actions) < -1e-12
                    or np.max(np.abs(actions.sum(axis=1) - 1)) > 1e-10):
                raise ValueError("Adaptation checkpoint action is outside the simplex.")
            if np.max(np.abs(occupancy - actions.T @ opponent / config["horizon"])) > 1e-12:
                raise ValueError("Adaptation checkpoint occupancy does not replay.")
            if (summaries[index]["method"] != method
                    or tensor_hash(actions) != summaries[index]["actions_sha256"]
                    or tensor_hash(opponent) != summaries[index]["opponent_path_sha256"]):
                raise ValueError("Adaptation checkpoint action/path digest mismatch.")
            occupancies.append(occupancy)
    return np.stack(occupancies), summaries


def _run_paths(original, config, output_root, workers):
    group = Path(output_root) / config["name"]
    paths = group / "paths"
    paths.mkdir(parents=True, exist_ok=True)
    signature = _signature(config)
    metadata_path = group / "meta.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("context_signature") != signature:
            raise ValueError("Existing adaptation paths use another frozen configuration; use a new directory.")
    jobs, complete = [], []
    for seed in config["path_seeds"]:
        path = paths / f"seed_{seed}.npz"
        if path.exists():
            _read_path(path, config, seed, signature)
            complete.append(seed)
        else:
            jobs.append((original.tensor, original.weights, config, seed, str(path), signature))
    started = time.perf_counter()
    metadata = dict(config, status="running", context_signature=signature,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    completed_seeds=sorted(complete), resumed_seeds=len(complete),
                    heldout_feedback_to_learners_or_attacker=False, target_geometry_evaluated=False)
    _atomic_json(metadata_path, metadata)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_simulate_path, job) for job in jobs]
            for future in as_completed(futures):
                seed = future.result()
                _read_path(paths / f"seed_{seed}.npz", config, seed, signature)
                complete.append(seed)
                metadata["completed_seeds"] = sorted(complete)
                _atomic_json(metadata_path, metadata)
                print(f"CAGE adaptation {config['name']}: {len(complete)}/{len(config['path_seeds'])} paths", flush=True)
    occupancies, summary_rows, hashes = [], [], {}
    for seed in config["path_seeds"]:
        path = paths / f"seed_{seed}.npz"
        occupancy, summary = _read_path(path, config, seed, signature)
        occupancies.append(occupancy)
        summary_rows.append(summary)
        hashes[path.name] = _sha256(path)
    def summary_array(key, default=0):
        return np.asarray([[item.get(key, default) for item in row] for row in summary_rows])
    aggregate_path = group / "occupancies.npz"
    _atomic_npz(
        aggregate_path, occupancies=np.stack(occupancies),
        methods=np.asarray([method["name"] for method in config["configurations"]]),
        seeds=np.asarray(config["path_seeds"], dtype=np.int64),
        configurations_json=np.asarray(json.dumps(config["configurations"], sort_keys=True)),
        switch_counts=np.asarray([[len(item["switch_rounds"]) for item in row] for row in summary_rows]),
        restart_counts=np.asarray([[len(item["restart_rounds"]) for item in row] for row in summary_rows]),
        max_saddle_gaps=summary_array("max_saddle_gap"),
        max_projection_gaps=summary_array("max_projection_gap"),
        scalar_fallback_counts=summary_array("scalar_fallback_count"),
        max_scalar_gap_excesses=summary_array("max_scalar_gap_excess"),
        max_scalar_contract_excesses=summary_array("max_scalar_contract_excess"),
        max_betas=summary_array("max_beta"),
        max_scalar_oracle_actual_gaps=summary_array("max_scalar_oracle_actual_gap"),
        summaries_json=np.asarray(json.dumps(_jsonable(summary_rows), sort_keys=True)),
        context_signature=np.asarray(signature))
    _atomic_npz(group / "training_fit.npz", tensor=original.tensor, train_mean=original.train_mean,
                train_seeds=original.train_seeds, normalization_scale=original.normalization_scale)
    metadata.update(status="complete", completed_seeds=sorted(complete),
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds_this_run=time.perf_counter() - started,
                    aggregate_sha256=_sha256(aggregate_path), path_sha256=hashes)
    _atomic_json(metadata_path, metadata)
    return metadata


def _aggregate(root, name, identity):
    group = Path(root) / name
    metadata = json.loads((group / "meta.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or any(metadata[key] != value for key, value in identity.items()):
        raise ValueError(f"Incomplete or mismatched adaptation group {name}.")
    if _sha256(group / "occupancies.npz") != metadata["aggregate_sha256"]:
        raise ValueError(f"Adaptation aggregate checksum mismatch for {name}.")
    with np.load(group / "occupancies.npz", allow_pickle=False) as values:
        aggregate = {key: values[key].copy() for key in values.files}
    if (aggregate["seeds"].tolist() != metadata["path_seeds"]
            or aggregate["methods"].tolist() != [item["name"] for item in metadata["configurations"]]):
        raise ValueError("Adaptation aggregate seed or method order changed.")
    return aggregate, metadata


def _group_config(name, section, configurations, protocol, identity, original):
    return dict(identity, name=name, scenario=section.get("scenario", "curriculum"),
                horizon=section["horizon"], path_seeds=_seeds(section, "path_seed_start", "path_count"),
                configurations=configurations, train_seeds=original.train_seeds.tolist(),
                initial_red_index=protocol["fixed"]["initial_red_index"],
                phase_fractions=protocol["fixed"]["phase_fractions"],
                attacker_exploration=protocol["fixed"]["attacker_exploration"],
                normalization_scale=original.normalization_scale)


def _locked_selection(root, identity):
    path = Path(root) / "selection.json"
    checksum_path = Path(root) / "selection.sha256.json"
    if not path.is_file() or not checksum_path.is_file():
        raise ValueError("Run --stage select to lock configuration before final testing or analysis.")
    digest = _sha256(path)
    if json.loads(checksum_path.read_text())["sha256"] != digest:
        raise ValueError("Locked selection was modified.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if any(selection.get(key) != value for key, value in identity.items()):
        raise ValueError("Locked selection uses another protocol, calibration or implementation.")
    if datetime.fromisoformat(selection["frozen_at"]) > datetime.now(timezone.utc):
        raise ValueError("Locked selection has an invalid future timestamp.")
    validation_path = Path(root) / "validation" / "occupancies.npz"
    if _sha256(validation_path) != selection["validation_aggregate_sha256"]:
        raise ValueError("Validation aggregate changed after selection.")
    return selection, digest


def _select(root, bank_root, protocol, identity, original):
    aggregate, metadata = _aggregate(root, "validation", identity)
    expected_paths = _seeds(protocol["validation"], "path_seed_start", "path_count")
    grid = _grid(protocol)
    if metadata["path_seeds"] != expected_paths or metadata["configurations"] != grid:
        raise ValueError("Selection requires every pre-specified validation path and grid configuration.")
    bank = _load_bank(bank_root, "validation", protocol, original)
    scalar_bank = bank["episode_losses"].sum(axis=-1)
    scores = np.einsum("pmij,eij->pm", aggregate["occupancies"], scalar_bank, optimize=True) / len(scalar_bank)
    means = scores.mean(axis=0)
    rows = [dict(config, mean_native_loss=float(means[index])) for index, config in enumerate(grid)]
    selected = {}
    for family in ("scalar", "window", "hedge"):
        indices = [index for index, config in enumerate(grid) if config["family"] == family]
        winner = min(indices, key=lambda index: (float(means[index]), index))
        selected[family] = grid[winner]
    selection = dict(identity, schema_version=1, frozen_at=datetime.now(timezone.utc).isoformat(),
                     validation_bank_sha256=bank["manifest"]["bank_npz_sha256"],
                     validation_aggregate_sha256=metadata["aggregate_sha256"],
                     validation_seeds=bank["seeds"].tolist(), validation_path_seeds=expected_paths,
                     selected=selected, scores=rows,
                     selection_rule=protocol["validation"]["selection"],
                     final_bank_loaded=False, old_test_data_used=False)
    root = Path(root)
    if (root / "selection.json").exists():
        previous, _ = _locked_selection(root, identity)
        for key in ("selected", "scores", "validation_bank_sha256", "validation_aggregate_sha256"):
            if previous[key] != selection[key]:
                raise ValueError("Selection inputs/results changed; existing selection will not be overwritten.")
        return previous
    _atomic_json(root / "validation_grid.json", dict(identity, rows=rows, selected=selected))
    _atomic_json(root / "selection.json", selection)
    _atomic_json(root / "selection.sha256.json", {"sha256": _sha256(root / "selection.json")})
    return selection


def _selected_configs(selection, restart_length=None):
    selected = selection["selected"]
    configs = [dict(selected["scalar"], name="selected_scalar_one_switch"),
               dict(selected["window"], name="selected_window"),
               dict(selected["hedge"], name="selected_hedge")]
    if restart_length is None:
        return configs + [dict(name="original_one_switch", kind="original"),
                          dict(name="original_window16", kind="window", window=16),
                          dict(name="original_hedge", kind="hedge", eta_multiplier=1)]
    scalar = selected["scalar"]
    restarts = {}
    for retain, label in ((False, "fresh"), (True, "retained")):
        for aware, prefix in ((False, ""), (True, "scalar_")):
            name = f"{prefix}{label}_restart{restart_length}"
            restarts[name] = dict(name=name, kind="restart", block_length=restart_length,
                                  retain_history=retain, scalar_aware=aware,
                                  window=scalar["window"], rho=scalar["rho"])
    by_name = {item["name"]: item for item in configs}
    by_name.update(restarts)
    by_name["original_one_switch"] = dict(name="original_one_switch", kind="original")
    names = ["original_one_switch", f"fresh_restart{restart_length}", f"retained_restart{restart_length}",
             "selected_scalar_one_switch", f"scalar_fresh_restart{restart_length}",
             f"scalar_retained_restart{restart_length}", "selected_window", "selected_hedge"]
    return [by_name[name] for name in names]


def _evaluate_group(aggregate, bank):
    components = np.einsum("pmij,eijd->epmd", aggregate["occupancies"], bank["episode_losses"], optimize=True)
    methods = aggregate["methods"].tolist()
    result = {}
    for index, method in enumerate(methods):
        mean_components = components[:, :, index].mean(axis=(0, 1))
        result[method] = {"mean_components": mean_components.tolist(),
                          "mean_native_loss": float(mean_components.sum()),
                          "switch_count": int(aggregate["switch_counts"][:, index].sum()),
                          "restart_count": int(aggregate["restart_counts"][:, index].sum()),
                          "scalar_fallback_count": int(aggregate["scalar_fallback_counts"][:, index].sum()),
                          "max_scalar_gap_excess": float(aggregate["max_scalar_gap_excesses"][:, index].max()),
                          "max_scalar_contract_excess": float(aggregate["max_scalar_contract_excesses"][:, index].max()),
                          "max_beta": float(aggregate["max_betas"][:, index].max()),
                          "max_scalar_oracle_actual_gap": float(aggregate["max_scalar_oracle_actual_gaps"][:, index].max())}
    return result


def _contrast(aggregate, bank, method, reference, *, samples, seed, alpha):
    methods = aggregate["methods"].tolist()
    result = paired_two_way_bootstrap(
        aggregate["occupancies"][:, methods.index(method)],
        aggregate["occupancies"][:, methods.index(reference)], bank["episode_losses"],
        samples=samples, seed=seed, alpha=alpha, return_differences=False)
    return _jsonable(result)


def _report(root, bank_root, protocol, identity, original):
    # The lock and all trajectories are checked BEFORE final-bank access.
    selection, selection_digest = _locked_selection(root, identity)
    names = ["test", "restarts_curriculum", "restarts_alternating500"]
    groups = {name: _aggregate(root, name, identity) for name in names}
    for _, metadata in groups.values():
        if metadata["selection_sha256"] != selection_digest:
            raise ValueError("Final trajectories were not produced by the locked selection.")
    analysis_started = datetime.now(timezone.utc).isoformat()
    if datetime.fromisoformat(selection["frozen_at"]) >= datetime.fromisoformat(analysis_started):
        raise ValueError("Selection must precede final outcome analysis.")
    validation = _load_bank(bank_root, "validation", protocol, original)
    if validation["manifest"]["bank_npz_sha256"] != selection["validation_bank_sha256"]:
        raise ValueError("Validation bank changed after selection.")
    final = _load_bank(bank_root, "test", protocol, original)
    primary = protocol["primary_test"]
    aggregate, metadata = groups["test"]
    alpha = primary["family_alpha"] / len(primary["contrasts"])
    contrasts = {}
    for reference in ("selected_window", "selected_hedge"):
        contrast = _contrast(aggregate, final, "selected_scalar_one_switch", reference,
                             samples=primary["bootstrap_samples"], seed=primary["bootstrap_seed"], alpha=alpha)
        contrast["advantage_supported"] = contrast["upper"] < 0
        contrasts[reference] = contrast
    adjusted = holm_adjust({reference: result["approximate_two_sided_pvalue"]
                            for reference, result in contrasts.items()})
    for reference in contrasts:
        contrasts[reference]["approximate_holm_adjusted_pvalue"] = adjusted[reference]
    restarts = {}
    for scenario in protocol["restarts"]["scenarios"]:
        aggregate, metadata = groups[f"restarts_{scenario}"]
        comparisons = {}
        for method, reference in (
            ("fresh_restart1000", "original_one_switch"),
            ("retained_restart1000", "original_one_switch"),
            ("scalar_fresh_restart1000", "selected_scalar_one_switch"),
            ("scalar_retained_restart1000", "selected_scalar_one_switch")):
            comparisons[f"{method}-minus-{reference}"] = _contrast(
                aggregate, final, method, reference, samples=primary["bootstrap_samples"],
                seed=primary["bootstrap_seed"] + 1, alpha=.05)
        restarts[scenario] = {"methods": _evaluate_group(aggregate, final),
                              "contrasts": comparisons, "descriptive_only": True,
                              "path_count": len(aggregate["seeds"]), "horizon": metadata["horizon"]}
    primary_aggregate, primary_metadata = groups["test"]
    analysis = dict(identity, schema_version=1, analysis_created_utc=analysis_started,
                    selection_sha256=selection_digest, selection_frozen_at=selection["frozen_at"],
                    selected=selection["selected"],
                    validation_bank_sha256=selection["validation_bank_sha256"],
                    final_bank_sha256=final["manifest"]["bank_npz_sha256"],
                    validation_seeds=validation["seeds"].tolist(), test_seeds=final["seeds"].tolist(),
                    primary={"methods": _evaluate_group(primary_aggregate, final), "contrasts": contrasts,
                             "alpha_per_comparison": alpha, "family_alpha": primary["family_alpha"],
                             "path_count": len(primary_aggregate["seeds"]), "horizon": primary_metadata["horizon"]},
                    restarts=restarts, inference=protocol["inference"],
                    interpretation="Expected native episode losses in the frozen empirical CAGE model; bootstrap intervals conditional on original calibration and validation-selected parameters.")
    _atomic_json(Path(root) / "analysis.json", analysis)
    return analysis


def run_stage(calibration_path, bank_root, output_root, protocol_path, *, stage, workers=8, smoke=False):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    original = _load_original(calibration_path, protocol)
    identity = _identity(protocol_path, original)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if smoke:
        if stage != "validation":
            raise ValueError("Smoke mode uses training-only validation-shaped paths, never bank outcomes.")
        config = _group_config("smoke_validation", protocol["validation"], _grid(protocol)[:2],
                               protocol, identity, original)
        config.update(horizon=8, path_seeds=config["path_seeds"][:2], smoke_override=True)
        return _run_paths(original, config, root, workers)
    stages = ("validation", "select", "test", "restarts", "report") if stage == "all" else (stage,)
    results = {}
    for current in stages:
        if current == "validation":
            config = _group_config("validation", protocol["validation"], _grid(protocol),
                                   protocol, identity, original)
            results[current] = _run_paths(original, config, root, workers)
        elif current == "select":
            results[current] = _select(root, bank_root, protocol, identity, original)
        elif current in ("test", "restarts"):
            selection, digest = _locked_selection(root, identity)
            if current == "test":
                config = _group_config("test", protocol["primary_test"], _selected_configs(selection),
                                       protocol, identity, original)
                config["selection_sha256"] = digest
                results[current] = _run_paths(original, config, root, workers)
            else:
                section = protocol["restarts"]
                configurations = _selected_configs(selection, section["block_length"])
                if [config["name"] for config in configurations] != section["methods"]:
                    raise ValueError("Restart method menu differs from the frozen protocol.")
                results[current] = {}
                for scenario, settings in section["scenarios"].items():
                    parameters = dict(settings, horizon=section["horizon"], scenario=scenario)
                    config = _group_config(f"restarts_{scenario}", parameters, configurations,
                                           protocol, identity, original)
                    config.update(selection_sha256=digest)
                    if scenario == "alternating500":
                        config.update(attack_block_length=settings["attack_block_length"],
                                      initial_red_index=settings["initial_red_index"],
                                      other_red_index=settings["other_red_index"])
                    results[current][scenario] = _run_paths(original, config, root, workers)
        elif current == "report":
            results[current] = _report(root, bank_root, protocol, identity, original)
        else:
            raise ValueError("Unknown adaptation-study stage.")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/cage2"))
    parser.add_argument("--bank-root", type=Path, default=Path("results/runs/cage_adaptation/banks"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/cage_adaptation"))
    parser.add_argument("--protocol", type=Path, default=REPOSITORY / "docs/cage_adaptation_protocol.json")
    parser.add_argument("--stage", choices=("validation", "select", "test", "restarts", "report", "all"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    run_stage(args.calibration, args.bank_root, args.output, args.protocol,
              stage=args.stage, workers=args.workers, smoke=args.smoke)
    print(f"Completed requested adaptation stage: {args.stage}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
