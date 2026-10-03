"""Run the pre-specified extended CAGE policy study with resumable trajectories.

Only calibration tensors enter policy selection.  Held-out banks can still be
collecting while the original-fit trajectories run; their seed lists are fixed
by the protocol and the reporting step evaluates outcomes afterwards.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

from lowdim_games.cage import load_cage_calibration
from lowdim_games.cage_study import simulate_policy_selection
from lowdim_games.experiment import tensor_hash
from lowdim_games.policy_experiment import _fixed_names
import lowdim_games.cage_study as study_module
import lowdim_games.policy_experiment as policy_module
import lowdim_games.learners as learner_module
import lowdim_games.game as game_module
import lowdim_games.geometry as geometry_module

# The collector is intentionally a repository script, not an installed module.
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.collect_cage_bank import _atomic_json, _atomic_npz, _read_shard, load_episode_bank


DIAGNOSTIC_NAMES = ("residual", "cumulative_residual", "h_t", "saddle_gap",
                    "projection_gap", "switch")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _methods(blue_names):
    return ["one_switch", "shared_past_hull", "block_safe", "uniform",
            "historical_best", "last_window", "hedge", *_fixed_names(blue_names)]


def _plans(protocol, stage):
    primary, secondary = protocol["primary"], protocol["secondary"]
    starts = secondary["path_seed_starts"]
    groups = []

    def add(name, scenario, count, seed_start, *, horizon=None, steps=50,
            calibration_key="original", evaluation_bank_key="primary50",
            training_seeds=None, test_seeds=None):
        groups.append({"name": name, "scenario": scenario, "path_count": int(count),
                       "path_seeds": list(range(seed_start, seed_start + count)),
                       "horizon": int(primary["meta_horizon"] if horizon is None else horizon),
                       "episode_steps": int(steps), "calibration_key": calibration_key,
                       "evaluation_bank_key": evaluation_bank_key,
                       "training_seeds": training_seeds, "test_seeds": test_seeds})

    if stage in ("primary", "all"):
        add("primary", primary["scenario"], primary["path_seed_count"], primary["path_seed_start"])
    if stage in ("secondary", "all"):
        for scenario in secondary["scenarios"]:
            add(scenario, scenario, secondary["path_count"], starts[scenario])
        for horizon in secondary["meta_horizons"]:
            add(f"horizon{horizon}", "curriculum", secondary["horizon_path_count"],
                starts[f"horizon_{horizon}"], horizon=horizon)
    if stage in ("replications", "all"):
        replication = secondary["calibration_replications"]
        for group in range(replication["groups"]):
            first = replication["seed_start"] + group * replication["seeds_per_group"]
            training = list(range(first, first + replication["seeds_per_group"]))
            add(f"replication{group}", "curriculum", replication["paths_per_group"],
                starts[f"calibration_replication_{group}"],
                calibration_key=f"calibration50_group{group}", training_seeds=training)
    if stage in ("sensitivities", "all"):
        for sensitivity in secondary["episode_sensitivities"]:
            steps, first = sensitivity["steps"], sensitivity["seed_start"]
            split = first + sensitivity["training_count"]
            training = list(range(first, split))
            testing = list(range(split, split + sensitivity["test_count"]))
            add(f"steps{steps}", "curriculum", sensitivity["path_count"],
                starts[f"episode_{steps}"], steps=steps,
                calibration_key=f"step{steps}_first{sensitivity['training_count']}",
                evaluation_bank_key=f"step{steps}", training_seeds=training, test_seeds=testing)
    return groups


def _training_rows(bank_dir, seeds):
    """Load only the declared training rows, including completed shard subsets."""
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    context = manifest["context"]
    if manifest.get("status") == "complete":
        bank = load_episode_bank(bank_dir)
        indices = {int(seed): index for index, seed in enumerate(bank["seeds"])}
        if not set(seeds).issubset(indices):
            raise ValueError(f"Declared calibration seeds missing from {bank_dir}.")
        rows = bank["episode_losses"][[indices[seed] for seed in seeds]]
    else:
        rows = []
        for seed in seeds:
            path = bank_dir / "shards" / f"seed_{seed}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Calibration shard is not complete yet: {path}.")
            losses, _, _ = _read_shard(path, seed, manifest["context_signature"],
                                       context["component_bounds"])
            rows.append(losses)
        rows = np.stack(rows)
    return rows, context


def _fit(original, plan, bank_root, protocol):
    if plan["calibration_key"] == "original":
        tensor = original.tensor.copy()
        training_seeds = original.train_seeds.tolist()
        training_mean = original.train_mean.copy()
        training_data_hash = tensor_hash(original.train_episode_losses)
        source = original.provenance["source"]
    else:
        training_seeds = plan["training_seeds"]
        key = ("calibration50" if plan["calibration_key"].startswith("calibration50")
               else plan["evaluation_bank_key"])
        rows, context = _training_rows(Path(bank_root) / key, training_seeds)
        if (tuple(context["blue_names"]) != original.blue_names
                or tuple(context["red_names"]) != original.red_names
                or tuple(context["metric_names"]) != original.metric_names):
            raise ValueError("Calibration bank changed the original frozen policies or reward components.")
        if context["episode_steps"] != plan["episode_steps"]:
            raise ValueError("Calibration bank episode length differs from the declared sensitivity.")
        if abs(context["normalization_scale"] - original.normalization_scale) > 1e-12:
            raise ValueError("Calibration bank changed the a priori normalization.")
        training_mean = rows.mean(axis=0)
        tensor = training_mean / original.normalization_scale
        training_data_hash = tensor_hash(rows)
        source = context["source"]
    primary = protocol["primary"]
    testing_seeds = plan["test_seeds"]
    if testing_seeds is None:
        testing_seeds = list(range(primary["heldout_seed_start"],
                                   primary["heldout_seed_start"] + primary["heldout_seed_count"]))
    if set(training_seeds) & set(testing_seeds):
        raise ValueError("A study group reuses its calibration seeds for evaluation.")
    weights = np.asarray(protocol["fixed_choices"]["response_weights"], dtype=float)
    if not np.array_equal(weights, original.weights):
        raise ValueError("Protocol weights differ from the original fit.")
    # This deliberately contains no held-out means, episode bank or seed losses.
    calibration = {"tensor": tensor, "weights": weights,
                   "blue_names": original.blue_names, "red_names": original.red_names,
                   "metric_names": original.metric_names}
    metadata = {"train_seeds": training_seeds, "test_seeds": testing_seeds,
                "training_data_sha256": training_data_hash,
                "tensor_sha256": tensor_hash(tensor), "source": source,
                "normalization_scale": original.normalization_scale,
                "blue_names": list(original.blue_names), "red_names": list(original.red_names),
                "metric_names": list(original.metric_names), "response_weights": weights.tolist()}
    return calibration, training_mean, metadata


def _explicit_path(scenario, horizon, modes):
    if scenario == "fixed_meander":
        return np.repeat(np.eye(modes)[1][None, :], horizon, axis=0)
    if scenario == "fixed_b_line":
        return np.repeat(np.eye(modes)[2][None, :], horizon, axis=0)
    if scenario == "weak_to_strong":
        indices = np.where(np.arange(horizon) < horizon // 2, 1, 2)
        return np.eye(modes)[indices]
    if scenario == "alternating_64":
        return np.eye(modes)[1 + (np.arange(horizon) // 64) % 2]
    return None


def _simulate_one(job):
    calibration, config, seed, path, signature = job
    started = time.perf_counter()
    initial_index = 2 if config["scenario"] == "fixed_b_line" else config["initial_red_index"]
    initial = np.eye(len(calibration["red_names"]))[initial_index]
    explicit = _explicit_path(config["scenario"], config["horizon"], len(initial))
    run = simulate_policy_selection(
        calibration, scenario=config["scenario"], horizon=config["horizon"],
        seed=int(seed), opponent_path=explicit, initial_distribution=initial,
        phase_fractions=config["phase_fractions"],
        attacker_exploration=config["attacker_exploration"],
        methods=config["methods"], window=config["window"])
    arrays = {"seed": np.asarray(seed, dtype=np.int64),
              "context_signature": np.asarray(signature),
              "methods": np.asarray(config["methods"]),
              "diagnostic_names": np.asarray(DIAGNOSTIC_NAMES),
              "elapsed_seconds": np.asarray(time.perf_counter() - started)}
    summaries = []
    for index, method in enumerate(config["methods"]):
        trajectory = run["trajectories"][method]
        for key in ("actions", "opponent_actions", "occupancy", "train_payoffs"):
            arrays[f"method_{index}_{key}"] = trajectory[key]
        arrays[f"method_{index}_diagnostics"] = np.column_stack(
            [trajectory["diagnostics"][name] for name in DIAGNOSTIC_NAMES])
        summaries.append({
            "method": method, "switch_round": trajectory["switch_round"],
            "switch_threshold": trajectory["switch_threshold"],
            "max_saddle_gap": trajectory["max_saddle_gap"],
            "max_past_hull_projection_gap": trajectory["max_past_hull_projection_gap"],
            "realized_affine_dimension": trajectory["realized_affine_dimension"],
            "opponent_path_sha256": trajectory["opponent_path_sha256"],
            "actions_sha256": trajectory["actions_sha256"]})
    arrays["summary_json"] = np.asarray(json.dumps(summaries, sort_keys=True))
    _atomic_npz(path, **arrays)
    return int(seed)


def _read_trajectory(path, config, expected_seed, signature):
    with np.load(path, allow_pickle=False) as values:
        if int(values["seed"]) != expected_seed or str(values["context_signature"]) != signature:
            raise ValueError(f"Trajectory checkpoint uses another seed/configuration: {path}.")
        if values["methods"].tolist() != config["methods"]:
            raise ValueError(f"Trajectory checkpoint changed the method menu: {path}.")
        summaries = json.loads(str(values["summary_json"]))
        occupancies = []
        for index, method in enumerate(config["methods"]):
            actions = values[f"method_{index}_actions"]
            opponent = values[f"method_{index}_opponent_actions"]
            occupancy = values[f"method_{index}_occupancy"].copy()
            if (actions.shape != (config["horizon"], len(config["blue_names"]))
                    or opponent.shape != (config["horizon"], len(config["red_names"]))):
                raise ValueError(f"Trajectory dimensions mismatch: {path}.")
            if (not np.all(np.isfinite(actions)) or not np.all(np.isfinite(opponent))
                    or np.any(actions < -1e-12) or np.any(opponent < -1e-12)
                    or not np.allclose(actions.sum(axis=1), 1, atol=1e-10)
                    or not np.allclose(opponent.sum(axis=1), 1, atol=1e-10)):
                raise ValueError(f"Trajectory violates the probability simplexes: {path}.")
            if not np.allclose(occupancy, actions.T @ opponent / config["horizon"], atol=1e-13):
                raise ValueError(f"Occupancy does not match its saved actions/path: {path}.")
            if (summaries[index]["method"] != method
                    or summaries[index]["actions_sha256"] != tensor_hash(actions)
                    or summaries[index]["opponent_path_sha256"] != tensor_hash(opponent)):
                raise ValueError(f"Trajectory hash/summary mismatch: {path}.")
            occupancies.append(occupancy)
        elapsed = float(values["elapsed_seconds"])
    return np.stack(occupancies), summaries, elapsed


def _run_group(calibration, training_mean, config, output_dir, *, workers):
    group_dir = Path(output_dir) / config["name"]
    paths_dir = group_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    signature = _signature(config)
    meta_path = group_dir / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous["context_signature"] != signature:
            raise ValueError(f"Study configuration changed for {config['name']}; choose a new output directory.")
    fixed = config["scenario"] in ("fixed_meander", "fixed_b_line")
    computed_seeds = config["path_seeds"][:1] if fixed else config["path_seeds"]
    pending, completed = [], []
    for seed in computed_seeds:
        path = paths_dir / f"seed_{seed}.npz"
        if path.is_file():
            _read_trajectory(path, config, seed, signature)
            completed.append(seed)
        else:
            pending.append((calibration, config, seed, str(path), signature))
    started = time.perf_counter()
    metadata = dict(config)
    metadata.update({"status": "running", "context_signature": signature,
                     "started_utc": datetime.now(timezone.utc).isoformat(),
                     "completed_computations": sorted(completed), "resumed_computations": len(completed),
                     "identical_fixed_paths_collapsed": fixed,
                     "distinct_trajectories_computed": len(computed_seeds),
                     "heldout_feedback_to_learners_or_attacker": False,
                     "target_geometry_evaluated": False,
                     "workers": workers})
    _atomic_json(meta_path, metadata)
    _atomic_npz(group_dir / "training_fit.npz", tensor=calibration["tensor"],
                train_mean=training_mean, train_seeds=np.asarray(config["train_seeds"]),
                normalization_scale=config["normalization_scale"])
    if pending:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_simulate_one, job): job[2] for job in pending}
            for future in as_completed(futures):
                seed = future.result()
                _read_trajectory(paths_dir / f"seed_{seed}.npz", config, seed, signature)
                completed.append(seed)
                metadata["completed_computations"] = sorted(completed)
                metadata["elapsed_seconds_this_run"] = time.perf_counter() - started
                _atomic_json(meta_path, metadata)
                print(f"CAGE study {config['name']}: {len(completed)}/{len(computed_seeds)} paths", flush=True)
    occupancy_rows, summary_rows, durations, aliases = [], [], [], []
    hashes = {}
    for seed in config["path_seeds"]:
        actual_seed = computed_seeds[0] if fixed else seed
        path = paths_dir / f"seed_{actual_seed}.npz"
        occupancies, summary, elapsed = _read_trajectory(path, config, actual_seed, signature)
        occupancy_rows.append(occupancies)
        summary_rows.append(summary)
        durations.append(elapsed)
        aliases.append(actual_seed)
        hashes[path.name] = _sha256(path)
    def summary_array(key, default):
        return np.asarray([[entry[key] if entry[key] is not None else default for entry in row]
                           for row in summary_rows])
    aggregate_path = group_dir / "occupancies.npz"
    _atomic_npz(
        aggregate_path, occupancies=np.stack(occupancy_rows), methods=np.asarray(config["methods"]),
        seeds=np.asarray(config["path_seeds"], dtype=np.int64),
        checkpoint_seeds=np.asarray(aliases, dtype=np.int64),
        switch_rounds=summary_array("switch_round", -1).astype(np.int64),
        switch_thresholds=summary_array("switch_threshold", np.nan).astype(float),
        max_saddle_gaps=summary_array("max_saddle_gap", 0),
        max_past_hull_projection_gaps=summary_array("max_past_hull_projection_gap", 0),
        realized_affine_dimensions=summary_array("realized_affine_dimension", 0),
        opponent_path_hashes=summary_array("opponent_path_sha256", ""),
        actions_hashes=summary_array("actions_sha256", ""),
        context_signature=np.asarray(signature))
    metadata.update({"status": "complete", "completed_computations": sorted(completed),
                     "completed_utc": datetime.now(timezone.utc).isoformat(),
                     "elapsed_seconds_this_run": time.perf_counter() - started,
                     "aggregate_sha256": _sha256(aggregate_path), "checkpoint_sha256": hashes,
                     "training_fit_sha256": _sha256(group_dir / "training_fit.npz"),
                     "aggregate_occupancy_layout": ["path", "method", "blue_policy", "red_policy"],
                     "same_opponent_path_across_methods": config["scenario"] != "interactive",
                     "phase_description": {
                         "fixed_meander": "All rounds use Meander.",
                         "fixed_b_line": "All rounds use B_line.",
                         "weak_to_strong": "First floor(T/2) rounds use Meander, then B_line.",
                         "alternating_64": "64-round Meander/B_line blocks starting with Meander.",
                         "interactive": "Each method faces the same causal attacker rule but induces its own path.",
                         "curriculum": "Original three-phase curriculum, with common path learned against uniform defence.",
                     }[config["scenario"]]})
    _atomic_json(meta_path, metadata)
    return metadata


def run_study(calibration_dir, bank_root, output_dir, *, protocol_path=REPOSITORY / "docs/cage_study_protocol.json",
              stage="all", workers=8, smoke=False):
    if workers < 1:
        raise ValueError("workers must be positive.")
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    original = load_cage_calibration(calibration_dir)
    expected_hash = protocol["primary"]["calibration_npz_sha256"].lower()
    if original.provenance["calibration_npz_sha256"].lower() != expected_hash:
        raise ValueError("Original calibration differs from the frozen study protocol.")
    plans = _plans(protocol, stage)
    if not plans:
        raise ValueError("Unknown or empty study stage.")
    if smoke:
        if stage != "primary":
            raise ValueError("The smoke override is available only for --stage primary.")
        plans[0].update(name="smoke_primary", horizon=8, path_count=2,
                        path_seeds=plans[0]["path_seeds"][:2],
                        evaluation_bank_key="original_pilot_test_smoke",
                        test_seeds=original.test_seeds.tolist())
    implementation_hashes = {Path(module.__file__).name: _sha256(module.__file__)
                             for module in (study_module, policy_module, learner_module,
                                            game_module, geometry_module)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for plan in plans:
        calibration, training_mean, fit_metadata = _fit(original, plan, bank_root, protocol)
        config = dict(plan)
        config.pop("training_seeds")
        config.pop("test_seeds")
        config.update(fit_metadata)
        config.update({"schema_version": 1, "protocol_sha256": _sha256(protocol_path),
                       "runner_sha256": _sha256(__file__), "implementation_sha256": implementation_hashes,
                       "methods": _methods(original.blue_names),
                       "phase_fractions": protocol["primary"]["phase_fractions"],
                       "initial_red_index": protocol["primary"]["initial_red_index"],
                       "attacker_exploration": protocol["primary"]["attacker_exploration"],
                       "window": protocol["fixed_choices"]["window"], "smoke_override": smoke,
                       "original_calibration_npz_sha256": expected_hash})
        results.append(_run_group(calibration, training_mean, config, output_dir, workers=workers))
    index = []
    for meta_path in sorted(output_dir.glob("*/meta.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "complete" and metadata.get("protocol_sha256") == _sha256(protocol_path):
            index.append({"name": metadata["name"], "metadata": str(meta_path.relative_to(output_dir)),
                          "aggregate": f"{metadata['name']}/occupancies.npz",
                          "aggregate_sha256": metadata["aggregate_sha256"],
                          "calibration_key": metadata["calibration_key"],
                          "evaluation_bank_key": metadata["evaluation_bank_key"],
                          "path_count": metadata["path_count"]})
    _atomic_json(output_dir / "index.json", {"schema_version": 1, "protocol_sha256": _sha256(protocol_path),
                                             "completed_groups": index})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/cage2"))
    parser.add_argument("--bank-root", type=Path, default=Path("results/runs/cage_study/banks"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/cage_study/selection"))
    parser.add_argument("--protocol", type=Path, default=REPOSITORY / "docs/cage_study_protocol.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stage", choices=("primary", "secondary", "replications", "sensitivities", "all"),
                        default="all")
    parser.add_argument("--smoke", action="store_true", help="Primary-only two-path T=8 check, labelled as a smoke override.")
    args = parser.parse_args()
    results = run_study(args.calibration, args.bank_root, args.output, protocol_path=args.protocol,
                        stage=args.stage, workers=args.workers, smoke=args.smoke)
    print(f"Completed {len(results)} requested study groups", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
