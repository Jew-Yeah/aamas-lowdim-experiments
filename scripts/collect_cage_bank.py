"""Collect an independent, parallel, resumable CAGE episode bank.

The output is a bank of simulator observations, not a recalibrated game.
Each worker handles one seed and all frozen Blue/Red policy pairs.  Completed
seeds are checkpointed atomically and checked before a later run reuses them.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

from lowdim_games.cage import (
    BLUE_NAMES, RED_NAMES, METRIC_NAMES, POLICY_DESCRIPTIONS,
    _source_provenance, load_cage_calibration, run_cage_episode,
    scenario_component_bounds,
)
import lowdim_games.cage as cage_adapter


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _signature(context):
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checked_seeds(seeds, excluded_seeds=()):
    supplied = list(seeds)
    if not supplied or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                           for seed in supplied):
        raise ValueError("Provide a nonempty list of integer episode seeds.")
    if len(set(supplied)) != len(supplied):
        raise ValueError("Episode seeds must be distinct.")
    overlap = set(supplied) & set(excluded_seeds)
    if overlap:
        raise ValueError(f"Episode seeds overlap the explicitly excluded bank: {sorted(overlap)}.")
    return np.asarray(supplied, dtype=np.int64)


def _read_shard(path, expected_seed, expected_signature, expected_bounds):
    with np.load(path, allow_pickle=False) as data:
        if int(data["seed"]) != int(expected_seed) or str(data["context_signature"]) != expected_signature:
            raise ValueError(f"Checkpoint seed/configuration mismatch: {path}.")
        loss = data["episode_losses"].copy()
        errors = data["reward_errors"].copy()
        elapsed = float(data["elapsed_seconds"])
    expected_shape = (len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES))
    if loss.shape != expected_shape or errors.shape != expected_shape[:2]:
        raise ValueError(f"Checkpoint dimensions mismatch: {path}.")
    if (not np.all(np.isfinite(loss)) or np.any(loss < -1e-12)
            or np.any(loss > np.asarray(expected_bounds) + 1e-10)):
        raise ValueError(f"Checkpoint loss violates native component bounds: {path}.")
    if (not np.all(np.isfinite(errors)) or np.any(errors < 0)
            or np.any(errors > 1e-8) or not np.isfinite(elapsed) or elapsed < 0):
        raise ValueError(f"Checkpoint reward accounting/timing invalid: {path}.")
    return loss, errors, elapsed


def _collect_seed(job):
    source_dir, seed, episode_steps, shard_path, context_signature = job
    started = time.perf_counter()
    losses = np.empty((len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES)))
    errors = np.empty((len(BLUE_NAMES), len(RED_NAMES)))
    for blue_index, blue in enumerate(BLUE_NAMES):
        for red_index, red in enumerate(RED_NAMES):
            episode = run_cage_episode(source_dir, blue, red, int(seed), episode_steps)
            losses[blue_index, red_index] = episode.mean_loss
            errors[blue_index, red_index] = episode.max_reward_reconstruction_error
    _atomic_npz(shard_path, seed=np.asarray(seed, dtype=np.int64),
                context_signature=np.asarray(context_signature), episode_losses=losses,
                reward_errors=errors, elapsed_seconds=time.perf_counter() - started)
    return int(seed)


def collect_cage_bank(source_dir, seeds, output_dir, *, episode_steps=50,
                      workers=4, partition="heldout", excluded_seeds=()):
    """Collect or resume a bank without changing an existing calibration.

    Expanding the requested seed list is supported.  The source revision,
    adapter and collector hashes, episode length, policies and partition must
    match previous checkpoints.  Every seed is paired across policy cells.
    """
    seeds = _checked_seeds(seeds, excluded_seeds)
    if episode_steps < 1 or workers < 1:
        raise ValueError("episode_steps and workers must be positive.")
    if partition not in ("heldout", "calibration", "external"):
        raise ValueError("partition must be heldout, calibration, or external.")
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source = _source_provenance(source_dir)
    bounds = scenario_component_bounds(source_dir)
    context = {
        "schema_version": 1, "source": source,
        "adapter_sha256": _sha256(cage_adapter.__file__),
        "collector_sha256": _sha256(__file__), "episode_steps": int(episode_steps),
        "partition": partition, "blue_names": list(BLUE_NAMES),
        "red_names": list(RED_NAMES), "metric_names": list(METRIC_NAMES),
        "blue_policy_descriptions": POLICY_DESCRIPTIONS,
        "component_bounds": bounds.tolist(),
        "normalization_scale": float(np.linalg.norm(bounds)),
        "loss_unit": "native positive loss averaged per simulator step",
        "seeded_before_environment_construction": True,
        "fresh_environment_every_episode": True,
        "simulation_only": True,
    }
    context_signature = _signature(context)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("context_signature") != context_signature:
            raise ValueError("Existing episode bank uses another source, adapter, collector, "
                             "episode length, menu or partition. Choose a new output directory.")
    started = time.perf_counter()
    completed, pending = [], []
    for seed in seeds:
        path = shard_dir / f"seed_{int(seed)}.npz"
        if path.exists():
            _read_shard(path, seed, context_signature, bounds)
            completed.append(int(seed))
        else:
            pending.append((str(source_dir), int(seed), int(episode_steps), str(path), context_signature))
    packages = {}
    for package in ("numpy", "scipy", "PyYAML", "paramiko", "prettytable"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "context": context, "context_signature": context_signature,
        "requested_seeds": seeds.tolist(), "excluded_seeds": sorted(set(map(int, excluded_seeds))),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "completed_seeds": sorted(completed),
        "resumed_seeds": len(completed), "workers": int(workers),
        "python": sys.version.split()[0], "packages": packages,
        "episode_loss_layout": ["seed", "blue_policy", "red_policy", "component"],
        "no_recalibration": True,
        "pairing": "One episode seed is reused across frozen policy cells; policy-dependent RNG consumption still changes trajectories.",
    }
    _atomic_json(manifest_path, manifest)
    print(f"CAGE bank: {len(completed)}/{len(seeds)} seed checkpoints reused; "
          f"{len(pending)} seeds to collect with {workers} workers", flush=True)
    if pending:
        # Explicit spawn gives identical worker semantics on Windows and Unix.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_collect_seed, job): job[1] for job in pending}
            for future in as_completed(futures):
                seed = future.result()
                _read_shard(shard_dir / f"seed_{seed}.npz", seed, context_signature, bounds)
                completed.append(seed)
                manifest["completed_seeds"] = sorted(completed)
                manifest["elapsed_seconds_this_run"] = time.perf_counter() - started
                _atomic_json(manifest_path, manifest)
                print(f"CAGE bank: {len(completed)}/{len(seeds)} seeds complete", flush=True)
    bank, reward_errors, durations, shard_hashes = [], [], [], {}
    for seed in seeds:
        path = shard_dir / f"seed_{int(seed)}.npz"
        losses, errors, elapsed = _read_shard(path, seed, context_signature, bounds)
        bank.append(losses)
        reward_errors.append(errors)
        durations.append(elapsed)
        shard_hashes[path.name] = _sha256(path)
    bank = np.stack(bank)
    bank_path = output_dir / "bank.npz"
    _atomic_npz(bank_path, episode_losses=bank, seeds=seeds,
                blue_names=np.asarray(BLUE_NAMES), red_names=np.asarray(RED_NAMES),
                metric_names=np.asarray(METRIC_NAMES), component_bounds=bounds,
                normalization_scale=float(np.linalg.norm(bounds)),
                reward_errors=np.asarray(reward_errors),
                seed_elapsed_seconds=np.asarray(durations),
                context_signature=np.asarray(context_signature))
    std = bank.std(axis=0, ddof=1) if len(seeds) >= 2 else np.full(bank.shape[1:], np.nan)
    statistics = {"episodes_per_cell": len(seeds), "mean": bank.mean(axis=0).tolist(),
                  "standard_deviation": std.tolist() if len(seeds) >= 2 else None,
                  "standard_error": (std / np.sqrt(len(seeds))).tolist() if len(seeds) >= 2 else None}
    _atomic_json(output_dir / "cell_statistics.json", statistics)
    manifest.update({
        "status": "complete", "completed_seeds": sorted(completed),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "seed_worker_seconds_total": float(sum(durations)),
        "episodes_total": int(len(seeds) * len(BLUE_NAMES) * len(RED_NAMES)),
        "simulator_steps_total": int(len(seeds) * len(BLUE_NAMES) * len(RED_NAMES) * episode_steps),
        "reward_reconstruction_max_abs_error": float(np.max(reward_errors)),
        "bank_npz_sha256": _sha256(bank_path), "shard_sha256": shard_hashes,
        "statistics_sha256": _sha256(output_dir / "cell_statistics.json"),
    })
    _atomic_json(manifest_path, manifest)
    return bank_path


def load_episode_bank(path):
    """Read the completed aggregate after checking its checksum and layout."""
    path = Path(path)
    if path.is_dir():
        path = path / "bank.npz"
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The episode-bank collection is incomplete.")
    if _sha256(path) != manifest["bank_npz_sha256"]:
        raise ValueError("Episode-bank checksum mismatch.")
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    if str(values["context_signature"]) != manifest["context_signature"]:
        raise ValueError("Episode-bank configuration signature mismatch.")
    if values["seeds"].tolist() != manifest["requested_seeds"]:
        raise ValueError("Episode-bank seeds do not match the manifest.")
    expected = (len(values["seeds"]), len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES))
    if values["episode_losses"].shape != expected:
        raise ValueError("Episode-bank dimensions do not match the frozen menu.")
    values["manifest"] = manifest
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".external/cage-challenge-2"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=24261003,
                        help="First seed of a contiguous bank; ignored when --seeds is supplied.")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--output", type=Path, default=Path("data/cage2_external"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--partition", choices=("heldout", "calibration", "external"), default="heldout")
    parser.add_argument("--exclude-calibration", type=Path,
                        help="Reject both train and test seeds recorded by an existing calibration.")
    parser.add_argument("--forbidden-seeds", type=int, nargs="+", default=[])
    args = parser.parse_args()
    if args.seeds is None and args.count < 1:
        parser.error("--count must be positive.")
    seeds = args.seeds if args.seeds is not None else range(args.seed, args.seed + args.count)
    excluded = list(args.forbidden_seeds)
    if args.exclude_calibration is not None:
        original = load_cage_calibration(args.exclude_calibration)
        excluded.extend(original.train_seeds.tolist())
        excluded.extend(original.test_seeds.tolist())
    path = collect_cage_bank(args.source, seeds, args.output, episode_steps=args.steps,
                             workers=args.workers, partition=args.partition,
                             excluded_seeds=excluded)
    print(f"Saved independent episode bank: {path}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
