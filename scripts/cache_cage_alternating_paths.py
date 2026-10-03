"""Memoize the deterministic alternating scenario without changing the study.

The frozen runner receives twenty declared path slots.  In alternating500 its
path seed is not used by the scenario or any learner.  One complete trajectory
can therefore populate those slots.  cache.json records this explicitly; the
slots provide no additional independent attack-path variation.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts import run_cage_adaptation_study as runner


def frozen_configuration(calibration, selection_dir, protocol_path):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    original = runner._load_original(calibration, protocol)
    identity = runner._identity(protocol_path, original)
    selection, digest = runner._locked_selection(selection_dir, identity)
    section = protocol["restarts"]
    settings = section["scenarios"]["alternating500"]
    configurations = runner._selected_configs(selection, section["block_length"])
    if [config["name"] for config in configurations] != section["methods"]:
        raise ValueError("Memoization method menu differs from the frozen protocol.")
    parameters = dict(settings, horizon=section["horizon"], scenario="alternating500")
    config = runner._group_config("restarts_alternating500", parameters, configurations,
                                  protocol, identity, original)
    config.update(selection_sha256=digest,
                  attack_block_length=settings["attack_block_length"],
                  initial_red_index=settings["initial_red_index"],
                  other_red_index=settings["other_red_index"])
    return original, config, runner._signature(config)


def _require_inactive(group):
    meta = Path(group) / "meta.json"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("status") == "running":
        raise RuntimeError("The main runner is already executing alternating500. "
                           "Pause that execution before committing memoized checkpoints.")


def _arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _assert_equal_arrays(first, second, *, independent=False):
    if first.keys() != second.keys():
        raise ValueError("Deterministic checkpoints changed their saved fields.")
    ignored = {"seed", "elapsed_seconds"} if independent else {"seed"}
    for key in first:
        if key not in ignored and not np.array_equal(first[key], second[key]):
            raise ValueError(f"Deterministic memoization parity failed for {key}.")


def prepare(calibration, selection_dir, protocol_path):
    """Compute one frozen trajectory, preserving a candidate if the runner races."""
    original, config, signature = frozen_configuration(calibration, selection_dir, protocol_path)
    group = Path(selection_dir) / config["name"]
    paths = group / "paths"
    paths.mkdir(parents=True, exist_ok=True)
    _require_inactive(group)
    seed = config["path_seeds"][0]
    source = paths / f"seed_{seed}.npz"
    info_path = group / "cache_prepare.json"
    if source.exists():
        runner._read_path(source, config, seed, signature)
        if not info_path.exists():
            runner._atomic_json(info_path, {"helper_actual_computations": 0,
                                            "source_preexisted": True,
                                            "source_seed": seed})
        return source
    candidate = paths / f".memoization_candidate_{seed}.npz"
    if not candidate.exists():
        print(f"Computing deterministic CAGE path seed {seed}, T={config['horizon']}", flush=True)
        runner._simulate_path((original.tensor, original.weights, config, seed, str(candidate), signature))
    runner._read_path(candidate, config, seed, signature)
    # A complete candidate remains available if the original runner has moved
    # into this stage; no partially written expected checkpoint is exposed.
    _require_inactive(group)
    try:
        os.link(candidate, source)  # atomic, same-directory, no overwrite
    except FileExistsError:
        runner._read_path(source, config, seed, signature)
        _assert_equal_arrays(_arrays(candidate), _arrays(source), independent=True)
    candidate.unlink()
    runner._atomic_json(info_path, {"helper_actual_computations": 1,
                                    "source_preexisted": False, "source_seed": seed,
                                    "prepared_utc": datetime.now(timezone.utc).isoformat()})
    return source


def materialize(calibration, selection_dir, protocol_path):
    """Populate missing seed slots with identical arrays, changing only seed."""
    _, config, signature = frozen_configuration(calibration, selection_dir, protocol_path)
    group = Path(selection_dir) / config["name"]
    paths = group / "paths"
    source_seed = config["path_seeds"][0]
    source = paths / f"seed_{source_seed}.npz"
    _require_inactive(group)
    runner._read_path(source, config, source_seed, signature)
    source_arrays = _arrays(source)
    cache_path = group / "cache.json"
    previous = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
    previous_memoized = set(previous["materialized_seed_slots"]) if previous else set()
    existing, memoized = [], []
    for seed in config["path_seeds"]:
        _require_inactive(group)
        destination = paths / f"seed_{seed}.npz"
        if destination.exists():
            runner._read_path(destination, config, seed, signature)
            _assert_equal_arrays(source_arrays, _arrays(destination), independent=seed not in previous_memoized)
            existing.append(seed)
            continue
        values = dict(source_arrays)
        values["seed"] = np.asarray(seed, dtype=np.int64)
        # Keep publication of the expected filename atomic and nonoverwriting.
        candidate = paths / f".memoization_clone_{seed}.npz"
        runner._atomic_npz(candidate, **values)
        _require_inactive(group)
        try:
            os.link(candidate, destination)
            memoized.append(seed)
        except FileExistsError:
            existing.append(seed)
        candidate.unlink()
        runner._read_path(destination, config, seed, signature)
        _assert_equal_arrays(source_arrays, _arrays(destination), independent=seed in existing)
    info_path = group / "cache_prepare.json"
    prepared = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    all_memoized = sorted(previous_memoized | set(memoized))
    cache = {"schema_version": 1, "context_signature": signature,
             "protocol_sha256": config["protocol_sha256"],
             "selection_sha256": config["selection_sha256"],
             "helper_sha256": runner._sha256(__file__),
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "source_seed": source_seed, "source_path": f"paths/seed_{source_seed}.npz",
             "source_path_sha256": runner._sha256(source),
             "helper_actual_computations": prepared.get("helper_actual_computations", 0),
             "nominal_path_slots": len(config["path_seeds"]),
             "materialized_seed_slots": all_memoized,
             "independently_present_seed_slots": sorted(set(existing) - previous_memoized),
             "alias_map": {str(seed): source_seed for seed in config["path_seeds"]},
             "checkpoint_sha256": {f"seed_{seed}.npz": runner._sha256(paths / f"seed_{seed}.npz")
                                    for seed in config["path_seeds"]},
             "changed_array_fields": ["seed"],
             "seed_independent_branch": "alternating500 constructs its path deterministically and passes no path seed to any learner constructor",
             "added_independent_attack_path_variation": False,
             "simulator_episode_count_changed": False,
             "interpretation": "Deterministic memoization, not additional independent experiments; the frozen runner still aggregates all twenty declared slots."}
    runner._atomic_json(cache_path, cache)
    print(f"Prepared {len(config['path_seeds'])} deterministic slots; "
          f"{len(all_memoized)} memoized, no additional independent path variation", flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/cage2"))
    parser.add_argument("--selection-dir", type=Path, default=Path("results/runs/cage_adaptation/selection"))
    parser.add_argument("--protocol", type=Path, default=REPOSITORY / "docs/cage_adaptation_protocol.json")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--prepare", action="store_true", help="Compute/validate the one source trajectory only.")
    stages.add_argument("--fill", action="store_true", help="Only materialize remaining slots from a prepared trajectory.")
    args = parser.parse_args()
    if not args.fill:
        prepare(args.calibration, args.selection_dir, args.protocol)
    if not args.prepare:
        materialize(args.calibration, args.selection_dir, args.protocol)


if __name__ == "__main__":
    main()
