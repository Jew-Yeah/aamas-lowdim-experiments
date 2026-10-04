"""Frozen paired validation of the resource-service switching mechanism.

The independent episode, rather than a dependent round, is the sampling unit.
All scenarios, controls and non-switching episodes are retained. This constructed
game is a mechanism stress test, not a representative sample of real systems.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lowdim_games.resource_validation import (
    METHODS, ResourceValidationConfig, run_resource_validation_batch,
)

DESIGN = ROOT / "docs/resource_validation_protocol.json"
SOURCES = [Path(__file__), ROOT / "src/lowdim_games/resource_validation.py",
           ROOT / "src/lowdim_games/resource_target.py",
           ROOT / "src/lowdim_games/paired_inference.py",
           ROOT / "src/lowdim_games/learners.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in SOURCES}


def configs(design):
    allowed = {f.name for f in fields(ResourceValidationConfig)}
    fixed = design["fixed_settings"]
    for scenario in design["scenarios"]:
        kwargs = {key: value for key, value in fixed.items() if key in allowed}
        kwargs.update({key: value for key, value in scenario.items() if key in allowed})
        if "scenario_id" in allowed:
            kwargs["scenario_id"] = scenario["id"]
        # Explicit aliases are checked against the saved config in the analysis.
        if "change_round" in allowed and "H" in fixed:
            kwargs["change_round"] = fixed["H"]
        if "H" in allowed and "change_round" in fixed:
            kwargs["H"] = fixed["change_round"]
        yield scenario["name"], ResourceValidationConfig(**kwargs)


def flatten(values):
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update({f"{key}__{subkey}": array for subkey, array in value.items()})
        else:
            result[key] = value
    return result


def save_batch(folder, run):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "episodes.json", {"metadata": run["metadata"],
                                          "episodes": run["summaries"]})
    np.savez_compressed(folder / "dynamics.npz", **flatten(run["dynamics"]))
    for seed, data in run.get("representatives", {}).items():
        arrays = {}
        for key, value in data.items():
            if isinstance(value, dict):
                if key == "methods":
                    for method, logs in value.items():
                        arrays.update({f"{method}__{name}": array for name, array in logs.items()
                                       if isinstance(array, np.ndarray)})
                else:
                    arrays.update({f"{key}__{name}": array for name, array in value.items()
                                   if isinstance(array, np.ndarray)})
            elif isinstance(value, np.ndarray):
                arrays[key] = value
        np.savez_compressed(folder / f"representative_{seed}.npz", **arrays)


def freeze(output, design):
    receipt = output / "precommitment.json"
    if receipt.exists():
        raise ValueError("Preserve the existing precommitment; use a fresh output directory.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    write_json(receipt, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol_sha256": sha(DESIGN), "source_sha256": source_hashes(),
                        "git_commit": commit, "design": design,
                        "versions": {"python": platform.python_version(),
                                     "numpy": np.__version__, "scipy": scipy.__version__}})
    print(f"Frozen design and implementation: {sha(receipt)}", flush=True)


def run_stage(output, design, stage):
    spec = design["development_seeds" if stage == "pilot" else "validation_seeds"]
    seeds = list(range(spec["first"], spec["last"] + 1))
    destination = output / "development" if stage == "pilot" else output
    if stage == "validation":
        receipt = json.loads((output / "precommitment.json").read_text(encoding="utf-8"))
        if receipt["protocol_sha256"] != sha(DESIGN) or receipt["source_sha256"] != source_hashes():
            raise ValueError("Design or simulation source changed after freeze.")
    for name, config in configs(design):
        folder = destination / name
        if (folder / "episodes.json").exists():
            raise ValueError(f"Refusing to overwrite completed episodes: {folder}")
        print(f"Starting {stage}: {name}, n={len(seeds)}, config={asdict(config)}", flush=True)
        started = time.perf_counter()
        def progress(event):
            if event["round"] % 65536 == 0 or event["round"] == event["T"]:
                print(f"{name}: {event['round']}/{event['T']} rounds, "
                      f"{event['elapsed_seconds']:.1f} seconds", flush=True)
        run = run_resource_validation_batch(
            config, seeds, sample_stride=design["fixed_settings"]["sample_stride"],
            representative_seeds=(() if stage == "pilot" else design["representative_full_paths"]),
            progress_callback=progress)
        save_batch(folder, run)
        taus = [row["switch_round"] for row in run["summaries"] if row["switch_round"] is not None]
        print(f"Completed {name}: switches={len(taus)}/{len(seeds)}, "
              f"median={float(np.median(taus)) if taus else None}, "
              f"seconds={time.perf_counter()-started:.1f}", flush=True)


def metric(rows, method, key):
    return np.array([row["methods"][method][key] for row in rows], dtype=float)


def analyze(output, design):
    from lowdim_games.paired_inference import paired_difference_summary, holm_adjust
    overview = {}
    expected = list(range(design["validation_seeds"]["first"], design["validation_seeds"]["last"]+1))
    pairs = [("terminal_delta", "shared_past_hull"),
             ("terminal_delta", "block_safe"), ("terminal_delta", "lag_response"),
             ("terminal_delta", "last_window")]
    for name, config in configs(design):
        data = json.loads((output / name / "episodes.json").read_text(encoding="utf-8"))
        rows = data["episodes"]
        if [row["seed"] for row in rows] != expected or data["metadata"]["config"] != asdict(config):
            raise ValueError(f"Completeness or fixed-setting mismatch: {name}")
        # These keys are full-target distances, not the 2D lower surrogate.
        method_stats = {method: {key: {"mean": float(metric(rows, method, key).mean()),
                                              "std": float(metric(rows, method, key).std(ddof=1))}
                                 for key in ("terminal_delta", "pre_mean_delta",
                                             "mean_unserved_fraction", "mean_activation_cost", "mean_idle_cost",
                                             "mean_weighted_cost")}
                        for method in METHODS}
        diffs = {}
        for key, comparator in pairs:
            values = metric(rows, "one_switch", key) - metric(rows, comparator, key)
            diffs[f"{key}__{comparator}"] = paired_difference_summary(
                values, bootstrap_resamples=design["inference"]["robustness_resamples"],
                rng=design["inference"]["inference_seed"], family_size=4)
            summary = diffs[f"{key}__{comparator}"]
            lo = (metric(rows, "one_switch", "terminal_projection_lower") -
                  metric(rows, comparator, "terminal_projection_upper"))
            hi = (metric(rows, "one_switch", "terminal_projection_upper") -
                  metric(rows, comparator, "terminal_projection_lower"))
            critical = float(scipy.stats.t.ppf(1-.05/(2*4), len(rows)-1))
            summary["numerical_mean_difference_enclosure"] = [float(lo.mean()), float(hi.mean())]
            summary["numeric_and_sampling_ci_family"] = [
                float(lo.mean()-critical*lo.std(ddof=1)/np.sqrt(len(rows))),
                float(hi.mean()+critical*hi.std(ddof=1)/np.sqrt(len(rows)))]
        tau = np.array([row["switch_round"] for row in rows if row["switch_round"] is not None])
        ci = scipy.stats.binomtest(len(tau), len(rows)).proportion_ci(.95, method="exact")
        overview[name] = {"n": len(rows), "methods": method_stats, "paired_differences": diffs,
                          "deterministic_prephase_difference_vs_safe": float(
                              (metric(rows, "one_switch", "pre_mean_delta")-
                               metric(rows, "block_safe", "pre_mean_delta")).mean()),
                          "switch_count": len(tau), "switch_fraction": len(tau)/len(rows),
                          "switch_fraction_ci_95": [float(ci.low), float(ci.high)],
                          "switch_quantiles_05_50_95": np.quantile(tau, [.05,.5,.95]).tolist() if len(tau) else None,
                          "mean_safe_rounds": float(np.mean([row["safe_rounds"] for row in rows])),
                          "maximum_terminal_support_gap": max(row["methods"][m]["terminal_support_gap"]
                                                               for row in rows for m in METHODS),
                          "maximum_terminal_feasibility_error": max(
                              row["methods"][m]["terminal_feasibility_error"] for row in rows for m in METHODS),
                          "all_safe_tail_certificates_hold": all(
                              row.get("safe_tail_certificate_lhs",0) <= row.get("safe_tail_budget",0)+2e-7
                              for row in rows)}
    primary = overview["primary"]["paired_differences"]
    adjusted = holm_adjust([primary[key]["p_value"] for key in primary])
    for key, p in zip(primary, adjusted):
        primary[key]["holm_p_value"] = float(p)
        primary[key]["significant_at_family_0_05"] = bool(p < .05)
        low, high = primary[key]["numeric_and_sampling_ci_family"]
        primary[key]["direction_robust_to_numerical_uncertainty"] = bool(low > 0 or high < 0)
    write_json(output / "analysis.json", {"protocol_sha256": sha(DESIGN), "scenarios": overview,
               "primary_family": list(primary), "primary_scenario": "primary",
               "inference_scope": "Independent episodes of this fixed constructed generator only.",
               "interactive_paths_and_realized_targets_can_differ": True,
               "numerical_enclosures_are_distinct_from_statistical_intervals": True})
    print(json.dumps(overview["primary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "freeze", "validation", "analyze"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/resource_validation")
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if args.stage == "freeze":
        freeze(args.output, design)
    elif args.stage in ("pilot", "validation"):
        run_stage(args.output, design, args.stage)
    else:
        analyze(args.output, design)


if __name__ == "__main__":
    main()
