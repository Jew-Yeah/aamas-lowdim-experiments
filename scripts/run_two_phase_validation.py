"""Precommitted paired Monte Carlo validation of original-budget switching.

The design is synthetic and motivated by earlier exploratory results. Freeze
the implementation receipt after tests, then run the untouched validation seeds.
All scenarios, controls, and non-switching episodes are reported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lowdim_games.switching_validation import METHODS, ValidationConfig, run_validation_batch

DESIGN = ROOT / "docs/two_phase_validation_protocol.json"
SOURCES = [Path(__file__), ROOT / "src/lowdim_games/switching_validation.py",
           ROOT / "src/lowdim_games/paired_inference.py", ROOT / "src/lowdim_games/learners.py"]
NAMES = {"one_switch": "One-switch", "shared_past_hull": "Fast only",
         "block_safe": "Block safe only", "lag_response": "Lag response",
         "last_window": "Window (W=16)"}
COLORS = {"one_switch": "#00796b", "shared_past_hull": "#c62828",
          "block_safe": "#1965a6", "lag_response": "#7b1fa2", "last_window": "#ef8c00"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def configs(design):
    fixed = design["fixed_settings"]
    for scenario in design["scenarios"]:
        yield scenario["name"], ValidationConfig(
            T=fixed["T"], change_round=scenario["change_round"],
            pre_r_low=fixed["pre_r_low"], pre_r_high=fixed["pre_r_high"],
            a_min=fixed["a_min"], post_a_low=fixed["post_a_low"], post_a_high=fixed["post_a_high"],
            post_r_low=scenario["post_r_low"], post_r_high=scenario["post_r_high"],
            window=fixed["window"], scenario_id=scenario["id"], base_entropy=fixed["base_entropy"])


def flatten_dynamics(dynamics):
    result = {}
    for name, values in dynamics.items():
        if isinstance(values, dict):
            for key, array in values.items():
                result[f"{name}__{key}"] = array
        else:
            result[name] = values
    return result


def save_batch(directory, run):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "episodes.json", {"metadata": run["metadata"], "episodes": run["summaries"]})
    np.savez_compressed(directory / "dynamics.npz", **flatten_dynamics(run["dynamics"]))
    for seed, data in run["representatives"].items():
        arrays = {f"path__{key}": value for key, value in data["path"].items()
                  if isinstance(value, np.ndarray)}
        for method, logs in data["methods"].items():
            arrays.update({f"{method}__{key}": value for key, value in logs.items()})
        np.savez_compressed(directory / f"representative_{seed}.npz", **arrays)


def source_hashes():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in SOURCES}


def freeze(output, design):
    receipt = output / "precommitment.json"
    if receipt.exists():
        raise ValueError("A precommitment already exists; preserve it and use a separate output folder.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(receipt, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol_sha256": sha(DESIGN), "source_sha256": source_hashes(),
                        "git_commit": commit, "design": design,
                        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                     "scipy": scipy.__version__}})
    print(f"Frozen protocol and checked implementation: {sha(receipt)}", flush=True)


def run_stage(output, design, stage):
    seed_spec = design["development_seeds" if stage == "pilot" else "validation_seeds"]
    seeds = range(seed_spec["first"], seed_spec["last"] + 1)
    destination = output / "development" if stage == "pilot" else output
    if stage == "validation":
        receipt = json.loads((output / "precommitment.json").read_text(encoding="utf-8"))
        if receipt["protocol_sha256"] != sha(DESIGN) or receipt["source_sha256"] != source_hashes():
            raise ValueError("The protocol or implementation changed after precommitment.")
    for name, config in configs(design):
        folder = destination / name
        if (folder / "episodes.json").exists():
            raise ValueError(f"Refusing to overwrite completed episodes: {folder}")
        print(f"Starting {stage}: {name}, n={len(seeds)}, T={config.T}", flush=True)
        started = time.perf_counter()
        run = run_validation_batch(config, seeds, sample_stride=design["fixed_settings"]["sample_stride"],
                                   representative_seeds=(() if stage == "pilot" else design["representative_full_paths"]))
        save_batch(folder, run)
        taus = [row["switch_round"] for row in run["summaries"] if row["switch_round"] is not None]
        print(f"Completed {name}: switches={len(taus)}/{len(seeds)}, "
              f"median crossing={np.median(taus) if taus else None}, seconds={time.perf_counter()-started:.1f}", flush=True)


def metric(episodes, method, key):
    return np.array([row["methods"][method][key] for row in episodes], dtype=float)


def analyze(output, design):
    from lowdim_games.paired_inference import paired_difference_summary, holm_adjust
    all_rows = {}
    overview = {}
    for name, config in configs(design):
        data = json.loads((output / name / "episodes.json").read_text(encoding="utf-8"))
        rows = data["episodes"]
        expected = list(range(design["validation_seeds"]["first"], design["validation_seeds"]["last"] + 1))
        if [row["seed"] for row in rows] != expected or data["metadata"]["config"] != asdict(config):
            raise ValueError(f"Configuration or episode completeness mismatch: {name}")
        all_rows[name] = rows
        tau = np.array([row["switch_round"] for row in rows if row["switch_round"] is not None])
        methods = {}
        for method in METHODS:
            methods[method] = {key: {"mean": float(metric(rows, method, key).mean()),
                                   "std": float(metric(rows, method, key).std(ddof=1))}
                               for key in ("terminal_delta", "pre_mean_delta", "mean_absolute_error",
                                           "pre_terminal_delta", "pre_mean_absolute_error")}
        differences = {}
        for key, comparator in (("pre_mean_delta", "block_safe"),
                                ("terminal_delta", "shared_past_hull"), ("terminal_delta", "block_safe")):
            delta = metric(rows, "one_switch", key) - metric(rows, comparator, key)
            differences[f"{key}__{comparator}"] = paired_difference_summary(
                delta, bootstrap_resamples=design["inference"]["robustness_resamples"],
                rng=design["inference"]["inference_seed"])
        switch_ci = scipy.stats.binomtest(len(tau), len(rows)).proportion_ci(confidence_level=.95, method="exact")
        overview[name] = {"n": len(rows), "methods": methods, "paired_differences": differences,
                          "switch_count": len(tau), "switch_fraction": len(tau) / len(rows),
                          "switch_fraction_ci_95": [float(switch_ci.low), float(switch_ci.high)],
                          "switch_quantiles_05_50_95": np.quantile(tau, [.05, .5, .95]).tolist() if len(tau) else None,
                          "mean_safe_rounds": float(np.mean([row["safe_rounds"] for row in rows])),
                          "all_safe_tail_certificates_hold": all(abs(row.get("safe_tail_signed_sum", 0))
                              <= row.get("safe_tail_budget", 0) + 1e-8 for row in rows)}
    primary = overview["primary"]["paired_differences"]
    keys = list(primary)
    p_adjusted = holm_adjust([primary[key]["p_value"] for key in keys])
    for key, adjusted in zip(keys, p_adjusted):
        primary[key]["holm_p_value"] = float(adjusted)
        primary[key]["significant_at_family_0_05"] = bool(adjusted < .05)
    write_json(output / "analysis.json", {"protocol_sha256": sha(DESIGN), "scenarios": overview,
               "primary_family": keys, "primary_scenario": "primary",
               "inference_scope": design["game"]["scope"],
               "pointwise_intervals_are_not_simultaneous": True})
    print(json.dumps({"primary": overview["primary"]}, indent=2), flush=True)


def make_figures(output, design):
    # Figures resample complete episodes; no resampling of dependent rounds.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lowdim_games.paired_inference import bootstrap_mean_band
    overview = json.loads((output / "analysis.json").read_text(encoding="utf-8"))["scenarios"]
    data = np.load(output / "primary/dynamics.npz")
    t = data["t"]
    H = design["scenarios"][0]["change_round"]
    G = 24576
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 6.5, "pdf.fonttype": 42})
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.0), constrained_layout=True)
    def curve(ax, values, name, color):
        mean, low, high = bootstrap_mean_band(values, n_resamples=2000,
                                              rng=design["inference"]["curve_bootstrap_seed"])
        ax.plot(t, mean, label=name, color=color, linewidth=1)
        ax.fill_between(t, low, high, color=color, alpha=.15, linewidth=0)
    for key, title, color in (("master_E", "One-switch E", COLORS["one_switch"]),
                              ("fast_E", "Fast-only E", COLORS["shared_past_hull"])):
        curve(axes[0,0], data[key]/G, title, color)
    axes[0,0].axhline(1, color="black", linestyle="--", linewidth=.7, label="Certified budget")
    axes[0,0].set(title="(a) Budget crossing", ylabel="Cumulative residual / G")
    axes[0,0].legend()
    for method in METHODS:
        curve(axes[0,1], data[f"{method}__delta"], NAMES[method], COLORS[method])
    axes[0,1].set(title="(b) Full-target distance", ylabel=r"Mean $\delta_t$")
    axes[0,1].set_yscale("log")
    axes[0,1].legend(ncol=2)
    for ax, comparator, letter in zip(axes[1], ("shared_past_hull", "block_safe"), ("c", "d")):
        diff = data["one_switch__delta"] - data[f"{comparator}__delta"]
        curve(ax, diff, "Paired mean difference", COLORS["one_switch"])
        ax.axhline(0, color="black", linestyle="--", linewidth=.7)
        ax.set(title=f"({letter}) One-switch - {NAMES[comparator]}", ylabel=r"Difference in $\delta_t$")
    for ax in axes.flat:
        ax.axvline(H, color="#333333", linestyle=":", linewidth=.8)
        ax.set(xlabel="Decision round, t", xlim=(1, design["fixed_settings"]["T"]))
        ax.grid(alpha=.2)
    fig.suptitle("Independent two-phase resource-allocation model | original block-safe budget\n"
                 "200 paired episodes; pointwise 95% bootstrap intervals; dotted line: environmental change.", fontsize=9)
    fig.savefig(folder / "two_phase_dynamics.pdf", metadata={"Title": "Paired two-phase switching dynamics", "Author": ""})
    fig.savefig(folder / "two_phase_dynamics.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)
    cells = list(overview)
    display = ["Primary", "Late change", "Weaker change", "Stationary"]
    for ax, key, title in zip(axes.flat[:3], ("pre_mean_delta__block_safe",
                                             "terminal_delta__shared_past_hull", "terminal_delta__block_safe"),
                             ("(a) Before change: one-switch - block safe",
                              "(b) Terminal distance: one-switch - fast",
                              "(c) Terminal distance: one-switch - block safe")):
        for y, name in enumerate(cells):
            record = overview[name]["paired_differences"][key]
            mean = record["mean_difference"]
            lo, hi = record["t_ci_95"]
            ax.errorbar(mean, y, xerr=[[max(0,mean-lo)], [max(0,hi-mean)]], fmt="o",
                        color=COLORS["one_switch"], capsize=3, markersize=4)
        ax.axvline(0, color="black", linestyle="--", linewidth=.7)
        ax.set(title=title, yticks=range(len(cells)), yticklabels=display,
               xlabel="Paired mean difference (95% t interval)")
        ax.invert_yaxis()
        ax.grid(alpha=.2, axis="x")
    for name, label, color in zip(cells, display, ("#00796b", "#c62828", "#1965a6", "#7b1fa2")):
        rows = json.loads((output / name / "episodes.json").read_text(encoding="utf-8"))["episodes"]
        taus = np.sort([row["switch_round"] for row in rows if row["switch_round"] is not None])
        # Denominator includes non-switchers, so the curve ends below one if needed.
        if len(taus):
            axes[1,1].step(np.r_[1, taus, design["fixed_settings"]["T"]],
                           np.r_[0, np.arange(1,len(taus)+1)/len(rows), len(taus)/len(rows)],
                           where="post", label=f"{label}: {len(taus)}/{len(rows)}", color=color)
        else:
            axes[1,1].plot([1,design["fixed_settings"]["T"]], [0,0], label=f"{label}: 0/{len(rows)}", color=color)
    axes[1,1].set(title="(d) Switching probability by round", xlabel="Decision round, t",
                  ylabel="Fraction of all episodes switched", ylim=(-.02,1.02))
    axes[1,1].legend()
    axes[1,1].grid(alpha=.2)
    fig.suptitle("Fixed scenario sensitivity | 200 independent paired episodes per scenario\n"
                 "Negative differences favor one-switch. Sensitivity intervals are descriptive; no best-cell selection.", fontsize=9)
    fig.savefig(folder / "two_phase_sensitivity.pdf", metadata={"Title": "Two-phase sensitivity and switching", "Author": ""})
    fig.savefig(folder / "two_phase_sensitivity.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "freeze", "validation", "analyze", "figures"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/two_phase_validation")
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if args.stage == "freeze":
        freeze(args.output, design)
    elif args.stage in ("pilot", "validation"):
        run_stage(args.output, design, args.stage)
    elif args.stage == "analyze":
        analyze(args.output, design)
    else:
        make_figures(args.output, design)


if __name__ == "__main__":
    main()
