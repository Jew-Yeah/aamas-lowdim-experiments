"""Replay switching studies, including every causal comparator and full trajectory.

The original CAGE analysis remains frozen. The new real-data tracking game uses
a separately proved lag safe base. A constructed diagnostic retains the original
block-safe routine and budget. No threshold, window or scenario is tuned here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lowdim_games.switching_tracking import run_tracking_path
from lowdim_games.switching_stress import run_default_budget_stress

NAMES = {"one_switch": "One-switch", "shared_past_hull": "Fast only",
         "lag_safe": "Lag safe only", "last_window": "Window (W=16)",
         "block_safe": "Block safe only"}
COLORS = {"one_switch": "#00796b", "shared_past_hull": "#c62828",
          "lag_safe": "#7b1fa2", "last_window": "#ef8c00", "block_safe": "#1965a6"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def load_nyc(directory):
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    dates, counts = [], []
    for year in (2021, 2022):
        file = directory / provenance["years"][str(year)]["csv"]
        if sha(file) != provenance["years"][str(year)]["csv_sha256"]:
            raise ValueError(f"Source aggregate checksum mismatch: {file}")
        with file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                dates.append(row["date"])
                counts.append([int(row[name]) for name in provenance["columns"][1:]])
    counts = np.asarray(counts, dtype=np.int64)
    pools = np.column_stack((counts[:, 1], counts[:, 3], counts[:, [0, 2, 4]].sum(axis=1)))
    total = pools.sum(axis=1)
    shares = np.full(pools.shape, 1 / 3, dtype=float)
    np.divide(pools, total[:, None], out=shares, where=total[:, None] > 0)
    return np.asarray(dates), counts, shares, provenance


def save_run(directory, run, *, dates=None):
    directory.mkdir(parents=True, exist_ok=True)
    for name, trajectory in run["trajectories"].items():
        arrays = {key: value for key, value in trajectory.items()
                  if isinstance(value, np.ndarray) and value.dtype != object}
        if dates is not None:
            arrays["dates"] = dates
        np.savez_compressed(directory / f"{name}.npz", **arrays)
    if "opponent_actions" in run:
        np.savez_compressed(directory / "game_path.npz", path=run["opponent_actions"],
                            tensor=run["tensor"], dates=dates)
    write_json(directory / "summary.json", {"metadata": run["metadata"], "methods": run["summary"]})


def setup_axes(axes, tau, horizon, *, quiet=None):
    for ax in axes.flat:
        ax.grid(alpha=.2, linewidth=.5)
        ax.set_xlim(1, horizon)
        ax.axvline(tau, color="#333333", linestyle=":", linewidth=1)
        ax.axvspan(tau + 1, horizon, color=COLORS["one_switch"], alpha=.055)
        if quiet is not None:
            ax.axvline(quiet, color="#999999", linestyle="--", linewidth=.8)
        ax.set_xlabel("Decision round, t")
        ax.tick_params(labelsize=7)


def plot_run(run, output, kind):
    tracking = kind == "nyc"
    traces, methods = run["trajectories"], run["summary"]
    master = traces["one_switch"]
    rounds = master["t"]
    tau = methods["one_switch"]["switch_round"]
    assert tau is not None, "A switch figure requires an observed crossing."
    G = run["metadata"]["master_G_T" if tracking else "switch_threshold"]
    Ekey, dkey = ("E", "distances") if tracking else ("cumulative_residual", "distance")
    mismatch = {name: (np.linalg.norm(values["payoffs"], axis=1) if tracking
                      else values["absolute_error"]) for name, values in traces.items()}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45), constrained_layout=True)
    setup_axes(axes, tau, len(rounds), quiet=None if tracking else run["metadata"]["quiet_rounds"])
    ax = axes[0, 0]
    ax.plot(rounds, master[Ekey], color=COLORS["one_switch"], label="One-switch monitored E(t)")
    ax.plot(rounds, traces["shared_past_hull"][Ekey], color=COLORS["shared_past_hull"],
            linestyle="--", label="Fast-only counterfactual E(t)")
    ax.axhline(G, color="black", linewidth=.8, label=f"Certified budget G = {G:,.2f}")
    ax.plot(tau, master[Ekey][tau - 1], "o", color=COLORS["one_switch"], markersize=3)
    ax.set_title(f"(a) Budget crossing at t = {tau:,}")
    ax.set_ylabel("Cumulative fast residual, E(t)")
    ax.legend(loc="best")
    ax = axes[0, 1]
    for name in NAMES:
        vals = traces[name][dkey]
        display_values = np.maximum(vals, 1e-5) if tracking else np.ma.masked_less_equal(vals, 0)
        ax.plot(rounds, display_values, label=NAMES[name], color=COLORS[name],
                linewidth=1.05 if name == "one_switch" else .85,
                linestyle="-" if name in ("one_switch", "shared_past_hull") else "--")
    ax.set_yscale("log")
    ax.set_title("(b) Distance to the full strict target")
    ax.set_ylabel(r"Average-payoff distance, $\delta_t$")
    ax.legend(loc="upper right", ncol=1, fontsize=6.3)
    if tracking:
        ax.text(.02, .03, "Display floor: 1e-5", transform=ax.transAxes, fontsize=6.3)
    comparators = ("shared_past_hull", "last_window" if tracking else "block_safe")
    for ax, comparator, tag in zip(axes[1], comparators, ("c", "d")):
        differences = np.cumsum(mismatch["one_switch"] - mismatch[comparator])
        ax.axhline(0, color="black", linewidth=.7, linestyle="--")
        ax.plot(rounds, differences, color=COLORS["one_switch"], linewidth=1.25)
        ax.set_title(f"({tag}) One-switch - {NAMES[comparator]}")
        ax.set_ylabel("Cumulative daily mismatch difference" if tracking
                      else "Cumulative absolute\nimbalance difference")
        ax.text(.98, .08, f"Final difference: {differences[-1]:+,.2f}",
                transform=ax.transAxes, ha="right", fontsize=7)
    label = ("NYC request-share tracking, 2021-2022 | certified lag safe base, G = 1"
             if tracking else "Constructed resource-balance stress | original block-safe routine and budget")
    fig.suptitle(label + "\nDotted line: crossing round; safe mode starts next round. "
                 "Negative differences favor one-switch.", fontsize=8.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), metadata={"Title": label, "Author": ""})
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/switching")
    parser.add_argument("--replot", action="store_true", help="Recompute deterministic traces and export figures.")
    args = parser.parse_args()
    output = args.output
    dates, counts, shares, provenance = load_nyc(ROOT / "data/switching_nyc")
    protocol = {
        "schema_version": 1, "study": "post-review full switching trajectory audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Exploratory scenario additions after reviewing the non-switching CAGE experiment; no confirmatory holdout claim.",
        "tracking_windows": [16], "tracking_years": [2021, 2022], "tracking_primary": "entire chronological 2021-2022 trace",
        "tracking_safe_base": "Exact lag telescoping, B0(h)=1 for h>0; G=1 known before each run.",
        "stress": {"T": 16384, "quiet_rounds": 128, "window": 16, "budget": "original 6*T^(3/4)",
                   "payoff_equivalent_novel_labels": True},
        "inference": "Descriptive deterministic traces. No iid-day bootstrap, p-values, or confidence bands.",
        "source": provenance,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in
                          [Path(__file__), ROOT / "src/lowdim_games/learners.py",
                           ROOT / "src/lowdim_games/geometry.py", ROOT / "src/lowdim_games/switching_tracking.py",
                           ROOT / "src/lowdim_games/switching_stress.py"]},
    }
    write_json(output / "protocol.json", protocol)
    all_summaries = {}
    primary = None
    for title, mask in [("nyc_2021_2022", np.ones(len(dates), dtype=bool)),
                        ("nyc_2021", np.char.startswith(dates, "2021")),
                        ("nyc_2022", np.char.startswith(dates, "2022"))]:
        run = run_tracking_path(shares[mask], window=16)
        run["metadata"].update({"dates": [str(dates[mask][0]), str(dates[mask][-1])],
                                "zero_count_days": int((counts[mask].sum(axis=1) == 0).sum()),
                                "data_origin": "registered NYC Hazard request shares, modeled allocation objective"})
        tau = run["summary"]["one_switch"]["switch_round"]
        run["summary"]["one_switch"]["switch_date"] = None if tau is None else str(dates[mask][tau - 1])
        save_run(output / title, run, dates=dates[mask])
        all_summaries[title] = {"metadata": run["metadata"], "methods": run["summary"]}
        print(title, json.dumps(run["summary"]["one_switch"]), flush=True)
        if title == "nyc_2021_2022":
            primary = run
    stress = run_default_budget_stress(T=16384, quiet_rounds=128, window=16)
    save_run(output / "original_budget_stress", stress)
    all_summaries["original_budget_stress"] = {"metadata": stress["metadata"], "methods": stress["summary"]}
    print("original_budget_stress", json.dumps(stress["summary"]["one_switch"]), flush=True)
    write_json(output / "analysis.json", all_summaries)
    plot_run(primary, output / "figures/switching_nyc_dynamics", "nyc")
    plot_run(stress, output / "figures/switching_original_budget", "stress")
    members = {str(p.relative_to(output)).replace("\\", "/"): sha(p)
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"
               and p.suffix in {".json", ".npz", ".pdf", ".png"}}
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "scope": "Programmatic results and figures; prose reports are tracked by package manifests.",
        "files_sha256": members,
    })
    print("Completed deterministic replay and both English dynamics figures.", flush=True)


if __name__ == "__main__":
    main()
