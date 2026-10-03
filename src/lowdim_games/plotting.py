"""Static, exportable figures from recorded benchmark results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

LABELS = {"one_switch": "One-switch", "shared_past_hull": "Shared past-hull",
          "block_safe": "Block safe", "uniform": "Uniform quotas",
          "historical_share": "Historical shares", "reserve": "Fixed reserve", "last_week": "Previous week"}


def display_title(name):
    nyc = re.fullmatch(r"nyc_M(\d+)_capacity([\d.]+)", name)
    if nyc:
        return f"NYC Forestry: {nyc[1]} profiles, baseline capacity ratio {nyc[2]}"
    synthetic = re.fullmatch(r"synthetic_q(\d+)_seed(\d+)_T(\d+)", name)
    if synthetic:
        return f"Unknown regimes: q={synthetic[1]}, T={synthetic[3]}, seed={synthetic[2]}"
    return name


def plot_comparison(manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in manifest["methods"]:
        points = [x for x in manifest["curves"] if x["method"] == method]
        ax.plot([x["round"] for x in points], [x["delta"] for x in points],
                marker="o", markersize=3, label=LABELS[method],
                linestyle="--" if method == "one_switch" else "-")
    ax.set_xscale("log")
    ax.set_xlabel("Round")
    ax.set_ylabel("Distance to the full realized-hull target")
    ax.set_title(display_title(manifest["experiment"]))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f'{manifest["experiment"]}__target.{extension}', dpi=180)
    plt.close(fig)
    summaries = manifest["summaries"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    labels = [LABELS[x["method"]] for x in summaries]
    for ax, key, label in zip(axes, ["mean_unmet_requests", "mean_resource_cost", "mean_disparity_requests"],
                             ["Mean unmet requests / round", "Mean modeled resource cost", "Mean deficit disparity / round"]):
        ax.bar(np.arange(len(labels)), [x[key] for x in summaries])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(display_title(manifest["experiment"]))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f'{manifest["experiment"]}__service.{extension}', dpi=180)
    plt.close(fig)


def plot_geometry(records, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for q in sorted({x["q_cap"] for x in records}):
        points = sorted([x for x in records if x["q_cap"] == q], key=lambda x: x["horizon"])
        axes[0].plot([x["horizon"] for x in points], [x["V_T"] for x in points], marker="o", label=f"q={q}")
        axes[1].plot([x["horizon"] for x in points], [x["moment_sum"] for x in points], marker="o", label=f"q={q}")
    for ax, ylabel in zip(axes, ["Sum of hull increments V_T", "Critical moment sum"]):
        ax.set_xlabel("Number of points")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"geometry_moments.{extension}", dpi=180)
    plt.close(fig)
