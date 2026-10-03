"""Exportable figures for simulator-calibrated policy games."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "one_switch": "One-switch",
    "shared_past_hull": "Shared past-hull",
    "block_safe": "Block safe",
    "uniform": "Uniform mixture",
    "historical_best": "Initial-mode response",
    "last_window": "Previous 16 rounds",
    "hedge": "Hedge",
    "fixed_monitor": "Fixed monitor",
    "fixed_react_remove": "Fixed reactive removal",
    "fixed_react_restore": "Fixed reactive restore",
    "fixed_decoy_cycle": "Fixed decoy cycle",
    "fixed_critical_restore": "Fixed critical restore",
    "fixed_decoy_react": "Fixed decoy + reaction",
}


def _save(fig, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def plot_cage_calibration(calibration, output_dir):
    """Native scalar loss is the sum of the four recorded reward components."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    matrices = [calibration.train_mean.sum(axis=-1), calibration.test_mean.sum(axis=-1)]
    maximum = max(float(x.max()) for x in matrices)
    for ax, matrix, label in zip(axes, matrices, ("Calibration episodes", "Held-out episodes")):
        artist = ax.imshow(matrix, vmin=0, vmax=maximum, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(calibration.red_names)), calibration.red_names)
        ax.set_yticks(range(len(calibration.blue_names)),
                      [name.replace("_", " ") for name in calibration.blue_names])
        ax.set_title(label)
        ax.set_xlabel("Attacking policy")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > .6 * maximum else "black", fontsize=9)
        fig.colorbar(artist, ax=ax, label="Mean native loss / simulator step", shrink=.8)
    fig.suptitle("CAGE 2: independently estimated policy payoff tables")
    _save(fig, output_dir, "cage_calibration")


def plot_policy_comparison(manifest, output_dir):
    name = manifest["experiment"]
    title = f"CAGE 2: {manifest['scenario']}, T={manifest['horizon']}, seed={manifest['seed']}"
    summaries = manifest["summaries"]
    components = len(manifest["metric_names"])
    labels = [LABELS.get(x["method"], x["method"]) for x in summaries]
    native_losses = [sum(x["mean_test_metrics"].values()) for x in summaries]
    # The recorded weights are equal across four native reward components.
    # Weighted-loss intervals can therefore be rescaled to the official total.
    equal_weights = np.allclose(manifest["weights"], np.full(components, 1 / components))
    errors = np.zeros((2, len(summaries)))
    if equal_weights:
        for i, summary in enumerate(summaries):
            interval = summary.get("heldout_weighted_loss_ci95", {})
            if interval.get("lower") is not None:
                errors[:, i] = [max(0, native_losses[i] - components * interval["lower"]),
                                max(0, components * interval["upper"] - native_losses[i])]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(np.arange(len(summaries)), native_losses, yerr=errors, capsize=3,
           color=["#b34a2d" if x["method"] == "one_switch" else "#397c9d" for x in summaries])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Held-out native loss / simulator step (lower is better)")
    ax.set_title(title + "\n95% intervals across held-out simulator seeds, conditional on calibration")
    ax.grid(axis="y", alpha=.2)
    _save(fig, output_dir, name + "__loss")

    methods = [x for x in manifest["methods"] if not x.startswith("fixed_")]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for method in methods:
        points = [x for x in manifest["curves"] if x["method"] == method]
        rounds = [x["round"] for x in points]
        label = LABELS.get(method, method)
        style = "--" if method == "one_switch" else "-"
        axes[0].plot(rounds, [sum(x["mean_test_metrics"]) for x in points],
                     label=label, linestyle=style, marker=".")
        axes[1].plot(rounds, [x["delta_realized_hull"] for x in points],
                     label=label, linestyle=style, marker=".")
    axes[0].set_ylabel("Cumulative mean held-out native loss / step")
    axes[1].set_ylabel("Distance to the full realized-hull target\n(calibration game)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Meta-round (reset episode)")
        ax.grid(alpha=.2)
        for boundary in manifest["phase_boundaries"][1:-1]:
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, output_dir, name + "__curves")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for method in methods:
        trace_file = Path(output_dir) / f"{name}__{method}.npz"
        with np.load(trace_file, allow_pickle=False) as trace:
            losses = trace["test_payoffs"].sum(axis=1)
        window = min(32, len(losses))
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(losses) + 1), smooth,
                label=LABELS.get(method, method),
                linestyle="--" if method == "one_switch" else "-")
    for boundary in manifest["phase_boundaries"][1:-1]:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Meta-round (reset episode)")
    ax.set_ylabel("Held-out native loss / step, trailing 32 rounds")
    ax.set_title(title)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, output_dir, name + "__adaptation")
