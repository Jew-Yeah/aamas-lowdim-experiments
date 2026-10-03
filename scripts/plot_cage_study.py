"""Export publication-ready figures from the completed extended CAGE report."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_cage_report import METHOD_LABELS


def label(method):
    return METHOD_LABELS[method][0]


def export(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/cage_study"))
    args = parser.parse_args()
    output = args.report
    report = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    primary = report["primary"]
    contrasts = primary["contrasts"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    panels = [(["fixed_monitor", "fixed_react_remove", "fixed_react_restore"],
               "Official Blue agents"),
              (["hedge", "last_window", "fixed_decoy_react"],
               "Adaptive and custom references")]
    for ax, (references, title) in zip(axes, panels):
        for index, reference in enumerate(references):
            row = contrasts[reference]
            color = "#157f60" if row["upper"] < 0 else "#bc5a28" if row["lower"] > 0 else "#607080"
            ax.plot([row["lower"], row["upper"]], [index, index], color=color, lw=2)
            ax.scatter(row["estimate"], index, color=color, s=36, zorder=3)
        ax.axvline(0, color="#444444", lw=1, ls="--")
        ax.set_yticks(range(len(references)), [label(r) for r in references])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Our loss minus reference (native loss / step)")
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Primary CAGE comparison: paired simultaneous intervals")
    fig.text(.5, .01, "400 held-out seed tables × 50 common paths; 10,000 crossed bootstrap draws. "
             "Bonferroni family level 95%; separate panel scales.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .93))
    export(fig, output, "primary_comparisons")

    scores = sorted(primary["scores"], key=lambda r: r["mean_native_loss"])
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(scores))
    security = [sum(r["mean_components"][m] for m in
                    ("host_compromise", "server_compromise", "operational_disruption")) for r in scores]
    restore = [r["mean_components"]["restore"] for r in scores]
    ax.barh(y, security, color="#dc8b56", label="Security loss")
    ax.barh(y, restore, left=security, color="#4b80bd", label="Restoration cost")
    for index, row in enumerate(scores):
        ci = row["native_loss_ci95_descriptive"]
        ax.plot([ci["lower"], ci["upper"]], [index, index], color="#222222", lw=1.1)
    ax.set_yticks(y, [label(r["method"]) for r in scores])
    for tick, row in zip(ax.get_yticklabels(), scores):
        if row["method"] == "one_switch":
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Native loss per simulator step (lower is better)")
    ax.set_title("Primary curriculum: all 13 methods and loss components")
    ax.grid(axis="x", alpha=.15)
    ax.legend(loc="lower right")
    fig.text(.5, .01, "Total-loss whiskers are descriptive pointwise 95% intervals; "
             "primary pairwise decisions use simultaneous intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    export(fig, output, "primary_losses")

    chosen = ["one_switch", "fixed_react_restore", "hedge", "last_window", "fixed_decoy_react"]
    names = list(report["groups"])
    values = np.array([[next(r["mean_native_loss"] for r in report["groups"][name]["scores"]
                             if r["method"] == method) for method in chosen] for name in names])
    fig, ax = plt.subplots(figsize=(10, 6.3))
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(chosen)), [label(m) for m in chosen], rotation=18, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(chosen)):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i,j] > values.max()*.65 else "black")
    fig.colorbar(im, ax=ax, label="Native loss per simulator step")
    ax.set_title("Primary comparison and exploratory robustness checks")
    fig.tight_layout()
    export(fig, output, "robustness_losses")
    hashes = {str(path.relative_to(output)).replace("\\", "/"):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob("*"))
              if path.is_file() and path.name != "SHA256SUMS.json"}
    (output / "SHA256SUMS.json").write_text(
        json.dumps({"algorithm": "sha256", "files": hashes}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"Exported six PNG/PDF figures to {output}")


if __name__ == "__main__":
    main()
