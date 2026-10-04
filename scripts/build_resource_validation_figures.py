"""English scientific figures for the frozen vector resource validation.

Whole-episode bootstrap intervals are pointwise and descriptive. Full-target
numerical enclosures are retained explicitly; no surrogate is called exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lowdim_games.paired_inference import bootstrap_mean_band

METHODS = ("one_switch", "shared_past_hull", "block_safe", "lag_response",
           "last_window", "uniform", "request_trigger", "fast_recent_saddle")
NAMES = {"one_switch":"One-switch", "shared_past_hull":"Fast: origin saddle",
         "block_safe":"Block safe", "lag_response":"Lag response",
         "last_window":"Window (W=16)", "uniform":"Uniform",
         "request_trigger":"Request trigger", "fast_recent_saddle":"Fast: recent saddle"}
SHORT = ["One-\nswitch", "Fast\norigin", "Block\nsafe", "Lag", "Window\n16",
         "Fixed\np=0.5", "Request\ntrigger", "Fast\nrecent"]
COLORS = {"one_switch":"#00796b", "shared_past_hull":"#b71c1c",
          "block_safe":"#1965a6", "lag_response":"#7b1fa2",
          "last_window":"#e67e00", "uniform":"#777777",
          "request_trigger":"#333333", "fast_recent_saddle":"#497fb1"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_figures(output):
    design = load_json(ROOT / "docs/resource_validation_protocol.json")
    primary = load_json(output / "primary/episodes.json")
    rows = primary["episodes"]
    data = np.load(output / "primary/dynamics.npz")
    t = data["t"]
    T, H = design["fixed_settings"]["T"], design["fixed_settings"]["change_round"]
    G = primary["metadata"]["G_T"]
    inference = design["inference"]
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":8, "axes.titlesize":9, "axes.labelsize":8,
                         "legend.fontsize":6.2, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})

    def band(values):
        return bootstrap_mean_band(values, n_resamples=inference["curve_resamples"],
                                   rng=inference["curve_bootstrap_seed"])

    def enclosed_curve(ax, lo, hi, label, color, style="-"):
        lm, ll, _ = band(lo)
        hm, _, hh = band(hi)
        ax.plot(t/1000, (lm+hm)/2, label=label, color=color, linewidth=1.1, linestyle=style)
        # Outer band includes episode sampling plus the oracle enclosure.
        ax.fill_between(t/1000, ll, hh, color=color, alpha=.10, linewidth=0)
        # A second, darker band shows numerical uncertainty alone.
        ax.fill_between(t/1000, lm, hm, color=color, alpha=.25, linewidth=0)

    fig, axes = plt.subplots(2,2,figsize=(8,5.1),constrained_layout=True)
    for key, label, color in (("master_E","One-switch",COLORS["one_switch"]),
                              ("fast_E","Fast: origin saddle",COLORS["shared_past_hull"])):
        enclosed_curve(axes[0,0],data[key]/G,data[key]/G,label,color)
    axes[0,0].axhline(1,color="black",linestyle="--",linewidth=.7,label="Certified budget")
    axes[0,0].set(title="(a) Monitored residual and switching budget",ylabel=r"$E_t/G_T$")
    axes[0,0].legend()
    for method in METHODS:
        enclosed_curve(axes[0,1],data[f"{method}__delta_lower"],data[f"{method}__delta_upper"],
                       NAMES[method],COLORS[method],"--" if method=="fast_recent_saddle" else "-")
    axes[0,1].set(title="(b) Full realized-target distance",ylabel=r"Mean enclosure midpoint for $\delta_t$")
    axes[0,1].set_yscale("symlog",linthresh=1e-5)
    axes[0,1].set_yticks([0,1e-5,1e-3,1e-1,.5],labels=["0",r"$10^{-5}$",r"$10^{-3}$","0.1","0.5"])
    handles, labels = axes[0,1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center")
    def difference(ax, comparator, label, color, style="-"):
        lo=data["one_switch__delta_lower"]-data[f"{comparator}__delta_upper"]
        hi=data["one_switch__delta_upper"]-data[f"{comparator}__delta_lower"]
        enclosed_curve(ax,lo,hi,label,color,style)
    difference(axes[1,0],"shared_past_hull","One-switch minus fast (origin)",COLORS["one_switch"])
    axes[1,0].set(title="(c) Paired difference against continued fast",ylabel=r"Difference in $\delta_t$")
    for comparator, style in (("block_safe","-"),("last_window","-"),("fast_recent_saddle","--")):
        difference(axes[1,1],comparator,"Minus "+NAMES[comparator],COLORS[comparator],style)
    axes[1,1].set(title="(d) Paired differences against other controls",ylabel=r"Difference in $\delta_t$")
    axes[1,1].legend()
    for ax in axes[1]:
        ax.axhline(0,color="black",linestyle="--",linewidth=.7)
    for ax in axes.flat:
        ax.axvline(H/1000,color="#333333",linestyle=":",linewidth=.8)
        ax.set(xlabel="Decision round (thousands)",xlim=(0,T/1000))
        ax.grid(alpha=.18)
    for ext in ("pdf","png"):
        fig.savefig(folder/f"resource_switching_dynamics.{ext}",dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(8,5.1),constrained_layout=True)
    x=np.arange(len(METHODS))
    for ax,key,title,ylabel in ((axes[0,0],"terminal_delta","(a) Terminal full-target distance",r"Mean $\delta_T$"),
                              (axes[0,1],"mean_unserved_fraction","(b) Physical unmet demand","Mean unserved request units / round")):
        means,lows,highs,membership=[],[],[],[]
        for method in METHODS:
            v=np.array([row["methods"][method][key] for row in rows])[:,None]
            inside = key == "terminal_delta" and all(
                row["methods"][method]["terminal_delta_upper"] == 0 for row in rows)
            membership.append(inside)
            if inside:
                # The analytical full-target membership inequality supplies zero.
                # Preserve the small floating-point KKT displacement in raw data.
                v = np.zeros_like(v)
            m,lo,hi=band(v)
            means.append(m[0]);lows.append(lo[0]);highs.append(hi[0])
        means=np.array(means)
        ax.bar(x,means,color=[COLORS[m] for m in METHODS],alpha=.82,width=.72)
        ax.errorbar(x,means,yerr=np.maximum(0,np.array([means-np.array(lows),np.array(highs)-means])),
                    fmt="none",color="black",linewidth=.8,capsize=2)
        if key == "terminal_delta":
            for index, value in enumerate(means):
                ax.annotate("0\N{DAGGER}" if membership[index] else f"{value:.4g}",
                            (index, value), xytext=(0,4), textcoords="offset points",
                            ha="center", fontsize=6.2)
        ax.set(title=title,ylabel=ylabel,xticks=x,xticklabels=SHORT)
        ax.tick_params(axis="x",labelsize=7)
        ax.grid(axis="y",alpha=.18)
    idle=np.array([np.mean([row["methods"][m]["mean_idle_cost"] for row in rows]) for m in METHODS])
    busy=np.array([np.mean([row["methods"][m]["mean_loaded_cost"] for row in rows]) for m in METHODS])
    axes[1,0].bar(x,idle,color=[COLORS[m] for m in METHODS],alpha=.82,width=.72,label="Idle reservation cost")
    axes[1,0].bar(x,busy,bottom=idle,color=[COLORS[m] for m in METHODS],alpha=.45,
                  hatch="///",width=.72,label="Loaded overhead")
    axes[1,0].set(title="(c) Modeled provisioning cost",ylabel="Mean cost units / round",xticks=x,xticklabels=SHORT)
    axes[1,0].tick_params(axis="x",labelsize=7)
    axes[1,0].legend()
    axes[1,0].grid(axis="y",alpha=.18)
    scenario_names={"primary":"Reactive: memory 1","memory16":"Reactive: memory 16",
                    "exogenous":"Exogenous arrivals","stationary":"Stationary control"}
    for name,color in zip(scenario_names,("#00796b","#7b1fa2","#1965a6","#777777")):
        sr=load_json(output/name/"episodes.json")["episodes"]
        tau=np.array([r["switch_round"] if r["switch_round"] is not None else np.inf for r in sr])
        cdf=(tau[:,None]<=t).mean(axis=0)
        axes[1,1].step(t/1000,cdf,where="post",color=color,linewidth=1.2,label=scenario_names[name])
    axes[1,1].axvline(H/1000,color="#333333",linestyle=":",linewidth=.8)
    axes[1,1].set(title="(d) Switching probability by scenario",xlabel="Decision round (thousands)",
                  ylabel="Fraction of all episodes crossed",xlim=(0,T/1000),ylim=(-.02,1.05))
    axes[1,1].legend(loc="upper left")
    axes[1,1].grid(alpha=.18)
    for ext in ("pdf","png"):
        fig.savefig(folder/f"resource_outcomes_and_sensitivity.{ext}",dpi=220)
    plt.close(fig)
    captions={
        "resource_switching_dynamics":
            "Primary reactive scenario, 256 complete paired episodes. The dotted vertical line marks H=98304; "
            "the changed environment starts on H+1. Master and fast-only have their own closed-loop paths; "
            "the master's E freezes after crossing. Curves use full-target certified enclosures, with their "
            "mean midpoint shown as a line. Dark fill is numerical enclosure; light outer fill includes "
            "95% pointwise whole-episode bootstrap uncertainty. These are not simultaneous bands. "
            "Negative paired differences favor one-switch. The symlog distance axis includes exact zero. "
            "The target expands when the first request is observed: a sharp drop near H is partly a "
            "change of the realized target, and zero distance does not imply zero unmet demand. "
            "The two valid fast saddle tie choices are distinct controls.",
        "resource_outcomes_and_sensitivity":
            "Panels a-c: primary scenario, all eight methods, 256 episodes. Terminal distances use "
            "full-target KKT evaluations with numerical residual checks. A zero marked with a dagger "
            "denotes full-target membership from the analytical endpoint inequalities in every episode; "
            "raw KKT roundoff is retained in the report, without ranking those small differences. "
            "Error bars in a,b are descriptive "
            "95% whole-episode bootstrap mean intervals; physical outcomes are not the same metric as "
            "target distance. Modeled provisioning cost splits idle reservation waste and loaded overhead. "
            "Panel d retains all 256 episodes per scenario in the denominator, including no switches; "
            "a crossed round is still fast, with safe mode starting next round. Targets can differ across "
            "interactive methods. The model is constructed, not observed operating data."
    }
    (folder/"captions.json").write_text(json.dumps(captions,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("Created two PDF/PNG figure pairs and explicit English captions.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/resource_validation")
    make_figures(parser.parse_args().output)
