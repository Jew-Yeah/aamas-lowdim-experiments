"""Create English, two-column AAMAS figure exports from unchanged public results.

This script changes presentation only. It performs no trajectory simulation,
parameter selection, bootstrap resampling, or geometric projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[1]
METHODS = ("selected_scalar_one_switch", "selected_window", "selected_hedge")
METHOD_LABELS = (
    r"Our one-switch ($W=16$, $\rho=0.25$)",
    r"Window ($W=16$)",
    r"Hedge (rate $\times 4$)",
)
COLORS = ("#2673b8", "#d68820", "#3a946c")
MARKERS = ("o", "s", "^")
COMPONENT_LABELS = ("Host compromise", "Server compromise", "Operational disruption", "Restore cost")
FLOOR = 1e-10

PRIMARY_CAPTION = (
    r"Native simulator cost at $T=512$. (a) Paired mean differences between our "
    r"one-switch method and each baseline. Error bars are the original crossed-bootstrap "
    r"intervals (10,000 draws; 400 held-out simulator seeds and 50 common opponent paths), "
    r"with approximate simultaneous 95\% coverage for the two prespecified contrasts, conditional on "
    r"calibration. (b) Additive decomposition into the four native cost components; bars "
    r"show means without new inferential intervals. Negative differences favor our method. "
    r"Cost is reported per native simulator step, using separately reset 50-step episodes."
)
GEOMETRY_CAPTION = (
    r"Full-target vector error in the fixed normalized calibration game. (a) Prefixes "
    r"of all 50 primary paths ($T=512$), using the full realized-hull target $S(Q_t)$. "
    r"Vertical lines mark curriculum phases; target expansion can also reduce error. "
    r"(b) Separate announced-horizon runs with 20 common paths each; curriculum lengths "
    r"and learning-rate formulas scale with $T$. Lines are medians and bands are empirical "
    r"10--90\% path ranges, not confidence intervals. Lower strips report the share of "
    r"numerical upper distances at or below $10^{-10}$; these values are omitted from log "
    r"axes without adding an epsilon. Numerical checks are not exact-arithmetic proofs, "
    r"and these data do not identify an asymptotic decay order."
)
CUMULATIVE_CAPTION = (
    r"Expected cumulative native simulator cost differences, "
    r"$\Delta C_b(t)=50\sum_{s=1}^{t}(c_s^{(\mathrm{ours})}-c_s^{(b)})$, "
    r"where $c_s^{(m)}$ is method $m$'s mean native per-step cost at round $s$. "
    r"Negative values favor our method. "
    r"Curves use all 50 common paths and 400 held-out simulator-seed tables; bands are "
    r"descriptive pointwise 95\% crossed-bootstrap intervals (2,000 draws), conditional "
    r"on calibration and fixed configurations. Panels use different vertical scales. "
    r"Dotted lines separate the initial mode, "
    r"exploration, and mode-learning phases. The attacker learns against a uniform reference "
    r"defense, generating the same exogenous paths for all compared methods. The Window "
    r"gap is entirely due to the first action and remains constant thereafter."
)
DESCRIPTIONS = {
    "primary_comparison": (
        "Two panels show native per-step cost of our one-switch method minus each baseline. "
        "Against Window the mean difference is positive, about 0.00265; against Hedge it "
        "is negative, about minus 0.06182. Error bars retain the original approximate "
        "simultaneous intervals. The component panel decomposes the same differences into "
        "host compromise, server compromise, operational disruption, and restore cost. "
        "Relative to Hedge, our method reduces disruption and server compromise but "
        "increases host compromise and restore cost. Negative differences favor our method."
    ),
    "calibrated_geometry": (
        "Two upper logarithmic panels compare absolute full-target vector distances "
        "for our one-switch method, Window, and Hedge. The left panel uses all 50 "
        "primary-path prefixes; the right uses 20 common paths at each separately "
        "announced horizon. Medians and empirical path ranges are shown. At horizons "
        "of at least 128, all new one-switch and Window paths have numerical upper "
        "distances at or below the display floor. Hedge retains a positive distance "
        "that decreases with horizon. Lower strips count near-zero paths rather "
        "than replacing their errors with an artificial positive value. The metric "
        "uses the fixed normalized training game and is distinct from native cost differences."
    ),
    "cumulative_cost_difference": (
        "Two panels show expected cumulative native cost of our one-switch method "
        "minus each baseline over 512 policy-selection rounds. Against Window, "
        "the positive difference remains about 67.74 after the first round because "
        "later actions coincide. Against Hedge, the difference becomes negative "
        "and reaches about minus 1582.54. Negative differences indicate lower cost "
        "for our method. Each round corresponds to separately reset 50-step "
        "episodes. Shading shows descriptive pointwise uncertainty, and dotted "
        "vertical lines separate the initial-mode, exploration, and mode-learning "
        "phases. The panels use different vertical scales."
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_inputs(root: Path) -> dict:
    """Require the locked methods and exact original scalar decomposition."""
    report = read_json(root / "analysis.json")
    primary = report["primary"]
    if primary["horizon"] != 512 or primary["path_count"] != 50:
        raise ValueError("The locked primary study must use T=512 and all 50 paths.")
    costs = np.array([primary["methods"][name]["mean_native_loss"] for name in METHODS])
    components = np.array([primary["methods"][name]["mean_components"] for name in METHODS])
    if components.shape != (3, 4) or not np.allclose(components.sum(axis=1), costs, atol=1e-12, rtol=0):
        raise ValueError("Four component costs must sum to the original primary metric.")
    for reference in METHODS[1:]:
        contrast = primary["contrasts"][reference]
        idx = METHODS.index(reference)
        if abs(contrast["estimate"] - (costs[0] - costs[idx])) > 1e-12:
            raise ValueError("Primary contrast direction or estimate does not match costs.")
        if not (contrast["lower"] <= contrast["estimate"] <= contrast["upper"]):
            raise ValueError("Primary confidence interval does not contain its estimate.")
        if contrast["samples"] != 10000 or contrast["episode_seed_count"] != 400 or contrast["path_seed_count"] != 50:
            raise ValueError("Original crossed-bootstrap sampling units changed.")
    geometries = {}
    for name in ("prefix_geometry", "horizon_geometry"):
        with np.load(root / "geometry" / f"{name}.npz", allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        if tuple(data["methods"]) != METHODS:
            raise ValueError("Geometry method order does not match the selected comparison.")
        fields = list(data["fields"])
        rows = data["rows"]
        if not np.all(np.isfinite(rows)):
            raise ValueError("Nonfinite geometry diagnostics.")
        lower = rows[..., fields.index("lower_distance")]
        upper = rows[..., fields.index("distance")]
        success = rows[..., fields.index("success")]
        if not np.all(success == 1) or np.any(lower < 0) or np.any(upper < lower - 1e-14):
            raise ValueError("Failed or inconsistent full-target numerical projection.")
        geometries[name] = data
    with np.load(root / "dynamics" / "plot_data.npz", allow_pickle=False) as archive:
        dynamics = {key: archive[key].copy() for key in archive.files}
    if tuple(dynamics["methods"]) != METHODS:
        raise ValueError("Dynamic method order does not match the selected comparison.")
    cumulative = dynamics["cumulative_difference"]
    bands = dynamics["cumulative_difference_interval"]
    if cumulative.shape != (512, 2) or bands.shape != (2, 512, 2):
        raise ValueError("The complete locked cumulative curves and intervals are required.")
    target = 50 * 512 * (costs[0] - costs[1:])
    if not np.allclose(cumulative[-1], target, rtol=0, atol=1e-8):
        raise ValueError("Cumulative native episode costs must equal 50 times summed per-step costs.")
    if not np.allclose(cumulative[:, 0], cumulative[0, 0], rtol=0, atol=1e-10):
        raise ValueError("The locked Window gap must remain constant after the initial action.")
    return {"report": report, "costs": costs, "components": components, "dynamics": dynamics, **geometries}


def log_summary(distances: np.ndarray, axis: int, floor: float = FLOOR) -> dict:
    """Censor unresolved values without modifying the published distances."""
    values = np.asarray(distances, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    median = np.median(values, axis=axis)
    low, high = np.quantile(values, [.1, .9], axis=axis)
    return {
        "median": np.where(median > floor, median, np.nan),
        "low": np.where(low > floor, low, np.nan),
        "high": np.where(high > floor, high, np.nan),
        "percent_unresolved": 100 * np.mean(values <= floor, axis=axis),
    }


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def primary_figure(inputs: dict, output: Path) -> None:
    primary = inputs["report"]["primary"]
    components = inputs["components"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.155, right=.985, bottom=.23, top=.75, wspace=.70)
    left, right = axes
    for j, reference in enumerate(METHODS[1:]):
        row = primary["contrasts"][reference]
        estimate = row["estimate"]
        color = COLORS[j + 1]
        left.errorbar(estimate, j, xerr=[[estimate-row["lower"]], [row["upper"]-estimate]],
                      fmt=MARKERS[j+1], color=color, markersize=5, capsize=3, linewidth=1.5)
        left.text(.02, .12 if j == 0 else .53,
                  f"{estimate:+.5f}\n[{row['lower']:+.5f}, {row['upper']:+.5f}]",
                  transform=left.transAxes, fontsize=8, color=color, ha="left", va="center")
    left.set_yticks([0, 1], ["vs Window", "vs Hedge"])
    left.set_ylim(-.55, 1.55)
    left.set_xlim(-.091, .014)
    left.axvline(0, color=".35", linewidth=.9, linestyle="--")
    left.set_xticks([-.08, -.04, 0], ["−0.08", "−0.04", "0"])
    left.set_xlabel(r"Mean total cost difference, $J_{\mathrm{ours}}-J_{\mathrm{baseline}}$")
    left.set_title("(a) Primary paired comparisons", loc="left")
    left.grid(axis="x", color=".9", linewidth=.6)

    differences = components[0] - components[1:]
    positions = np.arange(4)
    for j in range(2):
        right.barh(positions + (j-.5)*.28, differences[j], height=.25,
                   color=COLORS[j+1], label=f"Our method − {'Window' if j == 0 else 'Hedge'}")
    right.axvline(0, color=".35", linewidth=.9, linestyle="--")
    right.set_yticks(positions, COMPONENT_LABELS)
    right.invert_yaxis()
    right.set_xlim(-.081, .034)
    right.set_xticks([-.08, -.04, 0, .03], ["−0.08", "−0.04", "0", "+0.03"])
    right.set_xlabel("Mean component cost difference")
    right.set_title("(b) Component cost differences", loc="left")
    right.grid(axis="x", color=".9", linewidth=.6)
    right.set_axisbelow(True)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.99, .92),
               frameon=False, handlelength=1.1)
    fig.text(.50, .98,
             r"Our one-switch ($W=16$, $\rho=0.25$) | Window ($W=16$) | Hedge (rate $\times4$)",
             ha="center", va="top", fontsize=8)
    fig.text(.50, .06, "Negative values favor our method; costs are per native simulator step.",
             ha="center", fontsize=8)
    save(fig, output, "primary_comparison")


def geometry_figure(inputs: dict, output: Path) -> None:
    fig = plt.figure(figsize=(7.0, 3.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.1, 1], hspace=.13, wspace=.27,
                           left=.10, right=.975, bottom=.18, top=.80)
    for panel, name in enumerate(("prefix_geometry", "horizon_geometry")):
        data = inputs[name]
        fields = list(data["fields"])
        distance = data["rows"][..., fields.index("distance")]
        if panel == 0:
            x = data["checkpoints"]
            summary = log_summary(distance, axis=0)
            summary = {key: value.T for key, value in summary.items()}
        else:
            x = data["horizons"]
            summary = log_summary(distance, axis=1)
        top = fig.add_subplot(grid[0, panel])
        bottom = fig.add_subplot(grid[1, panel], sharex=top)
        for method in range(3):
            median = summary["median"][:, method]
            low = summary["low"][:, method]
            high = summary["high"][:, method]
            top.plot(x, median, color=COLORS[method], marker=MARKERS[method],
                     markersize=3, linewidth=1.4, label=METHOD_LABELS[method])
            valid = np.isfinite(low) & np.isfinite(high)
            top.fill_between(x, low, high, where=valid, color=COLORS[method], alpha=.15, linewidth=0)
            bottom.plot(x, summary["percent_unresolved"][:, method], color=COLORS[method],
                        marker=MARKERS[method], markersize=2.5, linewidth=1,
                        linestyle=("-", "--", ":")[method])
        top.set_xscale("log", base=2)
        top.set_yscale("log")
        top.set_ylim(3e-5, .18)
        top.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
        top.tick_params(axis="x", which="both", labelbottom=False)
        top.grid(color=".90", linewidth=.6)
        bottom.set_ylim(-8, 108)
        bottom.set_yticks([0, 100], ["0", "100"])
        bottom.grid(axis="y", color=".90", linewidth=.6)
        ticks = [1, 16, 128, 512] if panel == 0 else [64, 128, 256, 512, 1024, 2048]
        bottom.xaxis.set_major_locator(FixedLocator(ticks))
        bottom.xaxis.set_major_formatter(FuncFormatter(lambda value, _: str(int(value))))
        bottom.tick_params(axis="x", labelsize=8)
        if panel == 0:
            top.set_title("(a) Primary-run prefixes (50 paths)", loc="left")
            top.set_ylabel(r"Full-target vector error, $\delta_t$")
            bottom.set_xlabel("Policy-selection round, t")
            bottom.set_ylabel("At floor (%)")
            for boundary in (128, 384):
                top.axvline(boundary, color=".4", linewidth=.7, linestyle=":")
                bottom.axvline(boundary, color=".4", linewidth=.7, linestyle=":")
        else:
            top.set_title("(b) Separate horizons (20 paths each)", loc="left")
            top.set_ylabel(r"Terminal vector error, $\delta_T$")
            bottom.set_xlabel("Announced horizon, T")
            bottom.set_ylabel("At floor (%)")
        top.set_xlim(float(x[0]), float(x[-1]))
    handles = [Line2D([], [], color=color, marker=marker, markersize=4, linewidth=1.3,
                      label=label) for color, marker, label in zip(COLORS, MARKERS, METHOD_LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .965),
               ncol=3, frameon=False, columnspacing=1.15, handlelength=1.7)
    fig.text(.50, .025,
             r"Fixed normalized calibration game; display floor $10^{-10}$; no epsilon added.",
             ha="center", fontsize=8)
    save(fig, output, "calibrated_geometry")


def cumulative_figure(inputs: dict, output: Path) -> None:
    data = inputs["dynamics"]
    x = data["rounds"]
    mean = data["cumulative_difference"]
    low, high = data["cumulative_difference_interval"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(left=.105, right=.985, bottom=.28, top=.84, wspace=.29)
    for panel, axis in enumerate(axes):
        color = COLORS[panel+1]
        axis.fill_between(x, low[:, panel], high[:, panel], color=color, alpha=.18, linewidth=0)
        axis.plot(x, mean[:, panel], color=color, linewidth=1.65)
        axis.axhline(0, color=".35", linewidth=.8, linestyle="--")
        for boundary in (128, 384):
            axis.axvline(boundary, color=".4", linewidth=.75, linestyle=":")
        axis.set_xlim(1, 512)
        axis.set_xticks([1, 128, 256, 384, 512])
        axis.set_xlabel("Policy-selection round, t")
        axis.grid(axis="y", color=".90", linewidth=.6)
        axis.set_axisbelow(True)
        axis.set_title(f"({'a' if panel == 0 else 'b'}) Our one-switch − {'Window' if panel == 0 else 'Hedge'}", loc="left")
        axis.text(.125, .97, "Initial mode", transform=axis.transAxes, ha="center", va="top", fontsize=8)
        axis.text(.50, .97, "Exploration", transform=axis.transAxes, ha="center", va="top", fontsize=8)
        axis.text(.875, .97, "Mode learning", transform=axis.transAxes, ha="center", va="top", fontsize=8)
        if panel == 0:
            axis.set_ylim(-5, 87)
            axis.set_yticks([0, 20, 40, 60, 80])
            axis.set_ylabel(r"Expected cumulative cost difference, $\Delta C_b(t)$")
            axis.text(.52, .40, "First-action gap;\nlater actions coincide", transform=axis.transAxes,
                      ha="center", va="center", fontsize=8)
        else:
            axis.set_ylim(-2200, 220)
            axis.set_yticks([0, -500, -1000, -1500, -2000])
            axis.text(.98, .06, f"Final mean: {mean[-1, panel]:,.1f}", transform=axis.transAxes,
                      ha="right", va="bottom", fontsize=8, color=color)
    fig.text(.50, .98,
             r"Our one-switch ($W=16$, $\rho=0.25$) | Window ($W=16$) | Hedge (rate $\times4$)",
             ha="center", va="top", fontsize=8)
    fig.text(.5, .075,
             "Expected native cost for reset 50-step episodes; negative values favor our method.\n"
             "Bands: descriptive pointwise 95% intervals, conditional on calibration.",
             ha="center", fontsize=8, linespacing=1.45)
    save(fig, output, "cumulative_cost_difference")


def readme_text() -> str:
    return """[Русский](README.ru.md) | [Full English gallery](../figures/README.md)

# English figure exports for an AAMAS manuscript

These two-column figures use unchanged public results. This is a presentation update, with no new experiments, parameter tuning, or statistical analyses. PDF exports embed fonts and PNG previews use 300 dpi. Labels use at least 8 pt at the exported 7-inch width.

## Supplementary figure: primary comparisons and explanation

![Primary paired comparisons and cost decomposition](primary_comparison.png)

[Vector PDF](primary_comparison.pdf). Panel (a) shows **our method minus each baseline** in the original sum-of-four-components native cost per simulator step. Negative values favor our method. Intervals are the original 10,000-draw paired crossed-bootstrap intervals over 400 complete held-out simulator-seed tables and 50 complete common paths, providing approximate simultaneous 95% coverage across the two prespecified contrasts, conditional on the fixed calibration and selected configurations. The positive window difference is retained. Panel (b) decomposes exactly the same differences into native components; these descriptive means carry no additional hypothesis tests or uncertainty claim.

The locked configurations are **Our one-switch (W=16, rho=0.25)**, **Window (W=16)**, and **Hedge (rate ×4)**. The overall cost differences are −0.0618179 versus Hedge and +0.00264599 versus Window. Improvement is supported versus Hedge, not versus Window. All three methods share the same exogenous attacker paths. Each policy-selection round evaluates expected mixtures of separately reset 50-step simulator episodes.

## Optional main-paper figure: calibrated vector geometry

![Full-target vector error with explicit numerical floor counts](calibrated_geometry.png)

[Vector PDF](calibrated_geometry.pdf). This is the **absolute Euclidean distance to the full realized-hull response target**, using the frozen normalized training tensor. It is not a native scalar cost difference. Panel (a) retains all 50 primary paths and panel (b) retains all 20 new common paths at each of the six announced horizons. Lines are medians; translucent bands are empirical 10–90% ranges across paths, not confidence intervals. All numerical projection diagnostics remain available in the [original geometry data](../geometry/README.md).

The lower strips count values whose numerical upper distance is at or below the declared display floor 1e−10. Such values are omitted from logarithmic axes without substituting an epsilon. A band is shown only where both of its endpoints exceed the floor. At T≥128 the terminal error of our method and Window is at numerical precision on all tested paths; the decay order cannot be estimated. The target in panel (a) expands when a new opponent mode first appears, and the separate announced-horizon runs in panel (b) scale curriculum lengths and learning-rate formulas with T. The same 20 path-seed identifiers are reused across horizons; runs across horizons are not claimed to be statistically independent. This figure does not prove a convergence rate, transfer from calibration to deployment, or superiority over Window.

## Placement and captions

Use a compact primary-results table and the following cumulative difference figure in the main text. The paired-comparison/component figure is supplementary to avoid duplicating that table. Include the calibrated-geometry figure only if space permits, alongside its mathematical metric definition and numerical limitations. The existing [loss dynamics](../dynamics/loss_dynamics.pdf), [complete validation sensitivity](../dynamics/validation_sensitivity.pdf), and [path distribution](../dynamics/paired_path_distribution.pdf) are useful supplementary figures.

![Expected cumulative paired native cost differences](cumulative_cost_difference.png)

[Vector PDF](cumulative_cost_difference.pdf). Both panels subtract the indicated baseline from our method. The vertical axis is **expected cumulative native cost**, equal to 50 times the sum of mean per-step costs over policy-selection rounds: `Delta C_b(t) = 50 sum_s (c_s^(ours) - c_s^(b))`. Here `c_s^(m)` is instantaneous mean cost at round s; the terminal mean is `J_m = (1/T) sum_s c_s^(m)`. Bands are the existing 2,000-draw descriptive pointwise 95% crossed-bootstrap intervals; they do not replace the original simultaneous primary intervals. The three marked phases are initial mode, exploration, and mode learning against a uniform reference defense. The same exogenous paths are shared by all compared learners. The Window gap becomes constant after the initial round; its 1/t average dilution is not the convergence rate of vector error.

English LaTeX captions and plain-text accessibility descriptions are in [captions.tex](captions.tex). Each description is shorter than 2,000 characters. Figure labels, caption definitions, and the direction of subtraction must be retained when copying the artwork into a manuscript.

## Reproduction and provenance

From the repository root:

```powershell
python scripts/build_aamas_paper_figures.py --output results/runs/aamas_presentation_rebuild
```

The script reads only the original analysis and computed numerical projection arrays. It performs no bootstrap draws, simulator episodes, projections, or learner updates. `provenance.json` records all source hashes, exact estimates, captions, units, and software versions; `SHA256SUMS.json` covers the presentation files. Existing study protocols, algorithms, banks, and statistical results are unchanged.
"""


def russian_readme_text() -> str:
    return """[English](README.md) | [Полная галерея](../figures/README.ru.md)

# Английские графики для статьи AAMAS

Это обновление оформления опубликованных результатов. Новые эксперименты, подбор параметров и статистические расчёты не выполняются. PDF содержит встроенные шрифты, PNG — 300 dpi; минимальный размер подписей при ширине 7 дюймов — 8 pt.

В основной текст рекомендуется компактная таблица исходных результатов и [cumulative_cost_difference.pdf](cumulative_cost_difference.pdf). В каждой панели вычитается **наш метод минус соответствующий baseline**; отрицательная разность означает преимущество нашего метода. Вертикальная ось — ожидаемая накопленная нативная стоимость: 50 × сумма средних стоимостей на шаг по всем раундам выбора политики. Полосы — исходные 2 000 парных bootstrap-перевыборок, описательные поточечные интервалы 95% при фиксированной калибровке. Они не заменяют одновременные интервалы основного статистического анализа. Три фазы: начальный режим, исследование и обучение выбору режима против равномерной эталонной защиты. Все сравниваемые алгоритмы получают одинаковые внешние траектории противника.

В дополнительный материал рекомендуется [primary_comparison.pdf](primary_comparison.pdf): слева исходные одновременные интервалы для двух заранее заданных сравнений, справа разложение той же разности на четыре компоненты стоимости. Интервалы слева используют исходные 10 000 парных bootstrap-перевыборок целых 400 таблиц simulator seeds и 50 общих траекторий; приближённое одновременное покрытие для двух сравнений составляет 95% при фиксированной калибровке. Справа показаны средние без новых статистических утверждений.

Преимущество перед Hedge подтверждено, преимущество перед Window — нет: −0.0618179 против Hedge и +0.00264599 против Window. Стоимость дана на один шаг симулятора и складывается из четырёх исходных компонент. Один раунд выбора политики соответствует ожидаемому результату независимо перезапущенных эпизодов по 50 шагов, а не одному непрерывному развёртыванию сети.

Если позволяет объём статьи, добавьте [calibrated_geometry.pdf](calibrated_geometry.pdf). Это **абсолютная евклидова дистанция до полной целевой области по реализованной оболочке** в фиксированной нормированной калибровочной игре, а не разность скалярных стоимостей. Слева все 50 исходных траекторий, справа отдельные запуски по 20 общих траекторий на каждый заранее объявленный горизонт. Для разных горизонтов повторно используются те же 20 идентификаторов path seeds; статистическая независимость между горизонтами не утверждается. Линии — медианы, полосы — эмпирический диапазон 10–90% по траекториям, не доверительный интервал.

Нижние панели явно показывают долю численных верхних оценок дистанции не выше порога отображения 1e−10. Эти значения не сдвигаются искусственным epsilon. При T≥128 терминальная ошибка нашего метода и Window на всех проверенных траекториях достигает численной точности; порядок убывания по этим данным не оценивается. Целевая область слева расширяется при появлении новых режимов противника; справа длины фаз и формулы learning rate масштабируются с горизонтом. График не доказывает асимптотическую скорость или превосходство над Window.

Английские подписи и текстовые описания доступности для LaTeX находятся в [captions.tex](captions.tex); каждое описание короче 2 000 символов. Обозначение `c_s^(m)` относится к мгновенной средней стоимости на раунде s, а `J_m` — к терминальной средней по всем T раундам. Динамику потерь, полный валидационный перебор и распределение по траекториям можно оставить в дополнительном материале. Накопленная разность против Window постоянна после первого раунда; её деление на t не является скоростью сходимости векторной ошибки.

Воспроизведение из корня репозитория:

```powershell
python scripts/build_aamas_paper_figures.py --output results/runs/aamas_presentation_rebuild
```

Скрипт читает только исходный статистический анализ и уже рассчитанные численные проекции. Хэши входных файлов, значения, единицы, подписи и версии программ записаны в `provenance.json`; файлы оформления покрыты `SHA256SUMS.json`. Полное объяснение и иллюстрации доступны через переключатель на английскую версию.
"""


def build(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    inputs = load_inputs(source)
    output.mkdir(parents=True, exist_ok=True)
    style()
    primary_figure(inputs, output)
    geometry_figure(inputs, output)
    cumulative_figure(inputs, output)
    (output / "README.md").write_text(readme_text(), encoding="utf-8", newline="\n")
    (output / "README.ru.md").write_text(russian_readme_text(), encoding="utf-8", newline="\n")
    if any(len(description) > 2000 for description in DESCRIPTIONS.values()):
        raise ValueError("Figure accessibility descriptions must not exceed 2000 characters.")
    captions = (
        "% English captions for a two-column AAMAS manuscript. Adapt figure paths only.\n"
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{figures/primary_comparison.pdf}\n"
        f"\\caption{{{PRIMARY_CAPTION}}}\n"
        f"\\Description{{{DESCRIPTIONS['primary_comparison']}}}\n"
        "\\label{fig:cage-primary-comparison}\n\\end{figure*}\n\n"
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{figures/calibrated_geometry.pdf}\n"
        f"\\caption{{{GEOMETRY_CAPTION}}}\n"
        f"\\Description{{{DESCRIPTIONS['calibrated_geometry']}}}\n"
        "\\label{fig:cage-calibrated-geometry}\n\\end{figure*}\n\n"
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{figures/cumulative_cost_difference.pdf}\n"
        f"\\caption{{{CUMULATIVE_CAPTION}}}\n"
        f"\\Description{{{DESCRIPTIONS['cumulative_cost_difference']}}}\n"
        "\\label{fig:cage-cumulative-cost}\n\\end{figure*}\n"
    )
    (output / "captions.tex").write_text(captions, encoding="utf-8", newline="\n")
    sources = (
        "analysis.json", "selection.json", "protocol.json",
        "geometry/protocol.json", "geometry/prefix_geometry.npz", "geometry/horizon_geometry.npz",
        "dynamics/protocol.json", "dynamics/plot_data.npz",
    )
    provenance = {
        "schema_version": 1,
        "scope": "English manuscript presentation of unchanged original results",
        "builder_sha256": sha(Path(__file__)),
        "source_sha256": {name: sha(source/name) for name in sources},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "methods": list(METHODS), "method_labels": list(METHOD_LABELS),
        "primary_metric": "sum of four native costs per native simulator step",
        "difference_direction": "our method minus reference; negative favors our method",
        "primary_intervals": "original 10,000-draw paired crossed bootstrap, approximate simultaneous 95% for two prespecified comparisons; conditional on fixed calibration and selected configurations",
        "primary_contrasts": inputs["report"]["primary"]["contrasts"],
        "component_differences": (inputs["components"][0]-inputs["components"][1:]).tolist(),
        "geometry_metric": "absolute Euclidean distance to full realized-hull target in fixed normalized training game",
        "geometry_ranges": "empirical 10-90% path quantiles, not confidence intervals",
        "numerical_display_floor": FLOOR, "epsilon_added": False,
        "new_simulator_episodes": 0, "new_learner_runs": 0, "new_bootstrap_draws": 0,
        "cumulative_metric": "expected cumulative native simulator cost = 50 times summed within-episode per-step cost",
        "notation": {"c_s^(m)": "method m instantaneous mean native per-step cost at round s",
                     "J_m": "terminal mean native per-step cost averaged over all T rounds",
                     "Delta C_b(t)": "expected cumulative native cost of ours minus baseline b through round t"},
        "cumulative_intervals": "existing 2,000-draw descriptive pointwise 95% crossed-bootstrap intervals; not simultaneous curve coverage",
        "captions": {"primary_comparison": PRIMARY_CAPTION, "calibrated_geometry": GEOMETRY_CAPTION,
                     "cumulative_cost_difference": CUMULATIVE_CAPTION},
        "accessibility_descriptions": DESCRIPTIONS,
        "exports": {"width_inches": 7.0, "minimum_font_points": 8, "png_dpi": 300, "pdf_font_type": 42},
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "SHA256SUMS.json", {
        "algorithm": "sha256",
        "files": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"},
    })
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "results/cage_adaptation")
    parser.add_argument("--output", type=Path, default=REPO / "results/cage_adaptation/aamas")
    args = parser.parse_args()
    build(args.source, args.output)
    print(f"English AAMAS presentation figures written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
