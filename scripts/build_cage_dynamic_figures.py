"""Post-study scientific figures from every frozen primary CAGE trajectory.

No learner or simulator is run. Whole held-out simulator-seed tables and whole
common paths are the two crossed sampling units. The extra intervals are
descriptive and pointwise; the original two simultaneous contrasts are unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
METHODS = ("selected_scalar_one_switch", "selected_window", "selected_hedge")
COLORS = ("#167b77", "#c17724", "#6d65a9")
BOOTSTRAP_SEED = 46000000
BOOTSTRAP_SAMPLES = 2000
SMOOTHING_WINDOW = 16
EPISODE_STEPS = 50


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".dynamics-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def trailing_mean(values, window=SMOOTHING_WINDOW, axis=0):
    """Trailing visualization smoothing with shorter windows at the beginning."""
    values = np.asarray(values, dtype=float)
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError("The smoothing window must be a positive integer.")
    moved = np.moveaxis(values, axis, 0)
    padded = np.concatenate([np.zeros_like(moved[:1]), np.cumsum(moved, axis=0)], axis=0)
    end = np.arange(1, len(moved) + 1)
    start = np.maximum(0, end - int(window))
    denominator = (end - start).reshape((-1,) + (1,) * (moved.ndim - 1))
    return np.moveaxis((padded[end] - padded[start]) / denominator, 0, axis)


def joint_weights(actions, opponent_actions):
    """Expected whole-episode mixtures, indexed by path, round, method, cell."""
    actions = np.asarray(actions, dtype=float)
    opponent_actions = np.asarray(opponent_actions, dtype=float)
    if actions.ndim != 4 or opponent_actions.ndim != 3:
        raise ValueError("Actions must be [path,method,round,blue] and opponents [path,round,red].")
    if (actions.shape[0], actions.shape[2]) != opponent_actions.shape[:2]:
        raise ValueError("Action paths and horizons must agree.")
    for value in (actions, opponent_actions):
        if (not np.isfinite(value).all() or value.min() < -1e-12
                or not np.allclose(value.sum(axis=-1), 1, rtol=0, atol=1e-10)):
            raise ValueError("Every action must be a finite probability distribution.")
    joint = np.einsum("pmti,ptj->ptmij", actions, opponent_actions, optimize=True)
    return joint.reshape(joint.shape[:3] + (-1,))


def crossed_bootstrap(joint, bank, *, samples=BOOTSTRAP_SAMPLES,
                      seed=BOOTSTRAP_SEED, batch_size=64):
    """Paired whole-table/whole-path draws; never resample rounds or cells.

    Factor contraction avoids constructing the 400 x 50 x 512 x 3 array.
    The same seed and path weights are used for every method and component.
    """
    joint = np.asarray(joint, dtype=float)
    bank = np.asarray(bank, dtype=float)
    if joint.ndim != 4 or bank.ndim != 3 or joint.shape[-1] != bank.shape[1]:
        raise ValueError("Expected joint[path,round,method,cell] and bank[seed,cell,component].")
    if (min(joint.shape) < 1 or min(bank.shape) < 1 or not np.isfinite(joint).all()
            or not np.isfinite(bank).all() or samples < 1 or batch_size < 1):
        raise ValueError("Bootstrap inputs and draw counts must be finite and nonempty.")
    paths, horizon, methods, cells = joint.shape
    episode_seeds = len(bank)
    rng = np.random.default_rng(seed)
    series = np.empty((samples, horizon, methods))
    components = np.empty((samples, methods, bank.shape[-1]))
    flattened = joint.reshape(paths, -1)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        size = stop - start
        path_weights = rng.multinomial(paths, np.full(paths, 1 / paths), size=size) / paths
        seed_weights = rng.multinomial(episode_seeds, np.full(episode_seeds, 1 / episode_seeds), size=size) / episode_seeds
        weighted_joint = (path_weights @ flattened).reshape(size, horizon, methods, cells)
        weighted_bank = np.einsum("be,ejc->bjc", seed_weights, bank, optimize=True)
        series[start:stop] = np.einsum("btmj,bj->btm", weighted_joint,
                                     weighted_bank.sum(axis=-1), optimize=True)
        components[start:stop] = np.einsum("bmj,bjc->bmc", weighted_joint.mean(axis=1),
                                         weighted_bank, optimize=True)
    return series, components


def analyze(joint, bank, *, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    bank_mean = bank.mean(axis=0)
    mean = np.einsum("tmj,j->tm", joint.mean(axis=0), bank_mean.sum(axis=-1), optimize=True)
    bootstrap, boot_components = crossed_bootstrap(joint, bank, samples=samples, seed=seed)
    components = np.einsum("mj,jc->mc", joint.mean(axis=(0, 1)), bank_mean, optimize=True)
    path_means = np.einsum("pmj,j->pm", joint.mean(axis=1), bank_mean.sum(axis=-1), optimize=True)
    differences = mean[:, :1] - mean[:, 1:]
    boot_differences = bootstrap[:, :, :1] - bootstrap[:, :, 1:]
    cumulative = EPISODE_STEPS * np.cumsum(differences, axis=0)
    boot_cumulative = EPISODE_STEPS * np.cumsum(boot_differences, axis=1)
    boot_smoothed = trailing_mean(bootstrap, axis=1)
    boot_component_diff = boot_components[:, :1] - boot_components[:, 1:]
    bounds = lambda values: np.quantile(values, [.025, .975], axis=0)
    return {
        "rounds": np.arange(1, joint.shape[1] + 1),
        "mean_native_loss": mean,
        "smoothed_native_loss": trailing_mean(mean),
        "smoothed_native_loss_interval": bounds(boot_smoothed),
        "cumulative_difference": cumulative,
        "cumulative_difference_interval": bounds(boot_cumulative),
        "mean_components": components,
        "mean_components_interval": bounds(boot_components),
        "component_difference": components[:1] - components[1:],
        "component_difference_interval": bounds(boot_component_diff),
        "path_mean_native_loss": path_means,
        "path_mean_difference": path_means[:, :1] - path_means[:, 1:],
    }


def load_checked_inputs(input_path, report, bank_path):
    """Validate public action provenance and the original full held-out bank."""
    meta = read(report / "groups/test/meta.json")
    protocol = read(report / "protocol.json")
    analysis = read(report / "analysis.json")
    selection = read(report / "selection.json")
    if sha(report / "selection.json") != analysis["selection_sha256"]:
        raise ValueError("The locked selection checksum differs from the original primary analysis.")
    if sha(bank_path) != analysis["final_bank_sha256"]:
        raise ValueError("The held-out episode bank checksum differs from the original analysis.")
    spec = protocol["primary_test"]
    expected_seeds = list(range(spec["path_seed_start"], spec["path_seed_start"] + spec["path_count"]))
    with np.load(input_path, allow_pickle=False) as data:
        names = data["methods"].tolist()
        indices = [names.index(name) for name in METHODS]
        actions = data["actions"][:, indices].copy()
        path = data["opponent_actions"].copy()
        path_seeds = data["path_seeds"].tolist()
    source_file = input_path.parent / "primary_sources.json"
    source_hashes = read(source_file)["source_checkpoint_sha256"]
    if (path_seeds != expected_seeds or meta["path_seeds"] != expected_seeds
            or meta["completed_seeds"] != expected_seeds or meta["status"] != "complete"):
        raise ValueError("Figures must retain all predeclared completed primary paths in order.")
    expected_hashes = {f"seed_{seed}.npz": meta["path_sha256"][f"seed_{seed}.npz"] for seed in expected_seeds}
    if source_hashes != expected_hashes or len(source_hashes) != len(expected_seeds):
        raise ValueError("Public input provenance must cover every original checkpoint without omissions.")
    joint = joint_weights(actions, path)
    if joint.shape[1] != spec["horizon"]:
        raise ValueError("Public trajectory horizon differs from the frozen primary study.")
    if sha(report / "groups/test/occupancies.npz") != meta["aggregate_sha256"]:
        raise ValueError("Primary occupancy checksum mismatch.")
    with np.load(report / "groups/test/occupancies.npz", allow_pickle=False) as data:
        method_indices = [data["methods"].tolist().index(name) for name in METHODS]
        occupancies = data["occupancies"][:, method_indices]
        if not np.allclose(joint.mean(axis=1).reshape(occupancies.shape), occupancies, atol=1e-12, rtol=0):
            raise ValueError("Public trajectories do not reconstruct the full original occupancies.")
    with np.load(bank_path, allow_pickle=False) as data:
        bank = data["episode_losses"].copy()
        episode_seeds = data["seeds"].tolist()
        metric_names = data["metric_names"].tolist()
        red_names = data["red_names"].tolist()
    if episode_seeds != analysis["test_seeds"]:
        raise ValueError("Every held-out simulator seed must be retained.")
    if read(bank_path.parent / "manifest.json")["context"]["episode_steps"] != EPISODE_STEPS:
        raise ValueError("Cumulative native costs require the original 50-step episodes.")
    flattened_bank = bank.reshape(len(bank), -1, bank.shape[-1])
    components = np.einsum("pmj,ejc->mc", joint.mean(axis=1), flattened_bank, optimize=True) / (len(joint) * len(bank))
    for index, name in enumerate(METHODS):
        expected = analysis["primary"]["methods"][name]
        if not np.allclose(components[index], expected["mean_components"], rtol=0, atol=1e-12):
            raise ValueError("Full public trajectories do not reproduce the reported component means.")
    if not np.array_equal(actions[:, 0, 1:], actions[:, 1, 1:]):
        raise ValueError("Public traces no longer support the frozen action-equality diagnostic.")
    return joint, flattened_bank, path, metric_names, red_names, {
        "primary_inputs_sha256": sha(input_path),
        "primary_sources_sha256": sha(source_file),
        "primary_occupancies_sha256": sha(report / "groups/test/occupancies.npz"),
        "primary_meta_sha256": sha(report / "groups/test/meta.json"),
        "heldout_bank_sha256": sha(bank_path),
        "selection_sha256": sha(report / "selection.json"),
        "validation_grid_sha256": sha(report / "validation_grid.json"),
        "checkpoint_sha256": expected_hashes,
        "path_seeds": path_seeds,
        "episode_seeds": episode_seeds,
        "selected": selection["selected"],
    }


def validation_matrix(grid):
    rows = [row for row in grid["rows"] if row["family"] == "scalar"]
    windows = sorted({row["window"] for row in rows})
    rhos = sorted({row["rho"] for row in rows})
    if len(rows) != len(windows) * len(rhos):
        raise ValueError("The full declared scalar validation grid must be present.")
    matrix = np.full((len(windows), len(rhos)), np.nan)
    for row in rows:
        index = windows.index(row["window"]), rhos.index(row["rho"])
        if np.isfinite(matrix[index]):
            raise ValueError("Validation grid contains a duplicate configuration.")
        matrix[index] = row["mean_native_loss"]
    if not np.isfinite(matrix).all():
        raise ValueError("Validation losses must be finite.")
    return windows, rhos, matrix


def save_figure(fig, output, name, created):
    for suffix in ("png", "pdf"):
        metadata = ({"Software": "Matplotlib"} if suffix == "png" else
                    {"Creator": "CAGE 2 descriptive dynamic figures", "CreationDate": created, "ModDate": created})
        with tempfile.NamedTemporaryFile(dir=output, prefix=".figure-", suffix="." + suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, dpi=200, bbox_inches="tight", metadata=metadata)
            temporary.replace(output / (name + "." + suffix))
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def phase_marks(ax, ru, *, text=True):
    boundaries = (0, 128, 384, 512)
    labels = ("Начальный режим", "Разнообразные атаки", "Обучение противника") if ru else (
        "Initial mode", "Broad exploration", "Attacker learning")
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(start + .5, stop + .5, color=("#64748b", "#aaaaaa", "#bd975e")[index], alpha=.045, zorder=0)
        if text:
            ax.text((start + stop + 1) / 2, 1.035, labels[index], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
    for position in boundaries[1:-1]:
        ax.axvline(position + .5, color="#888", linewidth=.8, linestyle=":")
    ax.set_xlim(.5, 512.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.25)


def figures(data, grid, selection, output, created):
    rounds = data["rounds"]
    windows, rhos, matrix = validation_matrix(grid)
    for ru in (False, True):
        suffix = ".ru" if ru else ""
        labels = ("Наш one-switch", "Окно (W=16)", "Hedge (множитель η=4)") if ru else (
            "Our one-switch", "Window (W=16)", "Hedge (η multiplier=4)")
        fig, ax = plt.subplots(figsize=(8.8, 4.9))
        band = data["smoothed_native_loss_interval"]
        for index, label in enumerate(labels):
            ax.fill_between(rounds, band[0, :, index], band[1, :, index], color=COLORS[index], alpha=.14, linewidth=0)
            ax.plot(rounds, data["smoothed_native_loss"][:, index], label=label, color=COLORS[index],
                    linewidth=2.1, linestyle=("-", "--", "-")[index], alpha=.95)
        phase_marks(ax, ru)
        ax.set_xlabel("Раунд выбора политики" if ru else "Policy-selection round")
        ax.set_ylabel("Потери на шаг симулятора" if ru else "Native loss per simulator step")
        ax.set_title("Динамика потерь выбранных методов" if ru else "Loss dynamics of the selected methods", pad=35)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        fig.text(.5, .015, ("Среднее по всем 50 путям и 400 семенам. Сглаживание: последние 16 раундов.\n"
                            "Полосы: описательные поточечные интервалы 95%; окно совпадает с нашим методом после раунда 1.") if ru else
                 ("Mean over all 50 paths and 400 seeds. Visualization: trailing 16-round average.\n"
                  "Bands: descriptive pointwise 95% intervals; window and our method coincide after round 1."),
                 ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, .095, 1, 1))
        save_figure(fig, output, "loss_dynamics" + suffix, created)

        fig, axes = plt.subplots(2, 1, figsize=(8.8, 6), sharex=True)
        cumulative = data["cumulative_difference"]
        bounds = data["cumulative_difference_interval"]
        for index, ax in enumerate(axes):
            ax.fill_between(rounds, bounds[0, :, index], bounds[1, :, index], color=COLORS[index + 1], alpha=.2, linewidth=0)
            ax.plot(rounds, cumulative[:, index], color=COLORS[index + 1], linewidth=2,
                    label=("Наш метод − " if ru else "Our method − ") + labels[index + 1])
            ax.axhline(0, color="#333", linewidth=.9)
            phase_marks(ax, ru, text=index == 0)
            ax.legend(loc="best", fontsize=9)
        axes[0].set_ylim(-5, 80)
        axes[0].text(.98, .15, ("После первого раунда разность постоянна:\nдействия методов далее совпадают.") if ru else
                     ("The difference is constant after round 1:\nsubsequent actions are identical."),
                     transform=axes[0].transAxes, ha="right", fontsize=9)
        axes[1].set_xlabel("Раунд выбора политики" if ru else "Policy-selection round")
        fig.supylabel("Накопленная стоимость: наш метод − метод сравнения" if ru else
                      "Cumulative native cost: our method − reference", fontsize=10)
        axes[0].set_title("Накопленные парные различия" if ru else "Cumulative paired loss differences", pad=35)
        fig.text(.5, .015, ("Отрицательная разность означает меньшие потери нашего метода. Интервалы 95% — поточечные.\n"
                            "Стоимость = 50 × сумма средних потерь эпизодов; каждый раунд — отдельный 50-шаговый эпизод.") if ru else
                 ("Negative differences favor our method. Intervals are descriptive and pointwise 95%.\n"
                  "Cost = 50 × sum of episode-average losses; each round is a reset 50-step episode."),
                 ha="center", fontsize=8.5)
        fig.tight_layout(rect=(.02, .085, 1, 1))
        save_figure(fig, output, "cumulative_differences" + suffix, created)

        names = ("Компрометация хостов", "Компрометация серверов", "Нарушения работы", "Восстановление") if ru else (
            "Host compromise", "Server compromise", "Operational disruption", "Restore cost")
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6))
        components = data["mean_components"]
        component_bounds = data["mean_components_interval"]
        for component, ax in enumerate(axes.flat):
            estimates = components[:, component]
            error = np.stack([estimates - component_bounds[0, :, component],
                              component_bounds[1, :, component] - estimates])
            ax.bar(np.arange(3), estimates, color=COLORS, width=.6)
            ax.errorbar(np.arange(3), estimates, yerr=np.maximum(error, 0), fmt="none", color="#333", capsize=4, linewidth=1)
            ax.set_xticks(np.arange(3), ["Наш" if ru else "Ours", "Окно" if ru else "Window", "Hedge"])
            ax.set_title(names[component], fontsize=11)
            ax.set_ylim(0, max(float(component_bounds[1, :, component].max()) * 1.24, .01))
            for index, value in enumerate(estimates):
                ax.text(index, float(component_bounds[1, index, component]) + ax.get_ylim()[1] * .025,
                        f"{value:.4f}", ha="center", va="bottom", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.supylabel("Средние потери на шаг симулятора" if ru else "Mean native loss per simulator step", fontsize=10)
        fig.suptitle("Четыре компоненты потерь" if ru else "The four loss components", fontsize=13)
        fig.text(.5, .01, ("Все 50 путей × 400 таблиц семян. Описательные интервалы 95% при фиксированной калибровке.\n"
                           "Сумма четырёх компонент равна основной метрике; шкалы панелей различаются.") if ru else
                 ("All 50 paths × 400 seed tables. Descriptive 95% intervals conditional on fixed calibration.\n"
                  "The four components sum to the primary metric; panel scales differ."), ha="center", fontsize=8.5)
        fig.tight_layout(rect=(.02, .09, 1, .98))
        save_figure(fig, output, "loss_components" + suffix, created)

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
        for index, ax in enumerate(axes):
            values = np.sort(data["path_mean_difference"][:, index])
            if np.ptp(values) < 1e-12:
                value = float(values.mean())
                ax.plot([value, value], [0, 1], color=COLORS[index + 1], linewidth=2)
                ax.scatter([value], [.5], s=60, color=COLORS[index + 1], zorder=5)
                ax.annotate(("Все 50 путей:\n" if ru else "All 50 paths:\n") + f"Δ = {value:+.5f}",
                            xy=(value, .5), xytext=(.2, .75), textcoords="axes fraction", fontsize=10,
                            arrowprops={"arrowstyle": "-", "color": "#555"})
                ax.set_xlim(-.001, .006)
            else:
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS[index + 1], linewidth=2)
                ax.plot(values, np.zeros_like(values) + .025, "|", color=COLORS[index + 1], alpha=.6)
            ax.axvline(0, color="#555", linewidth=.9)
            ax.set_title(("Наш метод − " if ru else "Our method − ") + labels[index + 1], fontsize=11)
            ax.set_xlabel("Разность средних потерь" if ru else "Difference in mean native loss")
            ax.set_ylim(0, 1.04)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Доля путей с разностью ≤ x" if ru else "Fraction of paths with difference ≤ x")
        fig.suptitle("Парные различия по всем 50 траекториям" if ru else "Paired differences across all 50 paths", fontsize=13)
        fig.text(.5, .01, ("Каждая точка — целый путь, усреднённый по одному и тому же банку 400 семян.\n"
                           "Это описательный межпутевой разброс; отрицательное значение означает преимущество нашего метода.") if ru else
                 ("Each observation is a whole path, averaged over the same bank of 400 seeds.\n"
                  "This is descriptive between-path variation; a negative value favors our method."), ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, .12, 1, .95))
        save_figure(fig, output, "paired_path_distribution" + suffix, created)

        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
        ax.set_xticks(np.arange(len(rhos)), [str(value) for value in rhos])
        ax.set_yticks(np.arange(len(windows)), [str(value) for value in windows])
        ax.set_xlabel("Допустимая доля зазора ρ" if ru else "Permitted gap fraction ρ")
        ax.set_ylabel("Окно прогноза W" if ru else "Forecast window W")
        for i in range(len(windows)):
            for j in range(len(rhos)):
                normalized = (matrix[i, j] - matrix.min()) / max(np.ptp(matrix), 1e-15)
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=10,
                        color="white" if normalized > .5 else "#111")
        chosen = selection["selected"]["scalar"]
        row, column = windows.index(chosen["window"]), rhos.index(chosen["rho"])
        ax.add_patch(Rectangle((column - .48, row - .48), .96, .96, fill=False, linewidth=2.5, edgecolor="#ef5b48"))
        ax.set_title("Чувствительность параметров: только валидация" if ru else
                     "Parameter sensitivity: validation only", pad=15)
        colorbar = fig.colorbar(image, ax=ax, fraction=.055)
        colorbar.set_label("Средние потери; меньше — лучше" if ru else "Mean native loss; lower is better", fontsize=9)
        fig.text(.5, .01, ("Полная предзаданная сетка: 20 настроек, 100 семян и 20 путей.\n"
                           "Рамка: W=16, ρ=0,25, выбранные до финального теста. Нового подбора параметров нет.") if ru else
                 ("Complete predeclared grid: 20 configurations, 100 seeds and 20 paths.\n"
                  "Frame: W=16, ρ=0.25, locked before the final test. No parameters were retuned."),
                 ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, .11, 1, 1))
        save_figure(fig, output, "validation_sensitivity" + suffix, created)


def documentation(output, summary):
    for ru in (False, True):
        suffix = ".ru" if ru else ""
        link = "[English](README.md) | [Русский](README.ru.md)\n\n"
        if ru:
            text = """# Динамика выбранного метода и методов сравнения

Эти дополнительные графики построены после завершения исследования. Они используют все 50 предзаданных финальных путей и все 400 таблиц тестовых семян. Алгоритмы, калибровка, выбранные параметры и два исходных основных сравнения не изменены. Никакие новые эпизоды симулятора не запускались.

## Динамика потерь

![Динамика потерь](loss_dynamics.ru.png)

Линии показывают потери, усреднённые по всем путям и таблицам семян, со скользящим средним по последним 16 раундам. В начале используются доступные раунды. Сглаживание служит только визуализации. Фазы противника: начальный режим (раунды 1–128), разнообразные атаки (129–384), обучение выбора известных режимов против равномерной опорной защиты (385–512). Это общий внешний путь для всех сравниваемых защит. Противник не обучается отдельно против каждой из них.

Полосы — описательные поточечные интервалы 95%, не одновременные полосы по всей кривой. Выполнено 2000 парных скрещённых бутстрэп-перевыборок (семя 46000000). Единицы перевыборки — целые 400 таблиц семян и целые 50 путей, с одинаковыми весами для всех методов. Раунды и отдельные ячейки таблиц не считаются независимыми наблюдениями. Калибровка и выбор параметров фиксированы; их неопределённость не включена.

## Накопленные парные различия

![Накопленные различия](cumulative_differences.ru.png)

Разность определена как наш метод минус метод сравнения: отрицательные значения означают меньшие потери нашего метода. По вертикали — ожидаемая накопленная стоимость: 50 × сумма средних потерь эпизодов по раундам. Каждый раунд соответствует смеси ожидаемых результатов отдельных 50-шаговых эпизодов с перезапуском симулятора. Это сумма стоимости нескольких эпизодов, а не одного непрерывного сетевого развёртывания.

Разность с окном постоянна после первого раунда: далее действия совпадают во всех 50 путях. Если разделить эту константу на число раундов, получится убывание 1/t. Это следствие первого хода, а не экспериментальная скорость убывания векторной ошибки из теоремы.

## Компоненты стоимости

![Четыре компоненты](loss_components.ru.png)

Показаны исходные положительные стоимости без изменения весов: компрометация хостов, компрометация серверов, нарушения работы и восстановление. Их сумма равна основной скалярной метрике. Все столбцы начинаются от нуля; шкалы панелей различаются. Интервалы 95% описательные, построены по тем же скрещённым перевыборкам. Парные компонентные разности и их интервалы также сохранены в `summary.json`; проверки новых гипотез и коррекция по компонентам не выполнялись.

## Различия между путями

![Распределение парных различий](paired_path_distribution.ru.png)

Это распределение 50 парных разностей. Каждая разность сначала усредняется по одному и тому же банку из 400 семян. График показывает межпутевой разброс условно этому банку; 50 × 400 оценок не трактуются как 20 000 независимых наблюдений. Для окна все 50 значения совпадают, поэтому показана точка и ступень, без сглаженной плотности.

## Чувствительность параметров

![Валидационная сетка](validation_sensitivity.ru.png)

Полная предзаданная сетка из 20 настроек нашего оракула использует только валидационные данные (100 семян и 20 путей). Выбранная до теста настройка W=16, ρ=0,25 отмечена рамкой. После просмотра финальных результатов параметры не менялись.

## Воспроизводимость

Общие входные траектории находятся в `../geometry/primary_inputs.npz`: действия всех трёх выбранных методов и общий путь для каждого из 50 первичных семян. Они проверены по контрольным суммам исходных завершённых контрольных точек; средние заново сверены с неизменённым основным отчётом. Банк тестовых таблиц находится в `../../../data/cage2_adaptation/test50/bank.npz`.

`protocol.json` фиксирует дополнительный план визуализации, `summary.json` — численные итоги и происхождение входов, `plot_data.npz` — рассчитанные линии и интервалы. Интервалы этого дополнения не заменяют исходные одновременные интервалы двух предзаданных сравнений в [основном отчёте](../README.ru.md).

Из корня репозитория:

```powershell
python scripts/build_cage_dynamic_figures.py --output results/runs/dynamic_rebuild
```

Экспорт для статьи: [динамика](loss_dynamics.ru.pdf), [накопленные разности](cumulative_differences.ru.pdf), [компоненты](loss_components.ru.pdf), [распределение](paired_path_distribution.ru.pdf), [параметры](validation_sensitivity.ru.pdf). Английские PNG/PDF доступны через переключатель языка.
"""
        else:
            text = """# Dynamics of the selected method and references

These additional figures were created after the study was completed. They use all 50 predeclared final paths and all 400 held-out simulator-seed tables. Algorithms, calibration, selected configurations and the original two primary contrasts are unchanged. No new simulator episodes were run.

## Loss dynamics

![Loss dynamics](loss_dynamics.png)

Lines average over every path and seed table, with a trailing 16-round moving average. The initial windows use all rounds available so far. Smoothing is for visualization only. Attacker phases are the initial mode (rounds 1–128), broad exploration (129–384), and learning to choose among the frozen attack modes against a uniform reference defense (385–512). All defenses face the same exogenous path; the attacker does not learn separately against each compared method.

Bands are descriptive pointwise 95% intervals, not simultaneous bands over the complete curve. There are 2000 paired crossed bootstrap draws (seed 46000000). Entire simulator-seed tables and entire common paths are resampled, with the same weights for all methods. Rounds and table cells are not independent sampling units. Inference conditions on the calibration and locked configurations, excluding their uncertainty.

## Cumulative paired differences

![Cumulative differences](cumulative_differences.png)

The sign is our method minus reference: negative values favor our method. The vertical axis is expected cumulative native cost: 50 × the sum of within-episode average losses over policy-selection rounds. Each round represents expected mixture outcomes from independently reset 50-step episodes, rather than one persistent network deployment.

The window difference is constant after round 1 because all subsequent actions coincide on all 50 paths. Dividing this constant by the number of rounds yields a 1/t decline. This is an arithmetic first-action effect, not the convergence rate of the theorem's vector error.

## Cost components

![Four cost components](loss_components.png)

The four original positive native costs are host compromise, server compromise, operational disruption, and restore cost. Their sum is the primary scalar metric. Every bar axis starts at zero; panel scales differ. Descriptive 95% intervals use the same crossed draws. Paired component differences and intervals are also retained in `summary.json`; no new hypothesis tests or component-wise multiplicity correction are performed.

## Between-path differences

![Paired path distribution](paired_path_distribution.png)

The distribution contains 50 paired path differences. Each first averages over the same bank of 400 seeds. It describes between-path variation conditional on that bank; 50 × 400 evaluations are not treated as 20,000 independent observations. All 50 window differences coincide and are shown as a point and jump, without a smoothed density.

## Parameter sensitivity

![Validation grid](validation_sensitivity.png)

The complete predeclared 20-configuration scalar-oracle grid uses only validation data (100 seeds and 20 paths). The W=16, rho=0.25 configuration locked before test is framed. No parameters were retuned after inspecting final outcomes.

## Reproducibility

Shared trajectory inputs are in `../geometry/primary_inputs.npz`: actions of all three selected methods and the common path for every primary seed. Their provenance is checked against all completed original checkpoint hashes; reconstructed means are checked against the unchanged primary report. The held-out seed bank is `../../../data/cage2_adaptation/test50/bank.npz`.

`protocol.json` records the additional visualization plan, `summary.json` records estimates and input provenance, and `plot_data.npz` contains computed curves and intervals. These extra intervals do not replace the original simultaneous intervals for the two predeclared comparisons in the [primary report](../README.md).

From the repository root:

```powershell
python scripts/build_cage_dynamic_figures.py --output results/runs/dynamic_rebuild
```

Publication exports: [dynamics](loss_dynamics.pdf), [cumulative differences](cumulative_differences.pdf), [components](loss_components.pdf), [distribution](paired_path_distribution.pdf), [sensitivity](validation_sensitivity.pdf). Russian PNG/PDF figures are available through the language switch.
"""
        atomic_bytes(output / ("README" + suffix + ".md"), (link + text).encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "results/cage_adaptation/geometry/primary_inputs.npz")
    parser.add_argument("--report", type=Path, default=REPO / "results/cage_adaptation")
    parser.add_argument("--bank", type=Path, default=REPO / "data/cage2_adaptation/test50/bank.npz")
    parser.add_argument("--output", type=Path, default=REPO / "results/cage_adaptation/dynamics")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    canonical_protocol = REPO / "results/cage_adaptation/dynamics/protocol.json"
    if protocol_path.exists():
        plot_protocol = read(protocol_path)
    elif canonical_protocol.exists():
        plot_protocol = read(canonical_protocol)
        write(protocol_path, plot_protocol)
    else:
        plot_protocol = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "Post-study descriptive visualization; original test outcomes already known",
            "methods": list(METHODS),
            "all_predeclared_paths": True,
            "all_heldout_episode_seeds": True,
            "new_simulator_episodes": 0,
            "new_parameter_selection": False,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_units": ["whole held-out simulator-seed tables", "whole common paths"],
            "paired_across_methods_and_components": True,
            "interval_type": "descriptive pointwise 95% percentile intervals; not simultaneous",
            "smoothing": {"type": "trailing moving average for visualization only", "window": SMOOTHING_WINDOW},
            "phase_boundaries": [0, 128, 384, 512],
            "calibration_and_configuration_uncertainty_included": False,
            "primary_inference_replaced": False,
            "vector_error_evaluated_by_this_script": False,
        }
        write(protocol_path, plot_protocol)
    expected_fields = {"bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                       "methods": list(METHODS), "phase_boundaries": [0, 128, 384, 512]}
    if any(plot_protocol.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Existing descriptive plot protocol conflicts with the fixed implementation.")
    joint, bank, path, metrics, red_names, provenance = load_checked_inputs(args.input, args.report, args.bank)
    print(f"Checked all {len(joint)} paths and {len(bank)} complete held-out seed tables.", flush=True)
    data = analyze(joint, bank)
    summary = {
        "schema_version": 1,
        "plot_protocol_sha256": sha(protocol_path),
        "script_sha256": sha(__file__),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "source": provenance,
        "study_figure_protocol_sha256": sha(REPO / "docs/cage_figures_protocol.json"),
        "cumulative_native_cost_multiplier": EPISODE_STEPS,
        "methods": list(METHODS),
        "components": metrics,
        "final_mean_native_loss": data["mean_native_loss"].mean(axis=0).tolist(),
        "mean_components": data["mean_components"].tolist(),
        "paired_component_difference": data["component_difference"].tolist(),
        "paired_component_difference_interval": data["component_difference_interval"].tolist(),
        "cumulative_difference_at_horizon": data["cumulative_difference"][-1].tolist(),
        "cumulative_difference_pointwise_interval_at_horizon": data["cumulative_difference_interval"][:, -1].tolist(),
        "path_mean_difference_min": data["path_mean_difference"].min(axis=0).tolist(),
        "path_mean_difference_max": data["path_mean_difference"].max(axis=0).tolist(),
        "paths_with_lower_loss_than_reference": (data["path_mean_difference"] < 0).sum(axis=0).tolist(),
        "window_actions_equal_after_round_1": True,
        "interpretation": "Descriptive post-study figures; no new hypothesis tests or simultaneous curve claims",
    }
    with tempfile.NamedTemporaryFile(dir=args.output, prefix=".data-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **data, red_mode_fraction=path.mean(axis=0), red_names=np.array(red_names),
                            methods=np.array(METHODS), metric_names=np.array(metrics))
        temporary.replace(args.output / "plot_data.npz")
    finally:
        temporary.unlink(missing_ok=True)
    summary["plot_data_sha256"] = sha(args.output / "plot_data.npz")
    write(args.output / "summary.json", summary)
    created = datetime.fromisoformat(plot_protocol["created_utc"])
    figures(data, read(args.report / "validation_grid.json"), read(args.report / "selection.json"), args.output, created)
    documentation(args.output, summary)
    print(json.dumps({"output": str(args.output), "final_mean_native_loss": summary["final_mean_native_loss"],
                      "paths_with_lower_loss_than_reference": summary["paths_with_lower_loss_than_reference"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
