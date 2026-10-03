"""Publish checked aggregate artifacts and bilingual adaptation results."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.collect_cage_bank import load_episode_bank

PROTOCOL_SHA = "a1c8976415eb867bb179dfada83c4a4e9fa0b8667f2c1883aa4044f199fb7de3"
LABELS = {
    "selected_scalar": "Selected scalar-aware oracle",
    "selected_scalar_one_switch": "Selected scalar-aware oracle",
    "scalar_selected": "Selected scalar-aware oracle",
    "selected_window": "Selected window",
    "window_selected": "Selected window",
    "selected_hedge": "Selected Hedge",
    "hedge_selected": "Selected Hedge",
    "original_one_switch": "Original one-switch",
    "original_window16": "Original window (16)",
    "original_hedge": "Original Hedge",
    "fresh_restart1000": "Original: fresh restart",
    "retained_restart1000": "Original: retained history",
    "scalar_fresh_restart1000": "Scalar: fresh restart",
    "scalar_retained_restart1000": "Scalar: retained history",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, obj):
    Path(path).write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

def labels(name):
    return LABELS.get(name, name)

def means(group):
    return {name: float(value["mean_native_loss"]) for name, value in group["methods"].items()}

def result_label(value, ru):
    if value["upper"] < 0:
        return "преимущество" if ru else "advantage"
    if value["lower"] > 0:
        return "хуже" if ru else "disadvantage"
    return "не подтверждено" if ru else "inconclusive"

def save_figure(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def figures(analysis, grid, output):
    primary = analysis["primary"]
    contrasts = primary["contrasts"]
    keys = list(contrasts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.array([contrasts[k]["estimate"] for k in keys])
    lo = np.array([contrasts[k]["lower"] for k in keys])
    hi = np.array([contrasts[k]["upper"] for k in keys])
    axes[0].errorbar(x, np.arange(len(keys)), xerr=np.vstack((x-lo, hi-x)), fmt="o", color="#236f99", capsize=4)
    axes[0].axvline(0, color="#555", lw=1)
    axes[0].set_yticks(np.arange(len(keys)), [labels(k) for k in keys])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Selected scalar oracle minus reference\nNative loss per simulator step")
    axes[0].set_title("Primary simultaneous 95% intervals")
    values = means(primary)
    names = list(values)
    axes[1].barh(np.arange(len(names)), list(values.values()), color=["#267b7a" if "scalar" in n else "#8895a3" for n in names])
    axes[1].set_yticks(np.arange(len(names)), [labels(n) for n in names])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean native loss per simulator step")
    axes[1].set_title("Final test means (lower is better)")
    fig.tight_layout()
    save_figure(fig, output, "primary_comparisons")

    restarts = analysis["restarts"]
    scenarios = list(restarts)
    names = list(restarts[scenarios[0]]["methods"])
    matrix = np.array([[means(restarts[s])[n] for n in names] for s in scenarios])
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(names)), [labels(n) for n in names], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(scenarios)), scenarios)
    for i in range(len(scenarios)):
        for j in range(len(names)):
            color = "white" if matrix[i,j] > matrix.min() + .55 * np.ptp(matrix) else "#111"
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=10, color=color)
    ax.set_title("Exploratory block study: 3000 meta-rounds, blocks of 1000\nMean native loss per simulator step; lower is better")
    fig.colorbar(im, ax=ax, fraction=.025)
    fig.tight_layout()
    save_figure(fig, output, "restart_losses")

    rows = grid["rows"] if isinstance(grid, dict) and "rows" in grid else grid
    scalar = [r for r in rows if r["family"] == "scalar"]
    windows = sorted({int(r["window"]) for r in scalar})
    rhos = sorted({float(r["rho"]) for r in scalar})
    matrix = np.array([[next(float(r["mean_native_loss"]) for r in scalar if int(r["window"]) == w and float(r["rho"]) == rho) for w in windows] for rho in rhos])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(windows)), windows)
    ax.set_yticks(np.arange(len(rhos)), rhos)
    ax.set_xlabel("Forecast window W")
    ax.set_ylabel("Permitted gap fraction rho")
    ax.set_title("Validation grid: scalar-aware saddle oracle\nSelection data; no test inference")
    for i in range(len(rhos)):
        for j in range(len(windows)):
            color = "white" if matrix[i,j] > matrix.min() + .55 * np.ptp(matrix) else "#111"
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_figure(fig, output, "validation_grid")

def markdown(analysis, selection, protocol, ru, trace_diagnostics=None):
    language = "[English](README.md) | [Русский](README.ru.md)"
    selected = selection["selected"]
    title = "Настройка оракула и перезапуски CAGE 2" if ru else "CAGE 2 oracle tuning and block restarts"
    intro = (
        "Результаты новой проверки после настройки на отдельной валидации. Исходная статья и прежние результаты сохранены. Метрика — сумма четырёх нативных потерь на шаг симулятора; меньше означает лучше."
        if ru else
        "Results from a new test after tuning on separate validation data. The original article and previous results are preserved. The outcome is the sum of four native loss components per simulator step; lower is better."
    )
    lines = [language, "", "# " + title, "", intro, "",
             ("Параметры выбраны до чтения финальных результатов. Валидация: 100 семян и 20 общих путей. Финальный тест: 400 новых семян и 50 новых путей при горизонте 512. Всего собраны 9000 новых 50-шаговых эпизодов."
              if ru else "Configurations were locked before final outcomes were read. Validation uses 100 seeds and 20 common paths. Final testing uses 400 new seeds and 50 new paths at horizon 512. A total of 9000 new 50-step simulator episodes were collected."),
             "", "## " + ("Выбранные параметры" if ru else "Selected configurations"), ""]
    for family, config in selected.items():
        lines.append("- " + family + ": " + json.dumps(config, ensure_ascii=False, sort_keys=True))
    lines += ["", "## " + ("Основной тест" if ru else "Primary test"), "",
              ("| Метод | Средние потери |" if ru else "| Method | Mean loss |"), "|---|---:|"]
    for name, value in means(analysis["primary"]).items():
        lines.append(f"| {name} | {value:.5f} |")
    lines += ["", ("| Наш выбранный оракул минус базовый | Разность | Одновременный интервал | Решение |"
                   if ru else "| Selected scalar oracle minus reference | Difference | Simultaneous interval | Decision |"),
              "|---|---:|---:|---|"]
    for name, value in analysis["primary"]["contrasts"].items():
        lines.append(f'| {name} | {value["estimate"]:.5f} | [{value["lower"]:.5f}, {value["upper"]:.5f}] | {result_label(value, ru)} |')
    lines += ["",
              ("Интервалы приближённые: 10 000 парных перекрёстных бутстрэп-выборок с одновременным семейным уровнем 95% для двух сравнений. Сохраняется зависимость внутри таблицы одного семени и внутри пути. Оценка условна на исходной калибровке и выбранных параметрах. Сравнения с исходными ненастроенными методами описательные."
               if ru else "Intervals are approximate: 10,000 paired crossed bootstrap draws with simultaneous family level 95% for two contrasts. Dependence within each seed table and each path is preserved. Inference is conditional on the original calibration and locked configurations. Comparisons with original untuned methods are descriptive."),
              ""]
    if trace_diagnostics and trace_diagnostics["max_action_difference_after_first_round"] == 0:
        lines += [("Во всех 50 финальных путях действия выбранного нашего оракула и выбранного окна в точности совпадают со второго раунда. Разность потерь целиком объясняется первым равномерным ходом нашего алгоритма: окно сразу отвечает на известный начальный режим. Замена произвольного первого хода допустима теоретически, но не входила в зафиксированную настройку. Это наблюдение не доказывает превосходства над окном."
                   if ru else "Across all 50 final paths, the selected scalar oracle and selected window actions are exactly equal after the first round. The loss difference is entirely due to the oracle's first uniform action: the window responds immediately to the known initial mode. Changing the arbitrary first action is theoretically permitted but was outside the locked tuning plan. This observation does not establish superiority over the window."), ""]
    lines += ["## " + ("Перезапуски" if ru else "Restarts"), "",
              ("Блок содержит 1000 раундов выбора полной политики, то есть 50 000 внутренних шагов в отдельно инициализированных эпизодах. Горизонт 3000 содержит три блока. Сохраняется калибровка; сбрасываются направление, остаток, локальные часы и состояние переключения. Проверены свежая история и сохранение всей истории. Эти сравнения исследовательские."
               if ru else "Each block contains 1000 complete-policy selection rounds, corresponding to 50,000 internal steps across reset episodes. Horizon 3000 contains three blocks. Calibration is preserved; direction, residual, local clock and switch state are reset. Both fresh and retained history are tested. These comparisons are exploratory."),
              "",
              ("Детерминированные чередующиеся пути совпадают; их повторение не увеличивает независимую вариативность путей." if ru else "Deterministic alternating paths are identical; repetitions do not increase independent path variation."),
              ""]
    scenarios = list(analysis["restarts"])
    lines += ["| " + ("Метод" if ru else "Method") + " | " + " | ".join(scenarios) + " |", "|---|" + "---:|" * len(scenarios)]
    names = list(analysis["restarts"][scenarios[0]]["methods"])
    for name in names:
        lines.append("| " + name + " | " + " | ".join(f'{means(analysis["restarts"][s])[name]:.5f}' for s in scenarios) + " |")
    lines += [""]
    for scenario in scenarios:
        group = analysis["restarts"][scenario]
        switches = {name: value["switch_count"] for name, value in group["methods"].items()}
        restarts = {name: value["restart_count"] for name, value in group["methods"].items()}
        lines.append(scenario + ": " + ("переключения " if ru else "switch counts ") + json.dumps(switches, ensure_ascii=False) + "; " + ("перезапуски " if ru else "restart counts ") + json.dumps(restarts, ensure_ascii=False) + ".")
        lines.append("" )
        lines += [("| Перезапуск минус непрерывный вариант | Разность | Описательный интервал 95% |" if ru else "| Restart minus uninterrupted variant | Difference | Descriptive 95% interval |"), "|---|---:|---:|"]
        for name, value in group["contrasts"].items():
            lines.append(f'| {name} | {value["estimate"]:.5f} | [{value["lower"]:.5f}, {value["upper"]:.5f}] |')
        lines.append("")
    lines += ["",
              ("Перезапуск не меняет теорему исходного непрерывного алгоритма. Для него нужна оценка суммы сегментов; при фиксированной длине блока исходная скорость по общему горизонту не переносится. Сброс одного счётчика без переключений не меняет действия. При трёх известных чистых режимах новизна ограничена, поэтому ожидать эффекта от одного сброса бюджета не следует."
               if ru else "Restarting does not modify the theorem for the original uninterrupted algorithm. A restarted variant needs a sum-of-segments bound; a fixed block length does not inherit the original global-horizon rate. Resetting only a counter changes no actions when no switch occurs. Novelty is bounded for three known pure modes, so counter resets alone need not have an effect."),
              "", "## " + ("Воспроизводимость и ограничения" if ru else "Reproducibility and limitations"), "",
              ("Это симулятор с известной обучающей моделью и раскрытием режима атаки после выбора защиты. Смеси — ожидаемые результаты целых эпизодов; геометрия векторной цели в этом расширении не оценивается. Выигрыш по скалярной метрике не является новой теоремой."
               if ru else "This is a simulator experiment with a known training model and disclosure of the attack mode after defense selection. Mixtures represent expected complete-episode outcomes; vector target geometry is not evaluated in this extension. A scalar improvement is not a new theorem."),
              "",
              ("[Методика](../../docs/cage_adaptation_ru.md) · [Теория перезапусков](../../docs/cage_restart_theory_ru.md)"
               if ru else "[Methodology](../../docs/cage_adaptation_en.md) · [Restart theory](../../docs/cage_restart_theory_en.md)"),
              "",
              "[Protocol](protocol.json) · [Locked selection](selection.json) · [Validation scores](validation_grid.json) · [Analysis](analysis.json) · [Group aggregates](groups/) · [Input banks](../../data/cage2_adaptation/)",
              "",
              "![Primary comparisons](primary_comparisons.png)", "",
              "![Restart means](restart_losses.png)", "",
              "![Validation grid](validation_grid.png)", ""]
    return "\n".join(lines)

def build(run_dir, bank_root, output, protocol_path):
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("Frozen protocol changed.")
    protocol = read(protocol_path)
    analysis_path = run_dir / "analysis.json"
    selection_path = run_dir / "selection.json"
    analysis, selection = read(analysis_path), read(selection_path)
    if analysis["selection_sha256"] != sha(selection_path):
        raise ValueError("Final analysis selection hash does not match lock.")
    if selection["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Selection used a different protocol.")
    frozen = datetime.fromisoformat(selection["frozen_at"].replace("Z", "+00:00"))
    created = datetime.fromisoformat(analysis["analysis_created_utc"].replace("Z", "+00:00"))
    if frozen >= created:
        raise ValueError("Selection must be locked before final analysis.")
    if sha(REPO / "data/cage2/calibration.npz") != protocol["training"]["calibration_npz_sha256"]:
        raise ValueError("Original calibration changed.")
    for key, spec in protocol["banks"].items():
        name = "validation50" if key == "validation" else "test50"
        bank = load_episode_bank(bank_root / name)
        if bank["seeds"].tolist() != list(range(spec["seed_start"], spec["seed_start"] + spec["count"])):
            raise ValueError("Seed bank differs from protocol.")
        if bank["manifest"]["context"]["episode_steps"] != spec["episode_steps"]:
            raise ValueError("Episode length differs from protocol.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("selection.json", "selection.sha256.json", "validation_grid.json", "analysis.json"):
        if b"\r\n" in (run_dir / name).read_bytes():
            raise ValueError("Hash-bound runner JSON must use canonical LF bytes.")
        shutil.copy2(run_dir / name, output / name)
    shutil.copy2(protocol_path, output / "protocol.json")
    groups = output / "groups"
    groups.mkdir(exist_ok=True)
    for name in ("validation", "test", "restarts_curriculum", "restarts_alternating500"):
        dest = groups / name
        dest.mkdir(exist_ok=True)
        for filename in ("occupancies.npz", "meta.json", "training_fit.npz", "cache.json", "cache_prepare.json"):
            src = run_dir / name / filename
            if src.is_file():
                shutil.copy2(src, dest / filename)
    data_root = REPO / "data/cage2_adaptation"
    for name in ("validation50", "test50"):
        dest = data_root / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ("bank.npz", "manifest.json", "cell_statistics.json"):
            shutil.copy2(bank_root / name / filename, dest / filename)
        write(dest / "cell_statistics.json", read(dest / "cell_statistics.json"))
        manifest = read(dest / "manifest.json")
        manifest["statistics_sha256"] = sha(dest / "cell_statistics.json")
        write(dest / "manifest.json", manifest)
    grid = read(run_dir / "validation_grid.json")
    comparisons, checkpoint_hashes = [], {}
    for path in sorted((run_dir / "test/paths").glob("seed_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            names = data["methods"].tolist()
            left = data[f'method_{names.index("selected_scalar_one_switch")}_actions']
            right = data[f'method_{names.index("selected_window")}_actions']
            comparisons.append(float(np.max(np.abs(left[1:] - right[1:]))))
        checkpoint_hashes[path.name] = sha(path)
    if len(comparisons) != protocol["primary_test"]["path_count"]:
        raise ValueError("Expected all final checkpoints for action diagnostics.")
    trace_diagnostics = {"path_count": len(comparisons),
                         "max_action_difference_after_first_round": max(comparisons),
                         "test_checkpoint_sha256": checkpoint_hashes,
                         "interpretation": "Post-selection descriptive trace check, not another tuned configuration"}
    write(output / "trace_diagnostics.json", trace_diagnostics)
    figures(analysis, grid, output)
    for ru, filename in ((False, "README.md"), (True, "README.ru.md")):
        (output / filename).write_bytes(markdown(analysis, selection, protocol, ru, trace_diagnostics).encode("utf-8"))
    for path in output.rglob("*.json"):
        if b"\r\n" in path.read_bytes():
            raise ValueError("Published hash-bound JSON must use canonical LF bytes.")
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    write(output / "SHA256SUMS.json", {"algorithm": "sha256", "files": files})
    print(f"Published {len(files)} checked report artifacts to {output}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "results/runs/cage_adaptation/selection")
    parser.add_argument("--bank-root", type=Path, default=REPO / "results/runs/cage_adaptation/banks")
    parser.add_argument("--output", type=Path, default=REPO / "results/cage_adaptation")
    parser.add_argument("--protocol", type=Path, default=REPO / "docs/cage_adaptation_protocol.json")
    args = parser.parse_args()
    build(args.run_dir, args.bank_root, args.output, args.protocol)

if __name__ == "__main__":
    main()
