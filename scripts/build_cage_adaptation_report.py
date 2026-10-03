"""Build the focused CAGE 2 report from frozen primary-test aggregates.

This is a presentation projection of the completed, archived study. It does not
select new parameters, discard test paths, or rerun the simulator. Either the
original ignored runner directory or the committed report directory can be used
as --run-dir; the latter reproduces the report from public aggregate inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.collect_cage_bank import load_episode_bank

PROTOCOL_SHA = "a1c8976415eb867bb179dfada83c4a4e9fa0b8667f2c1883aa4044f199fb7de3"
ARCHIVE_COMMIT = "e8c85cb2e7be49032c1c2d014b6f19084ea064a8"
ARCHIVE_BRANCH = "archive/full-study-2026-10-04"
ARCHIVE_URL = ("https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/"
               + ARCHIVE_COMMIT)
PRIMARY_METHODS = ("selected_scalar_one_switch", "selected_window", "selected_hedge")
PRIMARY_REFERENCES = ("selected_window", "selected_hedge")
DISPLAY_LABELS = {
    False: ("Our one-switch (W=16, rho=0.25)", "Window (W=16)", "Hedge (eta multiplier=4)"),
    True: ("Наш one-switch (W=16, rho=0,25)", "Окно (W=16)", "Hedge (множитель eta=4)"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".report-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path, obj):
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def group_directory(run_dir, name):
    direct = run_dir / name
    return direct if direct.is_dir() else run_dir / "groups" / name


def focused_analysis(analysis, source_sha256):
    """Keep the predeclared comparisons; record that this is a post-study projection."""
    result = deepcopy(analysis)
    result.pop("restarts", None)
    result["primary"]["methods"] = {
        name: deepcopy(analysis["primary"]["methods"][name]) for name in PRIMARY_METHODS
    }
    result["primary"]["contrasts"] = {
        name: deepcopy(analysis["primary"]["contrasts"][name]) for name in PRIMARY_REFERENCES
    }
    if "publication_scope" not in result:
        result["publication_scope"] = {
            "type": "post-study primary presentation projection",
            "source_full_analysis_sha256": source_sha256,
            "archive_commit": ARCHIVE_COMMIT,
            "archive_branch": ARCHIVE_BRANCH,
            "archive_url": ARCHIVE_URL,
            "displayed_methods": list(PRIMARY_METHODS),
            "displayed_contrasts": list(PRIMARY_REFERENCES),
            "primary_paths_retained": analysis["primary"]["path_count"],
            "test_aggregate": "Exact original six-method aggregate retained as provenance; no test path omitted",
            "selection": "Original validation selection is unchanged; no retuning after final test",
            "omitted_from_presentation": "Untuned descriptive comparisons and exploratory restart studies remain in the archive",
            "window_equivalence_claim": False,
            "window_superiority_claim": False,
        }
    return result


def check_aggregates(run_dir, bank, analysis, selection, protocol):
    """Independently reconstruct displayed means from every held-out seed and path."""
    validation_dir = group_directory(run_dir, "validation")
    test_dir = group_directory(run_dir, "test")
    if sha(validation_dir / "occupancies.npz") != selection["validation_aggregate_sha256"]:
        raise ValueError("Validation aggregate differs from the locked selection.")
    meta = read(test_dir / "meta.json")
    if meta["status"] != "complete":
        raise ValueError("Primary aggregate is incomplete.")
    if sha(test_dir / "occupancies.npz") != meta["aggregate_sha256"]:
        raise ValueError("Primary aggregate checksum mismatch.")
    spec = protocol["primary_test"]
    expected_seeds = list(range(spec["path_seed_start"], spec["path_seed_start"] + spec["path_count"]))
    with np.load(test_dir / "occupancies.npz", allow_pickle=False) as data:
        if data["seeds"].tolist() != expected_seeds or meta["path_seeds"] != expected_seeds:
            raise ValueError("Report must retain every predeclared primary path.")
        names = data["methods"].tolist()
        occupancy = data["occupancies"]
        if np.min(occupancy) < -1e-12 or not np.allclose(occupancy.sum(axis=(2, 3)), 1, atol=1e-10, rtol=0):
            raise ValueError("Primary occupancy arrays are not valid joint distributions.")
        reconstructed = np.einsum("pmbr,ebrc->pmec", occupancy, bank["episode_losses"]).mean(axis=(0, 2))
        for name in PRIMARY_METHODS:
            components = reconstructed[names.index(name)]
            original = analysis["primary"]["methods"][name]
            if not np.allclose(components, original["mean_components"], atol=1e-12, rtol=0):
                raise ValueError("Primary reported component means do not match full aggregate.")
            if abs(float(components.sum()) - original["mean_native_loss"]) > 1e-12:
                raise ValueError("Primary reported scalar means do not match full aggregate.")
    if meta["horizon"] != spec["horizon"] or analysis["primary"]["path_count"] != spec["path_count"]:
        raise ValueError("Primary horizon or path count differs from protocol.")
    return meta


def action_diagnostics(run_dir, protocol, meta):
    test_dir = group_directory(run_dir, "test")
    paths = sorted((test_dir / "paths").glob("seed_*.npz"))
    if paths:
        comparisons, checkpoint_hashes = [], {}
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                names = data["methods"].tolist()
                left = data[f'method_{names.index("selected_scalar_one_switch")}_actions']
                right = data[f'method_{names.index("selected_window")}_actions']
                comparisons.append(float(np.max(np.abs(left[1:] - right[1:]))))
            checkpoint_hashes[path.name] = sha(path)
        diagnostic = {
            "path_count": len(comparisons),
            "max_action_difference_after_first_round": max(comparisons),
            "test_checkpoint_sha256": checkpoint_hashes,
            "interpretation": "Post-selection descriptive trace check, not another tuned configuration",
        }
    else:
        diagnostic = read(run_dir / "trace_diagnostics.json")
    if diagnostic["path_count"] != protocol["primary_test"]["path_count"]:
        raise ValueError("Action diagnostic must cover every primary path.")
    if diagnostic["test_checkpoint_sha256"] != meta["path_sha256"]:
        raise ValueError("Action diagnostic hashes do not match completed primary checkpoints.")
    return diagnostic


def save_figure(fig, output, name, created):
    """Fix PDF dates so aggregate-only regeneration produces stable artifacts."""
    for suffix in ("png", "pdf"):
        metadata = ({"Software": "Matplotlib"} if suffix == "png" else
                    {"Creator": "CAGE 2 focused report", "CreationDate": created, "ModDate": created})
        with tempfile.NamedTemporaryFile(dir=output, prefix=".figure-", suffix="." + suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, dpi=180, bbox_inches="tight", metadata=metadata)
            temporary.replace(output / (name + "." + suffix))
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def figures(analysis, output, trace):
    primary = analysis["primary"]
    means = np.array([primary["methods"][name]["mean_native_loss"] for name in PRIMARY_METHODS])
    ours = means[0]
    hedge_improvement = 100 * (means[2] - ours) / means[2]
    window_disadvantage = 100 * (ours - means[1]) / means[1]
    created = datetime.fromisoformat(analysis["analysis_created_utc"].replace("Z", "+00:00"))
    for ru in (False, True):
        ending = ".ru" if ru else ""
        fig, ax = plt.subplots(figsize=(9, 4.1))
        y = np.arange(len(PRIMARY_METHODS))
        ax.barh(y, means, height=.54, color=["#267b7a", "#64748b", "#8895a3"])
        ax.set_yticks(y, DISPLAY_LABELS[ru], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15 * float(means.max()))
        for i, value in enumerate(means):
            ax.text(value + .016, i, f"{value:.5f}", va="center", fontsize=11)
        ax.set_xlabel("Средние потери на шаг симулятора; меньше — лучше" if ru else
                      "Mean native loss per simulator step; lower is better")
        ax.set_title("Основной тест: 400 семян, 50 путей, горизонт 512" if ru else
                     "Primary test: 400 seeds, 50 paths, horizon 512", pad=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#ddd", alpha=.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save_figure(fig, output, "primary_means" + ending, created)

        fig, ax = plt.subplots(figsize=(10.2, 4.8))
        colors = ["#b05e2b", "#267b7a"]
        for i, name in enumerate(PRIMARY_REFERENCES):
            value = primary["contrasts"][name]
            estimate, low, high = value["estimate"], value["lower"], value["upper"]
            ax.errorbar(estimate, i, xerr=np.array([[estimate-low], [high-estimate]]),
                        fmt="o", color=colors[i], capsize=5, markersize=7, linewidth=2)
            ax.annotate(f"{estimate:+.5f} [{low:+.5f}, {high:+.5f}]", (estimate, i),
                        xytext=(0, 20), textcoords="offset points", ha="center", fontsize=10,
                        color=colors[i])
        ax.axvline(0, color="#555", linewidth=1.1)
        ax.set_yticks([0, 1], ["Окно (W=16)", "Hedge (множитель eta=4)"] if ru else
                      ["Window (W=16)", "Hedge (eta multiplier=4)"], fontsize=11)
        ax.set_ylim(1.55, -.65)
        ax.set_xlim(-.108, .045)
        ax.set_xlabel("Наш метод минус метод сравнения: потери на шаг симулятора\nОтрицательная разность означает преимущество нашего метода" if ru else
                      "Our method minus reference: native loss per simulator step\nA negative difference favors our method")
        ax.set_title("Два предзаданных сравнения: одновременные интервалы 95%" if ru else
                     "Two predeclared comparisons: simultaneous 95% intervals", pad=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#ddd", alpha=.6)
        if ru:
            caption = (f"Потери ниже Hedge на {hedge_improvement:.2f}%; выше окна на {window_disadvantage:.2f}%.\n"
                       "10 000 парных перекрёстных бутстрэп-выборок; калибровка и параметры фиксированы.")
            if trace["max_action_difference_after_first_round"] == 0:
                caption += "\nВо всех 50 путях действия нашего метода и окна совпадают с раунда 2."
        else:
            caption = (f"Loss is {hedge_improvement:.2f}% lower than Hedge; {window_disadvantage:.2f}% higher than window.\n"
                       "10,000 paired crossed bootstrap draws; calibration and configurations held fixed.")
            if trace["max_action_difference_after_first_round"] == 0:
                caption += "\nOur method and window take identical actions from round 2 in all 50 paths."
        fig.text(.5, .012, caption, ha="center", va="bottom", fontsize=9.3)
        fig.tight_layout(rect=(0, .19, 1, 1))
        save_figure(fig, output, "primary_comparisons" + ending, created)


def markdown(analysis, selection, protocol, ru, trace, supplementary=False):
    names = dict(zip(PRIMARY_METHODS, DISPLAY_LABELS[ru]))
    primary = analysis["primary"]
    mean = {name: primary["methods"][name]["mean_native_loss"] for name in PRIMARY_METHODS}
    improvement = 100 * (mean["selected_hedge"] - mean["selected_scalar_one_switch"]) / mean["selected_hedge"]
    disadvantage = 100 * (mean["selected_scalar_one_switch"] - mean["selected_window"]) / mean["selected_window"]
    ending = ".ru" if ru else ""
    lines = ["[English](README.md) | [Русский](README.ru.md)", "",
             "# " + ("CAGE 2: выбранный one-switch и основные методы сравнения" if ru else
                        "CAGE 2: selected one-switch and primary references"), ""]
    lines += [(f"Наш one-switch с оракулом, учитывающим скалярные потери (W=16, rho=0,25), даёт на {improvement:.2f}% меньшие средние потери, чем настроенный Hedge. От настроенного окна он отстаёт на {disadvantage:.2f}%. Оба вывода поддерживаются предзаданными одновременными интервалами. В этом эксперименте превосходство над окном и статистическая эквивалентность не установлены."
               if ru else
               f"Our one-switch with a scalar-aware oracle (W=16, rho=0.25) has {improvement:.2f}% lower mean loss than tuned Hedge and {disadvantage:.2f}% higher mean loss than tuned window. Both findings are supported by the predeclared simultaneous intervals. This experiment establishes neither superiority over window nor statistical equivalence."), "",
              ("Метрика — сумма четырёх нативных потерь на шаг симулятора: компрометация хостов, компрометация сервера, нарушение работы и восстановление. Меньше означает лучше."
               if ru else "The metric is the sum of four native loss components per simulator step: host compromise, server compromise, operational disruption, and restore cost. Lower is better."), "",
              "## " + ("Основной тест" if ru else "Primary test"), "",
              ("Параметры всех трёх методов выбраны на отдельных данных и зафиксированы до чтения финальных результатов. Валидация: 100 семян симулятора и 20 общих путей. Финальный тест: 400 новых семян и все 50 предзаданных новых путей, горизонт 512. Для двух банков собрано 9000 новых 50-шаговых эпизодов. Каждая пара из шести защит и трёх режимов атаки оценена на каждом семени."
               if ru else "All three configurations were selected on separate data and locked before final outcomes were read. Validation uses 100 simulator seeds and 20 common paths. The final test uses 400 new seeds and all 50 predeclared new paths at horizon 512. The two banks contain 9000 new 50-step episodes. Every pair of six defenses and three attack modes is evaluated on every simulator seed."), "",
              ("| Метод | Средние потери |" if ru else "| Method | Mean loss |"), "|---|---:|"]
    for name in PRIMARY_METHODS:
        lines.append(f"| {names[name]} | {mean[name]:.5f} |")
    lines += ["", ("| Наш метод минус метод сравнения | Разность | Одновременный интервал | Вывод |" if ru else
                   "| Our method minus reference | Difference | Simultaneous interval | Finding |"), "|---|---:|---:|---|"]
    for name in PRIMARY_REFERENCES:
        value = primary["contrasts"][name]
        if value["upper"] < 0:
            decision = "Наш метод лучше" if ru else "Our method has lower loss"
        elif value["lower"] > 0:
            decision = "Наш метод хуже" if ru else "Our method has higher loss"
        else:
            decision = "Различие не подтверждено" if ru else "Difference inconclusive"
        lines.append(f'| {names[name]} | {value["estimate"]:+.5f} | [{value["lower"]:+.5f}, {value["upper"]:+.5f}] | {decision} |')
    lines += ["", ("Интервалы приближённые: 10 000 парных перекрёстных бутстрэп-выборок с семейным уровнем 95% для двух сравнений (поправка Бонферрони). Перевыбираются целиком таблицы семян симулятора и целиком общие пути, сохраняя зависимость внутри них. Вывод условен на исходной калибровке и зафиксированных параметрах; неопределённость их выбора и обучения не включена."
                   if ru else "The intervals are approximate: 10,000 paired crossed bootstrap draws with simultaneous family level 95% for two comparisons (Bonferroni correction). Whole simulator-seed tables and whole common paths are resampled, preserving dependence within them. Inference is conditional on the original calibration and locked configurations; calibration and selection uncertainty are excluded."), "",
              f"![{'Средние потери' if ru else 'Mean losses'}](primary_means{ending}.png)", "",
              ("График средних начинается с нуля. Неопределённость предзаданных разностей показана отдельно ниже."
               if ru else "The mean-loss chart starts at zero. Uncertainty of the predeclared differences is shown separately below."), "",
              f"![{'Предзаданные сравнения' if ru else 'Predeclared comparisons'}](primary_comparisons{ending}.png)", "",
              "## " + ("Почему результат близок к окну" if ru else "Why performance is close to window"), ""]
    if trace["max_action_difference_after_first_round"] == 0:
        lines += [("Во всех 50 финальных путях действия нашего метода и окна в точности совпадают со второго раунда. Оба используют прогноз по последним 16 уже раскрытым режимам атаки и одну обученную таблицу потерь. Наш метод дополнительно соблюдает условие допустимости оракула; на этих данных оно не меняет последующие действия."
                   if ru else "In all 50 final paths, our method and window take exactly the same actions from round 2. Both use a forecast from the last 16 already disclosed attack modes and the same trained loss table. Our method additionally enforces the oracle admissibility condition; it does not alter subsequent actions on these data."), "",
                  ("Разность потерь целиком объясняется первым равномерным ходом нашего алгоритма. Окно сразу отвечает на известный начальный режим. Изменение первого хода не входило в зафиксированную настройку; параметры и ход не изменялись после просмотра теста. Диагностика совпадения действий описательная и выполнена после выбора параметров."
                   if ru else "The loss difference is entirely due to our algorithm's first uniform action. Window immediately responds to the known initial mode. Changing the first action was outside the locked tuning plan; neither the parameters nor this action were changed after test inspection. The action-equality diagnostic is descriptive and was performed after configuration selection."), ""]
    lines += ["## " + ("Реализация и границы вывода" if ru else "Implementation and limits"), "",
              ("Используется непрерывный one-switch без перезапусков: прогнозное окно W=16, доля допустимого зазора rho=0,25. Окно также выбрано с W=16; Hedge — с множителем скорости обучения 4. Полная сетка выбора сохранена. В основном тесте не было безопасных переключений, возвратов к исходному оракулу или превышений номинального допуска оракула. Численная проверка остатка служит численным свидетельством, а не сертификатом в точной арифметике."
               if ru else "The implementation is continuous one-switch without restarts: forecast window W=16 and permitted gap fraction rho=0.25. Window is also selected with W=16; Hedge uses learning-rate multiplier 4. The full selection grid is retained. The primary test has no safe switches, scalar-oracle fallbacks, or nominal oracle-contract violations. Numerical residual checks provide numerical evidence rather than an exact-arithmetic certificate."), "",
              ("Это эксперимент в симуляторе с известной обучающей моделью и раскрытием режима атаки после выбора защиты. Смеси обозначают ожидаемые исходы полных независимо инициализированных эпизодов. Противник выбирает среди трёх зафиксированных режимов; новые тактики не изучает. Основные статистические сравнения относятся к скалярным потерям. Геометрическая ошибка в исходном основном анализе не вычислялась; дополнительные численные проверки калиброванной игры представлены отдельно. Близость скалярных потерь к окну сама по себе не демонстрирует практическую пользу векторной гарантии статьи. Результат не означает общего превосходства над другими методами CAGE или на живой сети."
               if ru else "This is a simulator experiment with a known training model and attack-mode disclosure after defense selection. Mixtures represent expected outcomes of whole independently reset episodes. The attacker selects among three frozen modes and learns no new tactics. The primary statistical comparisons concern scalar loss. Geometric error was not computed in the original primary analysis; supplementary numerical checks of the calibrated game are presented separately. Scalar loss close to window does not by itself demonstrate the practical benefit of the paper's vector guarantee. These results do not establish general superiority over other CAGE methods or on a live network."), "",
              "## " + ("Воспроизводимость и архив" if ru else "Reproducibility and archive"), "",
              ("Это сокращённое представление уже завершённого исследования, созданное после просмотра его результатов. Состав основного теста, все 50 путей, параметры и два предзаданных сравнения сохранены. Побочные проверки перезапусков и ненастроенных вариантов доступны в полном архиве."
               if ru else "This is a focused presentation of an already completed study, created after its outcomes were inspected. The primary test, all 50 paths, configurations, and both predeclared comparisons are preserved. Exploratory restart checks and untuned variants remain available in the full archive."), "",
              f"[{'Полный архив исследования' if ru else 'Full study archive'}]({ARCHIVE_URL}/results/cage_adaptation) · `archive/full-study-2026-10-04` ({'ветка' if ru else 'branch'})", "",
              ("Сохранённый основной NPZ содержит все шесть первоначально оценённых методов как данные происхождения. Отчёт и графики показывают только три метода, выбранных на валидации. `analysis.json` явно записывает это сокращение и SHA-256 полного исходного анализа; исходные NPZ, protocol и selection не подменяются."
               if ru else "The retained primary NPZ contains all six originally evaluated methods as provenance data. This report and its plots display only the three validation-selected methods. `analysis.json` explicitly records this projection and the full source-analysis SHA-256; the original NPZ, protocol, and selection are not replaced."), "",
              ("[Методика](../../docs/cage_adaptation_ru.md)" if ru else "[Methodology](../../docs/cage_adaptation_en.md)"), "",
              "[Protocol](protocol.json) · [Locked selection](selection.json) · [Selection checksum](selection.sha256.json) · [Full validation grid](validation_grid.json) · [Primary analysis](analysis.json) · [Action diagnostic](trace_diagnostics.json) · [Aggregate inputs](groups/) · [Episode banks](../../data/cage2_adaptation/) · [Artifact checksums](SHA256SUMS.json)", "",
              ("Для пересоздания этого отчёта из опубликованных агрегатов без запуска симулятора:"
               if ru else "To regenerate this report from published aggregates without running the simulator:"), "", "```powershell",
              "python scripts/build_cage_adaptation_report.py --run-dir results/cage_adaptation --bank-root data/cage2_adaptation --output results/runs/focused_report", "```", "",
              ("PDF для статьи: [средние](primary_means.ru.pdf), [разности и интервалы](primary_comparisons.ru.pdf)."
               if ru else "PDF figures: [means](primary_means.pdf), [differences and intervals](primary_comparisons.pdf)."), ""]
    if supplementary:
        lines += ["## " + ("Дополнительные графики" if ru else "Supplementary figures"), "",
                  ("[Галерея графиков и PDF](figures/README.ru.md): динамика потерь, накопленная разность, четыре компоненты, разброс по траекториям, валидационная чувствительность и численная ошибка до полного векторного целевого множества."
                   if ru else "[Figure gallery and PDFs](figures/README.md): loss dynamics, cumulative differences, four components, path variability, validation sensitivity, and numerical distance to the full vector target."), "",
                  ("Эти проверки описательные и добавлены после завершения основного теста. Параметры и два исходных статистических сравнения сохранены."
                   if ru else "These checks are descriptive and were added after the primary test. The locked parameters and two original statistical comparisons are preserved."), ""]
    lines += ["## " + ("Материалы AAMAS" if ru else "AAMAS paper materials"), "",
              ("[Английские графики для статьи](aamas/README.ru.md) содержат накопленную разницу потерь, основные сравнения с компонентами и векторную геометрию. [Раздел LaTeX](../../paper/README.ru.md) предлагает таблицу и накопленную разность для основного текста. [Требования к подаче](../../docs/aamas_submission_ru.md) описывают формат AAMAS 2027 и анонимный пакет."
               if ru else "[English publication figures](aamas/README.md) include cumulative cost differences, primary contrasts with components, and vector geometry. [The LaTeX section](../../paper/README.md) recommends a table and cumulative comparison for main text. [Submission guidance](../../docs/aamas_submission_en.md) documents AAMAS 2027 formatting and the anonymous package."), ""]
    return "\n".join(lines)


def cleanup_legacy_report(output):
    """Remove only known report-owned side artifacts inside the verified output root."""
    root = output.resolve()
    for name in ("restart_losses.png", "restart_losses.pdf", "validation_grid.png", "validation_grid.pdf"):
        target = (root / name).resolve()
        if target.parent != root:
            raise ValueError("Legacy report path escaped its output directory.")
        target.unlink(missing_ok=True)
    for name in ("restarts_curriculum", "restarts_alternating500"):
        target = (root / "groups" / name).resolve()
        if not target.is_relative_to(root / "groups"):
            raise ValueError("Legacy restart group escaped the report output.")
        if target.is_dir():
            # Exact known filenames: no computed recursive deletion.
            for filename in ("occupancies.npz", "meta.json", "training_fit.npz", "cache.json", "cache_prepare.json"):
                (target / filename).unlink(missing_ok=True)
            if any(target.iterdir()):
                raise ValueError("Unexpected restart files remain; refusing to delete them.")
            target.rmdir()


def build(run_dir, bank_root, output, protocol_path):
    run_dir, bank_root, output = Path(run_dir), Path(bank_root), Path(output)
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError("Frozen protocol changed.")
    protocol = read(protocol_path)
    analysis_path, selection_path = run_dir / "analysis.json", run_dir / "selection.json"
    original, selection = read(analysis_path), read(selection_path)
    if original["selection_sha256"] != sha(selection_path):
        raise ValueError("Final analysis selection hash does not match lock.")
    if selection["protocol_sha256"] != PROTOCOL_SHA or original["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Selection or analysis used a different protocol.")
    if original["selected"] != selection["selected"]:
        raise ValueError("Displayed configurations differ from validation selection.")
    frozen = datetime.fromisoformat(selection["frozen_at"].replace("Z", "+00:00"))
    created = datetime.fromisoformat(original["analysis_created_utc"].replace("Z", "+00:00"))
    if frozen >= created:
        raise ValueError("Selection must be locked before final analysis.")
    if sha(REPO / "data/cage2/calibration.npz") != protocol["training"]["calibration_npz_sha256"]:
        raise ValueError("Original calibration changed.")
    banks = {}
    for key, spec in protocol["banks"].items():
        name = "validation50" if key == "validation" else "test50"
        bank = load_episode_bank(bank_root / name)
        if bank["seeds"].tolist() != list(range(spec["seed_start"], spec["seed_start"] + spec["count"])):
            raise ValueError("Seed bank differs from protocol.")
        if bank["manifest"]["context"]["episode_steps"] != spec["episode_steps"]:
            raise ValueError("Episode length differs from protocol.")
        expected_sha = original["validation_bank_sha256" if key == "validation" else "final_bank_sha256"]
        if sha(bank_root / name / "bank.npz") != expected_sha:
            raise ValueError("Input bank differs from the completed analysis.")
        banks[key] = bank
    meta = check_aggregates(run_dir, banks["test"], original, selection, protocol)
    trace = action_diagnostics(run_dir, protocol, meta)
    analysis = focused_analysis(original, sha(analysis_path))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("selection.json", "selection.sha256.json", "validation_grid.json"):
        content = (run_dir / name).read_bytes()
        if b"\r\n" in content:
            raise ValueError("Hash-bound runner JSON must use canonical LF bytes.")
        atomic_bytes(output / name, content)
    atomic_bytes(output / "protocol.json", Path(protocol_path).read_bytes())
    for name in ("validation", "test"):
        source, destination = group_directory(run_dir, name), output / "groups" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("occupancies.npz", "meta.json", "training_fit.npz"):
            atomic_bytes(destination / filename, (source / filename).read_bytes())
    # A public-input rebuild also preserves the separately reproducible
    # supplements. They do not change the locked primary analysis.
    for name in ("geometry", "dynamics", "figures", "aamas"):
        source, destination = run_dir / name, output / name
        if source.is_dir() and source.resolve() != destination.resolve():
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    atomic_bytes(destination / path.relative_to(source), path.read_bytes())
    write(output / "analysis.json", analysis)
    write(output / "trace_diagnostics.json", trace)
    figures(analysis, output, trace)
    for ru, filename in ((False, "README.md"), (True, "README.ru.md")):
        supplementary = (output / "figures" / "README.md").exists()
        atomic_bytes(output / filename, markdown(analysis, selection, protocol, ru, trace,
                                                 supplementary=supplementary).encode("utf-8"))
    cleanup_legacy_report(output)
    for path in output.rglob("*.json"):
        # Preserve the exact pre-sweep fingerprint of the supplementary source
        # ledger, which was originally serialized with Windows line endings.
        # Its own frozen protocol checks the bytes; normalizing it would break
        # provenance. Primary runner JSON and all other report JSON remain LF.
        historical_source = path.relative_to(output).as_posix() == "geometry/primary_sources.json"
        if b"\r\n" in path.read_bytes() and not historical_source:
            raise ValueError("Published hash-bound JSON must use canonical LF bytes.")
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
             if p.is_file() and p.name != "SHA256SUMS.json"}
    write(output / "SHA256SUMS.json", {"algorithm": "sha256", "files": files})
    print(f"Published {len(files)} primary report artifacts; all 50 primary paths retained: {output}")


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
