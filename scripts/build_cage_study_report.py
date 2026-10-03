"""Build a checked bilingual report from the completed frozen CAGE study.

No experiments or calibration updates are performed here. Incomplete banks or
selection groups are rejected before any published report files are written.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.collect_cage_bank import load_episode_bank
from lowdim_games.cage import BLUE_NAMES, RED_NAMES, METRIC_NAMES, load_cage_calibration
from lowdim_games.cage_study import holm_adjust, paired_two_way_bootstrap
from lowdim_games.experiment import software_versions, tensor_hash, write_json


FROZEN_PROTOCOL_SHA256 = "9a3ecc45704e42976c3fd3a2333b0b0ec13e1310fb4b2b5e57b65be0ad08357a"
METHODS = ("one_switch", "shared_past_hull", "block_safe", "uniform",
           "historical_best", "last_window", "hedge", *[f"fixed_{name}" for name in BLUE_NAMES])
GROUPS = {
    "primary": ("primary50", 50, "primary"),
    "fixed_meander": ("primary50", 20, "fixed_meander"),
    "fixed_b_line": ("primary50", 20, "fixed_b_line"),
    "weak_to_strong": ("primary50", 20, "weak_to_strong"),
    "alternating_64": ("primary50", 20, "alternating_64"),
    "interactive": ("primary50", 20, "interactive"),
    "horizon128": ("primary50", 10, "horizon_128"),
    "horizon2048": ("primary50", 10, "horizon_2048"),
    "replication0": ("primary50", 10, "calibration_replication_0"),
    "replication1": ("primary50", 10, "calibration_replication_1"),
    "replication2": ("primary50", 10, "calibration_replication_2"),
    "steps100": ("step100", 10, "episode_100"),
    "steps150": ("step150", 10, "episode_150"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _t_interval(values, alpha=0.05):
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "lower": None, "upper": None, "samples": len(values)}
    half = float(student_t.ppf(1 - alpha / 2, len(values) - 1)
                 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "samples": len(values)}


def _absolute_crossed_interval(matrix, samples, seed, alpha=0.05):
    """A descriptive crossed interval for a mean, preserving complete clusters."""
    episode_count, path_count = matrix.shape
    episode_rng, path_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(2))
    draws = np.empty(samples)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        seed_weights = episode_rng.multinomial(episode_count, np.full(episode_count, 1 / episode_count), size=count) / episode_count
        path_weights = path_rng.multinomial(path_count, np.full(path_count, 1 / path_count), size=count) / path_count
        draws[start:start + count] = np.sum((seed_weights @ matrix) * path_weights, axis=1)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], method="linear")
    return {"estimate": float(matrix.mean()), "lower": float(lower), "upper": float(upper),
            "bootstrap_std": float(draws.std(ddof=1)), "alpha": alpha,
            "samples": samples, "seed": seed, "descriptive": True}


def _load_inputs(protocol_path, bank_root, selection_dir, calibration_dir):
    if sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("Frozen protocol checksum changed; report cannot silently change the study design.")
    protocol = _read_json(protocol_path)
    original = load_cage_calibration(calibration_dir)
    if original.provenance["calibration_npz_sha256"] != protocol["primary"]["calibration_npz_sha256"]:
        raise ValueError("Primary calibration differs from the pre-specified original fit.")
    specifications = {
        "primary50": (32000000, 400, 50), "calibration50": (33000000, 120, 50),
        "step100": (34000000, 140, 100), "step150": (35000000, 140, 150),
    }
    banks = {}
    used_seeds = set(original.train_seeds.tolist() + original.test_seeds.tolist())
    for key, (seed_start, count, steps) in specifications.items():
        values = load_episode_bank(bank_root / key)
        seeds = values["seeds"].tolist()
        if seeds != list(range(seed_start, seed_start + count)):
            raise ValueError(f"Bank {key} has an incomplete or unexpected seed list.")
        if set(seeds) & used_seeds:
            raise ValueError(f"Bank {key} overlaps an earlier calibration or pilot bank.")
        used_seeds.update(seeds)
        context = values["manifest"]["context"]
        if context["episode_steps"] != steps:
            raise ValueError(f"Bank {key} has another episode length.")
        if (tuple(values["blue_names"]) != BLUE_NAMES or tuple(values["red_names"]) != RED_NAMES
                or tuple(values["metric_names"]) != METRIC_NAMES):
            raise ValueError(f"Bank {key} has another frozen policy menu.")
        if context["source"]["revision"] != original.provenance["source"]["revision"]:
            raise ValueError(f"Bank {key} uses another simulator revision.")
        losses = values["episode_losses"]
        if (not np.isfinite(losses).all() or np.min(losses) < -1e-12
                or np.any(losses > values["component_bounds"] + 1e-10)):
            raise ValueError(f"Bank {key} violates the native component bounds.")
        if np.max(values["reward_errors"]) > 1e-8:
            raise ValueError(f"Bank {key} failed native reward reconstruction.")
        banks[key] = values
    groups = {}
    for group_name, (bank_key, count, seed_key) in GROUPS.items():
        folder = selection_dir / group_name
        metadata = _read_json(folder / "meta.json")
        if metadata.get("status") != "complete":
            raise ValueError(f"Selection group {group_name} is incomplete.")
        aggregate_path = folder / "occupancies.npz"
        if sha256(aggregate_path) != metadata["aggregate_sha256"]:
            raise ValueError(f"Selection group {group_name} aggregate checksum mismatch.")
        if metadata["protocol_sha256"].lower() != FROZEN_PROTOCOL_SHA256:
            raise ValueError(f"Selection group {group_name} used another protocol.")
        with np.load(aggregate_path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        if tuple(values["methods"].tolist()) != METHODS:
            raise ValueError(f"Selection group {group_name} does not contain the exact 13 methods.")
        if values["occupancies"].shape != (count, len(METHODS), len(BLUE_NAMES), len(RED_NAMES)):
            raise ValueError(f"Selection group {group_name} has another path count or shape.")
        start = (protocol["primary"]["path_seed_start"] if seed_key == "primary"
                 else protocol["secondary"]["path_seed_starts"][seed_key])
        if values["seeds"].tolist() != list(range(start, start + count)):
            raise ValueError(f"Selection group {group_name} has another path-seed list.")
        occupations = values["occupancies"]
        if (not np.isfinite(occupations).all() or np.min(occupations) < -1e-12
                or not np.allclose(occupations.sum(axis=(2, 3)), 1, atol=1e-8, rtol=0)):
            raise ValueError(f"Selection group {group_name} contains invalid occupancies.")
        if metadata["evaluation_bank_key"] != bank_key:
            raise ValueError(f"Selection group {group_name} has another evaluation-bank key.")
        if metadata["path_count"] != count:
            raise ValueError(f"Selection group {group_name} metadata path count differs.")
        expected_horizon = (int(group_name[len("horizon"):]) if group_name.startswith("horizon")
                            else protocol["primary"]["meta_horizon"])
        expected_steps = int(group_name[len("steps"):]) if group_name.startswith("steps") else 50
        if metadata["horizon"] != expected_horizon or metadata["episode_steps"] != expected_steps:
            raise ValueError(f"Selection group {group_name} has another horizon or episode length.")
        evaluation = banks[bank_key]
        test_start = 0 if bank_key == "primary50" else 40
        if metadata["test_seeds"] != evaluation["seeds"][test_start:].tolist():
            raise ValueError(f"Selection group {group_name} has another evaluation split.")
        expected_train = (original.train_seeds.tolist() if group_name not in ("replication0", "replication1", "replication2", "steps100", "steps150")
                          else banks["calibration50"]["seeds"][40 * int(group_name[-1]):40 * (int(group_name[-1]) + 1)].tolist()
                          if group_name.startswith("replication") else banks[bank_key]["seeds"][:40].tolist())
        if metadata["train_seeds"] != expected_train:
            raise ValueError(f"Selection group {group_name} has another calibration split.")
        if set(metadata["train_seeds"]) & set(metadata["test_seeds"]):
            raise ValueError(f"Selection group {group_name} leaks calibration into evaluation.")
        if group_name.startswith("replication"):
            index = int(group_name[-1])
            training = banks["calibration50"]["episode_losses"][40 * index:40 * (index + 1)]
            expected_tensor = training.mean(axis=0) / original.normalization_scale
        elif group_name.startswith("steps"):
            training = banks[bank_key]["episode_losses"][:40]
            expected_tensor = training.mean(axis=0) / original.normalization_scale
        else:
            expected_tensor = original.tensor
        if metadata["tensor_sha256"] != tensor_hash(expected_tensor):
            raise ValueError(f"Selection group {group_name} tensor differs from its declared calibration rows.")
        if sha256(folder / "training_fit.npz") != metadata["training_fit_sha256"]:
            raise ValueError(f"Selection group {group_name} training-fit checksum mismatch.")
        max_action_difference = 0.0
        for filename, expected_hash in metadata["checkpoint_sha256"].items():
            checkpoint = folder / "paths" / filename
            if sha256(checkpoint) != expected_hash:
                raise ValueError(f"Selection group {group_name} trajectory checksum mismatch.")
            with np.load(checkpoint, allow_pickle=False) as data:
                master = data[f"method_{METHODS.index('one_switch')}_actions"]
                fast = data[f"method_{METHODS.index('shared_past_hull')}_actions"]
                max_action_difference = max(max_action_difference, float(np.max(np.abs(master - fast))))
        groups[group_name] = {"metadata": metadata, "arrays": values,
                              "bank": evaluation["episode_losses"][test_start:],
                              "one_switch_fast_action_max_abs_difference": max_action_difference}
    return protocol, original, banks, groups


def _group_statistics(name, item, protocol):
    occupations = item["arrays"]["occupancies"]
    bank = item["bank"]
    seed = protocol["primary"]["bootstrap_seed"]
    samples = protocol["primary"]["bootstrap_samples"]
    components = np.einsum("pmij,eijd->epmd", occupations, bank, optimize=True)
    total = components.sum(axis=-1)
    references = protocol["primary"]["references"]
    learner_index = METHODS.index(protocol["primary"]["learner"])
    scores = []
    for index, method in enumerate(METHODS):
        matrix = total[:, :, index]
        scores.append({"method": method, "mean_native_loss": float(matrix.mean()),
                       "mean_components": dict(zip(METRIC_NAMES, components[:, :, index].mean(axis=(0, 1)))),
                       "native_loss_ci95_descriptive": _absolute_crossed_interval(matrix, samples, seed),
                       "conditional_episode_ci95": _t_interval(matrix.mean(axis=1)),
                       "conditional_path_ci95": _t_interval(matrix.mean(axis=0)),
                       "std_across_episode_seed_means": float(matrix.mean(axis=1).std(ddof=1)),
                       "std_across_path_means": float(matrix.mean(axis=0).std(ddof=1))})
    contrasts = {}
    primary = name == "primary"
    for reference in references:
        result = paired_two_way_bootstrap(
            occupations[:, learner_index], occupations[:, METHODS.index(reference)], bank,
            samples=samples, seed=seed,
            alpha=protocol["primary"]["simultaneous_ci_alpha"] if primary else 0.05,
            return_differences=False)
        result["reference"] = reference
        result["descriptive"] = not primary
        result["decision"] = ("supported_advantage" if result["upper"] < 0
                              else "supported_disadvantage" if result["lower"] > 0 else "inconclusive") if primary else "descriptive_only"
        contrasts[reference] = result
    if primary:
        adjusted = holm_adjust({key: value["approximate_two_sided_pvalue"] for key, value in contrasts.items()})
        for key, value in contrasts.items():
            value["holm_adjusted_pvalue"] = adjusted[key]
    arrays = item["arrays"]
    if "switch_rounds" not in arrays or arrays["switch_rounds"].shape != occupations.shape[:2]:
        raise ValueError(f"Group {name} is missing complete switch diagnostics.")
    switches = {method: int(np.sum(arrays["switch_rounds"][:, index] >= 0)) for index, method in enumerate(METHODS)}
    bit_equal = None
    if "actions_hashes" in arrays:
        bit_equal = bool(np.all(arrays["actions_hashes"][:, METHODS.index("one_switch")] == arrays["actions_hashes"][:, METHODS.index("shared_past_hull")]))
    max_action_difference = item.get("one_switch_fast_action_max_abs_difference")
    numerically_equal = None if max_action_difference is None else max_action_difference <= 1e-14
    unique_paths = None
    if "opponent_path_hashes" in arrays:
        unique_paths = {method: len(np.unique(arrays["opponent_path_hashes"][:, index])) for index, method in enumerate(METHODS)}
    metadata = dict(item["metadata"])
    metadata["effective_initial_red_index"] = 2 if name == "fixed_b_line" else 1
    metadata["initial_red_index_interpretation"] = "initial_red_index records the primary default; fixed_b_line overrides it with known B_line index 2 before any observations"
    summary = {"group": name, "metadata": metadata, "path_count": occupations.shape[0],
               "heldout_episode_seed_count": len(bank), "scores": scores,
               "contrasts": contrasts, "switch_counts": switches,
               "one_switch_and_fast_actions_bit_identical": bit_equal,
               "one_switch_and_fast_actions_numerically_equal": numerically_equal,
               "one_switch_fast_action_max_abs_difference": max_action_difference,
               "action_equivalence_absolute_tolerance": 1e-14,
               "distinct_realized_paths_by_method": unique_paths,
               "interactive_paths": name == "interactive",
               "same_opponent_path_across_methods": name != "interactive"}
    if primary:
        component_contrasts = {}
        metrics = {"security_total": np.array([1, 1, 1, 0]),
                   **{metric: np.eye(4)[index] for index, metric in enumerate(METRIC_NAMES)}}
        for reference in references:
            component_contrasts[reference] = {}
            for metric, weights in metrics.items():
                result = paired_two_way_bootstrap(occupations[:, learner_index], occupations[:, METHODS.index(reference)], bank,
                                                  weights=weights, samples=samples, seed=seed, alpha=0.05, return_differences=False)
                result.update({"descriptive": True, "primary_family_member": False})
                component_contrasts[reference][metric] = result
        summary["descriptive_component_contrasts"] = component_contrasts
    return summary


def _fmt_interval(value):
    return f"{value['estimate']:.5f} [{value['lower']:.5f}, {value['upper']:.5f}]"


def _render_report(aggregate, russian):
    protocol = aggregate["protocol"]
    primary = aggregate["groups"]["primary"]
    contrasts = primary["contrasts"]
    wins = sum(item["decision"] == "supported_advantage" for item in contrasts.values())
    losses = sum(item["decision"] == "supported_disadvantage" for item in contrasts.values())
    unresolved = len(contrasts) - wins - losses
    lines = ["[English](README.md) | [Русский](README.ru.md)", "",
             "# Расширенное исследование CAGE 2" if russian else "# Extended CAGE 2 study", ""]
    if russian:
        lines += [f"По заранее зафиксированному основному критерию получено: преимущество в {wins} сравнениях, худший результат в {losses}, неопределённый результат в {unresolved}. Решение основано на одновременных бутстрэп-интервалах для шести сравнений, условных на исходной калибровке.", "",
                  "Это расширение пилотного исследования на официальном симуляторе CAGE 2. Первичный анализ использует 400 новых семян симулятора и 50 общих последовательностей атак. Каждый эпизод содержит 50 шагов. Основная метрика — сумма четырёх нативных потерь на шаг; меньше означает лучше. Меню защит, веса, калибровка и бюджет переключения сохранены.", "",
                  "## Основные результаты", "", "| Метод | Нативная потеря | Компрометация хостов | Серверы | Нарушение работы | Восстановление |", "|---|---:|---:|---:|---:|---:|"]
    else:
        lines += [f"The pre-specified primary decision supports lower loss in {wins} comparisons, higher loss in {losses}, and leaves {unresolved} inconclusive. Decisions use simultaneous bootstrap intervals across six comparisons, conditional on the original calibration.", "",
                  "This extends the pilot study in the official CAGE 2 simulator. Primary analysis uses 400 new simulator seeds and 50 common attack paths. Episodes contain 50 steps. The primary endpoint is the sum of four native loss components per step; lower is better. The defense menu, weights, calibration, and switch budget are unchanged.", "",
                  "## Primary results", "", "| Method | Native total loss | Host compromise | Server compromise | Disruption | Restore |", "|---|---:|---:|---:|---:|---:|"]
    for score in primary["scores"]:
        values = [score["mean_components"][name] for name in METRIC_NAMES]
        lines.append(f"| `{score['method']}` | {score['mean_native_loss']:.5f} | " + " | ".join(f"{value:.5f}" for value in values) + " |")
    lines += ["", "| " + ("Сравнение: наш метод минус базовый" if russian else "Our method minus reference") + " | " + ("Разность и одновременный интервал" if russian else "Difference and simultaneous interval") + " | " + ("Holm p" if russian else "Holm p") + " | " + ("Решение" if russian else "Decision") + " |", "|---|---:|---:|---|"]
    labels = {"supported_advantage": "преимущество" if russian else "lower loss",
              "supported_disadvantage": "хуже" if russian else "higher loss",
              "inconclusive": "неопределённо" if russian else "inconclusive"}
    for reference, item in contrasts.items():
        lines.append(f"| `{reference}` | {_fmt_interval(item)} | {item['holm_adjusted_pvalue']:.4g} | {labels[item['decision']]} |")
    if russian:
        lines += ["", "Отрицательная разность означает меньшие потери нашего метода. Интервалы используют 10 000 общих для всех сравнений выборок с повторением и уровень ошибки 0.05/6 для каждого сравнения. Holm p — дополнительная приближённая оценка; основное решение следует правилу интервала, зафиксированному до новых запусков.", "",
                  "`fixed_monitor`, `fixed_react_remove` и `fixed_react_restore` — неизменённые политики из официального кода. `fixed_decoy_react` — заранее заданная эвристика этого репозитория. Hedge и `last_window` сравниваются при одинаковой информации и весах. Остальные чистые политики также приведены в таблице, хотя они не входят в первичное семейство проверок.", "",
                  "## Чувствительность и вторичные сценарии", "", "Ниже приведены описательные интервалы 95%; они не дают дополнительных первичных заявлений о преимуществе. Колонки показывают разность `one_switch` и соответствующего базового метода."]
    else:
        lines += ["", "A negative difference means lower loss for our method. Intervals use 10,000 shared resampling draws and per-comparison error 0.05/6. Holm p-values are additional approximate inference; primary decisions follow the interval rule fixed before the new runs.", "",
                  "`fixed_monitor`, `fixed_react_remove`, and `fixed_react_restore` are unchanged official policies. `fixed_decoy_react` is this repository's predeclared heuristic. Hedge and `last_window` receive the same information and weights. Other pure policies appear above but are outside the primary comparison family.", "",
                  "## Sensitivities and secondary scenarios", "", "The following 95% intervals are descriptive and do not add primary advantage claims. Columns show `one_switch` minus the named reference."]
    refs = protocol["primary"]["references"]
    lines += ["", "| " + ("Группа" if russian else "Group") + " | " + " | ".join(f"`{ref}`" for ref in refs) + " |", "|---|" + "---:|" * len(refs)]
    for name, group in aggregate["groups"].items():
        if name != "primary":
            lines.append(f"| `{name}` | " + " | ".join(_fmt_interval(group["contrasts"][ref]) for ref in refs) + " |")
    primary_switches = primary["switch_counts"]["one_switch"]
    all_switches = sum(group["switch_counts"]["one_switch"] for group in aggregate["groups"].values())
    equal = all(group["one_switch_and_fast_actions_numerically_equal"] is True for group in aggregate["groups"].values())
    if russian:
        lines += ["", "## Ограничения вывода", "",
                  f"Реальное число переключений master: {primary_switches} в первичной группе и {all_switches} во всех группах. Численное совпадение действий `one_switch` и `shared_past_hull` во всех группах с абсолютным допуском 1e-14: {'да' if equal else 'нет'}. Дополнительная нормировка master может давать округление на уровне машинной точности; исходные массивы сохранены. Для трёх чистых режимов остаток растёт только при первых появлениях и не достигает исходного порога. Исследование оценивает адаптацию быстрого метода и не демонстрирует работу безопасного переключения.", "",
                  "400 семян симулятора и 50 последовательностей — два пересекающихся набора единиц выборки. Их 20 000 комбинаций не являются 20 000 независимых эпизодов. Каждая бутстрэп-выборка сохраняет целую таблицу одного семени и целую последовательность, включая парность методов. Калибровочная неопределённость исключена из основного интервала; три новые калибровки показывают описательную устойчивость.", "",
                  "Для интерактивной группы один причинный алгоритм атакующего применяется к каждой защите отдельно, поэтому пути атак различаются. Разности сравнивают полные взаимодействия, а не ответы на один общий путь. В фиксированных и детерминированных группах повторения путей могут совпадать; это не создаёт новую независимую вариативность.", "",
                  "В `fixed_b_line` исходный режим известен как B-line: `historical_best` и `last_window` начинают с индекса 2. Опубликованное поле `effective_initial_red_index` явно отражает это правило; исходное `initial_red_index=1` обозначает общий начальный режим основного сценария. Действия и исходные результаты не пересчитывались.", "",
                  "Обучается выбор среди трёх известных политик атакующего, а не новая тактика внутри них. Метка атаки раскрывается после выбора защиты; это предположение модели с полной информацией. Смеси означают ожидаемые результаты целых эпизодов с восстановлением сети. Результаты относятся к симулятору и данному меню политик; записи настоящих атак здесь не используются.", "",
                  "Все 13 методов оценены по одним заранее выбранным метрикам. Первичные решения не подбирались после анализа новых тестовых эпизодов. Разности по безопасности и восстановлению в `aggregate.json` описательные и не входят в шесть первичных проверок.", "",
                  "## Воспроизводимость", "",
                  "[Зафиксированный протокол](protocol.json), [машиночитаемые результаты](aggregate.json), [агрегаты групп](groups/), [методика](../../docs/cage_study_ru.md). Старый пилотный отчёт сохранён отдельно в `results/cage_reference`."]
    else:
        lines += ["", "## Interpretation limits", "",
                  f"Actual master switch counts are {primary_switches} in the primary group and {all_switches} across all groups. `one_switch` and `shared_past_hull` actions are numerically equal in every group at absolute tolerance 1e-14: {'yes' if equal else 'no'}. The master's extra normalization can introduce machine-precision rounding; original arrays are retained. With three pure modes, residual growth occurs only on first appearances and cannot reach the original switch threshold. This study evaluates fast-policy adaptation and does not demonstrate safe switching.", "",
                  "The 400 simulator seeds and 50 paths are two crossed sets of sampling units. Their 20,000 combinations are not 20,000 independent episodes. Each bootstrap draw retains a whole seed table and whole path, including method pairing. Primary intervals exclude calibration uncertainty; three new fits provide descriptive robustness checks.", "",
                  "In the interactive group, the same causal attacker rule runs separately against each defense, inducing different paths. Differences compare complete interactions, not responses to one common path. Fixed and deterministic groups can contain identical path repetitions; these do not create independent path variation.", "",
                  "In `fixed_b_line`, the initial mode is known to be B-line: `historical_best` and `last_window` start from index 2. Published `effective_initial_red_index` makes this existing rule explicit; the original `initial_red_index=1` denotes the primary scenario default. Actions and original results were not recomputed.", "",
                  "Learning selects among three known attacking policies; it does not learn new tactics inside them. The current attack label is revealed after defense selection, a full-information modeling assumption. Mixtures are expected outcomes of complete reset episodes. Findings concern this simulator and policy menu, rather than recorded operational attacks.", "",
                  "All 13 methods use the same predeclared endpoints. Primary decisions were not selected after inspecting the new held-out episodes. Security and restore differences in `aggregate.json` are descriptive and outside the six primary tests.", "",
                  "## Reproduction", "",
                  "[Frozen protocol](protocol.json), [machine-readable results](aggregate.json), [group aggregates](groups/), [methodology](../../docs/cage_study_en.md). The pilot report remains separately available under `results/cage_reference`."]
    lines += ["", "![Primary paired comparisons](primary_comparisons.png)", "",
              "![All methods and loss components](primary_losses.png)", "",
              "![Exploratory robustness checks](robustness_losses.png)", "",
              "```bash", "python scripts/build_cage_study_report.py",
              "python scripts/plot_cage_study.py", "```", "",
              f"New simulator episodes: {aggregate['new_simulator_episodes']:,}; internal steps: {aggregate['new_simulator_steps']:,}." if not russian
              else f"Новых эпизодов симулятора: {aggregate['new_simulator_episodes']:,}; внутренних шагов: {aggregate['new_simulator_steps']:,}.", ""]
    return "\n".join(lines)


def build_report(selection_dir, bank_root, output_dir, data_output_dir, protocol_path, calibration_dir):
    protocol, original, banks, groups = _load_inputs(protocol_path, bank_root, selection_dir, calibration_dir)
    summaries = {name: _group_statistics(name, item, protocol) for name, item in groups.items()}
    aggregate = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": FROZEN_PROTOCOL_SHA256, "protocol": protocol,
                 "software": software_versions(), "methods": METHODS, "groups": summaries,
                 "primary": summaries["primary"],
                 "new_simulator_episodes": sum(bank["manifest"]["episodes_total"] for bank in banks.values()),
                 "new_simulator_steps": sum(bank["manifest"]["simulator_steps_total"] for bank in banks.values()),
                 "new_bank_seed_counts": {name: len(bank["seeds"]) for name, bank in banks.items()},
                 "bank_hashes": {name: bank["manifest"]["bank_npz_sha256"] for name, bank in banks.items()},
                 "path_slots_total": sum(item["arrays"]["occupancies"].shape[0] for item in groups.values()),
                 "all_primary_data_collected_before_inference": True,
                 "primary_training_fit_sha256": original.provenance["calibration_npz_sha256"],
                 "primary_family_size": len(protocol["primary"]["references"]),
                 "primary_decision_rule": protocol["primary"]["primary_decision"],
                 "formal_finite_sample_guarantee": False,
                 "calibration_uncertainty_in_primary_intervals": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "aggregate.json", aggregate)
    shutil.copyfile(protocol_path, output_dir / "protocol.json")
    for name in groups:
        destination = output_dir / "groups" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("occupancies.npz", "training_fit.npz"):
            shutil.copyfile(selection_dir / name / filename, destination / filename)
        metadata = dict(summaries[name]["metadata"])
        metadata["source_meta_sha256"] = sha256(selection_dir / name / "meta.json")
        metadata["publication_metadata_note"] = "Derived effective_initial_red_index added to document the existing fixed_b_line override; original selection inputs unchanged."
        write_json(destination / "meta.json", metadata)
    for name in banks:
        destination = data_output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("bank.npz", "manifest.json", "cell_statistics.json"):
            shutil.copyfile(bank_root / name / filename, destination / filename)
    primary = groups["primary"]
    occupancies = primary["arrays"]["occupancies"]
    scalar_bank = primary["bank"].sum(axis=-1)
    differences = {reference: np.einsum("pij,eij->ep", occupancies[:, METHODS.index("one_switch")] - occupancies[:, METHODS.index(reference)], scalar_bank, optimize=True)
                   for reference in protocol["primary"]["references"]}
    np.savez_compressed(output_dir / "primary_paired_differences.npz", **differences)
    for russian, filename in ((False, "README.md"), (True, "README.ru.md")):
        (output_dir / filename).write_text(_render_report(aggregate, russian), encoding="utf-8", newline="\n")
    # Byte hashes must survive Git's LF checkout normalization on Windows.
    for path in output_dir.rglob("*.json"):
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"Report saved to {output_dir}; all {len(groups)} groups checked before analysis.", flush=True)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=Path("results/runs/cage_study/selection"))
    parser.add_argument("--bank-root", type=Path, default=Path("results/runs/cage_study/banks"))
    parser.add_argument("--output", type=Path, default=Path("results/cage_study"))
    parser.add_argument("--data-output", type=Path, default=Path("data/cage2_study"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/cage_study_protocol.json"))
    parser.add_argument("--calibration", type=Path, default=Path("data/cage2"))
    args = parser.parse_args()
    build_report(args.selection_dir, args.bank_root, args.output, args.data_output, args.protocol, args.calibration)


if __name__ == "__main__":
    main()
