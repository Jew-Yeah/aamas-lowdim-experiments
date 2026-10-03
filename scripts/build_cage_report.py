"""Build bilingual reports from completed, simulator-backed CAGE runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from lowdim_games.cage import load_cage_calibration
from lowdim_games.experiment import write_json
from lowdim_games.policy_experiment import mean_ci95


METHOD_LABELS = {
    "one_switch": ("One-switch", "Наш метод"),
    "shared_past_hull": ("Shared past-hull", "Общий метод past-hull"),
    "block_safe": ("Block safe", "Безопасная процедура"),
    "uniform": ("Uniform mixture", "Равномерная смесь"),
    "historical_best": ("Initial-mode response", "Ответ на начальный режим"),
    "last_window": ("Previous 16 rounds", "Ответ по последним 16 атакам"),
    "hedge": ("Hedge", "Hedge"),
    "fixed_monitor": ("Fixed monitor", "Фиксированное наблюдение"),
    "fixed_react_remove": ("Fixed reactive removal", "Фиксированное удаление"),
    "fixed_react_restore": ("Fixed reactive restore", "Фиксированное восстановление"),
    "fixed_decoy_cycle": ("Fixed decoy cycle", "Фиксированные приманки"),
    "fixed_critical_restore": ("Fixed critical restore", "Восстановление важных серверов"),
    "fixed_decoy_react": ("Fixed decoy + reaction", "Приманки с реакцией"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs/cage2")
    parser.add_argument("--calibration", default="data/cage2")
    parser.add_argument("--output", default="results/cage_reference")
    args = parser.parse_args()
    runs, output = Path(args.runs), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = load_cage_calibration(args.calibration)
    manifests = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(runs.glob("cage2_*_T512.json"))]
    assert len(manifests) == 9, "Expected all three scenarios and three attacker seeds"
    scenarios = ("fixed", "curriculum", "interactive")
    methods = manifests[0]["methods"]
    assert len(methods) == 13
    assert all(m["methods"] == methods for m in manifests)
    assert len({m["tensor_sha256"] for m in manifests}) == 1
    assert all(m["max_target_projection_gap"] < 1e-9 for m in manifests)
    assert all(len(m["summaries"]) == len(methods) for m in manifests)
    aggregates = {}
    max_action_difference = 0.0
    for scenario in scenarios:
        selected = [m for m in manifests if m["scenario"] == scenario]
        assert len(selected) == 3
        samples, rows = {}, {}
        for method in methods:
            occupancies = []
            for manifest in selected:
                with np.load(runs / f"{manifest['experiment']}__{method}.npz",
                             allow_pickle=False) as trace:
                    occupancy = trace["actions"].T @ trace["opponent_actions"] / manifest["horizon"]
                    np.testing.assert_allclose(occupancy, trace["occupancy"], atol=1e-13)
                    if method == "one_switch":
                        with np.load(runs / f"{manifest['experiment']}__shared_past_hull.npz") as shared:
                            np.testing.assert_allclose(trace["actions"], shared["actions"], rtol=0, atol=1e-14)
                            np.testing.assert_array_equal(trace["opponent_actions"], shared["opponent_actions"])
                            max_action_difference = max(max_action_difference,
                                                        float(np.max(np.abs(trace["actions"] - shared["actions"]))))
                occupancies.append(occupancy)
            mean_occupancy = np.mean(occupancies, axis=0)
            replicates = np.einsum("ij,eijd->ed", mean_occupancy, calibration.test_episode_losses)
            samples[method] = replicates
            summaries = [next(x for x in m["summaries"] if x["method"] == method) for m in selected]
            assert all(x["switch_round"] is None for x in summaries)
            row = {"method": method,
                   "mean_native_loss": float(replicates.sum(axis=1).mean()),
                   "native_loss_ci95": mean_ci95(replicates.sum(axis=1)),
                   "mean_security_loss": float(replicates[:, :3].sum(axis=1).mean()),
                   "mean_restore_loss": float(replicates[:, 3].mean()),
                   "mean_components": dict(zip(calibration.metric_names, replicates.mean(axis=0))),
                   "mean_delta_realized_hull": float(np.mean([x["delta_realized_hull"] for x in summaries])),
                   "native_loss_range_across_attack_seeds": [min(sum(x["mean_test_metrics"].values()) for x in summaries),
                                                            max(sum(x["mean_test_metrics"].values()) for x in summaries)],
                   "max_fast_residual_sum": max(x["final_fast_residual_sum"] for x in summaries),
                   "phase_native_loss": {phase: float(np.mean([sum(x["phases"][index]["mean_test_metrics"].values()) for x in summaries]))
                                          for index, phase in enumerate(("weak", "broad", "learning"))}}
            rows[method] = row
        for method in methods:
            rows[method]["paired_native_difference_ci95"] = {
                reference: mean_ci95((samples[method] - samples[reference]).sum(axis=1))
                for reference in methods if reference != method}
            rows[method]["paired_component_difference_ci95"] = {
                reference: mean_ci95(samples[method] - samples[reference])
                for reference in methods if reference != method}
            rows[method]["paired_security_difference_ci95"] = {
                reference: mean_ci95((samples[method] - samples[reference])[:, :3].sum(axis=1))
                for reference in methods if reference != method}
        aggregates[scenario] = list(rows.values())
    write_json(output / "aggregate.json", {
        "scenarios": aggregates, "source_experiments": [m["experiment"] for m in manifests],
        "simulator_episodes": calibration.provenance["episodes_total"],
        "heldout_seed_count": len(calibration.test_seeds),
        "interval_interpretation": "40 paired held-out seed clusters; conditional on calibration and the three attack paths; not 120 independent simulator samples",
        "calibration_uncertainty_included": False,
        "interactive_comparison": "Different opponent paths under the same causal attacker rule",
        "source_revision": calibration.provenance["source"]["revision"],
        "all_one_switch_actions_match_shared_past_hull_to_precision": True,
        "max_one_switch_shared_action_difference": max_action_difference,
        "max_target_projection_gap": max(m["max_target_projection_gap"] for m in manifests)})
    shutil.copy2(runs / "cage_summary.json", output / "all_run_summaries.json")
    for manifest in manifests:
        if manifest["seed"] != 20261003:
            continue
        for path in runs.glob(manifest["experiment"] + "*"):
            shutil.copy2(path, output / path.name)
    for path in runs.glob("cage_calibration.*"):
        shutil.copy2(path, output / path.name)
    for language in ("en", "ru"):
        ru = language == "ru"
        lines = ["[English](README.md) | [Русский](README.ru.md)", "",
                 "# CAGE 2: результаты первого опыта" if ru else "# CAGE 2: initial recorded results", ""]
        lines += (["Выполнено 1440 эпизодов официального симулятора по 50 шагов: 40 калибровочных и 40 проверочных seed для каждой из 18 пар политик. На зафиксированной калиброванной игре проведено девять сравнений: три сценария, три seed атакующего, 512 раундов и 13 методов."] if ru else
                  ["Completed 1,440 official simulator episodes of 50 steps: 40 calibration and 40 held-out seeds for each of 18 policy pairs. The frozen calibrated game was used for nine comparisons: three scenarios, three attacker seeds, 512 rounds, and 13 methods."])
        lines += ["", "**Средняя штатная потеря CAGE за шаг; меньше лучше.** Значения усреднены по трём заданным траекториям атакующего." if ru else
                  "**Mean native CAGE loss per simulator step; lower is better.** Values average the three specified attacker paths.", "",
                  "| Метод | Постоянный Meander | Смена режимов | Интерактивный атакующий |" if ru else
                  "| Method | Fixed Meander | Common curriculum | Interactive attacker |",
                  "|---|---:|---:|---:|"]
        for method in methods:
            values = [next(x for x in aggregates[s] if x["method"] == method)["mean_native_loss"] for s in scenarios]
            lines.append(f"| {METHOD_LABELS[method][int(ru)]} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        lines += ["", "Компоненты потерь в общем сценарии смены режимов:" if ru else
                  "Loss components in the common curriculum:", "",
                  "| Метод | Ущерб безопасности | Восстановление | Сумма |" if ru else
                  "| Method | Security loss | Restoration | Total |", "|---|---:|---:|---:|"]
        for method in ("one_switch", "block_safe", "hedge", "last_window"):
            row = next(x for x in aggregates["curriculum"] if x["method"] == method)
            lines.append(f"| {METHOD_LABELS[method][int(ru)]} | {row['mean_security_loss']:.4f} | "
                         f"{row['mean_restore_loss']:.4f} | {row['mean_native_loss']:.4f} |")
        lines += ["", "## Интерпретация" if ru else "## Interpretation", ""]
        lines += ([
            "Наш метод заметно лучше консервативной безопасной процедуры на этом меню политик. Его средняя потеря также ниже Hedge, однако для меняющихся сценариев условный 95% интервал парной разности с Hedge включает ноль. При смене атак ущерб безопасности ниже Hedge, но затраты на восстановление выше. Ответ по последним 16 атакам и хорошая фиксированная политика дают меньшую среднюю суммарную потерю, чем наш метод. Эксперимент не устанавливает общего превосходства.",
            "", "Переключений не было; действия `one_switch` совпадают с `shared_past_hull` с точностью вычислений с плавающей точкой. При трёх чистых режимах новизна возникает только при первом появлении двух оставшихся режимов. Накопленная невязка ограничена 4, тогда как порог для K=6 и T=512 равен примерно 1444. Это опыт быстрой адаптации, без подтверждения преимущества механизма переключения.",
            "", "Проверочные потери получены из независимого банка эпизодов после фиксации решений игроков. Разница между калибровочной игрой и проверочным банком сохраняется. Доверительные интервалы используют 40 seed как парные блоки по всей таблице; три траектории атакующего не превращают их в 120 независимых наблюдений. Интервалы условны на калибровку и эти траектории, не включают её неопределённость и не корректируются для множества сравнений.",
            "", "В интерактивном сценарии методы встречают разные траектории при одном и том же правиле атакующего. Расстояния до собственных реализованных целей и regret по собственной траектории следует читать с этой оговоркой. Общий сценарий смены режимов даёт прямое сравнение на одинаковом пути.",
            "", "Это выбор между фиксированными политиками по известной оценённой модели с раскрытием режима атаки после решения. Внутри эпизода защиты используют частичные наблюдения. Новые тактики нападения не обучаются; опыт не воспроизводит исторические атаки на действующую сеть."
        ] if ru else [
            "The method improves substantially over the conservative safe routine on this policy menu. Its mean loss is also below Hedge, but the conditional 95% paired difference interval includes zero in the changing scenarios. Security losses are lower than Hedge in the changing cases, while restoration costs are higher. The previous-16-round response and a good fixed policy have lower mean total loss than our method. This experiment does not establish general superiority.",
            "", "There were no switches: one_switch actions match shared_past_hull to floating-point precision. With three pure modes, novelty occurs only on the first appearance of the two remaining modes. The residual sum is at most 4; the K=6, T=512 threshold is about 1,444. This tests fast adaptation without validating an advantage of the switching mechanism.",
            "", "Held-out losses use the independent episode bank after player decisions are fixed. The calibration game and held-out table remain distinct. Confidence intervals use 40 seed clusters across the whole payoff table; the three attacker paths do not create 120 independent simulator samples. Intervals condition on calibration and these paths, exclude calibration uncertainty, and are not adjusted for multiple comparisons.",
            "", "Interactive methods face different paths under the same attacker rule. Distances to their own realized targets and regret on their own paths must be interpreted accordingly. The common curriculum provides direct comparisons on the same opponent path.",
            "", "This is selection among frozen policies with a known estimated model and attack-mode disclosure after choosing. Within episodes, defense policies use partial observations. New attack tactics are not trained, and this experiment does not reproduce historical attacks on a deployed network."
        ])
        lines += ["", "## Файлы и воспроизведение" if ru else "## Files and reproduction", "",
                  "[Методология и команды](../../docs/cage_ru.md)." if ru else "[Methodology and commands](../../docs/cage_en.md).", ""]
        lines += (["- [Все девять сравнений](all_run_summaries.json)",
                   "- [Агрегаты, компоненты потерь и парные интервалы](aggregate.json)",
                   "- [Банк эпизодов и происхождение](../../data/cage2/provenance.json)"] if ru else
                  ["- [All nine comparisons](all_run_summaries.json)",
                   "- [Aggregates, loss components, and paired intervals](aggregate.json)",
                   "- [Episode-bank provenance](../../data/cage2/provenance.json)"])
        lines += ["", "![Policy payoff tables](cage_calibration.png)", "",
                  "![Common-curriculum comparison](cage2_curriculum_seed20261003_T512__loss.png)", ""]
        path = output / ("README.ru.md" if ru else "README.md")
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    # Canonical LF text keeps byte checksums stable after a Git checkout.
    for path in output.glob("*.json"):
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    write_json(output / "SHA256SUMS.json", {"algorithm": "sha256", "files": hashes})
    print(f"Wrote bilingual CAGE report with {len(hashes)} hashed artifacts to {output}")


if __name__ == "__main__":
    main()
