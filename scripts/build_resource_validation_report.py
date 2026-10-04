"""Build bilingual reports from a complete, frozen resource validation batch.

This postprocessor does not run, select, or overwrite simulation episodes. It
requires all prespecified scenarios and checks the pre-validation source receipt.
The analytical membership diagnostic and numerical KKT residual are reported
separately; tiny off-manifold rounding displacements are not ranked as results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/resource_validation_protocol.json"
METHOD_NAMES = {
    "one_switch": ("One-switch", "One-switch"),
    "shared_past_hull": ("Fast: origin saddle", "Fast: седловая точка в нуле"),
    "block_safe": ("Block safe only", "Только блочный safe"),
    "lag_response": ("Lag response", "Отклик на предыдущий запрос"),
    "last_window": ("Window, W=16", "Окно, W=16"),
    "uniform": ("Uniform, p=0.5", "Постоянная активация, p=0.5"),
    "request_trigger": ("Request-triggered balanced control", "Баланс после первого запроса"),
    "fast_recent_saddle": ("Fast: recent-request saddle", "Fast: седловая точка последнего запроса"),
}
SCENARIO_NAMES = {
    "primary": ("Reactive, memory 1 (primary)", "Реактивная среда, память 1 (основной сценарий)"),
    "memory16": ("Reactive, memory 16", "Реактивная среда, память 16"),
    "exogenous": ("Exogenous arrivals, q=0.5", "Внешние поступления, q=0.5"),
    "stationary": ("Stationary origin control", "Стационарный контроль без запросов"),
}
COMPARATORS = ("shared_past_hull", "block_safe", "lag_response", "last_window")
FIGURES = ("resource_switching_dynamics", "resource_outcomes_and_sensitivity")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def values(rows, method, key):
    result = np.asarray([row["methods"][method][key] for row in rows], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite episode metric: {method}/{key}")
    return result


def exact_binomial_interval(count, n):
    result = stats.binomtest(int(count), int(n)).proportion_ci(.95, method="exact")
    return [float(result.low), float(result.high)]


def count_classification(zero, positive):
    zero, positive = np.asarray(zero, bool), np.asarray(positive, bool)
    if zero.shape != positive.shape or np.any(zero & positive):
        raise ValueError("Overlapping zero/positive endpoint classifications")
    n = len(zero)
    unresolved = ~(zero | positive)
    nz, np_, nu = (int(np.sum(x)) for x in (zero, positive, unresolved))
    cp_positive = exact_binomial_interval(np_, n)
    possible = exact_binomial_interval(np_ + nu, n)
    return {
        "n": n, "zero": nz, "positive": np_, "unresolved": nu,
        "positive_fraction": np_ / n,
        "positive_fraction_ci_95": cp_positive,
        "zero_fraction_ci_95": exact_binomial_interval(nz, n),
        "possible_positive_fraction_upper_count": np_ + nu,
        "possible_positive_population_fraction_enclosure_95": [cp_positive[0], possible[1]],
    }


def distribution_diagnostic(rows, method, beta, T):
    delta = values(rows, method, "terminal_delta")
    lower = values(rows, method, "terminal_projection_lower")
    upper = values(rows, method, "terminal_projection_upper")
    if np.any(lower < 0) or np.any(lower > upper) or np.any(delta < lower) or np.any(delta > upper):
        raise ValueError(f"Invalid numerical endpoint enclosure: {method}")
    count = values(rows, method, "observed_requests")
    A = values(rows, method, "cumulative_aggregate") / T
    E = values(rows, method, "cumulative_energy") / T
    margin = beta - 2 * A - E
    membership = ((count > 0) & (2 * A + E <= beta)) | ((count == 0) & (A == 0) & (E == 0))
    saved_membership = ((values(rows, method, "terminal_delta_lower") == 0) &
                        (values(rows, method, "terminal_delta_upper") == 0))
    if not np.array_equal(membership, saved_membership):
        raise ValueError(f"Stored analytical membership disagrees with endpoint totals: {method}")
    # KKT bounds and aggregate membership arise from distinct floating sums.
    # Keep both: treating a positive residual as a physical failure would erase
    # this distinction, while calling stored membership a formal proof over
    # exact real arithmetic would overstate what these floats establish.
    kkt = count_classification(upper == 0, lower > 0)
    combined = count_classification(membership, (~membership) & (lower > 0))
    if np.any(membership & (lower > 0)):
        raise ValueError(f"Positive KKT lower bound contradicts membership: {method}")
    return {
        "terminal_delta_mean": float(np.mean(delta)),
        "terminal_delta_std": float(np.std(delta, ddof=1)),
        "terminal_delta_quantiles_05_50_95_99": np.quantile(delta, [.05, .5, .95, .99]).tolist(),
        "maximum_kkt_distance_residual": float(np.max(delta)),
        "maximum_support_gap": float(np.max(values(rows, method, "terminal_support_gap"))),
        "maximum_feasibility_error": float(np.max(values(rows, method, "terminal_feasibility_error"))),
        "maximum_projection_interval_width": float(np.max(upper - lower)),
        "kkt_certificate_classification": kkt,
        "stored_float_membership_diagnostic": {
            "inside": int(np.sum(membership)), "outside": int(np.sum(~membership)),
            "inside_fraction_ci_95": exact_binomial_interval(np.sum(membership), len(rows)),
            "outside_fraction_ci_95": exact_binomial_interval(np.sum(~membership), len(rows)),
            "minimum_membership_margin_when_requests_observed":
                float(np.min(margin[count > 0])) if np.any(count > 0) else None,
            "maximum_kkt_residual_among_members":
                float(np.max(delta[membership])) if np.any(membership) else None,
        },
        "combined_machine_float_classification": combined,
    }


def outward_intervals(rows, comparator):
    lower = (values(rows, "one_switch", "terminal_projection_lower") -
             values(rows, comparator, "terminal_projection_upper"))
    upper = (values(rows, "one_switch", "terminal_projection_upper") -
             values(rows, comparator, "terminal_projection_lower"))
    n, df = len(rows), len(rows) - 1
    result = {"numerical_mean_difference_enclosure": [float(lower.mean()), float(upper.mean())]}
    for level, key in ((.95, "numeric_and_sampling_ci_95"), (.9875, "numeric_and_sampling_ci_family")):
        critical = float(stats.t.ppf((1 + level) / 2, df))
        result[key] = [float(lower.mean() - critical * lower.std(ddof=1) / math.sqrt(n)),
                       float(upper.mean() + critical * upper.std(ddof=1) / math.sqrt(n))]
    # Observed certified comparisons use episode bounds, not the sign of tiny
    # differences between two independent floating-summation residuals.
    wins = upper < 0
    losses = lower > 0
    exact_zero_ties = (
        (values(rows, "one_switch", "terminal_delta_upper") == 0) &
        (values(rows, comparator, "terminal_delta_upper") == 0))
    result["kkt_bound_comparisons"] = {
        "wins": int(np.sum(wins)), "losses": int(np.sum(losses)),
        "unresolved": int(n - np.sum(wins) - np.sum(losses)),
    }
    result["both_stored_membership_zero"] = int(np.sum(exact_zero_ties))
    return result


def log_holm(paired):
    keys = list(paired)
    logs = np.asarray([0. if paired[key]["log_p_value"] is None else
                       float(paired[key]["log_p_value"]) for key in keys])
    if not np.isfinite(logs).all() or np.any(logs > 0):
        raise ValueError("Invalid saved log p-values")
    order = np.argsort(logs, kind="stable")
    adjusted = np.minimum(0., np.maximum.accumulate(logs[order] + np.log(np.arange(len(keys), 0, -1))))
    result = {}
    for rank, index in enumerate(order):
        key = keys[index]
        result[key] = {
            "log_holm_p_value": float(adjusted[rank]),
            "log10_holm_p_value": float(adjusted[rank] / math.log(10)),
            "holm_test_defined": paired[key]["log_p_value"] is not None,
            "holm_log_convention": "Undefined tests count in the planned family with p=1.",
        }
    return result


def verify_inputs(output):
    design = load_json(PROTOCOL)
    receipt = load_json(output / "precommitment.json")
    analysis = load_json(output / "analysis.json")
    if receipt["protocol_sha256"] != sha256(PROTOCOL) or receipt["design"] != design:
        raise ValueError("Protocol changed after the frozen precommitment")
    if analysis["protocol_sha256"] != receipt["protocol_sha256"]:
        raise ValueError("Analysis protocol hash differs from the frozen receipt")
    checked_sources = {}
    for name, expected in receipt["source_sha256"].items():
        actual = sha256(ROOT / name)
        if actual != expected:
            raise ValueError(f"Critical source changed after freeze: {name}")
        checked_sources[name] = actual
    methods = design["methods"]
    if set(methods) != set(METHOD_NAMES) or len(methods) != 8:
        raise ValueError("The full eight prespecified controls are required")
    seedspec = design["validation_seeds"]
    expected_seeds = list(range(seedspec["first"], seedspec["last"] + 1))
    if len(expected_seeds) != seedspec["count_per_scenario"]:
        raise ValueError("Registered sample size and seed range disagree")
    batches = {}
    fixed = design["fixed_settings"]
    for scenario in design["scenarios"]:
        name = scenario["name"]
        batch = load_json(output / name / "episodes.json")
        rows, meta = batch["episodes"], batch["metadata"]
        if [row["seed"] for row in rows] != expected_seeds or meta["independent_seeds"] != expected_seeds:
            raise ValueError(f"Missing, replaced, reordered, or extra episodes: {name}")
        for key, value in meta["config"].items():
            expected = scenario["id"] if key == "scenario_id" else scenario.get(key, fixed.get(key))
            if expected is None or value != expected:
                raise ValueError(f"Scenario setting differs from protocol: {name}/{key}")
        if meta["methods"] != methods or meta["sample_stride"] != fixed["sample_stride"]:
            raise ValueError(f"Changed methods or curve grid: {name}")
        if meta["representative_seeds"] != design["representative_full_paths"]:
            raise ValueError(f"Changed representative episode selection: {name}")
        if meta["master_restarts"] or not meta["fresh_safe_tail_after_crossing"]:
            raise ValueError(f"Changed master switching rule: {name}")
        if meta["beta"] != design["game"]["beta"]:
            raise ValueError(f"Changed full-target game constant: {name}")
        if analysis["scenarios"][name]["n"] != len(rows):
            raise ValueError(f"Analysis has a different denominator: {name}")
        for row in rows:
            if set(row["methods"]) != set(methods):
                raise ValueError(f"Missing controls at seed {row['seed']}: {name}")
            tau = row["switch_round"]
            if row["safe_rounds"] != (0 if tau is None else fixed["T"] - tau):
                raise ValueError(f"Safe-tail round count disagrees with crossing: {name}/{row['seed']}")
            if tau is not None and row["master_final_E"] <= row["G_T"]:
                raise ValueError(f"No strict budget crossing: {name}/{row['seed']}")
            if tau is None and row["master_final_E"] > row["G_T"]:
                raise ValueError(f"Crossing episode incorrectly marked no-switch: {name}/{row['seed']}")
        for method in methods:
            for metric, stat in analysis["scenarios"][name]["methods"][method].items():
                observed = values(rows, method, metric)
                if not np.isclose(observed.mean(), stat["mean"], rtol=1e-13, atol=1e-18):
                    raise ValueError(f"Analysis mean disagrees with episodes: {name}/{method}/{metric}")
        for stem in FIGURES:
            for extension in ("pdf", "png"):
                if not (output / "figures" / f"{stem}.{extension}").is_file():
                    raise FileNotFoundError("Create the completed-batch figures before building this report")
        batches[name] = batch
    if sum(len(batch["episodes"]) for batch in batches.values()) != seedspec["total_episodes"]:
        raise ValueError("Total validation episode count differs from protocol")
    return design, receipt, analysis, batches, checked_sources


def build_diagnostics(design, analysis, batches):
    result = {
        "schema_version": 1,
        "protocol_sha256": analysis["protocol_sha256"],
        "classification_conventions": {
            "kkt": "Zero iff KKT projection upper=0; positive iff KKT projection lower>0; unresolved otherwise. Floating numerical certificates are not formal exact-arithmetic proofs.",
            "membership": "A=cumulative_aggregate/T and E=cumulative_energy/T: inside iff observed_requests>0 and 2*A+E<=beta, or no requests and A=E=0. This is a diagnostic on stored floats using the proven target form and the model identity A=sum(w). Independent floating summation of w may leave a tiny KKT residual.",
            "combined": "Stored-float analytical members are zero; outside members with KKT lower>0 are positive; remaining outside members are unresolved. This diagnostic is separate from the KKT classification.",
            "binomial": "Two-sided 95% Clopper-Pearson intervals over independent episodes. Possible-positive enclosure takes the lower endpoint for definite positives and upper endpoint for definite positives plus unresolved; ambiguity can make it uninformative.",
            "no_equivalence": "No positive distances observed is not population equality or established practical equivalence. No arbitrary tolerance relabels numerical residuals as failures.",
        },
        "interval_conventions": {
            "ordinary_confidence_level": .95, "family_marginal_confidence_level": .9875,
            "family_size": 4, "unit": "complete independent episode",
            "formula": "l=L_master-U_control, u=U_master-L_control; [mean(l)-tcrit*sd(l)/sqrt(n), mean(u)+tcrit*sd(u)/sqrt(n)]",
            "raw_p_values": "Saved t/Holm/bootstrap on numerical KKT distances remain diagnostics; a raw small p-value does not rank algorithms when endpoints share analytical membership or outward intervals include zero.",
        },
        "scenarios": {},
    }
    beta, T = design["game"]["beta"], design["fixed_settings"]["T"]
    for name, batch in batches.items():
        rows = batch["episodes"]
        early_one = values(rows, "one_switch", "pre_mean_delta")
        early_safe = values(rows, "block_safe", "pre_mean_delta")
        if not np.all(early_one == early_one[0]) or not np.all(early_safe == early_safe[0]):
            raise ValueError(f"Expected deterministic early phase changed: {name}")
        saved = analysis["scenarios"][name]
        contrasts = {}
        for control in COMPARATORS:
            key = f"terminal_delta__{control}"
            extra = outward_intervals(rows, control)
            if not np.allclose(extra["numeric_and_sampling_ci_family"],
                               saved["paired_differences"][key]["numeric_and_sampling_ci_family"],
                               rtol=2e-12, atol=1e-18):
                raise ValueError(f"Outward family interval differs from saved analysis: {name}/{control}")
            contrasts[key] = {**saved["paired_differences"][key], **extra}
        tau = np.asarray([row["switch_round"] for row in rows if row["switch_round"] is not None])
        crossing = {
            "n_all": len(rows), "n_switched": len(tau), "n_not_switched": len(rows) - len(tau),
            "fraction": len(tau) / len(rows), "fraction_ci_95": exact_binomial_interval(len(tau), len(rows)),
            "conditional_round_quantiles_05_50_95": np.quantile(tau, [.05, .5, .95]).tolist() if len(tau) else None,
            "conditional_delay_from_H_quantiles_05_50_95":
                np.quantile(tau - design["fixed_settings"]["H"], [.05, .5, .95]).tolist() if len(tau) else None,
            "mean_safe_rounds_all_episodes": float(np.mean([row["safe_rounds"] for row in rows])),
            "crossing_round_remains_fast": True, "fresh_safe_starts_next_round": True,
        }
        tail_checks = all(row.get("safe_tail_certificate_lhs", 0) <=
                          row.get("safe_tail_budget", 0) + 2e-7 for row in rows)
        if tail_checks != saved["all_safe_tail_certificates_hold"]:
            raise ValueError(f"Analysis conditional-tail checks differ from episodes: {name}")
        if crossing["n_switched"] != saved["switch_count"]:
            raise ValueError(f"Analysis switch denominator differs: {name}")
        result["scenarios"][name] = {
            "n": len(rows),
            "methods": {method: distribution_diagnostic(rows, method, beta, T) for method in design["methods"]},
            "paired_differences": contrasts,
            "early_phase": {"one_switch_mean_distance": float(early_one[0]),
                            "block_safe_mean_distance": float(early_safe[0]),
                            "difference": float(early_one[0] - early_safe[0]),
                            "constant_in_all_episodes": True,
                            "inference": "deterministic fixed-design quantity; no sampling test"},
            "switching": crossing,
            "conditional_safe_tail_certificates_hold": bool(saved["all_safe_tail_certificates_hold"]),
        }
    primary = result["scenarios"]["primary"]["paired_differences"]
    for key, extra in log_holm(primary).items():
        primary[key].update(extra)
        if extra["log_holm_p_value"] > math.log(np.finfo(float).tiny):
            saved_p = primary[key]["holm_p_value"]
            if not np.isclose(math.exp(extra["log_holm_p_value"]), saved_p, rtol=1e-10, atol=0.):
                raise ValueError(f"Log-scale Holm adjustment differs from saved analysis: {key}")
    return result


def number(value):
    return "—" if value is None else f"{value:.6g}"


def interval(value):
    return "[" + ", ".join(number(x) for x in value) + "]"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def terminal_cell(diagnostic, language):
    membership = diagnostic["stored_float_membership_diagnostic"]
    n = diagnostic["kkt_certificate_classification"]["n"]
    if membership["inside"] == n:
        return f"0† (KKT ≤ {number(diagnostic['maximum_kkt_distance_residual'])})"
    return number(diagnostic["terminal_delta_mean"])


def contrast_conclusion(contrast, language):
    low, high = contrast["numeric_and_sampling_ci_family"]
    # A mean interval that excludes zero on rounding residuals is not scientific
    # separation when every pair analytically belongs to its own target.
    if contrast["both_stored_membership_zero"] == contrast["n"]:
        return ("Both methods satisfy terminal membership on stored floats; no scientific ranking from KKT residuals."
                if language == 0 else "Оба метода удовлетворяют условию принадлежности на сохранённых числах; остатки KKT не дают научного ранжирования.")
    if high < 0:
        return "Resolved lower own-target distance for one-switch." if language == 0 else "Разрешимо меньшее расстояние до собственной цели у one-switch."
    if low > 0:
        return "Resolved higher own-target distance for one-switch." if language == 0 else "Разрешимо большее расстояние до собственной цели у one-switch."
    return "Outward family interval includes zero; no resolved directional difference." if language == 0 else "Интервал с численной погрешностью включает ноль; направление различия не разрешено."


def render_report(design, receipt, analysis, diagnostics, batches, language):
    en = language == 0
    text = []
    add = text.append
    n = design["validation_seeds"]["count_per_scenario"]
    T, H = design["fixed_settings"]["T"], design["fixed_settings"]["H"]
    primary = diagnostics["scenarios"]["primary"]
    source_url = "https://github.com/Jew-Yeah/aamas-lowdim-experiments/commit/" + receipt["git_commit"]
    add("# Resource-service switching validation" if en else "# Проверка переключения в модели обслуживания ресурсов")
    add("[English](README.md) | [Русский](README.ru.md)")
    add((f"Complete fixed validation: **{n} independent paired episodes per scenario, "
         f"{design['validation_seeds']['total_episodes']} total**, all eight prespecified methods. "
         "The following are results of a constructed vector game, with inference conditional on its fixed stochastic generator."
         if en else f"Завершена фиксированная проверка: **{n} независимых парных эпизодов на сценарий, "
         f"всего {design['validation_seeds']['total_episodes']}**, все восемь заранее заданных методов. "
         "Результаты относятся к искусственно построенной векторной игре и её фиксированному стохастическому генератору."))
    add("## What the comparison establishes" if en else "## Что устанавливает сравнение")
    for control in COMPARATORS:
        contrast = primary["paired_differences"][f"terminal_delta__{control}"]
        add(f"- **One-switch − {METHOD_NAMES[control][language]}**: " + contrast_conclusion(contrast, language) +
            (f" Outward 98.75% interval {interval(contrast['numeric_and_sampling_ci_family'])}." if en else
             f" Интервал 98,75% с численной погрешностью {interval(contrast['numeric_and_sampling_ci_family'])}."))
    early = primary["early_phase"]
    add((f"On the deterministic benign prefix, the exact average distance is {number(early['one_switch_mean_distance'])} "
         f"for one-switch and {number(early['block_safe_mean_distance'])} for safe only: difference "
         f"{number(early['difference'])}. This is a fixed-design computation, without a p-value or a claim of sampling superiority."
         if en else f"На детерминированном спокойном префиксе точное среднее расстояние равно "
         f"{number(early['one_switch_mean_distance'])} для one-switch и {number(early['block_safe_mean_distance'])} "
         f"для safe; разность {number(early['difference'])}. Это вычисление для фиксированной постановки, "
         "без p-value и без утверждения о статистическом превосходстве."))
    add(("The recent-request fast saddle and the request-triggered balanced control are retained below. "
         "An improvement over origin-saddle fast does not establish that every valid fast oracle needs switching, "
         "or that one-switch is preferable to all causal controls. Physical shortage and activation costs can favor different policies."
         if en else "Ниже сохранены fast с седловой точкой последнего запроса и балансирующий метод после первого запроса. "
         "Выигрыш у fast с выбором нуля не устанавливает необходимость переключения для любого допустимого fast-оракула "
         "или преимущество one-switch над всеми причинными методами. По дефициту обслуживания и затратам активации "
         "предпочтительными могут оказаться разные стратегии."))
    add("## Fixed game and protocol" if en else "## Фиксированная игра и протокол")
    add((f"The horizon is T={T}, the origin-only prefix lasts H={H} rounds, and the environment changes on H+1. "
         "The learner chooses p∈[0,1]. A request uses a genuine distinct resource coordinate from a catalogue fixed before play; "
         "an idle round is ℓ=0. There are M=T−H=163840 request coordinates."
         if en else f"Горизонт T={T}, префикс без запросов длится H={H} раундов, среда меняется на H+1. "
         "Игрок выбирает p∈[0,1]. Запрос задействует собственную координату ресурса из каталога, "
         "заданного до начала игры; отсутствие запроса соответствует ℓ=0. Координат запросов M=T−H=163840."))
    add("```text\nu(p, ℓ) = ((1−p) ε r, (1−p) ε ℓ, p(γ₀−(γ₀−γ₁)r)), r=Σℓᵢ\nε=1/√2, γ₀=1, γ₁=0.2, β=0.6386979044864147\np*(ℓ)=1 iff 2εr > γ₀−(γ₀−γ₁)r; equality chooses 0\nS(Q)={0} before a request; after a request and origin:\nS(Q)={(A,w,E): w≥0, Σwᵢ=A, E≥0, 2A+E≤β, support(w)⊆observed classes}\n```")
    add(("This is the full strict target, including benchmark responses at every interior hull mixture and tie-boundary limit. "
         "Endpoint distance is computed per episode before averaging. The analytical KKT projection retains its support gap "
         "and feasibility error; curves use certified full-target enclosures rather than an endpoint-only target surrogate. "
         "The fixed known Lipschitz bound is max(ε√(M+1),0.8√M)."
         if en else "Это полная строгая цель, включающая отклики на все внутренние смеси наблюдённой оболочки "
         "и пределы на границе равенства. Расстояние в конце вычисляется отдельно в каждом эпизоде, затем усредняется. "
         "Для аналитической проекции KKT сохранены зазор опорной функции и ошибка допустимости; кривые используют "
         "границы расстояния до полной цели. Известная заранее граница Липшица равна max(ε√(M+1),0.8√M)."))
    add(("The original budget **G=6T^(3/4)=69511.42501776238** and original block-safe base are used. "
         "The crossing round still uses fast; a fresh safe run starts on the next round. The master never periodically restarts. "
         "Original fast selects the observed origin in a tied saddle set; the recent-request control selects another valid tied saddle."
         if en else "Используются исходный бюджет **G=6T^(3/4)=69511.42501776238** и исходный блочный safe. "
         "Раунд превышения бюджета ещё выполняется в режиме fast; новый safe начинается со следующего раунда. "
         "Периодических перезапусков master нет. Основной fast выбирает наблюдённый ноль среди равнозначных "
         "седловых точек; альтернативный fast использует другую допустимую седловую точку."))
    add("```text\nAfter H: qₜ=0.05+0.9 sigmoid((θ−mean(previous p over memory))/0.03)\nθ∼Uniform[0.45,0.55]; request iff shared Uₜ<qₜ, Uₜ∼Uniform[0,1]\n```")
    add(("Only past learner actions enter qₜ. Methods share θ and all exogenous uniforms, with a separate action history "
         "for each policy. Therefore reactive realized paths and own-hull targets can differ. "
         "Own-target distance contrasts measure adaptation to each policy's realized hull, not success against one identical "
         "service target. Exogenous q=0.5 and stationary-origin cells retain common realized paths. "
         "All policies use the same known game; θ, future uniforms, and evaluator outputs are unavailable to them."
         if en else "В qₜ входят только прошлые действия игрока. Методы получают общие θ и внешние равномерные числа, "
         "но у каждого своя история действий. Поэтому реактивные траектории и собственные цели могут различаться. "
         "Сравнение расстояний до собственных целей измеряет адаптацию к реализованным оболочкам, "
         "а не качество обслуживания относительно одной общей цели. При внешних поступлениях q=0.5 и в стационарном "
         "контроле траектории общие. Игра известна всем методам; θ, будущие случайные числа и ответы оценщика им недоступны."))
    add((f"The [fixed protocol](../../docs/resource_validation_protocol.json) and [public source commit]({source_url}) "
         f"are identified by the [precommitment receipt](precommitment.json), frozen at {receipt['frozen_at_utc']}. "
         "The code and protocol were published before validation; the receipt was recorded locally before validation "
         "and published with these results. This is not registration in an external study registry. "
         "The design follows the earlier exploratory scalar studies retained in this repository. "
         "Development seeds 30000–30015 are excluded. Validation uses all seeds 40000–40255 in each cell; "
         "no seed replacement, optional stopping, scenario selection, or restriction to switched episodes is applied."
         if en else f"[Фиксированный протокол](../../docs/resource_validation_protocol.json) и "
         f"[публичная версия исходников]({source_url}) указаны в [квитанции фиксации](precommitment.json), "
         f"созданной {receipt['frozen_at_utc']}. Код и протокол опубликованы до итогового запуска; квитанция "
         "сохранена локально до запуска и опубликована с результатами. Это не регистрация во внешнем реестре. "
         "План следует за предыдущими исследовательскими скалярными тестами, сохранёнными в репозитории. "
         "Подготовительные seed 30000–30015 исключены. "
         "В каждом сценарии использованы все seed 40000–40255: без замены seed, досрочной остановки, "
         "выбора выигрышного сценария или отбора только переключившихся эпизодов."))
    add("## All prespecified outcomes" if en else "## Результаты всех заранее заданных методов")
    add(("Entries are episode means. Unserved is total unmet request units divided by **all T rounds**, "
         "not the fraction conditional on requests. Activation includes idle cost plus loaded overhead; idle is shown separately. "
         "These are modeled costs, without measured operating-energy or fairness claims. † means every stored episode "
         "satisfies analytical target membership; the parenthesis reports the largest numerical KKT displacement. "
         "Raw KKT means, quantiles, and all certificates remain in JSON."
         if en else "В таблицах — средние по эпизодам. Дефицит означает сумму необслуженных единиц запроса, "
         "делённую на **все T раундов**, а не долю среди поступивших запросов. Активация включает стоимость простоя "
         "и накладные расходы при нагрузке; простой приведён отдельно. Затраты заданы моделью; измерение энергопотребления "
         "или справедливости не заявляется. † означает выполнение аналитического условия принадлежности "
         "на сохранённых числах во всех эпизодах; в скобках — наибольший численный остаток KKT. "
         "Исходные средние и квантили KKT, а также сертификаты сохранены в JSON."))
    for scenario in design["scenarios"]:
        name = scenario["name"]
        add("### " + SCENARIO_NAMES[name][language])
        rows = []
        for method in design["methods"]:
            stat = analysis["scenarios"][name]["methods"][method]
            diag = diagnostics["scenarios"][name]["methods"][method]
            rows.append([METHOD_NAMES[method][language], terminal_cell(diag, language),
                         number(stat["mean_unserved_fraction"]["mean"]),
                         number(stat["mean_activation_cost"]["mean"]),
                         number(stat["mean_idle_cost"]["mean"])])
        add(table(["Method", "Terminal distance", "Unserved / round", "Activation / round", "Idle / round"] if en else
                  ["Метод", "Итоговое расстояние", "Дефицит / раунд", "Активация / раунд", "Простой / раунд"], rows))
    add("## Primary paired inference" if en else "## Основные парные сравнения")
    add(("Differences are **one-switch minus comparator**; negative favors one-switch. The independent complete episode "
         "is the sampling unit. The table retains numerical t/Holm/bootstrap diagnostics for all four planned tests, "
         "and separately gives outward intervals incorporating endpoint numerical uncertainty. Ordinary intervals use 95%; "
         "the four-comparison family uses 98.75% marginal intervals. Tiny differences between two membership-zero methods "
         "are floating-summation effects and have no scientific direction, even when a raw numerical p-value is small."
         if en else "Разность — **one-switch минус сравниваемый метод**; отрицательное значение в пользу one-switch. "
         "Единица наблюдения — полный независимый эпизод. Сохранены численные диагностики t/Holm/bootstrap "
         "для всех четырёх заданных тестов; отдельно приведены интервалы с учётом численной неопределённости. "
         "Обычный уровень — 95%, для семейства из четырёх сравнений — 98,75% на каждый интервал. "
         "Малые различия между методами с нулём по принадлежности возникают из суммирования чисел "
         "и не имеют научного направления, даже если численное p-value мало."))
    ci_rows, diagnostic_rows = [], []
    for control in COMPARATORS:
        contrast = primary["paired_differences"][f"terminal_delta__{control}"]
        ci_rows.append([METHOD_NAMES[control][language], number(contrast["mean_difference"]),
                        interval(contrast["numeric_and_sampling_ci_95"]),
                        interval(contrast["numeric_and_sampling_ci_family"]),
                        interval(contrast["bootstrap_mean_ci_95"]),
                        interval(contrast["bootstrap_mean_ci_family"])])
        p_display = ("undefined" if en else "не определён") if not contrast["holm_test_defined"] else (
            f"log10 p={number(contrast['log10_holm_p_value'])}")
        rawp = number(contrast["p_value"]) if not contrast["p_value_underflow"] else (
            "underflow; log p=" + number(contrast["log_p_value"]))
        robust = contrast["kkt_bound_comparisons"]
        diagnostic_rows.append([METHOD_NAMES[control][language], rawp, p_display,
                                contrast["inference_status"],
                                f"{contrast['wins']}/{contrast['ties']}/{contrast['losses']}",
                                f"{robust['wins']}/{robust['unresolved']}/{robust['losses']}",
                                contrast["both_stored_membership_zero"]])
    add(table(["Comparator", "Raw numerical mean Δ", "Outward 95% CI", "Outward 98.75% CI", "Bootstrap 95%", "Bootstrap 98.75%"] if en else
              ["Сравнение", "Численное среднее Δ", "95% + численная ошибка", "98,75% + численная ошибка", "Bootstrap 95%", "Bootstrap 98,75%"], ci_rows))
    add(table(["Comparator", "Raw p diagnostic", "Holm diagnostic", "Status", "Raw W/T/L", "KKT W/U/L", "Both membership-zero"] if en else
              ["Сравнение", "Исходный p (диагностика)", "Holm (диагностика)", "Статус", "Численные W/T/L", "KKT W/U/L", "Оба нуля по принадлежности"], diagnostic_rows))
    add(("W/T/L are numerical wins/exact-floating ties/losses; W/U/L use KKT endpoint bounds with unresolved comparisons. "
         "Both-membership-zero counts use the separate analytical diagnostic. P-values that underflow are retained on a "
         "finite log scale; a displayed underflow is not mathematical p=0. Constant differences have undefined tests. "
         "Bootstrap uses 9999 paired whole-episode draws with fixed seed 2026100403 and percentile intervals. "
         "Its raw intervals do not account for oracle uncertainty; compare the outward intervals. "
         "The remaining scenarios and uniform/trigger/recent-saddle controls are descriptive."
         if en else "W/T/L — выигрыши, точные равенства сохранённых чисел и проигрыши; W/U/L вычислены "
         "по границам KKT, где U — неразрешённое сравнение. Оба нуля по принадлежности — отдельная аналитическая диагностика. "
         "Слишком малые p-value сохранены в конечном логарифмическом масштабе; underflow не означает математический p=0. "
         "Для постоянной разности тест не определён. Bootstrap использует 9999 парных перевыборок полных эпизодов "
         "с seed 2026100403 и процентильные интервалы. Эти интервалы не учитывают численную погрешность оракула; "
         "нужно сопоставлять их с интервалами, расширенными на эту погрешность. Остальные сценарии "
         "и методы uniform/trigger/recent-saddle описательные."))
    add("## Endpoint distribution and numerical resolution" if en else "## Распределение итогов и численная разрешимость")
    add(("Two separate checks are retained. **KKT Z/P/U** uses upper=0 / lower>0 / unresolved numerical distance. "
         "**Membership** evaluates the proven full-target inequality on stored aggregate/activation totals. "
         "Within the model A=Σw, yet summing these two forms in different orders can leave a nonzero KKT displacement. "
         "Membership-zero does not mean a formal exact-real-arithmetic certificate. No arbitrary cutoff chooses which algorithm wins. "
         "CP denotes exact two-sided 95% Clopper–Pearson intervals; their sampling exactness does not validate floating arithmetic."
         if en else "Сохранены две отдельные проверки. **KKT Z/P/U** означает верхнюю границу 0 / нижнюю границу >0 / "
         "неразрешённое численное расстояние. **Принадлежность** проверяет доказанное неравенство полной цели "
         "на сохранённых суммах дефицита и активации. В модели A=Σw, но разные порядки суммирования "
         "могут давать ненулевой остаток KKT. Ноль по принадлежности не является формальным сертификатом "
         "в точной вещественной арифметике. Произвольный порог для выбора победителя не вводится. "
         "CP — точные двусторонние 95% интервалы Clopper–Pearson; их статистическая точность не проверяет арифметику чисел."))
    for scenario in design["scenarios"]:
        name = scenario["name"]
        add("### " + SCENARIO_NAMES[name][language])
        diagnostic_rows = []
        for method in design["methods"]:
            diag = diagnostics["scenarios"][name]["methods"][method]
            kkt, membership = diag["kkt_certificate_classification"], diag["stored_float_membership_diagnostic"]
            diagnostic_rows.append([METHOD_NAMES[method][language], f"{kkt['zero']}/{kkt['positive']}/{kkt['unresolved']}",
                                    interval(kkt["positive_fraction_ci_95"]),
                                    interval(kkt["possible_positive_population_fraction_enclosure_95"]),
                                    f"{membership['inside']}/{n}", interval(membership["outside_fraction_ci_95"]),
                                    number(membership["minimum_membership_margin_when_requests_observed"])])
        add(table(["Method", "KKT Z/P/U", "CP definite-positive", "CP possible-positive enclosure", "Membership zero", "CP outside-membership", "Min β−2A−E"] if en else
                  ["Метод", "KKT Z/P/U", "CP положительных KKT", "CP с неразрешёнными", "Ноль по принадлежности", "CP вне цели по числам", "Мин. β−2A−E"], diagnostic_rows))
    add(("If every endpoint is unresolved under KKT bounds, the possible-positive population enclosure may be [0,1]; "
         "the independent analytical membership check supplies more information in this game. Membership margins are shown "
         "only for episodes with requests. No observed outside-membership episodes does not prove equality or practical "
         "equivalence in the generator population. A 0/256 count still has a positive upper CP limit. "
         "The JSON also retains combined stored-float zero/positive/unresolved counts and all residual maxima."
         if en else "Если все итоги неразрешимы по границам KKT, интервал доли потенциально положительных расстояний "
         "может быть [0,1]; отдельная аналитическая проверка принадлежности даёт больше информации для этой игры. "
         "Запас неравенства показан только для эпизодов с запросами. Отсутствие наблюдений вне цели "
         "не доказывает равенство или практическую эквивалентность в генеральной совокупности генератора. "
         "При 0/256 верхняя граница CP остаётся положительной. JSON дополнительно содержит совместные "
         "численные категории ноль/положительное/неразрешённое и максимумы остатков."))
    add("## Budget crossing and conditional safe tails" if en else "## Превышение бюджета и условные хвосты safe")
    switching_rows = []
    for scenario in design["scenarios"]:
        name = scenario["name"]
        case = diagnostics["scenarios"][name]
        crossing = case["switching"]
        quantiles = crossing["conditional_round_quantiles_05_50_95"]
        switching_rows.append([SCENARIO_NAMES[name][language], f"{crossing['n_switched']}/{crossing['n_all']}",
                               interval(crossing["fraction_ci_95"]),
                               "—" if quantiles is None else "/".join(number(x) for x in quantiles),
                               number(crossing["mean_safe_rounds_all_episodes"]),
                               "pass" if case["conditional_safe_tail_certificates_hold"] else "FAIL"])
    add(table(["Scenario", "Switched / all", "Exact 95% fraction CI", "τ 5/50/95% (switched only)", "Mean safe rounds (all)", "Tail bound check"] if en else
              ["Сценарий", "Переключились / все", "Точный 95% интервал доли", "τ 5/50/95% (переключившиеся)", "Средний хвост safe (все)", "Проверка границы"], switching_rows))
    add(("The probability denominator retains every episode. Crossing quantiles are conditional descriptions, not a "
         "filtered performance comparison. Safe-tail checks use the tail's own observed hull and h·δ_tail≤6h^(3/4) "
         "with the saved numerical tolerance 2e−7. They check the original instantiated bound computationally; "
         "the data do not replace its mathematical proof. Master's E freezes after crossing; fast-only E belongs to a "
         "different counterfactual closed-loop path and is labeled separately."
         if en else "В знаменателе вероятности остаются все эпизоды. Квантили переключения — условное описание, "
         "а не сравнение качества после отбора. Хвост safe проверяется относительно собственной оболочки хвоста: "
         "h·δ_tail≤6h^(3/4), с сохранённым численным допуском 2e−7. Это вычислительная проверка исходной границы; "
         "эксперимент не заменяет её доказательство. E master фиксируется после переключения; E непрерывного fast "
         "относится к другой контрфактической траектории и подписано отдельно."))
    add("## Figures" if en else "## Графики")
    add(("- [Switching dynamics and paired distance differences](figures/resource_switching_dynamics.pdf): "
         "fixed phase boundary, master/fast residuals relative to the budget, distance enclosures and paired differences.\n"
         "- [Endpoint outcomes and scenario sensitivity](figures/resource_outcomes_and_sensitivity.pdf): "
         "all controls, terminal distance, physical unmet demand, idle/loaded activation cost and crossing probability."
         if en else "- [Динамика переключения и парные разности расстояний](figures/resource_switching_dynamics.pdf): "
         "граница фаз, остатки master/fast относительно бюджета, границы расстояний и парные разности.\n"
         "- [Итоговые результаты и чувствительность к среде](figures/resource_outcomes_and_sensitivity.pdf): "
         "все методы, итоговое расстояние, дефицит запросов, простой/нагрузка и вероятность переключения."))
    add(("The English [captions](figures/captions.json) identify every quantity. Dark curve fill is numerical target "
         "enclosure; the outer light fill additionally includes 95% pointwise whole-episode bootstrap uncertainty "
         "(2000 draws, seed 2026100404). Neither is a simultaneous confidence band. A curve's midpoint is a displayed "
         "enclosure midpoint, not an exact distance observation. The fixed full-trace seeds are 40000 and 40001."
         if en else "Английские [подписи](figures/captions.json) определяют все величины. Тёмная область кривой — "
         "численные границы расстояния; внешняя светлая дополнительно включает точечную 95% неопределённость "
         "bootstrap полных эпизодов (2000 перевыборок, seed 2026100404). Это не одновременные доверительные полосы. "
         "Средняя линия области — середина границ, а не точное наблюдение расстояния. Seed полных трасс "
         "заданы заранее: 40000 и 40001."))
    add(("The first request expands the target from {0} to the full resource polytope. A sharp distance drop near H+1 "
         "can therefore reflect target expansion, without immediate physical learning or elimination of unmet demand. "
         "Zero target distance is compatible with positive physical shortage; inspect both the distance and shortage/cost panels."
         if en else "Первый запрос расширяет цель из {0} до полного многогранника ресурсов. Поэтому резкое падение "
         "расстояния около H+1 может отражать расширение цели, без мгновенного обучения или исчезновения "
         "необслуженных запросов. Нулевое расстояние совместимо с положительным физическим дефицитом; "
         "нужно смотреть одновременно на расстояние и панели дефицита/затрат."))
    add("## Use in the paper" if en else "## Использование в статье")
    add(("Use this as a synthetic mechanism test of the specified one-switch instantiation. The main figure is "
         "the switching-dynamics figure; keep all eight controls and all four scenarios in a full table in the "
         "appendix or supplementary material, with the strong controls disclosed in the main discussion. "
         "Discuss the benign-prefix benefit separately from terminal fallback, and disclose target expansion "
         "and policy-dependent realized hulls. The following English paragraph is generated from the completed results. "
         "It does not alter the manuscript or assert year-specific conference formatting requirements."
         if en else "Используйте результат как синтетическую проверку механизма конкретной реализации one-switch. "
         "Основной рисунок — динамика переключения; полную таблицу восьми методов и четырёх сценариев "
         "сохраните в приложении или дополнительных материалах, а сильные методы обозначьте в основной дискуссии. "
         "Преимущество на спокойном префиксе обсуждайте отдельно от итогового fallback; раскройте расширение цели "
         "и зависимость реализованной оболочки от политики. Следующий английский абзац построен по завершённым результатам. "
         "Он не редактирует рукопись и не заявляет правила формата конференции для определённого года."))
    origin = primary["paired_differences"]["terminal_delta__shared_past_hull"]
    one_members = primary["methods"]["one_switch"]["stored_float_membership_diagnostic"]["inside"]
    recent_members = primary["methods"]["fast_recent_saddle"]["stored_float_membership_diagnostic"]["inside"]
    trigger_members = primary["methods"]["request_trigger"]["stored_float_membership_diagnostic"]["inside"]
    origin_mean = analysis["scenarios"]["primary"]["methods"]["shared_past_hull"]["terminal_delta"]["mean"]
    paragraph = (
        f"We evaluated a fixed synthetic vector resource-service game using {n} independent paired episodes "
        f"in each of four prespecified scenarios (eight causal policies; T={T}, H={H}, "
        "G=6T^{3/4}). The primary reactive environment used only past service activation. "
        f"One-switch satisfied analytical terminal membership on stored floats in {one_members}/{n} episodes, "
        f"whereas fast with the specified origin saddle tie had mean full-target distance {number(origin_mean)}. "
        f"The numerical paired mean difference was {number(origin['mean_difference'])}, with an outward "
        f"98.75% interval {interval(origin['numeric_and_sampling_ci_family'])} accounting for endpoint numerical uncertainty. "
        f"On the deterministic origin-only prefix, mean distance was {number(early['one_switch_mean_distance'])} "
        f"for one-switch and {number(early['block_safe_mean_distance'])} for block safe; this comparison has no sampling p-value. "
        f"An alternative valid fast saddle choice and the request-triggered balanced control also satisfied "
        f"terminal membership in {recent_members}/{n} and {trigger_members}/{n} episodes, respectively, "
        "limiting any claim that switching is generally necessary. Reactive policies may induce different paths "
        "and own-hull targets; zero target distance can coexist with unmet demand, and the first request itself "
        "expands the target. We therefore report all controls and separate physical shortage and activation costs."
    )
    add("> " + paragraph)
    add("## Reproduction and limits" if en else "## Воспроизведение и ограничения")
    add(("From the repository root, use this result release, whose critical simulator and protocol match the frozen "
         "source commit, and a fresh output directory; the runner refuses "
         "to overwrite completed episodes. The pilot is optional for reproduction and does not contribute to inference."
         if en else "Из корня репозитория используйте этот выпуск результатов: его критические исходники и протокол "
         "совпадают с зафиксированной версией. Укажите новый каталог результатов; "
         "runner отказывается перезаписывать завершённые эпизоды. Подготовительный запуск необязателен "
         "для воспроизведения и не входит в статистические выводы."))
    add("```sh\npython -m pip install -r requirements-lock.txt\npython -m pip install -e \".[test]\"\npython -m pytest tests/test_resource_validation.py tests/test_resource_target.py\npython scripts/run_resource_validation.py --stage pilot --output results/resource_validation_replay\npython scripts/run_resource_validation.py --stage freeze --output results/resource_validation_replay\npython scripts/run_resource_validation.py --stage validation --output results/resource_validation_replay\npython scripts/run_resource_validation.py --stage analyze --output results/resource_validation_replay\npython scripts/build_resource_validation_figures.py --output results/resource_validation_replay\npython scripts/build_resource_validation_report.py --output results/resource_validation_replay\n```")
    add(("The report and figure builders are postprocessors supplied with this result release; the frozen receipt covers "
         "the protocol, simulator, target oracle, paired-inference code, and original learners. "
         "[Statistical summary](analysis.json), [distribution diagnostics](distribution_diagnostics.json), and "
         "[SHA-256 manifest](reproducibility_manifest.json) retain exact filenames and hashes. Episode tables reside in "
         "`primary/episodes.json`, `memory16/episodes.json`, `exogenous/episodes.json`, and `stationary/episodes.json`. "
         "NPZ dynamics and representative raw arrays are kept locally as large generated artifacts; regenerate them "
         "with the fixed seeds and verify their saved SHA-256 values rather than assuming that a public checkout contains them. "
         "Frozen runtime versions: " + ", ".join(f"{key} {value}" for key, value in receipt["versions"].items()) + "."
         if en else "Построители отчёта и графиков — постпроцессоры этого выпуска результатов; квитанция фиксирует "
         "протокол, симулятор, оракул цели, парную статистику и исходные алгоритмы. "
         "[Сводка эпизодов](analysis.json), [диагностика распределений](distribution_diagnostics.json) и "
         "[манифест SHA-256](reproducibility_manifest.json) сохраняют имена файлов и хеши. Таблицы эпизодов находятся в "
         "`primary/episodes.json`, `memory16/episodes.json`, `exogenous/episodes.json`, `stationary/episodes.json`. "
         "Динамика NPZ и выбранные полные трассы хранятся локально как крупные генерируемые артефакты; "
         "их следует воспроизвести с фиксированными seed и проверить по сохранённым SHA-256, "
         "а не предполагать наличие в публичной копии. Версии среды при фиксации: " +
         ", ".join(f"{key} {value}" for key, value in receipt["versions"].items()) + "."))
    add(("These findings concern one finite synthetic game and one specified nonanticipating reaction rule. "
         "They do not establish representativeness for real resource systems, attacker learning, empirical minimax "
         "rates, or a q=4 result. Changing T would change the catalogue dimension and Lipschitz constant. "
         "All controls and sensitivity cells remain reportable, including zero-distance ties and negative comparisons; "
         "there is no post hoc selection of a winning method or data regime."
         if en else "Выводы относятся к одной конечной синтетической игре и заданному правилу реакции, "
         "не использующему будущее. Они не устанавливают репрезентативность для реальных систем, "
         "обучение атакующего, эмпирические минимаксные скорости или результат для q=4. "
         "При изменении T менялись бы размер каталога и константа Липшица. Сохранены все методы и сценарии, "
         "включая нулевые расстояния и неблагоприятные сравнения; победитель и режим данных "
         "не отбираются после просмотра результатов."))
    return "\n\n".join(text) + "\n"


def build_manifest(output, receipt, checked_sources, design):
    artifact_hashes, raw_artifacts = {}, []
    target = output / "reproducibility_manifest.json"
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == target:
            continue
        relative = path.relative_to(output)
        if "development" in relative.parts or any(part.startswith(".") for part in relative.parts):
            continue
        name = relative.as_posix()
        artifact_hashes[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".npz":
            raw_artifacts.append(name)
    postprocessors = {}
    for name in ("scripts/build_resource_validation_report.py", "scripts/build_resource_validation_figures.py"):
        postprocessors[name] = sha256(ROOT / name)
    return {
        "schema_version": 1, "hash_algorithm": "SHA-256",
        "frozen_at_utc": receipt["frozen_at_utc"],
        "source_commit": receipt["git_commit"],
        "source_commit_url": "https://github.com/Jew-Yeah/aamas-lowdim-experiments/commit/" + receipt["git_commit"],
        "protocol_path": "docs/resource_validation_protocol.json",
        "protocol_sha256": receipt["protocol_sha256"],
        "critical_source_receipt": receipt["source_sha256"],
        "critical_source_hashes_verified": checked_sources,
        "postprocessor_sha256": postprocessors,
        "software_versions_at_freeze": receipt["versions"],
        "seeds": design["validation_seeds"], "representative_seeds": design["representative_full_paths"],
        "artifact_paths_relative_to": "the report output directory",
        "excluded": ["reproducibility_manifest.json itself", "development/**", "hidden files/directories"],
        "raw_artifact_policy": "Listed NPZ arrays are local generated artifacts, not automatically published. Recreate the batch with the fixed protocol and seeds; compare SHA-256. Published JSON and figures retain the complete episode denominators.",
        "local_raw_artifacts": raw_artifacts,
        "artifact_sha256": artifact_hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/resource_validation")
    args = parser.parse_args()
    output = args.output.resolve()
    design, receipt, analysis, batches, checked = verify_inputs(output)
    diagnostics = build_diagnostics(design, analysis, batches)
    write_json(output / "distribution_diagnostics.json", diagnostics)
    for language, name in ((0, "README.md"), (1, "README.ru.md")):
        (output / name).write_text(render_report(design, receipt, analysis, diagnostics, batches, language),
                                   encoding="utf-8", newline="\n")
    write_json(output / "reproducibility_manifest.json", build_manifest(output, receipt, checked, design))
    print("Created English/Russian reports, distribution diagnostics, and a verified SHA-256 manifest.")


if __name__ == "__main__":
    main()
