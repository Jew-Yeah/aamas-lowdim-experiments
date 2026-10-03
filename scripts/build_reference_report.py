"""Build the bilingual reference report from completed, recorded runs."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUTPUT = ROOT / "results" / "reference_run"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nyc = read(RUNS / "nyc" / "nyc_summary.json")
    synthetic = read(RUNS / "synthetic" / "synthetic_summary.json")
    geometry = read(RUNS / "geometry" / "geometry.json")
    example = read(RUNS / "nyc" / "nyc_M6_capacity0.80.json")
    if len(nyc["summaries"]) != 63 or len(synthetic["summaries"]) != 105 or len(geometry["records"]) != 15:
        raise RuntimeError("Incomplete suite: expected 9/15 seven-policy comparisons and 15 geometry checks.")
    files = [RUNS / "nyc" / "nyc_summary.json", RUNS / "synthetic" / "synthetic_summary.json",
             RUNS / "geometry" / "geometry.json", RUNS / "geometry" / "geometry_moments.png",
             RUNS / "geometry" / "geometry_moments.pdf"]
    for directory, prefix in [("nyc", "nyc_M6_capacity0.80"), ("synthetic", "synthetic_q4_seed20261003_T128")]:
        files.extend(sorted((RUNS / directory).glob(prefix + "*")))
    for m in (3, 6, 12):
        files.extend(sorted((RUNS / "nyc").glob(f"nyc_preprocessing_M{m}.*")))
    for source in files:
        shutil.copy2(source, OUTPUT / source.name)
    action_errors = []
    for source in sorted((RUNS / "nyc").glob("nyc_M*_capacity*.json")):
        manifest = read(source)
        if not all(x["switch_round"] is None for x in manifest["summaries"]):
            raise RuntimeError("Unexpected switch under the stated short-horizon bound.")
        stem = source.stem
        first = np.load(source.with_name(stem + "__one_switch.npz"))["actions"]
        second = np.load(source.with_name(stem + "__shared_past_hull.npz"))["actions"]
        action_errors.append(float(np.max(np.abs(first-second))))
    quantization = "\n".join(f"| {x['profiles']} | {x['realized_simplex_affine_dimension']} | {100*x['relative_l2_rms']:.2f}% | {100*x['relative_l1_total']:.2f}% |" for x in nyc["quantization"])
    rows = "\n".join(f"| `{x['method']}` | {x['delta']:.6g} | {x['mean_unmet_requests']:.3f} | {x['mean_resource_cost']:.5f} | {x['mean_disparity_requests']:.3f} |" for x in example["summaries"])
    max_gap = max(x["projection_gap"] for x in example["curves"])
    common = """```bash
python -m pytest -q
python -m lowdim_games.cli synthetic --q 1 2 3 4 5 --horizon 128 --seeds 20261003 20261004 20261005 --output results/runs/synthetic --plots
python -m lowdim_games.cli geometry --q 1 2 3 4 5 --horizons 32 64 128 --output results/runs/geometry
python -m lowdim_games.cli nyc --profiles 3 6 12 --capacity-ratios 0.6 0.8 1.0 --output results/runs/nyc
python scripts/build_reference_report.py
```"""
    english = f"""[English](README.md) | [Русский](README.ru.md) | [Home](../../README.md)

# Recorded reference run

The initial suite was completed on 2026-10-03. It contains 15 controlled
synthetic comparisons (three seeds, q=1,...,5, T=128), nine NYC comparisons
(M=3,6,12; baseline capacity ratios 0.6,0.8,1.0; T=730), and 15 geometric
checks (q=1,...,5; T=32,64,128). Each comparison contains seven policies.
Correctness tests and the numerical regressions are separate from these runs.

## NYC example: six profiles and baseline ratio 0.8

The mathematical distance uses the quantized game. The remaining columns use
original held-out counts and the stated service model. Resource cost is
dimensionless. Tiny distances below numerical resolution are approximately zero.

| Method | Target distance | Unmet requests/day | Resource cost | Deficit disparity/day |
|---|---:|---:|---:|---:|
{rows}

Maximum full-target squared projection gap in this example: {max_gap:.3g}.

The one-switch and shared past-hull methods coincide throughout the NYC suite;
the maximum action-coordinate difference is {max(action_errors):.3g}. The
theoretical budget cannot trigger on this horizon. The constructed switch
regression separately validates the fallback logic.

The initial results do not establish superiority over allocation heuristics.
The previous-week and fixed-reserve references are useful controls: extra
capacity can improve service independently of adaptation. Target distance,
unmet demand, resource cost, and disparity measure different objectives.
The comparisons are exploratory results for the stated model.

## Quantization sensitivity

| Profiles M | Realized q | Relative L2 RMS | Relative total L1 |
|---:|---:|---:|---:|
{quantization}

The approximation error remains substantial. Dimension is a property of the
finite profile representation; this run does not establish low dimension in
the original demand trace. Target sets and normalization differ between profile
libraries, so cross-M target distances do not isolate a dimensional effect.

## Synthetic and geometric checks

The [synthetic summary](synthetic_summary.json) preserves all seeds and policies.
Regimes are introduced explicitly and then mixed, making adaptation visible.
Near-zero target distance in some runs can reflect the growing target set;
these paths do not estimate worst-case convergence exponents.
All 15 [critical-moment checks](geometry.json) satisfy the stated geometric bound
to the reported numerical accuracy. Finite-horizon checks complement the proof;
they do not prove the asymptotic bound or a game minimax lower bound.

![Full-hull target distance](nyc_M6_capacity0.80__target.png)

![Service metrics on original demand](nyc_M6_capacity0.80__service.png)

![Geometric moment checks](geometry_moments.png)

## Reproduce

{common}

The committed files include summaries, representative complete action traces,
game tensors, preprocessing arrays, and exportable PNG/PDF figures. Other
complete runs are regenerated by the commands above. Input CSV hashes and
software versions are recorded in JSON; exact installed dependencies are in
`requirements-lock.txt` at the repository root.
"""
    russian = f"""[English](README.md) | [Русский](README.ru.md) | [Главная](../../README.ru.md)

# Опорный запуск

Первоначальный набор экспериментов выполнен 2026-10-03. Он содержит 15
синтетических сравнений (три начальных значения ГСЧ, q=1,...,5, T=128),
девять сравнений NYC (M=3,6,12; отношения базовой мощности 0.6,0.8,1.0;
T=730) и 15 геометрических проверок (q=1,...,5; T=32,64,128).
В каждом сравнении участвуют семь стратегий. Тесты корректности и численные
регрессии выполняются отдельно от этих запусков.

## Пример NYC: шесть профилей и базовое отношение 0.8

Расстояние до цели относится к квантованной игре. Остальные столбцы используют
исходный тестовый спрос и заданную модель обслуживания. Стоимость ресурса
безразмерна. Расстояния ниже численного разрешения следует считать близкими к нулю.

| Метод | Расстояние до цели | Необслуженные заявки/день | Стоимость ресурса | Различие дефицитов/день |
|---|---:|---:|---:|---:|
{rows}

Максимальный разрыв прямой и двойственной оценок квадрата расстояния
для этого примера: {max_gap:.3g}.

Алгоритм с переключением и метод прошлой выпуклой оболочки совпадают во всех
запусках NYC; максимальная разность координат действий равна {max(action_errors):.3g}.
Теоретический бюджет не позволяет пересечь порог на этом горизонте.
Отдельная регрессия с построенной игрой проверяет логику безопасного продолжения.

Начальные результаты не устанавливают превосходство над эвристиками.
Стратегия предыдущей недели и фиксированный резерв позволяют проверить,
насколько улучшение обслуживания связано с дополнительной мощностью.
Расстояние до цели, дефицит, стоимость и различие округов описывают разные
цели. Эти результаты представляют предварительное исследование заданной модели.

## Чувствительность к квантованию

| Число профилей M | Реализованная q | Относительная L2 RMS | Относительная суммарная L1 |
|---:|---:|---:|---:|
{quantization}

Ошибка аппроксимации остаётся существенной. Размерность относится к конечному
представлению профилей; низкая размерность исходного спроса не установлена.
При изменении M меняются целевые множества и нормировка, поэтому сравнение
расстояний между разными библиотеками не выделяет чистый эффект размерности.

## Синтетические и геометрические проверки

[Синтетическая сводка](synthetic_summary.json) сохраняет все стратегии и
начальные значения ГСЧ. Новые режимы вводятся явно, после чего смешиваются.
Близкое к нулю расстояние в части опытов может объясняться расширением цели;
эти последовательности не оценивают худшие асимптотические показатели.
Все 15 [проверок критического момента](geometry.json) удовлетворяют оценке
в пределах указанной численной точности. Проверки на конечных горизонтах
дополняют доказательство, без установления асимптотики или игровой минимаксной границы.

![Расстояние до полной цели](nyc_M6_capacity0.80__target.png)

![Обслуживание исходного спроса](nyc_M6_capacity0.80__service.png)

![Геометрические моменты](geometry_moments.png)

## Воспроизведение

{common}

Сохранены сводки, полные действия для выбранных примеров, тензоры игр,
массивы обработки данных и графики PNG/PDF. Остальные полные запуски
восстанавливаются приведёнными командами. JSON содержит хеши входных CSV
и версии программ; точные установленные зависимости записаны в
`requirements-lock.txt` в корне репозитория.
"""
    (OUTPUT / "README.md").write_text(english, encoding="utf-8")
    (OUTPUT / "README.ru.md").write_text(russian, encoding="utf-8")
    # Canonical text bytes keep the hash manifest valid on Windows and Unix.
    for path in OUTPUT.iterdir():
        if path.suffix in {".md", ".json"}:
            path.write_bytes(path.read_text(encoding="utf-8").encode("utf-8"))
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(OUTPUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    (OUTPUT / "SHA256SUMS.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"Reference report built: {len(manifest)} files")


if __name__ == "__main__":
    main()
