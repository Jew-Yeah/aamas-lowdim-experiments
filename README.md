[English](README.md) | [Русский](README.ru.md)

# One-switch adaptation in CAGE 2

Reproducible experiments for a one-switch learner with a scalar-aware admissible
saddle oracle. The recommended implementation uses a **16-round forecast window
and `rho = 0.25`**, selected on separate validation data. The final benchmark
compares this implementation with independently tuned window and Hedge policies
in the official CAGE 2 simulator.

## Validated result

Mean native loss per simulator step, averaged over 400 held-out simulator seeds
and 50 common attacker paths of 512 policy-selection rounds. Lower is better.

| Method | Mean loss |
|---|---:|
| Selected scalar-aware one-switch | **1.10418** |
| Selected window, 16 rounds | **1.10154** |
| Selected Hedge, learning-rate multiplier 4 | 1.16600 |

The selected one-switch reduces loss by **5.3% against tuned Hedge**. The
simultaneous 95% interval for its paired loss difference is
`[-0.08010, -0.04421]`. Window performs **0.24% better**: the interval for
one-switch minus window is `[0.00254, 0.00275]`. All 50 test paths have identical
one-switch and window actions after the first round. Their first actions differ:
one-switch starts uniformly, while window responds to the known initial mode.

These results support near-window scalar performance and an advantage over
tuned Hedge in this benchmark. They do not establish superiority over window.
The [complete primary report](results/cage_adaptation/README.md) includes figures,
input hashes, locked parameters, and the two primary comparisons.

The [figure gallery](results/cage_adaptation/figures/README.md) adds loss dynamics,
cumulative paired differences, four cost components, all-path distributions,
validation sensitivity, and full-target vector error. English and Russian
PNG/PDF exports are available. These descriptive additions retain the selected
parameters and the original primary comparisons.

## Use and reproduce

Python 3.10 or later is required.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,cage]"
python -m pytest -q
python -m lowdim_games.cli cage-selected --horizon 16 --seeds 41000000 --output results/runs/cage_selected_smoke
```

The [CAGE setup](docs/cage_en.md) explains the pinned simulator and the frozen
policy menu. The [validation protocol](docs/cage_adaptation_en.md) records the
grid, seed ranges, selection chronology, statistical analysis, and reproduction
commands. Published aggregate banks allow scalar results to be reconstructed
without collecting the simulator episodes again.

`cage-selected` runs the recommended `RecommendedOneSwitchLearner` and the two
selected baselines without retuning. The short command above is a functional
check; omit `--horizon` and `--seeds` to use the recorded 512-round, 50-path setup.

## Method and scope

The learner chooses before the current attack mode is revealed. Its forecast
uses only previous observations. The oracle minimizes forecast scalar loss
among saddle responses satisfying the manuscript's allowed error. The hull,
direction update, response witness, and theorem-derived switch budget are kept.
See the [mathematical implementation](docs/algorithms_en.md).

Every compared method receives the same calibrated loss table. Attack-mode
labels are disclosed after each choice. A round chooses a complete defense
policy for a separately reset 50-step simulator episode; mixture scores average
whole-episode outcomes. Red selects among three fixed attack modes. These are
model-informed simulator experiments, not deployed-network attack logs.

The primary endpoint is scalar loss. The supplementary figures separately
measure full-target vector distance in the fixed normalized training game,
including new runs at six announced horizons. Errors at numerical precision
do not identify an asymptotic decay rate or demonstrate a benefit from the
vector guarantee. No safe switches occurred in the primary paths.
Numerical oracle residuals provide floating-point checks rather than formal
exact-arithmetic certification.

## Previous studies and reuse

The main branch focuses on the selected implementation and the primary
comparison. The [full-study archive](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/archive/full-study-2026-10-04)
preserves earlier implementations, results, and unsuccessful comparisons,
including the original one-switch and exploratory restarts. Restarts had
scenario-dependent effects and are not part of the recommended configuration.
The [snapshot commit](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/e8c85cb2e7be49032c1c2d014b6f19084ea064a8)
fixes the complete pre-cleanup study independently of future branch changes.

Algorithmic context: [Marinov et al., Efficient Opportunistic Approachability](https://proceedings.mlr.press/v313/marinov26a.html).
Simulator: [official CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2).
Code is available under the [MIT License](LICENSE). Source attributions and
provenance accompany the data. The original manuscript is outside this repository.
