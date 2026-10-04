[English](README.md) | [Русский](README.ru.md)

# Full one-switch trajectories

## Independent two-phase validation

The [new paired study](results/two_phase_validation/README.md) runs **800
independent episodes**, four fixed scenarios with 200 episodes each, using the
original block-safe procedure and budget. Its locally frozen design separates
development seeds from validation and reports all five methods. In the primary
synthetic scenario one-switch has lower prechange mean target distance than
block-safe, and lower terminal target distance than fast-only (0.395181 versus
0.520319). Standalone block-safe is better at the horizon (0.049686), and lag
and window are stronger still. Holm-adjusted paired inference supports these
different directions; this is not a claim of overall or real-world superiority.
See the [method notes](docs/two_phase_validation_en.md) and all fixed sensitivity
results, including episodes without switching. Full raw arrays are generated
locally from the published seeds and code rather than committed to Git.

## Real-data illustration

The manuscript's real-data illustration follows **730 chronological days of NYC request
shares**, including the first certified crossing and the complete safe tail.
It uses `u(p,ell)=(p-ell)/sqrt(2)`, benchmark `p*(ell)=ell`, and the full target
`{0}`. A separately proved lag safe base gives the conservative, data-independent
budget **G=1**. This is a different safe base from the original block routine.
All methods start uniformly and choose before the current share is revealed;
no learner restarts within the primary run.

| Method | Final vector distance δ | Mean daily mismatch norm |
|---|---:|---:|
| One-switch | 0.000939885 | 0.155910247 |
| Fast only | 0.000644365 | 0.573865013 |
| Lag safe only | 0.000167400 | 0.131152906 |
| Window, W=16 | 0.001311085 | 0.104084214 |
| Block safe only | 0.034672805 | 0.109093125 |

The master switches at **44**, on **2021-02-13**, followed by **686 safe rounds**.
It improves daily mismatch relative to fast-only, while fast has a smaller final
vector distance. Lag is best on final distance; Window is best on daily mismatch.
Signed errors can cancel in the vector average, so these metrics are distinct.
Registered request shares are a modeled allocation objective, not measured
staffing requirements or service outcomes.

A constructed, high-dimensional resource-balance diagnostic separately retains
the **original block-safe routine and G=6 T^(3/4)**. At `T=16384` it switches at
**8817**. Final distance is **0.602296** for the master, **0.992126** for fast,
and **0.124300** for standalone block-safe; lag and Window are stronger still.
Its novel labels are payoff-equivalent. This is a mechanism test, not a
low-dimensional rate experiment or a claim of general dominance.

[Full results and both dynamics figures](results/switching/README.md) retain
all five methods, both annual checks, per-round arrays and hashes.
[Method notes](docs/switching_en.md) prove the budgets and describe observation
order. These additions are post-review exploratory; fixed chronological traces
carry no confidence intervals, p-values, or confirmatory holdout claims.

## Reproduce the switching studies

Python 3.10 or later is required. Only base scientific dependencies are needed.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/run_switching_study.py
```

Use `--output results/runs/switching_replay` for a separate replay. The runner
verifies CSV hashes, replays all three NYC traces and the original-budget stress,
and exports English PNG/PDF figures. Recorded verification: **152 tests passed**
in the full development environment; optional simulator checks may skip without
CAGE. [Pinned base dependencies](requirements-lock.txt) and
[data provenance](data/switching_nyc/provenance.json) are retained.

## Frozen CAGE 2 result

The following retained study uses a scalar-aware admissible saddle oracle with
**W=16 and `rho=0.25`**, selected on separate validation. Its scientific data,
selection, results and original inference remain unchanged. CAGE is the simulator;
Window and Hedge are the comparison policies.

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
The [original CAGE report](results/cage_adaptation/README.md) includes figures,
input hashes, locked parameters, and the two primary comparisons.

The [figure gallery](results/cage_adaptation/figures/README.md) adds loss dynamics,
cumulative paired differences, four cost components, all-path distributions,
validation sensitivity, and full-target vector error. English and Russian
PNG/PDF exports are available. These descriptive additions retain the selected
parameters and the original primary comparisons.

## AAMAS paper materials

[LaTeX integration notes](paper/README.md) describe the current switching main
section and detailed supplementary methods. Current figures are in
[the switching report](results/switching/README.md); the
[frozen CAGE figure pack](results/cage_adaptation/aamas/README.md) remains secondary.
[Submission guidance](docs/aamas_submission_en.md) distinguishes official rules
from editorial choices. The final reviewer package must include the new switching
code, inputs, results and figures; an earlier CAGE-only package is insufficient.
The compiled whole manuscript determines page compliance.

## Reproduce the frozen CAGE study

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

## CAGE method and scope

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

The retained CAGE comparison uses its original selected implementation. The
[full-study archive](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/archive/full-study-2026-10-04)
preserves earlier implementations, results, and unsuccessful comparisons,
including the original one-switch and exploratory restarts. Restarts had
scenario-dependent effects and are not part of the recommended configuration.
The [snapshot commit](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/e8c85cb2e7be49032c1c2d014b6f19084ea064a8)
fixes the complete pre-cleanup study independently of future branch changes.

Algorithmic context: [Marinov et al., Efficient Opportunistic Approachability](https://proceedings.mlr.press/v313/marinov26a.html).
Simulator: [official CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2).
Code is available under the [MIT License](LICENSE). Source attributions and
provenance accompany the data. The original manuscript is outside this repository.
