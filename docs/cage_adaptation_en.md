[English](cage_adaptation_en.md) | [Русский](cage_adaptation_ru.md) | [Home](../README.md)

# Selected implementation: validation and final test

The main report presents continuous scalar-aware one-switch with **W = 16,
rho = 0.25**, the selected 16-round window, and selected Hedge with learning-rate
multiplier 4. Parameters were chosen on validation before final outcomes were
read. The [frozen protocol](cage_adaptation_protocol.json),
[locked selection](../results/cage_adaptation/selection.json), and
[validation grid](../results/cage_adaptation/validation_grid.json) are preserved.

## Oracle and baselines

The scalar-aware oracle minimizes the calibrated loss against recent empirical
mode frequencies among saddle responses satisfying
`B.T @ p <= L + rho / sqrt(t)`. It preserves the original minimax dual witness,
full past hull, direction update, uniform first move, and theorem-derived switch
budget. Failed candidate checks fall back to the original saddle pair. See the
[mathematical implementation](algorithms_en.md).

Window responds directly to the frequencies of the last W disclosed modes.
Before any observation it responds to the known initial Meander mode. Hedge
updates its exponential defense weights using the same calibrated scalar
losses. It uses the original learning-rate formula multiplied by the selected
factor. All methods share training information and post-choice mode disclosure.
None receives final-bank outcomes during its updates.

## Separate banks and locked selection

The unchanged training fit uses 40 simulator seeds in `data/cage2` for each of
18 defense/attack cells. Every episode has 50 native steps. The primary outcome
is the sum of the four native loss coordinates per simulator step; lower is better.

The new banks contain:

| Bank | Simulator seeds | Role |
|---|---|---|
| Validation | 38000000–38000099, 100 seeds | Configuration selection |
| Final test | 39000000–39000399, 400 seeds | Final performance measurement |

These banks total **9000 new simulator episodes and 450000 native steps**.
Seed indices are shared across cells. Whole cell tables remain paired during
analysis; different policies may consume random numbers differently.

Validation evaluates 20 common curriculum paths, seeds 40000000–40000019, at
horizon 512. Its declared grid has 30 configurations:

- Scalar-aware oracle: W in {4, 8, 16, 32, 64}, rho in {0, 0.25, 0.5, 1}.
- Window: the same five W values.
- Hedge: multipliers {0.25, 0.5, 1, 2, 4} of the original learning rate.

Each family's lowest mean validation loss selects its configuration. Exact ties
use the declared grid order. Input and implementation hashes and selected
parameters were locked before accessing final-bank scores. Original and
exploratory comparisons do not determine the final choice. No further parameter
search or first-action change was made after inspecting the final results.

## Final comparisons

Fifty new common curriculum paths, seeds 41000000–41000049, evaluate the locked
configurations at horizon 512. The two primary contrasts are selected one-switch
minus selected window, and selected one-switch minus selected Hedge. A paired
crossed bootstrap resamples whole simulator seed tables and whole paths, with
10000 draws and seed 44000000. Bonferroni adjustment gives simultaneous family
coverage of approximately 95% for the two reported intervals.

| Method | Mean native loss |
|---|---:|
| Selected one-switch | 1.1041833342 |
| Selected window | 1.1015373414 |
| Selected Hedge | 1.1660012489 |

| Contrast | Mean difference | Simultaneous 95% interval |
|---|---:|---:|
| One-switch − window | +0.0026459928 | [+0.0025386010, +0.0027535403] |
| One-switch − Hedge | −0.0618179146 | [−0.0800977015, −0.0442132772] |

The selected method improves on tuned Hedge by about 5.3%; window has about
0.24% lower loss. All 50 test paths have **exactly identical one-switch and
window actions after round 1**. The first uniform one-switch move explains the
entire gap. A prior-informed first move was not part of the locked experiment.
There are no safe switches, fallbacks, or excesses over the nominal saddle-gap
allowance on the primary test. [Trace diagnostics](../results/cage_adaptation/trace_diagnostics.json)
record the action comparison and checkpoint hashes.

The intervals are approximate and conditional on the original fitted training
model and selected parameters. They omit training and model-selection uncertainty.
Reusing cell means over 512 rounds does not create 512 new simulator samples.

## Reproduce the published analysis

After [installing the package](cage_en.md), rebuild the focused report from the
published aggregate inputs:

```bash
python scripts/build_cage_adaptation_report.py --run-dir results/cage_adaptation --bank-root data/cage2_adaptation --output results/runs/focused_report
```

The builder checks bank, selection, and aggregate hashes and independently
reconstructs the primary means. It keeps both primary comparisons and creates
PNG/PDF figures. The raw aggregate file retains the original six-method primary
run; the presentation highlights the three selected methods.

To rerun the selected trajectories without retuning:

```bash
python -m lowdim_games.cli cage-selected --output results/runs/cage_selected
```

For a short functional check use `--horizon 16 --seeds 41000000`. A short run
is not evidence for the locked statistical comparison.

To reproduce the original validation and test stages from the public banks:

```bash
python scripts/run_cage_adaptation_study.py --stage validation --bank-root data/cage2_adaptation --output results/runs/cage_adaptation_replay --workers 8
python scripts/run_cage_adaptation_study.py --stage select --bank-root data/cage2_adaptation --output results/runs/cage_adaptation_replay
python scripts/run_cage_adaptation_study.py --stage test --bank-root data/cage2_adaptation --output results/runs/cage_adaptation_replay --workers 8
```

A clean pinned simulator checkout is required only for collecting episodes again:

```bash
python scripts/collect_cage_bank.py --steps 50 --seed 38000000 --count 100 --partition external --exclude-calibration data/cage2 --workers 4 --output results/runs/recollected_banks/validation50
python scripts/collect_cage_bank.py --steps 50 --seed 39000000 --count 400 --partition heldout --exclude-calibration data/cage2 --workers 8 --output results/runs/recollected_banks/test50
```

Full trajectory checkpoints remain in ignored runtime directories. Changed
configurations require a new directory. The frozen runner and learning modules
remain byte-identical to those used for selection and testing.

## Supplementary figures

The [gallery](../results/cage_adaptation/figures/README.md) provides English and
Russian PNG/PDF figures and links to their underlying arrays. The supplementary
[figure protocol](cage_figures_protocol.json) was frozen before the new horizon
runs. It does not alter the original validation, selection, or two primary
contrasts. Loss curves and component intervals use 2000 paired crossed bootstrap
draws (seed 46000000); their 95% bands are descriptive and pointwise.

Vector distance is computed against the full response target in the fixed
normalized training game, with numerical lower/upper distances and feasibility
checks. The moving realized-hull target and a retrospective fixed final target
are both displayed. A separate sweep uses all 20 new seeds 45000000–45000019
at each horizon 64, 128, 256, 512, 1024 and 2048, retaining the selected settings.
Phase lengths and the original learning-rate formulas scale with the announced
horizon. No additional simulator episodes or parameter search are involved.
Numerical-zero errors cannot support a fitted decay order.

```bash
python scripts/build_cage_dynamic_figures.py --output results/runs/dynamic_rebuild
python scripts/build_cage_geometry_figures.py --stage report
```

The figure-specific README files explain how to recompute the arrays as well
as redraw the figures. Uncertainty for vector distance concerns whole attacker
paths conditional on the training game; held-out simulator tables are used
only for the native-loss analysis.

## Presentation cleanup and limitations

The main branch focuses on the validated primary result. The
[immutable full-study snapshot](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/e8c85cb2e7be49032c1c2d014b6f19084ea064a8)
and [archive branch](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/archive/full-study-2026-10-04)
preserve previous applications, original-method results, and all restart studies.
The original one-switch had higher scalar loss. Restarts helped the original
implementation under alternating attacks but hurt it under gradual changes;
they are exploratory and are excluded from the recommended configuration.
The original protocol and primary raw aggregates remain available on main.

This study measures model-informed policy selection in independently reset
simulator episodes. Red selects fixed modes, and its mode is disclosed after
choice. It does not evaluate deployed attack logs, persistent network state, new
learned tactics, or vector target-distance superiority. Oracle diagnostics are
floating-point evidence. The original manuscript has not been edited.
