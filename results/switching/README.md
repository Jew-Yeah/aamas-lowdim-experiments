[English](README.md) | [Русский](README.ru.md) | [Home](../../README.md)

# Full switching results

These are descriptive, post-review exploratory additions to the frozen CAGE
study. They examine the crossing action and the complete safe tail. No parameters
are selected from these outcomes; the fixed traces carry no confidence intervals,
p-values, or confirmatory holdout claims. All five methods are retained, including
stronger controls. Tables are rounded; [analysis.json](analysis.json) preserves
the complete recorded numbers.

## NYC: chronological 2021–2022

The 730 daily Hazard-request aggregates define shares for Brooklyn, Queens, and
the other three boroughs. The single zero-count day uses uniform shares. These
are registered requests, not staffing requirements or observed service outcomes.
[Source provenance](../../data/switching_nyc/provenance.json) records the official
dataset, aggregate query, and original CSV hashes. No fitted profiles or
quantization enter this study.

The game is `P=L=Delta_3`, `u(p,ell)=(p-ell)/sqrt(2)`, with response `p*(ell)=ell`.
The **full** strict target is `{0}`, including responses at all unobserved hull
mixtures. Every method chooses before the current observation and starts
uniformly. There are no learner restarts within the primary run.

The master uses a game-specific lag safe base: a fresh uniform first action,
then the previous share. Its exact telescoping identity
`sum u=(p_initial-ell_last)/sqrt(2)` gives the conservative certificate
`B0(0)=0`, `B0(h)=1` for `h>0`, hence **G=1** before play. This is a separate
safe base rather than a fitted threshold for the original block routine.

Metrics are distinct:

- `delta_t = ||(1/t) sum_s u_s||_2`: distance of average payoff to the target.
- Mean daily mismatch norm: `(1/T) sum_t ||u_t||_2`.
- Mean L1 share error: `(1/T) sum_t ||p_t-ell_t||_1`, without the `sqrt(2)` divisor.
- Cumulative daily-norm differences: `sum_s (||u_s^master||_2-||u_s^baseline||_2)`.
  Negative values favor the master; these are not differences of `delta_t`.

| Method | Final δ | Mean daily norm | Mean L1 share error |
|---|---:|---:|---:|
| One-switch | 0.000939885298 | 0.155910247468 | 0.346483011179 |
| Fast only | 0.000644365043 | 0.573865013191 | 1.309584310923 |
| Lag safe only | 0.000167400025 | 0.131152905539 | 0.289748910998 |
| Window, W=16 | 0.001311085014 | 0.104084213940 | 0.230298469800 |
| Block safe only | 0.034672804800 | 0.109093125205 | 0.242108870049 |

Crossing occurs at **44**, **2021-02-13**:
`E_43=0.992324057295`, `E_44=1.090736034998`. Round 44 remains fast; the fresh
lag base starts at 45 and lasts **686 rounds**. The master's E stays at its
crossing value. The dashed E curve is a separate fast-only run, ending at
`2.129676313323`. The two methods' actions coincide through the crossing.
Its numerical projection-error charge is about `1.33e-7`, far below both
threshold margins; safe-tail prefix telescope errors are below `5.2e-16`.

Cumulative daily-norm differences end at **−305.106978978** versus fast and
**+37.833004475** versus Window. Master improves daily mismatch over fast,
while fast has the smaller final vector distance. Lag is best on final δ;
Window is best on daily mismatch. Signed errors can cancel in the vector average.

![NYC complete switching dynamics](figures/switching_nyc_dynamics.png)

[Vector PDF](figures/switching_nyc_dynamics.pdf). The dotted line marks crossing;
safe starts next round. Panel (b) clips values below its disclosed `1e-5` display
floor: four lag-safe points are affected, with raw arrays unchanged. Other panels
show E and cumulative daily-norm differences. There are no confidence bands.

Annual checks are separate fresh runs, not block restarts of the primary trace.
Master crosses at **44** in 2021 and **26** in 2022, with
`E_26=1.001078750041` in the latter. All controls are retained:

| Year | Method | Final δ | Mean daily norm |
|---|---|---:|---:|
| 2021 | One-switch | 0.001437149006 | 0.180284970600 |
| 2021 | Fast only | 0.000787605164 | 0.568608734052 |
| 2021 | Lag safe only | 0.000643846251 | 0.130770286743 |
| 2021 | Window, W=16 | 0.001487341298 | 0.106465936245 |
| 2021 | Block safe only | 0.032315630757 | 0.108895989674 |
| 2022 | One-switch | 0.000991389611 | 0.155649632210 |
| 2022 | Fast only | 0.002138496664 | 0.567110706230 |
| 2022 | Lag safe only | 0.000334800051 | 0.130959985497 |
| 2022 | Window, W=16 | 0.000332890973 | 0.101496707311 |
| 2022 | Block safe only | 0.044824485958 | 0.112273125354 |

## Original block-safe budget: constructed diagnostic

Capacity is `p in [0,1]`; the opponent set is `conv{0,e_1,...,e_M}` in its
original Euclidean space. Demand is `r=sum(ell)`, payoff `u=p-r`, and response
`p*=r`. The full target is again `{0}`. The announced horizon is **16384**:
128 zero-demand rounds, then a new orthogonal unit-demand label each round.
The realized affine dimension is **16256**. Novel labels are payoff-equivalent.

The master keeps the original block-safe implementation and
**G=6 T^(3/4)=8688.928127220297**. New orthogonal labels project onto zero;
fast operations use the exact closed-form equations in that geometry. Only
safe play uses the lossless payoff compression `(1-r,r)`.

Here δ is `abs(mean(p-r))`, mean absolute imbalance is `mean(abs(p-r))`, and
mean capacity is `mean(p)`. Cumulative absolute-imbalance differences sum the
per-round absolute errors, master minus baseline.

| Method | Final δ | Mean absolute imbalance | Mean capacity |
|---|---:|---:|---:|
| One-switch | 0.602296147684 | 0.602418217996 | 0.389891352316 |
| Fast only | 0.992126464844 | 0.992248535156 | 0.000061035156 |
| Block safe only | 0.124300406158 | 0.132112906158 | 0.867887093842 |
| Lag safe only | 0.000030517578 | 0.000091552734 | 0.992156982422 |
| Window, W=16 | 0.000488281250 | 0.000549316406 | 0.991699218750 |

Crossing occurs at **8817**, with `E_8816=8688`, `E_8817=8689`. The fresh safe
tail starts at 8818 and lasts **7567 rounds**. Its cumulative payoff
`−1180.020083653083` is within budget `4867.926932557328`. The fast prefix
payoff sum is `−8688`; the final signed sum is `−9868.020083653070`.

Final cumulative absolute-imbalance differences are **−6386.979916347** versus
fast and **+7705.482229165** versus standalone block-safe. Lag and Window are
stronger still. This deliberately artificial test shows benefit over continuing
fast play, not general dominance, real attacker learning, low-dimensional
rates, or a q=4 empirical result.

![Original-budget complete switching diagnostic](figures/switching_original_budget.png)

[Vector PDF](figures/switching_original_budget.pdf). All five distances are shown.
The bottom panels compare cumulative absolute imbalance against fast and
standalone block-safe, without confidence intervals or significance claims.

## Reproduction and verification

[Protocol](protocol.json), [analysis](analysis.json), and [artifact hashes](manifest.json)
retain settings, versions, sources and outputs. Per-round arrays are available
for [full NYC](nyc_2021_2022/one_switch.npz), [2021](nyc_2021/one_switch.npz),
[2022](nyc_2022/one_switch.npz), and [original-budget stress](original_budget_stress/one_switch.npz).
Each directory also contains all four comparator files and a summary. Fast-only
retains complete counterfactual E after master crossing. Safe residual zeros
are placeholders; use the fast-mode mask for monitored residuals.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/run_switching_study.py
```

Only base dependencies are needed; use `--output results/runs/switching_replay`
for a separate replay. The recorded full development suite passed **152 tests**.
Independent array checks confirm first crossing, fast prefix, fresh safe clocks,
stopped E, analytical target distances, metric reconstruction and source hashes.
Floating-point oracle checks are numerical evidence; target and budget identities
are analytical. See [method notes](../../docs/switching_en.md),
[AAMAS integration](../../paper/README.md), and the
[frozen CAGE comparison](../cage_adaptation/README.md).
