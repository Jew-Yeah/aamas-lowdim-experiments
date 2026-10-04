[English](README.md) | [Русский](README.ru.md) | [Home](../../README.md)

# Independent two-phase switching validation

**The mechanism succeeds against leaving the fast policy uninterrupted, but it does not beat standalone block safe over the full horizon.** In the primary scenario, one-switch has a smaller prechange mean target distance than block safe and a 24.05% smaller terminal target distance than fast only. Its terminal distance is substantially larger than block safe, lag response, and window. This is a mixed result, and all controls and fixed scenarios are retained.

## What was fixed before validation

The [protocol](../../docs/two_phase_validation_protocol.json) and [precommitment receipt](precommitment.json) were frozen before the 800 validation episodes. The receipt time is `2026-10-04T05:56:14.532111+00:00` and the protocol SHA-256 is `9fafd7cce7c9250fbc3981bba40f792791937ab9903b2461dcbcb1f5c3c1930b`. Development used seeds 0–7; validation used all seeds 10000–10199, 200 per scenario. No validation episode was omitted, and there was no optional stopping or selection of successful cells. This is an auditable local precommitment following exploratory design, not an external preregistration.

Every episode has T=65536 decisions and all five methods face exactly the same exogenous path. The first decision is a zero-demand context. Through the fixed boundary H, a single known type repeats, with its required capacity ratio sampled once per episode from Uniform[0.02,0.15]. After H, each round has a new orthogonal task type, with efficiency a sampled from Uniform[0.5,1] and a sampled capacity ratio r. The primary/late ranges are r in [0.85,1]; the weaker range is [0.5,0.85]. The stationary control has H=T and no changed phase.

The constructed allocation game has p in [0,1], payoff `u=a*p-b`, and `b=a*r`. At the geometric origin, (a,b)=(1,0). The response `p_star=b/a` gives zero payoff on **the entire strict target**, including unobserved interior mixtures: `S(Q)={0}`. The original block-safe base and certified budget `G=6*T^(3/4)=24576` are preserved. Fast projection keeps the original geometry `conv{0,e_1,...,e_M}`; only safe payoff calculations are compressed losslessly. New types have different payoff maps, unlike the older payoff-equivalent stress labels. The allocation interpretation is synthetic: no real demand data or measured deployment outcomes are used here.

All actions precede the current observation. The crossing action remains fast; a fresh safe run starts on the next decision. There are no periodic restarts. Lag response uses the previous observed ratio; window uses the mean of the previous at most 16 ratios. Neither comparator knows H or future contexts. There is no claimed telescoping certificate for the heterogeneous-efficiency lag rule.

## Metrics and all controls

`delta_t = abs(sum_{s<=t} u_s)/t` is the exact distance to the full strict target. The prechange metric is the mean of this full-prefix distance on every round through H; it is not mean instantaneous error. The terminal metric is delta_T. Mean per-round imbalance is `mean(abs(u_t))`, an additional metric where cancellation cannot hide errors. Lower is better for each metric.

Each table entry is **mean ± standard deviation across 200 independent complete episodes**. The standard deviation describes episode variation, not uncertainty in the mean. For the stationary control, the through-H column spans its entire horizon. Every episode is included in the published summaries. The fixed n=200 was chosen for approximately 80% power at a paired standardized effect of 0.23 under a normal approximation and conservative alpha=0.05/3; it is not a guarantee of power for an important absolute effect.

Small prefix distance can coexist with appreciable per-round imbalance because opposite signs cancel. The stationary result makes this visible: fast-only has tiny terminal distance but much larger mean per-round imbalance than lag/window. Early target-distance superiority therefore does not imply better demand tracking on each decision.

### Primary change: H=16384, n=200

| Method | Mean prefix distance through H | Terminal distance | Mean per-round imbalance |
|---|---:|---:|---:|
| One-switch | 0.000366017 ± 0.000064130 | 0.395181231 ± 0.000136154 | 0.440722146 ± 0.015114971 |
| Fast only | 0.000366017 ± 0.000064130 | 0.520319017 ± 0.000490806 | 0.557829497 ± 0.015112424 |
| Block safe only | 0.219445874 ± 0.022735213 | 0.049686434 ± 0.000204451 | 0.157557023 ± 0.004625057 |
| Lag response | 0.000266542 ± 0.000020583 | 0.000026063 ± 0.000019183 | 0.028137824 ± 0.000100556 |
| Window, W=16 | 0.000170085 ± 0.000062557 | 0.000076849 ± 0.000022431 | 0.021616600 ± 0.000062283 |

### Late change: H=32768, n=200

| Method | Mean prefix distance through H | Terminal distance | Mean per-round imbalance |
|---|---:|---:|---:|
| One-switch | 0.000210792 ± 0.000038084 | 0.346833046 ± 0.000370610 | 0.421702246 ± 0.030977798 |
| Fast only | 0.000210792 ± 0.000038084 | 0.346833046 ± 0.000370610 | 0.421702246 ± 0.030977798 |
| Block safe only | 0.159085972 ± 0.020917923 | 0.041901572 ± 0.000560947 | 0.171184443 ± 0.003626250 |
| Lag response | 0.000142129 ± 0.000011310 | 0.000019660 ± 0.000016841 | 0.018767338 ± 0.000083106 |
| Window, W=16 | 0.000089790 ± 0.000034657 | 0.000078131 ± 0.000018929 | 0.014439647 ± 0.000052863 |

### Weaker change: H=16384, n=200

| Method | Mean prefix distance through H | Terminal distance | Mean per-round imbalance |
|---|---:|---:|---:|
| One-switch | 0.000366464 ± 0.000062987 | 0.375868725 ± 0.000064166 | 0.413857235 ± 0.016302742 |
| Fast only | 0.000366464 ± 0.000062987 | 0.379759840 ± 0.000425651 | 0.417643964 ± 0.016315901 |
| Block safe only | 0.218819354 ± 0.024556288 | 0.014584398 ± 0.000077953 | 0.129917210 ± 0.004659328 |
| Lag response | 0.000265883 ± 0.000022266 | 0.000054311 ± 0.000038409 | 0.065681540 ± 0.000247870 |
| Window, W=16 | 0.000168073 ± 0.000067696 | 0.000063329 ± 0.000042826 | 0.050313198 ± 0.000142034 |

### Stationary control: H=65536, n=200

| Method | Mean prefix distance through H | Terminal distance | Mean per-round imbalance |
|---|---:|---:|---:|
| One-switch | 0.000117474 ± 0.000018433 | 0.000021786 ± 0.000005660 | 0.157998926 ± 0.061124822 |
| Fast only | 0.000117474 ± 0.000018433 | 0.000021786 ± 0.000005660 | 0.157998926 ± 0.061124822 |
| Block safe only | 0.107067686 ± 0.018085568 | 0.044780687 ± 0.014168784 | 0.135057837 ± 0.014994841 |
| Lag response | 0.000074672 ± 0.000005992 | 0.000006285 ± 0.000000562 | 0.000008974 ± 0.000000562 |
| Window, W=16 | 0.000044727 ± 0.000018500 | 0.000003084 ± 0.000001899 | 0.000012175 ± 0.000001899 |


## Three primary paired comparisons

These comparisons were fixed for the primary scenario only. Every difference is one-switch minus the named comparator, so negative favors one-switch. The before-change comparison uses the prechange mean of delta_t; the two terminal comparisons use delta_T. The intervals use episode-level paired differences, not independent rounds. Bonferroni marginal intervals at 98.3333% provide a conservative 95% family coverage for the three means; Holm adjusts the two-sided p-values for the same family. Whole-episode percentile bootstrap uses 9999 draws and its fixed protocol RNG.

| Comparison | Paired mean difference | 95% t interval | 98.3333% family interval | Wins / ties / losses |
|---|---:|---|---|---:|
| Before change: one-switch − block safe | -0.219079857 | [-0.222242347, -0.215917367] | [-0.222951889, -0.215207825] | 200 / 0 / 0 |
| Terminal: one-switch − fast only | -0.125137787 | [-0.125200613, -0.125074961] | [-0.125214709, -0.125060865] | 200 / 0 / 0 |
| Terminal: one-switch − block safe | 0.345494797 | [0.345461709, 0.345527884] | [0.345454286, 0.345535308] | 0 / 0 / 200 |

| Comparison | 95% bootstrap mean interval | Holm-adjusted p | −log10 raw p | Primary-scenario conclusion |
|---|---|---:|---:|---|
| Before change: one-switch − block safe | [-0.222145598, -0.215952728] | 1.183e-198 | 197.93 | Benefit over block safe before change |
| Terminal: one-switch − fast only | [-0.125200983, -0.125074951] | < 1e−300 | 487.75 | Benefit over fast only at the horizon |
| Terminal: one-switch − block safe | [0.345461619, 0.345527286] | < 1e−300 | 630.93 | Loss to block safe at the horizon |

The latter two raw p-values underflow ordinary floating-point representation; they are **not mathematical zeros**. [analysis.json](analysis.json) retains finite log p-values, `negative_log10_p_value`, and the explicit `p_value_underflow` flag. The `<1e−300` presentation remains true after Holm correction. Standardized paired effects d_z and standard errors are also retained in the JSON; the very large d_z values reflect narrow variance under this fixed synthetic generator.

The primary mechanism claim is supported: fast has an early advantage over block safe, and switching later improves the terminal outcome relative to continuing fast. The distinct claim of full-horizon superiority over safe-only fails in the opposite direction: **one-switch loses to block safe in all 200 primary episodes**. The lag and window controls also have smaller primary prechange and terminal target distances. There is no general superiority claim.

## Switching and fixed sensitivity checks

| Scenario | Switches | 95% switch-probability interval | τ quantiles: 5% / 50% / 95% | Mean safe-tail length |
|---|---:|---|---|---:|
| Primary change | 200 / 200 | [0.981725, 1.000000] | 51749.95 / 51808.00 / 51872.15 | 13729.58 |
| Late change | 0 / 200 | [0.000000, 0.018275] | — | 0.00 |
| Weaker change | 200 / 200 | [0.981725, 1.000000] | 64840.85 / 64923.50 / 65007.15 | 613.16 |
| Stationary control | 0 / 200 | [0.000000, 0.018275] | — | 0.00 |

Intervals in the switching table are exact Clopper–Pearson intervals across independent episode indicators. τ quantiles describe only actual switch times, while all episodes remain in metric estimates and the plotted switching fraction uses all 200 episodes as its denominator. The primary median delay from environmental change to crossing is 35424 rounds: the conservative budget allows substantial fast-prefix error before the safe tail begins. In the weaker scenario, switching occurs near the horizon and the improvement over fast-only is much smaller. The late scenario has no switch in this batch, and therefore one-switch equals fast-only. Sensitivity scenarios are descriptive, not extra confirmatory significance tests.

Both the late and stationary one-switch-minus-fast differences are identically zero. Their t statistics, p-values, and d_z are undefined and recorded as `null`; no invented significant result is reported. Their empirical intervals collapse to zero and are explicitly marked degenerate. All computed safe-tail certificates hold, including the no-tail cases.

## Figures and reproduction

[Full switching dynamics (PDF)](figures/two_phase_dynamics.pdf) shows budget crossing, full-target distance for every method, and paired distance differences against fast-only and block-safe-only. [Fixed sensitivity checks (PDF)](figures/two_phase_sensitivity.pdf) shows paired endpoint intervals in every scenario and the switching fraction by round. Curve intervals resample whole paired episodes with common weights across time, using 2000 draws; they are **pointwise 95% mean intervals**, not simultaneous bands or individual-trajectory ranges. The dotted line marks environmental change, not the data-dependent switch.

[Detailed method notes](../../docs/two_phase_validation_en.md), [analysis](analysis.json), and the four full episode tables ([primary](primary/episodes.json), [late](late_change/episodes.json), [weaker](weaker_change/episodes.json), [stationary](stationary_control/episodes.json)) retain exact numerical precision and every control. The public repository includes code, the frozen protocol and receipt, all 800 episode summaries, analysis, and figures. Larger per-episode grid arrays and full representative paths are retained locally and regenerated by the runner; they are not included in the compact GitHub upload. The full-path seeds 10000 and 10001 were fixed before validation, rather than selected for favorable outcomes.

```powershell
python -m pytest -q
python scripts/run_two_phase_validation.py --stage pilot --output results/two_phase_replay
python scripts/run_two_phase_validation.py --stage freeze --output results/two_phase_replay
python scripts/run_two_phase_validation.py --stage validation --output results/two_phase_replay
python scripts/run_two_phase_validation.py --stage analyze --output results/two_phase_replay
python scripts/run_two_phase_validation.py --stage figures --output results/two_phase_replay
```

The runner refuses to overwrite completed episode files and rejects changed source/protocol hashes after freezing. Use a new output folder to reproduce the study. Statistical inference describes the stated stochastic mechanism model only; it does not establish representativeness for NYC, real attackers, arbitrary games, or low-dimensional convergence exponents. The representation dependence and late conservative switch remain material limitations.
