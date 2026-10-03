[English](README.md) | [Русский](README.ru.md)

# CAGE 2: initial recorded results

Completed 1,440 official simulator episodes of 50 steps: 40 calibration and 40 held-out seeds for each of 18 policy pairs. The frozen calibrated game was used for nine comparisons: three scenarios, three attacker seeds, 512 rounds, and 13 methods.

**Mean native CAGE loss per simulator step; lower is better.** Values average the three specified attacker paths.

| Method | Fixed Meander | Common curriculum | Interactive attacker |
|---|---:|---:|---:|
| One-switch | 0.7354 | 1.2020 | 1.2018 |
| Shared past-hull | 0.7354 | 1.2020 | 1.2018 |
| Block safe | 2.1666 | 2.7651 | 2.7651 |
| Uniform mixture | 2.2033 | 2.8459 | 2.8459 |
| Initial-mode response | 0.7296 | 1.1245 | 1.1237 |
| Previous 16 rounds | 0.7296 | 1.0994 | 1.0989 |
| Hedge | 0.8604 | 1.3072 | 1.3064 |
| Fixed monitor | 5.4344 | 6.1615 | 6.1615 |
| Fixed reactive removal | 2.4577 | 3.4792 | 3.4792 |
| Fixed reactive restore | 0.7406 | 1.3046 | 1.3046 |
| Fixed decoy cycle | 1.9796 | 3.4195 | 3.4195 |
| Fixed critical restore | 1.8781 | 1.5862 | 1.5959 |
| Fixed decoy + reaction | 0.7296 | 1.1245 | 1.1237 |

Loss components in the common curriculum:

| Method | Security loss | Restoration | Total |
|---|---:|---:|---:|
| One-switch | 0.9525 | 0.2495 | 1.2020 |
| Block safe | 2.6349 | 0.1302 | 2.7651 |
| Hedge | 1.1231 | 0.1841 | 1.3072 |
| Previous 16 rounds | 0.9009 | 0.1984 | 1.0994 |

## Interpretation

The method improves substantially over the conservative safe routine on this policy menu. Its mean loss is also below Hedge, but the conditional 95% paired difference interval includes zero in the changing scenarios. Security losses are lower than Hedge in the changing cases, while restoration costs are higher. The previous-16-round response and a good fixed policy have lower mean total loss than our method. This experiment does not establish general superiority.

There were no switches: one_switch actions match shared_past_hull to floating-point precision. With three pure modes, novelty occurs only on the first appearance of the two remaining modes. The residual sum is at most 4; the K=6, T=512 threshold is about 1,444. This tests fast adaptation without validating an advantage of the switching mechanism.

Held-out losses use the independent episode bank after player decisions are fixed. The calibration game and held-out table remain distinct. Confidence intervals use 40 seed clusters across the whole payoff table; the three attacker paths do not create 120 independent simulator samples. Intervals condition on calibration and these paths, exclude calibration uncertainty, and are not adjusted for multiple comparisons.

Interactive methods face different paths under the same attacker rule. Distances to their own realized targets and regret on their own paths must be interpreted accordingly. The common curriculum provides direct comparisons on the same opponent path.

This is selection among frozen policies with a known estimated model and attack-mode disclosure after choosing. Within episodes, defense policies use partial observations. New attack tactics are not trained, and this experiment does not reproduce historical attacks on a deployed network.

## Files and reproduction

[Methodology and commands](../../docs/cage_en.md).

- [All nine comparisons](all_run_summaries.json)
- [Aggregates, loss components, and paired intervals](aggregate.json)
- [Episode-bank provenance](../../data/cage2/provenance.json)

![Policy payoff tables](cage_calibration.png)

![Common-curriculum comparison](cage2_curriculum_seed20261003_T512__loss.png)
