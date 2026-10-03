[English](cage_study_en.md) | [Русский](cage_study_ru.md)

# Extended CAGE study

This study checks native simulator losses more precisely than the initial
[pilot](../results/cage_reference/README.md). CAGE is a benchmark environment.
The comparison methods are its unchanged BlueMonitor, BlueReactRemove and
BlueReactRestore agents, three frozen custom policies, Hedge, a response to
the previous 16 attacks, and the manuscript learners.

The [local protocol](cage_study_protocol.json) fixes the seed ranges, endpoint,
comparisons and analysis before the new banks are evaluated. It is a recorded
analysis plan, without an external preregistration service. The original pilot
guided the design and remains available. No significance-based stopping is used.

## Primary comparison

The original 40 training seed tables and 50-step model are frozen. We collect
400 new held-out seed tables, each containing all six defense policies against
all three attack modes. Seeds 32000000–32000399 are disjoint from the original
training and test seeds.

The common curriculum has 512 meta-rounds and 50 new attacker seeds
(36000000–36000049). Its first quarter uses Meander; the middle half exposes
and samples all modes; the last quarter uses exponential weights trained
against a fixed uniform reference. All methods face the same path in this
primary comparison. The primary outcome is the sum of four native positive
loss components, averaged per simulator step.

Six comparisons were declared: our method against monitor, reactive removal,
reactive restore, the frozen decoy-and-reaction policy, Hedge and the window
response. A negative difference means our method has lower loss.

## Statistical units and inference

Each simulator seed produces a complete policy payoff table. Reusing its seed
across policy cells preserves paired comparisons; policy-dependent random-number
consumption still produces different episode trajectories. Each attacker seed
produces a complete meta-game path. The 400 × 50 evaluations have crossed
dependence and are not 20,000 independent simulator observations.

We use 10,000 paired bootstrap replicates, independently resampling whole
simulator seed tables and whole attacker paths. Identical draws are used for
all six contrasts. This follows the row-and-column resampling approach of the
[pigeonhole bootstrap](https://arxiv.org/abs/0712.1111).

Percentile intervals use linear quantiles and a Bonferroni level of 0.05/6 per
contrast. A primary advantage is supported when the entire simultaneous
interval is below zero. Centered approximate two-sided bootstrap p-values
are also reported with Holm adjustment across the six comparisons. The two
procedures can differ. Bootstrap inference is approximate, with no exact
finite-sample coverage claim.

These intervals condition on the frozen training fit. They describe variation
over the new simulator-seed and attacker-path distributions. They exclude
calibration uncertainty. Security damage and restoration costs are secondary
outcomes and cannot replace the declared total-loss endpoint.

## Robustness checks

- Constant Meander and B_line; an abrupt Meander-to-B_line shift; alternating
  64-round blocks; and an interactive attacker that learns from past actions.
- Meta-horizons of 128 and 2048 rounds.
- Three new independent 40-seed training fits, using seeds 33000000–33000119,
  evaluated on the same primary held-out bank. Three fits provide a descriptive
  sensitivity check, not an interval covering all training uncertainty.
- Episodes of 100 and 150 steps, each with its own new 40-seed calibration and
  100-seed test bank. Their seeds start at 34000000 and 35000000, respectively.

Deterministic paths may be repeated by configuration; repeated identical
trajectories do not add independent information. Interactive methods can
induce different paths under the same causal attacker rule. All robustness
comparisons are exploratory and use pointwise intervals.

## Reproduction

Use the [pinned simulator and environment](cage_en.md). Run from the repository
root. Collection is parallel and resumes checked seed checkpoints.

```bash
python scripts/collect_cage_bank.py --steps 50 --seed 32000000 --count 400 --workers 8 --exclude-calibration data/cage2 --output results/runs/cage_study/banks/primary50
python scripts/collect_cage_bank.py --steps 50 --seed 33000000 --count 120 --partition calibration --workers 8 --exclude-calibration data/cage2 --output results/runs/cage_study/banks/calibration50
python scripts/collect_cage_bank.py --steps 100 --seed 34000000 --count 140 --partition external --workers 8 --exclude-calibration data/cage2 --output results/runs/cage_study/banks/step100
python scripts/collect_cage_bank.py --steps 150 --seed 35000000 --count 140 --partition external --workers 8 --exclude-calibration data/cage2 --output results/runs/cage_study/banks/step150
python scripts/run_cage_study.py --stage all --workers 8
python scripts/build_cage_study_report.py
python scripts/plot_cage_study.py
```

The new collection contains 14,400 simulator episodes and 1,098,000 internal
steps. The [report](../results/cage_study/README.md) records completed comparisons,
means, intervals, provenance and group occupancy arrays.

The policies, response weights, normalization and switch budget remain fixed.
With three pure attack modes, novelty is bounded and no switch is expected.
This tests adaptation among frozen policies with an estimated known model and
attack-mode disclosure after choosing. It does not train new attack tactics
or reproduce logs from a deployed network. Target-set geometry is omitted
from the extended scalar analysis; the pilot reports it separately.
