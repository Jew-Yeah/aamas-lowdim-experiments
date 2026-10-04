[English](two_phase_validation_en.md) | [Русский](two_phase_validation_ru.md)

# Independent two-phase validation

The [fixed design](two_phase_validation_protocol.json) was written before the
new development and validation batches. This is a local auditable commitment,
not an external preregistration. Its purpose is to distinguish an early
low-dimensional benefit from later protection against continuing fast play.
The design was motivated by the earlier exploratory switching audit.

## Resource-allocation model and exact target

Capacity is `p in [0,1]`. A task context specifies efficiency `a` and required
work `b`, with `0.5 <= a <= 1` and `0 <= b <= a`. The signed normalized
imbalance is `u=a*p-b`. The prescribed benchmark allocates `p*=b/a`.
The opponent set is `conv{0,e_1,...,e_m}` in its original Euclidean space;
the origin context has `(a,b)=(1,0)`, and the coefficients extend affinely to
mixtures. Efficiency remains positive at every mixture. Therefore the response
payoff is exactly zero at every point of the entire realized hull, and the
strict target is exactly `{0}`. Vertex payoff absolute values are at most one.

The first context is origin. A single other context then repeats in the
first phase, with efficiency one and a capacity ratio independently drawn
per episode from `[0.02,0.15]`. The changed phase presents fresh orthogonal
contexts with independently sampled efficiencies and capacity requirements.
New contexts have different actual payoff maps, rather than merely different
names with identical payoffs. Context novelty is generated externally;
it does not model an attacker learning a strategy.

The high-dimensional context representation is nevertheless an explicit
limitation. Payoffs depend on only two affine coefficients. Fresh orthogonal
contexts project to the previously observed origin in the specified geometry,
even when their required work is similar. Thus the experiment also measures
the fast branch's sensitivity to representation. It is a constructed mechanism
study, rather than representative evidence about NYC or real adversaries.

## Original algorithms and controls

The master retains `G=6*T^(3/4)=24576` at `T=65536`, the original scalar
`k=1` block-safe procedure, and its fresh tail with horizon `T-tau`. The
crossing action remains fast, with safe play starting on the next round.
No arbitrary reduced threshold or repeated learner restart is used.

The scalar fast implementation solves the original hull projection and
saddle problems analytically. An unseen unit vertex projects to origin
once origin is observed. The minimizing capacity is zero for positive
direction, one for negative direction, and one half at a tie. Independent
dense projection and saddle solvers verify the specialization on small paths.

Safe updates use four `(a,b)` corners only as a lossless computational
representation of payoffs, gradients and block responses. This compression
is never used for fast geometry. Its scalar recurrence was independently
compared with `SafeBlockLearner` on arbitrary weighted paths.

Every episode compares one-switch, fast-only, block-safe-only, previous
observed response, and a trailing mean of the previous 16 response ratios.
All choose before observing the current context and start at capacity one
half. With variable efficiency the lag heuristic has no asserted telescoping
safety guarantee. No method knows the realized phase-change observations
before they occur.

## Fixed sampling and inference

There are four fixed scenarios: primary change at round 16384, later change
at 32768, weaker change at 16384, and a stationary control. Each uses 200
independent validation episodes, seeds 10000--10199, with separate scenario
RNG namespaces. Eight development seeds per scenario check implementation
and runtime; they do not enter inference. Settings are not selected from
development outcomes. The complete independent episode is the statistical
unit. Methods are paired on the same exogenous episode path.

The primary family has three comparisons in the primary scenario:

1. One-switch minus block-safe-only in mean prefix distance before change.
2. One-switch minus fast-only in terminal full-target distance.
3. One-switch minus block-safe-only in terminal full-target distance.

Distance is `delta_t=abs(sum_{s<=t}u_s)/t`. Mean prefix distance is computed
on every round before the fixed environmental boundary. Mean daily absolute
imbalance is additional and reported separately; it is never substituted for
theoretical target distance. No-switch episodes remain in every analysis.

Paired mean differences use two-sided t tests and Holm adjustment across the
three primary comparisons. Reports include 95% t intervals, conservative
family intervals, and whole-episode paired bootstrap checks. Sensitivity
scenarios are descriptive and all are retained. Curve bands resample complete
episodes, are pointwise intervals for the mean, and are neither simultaneous
bands nor individual-trajectory ranges. A stationary curve's prechange metric
covers its entire horizon because it has no changed phase.

At `n=200`, a normal approximation with conservative `alpha=0.05/3` gives
about 80% power for a standardized paired mean effect around 0.23. This does
not guarantee detection of a practically important absolute effect or make
the synthetic generator representative of a real-world population.

Methods follow [NIST's paired-observation analysis](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)
and the paired episode-resampling principle documented by
[SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

## Reproduction

Use Python and the dependencies in `pyproject.toml`. The runner refuses to
overwrite completed episode batches or reuse a changed frozen implementation.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/run_two_phase_validation.py --stage pilot
python scripts/run_two_phase_validation.py --stage freeze
python scripts/run_two_phase_validation.py --stage validation
python scripts/run_two_phase_validation.py --stage analyze
python scripts/run_two_phase_validation.py --stage figures
```

Pass `--output results/runs/my_two_phase_replay` to every runner invocation
for a separate replay. The repository publishes the frozen design/source
receipt, all episode summaries and final figures. Sampled per-episode dynamics
and complete raw traces for the two seeds chosen before validation are retained
locally and regenerated by the same runner; these larger arrays are excluded
from Git. All reported summary metrics are computed at full resolution;
plotting downsampling does not enter inference.

The [result report](../results/two_phase_validation/README.md) states the actual
findings, including comparisons that favor another method. This study does
not estimate a convergence exponent or add the declined `q=4` experiment.
