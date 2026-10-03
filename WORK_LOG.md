# Work log

## Recorded full study

The complete exploratory history, earlier NYC/synthetic/CAGE results, restart
studies and their documentation are preserved at commit
`e8c85cb2e7be49032c1c2d014b6f19084ea064a8` and the branch
[archive/full-study-2026-10-04](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/archive/full-study-2026-10-04).
The archived work log contains the complete original chronology. A local
snapshot is saved at `C:/Math/aamas-full-study-2026-10-04.zip`.

## 2026-10-04: focused implementation and presentation

At the user's request, current main now focuses on the selected continuous
scalar-aware one-switch implementation: W=16, rho=0.25, uniform first action,
original safe threshold and no restarts. The frozen learner and study-runner
bytes are unchanged. RecommendedOneSwitchLearner and the cage-selected CLI
expose the already selected implementation without another tuning stage.

Earlier result directories, their auxiliary inputs and obsolete report scripts
were removed from current main after archiving. The primary validation/selection
and complete latest test aggregates remain. The published primary analysis and
figures compare only the three selected methods, retaining both predeclared
contrasts and all 50 test paths. The six-method source aggregate is kept as
unchanged provenance; ancillary original and restart results remain accessible
in the archived snapshot. Filtering the report does not alter any observations,
parameters, confidence intervals or conclusions.

Primary mean native losses: ours 1.1041833342447924, window
1.1015373414062506, tuned Hedge 1.1660012488598512. Our losses are about
5.3% lower than tuned Hedge and 0.24% higher than the window. All 50 paths
have exactly identical selected-ours/window actions after round one. The
figures display the unfavourable window contrast as well as the Hedge benefit;
no superiority, equivalence or noninferiority to window is claimed.

Validation remains independent: 100 simulator seeds and 20 paths; final
assessment uses 400 new simulator seeds and 50 paths, horizon 512. The locked
selection preceded final outcome analysis. Earlier test results were not used
for selection. The published episode banks represent 9000 newly collected
50-step simulator episodes. This is a calibrated, known-model policy game with
revealed opponent modes and reset episodes; it does not evaluate live networks.
The scalar metric does not establish a new vector-target theorem.

Verification: exact recommended/explicit learner parity (actions, residuals,
gaps and fallback), uniform first action with a nonuniform prior, CLI routing,
and an actual 16-round cached-bank smoke run passed. A 512-round replay of the
first locked test path exactly matched all three original methods' actions,
opponent path and occupancies; native-sum scores matched the original aggregate.
The smoke scores are not
substituted for the locked 512-round primary results. All 92 tests passed;
22 report artifact checksums and 82 local Markdown links were verified.
All frozen source hashes, selection bytes and the article checksum are unchanged.
Remote CI runs on publication.
The original article is unchanged.

## 2026-10-04: descriptive figures and calibrated geometry

Added eight supplementary figure sets in English and Russian, each with PNG
and vector PDF exports: native loss dynamics, cumulative paired differences,
four cost components, all 50 paired-path distributions, full validation
sensitivity, moving full-target vector distance, retrospective fixed-target
distance, and separate terminal-horizon experiments. The bilingual gallery
links figures, source arrays, methodological notes and reproduction commands.
The article and selected algorithms/settings are unchanged.

Native plots reconstruct every primary path against all 400 simulator tables.
Their 2000-draw crossed bootstrap (seed 46000000) resamples whole tables and
whole paths with shared weights across methods/components. Bands are descriptive
pointwise 95%, conditional on fixed calibration and selection; they do not
replace the original two simultaneous primary comparisons. Cumulative costs
include the 50-step episode multiplier. All 50 paths favor ours over Hedge and
favor window over ours. The cumulative window gap plateaus at 67.7374 after
the first action; the Hedge difference ends at -1582.5386. Components show
less disruption and server compromise than Hedge, alongside greater host
compromise and restoration cost. No new parameter selection took place.

The supplementary protocol was frozen before new runs. Full-target projections
use the normalized training game, including response cells for interior hull
mixtures, at tolerance 1e-11. All 50 primary paths and 39 checkpoints retain
5850 moving-target and 5850 fixed-target projections. The separate sweep uses
20 new common path seeds 45000000–45000019 at each of six announced horizons
64–2048: 120 paths, 360 method trajectories, no new simulator episodes. Original
curriculum and learning-rate formulas scale with each announced horizon.
Every numerical lower/upper distance, gap, alpha, feasibility, success and hull
dimension is published. All projections passed their numerical checks.

At T=64 the selected method's median upper distance is 0.0001871 (5/20 paths
at the display floor). At every T>=128 all selected-method and window paths
have distances at numerical precision; a decay slope for them is unidentified.
Hedge medians decline from 0.01535 to 0.002112; any fitted slope describes this
horizon-scaled family only. Log plots do not substitute epsilon for zeros.
The floor panel retains every censored observation, and captions explain
omitted ranges when their lower quantile reaches the display floor. Target
expansion and dilution of the initial-action difference are explicitly explained.

Verification: all 108 tests passed, including full-target interior mixtures,
failed projection rejection, causal replay, unavailable-checkpoint public
replay, whole-unit bootstrap and native component/cumulative reconstruction.
The dynamic public rebuild reproduces all 25 artifacts byte for byte.
The geometry source ledger retains its exact pre-sweep Windows-line-ending
fingerprint; a documented serialization/presentation repair preserves frozen
conditions and numerical outputs. The primary builder accepts only this
specific historical JSON exception, with a matching Git attribute preserving
its exact bytes on checkout. Complete artifact checksums, source hashes,
local links, report reconstruction and remote CI are checked on publication.
