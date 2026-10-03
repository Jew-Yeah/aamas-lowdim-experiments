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
