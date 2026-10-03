# Implementation log

2026-10-03: Start an independent experiment repository. The original article remains unchanged.
Scope: finite allocation games, a full-hull target oracle, the paper's one-switch learner,
its safe routine, the shared past-hull policy, and interpretable allocation heuristics.
Real-data example: aggregate NYC Forestry demand with a training-only finite profile library.
The queueing and SLA model of Liu and Garg is distinct from this initial daily allocation experiment.
Documentation languages: English and Russian. Publication requested: public GitHub repository.

Completed 39 correctness tests, 15 synthetic comparisons (three seeds, q=1,...,5),
nine NYC comparisons (three profile counts and capacity ratios, 730 test days),
and 15 layered-paraboloid checks. Each comparison contains seven policies.
Independent review verified causality and raw service metrics. Projection regressions
led to hull-membership/active-face polishing and scaled LP support objectives;
the requested numerical tolerance remained 1e-9.

The initial NYC model does not establish empirical superiority over simple
allocation references. The one-switch and shared past-hull actions coincide
under the theorem's conservative threshold on the tested horizons.
Aggregate data, input provenance, complete representative traces, and bilingual
reference reports are included. The original main.tex SHA256 remains
48d067d64a16c76d674f2f8977fe0f425249d093c3c699249434a7f4194e2a74.

2026-10-03: Added the official CAGE Challenge 2 simulator at pinned revision
26ce1c1253fa9e2e73f25e6a7f2da32860c11257 without modifying upstream source.
Six frozen defensive policies and three official attacking policies produced
1,440 fresh 50-step episodes: 40 calibration and 40 held-out seeds per cell.
Native four-component losses reconstruct the scalar reward within 5.55e-17.
Nine 512-round policy-game comparisons include 13 methods, three scenarios,
and three attacker seeds; held-out data do not affect either player's actions.
The simulator bank is reused as 40 paired seed clusters for uncertainty.

In the common curriculum, mean native loss per step is 1.2020 for one-switch,
2.7651 for block safe, 1.3072 for Hedge, and 1.0994 for a 16-round response.
Changing-scenario total-loss differences against Hedge remain uncertain;
security losses are lower but restoration costs higher. Window response wins.
No switches occur and the master matches the shared fast policy to floating-point
precision. The experiment is exploratory and does not establish general superiority.

The tiny eight-step CI case exposed a lower-dimensional response-cell regression.
Eliminating proved affine equalities before row normalization fixed the oracle
without relaxing tolerances. All 55 tests pass. Rechecking 2,574 projections
from the nine main runs changes distances by at most 5.55e-17, so the saved
experiment outputs remain valid. Maximum checked support gap is 7.63e-11.

## Extended CAGE statistical study

The local analysis plan was fixed before evaluation of the new banks
(`docs/cage_study_protocol.json`, SHA256
`9a3ecc45704e42976c3fd3a2333b0b0ec13e1310fb4b2b5e57b65be0ad08357a`).
Collected 14,400 additional official simulator episodes, totalling 1,098,000
internal steps. Primary evaluation retains the original 40-seed calibration
and uses 400 new held-out seed tables and 50 common attack paths. Six primary
contrasts use shared crossed-bootstrap draws, simultaneous Bonferroni
intervals, and additional approximate Holm-adjusted p-values.

Primary native loss is 1.19373 for our method, 1.25799 for Hedge, 1.23157
for official reactive restore, 1.08057 for the window response, and 1.10039
for frozen decoys with reaction. The difference to Hedge is -0.06425, with
simultaneous interval [-0.10811, -0.02170]. The restore comparison remains
uncertain; both window response and fixed decoy-reaction outperform our method.
All primary inferences condition on the frozen fit. At meta-horizon 2048 the
descriptive Hedge contrast includes zero. Three independent calibrations,
episode lengths 100/150, and additional attack schedules are exploratory.

All 13 groups completed. Their 220 path slots correspond to 182 saved path
checkpoints and 2,366 method trajectories; repeated deterministic paths add
no independent observations. Independent replay of every occupancy gave zero
error, and vector-payoff replay error was at most 1.11e-16. No switches occurred;
master and shared fast actions agree within 2.22e-16. Serial replay of two
actual simulator cells exactly matched the parallel bank. Original article
and upstream tracked source hashes remain unchanged. All 66 local tests pass.
The bilingual report and figures are saved under `results/cage_study`.

## Scalar-aware oracle and restart study

Saved a separate protocol before validation analysis (`docs/cage_adaptation_protocol.json`,
SHA256 `a1c8976415eb867bb179dfada83c4a4e9fa0b8667f2c1883aa4044f199fb7de3`).
The original article, calibration, learners and previous reports remain unchanged.
Collected 9,000 new 50-step simulator episodes (450,000 internal steps): 100
validation seeds and 400 final-test seeds, each covering all 18 policy pairs.
Validation used 20 new 512-round paths and 30 configurations across the scalar
oracle, window and Hedge families. The selection was locked before final-test
outcome access (SHA256 `3e8f87ea1babcf7b2328fb6eed9aecb5314ffdc996be19f34667c31861ee4cc4`).
Selected parameters: oracle window 16, rho 0.25; window baseline 16; Hedge eta
multiplier 4. Fifty new final paths gave native losses 1.10418, 1.10154 and
1.16600, respectively; original one-switch gave 1.20665. Two simultaneous
approximate primary contrasts favor the selected oracle over tuned Hedge and
favor tuned window over the oracle. The oracle and window actions are exactly
equal after round one on all 50 paths; the oracle's initial uniform action
accounts for their entire difference. No initial-action retuning was performed.

Two secondary 3,000-round scenarios compare fresh and retained-history
1,000-round restarts. Both worsen losses on the curriculum. On deterministic
500-round attack alternation, original fresh restart improves the mean from
1.33833 to 1.29304; retained history gives 1.30769. These are exploratory
comparisons, not additional primary claims. Actual safe switches are zero;
each restarted path has two state restarts. The reset of the residual counter
alone therefore cannot explain improvement.

The alternating scenario has one deterministic path represented by 20 declared
slots. After curriculum completion, the queued alternating stage was stopped
before any alternating checkpoint completed; a checked helper computed one
source trajectory and memoized the other 19 slots. The frozen runner and
selection were unchanged. Published cache metadata discloses this reuse, which
adds no independent path variation and changes no simulator episode counts.
In total, 110 nominal path slots represent 91 computed path checkpoints and
1,068 method trajectories retained in those checkpoints (1,220 nominal
trajectory slots).

The forecast LP preserves the saddle dual, full hull and update and independently
checks its allowed gap with fallback. Separate notes derive segment bounds for
both restart histories and explain why fixed block length does not retain the
original global-horizon convergence rate. All 89 correctness tests passed.
The report under `results/cage_adaptation` includes bilingual findings, selection
and input hashes, group occupancies, three scientific figures and their PDF
exports. Plot layout and contrast were visually checked.
