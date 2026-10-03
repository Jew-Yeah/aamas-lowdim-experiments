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
