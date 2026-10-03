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
