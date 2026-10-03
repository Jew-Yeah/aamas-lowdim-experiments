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
