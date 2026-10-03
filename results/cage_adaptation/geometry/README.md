[Русский](README.ru.md) · [Main report](../README.md)

Full-target vector error of the selected implementation
======================================================

The error is delta_t = dist(mean_(s<=t) u(p_s, ell_s), S(Q_t)) in the frozen normalized TRAINING game. It is distinct from held-out native simulator cost. The target includes the full realized opponent hull, its mixtures and all actual response cells.

![Prefix error](prefix_error.png)

All 50 locked primary paths and all three selected methods are retained. The target expands upon first appearances of modes: a reduction can reflect both average-payoff dynamics and target expansion. The vertical lines mark curriculum phases. Early decay of our method also dilutes the one uniform initial action; it does not establish an asymptotic order.

![Fixed target](prefix_fixed_target.png)

The same payoff prefixes are compared with the fixed final S(Q_T). This is a retrospective diagnostic, using the final realized hull only for evaluation. It separates changes of the target from changes of the mean payoff and does not change any learner's information.

![Separate horizon sweep](horizon_error.png)

The independent terminal sweep runs 120 new common paths / 360 method trajectories: 20 seeds 45000000–45000019 at each prespecified horizon 64, 128, 256, 512, 1024 and 2048. Curriculum phase lengths and the original Hedge/opponent learning-rate formulas scale with the announced horizon. These are separate runs, not truncated longest-run prefixes. Settings W=16, rho=0.25 and Hedge multiplier 4 are unchanged. No parameter is retuned, no seed omitted and no new simulator episode collected.

Median lines and 10–90% empirical path bands are conditional on the frozen training calibration. Values at or below the declared numerical display floor 1e-10 are counted in the lower panel, never shifted by an epsilon for logarithmic plotting. Every numerical lower/upper distance, gap, alpha=sqrt(gap), success flag, feasibility residual and affine dimension is published in the NPZ arrays. Numerical checks are not exact-arithmetic proofs. The dashed T^(-1/2) line is a display-scaled guide for dimension at most two, not a fitted theorem constant.

Median terminal numerical upper distances:

| T | Our one-switch | Window | Hedge |
|---:|---:|---:|---:|
| 64 | 0.000187074 | ≤ 1e−10 | 0.0153491 |
| 128 | ≤ 1e−10 | ≤ 1e−10 | 0.0112626 |
| 256 | ≤ 1e−10 | ≤ 1e−10 | 0.00788035 |
| 512 | ≤ 1e−10 | ≤ 1e−10 | 0.00525658 |
| 1024 | ≤ 1e−10 | ≤ 1e−10 | 0.0034067 |
| 2048 | ≤ 1e−10 | ≤ 1e−10 | 0.00211179 |


Bands whose lower limit reaches the display floor are omitted on the logarithmic panel; the share of values at or below 1e-10 is shown separately.

The terminal sweep does not identify a decay order for our method or window when their errors are at numerical precision. These figures can show small full-target error relative to Hedge; they do not establish an advantage over the window strategy. See metrics.json for all values and any descriptive slope estimate; any such estimate concerns the changing horizon-scaled scenario, not a proof of an asymptotic rate.

Reproduce figures with the committed arrays:

    python scripts/build_cage_geometry_figures.py --stage report

To rerun original-path projections and new trajectories:

    python scripts/build_cage_geometry_figures.py --stage all --workers 4

The full stage replays published primary_inputs.npz if original ignored checkpoints are absent. It contains actions [50,3,512,6], opponent_actions [50,512,3], tensor, weights, methods and path_seeds. horizon_inputs.npz publishes every new path/action array keyed by horizon. prefix_geometry.npz and horizon_geometry.npz store all diagnostics. Protocols record settings, seeds and frozen source hashes before the new sweep; source files preserve checkpoint hashes; metrics.json records software; SHA256SUMS.json covers all artifacts.

Calibration uses the pinned [official CAGE Challenge 2 simulator](https://github.com/cage-challenge/cage-challenge-2/tree/26ce1c1253fa9e2e73f25e6a7f2da32860c11257). See [methodology](../../../docs/cage_adaptation_en.md). The original article, calibration, selected settings and algorithm core are unchanged.
