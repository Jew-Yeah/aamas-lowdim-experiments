[English](README.md) | [Русский](README.ru.md)

# CAGE 2 oracle tuning and block restarts

Results from a new test after tuning on separate validation data. The original article and previous results are preserved. The outcome is the sum of four native loss components per simulator step; lower is better.

Configurations were locked before final outcomes were read. Validation uses 100 seeds and 20 common paths. Final testing uses 400 new seeds and 50 new paths at horizon 512. A total of 9000 new 50-step simulator episodes were collected.

## Selected configurations

- scalar: {"family": "scalar", "grid_index": 9, "kind": "scalar", "name": "scalar_w16_r0p25", "rho": 0.25, "window": 16}
- window: {"family": "window", "grid_index": 22, "kind": "window", "name": "window_w16", "window": 16}
- hedge: {"eta_multiplier": 4, "family": "hedge", "grid_index": 29, "kind": "hedge", "name": "hedge_eta4"}

## Primary test

| Method | Mean loss |
|---|---:|
| selected_scalar_one_switch | 1.10418 |
| selected_window | 1.10154 |
| selected_hedge | 1.16600 |
| original_one_switch | 1.20665 |
| original_window16 | 1.10154 |
| original_hedge | 1.27458 |

| Selected scalar oracle minus reference | Difference | Simultaneous interval | Decision |
|---|---:|---:|---|
| selected_window | 0.00265 | [0.00254, 0.00275] | disadvantage |
| selected_hedge | -0.06182 | [-0.08010, -0.04421] | advantage |

Intervals are approximate: 10,000 paired crossed bootstrap draws with simultaneous family level 95% for two contrasts. Dependence within each seed table and each path is preserved. Inference is conditional on the original calibration and locked configurations. Comparisons with original untuned methods are descriptive.

Across all 50 final paths, the selected scalar oracle and selected window actions are exactly equal after the first round. The loss difference is entirely due to the oracle's first uniform action: the window responds immediately to the known initial mode. Changing the arbitrary first action is theoretically permitted but was outside the locked tuning plan. This observation does not establish superiority over the window.

## Restarts

Each block contains 1000 complete-policy selection rounds, corresponding to 50,000 internal steps across reset episodes. Horizon 3000 contains three blocks. Calibration is preserved; direction, residual, local clock and switch state are reset. Both fresh and retained history are tested. These comparisons are exploratory.

Deterministic alternating paths are identical; repetitions do not increase independent path variation.

| Method | curriculum | alternating500 |
|---|---:|---:|
| original_one_switch | 1.19995 | 1.33833 |
| fresh_restart1000 | 1.20448 | 1.29304 |
| retained_restart1000 | 1.20440 | 1.30769 |
| selected_scalar_one_switch | 1.10438 | 1.29484 |
| scalar_fresh_restart1000 | 1.10774 | 1.29424 |
| scalar_retained_restart1000 | 1.10648 | 1.29499 |
| selected_window | 1.10507 | 1.29439 |
| selected_hedge | 1.14614 | 1.39779 |

curriculum: switch counts {"original_one_switch": 0, "fresh_restart1000": 0, "retained_restart1000": 0, "selected_scalar_one_switch": 0, "scalar_fresh_restart1000": 0, "scalar_retained_restart1000": 0, "selected_window": 0, "selected_hedge": 0}; restart counts {"original_one_switch": 0, "fresh_restart1000": 40, "retained_restart1000": 40, "selected_scalar_one_switch": 0, "scalar_fresh_restart1000": 40, "scalar_retained_restart1000": 40, "selected_window": 0, "selected_hedge": 0}.

| Restart minus uninterrupted variant | Difference | Descriptive 95% interval |
|---|---:|---:|
| fresh_restart1000-minus-original_one_switch | 0.00452 | [0.00296, 0.00621] |
| retained_restart1000-minus-original_one_switch | 0.00445 | [0.00305, 0.00585] |
| scalar_fresh_restart1000-minus-selected_scalar_one_switch | 0.00336 | [0.00277, 0.00398] |
| scalar_retained_restart1000-minus-selected_scalar_one_switch | 0.00210 | [0.00158, 0.00263] |

alternating500: switch counts {"original_one_switch": 0, "fresh_restart1000": 0, "retained_restart1000": 0, "selected_scalar_one_switch": 0, "scalar_fresh_restart1000": 0, "scalar_retained_restart1000": 0, "selected_window": 0, "selected_hedge": 0}; restart counts {"original_one_switch": 0, "fresh_restart1000": 40, "retained_restart1000": 40, "selected_scalar_one_switch": 0, "scalar_fresh_restart1000": 40, "scalar_retained_restart1000": 40, "selected_window": 0, "selected_hedge": 0}.

| Restart minus uninterrupted variant | Difference | Descriptive 95% interval |
|---|---:|---:|
| fresh_restart1000-minus-original_one_switch | -0.04528 | [-0.04787, -0.04270] |
| retained_restart1000-minus-original_one_switch | -0.03063 | [-0.03158, -0.02966] |
| scalar_fresh_restart1000-minus-selected_scalar_one_switch | -0.00060 | [-0.00065, -0.00056] |
| scalar_retained_restart1000-minus-selected_scalar_one_switch | 0.00015 | [0.00012, 0.00018] |


Restarting does not modify the theorem for the original uninterrupted algorithm. A restarted variant needs a sum-of-segments bound; a fixed block length does not inherit the original global-horizon rate. Resetting only a counter changes no actions when no switch occurs. Novelty is bounded for three known pure modes, so counter resets alone need not have an effect.

## Reproducibility and limitations

This is a simulator experiment with a known training model and disclosure of the attack mode after defense selection. Mixtures represent expected complete-episode outcomes; vector target geometry is not evaluated in this extension. A scalar improvement is not a new theorem.

[Methodology](../../docs/cage_adaptation_en.md) · [Restart theory](../../docs/cage_restart_theory_en.md)

[Protocol](protocol.json) · [Locked selection](selection.json) · [Validation scores](validation_grid.json) · [Analysis](analysis.json) · [Group aggregates](groups/) · [Input banks](../../data/cage2_adaptation/)

![Primary comparisons](primary_comparisons.png)

![Restart means](restart_losses.png)

![Validation grid](validation_grid.png)
