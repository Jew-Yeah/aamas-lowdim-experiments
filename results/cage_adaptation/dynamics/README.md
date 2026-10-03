[English](README.md) | [Русский](README.ru.md)

# Dynamics of the selected method and references

These additional figures were created after the study was completed. They use all 50 predeclared final paths and all 400 held-out simulator-seed tables. Algorithms, calibration, selected configurations and the original two primary contrasts are unchanged. No new simulator episodes were run.

## Loss dynamics

![Loss dynamics](loss_dynamics.png)

Lines average over every path and seed table, with a trailing 16-round moving average. The initial windows use all rounds available so far. Smoothing is for visualization only. Attacker phases are the initial mode (rounds 1–128), broad exploration (129–384), and learning to choose among the frozen attack modes against a uniform reference defense (385–512). All defenses face the same exogenous path; the attacker does not learn separately against each compared method.

Bands are descriptive pointwise 95% intervals, not simultaneous bands over the complete curve. There are 2000 paired crossed bootstrap draws (seed 46000000). Entire simulator-seed tables and entire common paths are resampled, with the same weights for all methods. Rounds and table cells are not independent sampling units. Inference conditions on the calibration and locked configurations, excluding their uncertainty.

## Cumulative paired differences

![Cumulative differences](cumulative_differences.png)

The sign is our method minus reference: negative values favor our method. The vertical axis is expected cumulative native cost: 50 × the sum of within-episode average losses over policy-selection rounds. Each round represents expected mixture outcomes from independently reset 50-step episodes, rather than one persistent network deployment.

The window difference is constant after round 1 because all subsequent actions coincide on all 50 paths. Dividing this constant by the number of rounds yields a 1/t decline. This is an arithmetic first-action effect, not the convergence rate of the theorem's vector error.

## Cost components

![Four cost components](loss_components.png)

The four original positive native costs are host compromise, server compromise, operational disruption, and restore cost. Their sum is the primary scalar metric. Every bar axis starts at zero; panel scales differ. Descriptive 95% intervals use the same crossed draws. Paired component differences and intervals are also retained in `summary.json`; no new hypothesis tests or component-wise multiplicity correction are performed.

## Between-path differences

![Paired path distribution](paired_path_distribution.png)

The distribution contains 50 paired path differences. Each first averages over the same bank of 400 seeds. It describes between-path variation conditional on that bank; 50 × 400 evaluations are not treated as 20,000 independent observations. All 50 window differences coincide and are shown as a point and jump, without a smoothed density.

## Parameter sensitivity

![Validation grid](validation_sensitivity.png)

The complete predeclared 20-configuration scalar-oracle grid uses only validation data (100 seeds and 20 paths). The W=16, rho=0.25 configuration locked before test is framed. No parameters were retuned after inspecting final outcomes.

## Reproducibility

Shared trajectory inputs are in `../geometry/primary_inputs.npz`: actions of all three selected methods and the common path for every primary seed. Their provenance is checked against all completed original checkpoint hashes; reconstructed means are checked against the unchanged primary report. The held-out seed bank is `../../../data/cage2_adaptation/test50/bank.npz`.

`protocol.json` records the additional visualization plan, `summary.json` records estimates and input provenance, and `plot_data.npz` contains computed curves and intervals. These extra intervals do not replace the original simultaneous intervals for the two predeclared comparisons in the [primary report](../README.md).

From the repository root:

```powershell
python scripts/build_cage_dynamic_figures.py --output results/runs/dynamic_rebuild
```

Publication exports: [dynamics](loss_dynamics.pdf), [cumulative differences](cumulative_differences.pdf), [components](loss_components.pdf), [distribution](paired_path_distribution.pdf), [sensitivity](validation_sensitivity.pdf). Russian PNG/PDF figures are available through the language switch.
