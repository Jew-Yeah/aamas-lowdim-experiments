[English](cage_adaptation_en.md) | [Русский](cage_adaptation_ru.md)

# Scalar-aware oracle selection and block restarts

This study examines two implementation choices in the calibrated CAGE 2 policy-selection game: choosing a forecast-aware admissible saddle response and restarting one-switch in blocks. It preserves the original article, the original learners and the previous reports. The [protocol](cage_adaptation_protocol.json) was saved before validation scores were evaluated.

## Forecast-aware saddle response

Let B be the direction-weighted payoff matrix over the full past opponent hull. First compute the original minimax dual distribution z*, and set L = min_a (B z*)_a. Among probability vectors satisfying

```text
B.T @ p <= L + rho / sqrt(t),    0 <= rho <= 1,
```

choose one minimizing the calibrated scalar loss against the empirical distribution of the previous W attack modes. The same z* provides the target witness; the full hull, projection, direction update and certified switch budget remain in use. The initial action remains uniform, matching the original implementation.

The inequality bounds the saddle gap by rho/sqrt(t), within the manuscript's permitted error. The code recomputes the actual gap independently, records numerical excesses and falls back to the original pair when the forecast LP fails its checks. Floating-point checks are numerical evidence, rather than formal exact-arithmetic certificates. This choice changes the implemented oracle selection; it does not introduce a new convergence theorem.

## Validation and final evaluation

The training model remains the original 40-seed fit in data/cage2. All simulator episodes have 50 internal steps and use the frozen six Blue policies and three Red modes. The native sum of the four loss components is the outcome; lower is better.

A new 100-seed validation bank, seeds 38000000–38000099, selects parameters on 20 common curriculum paths, seeds 40000000–40000019, with 512 meta-rounds. The grid contains:

- Forecast-aware oracle: W in {4, 8, 16, 32, 64} and rho in {0, 0.25, 0.5, 1}.
- Window baseline: the same five window lengths.
- Hedge baseline: multipliers {0.25, 0.5, 1, 2, 4} of its original learning rate.

Each family's lowest mean validation native loss selects its configuration; exact ties follow the declared grid order. The configuration and validation input hashes are locked in selection.json before final outcomes are read.

The final bank contains 400 new seeds, 39000000–39000399. Fifty new common paths, 41000000–41000049, evaluate the locked configurations at horizon 512. The two primary contrasts compare the selected oracle with the selected window and selected Hedge. There are 10,000 paired crossed bootstrap draws, seed 44000000, with Bonferroni simultaneous family level 95%. Whole simulator seed tables and whole paths are resampled. The intervals are approximate and conditional on the fitted model and the selected configuration; they do not include training uncertainty. Advantage requires a simultaneous interval strictly below zero for method minus reference. Previous test outcomes do not participate in selection.

## Block experiments

A block is 1000 meta-rounds, each choosing a complete policy for a independently reset simulator episode. A horizon of 3000 therefore contains three blocks, rather than 3000 steps in one persistent network.

Both variants retain the calibration. At every block boundary they reset the direction, cumulative residual, safe mode and local clock, with the budget computed for the new block's actual length:

- Fresh history clears the past attack hull and forecast history.
- Retained history preserves all previously revealed attack modes, including those observed in a safe tail, while resetting the algorithm's state.

The first action of each block remains uniform. The original and selected scalar-aware methods are tested with both restart variants, along with selected window and Hedge. Twenty curriculum paths start at seed 42000000; twenty alternating paths start at 43000000 and alternate Meander and B-line every 500 rounds. These use the same new final simulator bank. Restart comparisons have descriptive paired 95% intervals and are secondary analyses.

The alternating path is deterministic. Its twenty repetitions are identical and do not create twenty independent sources of path variation.

Resetting only the residual counter, while preserving all other state, changes no actions when the switch threshold is never crossed. A full restart can change actions. Its guarantee accumulates over blocks; fixed-length restarts do not inherit the original global-horizon convergence rate. See the [restart theory note](cage_restart_theory_en.md).

## Interpretation

This remains a model-informed simulator experiment. Red learns to choose among three fixed modes, and its mode is disclosed after each policy choice. Policy mixtures mean expected outcomes of complete reset episodes. These tests do not establish performance on deployed attack logs or a network with persistent state across meta-rounds.

A tuned method can match a strong window baseline without proving an advantage over it. Report every primary comparison and the actual switch and restart counts. Favorable changes in scalar loss also do not establish a smaller vector target distance; the extended study evaluates scalar loss without rebuilding target geometry.

## Reproduction

Use the pinned simulator and dependencies described in [CAGE setup](cage_en.md). The following commands regenerate the separate validation/test banks and execute validation, selection, test trajectories, restarts and analysis in that order:

```bash
python scripts/collect_cage_bank.py --steps 50 --seed 38000000 --count 100 --partition external --exclude-calibration data/cage2 --workers 4 --output results/runs/cage_adaptation/banks/validation50
python scripts/collect_cage_bank.py --steps 50 --seed 39000000 --count 400 --partition heldout --exclude-calibration data/cage2 --workers 8 --output results/runs/cage_adaptation/banks/test50
python scripts/run_cage_adaptation_study.py --stage all --workers 8 --output results/runs/cage_adaptation/selection
python scripts/build_cage_adaptation_report.py
```

Full path checkpoints remain under the ignored runtime directory. Published bank inputs, selected parameters, validation scores, group occupancies and figures are sufficient to reconstruct the scalar comparisons. Existing checkpoints can be resumed only with the same code, protocol and calibration. Use a new output directory for a changed configuration.
