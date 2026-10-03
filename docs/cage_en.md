[English](cage_en.md) | [Русский](cage_ru.md) | [Home](../README.md)

# CAGE 2 environment and quick start

The benchmark selects complete defensive policies in the official CAGE Challenge
2 / CybORG manufacturing-network scenario. Inputs are simulator episodes with
public source code. Every policy-selection round starts a new 50-step episode.
The learner selects a distribution over six frozen policies before the current
attack mode is disclosed. Mixture losses average whole-episode outcomes.

## Frozen policies

| Blue policy | Behavior |
|---|---|
| `monitor` | Unmodified upstream `BlueMonitorAgent`. |
| `react_remove` | Unmodified upstream `BlueReactRemoveAgent`. |
| `react_restore` | Unmodified upstream `BlueReactRestoreAgent`. |
| `decoy_cycle` | Cycle native decoys across four specified critical hosts. |
| `critical_restore` | Restore `Op_Server0` and `Enterprise2` alternately on steps 1, 4, 7, …; monitor otherwise. |
| `decoy_react` | Respond to process alerts with rate-limited restores; deploy cyclic decoys otherwise. |

The last three are fixed heuristics defined in this repository. They use allowed
partial Blue observations. Process alerts can refer to legitimate activity;
unsupported decoy placements and failed restores remain in the measured scores.
The Red menu contains the upstream `SleepAgent`, `RedMeanderAgent`, and
`B_lineAgent`. The attacker learns to choose among these modes; their internal
tactics remain fixed. The curriculum starts with Meander.

## Loss model and feedback

The four positive loss coordinates are host compromise, server compromise,
operational disruption, and Restore cost. Their negative sum is checked against
the simulator's native reward at every step. The primary reported endpoint is
this **sum per native simulator step**, in simulator reward units.

The component bounds `(0.8, 4, 10, 1)` give the a priori vector normalization
`sqrt(117.64)`. Forty calibration episodes per defense/attack cell estimate the
known vector-loss tensor. Only that training model is used for learner updates
and attack-mode selection. The public validation and final-test banks are kept
separate from calibration. Their provenance includes upstream revision, seed
ranges, software versions, source hashes, and reward-accounting checks.

All compared methods receive the same calibrated table and learn the realized
attack label after choosing. This feedback supplies a mode label and permits
model-based counterfactual scoring; ordinary partial network observations would
require an additional mode-recognition model. Within an episode, the frozen Blue
policies retain their ordinary partial observations.

The final curriculum has an initial Meander phase, a middle phase exposing and
sampling the three modes, and a final exponential-weights attacker trained
against a fixed uniform defender reference. The final attacker mixes 90% learned
weights with 10% uniform exploration. All methods face the same realized path.

## Recommended implementation

The public API `lowdim_games.RecommendedOneSwitchLearner` fixes the validated
forecast window to 16 and `rho` to 0.25. It runs continuously, with a uniform
first action. The constructor is:

```python
from lowdim_games import RecommendedOneSwitchLearner

learner = RecommendedOneSwitchLearner(
    tensor, response, horizon,
    weights=weights, initial_prior=initial_prior,
)
# On each round: p = learner.choose(); then learner.observe(ell).
# ell is the one-hot vector of the mode revealed after choosing p.
```

`tensor` and `response` describe the training game. Forecast weights and an
initial prior may be supplied; the prior does not replace the uniform first
move. The scalar-aware choice retains the original hull, witness, direction
update, switch threshold, and safe tail. [Mathematical details](algorithms_en.md).
The recorded validation establishes the defaults for this CAGE experiment;
other applications need separate validation.

## Run from the published banks

Run from the repository root with Python 3.10 or later:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test,cage]"
python -m pytest -q
python -m lowdim_games.cli cage-selected --horizon 16 --seeds 41000000 --output results/runs/cage_selected_smoke
```

This is a short reproducibility check using the unchanged calibration, selected
configurations, and public final-test bank. It performs no new parameter search.
The full recorded horizon and 50 path seeds are the command defaults:

```bash
python -m lowdim_games.cli cage-selected --output results/runs/cage_selected
```

The runner produces the selected one-switch, selected window, and selected
Hedge results. A changed horizon or seed set is a new run, distinct from the
locked primary analysis. See the [primary report](../results/cage_adaptation/README.md)
and [validation and analysis protocol](cage_adaptation_en.md).

## Collect simulator episodes again

Episode collection additionally requires a clean pinned upstream checkout:

```bash
git clone https://github.com/cage-challenge/cage-challenge-2 .external/cage-challenge-2
git -C .external/cage-challenge-2 checkout 26ce1c1253fa9e2e73f25e6a7f2da32860c11257
```

Collection rejects tracked modifications to the pinned simulator. Exact bank
commands are recorded in the validation protocol. Published aggregate inputs
are sufficient for policy-selection reruns; the external checkout is needed
only when invoking the simulator collector.

## Interpretation

The opponent's affine dimension is at most two because the menu has three modes.
This is a property of the defined game. Scores are conditional on a finite
calibrated model and frozen policies, and do not establish guarantees for the
simulator's unknown exact expectations. They are not directly comparable with
trained challenge leaderboard agents.

The reported final study evaluates scalar loss. It does not rebuild vector
target geometry. There are no safe switches: with pure observations from three
modes, the cumulative novelty residual is at most four, below the standard
budget. The measured behavior therefore exercises the fast oracle.

## Sources

- [Official CAGE Challenge 2 source and protocol](https://github.com/cage-challenge/cage-challenge-2).
- [Pinned simulator source](https://github.com/cage-challenge/cage-challenge-2/tree/26ce1c1253fa9e2e73f25e6a7f2da32860c11257/CybORG).
- [Kiely et al., On Autonomous Agents in a Cyber Defence Environment](https://arxiv.org/abs/2309.07388).
