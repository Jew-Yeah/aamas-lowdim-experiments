[English](cage_en.md) | [Русский](cage_ru.md) | [Home](../README.md)

# CAGE 2: defense against changing attack policies

This benchmark tests whether selecting among complete defensive policies helps
when an attacker changes its behavior across episodes. It uses the official
CAGE Challenge 2 / CybORG simulator and its manufacturing-network scenario.
The inputs are simulator episodes with public source code; they are not a
historical record of attacks on an operational network.

The experiment adds a policy-selection game above the simulator. A round is a
fresh episode of 50 simulator steps. The defender chooses a distribution over
six frozen policies before the current attacking policy is revealed. Each
selected policy controls the entire episode. Mixture losses average complete
episode outcomes; they do not describe switching native actions independently
at every internal simulator step.

## Frozen policy menu

| Defender policy | Behavior |
|---|---|
| `monitor` | Unmodified upstream `BlueMonitorAgent`. |
| `react_remove` | Unmodified upstream `BlueReactRemoveAgent`. |
| `react_restore` | Unmodified upstream `BlueReactRestoreAgent`. |
| `decoy_cycle` | Cycle eight native decoy types across `Op_Server0`, `Enterprise2`, `Enterprise1`, and `Enterprise0`; do not restore. |
| `critical_restore` | Restore `Op_Server0` and `Enterprise2` alternately on steps 1, 4, 7, ...; monitor on other steps. |
| `decoy_react` | Restore hosts with a process identifier in the latest partial Blue observation, prioritizing the four listed critical hosts; allow at least three steps between restores of the same host. Deploy cyclic decoys otherwise. |

The last three policies are explicitly defined heuristics in this repository.
They are fixed before calibration and use no simulator true-state compromise
oracle. A process alert can represent legitimate activity, so `decoy_react` can
restore unnecessarily. Unsupported decoy placements can fail under the native
simulator rules; these failures are included in the measured outcomes.

The three attacking policies are the upstream `SleepAgent`, `RedMeanderAgent`,
and `B_lineAgent`, in that order. Sleep is inactive. Meander explores the network;
B-line uses prior network knowledge to pursue the operational server. The
experiment learns which fixed attacking policy to select. It does not train
new attack tactics within those policies. Meander is the default initial mode;
the label "weak" denotes its position in the curriculum and does not assert
that it causes less loss against every defense.

## Losses and calibration

The four positive loss coordinates reproduce the native negative reward:

1. `host_compromise`: privileged attacker access to non-server hosts, including
   `Defender`;
2. `server_compromise`: privileged access to the three Enterprise servers and
   `Op_Server0`;
3. `operational_disruption`: the penalty on every step while the operational
   service is absent;
4. `restore`: the executed Restore action's cost, including failed restores.

Every simulator step checks that the negative sum of the reconstructed vector
equals the native scalar reward. These are simulator reward units, not money.
Episode losses are averaged per step. Multiplying the sum of their four
coordinates by the episode length gives the negative official episode reward.

The pinned scenario gives component bounds `(0.8, 4, 10, 1)`. The entire vector
is divided by their Euclidean norm, `sqrt(117.64)`, so normalized vertex losses
have norm at most one. This bound is derived before observing calibration or
held-out outcomes. It also covers unsuccessful Restore actions.

For each of the 18 defense/attack cells, 40 calibration episodes estimate its
mean vector loss. A separate bank of 40 held-out episodes per cell evaluates
the frozen selection policies. Seeds are set before constructing a new simulator
instance, because construction randomizes the initial network. Seed indices
are shared across cells for paired evaluation; different policies can consume
random numbers differently, so paired episodes need not follow identical paths.
The default bank contains 1,440 simulator episodes and 72,000 internal steps.

Only calibration means define the known tensor, response map, and learning
updates. The response minimizes the equally weighted scalar loss, with the
smallest action index resolving a tie. Internally these weights are normalized
to sum to one. Consequently, `mean_test_weighted_loss` is the mean of the four
raw coordinates, whereas their sum is the native per-step total loss.

The held-out bank is never supplied to either learner or attacker updates.
Calibration, policy menus, scalar weights, and normalization remain fixed
throughout the policy-selection run. Changing them after looking at held-out
performance requires a new independent test bank.

## Causal scenarios

The default horizon is 512 meta-rounds. The three phase boundaries are at 25%,
75%, and 100% of the horizon.

| Scenario | Opponent protocol |
|---|---|
| `fixed` | Use the initial mode throughout; by default this is Meander. |
| `curriculum` | Begin with the initial mode. In the middle phase, explicitly expose each of the three modes and then sample modes uniformly. In the final phase, use an exponential-weights attacker trained against a fixed uniform defender reference. All methods face the same realized path. |
| `interactive` | Use the same initial and broad phases. In the final phase, the exponential-weights attacker selects modes using losses of that method's previous defender mixtures. Each method therefore has its own interaction path. |

The attacker accumulates calibration-based information during all three phases.
Its final-phase distribution combines 90% exponential weights and 10% uniform
exploration. It maximizes the equally weighted calibration scalar loss. Its
learning rate is `sqrt(8 log(M) / T) / R`, where `R` is the global scalar
calibration loss range. It commits to the current mode using only past
information. The learner also chooses before receiving that mode.

The three attacking policies and their estimated loss table are known to all selection
methods. The current mode is revealed after the choice. This is a full-feedback
meta-game assumption: the original challenge's partial Blue observations do
not themselves reveal an attack-policy label or all counterfactual losses.
Inside each episode the frozen defensive policy still uses only its allowed
partial observations. Thus the experiment evaluates selection between complete
policies under model-informed feedback.

## Comparisons

The selection methods are the manuscript's `one_switch`, `shared_past_hull`,
and `block_safe`; the uniform distribution over defense policies; the fixed
calibration response to the initial attack mode (`historical_best`); the response
to the last 16 revealed modes (`last_window`); Hedge; and all six pure fixed
defense policies. The scalar weights and feedback are common across methods.

Hedge minimizes calibration scalar loss with learning rate
`sqrt(8 log(K) / T) / R`. Its reported bound is
`R sqrt(T log(K) / 2)` for cumulative calibration regret to the best fixed
policy. Held-out regret is measured separately and is not covered by that bound.
No learning rate is selected using the held-out bank.

The fast past-hull policy is shared with the geometric method discussed in the
article's comparison with Marinov et al. The one-switch master uses the original
budget `G_T = 6 sqrt(K - 1) T^(3/4)`. In this benchmark every revealed opponent
action is a pure mode. Only first appearances can lie outside the past hull,
so the cumulative residual is at most `2(M - 1) = 4`. The default switch is
therefore impossible for these three-mode paths, at any horizon. The one-switch
and shared past-hull actions should coincide. These runs examine adaptation
of the fast policy; they do not demonstrate its safe-switch mechanism.

## Evaluation and interpretation

The practical outputs are the four held-out losses, their scalar combination,
and phase-specific outcomes. Every method is also compared with the best fixed
policy on its realized path. For interactive runs this is a descriptive replay:
deploying that fixed policy would generally change the attacker's future path.
Cross-method interactive scores compare complete interactions under the same
attacker rule, rather than responses to a common attack sequence.

Target distances use the calibration game and its fixed response map. They
include the full response target over the realized hull, with independent
support-LP residual checks. A second distance uses the full three-mode library
as a common target. This second target can be less restrictive than a realized
hull target. Report both distances alongside practical losses; a smaller target
distance does not by itself establish better security or lower restore cost.
The opponent's affine dimension is at most two because of the stated menu.
This is a design property, not evidence that real cyber attacks are inherently
two-dimensional.

The held-out episode bank provides conditional 95% Student-t intervals. For a
fixed learned path, occupancy weights combine the entire payoff table for each
held-out seed. Paired differences use matching seed indices. The independent
sample count is the held-out seed count; replaying its cell means for 512 rounds
does not generate 512 new simulator samples. These intervals condition on the
calibrated tensor and paths and omit calibration uncertainty. They are
approximate sampling intervals, not formal numerical certificates.

The finite tensor is a Monte Carlo estimate. Guarantees and numerical target
checks concern the resulting calibrated game; they do not prove guarantees for
the unknown exact simulator expectations. This is an exploratory benchmark
with a small menu of frozen heuristics. Its scores are not directly comparable
with the original challenge leaderboard or its trained RL submissions.

## Reproduction

Run the following commands from the repository root. The source checkout is
revision-pinned and calibration rejects tracked modifications to it.

```bash
python -m pip install -e ".[test,cage]"
git clone https://github.com/cage-challenge/cage-challenge-2 .external/cage-challenge-2
git -C .external/cage-challenge-2 checkout 26ce1c1253fa9e2e73f25e6a7f2da32860c11257
python -m lowdim_games.cli cage-calibrate --steps 50 --train-episodes 40 --test-episodes 40 --seed 20261003 --output data/cage2
python -m lowdim_games.cli cage --calibration data/cage2 --scenarios fixed curriculum interactive --horizon 512 --seeds 20261003 20261004 20261005 --initial-red-index 1 --plots
python -m pytest -q
```

Calibration seeds are `20261003` through `20261042`; held-out seeds are
`21261003` through `21261042`. The three policy-selection seeds govern mode
sampling and do not create additional calibration episode banks.

`data/cage2/calibration.npz` stores the separate episode banks, cell means,
normalized tensor, policy names, weights, and seeds. `provenance.json` records
the clean upstream revision, source hashes, reward-accounting checks, units,
and software versions. `cell_statistics.json` contains means, standard
deviations, and standard errors. Each run saves its manifest and actions,
opponent modes, payoffs, occupancy weights, and oracle diagnostics under
`results/runs/cage2`. Runtime includes post-hoc evaluation.

The [reference report](../results/cage_reference/README.md) records completed
runs and measured results. Methodology alone does not assert an advantage for
any method.

## Sources

- [Official CAGE Challenge 2 source and evaluation protocol](https://github.com/cage-challenge/cage-challenge-2).
- [Pinned simulator source](https://github.com/cage-challenge/cage-challenge-2/tree/26ce1c1253fa9e2e73f25e6a7f2da32860c11257/CybORG).
- [Kiely et al., On Autonomous Agents in a Cyber Defence Environment](https://arxiv.org/abs/2309.07388), the challenge's recommended citation.
