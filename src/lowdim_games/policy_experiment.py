"""Causal policy selection in an independently calibrated finite vector game.

Rounds are reset simulator episodes. Mixed policies represent expected episode
outcomes, not a policy that mixes native simulator actions at each internal step.
The frozen calibration tensor is available to learners; held-out episode means
and episode banks are used only after actions and opponent paths are fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

from .benchmarks import affine_dimension
from .experiment import checkpoints_for, software_versions, tensor_hash, write_json
from .game import FiniteGame
from .learners import FastHullLearner, OneSwitchLearner, SafeBlockLearner


def _field(calibration, name, default=None):
    return calibration.get(name, default) if isinstance(calibration, Mapping) else getattr(calibration, name, default)


def _simplex(value, size):
    vector = np.asarray(value, dtype=float)
    if (vector.shape != (size,) or not np.isfinite(vector).all()
            or np.min(vector) < 0 or abs(float(vector.sum()) - 1) > 1e-8):
        raise ValueError(f"A nonnegative probability vector of length {size} is required.")
    return vector / vector.sum()


class _Baseline:
    def __init__(self, game, horizon):
        self.game, self.horizon = game, horizon
        self.round = 0
        self._pending = None

    def _start(self, p):
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self._pending is None:
            self._pending = _simplex(p, self.game.K).copy()
        return self._pending.copy()

    def observe(self, ell):
        if self._pending is None:
            raise RuntimeError("choose() must precede observe().")
        z = _simplex(ell, self.game.M)
        self._update(z)
        self.round += 1
        self._pending = None
        return {}

    def _update(self, ell):
        pass


class FixedPolicy(_Baseline):
    """A fixed distribution over complete defender policies."""

    def __init__(self, game, horizon, distribution):
        super().__init__(game, horizon)
        self.distribution = _simplex(distribution, game.K)

    def choose(self):
        return self._start(self.distribution)


class PastWindowResponse(_Baseline):
    """Weighted best response to previous revealed red modes only."""

    def __init__(self, game, horizon, initial_distribution, window=16):
        super().__init__(game, horizon)
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.initial_distribution = _simplex(initial_distribution, game.M)
        self.window = window
        self.history = []

    def choose(self):
        mixture = self.initial_distribution if not self.history else np.mean(self.history[-self.window:], axis=0)
        return self._start(self.game.response(mixture))

    def _update(self, ell):
        self.history.append(ell.copy())


class HedgeLearner(_Baseline):
    """Full-information exponential weights on the frozen scalar loss table.

    eta minimizes log(K)/eta + eta*T*R^2/8, where R is the global scalar
    calibration loss range. The stated bound concerns calibration losses;
    estimated held-out losses and their empirical regret are separate metrics.
    """

    def __init__(self, game, horizon):
        super().__init__(game, horizon)
        self.loss_range = float(np.ptp(game.scalar_costs))
        self.learning_rate = (np.sqrt(8 * np.log(game.K) / horizon) / self.loss_range
                              if game.K > 1 and self.loss_range > 0 else 0.0)
        self.regret_bound = self.loss_range * np.sqrt(horizon * np.log(game.K) / 2)
        self.log_weights = np.zeros(game.K)

    def choose(self):
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        return self._start(weights / weights.sum())

    def _update(self, ell):
        losses = self.game.scalar_costs @ ell
        self.log_weights -= self.learning_rate * losses
        self.log_weights -= np.max(self.log_weights)


def phase_boundaries(horizon, fractions=(0.25, 0.5, 0.25)):
    fractions = np.asarray(fractions, dtype=float)
    if (horizon < 3 or fractions.shape != (3,) or not np.isfinite(fractions).all()
            or np.min(fractions) <= 0 or abs(float(fractions.sum()) - 1) > 1e-8):
        raise ValueError("At least three rounds and three positive fractions summing to one are required.")
    first = max(1, min(horizon - 2, int(np.floor(horizon * fractions[0]))))
    second = max(first + 1, min(horizon - 1, int(np.floor(horizon * (fractions[0] + fractions[1])))))
    return (0, first, second, horizon)


class AdaptivePolicyOpponent:
    """Known red-policy library with a weak, broad, then learning curriculum.

    Its last phase uses exponential weights to maximize the fixed calibration
    scalar losses of *previous* defender mixtures. No current defender action,
    held-out outcome, or current simulator state enters choose(). A fixed
    exploration mixture prevents an unsupported assertion of optimal attack.
    """

    def __init__(self, tensor, weights, horizon, seed, *, initial_distribution=None,
                 phase_fractions=(0.25, 0.5, 0.25), exploration=0.1,
                 learning_rate=None):
        self.game = FiniteGame(tensor, weights)
        self.horizon = int(horizon)
        self.boundaries = phase_boundaries(self.horizon, phase_fractions)
        if not 0 <= exploration <= 1:
            raise ValueError("exploration must lie in [0,1].")
        self.exploration = float(exploration)
        self.initial_distribution = _simplex(
            np.eye(self.game.M)[0] if initial_distribution is None else initial_distribution,
            self.game.M)
        reward_range = float(np.ptp(self.game.scalar_costs))
        self.learning_rate = (np.sqrt(8 * np.log(self.game.M) / self.horizon) / reward_range
                              if self.game.M > 1 and reward_range > 0 else 0.0)
        if learning_rate is not None:
            if not np.isfinite(learning_rate) or learning_rate < 0:
                raise ValueError("attacker learning_rate must be finite and nonnegative.")
            self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.round = 0
        self.log_weights = np.zeros(self.game.M)
        self._pending = None

    def choose(self):
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self._pending is not None:
            return self._pending.copy()
        _, weak_end, broad_end, _ = self.boundaries
        if self.round < weak_end:
            distribution = self.initial_distribution
            index = int(self.rng.choice(self.game.M, p=distribution))
        elif self.round < broad_end:
            # Expose each known mode explicitly before random broad exploration.
            broad_round = self.round - weak_end
            index = broad_round if broad_round < self.game.M else int(self.rng.integers(self.game.M))
        else:
            weights = np.exp(self.log_weights - np.max(self.log_weights))
            distribution = ((1 - self.exploration) * weights / weights.sum()
                            + self.exploration / self.game.M)
            index = int(self.rng.choice(self.game.M, p=distribution))
        self._pending = np.eye(self.game.M)[index]
        return self._pending.copy()

    def observe(self, previous_defender_action):
        if self._pending is None:
            raise RuntimeError("Opponent choose() must precede observe().")
        p = _simplex(previous_defender_action, self.game.K)
        rewards = p @ self.game.scalar_costs
        self.log_weights += self.learning_rate * rewards
        self.log_weights -= np.max(self.log_weights)
        self.round += 1
        self._pending = None


def curriculum_path(tensor, weights, horizon, seed, **kwargs):
    """A common exogenous attack path learned against the uniform reference."""
    opponent = AdaptivePolicyOpponent(tensor, weights, horizon, seed, **kwargs)
    reference = np.full(opponent.game.K, 1 / opponent.game.K)
    path = []
    for _ in range(horizon):
        path.append(opponent.choose())
        opponent.observe(reference)
    return np.asarray(path)


def _fixed_names(blue_names):
    names = ["fixed_" + re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_") for name in blue_names]
    if len(set(names)) != len(names):
        raise ValueError("Defender names must produce distinct fixed-policy identifiers.")
    return names


def make_policy_learner(name, game, horizon, blue_names, initial_distribution, window):
    if name == "one_switch":
        return OneSwitchLearner(game.tensor, game.response, horizon)
    if name == "shared_past_hull":
        return FastHullLearner(game.tensor, game.response, horizon)
    if name == "block_safe":
        return SafeBlockLearner(game.tensor, game.response, horizon)
    if name == "uniform":
        return FixedPolicy(game, horizon, np.full(game.K, 1 / game.K))
    if name == "historical_best":
        return FixedPolicy(game, horizon, game.response(initial_distribution))
    if name == "last_window":
        return PastWindowResponse(game, horizon, initial_distribution, window)
    if name == "hedge":
        return HedgeLearner(game, horizon)
    fixed = _fixed_names(blue_names)
    if name in fixed:
        return FixedPolicy(game, horizon, np.eye(game.K)[fixed.index(name)])
    raise ValueError(f"Unknown policy-selection method: {name}")


def mean_ci95(samples):
    """Student t interval across held-out simulator seeds, conditional on path."""
    values = np.asarray(samples, dtype=float)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return {"mean": mean, "lower": None, "upper": None, "seeds": len(values)}
    half_width = student_t.ppf(0.975, len(values) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return {"mean": mean, "lower": mean - half_width, "upper": mean + half_width, "seeds": len(values)}


def _phase_metrics(actions, path, raw_test, train_tensor, weights, metric_names, boundaries):
    result = []
    for label, start, stop in zip(("weak", "broad", "learning"), boundaries[:-1], boundaries[1:]):
        p, ell = actions[start:stop], path[start:stop]
        count = stop - start
        vector = np.einsum("ti,ijd,tj->d", p, raw_test, ell) / count
        fixed = np.einsum("ijd,tj,d->i", raw_test, ell, weights) / count
        scalar = float(vector @ weights)
        train_scalar = float(np.einsum("ti,ijd,tj,d->", p, train_tensor, ell, weights) / count)
        result.append({"phase": label, "first_round": start + 1, "last_round": stop,
                       "mean_test_metrics": dict(zip(metric_names, vector)),
                       "mean_test_weighted_loss": scalar,
                       "mean_train_weighted_loss": train_scalar,
                       "best_fixed_test_index": int(np.argmin(fixed)),
                       "test_regret_to_best_fixed_on_own_path": float(count * (scalar - fixed.min())),
                       "opponent_mode_counts": ell.sum(axis=0)})
    return result


def run_policy_comparison(calibration, *, scenario="curriculum", horizon=512, seed=20261003,
                          output_dir, name=None, methods=None,
                          initial_distribution=None, phase_fractions=(0.25, 0.5, 0.25),
                          window=16, attacker_lr=None, attacker_exploration=0.1, config=None):
    """Run a calibration-only causal comparison and evaluate held-out outcomes.

    ``calibration`` can be a mapping or object. Required fields are tensor,
    train_mean, test_mean, weights, blue_names, red_names, metric_names. Optional
    test_episode_losses has shape [seed,K,M,d] and enables paired conditional
    seed intervals. The scenarios curriculum/fixed share an exogenous path;
    interactive runs the same attacker rule separately against each learner.
    """
    tensor = np.asarray(_field(calibration, "tensor"), dtype=float)
    game = FiniteGame(tensor, _field(calibration, "weights"))
    train_mean = np.asarray(_field(calibration, "train_mean"), dtype=float)
    test_mean = np.asarray(_field(calibration, "test_mean"), dtype=float)
    if any(a.shape != tensor.shape or not np.isfinite(a).all() for a in (train_mean, test_mean)):
        raise ValueError("Calibration and held-out means must be finite and match tensor shape.")
    if game.normalization_bound > 1 + 1e-8:
        raise ValueError("Calibration tensor vertex norms must be at most one.")
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 3:
        raise ValueError("horizon must be an integer of at least three.")
    horizon = int(horizon)
    boundaries = phase_boundaries(horizon, phase_fractions)
    blue_names = tuple(_field(calibration, "blue_names", [str(i) for i in range(game.K)]))
    red_names = tuple(_field(calibration, "red_names", [str(i) for i in range(game.M)]))
    metric_names = tuple(_field(calibration, "metric_names", [str(i) for i in range(game.d)]))
    if (len(blue_names), len(red_names), len(metric_names)) != tensor.shape:
        raise ValueError("Policy and metric names must match the tensor axes.")
    initial = _simplex(np.eye(game.M)[0] if initial_distribution is None else initial_distribution, game.M)
    if scenario not in ("fixed", "curriculum", "interactive"):
        raise ValueError("scenario must be fixed, curriculum, or interactive.")
    bank = _field(calibration, "test_episode_losses")
    if bank is not None:
        bank = np.asarray(bank, dtype=float)
        if bank.ndim != 4 or bank.shape[1:] != tensor.shape or not np.isfinite(bank).all() or len(bank) == 0:
            raise ValueError("test_episode_losses must have shape [seed,K,M,d].")
        if not np.allclose(bank.mean(axis=0), test_mean, rtol=1e-10, atol=1e-10):
            raise ValueError("Held-out episode bank does not average to test_mean.")
    fixed_names = _fixed_names(blue_names)
    methods = list(methods) if methods is not None else [
        "one_switch", "shared_past_hull", "block_safe", "uniform",
        "historical_best", "last_window", "hedge", *fixed_names]
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("methods must be a nonempty list of distinct identifiers.")
    opponent_kwargs = dict(initial_distribution=initial, phase_fractions=phase_fractions,
                           exploration=attacker_exploration, learning_rate=attacker_lr)
    common_path = None
    if scenario == "curriculum":
        common_path = curriculum_path(tensor, game.weights, horizon, seed, **opponent_kwargs)
    elif scenario == "fixed":
        rng = np.random.default_rng(seed)
        common_path = np.eye(game.M)[rng.choice(game.M, size=horizon, p=initial)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"policy_{scenario}_T{horizon}_seed{seed}"
    np.savez_compressed(output_dir / f"{name}__game.npz", tensor=tensor,
                        train_mean=train_mean, test_mean=test_mean, weights=game.weights,
                        blue_names=blue_names, red_names=red_names, metric_names=metric_names)
    checkpoints = sorted(set(checkpoints_for(horizon)) | set(boundaries[1:]))
    summaries, curves, conditional_samples = [], [], {}
    max_target_gap = 0.0
    for method in methods:
        learner = make_policy_learner(method, game, horizon, blue_names, initial, window)
        opponent = (AdaptivePolicyOpponent(tensor, game.weights, horizon, seed, **opponent_kwargs)
                    if scenario == "interactive" else None)
        actions, path, vectors, test_vectors, records = [], [], [], [], []
        started = perf_counter()
        for index in range(horizon):
            # Both choose operations use past information. In particular no p_t
            # is supplied to the opponent before it commits to its current ell.
            ell = opponent.choose() if opponent is not None else common_path[index].copy()
            p = learner.choose()
            record = learner.observe(ell)
            if opponent is not None:
                opponent.observe(p)
            actions.append(p.copy())
            path.append(ell.copy())
            vectors.append(game.payoff(p, ell))
            test_vectors.append(np.einsum("i,ijd,j->d", p, test_mean, ell))
            records.append(record)
            t = index + 1
            if t in checkpoints:
                mean = np.mean(vectors, axis=0)
                realized = game.target_projection(mean, np.asarray(path), enumerate_vertices=game.M <= 6)
                common_target = game.target_projection(mean, np.eye(game.M), enumerate_vertices=game.M <= 6)
                if not realized.success or not common_target.success:
                    raise RuntimeError(f"Full-target numerical projection failed: {name}/{method}/{t}.")
                max_target_gap = max(max_target_gap, realized.gap, common_target.gap)
                curves.append({"experiment": name, "method": method, "round": t, "delta_realized_hull": realized.distance,
                               "delta_realized_lower": realized.lower_distance,
                               "delta_realized_error_bound": realized.alpha,
                               "delta_full_library": common_target.distance,
                               "delta_full_library_lower": common_target.lower_distance,
                               "delta_full_library_error_bound": common_target.alpha,
                               "realized_projection_gap": realized.gap,
                               "full_library_projection_gap": common_target.gap,
                               "mean_test_metrics": np.mean(test_vectors, axis=0),
                               "mean_train_weighted_loss": float(np.mean(vectors, axis=0) @ game.weights)})
        actions, path, vectors, test_vectors = map(np.asarray, (actions, path, vectors, test_vectors))
        occupancy = actions.T @ path / horizon
        mean_test = test_vectors.mean(axis=0)
        fixed_test = np.einsum("ijd,tj,d->i", test_mean, path, game.weights) / horizon
        fixed_train = game.scalar_costs @ path.sum(axis=0) / horizon
        final = curves[-1]
        scalar_test = float(mean_test @ game.weights)
        scalar_train = float(vectors.mean(axis=0) @ game.weights)
        summary = {"experiment": name, "scenario": scenario, "seed": seed,
                   "method": method, "horizon": horizon,
                   "mean_test_metrics": dict(zip(metric_names, mean_test)),
                   "mean_test_weighted_loss": scalar_test,
                   "mean_train_weighted_loss": scalar_train,
                   "test_regret_to_best_fixed_on_own_path": float(horizon * (scalar_test - fixed_test.min())),
                   "train_regret_to_best_fixed_on_own_path": float(horizon * (scalar_train - fixed_train.min())),
                   "best_fixed_test_policy": blue_names[int(np.argmin(fixed_test))],
                   "mean_fixed_test_weighted_losses_on_own_path": dict(zip(blue_names, fixed_test)),
                   "mean_uniform_reference_test_loss_on_own_path": float(fixed_test.mean()),
                   "realized_affine_dimension": affine_dimension(path),
                   "opponent_mode_counts": dict(zip(red_names, path.sum(axis=0))),
                   "opponent_path_sha256": tensor_hash(path),
                   "actions_sha256": tensor_hash(actions),
                   "switch_round": getattr(learner, "switch_round", None),
                   "switch_threshold": getattr(learner, "G_T", None),
                   "final_fast_residual_sum": float(records[-1].get("cumulative_residual", 0)),
                   "max_saddle_gap": max(r.get("saddle_gap", 0) for r in records),
                   "max_past_hull_projection_gap": max(r.get("projection_gap", 0) for r in records),
                   "delta_realized_hull": final["delta_realized_hull"],
                   "delta_realized_error_bound": final["delta_realized_error_bound"],
                   "delta_full_library": final["delta_full_library"],
                   "delta_full_library_error_bound": final["delta_full_library_error_bound"],
                   "phases": _phase_metrics(actions, path, test_mean, tensor, game.weights, metric_names, boundaries),
                   "wall_seconds_including_evaluation": perf_counter() - started}
        if isinstance(learner, HedgeLearner):
            summary["hedge_learning_rate"] = learner.learning_rate
            summary["hedge_train_regret_bound"] = learner.regret_bound
        if bank is not None:
            replicates = np.einsum("ij,eijd->ed", occupancy, bank)
            conditional_samples[method] = replicates
            summary["heldout_metric_ci95"] = mean_ci95(replicates)
            summary["heldout_weighted_loss_ci95"] = mean_ci95(replicates @ game.weights)
        diagnostics = {key: np.array([record.get(key, 0) or 0 for record in records])
                       for key in ("residual", "cumulative_residual", "h_t", "saddle_gap", "projection_gap", "switch")}
        np.savez_compressed(output_dir / f"{name}__{method}.npz", actions=actions,
                            opponent_actions=path, train_payoffs=vectors, test_payoffs=test_vectors,
                            occupancy=occupancy, **diagnostics)
        summaries.append(summary)
    if bank is not None:
        for summary in summaries:
            samples = conditional_samples[summary["method"]]
            summary["heldout_paired_weighted_loss_difference_ci95"] = {
                reference: mean_ci95((samples - conditional_samples[reference]) @ game.weights)
                for reference in ("uniform", "hedge", *fixed_names) if reference in conditional_samples}
    manifest = {"experiment": name, "scenario": scenario, "horizon": horizon, "seed": seed,
                "config": config or {}, "blue_names": blue_names, "red_names": red_names,
                "metric_names": metric_names, "weights": game.weights,
                "normalization_scale": _field(calibration, "normalization_scale"),
                "initial_distribution": initial, "phase_boundaries": boundaries,
                "phase_fractions": phase_fractions, "window": window,
                "attacker_learning_rate_override": attacker_lr, "attacker_exploration": attacker_exploration,
                "same_opponent_path_across_methods": common_path is not None,
                "attacker_feedback": "calibration scalar losses of past defender mixtures",
                "common_curriculum_reference": "uniform defender" if scenario == "curriculum" else None,
                "heldout_feedback_to_learners_or_attacker": False,
                "mixture_interpretation": "expected reset-episode outcomes of complete policies",
                "interval_interpretation": "paired held-out simulator seeds, conditional on calibrated policies and each fixed path; calibration uncertainty excluded",
                "interactive_comparison_caveat": "same attacker rule and RNG seed; different defender histories can produce different paths and realized targets" if scenario == "interactive" else None,
                "tensor_sha256": tensor_hash(tensor), "train_mean_sha256": tensor_hash(train_mean),
                "test_mean_sha256": tensor_hash(test_mean),
                "test_episode_bank_sha256": tensor_hash(bank) if bank is not None else None,
                "calibration_provenance": _field(calibration, "provenance", {}),
                "max_target_projection_gap": max_target_gap, "software": software_versions(),
                "methods": methods, "summaries": summaries, "curves": curves}
    write_json(output_dir / f"{name}.json", manifest)
    return manifest
