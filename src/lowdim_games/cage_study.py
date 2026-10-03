"""Scalar CAGE policy studies and paired seed/path uncertainty estimates.

This module reuses the calibrated game and original learners. It omits target
geometry from extended studies whose primary endpoint is native total loss.
Held-out simulator seeds and attack-path seeds are crossed sampling units;
bootstrap draws resample whole seeds and whole paths, never individual cells.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from .benchmarks import affine_dimension
from .experiment import tensor_hash
from .game import FiniteGame
from .policy_experiment import (
    AdaptivePolicyOpponent, _field, _fixed_names, _simplex,
    curriculum_path, make_policy_learner, phase_boundaries,
)


def simulate_policy_selection(calibration, *, scenario="curriculum", horizon=None,
                              seed=0, opponent_path=None, initial_distribution=None,
                              phase_fractions=(0.25, 0.5, 0.25),
                              attacker_exploration=0.1, attacker_lr=None,
                              methods=None, window=16):
    """Return causal actions, paths, occupancies, and calibration diagnostics.

    Only ``tensor``, ``weights``, policy names and metric names are read from
    calibration. No held-out outcomes enter this function. Supplying an
    explicit path permits named common shift or alternating-block scenarios;
    otherwise fixed, curriculum and interactive follow the original protocol.
    """
    tensor = np.asarray(_field(calibration, "tensor"), dtype=float)
    game = FiniteGame(tensor, _field(calibration, "weights"))
    if game.normalization_bound > 1 + 1e-8:
        raise ValueError("Calibration payoff norms must be at most one.")
    blue_names = tuple(_field(calibration, "blue_names", [str(i) for i in range(game.K)]))
    red_names = tuple(_field(calibration, "red_names", [str(i) for i in range(game.M)]))
    metric_names = tuple(_field(calibration, "metric_names", [str(i) for i in range(game.d)]))
    if (len(blue_names), len(red_names), len(metric_names)) != tensor.shape:
        raise ValueError("Policy and metric names must match the tensor axes.")
    initial = _simplex(np.eye(game.M)[0] if initial_distribution is None else initial_distribution, game.M)
    common_path = None
    if opponent_path is not None:
        path = np.asarray(opponent_path, dtype=float)
        if path.ndim != 2 or path.shape[1] != game.M or len(path) < 3:
            raise ValueError("An explicit path must have shape [T,M] with at least three rounds.")
        if horizon is not None and horizon != len(path):
            raise ValueError("Explicit path length must match the specified horizon.")
        common_path = np.array([_simplex(ell, game.M) for ell in path])
        horizon = len(common_path)
    if horizon is None:
        horizon = 512
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 3:
        raise ValueError("horizon must be an integer of at least three.")
    horizon = int(horizon)
    boundaries = phase_boundaries(horizon, phase_fractions)
    opponent_kwargs = dict(initial_distribution=initial, phase_fractions=phase_fractions,
                           exploration=attacker_exploration, learning_rate=attacker_lr)
    if common_path is None:
        if scenario == "curriculum":
            common_path = curriculum_path(tensor, game.weights, horizon, seed, **opponent_kwargs)
        elif scenario == "fixed":
            rng = np.random.default_rng(seed)
            common_path = np.eye(game.M)[rng.choice(game.M, size=horizon, p=initial)]
        elif scenario != "interactive":
            raise ValueError("Without an explicit path use fixed, curriculum, or interactive.")
    methods = list(methods) if methods is not None else [
        "one_switch", "shared_past_hull", "block_safe", "uniform",
        "historical_best", "last_window", "hedge", *_fixed_names(blue_names)]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a nonempty list of distinct identifiers.")
    trajectories = {}
    for method in methods:
        learner = make_policy_learner(method, game, horizon, blue_names, initial, window)
        opponent = (AdaptivePolicyOpponent(tensor, game.weights, horizon, seed, **opponent_kwargs)
                    if common_path is None else None)
        actions = np.empty((horizon, game.K))
        path = np.empty((horizon, game.M))
        payoffs = np.empty((horizon, game.d))
        records = []
        for index in range(horizon):
            ell = opponent.choose() if opponent is not None else common_path[index].copy()
            p = learner.choose()
            record = learner.observe(ell)
            if opponent is not None:
                opponent.observe(p)
            actions[index], path[index] = p, ell
            payoffs[index] = game.payoff(p, ell)
            records.append(record)
        occupancy = actions.T @ path / horizon
        diagnostic_arrays = {
            key: np.array([record.get(key, 0) or 0 for record in records])
            for key in ("residual", "cumulative_residual", "h_t", "saddle_gap", "projection_gap", "switch")}
        trajectories[method] = {
            "actions": actions, "opponent_actions": path, "occupancy": occupancy,
            "train_payoffs": payoffs, "diagnostics": diagnostic_arrays,
            "switch_round": getattr(learner, "switch_round", None),
            "switch_threshold": getattr(learner, "G_T", None),
            "realized_affine_dimension": affine_dimension(path),
            "opponent_path_sha256": tensor_hash(path),
            "actions_sha256": tensor_hash(actions),
            "max_saddle_gap": max(record.get("saddle_gap", 0) for record in records),
            "max_past_hull_projection_gap": max(record.get("projection_gap", 0) for record in records),
        }
    return {"scenario": scenario, "horizon": horizon, "seed": seed,
            "blue_names": blue_names, "red_names": red_names, "metric_names": metric_names,
            "initial_distribution": initial, "phase_boundaries": boundaries,
            "phase_fractions": phase_fractions, "window": window,
            "attacker_exploration": attacker_exploration,
            "attacker_learning_rate_override": attacker_lr,
            "same_opponent_path_across_methods": common_path is not None,
            "explicit_opponent_path": opponent_path is not None,
            "tensor_sha256": tensor_hash(tensor), "trajectories": trajectories,
            "heldout_feedback_to_learners_or_attacker": False,
            "target_geometry_evaluated": False}


def _episode_bank(bank):
    bank = np.asarray(bank, dtype=float)
    if bank.ndim != 4 or min(bank.shape) < 1 or not np.isfinite(bank).all():
        raise ValueError("Episode bank must be a nonempty finite array [seed,K,M,d].")
    return bank


def _metric_weights(weights, dimension):
    weights = np.ones(dimension) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (dimension,) or not np.isfinite(weights).all():
        raise ValueError("Metric weights must be finite and match the episode components.")
    return weights


def evaluate_episode_bank(run, bank, *, weights=None):
    """Combine each entire held-out seed table with a fixed occupancy.

    Default scalar loss is the native *sum* of components. The function does
    not rerun or update any learner. Seed pairing is retained for comparisons.
    """
    bank = _episode_bank(bank)
    weights = _metric_weights(weights, bank.shape[-1])
    evaluations = {}
    for method, trajectory in run["trajectories"].items():
        occupancy = np.asarray(trajectory["occupancy"], dtype=float)
        if occupancy.shape != bank.shape[1:3]:
            raise ValueError("Occupancy shape must match the episode-bank policy axes.")
        components = np.einsum("ij,eijd->ed", occupancy, bank, optimize=True)
        evaluations[method] = {"by_seed_components": components,
                               "by_seed_native_loss": components @ weights,
                               "mean_components": components.mean(axis=0),
                               "mean_native_loss": float(components.mean(axis=0) @ weights)}
    return {"methods": evaluations, "episode_seeds": len(bank), "weights": weights,
            "same_opponent_path_across_methods": run["same_opponent_path_across_methods"],
            "interpretation": "expected episode loss conditional on calibration and fixed occupancy"}


def _conditional_t_interval(values, alpha):
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "lower": None, "upper": None, "samples": len(values)}
    half = float(student_t.ppf(1 - alpha / 2, len(values) - 1)
                 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "samples": len(values)}


def paired_two_way_bootstrap(occupancies, reference_occupancies, bank, *, weights=None,
                             samples=10_000, seed=0, alpha=0.05, batch_size=256,
                             return_differences=True):
    """Paired crossed bootstrap of native-loss differences, method minus reference.

    Occupancies have shape [path,K,M], and bank has shape [episode_seed,K,M,d].
    Each replicate independently resamples whole episode seeds and whole path
    indices, retaining both methods' seed/path pairing and all cell dependence.
    The percentile interval and centered two-sided p-value are approximate,
    conditional on the fitted calibration game. Caller applies multiplicity.
    """
    bank = _episode_bank(bank)
    occupations = np.asarray(occupancies, dtype=float)
    reference = np.asarray(reference_occupancies, dtype=float)
    if occupations.ndim == 2:
        occupations = occupations[None, :, :]
    if reference.ndim == 2:
        reference = reference[None, :, :]
    if (occupations.shape != reference.shape or occupations.ndim != 3
            or occupations.shape[1:] != bank.shape[1:3] or not len(occupations)
            or not np.isfinite(occupations).all() or not np.isfinite(reference).all()):
        raise ValueError("Paired occupancies must be finite arrays [path,K,M] matching the bank.")
    for name, array in (("method", occupations), ("reference", reference)):
        if np.min(array) < -1e-12 or not np.allclose(array.sum(axis=(1, 2)), 1, atol=1e-8, rtol=0):
            raise ValueError(f"Every {name} occupancy must be nonnegative and sum to one.")
    if len(bank) < 2:
        raise ValueError("At least two independent held-out episode seeds are required.")
    if (isinstance(samples, bool) or int(samples) != samples or samples < 2
            or isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1):
        raise ValueError("samples >= 2 and batch_size >= 1 must be integers.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    samples, batch_size = int(samples), int(batch_size)
    weights = _metric_weights(weights, bank.shape[-1])
    scalar_bank = np.einsum("eijd,d->eij", bank, weights, optimize=True)
    differences = np.einsum("pij,eij->ep", occupations - reference, scalar_bank, optimize=True)
    estimate = float(differences.mean())
    episode_count, path_count = differences.shape
    # Separate RNG streams preserve the resampling distribution and make the
    # result independent of computation batch size.
    streams = np.random.SeedSequence(seed).spawn(2)
    episode_rng, path_rng = map(np.random.default_rng, streams)
    bootstrap = np.empty(samples)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        episode_weights = episode_rng.multinomial(
            episode_count, np.full(episode_count, 1 / episode_count), size=count) / episode_count
        path_weights = path_rng.multinomial(
            path_count, np.full(path_count, 1 / path_count), size=count) / path_count
        bootstrap[start:start + count] = np.sum((episode_weights @ differences) * path_weights, axis=1)
    lower, upper = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2], method="linear")
    pvalue = float((1 + np.sum(np.abs(bootstrap - estimate) >= abs(estimate))) / (samples + 1))
    result = {"estimate": estimate, "lower": float(lower), "upper": float(upper),
              "bootstrap_std": float(bootstrap.std(ddof=1)),
              "approximate_two_sided_pvalue": pvalue, "alpha": float(alpha),
              "samples": samples, "seed": seed, "episode_seed_count": episode_count,
              "path_seed_count": path_count, "weights": weights,
              "conditional_episode_t_interval": _conditional_t_interval(differences.mean(axis=1), alpha),
              "conditional_path_t_interval": _conditional_t_interval(differences.mean(axis=0), alpha),
              "difference_direction": "method minus reference; negative means lower loss",
              "interpretation": "paired episode-seed/path-seed crossed bootstrap; conditional on calibrated model; calibration uncertainty excluded"}
    if return_differences:
        result["paired_seed_path_differences"] = differences
    return result


def holm_adjust(pvalues):
    """Holm adjusted p-values for a caller-declared family of comparisons."""
    values = {name: float(value) for name, value in pvalues.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Every p-value must be finite and lie in [0,1].")
    ordered = sorted(values, key=values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * values[name])
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in values}


def bonferroni_alpha(comparisons, family_alpha=0.05):
    if (isinstance(comparisons, bool) or int(comparisons) != comparisons or comparisons < 1
            or not np.isfinite(family_alpha) or not 0 < family_alpha < 1):
        raise ValueError("A positive integer comparison count and family alpha in (0,1) are required.")
    return float(family_alpha / comparisons)
