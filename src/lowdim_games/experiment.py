"""Run a common causal protocol and measure full-hull target distances."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import pdist

from .benchmarks import affine_dimension, allocation_losses, simplex_lipschitz_bound
from .game import FiniteGame
from .geometry import project_convex_hull
from .learners import FastHullLearner, OneSwitchLearner, SafeBlockLearner


class FixedSchedule:
    def __init__(self, game, schedule):
        self.game, self.schedule = game, schedule

    def choose(self):
        return np.eye(len(self.game.tensor))[self.schedule]

    def observe(self, ell):
        return {}


class LastWeekResponse:
    """Best weighted response to up to seven previous daily profiles."""
    def __init__(self, game):
        self.game = game
        self.history = []

    def choose(self):
        if not self.history:
            return np.eye(len(self.game.tensor))[0]
        return self.game.response(np.mean(self.history[-7:], axis=0))

    def observe(self, ell):
        self.history.append(np.asarray(ell).copy())
        return {}


METHODS = ("one_switch", "shared_past_hull", "block_safe", "uniform",
           "historical_share", "reserve", "last_week")


def make_learner(name, game, horizon):
    if name == "one_switch":
        return OneSwitchLearner(game.tensor, game.response, horizon)
    if name == "shared_past_hull":
        return FastHullLearner(game.tensor, game.response, horizon)
    if name == "block_safe":
        return SafeBlockLearner(game.tensor, game.response, horizon)
    if name in ("uniform", "historical_share"):
        return FixedSchedule(game, 0 if name == "uniform" else 1)
    if name == "reserve":
        return FixedSchedule(game, 7)
    if name == "last_week":
        return LastWeekResponse(game)
    raise ValueError(f"Unknown method: {name}")


def checkpoints_for(horizon):
    return sorted({1, horizon, *[2**i for i in range(1, 16) if 2**i < horizon]})


def _finite_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=_finite_json, allow_nan=False) + "\n", encoding="utf-8")


def software_versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "scipy", "matplotlib", "lowdim-opponent-games")}}


def tensor_hash(tensor):
    return hashlib.sha256(np.asarray(tensor, dtype="<f8").tobytes()).hexdigest()


def run_comparison(instance, path, *, name, output_dir, raw_demands=None,
                   config=None, methods=METHODS):
    """Precompute evaluation targets; learners only see current ell after choose.

    Target construction uses future data only for post-hoc evaluation at the
    corresponding checkpoint. It is never passed to a learner.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = np.asarray(path, dtype=float)
    horizon = len(path)
    game = FiniteGame(instance.tensor, weights=instance.weights)
    np.savez_compressed(output_dir / f"{name}__game.npz", tensor=instance.tensor,
                        schedules=instance.schedules, profiles=instance.profiles,
                        weights=instance.weights, capacity=instance.capacity, scale=instance.scale)
    checkpoints = checkpoints_for(horizon)
    targets = {}
    target_time = perf_counter()
    enumerate_vertices = instance.tensor.shape[1] <= 6
    for t in checkpoints:
        targets[t] = game.target_vertices(path[:t]) if enumerate_vertices else None
    target_seconds = perf_counter() - target_time
    summaries, curves = [], []
    max_target_gap = 0.0
    for method in methods:
        learner = make_learner(method, game, horizon)
        cumulative = np.zeros(instance.tensor.shape[-1])
        deficits = np.zeros(5)
        cost = disparity = 0.0
        demand_totals = np.zeros(5)
        weighted_loss = 0.0
        max_saddle_gap = max_projection_gap = 0.0
        projected_alpha_sum = cumulative_h = 0.0
        started = perf_counter()
        actions = []
        for index, ell in enumerate(path):
            # The order here is part of the protocol, not a plotting convention.
            p = learner.choose()
            actions.append(p.copy())
            payoff = game.payoff(p, ell)
            diagnostics = learner.observe(ell)
            cumulative += payoff
            max_saddle_gap = max(max_saddle_gap, diagnostics.get("saddle_gap", 0.0))
            max_projection_gap = max(max_projection_gap, diagnostics.get("projection_gap", 0.0))
            projected_alpha_sum += diagnostics.get("alpha_t", 0.0)
            cumulative_h += diagnostics.get("h_t") or 0.0
            if raw_demands is None:
                modeled = payoff * instance.scale
            else:
                # Applied metrics use the held-out counts, independently of profile quantization.
                modeled = p @ allocation_losses(instance.schedules,
                                                  np.asarray(raw_demands[index])[None, :],
                                                  instance.capacity)[:, 0, :]
            deficits += modeled[:5] * instance.capacity
            cost += modeled[5]
            disparity += modeled[6] * instance.capacity
            demand_totals += ell @ instance.profiles if raw_demands is None else raw_demands[index]
            weighted_loss += float(modeled @ instance.weights / instance.weights.sum())
            t = index + 1
            if t in targets:
                projection = game.target_projection(cumulative / t, path[:t],
                                                    enumerate_vertices=enumerate_vertices)
                if not projection.success:
                    raise RuntimeError(f"Full-target projection failed for {name}/{method}/{t}")
                max_target_gap = max(max_target_gap, projection.gap)
                curves.append({"experiment": name, "method": method, "round": t,
                               "delta": projection.distance, "projection_gap": projection.gap,
                               "delta_lower": projection.lower_distance,
                               "delta_error_bound": projection.alpha,
                               "target_representation_size": len(projection.weights),
                               "mean_unmet_requests": float(deficits.sum() / t),
                               "mean_resource_cost": float(cost / t),
                               "mean_disparity_requests": float(disparity / t),
                               "mean_weighted_loss": float(weighted_loss / t)})
        wall_seconds = perf_counter() - started
        final = curves[-1]
        summaries.append({"experiment": name, "method": method, "horizon": horizon,
                          "delta": final["delta"],
                          "delta_lower": final["delta_lower"],
                          "delta_error_bound": final["delta_error_bound"],
                          "mean_unmet_requests": final["mean_unmet_requests"],
                          "mean_resource_cost": final["mean_resource_cost"],
                          "mean_disparity_requests": final["mean_disparity_requests"],
                          "mean_weighted_loss": final["mean_weighted_loss"],
                          "mean_deficits_by_borough": deficits / horizon,
                          "service_fraction_by_borough": 1 - np.divide(deficits, demand_totals,
                                                                       out=np.zeros(5), where=demand_totals > 0),
                          "switch_round": getattr(learner, "switch_round", None),
                          "max_saddle_gap": max_saddle_gap,
                          "max_past_hull_projection_gap": max_projection_gap,
                          "sum_projection_alpha": projected_alpha_sum,
                          "sum_h_for_fast_rounds": cumulative_h,
                          "wall_seconds_including_evaluation": wall_seconds})
        np.savez_compressed(output_dir / f"{name}__{method}.npz",
                            actions=np.asarray(actions), opponent_actions=path)
    diameter = float(pdist(path).max()) if len(path) > 1 else 0.0
    manifest = {"experiment": name, "horizon": horizon,
                "realized_affine_dimension": affine_dimension(path),
                "opponent_diameter": diameter, "payoff_lipschitz_bound": simplex_lipschitz_bound(instance.tensor),
                "normalization_scale": instance.scale, "capacity": instance.capacity,
                "weights": instance.weights, "tensor_sha256": tensor_hash(instance.tensor),
                "target_construction_seconds": target_seconds,
                "target_oracle_mode": "polyhedral_vertices_and_support_LP" if enumerate_vertices else "full_support_LP_column_generation",
                "max_target_projection_gap": max_target_gap,
                "software": software_versions(), "config": config or {},
                "methods": list(methods), "summaries": summaries, "curves": curves}
    write_json(output_dir / f"{name}.json", manifest)
    return manifest
