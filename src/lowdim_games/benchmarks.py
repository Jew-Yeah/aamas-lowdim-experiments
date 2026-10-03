"""Fixed finite allocation games and controlled opponent paths."""

from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass(frozen=True)
class AllocationInstance:
    tensor: np.ndarray
    schedules: np.ndarray
    profiles: np.ndarray
    capacity: float
    scale: float
    weights: np.ndarray


def allocation_losses(schedules, demands, capacity):
    """Losses in capacity units: five deficits, reserve cost, disparity.

    Rows index schedules and columns index demand profiles. Reserve cost is
    a stated model parameter: 0.25 times allocated capacity relative to the
    baseline capacity. It is not a measured NYC operating cost.
    """
    schedules = np.asarray(schedules, dtype=float)
    demands = np.asarray(demands, dtype=float)
    shortfall = np.maximum(demands[None, :, :] - schedules[:, None, :], 0.0)
    cost = np.broadcast_to(0.25 * schedules.sum(axis=1)[:, None] / capacity,
                           shortfall.shape[:2])
    disparity = (shortfall.max(axis=2) - shortfall.min(axis=2)) / capacity
    return np.concatenate([shortfall / capacity, cost[..., None],
                           disparity[..., None]], axis=2)


def make_allocation(profiles, training_counts=None, capacity_ratio=0.8):
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != 5:
        raise ValueError("The initial allocation benchmark uses five boroughs.")
    if training_counts is None:
        training_counts = profiles
    training_counts = np.asarray(training_counts, dtype=float)
    if not 0 < capacity_ratio <= 2:
        raise ValueError("capacity_ratio must be in (0,2].")
    capacity = max(1.0, float(np.median(training_counts.sum(axis=1))) * capacity_ratio)
    shares = training_counts.mean(axis=0)
    shares = shares / shares.sum() if shares.sum() else np.full(5, 0.2)
    plans = [np.full(5, capacity / 5), capacity * shares]
    for borough in range(5):
        focus = 0.5 * shares
        focus[borough] += 0.5
        plans.append(capacity * focus)
    plans.append(1.25 * capacity * shares)
    schedules = np.stack(plans)
    losses = allocation_losses(schedules, profiles, capacity)
    scale = max(1.0, float(np.linalg.norm(losses, axis=2).max()))
    # Sum of deficits, resource cost, and disparity. Fixed before test play.
    weights = np.array([1.0] * 5 + [1.0, 0.5])
    return AllocationInstance(losses / scale, schedules, profiles, capacity,
                              scale, weights)


def make_synthetic(seed=0, ambient_profiles=8):
    rng = np.random.default_rng(seed)
    totals = rng.uniform(70, 150, ambient_profiles)
    profiles = rng.dirichlet(np.full(5, 0.7), size=ambient_profiles) * totals[:, None]
    return make_allocation(profiles)


def regime_path(q, horizon, ambient_profiles=8, seed=0):
    """Unknown independent regimes, introduced in chronological blocks.

    At the start of each block its new regime is observed explicitly. Later
    rounds mix only regimes already introduced. The realized hull therefore
    has at most q+1 vertices and exact affine dimension q after all blocks.
    This is an adaptation experiment, not a worst-case exponent estimator.
    """
    if q < 1 or q >= ambient_profiles or horizon < q + 1:
        raise ValueError("Require 1 <= q < ambient_profiles and horizon >= q+1.")
    rng = np.random.default_rng(seed)
    regimes = 0.8 * np.eye(ambient_profiles)[:q + 1] + 0.2 / ambient_profiles
    # Random permutation keeps the distinguished profile identities unknown.
    regimes = regimes[:, rng.permutation(ambient_profiles)]
    path = []
    starts = np.linspace(0, horizon, q + 2, dtype=int)
    for block in range(q + 1):
        length = starts[block + 1] - starts[block]
        path.append(regimes[block].copy())
        if length > 1:
            coefficients = rng.dirichlet(np.ones(block + 1), size=length - 1)
            path.extend(coefficients @ regimes[:block + 1])
    return np.asarray(path), regimes


def sphere_path(q, horizon, seed=0):
    rng = np.random.default_rng(seed)
    if q == 1:
        return np.linspace(-1, 1, horizon)[:, None]
    points = rng.normal(size=(horizon, q))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def layered_paraboloid(q, horizon):
    """Diameter-one prefix of the manuscript's geometric construction."""
    if q < 1 or horizon < 2:
        raise ValueError("Require q>=1 and horizon>=2.")
    r = int(np.ceil(horizon ** (1 / (q + 1))))
    while r ** (q + 1) < horizon:
        r += 1
    points = []
    for layer in range(r * r):
        for grid in product(range(r), repeat=q - 1):
            horizontal = np.asarray(grid, dtype=float) / r
            points.append(np.r_[horizontal, -horizontal @ horizontal + layer / r**2])
            if len(points) == horizon:
                points = np.asarray(points)
                from scipy.spatial.distance import pdist
                diameter = float(pdist(points).max())
                return points / diameter
    raise RuntimeError("The construction generated an insufficient number of points.")


def affine_dimension(points, tol=1e-9):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0
    return int(np.linalg.matrix_rank(points - points[0], tol=tol))


def simplex_lipschitz_bound(tensor):
    """Spectral-norm bound for the payoff on the simplex tangent space."""
    tensor = np.asarray(tensor, dtype=float)
    m = tensor.shape[1]
    tangent_projection = np.eye(m) - np.ones((m, m)) / m
    return max(float(np.linalg.norm(a.T @ tangent_projection, 2)) for a in tensor)
