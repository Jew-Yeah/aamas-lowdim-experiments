"""Finite biaffine games and the full realized-hull response target.

The target oracle intersects the *entire* opponent hull with response cells.
It never replaces S(Q) by responses at the observed actions. A support LP over
original convex-combination weights independently checks target projection.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import linprog

from .geometry import (
    OracleCertificationError, ProjectionResult, affine_hull_polytope,
    polytope_vertices, project_convex_hull,
)


@dataclass(frozen=True)
class TargetCell:
    action: int
    inequalities: np.ndarray
    strict_margin: float
    strict_margin_upper: float
    witness_weights: np.ndarray


@dataclass(frozen=True)
class TargetSupport:
    point: np.ndarray
    value: float
    upper_value: float
    gap: float
    action: int
    opponent: np.ndarray
    feasibility_error: float


@dataclass
class _TargetData:
    opponent_points: np.ndarray
    cells: list[TargetCell]
    vertices: np.ndarray | None = None


class FiniteGame:
    """A[K,M,d], with smallest-index weighted-loss minimizer as benchmark."""

    def __init__(self, tensor: np.ndarray, weights: np.ndarray | None = None,
                 oracle_tol: float = 1e-9, cache_size: int = 16):
        tensor = np.asarray(tensor, dtype=float)
        if tensor.ndim != 3 or min(tensor.shape) < 1 or not np.all(np.isfinite(tensor)):
            raise ValueError("tensor must be a finite K by M by d array")
        self.tensor = tensor.copy()
        self.tensor.setflags(write=False)
        self.K, self.M, self.d = self.tensor.shape
        if weights is None:
            weights = np.ones(self.d) / self.d
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (self.d,) or not np.all(np.isfinite(weights))
                or np.min(weights) < 0 or weights.sum() <= 0):
            raise ValueError("weights must be a nonnegative nonzero d-vector")
        self.weights = weights.copy() / weights.sum()
        self.weights.setflags(write=False)
        self.scalar_costs = np.einsum("kmd,d->km", self.tensor, self.weights)
        self.oracle_tol = float(oracle_tol)
        self.cache_size = int(cache_size)
        self._target_cache: OrderedDict[bytes, _TargetData] = OrderedDict()

    @property
    def A(self) -> np.ndarray:
        return self.tensor

    @property
    def normalization_bound(self) -> float:
        return float(np.max(np.linalg.norm(self.tensor, axis=2)))

    def action_payoffs(self, opponent: np.ndarray) -> np.ndarray:
        opponent = np.asarray(opponent, dtype=float)
        if opponent.shape != (self.M,):
            raise ValueError("opponent must be an M-vector")
        return np.einsum("kmd,m->kd", self.tensor, opponent)

    def payoff(self, learner: np.ndarray, opponent: np.ndarray) -> np.ndarray:
        learner = np.asarray(learner, dtype=float)
        if learner.shape != (self.K,):
            raise ValueError("learner must be a K-vector")
        return learner @ self.action_payoffs(opponent)

    def response_index(self, opponent: np.ndarray) -> int:
        # np.argmin implements the first-index rule without a fuzzy tie band.
        return int(np.argmin(self.scalar_costs @ np.asarray(opponent, dtype=float)))

    def response(self, opponent: np.ndarray) -> np.ndarray:
        learner = np.zeros(self.K)
        learner[self.response_index(opponent)] = 1.0
        return learner

    def response_payoff(self, opponent: np.ndarray) -> np.ndarray:
        action = self.response_index(opponent)
        return self.tensor[action].T @ np.asarray(opponent, dtype=float)

    def _key(self, points: np.ndarray) -> bytes:
        # Exact byte keys: rounding could change discontinuous response cells.
        return hashlib.sha256(np.asarray(points.shape, np.int64).tobytes() + points.tobytes()).digest()

    def _target_data(self, opponent_points: np.ndarray) -> _TargetData:
        points = np.asarray(opponent_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.M or len(points) == 0 or not np.all(np.isfinite(points)):
            raise ValueError("opponent_points must be a finite nonempty N by M array")
        points = np.unique(points, axis=0)
        key = self._key(points)
        if key in self._target_cache:
            self._target_cache.move_to_end(key)
            return self._target_cache[key]
        cells = []
        count = len(points)
        for action in range(self.K):
            rivals = [other for other in range(self.K) if other != action]
            differences = self.scalar_costs[action] - self.scalar_costs[rivals]
            inequalities = differences @ points.T
            # A tiny but nonzero score gap still changes a discontinuous
            # benchmark. Rescale each inequality before HiGHS so its absolute
            # feasibility tolerance does not erase such a gap.
            row_scale = np.max(np.abs(inequalities), axis=1) if len(rivals) else np.empty(0)
            inequalities = inequalities / np.where(row_scale > 0.0, row_scale, 1.0)[:, None]
            if action == 0:
                result = linprog(np.zeros(count), A_ub=inequalities if len(rivals) else None,
                                 b_ub=np.zeros(len(rivals)) if len(rivals) else None,
                                 A_eq=np.ones((1, count)), b_eq=[1.0],
                                 bounds=[(0.0, None)] * count, method="highs")
                if result.status == 2:
                    continue
                if not result.success:
                    raise OracleCertificationError("Response-cell feasibility LP failed")
                margin, upper = float("inf"), float("inf")
                witness = result.x
            else:
                earlier = np.array([float(other < action) for other in rivals])
                augmented = np.column_stack((inequalities, earlier))
                result = linprog(np.r_[np.zeros(count), -1.0],
                                 A_ub=augmented, b_ub=np.zeros(len(rivals)),
                                 A_eq=np.r_[np.ones(count), 0.0][None, :], b_eq=[1.0],
                                 bounds=[(0.0, None)] * count + [(0.0, 1.0)], method="highs")
                if result.status == 2:
                    continue
                if not result.success:
                    raise OracleCertificationError("Strict response-cell margin LP failed")
                margin = max(0.0, float(result.x[-1]))
                # Independent feasible dual: lambda<=0, convex weights sum to
                # one, delta in [0,1]. This bounds the maximum strict margin.
                multipliers = np.minimum(result.ineqlin.marginals, 0.0)
                nu = float(np.min(-inequalities.T @ multipliers))
                delta_reduced = -1.0 - float(earlier @ multipliers)
                upper = max(0.0, -nu - min(0.0, delta_reduced))
                if margin <= 0.0:
                    if upper > 0.0:
                        raise OracleCertificationError(
                            f"Numerically ambiguous strict response cell {action}: margin upper={upper:g}")
                    continue
                witness = result.x[:-1]
            witness = np.maximum(witness, 0.0)
            witness /= witness.sum()
            if len(rivals) and np.max(inequalities @ witness) > self.oracle_tol:
                raise OracleCertificationError("Response-cell witness is not primal feasible")
            cells.append(TargetCell(action, inequalities, margin, upper, witness))
        if not cells:
            raise OracleCertificationError("No response winner exists on a nonempty hull")
        data = _TargetData(points, cells)
        if self.cache_size > 0:
            self._target_cache[key] = data
            while len(self._target_cache) > self.cache_size:
                self._target_cache.popitem(last=False)
        return data

    def target_cells(self, opponent_points: np.ndarray) -> tuple[TargetCell, ...]:
        return tuple(self._target_data(opponent_points).cells)

    def target_vertices(self, opponent_points: np.ndarray) -> np.ndarray:
        """Vertices spanning cl conv of responses over the full opponent hull.

        A cell with earlier tied actions and no strict interior relative to
        those inequalities is excluded by its max-margin LP. Nonempty actual
        winner cells contribute their closures, including boundary limits.
        """
        data = self._target_data(opponent_points)
        if data.vertices is not None:
            return data.vertices.copy()
        hull = affine_hull_polytope(data.opponent_points)
        payoff_points = []
        for cell in data.cells:
            action = cell.action
            differences = self.scalar_costs[action] - np.delete(self.scalar_costs, action, axis=0)
            original_rows = differences @ data.opponent_points.T
            row_scale = np.max(np.abs(original_rows), axis=1) if len(differences) else np.empty(0)
            # Rows zero on all observed vertices vanish on their entire hull.
            # Removing them avoids an artificial constraint caused by SVD
            # rounding of an exactly tied affine hull.
            differences = differences[row_scale > 0.0] / row_scale[row_scale > 0.0, None]
            matrix = np.vstack((hull.matrix, differences @ hull.basis))
            bound = np.r_[hull.bound, -differences @ hull.origin]
            coordinates = polytope_vertices(matrix, bound, self.oracle_tol)
            if not len(coordinates):
                raise OracleCertificationError("A nonempty response cell lost its vertices")
            opponents = hull.lift(coordinates)
            payoff_points.extend(opponents @ self.tensor[action])
        vertices = np.asarray(payoff_points, dtype=float)
        if not len(vertices):
            raise OracleCertificationError("No full-target payoff vertices were constructed")
        vertices = np.unique(vertices, axis=0)
        data.vertices = vertices
        return vertices.copy()

    def target_support(self, direction: np.ndarray, opponent_points: np.ndarray) -> TargetSupport:
        """Support of the full target via LPs on original hull weights.

        ``upper_value`` is computed from a feasible dual, independent of
        vertex enumeration and affine-rank truncation.
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (self.d,):
            raise ValueError("direction must be a d-vector")
        data = self._target_data(opponent_points)
        count = len(data.opponent_points)
        best = None
        global_upper = -np.inf
        for cell in data.cells:
            payoffs = data.opponent_points @ self.tensor[cell.action]
            scores = payoffs @ direction
            # A projection residual can be tiny even when the game is well
            # scaled. HiGHS' absolute objective tolerance would then leave a
            # poor dual bound. A constant shift is valid because hull weights
            # sum to one; scaling the remaining variation makes the LP's
            # accuracy independent of the residual's magnitude.
            offset = float(scores.mean())
            centered_scores = scores - offset
            objective_scale = float(np.max(np.abs(centered_scores)))
            normalized_scores = (centered_scores / objective_scale
                                 if objective_scale > 0.0 else np.zeros_like(scores))
            result = linprog(-normalized_scores,
                             A_ub=cell.inequalities if len(cell.inequalities) else None,
                             b_ub=np.zeros(len(cell.inequalities)) if len(cell.inequalities) else None,
                             A_eq=np.ones((1, count)), b_eq=[1.0],
                             bounds=[(0.0, None)] * count, method="highs",
                             options={"primal_feasibility_tolerance": 1e-10,
                                      "dual_feasibility_tolerance": 1e-10})
            if not result.success:
                raise OracleCertificationError("Full-target support LP failed")
            weights = np.maximum(result.x, 0.0)
            weights /= weights.sum()
            payoff = weights @ payoffs
            value = float(direction @ payoff)
            if len(cell.inequalities):
                multipliers = np.minimum(result.ineqlin.marginals, 0.0)
                upper = offset + objective_scale * float(np.max(
                    normalized_scores + cell.inequalities.T @ multipliers))
                feasibility = max(0.0, float(np.max(cell.inequalities @ weights)))
            else:
                upper, feasibility = float(scores.max()), 0.0
            global_upper = max(global_upper, upper)
            if best is None or value > best[0]:
                best = (value, payoff, cell.action, weights @ data.opponent_points, feasibility)
        value, payoff, action, opponent, feasibility = best
        return TargetSupport(payoff, value, max(global_upper, value),
                             max(0.0, global_upper - value), action, opponent, feasibility)

    def target_projection(self, average_payoff: np.ndarray, opponent_points: np.ndarray,
                          tol: float | None = None, max_iter: int = 250,
                          enumerate_vertices: bool = True) -> ProjectionResult:
        """Numerical target projection with an independent *full-target* gap.

        Missing numerical vertices cannot silently certify an incorrect
        distance: support LPs over all actual winner cells check the result.
        If needed, their extreme points augment the projection hull. Setting
        enumerate_vertices=False starts from actual observed response outcomes
        and uses LP column generation; the same full-target certificate is
        required. The returned weights then count a representation, not all
        target vertices.
        """
        tolerance = self.oracle_tol if tol is None else float(tol)
        query = np.asarray(average_payoff, float)
        if enumerate_vertices:
            vertices = self.target_vertices(opponent_points)
        else:
            data = self._target_data(opponent_points)
            vertices = np.unique(np.asarray([self.response_payoff(point)
                                             for point in data.opponent_points]), axis=0)
        scale = max(1.0, float(query @ query), float(np.max(np.sum(vertices**2, axis=1))))
        for iteration in range(max_iter):
            result = project_convex_hull(query, vertices, tolerance * 0.1)
            direction = query - result.point
            support = self.target_support(direction, opponent_points)
            full_gap = max(0.0, 2.0 * (support.upper_value - float(direction @ result.point)))
            feasibility = max(result.feasibility_error, support.feasibility_error)
            if full_gap <= tolerance * scale and feasibility <= tolerance:
                return ProjectionResult(result.point, result.weights, result.distance, full_gap,
                                        True, feasibility, result.iterations + iteration)
            if support.feasibility_error > tolerance:
                raise OracleCertificationError("Full-target support point is not primal feasible")
            if np.min(np.linalg.norm(vertices - support.point, axis=1)) <= tolerance * 0.01:
                return ProjectionResult(result.point, result.weights, result.distance, full_gap,
                                        False, feasibility, result.iterations + iteration)
            vertices = np.vstack((vertices, support.point))
        return ProjectionResult(result.point, result.weights, result.distance, full_gap,
                                False, feasibility, result.iterations + max_iter)
