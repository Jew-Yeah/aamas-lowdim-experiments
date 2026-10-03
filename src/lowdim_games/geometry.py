"""Small-dimensional convex geometry with numerical primal/dual certificates.

The certificates are for floating-point input data, up to the reported solver
tolerance. They do not assert exact rational or real-arithmetic computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


class OracleCertificationError(RuntimeError):
    """An oracle could not verify its requested numerical accuracy."""


@dataclass(frozen=True)
class ProjectionResult:
    point: np.ndarray
    weights: np.ndarray
    distance: float
    gap: float
    success: bool
    feasibility_error: float = 0.0
    iterations: int = 0

    @property
    def alpha(self) -> float:
        """Additive distance error certified by the squared-objective gap."""
        return float(np.sqrt(max(0.0, self.gap)))

    @property
    def lower_distance(self) -> float:
        return float(np.sqrt(max(0.0, self.distance**2 - self.gap)))


def _points(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or len(vertices) == 0 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be a nonempty finite N by d array")
    return vertices


def project_convex_hull(
    point: np.ndarray, vertices: np.ndarray, tol: float = 1e-9,
    initial_weights: np.ndarray | None = None, max_iter: int = 1500,
    reduce_hull: bool = True,
) -> ProjectionResult:
    """Project onto the complete supplied hull, and check a dual lower bound.

    ``gap`` is U-L for the *squared* distance objective. With
    y = point - projected, L = 2<y,point>-||y||²-2 max_v <y,v>.
    Thus sqrt(gap) is a certified additive error in hull distance. Success
    depends on feasibility and this gap, rather than SLSQP's status alone.
    Duplicate and interior points can be removed from the numerical solve;
    returned weights and the final certificate always use all original points.
    """
    vertices = _points(vertices)
    point = np.asarray(point, dtype=float)
    if point.shape != (vertices.shape[1],) or not np.all(np.isfinite(point)):
        raise ValueError("point has incompatible shape or contains nonfinite values")
    if tol <= 0:
        raise ValueError("tol must be positive")
    original_vertices = vertices
    original_count = len(vertices)
    scale = max(1.0, float(point @ point), float(np.max(np.sum(vertices**2, axis=1))))
    tolerance = tol * scale
    unique_vertices, original_indices, inverse = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True)
    unique_count = len(unique_vertices)
    retained = np.arange(unique_count)
    if reduce_hull and unique_count > 1:
        centered = unique_vertices - unique_vertices[0]
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(np.sum(singular > 1e-12 * max(1.0, float(singular[0]))))
        if rank == 0:
            retained = np.array([0])
        elif rank == 1:
            coordinates = centered @ vt[0]
            retained = np.unique([np.argmin(coordinates), np.argmax(coordinates)])
        elif rank <= 6 and unique_count > rank + 1:
            try:
                retained = ConvexHull(centered @ vt[:rank].T).vertices
            except QhullError:
                # A reduction is only an optimization, never a requirement.
                pass
    original_indices = original_indices[retained]
    vertices = unique_vertices[retained]
    count = len(vertices)
    if initial_weights is None:
        weights = np.zeros(count)
        weights[np.argmin(np.sum((vertices - point)**2, axis=1))] = 1.0
    else:
        weights = np.asarray(initial_weights, dtype=float).copy()
        if weights.shape != (original_count,) or not np.all(np.isfinite(weights)):
            raise ValueError("initial_weights have incompatible shape")
        weights = np.maximum(weights, 0.0)
        weights = np.bincount(inverse, weights=weights, minlength=unique_count)[retained]
        if weights.sum() == 0:
            weights[:] = 1.0 / count
        else:
            weights /= weights.sum()

    def certificate(w: np.ndarray, iterations: int) -> ProjectionResult:
        full_weights = np.zeros(original_count)
        full_weights[original_indices] = w
        projected = full_weights @ original_vertices
        direction = point - projected
        # Computing differences first avoids subtracting two nearly equal
        # support-function values when the query is already in the hull.
        gap = max(0.0, 2.0 * float(np.max((original_vertices - projected) @ direction)))
        feasibility = max(abs(float(full_weights.sum()) - 1.0),
                          max(0.0, -float(full_weights.min())))
        return ProjectionResult(projected, full_weights, float(np.linalg.norm(direction)), gap,
                                bool(gap <= tolerance and feasibility <= tol), feasibility, iterations)

    iterations = 0
    if count > 1 and np.all(point >= vertices.min(axis=0) - tol) and np.all(point <= vertices.max(axis=0) + tol):
        # Interior queries have a zero objective. Generic QP termination on
        # changes in that objective can stop with a much larger support gap.
        # An equality LP finds a sparse exact membership witness instead. Its
        # status is not trusted without the independent full-hull certificate.
        centered = vertices - vertices[0]
        membership = linprog(
            np.zeros(count), A_eq=np.vstack((centered.T, np.ones(count))),
            b_eq=np.r_[point - vertices[0], 1.0], bounds=[(0.0, None)] * count,
            method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                      "dual_feasibility_tolerance": 1e-10},
        )
        if membership.success:
            member_weights = np.maximum(membership.x, 0.0)
            member_weights /= member_weights.sum()
            member_certificate = certificate(member_weights, 1)
            if member_certificate.success:
                return member_certificate
            weights = member_weights
    if count > 1:
        def objective(w: np.ndarray) -> float:
            residual = w @ vertices - point
            return float(residual @ residual)

        def gradient(w: np.ndarray) -> np.ndarray:
            return 2.0 * (vertices @ (w @ vertices - point))

        result = minimize(
            objective, weights, jac=gradient, method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                         "jac": lambda w: np.ones(count)},
            options={"ftol": min(1e-12, tolerance * 0.001), "maxiter": max_iter},
        )
        if np.all(np.isfinite(result.x)):
            weights = np.maximum(result.x, 0.0)
            weights /= weights.sum()
        iterations = int(result.nit)

    # Wolfe-style active-face polishing solves affine least-squares systems
    # rather than taking thousands of tiny Frank-Wolfe steps near an interior
    # or face optimum. Minor cycles move to a simplex boundary whenever an
    # affine minimizer has a negative weight.
    weight_threshold = min(1e-13, tol * 0.001)
    for polishing in range(max_iter):
        projected = weights @ vertices
        dual_direction = point - projected
        values = (vertices - projected) @ dual_direction
        best = int(np.argmax(values))
        gap = max(0.0, 2.0 * float(values[best]))
        if gap <= tolerance:
            break
        active = np.unique(np.r_[np.flatnonzero(weights > weight_threshold), best])
        old_weights = weights.copy()
        for minor in range(count + 1):
            base = vertices[active[0]]
            if len(active) == 1:
                affine_weights = np.ones(1)
            else:
                coefficients = np.linalg.lstsq(
                    (vertices[active[1:]] - base).T, point - base, rcond=1e-13)[0]
                affine_weights = np.r_[1.0 - coefficients.sum(), coefficients]
            if affine_weights.min() >= -weight_threshold:
                affine_weights = np.maximum(affine_weights, 0.0)
                affine_weights /= affine_weights.sum()
                weights[:] = 0.0
                weights[active] = affine_weights
                break
            current = weights[active]
            negative = affine_weights < -weight_threshold
            fraction = float(np.min(current[negative] / (current[negative] - affine_weights[negative])))
            current = (1.0 - fraction) * current + fraction * affine_weights
            current[np.abs(current) <= weight_threshold] = 0.0
            current = np.maximum(current, 0.0)
            current /= current.sum()
            weights[:] = 0.0
            weights[active] = current
            active = active[current > weight_threshold]
        # Rank degeneracy can stall a minor cycle. A true line-search descent
        # remains a safe fallback; the same independent gap must still pass.
        if np.linalg.norm(weights - old_weights) <= np.finfo(float).eps:
            step_direction = vertices[best] - projected
            denominator = float(step_direction @ step_direction)
            if denominator <= np.finfo(float).tiny:
                break
            fraction = float(np.clip(dual_direction @ step_direction / denominator, 0.0, 1.0))
            weights *= 1.0 - fraction
            weights[best] += fraction
        iterations += 1
    result = certificate(weights, iterations)
    if not result.success and reduce_hull and count < unique_count:
        # An SVD-based reduction must not hide small genuine hull directions.
        return project_convex_hull(point, original_vertices, tol, result.weights,
                                   max_iter, reduce_hull=False)
    return result


@dataclass(frozen=True)
class AffinePolytope:
    origin: np.ndarray
    basis: np.ndarray
    coordinates: np.ndarray
    matrix: np.ndarray
    bound: np.ndarray
    residual: float

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def lift(self, coordinates: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(coordinates) @ self.basis.T


def affine_hull_polytope(points: np.ndarray, tol: float = 1e-11) -> AffinePolytope:
    """Represent an observed hull in orthonormal affine coordinates.

    No Qhull joggling is used: artificial positive-dimensional volumes would
    change response cells on exactly degenerate observed paths.
    """
    points = _points(points)
    origin = points.mean(axis=0)
    centered = points - origin
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    threshold = tol * max(1.0, float(singular[0]) if len(singular) else 0.0)
    dimension = int(np.sum(singular > threshold))
    basis = vt[:dimension].T
    coordinates = centered @ basis
    residual = float(np.max(np.linalg.norm(centered - coordinates @ basis.T, axis=1)))
    if dimension == 0:
        matrix, bound = np.empty((0, 0)), np.empty(0)
    elif dimension == 1:
        matrix = np.array([[1.0], [-1.0]])
        bound = np.array([coordinates[:, 0].max(), -coordinates[:, 0].min()])
    else:
        try:
            hull = ConvexHull(coordinates)
        except QhullError as error:
            raise OracleCertificationError("Observed affine hull could not be certified") from error
        matrix = hull.equations[:, :-1]
        bound = -hull.equations[:, -1]
        # Triangulated coplanar facets can repeat; exact duplicates suffice.
        rows = np.unique(np.column_stack((matrix, bound)), axis=0)
        matrix, bound = rows[:, :-1], rows[:, -1]
    return AffinePolytope(origin, basis, coordinates, matrix, bound, residual)


def _unique_rows(values: np.ndarray, tol: float) -> np.ndarray:
    if len(values) <= 1:
        return values
    retained: list[np.ndarray] = []
    for value in values:
        if not any(np.linalg.norm(value - other) <= tol for other in retained):
            retained.append(value)
    return np.asarray(retained)


def polytope_vertices(
    matrix: np.ndarray, bound: np.ndarray, tol: float = 1e-9,
    max_combinations: int = 200000,
) -> np.ndarray:
    """Vertices of a bounded halfspace intersection, including lower dimensions.

    A Chebyshev-center LP normally supplies a strict interior to Qhull. For
    lower-dimensional cells, LPs identify inequalities tight everywhere and
    recursively restrict to their common affine subspace.
    """
    matrix, bound = np.asarray(matrix, float), np.asarray(bound, float)
    if matrix.ndim != 2 or bound.shape != (matrix.shape[0],):
        raise ValueError("incompatible halfspace arrays")
    dimension = matrix.shape[1]
    if dimension == 0:
        return np.empty((1, 0)) if np.all(bound >= -tol) else np.empty((0, 0))
    row_norm = np.linalg.norm(matrix, axis=1)
    zero = row_norm == 0.0
    if np.any(bound[zero] < -tol):
        return np.empty((0, dimension))
    matrix, bound = matrix[~zero], bound[~zero]
    row_norm = row_norm[~zero]
    if not len(matrix):
        raise OracleCertificationError("The cell is unbounded")
    matrix, bound = matrix / row_norm[:, None], bound / row_norm
    center_result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=np.column_stack((matrix, np.ones(len(matrix)))), b_ub=bound,
        bounds=[(None, None)] * dimension + [(0.0, None)], method="highs",
    )
    if center_result.status == 2:
        return np.empty((0, dimension))
    if not center_result.success:
        raise OracleCertificationError("Cell interior LP failed: " + center_result.message)
    center = center_result.x[:-1]
    radius = float(center_result.x[-1])
    if dimension == 1:
        positive, negative = matrix[:, 0] > 0, matrix[:, 0] < 0
        high = float(np.min(bound[positive] / matrix[positive, 0])) if positive.any() else np.inf
        low = float(np.max(bound[negative] / matrix[negative, 0])) if negative.any() else -np.inf
        if not np.isfinite(low + high):
            raise OracleCertificationError("Unbounded one-dimensional cell")
        return _unique_rows(np.array([[low], [high]]), tol)
    if radius > tol:
        try:
            vertices = HalfspaceIntersection(np.column_stack((matrix, -bound)), center).intersections
            if np.max(matrix @ vertices.T - bound[:, None]) > 100 * tol:
                raise OracleCertificationError("Qhull returned infeasible cell vertices")
            return _unique_rows(vertices, tol)
        except QhullError:
            pass
    else:
        candidate_tight = np.flatnonzero(bound - matrix @ center <= 10 * tol)
        globally_tight = []
        for index in candidate_tight:
            result = linprog(matrix[index], A_ub=matrix, b_ub=bound,
                             bounds=[(None, None)] * dimension, method="highs")
            if not result.success:
                raise OracleCertificationError("Cell affine-hull LP failed")
            if bound[index] - result.fun <= tol:
                globally_tight.append(index)
        if globally_tight:
            tight = matrix[globally_tight]
            local_basis = null_space(tight, rcond=1e-11)
            if local_basis.shape[1] < dimension:
                # These rows define the affine hull, so they must be eliminated
                # from the reduced inequalities. Their nominally zero product
                # with the nullspace can be ~1e-17; normalizing that roundoff
                # would invent an order-one halfspace and destroy a real edge.
                correction = np.linalg.lstsq(
                    tight, bound[globally_tight] - tight @ center, rcond=1e-11)[0]
                affine_origin = center + correction
                if np.max(np.abs(tight @ affine_origin - bound[globally_tight])) > tol:
                    raise OracleCertificationError("Cell affine equalities could not be aligned")
                remaining = np.ones(len(matrix), dtype=bool)
                remaining[globally_tight] = False
                reduced = polytope_vertices(
                    matrix[remaining] @ local_basis,
                    bound[remaining] - matrix[remaining] @ affine_origin,
                    tol, max_combinations)
                lifted = affine_origin + reduced @ local_basis.T
                if len(lifted) and np.max(matrix @ lifted.T - bound[:, None]) > 10 * tol:
                    raise OracleCertificationError("Lifted cell vertices violate original inequalities")
                return lifted

    # A finite deterministic fallback, used only on genuinely small cells.
    import math
    count = math.comb(len(matrix), dimension)
    if count > max_combinations:
        raise OracleCertificationError("Cell vertex enumeration exceeded its explicit budget")
    vertices = []
    for active in combinations(range(len(matrix)), dimension):
        active_matrix = matrix[list(active)]
        if np.linalg.matrix_rank(active_matrix, tol=1e-11) < dimension:
            continue
        vertex = np.linalg.solve(active_matrix, bound[list(active)])
        if np.all(matrix @ vertex <= bound + 10 * tol):
            vertices.append(vertex)
    if not vertices:
        raise OracleCertificationError("No vertices found for a nonempty bounded cell")
    return _unique_rows(np.asarray(vertices), tol)
