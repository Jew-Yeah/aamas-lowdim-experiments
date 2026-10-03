"""Causal learners from the paper, in finite simplex games.

``choose()`` must precede ``observe(ell)`` on every round.  The first method
uses only past opponent actions.  The fast past-hull policy is shared with
the geometric approach of Marinov et al.; it is not a distinct competitor
under a different label.  The safe routine implements the manuscript's
explicit block routine, which instantiates the epoch framework underlying
Marinov et al. Theorem 14.  The exponential-cover Theorem 21 algorithm is
not implemented here.

LP/QP outputs have checked floating-point residuals.  These diagnostics
are numerical estimates, not formal exact-arithmetic oracle certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linprog

from .geometry import project_convex_hull


Response = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SaddleResult:
    p: np.ndarray
    opponent_weights: np.ndarray
    gap: float
    value: float
    success: bool


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1 or len(v) == 0 or not np.isfinite(v).all():
        raise ValueError("A nonempty finite one-dimensional vector is required.")
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    active = np.flatnonzero(ordered - cumulative / np.arange(1, len(v) + 1) > 0)
    theta = cumulative[active[-1]] / (active[-1] + 1)
    projected = np.maximum(v - theta, 0.0)
    return projected / projected.sum()


def _ball_projection(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(1.0, norm)


def _simplex_vector(vector: np.ndarray, length: int, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (length,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must have shape ({length},) and finite entries.")
    if np.min(v) < -1e-8 or abs(float(v.sum()) - 1.0) > 1e-8:
        raise ValueError(f"{name} must lie in the probability simplex.")
    v = np.maximum(v, 0.0)
    return v / v.sum()


def solve_saddle_lp(matrix: np.ndarray, tol: float = 1e-9) -> SaddleResult:
    """Minimizing row mixture and maximizing column mixture of a matrix game.

    The returned gap is ``max(p @ B) - min(B @ z)``, independently
    recomputed from the returned feasible mixtures rather than an LP status.
    """
    b = np.asarray(matrix, dtype=float)
    if b.ndim != 2 or min(b.shape) == 0 or not np.isfinite(b).all():
        raise ValueError("The game matrix must be nonempty and finite.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    k, n = b.shape
    if np.ptp(b) <= np.finfo(float).eps:
        p, z = np.full(k, 1.0 / k), np.full(n, 1.0 / n)
        return SaddleResult(p, z, 0.0, float(b.mean()), True)
    options = {
        "dual_feasibility_tolerance": max(tol, 1e-10),
        "primal_feasibility_tolerance": max(tol, 1e-10),
    }
    row = linprog(
        np.r_[np.zeros(k), 1.0],
        A_ub=np.c_[b.T, -np.ones(n)], b_ub=np.zeros(n),
        A_eq=np.r_[np.ones(k), 0.0][None, :], b_eq=[1.0],
        bounds=[(0, None)] * k + [(None, None)], method="highs", options=options,
    )
    column = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[-b, np.ones(k)], b_ub=np.zeros(k),
        A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[1.0],
        bounds=[(0, None)] * n + [(None, None)], method="highs", options=options,
    )
    if not row.success or not column.success:
        raise RuntimeError(f"Saddle LP failed: row={row.message}; column={column.message}")
    p = _simplex_vector(row.x[:k], k, "LP row mixture")
    z = _simplex_vector(column.x[:n], n, "LP column mixture")
    upper, lower = float(np.max(p @ b)), float(np.min(b @ z))
    gap = max(0.0, upper - lower)
    success = gap <= max(20 * tol, 1e-8)
    return SaddleResult(p, z, gap, (upper + lower) / 2, success)


def paper_safe_budget(horizon: int, learner_dimension: int) -> float:
    """The supplied manuscript bound B_0(h)=6 sqrt(k) h^(3/4)."""
    if horizon < 0 or learner_dimension < 0:
        raise ValueError("Horizon and learner dimension must be nonnegative.")
    return float(6 * np.sqrt(learner_dimension) * horizon**0.75)


class _FiniteLearner:
    def __init__(self, tensor: np.ndarray, response: Response, horizon: int):
        a = np.asarray(tensor, dtype=float)
        if a.ndim != 3 or min(a.shape) == 0 or not np.isfinite(a).all():
            raise ValueError("tensor must be a finite array with shape [K,M,d].")
        if float(np.max(np.linalg.norm(a, axis=2))) > 1.0 + 1e-8:
            raise ValueError("Payoff tensor must be normalized to vertex norms at most one.")
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        self.tensor = a.copy()
        self.response = response
        self.horizon = int(horizon)
        self.K, self.M, self.d = a.shape
        self.k = self.K - 1
        self.round = 0
        self.records: list[dict] = []
        self._pending: np.ndarray | None = None

    def payoff(self, p: np.ndarray, ell: np.ndarray) -> np.ndarray:
        return np.einsum("a,ajd,j->d", p, self.tensor, ell)

    def _response(self, ell: np.ndarray) -> np.ndarray:
        return _simplex_vector(self.response(ell.copy()), self.K, "response output")

    def _start(self, p: np.ndarray) -> np.ndarray:
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self._pending is None:
            self._pending = _simplex_vector(p, self.K, "learner action")
        return self._pending.copy()

    def _observation(self, ell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is None:
            raise RuntimeError("choose() must be called before observe().")
        z = _simplex_vector(ell, self.M, "opponent action")
        return self._pending.copy(), z

    def _record(self, record: dict) -> dict:
        self.round += 1
        self._pending = None
        self.records.append(record)
        return record


class FastHullLearner(_FiniteLearner):
    """One uninterrupted run of the manuscript's fast past-hull policy."""

    algorithm_name = "shared_past_hull"
    description = "Shared causal past-hull approachability policy; not a distinct prior-paper competitor."

    def __init__(self, tensor: np.ndarray, response: Response, horizon: int,
                 *, oracle_tol: float = 1e-9):
        super().__init__(tensor, response, horizon)
        if oracle_tol <= 0:
            raise ValueError("oracle_tol must be positive.")
        self.oracle_tol = oracle_tol
        self.direction = np.zeros(self.d)  # lambda_2=0 after the first round.
        self.history: list[np.ndarray] = []
        self.cumulative_residual = 0.0
        self._saddle: SaddleResult | None = None
        self._response_point: np.ndarray | None = None

    def choose(self) -> np.ndarray:
        if self._pending is not None:
            return self._pending.copy()
        if self.round == 0:
            return self._start(np.full(self.K, 1.0 / self.K))
        vertices = np.asarray(self.history)
        action_payoffs = np.einsum("ajd,nj->and", self.tensor, vertices)
        matrix = np.einsum("and,d->an", action_payoffs, self.direction)
        self._saddle = solve_saddle_lp(matrix, self.oracle_tol)
        if not self._saddle.success:
            raise RuntimeError(f"Saddle gap {self._saddle.gap:g} exceeds the numerical tolerance.")
        self._response_point = self._saddle.opponent_weights @ vertices
        return self._start(self._saddle.p)

    def observe(self, ell: np.ndarray) -> dict:
        p, z = self._observation(ell)
        t = self.round + 1
        actual = self.payoff(p, z)
        direction_before = self.direction.copy()
        if t == 1:
            target = self.payoff(self._response(z), z)
            h_t = projection_gap = alpha = saddle_gap = residual = beta = 0.0
            response_point = z.copy()
            projected = z.copy()
        else:
            assert self._response_point is not None and self._saddle is not None
            projection = project_convex_hull(z, np.asarray(self.history), tol=self.oracle_tol)
            if not projection.success:
                raise RuntimeError("Past-hull projection solver failed its numerical checks.")
            projected = projection.point
            h_t, projection_gap = float(projection.distance), float(projection.gap)
            alpha = float(np.sqrt(max(projection_gap, 0.0)))
            response_point = self._response_point.copy()
            target = self.payoff(self._response(response_point), response_point)
            a_t = self.payoff(p, projected) - target
            residual = float(np.linalg.norm(actual - self.payoff(p, projected)))
            self.cumulative_residual += residual
            self.direction = _ball_projection(self.direction + a_t / (2 * np.sqrt(t)))
            saddle_gap = self._saddle.gap
            beta = max(0.0, saddle_gap - t**-0.5)
        self.history.append(z.copy())
        return self._record({
            "t": t, "mode": "fast", "p": p, "ell": z,
            "payoff": actual, "h_t": h_t, "residual": residual,
            "cumulative_residual": self.cumulative_residual,
            "switch": False, "saddle_gap": saddle_gap,
            "projection_gap": projection_gap, "alpha_t": alpha, "beta_t": beta,
            "direction": direction_before, "target_witness": target,
            "response_point": response_point, "projected_ell": projected,
        })


class SafeBlockLearner(_FiniteLearner):
    """Fresh h-round implementation of Proposition 'Block routine safety'.

    A regular K-simplex centered at its uniform mixture has inradius
    1/sqrt(K(K-1)).  Scaling its affine coordinates by sqrt(K(K-1))
    gives the manuscript's B_k(1) subset P subset B_k(k) normalization.
    Therefore its normalized-coordinate OGD step k/sqrt(n_h) becomes
    exactly 1/(K sqrt(n_h)) in probability coordinates.
    """

    algorithm_name = "paper_block_safe"
    description = "The manuscript's explicit safe block routine; no separate prior unrestricted algorithm is claimed."

    def __init__(self, tensor: np.ndarray, response: Response, horizon: int):
        super().__init__(tensor, response, horizon)
        self.m_h = max(1, int(np.floor(np.sqrt(self.horizon) / self.k))) if self.k else 1
        self.n_h = self.horizon // self.m_h
        self.r_h = self.horizon - self.m_h * self.n_h
        self.direction = np.zeros(self.d)
        self._inner_p = np.full(self.K, 1.0 / self.K)
        self._block_ell = np.zeros(self.M)
        self._block_payoff = np.zeros(self.d)
        self.budget = paper_safe_budget(self.horizon, self.k)

    def choose(self) -> np.ndarray:
        p = self._inner_p if self.round < self.m_h * self.n_h else np.full(self.K, 1.0 / self.K)
        return self._start(p)

    def observe(self, ell: np.ndarray) -> dict:
        p, z = self._observation(ell)
        t = self.round + 1
        actual = self.payoff(p, z)
        direction_before = self.direction.copy()
        in_block = t <= self.m_h * self.n_h
        block_index = (t - 1) // self.n_h if in_block else -1
        target = response_point = None
        if in_block:
            self._block_ell += z
            self._block_payoff += actual
            gradient = np.einsum("ajd,j,d->a", self.tensor, z, self.direction)
            self._inner_p = project_simplex(p - gradient / (self.K * np.sqrt(self.n_h)))
            if t % self.n_h == 0:
                response_point = self._block_ell / self.n_h
                target = self.payoff(self._response(response_point), response_point)
                v_e = self._block_payoff / self.n_h - target
                self.direction = _ball_projection(self.direction + v_e / (2 * np.sqrt(self.m_h)))
                self._block_ell.fill(0.0)
                self._block_payoff.fill(0.0)
                self._inner_p = np.full(self.K, 1.0 / self.K)
        else:
            response_point = z.copy()
            target = self.payoff(self._response(z), z)
        return self._record({
            "t": t, "mode": "safe", "p": p, "ell": z,
            "payoff": actual, "h_t": None, "residual": 0.0,
            "cumulative_residual": 0.0, "switch": False,
            "saddle_gap": 0.0, "projection_gap": 0.0,
            "alpha_t": 0.0, "beta_t": 0.0,
            "direction": direction_before, "block_index": block_index,
            "target_witness": target, "response_point": response_point,
        })


class OneSwitchLearner(_FiniteLearner):
    """Fast prefix plus a fresh safe tail with the paper's known-T threshold."""

    algorithm_name = "paper_one_switch"
    description = "The manuscript's one-switch master, using its explicit known-horizon safe budget by default."

    def __init__(self, tensor: np.ndarray, response: Response, horizon: int,
                 *, oracle_tol: float = 1e-9,
                 safe_factory: Callable | None = None,
                 safe_budget: Callable[[int], float] | None = None):
        """Optionally supply a legitimate abstract safe base and its B_0(h).

        Both custom arguments are required together.  The caller is responsible
        for the safe-base guarantee; a smaller guessed threshold is not valid.
        Experiments should use the default manuscript base and budget.
        """
        super().__init__(tensor, response, horizon)
        if (safe_factory is None) != (safe_budget is None):
            raise ValueError("A custom safe base requires both its factory and its guaranteed budget.")
        self._safe_factory = safe_factory or SafeBlockLearner
        if safe_budget is None:
            self.G_T = paper_safe_budget(self.horizon, self.k)
        else:
            budgets = np.asarray([safe_budget(h) for h in range(self.horizon + 1)], dtype=float)
            if not np.isfinite(budgets).all() or np.min(budgets) < 0 or budgets[0] != 0:
                raise ValueError("B_0 must be finite, nonnegative, and satisfy B_0(0)=0.")
            self.G_T = float(np.max(budgets))
        self.fast = FastHullLearner(tensor, response, horizon, oracle_tol=oracle_tol)
        self.safe: SafeBlockLearner | None = None
        self.switch_round: int | None = None
        self.cumulative_residual = 0.0

    def choose(self) -> np.ndarray:
        base = self.safe if self.safe is not None else self.fast
        return self._start(base.choose())

    def observe(self, ell: np.ndarray) -> dict:
        _, z = self._observation(ell)
        t = self.round + 1
        if self.safe is None:
            record = dict(self.fast.observe(z))
            self.cumulative_residual = self.fast.cumulative_residual
            if self.cumulative_residual > self.G_T:
                self.switch_round = t
                record["switch"] = True
                # The threshold-crossing round remains in the fast prefix.
                if t < self.horizon:
                    self.safe = self._safe_factory(self.tensor, self.response, self.horizon - t)
        else:
            record = dict(self.safe.observe(z))
            record["safe_local_t"] = record["t"]
        record["t"] = t
        record["cumulative_residual"] = self.cumulative_residual
        record["G_T"] = self.G_T
        return self._record(record)


class LagResponseLearner(_FiniteLearner):
    """A transparent lag-response heuristic, with no paper safety guarantee."""

    algorithm_name = "lag_response_heuristic"
    description = "A one-step lag response heuristic, without an asserted approachability certificate."

    def __init__(self, tensor: np.ndarray, response: Response, horizon: int):
        super().__init__(tensor, response, horizon)
        self._last_ell: np.ndarray | None = None

    def choose(self) -> np.ndarray:
        p = np.full(self.K, 1.0 / self.K) if self._last_ell is None else self._response(self._last_ell)
        return self._start(p)

    def observe(self, ell: np.ndarray) -> dict:
        p, z = self._observation(ell)
        self._last_ell = z.copy()
        return self._record({
            "t": self.round + 1, "mode": "heuristic", "p": p, "ell": z,
            "payoff": self.payoff(p, z), "h_t": None, "residual": 0.0,
            "cumulative_residual": 0.0, "switch": False,
            "saddle_gap": 0.0, "projection_gap": 0.0,
            "alpha_t": 0.0, "beta_t": 0.0,
        })
