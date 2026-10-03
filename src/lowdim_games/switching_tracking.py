"""A resource-share tracking game with a certified lag safe base.

The opponent action is the observed demand share, not a quantized profile.
For P=L=Delta_m, u(p,ell)=(p-ell)/sqrt(2) and p*(ell)=ell, every
response payoff is zero, so the full target S(Q) is exactly {0} for any Q.
The lag base satisfies sum_t u(p_t,ell_t)=(p_initial-ell_last)/sqrt(2),
whose norm is at most one. Its budget is known before observing a path.

This is a separate instantiation of the abstract one-switch master. Its lag
base and budget must not be described as the original block safe routine.
"""
from __future__ import annotations

import numpy as np

from .learners import (
    _FiniteLearner, FastHullLearner, OneSwitchLearner, SafeBlockLearner,
    paper_safe_budget,
)


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def tracking_tensor(m=3):
    """Return A[i,j]=(e_i-e_j)/sqrt(2), with vertex norms at most one."""
    m = _positive_integer(m, "m")
    basis = np.eye(m)
    return (basis[:, None, :] - basis[None, :, :]) / np.sqrt(2.0)


def tracking_response(ell):
    """The fixed, continuous benchmark response p*(ell)=ell."""
    return np.asarray(ell, dtype=float).copy()


def tracking_safe_budget(h):
    """The exact telescoping certificate B0(0)=0 and B0(h)=1 for h>=1."""
    if isinstance(h, bool) or int(h) != h or h < 0:
        raise ValueError("h must be a nonnegative integer.")
    return 0.0 if h == 0 else 1.0


class TrackingLagSafeLearner(_FiniteLearner):
    """Uniform first move, then the last observed demand share.

    The factory signature matches OneSwitchLearner. The supplied response
    must be the module's fixed identity benchmark; arbitrary response maps
    would change S(Q) and invalidate the stated certificate.
    """

    algorithm_name = "certified_tracking_lag_base"
    description = "Previous demand share, with an exact telescoping safe budget."

    def __init__(self, tensor, response, horizon):
        tensor = np.asarray(tensor, dtype=float)
        if (tensor.ndim != 3 or tensor.shape[0] < 1
                or tensor.shape != (tensor.shape[0],) * 3
                or not np.array_equal(tensor, tracking_tensor(tensor.shape[0]))):
            raise ValueError("The lag certificate requires the fixed tracking tensor.")
        if response is not tracking_response:
            raise ValueError("The lag certificate requires tracking_response as benchmark.")
        super().__init__(tensor, tracking_response, horizon)
        self.previous = None
        self.payoff_sum = np.zeros(self.d)

    @property
    def budget(self):
        return tracking_safe_budget(self.horizon)

    def choose(self):
        action = (np.full(self.K, 1.0 / self.K) if self.previous is None
                  else self.previous)
        return self._start(action)

    def observe(self, ell):
        p, z = self._observation(ell)
        payoff = self.payoff(p, z)
        self.previous = z.copy()
        self.payoff_sum += payoff
        return self._record({
            "t": self.round + 1, "mode": "safe", "p": p, "ell": z,
            "payoff": payoff, "h_t": None, "residual": 0.0,
            "cumulative_residual": 0.0, "switch": False,
            "saddle_gap": 0.0, "projection_gap": 0.0,
            "alpha_t": 0.0, "beta_t": 0.0, "direction": np.zeros(self.d),
            "target_witness": np.zeros(self.d), "response_point": z.copy(),
            "safe_prefix_sum_norm": float(np.linalg.norm(self.payoff_sum)),
            "safe_prefix_budget": tracking_safe_budget(self.round + 1),
        })


class _TrackingWindowLearner(_FiniteLearner):
    """Past-window demand mean; identical uniform initialization."""

    def __init__(self, tensor, response, horizon, window):
        super().__init__(tensor, response, horizon)
        self.window = _positive_integer(window, "window")
        self.history = []

    def choose(self):
        action = (np.mean(self.history[-self.window:], axis=0) if self.history
                  else np.full(self.K, 1.0 / self.K))
        return self._start(action)

    def observe(self, ell):
        p, z = self._observation(ell)
        self.history.append(z.copy())
        return self._record({
            "t": self.round + 1, "mode": "window", "p": p, "ell": z,
            "payoff": self.payoff(p, z), "switch": False,
        })


def _path_array(path):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or min(path.shape) < 1 or not np.isfinite(path).all():
        raise ValueError("path must be a finite nonempty T by m array.")
    if np.min(path) < -1e-12 or np.max(np.abs(path.sum(axis=1) - 1.0)) > 1e-10:
        raise ValueError("Every demand share must lie in the probability simplex.")
    # The same canonical simplex values reach every learner and the evaluator.
    path = np.maximum(path, 0.0)
    return path / path.sum(axis=1, keepdims=True)


def _diagnostics(records, key, *, missing=np.nan):
    return np.asarray([missing if r.get(key) is None else r[key] for r in records], dtype=float)


def run_tracking_path(path, window=16):
    """Run the original master/fast implementation and causal comparators.

    All methods face the same given chronological path and start uniformly.
    Evaluation uses the exact singleton target. The daily mismatch norms are
    additional descriptive metrics; they are different from the norm of the
    average vector payoff. Safe-mode residual zeros in original records are
    placeholders: ``fast_residual_defined`` identifies the monitored rounds.
    """
    path = _path_array(path)
    window = _positive_integer(window, "window")
    horizon, m = path.shape
    tensor = tracking_tensor(m)
    learners = {
        "one_switch": OneSwitchLearner(
            tensor, tracking_response, horizon,
            safe_factory=TrackingLagSafeLearner, safe_budget=tracking_safe_budget),
        "shared_past_hull": FastHullLearner(tensor, tracking_response, horizon),
        "lag_safe": TrackingLagSafeLearner(tensor, tracking_response, horizon),
        "last_window": _TrackingWindowLearner(tensor, tracking_response, horizon, window),
        "block_safe": SafeBlockLearner(tensor, tracking_response, horizon),
    }
    summary, trajectories = {}, {}
    rounds = np.arange(1, horizon + 1)
    for name, learner in learners.items():
        for ell in path:
            learner.choose()
            learner.observe(ell)
        records = learner.records
        actions = np.asarray([r["p"] for r in records])
        payoffs = np.asarray([r["payoff"] for r in records])
        payoff_sums = np.cumsum(payoffs, axis=0)
        prefix_averages = payoff_sums / rounds[:, None]
        distances = np.linalg.norm(prefix_averages, axis=1)
        mode = np.asarray([r["mode"] for r in records])
        defined = (mode == "fast") & (rounds >= 2)
        switch_round = getattr(learner, "switch_round", None)
        G = getattr(learner, "G_T", None)
        trajectory = {
            "actions": actions, "payoffs": payoffs, "payoff_sums": payoff_sums,
            "prefix_averages": prefix_averages, "distances": distances,
            "t": rounds.copy(), "mode": mode,
            "switch": np.asarray([bool(r.get("switch", False)) for r in records]),
            "E": _diagnostics(records, "cumulative_residual"),
            "residual": _diagnostics(records, "residual"),
            "h_t": _diagnostics(records, "h_t"),
            "G_T": _diagnostics(records, "G_T"),
            "safe_local_t": _diagnostics(records, "safe_local_t"),
            "fast_residual_defined": defined,
            "saddle_gap": _diagnostics(records, "saddle_gap"),
            "projection_gap": _diagnostics(records, "projection_gap"),
            "alpha_t": _diagnostics(records, "alpha_t"),
            "beta_t": _diagnostics(records, "beta_t"),
            "safe_prefix_sum_norm": _diagnostics(records, "safe_prefix_sum_norm"),
            "safe_prefix_budget": _diagnostics(records, "safe_prefix_budget"),
            "records": records,
        }
        if name == "lag_safe":
            # Standalone local time is known even though the master alone adds
            # safe_local_t to records copied from its freshly created tail.
            trajectory["safe_local_t"] = rounds.astype(float)
        trajectories[name] = trajectory
        summary[name] = {
            "horizon": horizon, "terminal_delta": float(distances[-1]),
            "terminal_sum_norm": float(np.linalg.norm(payoff_sums[-1])),
            "mean_instantaneous_norm": float(np.linalg.norm(payoffs, axis=1).mean()),
            "mean_l1_share_error": float(np.abs(actions - path).sum(axis=1).mean()),
            "switch_round": switch_round, "switch_threshold": G,
            "safe_round_count": int(np.sum(mode == "safe")),
            "last_fast_residual_sum": float(getattr(learner, "cumulative_residual", 0.0)),
            "sum_projection_alpha": float(np.nansum(trajectory["alpha_t"])),
            "sum_saddle_beta": float(np.nansum(trajectory["beta_t"])),
            "max_saddle_gap": float(np.nanmax(trajectory["saddle_gap"]))
                if np.isfinite(trajectory["saddle_gap"]).any() else None,
            "max_projection_gap": float(np.nanmax(trajectory["projection_gap"]))
                if np.isfinite(trajectory["projection_gap"]).any() else None,
        }
    return {
        "metadata": {
            "game": "resource-share tracking", "horizon": horizon, "m": m,
            "payoff": "(p-ell)/sqrt(2)", "response": "p_star(ell)=ell",
            "target": "S(Q)={0} for every nonempty Q; exact analytical evaluator",
            "realized_affine_dimension": int(np.linalg.matrix_rank(path - path[0], tol=1e-12)),
            "max_vertex_norm": float(np.max(np.linalg.norm(tensor, axis=2))),
            "window": window, "initialization": "uniform for every method",
            "same_opponent_path_across_methods": True,
            "learner_restarts": False,
            "master_safe_base": "tracking lag; B0(0)=0, B0(h)=1 for h>=1",
            "master_G_T": 1.0, "block_safe_budget": paper_safe_budget(horizon, m - 1),
            "numerical_oracles": "original checked floating-point fast hull and saddle solvers",
            "monitored_residual_contains_projection_error": True,
            "daily_mismatch_is_additional_metric": True,
        },
        "summary": summary, "trajectories": trajectories,
        "opponent_actions": path.copy(), "tensor": tensor,
    }


def controlled_tracking_path(seed=0, horizon=512):
    """Predeclared geometric expansion, with independent Dirichlet mixtures.

    At T=512: e1 on rounds 1--128; e2 at round 129; mixtures of the first
    two shares through round 256; e3 at round 257; all-three mixtures after
    that. Quarter boundaries scale with other announced horizons. This is
    an exogenous geometric mechanism test, with no claim of attacker learning.
    In exact arithmetic E first reaches the threshold 1 at round 129. With
    numerical projections, tiny subsequent residuals can cause a strict
    crossing before round 257; an early crossing must not be attributed to
    the later geometric expansion.
    """
    horizon = _positive_integer(horizon, "horizon")
    if horizon < 4:
        raise ValueError("The controlled path requires horizon >= 4.")
    rng = np.random.default_rng(seed)
    first, second = horizon // 4, horizon // 2
    path = np.zeros((horizon, 3))
    path[:first, 0] = 1.0
    path[first] = [0.0, 1.0, 0.0]
    if second > first + 1:
        path[first + 1:second, :2] = rng.dirichlet(np.ones(2), size=second - first - 1)
    path[second] = [0.0, 0.0, 1.0]
    if horizon > second + 1:
        path[second + 1:] = rng.dirichlet(np.ones(3), size=horizon - second - 1)
    return path
