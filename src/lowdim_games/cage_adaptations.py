"""Experimental scalar-aware saddle selection and block restarts.

The original manuscript learners remain unchanged. The scalar LP selects only
the primal mixture while keeping the exact saddle solver's dual mixture. Its
actual gap is checked independently; reported residuals are floating-point
evidence. Restarted blocks use the original safe budget for their actual local
horizons. Retained-history witnesses concern the global opponent hull.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .game import FiniteGame
from .learners import (
    FastHullLearner, OneSwitchLearner, SaddleResult, _FiniteLearner,
    _simplex_vector, solve_saddle_lp,
)


def _unique_history(history):
    """Keep exact distinct observations in their first-appearance order."""
    result, seen = [], set()
    for observation in history:
        point = np.asarray(observation, dtype=float)
        key = point.tobytes()
        if key not in seen:
            seen.add(key)
            result.append(point.copy())
    return result


class ScalarAwareFastHullLearner(FastHullLearner):
    """Past-hull learner with a causal secondary scalar-loss objective.

    At local round t>=2, retain the original saddle dual z*, and minimize the
    forecast scalar loss subject to max(p B)-min(B z*)<=rho/sqrt(t). Candidate
    gaps are independently recomputed. A rejected candidate falls back to the
    original numerical saddle pair. Initial action remains uniform.
    """

    algorithm_name = "experimental_scalar_aware_past_hull"

    def __init__(self, tensor, response, horizon, *, weights=None, window=16,
                 rho=1.0, initial_prior=None, oracle_tol=1e-9):
        super().__init__(tensor, response, horizon, oracle_tol=oracle_tol)
        if isinstance(window, bool) or int(window) != window or window < 1:
            raise ValueError("window must be a positive integer.")
        if not np.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("rho must lie in [0,1].")
        self.window, self.rho = int(window), float(rho)
        self.weights = FiniteGame(self.tensor, weights).weights.copy()
        self.scalar_costs = np.einsum("ajd,d->aj", self.tensor, self.weights)
        self.initial_prior = _simplex_vector(
            np.full(self.M, 1 / self.M) if initial_prior is None else initial_prior,
            self.M, "initial prior")
        self.chronological_history = []
        self._retained_seed = []
        self._oracle_details = {}

    def seed_retained_history(self, history):
        """Supply previously revealed actions; seed geometry after local round 1."""
        if self.round != 0 or self._pending is not None:
            raise RuntimeError("Retained history must be supplied before the first local choose().")
        points = [_simplex_vector(point, self.M, "retained opponent action") for point in history]
        self._retained_seed = _unique_history(points)
        self.chronological_history = [point.copy() for point in points]

    def _forecast(self):
        if not self.chronological_history:
            return self.initial_prior.copy()
        return np.mean(self.chronological_history[-self.window:], axis=0)

    def _solve_scalar_lp(self, matrix, lower_value, epsilon, forecast_costs):
        objective = forecast_costs - np.mean(forecast_costs)
        scale = float(np.max(np.abs(objective)))
        if scale > 0:
            objective = objective / scale
        return linprog(objective, A_ub=matrix.T,
                       b_ub=np.full(matrix.shape[1], lower_value + epsilon),
                       A_eq=np.ones((1, self.K)), b_eq=[1.0],
                       bounds=[(0.0, None)] * self.K, method="highs",
                       options={"primal_feasibility_tolerance": max(self.oracle_tol, 1e-10),
                                "dual_feasibility_tolerance": max(self.oracle_tol, 1e-10)})

    def choose(self):
        if self._pending is not None:
            return self._pending.copy()
        if self.round == 0:
            self._oracle_details = {"scalar_oracle_used": False,
                                    "scalar_oracle_fallback": False,
                                    "scalar_oracle_epsilon": 0.0,
                                    "scalar_oracle_actual_gap": 0.0}
            return self._start(np.full(self.K, 1 / self.K))
        vertices = np.asarray(self.history)
        action_payoffs = np.einsum("ajd,nj->and", self.tensor, vertices)
        matrix = np.einsum("and,d->an", action_payoffs, self.direction)
        original = solve_saddle_lp(matrix, self.oracle_tol)
        if not original.success:
            raise RuntimeError("The original saddle pair failed its numerical gap check.")
        lower = float(np.min(matrix @ original.opponent_weights))
        local_t = self.round + 1
        epsilon = self.rho / np.sqrt(local_t)
        forecast = self._forecast()
        costs = self.scalar_costs @ forecast
        result = self._solve_scalar_lp(matrix, lower, epsilon, costs)
        fallback = not result.success
        reason = "solver_failed" if fallback else None
        candidate_gap = None
        selected = original
        if result.success:
            try:
                p = _simplex_vector(result.x, self.K, "scalar LP action")
                candidate_gap = max(0.0, float(np.max(p @ matrix)) - lower)
                tolerance = max(20 * self.oracle_tol, 1e-8)
                if candidate_gap <= epsilon + tolerance:
                    selected = SaddleResult(p, original.opponent_weights.copy(), candidate_gap,
                                            (float(np.max(p @ matrix)) + lower) / 2, True)
                else:
                    fallback, reason = True, "candidate_gap_exceeds_allowance"
            except ValueError:
                fallback, reason = True, "candidate_is_not_a_simplex_action"
        self._saddle = selected
        self._response_point = selected.opponent_weights @ vertices
        self._oracle_details = {
            "scalar_oracle_used": not fallback,
            "scalar_oracle_fallback": fallback,
            "scalar_oracle_fallback_reason": reason,
            "scalar_oracle_epsilon": float(epsilon),
            "scalar_oracle_actual_gap": selected.gap,
            "scalar_oracle_candidate_gap": candidate_gap,
            "scalar_oracle_original_gap": original.gap,
            "scalar_oracle_gap_excess": max(0.0, selected.gap - epsilon),
            "scalar_oracle_contract_excess": max(0.0, selected.gap - local_t ** -0.5),
            "scalar_forecast": forecast.copy(),
            "scalar_forecast_selected_loss": float(costs @ selected.p),
            "scalar_forecast_original_loss": float(costs @ original.p),
            "scalar_oracle_window": self.window, "scalar_oracle_rho": self.rho,
        }
        return self._start(selected.p)

    def observe(self, ell):
        record = super().observe(ell)
        self.chronological_history.append(record["ell"].copy())
        if self.round == 1 and self._retained_seed:
            self.history = self._retained_seed + self.history
        self.history = _unique_history(self.history)
        record.update(self._oracle_details)
        record["geometry_history_size"] = len(self.history)
        record["forecast_history_size"] = len(self.chronological_history)
        return record


class ScalarAwareOneSwitchLearner(OneSwitchLearner):
    """Original master threshold and safe base with the scalar-aware fast oracle."""

    algorithm_name = "experimental_scalar_aware_one_switch"

    def __init__(self, tensor, response, horizon, *, weights=None, window=16,
                 rho=1.0, initial_prior=None, oracle_tol=1e-9):
        super().__init__(tensor, response, horizon, oracle_tol=oracle_tol)
        self.window, self.rho = window, rho
        self.fast = ScalarAwareFastHullLearner(
            tensor, response, horizon, weights=weights, window=window, rho=rho,
            initial_prior=initial_prior, oracle_tol=oracle_tol)

    def seed_retained_history(self, history):
        self.fast.seed_retained_history(history)


class BlockRestartLearner(_FiniteLearner):
    """Restart complete one-switch masters at predeclared block boundaries.

    Fresh blocks discard all history. Retained blocks keep every globally
    revealed action, including observations from earlier safe tails, and seed
    the new fast hull after its first uniform round. Local clocks, directions,
    residual sums, safe state and thresholds reset for each actual block length.
    These variants are distinct from the original uninterrupted master.
    """

    algorithm_name = "experimental_block_restart_one_switch"

    def __init__(self, tensor, response, horizon, *, block_length=1000,
                 retain_history=False, scalar_aware=False, weights=None,
                 window=16, rho=1.0, initial_prior=None, oracle_tol=1e-9):
        super().__init__(tensor, response, horizon)
        if isinstance(block_length, bool) or int(block_length) != block_length or block_length < 1:
            raise ValueError("block_length must be a positive integer.")
        self.block_length = int(block_length)
        self.retain_history, self.scalar_aware = bool(retain_history), bool(scalar_aware)
        self.parameters = {"block_length": self.block_length, "retain_history": self.retain_history,
                           "scalar_aware": self.scalar_aware, "window": window, "rho": rho}
        self._scalar_parameters = dict(weights=weights, window=window, rho=rho,
                                       initial_prior=initial_prior, oracle_tol=oracle_tol)
        # Validate scalar parameters eagerly rather than during a later block.
        if self.scalar_aware:
            ScalarAwareFastHullLearner(tensor, response, 1, **self._scalar_parameters)
        self.oracle_tol = oracle_tol
        self.global_history = []
        self.switch_rounds = []
        self.switch_round = None
        self.active_master = None
        self.block_index = -1
        self.block_horizon = 0
        self._block_start = 0
        self._retained_seed = []

    def _start_block(self):
        self.block_index += 1
        self._block_start = self.round
        self.block_horizon = min(self.block_length, self.horizon - self.round)
        if self.scalar_aware:
            self.active_master = ScalarAwareOneSwitchLearner(
                self.tensor, self.response, self.block_horizon, **self._scalar_parameters)
        else:
            self.active_master = OneSwitchLearner(
                self.tensor, self.response, self.block_horizon, oracle_tol=self.oracle_tol)
        self._retained_seed = [point.copy() for point in self.global_history] if self.retain_history else []
        if self.scalar_aware and self._retained_seed:
            self.active_master.seed_retained_history(self._retained_seed)

    def choose(self):
        if self._pending is not None:
            return self._pending.copy()
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self.active_master is None:
            self._start_block()
        return self._start(self.active_master.choose())

    def observe(self, ell):
        _, z = self._observation(ell)
        local_record = dict(self.active_master.observe(z))
        local_t = local_record["t"]
        # Standard FastHullLearner has no chronological forecast to initialize.
        if local_t == 1 and self._retained_seed and not self.scalar_aware:
            self.active_master.fast.history = _unique_history(
                self._retained_seed + self.active_master.fast.history)
        global_t = self.round + 1
        self.global_history.append(z.copy())
        if local_record["switch"]:
            self.switch_rounds.append(global_t)
            if self.switch_round is None:
                self.switch_round = global_t
        local_record.update({"t": global_t, "local_t": local_t,
                             "block_index": self.block_index,
                             "block_horizon": self.block_horizon,
                             "restart": local_t == 1 and self.block_index > 0,
                             "local_switch": bool(local_record["switch"]),
                             "global_switch": bool(local_record["switch"]),
                             "global_switch_round": global_t if local_record["switch"] else None,
                             "history_retained": self.retain_history,
                             "local_switch_round": self.active_master.switch_round,
                             "local_switch_threshold": self.active_master.G_T})
        record = self._record(local_record)
        if local_t == self.block_horizon:
            self.active_master = None
        return record
