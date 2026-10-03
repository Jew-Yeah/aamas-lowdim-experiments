"""An original-budget one-switch mechanism diagnostic, with exact fast oracles.

This is a constructed resource-balancing game, not CAGE or an empirical claim
about realistic attack complexity.  The opponent set is the *subprobability*
simplex ``conv{0,e_1,...,e_M}`` in Euclidean ``R^M``.  It is deliberately not
embedded into ``Delta_(M+1)``: that embedding changes hull projections.

Capacity ``p`` lies in [0,1], demand is ``r(ell)=sum(ell)``, and the normalized
scalar payoff is ``u(p,ell)=p-r(ell)``.  The prescribed mixed response is
``p_star(ell)=r(ell)``, so the full strict target is exactly {0}, including
responses at every unobserved mixture.  This is not a weighted argmin response.

New orthogonal types have identical aggregate demand.  Such payoff-equivalent
labels deliberately expose the fast method's sensitivity to the specified
opponent geometry.  The original block-safe update and original budget are
unchanged.  Standalone block-safe and a stronger certified lag control are
reported; the example establishes a benefit against fast-only, not against all
controls and not the low-dimensional convergence exponents.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .learners import SafeBlockLearner, paper_safe_budget


MODE_FAST, MODE_SAFE, MODE_BASELINE = 0, 1, 2
MODE_NAMES = {MODE_FAST: "fast", MODE_SAFE: "safe", MODE_BASELINE: "baseline"}

# Only the safe routine uses this lossless payoff compression.  Its tensor,
# response, block means, gradients, initial actions and steps are identical to
# those in the full resource game.  Fast geometry must retain the original labels.
_SAFE_TENSOR = np.array([[[0.0], [-1.0]], [[1.0], [0.0]]])


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def resource_response(compressed_opponent):
    """The exact balancing mixture, not the scalar minimizer of the tensor."""
    z = np.asarray(compressed_opponent, dtype=float)
    if (z.shape != (2,) or not np.isfinite(z).all() or np.min(z) < 0
            or abs(float(z.sum()) - 1.0) > 1e-10):
        raise ValueError("The compressed opponent must lie in Delta_2.")
    return np.array([z[0], z[1]])


def make_resource_safe(horizon):
    """Instantiate the unchanged original block-safe routine (k=1)."""
    return SafeBlockLearner(_SAFE_TENSOR, resource_response, horizon)


class ResourceBalanceFastLearner:
    """Exact specialization of the original fast policy in subprobability space.

    Label 0 denotes the zero vector and positive integer i denotes e_i.
    These compact labels avoid a quadratic dense storage of orthogonal vectors.
    The exact scalar saddle selects capacity 0 for positive direction and 1 for
    negative direction; at direction zero it uses the original uniform mixture.
    Hull projection has a closed form.  No future labels enter choose().
    """

    def __init__(self, horizon):
        self.horizon = _positive_integer(horizon, "horizon")
        self.round = 0
        self.direction = 0.0
        self.cumulative_residual = 0.0
        self._seen_nonzero = set()
        self._zero_seen = False
        self._demand_sum = 0
        self._pending = None
        self._saddle_demand = None

    def choose(self):
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self._pending is not None:
            return self._pending
        if self.round == 0 or self.direction == 0:
            capacity = 0.5
            self._saddle_demand = (self._demand_sum / self.round
                                   if self.round else None)
        elif self.direction > 0:
            capacity = 0.0
            self._saddle_demand = 0.0 if self._zero_seen else 1.0
        else:
            capacity = 1.0
            self._saddle_demand = 1.0 if self._seen_nonzero else 0.0
        self._pending = capacity
        return capacity

    def _projection(self, label):
        """Return projected demand, hull distance and exact projection kind."""
        if label == 0:
            if self._zero_seen:
                return 0.0, 0.0, "observed_zero"
            return 1.0, 1.0 / np.sqrt(len(self._seen_nonzero)), "uniform_seen_basis"
        if label in self._seen_nonzero:
            return 1.0, 0.0, "observed_basis"
        if self._zero_seen:
            return 0.0, 1.0, "zero_for_new_orthogonal_basis"
        return (1.0, np.sqrt(1.0 + 1.0 / len(self._seen_nonzero)),
                "uniform_seen_basis")

    def observe(self, label):
        if self._pending is None:
            raise RuntimeError("choose() must precede observe().")
        if isinstance(label, bool) or int(label) != label or label < 0:
            raise ValueError("label must be a nonnegative integer.")
        label = int(label)
        capacity = self._pending
        demand = float(label != 0)
        t = self.round + 1
        before = self.direction
        if t == 1:
            projected, distance, residual = demand, 0.0, 0.0
            saddle_demand, saddle_gap = demand, 0.0
            projection_kind = "first_round"
        else:
            projected, distance, projection_kind = self._projection(label)
            saddle_demand = self._saddle_demand
            # For either direction sign, the selected opponent demand is the
            # maximizing endpoint.  Check the actual primal/dual values.
            min_demand = 0.0 if self._zero_seen else 1.0
            max_demand = 1.0 if self._seen_nonzero else 0.0
            upper = max(before * (capacity - min_demand),
                        before * (capacity - max_demand))
            lower = min(before * (0.0 - saddle_demand),
                        before * (1.0 - saddle_demand))
            saddle_gap = max(0.0, upper - lower)
            if saddle_gap > 1e-12:
                raise RuntimeError("The analytic saddle failed its gap check.")
            # The response payoff at every saddle input is exactly zero.
            update = before + (capacity - projected) / (2.0 * np.sqrt(t))
            self.direction = update / max(1.0, abs(update))
            residual = abs(demand - projected)
            self.cumulative_residual += residual
        if label:
            self._seen_nonzero.add(label)
            self._demand_sum += 1
        else:
            self._zero_seen = True
        self.round = t
        self._pending = None
        return {
            "t": t, "label": label, "capacity": capacity, "demand": demand,
            "payoff": capacity - demand, "mode": MODE_FAST,
            "direction": before, "direction_after": self.direction,
            "projected_demand": projected, "response_point_demand": saddle_demand,
            "target_witness": 0.0, "h_t": distance, "residual": residual,
            "cumulative_residual": self.cumulative_residual,
            "saddle_gap": saddle_gap, "projection_gap": 0.0,
            "projection_kind": projection_kind, "switch": False,
            "safe_local_t": 0, "safe_block_index": -2,
        }


def _allocate_logs(T, labels, demand):
    logs = {
        "t": np.arange(1, T + 1, dtype=np.int64),
        "labels": labels.copy(), "demand": demand.copy(),
        "capacity": np.empty(T), "payoff": np.empty(T),
        "mode": np.full(T, MODE_BASELINE, dtype=np.int8),
        "switch": np.zeros(T, dtype=bool),
        "safe_local_t": np.zeros(T, dtype=np.int64),
        "safe_block_index": np.full(T, -2, dtype=np.int64),
    }
    for key in ("h_t", "direction", "direction_after", "projected_demand",
                "response_point_demand", "target_witness", "saddle_gap",
                "projection_gap"):
        logs[key] = np.full(T, np.nan)
    for key in ("residual", "cumulative_residual"):
        logs[key] = np.zeros(T)
    return logs


def _store(logs, index, record):
    for key, values in logs.items():
        if key in record:
            values[index] = record[key]


def _safe_record(safe, compressed, global_t, cumulative_residual=0.0):
    p = safe.choose()  # Must precede that round's observation.
    record = safe.observe(compressed)
    return {
        "t": global_t, "capacity": float(p[1]),
        "payoff": float(record["payoff"][0]), "mode": MODE_SAFE,
        "direction": float(record["direction"][0]),
        "direction_after": float(safe.direction[0]),
        "cumulative_residual": cumulative_residual,
        "safe_local_t": record["t"], "safe_block_index": record["block_index"],
        "target_witness": (0.0 if record["target_witness"] is not None else np.nan),
        "response_point_demand": (float(record["response_point"][1])
                                   if record["response_point"] is not None else np.nan),
    }


def _finish(logs, threshold=None, switch_round=None):
    logs["cumulative_payoff"] = np.cumsum(logs["payoff"])
    logs["average_payoff"] = logs["cumulative_payoff"] / logs["t"]
    # This uses the entire strict response target, analytically equal to {0}.
    logs["distance"] = np.abs(logs["average_payoff"])
    logs["absolute_error"] = np.abs(logs["payoff"])
    logs["mean_absolute_error"] = np.cumsum(logs["absolute_error"]) / logs["t"]
    logs["full_target_projection"] = np.zeros(len(logs["t"]))
    if threshold is not None:
        logs["threshold"] = np.full(len(logs["t"]), threshold)
    return {
        "final_distance": float(logs["distance"][-1]),
        "signed_cumulative_payoff": float(logs["cumulative_payoff"][-1]),
        "mean_absolute_error": float(logs["mean_absolute_error"][-1]),
        "mean_capacity": float(logs["capacity"].mean()),
        "final_cumulative_residual": float(logs["cumulative_residual"][-1]),
        "switch_round": switch_round,
        "safe_rounds": int(np.count_nonzero(logs["mode"] == MODE_SAFE)),
        "max_saddle_gap": (float(np.nanmax(logs["saddle_gap"]))
                           if np.any(np.isfinite(logs["saddle_gap"])) else None),
        "max_projection_gap": (float(np.nanmax(logs["projection_gap"]))
                               if np.any(np.isfinite(logs["projection_gap"])) else None),
    }


def run_default_budget_stress(T=16384, quiet_rounds=128, window=16):
    """Run the common causal path with original master budget and five controls.

    Return ``metadata``, ``summary`` and ``trajectories``.  Each trajectory is a
    dictionary of compact one-dimensional NumPy arrays, suitable for NPZ and
    plotting.  Summary and metadata contain only JSON-serializable values.
    Labels 1,...,T-quiet_rounds are distinct orthogonal opponent vectors, not
    additional demand levels.  The horizon is announced before play.
    """
    T = _positive_integer(T, "T")
    quiet_rounds = _positive_integer(quiet_rounds, "quiet_rounds")
    window = _positive_integer(window, "window")
    if not 2 <= quiet_rounds < T:
        raise ValueError("Require 2 <= quiet_rounds < T.")
    labels = np.r_[np.zeros(quiet_rounds, dtype=np.int64),
                   np.arange(1, T - quiet_rounds + 1, dtype=np.int64)]
    demand = (labels != 0).astype(float)
    methods = ("one_switch", "shared_past_hull", "block_safe", "lag_safe", "last_window")
    trajectories = {name: _allocate_logs(T, labels, demand) for name in methods}
    G = paper_safe_budget(T, 1)
    master_fast, fast = ResourceBalanceFastLearner(T), ResourceBalanceFastLearner(T)
    safe_all = make_resource_safe(T)
    safe_tail = None
    switch_round = None
    frozen_residual = 0.0
    past_demand = None
    recent = deque(maxlen=window)
    for index, label in enumerate(labels):
        t = index + 1
        # All choices depend on prior state only.  No learner receives an
        # unrevealed label or its aggregate demand to select the current action.
        fast.choose()
        if safe_tail is None:
            master_fast.choose()
        lag_capacity = 0.5 if past_demand is None else past_demand
        window_capacity = 0.5 if not recent else float(np.mean(recent))
        # The original safe choose/observe protocol is enforced in _safe_record.
        compressed = np.array([1.0 - demand[index], demand[index]])
        _store(trajectories["shared_past_hull"], index, fast.observe(label))
        if safe_tail is None:
            record = master_fast.observe(label)
            frozen_residual = master_fast.cumulative_residual
            if frozen_residual > G:
                switch_round = t
                record["switch"] = True
                # Keep the threshold-crossing action in the fast prefix.
                if t < T:
                    safe_tail = make_resource_safe(T - t)
            _store(trajectories["one_switch"], index, record)
        else:
            _store(trajectories["one_switch"], index,
                   _safe_record(safe_tail, compressed, t, frozen_residual))
        _store(trajectories["block_safe"], index, _safe_record(safe_all, compressed, t))
        for name, capacity in (("lag_safe", lag_capacity), ("last_window", window_capacity)):
            _store(trajectories[name], index,
                   {"capacity": capacity, "payoff": capacity - demand[index]})
        past_demand = float(demand[index])
        recent.append(past_demand)
    summary = {name: _finish(logs, G if name == "one_switch" else None,
                             switch_round if name == "one_switch" else None)
               for name, logs in trajectories.items()}
    if switch_round is not None and switch_round < T:
        prefix_sum = float(trajectories["one_switch"]["payoff"][:switch_round].sum())
        tail_sum = float(trajectories["one_switch"]["payoff"][switch_round:].sum())
        tail_horizon = T - switch_round
        summary["one_switch"].update({
            "fast_prefix_cumulative_payoff": prefix_sum,
            "safe_tail_cumulative_payoff": tail_sum,
            "safe_tail_horizon": tail_horizon,
            "safe_tail_budget": paper_safe_budget(tail_horizon, 1),
            "prefix_plus_safe_bound": (abs(prefix_sum) + paper_safe_budget(tail_horizon, 1)) / T,
        })
    return {
        "metadata": {
            "benchmark": "resource_balance_original_budget_mechanism_stress",
            "T": T, "quiet_rounds": quiet_rounds, "window": window,
            "learner_affine_dimension": 1, "switch_threshold": G,
            "safe_budget": "B0(h)=6*h^(3/4)",
            "safe_implementation": "unchanged SafeBlockLearner",
            "opponent_geometry": "conv{0,e_1,...,e_M} in Euclidean R^M",
            "opponent_dimension": T - quiet_rounds,
            "realized_affine_dimension": T - quiet_rounds,
            "payoff": "p-sum(ell)", "response": "p_star(ell)=sum(ell)",
            "vertex_payoff_norm_bound": 1.0,
            "full_strict_target": "{0}, for all mixtures in the full hull",
            "fast_oracles": "exact closed-form subprobability-space saddle and hull projection",
            "safe_compression": "[1-r,r], lossless for safe payoffs and updates only",
            "same_opponent_path_across_methods": True,
            "mode_names": MODE_NAMES,
            "lag_safe_certificate": "B0(0)=0; B0(h)=1 for h>=1: sum(p_t-r_t)=p_1-r_h",
            "limitations": [
                "Constructed diagnostic; no real CAGE data or empirical attack-learning claim.",
                "Novel labels have payoff-equivalent unit demands; opponent geometry determines fast residuals.",
                "High realized dimension; no low-dimensional rate or q=4 validation.",
                "One-switch improvement is against fast-only; standalone safe and lag can perform better.",
                "Distance measures signed average imbalance; mean absolute error is reported separately.",
                "Floating-point implementation of exact closed-form oracles and the original safe routine.",
            ],
        },
        "summary": summary,
        "trajectories": trajectories,
    }
