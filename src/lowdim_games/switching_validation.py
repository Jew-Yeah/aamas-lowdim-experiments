"""Independent stochastic paths for the original one-switch mechanism.

This is a constructed weighted capacity-balancing game, not a real-data
benchmark. Opponent geometry is ``conv{0,e_1,...,e_M}`` in Euclidean R^M.
At origin, (a,b)=(1,0); type i has a_i in [a_min,1] and b_i=a_i*r_i.
For mixtures the parameters are affine, and ``u(p,ell)=a(ell)*p-b(ell)``.
The fixed response b/a makes the *entire* strict response target {0}.

Unlike the older stress diagnostic, fresh types have different payoff maps.
Unlike unweighted tracking, responding to the previous ratio does not have
an asserted telescoping safety certificate. The original k=1 block learner
and budget 6*T**(3/4) are unchanged. Only its payoff computation is compressed
to four corners; fast hull geometry retains the original orthogonal types.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np

from .learners import SafeBlockLearner, paper_safe_budget


METHODS = ("one_switch", "shared_past_hull", "block_safe", "lag_response", "last_window")


@dataclass(frozen=True)
class ValidationConfig:
    T: int = 65536
    change_round: int = 16384  # This many prefix rounds; changed phase starts +1.
    a_min: float = 0.5
    post_a_low: float = 0.5
    post_a_high: float = 1.0
    post_r_low: float = 0.85
    post_r_high: float = 1.0
    pre_r_low: float = 0.02
    pre_r_high: float = 0.15
    window: int = 16
    scenario_id: int = 0
    base_entropy: int = 20261004

    def __post_init__(self):
        for name in ("T", "change_round", "window"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self,name,int(value))
        if not 2 <= self.change_round <= self.T:
            raise ValueError("Require 2 <= change_round <= T.")
        for name in ("scenario_id", "base_entropy"):
            value = getattr(self,name)
            if isinstance(value,bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
            object.__setattr__(self,name,int(value))
        if not 0 < self.a_min < 1:
            raise ValueError("Require 0 < a_min < 1.")
        if not self.a_min <= self.post_a_low <= self.post_a_high <= 1:
            raise ValueError("Post coefficients must lie in [a_min,1].")
        if not 0 <= self.pre_r_low <= self.pre_r_high <= 1:
            raise ValueError("Prefix ratios must lie in [0,1].")
        if not 0 <= self.post_r_low <= self.post_r_high <= 1:
            raise ValueError("Post ratios must lie in [0,1].")


def capacity_tensor(a_min=0.5):
    """Two capacity actions and four (a,b) corner payoff maps."""
    if not 0 < a_min < 1:
        raise ValueError("Require 0 < a_min < 1.")
    a = np.array([a_min, 1.0, a_min, 1.0])
    b = np.array([0.0, 0.0, a_min, 1.0])
    return np.stack([-b, a-b], axis=0)[..., None]


def capacity_response(ell, a_min=0.5):
    z = np.asarray(ell, dtype=float)
    if (z.shape != (4,) or not np.isfinite(z).all() or np.min(z) < -1e-12
            or abs(float(z.sum()) - 1.0) > 1e-10):
        raise ValueError("The compressed opponent must be a probability vector of length four.")
    a = float(z @ np.array([a_min, 1.0, a_min, 1.0]))
    b = float(z @ np.array([0.0, 0.0, a_min, 1.0]))
    return np.array([1-b/a, b/a])


def compress_capacity(a, b, a_min=0.5):
    """Losslessly preserve a,b for safe updates; never use for fast geometry."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < a_min-1e-12) or np.any(a > 1+1e-12)
            or np.any(b < -1e-12) or np.any(b > a+1e-12)):
        raise ValueError("Require a_min <= a <= 1 and 0 <= b <= a.")
    high = np.clip((a-a_min)/(1-a_min), 0, 1)
    ratio = np.clip(b/a, 0, 1)
    return np.stack([(1-high)*(1-ratio), high*(1-ratio),
                     (1-high)*ratio, high*ratio], axis=-1)


def make_capacity_safe(horizon, a_min=0.5):
    return SafeBlockLearner(capacity_tensor(a_min),
                            lambda ell: capacity_response(ell, a_min), horizon)


class HeterogeneousCapacityFastLearner:
    """Exact causal past-hull specialization for origin and unit vertices.

    The vertices are geometrical labels, not payoff coordinates. The projection
    of an unseen e_i is origin if origin was observed, otherwise the uniform
    mean of distinct observed unit vertices. Repetition does not alter that
    projection. The saddle maximizes lambda*(a*p-b) over observed vertices.
    """
    def __init__(self, horizon):
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        self.horizon = int(horizon)
        self.round = 0
        self.direction = 0.0
        self.cumulative_residual = 0.0
        self.vertices = {}
        self.counts = {}
        self._pending = None
        self._saddle_ab = None

    def choose(self):
        if self._pending is not None:
            return self._pending
        if self.round >= self.horizon:
            raise RuntimeError("The announced horizon has been exhausted.")
        if self.round == 0 or self.direction == 0:
            p = 0.5
            if self.vertices:
                total = sum(self.counts.values())
                self._saddle_ab = tuple(sum(self.counts[k]*v[j]
                                            for k, v in self.vertices.items())/total
                                        for j in (0, 1))
        elif self.direction > 0:
            p = 0.0
            self._saddle_ab = min(self.vertices.values(), key=lambda v: v[1])
        else:
            p = 1.0
            self._saddle_ab = min(self.vertices.values(), key=lambda v: v[0]-v[1])
        self._pending = p
        return p

    def _project(self, label):
        if label in self.vertices:
            return self.vertices[label], 0.0, "observed_vertex"
        if 0 in self.vertices:
            return self.vertices[0], 1.0, "origin_for_new_vertex"
        values = np.asarray(list(self.vertices.values()))
        projected = tuple(values.mean(axis=0))
        distance = (1/np.sqrt(len(values)) if label == 0
                    else np.sqrt(1+1/len(values)))
        return projected, float(distance), "uniform_distinct_vertices"

    def observe(self, label, a, b):
        if self._pending is None:
            raise RuntimeError("choose() must precede observe().")
        if isinstance(label, bool) or int(label) != label or label < 0:
            raise ValueError("label must be a nonnegative integer.")
        label, a, b = int(label), float(a), float(b)
        if not np.isfinite([a,b]).all() or not 0 < a <= 1 or not 0 <= b <= a:
            raise ValueError("Require 0 < a <= 1 and 0 <= b <= a.")
        if label == 0 and (a != 1.0 or b != 0.0):
            raise ValueError("Origin must have (a,b)=(1,0).")
        if label in self.vertices and self.vertices[label] != (a,b):
            raise ValueError("An observed vertex cannot change its payoff map.")
        p, before, t = self._pending, self.direction, self.round+1
        if t == 1:
            projected, h, residual, gap, kind = (a,b), 0.0, 0.0, 0.0, "first_round"
            saddle = (a,b)
        else:
            projected, h, kind = self._project(label)
            saddle = self._saddle_ab
            upper = max(before*(x*p-y) for x,y in self.vertices.values())
            lower = min(-before*saddle[1], before*(saddle[0]-saddle[1]))
            gap = max(0.0, upper-lower)
            if gap > 1e-12:
                raise RuntimeError("Exact saddle failed its recomputed gap check.")
            projected_payoff = projected[0]*p-projected[1]
            residual = abs((a*p-b)-projected_payoff)
            self.cumulative_residual += residual
            self.direction = float(np.clip(before+projected_payoff/(2*np.sqrt(t)), -1, 1))
        self.vertices[label] = (a,b)
        self.counts[label] = self.counts.get(label,0)+1
        self.round, self._pending = t, None
        return {"t":t, "label":label, "a":a, "b":b, "ratio":b/a,
                "capacity":p, "payoff":a*p-b, "direction":before,
                "direction_after":self.direction, "projected_a":projected[0],
                "projected_b":projected[1], "h_t":h, "residual":residual,
                "cumulative_residual":self.cumulative_residual,
                "target_witness":0.0, "saddle_a":saddle[0], "saddle_b":saddle[1],
                "saddle_gap":gap, "projection_gap":0.0, "projection_kind":kind}


def generate_capacity_path(config: ValidationConfig, seed: int):
    """Independent RNG streams per seed; fixed design, no path selection."""
    if isinstance(seed,bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(np.random.SeedSequence([config.base_entropy,config.scenario_id,int(seed)]))
    T, change = config.T, config.change_round
    pre_ratio = float(rng.uniform(config.pre_r_low, config.pre_r_high))
    a = np.ones(T)
    ratio = np.full(T, pre_ratio)
    ratio[0] = 0.0
    count = T-change
    a[change:] = rng.uniform(config.post_a_low, config.post_a_high, size=count)
    ratio[change:] = rng.uniform(config.post_r_low, config.post_r_high, size=count)
    labels = np.ones(T, dtype=np.int64)
    labels[0] = 0
    if pre_ratio == 0:
        labels[:change] = 0
    labels[change:] = np.arange(2, count+2)
    b = a*ratio
    digest = hashlib.sha256()
    for name, values, dtype in (("a",a,"<f8"),("b",b,"<f8"),("labels",labels,"<i8")):
        digest.update(name.encode("ascii")+b"\0")
        digest.update(np.asarray(values,dtype=dtype).tobytes(order="C"))
    return {"a":a, "b":b, "ratio":ratio, "labels":labels,
            "pre_ratio":pre_ratio,"path_sha256":digest.hexdigest()}


class _BatchSafeState:
    """Scalar K=2 recurrence, numerically equivalent to SafeBlockLearner."""
    def __init__(self, horizons):
        horizons = np.asarray(horizons, dtype=np.int64)
        self.horizons = horizons.copy()
        self.m = np.maximum(1, np.floor(np.sqrt(horizons)).astype(np.int64))
        self.n = np.maximum(1, horizons//self.m)
        self.local_t = np.zeros(len(horizons), dtype=np.int64)
        self.direction = np.zeros(len(horizons))
        self.capacity = np.full(len(horizons), 0.5)
        self.block_sum = np.zeros(len(horizons))

    def initialize(self, mask, horizons):
        self.horizons[mask] = horizons
        self.m[mask] = np.maximum(1, np.floor(np.sqrt(horizons)).astype(np.int64))
        self.n[mask] = horizons//self.m[mask]
        self.local_t[mask] = 0
        self.direction[mask] = 0.0
        self.capacity[mask] = 0.5
        self.block_sum[mask] = 0.0

    def choose(self):
        return np.where(self.local_t < self.m*self.n, self.capacity, 0.5)

    def observe(self, a, b, mask=None):
        if mask is None:
            mask = np.ones(len(self.local_t), dtype=bool)
        capacity = self.choose()
        payoff = a*capacity-b
        self.local_t[mask] += 1
        active = mask & (self.local_t <= self.m*self.n)
        self.block_sum[active] += payoff[active]
        self.capacity[active] = np.clip(capacity[active]-self.direction[active]*a[active]
                                       /(4*np.sqrt(self.n[active])),0,1)
        boundary = active & (self.local_t % self.n == 0)
        self.direction[boundary] = np.clip(self.direction[boundary]+self.block_sum[boundary]
                                           /(self.n[boundary]*2*np.sqrt(self.m[boundary])), -1,1)
        self.block_sum[boundary] = 0.0
        self.capacity[boundary] = 0.5
        return payoff


def run_validation_batch(config: ValidationConfig, seeds: Iterable[int], *,
                         sample_stride=128, representative_seeds=()):
    """Simulate independent full paths, all controls, with unchanged budget.

    Each seed gets its own RNG. All five methods face the identical exogenous
    path. The simulation accumulates every round before downsampling dynamics;
    representative seeds are chosen *before* play and saved at full resolution.
    No restarts, validation selection, best-seed selection or oracle lookahead.
    Summary records one independent seed, not one time point, as the unit.
    """
    raw_seeds = list(seeds)
    if any(isinstance(seed,bool) or int(seed) != seed or seed < 0 for seed in raw_seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    seeds = np.asarray(raw_seeds,dtype=np.int64)
    if not len(seeds) or len(np.unique(seeds)) != len(seeds) or np.any(seeds < 0):
        raise ValueError("Require a nonempty list of distinct nonnegative seeds.")
    if isinstance(sample_stride,bool) or int(sample_stride) != sample_stride or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer.")
    representatives = set(int(seed) for seed in representative_seeds)
    if not representatives.issubset(set(seeds.tolist())):
        raise ValueError("Representative seeds must belong to the batch.")
    T, change, N = config.T, config.change_round, len(seeds)
    paths = [generate_capacity_path(config,int(seed)) for seed in seeds]
    a = np.asarray([path["a"] for path in paths])
    b = np.asarray([path["b"] for path in paths])
    ratio = np.asarray([path["ratio"] for path in paths])
    pre_ratio = np.array([path["pre_ratio"] for path in paths])
    # Once origin and the one prefix type are seen, all post-change vertices
    # are new and project exactly to origin. These three cases implement the
    # full orthogonal geometry without materializing a T-dimensional basis.
    fast_direction, fast_E = np.zeros(N), np.zeros(N)
    frozen_E, switches = np.zeros(N), np.zeros(N,dtype=np.int64)
    safe_all = _BatchSafeState(np.full(N,T,dtype=np.int64))
    safe_tail = _BatchSafeState(np.ones(N,dtype=np.int64))
    G = paper_safe_budget(T,1)
    recent = np.zeros((N,config.window))
    recent_sum, recent_count = np.zeros(N), 0
    previous_ratio = np.full(N,0.5)
    sums = {method:np.zeros(N) for method in METHODS}
    abs_sums = {method:np.zeros(N) for method in METHODS}
    auc = {method:np.zeros(N) for method in METHODS}
    pre_sums, pre_abs, pre_auc = {}, {}, {}
    prefix_at_switch = np.zeros(N)
    times = np.unique(np.r_[1,2,3,np.arange(sample_stride,T+1,sample_stride), change,
                            min(change+1,T),T]).astype(np.int64)
    dynamics = {"t":times, "seeds":seeds.copy()}
    for method in METHODS:
        dynamics[method] = {key:np.empty((N,len(times)))
                            for key in ("delta","cumulative_payoff","cumulative_absolute_error")}
    dynamics.update({"fast_E":np.empty((N,len(times))), "master_E":np.empty((N,len(times))),
                     "safe_mode":np.empty((N,len(times)),dtype=bool)})
    rep_indices = [i for i,seed in enumerate(seeds) if int(seed) in representatives]
    reps = {str(int(seeds[i])):{"path":paths[i], "methods":{method:{
        key:np.empty(T,dtype=(bool if key == "safe_mode" else np.int64 if key == "safe_local_t" else float))
        for key in ("capacity","payoff","delta","cumulative_payoff","cumulative_absolute_error",
                    "E","residual","direction","safe_mode","safe_local_t")}
        for method in METHODS}} for i in rep_indices}
    grid_index = 0
    for index in range(T):
        t = index+1
        ai, bi, ri = a[:,index], b[:,index], ratio[:,index]
        before = fast_direction.copy()
        fp = np.where(before > 0,0.0,np.where(before < 0,1.0,0.5))
        mode = switches > 0  # A crossing round is still played by fast.
        tail_p = safe_tail.choose()
        bp = safe_all.choose()
        wp = np.full(N,0.5) if not recent_count else recent_sum/min(recent_count,config.window)
        actions = {"shared_past_hull":fp, "one_switch":np.where(mode,tail_p,fp),
                   "block_safe":bp, "lag_response":previous_ratio.copy(), "last_window":wp}
        # Prefix known type projects to itself from round 3 onward. On round
        # 2 (its first observation), and every post-change fresh type, projection
        # is origin. The first round has no residual or direction update.
        projected_b = np.zeros(N) if t == 2 or t > change else bi
        projected_a = np.ones(N) if t == 2 or t > change else ai
        residual = np.zeros(N) if t == 1 else np.abs(ai*fp-bi-(projected_a*fp-projected_b))
        if t > 1:
            fast_E += residual
            fast_direction = np.clip(before+(projected_a*fp-projected_b)/(2*np.sqrt(t)),-1,1)
        frozen_E[~mode] = fast_E[~mode]
        # Observe with the original safe recurrences only after all choices.
        safe_all.observe(ai,bi)
        safe_tail.observe(ai,bi,mode)
        for method in METHODS:
            payoff = ai*actions[method]-bi
            sums[method] += payoff
            abs_sums[method] += np.abs(payoff)
            auc[method] += np.abs(sums[method])/t
            for i in rep_indices:
                log = reps[str(int(seeds[i]))]["methods"][method]
                log["capacity"][index] = actions[method][i]
                log["payoff"][index] = payoff[i]
                log["delta"][index] = abs(sums[method][i])/t
                log["cumulative_payoff"][index] = sums[method][i]
                log["cumulative_absolute_error"][index] = abs_sums[method][i]
                log["E"][index] = frozen_E[i] if method == "one_switch" else fast_E[i] if method == "shared_past_hull" else 0
                log["residual"][index] = residual[i] if method == "shared_past_hull" or (method == "one_switch" and not mode[i]) else np.nan
                log["direction"][index] = before[i] if method == "shared_past_hull" or (method == "one_switch" and not mode[i]) else np.nan
                log["safe_mode"][index] = mode[i] if method == "one_switch" else method == "block_safe"
                log["safe_local_t"][index] = safe_tail.local_t[i] if method == "one_switch" and mode[i] else t if method == "block_safe" else 0
        crossing = (~mode) & (fast_E > G)
        if crossing.any():
            switches[crossing] = t
            prefix_at_switch[crossing] = sums["one_switch"][crossing]
            if t < T:
                safe_tail.initialize(crossing,T-t+np.zeros(crossing.sum(),dtype=np.int64))
        if t == change:
            pre_sums = {method:values.copy() for method,values in sums.items()}
            pre_abs = {method:values.copy() for method,values in abs_sums.items()}
            pre_auc = {method:values.copy() for method,values in auc.items()}
        if grid_index < len(times) and t == times[grid_index]:
            for method in METHODS:
                dynamics[method]["delta"][:,grid_index] = np.abs(sums[method])/t
                dynamics[method]["cumulative_payoff"][:,grid_index] = sums[method]
                dynamics[method]["cumulative_absolute_error"][:,grid_index] = abs_sums[method]
            dynamics["fast_E"][:,grid_index] = fast_E
            dynamics["master_E"][:,grid_index] = frozen_E
            dynamics["safe_mode"][:,grid_index] = mode
            grid_index += 1
        slot = index % config.window
        recent_sum += ri-recent[:,slot]
        recent[:,slot] = ri
        recent_count += 1
        previous_ratio = ri.copy()
    summaries = []
    for i,seed in enumerate(seeds):
        result = {"seed":int(seed), "pre_ratio":float(pre_ratio[i]),
                  "path_sha256":paths[i]["path_sha256"],
                  "switch_round":int(switches[i]) if switches[i] else None,
                  "safe_rounds":int(T-switches[i]) if switches[i] else 0,
                  "master_final_E":float(frozen_E[i]), "fast_final_E":float(fast_E[i]),
                  "G_T":G, "methods":{}}
        for method in METHODS:
            result["methods"][method] = {
                "terminal_delta":float(abs(sums[method][i])/T),
                "signed_cumulative_payoff":float(sums[method][i]),
                "mean_absolute_error":float(abs_sums[method][i]/T),
                "mean_prefix_distance":float(auc[method][i]/T),
                "pre_terminal_delta":float(abs(pre_sums[method][i])/change),
                "pre_mean_absolute_error":float(pre_abs[method][i]/change),
                "pre_mean_prefix_distance":float(pre_auc[method][i]/change),
                "pre_mean_delta":float(pre_auc[method][i]/change),
                "post_mean_delta":float((auc[method][i]-pre_auc[method][i])/(T-change)) if T > change else None,
                "post_terminal_delta":float(abs(sums[method][i]-pre_sums[method][i])/(T-change)) if T > change else None,
                "post_mean_absolute_error":float((abs_sums[method][i]-pre_abs[method][i])/(T-change)) if T > change else None,
            }
        if switches[i] and switches[i] < T:
            tail_h = T-int(switches[i])
            tail_sum = sums["one_switch"][i]-prefix_at_switch[i]
            result.update({"fast_prefix_signed_sum":float(prefix_at_switch[i]),
                           "safe_tail_signed_sum":float(tail_sum),
                           "safe_tail_delta":float(abs(tail_sum)/tail_h),
                           "safe_tail_budget":paper_safe_budget(tail_h,1)})
        summaries.append(result)
    return {"metadata":{
        "benchmark":"stochastic_weighted_capacity_two_phase",
        "config":asdict(config), "independent_seeds":seeds.tolist(),
        "representative_seeds":sorted(representatives), "sample_stride":int(sample_stride),
        "payoff":"a(ell)*p-b(ell)", "response":"p_star(ell)=b(ell)/a(ell)",
        "full_strict_target":"{0}, exactly for every mixture",
        "learner_affine_dimension":1, "vertex_payoff_norm_bound":1.0,
        "opponent_geometry":"conv{0,e_1,...,e_M} in Euclidean R^M",
        "safe_base":"original SafeBlockLearner scalar K=2 recurrence",
        "safe_budget":"B0(h)=6*h^(3/4)", "G_T":G,
        "safe_compression":"Four (a,b) corners; exact for payoffs, gradients, block responses only",
        "fast_oracles":"Exact analytic origin/unit-vertex hull projections and scalar saddle",
        "same_exogenous_path_across_methods":True, "master_restarts":False,
        "fresh_safe_tail_after_crossing":True,
        "phase_change_starts":change+1 if change < T else None,
        "rng":"default_rng(SeedSequence([base_entropy, scenario_id, seed])); independent per episode/cell",
        "limitations":["Constructed stochastic opponent, not real data or real attacker learning.",
                       "Fresh orthogonal contexts make realized opponent dimension large; no q=4 study or rate validation.",
                       "Fast sensitivity to specified opponent geometry remains; payoff-equivalent compression is not used by fast.",
                       "Lag/window are causal heuristic controls without asserted safety certificates for weighted game.",
                       "Signed average-payoff target distance and mean absolute round error are different metrics.",
                       "Independent seeds represent this generator, not a broader empirical population."]},
        "summaries":summaries,"dynamics":dynamics,"representatives":reps}
