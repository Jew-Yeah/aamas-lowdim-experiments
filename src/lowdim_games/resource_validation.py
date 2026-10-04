"""A genuinely vector, fixed resource game with a causal reactive opponent.

The learner mixes an inactive and an active service plan. Opponent actions are
the origin and distinct, pre-existing unit resource requests. Resource i has
its own deficit coordinate; its payoff map is never created or modified by
the opponent. The full strict response target is evaluated by resource_target.

Only sparse physics is compressed for the original block-safe routine. The
fast routine retains the specified Euclidean resource geometry and explicitly
selects the observed origin among dual saddle optima. With this fixed tie
rule, each new unit request projects to the origin and has residual one.
Different valid dual saddle tie rules can change the fast dynamics in this
vector game. The budget remains 6*T**(3/4), without retuning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Iterable

import numpy as np
from scipy.special import expit

from .learners import paper_safe_budget
from .resource_target import exact_target_projection, target_bounds


EPSILON = float(1 / np.sqrt(2))
METHODS = ("one_switch", "shared_past_hull", "block_safe", "lag_response",
           "last_window", "uniform", "request_trigger", "fast_recent_saddle")


@dataclass(frozen=True)
class ResourceValidationConfig:
    T: int = 262144
    change_round: int = 98304
    mode: str = "reactive"
    memory: int = 1
    window: int = 16
    theta_low: float = .45
    theta_high: float = .55
    reaction_floor: float = .05
    reaction_gain: float = .9
    reaction_scale: float = .03
    exogenous_probability: float = .5
    gamma0: float = 1.
    gamma1: float = .2
    scenario_id: int = 0
    base_entropy: int = 2026100402

    def __post_init__(self):
        for name in ("T", "change_round", "memory", "window"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if not 2 <= self.change_round <= self.T:
            raise ValueError("Require 2 <= change_round <= T.")
        if self.mode not in ("reactive", "exogenous", "stationary"):
            raise ValueError("mode must be reactive, exogenous or stationary.")
        for name in ("scenario_id", "base_entropy"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
            object.__setattr__(self, name, int(value))
        values = [self.theta_low, self.theta_high, self.reaction_floor,
                  self.reaction_gain, self.reaction_scale,
                  self.exogenous_probability, self.gamma0, self.gamma1]
        if not np.isfinite(values).all():
            raise ValueError("All real parameters must be finite.")
        if not 0 <= self.theta_low <= self.theta_high <= 1:
            raise ValueError("Episode thresholds must lie in [0,1].")
        if (not 0 <= self.reaction_floor <= 1 or self.reaction_gain < 0
                or self.reaction_floor + self.reaction_gain > 1
                or self.reaction_scale <= 0):
            raise ValueError("The reactive probability must lie in [0,1].")
        if not 0 <= self.exogenous_probability <= 1:
            raise ValueError("The exogenous probability must lie in [0,1].")
        if not 0 <= self.gamma1 <= self.gamma0 <= 1 or self.gamma0 <= 0:
            raise ValueError("Require 0 <= gamma1 <= gamma0 <= 1 and gamma0 > 0.")
        if not 0 < self.response_threshold < 1:
            raise ValueError("The response threshold must be strictly between zero and one.")

    @property
    def response_threshold(self):
        return self.gamma0 / (2 * EPSILON + self.gamma0 - self.gamma1)

    @property
    def beta(self):
        return 2 * EPSILON * self.response_threshold

    @property
    def balanced_capacity(self):
        return self.beta / self.gamma0


def generate_resource_innovations(config: ResourceValidationConfig, seed: int):
    """Episode threshold and common random numbers, fixed before all play."""
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(np.random.SeedSequence(
        [config.base_entropy, config.scenario_id, int(seed)]))
    theta = float(rng.uniform(config.theta_low, config.theta_high))
    uniforms = rng.random(config.T - config.change_round)
    digest = hashlib.sha256()
    digest.update(b"theta\0" + np.asarray([theta], dtype="<f8").tobytes())
    digest.update(b"uniforms\0" + np.asarray(uniforms, dtype="<f8").tobytes())
    return {"theta": theta, "uniforms": uniforms,
            "innovation_sha256": digest.hexdigest()}


def arrival_probability(config, theta, past_capacity_mean):
    """Read only previously played capacities, never the current choice."""
    theta, mean = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(past_capacity_mean, dtype=float))
    if config.mode == "stationary":
        return np.zeros_like(mean)
    if config.mode == "exogenous":
        return np.full_like(mean, config.exogenous_probability)
    return config.reaction_floor + config.reaction_gain * expit(
        (theta - mean) / config.reaction_scale)


class _SparseResourceSafeState:
    """Exact sparse specialization of the original K=2 block-safe learner.

    A newly revealed resource has zero dual coordinate. It will never reappear,
    so later choices need only aggregate/energy dual coordinates. Past resource
    coordinates are retained through their squared norm, including the common
    rescaling of the Euclidean ball projection. This is not a dimension change
    in either the payoff or the opponent geometry.
    """
    def __init__(self, horizons, config):
        horizons = np.asarray(horizons, dtype=np.int64)
        if horizons.ndim != 1 or np.any(horizons < 1):
            raise ValueError("Safe horizons must be a positive integer vector.")
        self.config = config
        self.horizons = horizons.copy()
        self.m = np.maximum(1, np.floor(np.sqrt(horizons)).astype(np.int64))
        self.n = horizons // self.m
        self.local_t = np.zeros(len(horizons), dtype=np.int64)
        self.capacity = np.full(len(horizons), .5)
        self.direction_A = np.zeros(len(horizons))
        self.direction_E = np.zeros(len(horizons))
        self.resource_direction_sq = np.zeros(len(horizons))
        self.block_requests = np.zeros(len(horizons))
        self.block_request_p = np.zeros(len(horizons))
        self.block_request_p_sq = np.zeros(len(horizons))
        self.block_energy = np.zeros(len(horizons))

    def initialize(self, mask, horizons):
        horizons = np.asarray(horizons, dtype=np.int64)
        self.horizons[mask] = horizons
        self.m[mask] = np.maximum(1, np.floor(np.sqrt(horizons)).astype(np.int64))
        self.n[mask] = horizons // self.m[mask]
        for values in (self.local_t, self.direction_A, self.direction_E,
                       self.resource_direction_sq, self.block_requests,
                       self.block_request_p, self.block_request_p_sq, self.block_energy):
            values[mask] = 0
        self.capacity[mask] = .5

    def choose(self):
        return np.where(self.local_t < self.m * self.n, self.capacity, .5)

    def observe(self, requests, mask=None):
        req = np.asarray(requests, dtype=bool)
        if req.shape != self.local_t.shape:
            raise ValueError("One request flag per safe instance is required.")
        if mask is None:
            mask = np.ones(len(self.local_t), dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
        p = self.choose()
        gamma = self.config.gamma0 - (self.config.gamma0-self.config.gamma1)*req
        self.local_t[mask] += 1
        active = mask & (self.local_t <= self.m*self.n)
        self.block_requests[active] += req[active]
        self.block_request_p[active] += req[active]*p[active]
        self.block_request_p_sq[active] += req[active]*p[active]**2
        self.block_energy[active] += gamma[active]*p[active]
        gradient = gamma*self.direction_E - EPSILON*req*self.direction_A
        self.capacity[active] = np.clip(p[active]-gradient[active]
                                       /(4*np.sqrt(self.n[active])), 0, 1)
        boundary = active & (self.local_t % self.n == 0)
        if boundary.any():
            n, m = self.n[boundary], self.m[boundary]
            count = self.block_requests[boundary]
            p_sum = self.block_request_p[boundary]
            p_sq = self.block_request_p_sq[boundary]
            response = (count/n > self.config.response_threshold).astype(float)
            vA = EPSILON*(response*count-p_sum)/n
            mean_gamma = self.config.gamma0-(self.config.gamma0-self.config.gamma1)*count/n
            vE = self.block_energy[boundary]/n-response*mean_gamma
            fair_sq = EPSILON**2*(response*count-2*response*p_sum+p_sq)/n**2
            fair_sq = np.maximum(fair_sq, 0.)
            self.direction_A[boundary] += vA/(2*np.sqrt(m))
            self.direction_E[boundary] += vE/(2*np.sqrt(m))
            self.resource_direction_sq[boundary] += fair_sq/(4*m)
            norm = np.sqrt(self.direction_A[boundary]**2+self.direction_E[boundary]**2
                           +self.resource_direction_sq[boundary])
            scale = np.maximum(1., norm)
            self.direction_A[boundary] /= scale
            self.direction_E[boundary] /= scale
            self.resource_direction_sq[boundary] /= scale**2
            self.capacity[boundary] = .5
            for values in (self.block_requests, self.block_request_p,
                           self.block_request_p_sq, self.block_energy):
                values[boundary] = 0
        return EPSILON*req*(1-p), gamma*p


def resource_tensor(resource_count, config=None):
    """Small dense reference game; production uses the same sparse maps."""
    config = config or ResourceValidationConfig()
    if isinstance(resource_count, bool) or int(resource_count) != resource_count or resource_count < 1:
        raise ValueError("resource_count must be a positive integer.")
    resource_count = int(resource_count)
    tensor = np.zeros((2, resource_count+1, resource_count+2))
    tensor[0, 1:, 0] = EPSILON
    tensor[0, np.arange(1, resource_count+1), np.arange(1, resource_count+1)] = EPSILON
    tensor[1, 0, -1] = config.gamma0
    tensor[1, 1:, -1] = config.gamma1
    return tensor


def resource_response(opponent, config=None):
    config = config or ResourceValidationConfig()
    ell = np.asarray(opponent, dtype=float)
    if (ell.ndim != 1 or not np.isfinite(ell).all() or np.any(ell < -1e-12)
            or abs(float(ell.sum())-1) > 1e-10):
        raise ValueError("The reference opponent must be a simplex mixture.")
    service = float(ell[1:].sum() > config.response_threshold)
    return np.array([1-service, service])


def _path_hash(bits, T, change):
    digest = hashlib.sha256()
    digest.update(np.asarray([T, change], dtype="<i8").tobytes())
    digest.update(np.asarray(bits, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def run_resource_batch(config: ResourceValidationConfig, seeds: Iterable[int], *,
                       sample_stride=512, representative_seeds=(), progress_callback=None):
    """Complete paired closed-loop episodes with own paths and strict targets.

    Common episode thresholds and innovations are paired across all policies.
    Reactive policies have separate capacity histories and realized paths;
    the exogenous control has a genuinely shared path. Exact full-target
    projection is used for terminal endpoints. All-round distances are saved
    as certified lower/upper enclosures, while the origin-prefix is exact.
    No episode is discarded for failing to switch.
    """
    original_seeds = list(seeds)
    if (not original_seeds or any(isinstance(s,bool) or int(s) != s or s < 0 for s in original_seeds)
            or len(set(original_seeds)) != len(original_seeds)):
        raise ValueError("Require distinct nonnegative integer seeds.")
    seeds = np.asarray(original_seeds, dtype=np.int64)
    if isinstance(sample_stride,bool) or int(sample_stride) != sample_stride or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer.")
    sample_stride = int(sample_stride)
    representatives = set(int(s) for s in representative_seeds)
    if not representatives.issubset(set(seeds.tolist())):
        raise ValueError("Representatives must belong to the batch.")
    N, T, H, J = len(seeds), config.T, config.change_round, len(METHODS)
    post = T-H
    innovations = [generate_resource_innovations(config,int(s)) for s in seeds]
    theta = np.array([value["theta"] for value in innovations])
    uniforms = np.asarray([value["uniforms"] for value in innovations])
    # Two continuous-weight policies need full resource weights only once,
    # for exact terminal projection. Other controls have finite histograms.
    continuous = {name:np.zeros((N,post)) for name in ("one_switch","block_safe")}
    hist_unserved = np.zeros((N,J), dtype=np.int64)
    hist_half = np.zeros((N,J), dtype=np.int64)
    path_bits = np.zeros((N,J,(post+7)//8), dtype=np.uint8)
    safe = _SparseResourceSafeState(np.full(N,T),config)
    tail = _SparseResourceSafeState(np.ones(N,dtype=np.int64),config)
    switches = np.zeros(N,dtype=np.int64)
    master_E, fast_E = np.zeros(N), np.zeros(N)
    recent_fast_E, recent_direction_E = np.zeros(N),np.zeros(N)
    G = paper_safe_budget(T,1)
    action_ring = np.zeros((N,J,config.memory))
    action_sum = np.zeros((N,J))
    request_ring = np.zeros((N,config.window),dtype=bool)
    window_sum = np.zeros(N,dtype=np.int64)
    last_request = np.zeros((N,J),dtype=bool)
    sum_A, sum_E, sum_q_sq = (np.zeros((N,J)) for _ in range(3))
    sum_idle, sum_loaded = np.zeros((N,J)),np.zeros((N,J))
    sum_norm, distance_lo_sum, distance_hi_sum = (np.zeros((N,J)) for _ in range(3))
    request_count = np.zeros((N,J),dtype=np.int64)
    tail_A, tail_E, tail_q_sq = (np.zeros(N) for _ in range(3))
    tail_requests = np.zeros(N,dtype=np.int64)
    prefix_at_switch_A, prefix_at_switch_E = np.zeros(N),np.zeros(N)
    pre = None
    times = np.unique(np.r_[1,2,3,np.arange(sample_stride,T+1,sample_stride),H,min(H+1,T),T]).astype(np.int64)
    dynamics = {"t":times,"seeds":seeds.copy(),"master_E":np.empty((N,len(times))),
                "fast_E":np.empty((N,len(times))),"safe_mode":np.empty((N,len(times)),dtype=bool)}
    dynamics["recent_fast_E"] = np.empty((N,len(times)))
    dynamic_keys = ("delta_lower","delta_upper","cumulative_aggregate","cumulative_energy",
                    "cumulative_idle_cost","cumulative_loaded_cost","mean_aggregate_deficit",
                    "resource_sq_sum","observed_count","cumulative_round_norm",
                    "capacity","request_probability")
    for method in METHODS:
        dynamics[method] = {key:np.empty((N,len(times)),dtype=np.int64 if key=="observed_count" else float)
                            for key in dynamic_keys}
    rep_indices = [i for i,s in enumerate(seeds) if int(s) in representatives]
    rep_keys = (*dynamic_keys,"payoff_aggregate","payoff_energy","payoff_resource","request",
                "label","E","residual","safe_mode","safe_local_t",
                "direction_aggregate","direction_energy","direction_resource_norm")
    reps = {}
    for i in rep_indices:
        seed = str(int(seeds[i]))
        reps[seed] = {"path":{"uniforms":innovations[i]["uniforms"].copy(),
                              "theta":theta[i]},"methods":{}}
        for method in METHODS:
            reps[seed]["methods"][method] = {
                key:np.empty(T,dtype=bool if key in ("request","safe_mode") else
                             np.int64 if key in ("label","safe_local_t","observed_count") else float)
                for key in rep_keys}
    # The stacked matrix now owns all innovations; retain only seed receipts
    # and the two prespecified representative copies, avoiding a second full
    # N-by-post matrix in the per-episode generator dictionaries.
    for innovation in innovations:
        innovation["uniforms"] = None
    grid_index = 0
    started = time.perf_counter()
    for index in range(T):
        t = index+1
        mode = switches > 0
        fast_p = .5 if t <= 2 else 0.
        actions = np.empty((N,J))
        actions[:,0] = np.where(mode,tail.choose(),fast_p)
        actions[:,1] = fast_p
        actions[:,2] = safe.choose()
        actions[:,3] = .5 if t==1 else last_request[:,3]
        actions[:,4] = .5 if t==1 else window_sum/min(t-1,config.window) > config.response_threshold
        actions[:,5] = .5
        actions[:,6] = .5 if t==1 else np.where(request_count[:,6]>0,config.balanced_capacity,0.)
        actions[:,7] = np.where(recent_direction_E>0,0.,np.where(recent_direction_E<0,1.,.5))
        recent_target_energy = np.where(request_count[:,7]>0,config.gamma1,0.)
        old_recent_direction_E = recent_direction_E.copy()
        # This mean contains only actions from rounds <t. The current choices
        # above are deliberately absent from the environment's information.
        past_mean = action_sum/min(index,config.memory) if index else np.full((N,J),.5)
        probabilities = np.zeros((N,J))
        requests = np.zeros((N,J),dtype=bool)
        if t > H:
            probabilities = arrival_probability(config,theta[:,None],past_mean)
            requests = uniforms[:,index-H,None] < probabilities
            path_bits[:,:, (index-H)//8] |= requests.astype(np.uint8) << ((index-H)%8)
        old_master_direction_A = np.where(mode,tail.direction_A,0.)
        old_master_direction_E = np.where(mode,tail.direction_E,
            config.gamma0/(4*np.sqrt(2)) if t>=3 else 0.)
        old_master_direction_sq = np.where(mode,tail.resource_direction_sq,0.)
        old_safe_A,old_safe_E,old_safe_sq = safe.direction_A.copy(),safe.direction_E.copy(),safe.resource_direction_sq.copy()
        safe.observe(requests[:,2])
        tail.observe(requests[:,0],mode)
        gamma = config.gamma0-(config.gamma0-config.gamma1)*requests
        q = EPSILON*requests*(1-actions)
        energy = gamma*actions
        sum_A += q
        sum_E += energy
        sum_idle += config.gamma0*actions*(~requests)
        sum_loaded += config.gamma1*actions*requests
        sum_q_sq += q*q
        sum_norm += np.sqrt(2*q*q+energy*energy)
        request_count += requests
        hist_unserved += requests & (actions==0)
        hist_half += requests & (actions==.5)
        if t > H:
            continuous["one_switch"][:,index-H] = q[:,0]
            continuous["block_safe"][:,index-H] = q[:,2]
        fast_E += requests[:,1]
        recent_residual = requests[:,7]*np.sqrt((1-actions[:,7])**2
                             +((config.gamma0-config.gamma1)*actions[:,7])**2)
        if t>1:
            recent_fast_E += recent_residual
            recent_direction_E = np.clip(recent_direction_E
                +(config.gamma0*actions[:,7]-recent_target_energy)/(2*np.sqrt(t)),-1,1)
        master_E[~mode] += requests[~mode,0]
        tail_A[mode] += q[mode,0]
        tail_E[mode] += energy[mode,0]
        tail_q_sq[mode] += q[mode,0]**2
        tail_requests[mode] += requests[mode,0]
        if t<=H or config.mode=="stationary":
            # This realized hull is exactly the origin. Its full strict target
            # is {0}, so the Euclidean distance is the positive energy mean.
            prefix_distance = sum_E/t
            bounds = {"lower":prefix_distance,"upper":prefix_distance}
        else:
            bounds = target_bounds(sum_A/t,sum_E/t,sum_q_sq/t**2,request_count,config.beta)
        distance_lo_sum += bounds["lower"]
        distance_hi_sum += bounds["upper"]
        for i in rep_indices:
            logs = reps[str(int(seeds[i]))]["methods"]
            for j,method in enumerate(METHODS):
                log = logs[method]
                values = {"capacity":actions[i,j],"request_probability":probabilities[i,j],
                    "payoff_aggregate":q[i,j],"payoff_resource":q[i,j],"payoff_energy":energy[i,j],
                    "request":requests[i,j],"label":t-H if requests[i,j] else 0,
                    "cumulative_aggregate":sum_A[i,j],"cumulative_energy":sum_E[i,j],
                    "cumulative_idle_cost":sum_idle[i,j],"cumulative_loaded_cost":sum_loaded[i,j],
                    "mean_aggregate_deficit":sum_A[i,j]/(EPSILON*t),
                    "resource_sq_sum":sum_q_sq[i,j],"observed_count":request_count[i,j],
                    "cumulative_round_norm":sum_norm[i,j],"delta_lower":bounds["lower"][i,j],
                    "delta_upper":bounds["upper"][i,j],"E":master_E[i] if j==0 else fast_E[i] if j==1 else recent_fast_E[i] if j==7 else 0.,
                    "residual":float(requests[i,j]) if j==1 or (j==0 and not mode[i]) else recent_residual[i] if j==7 else np.nan,
                    "safe_mode":bool(mode[i]) if j==0 else j==2,
                    "safe_local_t":int(tail.local_t[i]) if j==0 and mode[i] else t if j==2 else 0,
                    "direction_aggregate":old_master_direction_A[i] if j==0 else old_safe_A[i] if j==2 else 0. if j in (1,7) else np.nan,
                    "direction_energy":old_master_direction_E[i] if j==0 else old_safe_E[i] if j==2 else old_recent_direction_E[i] if j==7 else config.gamma0/(4*np.sqrt(2)) if j==1 and t>=3 else 0. if j==1 else np.nan,
                    "direction_resource_norm":np.sqrt(old_master_direction_sq[i]) if j==0 else np.sqrt(old_safe_sq[i]) if j==2 else 0. if j in (1,7) else np.nan}
                for key,value in values.items():
                    log[key][index] = value
        crossing = (~mode) & (master_E > G)
        if crossing.any():
            switches[crossing] = t
            prefix_at_switch_A[crossing] = sum_A[crossing,0]
            prefix_at_switch_E[crossing] = sum_E[crossing,0]
            if t < T:
                tail.initialize(crossing,np.full(crossing.sum(),T-t))
        if t == H:
            pre = {"A":sum_A.copy(),"E":sum_E.copy(),"sq":sum_q_sq.copy(),
                   "norm":sum_norm.copy(),"lower":distance_lo_sum.copy(),"upper":distance_hi_sum.copy()}
        if grid_index < len(times) and t==times[grid_index]:
            for j,method in enumerate(METHODS):
                values = {"delta_lower":bounds["lower"][:,j],"delta_upper":bounds["upper"][:,j],
                          "cumulative_aggregate":sum_A[:,j],"cumulative_energy":sum_E[:,j],
                          "cumulative_idle_cost":sum_idle[:,j],"cumulative_loaded_cost":sum_loaded[:,j],
                          "mean_aggregate_deficit":sum_A[:,j]/(EPSILON*t),
                          "resource_sq_sum":sum_q_sq[:,j],"observed_count":request_count[:,j],
                          "cumulative_round_norm":sum_norm[:,j],"capacity":actions[:,j],
                          "request_probability":probabilities[:,j]}
                for key,value in values.items():
                    dynamics[method][key][:,grid_index] = value
            dynamics["master_E"][:,grid_index] = master_E
            dynamics["fast_E"][:,grid_index] = fast_E
            dynamics["recent_fast_E"][:,grid_index] = recent_fast_E
            dynamics["safe_mode"][:,grid_index] = mode
            grid_index += 1
        slot = index % config.memory
        action_sum += actions-action_ring[:,:,slot]
        action_ring[:,:,slot] = actions
        request_slot = index % config.window
        window_sum += requests[:,4].astype(np.int64)-request_ring[:,request_slot].astype(np.int64)
        request_ring[:,request_slot] = requests[:,4]
        last_request = requests.copy()
        if progress_callback is not None and (t%16384==0 or t==T):
            progress_callback({"round":t,"T":T,"elapsed_seconds":time.perf_counter()-started})
    summaries = []
    for i,seed in enumerate(seeds):
        tau = int(switches[i]) if switches[i] else None
        row = {"seed":int(seed),"theta":float(theta[i]),
               "innovation_sha256":innovations[i]["innovation_sha256"],
               "switch_round":tau,"safe_rounds":T-tau if tau else 0,
               "G_T":G,"master_final_E":float(master_E[i]),"fast_final_E":float(fast_E[i]),
               "recent_fast_final_E":float(recent_fast_E[i]),"methods":{}}
        for j,method in enumerate(METHODS):
            count = int(request_count[i,j])
            if method in continuous:
                values = continuous[method][i]
                multiplicity = None
            elif method in ("lag_response","last_window"):
                values = np.array([0.,EPSILON])
                multiplicity = np.array([count-hist_unserved[i,j],hist_unserved[i,j]])
            elif method == "uniform":
                values,multiplicity = np.array([EPSILON/2]),np.array([count])
            elif method == "request_trigger":
                values = np.array([EPSILON,EPSILON*(1-config.balanced_capacity)])
                multiplicity = np.array([int(count>0),max(count-1,0)])
            elif method == "fast_recent_saddle":
                values = np.array([0.,EPSILON/2,EPSILON])
                multiplicity = np.array([count-hist_half[i,j]-hist_unserved[i,j],
                                         hist_half[i,j],hist_unserved[i,j]])
            else:
                values,multiplicity = np.array([EPSILON]),np.array([count])
            terminal = exact_target_projection(values,sum_A[i,j],sum_E[i,j],T,config.beta,
                                              observed_count=count,counts=multiplicity)
            if terminal["support_gap"] > 2e-8 or terminal["feasibility_error"] > 2e-8:
                raise RuntimeError("The full-target terminal oracle failed its residual checks.")
            result = {"terminal_delta":terminal["distance"],
                "terminal_support_gap":terminal["support_gap"],
                "terminal_feasibility_error":terminal["feasibility_error"],
                "terminal_projection_lower":float(np.sqrt(max(terminal["distance_squared"]-terminal["support_gap"],0.))),
                "terminal_projection_upper":float(terminal["distance"]+terminal["feasibility_error"]),
                "terminal_delta_lower":float(bounds["lower"][i,j]),
                "terminal_delta_upper":float(bounds["upper"][i,j]),
                "pre_terminal_delta":float(pre["E"][i,j]/H),
                "pre_mean_delta":float(pre["lower"][i,j]/H),
                "pre_mean_delta_upper":float(pre["upper"][i,j]/H),
                "mean_prefix_distance_lower":float(distance_lo_sum[i,j]/T),
                "mean_prefix_distance_upper":float(distance_hi_sum[i,j]/T),
                "mean_round_payoff_norm":float(sum_norm[i,j]/T),
                "mean_aggregate_deficit":float(sum_A[i,j]/(EPSILON*T)),
                "mean_aggregate_deficit_normalized":float(sum_A[i,j]/T),
                "mean_unserved_fraction":float(sum_A[i,j]/(EPSILON*T)),
                "mean_activation_cost":float(sum_E[i,j]/T),
                "mean_idle_cost":float(sum_idle[i,j]/T),
                "mean_loaded_cost":float(sum_loaded[i,j]/T),
                "mean_weighted_cost":float((2*sum_A[i,j]+sum_E[i,j])/T),
                "resource_deficit_norm":float(np.sqrt(sum_q_sq[i,j])/T),
                "observed_requests":count,"request_fraction":count/T,
                "request_path_sha256":_path_hash(path_bits[i,j],T,H),
                "cumulative_aggregate":float(sum_A[i,j]),"cumulative_energy":float(sum_E[i,j]),
                "resource_sq_sum":float(sum_q_sq[i,j])}
            if post:
                result.update({"post_mean_delta_lower":float((distance_lo_sum[i,j]-pre["lower"][i,j])/post),
                               "post_mean_delta_upper":float((distance_hi_sum[i,j]-pre["upper"][i,j])/post)})
            else:
                result.update({"post_mean_delta_lower":None,"post_mean_delta_upper":None})
            if not result["terminal_delta_lower"]-1e-9 <= terminal["distance"] <= result["terminal_delta_upper"]+1e-9:
                raise RuntimeError("Exact terminal distance escaped its full-target enclosure.")
            row["methods"][method] = result
        if tau and tau<T:
            h,count = T-tau,int(tail_requests[i])
            if count == h:
                distance = float(np.sqrt((tail_A[i]/h)**2+tail_q_sq[i]/h**2
                                         +(tail_E[i]/h-config.gamma1)**2))
                tail_target = "{gamma1*e_energy}; tail contains requests only"
            else:
                raw = continuous["one_switch"][i,tau-H:]
                projection = exact_target_projection(raw,tail_A[i],tail_E[i],h,config.beta,observed_count=count)
                distance = projection["distance"]
                tail_target = "{0}" if not count else "full origin-plus-visited-resource simplex"
            budget = paper_safe_budget(h,1)
            if h*distance > budget+2e-7:
                raise RuntimeError("The original safe-tail certificate was violated.")
            row.update({"fast_prefix_cumulative_aggregate":float(prefix_at_switch_A[i]),
                        "fast_prefix_cumulative_energy":float(prefix_at_switch_E[i]),
                        "safe_tail_cumulative_aggregate":float(tail_A[i]),
                        "safe_tail_cumulative_energy":float(tail_E[i]),
                        "safe_tail_resource_sq_sum":float(tail_q_sq[i]),
                        "safe_tail_requests":count,"safe_tail_target":tail_target,
                        "safe_tail_delta":distance,"safe_tail_budget":budget,
                        "safe_tail_certificate_lhs":h*distance})
        summaries.append(row)
    return {"metadata":{
        "benchmark":"vector_resource_reactive_original_budget","config":asdict(config),
        "independent_seeds":seeds.tolist(),"representative_seeds":sorted(representatives),
        "sample_stride":sample_stride,"methods":list(METHODS),
        "epsilon":EPSILON,"response_threshold":config.response_threshold,"beta":config.beta,
        "learner_affine_dimension":1,"payoff_dimension":post+2,"opponent_dimension":post,
        "known_payoff_lipschitz_bound":max(EPSILON*np.sqrt(post+1),(config.gamma0-config.gamma1)*np.sqrt(post)) if post else 0.,
        "opponent_geometry":"conv{0,e_1,...,e_M} in its original Euclidean coordinates",
        "payoff":"((1-p)*epsilon*r, (1-p)*epsilon*ell, p*(gamma0-(gamma0-gamma1)*r))",
        "response":"pure active iff r>gamma0/(2*epsilon+gamma0-gamma1); inactive on a tie",
        "response_weights":"Equal positive weights on all coordinates; normalization does not change the minimizer",
        "full_target":"{0} before any request; thereafter resource coordinates nonnegative, A=sum(resource), E>=0, 2*A+E<=beta, supported on observed resources",
        "full_target_includes_interior_mixtures":True,"vertex_payoff_norm_bound":1.,
        "safe_base":"unchanged original K=2 SafeBlockLearner, exact sparse state recurrence",
        "safe_budget":"B0(h)=6*h^(3/4)","G_T":G,
        "fast_oracles":"exact origin projection of each fresh unit resource; exact energy-direction saddle with observed-origin dual tie break",
        "fast_dual_tie_rule":"Select the observed origin among dual saddle optima; this is a fixed valid oracle choice, not a claim that every generic LP solver selects the same optimum",
        "alternate_fast_dual_tie_rule":"fast_recent_saddle selects an observed request after the first one; otherwise origin. Its benchmark energy witness is gamma1 after a past request and zero before; projected-current origin energy is gamma0*p",
        "safe_compression":"aggregate/energy dual coordinates plus full resource-dual squared norm; disjoint new-resource block updates preserve the original Euclidean ball projection",
        "environment_information":"past capacities only; current action and private learner state are not read",
        "environment_memory":config.memory,"same_environment_rule_and_innovations":True,
        "same_realized_path_across_methods":config.mode in ("exogenous","stationary"),
        "per_policy_histories":True,"master_restarts":False,"fresh_safe_tail_after_crossing":True,
        "phase_change_starts":H+1 if config.mode!="stationary" and post else None,
        "curve_distances":"all-round analytic lower/upper enclosures of the FULL strict target; prechange exact; terminal full-target KKT projection with support/feasibility diagnostics",
        "physical_weighted_cost":"2*aggregate deficit + provisioning cost, equal unnormalized component weights",
        "request_trigger":"Causal heuristic: off after first uniform action until the first request, then fixed balanced_capacity=beta/gamma0; no safety certificate claimed",
        "rng":"default_rng(SeedSequence([base_entropy,scenario_id,episode_seed])); same theta and uniforms across methods",
        "limitations":["Constructed large-resource mechanism model, not a real-world data population.",
            "Reactive policies generate different paths and strict targets; paired common innovations do not make paths identical.",
            "The fixed catalog has truly distinct resource-deficit coordinates; large realized dimension is deliberate, with no low-dimensional rate or q=4 validation.",
            "The fast dynamics depend on the disclosed origin dual-saddle tie rule; other valid saddle tie rules can behave differently.",
            "All-round distance enclosures are oracle uncertainty, not sampling confidence intervals.",
            "Provisioning cost models idle reservation waste and loaded overhead, not measured total electrical energy.",
            "The target distance, unmet demand, provisioning cost and per-round payoff norm are different outcomes.",
            "Finite floating arithmetic is checked numerically, not formally certified exact arithmetic."]},
        "summaries":summaries,"dynamics":dynamics,"representatives":reps}


# Retain the spelling used by the study runner during parallel implementation.
run_resource_validation_batch = run_resource_batch
