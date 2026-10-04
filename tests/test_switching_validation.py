"""Independent geometry/original-base checks for stochastic validation."""
from dataclasses import replace

import numpy as np
import pytest

from lowdim_games.geometry import project_convex_hull
from lowdim_games.learners import paper_safe_budget, solve_saddle_lp
from lowdim_games.switching_validation import (
    METHODS, HeterogeneousCapacityFastLearner, ValidationConfig, _BatchSafeState,
    capacity_response, capacity_tensor, compress_capacity, generate_capacity_path,
    make_capacity_safe, run_validation_batch,
)


def test_four_corner_compression_preserves_all_payoffs_and_strict_target():
    rng = np.random.default_rng(71)
    a = rng.uniform(.5,1,size=70)
    b = a*rng.uniform(0,1,size=70)
    q = compress_capacity(a,b)
    np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-14)
    tensor = capacity_tensor()
    for p in (.0,.2,.5,.9,1.):
        payoff = np.einsum("i,ijd,nj->nd",[1-p,p],tensor,q)[:,0]
        np.testing.assert_allclose(payoff,a*p-b,atol=3e-16)
    # Interior mixtures of distinct contexts matter: response is ratio of
    # mixed coefficients, not a mixture of per-context response ratios.
    mixed = rng.dirichlet(np.ones(len(q))) @ q
    response = capacity_response(mixed)
    target = np.einsum("i,ijd,j->d",response,tensor,mixed)
    np.testing.assert_allclose(target,0,atol=3e-16)
    assert np.max(np.linalg.norm(tensor,axis=2)) <= 1


@pytest.mark.parametrize("T",[1,2,17,31,65,119])
def test_scalar_batch_safe_is_original_safe_on_arbitrary_weighted_paths(T):
    rng = np.random.default_rng(74)
    a = rng.uniform(.5,1,size=(3,T))
    b = a*rng.uniform(0,1,size=(3,T))
    batch = _BatchSafeState(np.full(3,T,dtype=int))
    originals = [make_capacity_safe(T) for _ in range(3)]
    for t in range(T):
        actions = batch.choose()
        original_actions = [original.choose()[1] for original in originals]
        np.testing.assert_allclose(actions,original_actions,rtol=0,atol=2e-14)
        payoff = batch.observe(a[:,t],b[:,t])
        records = [original.observe(compress_capacity(a[i,t],b[i,t]))
                   for i,original in enumerate(originals)]
        np.testing.assert_allclose(payoff,[record["payoff"][0] for record in records],atol=2e-14)
        np.testing.assert_allclose(batch.direction,[original.direction[0] for original in originals],atol=2e-14)


@pytest.mark.parametrize("labels",[
    [0,1,1,2,0,3,3,4], [1,2,2,3,0,4,1,0], [1,1,1,2,3,4,0,5],
])
def test_exact_fast_geometry_and_saddle_match_independent_dense_oracles(labels):
    rng = np.random.default_rng(18)
    dimension = max(labels)
    vertices = np.vstack([np.zeros(dimension),np.eye(dimension)])
    a_coeff = np.r_[1.,rng.uniform(.5,1,size=dimension)]
    b_coeff = np.r_[0.,a_coeff[1:]*rng.uniform(0,1,size=dimension)]
    fast = HeterogeneousCapacityFastLearner(len(labels))
    history = []
    direction = 0.0
    residual_sum = 0.0
    for t,label in enumerate(labels,1):
        p = fast.choose()
        if t > 1:
            observed = np.asarray(history)
            projected = project_convex_hull(vertices[label],vertices[observed],tol=1e-10)
            assert projected.success
            projected_a = 1+(a_coeff[1:]-1) @ projected.point
            projected_b = b_coeff[1:] @ projected.point
            matrix = direction*np.stack([-b_coeff[observed],a_coeff[observed]-b_coeff[observed]])
            saddle = solve_saddle_lp(matrix,tol=1e-10)
            assert saddle.success
            np.testing.assert_allclose(p,saddle.p[1],atol=1e-8)
            residual = abs((a_coeff[label]*p-b_coeff[label])-(projected_a*p-projected_b))
            residual_sum += residual
            direction = np.clip(direction+(projected_a*p-projected_b)/(2*np.sqrt(t)),-1,1)
        record = fast.observe(label,a_coeff[label],b_coeff[label])
        if t > 1:
            np.testing.assert_allclose(record["h_t"],projected.distance,atol=2e-7)
            np.testing.assert_allclose([record["projected_a"],record["projected_b"]],
                                       [projected_a,projected_b],atol=2e-7)
            np.testing.assert_allclose(fast.direction,direction,atol=2e-7)
            np.testing.assert_allclose(fast.cumulative_residual,residual_sum,atol=2e-7)
        assert record["saddle_gap"] <= 1e-12
        history.append(label)


def test_negative_direction_saddle_uses_minimum_a_minus_b():
    fast = HeterogeneousCapacityFastLearner(5)
    for label,a,b in [(0,1.,0.),(1,.8,.7),(2,.6,.1)]:
        fast.choose()
        fast.observe(label,a,b)
    fast.direction = -.2
    assert fast.choose() == 1
    result = fast.observe(2,.6,.1)
    assert result["saddle_a"] == .8
    assert result["saddle_b"] == .7
    assert result["saddle_gap"] == 0


def test_repeated_vertices_do_not_change_geometric_projection_weights():
    fast = HeterogeneousCapacityFastLearner(10)
    for label,a,b in [(1,.5,.1),(1,.5,.1),(1,.5,.1),(2,1.,.8)]:
        fast.choose()
        fast.observe(label,a,b)
    fast.choose()
    result = fast.observe(3,.7,.4)
    np.testing.assert_allclose([result["projected_a"],result["projected_b"]],[.75,.45])
    np.testing.assert_allclose(result["h_t"],np.sqrt(1.5))


def test_seed_namespaces_are_reproducible_and_independent_cells():
    config = ValidationConfig(T=90,change_round=20)
    first = generate_capacity_path(config,9)
    again = generate_capacity_path(config,9)
    other = generate_capacity_path(replace(config,scenario_id=1),9)
    np.testing.assert_array_equal(first["a"],again["a"])
    assert first["path_sha256"] == again["path_sha256"]
    assert first["path_sha256"] != other["path_sha256"]
    assert not np.array_equal(first["a"],other["a"])
    # Seeding a batch does not depend on order or which other episodes run.
    out = run_validation_batch(config,[7,9],representative_seeds=[9],sample_stride=17)
    single = run_validation_batch(config,[9],representative_seeds=[9],sample_stride=17)
    assert out["summaries"][1] == single["summaries"][0]
    for method in METHODS:
        np.testing.assert_array_equal(out["representatives"]["9"]["methods"][method]["payoff"],
                                      single["representatives"]["9"]["methods"][method]["payoff"])


def test_full_resolution_batch_matches_reference_and_reports_all_metrics():
    config = ValidationConfig(T=97,change_round=30)
    result = run_validation_batch(config,[31],sample_stride=16,representative_seeds=[31])
    rep = result["representatives"]["31"]
    path = rep["path"]
    fast = HeterogeneousCapacityFastLearner(config.T)
    safe = make_capacity_safe(config.T)
    for index,(label,a,b) in enumerate(zip(path["labels"],path["a"],path["b"])):
        assert fast.choose() == rep["methods"]["shared_past_hull"]["capacity"][index]
        record = fast.observe(label,a,b)
        np.testing.assert_allclose(record["cumulative_residual"],
                                   rep["methods"]["shared_past_hull"]["E"][index],atol=1e-13)
        action = safe.choose()[1]
        np.testing.assert_allclose(action,rep["methods"]["block_safe"]["capacity"][index],atol=2e-13)
        safe.observe(compress_capacity(a,b))
    summary = result["summaries"][0]
    for method in METHODS:
        log = rep["methods"][method]
        metrics = summary["methods"][method]
        assert metrics["pre_mean_delta"] == pytest.approx(log["delta"][:config.change_round].mean(),abs=1e-14)
        assert metrics["post_mean_delta"] == pytest.approx(log["delta"][config.change_round:].mean(),abs=1e-14)
        assert metrics["terminal_delta"] == pytest.approx(abs(log["payoff"].sum())/config.T,abs=1e-14)
        np.testing.assert_allclose(result["dynamics"][method]["delta"][0],
                                   log["delta"][result["dynamics"]["t"]-1],atol=1e-14)
    assert config.change_round+1 in result["dynamics"]["t"]


def test_original_budget_crossing_and_fresh_safe_tail_reference():
    config = ValidationConfig(T=2048,change_round=8,pre_r_low=0,pre_r_high=0,
                              post_a_low=1,post_a_high=1,post_r_low=1,post_r_high=1)
    result = run_validation_batch(config,[23],sample_stride=32,representative_seeds=[23])
    summary = result["summaries"][0]
    G = paper_safe_budget(config.T,1)
    expected = config.change_round+int(np.floor(G))+1
    assert summary["G_T"] == G
    assert summary["switch_round"] == expected
    assert summary["safe_rounds"] == config.T-expected
    master = result["representatives"]["23"]["methods"]["one_switch"]
    fast = result["representatives"]["23"]["methods"]["shared_past_hull"]
    np.testing.assert_array_equal(master["capacity"][:expected],fast["capacity"][:expected])
    assert not master["safe_mode"][expected-1]
    assert master["safe_mode"][expected]
    assert master["capacity"][expected] == .5
    assert master["safe_local_t"][expected] == 1
    assert master["E"][expected-2] <= G < master["E"][expected-1]
    np.testing.assert_array_equal(master["E"][expected:],np.full(config.T-expected,master["E"][expected-1]))
    assert np.isnan(master["residual"][expected:]).all()
    original = make_capacity_safe(config.T-expected)
    for index in range(expected,config.T):
        np.testing.assert_allclose(original.choose()[1],master["capacity"][index],atol=2e-13)
        original.observe(compress_capacity(1.,1.))
    assert abs(master["payoff"][expected:].sum()) <= paper_safe_budget(config.T-expected,1)


def test_stationary_no_change_no_switch_and_none_post_metrics():
    config = ValidationConfig(T=128,change_round=128,scenario_id=3)
    result = run_validation_batch(config,[3,4],representative_seeds=[3])
    for summary in result["summaries"]:
        assert summary["switch_round"] is None
        assert summary["safe_rounds"] == 0
        for method in METHODS:
            assert summary["methods"][method]["post_mean_delta"] is None
            assert summary["methods"][method]["post_mean_absolute_error"] is None
    assert result["metadata"]["phase_change_starts"] is None
    rep = result["representatives"]["3"]["methods"]
    np.testing.assert_array_equal(rep["one_switch"]["capacity"],rep["shared_past_hull"]["capacity"])


def test_actions_before_observation_do_not_depend_on_current_or_future_coefficients():
    fast = HeterogeneousCapacityFastLearner(5)
    for label,a,b in [(0,1,0),(1,.8,.1),(1,.8,.1)]:
        fast.choose()
        fast.observe(label,a,b)
    first = fast.choose()
    second = fast.choose()
    assert first == second
    # The next unseen vertex can have any valid coefficients without affecting
    # a choice already made. Its observation can affect only subsequent choices.
    result = fast.observe(2,.5,.49)
    assert result["capacity"] == first


@pytest.mark.parametrize("kwargs",[{"change_round":1},{"change_round":65537},{"a_min":0},
                                    {"post_a_low":.4},{"post_r_high":1.1},{"scenario_id":-1}])
def test_invalid_protocol_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ValidationConfig(**kwargs)
