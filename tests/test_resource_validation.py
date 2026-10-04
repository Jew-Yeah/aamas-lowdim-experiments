"""Independent dense-game, sparse-safe and closed-loop timing checks."""
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from lowdim_games.game import FiniteGame
from lowdim_games.geometry import project_convex_hull
from lowdim_games.learners import SafeBlockLearner, paper_safe_budget, solve_saddle_lp
from lowdim_games.resource_target import exact_target_projection
from lowdim_games.resource_validation import (
    EPSILON, METHODS, ResourceValidationConfig, _SparseResourceSafeState,
    arrival_probability, generate_resource_innovations, resource_response,
    resource_tensor, run_resource_batch,
)


def test_fixed_vector_game_normalization_response_and_known_lipschitz_bound():
    config = ResourceValidationConfig(T=20,change_round=8)
    M = config.T-config.change_round
    tensor = resource_tensor(M,config)
    assert tensor.shape == (2,M+1,M+2)
    np.testing.assert_allclose(np.linalg.norm(tensor[0,1:],axis=1),1,atol=2e-16)
    assert np.max(np.linalg.norm(tensor,axis=2)) <= 1+2e-16
    game = FiniteGame(tensor)
    rng = np.random.default_rng(76)
    for ell in rng.dirichlet(np.ones(M+1),size=50):
        np.testing.assert_array_equal(resource_response(ell,config),game.response(ell))
    # Every resource has a genuinely different vector payoff coordinate.
    assert len(np.unique(tensor[0,1:],axis=0)) == M
    tangent0 = np.vstack([EPSILON*np.ones(M),EPSILON*np.eye(M),np.zeros(M)])
    tangent1 = np.zeros_like(tangent0)
    tangent1[-1] = config.gamma1-config.gamma0
    expected = max(EPSILON*np.sqrt(M+1),(config.gamma0-config.gamma1)*np.sqrt(M))
    assert max(np.linalg.norm(tangent0,2),np.linalg.norm(tangent1,2)) == pytest.approx(expected)


def test_full_target_includes_response_at_unobserved_boundary_mixtures():
    config = ResourceValidationConfig(T=16,change_round=8)
    tensor = resource_tensor(4,config)
    game = FiniteGame(tensor,oracle_tol=1e-10)
    opponent = np.eye(5)[[0,1,3]]
    point = np.zeros(6)
    point[-1] = config.beta
    full = game.target_projection(point,opponent,tol=1e-9)
    assert full.success and full.distance < 2e-8
    endpoints = np.array([game.response_payoff(ell) for ell in opponent])
    surrogate = project_convex_hull(point,endpoints)
    assert surrogate.distance == pytest.approx(config.beta-config.gamma1,abs=1e-12)
    # The explicit target resource corner is the response payoff at an
    # unobserved interior origin/request mixture on the benchmark boundary.
    interior = (1-config.response_threshold)*opponent[0]+config.response_threshold*opponent[1]
    corner = tensor[0].T @ interior
    corner_expected = np.zeros(6)
    corner_expected[[0,1]] = config.beta/2
    np.testing.assert_allclose(corner,corner_expected,atol=1e-15)


@pytest.mark.parametrize("horizons",[[1,1],[2,3],[17,31],[65,97]])
def test_sparse_safe_exactly_matches_original_dense_safe(horizons):
    config = ResourceValidationConfig(T=max(horizons)+3,change_round=2)
    sparse = _SparseResourceSafeState(np.array(horizons),config)
    tensor = resource_tensor(max(horizons),config)
    dense = [SafeBlockLearner(tensor,lambda ell:resource_response(ell,config),h) for h in horizons]
    rng = np.random.default_rng(900)
    for index in range(max(horizons)):
        mask = index < np.array(horizons)
        requests = rng.random(len(horizons)) < .65
        p = sparse.choose()
        for i in np.flatnonzero(mask):
            np.testing.assert_allclose(p[i],dense[i].choose()[1],rtol=0,atol=4e-14)
        sparse.observe(requests,mask)
        for i in np.flatnonzero(mask):
            ell = np.eye(max(horizons)+1)[index+1 if requests[i] else 0]
            dense[i].observe(ell)
            np.testing.assert_allclose(sparse.direction_A[i],dense[i].direction[0],atol=4e-14)
            np.testing.assert_allclose(sparse.direction_E[i],dense[i].direction[-1],atol=4e-14)
            np.testing.assert_allclose(sparse.resource_direction_sq[i],np.sum(dense[i].direction[1:-1]**2),atol=4e-14)


def test_fast_origin_dual_tie_is_valid_but_different_ties_change_directions():
    config = ResourceValidationConfig(T=32,change_round=8)
    tensor = resource_tensor(2,config)
    direction = np.zeros(4)
    direction[-1] = config.gamma0/(4*np.sqrt(2))
    matrix = np.einsum("ijd,d->ij",tensor,direction)
    saddle = solve_saddle_lp(matrix)
    assert saddle.success and saddle.p[1] == 0
    # In ORIGINAL opponent R^2, the fresh unit projects to the origin.
    projection = project_convex_hull(np.array([0.,1.]),np.array([[0.,0.],[1.,0.]]))
    np.testing.assert_array_equal(projection.point,[0.,0.])
    assert projection.distance == 1
    for column in (0,1):
        lower = min(matrix[:,column])
        upper = max(matrix[0])
        assert upper-lower == 0  # both dual choices are exact saddle optima
    origin_target = tensor[0,0]
    request_target = tensor[1,1]
    assert np.linalg.norm(origin_target) == 0
    assert np.linalg.norm(request_target) == config.gamma1
    # Thus the disclosed origin tie is substantive, not interchangeable with
    # a claim that every generic LP implementation produces the same fast run.
    assert not np.array_equal(direction-origin_target,direction-request_target)


def test_recent_saddle_control_matches_independent_exact_dual_and_update():
    config = ResourceValidationConfig(T=129,change_round=8,mode='exogenous',exogenous_probability=1.)
    out = run_resource_batch(config,[830],sample_stride=17,representative_seeds=[830])
    log = out['representatives']['830']['methods']['fast_recent_saddle']
    original = out['representatives']['830']['methods']['shared_past_hull']
    lam = 0.
    past_gamma = [config.gamma0]
    E = 0.
    previous_request = False
    for index in range(config.T):
        t = index+1
        p = 0. if lam>0 else 1. if lam<0 else .5
        assert log['capacity'][index] == p
        assert log['direction_energy'][index] == pytest.approx(lam,abs=2e-14)
        if t>1:
            chosen_gamma = config.gamma1 if previous_request else config.gamma0
            # Original geometry projects current origin or a fresh unit to
            # origin. Both exact matrix-game primal and chosen-dual gaps are
            # independently recomputed on previously observed columns.
            B = lam*np.vstack([np.zeros(len(past_gamma)),past_gamma])
            upper = np.max(np.array([1-p,p]) @ B)
            lower = min(0.,lam*chosen_gamma)
            assert upper-lower == pytest.approx(0.,abs=1e-14)
            witness = config.gamma1 if previous_request else 0.
            lam = float(np.clip(lam+(config.gamma0*p-witness)/(2*np.sqrt(t)),-1,1))
            residual = log['request'][index]*np.sqrt((1-p)**2+((config.gamma0-config.gamma1)*p)**2)
            E += residual
            assert log['residual'][index] == pytest.approx(residual,abs=1e-14)
        assert log['E'][index] == pytest.approx(E,abs=1e-13)
        if log['request'][index]:
            previous_request = True
            past_gamma.append(config.gamma1)
    assert np.any(log['capacity'][config.change_round:] == 1.)
    assert not np.array_equal(log['capacity'],original['capacity'])
    assert 'origin' in out['metadata']['fast_dual_tie_rule']
    assert 'request' in out['metadata']['alternate_fast_dual_tie_rule']


@pytest.mark.parametrize("mode,memory",[("reactive",1),("reactive",16),("exogenous",1),("stationary",1)])
def test_common_innovations_and_nonanticipating_own_policy_histories(mode,memory):
    config = ResourceValidationConfig(T=129,change_round=48,mode=mode,memory=memory)
    out = run_resource_batch(config,[30001,30002],sample_stride=17,representative_seeds=[30001])
    rep = out['representatives']['30001']
    innovation = generate_resource_innovations(config,30001)
    np.testing.assert_array_equal(rep['path']['uniforms'],innovation['uniforms'])
    for method in METHODS:
        log = rep['methods'][method]
        assert log['capacity'][0] == .5
        for index in range(config.T):
            if index < config.change_round:
                assert not log['request'][index] and log['request_probability'][index] == 0
            else:
                past = log['capacity'][max(0,index-memory):index].mean()
                q = arrival_probability(config,innovation['theta'],past)
                assert log['request_probability'][index] == pytest.approx(q,abs=2e-14)
                expected = innovation['uniforms'][index-config.change_round] < q
                assert log['request'][index] == expected
            expected_label = index+1-config.change_round if log['request'][index] else 0
            assert log['label'][index] == expected_label
    if mode != 'reactive':
        first = rep['methods'][METHODS[0]]['request']
        for method in METHODS:
            np.testing.assert_array_equal(first,rep['methods'][method]['request'])


def test_full_raw_statistics_and_projection_match_independent_reconstruction():
    config = ResourceValidationConfig(T=119,change_round=45)
    out = run_resource_batch(config,[81],sample_stride=13,representative_seeds=[81])
    row = out['summaries'][0]
    rep = out['representatives']['81']['methods']
    times = out['dynamics']['t']
    for method in METHODS:
        log = rep[method]
        request,p = log['request'],log['capacity']
        q = EPSILON*request*(1-p)
        energy = p*(config.gamma0-(config.gamma0-config.gamma1)*request)
        idle = config.gamma0*p*(~request)
        loaded = config.gamma1*p*request
        np.testing.assert_array_equal(log['payoff_resource'],q)
        np.testing.assert_array_equal(log['payoff_aggregate'],q)
        np.testing.assert_array_equal(log['payoff_energy'],energy)
        np.testing.assert_allclose(log['cumulative_aggregate'],np.cumsum(q),atol=1e-13)
        np.testing.assert_allclose(log['cumulative_energy'],np.cumsum(energy),atol=1e-13)
        np.testing.assert_allclose(log['resource_sq_sum'],np.cumsum(q*q),atol=1e-13)
        metric = row['methods'][method]
        assert metric['mean_aggregate_deficit'] == pytest.approx((request*(1-p)).mean())
        assert metric['mean_activation_cost'] == pytest.approx(energy.mean())
        assert metric['mean_idle_cost'] == pytest.approx(idle.mean())
        assert metric['mean_loaded_cost'] == pytest.approx(loaded.mean())
        assert metric['mean_weighted_cost'] == pytest.approx((2*q+energy).mean())
        assert metric['pre_mean_delta'] == pytest.approx(log['delta_lower'][:config.change_round].mean())
        np.testing.assert_array_equal(log['delta_lower'][:config.change_round],log['delta_upper'][:config.change_round])
        exact = exact_target_projection(q,float(q.sum()),float(energy.sum()),config.T,config.beta,
                                        observed_count=int(request.sum()))
        assert metric['terminal_delta'] == pytest.approx(exact['distance'],abs=1e-12)
        assert metric['terminal_projection_lower']-1e-12 <= exact['distance'] <= metric['terminal_projection_upper']+1e-12
        for key in ('delta_lower','delta_upper','cumulative_aggregate','cumulative_energy','observed_count'):
            np.testing.assert_array_equal(out['dynamics'][method][key][0],log[key][times-1])
        digest=hashlib.sha256()
        digest.update(np.asarray([config.T,config.change_round],dtype='<i8').tobytes())
        digest.update(np.packbits(request[config.change_round:],bitorder='little').tobytes())
        assert metric['request_path_sha256'] == digest.hexdigest()


def test_original_budget_crossing_fresh_tail_and_request_only_tail_target():
    config = ResourceValidationConfig(T=2048,change_round=8,mode='exogenous',exogenous_probability=1.)
    out = run_resource_batch(config,[52],sample_stride=64,representative_seeds=[52])
    row = out['summaries'][0]
    G = paper_safe_budget(config.T,1)
    tau = config.change_round+int(np.floor(G))+1
    assert row['G_T'] == G and row['switch_round'] == tau
    master = out['representatives']['52']['methods']['one_switch']
    fast = out['representatives']['52']['methods']['shared_past_hull']
    np.testing.assert_array_equal(master['capacity'][:tau],fast['capacity'][:tau])
    assert not master['safe_mode'][tau-1] and master['safe_mode'][tau]
    assert master['capacity'][tau] == .5 and master['safe_local_t'][tau] == 1
    assert master['E'][tau-2] <= G < master['E'][tau-1]
    np.testing.assert_array_equal(master['E'][tau:],np.full(config.T-tau,master['E'][tau-1]))
    assert np.isnan(master['residual'][tau:]).all()
    assert row['safe_tail_requests'] == config.T-tau
    assert row['safe_tail_target'].startswith('{gamma1*e_energy}')
    q = master['payoff_resource'][tau:]
    E = master['payoff_energy'][tau:]
    h = config.T-tau
    target_distance = np.sqrt((q.sum()/h)**2+(q*q).sum()/h**2+(E.sum()/h-config.gamma1)**2)
    assert row['safe_tail_delta'] == pytest.approx(target_distance,abs=1e-13)
    assert row['safe_tail_certificate_lhs'] <= row['safe_tail_budget']+1e-9
    # Re-run the actual original dense base on this fresh suffix, not the
    # compressed recurrence used by production.
    tensor = resource_tensor(h,config)
    original = SafeBlockLearner(tensor,lambda ell:resource_response(ell,config),h)
    for local,index in enumerate(range(tau,config.T)):
        np.testing.assert_allclose(original.choose()[1],master['capacity'][index],atol=2e-13)
        original.observe(np.eye(h+1)[local+1])


def test_stationary_no_switch_origin_target_and_seed_order_invariance():
    config = ResourceValidationConfig(T=64,change_round=24,mode='stationary')
    batch = run_resource_batch(config,[7,9],representative_seeds=[9])
    single = run_resource_batch(config,[9],representative_seeds=[9])
    assert batch['summaries'][1] == single['summaries'][0]
    for method in METHODS:
        np.testing.assert_array_equal(batch['representatives']['9']['methods'][method]['capacity'],
                                      single['representatives']['9']['methods'][method]['capacity'])
        metric = single['summaries'][0]['methods'][method]
        assert metric['observed_requests'] == 0
        assert metric['terminal_delta'] == pytest.approx(metric['mean_activation_cost'])
    assert single['summaries'][0]['switch_round'] is None


@pytest.mark.parametrize('kwargs',[{'change_round':1},{'change_round':262145},{'mode':'future'},
                                  {'memory':0},{'theta_low':.6},{'reaction_scale':0},
                                  {'reaction_floor':.8},{'gamma0':1.1},{'gamma1':-1},
                                  {'exogenous_probability':1.1},{'scenario_id':-1}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ResourceValidationConfig(**kwargs)
