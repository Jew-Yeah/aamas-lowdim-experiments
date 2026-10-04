"""Independent full-hull LP and dense-QP checks of the analytical target."""
import numpy as np
import pytest
from scipy.optimize import minimize

from lowdim_games.game import FiniteGame
from lowdim_games.resource_target import exact_target_projection, target_bounds


EPSILON = 1/np.sqrt(2)
GAMMA0, GAMMA1 = 1.0, 0.2
RSTAR = GAMMA0/(2*EPSILON+GAMMA0-GAMMA1)
BETA = 2*EPSILON*RSTAR


def independent_game(n):
    # Opponent simplex coordinate zero is the no-demand action. All loss
    # tensor entries are built from the physical model, without the target
    # equations under test. Equal weights implement its stated benchmark.
    tensor = np.zeros((2, n+1, n+2))
    tensor[1, 0, -1] = GAMMA0
    for i in range(n):
        tensor[0, i+1, 0] = EPSILON
        tensor[0, i+1, i+1] = EPSILON
        tensor[1, i+1, -1] = GAMMA1
    return FiniteGame(tensor, oracle_tol=1e-9)


def dense_projection(w, E):
    A = w.sum()
    def objective(v):
        return (v[:-1].sum()-A)**2+np.sum((v[:-1]-w)**2)+(v[-1]-E)**2
    def gradient(v):
        return np.r_[2*(v[:-1].sum()-A)+2*(v[:-1]-w), 2*(v[-1]-E)]
    initial = np.r_[w, E]*min(1.0, 0.8*BETA/max(2*A+E, 1e-30))
    result = minimize(objective, initial, jac=gradient, method="SLSQP",
                      bounds=[(0.0, None)]*(len(w)+1),
                      constraints={"type": "ineq", "fun": lambda v: BETA-2*v[:-1].sum()-v[-1],
                                   "jac": lambda v: np.r_[np.full(len(w), -2.0), -1.0]},
                      options={"ftol": 1e-12, "maxiter": 500})
    assert result.success, result.message
    return result


@pytest.mark.parametrize("seen", [[0], [0, 1], [0, 1, 3], [0, 1, 2, 3]])
def test_full_response_cell_support_matches_simplex_including_unseen_mixtures(seen):
    game = independent_game(3)
    points = np.eye(4)[seen]
    resources = [i-1 for i in seen if i != 0]
    expected = [np.zeros(game.d)]
    if resources:
        energy = np.zeros(game.d); energy[-1] = BETA
        expected.append(energy)
        for i in resources:
            v = np.zeros(game.d); v[0] = BETA/2; v[i+1] = BETA/2
            expected.append(v)
    expected = np.asarray(expected)
    rng = np.random.default_rng(315+len(seen))
    for direction in rng.normal(size=(25, game.d)):
        support = game.target_support(direction, points)
        analytic = float(np.max(expected@direction))
        assert support.value == pytest.approx(analytic, abs=3e-10)
        assert support.upper_value == pytest.approx(analytic, abs=3e-10)


def test_discontinuous_response_requires_boundary_closure_not_only_observed_points():
    game = independent_game(2)
    points = np.eye(3)[[0, 1]]
    z = np.array([1-RSTAR, RSTAR, 0.0])
    # At the tie the inactive response owns the aggregate/resource corner.
    np.testing.assert_allclose(game.action_payoffs(z).sum(axis=1), [BETA, BETA], atol=2e-16)
    above = z.copy(); above[0] -= 1e-8; above[1] += 1e-8
    assert game.response_index(above) == 1
    assert game.response_payoff(above)[-1] == pytest.approx(BETA, abs=1e-8)
    direction = np.array([0.0, 0.0, 0.0, 1.0])
    # Responses at the two observed actions would expose only energy GAMMA1.
    observed = np.asarray([game.response_payoff(p) for p in points])
    assert np.max(observed@direction) == GAMMA1
    assert game.target_support(direction, points).value == pytest.approx(BETA, abs=1e-12)


def test_exact_projection_matches_dense_slsqp_on_random_points():
    rng = np.random.default_rng(4471)
    for _ in range(100):
        n = int(rng.integers(1, 18))
        w = rng.random(n); w *= rng.random()/w.sum()
        E = rng.random()
        oracle = exact_target_projection(w, w.sum(), E, 1, BETA, n,
                                         return_resource_projection=True)
        dense = dense_projection(w, E)
        assert oracle["distance_squared"] == pytest.approx(dense.fun, abs=5e-11)
        assert oracle["support_gap"] < 5e-12
        assert oracle["feasibility_error"] < 5e-12
        wp = oracle["resource_projection"]
        assert wp.sum() == pytest.approx(oracle["aggregate"], abs=1e-15)
        assert 2*wp.sum()+oracle["energy"] <= BETA+5e-12


def test_projection_full_support_lp_independently_certifies_every_resource_corner():
    game = independent_game(3)
    points = np.eye(4)
    rng = np.random.default_rng(8173)
    for _ in range(15):
        w = rng.uniform(0, 0.3, 3); A = w.sum(); E = rng.random()
        out = exact_target_projection(w, A, E, 1, BETA, 3, return_resource_projection=True)
        actual = np.r_[A, w, E]
        projected = np.r_[out["aggregate"], out["resource_projection"], out["energy"]]
        residual = actual-projected
        support = game.target_support(residual, points)
        full_gap = max(0.0, 2*(support.upper_value-residual@projected))
        assert full_gap < 2e-9
        assert out["support_gap"] == pytest.approx(full_gap, abs=2e-9)
        original = game.target_projection(actual, points, tol=1e-9)
        assert out["distance"] == pytest.approx(original.distance, abs=3e-7)


@pytest.mark.parametrize("w,E", [
    ([], 0.8), ([0.0, 0.0], 0.8), ([0.2, 0.01], 0.1),
    ([0.01, 0.01], 1.0), ([0.5, 0.2, 0.001], 0.0),
    ([0.02, 0.02, 0.02], 0.01), ([BETA/8]*4, 0.0),
    ([0.2, 0.02, 0.001], 0.8),
])
def test_all_corner_and_clipping_cases(w, E):
    w = np.asarray(w)
    # Even an empty positive-value representation may have observed a fully
    # served request; that gives the energy corner, unlike the origin prefix.
    n = max(1, len(w))
    out = exact_target_projection(w, w.sum(), E, 1, BETA, n,
                                  return_resource_projection=True)
    if len(w):
        dense = dense_projection(w, E)
        assert out["distance_squared"] == pytest.approx(dense.fun, abs=5e-11)
    else:
        assert out["distance"] == pytest.approx(max(0.0, E-BETA))
    assert out["support_gap"] < 5e-12
    assert out["feasibility_error"] < 5e-12


def test_origin_only_target_is_zero_even_if_energy_below_later_beta():
    out = exact_target_projection([], 0.0, 5.0, 20, BETA, observed_count=0)
    assert out["distance"] == 0.25
    assert out["energy"] == 0.0
    bounds = target_bounds(0.0, 0.25, 0.0, 0, BETA)
    assert bounds["lower"] == bounds["upper"] == 0.25
    assert bool(bounds["exactflag"])
    assert not bool(bounds["exact_zero"])


def test_vectorized_bounds_enclose_full_target_and_upper_is_actual_feasible_point():
    rng = np.random.default_rng(19971)
    data = []
    for i in range(90):
        n = int(rng.integers(1, 30)) if i % 10 else 0
        w = rng.random(n)
        if n: w *= rng.random()/w.sum()
        data.append((w, rng.random()))
    A = np.array([w.sum() for w, E in data])
    E = np.array([E for w, E in data])
    sq = np.array([w@w for w, E in data])
    n = np.array([len(w) for w, E in data])
    bounds = target_bounds(A, E, sq, n, BETA)
    assert bounds["lower"].shape == (90,)
    for i, (w, e) in enumerate(data):
        exact = exact_target_projection(w, A[i], e, 1, BETA, n[i])
        assert bounds["lower"][i] <= exact["distance"]+1e-12
        assert exact["distance"] <= bounds["upper"][i]+1e-12
        ap, ep = bounds["candidate_A"][i], bounds["candidate_E"][i]
        if n[i]:
            wp = w*ap/A[i]
            assert wp.sum() == pytest.approx(ap)
            assert 2*ap+ep <= BETA+1e-14
        else:
            wp = w
            assert ap == ep == 0
        distance = np.linalg.norm(np.r_[ap-A[i], wp-w, ep-e])
        assert distance == pytest.approx(bounds["upper"][i], abs=2e-15)
        assert bool(bounds["exact_zero"][i]) == (n[i] > 0 and 2*A[i]+e <= BETA)


def test_histogram_exactly_matches_expanded_array_and_omitted_zero_coordinates():
    levels = np.array([0.0, 0.1, 0.37, 0.62, 0.62])
    counts = np.array([150, 71, 20, 5000, 9])
    expanded = np.repeat(levels, counts)
    total = levels@counts; t = 262144; energy = 0.9*t
    a = exact_target_projection(levels, total, energy, t, BETA,
                                observed_count=counts.sum(), counts=counts)
    b = exact_target_projection(expanded, total, energy, t, BETA, len(expanded))
    for key in ["distance", "distance_squared", "aggregate", "energy", "threshold", "support_gap"]:
        assert a[key] == pytest.approx(b[key], abs=2e-13)
    c = exact_target_projection(levels[1:], total, energy, t, BETA,
                                observed_count=counts.sum(), counts=counts[1:])
    assert c["distance"] == pytest.approx(a["distance"], abs=1e-15)


def test_large_multiplicity_histogram_preserves_finite_precision_certificates():
    levels = np.array([0.0, EPSILON/2, EPSILON])
    counts = np.array([32768, 65536, 98304])
    t = 262144; total = levels@counts
    out = exact_target_projection(levels, total, 0.4*t, t, BETA,
                                  observed_count=int(counts.sum()), counts=counts)
    expanded = exact_target_projection(np.repeat(levels, counts), total, 0.4*t,
                                       t, BETA, int(counts.sum()))
    assert out["distance"] == pytest.approx(expanded["distance"], abs=5e-13)
    assert out["support_gap"] < 5e-12
    assert expanded["support_gap"] < 5e-12
    assert out["feasibility_error"] < 5e-12


@pytest.mark.parametrize("kwargs", [
    {"beta": 0}, {"beta": np.nan}, {"t": 0}, {"t": True},
    {"raw_resource_payoffs": [-0.1]}, {"raw_resource_payoffs": [[0.1]]},
    {"cumulative_aggregate": 0.8}, {"observed_count": 0},
    {"observed_count": 0.5}, {"counts": [0.5]}, {"counts": [2, 3]},
])
def test_exact_oracle_rejects_invalid_domains(kwargs):
    inputs = dict(raw_resource_payoffs=[0.1], cumulative_aggregate=0.1,
                  cumulative_energy=0.2, t=1, beta=BETA, observed_count=1)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        exact_target_projection(**inputs)


@pytest.mark.parametrize("A,E,sq,n", [
    (-0.1, 0.2, 0.01, 1), (0.1, np.inf, 0.01, 1),
    (0.1, 0.2, 0.01, 0), (0.1, 0.2, 0.01, 0.5),
    (0.1, 0.2, 0.1, 1), (0.1, 0.2, 0.0001, 2),
])
def test_bounds_reject_impossible_summary_inputs(A, E, sq, n):
    with pytest.raises(ValueError):
        target_bounds(A, E, sq, n, BETA)
