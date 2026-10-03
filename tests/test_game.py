import numpy as np

from lowdim_games.game import FiniteGame
from lowdim_games.geometry import project_convex_hull


def crossing_game():
    # Response scores are z_0 and z_1. At their crossing, the first response
    # gives (.5,0), while the closure of the other response gives (.5,2).
    tensor = np.array([[[1.0, 0.0], [0.0, 0.0]],
                       [[0.0, 2.0], [1.0, 2.0]]])
    return FiniteGame(tensor, weights=np.array([1.0, 0.0]))


def test_full_target_contains_unobserved_mixture_response_outcomes():
    game, points = crossing_game(), np.eye(2)
    query = np.array([0.5, 1.0])
    observed_only = np.array([game.response_payoff(z) for z in points])
    wrong_proxy = project_convex_hull(query, observed_only)
    assert abs(wrong_proxy.distance - 0.5) < 1e-10
    target = game.target_vertices(points)
    expected = np.array([[0.0, 0.0], [0.0, 2.0], [0.5, 0.0], [0.5, 2.0]])
    for vertex in expected:
        assert np.min(np.linalg.norm(target - vertex, axis=1)) < 1e-8
    result = game.target_projection(query, points)
    assert result.success and result.distance < 1e-7


def test_target_projection_full_support_certificate_outside_rectangle():
    game, points = crossing_game(), np.eye(2)
    result = game.target_projection(np.array([0.75, 1.0]), points)
    assert result.success and result.gap <= 1e-9 * 3
    assert abs(result.distance - 0.25) < 1e-7
    np.testing.assert_allclose(result.point, [0.5, 1.0], atol=1e-7)
    support = game.target_support(np.array([1.0, 1.0]), points)
    assert abs(support.value - 2.5) < 1e-9
    assert support.value <= support.upper_value + 1e-12


def test_implicit_full_target_projection_discovers_unobserved_responses():
    game, points = crossing_game(), np.eye(2)
    for query in (np.array([0.5, 1.0]), np.array([0.75, 1.0])):
        explicit = game.target_projection(query, points)
        implicit = game.target_projection(query, points, enumerate_vertices=False)
        assert implicit.success and implicit.gap <= 1e-9 * 3
        assert abs(explicit.distance - implicit.distance) < 1e-7
        np.testing.assert_allclose(explicit.point, implicit.point, atol=1e-7)


def test_never_selected_tied_action_is_excluded_from_closed_target():
    # Blindly adding every weak response cell would incorrectly add y=10.
    tensor = np.array([[[0.0, 0.0], [0.0, 0.0]],
                       [[0.0, 10.0], [0.0, 10.0]]])
    game = FiniteGame(tensor, weights=np.array([1.0, 0.0]))
    assert [cell.action for cell in game.target_cells(np.eye(2))] == [0]
    np.testing.assert_allclose(game.target_vertices(np.eye(2)), [[0.0, 0.0]])
    result = game.target_projection(np.array([0.0, 5.0]), np.eye(2))
    assert result.success and abs(result.distance - 5.0) < 1e-10


def test_winner_existing_only_on_tie_hyperplane_must_be_retained():
    # Action zero wins only at the crossing, because it has the first index.
    tensor = np.array([[[0.5, 3.0], [0.5, 3.0]],
                       [[1.0, 0.0], [0.0, 0.0]],
                       [[0.0, 0.0], [1.0, 0.0]]])
    game = FiniteGame(tensor, weights=np.array([1.0, 0.0]))
    target = game.target_vertices(np.eye(2))
    assert np.min(np.linalg.norm(target - np.array([0.5, 3.0]), axis=1)) < 1e-8
    support = game.target_support(np.array([0.0, 1.0]), np.eye(2))
    assert abs(support.value - 3.0) < 1e-10


def test_degenerate_hull_uses_actual_tie_winner_and_singleton_target():
    game = crossing_game()
    points = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert game.response_index(points[0]) == 0
    assert [cell.action for cell in game.target_cells(points)] == [0]
    np.testing.assert_allclose(game.target_vertices(points), [[0.5, 0.0]])


def test_tiny_nonzero_response_score_gap_is_not_treated_as_tie():
    tensor = np.array([[[1e-14, 0.0], [1e-14, 0.0]],
                       [[0.0, 10.0], [0.0, 10.0]]])
    game = FiniteGame(tensor, weights=np.array([1.0, 0.0]))
    assert game.response_index(np.array([0.5, 0.5])) == 1
    assert [cell.action for cell in game.target_cells(np.eye(2))] == [1]
    target = game.target_vertices(np.eye(2))
    assert np.max(np.linalg.norm(target - np.array([0.0, 10.0]), axis=1)) < 1e-9


def test_random_small_target_support_agrees_with_all_region_vertices():
    random = np.random.default_rng(184)
    tensor = random.uniform(size=(4, 3, 3))
    game, points = FiniteGame(tensor), np.eye(3)
    target = game.target_vertices(points)
    for direction in random.normal(size=(8, 3)):
        support = game.target_support(direction, points)
        assert abs(np.max(target @ direction) - support.value) < 1e-7
        assert support.gap < 1e-7
    query = random.normal(size=3)
    result = game.target_projection(query, points)
    assert result.success
    assert result.lower_distance <= result.distance


def test_small_residual_target_support_dual_is_accurate_on_frozen_allocation_case():
    # Offline regression from the M=6 capacity=.8 chronological benchmark.
    # Defaults formerly yielded a 5.22e-9 support-LP gap despite a 9.48e-19
    # geometric QP gap, failing full-target projection at tolerance 1e-9.
    from lowdim_games.benchmarks import allocation_losses
    profiles = np.array([
        [3.918103448275862, 11.818965517241379, 2.2844827586206895, 14.40948275862069, 3.7327586206896552],
        [8.826923076923077, 32.48076923076923, 5.298076923076923, 35.15384615384615, 11.663461538461538],
        [11.818181818181818, 58.59090909090909, 6.909090909090909, 67.86363636363636, 17.0],
        [14.75, 135.25, 16.75, 120.0, 81.75],
        [43.5, 97.0, 19.5, 359.5, 35.0],
        [135.0, 263.0, 26.0, 542.0, 67.0],
    ])
    schedules = np.array([
        [8.0, 8.0, 8.0, 8.0, 8.0],
        [3.7453538948200866, 13.296955318307631, 2.158956109134045, 16.185053380782918, 4.613681296955318],
        [21.872676947410042, 6.648477659153816, 1.0794780545670224, 8.092526690391459, 2.306840648477659],
        [1.8726769474100433, 26.64847765915382, 1.0794780545670224, 8.092526690391459, 2.306840648477659],
        [1.8726769474100433, 6.648477659153816, 21.079478054567026, 8.092526690391459, 2.306840648477659],
        [1.8726769474100433, 6.648477659153816, 1.0794780545670224, 28.09252669039146, 2.306840648477659],
        [1.8726769474100433, 6.648477659153816, 1.0794780545670224, 8.092526690391459, 22.30684064847766],
        [4.681692368525108, 16.621194147884538, 2.698695136417556, 20.231316725978647, 5.767101621194147],
    ])
    query = np.array([0.004679732201202055, 0.01409973004766392, 0.0016565884383541087,
                      0.02169800987529138, 0.005494646500968331, 0.012588008530625793,
                      0.02084532371682511])
    tensor = allocation_losses(schedules, profiles, capacity=40.0) / 20.196422769919707
    game = FiniteGame(tensor, np.array([1.0] * 6 + [0.5]))
    points = np.eye(6)
    vertices = game.target_vertices(points)
    geometric = project_convex_hull(query, vertices, tol=1e-10)
    support = game.target_support(query - geometric.point, points)
    assert support.gap < 1e-12
    assert abs(support.value - np.max(vertices @ (query - geometric.point))) < 1e-12
    explicit = game.target_projection(query, points, tol=1e-9)
    implicit = game.target_projection(query, points, tol=1e-9, enumerate_vertices=False)
    assert explicit.success and explicit.gap <= 1e-9
    assert implicit.success and implicit.gap <= 1e-9
    assert abs(explicit.distance - 0.00015623458164897865) < 1e-8
    assert abs(explicit.distance - implicit.distance) < 1e-8
