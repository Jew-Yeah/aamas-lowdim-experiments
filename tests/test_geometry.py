import numpy as np

from lowdim_games.geometry import (
    affine_hull_polytope, polytope_vertices, project_convex_hull,
)


def test_projection_segment_and_squared_gap_certificate():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0]])
    result = project_convex_hull(np.array([0.75, 1.0]), vertices)
    assert result.success
    np.testing.assert_allclose(result.point, [0.75, 0.0], atol=1e-8)
    assert abs(result.distance - 1.0) < 1e-10
    assert result.lower_distance <= 1.0 + 1e-10 <= result.distance + 1e-10
    assert result.alpha == np.sqrt(result.gap)
    np.testing.assert_allclose(result.weights @ vertices, result.point)


def test_projection_singleton_and_square_interior():
    singleton = project_convex_hull(np.array([3.0, 4.0]), np.zeros((1, 2)))
    assert singleton.distance == 5.0 and singleton.gap == 0.0 and singleton.success
    square = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    result = project_convex_hull(np.array([0.25, 0.75]), square)
    assert result.success
    assert result.distance < 1e-8


def test_interior_mixture_projection_with_redundant_history_meets_gap_tolerance():
    from lowdim_games.benchmarks import regime_path
    path, _ = regime_path(2, 12, seed=8)
    # This formerly stopped at distance 3.9e-8 after 1500 polishing steps,
    # failing the requested 1e-9 full-hull dual-gap tolerance.
    result = project_convex_hull(path[-1], path[:-1], tol=1e-10)
    assert result.success and result.gap <= 1e-10
    assert result.distance < 1e-12
    assert result.weights.shape == (len(path) - 1,)
    np.testing.assert_allclose(result.weights @ path[:-1], result.point, atol=1e-14)


def test_hull_reduction_preserves_original_weights_and_outside_certificate():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                         [0.2, 0.2], [0.3, 0.4], [1.0, 0.0]])
    result = project_convex_hull(np.array([1.0, 1.0]), vertices, tol=1e-11)
    assert result.success and result.gap <= 2e-11
    np.testing.assert_allclose(result.point, [0.5, 0.5], atol=1e-10)
    assert result.weights.shape == (len(vertices),)
    assert abs(result.distance - np.sqrt(0.5)) < 1e-10


def test_affine_hull_embedded_segment_and_point():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    hull = affine_hull_polytope(points)
    assert hull.dimension == 1
    np.testing.assert_allclose(hull.lift(hull.coordinates), points, atol=1e-13)
    assert hull.residual < 1e-13
    point_hull = affine_hull_polytope(np.array([[0.5, 0.5], [0.5, 0.5]]))
    assert point_hull.dimension == 0


def test_vertices_include_lower_dimensional_response_cell():
    # The intersection is a segment embedded in R^2, with empty interior.
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    bound = np.array([1.0, 0.0, 0.0, 0.0])
    vertices = polytope_vertices(matrix, bound)
    np.testing.assert_allclose(vertices[np.argsort(vertices[:, 0])], [[0.0, 0.0], [1.0, 0.0]], atol=1e-9)


def test_vertices_square_and_infeasible_cell():
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    vertices = polytope_vertices(matrix, np.array([1.0, 0.0, 1.0, 0.0]))
    assert len(vertices) == 4
    assert np.max(matrix @ vertices.T - np.array([1.0, 0.0, 1.0, 0.0])[:, None]) < 1e-8
    empty = polytope_vertices(np.array([[1.0], [-1.0]]), np.array([0.0, -1.0]))
    assert empty.shape == (0, 1)
