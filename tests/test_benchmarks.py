import numpy as np
import pytest
from lowdim_games.benchmarks import (make_synthetic, regime_path,
                                    affine_dimension, allocation_losses)


def test_allocation_has_bounded_biaffine_tensor_and_interpretable_cost():
    instance = make_synthetic(seed=3)
    assert instance.tensor.shape == (8, 8, 7)
    assert np.max(np.linalg.norm(instance.tensor, axis=2)) <= 1 + 1e-12
    assert np.all(instance.tensor >= 0)
    assert instance.schedules[-1].sum() == pytest.approx(1.25 * instance.capacity)
    raw = allocation_losses(instance.schedules, instance.profiles, instance.capacity)
    assert np.allclose(instance.tensor * instance.scale, raw)


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_unknown_regimes_have_requested_dimension(q):
    path, regimes = regime_path(q, 40, seed=11)
    assert path.shape == (40, 8)
    assert np.all(path >= 0)
    assert np.allclose(path.sum(axis=1), 1)
    assert affine_dimension(path) == q
    assert len(regimes) == q + 1


def test_profile_and_path_randomness_are_reproducible():
    assert np.array_equal(make_synthetic(8).tensor, make_synthetic(8).tensor)
    assert np.array_equal(regime_path(4, 80, seed=9)[0], regime_path(4, 80, seed=9)[0])
