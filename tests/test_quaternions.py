
import numpy as np
import pytest

from astra_geometry.quaternions import (
    is_unit_quaternion,
    normalize_quaternion,
)


@pytest.mark.parametrize(
    "q",
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [-1.0, 0.0, 0.0, 0.0],
    ],
)
def test_is_unit_quaternion_accepts_unit_inputs(q):
    assert is_unit_quaternion(q)


@pytest.mark.parametrize(
    "q",
    [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, np.nan],
        [1.0, 0.0, 0.0, np.inf],
        [1.0 + 0.0j, 0.0, 0.0, 0.0],
    ],
)
def test_is_unit_quaternion_rejects_invalid_inputs(q):
    assert not is_unit_quaternion(q)


def test_normalize_quaternion_returns_unit_norm():
    q = normalize_quaternion([2.0, 2.0, 0.0, 0.0])

    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(q, [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0])


def test_normalize_quaternion_preserves_orientation_sign():
    q = normalize_quaternion([-2.0, 0.0, 0.0, 0.0])

    assert np.allclose(q, [-1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "q",
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, np.nan],
        [1.0 + 0.0j, 0.0, 0.0, 0.0],
    ],
)
def test_normalize_quaternion_rejects_invalid_inputs(q):
    with pytest.raises(ValueError):
        normalize_quaternion(q)


def test_normalize_quaternion_handles_large_finite_values():
    q = normalize_quaternion([1e308, 1e308, 0.0, 0.0])

    assert np.all(np.isfinite(q))
    assert np.isclose(np.linalg.norm(q), 1.0)

