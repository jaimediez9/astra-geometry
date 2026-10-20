
import numpy as np
import pytest

from astra_geometry.quaternions import (
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
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


def test_identity_quaternion():
    q = np.array([1.0, 0.0, 0.0, 0.0])

    R = quaternion_to_rotation_matrix(q)

    np.testing.assert_allclose(R, np.eye(3), atol = 1e-12)
    assert R.dtype == np.float64

def test_quarter_turn_about_positive_z():
    s = np.sqrt(0.5)
    q = np.array([s, 0.0, 0.0, s])

    R = quaternion_to_rotation_matrix(q)

    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    np.testing.assert_allclose(R, expected, atol = 1e-12)


def test_q_and_negative_q_represent_same_rotation():
    q = np.array([0.5, 0.5, 0.5, 0.5])

    R1 = quaternion_to_rotation_matrix(q)
    R2 = quaternion_to_rotation_matrix(-q)

    np.testing.assert_allclose(R1, R2, atol=1e-12)


def test_output_is_a_valid_rotation_matrix():
    q = np.array([0.5, 0.5, 0.5, 0.5])

    R = quaternion_to_rotation_matrix(q)

    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "q",
    [
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0, 0.0],
        [np.inf, 0.0, 0.0, 0.0],
    ],
)
def test_invalid_quaternions_raise_value_error(q):
    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(q)


def test_input_is_not_modified():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    original = q.copy()

    quaternion_to_rotation_matrix(q)

    np.testing.assert_array_equal(q, original)
