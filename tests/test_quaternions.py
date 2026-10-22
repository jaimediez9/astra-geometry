
import numpy as np
import pytest

from astra_geometry.quaternions import (
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def _rotation_matrix_about(axis, theta):
    """Rodrigues rotation matrix, built independently of the library code.

    Used only to generate test inputs, so that the conversion under test is
    never fed a matrix produced by the conversion itself.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


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


# Tests for rotation_matrix_to_quaternion

def test_rotation_matrix_to_quaternion_identity():
    q = rotation_matrix_to_quaternion(np.eye(3))

    np.testing.assert_allclose(
        q, [1.0, 0.0, 0.0, 0.0], atol=1e-12
    )
    assert q.shape == (4,)
    assert q.dtype == np.float64
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("R", "expected"),
    [
        (
            np.diag([1.0, -1.0, -1.0]),
            [0.0, 1.0, 0.0, 0.0],
        ),
        (
            np.diag([-1.0, 1.0, -1.0]),
            [0.0, 0.0, 1.0, 0.0],
        ),
        (
            np.diag([-1.0, -1.0, 1.0]),
            [0.0, 0.0, 0.0, 1.0],
        ),
    ],
)
def test_rotation_matrix_to_quaternion_half_turns(R, expected):
    q = rotation_matrix_to_quaternion(R)

    np.testing.assert_allclose(q, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)


def test_rotation_matrix_to_quaternion_quarter_turn_about_z():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])

    s = np.sqrt(0.5)
    expected = [s, 0.0, 0.0, s]

    q = rotation_matrix_to_quaternion(R)

    np.testing.assert_allclose(q, expected, atol=1e-12)


@pytest.mark.parametrize(
    "axis",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    "angle",
    [
        np.pi - 1e-7,
        np.pi,
        np.pi + 1e-7,
        np.pi - 1e-12,
    ],
)
def test_rotation_matrix_to_quaternion_near_half_turn(axis, angle):
    R = _rotation_matrix_about(axis, angle)

    q = rotation_matrix_to_quaternion(R)

    assert np.all(np.isfinite(q))
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
    assert q[0] >= 0.0

    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(q),
        R,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "R",
    [
        np.zeros((3, 3)),
        np.diag([1.0, 1.0, -1.0]),  # Reflection: det = -1
        np.diag([2.0, 1.0, 1.0]),   # Non-orthogonal
        np.eye(2),
        np.eye(4),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
        np.eye(3, dtype=complex) + 1j,
    ],
)
def test_rotation_matrix_to_quaternion_rejects_invalid_matrices(R):
    with pytest.raises(ValueError):
        rotation_matrix_to_quaternion(R)


@pytest.mark.parametrize(
    "atol",
    [-1.0, np.nan, np.inf, -np.inf],
)
def test_rotation_matrix_to_quaternion_rejects_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        rotation_matrix_to_quaternion(np.eye(3), atol=atol)


def test_rotation_matrix_to_quaternion_does_not_modify_input():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    original = R.copy()

    rotation_matrix_to_quaternion(R)

    np.testing.assert_array_equal(R, original)

def test_quaternion_rotation_matrix_round_trip():
    """Converting q -> R -> q' -> R' preserves the rotation."""
    quaternions = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]),
        np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]),
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]

    for q in quaternions:
        R = quaternion_to_rotation_matrix(q)
        q_result = rotation_matrix_to_quaternion(R)
        R_result = quaternion_to_rotation_matrix(q_result)

        np.testing.assert_allclose(
            R_result,
            R,
            atol=1e-8,
            rtol=0.0,
        )

def test_quaternion_rotation_matrix_round_trip_sign_equivalence():
    """Opposite quaternion signs represent the same rotation."""
    q = np.array([0.5, 0.5, 0.5, 0.5])

    R_positive = quaternion_to_rotation_matrix(q)
    R_negative = quaternion_to_rotation_matrix(-q)

    np.testing.assert_allclose(
        R_negative,
        R_positive,
        atol=1e-8,
        rtol=0.0,
    )

    q_result = rotation_matrix_to_quaternion(R_negative)
    R_result = quaternion_to_rotation_matrix(q_result)

    np.testing.assert_allclose(
        R_result,
        R_positive,
        atol=1e-8,
        rtol=0.0,
    )


def test_rotation_matrix_to_quaternion_half_turn_equal_diagonal_entries():
    """pi about (1,1,1)/sqrt(3): the three diagonal entries are equal, so the
    branch selection is an exact tie and no branch wins on the diagonal test."""
    R = _rotation_matrix_about([1.0, 1.0, 1.0], np.pi)

    # Documents the tie: all three diagonal entries are identical.
    np.testing.assert_allclose(np.diag(R), [-1.0 / 3.0] * 3, atol=1e-12)

    q = rotation_matrix_to_quaternion(R)

    s = 1.0 / np.sqrt(3.0)
    np.testing.assert_allclose(q, [0.0, s, s, s], atol=1e-12)
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(q),
        R,
        atol=1e-12,
    )


def test_quaternion_matrix_round_trip_recovers_q_up_to_sign():
    """R -> q -> R is not enough: the quaternion itself must come back, up to
    the global sign that q and -q share (docs/conventions.md, section 3).
    """
    rng = np.random.default_rng(20261022)

    for _ in range(256):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)

        q_result = rotation_matrix_to_quaternion(
            quaternion_to_rotation_matrix(q)
        )

        assert np.isclose(np.linalg.norm(q_result), 1.0, atol=1e-12)
        assert min(
            np.max(np.abs(q_result - q)),
            np.max(np.abs(q_result + q)),
        ) < 1e-9


@pytest.mark.parametrize(
    "q",
    [
        np.array([-1.0, 0.0, 0.0, 0.0]),
        np.array([-0.5, 0.5, 0.5, 0.5]),
        np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    ],
)
def test_negative_w_quaternion_returns_positive_w_representative(q):
    """The w >= 0 rule is part of the documented output policy."""
    q_result = rotation_matrix_to_quaternion(
        quaternion_to_rotation_matrix(q)
    )

    np.testing.assert_allclose(q_result, -q, atol=1e-9)
    assert q_result[0] > 0.0


def test_half_turn_sign_is_unconstrained_when_w_is_zero():
    """At w == 0 both representatives are equally valid, so the w >= 0 rule
    cannot choose one and the sign is whatever the extraction branch yields.
    Either sign must still rebuild the same rotation."""
    R = np.diag([-1.0, -1.0, 1.0])  # pi about z

    q = rotation_matrix_to_quaternion(R)

    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    assert q[0] == 0.0
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(-q),
        R,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "atol",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        None,
        1j,
        "no-numerico",
        [1e-8],
    ],
)
def test_quaternion_to_rotation_matrix_rejects_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=atol,
        )
