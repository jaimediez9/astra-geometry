
import numpy as np
import pytest

from astra_geometry.quaternions import (
    conjugate_quaternion,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from _helpers import rotation_matrix_about as _rotation_matrix_about


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


def test_is_unit_quaternion_tolerance_is_keyword_only():
    with pytest.raises(TypeError):
        is_unit_quaternion([1.0, 0.0, 0.0, 0.0], 1e-8)  # type: ignore[reportCallIssue]

    assert is_unit_quaternion([1.0, 0.0, 0.0, 0.0], atol=1e-8)


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


def test_rotation_matrix_to_quaternion_tolerance_is_keyword_only():
    with pytest.raises(TypeError):
        rotation_matrix_to_quaternion(np.eye(3), 1e-8)  # type: ignore[reportCallIssue]

    rotation_matrix_to_quaternion(np.eye(3), atol=1e-8)


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


def test_quaternion_to_rotation_matrix_tolerance_is_keyword_only():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        quaternion_to_rotation_matrix(q, 1e-8)  # type: ignore[reportCallIssue]

    quaternion_to_rotation_matrix(q, atol=1e-8)


# Tests for conjugate_quaternion

def test_conjugate_quaternion_negates_the_vector_part():
    q = np.array([0.5, 0.5, 0.5, 0.5])

    q_conjugate = conjugate_quaternion(q)

    np.testing.assert_allclose(q_conjugate, [0.5, -0.5, -0.5, -0.5], atol=1e-15)
    assert q_conjugate.shape == (4,)
    assert q_conjugate.dtype == np.float64


def test_conjugate_quaternion_does_not_modify_input():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    original = q.copy()

    conjugate_quaternion(q)

    np.testing.assert_array_equal(q, original)


def test_conjugate_quaternion_is_the_inverse_for_unit_inputs():
    """docs/conventions.md, section 3: for a unit quaternion q^-1 = q*."""
    rng = np.random.default_rng(20261023)
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    for _ in range(64):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)

        product = quaternion_multiply(q, conjugate_quaternion(q))

        np.testing.assert_allclose(product, identity, atol=1e-12)


def test_conjugate_quaternion_general_case_gives_norm_squared_identity():
    """For a non-unit q, q* * q = q * q* = |q|^2, which is what makes the
    general inverse q* / |q|^2 later on."""
    q = np.array([2.0, -3.0, 0.5, 1.0])

    product = quaternion_multiply(q, conjugate_quaternion(q))

    np.testing.assert_allclose(product, [np.dot(q, q), 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "q",
    [
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, np.nan],
        [1.0, 0.0, 0.0, np.inf],
        [1.0 + 0.0j, 0.0, 0.0, 0.0],
        ["no", "numerico", 0.0, 0.0],
        np.zeros((2, 2)),
    ],
)
def test_conjugate_quaternion_rejects_invalid_inputs(q):
    with pytest.raises(ValueError):
        conjugate_quaternion(q)


# Tests for quaternion_multiply

def test_quaternion_multiply_by_the_identity():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.5, 0.5, 0.5, 0.5])

    np.testing.assert_allclose(quaternion_multiply(q, identity), q, atol=1e-15)
    np.testing.assert_allclose(quaternion_multiply(identity, q), q, atol=1e-15)


def test_two_quarter_turns_about_z_compose_to_a_half_turn():
    """Hand-computable case: q * q with q = (sqrt(.5), 0, 0, sqrt(.5))."""
    s = np.sqrt(0.5)
    q = np.array([s, 0.0, 0.0, s])

    q_composed = quaternion_multiply(q, q)

    np.testing.assert_allclose(q_composed, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(q_composed),
        np.diag([-1.0, -1.0, 1.0]),
        atol=1e-12,
    )


def test_quaternion_multiply_agrees_with_matrix_composition_order():
    """The decisive property: composing quaternions must equal composing the
    rotations, including which one is applied first.

    The matrices are built with the independent Rodrigues helper, so this
    checks the product formula and the argument order, not the conversion.
    """
    rng = np.random.default_rng(20261024)

    for _ in range(128):
        axis1 = rng.normal(size=3)
        axis1 = axis1 / np.linalg.norm(axis1)
        angle1 = rng.uniform(-np.pi, np.pi)

        axis2 = rng.normal(size=3)
        axis2 = axis2 / np.linalg.norm(axis2)
        angle2 = rng.uniform(-np.pi, np.pi)

        q1 = np.concatenate(([np.cos(angle1 / 2)], np.sin(angle1 / 2) * axis1))
        q2 = np.concatenate(([np.cos(angle2 / 2)], np.sin(angle2 / 2) * axis2))

        R1 = _rotation_matrix_about(axis1, angle1)
        R2 = _rotation_matrix_about(axis2, angle2)

        R_composed = quaternion_to_rotation_matrix(
            quaternion_multiply(q1, q2)
        )

        np.testing.assert_allclose(R_composed, R2 @ R1, atol=1e-12, rtol=0.0)

        # Conjugation reverses the order of a product: (q2 q1)* = q1* q2*.
        np.testing.assert_allclose(
            conjugate_quaternion(quaternion_multiply(q1, q2)),
            quaternion_multiply(
                conjugate_quaternion(q2),
                conjugate_quaternion(q1),
            ),
            atol=1e-12,
        )


def test_quaternion_multiply_is_not_commutative():
    s = np.sqrt(0.5)
    q_z = np.array([s, 0.0, 0.0, s])   # quarter turn about z
    q_x = np.array([s, s, 0.0, 0.0])   # quarter turn about x

    p_zx = quaternion_multiply(q_z, q_x)
    p_xz = quaternion_multiply(q_x, q_z)

    assert not np.allclose(p_zx, p_xz, atol=1e-9)

    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(p_zx),
        _rotation_matrix_about([1.0, 0.0, 0.0], np.pi / 2)
        @ _rotation_matrix_about([0.0, 0.0, 1.0], np.pi / 2),
        atol=1e-12,
    )


def test_quaternion_multiply_preserves_norm_up_to_product_of_norms():
    rng = np.random.default_rng(20261025)

    for _ in range(64):
        q1 = rng.normal(size=4)
        q2 = rng.normal(size=4)

        product = quaternion_multiply(q1, q2)

        assert np.isclose(
            np.linalg.norm(product),
            np.linalg.norm(q1) * np.linalg.norm(q2),
            atol=1e-12,
        )


def test_quaternion_multiply_is_associative():
    rng = np.random.default_rng(20261026)

    for _ in range(64):
        q1 = rng.normal(size=4)
        q1 = q1 / np.linalg.norm(q1)
        q2 = rng.normal(size=4)
        q2 = q2 / np.linalg.norm(q2)
        q3 = rng.normal(size=4)
        q3 = q3 / np.linalg.norm(q3)

        left = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        right = quaternion_multiply(q1, quaternion_multiply(q2, q3))

        np.testing.assert_allclose(left, right, atol=1e-12)


def test_quaternion_multiply_sign_equivalence():
    """q and -q are the same rotation, so flipping either factor flips the
    sign of the product but must not change the composed rotation."""
    rng = np.random.default_rng(20261027)

    for _ in range(32):
        q1 = rng.normal(size=4)
        q1 = q1 / np.linalg.norm(q1)
        q2 = rng.normal(size=4)
        q2 = q2 / np.linalg.norm(q2)

        product = quaternion_multiply(q1, q2)

        np.testing.assert_allclose(
            quaternion_multiply(-q1, q2),
            -product,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            quaternion_to_rotation_matrix(quaternion_multiply(-q1, q2)),
            quaternion_to_rotation_matrix(product),
            atol=1e-12,
        )


def test_quaternion_multiply_accepts_non_unit_inputs():
    """Documented contract: multiplication is algebraic, so unit norm is not
    required; only the shape, realness and finiteness are validated."""
    np.testing.assert_allclose(
        quaternion_multiply([2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]),
        [6.0, 0.0, 0.0, 0.0],
        atol=1e-15,
    )


def test_quaternion_multiply_returns_float64():
    product = quaternion_multiply([1, 0, 0, 0], (0.0, 1.0, 0.0, 0.0))

    assert product.shape == (4,)
    assert product.dtype == np.float64


def test_quaternion_multiply_does_not_modify_inputs():
    q1 = np.array([0.5, 0.5, 0.5, 0.5])
    q2 = np.array([0.5, -0.5, 0.5, -0.5])
    original1 = q1.copy()
    original2 = q2.copy()

    quaternion_multiply(q1, q2)

    np.testing.assert_array_equal(q1, original1)
    np.testing.assert_array_equal(q2, original2)


@pytest.mark.parametrize(
    "q1, q2",
    [
        ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0, np.nan], [1.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, np.inf]),
        ([1.0 + 0.0j, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        (["no", "numerico", 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        (np.zeros((4, 1)), [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_quaternion_multiply_rejects_invalid_inputs(q1, q2):
    with pytest.raises(ValueError):
        quaternion_multiply(q1, q2)
