"""Contract and property tests for SE(3) rigid transformations.

Test inputs are built with the independent Rodrigues helper in _helpers.py, so
that a function under test is never fed an input produced by the library.
"""

import numpy as np
import pytest

from _helpers import rotation_matrix_about, transform_from_blocks

from astra_geometry.rotation import is_rotation_matrix, rotate_points
from astra_geometry.transforms import (
    compose_transforms,
    identity_transform,
    inverse_transform,
    is_transformation_matrix,
    rotation_part,
    transform_points,
    transformation_matrix,
    translation_part,
)


def _sample_transform(axis=(0.0, 0.0, 1.0), angle=np.pi / 3, t=(1.0, -2.0, 0.5)):
    return transform_from_blocks(axis, angle, t)


# Tests for is_transformation_matrix

@pytest.mark.parametrize(
    "T",
    [
        np.eye(4),
        np.eye(4, dtype=np.float32),
        np.eye(4, dtype=int),
        _sample_transform(),
        _sample_transform(axis=(1.0, 2.0, -3.0), angle=-1.1, t=(0.0, 0.0, 0.0)),
        _sample_transform(axis=(1.0, 1.0, 1.0), angle=np.pi, t=(1e6, -1e6, 3.0)),
        identity_transform(),
    ],
)
def test_is_transformation_matrix_accepts_valid_inputs(T):
    assert is_transformation_matrix(T)


@pytest.mark.parametrize(
    "T",
    [
        np.eye(3),
        np.eye(5),
        np.zeros((4, 4)),
        np.diag([1.0, 1.0, -1.0, 1.0]),      # reflexion: det(R) = -1
        np.diag([-1.0, 1.0, 1.0, 1.0]),      # reflexion: det(R) = -1
        np.diag([2.0, 1.0, 1.0, 1.0]),       # R no ortogonal
        np.array([[1.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]),    # R no ortogonal
        np.eye(4, dtype=bool),
        np.eye(4, dtype=complex) + 1j,
        np.full((4, 4), np.nan),
        np.full((4, 4), np.inf),
        np.array([["a"] * 4] * 4),
        np.array([[None] * 4] * 4, dtype=object),
    ],
)
def test_is_transformation_matrix_rejects_invalid_inputs(T):
    assert not is_transformation_matrix(T)


@pytest.mark.parametrize(
    "delta",
    [1e-6, 1e-9, 1e-12, np.finfo(float).eps],
)
def test_is_transformation_matrix_rejects_any_noise_in_the_last_row(delta):
    """The last row is structure of the representation, not an approximated
    geometric property: no tolerance is admitted, not even a loose one."""
    T = _sample_transform()

    for index in (0, 1, 2):
        noisy = T.copy()
        noisy[3, index] = delta
        assert not is_transformation_matrix(noisy)
        assert not is_transformation_matrix(noisy, atol=1e-3)

    noisy = T.copy()
    noisy[3, 3] = 1.0 - delta
    assert not is_transformation_matrix(noisy)
    assert not is_transformation_matrix(noisy, atol=1e-3)


def test_is_transformation_matrix_accepts_signed_zero_in_the_last_row():
    """-0.0 == 0.0 in IEEE, so a signed zero is the same structure."""
    T = _sample_transform()
    T[3, 0] = -0.0
    T[3, 1] = -0.0

    assert is_transformation_matrix(T)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["rotation", "translation", "last_row"])
def test_is_transformation_matrix_rejects_non_finite_entries(bad, where):
    T = _sample_transform()

    if where == "rotation":
        T[0, 1] = bad
    elif where == "translation":
        T[1, 3] = bad
    else:
        T[3, 2] = bad

    assert not is_transformation_matrix(T)


def test_is_transformation_matrix_atol_applies_only_to_the_rotation_block():
    """A rotation slightly off SO(3) passes only with a tolerance that covers
    it; the last row is unaffected by atol either way."""
    T = _sample_transform()
    T[0, 1] += 1e-5

    assert not is_transformation_matrix(T)
    assert is_transformation_matrix(T, atol=1e-4)


@pytest.mark.parametrize(
    "atol",
    [-1.0, np.nan, np.inf, -np.inf, None, 1j, "no-numerico"],
)
def test_is_transformation_matrix_rejects_invalid_tolerance(atol):
    assert not is_transformation_matrix(np.eye(4), atol=atol)


# Tests for transformation_matrix

def test_transformation_matrix_builds_the_expected_layout():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([1.0, 2.0, 3.0])

    T = transformation_matrix(R, t)

    expected = np.array([
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    np.testing.assert_array_equal(T, expected)
    assert T.shape == (4, 4)
    assert T.dtype == np.float64


def test_transformation_matrix_last_row_is_exact():
    T = transformation_matrix(
        _sample_transform()[:3, :3],
        _sample_transform()[:3, 3],
    )

    assert np.array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_transformation_matrix_passes_the_predicate():
    assert is_transformation_matrix(_sample_transform())
    assert is_transformation_matrix(transformation_matrix(np.eye(3), [1, 2, 3]))


def test_transformation_matrix_does_not_alias_its_inputs():
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([1.0, 2.0, 3.0])
    original_R = R.copy()
    original_t = t.copy()

    T = transformation_matrix(R, t)
    T[:3, :3] = np.eye(3)
    T[:3, 3] = np.zeros(3)

    np.testing.assert_array_equal(R, original_R)
    np.testing.assert_array_equal(t, original_t)


@pytest.mark.parametrize(
    "R",
    [
        np.diag([1.0, 1.0, -1.0]),      # reflexion
        np.diag([2.0, 1.0, 1.0]),       # no ortogonal
        np.zeros((3, 3)),
        np.eye(2),
        np.eye(4),
        np.full((3, 3), np.nan),
        np.eye(3, dtype=complex) + 1j,
        "no es una matriz",
    ],
)
def test_transformation_matrix_rejects_invalid_rotation(R):
    with pytest.raises(ValueError):
        transformation_matrix(R, [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "t",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        np.zeros((3, 1)),
        [1.0, 2.0, np.nan],
        [1.0, 2.0, np.inf],
        [1.0 + 0.0j, 2.0, 3.0],
        ["a", "b", "c"],
    ],
)
def test_transformation_matrix_rejects_invalid_translation(t):
    with pytest.raises(ValueError):
        transformation_matrix(np.eye(3), t)


# Tests for identity_transform

def test_identity_transform_is_the_exact_identity():
    T = identity_transform()

    assert np.array_equal(T, np.eye(4))
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    assert np.array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_identity_transform_passes_the_predicate():
    assert is_transformation_matrix(identity_transform())


def test_identity_transform_returns_independent_arrays():
    first = identity_transform()
    second = identity_transform()

    first[0, 0] = 0.5

    assert second[0, 0] == 1.0


# Tests for rotation_part and translation_part

def test_rotation_part_returns_the_rotation_block():
    T = _sample_transform(axis=(1.0, 2.0, 3.0), angle=0.7, t=(4.0, 5.0, 6.0))

    R = rotation_part(T)

    np.testing.assert_allclose(R, T[:3, :3], atol=0.0)
    assert R.shape == (3, 3)
    assert R.dtype == np.float64
    assert is_rotation_matrix(R)


def test_translation_part_returns_the_translation():
    T = _sample_transform(axis=(1.0, 2.0, 3.0), angle=0.7, t=(4.0, 5.0, 6.0))

    t = translation_part(T)

    np.testing.assert_array_equal(t, np.array([4.0, 5.0, 6.0]))
    assert t.shape == (3,)
    assert t.dtype == np.float64


def test_accessors_return_independent_copies():
    T = _sample_transform()
    original = T.copy()

    R = rotation_part(T)
    t = translation_part(T)
    R[0, 0] = 0.0
    t[0] = 0.0

    np.testing.assert_array_equal(T, original)


def test_accessors_accept_list_input():
    T = _sample_transform().tolist()

    assert rotation_part(T).shape == (3, 3)
    assert translation_part(T).shape == (3,)


@pytest.mark.parametrize(
    "T",
    [
        np.eye(3),
        np.eye(5),
        np.zeros((4, 4)),
        np.diag([1.0, 1.0, -1.0, 1.0]),
        np.full((4, 4), np.nan),
        np.eye(4, dtype=complex) + 1j,
    ],
)
def test_accessors_reject_invalid_transforms(T):
    with pytest.raises(ValueError):
        rotation_part(T)

    with pytest.raises(ValueError):
        translation_part(T)


def test_accessors_reject_a_noisy_last_row():
    T = _sample_transform()
    T[3, 0] = 1e-12

    with pytest.raises(ValueError):
        rotation_part(T)

    with pytest.raises(ValueError):
        translation_part(T)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_accessors_reject_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        rotation_part(_sample_transform(), atol=atol)

    with pytest.raises(ValueError):
        translation_part(_sample_transform(), atol=atol)


# Tests for compose_transforms

def test_compose_transforms_agrees_with_block_composition():
    """The decisive property: composing transforms must equal composing the
    blocks by hand, including which transform is applied first.

    R1, R2 and the expected translation are built with the independent
    Rodrigues helper, so this checks the composition and its argument order,
    not the constructor.
    """
    rng = np.random.default_rng(20261026)

    for _ in range(64):
        axis1 = rng.normal(size=3)
        axis1 = axis1 / np.linalg.norm(axis1)
        angle1 = rng.uniform(-np.pi, np.pi)
        t1 = rng.normal(size=3)

        axis2 = rng.normal(size=3)
        axis2 = axis2 / np.linalg.norm(axis2)
        angle2 = rng.uniform(-np.pi, np.pi)
        t2 = rng.normal(size=3)

        T1 = transform_from_blocks(axis1, angle1, t1)
        T2 = transform_from_blocks(axis2, angle2, t2)

        R1 = rotation_matrix_about(axis1, angle1)
        R2 = rotation_matrix_about(axis2, angle2)

        expected = np.eye(4)
        expected[:3, :3] = R2 @ R1
        expected[:3, 3] = t2 + R2 @ t1

        result = compose_transforms(T1, T2)

        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0.0)
        assert result.dtype == np.float64
        assert np.array_equal(result[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_compose_transforms_is_not_commutative():
    T_z = transform_from_blocks([0.0, 0.0, 1.0], np.pi / 2, (0.0, 0.0, 0.0))
    T_x = transform_from_blocks([1.0, 0.0, 0.0], np.pi / 2, (1.0, 0.0, 0.0))

    T_zx = compose_transforms(T_z, T_x)
    T_xz = compose_transforms(T_x, T_z)

    assert not np.allclose(T_zx, T_xz, atol=1e-9)

    # compose_transforms(T_z, T_x) = T_x @ T_z
    R_z = rotation_matrix_about([0.0, 0.0, 1.0], np.pi / 2)
    R_x = rotation_matrix_about([1.0, 0.0, 0.0], np.pi / 2)

    np.testing.assert_allclose(T_zx[:3, :3], R_x @ R_z, atol=1e-12)
    np.testing.assert_allclose(T_zx[:3, 3], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(T_xz[:3, 3], np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_compose_transforms_with_the_identity():
    T = _sample_transform()
    I = identity_transform()

    np.testing.assert_array_equal(compose_transforms(T, I), T)
    np.testing.assert_array_equal(compose_transforms(I, T), T)


def test_compose_transforms_is_associative():
    rng = np.random.default_rng(20261027)

    for _ in range(32):
        T1 = transform_from_blocks(rng.normal(size=3), rng.uniform(-1, 1), rng.normal(size=3))
        T2 = transform_from_blocks(rng.normal(size=3), rng.uniform(-1, 1), rng.normal(size=3))
        T3 = transform_from_blocks(rng.normal(size=3), rng.uniform(-1, 1), rng.normal(size=3))

        left = compose_transforms(compose_transforms(T1, T2), T3)
        right = compose_transforms(T1, compose_transforms(T2, T3))

        np.testing.assert_allclose(left, right, atol=1e-12)


def test_compose_transforms_result_passes_the_predicate():
    T1 = _sample_transform(axis=(1.0, 2.0, 3.0), angle=0.4, t=(1.0, 0.0, -1.0))
    T2 = _sample_transform(axis=(-1.0, 0.5, 2.0), angle=-1.2, t=(0.0, 3.0, 0.5))

    assert is_transformation_matrix(compose_transforms(T1, T2))


def test_compose_transforms_preserves_the_last_row_exactly_over_a_long_chain():
    """The last row stays exact because the validated inputs have it exact.
    The rotation block may drift, which is why the chained result is checked
    with a looser tolerance than the default."""
    step = _sample_transform(axis=(1.0, 2.0, 3.0), angle=0.01, t=(0.001, 0.0, 0.0))

    result = identity_transform()
    for _ in range(100):
        result = compose_transforms(result, step)

    assert np.array_equal(result[3], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.all(np.isfinite(result))
    assert is_transformation_matrix(result, atol=1e-6)


def test_compose_transforms_does_not_modify_inputs():
    T1 = _sample_transform()
    T2 = _sample_transform(axis=(0.0, 1.0, 0.0), angle=0.9, t=(2.0, 2.0, 2.0))
    original1 = T1.copy()
    original2 = T2.copy()

    compose_transforms(T1, T2)

    np.testing.assert_array_equal(T1, original1)
    np.testing.assert_array_equal(T2, original2)


@pytest.mark.parametrize(
    "T1, T2",
    [
        (np.zeros((4, 4)), np.eye(4)),
        (np.eye(4), np.zeros((4, 4))),
        (np.eye(3), np.eye(4)),
        (np.eye(4), np.eye(3)),
        (np.diag([1.0, 1.0, -1.0, 1.0]), np.eye(4)),
        (np.eye(4), np.eye(4, dtype=complex) + 1j),
        (np.full((4, 4), np.nan), np.eye(4)),
    ],
)
def test_compose_transforms_rejects_invalid_inputs(T1, T2):
    with pytest.raises(ValueError):
        compose_transforms(T1, T2)


def test_compose_transforms_rejects_a_noisy_last_row():
    T = _sample_transform()
    T[3, 1] = 1e-12

    with pytest.raises(ValueError):
        compose_transforms(T, identity_transform())


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_compose_transforms_rejects_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        compose_transforms(identity_transform(), identity_transform(), atol=atol)


# Tests for inverse_transform

def test_inverse_transform_round_trip_on_both_sides():
    T = _sample_transform(axis=(1.0, 2.0, 3.0), angle=0.8, t=(4.0, -1.0, 2.0))
    T_inverse = inverse_transform(T)
    I = identity_transform()

    np.testing.assert_allclose(compose_transforms(T, T_inverse), I, atol=1e-12)
    np.testing.assert_allclose(compose_transforms(T_inverse, T), I, atol=1e-12)


def test_inverse_transform_matches_numpy_inverse():
    """np.linalg.inv is an independent reference: for a genuine SE(3) element
    the transpose shortcut must agree with a general matrix inverse."""
    rng = np.random.default_rng(20261028)

    for _ in range(32):
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        T = transform_from_blocks(axis, rng.uniform(-np.pi, np.pi), rng.normal(size=3))

        np.testing.assert_allclose(
            inverse_transform(T),
            np.linalg.inv(T),
            atol=1e-12,
        )


def test_inverse_transform_of_a_pure_translation():
    T = transform_from_blocks([0.0, 0.0, 1.0], 0.0, (1.0, 2.0, 3.0))

    np.testing.assert_allclose(
        inverse_transform(T)[:3, 3],
        np.array([-1.0, -2.0, -3.0]),
        atol=1e-15,
    )


def test_inverse_transform_of_a_pure_rotation():
    T = transform_from_blocks([1.0, 1.0, 0.0], 0.6, (0.0, 0.0, 0.0))

    T_inverse = inverse_transform(T)

    np.testing.assert_allclose(T_inverse[:3, :3], T[:3, :3].T, atol=1e-15)
    np.testing.assert_allclose(T_inverse[:3, 3], np.zeros(3), atol=1e-15)


def test_inverse_transform_is_an_involution():
    T = _sample_transform(axis=(-2.0, 1.0, 0.5), angle=-0.9, t=(3.0, 0.0, -2.0))

    np.testing.assert_allclose(inverse_transform(inverse_transform(T)), T, atol=1e-12)


def test_inverse_transform_is_independent_and_keeps_the_exact_last_row():
    T = _sample_transform()
    T_inverse = inverse_transform(T)

    assert np.array_equal(T_inverse[3], np.array([0.0, 0.0, 0.0, 1.0]))
    assert T_inverse.dtype == np.float64

    T_inverse[:3, :3] = np.eye(3)

    np.testing.assert_allclose(T[:3, :3], rotation_matrix_about([0.0, 0.0, 1.0], np.pi / 3), atol=1e-15)


def test_inverse_transform_does_not_modify_input():
    T = _sample_transform()
    original = T.copy()

    inverse_transform(T)

    np.testing.assert_array_equal(T, original)


def test_inverse_transform_rejects_matrices_outside_so3():
    """Section 5: the transpose shortcut must never be applied to a matrix
    whose rotation block is not in SO(3), even when the shape and the last
    row look right."""
    scaled = np.eye(4)
    scaled[:3, :3] = np.diag([2.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        inverse_transform(scaled)

    reflection = np.eye(4)
    reflection[:3, :3] = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError):
        inverse_transform(reflection)


@pytest.mark.parametrize(
    "T",
    [
        np.zeros((4, 4)),
        np.eye(3),
        np.eye(5),
        np.full((4, 4), np.inf),
        np.eye(4, dtype=complex) + 1j,
    ],
)
def test_inverse_transform_rejects_invalid_inputs(T):
    with pytest.raises(ValueError):
        inverse_transform(T)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_inverse_transform_rejects_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        inverse_transform(identity_transform(), atol=atol)


# Tests for transform_points

def test_transform_points_matches_manual_computation():
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    R = rotation_matrix_about([0.0, 0.0, 1.0], np.pi / 2)
    t = np.array([1.0, 2.0, 3.0])
    T = transform_from_blocks([0.0, 0.0, 1.0], np.pi / 2, t)

    result = transform_points(points, T)

    np.testing.assert_allclose(result, points @ R.T + t, atol=1e-15)
    assert result.shape == points.shape
    assert result.dtype == np.float64


def test_transform_points_matches_rotate_points_plus_translation():
    """Cross-check against rotation.py: with the same rotation block, the
    application must agree with rotate_points plus the translation."""
    rng = np.random.default_rng(20261029)
    points = rng.normal(size=(20, 3))

    axis = np.array([1.0, -2.0, 0.5])
    angle = 1.1
    t = np.array([0.5, -1.0, 2.0])
    T = transform_from_blocks(axis, angle, t)

    np.testing.assert_allclose(
        transform_points(points, T),
        rotate_points(points, axis, angle) + t,
        atol=1e-12,
    )


def test_transform_points_preserves_distances():
    rng = np.random.default_rng(20261030)
    points = rng.normal(size=(12, 3))
    T = _sample_transform(axis=(1.0, -1.0, 2.0), angle=1.3, t=(5.0, -3.0, 1.0))

    moved = transform_points(points, T)

    before = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    after = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=-1)

    np.testing.assert_allclose(after, before, atol=1e-12)


def test_transform_points_with_the_identity_is_the_identity():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])

    np.testing.assert_array_equal(transform_points(points, identity_transform()), points)


def test_transform_points_single_point_and_empty_input():
    single = np.array([[1.0, 2.0, 3.0]])
    assert transform_points(single, identity_transform()).shape == (1, 3)

    empty = np.zeros((0, 3))
    result = transform_points(empty, _sample_transform())

    assert result.shape == (0, 3)
    assert np.all(np.isfinite(result))


def test_transform_points_inverse_round_trip():
    rng = np.random.default_rng(20261031)
    points = rng.normal(size=(10, 3))
    T = _sample_transform(axis=(0.5, 1.0, -0.5), angle=-0.7, t=(2.0, 1.0, -1.0))

    moved = transform_points(points, T)
    back = transform_points(moved, inverse_transform(T))

    np.testing.assert_allclose(back, points, atol=1e-12)


def test_transform_points_does_not_modify_inputs():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    original_points = points.copy()
    T = _sample_transform()
    original_T = T.copy()

    transform_points(points, T)

    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(T, original_T)


def test_transform_points_accepts_lists():
    result = transform_points([[1, 2, 3]], identity_transform())

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 2)),
        np.zeros((3, 4)),
        np.zeros(3),
        np.zeros((2, 2, 3)),
        np.zeros((0,)),
        np.array([[1.0, 2.0, np.nan]]),
        np.array([[1.0, 2.0, np.inf]]),
        np.array([[1.0, 2.0, -np.inf]]),
        np.array([[1.0 + 0.0j, 2.0, 3.0]]),
        np.array([["a", "b", "c"]]),
        5.0,
    ],
)
def test_transform_points_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        transform_points(points, identity_transform())


@pytest.mark.parametrize(
    "T",
    [
        np.zeros((4, 4)),
        np.eye(3),
        np.diag([1.0, 1.0, -1.0, 1.0]),
        np.full((4, 4), np.nan),
    ],
)
def test_transform_points_rejects_invalid_transform(T):
    with pytest.raises(ValueError):
        transform_points(np.zeros((2, 3)), T)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf])
def test_transform_points_rejects_invalid_tolerance(atol):
    with pytest.raises(ValueError):
        transform_points(np.zeros((2, 3)), identity_transform(), atol=atol)
