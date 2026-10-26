"""Contract and property tests for SE(3) rigid transformations.

Test inputs are built with the independent Rodrigues helper in _helpers.py, so
that a function under test is never fed an input produced by the library.
"""

import numpy as np
import pytest

from _helpers import transform_from_blocks

from astra_geometry.rotation import is_rotation_matrix
from astra_geometry.transforms import (
    identity_transform,
    is_transformation_matrix,
    rotation_part,
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
