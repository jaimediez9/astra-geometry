"""
Verification tests
"""
import numpy as np
import pytest

from astra_geometry.rotation import(
    is_rotation_matrix,
    rotation_matrix3D,
    rotation_matrix2D,
    rotate_points,
)


def test_reflection_is_not_rotation():
    R = np.diag([1.0, 1.0, -1.0])
    assert not is_rotation_matrix(R)

def test_rotation_preserves_norm():
    p = np.array([[1.0, 2.0, 3.0]])
    rotated = rotate_points(p, [0, 0, 1], np.pi / 4)
    
    assert np.allclose(
        np.linalg.norm(rotated, axis=1),
        np.linalg.norm(p, axis=1),
    )

def test_rotation_around_z():
    p = np.array([[1.0, 0.0, 0.0]])
    rotated = rotate_points(p, [0, 0, 1], np.pi / 2)

    assert np.allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-8)

def test_zero_axis_is_rejected():
    with pytest.raises(ValueError):
        rotation_matrix3D([0, 0, 0], 1.0)

def test_rotation_matrix3d_belongs_to_so3():
    axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
    ]
    angles = [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]

    for axis in axes:
        for theta in angles:
            R = rotation_matrix3D(axis, theta)
            assert is_rotation_matrix(R)



"""
Identity matrix and incorrect dimensions
"""

def test_identity_is_rotation_matrix():
    assert is_rotation_matrix(np.eye(3))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (4, 4),
    (3, 2),
    (2, 3),
])
def test_invalid_matrix_shape_is_rejected(shape):
    R = np.zeros(shape)
    assert not is_rotation_matrix(R)


"""
Matrices with non-finite values
"""

@pytest.mark.parametrize("value", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_matrix_is_rejected(value):
    R = np.eye(3)
    R[0, 0] = value

    assert not is_rotation_matrix(R)

"""
Verificación de tipos
"""

@pytest.mark.parametrize("R", [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    np.eye(3, dtype=np.float32),
])
def test_valid_rotation_matrix_inputs(R):
    assert is_rotation_matrix(R)

@pytest.mark.parametrize("R", [
    "no es una matriz",
    None,
    [[1, 2], [3, 4]],
    np.eye(3, dtype=complex),
    np.full((3, 3), np.nan),
    np.full((3, 3), np.inf),
])
def test_invalid_rotation_matrix_inputs(R):
    assert not is_rotation_matrix(R)

@pytest.mark.parametrize("atol", [
    -1.0,
    np.nan,
    np.inf,
    -np.inf,
])
def test_invalid_tolerance_is_rejected(atol):
    assert not is_rotation_matrix(np.eye(3), atol=atol)


"""
Axis and angle validation
"""

@pytest.mark.parametrize("axis", [
    [1.0, 2.0],
    [1.0, 2.0, 3.0, 4.0],
    [np.nan, 0.0, 1.0],
    [np.inf, 0.0, 1.0],
])
def test_invalid_axis_is_rejected(axis):
    with pytest.raises(ValueError):
        rotation_matrix3D(axis, 1.0)


@pytest.mark.parametrize("theta", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_angle_is_rejected(theta):
    with pytest.raises(ValueError):
        rotation_matrix3D([0.0, 0.0, 1.0], theta)


@pytest.mark.parametrize("axis", [
    [1e308, 1e308, 1e308],
    [1e-300, 0.0, 0.0],
])
def test_rotation_matrix3d_handles_extreme_axes(axis):
    R = rotation_matrix3D(axis, np.pi / 4)
    assert is_rotation_matrix(R)


@pytest.mark.parametrize("theta", [
    [1.0, 2.0],
    np.array([1.0]),
    1.0 + 2.0j,
    "invalid",
])
def test_rotation_matrix3d_rejects_invalid_angles(theta):
    with pytest.raises(ValueError):
        rotation_matrix3D([0.0, 0.0, 1.0], theta)


"""
2D Rotations
"""

@pytest.mark.parametrize("theta", [
    0.0,
    np.pi / 4,
    np.pi / 2,
    np.pi,
    -np.pi / 3
])
def tets_rotation_matrix2d_is_valid(theta):
    R = rotation_matrix2D(theta)

    assert R.shape == (2, 2)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-8)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)

def test_rotation_matrix2d_rejects_non_finite_angle():
    with pytest.raises(ValueError):
        rotation_matrix2D(np.nan)

"""
Points validation
"""

@pytest.mark.parametrize("points", [
    np.array([1.0, 2.0, 3.0]),
    np.zeros((3, 2)),
    np.array([[np.nan, 0.0, 1.0]]),
    np.array([[np.inf, 0.0, 1.0]]),
    np.array([[1.0 + 2.0j, 0.0, 1.0]]),
    [["a", "b", "c"]],
])
def test_invalid_points_are_rejected(points):
    with pytest.raises(ValueError):
        rotate_points(points, [0.0, 0.0, 1.0], 1.0)

def test_rotate_points_accepts_empty_array():
    points = np.empty((0, 3))
    rotated = rotate_points(points, [0.0, 0.0, 1.0], 1.0)

    assert rotated.shape == (0, 3)
    assert rotated.dtype == np.float64

def test_rotate_points_preserves_pairwise_distances():
    points = np.array([
        [1.0, 2.0, 3.0],
        [-2.0, 1.0, 0.0],
        [0.5, -1.0, 4.0],
    ])

    rotated = rotate_points(points, [1.0, 2.0, 3.0], np.pi / 3)

    original_distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :],
        axis = 2,
    )
    rotated_distances = np.linalg.norm(
        rotated[:, None, :] - rotated[None, :, :],
        axis = 2,
    )

    assert np.allclose(rotated_distances, original_distances, atol=1e-8)
