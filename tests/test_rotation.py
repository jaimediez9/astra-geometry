"""
Pruebas de verificación
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
Matriz identidad y dimensiones incorrectas
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
Matrices con valores no finitos
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
Validación del eje y ángulo
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
        assert rotation_matrix3D([0.0, 0.0, 1.0], theta)


"""
Rotaciones 2D
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
    with pytest.raise(ValueError):
        rotation_matrix2D(np.nan)
