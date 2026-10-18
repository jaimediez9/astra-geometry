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


