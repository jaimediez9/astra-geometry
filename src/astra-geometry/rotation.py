"""
rotation.py — Rotaciones vectoriales y matriciales.

Alcance:
1. Validación de matrices de rotación.
2. Construcción de matrices de rotación alrededor de X, Y y Z.
3. Composición de rotaciones.
4. Aplicación de rotaciones a conjuntos de puntos.
5. Conversión entre matrices de rotación y cuaterniones.

Este módulo constituye una implementación experimental y una referencia
matemática. La forma de los arrays y las propiedades geométricas deben
validarse explícitamente; las anotaciones de tipos no las garantizan.
"""

import numpy as np
from numpy.typing import NDArray


def is_rotation_matrix(
    R: NDArray[np.float64],
    atol: float = 1e-8,
) -> bool:
    """Comprueba si una matriz satisface las propiedades de SO(3)."""
    if(R.shape != (3,3)):
        return False

    if np.iscomplexobj(R) or not np.issubdtype(R.dtype, np.number):
        return False
    
    if not np.all(np.isfinite(R)):
        return False

    if not np.isfinite(atol) or atol < 0:
        return False

    if not np.allclose(R.T @ R, np.eye(3), atol=atol, rtol=0.0):
        return False

    return bool(
        np.isclose(np.linalg.det(R), 1.0, atol=atol, rtol=0.0)
    )



def rotation_matrix3D(axis, theta: float) -> NDArray[np.float64]:
    """
    Devuelve la matriz de rotación asociada a una rotación alrededor de un eje dado 'axis' con un ángulo dado 'theta'
    Utilizamos la forma de conversión de cuaterniones a matrices de rotación para obtener la matriz de rotación.
    """
    axis = np.asarray(axis, dtype=np.float64)

    if axis.shape != (3,):
        raise ValueError("axis debe tener forma (3,)")

    if not np.all(np.isfinite(axis)) or not np.isfinite(theta):
        raise ValueError("axis y theta deben contener valores finitos")

    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("El eje de rotación no puede ser nulo")

    x, y, z = axis / norm 

    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    I = np.eye(3)

    return (
        I 
        + np.sin(theta) * K 
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def rotation_matrix2D(theta):
    """
    Construye una rotación 2D; theta se expresa en radianes
    """
    if not np.isfinite(theta):
        raise ValueError("theta debe tener un valor finito")

    c, s = np.cos(theta), np.sin(theta)
    
    return np.array([[c, -s],
                     [s, c]
    ])

def rotate_points(points, axis, theta):
    """
    Rota puntos 3D almacenados como filas, con forma (N, 3)
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points debe tener forma (N, 3)")

    if not np.all(np.infinite(points)):
        raise ValueError("Los puntos deben contener valores finitos")

    R = rotation_matrix3D(axis, theta)      # Obtenemos matriz de rotación 3D para aplicarla a los puntos
    return points @ R.T                     # Aplicammos la matriz de rotación a los puntos



"""
Pruebas de verificación
"""

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
    try:
        rotation_matrix3D([0, 0, 0], 1.0)
    except ValueError:
        pass 
    else:
        raise AssertionError("Se esperaba ValueError")
