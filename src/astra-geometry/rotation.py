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
    es_ortogonal = np.allclose(
        R.T @ R, 
        np.eye(3), 
        atol=atol,
        rtol=0-0,
    )
    if not es_ortogonal:
        return False
    det_es_uno = np.isclose(
        np.linalg.det(R), 
        1.0, 
        atol=atol,
        rtol=0.0,
    )
    return bool(det_es_uno)

'''
Pruebas de verificación
'''

def test_identity_is_rotation():
    assert is_rotation_matrix(np.eye(3))

def test_reflection_is_not_rotation():
    assert not is_rotation_matrix(np.diag([1.0],[1.0],[-1.0]))

def test_wrong_shape_is_rejected():
    assert not is_rotation_matrix(np.eye(4))

def test_nan_is_rejected():
    R = np.eye(3)
    R[0,0] = np.nan
    assert not is_rotation_matrix(R)


