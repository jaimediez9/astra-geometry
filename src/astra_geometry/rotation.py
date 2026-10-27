"""
rotation.py — Rotaciones vectoriales y matriciales.

Alcance:
1. Validación de matrices de rotación.
2. Construcción de matrices de rotación alrededor de X, Y y Z.
3. Composición de rotaciones.
4. Aplicación de rotaciones a conjuntos de puntos.

La conversión entre matrices de rotación y cuaterniones no vive aquí: está
en quaternions.py, que importa is_rotation_matrix de este módulo. La dirección
de la dependencia es rotation.py -> quaternions.py, nunca al contrario.

Este módulo constituye una implementación experimental y una referencia
matemática. La forma de los arrays y las propiedades geométricas deben
validarse explícitamente; las anotaciones de tipos no las garantizan.
"""

import numpy as np
from numpy.typing import NDArray


def is_rotation_matrix(
    R: NDArray[np.float64],
    *,
    atol: float = 1e-8,
) -> bool:
    """Comprueba si R representa una matriz de rotación de SO(3).

    Args:
        R: Matriz candidata de forma (3, 3).
        atol: Tolerancia absoluta no negativa.

    Returns:
        True si R satisface las propiedades de SO(3); False en caso contrario.
    """
    try:
        R = np.asarray(R)
        atol = float(atol)
    except (TypeError, ValueError, OverflowError):
        return False

    if R.shape != (3, 3):
        return False

    if np.iscomplexobj(R) or not np.issubdtype(R.dtype, np.number):
        return False

    if not np.isfinite(atol) or atol < 0:
        return False

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            R = R.astype(np.float64)

        if not np.all(np.isfinite(R)):
            return False

        orthogonal = np.allclose(
            R.T @ R,
            np.eye(3),
            atol=atol,
            rtol=0.0,
        )

        if not orthogonal:
            return False

        determinant = np.linalg.det(R)

        return bool(
            np.isfinite(determinant)
            and np.isclose(
                determinant,
                1.0,
                atol=atol,
                rtol=0.0,
            )
        )

    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False




def rotation_matrix3D(axis, theta: float) -> NDArray[np.float64]:
    """
    Construye una matriz de rotación 3D mediante la fórmula de Rodrigues.

    Args:
        axis: Vector de tres componentes que define el eje de rotación.
        theta: Ángulo de rotación en radianes.

    Returns:
        Matriz de rotación 3x3 perteneciente a SO(3).

    Raises:
        ValueError: Si el eje no tiene forma (3,), es nulo o contiene
                    valores no finitos, o si theta no es finito.
    """

    """
    Validación del ángulo
    """
    try:
        theta = np.asarray(theta)

        if theta.shape != () or np.iscomplexobj(theta):
            raise ValueError("theta debe ser un escalar real")

        theta = float(theta)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("theta debe ser un escalar real")

    if not np.isfinite(theta):
        raise ValueError("theta debe ser un escalar real finito")

    try:
        axis = np.asarray(axis)

        if np.iscomplexobj(axis):
            raise ValueError("axis debe contener valores reales")

        axis = axis.astype(np.float64)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "axis debe ser convertible a un array real float64"
        ) from exc

    if axis.shape != (3,):
        raise ValueError("axis debe tener forma (3,)")

    if not np.all(np.isfinite(axis)):
        raise ValueError("axis debe contener valores finitos")

    scale = np.max(np.abs(axis))

    if scale == 0.0:
        raise ValueError("El eje de rotación no puede ser nulo")

    axis = axis / scale
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

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


def rotation_matrix2D(theta: float) -> NDArray[np.float64]:
    """Construye una matriz de rotación 2D.

    Args:
        theta: Ángulo de rotación en radianes.

    Returns:
        Matriz de rotación de forma (2, 2).

    Raises:
        ValueError: Si theta no es un escalar real finito.
    """
    try:
        theta = np.asarray(theta)

        if theta.shape != () or np.iscomplexobj(theta):
            raise ValueError("theta debe ser un escalar real")

        theta = float(theta)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("theta debe ser un escalar real finito") from exc

    if not np.isfinite(theta):
        raise ValueError("theta debe ser un escalar real finito")

    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [c, -s],
        [s, c],
    ])


def rotate_points(
    points: NDArray[np.float64], 
    axis: NDArray[np.float64], 
    theta: float,
) -> NDArray[np.float64]:
    """
    Rota puntos 3D almacenados por filas.

    Args:
        points: Array de forma (N, 3), con un punto por fila.
        axis: Eje de rotación con forma (3,).
        theta: Ángulo de rotación en radianes.

    Returns:
        Array float64 de forma (N, 3)

    Raises:
        ValueError: Si los puntos no son numéricos reales, no tienen 
                    forma (N, 3) o contienen valores no finitos.
        ValueError: Si el eje o el ángulo no son válidos.
    """
    try:
        points = np.asarray(points)
        if np.iscomplexobj(points):
            raise ValueError("points debe contener valores reales")

        points = points.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "points debe ser convertible a un array real float64"
        ) from exc

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points debe tener forma (N, 3)")

    if not np.all(np.isfinite(points)):
        raise ValueError("points debe contener valores finitos")

    R = rotation_matrix3D(axis, theta)      # Obtenemos matriz de rotación 3D para aplicarla a los puntos
    return points @ R.T                     # Aplicammos la matriz de rotación a los puntos


def compose_rotations(
    R1: NDArray[np.float64],
    R2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compone dos rotaciones 3D.
    La rotación R1 se aplica primero y R2 después.
    Args:
        R1: Primera matriz de rotación, de forma (3, 3).
        R2: Segunda matriz de rotación, de forma (3, 3).
    Returns:
        Matriz de rotación compuesta R2 @ R1.
    Raises:
        ValueError: Si alguna entrada no es una matriz de rotación válida.
    """
    if not is_rotation_matrix(R1):
        raise ValueError("R1 debe ser una matriz de rotación válida")

    if not is_rotation_matrix(R2):
        raise ValueError("R2 debe ser una matriz de rotación válida")

    return R2 @ R1


def inverse_rotation(
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calcula la inversa de una matriz de rotación 3D.
    Args:
        R: Matriz de rotación válida de forma (3, 3).
    Returns:
        Matriz inversa R.T, como array independiente.
    Raises:
        ValueError: Si R no es una matriz de rotación válida.
    """
    if not is_rotation_matrix(R):
        raise ValueError("R debe ser una matriz de rotación válida")

    return R.T.copy()
