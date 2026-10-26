"""
transforms.py — Transformaciones rígidas SE(3).

Alcance:
1. Validación de matrices de transformación homogéneas.
2. Construcción de la identidad y desde (R, t).
3. Composición de transformaciones.
4. Inversión por el atajo de la traspuesta.
5. Aplicación de transformaciones a conjuntos de puntos.

Representación: matriz homogénea 4x4, T = [[R, t], [0, 1]], con R en SO(3)
y t expresado en el marco de destino (docs/conventions.md, secciones 4 y 5).

Convención de marcos: T_{A<-B} transforma coordenadas del marco B al marco A,
p̃_A = T_{A<-B} p̃_B. Los identificadores de marco se documentan en el
docstring de cada función; no son verificables numéricamente.

La última fila [0, 0, 0, 1] pertenece a la estructura de la representación y
no al bloque de rotación aproximado: se exige exacta, sin tolerancia. atol se
aplica únicamente al bloque de rotación.

Estado: bloques 1-3 implementados (validación, constructores y accesores).
Composición, inversión y aplicación a puntos, pendientes.

Este módulo importa rotation.py y nunca al contrario.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotation import is_rotation_matrix

__all__ = [
    "compose_transforms",
    "identity_transform",
    "inverse_transform",
    "is_transformation_matrix",
    "rotation_part",
    "transform_points",
    "transformation_matrix",
    "translation_part",
]


def is_transformation_matrix(
    T: ArrayLike,
    *,
    atol: float = 1e-8,
) -> bool:
    """Comprueba si T es una transformación rígida válida de SE(3).

    La última fila debe ser exactamente [0, 0, 0, 1]. No es una propiedad
    geométrica aproximada sino estructura de la representación, así que no se
    admite tolerancia en ella: una matriz con ruido en esa fila debe repararse
    de forma explícita, no validarse. Exigirla exacta es además lo que hace
    que el producto de dos transformaciones válidas la conserve exactamente.

    atol se aplica solo al bloque de rotación (ortogonalidad y determinante),
    reutilizando la validación de SO(3).

    Devuelve False ante entradas inválidas en lugar de lanzar excepción.
    """
    try:
        T = np.asarray(T)
        atol = float(atol)
    except (TypeError, ValueError, OverflowError):
        return False

    if T.shape != (4, 4):
        return False

    if np.iscomplexobj(T) or not np.issubdtype(T.dtype, np.number):
        return False

    if not np.isfinite(atol) or atol < 0:
        return False

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            T = T.astype(np.float64)

        if not np.all(np.isfinite(T)):
            return False

        if not np.array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0])):
            return False

        return bool(is_rotation_matrix(T[:3, :3], atol=atol))

    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return False


def transformation_matrix(
    R: ArrayLike,
    t: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Construye T = [[R, t], [0, 1]] a partir de una rotación y una traslación.

    La última fila se escribe como [0, 0, 0, 1] de forma exacta.

    Args:
        R: Matriz de rotación de forma (3, 3).
        t: Vector de traslación de forma (3,).

    Returns:
        Transformación float64 de forma (4, 4), independiente de las entradas.

    Raises:
        ValueError: Si R no es una rotación de SO(3) o t no es un vector
            real finito de forma (3,).
    """
    if not is_rotation_matrix(R, atol=atol):
        raise ValueError(
            "R debe representar una matriz de rotación válida de SO(3)"
        )

    t = _as_translation(t, "t")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = t

    return T


def identity_transform() -> NDArray[np.float64]:
    """Devuelve la transformación identidad de forma (4, 4)."""
    return np.eye(4, dtype=np.float64)


def rotation_part(T: ArrayLike, *, atol: float = 1e-8) -> NDArray[np.float64]:
    """Devuelve el bloque de rotación R de forma (3, 3).

    La salida es un array independiente: modificarla no altera T.

    Raises:
        ValueError: Si T no es una transformación rígida válida.
    """
    return _as_transform(T, "T", atol=atol)[:3, :3].copy()


def translation_part(T: ArrayLike, *, atol: float = 1e-8) -> NDArray[np.float64]:
    """Devuelve la traslación t de forma (3,).

    La salida es un array independiente: modificarla no altera T.

    Raises:
        ValueError: Si T no es una transformación rígida válida.
    """
    return _as_transform(T, "T", atol=atol)[:3, 3].copy()


def compose_transforms(
    T1: ArrayLike,
    T2: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Compone dos transformaciones rígidas.

    El orden de los argumentos sigue el de compose_rotations: T1 se aplica
    primero y T2 después, de modo que se devuelve T2 @ T1, equivalente a
    T_{A<-C} = T_{A<-B} T_{B<-C} (docs/conventions.md, sección 4.1).

    Raises:
        ValueError: Si alguna entrada no es una transformación válida.
    """
    raise NotImplementedError


def inverse_transform(T: ArrayLike, *, atol: float = 1e-8) -> NDArray[np.float64]:
    """Calcula T^-1 = [[R^T, -R^T t], [0, 1]].

    El atajo por la traspuesta es válido para R en SO(3), no para una matriz
    3x3 arbitraria; el contrato de entrada se valida antes de aplicarlo
    (docs/conventions.md, sección 5).

    Returns:
        Transformación inversa, como array independiente.

    Raises:
        ValueError: Si T no es una transformación rígida válida.
    """
    raise NotImplementedError


def transform_points(
    points: ArrayLike,
    T: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Aplica T a puntos 3D almacenados por filas.

    Implementa P' = P R^T + t para P de forma (N, 3), coherente con la
    convención de almacenamiento de NumPy de la sección 2.1.

    Raises:
        ValueError: Si points no tiene forma (N, 3) o T no es válida.
    """
    raise NotImplementedError


def _as_transform(
    T: ArrayLike,
    nombre: str,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Convierte T en un array float64 de forma (4, 4) validado.

    Helper privado: distingue los fallos estructurales (forma, tipo real,
    finitud) de la violación del contrato de SE(3), que delega en
    is_transformation_matrix. Es la semilla de la validación compartida que
    se trasladará a _validation.py junto con _as_quaternion.

    Raises:
        ValueError: Si T no es una transformación rígida válida.
    """
    try:
        T = np.asarray(T)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if np.iscomplexobj(T):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        T = T.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if T.shape != (4, 4):
        raise ValueError(f"{nombre} debe tener forma (4, 4)")

    if not np.all(np.isfinite(T)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    if not is_transformation_matrix(T, atol=atol):
        raise ValueError(
            f"{nombre} debe representar una transformación rígida válida de SE(3)"
        )

    return T


def _as_translation(
    t: ArrayLike,
    nombre: str,
) -> NDArray[np.float64]:
    """Convierte t en un array float64 de forma (3,) validado.

    Helper privado: comprueba forma, tipo real y finitud.

    Raises:
        ValueError: Si t no es un vector real finito de forma (3,).
    """
    try:
        t = np.asarray(t)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(t):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        t = t.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if t.shape != (3,):
        raise ValueError(f"{nombre} debe tener forma (3,)")

    if not np.all(np.isfinite(t)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return t
