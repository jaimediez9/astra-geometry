"""
estimation.py — Estimación de transformaciones rígidas SE(3) por SVD (Kabsch).

Alcance:
    Estimar la transformación rígida T_{B<-A} que mejor alinea, en mínimos
    cuadrados, un conjunto de puntos expresados en el marco A con sus
    correspondencias expresadas en el marco B.

Problema (Kabsch, 1976):
    Dadas N correspondencias p_i (marco A) y q_i (marco B), se busca
    (R, t) con R en SO(3) que minimice

        sum_i || q_i - (R p_i + t) ||^2

    La solución es única salvo cuando los puntos de origen están
    degenerados (colineales o coincidentes), en cuyo caso la rotación no
    queda determinada y la función lo rechaza.

Convención de almacenamiento (docs/conventions.md, sección 2.1):
    Los conjuntos de puntos se almacenan como arrays (N, 3), un punto por
    fila. Con vectores columna el modelo es q = R p + t; con puntos por fila
    el mismo modelo se escribe q_row = p_row R^T + t_row. La rotación devuelta
    es la misma en ambos casos y NO se traspone.

Matemáticas (por qué esta orientación de covarianza):
    Minimizar sum_i || q_i - R p_i ||^2 (tras centrar) equivale a maximizar
    sum_i q_i^T R p_i = trace(R H), con la matriz de covarianza cruzada

        H = sum_i p_i q_i^T = P^T Q        (3x3)

    donde P y Q son las nubes centradas, almacenadas como (N, 3). Con la SVD
    H = U S V^T, el maximizador sobre O(3) es R = V D U^T con
    D = diag(1, 1, det(V U^T)). El signo de det(V U^T) es lo que corrige el
    reflejo: si H induce una transformación con det = -1, el producto con D
    devuelve una rotación propia (det = +1). La traslación se recupera con
    t = q_bar - R p_bar.

Entrada 3D-3D (por qué no PnP ni geometría epipolar):
    Este capítulo resuelve el caso 3D-3D (dos nubes de puntos ya emparejadas).
    PnP (3D-2D, puntos 3D frente a píxeles) es un problema de minimización NO
    lineal y no se resuelve por SVD. La geometría epipolar (2D-2D) usa SVD para
    la matriz esencial, pero no devuelve directamente una transformación
    rígida: da una restricción entre vistas. Ambos quedan fuera de alcance.

Limitaciones:
    - No estima escala (solo transformaciones rígidas).
    - No maneja outliers: no hay RANSAC; una correspondencia errónea
      contamina la solución completa.
    - El criterio de degeneración es absoluto (atol), por lo que nubes de
      extensión muy pequeña pueden rechazarse aunque no sean colineales.

Este módulo importa transforms.py y nunca al contrario.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transforms import transformation_matrix

__all__ = ["estimate_rigid_transform"]

# Una rotación 3D queda determinada por correspondencias no colineales; con
# menos de tres puntos no puede haber un conjunto no colineal.
_MIN_CORRESPONDENCES = 3


def estimate_rigid_transform(
    source: ArrayLike,
    target: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Estima T_{B<-A} que alinea source (A) con target (B) por mínimos cuadrados.

    Resuelve el problema de Kabsch: minimiza sum_i || q_i - (R p_i + t) ||^2
    sobre R en SO(3), con p_i en source (marco A) y q_i en target (marco B).
    No estima escala.

    Args:
        source: Correspondencias de origen, forma (N, 3), un punto por fila,
            en el marco A.
        target: Correspondencias de destino, forma (N, 3), un punto por fila,
            en el marco B. La fila i corresponde a la fila i de source.
        atol: Umbral absoluto no negativo para el criterio de degeneración.
            Una configuración se rechaza si el segundo valor singular de las
            correspondencias de origen centradas no lo supera, es decir, si los
            puntos son colineales (o coincidentes) dentro de esa tolerancia.
            Es una tolerancia absoluta: su rtol implícito es 0
            (docs/conventions.md, sección 7.2).

    Returns:
        Transformación rígida float64 de forma (4, 4) que lleva del marco A al
        marco B: transform_points(source, T) ~= target.

    Raises:
        ValueError: Si source o target no son (N, 3) reales finitos, si tienen
            distinto número de correspondencias, si hay menos de tres
            correspondencias, si atol no es finito y no negativo, o si la
            configuración de origen es degenerada (colineal o coincidente).
    """
    source = _as_points(source, "source")
    target = _as_points(target, "target")
    atol = _as_atol(atol)

    if source.shape[0] != target.shape[0]:
        raise ValueError(
            "source y target deben tener el mismo número de correspondencias "
            f"({source.shape[0]} frente a {target.shape[0]})"
        )

    if source.shape[0] < _MIN_CORRESPONDENCES:
        raise ValueError(
            f"se necesitan al menos {_MIN_CORRESPONDENCES} correspondencias "
            f"({source.shape[0]} recibidas)"
        )

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)

    centered_source = source - source_centroid
    centered_target = target - target_centroid

    # Degeneración: la rotación solo queda determinada si el origen abarca algo
    # más que una recta. Colinealidad == rango(centered_source) < 2 == el
    # segundo valor singular es cero (dentro de atol).
    singular_values = np.linalg.svd(centered_source, compute_uv=False)
    if singular_values[1] <= atol:
        raise ValueError(
            "configuración degenerada: las correspondencias de source son "
            "colineales o coincidentes, y la rotación no queda determinada"
        )

    cross_covariance = centered_source.T @ centered_target  # H = P^T Q

    U, _, Vt = np.linalg.svd(cross_covariance)
    V = Vt.T

    # Corrección de reflexión: garantiza det(R) = +1 en lugar de -1.
    d = np.sign(np.linalg.det(V @ U.T))
    R = V @ np.diag([1.0, 1.0, d]) @ U.T

    t = target_centroid - R @ source_centroid

    return transformation_matrix(R, t)


def _as_points(points: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte points en un array float64 de forma (N, 3) validado.

    Helper privado, en línea con los de transforms.py y camera.py. Es parte de
    la validación que se trasladará a _validation.py (trabajo pendiente).

    Raises:
        ValueError: Si points no es un conjunto real finito de forma (N, 3).
    """
    try:
        points = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(points):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        points = points.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{nombre} debe tener forma (N, 3)")

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return points


def _as_atol(atol) -> float:
    """Convierte atol en un float real finito no negativo.

    Raises:
        ValueError: Si atol no es un escalar real finito y no negativo.
    """
    try:
        atol = np.asarray(atol)

        if atol.shape != () or np.iscomplexobj(atol):
            raise ValueError("atol debe ser un escalar real finito no negativo")

        atol = float(atol)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "atol debe ser un escalar real finito no negativo"
        ) from exc

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol debe ser un escalar real finito no negativo")

    return atol
