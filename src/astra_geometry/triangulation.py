"""
triangulation.py — Reconstrucción 3D a partir de observaciones en varias vistas.

Alcance:
    Reconstruir puntos 3D en el marco de la cámara 1 a partir de sus
    observaciones en dos vistas con cámaras conocidas.

Método (DLT lineal; docs/conventions.md, sección 6.3):
    Con la cámara 1 como referencia, las matrices de proyección son
        P_1 = K_1 [ I | 0 ],    P_2 = K_2 [ R | t ]
    donde (R, t) es la transformación T_{2<-1} (sección 6.2). Cada observación
    aporta dos ecuaciones lineales independientes sobre las coordenadas
    homogéneas X~ del punto, de modo que dos vistas dan las cuatro ecuaciones
    que fijan X~ salvo escala. La solución es el vector singular asociado al
    menor valor singular de la matriz A (4x4) construida con las dos vistas.

Signo y profundidad (sección 6.3):
    La solución homogénea está definida salvo escala, pero su SIGNO es
    significativo: tras dividir por la coordenada homogénea, la tercera
    coordenada es la profundidad a lo largo del eje óptico, y "profundidad
    positiva en ambas cámaras" es el requisito físico que usa la quiralidad
    (sección 6.2). Esta función NO impone ese signo: lo devuelve.

Degeneración:
    Dos casos se RECHAZAN (ValueError) en lugar de devolver un resultado sin
    sentido:

    * Rayos (casi) paralelos: el punto está (casi) en el infinito y la
      coordenada homogénea se anula.
    * Centros ópticos coincidentes: las filas de A son combinaciones de filas
      de P, y P·C = 0, así que el centro común de las dos cámaras pertenece al
      núcleo de A. El rango de A cae de 3 a 2 y la solución deja de ser única:
      el vector singular menor puede ser el propio centro de proyección, cuya
      profundidad es nula. Se detecta con un criterio de RANGO RELATIVO
      (véase _RANK_RATIO), no con una tolerancia absoluta.

Este módulo importa camera.py y transforms.py, y nunca al contrario.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .camera import _as_intrinsic
from .transforms import _as_transform

__all__ = [
    "projection_matrix",
    "triangulate",
    "triangulate_points",
]

# Umbral RELATIVO de rango para la triangulación DLT (véase _triangulate_one).
# Es distinto de `atol` en semántica: `atol` es absoluto y gobierna la
# coordenada homogénea del punto en el infinito sobre un vector de norma
# unitaria, mientras que este umbral es adimensional y se aplica al cociente
# entre valores singulares de A. Justificación numérica en _triangulate_one.
_RANK_RATIO = 1e-12


def projection_matrix(
    K: ArrayLike,
    T_cam_from_world: ArrayLike,
) -> NDArray[np.float64]:
    """Construye P = K [R | t] a partir de los intrínsecos y una pose SE(3).

    Args:
        K: Matriz intrínseca de forma (3, 3).
        T_cam_from_world: Transformación rígida (4, 4) que lleva del mundo a la
            cámara. Con T = [[R, t], [0, 1]] se obtiene P = K [R | t].

    Returns:
        Matriz de proyección float64 de forma (3, 4).

    Raises:
        ValueError: Si K no es una matriz intrínseca válida o T no es un SE(3).
    """
    K = _as_intrinsic(K, "K")
    T = _as_transform(T_cam_from_world, "T_cam_from_world")

    return K @ T[:3, :]


def triangulate_points(
    pixels1: ArrayLike,
    pixels2: ArrayLike,
    P1: ArrayLike,
    P2: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Triangula por DLT lineal puntos observados en dos cámaras.

    Las coordenadas devueltas están expresadas en el marco de la cámara 1
    (el de P1).

    Args:
        pixels1: Observaciones de forma (N, 2) en la cámara 1, en píxeles.
        pixels2: Observaciones de forma (N, 2) en la cámara 2, en píxeles.
        P1: Matriz de proyección (3, 4) de la cámara 1.
        P2: Matriz de proyección (3, 4) de la cámara 2.
        atol: Umbral absoluto no negativo para detectar el punto en el
            infinito: la solución homogénea se normaliza a norma unitaria, y si
            el valor absoluto de su coordenada homogénea no lo supera, la
            reconstrucción es degenerada (rayos paralelos). Es una tolerancia
            absoluta: su rtol implícito es 0 (docs/conventions.md, sección 7.2).

    Returns:
        Puntos float64 de forma (N, 3) en el marco de la cámara 1.

    Raises:
        ValueError: Si las entradas no cumplen su contrato, si el número de
            correspondencias no coincide, o si alguna reconstrucción es
            degenerada (punto en el infinito por paralaje nulo).
    """
    x1 = _as_pixels(pixels1, "pixels1")
    x2 = _as_pixels(pixels2, "pixels2")
    atol = _as_atol(atol)

    if x1.shape[0] != x2.shape[0]:
        raise ValueError(
            "pixels1 y pixels2 deben tener el mismo número de correspondencias "
            f"({x1.shape[0]} frente a {x2.shape[0]})"
        )

    P1 = _as_projection_matrix(P1, "P1")
    P2 = _as_projection_matrix(P2, "P2")

    points = np.empty((x1.shape[0], 3), dtype=np.float64)

    for i in range(x1.shape[0]):
        u1, v1 = x1[i]
        u2, v2 = x2[i]

        # Dos filas independientes por vista (la tercera es combinación lineal).
        A = np.vstack((
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ))

        # La solución es única si y solo si rank(A) = 3, esto es, si el SEGUNDO
        # valor singular menor no es despreciable frente al mayor. Con centros
        # ópticos coincidentes el centro común pertenece al núcleo de A y el
        # rango cae a 2: Vt[-1] deja de ser la solución y puede ser el propio
        # centro de proyección, de profundidad nula.
        #
        # El criterio es RELATIVO a sigma_0 porque la escala de A depende de la
        # magnitud de los píxeles; un umbral absoluto cambiaría de significado al
        # cambiar K. Análisis numérico (8 correspondencias, escena a 4-9, focales
        # 760-800):
        #   * centros coincidentes (P1 == P2; rotación pura 90 y 180 grados con
        #     intrínsecos distintos): sigma_2/sigma_0 <= 7e-17, es decir ruido de
        #     redondeo; el ratio es invariante de escala (0.1x a 100x en K).
        #   * configuraciones válidas: >= 4e-7 con una baseline de 1e-6, >= 2e-4
        #     añadiendo 0.5 px de ruido, y >= 4e-3 en 200 poses aleatorias.
        # 1e-12 separa ambos regímenes con ~4 órdenes de margen por arriba y ~5
        # por abajo, y no rechaza ninguna configuración de rango pleno.
        _, singular, Vt = np.linalg.svd(A)
        homogeneous = Vt[-1]  # norma unitaria

        if not np.all(np.isfinite(homogeneous)):
            raise ValueError(
                f"triangulación no finita para la correspondencia {i}"
            )

        if singular[2] <= _RANK_RATIO * singular[0]:
            raise ValueError(
                f"triangulación degenerada en la correspondencia {i}: los centros "
                "ópticos de las dos cámaras coinciden, el sistema no tiene rango "
                "suficiente (rango 2 en lugar de 3) y la solución no es única"
            )

        if abs(homogeneous[3]) <= atol:
            raise ValueError(
                f"triangulación degenerada en la correspondencia {i}: el punto "
                "está en el infinito (rayos paralelos, paralaje nulo)"
            )

        points[i] = homogeneous[:3] / homogeneous[3]

    return points


def triangulate(
    pixels1: ArrayLike,
    pixels2: ArrayLike,
    K1: ArrayLike,
    K2: ArrayLike,
    T_2_from_1: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Triangula a partir de los intrínsecos y la pose relativa T_{2<-1}.

    Construye P1 = K1 [I | 0] y P2 = K2 [R | t] con (R, t) = T_{2<-1}, y
    triangula. Es la forma que describe docs/conventions.md, sección 6.3.

    Args:
        pixels1: Observaciones (N, 2) en la cámara 1 (marco de referencia).
        pixels2: Observaciones (N, 2) en la cámara 2.
        K1: Intrínsecos (3, 3) de la cámara 1.
        K2: Intrínsecos (3, 3) de la cámara 2.
        T_2_from_1: Transformación rígida (4, 4) que lleva del marco 1 al 2.

    Returns:
        Puntos float64 de forma (N, 3) en el marco de la cámara 1.

    Raises:
        ValueError: Bajo las mismas condiciones que projection_matrix y
            triangulate_points.
    """
    T_2_from_1 = _as_transform(T_2_from_1, "T_2_from_1")

    P1 = projection_matrix(K1, np.eye(4))
    P2 = projection_matrix(K2, T_2_from_1)

    return triangulate_points(pixels1, pixels2, P1, P2, atol=atol)


def _as_pixels(pixels: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte pixels en un array float64 de forma (N, 2) validado.

    Helper privado, en línea con los de camera.py y estimation.py. Forma parte
    de la validación que se trasladará a _validation.py (trabajo pendiente).

    Raises:
        ValueError: Si pixels no es un conjunto real finito de forma (N, 2).
    """
    try:
        pixels = np.asarray(pixels)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(pixels):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        pixels = pixels.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"{nombre} debe tener forma (N, 2)")

    if not np.all(np.isfinite(pixels)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return pixels


def _as_projection_matrix(P: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte P en un array float64 de forma (3, 4) validado.

    Raises:
        ValueError: Si P no es real finito de forma (3, 4).
    """
    try:
        P = np.asarray(P)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if np.iscomplexobj(P):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        P = P.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if P.shape != (3, 4):
        raise ValueError(f"{nombre} debe tener forma (3, 4)")

    if not np.all(np.isfinite(P)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return P


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
