"""
camera.py — Geometría de cámara (modelo pinhole).

Alcance:
1. Validación y construcción de la matriz intrínseca K.
2. Conversión de puntos del mundo al marco de la cámara con una extrínseca SE(3).
3. Proyección 3D->2D y back-projection 2D->3D con profundidad.
4. Residual de reproyección.

Convenciones (docs/conventions.md, secciones 6, 6.1 y 7):
- K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]. La última fila y el elemento por
  debajo de la diagonal son estructura exacta (K[1, 0] = 0 y K[2, :] = [0, 0, 1]),
  sin tolerancia. El skew s puede ser no nulo.
- Proyección: lambda [u, v, 1]^T = K [X_c, Y_c, Z_c]^T, con lambda = Z_c. Los
  puntos con Z_c <= 0 (sobre o detrás del plano imagen) se rechazan.
- Píxeles (u, v) con origen en la esquina superior izquierda; u hacia la derecha,
  v hacia abajo. (cx, cy) es el punto principal, en píxeles.
- La pose extrínseca es un SE(3) T_{C<-W} que lleva coordenadas del mundo (W) al
  marco de la cámara (C): p_C = T_{C<-W} p_W.

Ninguna función toma tolerancia: todas las condiciones sobre K, los puntos y las
profundidades son estructurales o exactas; no hay ninguna propiedad geométrica
aproximada que comparar.

Este módulo importa transforms.py y nunca al contrario.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# _as_transform es el helper privado de transforms.py; se reutiliza para dar el
# mismo contrato y los mismos mensajes de error. Es, según su docstring, la
# semilla de la validación compartida que se trasladará a _validation.py.
from .transforms import _as_transform, transform_points

__all__ = [
    "backproject_pixels",
    "intrinsic_matrix",
    "is_intrinsic_matrix",
    "project",
    "project_points",
    "reprojection_residual",
    "world_to_camera",
]


def is_intrinsic_matrix(K: ArrayLike) -> bool:
    """Comprueba si K es una matriz intrínseca de cámara pinhole válida.

    Condiciones: forma (3, 3), real y finita; K[1, 0] == 0 y K[2, :] == [0, 0, 1]
    exactos (estructura de la representación); f_x > 0 y f_y > 0. El skew no se
    restringe salvo por finitud.

    No se aplica tolerancia a propósito: no hay ninguna propiedad geométrica
    aproximada que comparar, solo estructura y signos.

    Devuelve False ante entradas inválidas en lugar de lanzar excepción.
    """
    try:
        K = np.asarray(K)
    except (TypeError, ValueError):
        return False

    if K.shape != (3, 3):
        return False

    if np.iscomplexobj(K) or not np.issubdtype(K.dtype, np.number):
        return False

    try:
        K = K.astype(np.float64)
    except (TypeError, ValueError, OverflowError):
        return False

    if not np.all(np.isfinite(K)):
        return False

    if K[1, 0] != 0.0:
        return False

    if not np.array_equal(K[2], np.array([0.0, 0.0, 1.0])):
        return False

    return bool(K[0, 0] > 0.0 and K[1, 1] > 0.0)


def intrinsic_matrix(
    fx,
    fy,
    cx,
    cy,
    *,
    skew=0.0,
) -> NDArray[np.float64]:
    """Construye K = [[fx, skew, cx], [0, fy, cy], [0, 0, 1]].

    Args:
        fx: Distancia focal horizontal en píxeles; debe ser positiva.
        fy: Distancia focal vertical en píxeles; debe ser positiva.
        cx: Coordenada u del punto principal, en píxeles.
        cy: Coordenada v del punto principal, en píxeles.
        skew: Término de skew (por defecto 0.0); puede ser no nulo.

    Returns:
        Matriz intrínseca float64 de forma (3, 3).

    Raises:
        ValueError: Si algún argumento no es un escalar real finito, o si fx o
            fy no son positivos.
    """
    fx = _as_scalar(fx, "fx")
    fy = _as_scalar(fy, "fy")
    cx = _as_scalar(cx, "cx")
    cy = _as_scalar(cy, "cy")
    skew = _as_scalar(skew, "skew")

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx y fy deben ser positivos")

    return np.array(
        [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_to_camera(
    points_world: ArrayLike,
    T_cam_from_world: ArrayLike,
) -> NDArray[np.float64]:
    """Transforma puntos del mundo al marco de la cámara con un SE(3).

    Args:
        points_world: Puntos de forma (N, 3), uno por fila, en el marco del mundo.
        T_cam_from_world: Transformación rígida (4, 4) que lleva del mundo a la
            cámara.

    Returns:
        Puntos en el marco de la cámara, float64 de forma (N, 3).

    Raises:
        ValueError: Si points_world no es (N, 3) real finito o T no es un SE(3).
    """
    points_world = _as_points(points_world, "points_world", 3)
    T = _as_transform(T_cam_from_world, "T_cam_from_world")

    return transform_points(points_world, T)


def project_points(
    points_cam: ArrayLike,
    K: ArrayLike,
) -> NDArray[np.float64]:
    """Proyecta puntos del marco de la cámara al plano imagen.

    Implementa lambda [u, v, 1]^T = K [X_c, Y_c, Z_c]^T con lambda = Z_c,
    equivalente a (u, v) = (K p_C)[:2] / (K p_C)[2].

    Args:
        points_cam: Puntos de forma (N, 3), uno por fila, en el marco de la cámara.
        K: Matriz intrínseca de forma (3, 3).

    Returns:
        Coordenadas imagen float64 de forma (N, 2).

    Raises:
        ValueError: Si K no es una matriz intrínseca válida, si points_cam no es
            (N, 3) real finito, o si algún punto tiene Z_c <= 0.
    """
    K = _as_intrinsic(K, "K")
    points = _as_points(points_cam, "points_cam", 3)

    Z = points[:, 2]
    if not np.all(Z > 0.0):
        raise ValueError(
            "points_cam debe tener Z_c > 0 (puntos delante de la cámara)"
        )

    homogeneous = points @ K.T  # (N, 3) = [fx X + s Y + cx Z, fy Y + cy Z, Z]

    return homogeneous[:, :2] / homogeneous[:, 2:3]


def project(
    points_world: ArrayLike,
    K: ArrayLike,
    T_cam_from_world: ArrayLike,
) -> NDArray[np.float64]:
    """Proyecta puntos del mundo: aplica la extrínseca SE(3) y luego K.

    Equivale a project_points(world_to_camera(points_world, T), K).

    Raises:
        ValueError: Bajo las mismas condiciones que world_to_camera y
            project_points.
    """
    return project_points(world_to_camera(points_world, T_cam_from_world), K)


def backproject_pixels(
    pixels: ArrayLike,
    K: ArrayLike,
    depths: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruye puntos 3D en el marco de la cámara a partir de píxeles y Z.

    Invierte la proyección pinhole para Z_c conocido:
        Y = (v - cy) Z / fy
        X = ((u - cx) Z - s Y) / fx

    Args:
        pixels: Coordenadas imagen de forma (N, 2), uno por fila.
        K: Matriz intrínseca de forma (3, 3).
        depths: Profundidades Z_c de forma (N,), estrictamente positivas.

    Returns:
        Puntos en el marco de la cámara, float64 de forma (N, 3).

    Raises:
        ValueError: Si K no es válida, pixels no es (N, 2) real finito, depths no
            es (N,) real finito, las longitudes no coinciden, o algún Z <= 0.
    """
    K = _as_intrinsic(K, "K")
    pixels = _as_points(pixels, "pixels", 2)
    depths = _as_depths(depths, "depths", pixels.shape[0])

    fx, skew, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    u = pixels[:, 0]
    v = pixels[:, 1]

    Y = (v - cy) * depths / fy
    X = ((u - cx) * depths - skew * Y) / fx

    return np.column_stack((X, Y, depths))


def reprojection_residual(
    points_world: ArrayLike,
    pixels_observed: ArrayLike,
    K: ArrayLike,
    T_cam_from_world: ArrayLike,
) -> NDArray[np.float64]:
    """Distancia euclídea en píxeles entre la proyección y los píxeles medidos.

    Residual de reproyección por correspondencia, para calibración y validación:
    norm(project(points_world, K, T)[i] - pixels_observed[i]).

    Args:
        points_world: Puntos 3D del mundo, forma (N, 3).
        pixels_observed: Píxeles medidos, forma (N, 2).
        K: Matriz intrínseca de forma (3, 3).
        T_cam_from_world: Extrínseca SE(3) de forma (4, 4).

    Returns:
        Residuales float64 de forma (N,).

    Raises:
        ValueError: Si la proyección falla o si las longitudes no coinciden.
    """
    observed = _as_points(pixels_observed, "pixels_observed", 2)
    projected = project(points_world, K, T_cam_from_world)

    if projected.shape[0] != observed.shape[0]:
        raise ValueError(
            "points_world y pixels_observed deben tener el mismo número de filas"
        )

    return np.linalg.norm(projected - observed, axis=1)


def _as_scalar(value, nombre: str) -> float:
    """Convierte value en un float escalar real finito.

    Raises:
        ValueError: Si no es un escalar real finito.
    """
    try:
        value = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} debe ser un escalar real finito") from exc

    if value.shape != () or np.iscomplexobj(value):
        raise ValueError(f"{nombre} debe ser un escalar real finito")

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{nombre} debe ser un escalar real finito")

    return value


def _as_intrinsic(K: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte K en un array float64 de forma (3, 3) validado.

    Raises:
        ValueError: Si K no es una matriz intrínseca válida.
    """
    try:
        K = np.asarray(K)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if np.iscomplexobj(K):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        K = K.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a una matriz real float64"
        ) from exc

    if K.shape != (3, 3):
        raise ValueError(f"{nombre} debe tener forma (3, 3)")

    if not np.all(np.isfinite(K)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    if not is_intrinsic_matrix(K):
        raise ValueError(
            f"{nombre} debe representar una matriz intrínseca válida "
            "(K[1, 0] = 0, K[2, :] = [0, 0, 1], fx > 0, fy > 0)"
        )

    return K


def _as_points(
    points: ArrayLike,
    nombre: str,
    width: int,
) -> NDArray[np.float64]:
    """Convierte points en un array float64 de forma (N, width) validado.

    Raises:
        ValueError: Si points no es un conjunto real finito de forma (N, width).
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

    if points.ndim != 2 or points.shape[1] != width:
        raise ValueError(f"{nombre} debe tener forma (N, {width})")

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return points


def _as_depths(
    depths: ArrayLike,
    nombre: str,
    n_expected: int,
) -> NDArray[np.float64]:
    """Convierte depths en un array float64 de forma (N,) estrictamente positivo.

    Raises:
        ValueError: Si depths no es (N,) real finito con N == n_expected o si
            algún valor es <= 0.
    """
    try:
        depths = np.asarray(depths)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(depths):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        depths = depths.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if depths.ndim != 1:
        raise ValueError(f"{nombre} debe tener forma (N,)")

    if depths.shape[0] != n_expected:
        raise ValueError(
            f"{nombre} debe tener una profundidad por píxel "
            f"({n_expected} esperadas, {depths.shape[0]} recibidas)"
        )

    if not np.all(np.isfinite(depths)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    if not np.all(depths > 0.0):
        raise ValueError(f"{nombre} debe ser estrictamente positivo")

    return depths
