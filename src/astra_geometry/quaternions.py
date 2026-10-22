"""Quaternion operations for three-dimensional rotations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from .rotation import is_rotation_matrix


def is_unit_quaternion(
    q: ArrayLike,
    atol: float = 1e-8,
) -> bool:
    """Check whether q is a valid unit quaternion.

    Quaternions use scalar-first ordering: (w, x, y, z).
    """
    try:
        q = np.asarray(q)
        atol = float(atol)
    except (TypeError, ValueError, OverflowError):
        return False

    if q.shape != (4,):
        return False

    if np.iscomplexobj(q) or not np.issubdtype(q.dtype, np.number):
        return False

    if not np.isfinite(atol) or atol < 0:
        return False

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            q = q.astype(np.float64)

        if not np.all(np.isfinite(q)):
            return False

        norm = np.linalg.norm(q)

        return bool(
            np.isfinite(norm)
            and np.isclose(norm, 1.0, atol=atol, rtol=0.0)
        )

    except (TypeError, ValueError, OverflowError):
        return False


def normalize_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Return a normalized quaternion in (w, x, y, z) order.

    Raises:
        ValueError: If q is not a finite, nonzero real vector of shape (4,).
    """
    try:
        q = np.asarray(q)

        if np.iscomplexobj(q):
            raise ValueError("q must contain real values")

        q = q.astype(np.float64)

    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "q must be convertible to a real float64 array"
        ) from exc

    if q.shape != (4,):
        raise ValueError("q must have shape (4,)")

    if not np.all(np.isfinite(q)):
        raise ValueError("q must contain finite values")

    # Scale first to avoid overflow when computing the norm.
    scale = np.max(np.abs(q))

    if scale == 0.0:
        raise ValueError("A zero quaternion cannot be normalized")

    q = q / scale
    return q / np.linalg.norm(q)


def _as_quaternion(q: ArrayLike, nombre: str) -> NDArray[np.float64]:
    """Convierte q en un array float64 de forma (4,) validado.

    No comprueba la norma: se usa en operaciones algebraicas, que no exigen
    cuaterniones unitarios. Es la semilla de la validación compartida que se
    trasladará a _validation.py una vez cerrado el álgebra de cuaterniones.

    Args:
        q: Entrada candidata a cuaternión.
        nombre: Nombre del argumento, para el mensaje de error.

    Returns:
        Cuaternión float64 de forma (4,).

    Raises:
        ValueError: Si q no es un vector real finito de forma (4,).
    """
    try:
        q = np.asarray(q)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if np.iscomplexobj(q):
        raise ValueError(f"{nombre} debe contener valores reales")

    try:
        q = q.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{nombre} debe ser convertible a un array real float64"
        ) from exc

    if q.shape != (4,):
        raise ValueError(f"{nombre} debe tener forma (4,)")

    if not np.all(np.isfinite(q)):
        raise ValueError(f"{nombre} debe contener valores finitos")

    return q


def _hamilton_product(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Producto de Hamilton left * right, con operandos ya validados."""
    wl, xl, yl, zl = left
    wr, xr, yr, zr = right

    return np.array([
        wl * wr - xl * xr - yl * yr - zl * zr,
        wl * xr + xl * wr + yl * zr - zl * yr,
        wl * yr - xl * zr + yl * wr + zl * xr,
        wl * zr + xl * yr - yl * xr + zl * wr,
    ], dtype=np.float64)


def conjugate_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Devuelve el conjugado de un cuaternión.

    El conjugado de q = (w, x, y, z) es q* = (w, -x, -y, -z). Para un
    cuaternión unitario, el conjugado es su inverso (docs/conventions.md,
    sección 3). Para un cuaternión general el inverso es q* / |q|^2, que no
    se calcula aquí.

    No se exige norma unitaria: la conjugación es una operación algebraica.
    El resultado es unitario si la entrada lo es.

    Args:
        q: Cuaternión de forma (4,), en orden (w, x, y, z).

    Returns:
        Cuaternión conjugado float64 de forma (4,).

    Raises:
        ValueError: Si q no es un vector real finito de forma (4,).
    """
    q = _as_quaternion(q, "q")

    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(
    q1: ArrayLike,
    q2: ArrayLike,
) -> NDArray[np.float64]:
    """Compone dos cuaterniones mediante el producto de Hamilton.

    El orden de los argumentos sigue el de compose_rotations: q1 se aplica
    primero y q2 después, de modo que el producto devuelto es q2 * q1 y su
    rotación asociada es R2 @ R1 (docs/conventions.md, sección 3). El orden
    importa: el producto no es conmutativo.

    No se exige norma unitaria: es una operación algebraica, y en general
    |q2 * q1| = |q2| · |q1|. Si ambos operandos son unitarios, el resultado
    también lo es.

    Args:
        q1: Cuaternión aplicado primero, de forma (4,), en orden (w, x, y, z).
        q2: Cuaternión aplicado después, de forma (4,), en orden (w, x, y, z).

    Returns:
        Cuaternión float64 de forma (4,) que representa q2 * q1.

    Raises:
        ValueError: Si algún operando no es un vector real finito de
            forma (4,).
    """
    q1 = _as_quaternion(q1, "q1")
    q2 = _as_quaternion(q2, "q2")

    return _hamilton_product(q2, q1)


def rotation_matrix_to_quaternion(
    R: ArrayLike,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Convierte una matriz de rotación SO(3) en un cuaternión unitario.

    La matriz representa una rotación activa aplicada a vectores columna.
    El cuaternión de salida usa el orden escalar-primero (w, x, y, z).

    Args:
        R: Matriz de rotación de forma (3, 3).
        atol: Tolerancia absoluta para validar la matriz.

    Returns:
        Cuaternión unitario float64 de forma (4,), en orden (w, x, y, z).
        Se elige el representante con w >= 0 cuando sea posible.

    Notas:
        Para rotaciones de media vuelta exacta (w == 0) los representantes q
        y -q son igualmente válidos y la regla w >= 0 no puede decidir entre
        ellos: el signo queda determinado por la rama de extracción
        seleccionada. No debe asumirse estabilidad del signo en las
        proximidades de ese caso; la rotación reconstruida sí es estable.

    Raises:
        ValueError: Si la matriz no representa una rotación válida.
    """
    try:
        atol = float(atol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("atol debe ser un escalar real finito no negativo") from exc

    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol debe ser un escalar real finito no negativo")

    try:
        R = np.asarray(R)
    except (TypeError, ValueError) as exc:
        raise ValueError("R debe ser una matriz real de forma (3, 3)") from exc

    if not is_rotation_matrix(R, atol=atol):
        raise ValueError("R debe representar una matriz de rotación válida de SO(3)")

    R = R.astype(np.float64, copy=False)

    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + trace))
        q = np.array([
            0.25 * s,
            (m21 - m12) / s,
            (m02 - m20) / s,
            (m10 - m01) / s,
        ])

    elif m00 >= m11 and m00 >= m22:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
        q = np.array([
            (m21 - m12) / s,
            0.25 * s,
            (m01 + m10) / s,
            (m02 + m20) / s,
        ])

    elif m11 >= m22:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
        q = np.array([
            (m02 - m20) / s,
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
        ])

    else:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
        q = np.array([
            (m10 - m01) / s,
            (m02 + m20) / s,
            (m12 + m21) / s,
            0.25 * s,
        ])

    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)

    if q[0] < 0.0:
        q = -q

    return q


def quaternion_to_rotation_matrix(
    q: ArrayLike,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Convierte un cuaternión unitario en una matriz de rotación 3D.

    El cuaternión usa el orden escalar-primero (w, x, y, z).
    La matriz resultante representa una rotación activa aplicada a
    vectores columna.
    Los cuaterniones q y -q producen la misma matriz, ya que representan
    la misma rotación (docs/conventions.md, sección 3).

    Args:
        q: Cuaternión unitario de forma (4,), en orden (w, x, y, z).
        atol: Tolerancia absoluta para validar la norma del cuaternión.

    Returns:
        Matriz de rotación de tipo float64 y forma (3, 3).

    Raises:
        ValueError: Si q no es un cuaternión unitario válido o atol
            no es finito y no negativo.
    """
    try:
        atol = float(atol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("atol debe ser finito y no negativo") from exc

    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol debe ser finito y no negativo")

    try:
        q = np.asarray(q)
    except (TypeError, ValueError) as exc:
        raise ValueError("q debe ser un cuaternión real de forma (4,)") from exc

    if not is_unit_quaternion(q, atol=atol):
        raise ValueError("q debe ser un cuaternión unitario válido")

    q = q.astype(np.float64, copy=False)
    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )



