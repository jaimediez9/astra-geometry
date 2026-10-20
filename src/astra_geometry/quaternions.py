"""Quaternion operations for three-dimensional rotations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


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



def quaternion_to_rotation_matrix(
    q: ArrayLike,
    *,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Convert a unit quaternion to a 3D rotation matrix.

    Quaternion components use scalar-first Hamilton convention (w, x, y, z).
    The resulting matrix represents an active rotation acting on column vectors.

    Parameters
    ----------
    q : array-like of shape (4,)
        Unit quaternion in (w, x, y, z) order.
    atol : float, default=1e-8
        Absolute tolerance for unit-norm validation.

    Returns
    -------
    numpy.ndarray of shape (3, 3)
        Rotation matrix with dtype float64.

    Raises
    ------
    ValueError
        If q is malformed, contains non-finite values, or is not unit-length
        within the specified tolerance.
    """
    if not np.isscalar(atol) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite, non-negative value.")

    try:
        raw = np.asarray(q)
        if np.iscomplexobj(raw):
            raise ValueError("q must contain real-valued components.")
        q_array = np.asarray(q, dtype=np.float64)
    except(TypeError, OverflowError) as exc:
        raise ValueError("q must contain real-valued components.")

    if q_array.shape != (4,):
        raise ValueError("q must have shape (4,).")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q must contain only finite values.")
    if not is_unit_quaternion(q_array, atol=atol):
        raise ValueError("q must be a quaternion.")

    w, x, y, z = q_array

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
