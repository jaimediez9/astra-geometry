"""Benchmark case registry for the frozen astra-geometry API.

Each case builds its own inputs from a seeded RNG and an independent Rodrigues
(mirroring ``tests/_helpers.py``), so the function under measure never receives
input produced by the library itself.

``static_temporaries`` is a **hand count of array-shaped temporaries per call**,
not a measurement: NumPy's allocator is invisible to ``tracemalloc``, so this
approximates the allocation signal until the C++ core can be instrumented for
real. It complements the measured ``peak_bytes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from astra_geometry import (
    compose_rotations,
    inverse_rotation,
    is_rotation_matrix,
    rotate_points,
    rotation_matrix2D,
    rotation_matrix3D,
)
from astra_geometry.quaternions import (
    conjugate_quaternion,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from astra_geometry.transforms import (
    compose_transforms,
    identity_transform,
    inverse_transform,
    is_transformation_matrix,
    rotation_part,
    transform_points,
    transformation_matrix,
    translation_part,
)

# Sizes swept for the point-set operations. 1 is included so the per-call
# overhead (the regime where input validation dominates) is visible.
SIZES: tuple[int, ...] = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)


def _rotation(axis: Any, theta: float) -> np.ndarray:
    """Independent Rodrigues, so inputs are never built by the library."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _transform(axis: Any, theta: float, t: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rotation(axis, theta)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


# Fixed, deterministic inputs shared across cases.
_AXIS = np.array([0.3, -0.5, 0.81])
_THETA = 0.7
_R = _rotation(_AXIS, _THETA)
_R2 = _rotation([-0.2, 0.9, 0.4], 1.1)
_T = _transform(_AXIS, _THETA, [1.0, -2.0, 3.0])
_T2 = _transform([-0.2, 0.9, 0.4], 1.1, [-0.5, 0.25, 4.0])
_T_T = np.array([1.0, -2.0, 3.0])
_Q = np.array([0.5, 0.5, 0.5, 0.5])
_Q2 = np.array([0.5, -0.5, 0.5, -0.5])
_Q_NON_UNIT = np.array([2.0, 0.0, 0.0, 0.0])

_RNG = np.random.default_rng(1234)


def _points(n: int) -> np.ndarray:
    # Seeded per call: deterministic across runs, independent of the library.
    return np.random.default_rng(1234).random((n, 3))


@dataclass(frozen=True)
class Case:
    name: str
    module: str
    function: str
    category: str
    input_spec: str
    static_temporaries: int
    make: Callable[[int], Callable[[], Any]]
    sizes: tuple[int, ...] | None = None


def _const(fn: Callable[[], Any]) -> Callable[[int], Callable[[], Any]]:
    return lambda _n: fn


def _make_rotate_points(n: int) -> Callable[[], Any]:
    points = _points(n)  # built once, outside the timed call
    return lambda: rotate_points(points, _AXIS, _THETA)


def _make_transform_points(n: int) -> Callable[[], Any]:
    points = _points(n)  # built once, outside the timed call
    return lambda: transform_points(points, _T)


CASES: tuple[Case, ...] = (
    # --- SO(3) ---
    Case(
        "is_rotation_matrix", "rotation", "is_rotation_matrix", "so3-validation",
        "R valid (3,3)", static_temporaries=4,
        make=_const(lambda: is_rotation_matrix(_R)),
    ),
    Case(
        "rotation_matrix3D", "rotation", "rotation_matrix3D", "so3",
        "axis (3,), theta (float)", static_temporaries=6,
        make=_const(lambda: rotation_matrix3D(_AXIS, _THETA)),
    ),
    Case(
        "rotation_matrix2D", "rotation", "rotation_matrix2D", "so3",
        "theta (float)", static_temporaries=1,
        make=_const(lambda: rotation_matrix2D(_THETA)),
    ),
    Case(
        "compose_rotations", "rotation", "compose_rotations", "so3",
        "R1, R2 (3,3)", static_temporaries=1,
        make=_const(lambda: compose_rotations(_R, _R2)),
    ),
    Case(
        "inverse_rotation", "rotation", "inverse_rotation", "so3",
        "R (3,3)", static_temporaries=1,
        make=_const(lambda: inverse_rotation(_R)),
    ),
    Case(
        "rotate_points", "rotation", "rotate_points", "so3",
        "points (N,3), axis, theta", static_temporaries=2,
        make=_make_rotate_points, sizes=SIZES,
    ),
    # --- Quaternions ---
    Case(
        "is_unit_quaternion", "quaternions", "is_unit_quaternion",
        "quaternion-validation", "q unit (4,)", static_temporaries=2,
        make=_const(lambda: is_unit_quaternion(_Q)),
    ),
    Case(
        "normalize_quaternion", "quaternions", "normalize_quaternion",
        "quaternion", "q non-unit (4,)", static_temporaries=2,
        make=_const(lambda: normalize_quaternion(_Q_NON_UNIT)),
    ),
    Case(
        "conjugate_quaternion", "quaternions", "conjugate_quaternion",
        "quaternion", "q (4,)", static_temporaries=2,
        make=_const(lambda: conjugate_quaternion(_Q)),
    ),
    Case(
        "quaternion_multiply", "quaternions", "quaternion_multiply",
        "quaternion", "q1, q2 (4,)", static_temporaries=8,
        make=_const(lambda: quaternion_multiply(_Q, _Q2)),
    ),
    Case(
        "rotation_matrix_to_quaternion", "quaternions",
        "rotation_matrix_to_quaternion", "quaternion",
        "R (3,3)", static_temporaries=6,
        make=_const(lambda: rotation_matrix_to_quaternion(_R)),
    ),
    Case(
        "quaternion_to_rotation_matrix", "quaternions",
        "quaternion_to_rotation_matrix", "quaternion",
        "q (4,)", static_temporaries=6,
        make=_const(lambda: quaternion_to_rotation_matrix(_Q)),
    ),
    # --- SE(3) ---
    Case(
        "is_transformation_matrix", "transforms", "is_transformation_matrix",
        "se3-validation", "T valid (4,4)", static_temporaries=3,
        make=_const(lambda: is_transformation_matrix(_T)),
    ),
    Case(
        "transformation_matrix", "transforms", "transformation_matrix",
        "se3", "R (3,3), t (3,)", static_temporaries=1,
        make=_const(lambda: transformation_matrix(_R, _T_T)),
    ),
    Case(
        "identity_transform", "transforms", "identity_transform",
        "se3", "(none)", static_temporaries=1,
        make=_const(identity_transform),
    ),
    Case(
        "rotation_part", "transforms", "rotation_part",
        "se3", "T (4,4)", static_temporaries=2,
        make=_const(lambda: rotation_part(_T)),
    ),
    Case(
        "translation_part", "transforms", "translation_part",
        "se3", "T (4,4)", static_temporaries=2,
        make=_const(lambda: translation_part(_T)),
    ),
    Case(
        "compose_transforms", "transforms", "compose_transforms",
        "se3", "T1, T2 (4,4)", static_temporaries=1,
        make=_const(lambda: compose_transforms(_T, _T2)),
    ),
    Case(
        "inverse_transform", "transforms", "inverse_transform",
        "se3", "T (4,4)", static_temporaries=2,
        make=_const(lambda: inverse_transform(_T)),
    ),
    Case(
        "transform_points", "transforms", "transform_points",
        "se3", "points (N,3), T (4,4)", static_temporaries=2,
        make=_make_transform_points, sizes=SIZES,
    ),
)

# (validator, operation, size) — the headline validation_cost ratio.
# ``size`` is the N at which a point-set operation is compared (1 = the regime
# where validation must be cheap relative to the operation); None = fixed.
VALIDATION_PAIRS: tuple[tuple[str, str, int | None], ...] = (
    ("is_rotation_matrix", "compose_rotations", None),
    ("is_rotation_matrix", "rotate_points", 1),
    ("is_unit_quaternion", "quaternion_multiply", None),
    ("is_transformation_matrix", "compose_transforms", None),
    ("is_transformation_matrix", "transform_points", 1),
)


def select_cases(pattern: str | None) -> tuple[Case, ...]:
    """Return cases whose name contains ``pattern`` (substring), or all."""
    if not pattern:
        return CASES
    return tuple(c for c in CASES if pattern in c.name)
