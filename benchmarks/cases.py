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
from astra_geometry.camera import (
    is_intrinsic_matrix,
    project_points,
    reprojection_residual,
)
from astra_geometry.essential import (
    estimate_essential_matrix,
    estimate_relative_pose,
    is_essential_matrix,
)
from astra_geometry.estimation import estimate_rigid_transform
from astra_geometry.triangulation import triangulate
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

# Post-freeze operations have their own sweeps, for three different reasons:
# a real minimum N (Kabsch needs 3, the 8-point method needs 8), and — for the
# essential-matrix family — an O(N^2) intermediate in the current
# implementation (`np.linalg.svd(rows)` on an (N, 9) matrix keeps the full
# (N, N) U; 74.5 GiB at N=1e5). The shared SIZES tuple is NOT modified, so the
# original 20 cases keep exactly the sizes they were measured with.
ESTIMATION_SIZES: tuple[int, ...] = (3, 10, 100, 1_000, 10_000)
TRIANGULATION_SIZES: tuple[int, ...] = (1, 10, 100, 1_000)
# The essential-matrix sweep used to stop at 256 because the solve allocated an
# (N, N) U: at N = 1e5 that is 74.5 GiB. With the reduced-SVD fix the ceiling is
# gone, so the sweep now reaches the sizes where the old implementation was
# impossible. The small sizes are kept unchanged to guard against regressions.
ESSENTIAL_SIZES: tuple[int, ...] = (8, 16, 64, 256, 1_000, 10_000, 100_000)
RELATIVE_POSE_SIZES: tuple[int, ...] = (8, 32, 128, 512)


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


# --- Post-freeze chapters: camera, estimation, triangulation, essential. -----
# Inputs stay hand-built (Rodrigues above, an explicit pinhole formula), so no
# measured function receives input produced by the library under test.

# K written out, including the admitted non-zero skew.
_K_CAM = np.array([[800.0, 3.5, 640.0], [0.0, 780.0, 360.0], [0.0, 0.0, 1.0]])
_K_CAM2 = np.array([[760.0, 0.0, 610.0], [0.0, 770.0, 380.0], [0.0, 0.0, 1.0]])
_T_CAM = _transform([0.2, -0.7, 0.68], 0.35, [0.3, -0.2, 0.4])
_ESSENTIAL = np.array([
    [0.0, -_T_T[2], _T_T[1]],
    [_T_T[2], 0.0, -_T_T[0]],
    [-_T_T[1], _T_T[0], 0.0],
]) @ _R


def _scene_points(n: int, seed: int = 4321) -> np.ndarray:
    """World points in front of the cameras (Z well positive), deterministic."""
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(6.0, 12.0, n),
    ))


def _pinhole(points_world: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Hand-written pinhole projection, so pixels never come from `project`."""
    camera = (T[:3, :3] @ points_world.T).T + T[:3, 3]
    X, Y, Z = camera.T
    u = K[0, 0] * X / Z + K[0, 1] * Y / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return np.column_stack((u, v))


def _normalized(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """x_hat = K^-1 [u, v, 1]^T, first two coordinates (the calibrated case)."""
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    projected = homogeneous @ np.linalg.inv(K).T
    return projected[:, :2] / projected[:, 2:3]


def _make_project_points(n: int) -> Callable[[], Any]:
    points = _scene_points(n)  # built once, outside the timed call
    return lambda: project_points(points, _K_CAM)


def _make_reprojection_residual(n: int) -> Callable[[], Any]:
    points = _scene_points(n)
    observed = _pinhole(points, _K_CAM, _T_CAM) + np.array([0.4, -0.3])
    return lambda: reprojection_residual(points, observed, _K_CAM, _T_CAM)


def _make_estimate_rigid(n: int) -> Callable[[], Any]:
    source = _points(n)
    target = (source @ _R.T) + _T_T
    return lambda: estimate_rigid_transform(source, target)


def _make_triangulate(n: int) -> Callable[[], Any]:
    points = _scene_points(n)
    pixels1 = _pinhole(points, _K_CAM, np.eye(4))
    pixels2 = _pinhole(points, _K_CAM2, _T_CAM)
    return lambda: triangulate(pixels1, pixels2, _K_CAM, _K_CAM2, _T_CAM)


def _make_estimate_essential(n: int) -> Callable[[], Any]:
    points = _scene_points(n)
    x1 = _normalized(_pinhole(points, _K_CAM, np.eye(4)), _K_CAM)
    x2 = _normalized(_pinhole(points, _K_CAM2, _T_CAM), _K_CAM2)
    return lambda: estimate_essential_matrix(x1, x2)


def _make_relative_pose(n: int) -> Callable[[], Any]:
    points = _scene_points(n)
    pixels1 = _pinhole(points, _K_CAM, np.eye(4))
    pixels2 = _pinhole(points, _K_CAM2, _T_CAM)
    return lambda: estimate_relative_pose(pixels1, pixels2, _K_CAM, _K_CAM2)


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
    # --- Camera ---
    Case(
        "is_intrinsic_matrix", "camera", "is_intrinsic_matrix",
        "camera-validation", "K valid (3,3)", static_temporaries=2,
        make=_const(lambda: is_intrinsic_matrix(_K_CAM)),
    ),
    Case(
        "project_points", "camera", "project_points",
        "camera", "points_cam (N,3, Z>0), K (3,3)", static_temporaries=3,
        make=_make_project_points, sizes=SIZES,
    ),
    Case(
        "reprojection_residual", "camera", "reprojection_residual",
        "camera", "points_world (N,3), pixels (N,2), K, T", static_temporaries=6,
        make=_make_reprojection_residual, sizes=SIZES,
    ),
    # --- Estimation ---
    Case(
        "estimate_rigid_transform", "estimation", "estimate_rigid_transform",
        "estimation", "source (N,3), target (N,3), N>=3", static_temporaries=6,
        make=_make_estimate_rigid, sizes=ESTIMATION_SIZES,
    ),
    # --- Triangulation ---
    Case(
        "triangulate", "triangulation", "triangulate",
        "triangulation", "pixels1 (N,2), pixels2 (N,2), K1, K2, T", static_temporaries=8,
        make=_make_triangulate, sizes=TRIANGULATION_SIZES,
    ),
    # --- Epipolar ---
    Case(
        "is_essential_matrix", "essential", "is_essential_matrix",
        "essential-validation", "E valid (3,3)", static_temporaries=3,
        make=_const(lambda: is_essential_matrix(_ESSENTIAL)),
    ),
    Case(
        "estimate_essential_matrix", "essential", "estimate_essential_matrix",
        "essential", "x1 (N,2), x2 (N,2), N>=8", static_temporaries=7,
        make=_make_estimate_essential, sizes=ESSENTIAL_SIZES,
    ),
    Case(
        "estimate_relative_pose", "essential", "estimate_relative_pose",
        "essential", "pixels1 (N,2), pixels2 (N,2), K1, K2, N>=8",
        static_temporaries=12,
        make=_make_relative_pose, sizes=RELATIVE_POSE_SIZES,
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
    # Post-freeze validators, guarding the operation they protect.
    ("is_intrinsic_matrix", "project_points", 1),
    ("is_essential_matrix", "estimate_essential_matrix", 8),
)


def select_cases(pattern: str | None) -> tuple[Case, ...]:
    """Return cases whose name contains ``pattern`` (substring), or all."""
    if not pattern:
        return CASES
    return tuple(c for c in CASES if pattern in c.name)
