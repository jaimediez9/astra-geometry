"""Reference vectors for the C++ conformance harness.

Emits, per operation, a JSON document with serializable inputs and the exact
output produced by the Python reference implementation. The planned C++20
harness loads the same file and compares against it, so correctness is defined
once (in Python) and checked twice (Python + C++).

Each vector carries a ``compare`` mode the C++ side must honour:

- ``exact``  — bit-identical comparison (booleans, and the structural last row
  of a homogeneous matrix; see conventions.md section 7.3).
- ``close``  — element-wise comparison with the vector's ``atol`` and rtol = 0.

Two schema versions coexist, and a reader must handle both:

- **schema 1** — a vector carries ``inputs``, ``expected`` and ``compare`` (plus
  ``atol`` for ``close``). Still readable and executable; the vectors frozen
  before the conformance extension were written this way and are preserved
  verbatim in ``benchmarks/reference/vectors-schema1.json``.
- **schema 2** — adds ``expected_error``: a vector may declare the exception
  type a call must raise instead of an expected value. ``expected_error`` is
  **mutually exclusive** with ``expected`` and ``compare``; a vector has either
  a value contract or an error contract, never both.

``verify_document`` executes a document of EITHER schema against the library
(this is the reader the C++ conformance harness mirrors), so compatibility is
checked by running, not by inspection.

Inputs are reused from the case registry so the fixtures describe exactly the
operations the benchmark measures. Run::

    python -m benchmarks.reference --out benchmarks/reference/vectors.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

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

from . import cases

SCHEMA_VERSION = 2

# Schema versions this module can read and execute. Kept explicit so the
# compatibility contract is greppable and testable.
SUPPORTED_SCHEMAS = (1, 2)

CLOSE_ATOL = 1e-12


def _as_list(value):
    """JSON-serializable form: arrays become nested lists, containers recurse."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_as_list(item) for item in value]
    return value


def _vector(name, module, function, inputs, expected, *, compare="close", atol=CLOSE_ATOL):
    vector = {
        "name": name,
        "module": module,
        "function": function,
        "inputs": {k: _as_list(v) for k, v in inputs.items()},
        "expected": _as_list(expected),
        "compare": compare,
    }
    if compare == "close":
        vector["atol"] = atol
    return vector


def _error_vector(name, module, function, inputs, error):
    """A vector whose contract is an EXCEPTION, not a value (schema 2).

    Mutually exclusive with ``expected``/``compare`` by construction: the
    returned dict never carries those keys.
    """
    return {
        "name": name,
        "module": module,
        "function": function,
        "inputs": {k: _as_list(v) for k, v in inputs.items()},
        "expected_error": {"type": error},
    }


# ---------------------------------------------------------------------------
# Independent mathematical references for the chapters added after the freeze.
#
# Nothing in this block calls astra_geometry: these are the hand-written
# formulas the library is checked AGAINST. NumPy is used only as arithmetic
# (a matrix product, a linear solve), never as the implementation under test.
# ---------------------------------------------------------------------------


def _skew(t):
    """Explicit cross-product matrix, written out (no library call)."""
    t = np.asarray(t, dtype=np.float64)
    x, y, z = t
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def _rodrigues(axis, theta):
    """Independent Rodrigues, same construction the case registry uses."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _intrinsics(fx, fy, cx, cy, skew=0.0):
    """K written out, including the admitted non-zero skew."""
    return np.array([
        [fx, skew, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])


def _extrinsics(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _pinhole_camera_frame(points_cam, K):
    """[u, v] = [fx*X/Z + s*Y/Z + cx, fy*Y/Z + cy], term by term."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    X, Y, Z = points_cam.T
    fx, skew, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]
    u = fx * X / Z + skew * Y / Z + cx
    v = fy * Y / Z + cy
    return np.column_stack((u, v))


def _pinhole_world(points_world, K, T):
    """Project world points: X_cam = R X + t, then the pinhole formula."""
    points_world = np.asarray(points_world, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return _pinhole_camera_frame((R @ points_world.T).T + t, K)


def _normalized(pixels, K):
    """x_hat = K^-1 [u, v, 1]^T, first two coordinates (calibrated case)."""
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    projected = homogeneous @ np.linalg.inv(K).T
    return projected[:, :2] / projected[:, 2:3]


# Fixed, deterministic inputs for the post-freeze chapters.
_K1 = _intrinsics(800.0, 780.0, 640.0, 360.0, skew=3.5)
_K2 = _intrinsics(760.0, 770.0, 610.0, 380.0)
_POSE_AXIS, _POSE_THETA = (0.2, -0.7, 0.68), 0.4
_POSE_R = _rodrigues(_POSE_AXIS, _POSE_THETA)
_POSE_T = np.array([1.4, -0.6, 0.35])
_POSE = _extrinsics(_POSE_R, _POSE_T)
# Two cameras that differ only by a translation: a point at infinity then has
# exactly the same projection in both, i.e. zero parallax.
_TRANSLATION_ONLY = _extrinsics(np.eye(3), _POSE_T)


def _scene(n=5, seed=11):
    """World points in front of both cameras (Z positive in frame 1)."""
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(4.0, 9.0, n),
    ))


def _build_post_freeze_vectors() -> list[dict]:
    """Vectors for camera, estimation, triangulation and essential.

    Every ``expected`` here is produced by the independent formulas above, or
    is the ground truth the input was constructed from — never by calling the
    function under test.
    """
    scene = _scene(5, seed=11)
    scene3 = _scene(3, seed=12)

    # --- camera ---------------------------------------------------------
    camera_points = (scene3 @ _POSE_R.T) + _POSE_T  # hand-built camera frame
    observed_pixels = _pinhole_world(scene, _K1, _POSE)
    # A deterministic sub-pixel offset, so the residual vector checks a
    # NON-trivial value instead of the trivial zero of consistent data.
    pixel_offset = np.array([0.5, -0.25])
    perturbed_pixels = observed_pixels + pixel_offset

    # --- estimation: exact correspondence, so the answer is the pose itself
    estimate_source = _scene(6, seed=21)
    estimate_target = (estimate_source @ _POSE_R.T) + _POSE_T

    # --- triangulation: the answer is the scene the pixels were made from
    tri_pixels1 = _pinhole_world(scene3, _K1, _extrinsics(np.eye(3), np.zeros(3)))
    tri_pixels2 = _pinhole_world(scene3, _K2, _POSE)

    # A point at infinity, for the no-parallax degeneracy: a direction scaled
    # enormously, seen from two cameras that differ only by a translation. The
    # projections coincide, so the depth is unrecoverable and must be rejected.
    far_directions = _scene(3, seed=41)
    far_directions = far_directions / np.linalg.norm(far_directions, axis=1, keepdims=True)
    far_points = far_directions * 1e9
    far_pixels1 = _pinhole_world(far_points, _K1, _extrinsics(np.eye(3), np.zeros(3)))
    far_pixels2 = _pinhole_world(far_points, _K2, _TRANSLATION_ONLY)

    # --- essential: the answer is the pose the pixels were made from
    # At least 8 correspondences are required (the 8-point method); 24 gives a
    # well-conditioned, over-determined system.
    ess_scene = _scene(24, seed=31)
    ess_pixels1 = _pinhole_world(ess_scene, _K1, _extrinsics(np.eye(3), np.zeros(3)))
    ess_pixels2 = _pinhole_world(ess_scene, _K2, _POSE)
    ess_x1 = _normalized(ess_pixels1, _K1)
    ess_x2 = _normalized(ess_pixels2, _K2)
    E_true = _skew(_POSE_T) @ _POSE_R
    t_hat = _POSE_T / np.linalg.norm(_POSE_T)

    # The twisted pair, in closed form: R_tw = R * Rot(pi, R^T t_hat). This is
    # the second solution of E = [t]_x R for the same t; it is built here by
    # Rodrigues so the candidate set does not come from the module.
    R_tw = _POSE_R @ _rodrigues(_POSE_R.T @ t_hat, np.pi)
    candidates = [
        (_POSE_R, t_hat), (_POSE_R, -t_hat), (R_tw, t_hat), (R_tw, -t_hat),
    ]

    return [
        # --- camera ---
        _vector("camera.intrinsic_matrix", "camera", "intrinsic_matrix",
                {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0, "skew": 3.5},
                _intrinsics(800.0, 780.0, 640.0, 360.0, skew=3.5)),
        _vector("camera.is_intrinsic_matrix.valid", "camera", "is_intrinsic_matrix",
                {"K": _K1}, True, compare="exact"),
        _vector("camera.is_intrinsic_matrix.bad_last_row", "camera",
                "is_intrinsic_matrix",
                {"K": np.array([[800.0, 0.0, 640.0], [0.0, 780.0, 360.0], [0.0, 0.0, 1.0 + 1e-6]])},
                False, compare="exact"),
        _vector("camera.world_to_camera", "camera", "world_to_camera",
                {"points_world": scene3, "T_cam_from_world": _POSE},
                (scene3 @ _POSE_R.T) + _POSE_T),
        _vector("camera.project_points", "camera", "project_points",
                {"points_cam": camera_points, "K": _K1},
                _pinhole_camera_frame(camera_points, _K1)),
        _vector("camera.project", "camera", "project",
                {"points_world": scene, "K": _K1, "T_cam_from_world": _POSE},
                observed_pixels),
        _vector("camera.backproject_pixels", "camera", "backproject_pixels",
                {"pixels": observed_pixels, "K": _K1, "depths": np.array([5.0, 6.5, 4.0, 8.0, 7.25])},
                np.array([5.0, 6.5, 4.0, 8.0, 7.25])[:, None]
                * (_pinhole_ray(observed_pixels, _K1))),
        _vector("camera.reprojection_residual", "camera", "reprojection_residual",
                {"points_world": scene, "pixels_observed": perturbed_pixels,
                 "K": _K1, "T_cam_from_world": _POSE},
                # Euclidean pixel distance per point, which is what the function
                # returns: the norm of the constant offset, once per point.
                np.linalg.norm(perturbed_pixels - observed_pixels, axis=1)),
        # --- estimation ---
        _vector("estimation.estimate_rigid_transform", "estimation",
                "estimate_rigid_transform",
                {"source": estimate_source, "target": estimate_target},
                _extrinsics(_POSE_R, _POSE_T)),
        # --- triangulation ---
        _vector("triangulation.projection_matrix", "triangulation",
                "projection_matrix", {"K": _K1, "T_cam_from_world": _POSE},
                _K1 @ _POSE[:3, :]),
        _vector("triangulation.triangulate", "triangulation", "triangulate",
                {"pixels1": tri_pixels1, "pixels2": tri_pixels2,
                 "K1": _K1, "K2": _K2, "T_2_from_1": _POSE},
                scene3),
        # --- essential ---
        _vector("essential.essential_matrix", "essential", "essential_matrix",
                {"R": _POSE_R, "t": _POSE_T}, E_true),
        _vector("essential.is_essential_matrix.valid", "essential",
                "is_essential_matrix", {"E": E_true}, True, compare="exact"),
        _vector("essential.is_essential_matrix.scaled", "essential",
                "is_essential_matrix", {"E": 5.0 * E_true}, True, compare="exact"),
        _vector("essential.is_essential_matrix.rank_three", "essential",
                "is_essential_matrix", {"E": np.eye(3)}, False, compare="exact"),
        _vector("essential.estimate_relative_pose", "essential",
                "estimate_relative_pose",
                {"pixels1": ess_pixels1, "pixels2": ess_pixels2, "K1": _K1, "K2": _K2},
                (_POSE_R, t_hat), atol=1e-9),
        _vector("essential.select_candidate", "essential", "select_candidate",
                {"candidates": [[R, t] for R, t in candidates],
                 "pixels1": ess_pixels1, "pixels2": ess_pixels2,
                 "K1": _K1, "K2": _K2},
                (_POSE_R, t_hat), atol=1e-9),
        # --- degeneracies: the contract is the exception, not a value ---
        _error_vector("estimation.estimate_rigid_transform.collinear", "estimation",
                      "estimate_rigid_transform",
                      {"source": np.column_stack((np.arange(5.0), 2.0 * np.arange(5.0),
                                                  3.0 * np.arange(5.0))),
                       "target": np.column_stack((np.arange(5.0), 2.0 * np.arange(5.0),
                                                  3.0 * np.arange(5.0)))},
                      "ValueError"),
        _error_vector("triangulation.triangulate.no_parallax", "triangulation",
                      "triangulate",
                      {"pixels1": far_pixels1, "pixels2": far_pixels2,
                       "K1": _K1, "K2": _K2, "T_2_from_1": _TRANSLATION_ONLY},
                      "ValueError"),
        _error_vector("essential.essential_matrix.null_translation", "essential",
                      "essential_matrix", {"R": _POSE_R, "t": np.zeros(3)},
                      "ValueError"),
        _error_vector("essential.estimate_essential_matrix.too_few", "essential",
                      "estimate_essential_matrix",
                      {"x1": ess_x1[:7], "x2": ess_x2[:7]}, "ValueError"),
    ]


def _pinhole_ray(pixels, K):
    """K^-1 [u, v, 1]^T normalized to unit depth: the back-projection ray."""
    pixels = np.asarray(pixels, dtype=np.float64)
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    rays = homogeneous @ np.linalg.inv(K).T
    return rays / rays[:, 2:3]


def build_vectors() -> list[dict]:
    """Build the full reference-vector list in a stable order."""
    R, R2 = cases._R, cases._R2
    T, T2 = cases._T, cases._T2
    t = cases._T_T
    Q, Q2, Q_NON_UNIT = cases._Q, cases._Q2, cases._Q_NON_UNIT
    axis, theta = cases._AXIS, cases._THETA
    points = cases._points(3)

    # An invalid rotation (scaled, det != 1) and an SE(3) with a noisy last row,
    # to pin the predicate behaviour: the last row is structure, not geometry.
    R_bad = R * 2.0
    T_bad_row = T.copy()
    T_bad_row[3, 3] = 1.0 + 1e-6  # within atol, yet structurally invalid

    return [
        # --- SO(3) ---
        _vector("rotation_matrix3D", "rotation", "rotation_matrix3D",
                {"axis": axis, "theta": theta}, rotation_matrix3D(axis, theta)),
        _vector("rotation_matrix2D", "rotation", "rotation_matrix2D",
                {"theta": theta}, rotation_matrix2D(theta)),
        _vector("is_rotation_matrix.valid", "rotation", "is_rotation_matrix",
                {"R": R}, is_rotation_matrix(R), compare="exact"),
        _vector("is_rotation_matrix.invalid", "rotation", "is_rotation_matrix",
                {"R": R_bad}, is_rotation_matrix(R_bad), compare="exact"),
        _vector("compose_rotations", "rotation", "compose_rotations",
                {"R1": R, "R2": R2}, compose_rotations(R, R2)),
        _vector("inverse_rotation", "rotation", "inverse_rotation",
                {"R": R}, inverse_rotation(R)),
        _vector("rotate_points", "rotation", "rotate_points",
                {"points": points, "axis": axis, "theta": theta},
                rotate_points(points, axis, theta)),
        # --- Quaternions ---
        _vector("is_unit_quaternion.valid", "quaternions", "is_unit_quaternion",
                {"q": Q}, is_unit_quaternion(Q), compare="exact"),
        _vector("is_unit_quaternion.invalid", "quaternions", "is_unit_quaternion",
                {"q": Q_NON_UNIT}, is_unit_quaternion(Q_NON_UNIT), compare="exact"),
        _vector("normalize_quaternion", "quaternions", "normalize_quaternion",
                {"q": Q_NON_UNIT}, normalize_quaternion(Q_NON_UNIT)),
        _vector("conjugate_quaternion", "quaternions", "conjugate_quaternion",
                {"q": Q}, conjugate_quaternion(Q)),
        _vector("quaternion_multiply", "quaternions", "quaternion_multiply",
                {"q1": Q, "q2": Q2}, quaternion_multiply(Q, Q2)),
        _vector("rotation_matrix_to_quaternion", "quaternions",
                "rotation_matrix_to_quaternion", {"R": R},
                rotation_matrix_to_quaternion(R)),
        _vector("quaternion_to_rotation_matrix", "quaternions",
                "quaternion_to_rotation_matrix", {"q": Q},
                quaternion_to_rotation_matrix(Q)),
        # --- SE(3) ---
        _vector("is_transformation_matrix.valid", "transforms",
                "is_transformation_matrix", {"T": T},
                is_transformation_matrix(T), compare="exact"),
        _vector("is_transformation_matrix.bad_last_row", "transforms",
                "is_transformation_matrix", {"T": T_bad_row},
                is_transformation_matrix(T_bad_row), compare="exact"),
        _vector("transformation_matrix", "transforms", "transformation_matrix",
                {"R": R, "t": t}, transformation_matrix(R, t)),
        _vector("identity_transform", "transforms", "identity_transform",
                {}, identity_transform()),
        _vector("rotation_part", "transforms", "rotation_part",
                {"T": T}, rotation_part(T)),
        _vector("translation_part", "transforms", "translation_part",
                {"T": T}, translation_part(T)),
        _vector("compose_transforms", "transforms", "compose_transforms",
                {"T1": T, "T2": T2}, compose_transforms(T, T2)),
        _vector("inverse_transform", "transforms", "inverse_transform",
                {"T": T}, inverse_transform(T)),
        _vector("transform_points", "transforms", "transform_points",
                {"points": points, "T": T}, transform_points(points, T)),
    ] + _build_post_freeze_vectors()


def build_fixture_document() -> dict:
    return {"schema": SCHEMA_VERSION, "vectors": build_vectors()}


# ---------------------------------------------------------------------------
# Reader: execute a fixture document against the library.
#
# This is the reference behaviour the C++ conformance harness mirrors, and the
# proof that schema 1 and schema 2 are both readable and executable.
# ---------------------------------------------------------------------------

LIBRARY_MODULES = (
    "rotation",
    "quaternions",
    "transforms",
    "camera",
    "estimation",
    "triangulation",
    "essential",
)


def function_registry() -> dict[str, dict[str, object]]:
    """``{module: {name: callable}}`` resolved from the library, by name."""
    registry: dict[str, dict[str, object]] = {}
    for module_name in LIBRARY_MODULES:
        module = importlib.import_module(f"astra_geometry.{module_name}")
        registry[module_name] = {
            name: value
            for name, value in vars(module).items()
            if not name.startswith("_") and callable(value)
        }
    return registry


def decode_inputs(inputs: dict) -> dict:
    """JSON inputs back to call arguments.

    Nested numeric lists become arrays; ``candidates`` is the one input whose
    shape is semantically a sequence of pairs rather than a matrix.
    """
    decoded = {}
    for key, value in inputs.items():
        if key == "candidates":
            decoded[key] = tuple(
                (np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))
                for R, t in value
            )
        elif isinstance(value, list):
            decoded[key] = np.asarray(value, dtype=np.float64)
        else:
            decoded[key] = value
    return decoded


def assert_schema_invariants(document: dict) -> None:
    """A vector has a VALUE contract or an ERROR contract, never both."""
    assert document["schema"] in SUPPORTED_SCHEMAS, (
        f"unsupported schema {document['schema']!r}; "
        f"this reader handles {SUPPORTED_SCHEMAS}"
    )
    for vector in document["vectors"]:
        name = vector["name"]
        has_error = "expected_error" in vector
        has_value = "expected" in vector or "compare" in vector
        assert not (has_error and has_value), (
            f"{name}: expected_error is mutually exclusive with expected/compare"
        )
        assert has_error or ("expected" in vector and "compare" in vector), (
            f"{name}: a vector needs expected_error, or expected plus compare"
        )
        if has_error:
            assert set(vector["expected_error"]) == {"type"}, (
                f"{name}: expected_error carries only 'type'"
            )
        elif vector["compare"] == "close":
            assert "atol" in vector, f"{name}: a close vector needs atol"


def _assert_close(produced, expected, atol: float, name: str) -> None:
    produced = np.asarray(produced, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert produced.shape == expected.shape, (
        f"{name}: shape {produced.shape} != {expected.shape}"
    )
    if produced.size == 0:
        return
    delta = float(np.max(np.abs(produced - expected)))
    assert delta <= atol, f"{name}: max|delta| = {delta:.3e} exceeds atol = {atol:.0e}"


def _compare_value(produced, vector: dict) -> None:
    name = vector["name"]
    expected = vector["expected"]
    if vector["compare"] == "exact":
        if isinstance(expected, bool):
            assert bool(produced) is expected, (
                f"{name}: expected {expected}, got {bool(produced)}"
            )
            return
        assert np.array_equal(np.asarray(produced), np.asarray(expected)), (
            f"{name}: not bit-identical to the expected value"
        )
        return

    atol = float(vector.get("atol", CLOSE_ATOL))
    # A tuple result (a pose, a candidate) is compared part by part; anything
    # else is a single array.
    if isinstance(produced, tuple):
        assert len(produced) == len(expected), (
            f"{name}: {len(produced)} parts != {len(expected)} expected"
        )
        for index, (part, reference) in enumerate(zip(produced, expected)):
            _assert_close(part, reference, atol, f"{name}[{index}]")
        return
    _assert_close(produced, expected, atol, name)


def verify_vector(vector: dict, registry: dict | None = None) -> str:
    """Execute one vector against the library and check its contract.

    Returns ``"error"`` or ``"value"``; raises ``AssertionError`` on failure.

    For an error contract the match is on the **exact** exception class name,
    not on ``isinstance``: ``np.linalg.LinAlgError`` is a ``ValueError``
    subclass, so an ``isinstance`` check would accept numpy failing deep inside
    an SVD where the vector meant to pin the library's own rejection.
    """
    registry = registry if registry is not None else function_registry()
    name = vector["name"]
    try:
        function = registry[vector["module"]][vector["function"]]
    except KeyError:
        raise AssertionError(
            f"{name}: unknown operation {vector['module']}.{vector['function']}"
        ) from None

    inputs = decode_inputs(vector["inputs"])

    if "expected_error" in vector:
        expected_type = vector["expected_error"]["type"]
        try:
            function(**inputs)
        except Exception as exc:  # noqa: BLE001 - the type is the contract
            actual = type(exc).__name__
            assert actual == expected_type, (
                f"{name}: expected {expected_type}, got {actual}: {exc}"
            )
            return "error"
        raise AssertionError(f"{name}: expected {expected_type}, the call returned")

    _compare_value(function(**inputs), vector)
    return "value"


def verify_document(document: dict, registry: dict | None = None) -> dict[str, str]:
    """Execute every vector in a document of either schema.

    Returns ``{vector name: "value" | "error"}``.
    """
    assert_schema_invariants(document)
    registry = registry if registry is not None else function_registry()
    return {
        vector["name"]: verify_vector(vector, registry)
        for vector in document["vectors"]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/reference/vectors.json",
                        help="destination JSON path")
    args = parser.parse_args(argv)

    document = build_fixture_document()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")

    print(f"wrote {path} ({len(document['vectors'])} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
