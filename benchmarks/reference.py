"""Reference vectors for the C++ conformance harness.

Emits, per operation, a JSON document with serializable inputs and the exact
output produced by the Python reference implementation. The planned C++20
harness loads the same file and compares against it, so correctness is defined
once (in Python) and checked twice (Python + C++).

Each vector carries a ``compare`` mode the C++ side must honour:

- ``exact``  — bit-identical comparison (booleans, and the structural last row
  of a homogeneous matrix; see conventions.md section 7.3).
- ``close``  — element-wise comparison with the vector's ``atol`` and rtol = 0.

Inputs are reused from the case registry so the fixtures describe exactly the
operations the benchmark measures. Run::

    python -m benchmarks.reference --out benchmarks/reference/vectors.json
"""

from __future__ import annotations

import argparse
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

SCHEMA_VERSION = 1
CLOSE_ATOL = 1e-12


def _as_list(value):
    """JSON-serializable form: ndarrays become nested lists, scalars pass."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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
    ]


def build_fixture_document() -> dict:
    return {"schema": SCHEMA_VERSION, "vectors": build_vectors()}


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
