# Astra Geometry

**A 3D geometry, calibration, and validation toolkit for robotics and computer vision.**

Astra Geometry covers rigid-body transformations, 3D rotations, camera geometry, and numerical validation, with the goal of providing a reliable, testable foundation for geometric computations in robotics and autonomous perception systems.

The project prioritizes mathematical correctness, explicit coordinate-frame conventions, numerical robustness, and reproducible validation.

> **Status:** Early development · API and architecture subject to change.
>
> **Two tracks.** The repository currently contains a **Python reference implementation** — a readable, explicitly validated statement of the intended mathematics. The production target is a **C++20 core** intended for real-time operation on drones; that port has not been started. Ada is under consideration as a longer-term target. The Python code is the reference and specification aid, not the deployment artifact.

## Objectives

- Implement 3D rotations using rotation matrices and quaternions.
- Represent rigid-body transformations using homogeneous matrices.
- Support composition and inversion of transformations.
- Model pinhole cameras and camera calibration parameters.
- Estimate rigid transformations from point correspondences using SVD.
- Validate mathematical invariants and quantify numerical errors.
- Explore bounded-memory, predictable execution for selected real-time applications.

## Repository layout

```
src/astra_geometry/   Python reference implementation
tests/                Contract- and property-oriented test suite
docs/conventions.md   Mathematical conventions (normative for the API)
cpp/                  C++20 core — planned, currently empty
benchmarks/           Timing and memory measurements — planned, currently empty
examples/             Runnable examples — planned, currently empty
```

`docs/conventions.md` is the normative description of coordinate frames, signs, composition order, and tolerance handling. Where documentation and code disagree, that file defines the intended behavior and the code is the bug.

## Technical scope and current status

| Component | Planned capabilities | Python reference |
|---|---|---|
| Linear algebra | Fixed-size vectors and matrices | via NumPy |
| SO(3) | Rotation matrices and rotation validation | implemented |
| Quaternions | Normalization and unit-norm validation | implemented |
| Quaternions | Conversion to and from rotation matrices | implemented |
| Quaternions | Composition and inversion | implemented (unit inverse = conjugate) |
| SE(3) | Rigid transformations, composition, and inversion | pending |
| Camera geometry | Intrinsics, extrinsics, and 3D-to-2D projection | pending |
| Estimation | Rigid alignment from point correspondences | pending |
| Validation | Property-based tests, numerical error analysis, and reference comparisons | partial |
| Visualization | 3D coordinate frames, camera geometry, and projection results | pending |

## Technology stack

**Python reference implementation (current).**

- **Python 3.12+** — reference implementation and test suite.
- **NumPy 2.0+** — the only runtime dependency.
- **pytest 8.0+** — automated testing.

**C++20 core (planned, not started).**

- **C++20** — core implementation for the real-time target.
- **Eigen** — linear algebra.
- **CMake** — build system.
- **GoogleTest or Catch2** — automated testing.

The core library is intended to remain independent of visualization and experimentation tools. The Python package requires only NumPy at runtime; matplotlib is not currently a dependency.

## Design principles

### Explicit coordinate conventions

Coordinate frames, transformation direction, vector conventions, and units must be documented and consistent throughout the API. The full specification lives in [`docs/conventions.md`](docs/conventions.md).

### Mathematical invariants

Operations are validated against their defining properties. For example, a valid rotation matrix must satisfy:


$$R^\mathsf{T}R=I,\qquad \det(R)=1$$

### Numerical robustness

Tests cover floating-point tolerances, invalid inputs, degenerate configurations, and accumulated numerical error.

### Reproducible validation

Synthetic datasets with known ground truth are used to evaluate transformation recovery, projection accuracy, and implementation consistency.

### Real-time awareness

Selected operations will be evaluated for bounded memory usage and execution-time variability. Hard real-time suitability will not be claimed without platform-specific timing analysis and appropriate system-level verification.

## Public API

Rotation-matrix operations are exported at the package top level:

```python
from astra_geometry import (
    is_rotation_matrix,
    rotation_matrix2D,
    rotation_matrix3D,
    rotate_points,
    compose_rotations,
    inverse_rotation,
)
```

Quaternion operations are currently importable from their submodule only:

```python
from astra_geometry.quaternions import (
    conjugate_quaternion,
    is_unit_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
```

Validation predicates (`is_rotation_matrix`, `is_unit_quaternion`) return `False` for invalid input instead of raising; constructors and operations raise `ValueError` on violated contracts.

## Build and test

### Python reference implementation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

### C++ core (planned)

Build instructions will be finalized as the C++ implementation and dependency configuration stabilize. The intended workflow is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Project roadmap

- [x] Establish coordinate-frame and transformation conventions.
- [x] Implement and test SO(3) operations.
- [x] Complete quaternion support: composition, inversion, and conversion to and from rotation matrices. *Inversion is exposed as the conjugate, which is the inverse for unit quaternions; a general (non-unit) inverse is not implemented.*
- [ ] Implement SE(3) transformations.
- [ ] Implement pinhole camera projection.
- [ ] Implement rigid transformation estimation using SVD.
- [ ] Extend numerical validation and reference comparisons. *Validation predicates and explicit tolerance handling are in place.*
- [ ] Develop a reproducible 3D visualization example.
- [ ] Benchmark selected operations and investigate real-time constraints.
- [ ] Port the core to C++20 and verify the reference implementation against it.
- [ ] Evaluate optional language bindings and alternative implementations.

## Intended applications

- Robotics and autonomous systems.
- Camera calibration and sensor alignment.
- Visual odometry and perception pipelines.
- Geometric validation and numerical diagnostics.
- Research into predictable geometric computation for embedded systems.

## Project status and limitations

Astra Geometry is an experimental engineering project. Its API, numerical behavior, performance, and suitability for safety-critical or hard real-time systems remain subject to implementation and verification.

The Python implementation is a reference and specification aid. It is not intended for deployment on the real-time target, and no timing, memory, or determinism claims should be inferred from it. The C++ core does not yet exist; roadmap entries are intentions, not commitments.

The project aims to make correctness and limitations measurable rather than assume them.

## License

License to be determined. Until one is added, no permission is granted for use, modification, or redistribution.
