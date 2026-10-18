# Astra Geometry

**A 3D geometry, calibration, and validation toolkit for robotics and computer vision.**

Astra Geometry is a C++20 library for rigid-body transformations, 3D rotations, camera geometry, and numerical validation. Its goal is to provide a reliable, testable foundation for geometric computations in robotics and autonomous perception systems.

The project prioritizes mathematical correctness, explicit coordinate-frame conventions, numerical robustness, and reproducible validation.

> **Status:** Early development · API and architecture subject to change.

## Objectives

- Implement 3D rotations using rotation matrices and quaternions.
- Represent rigid-body transformations using homogeneous matrices.
- Support composition and inversion of transformations.
- Model pinhole cameras and camera calibration parameters.
- Estimate rigid transformations from point correspondences using SVD.
- Validate mathematical invariants and quantify numerical errors.
- Explore bounded-memory, predictable execution for selected real-time applications.

## Technical scope

| Component | Planned capabilities |
|---|---|
| Linear algebra | Fixed-size vectors and matrices |
| SO(3) | Rotation matrices and rotation validation |
| Quaternions | Representation, composition, and conversion |
| SE(3) | Rigid transformations, composition, and inversion |
| Camera geometry | Intrinsics, extrinsics, and 3D-to-2D projection |
| Estimation | Rigid alignment from point correspondences |
| Validation | Property-based tests, numerical error analysis, and reference comparisons |
| Visualization | 3D coordinate frames, camera geometry, and projection results |

## Technology stack

- **C++20** — Core implementation.
- **Eigen** — Linear algebra.
- **CMake** — Build system.
- **GoogleTest or Catch2** — Automated testing.
- **Python and Matplotlib** — Optional numerical experiments and visualization.

The core library is intended to remain independent of visualization and experimentation tools.

## Design principles

### Explicit coordinate conventions

Coordinate frames, transformation direction, vector conventions, and units must be documented and consistent throughout the API.

### Mathematical invariants

Operations are validated against their defining properties. For example, a valid rotation matrix must satisfy:


$$R^\mathsf{T}R=I,\qquad \det(R)=1$$

### Numerical robustness

Tests cover floating-point tolerances, invalid inputs, degenerate configurations, and accumulated numerical error.

### Reproducible validation

Synthetic datasets with known ground truth are used to evaluate transformation recovery, projection accuracy, and implementation consistency.

### Real-time awareness

Selected operations will be evaluated for bounded memory usage and execution-time variability. Hard real-time suitability will not be claimed without platform-specific timing analysis and appropriate system-level verification.

## Build and test

Requirements:

- A C++20-compatible compiler.
- CMake.
- Eigen.
- A supported test framework.

Build instructions will be finalized as the initial implementation and dependency configuration stabilize.

The intended workflow is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Project roadmap

- [x] Establish coordinate-frame and transformation conventions.
- [/] Implement and test SO(3) operations.
- [ ] Implement quaternion conversions and operations.
- [ ] Implement SE(3) transformations.
- [ ] Implement pinhole camera projection.
- [ ] Implement rigid transformation estimation using SVD.
- [ ] Add numerical validation and reference comparisons.
- [ ] Develop a reproducible 3D visualization example.
- [ ] Benchmark selected operations and investigate real-time constraints.
- [ ] Evaluate optional language bindings and alternative implementations.

## Intended applications

- Robotics and autonomous systems.
- Camera calibration and sensor alignment.
- Visual odometry and perception pipelines.
- Geometric validation and numerical diagnostics.
- Research into predictable geometric computation for embedded systems.

## Project status and limitations

Astra Geometry is an experimental engineering project. Its API, numerical behavior, performance, and suitability for safety-critical or hard real-time systems remain subject to implementation and verification.

The project aims to make correctness and limitations measurable rather than assume them.

## License

License to be determined.
