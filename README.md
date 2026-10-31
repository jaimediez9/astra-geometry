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
benchmarks/           Timing, memory and validation-cost measurements (stdlib harness)
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
| SE(3) | Rigid transformations, composition, and inversion | implemented |
| Camera geometry | Intrinsics, extrinsics, and 3D-to-2D projection | implemented (pinhole; lens distortion out of scope) |
| Triangulation | 3D reconstruction from two views | implemented (linear DLT) |
| Epipolar geometry | Essential matrix and relative pose up to scale | implemented (calibrated; essential only) |
| Estimation | Rigid alignment from point correspondences | implemented (3D-3D Kabsch; no scale, no outliers) |
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

SE(3) operations are importable from their submodule only, under the same policy:

```python
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
```

Camera operations are importable from their submodule only, under the same policy:

```python
from astra_geometry.camera import (
    backproject_pixels,
    intrinsic_matrix,
    is_intrinsic_matrix,
    project,
    project_points,
    reprojection_residual,
    world_to_camera,
)
```

Estimation operations are importable from their submodule only, under the same policy:

```python
from astra_geometry.estimation import estimate_rigid_transform
```

Triangulation and epipolar operations are importable from their submodules only:

```python
from astra_geometry.triangulation import projection_matrix, triangulate, triangulate_points
from astra_geometry.essential import (
    decompose_essential_matrix,
    essential_matrix,
    estimate_essential_matrix,
    estimate_relative_pose,
    is_essential_matrix,
    select_candidate,
)
```

Validation predicates (`is_rotation_matrix`, `is_unit_quaternion`, `is_transformation_matrix`) return `False` for invalid input instead of raising; constructors and operations raise `ValueError` on violated contracts.

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

## Benchmarks

`benchmarks/` measures the frozen SO(3), quaternion and SE(3) API with a standard-library harness (`time`, `tracemalloc`, `hashlib`). It is a **design signal** for the planned C++20 core — how many temporaries and allocations the API makes per call — not a prediction of on-target timing.

```bash
python -m benchmarks.run                                   # all cases, JSON to stdout
python -m benchmarks.run --repeat 300 --out benchmarks/results/bench.json
pytest -m bench                                            # harness smoke tests
```

Per case it records per-call timing (median/IQR/min/max), approximate peak memory, a static count of array-shaped temporaries, a bit-for-bit output checksum, and the scaling sweep for the point-set operations. It also reports the **validation-cost ratio**: the cost of each `is_*` predicate against the operation it guards — the input that decides whether validated and trusted paths must be separated in the core.

The registry covers the frozen SO(3), quaternion and SE(3) chapters plus camera (`project_points`, `reprojection_residual`), estimation, triangulation and epipolar geometry. The post-freeze operations have their **own size sweeps** — a real minimum `N` (Kabsch needs 3, the 8-point method needs 8) and, for the essential-matrix family, an `O(N²)` intermediate in the current implementation. The shared sweep used by the original cases is untouched, so those measurements stay comparable.

**Measurement caveat.** On a machine without a pinned CPU frequency, absolute times for identical code swung 2–4× between runs. Every measurement is therefore bracketed by a fixed calibration probe and reported **normalized** (case time / probe time). Raw nanoseconds are kept for the record only.

**How to read these numbers.** The normalized values are **indicative comparisons, not absolute performance measurements**. They rank operations within a run and compare a run against another on the same machine; they are not a characterisation of the implementation's cost.

- **The normalisation is approximate, and small moves in a ratio are noise.** The calibration probe has a different instruction mix from the case it brackets, so dividing by it cancels CPU-frequency drift only imperfectly: the probe's own reading varies by up to ~3× across a single run as the CPU ramps (`calibration_ns` per case is in the report). Measured dispersion of a validation ratio at `N = 1` — the noisiest point of a sweep, where a fixed-cost predicate is compared against a call dominated by per-call overhead — reaches **≈ 5–9 % between runs**, and the same ratio moved in both directions across repeated runs with no code change. That figure is a **description of the observed spread, not a tolerance**: nothing here is enforced as pass/fail, and no band should be read off it as a guarantee. Ratios away from `N = 1` are markedly tighter (see below).
- **The memory signal is approximate too.** `static_temporaries` is a **hand count of array-shaped temporaries per call, not a measurement** — it is an estimate made by reading the code, and it is labelled as such in `benchmarks/cases.py`. `peak_bytes` is measured, but with `tracemalloc`, which **cannot see NumPy's own allocator** and therefore **can understate the real reservation**; read it as a lower bound. Both stand in until the C++ core can be instrumented with a real allocator hook.

`benchmarks/reference/vectors.json` holds per-operation reference vectors — serializable inputs plus the contract each call must satisfy — generated by `python -m benchmarks.reference`. The planned C++20 conformance harness loads the same file, so correctness is defined once and checked twice; a test in the default suite fails if the committed file drifts from the library.

**Two schema versions, both readable.** *Schema 1* vectors carry `expected` plus a `compare` mode: `exact` for structure and booleans, `close` for floats compared with the vector's `atol` and `rtol = 0`. *Schema 2* adds `expected_error`, which lets a vector declare the exception type a call must raise **instead of** a value — the way degeneracies are pinned. `expected_error` is mutually exclusive with `expected` and `compare`, and the match is on the exact exception class name (`isinstance` would wrongly accept `np.linalg.LinAlgError`, a `ValueError` subclass, failing deep inside an SVD). `verify_document` executes a document of either version against the library — the reader the C++ harness mirrors — and the pre-extension schema-1 document is preserved verbatim as `vectors-schema1.json`, so compatibility is demonstrated by running it rather than by inspection.

**Reference vectors for the post-freeze chapters** (camera, estimation, triangulation, essential) are built from **independent mathematics** written in `benchmarks/reference.py` — an explicit cross-product matrix, Rodrigues, the hand-written pinhole formula, an exact ground-truth pose — never from the implementation's own output, so they check correctness instead of only detecting drift. A test guards that mechanically: the vector builder must not call any function of the package under test.

## Rigid estimation (Kabsch)

`estimation.py` solves the **3D-3D** problem: given two point clouds whose points are already in correspondence — `source` in frame A and `target` in frame B — find the rigid transform $T_{B\leftarrow A}$ that best aligns them in the least-squares sense:

$$\min_{R\in\mathrm{SO}(3),\,t}\sum_i\lVert q_i-(Rp_i+t)\rVert^2$$

```python
from astra_geometry.estimation import estimate_rigid_transform

T = estimate_rigid_transform(source, target)   # (N,3), (N,3) -> (4,4)
```

**What it needs.** Two `(N, 3)` arrays with the same number of rows and a known pairing, and at least three correspondences whose points are **not collinear**. Coplanar but non-collinear clouds are accepted. The algorithm is Kabsch's: centre both clouds, build the cross-covariance, take its SVD, and correct the sign of the last singular vector so that the result is a proper rotation ($\det R=+1$).

**What it returns.** A `4×4` rigid transform in the same representation as `transforms.py`, so it composes with `compose_transforms`, applies with `transform_points` and feeds `camera.py`. It never estimates scale.

**Limitations.** Rigid only (no scale); no outlier handling — a single bad correspondence contaminates the whole solution, because there is no RANSAC; and the degeneracy test is *absolute* (`atol`), so a cloud of extremely small extent can be rejected even when it is not collinear.

**Why 3D-3D, and how it differs from its neighbours.**

- **3D-3D (this chapter).** Two matched point clouds. There is a closed-form least-squares solution whose core is an SVD. This is what a LiDAR/stereo depth pair, a structured-light sensor, or a marker set gives you.
- **PnP (3D-2D).** 3D points against image pixels. Projection is projective, so the problem is **non-linear** and is solved by iterative least squares (Levenberg-Marquardt), not by SVD. *Out of scope.*
- **Epipolar geometry (2D-2D).** Two images. SVD yields the essential/fundamental matrix, but that is a *constraint between views*, not a rigid transform: it does not directly say where the camera is. *Out of scope.*

Consequently the reprojection residual in `camera.py` is not what Kabsch minimises — Kabsch minimises distances between 3D points. The residual serves to *validate* an estimated pose on synthetic data, and it would be the objective of a future PnP.

## Epipolar geometry and triangulation

Two chapters cover the two-view geometry that sits between the camera model and visual odometry.

### Triangulation — 2D → 3D

Given a point observed in two views, with both cameras known, reconstruct its 3D position. Taking frame 1 as the reference, the projection matrices are `P1 = K1 [I | 0]` and `P2 = K2 [R | t]`, with `R, t` from `T_{2←1}`. Each observation gives two independent equations, so two views give four equations for the homogeneous point, solved by SVD.

```python
from astra_geometry.triangulation import triangulate

points = triangulate(pixels1, pixels2, K1, K2, T_2_from_1)   # (N,2), (N,2) -> (N,3)
```

The result is defined only up to a scale whose **sign is meaningful**: the third coordinate is the depth along the optical axis, which is exactly what the cheirality test consumes. If the two rays are (nearly) parallel the point lies (nearly) at infinity; that case is **rejected** rather than returning an unbounded depth.

### Epipolar geometry — 2D → 2D

Given correspondences between two **calibrated** views, estimate the essential matrix and recover the relative pose. With `x_i` the normalized image coordinates (`x̂ = K⁻¹ [u, v, 1]ᵀ`) and `(R, t) = T_{2←1}`:

$$x_2^\mathsf{T} E\,x_1 = 0, \qquad E = [t]_\times R$$

> [!WARNING]
> The convention is normative (conventions.md 6.2) and **cannot be detected by a round-trip test**: `[t]_× R` and `R [t]_×` satisfy superficially similar constraints but belong to opposite pairings. Implementations must follow the stated definition.

```python
from astra_geometry.essential import estimate_relative_pose

R, t_hat = estimate_relative_pose(pixels1, pixels2, K1, K2)   # unit direction of t
```

Estimation uses the **normalized 8-point** algorithm (Hartley normalization, then projection onto the essential manifold: rank 2 with equal non-zero singular values). The decomposition of `E` yields four candidate poses; `decompose_essential_matrix` exposes all four and `select_candidate` picks the physically valid one by **cheirality** (positive depth in both cameras). The candidate list is a **set** — because the two non-zero singular values are equal, its ordering is not stable and must not be relied upon.

**Limits, collected in one place.**

- `E` is defined up to scale, so **only the direction of `t` is observable**; the metric baseline requires additional information. `estimate_relative_pose` therefore returns a **unit** `t̂`.
- The **fundamental matrix** (unknown intrinsics) is out of scope. The relation `F = K₂⁻ᵀ E K₁⁻¹` is documented for reference only.
- Only 2D-2D (epipolar) and 3D-3D (Kabsch) are covered. **PnP (3D-2D)** is a non-linear problem and is **not** implemented.
- Degenerate two-view configurations — pure rotation (`t ≈ 0`), planar scenes for a point-based estimator, collinear correspondences, negligible parallax — are rejected or reported.

## Project roadmap

- [x] Establish coordinate-frame and transformation conventions.
- [x] Implement and test SO(3) operations.
- [x] Complete quaternion support: composition, inversion, and conversion to and from rotation matrices. *Inversion is exposed as the conjugate, which is the inverse for unit quaternions; a general (non-unit) inverse is not implemented.*
- [x] Implement SE(3) transformations. *Validation, construction, composition, inversion and application to points are done. The last row of a homogeneous matrix is required to be exactly [0, 0, 0, 1]: it is structure, not an approximated geometric property, and carries no tolerance.*
- [x] Implement pinhole camera projection. *Intrinsics, extrinsics as SE(3), projection, back-projection with depth, and a reprojection residual. Lens distortion is out of scope (not in conventions.md section 6).*
- [x] Implement rigid transformation estimation using SVD. *Kabsch for 3D-3D correspondences. No scale estimation and no outlier handling (no RANSAC); degenerate (collinear) configurations are rejected.*
- [x] Implement triangulation from two views. *Linear DLT; degeneracy (point at infinity, no parallax) is rejected.*
- [x] Implement epipolar geometry. *Essential matrix for the calibrated case: normalized 8-point estimation, decomposition into the four pose candidates, and selection by cheirality. The fundamental matrix and the 5-point minimal solver are out of scope.*
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
