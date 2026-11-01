# Astra Geometry — C++20 conformance spike

A **vertical spike**, not the port. It exists to measure the risks of porting the
frozen mathematics to C++20 before committing to the port itself, and to answer
three questions with evidence:

1. Does the port reproduce the Python reference numerically? (§2)
2. Is the toolchain reproducible, and where does it bite? (§3)
3. Do the critical operations allocate, and are the guards that claim so
   trustworthy? (§4)

## What this is NOT claiming

- **Not** a real-time, WCET, latency or determinism result. Nothing here measures
  whether the code is fit for a hard real-time target; that needs a specific
  target and explicit time budgets, and it is a later milestone.
- **Not** performance work. There is no optimisation, no benchmarking of speed
  and no vectorisation effort. The only performance-adjacent claim is about
  memory, for the reason in §4.
- **Not** the full library. Only SO(3), SE(3) and the quaternion algebra are
  here: `camera`, `estimation`, `triangulation` and `essential` are out of scope.
- **Not** the frozen API of the Python package. The build never touches
  `src/astra_geometry`, the fixtures, `benchmarks/` or the README of the Python
  tree.

## 1. Layout

```
include/astra/tolerance.hpp   Tolerance as a distinct type (§7.2 of conventions)
include/astra/types.hpp       fixed-size aliases; points are (N, 3) row-major
include/astra/so3.hpp         6 operations of rotation.py
include/astra/quaternion.hpp  6 operations of quaternions.py
include/astra/se3.hpp         8 operations of transforms.py
tests/conformance.cpp         the runner: consumes the Python fixtures
tests/fixture_io.hpp          JSON, only in the harness
tests/dispatch.hpp            fixture function name -> core, 1:1 with Python
tests/test_*.cpp              native property tests and the API-shape contracts
tests/memory_guard.hpp        the three allocation layers
tools/interpose_malloc.c      LD_PRELOAD malloc counter (Linux/glibc only)
```

The core (`include/astra/**`) includes no JSON, no Catch2 and no I/O header. That
boundary is the point: the port must not grow a serialization dependency.

## 2. Conformance: the same fixtures, run twice

The runner consumes `benchmarks/reference/vectors.json` (schema 2) and
`vectors-schema1.json` (schema 1) **without modifying either**, and honours the
`compare` mode the fixture carries:

- `close`: equal shapes and `max|produced - expected| <= atol`, `rtol = 0`, using
  the fixture's own `atol` (1e-12 for the whole subset). No tolerance is invented.
- `exact`: booleans compared as booleans (all eleven `exact` vectors in the file
  are boolean).
- `expected_error`: matched by exact exception class name, never by `isinstance`.

A vector's `atol` is a **comparison** tolerance. It is never forwarded as an
argument: the reference calls `function(**inputs)`, and forwarding it would
silently change the operation's behaviour.

Schema compatibility is established **by execution**, never by inspecting the
documents: both files are run, and the 23 shared names must produce identical
outcomes (`--compare-outcomes-with`). Coverage is reported with its gaps:

```
vectors      : 44 total, 23 covered by this spike, 21 not covered
```

The 21 uncovered vectors are listed with their reason (`module 'camera' is
outside this spike`, `error contract: this spike covers value contracts only`),
so "everything passes" can never mean "I only ran what I wrote". Note the honest
limitation: the SO(3)/SE(3)/quaternion subset contains **no** `expected_error`
vector, so the error branch of the harness is exercised in the parser but not
dispatched at operation level.

## 3. Toolchain: what is pinned and what was learned

| Dependency | Version | Pin |
|---|---|---|
| Eigen | 3.4.0 | commit `3147391d946bb4b6c68edd901f2add6ac1f31f8c` |
| Catch2 | 3.16.1 | commit `08092139210881f01a34e60c5f2fd6afa0ea4024` |
| nlohmann/json | 3.12.0 | commit `55f93686c01528224f448c19128836e7df245f72` (harness only) |

`find_package` is tried first; FetchContent is the fallback, pinned by **commit
SHA** rather than tag, because tags are mutable and a commit is content-addressed
— which is what makes the pin survive a flaky host.

### Pin precedence and caches

Where a dependency version comes from, highest precedence first:

1. an explicit `-D<VAR>=<value>` on the configure line;
2. the default declared in `cmake/Dependencies.cmake`;
3. **never** a value left in a build directory by an earlier revision of that
   file — such an entry is purged, with a warning.

Point 3 is not hypothetical, and it cost a real measurement. The pins were
originally declared with `set(... CACHE STRING ...)`, and a pre-existing cache
entry takes precedence over such a declaration: replacing the nlohmann/json pin
had **no effect at all** on the four build directories that already existed, and
the generated build header kept reporting the old value while the source declared
the new one. A pin that cannot be changed is not a pin.

The mechanism separates the two kinds of cache entry by the type CMake itself
assigns: `-DVAR=value` creates an `UNINITIALIZED` entry, which is honoured, while
one written by a previous revision of the file is a `STRING` entry, which is
purged. The pins are no longer written to the cache at all, so a build directory
cannot hold a stale pin in the first place. The behaviour was measured with a
minimal CMake project before being adopted here — clean configure → file default;
`-D` → honoured; injected `STRING` → purged — and then verified on this project
by injecting the old tag-object SHA into a build directory and reconfiguring.

One consequence of CMake's own cache semantics: a `-D` override is **sticky** for
that build directory. It applies to later configures until the entry is removed
(delete the line from `CMakeCache.txt`, or configure a fresh directory).

Three toolchain findings, all measured here:

- **The official Eigen host (GitLab) was unreachable** during these builds
  (TLS error), so the SHA was read from the GitHub mirror. The configure script
  probes the official URL and falls back automatically, printing a warning. The
  SHA is what pins the content, not the host.
- **Eigen writes to the user package registry.** Running Eigen's own CMakeLists
  calls `export(PACKAGE Eigen3)`, writing `~/.cmake/packages/Eigen3` and storing
  `Eigen3_DIR` in the cache; the next configure then reported *"using system
  Eigen3 3.4.0"* on a machine with **no** Eigen installed, silently replacing the
  pinned commit with a directory inside the build tree. Fixed at the source:
  Eigen is populated with `SOURCE_SUBDIR` pointing at a directory that has no
  `CMakeLists.txt`, so its subproject is never added. Only its headers are needed.
- **Eigen registers ~950 of its own tests with CTest**, which buried the spike's
  six. The same `SOURCE_SUBDIR` fix removes that too.

Versions, flags and the fixture digests are recorded in
`build/<preset>/generated/astra_build_info.hpp` and in every conformance report.
The runner accepts `--expect-fixture-sha`, so a report produced from a stale
fixture can be rejected rather than believed.

## 4. Memory: three layers, and what each one can see

No single guard is sufficient, so there are three, and the positive controls
prove that each one actually fires:

| Layer | Mechanism | Portable? | Sees |
|---|---|---|---|
| 1 | replaced `operator new` / `new[]` | yes | std library and our code |
| 2 | `EIGEN_RUNTIME_NO_MALLOC` + `set_is_malloc_allowed(false)` | yes | Eigen heap use |
| 3 | `LD_PRELOAD` malloc interposition | **no — Linux/glibc only** | everything, including inside dependencies |

Two traps, both found by measurement rather than assumption:

- **Layer 2 is vacuous in a plain Release build.** Verified in Eigen 3.4.0's
  `Macros.h`: `NDEBUG` defines `EIGEN_NO_DEBUG`, which turns `eigen_assert` into a
  no-op — and `check_that_malloc_is_allowed()` *is* an `eigen_assert` (it is the
  first statement of `aligned_malloc`). The `memory_guards` target is therefore
  compiled **without `NDEBUG` while keeping `-O2`**, and a test fails loudly if
  that ever changes.
- **Eigen does not use `operator new`.** `aligned_malloc` reaches `std::malloc`
  (on glibc/LP64 it takes the `EIGEN_MALLOC_ALREADY_ALIGNED` path), so layer 1
  alone would report zero for every Eigen allocation. The positive control shows
  this directly: a dynamic `Eigen::MatrixXd` is seen by layers 2 and 3 and by
  layer 1 not at all.

**Layer 3 is a host-specific measurement, not a portable property.** It relies on
the dynamic linker resolving the symbols ahead of libc's and on glibc's internal
entry points. Where it cannot be used, the harness reports `malloc_layer = -1`
("not measured") and never zero.

### What was measured (Release, gcc, `-O2`, BLAS not involved)

Eighteen operations return fixed-size values and allocate **nothing**, in all
three layers:

```
is_rotation_matrix  rotation_matrix3D  rotation_matrix2D  compose_rotations
inverse_rotation    is_unit_quaternion normalize_quaternion conjugate_quaternion
quaternion_multiply rotation_matrix_to_quaternion quaternion_to_rotation_matrix
is_transformation_matrix transformation_matrix identity_transform
rotation_part translation_part compose_transforms inverse_transform
```

The two batch operations have a contract to **return a new (N, 3) array**, so the
honest claim is not "zero" but "constant in N":

```
rotate_points      n=3     eigen=1  malloc=1  cxx_new=0
rotate_points      n=2000  eigen=1  malloc=1  cxx_new=0
transform_points   n=3     eigen=1  malloc=1  cxx_new=0
transform_points   n=2000  eigen=1  malloc=1  cxx_new=0
```

One result buffer, no per-point scratch.

**Scope of this claim:** only the operations and the code paths actually executed
between `arm()` and `disarm()`. Error paths are excluded on purpose (throwing
allocates) and so is everything the harness does. The numbers characterise these
eighteen fixed-size operations on this host; they are not a statement about the
whole library.

## 5. Build and test

```sh
cmake --preset release-gcc      # or debug-gcc, release-clang, debug-clang
cmake --build --preset release-gcc
cd build/release-gcc && ctest --output-on-failure

# the third memory layer needs the interposer to be preloaded
LD_PRELOAD=./astra_interpose_malloc.so ./memory_guards

# conformance on both schemas, with the outcome comparison
./conformance --fixtures ../../../benchmarks/reference/vectors.json --label schema2 \
              --out schema2.json
./conformance --fixtures ../../../benchmarks/reference/vectors-schema1.json --label schema1 \
              --out schema1.json --compare-outcomes-with schema2.json
```

The fixtures are referenced in place from `../benchmarks/reference`; nothing is
copied into the C++ tree, so the two implementations cannot drift apart silently.

## 6. Traps found while building this (worth carrying into the port)

- **`NDEBUG` removes Eigen's own checks**, twice over: Eigen's shape assertions
  *and* the allocation check. A Release build is not a build where Eigen
  validates your expressions. An invalid expression was written here
  (`matrix_product + row_vector`, which Eigen does not broadcast) and, with the
  assertion compiled out, it read out of bounds and produced plausible wrong
  numbers instead of aborting: the Debug build aborts on it. **Run the Debug
  configuration in CI**, not just Release.
- **`determinant()` is not in `<Eigen/Core>`**; it lives in `<Eigen/LU>`. The
  link error is confusing until you know.
- **A tag object is not a commit.** `refs/tags/v3.12.0` of nlohmann/json is an
  *annotated* tag whose object SHA differs from the commit it points to, so
  pinning the tag object records an identifier that is not the checked-out tree —
  a reader comparing the two sees a mismatch that is not an error. Pin the peeled
  commit; `git ls-remote --tags` prints it on the `^{}` line. Eigen and Catch2
  turned out to have lightweight tags and commit pins, so only this one was
  affected, and it was found by auditing the checkout rather than the declaration.
- **Eigen's expression templates**: `auto x = A * B;` is not a matrix, it is an
  expression referring to `A` and `B`. Assign to concrete fixed-size types.
- **The optimiser will delete what you are measuring.** A first version of the
  positive control built a `std::string` from a literal and measured zero
  allocations, because the construction was folded away. Measurements need an
  opaque sink (and a `volatile` one for good measure).
- **`q * q*` is the identity to round-off, not bit for bit**: the scalar part is
  exactly 1, the vector part keeps ~1e-17 from cancelling cross terms. A test
  demanding bit equality there is a wrong test, not a failing port.
- **Argument order is the highest-risk convention** and the port keeps the
  reference's: `compose_rotations(R1, R2) == R2 @ R1` and
  `quaternion_multiply(q1, q2) == q2*q1`. Both are pinned by tests that also
  assert the *other* order disagrees, so the assertion cannot pass vacuously.

## 7. Recommendations for the full port

1. **Keep the header-only, fixed-size discipline.** It is what makes the
   eighteen operations allocation-free; a dynamic container in a hot signature
   silently reintroduces the heap.
2. **Keep the conformance runner as the acceptance gate**, and grow it module by
   module. The fixtures are already the single definition of correctness; the
   `--expect-fixture-sha` argument makes a stale input detectable.
3. **Do not add JSON or Catch2 to the core.** The gap between `include/astra` and
   `tests/` is the only reason the core has no serialization dependency.
4. **Build Debug in CI.** Two classes of error in this spike were invisible in
   Release and loud in Debug.
5. **Keep `Tolerance` a distinct type.** It reproduces the keyword-only rule of
   §7.2 in the type system, and the static assertions pin that the set of
   functions taking a tolerance is exactly the set that takes `atol` in Python.
6. **Decide the validation/trustworthy split before porting more modules.** The
   spike shows the validation cost per call is measurable; what it does not
   decide is whether the trusted path skips it.
