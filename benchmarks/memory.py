"""Peak-RSS measurement for the essential-matrix solve, per process.

What this measures and what it does not
--------------------------------------
Peak RSS (``ru_maxrss``) is a process-level high-water mark: it includes
whatever the linear-algebra backend actually reserves, which is exactly the
quantity the O(N^2) allocation showed up in. It is NOT an exact accounting of
LAPACK's internal allocations — it also carries the interpreter, NumPy and any
BLAS scratch — so every measurement is paired with a **sibling process that
builds the design matrix and stops**, and only the difference between the two
peaks is reported as the cost attributable to the factorization.

``tracemalloc`` cannot do this: NumPy's allocator is invisible to it (the
general harness says so too). Hence a separate module instead of an extra
column in the timing report, and the general harness is left untouched.

Why this has to run as its own process
-------------------------------------
``ru_maxrss`` is a high-water mark for the *process*, and a forked child starts
with its parent's resident set, which survives ``exec``: a child can therefore
never report a peak below whatever the parent held at fork time. If the forking
parent is already heavy, the child's own allocation is **masked** and the
increment is silently under-reported — measured here at ~55 MB instead of
~236 MB when the benchmark suite had warmed the parent. Linux's
``clear_refs=5`` (documented to reset the peak) leaves ``ru_maxrss`` unchanged on
this kernel, so the protocol relies on being forked from a *light* parent: run
this module as ``python -m benchmarks.memory`` (its CLI never touches a large
array) rather than importing it into a process that has already allocated one.

Every record carries ``startup_rss_mb`` — the floor the child inherited — so a
contaminated measurement is visible instead of silent.

Threads are pinned to one so BLAS scratch is not a moving part: the children
run with ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` and
``MKL_NUM_THREADS=1``.

Variants
--------
``baseline``  build the correspondences and the (N, 9) design matrix, then stop.
``library``   call ``estimate_essential_matrix``: the real, corrected path.
``reduced``   the same solve with an explicit reduced SVD.
``full``      the pre-correction allocation, kept for comparison only.

Run::

    python -m benchmarks.memory                 # table to stdout
    python -m benchmarks.memory --out path.json
    python -m benchmarks.memory --sizes 8,64,1000
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

# The sizes this correction is measured at.
SIZES: tuple[int, ...] = (8, 16, 64, 256, 1_000, 10_000, 100_000)

# The pre-correction path needs 8*N^2 bytes for U alone (74.5 GiB at N = 1e5),
# so it is only run where it fits comfortably. Above this it is reported as
# infeasible with its theoretical requirement instead.
FULL_MAX_N = 10_000

VARIANTS = ("baseline", "library", "reduced", "full")


def _intrinsics(fx, fy, cx, cy, skew=0.0):
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _correspondences(n, seed=4321):
    """Deterministic scene and its two views, built without the library."""
    rng = np.random.default_rng(seed)
    points = np.column_stack((
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(6.0, 12.0, n),
    ))
    k1 = _intrinsics(800.0, 780.0, 640.0, 360.0, skew=3.5)
    k2 = _intrinsics(760.0, 770.0, 610.0, 380.0)
    rotation = np.eye(3)
    translation = np.array([0.9, -0.3, 0.15])

    def pinhole(points_world, k, t):
        camera = (rotation @ points_world.T).T + t
        x_axis, y_axis, z_axis = camera.T
        u = k[0, 0] * x_axis / z_axis + k[0, 1] * y_axis / z_axis + k[0, 2]
        v = k[1, 1] * y_axis / z_axis + k[1, 2]
        return np.column_stack((u, v))

    pixels1 = pinhole(points, k1, np.zeros(3))
    pixels2 = pinhole(points, k2, translation)

    def normalized(pixels, k):
        homogeneous = np.hstack((pixels, np.ones((len(pixels), 1))))
        projected = homogeneous @ np.linalg.inv(k).T
        return projected[:, :2] / projected[:, 2:3]

    return normalized(pixels1, k1), normalized(pixels2, k2)


def _design_matrix(x1, x2):
    """The (N, 9) system, built with the library's own normalization."""
    from astra_geometry.essential import _hartley_normalize

    _, homogeneous1 = _hartley_normalize(x1)
    _, homogeneous2 = _hartley_normalize(x2)
    return np.einsum("ni,nj->nij", homogeneous2, homogeneous1).reshape(-1, 9)


def peak_rss_mb() -> float:
    """Process peak resident set size, in MiB (Linux reports kilobytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_child(variant: str, n: int) -> dict:
    """One measurement, in a fresh process, with BLAS threads pinned to one."""
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"

    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.memory", "--child", variant, str(n)],
        capture_output=True, text=True, env=environment,
        cwd=str(Path(__file__).resolve().parents[1]), check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _child(variant: str, n: int) -> int:
    # The inherited floor, sampled before any work. ``ru_maxrss`` is a
    # high-water mark for the PROCESS, and a forked child starts with its
    # parent's resident set — which survives the exec. Without subtracting this,
    # a parent that had already touched a large array made every child report
    # that floor and the increments collapsed to noise: measured 55 MB instead of
    # 236 MB when the benchmark suite had warmed the parent's memory. Each child
    # now reports its own floor, so the increment is independent of the parent.
    startup_rss = peak_rss_mb()

    x1, x2 = _correspondences(n)

    if variant == "library":
        from astra_geometry.essential import estimate_essential_matrix

        start = time.perf_counter()
        estimate_essential_matrix(x1, x2)
        elapsed = time.perf_counter() - start
        record = {"variant": variant, "n": n, "elapsed_s": elapsed,
                  "startup_rss_mb": startup_rss, "rss_mb": peak_rss_mb()}
    else:
        rows = _design_matrix(x1, x2)
        baseline_rss = peak_rss_mb()
        if variant == "baseline":
            record = {"variant": variant, "n": n, "elapsed_s": 0.0,
                      "startup_rss_mb": startup_rss, "rss_mb": baseline_rss,
                      "residual": None}
        else:
            start = time.perf_counter()
            if variant == "reduced":
                _, _, vt = np.linalg.svd(rows, full_matrices=False)
            else:
                _, _, vt = np.linalg.svd(rows, full_matrices=True)
            elapsed = time.perf_counter() - start
            residual = float(np.abs(rows @ vt[-1]).max() / np.linalg.norm(rows, 2))
            record = {"variant": variant, "n": n, "elapsed_s": elapsed,
                      "startup_rss_mb": startup_rss, "rss_mb": peak_rss_mb(),
                      "residual": residual}

    # Work done by this variant, above the floor it inherited.
    record["work_rss_mb"] = record["rss_mb"] - record["startup_rss_mb"]
    print(json.dumps(record, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, metavar=("VARIANT", "N"),
                        help="run one measurement in this process (internal)")
    parser.add_argument("--sizes", default=None,
                        help="comma-separated sizes (default: the full sweep)")
    parser.add_argument("--full-max", type=int, default=FULL_MAX_N,
                        help=f"skip the pre-correction variant above this N "
                             f"(default {FULL_MAX_N})")
    parser.add_argument("--out", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if args.child:
        variant, n = args.child[0], int(args.child[1])
        if variant not in VARIANTS:
            parser.error(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        return _child(variant, n)

    sizes = SIZES if not args.sizes else tuple(
        int(part) for part in args.sizes.split(",") if part
    )

    report: dict = {"schema": SCHEMA_VERSION, "sizes": list(sizes),
                    "full_max_n": args.full_max, "measurements": []}
    header = (f"{'N':>8} {'variant':>9} {'floor':>8} {'peak RSS':>10} "
              f"{'work':>8} {'factor.':>9} {'time':>10} {'residual':>10}")
    print(header)
    print("-" * len(header))
    print("floor = RSS inherited by the child at start; work = peak - floor;")
    print("factor. = work - work(baseline), the cost of the factorization itself.")

    for n in sizes:
        baseline = _run_child("baseline", n)
        baseline_work = baseline["work_rss_mb"]
        report["measurements"].append({"n": n, **baseline})
        print(f"{n:>8} {'baseline':>9} {baseline['startup_rss_mb']:>7.1f}M "
              f"{baseline['rss_mb']:>9.1f}M {baseline_work:>7.1f}M "
              f"{'-':>9} {'-':>10} {'-':>10}")

        variants = ["library", "reduced"]
        if n <= args.full_max:
            variants.append("full")
        for variant in variants:
            record = _run_child(variant, n)
            attributable = record["work_rss_mb"] - baseline_work
            record["reference_rss_mb"] = baseline["rss_mb"]
            record["attributable_rss_mb"] = attributable
            report["measurements"].append({"n": n, **record})
            residual = record.get("residual")
            residual_text = f"{residual:.2e}" if residual is not None else "-"
            print(f"{n:>8} {variant:>9} {record['startup_rss_mb']:>7.1f}M "
                  f"{record['rss_mb']:>9.1f}M {record['work_rss_mb']:>7.1f}M "
                  f"{attributable:>8.1f}M {record['elapsed_s']:>9.3f}s "
                  f"{residual_text:>10}")
        if n > args.full_max:
            theoretical = 8.0 * n * n / 2 ** 30
            print(f"{n:>8} {'full':>9} {'skipped':>8} {'-':>10} {'-':>8} "
                  f"{'-':>9} {'-':>10} needs {theoretical:.1f} GiB for U")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
