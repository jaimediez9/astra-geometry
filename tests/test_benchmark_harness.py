"""Smoke tests for the benchmark harness (marker: bench).

Kept out of the default ``pytest`` run (see ``tool.pytest.ini_options``) because
they execute real timing loops. Run them explicitly with ``pytest -m bench``.

They assert the harness CONTRACT, not the timings: structure of the record,
bit-for-bit reproducibility of a call, and that the point-set scaling sweep
behaves (per-point cost amortizes as N grows).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

# Make the repo-root ``benchmarks`` package importable regardless of pytest's
# rootdir/import-mode heuristics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import cases  # noqa: E402
from benchmarks import memory as memory_measurement  # noqa: E402
from benchmarks.harness import time_call  # noqa: E402
from benchmarks.run import _measure_case, _validation_ratios  # noqa: E402

pytestmark = pytest.mark.bench


def test_registry_covers_the_frozen_chapters():
    names = {c.name for c in cases.CASES}
    assert {
        "is_rotation_matrix",
        "quaternion_multiply",
        "is_transformation_matrix",
        "transform_points",
    } <= names


def test_registry_covers_the_post_freeze_chapters():
    by_module = {c.module for c in cases.CASES}
    assert {"camera", "estimation", "triangulation", "essential"} <= by_module
    assert {
        "is_intrinsic_matrix",
        "project_points",
        "reprojection_residual",
        "estimate_rigid_transform",
        "triangulate",
        "is_essential_matrix",
        "estimate_essential_matrix",
        "estimate_relative_pose",
    } <= {c.name for c in cases.CASES}


def test_every_case_builds_and_runs_at_its_smallest_size():
    """A case whose inputs violate its function's contract would die here."""
    for case in cases.CASES:
        size = case.sizes[0] if case.sizes else 0
        case.make(size)()  # must not raise


def test_post_freeze_sizes_respect_each_operations_minimum():
    def sizes_of(name: str) -> tuple[int, ...]:
        case = next(c for c in cases.CASES if c.name == name)
        assert case.sizes is not None, f"{name} must sweep sizes"
        return case.sizes

    assert min(sizes_of("estimate_rigid_transform")) >= 3   # Kabsch minimum
    assert min(sizes_of("estimate_essential_matrix")) >= 8  # 8-point minimum
    assert min(sizes_of("estimate_relative_pose")) >= 8
    # The shared sweep is untouched, so the original cases are comparable.
    assert cases.SIZES == (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)


def test_time_call_reports_positive_median_and_params():
    timing = time_call(lambda: sum(range(100)), repeats=5, warmup=2)
    assert timing.median_ns > 0.0
    assert timing.repeats == 5
    assert timing.warmup == 2


def test_measure_case_is_reproducible_bitwise():
    case = next(c for c in cases.CASES if c.name == "compose_transforms")
    record = _measure_case(case, repeats=5, warmup=2)
    assert record["reproducible_bitwise"] is True
    assert record["timing"]["median_ns"] > 0.0
    assert record["memory"]["peak_bytes"] >= 0


def test_point_set_scaling_amortizes_per_point_cost():
    case = next(c for c in cases.CASES if c.name == "transform_points")
    record = _measure_case(case, repeats=3, warmup=1)
    assert "1" in record["scaling"] and "100" in record["scaling"]
    # ns/point at N=100 must be below ns/point at N=1 (fixed overhead amortized)
    assert record["scaling"]["100"][1] < record["scaling"]["1"][1]


def test_validation_ratios_are_computed():
    wanted = {"is_transformation_matrix", "compose_transforms"}
    records = {
        c.name: _measure_case(c, repeats=3, warmup=1)
        for c in cases.CASES
        if c.name in wanted
    }
    ratios = _validation_ratios(records)
    key = "is_transformation_matrix / compose_transforms"
    assert key in ratios
    assert ratios[key]["ratio"] > 0.0


def _run_module(sizes: str, destination: pathlib.Path) -> dict:
    """Run this module as its own light process and read back its report.

    Spawning the children from *here* would inherit this process's resident set
    (see the module docstring), so the measurement goes through one extra,
    deliberately light layer: the CLI never holds a large array.
    """
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    subprocess.run(
        [sys.executable, "-m", "benchmarks.memory", "--sizes", sizes,
         "--out", str(destination)],
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        env=environment, check=True, capture_output=True, text=True,
    )
    return json.loads(destination.read_text(encoding="utf-8"))


def _by_variant(report: dict, n: int) -> dict:
    return {m["variant"]: m for m in report["measurements"] if m["n"] == n}


def test_memory_measurement_sees_the_pre_correction_allocation(tmp_path):
    """The measurement must SEE the O(N^2) difference it exists for.

    At N = 4000 the pre-correction full SVD reserves ~236 MB where the reduced
    one reserves ~1.5 MB — two orders of magnitude, far above the noise floor.
    """
    report = _run_module("4000", tmp_path / "memory.json")
    measured = _by_variant(report, 4_000)

    assert measured["baseline"]["residual"] is None  # it only builds the system
    assert measured["reduced"]["residual"] < 1e-14
    assert measured["full"]["residual"] < 1e-14
    assert measured["reduced"]["attributable_rss_mb"] < 10.0, \
        "the corrected path must stay flat"
    assert measured["full"]["attributable_rss_mb"] > 100.0, \
        "the pre-correction path must be visible"


def test_memory_measurement_is_robust_to_a_heavy_parent(tmp_path):
    """A heavy parent must not shrink the reported cost.

    Before the protocol was fixed, a parent holding a large array made every
    child report that floor and the measured cost of the pre-correction path
    collapsed from ~236 MB to ~55 MB — a silent under-report of exactly the bug
    this module exists to expose.
    """
    ballast = np.full(200_000_000 // 8, 1.0)  # ~200 MB held by THIS process
    assert ballast.sum() > 0.0                # touched, so it is resident

    report = _run_module("4000", tmp_path / "memory.json")
    measured = _by_variant(report, 4_000)

    # The extra process layer keeps the children's floor small...
    assert measured["baseline"]["startup_rss_mb"] < 150.0
    # ...so the attributable cost is the real one.
    assert measured["full"]["attributable_rss_mb"] > 100.0


def test_the_inherited_floor_documents_why_an_extra_layer_is_needed():
    """Pin the mechanism: a child forked from a heavy parent inherits its floor.

    This is why the module is run as a subprocess rather than imported here.
    """
    ballast = np.full(200_000_000 // 8, 1.0)
    assert ballast.sum() > 0.0
    record = memory_measurement._run_child("baseline", 8)
    assert record["startup_rss_mb"] > 150.0, \
        "the child should inherit this process's resident set"


def test_memory_measurement_exposes_the_wrong_short_case():
    """At N = 8 the reduced path is not the solution, and the module shows it."""
    baseline = memory_measurement._run_child("baseline", 8)
    reduced = memory_measurement._run_child("reduced", 8)
    full = memory_measurement._run_child("full", 8)

    assert baseline["residual"] is None
    assert reduced["residual"] > 1e-3   # a different vector, not the null one
    assert full["residual"] < 1e-14     # the actual solution


def test_memory_module_keeps_the_previous_sizes_in_its_sweep():
    for size in (8, 16, 64, 256):
        assert size in memory_measurement.SIZES
    assert memory_measurement.SIZES == (8, 16, 64, 256, 1_000, 10_000, 100_000)
