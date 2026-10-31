"""Smoke tests for the benchmark harness (marker: bench).

Kept out of the default ``pytest`` run (see ``tool.pytest.ini_options``) because
they execute real timing loops. Run them explicitly with ``pytest -m bench``.

They assert the harness CONTRACT, not the timings: structure of the record,
bit-for-bit reproducibility of a call, and that the point-set scaling sweep
behaves (per-point cost amortizes as N grows).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make the repo-root ``benchmarks`` package importable regardless of pytest's
# rootdir/import-mode heuristics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks import cases  # noqa: E402
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
