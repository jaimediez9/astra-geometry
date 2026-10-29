"""Reference fixtures for the C++ conformance harness must stay in sync.

Fast and deterministic, so this runs in the default suite: it regenerates the
fixture document from the library and asserts it matches the committed file.
Any behaviour change therefore fails here until the fixtures are regenerated
and reviewed — which is the intended signal, not an inconvenience.

The comparison follows the fixture's own contract rather than raw JSON
equality. Bit-identical regeneration is only guaranteed on the build that
generated the file: trigonometric and BLAS results can differ by a few ULP
(~1e-16) between machines and library builds. Exact comparison is reserved for
structure, metadata, booleans and vectors declared ``compare: "exact"``;
floating-point leaves declared ``close`` are compared with the vector's ``atol``
(rtol = 0), matching what the planned C++ conformance harness will do.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

# Make the repo-root ``benchmarks`` package importable under any pytest rootdir.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.reference import build_fixture_document  # noqa: E402

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "reference"
    / "vectors.json"
)

# Generated inputs are floating-point too, so they get the same tolerance the
# fixture declares for computed values.
INPUT_ATOL = 1e-12


def _assert_equivalent(committed, regenerated, path: str, atol: float) -> None:
    """Compare two JSON values: structure exactly, floats within ``atol``."""
    if isinstance(committed, dict):
        assert isinstance(regenerated, dict), f"{path}: expected an object"
        assert set(committed) == set(regenerated), (
            f"{path}: keys changed "
            f"(missing={sorted(set(committed) - set(regenerated))}, "
            f"extra={sorted(set(regenerated) - set(committed))})"
        )
        for key in committed:
            _assert_equivalent(committed[key], regenerated[key], f"{path}.{key}", atol)
        return

    if isinstance(committed, list):
        assert isinstance(regenerated, list), f"{path}: expected a list"
        assert len(committed) == len(regenerated), (
            f"{path}: length changed ({len(committed)} -> {len(regenerated)})"
        )
        for index, (left, right) in enumerate(zip(committed, regenerated)):
            _assert_equivalent(left, right, f"{path}[{index}]", atol)
        return

    if isinstance(committed, bool) or isinstance(regenerated, bool):
        assert committed is regenerated, f"{path}: {committed!r} != {regenerated!r}"
        return

    if isinstance(committed, (int, float)) and isinstance(regenerated, (int, float)):
        if isinstance(committed, int) and isinstance(regenerated, int):
            assert committed == regenerated, f"{path}: {committed} != {regenerated}"
            return
        if math.isnan(committed) or math.isnan(regenerated):
            assert math.isnan(committed) and math.isnan(regenerated), (
                f"{path}: {committed!r} != {regenerated!r}"
            )
            return
        delta = abs(float(committed) - float(regenerated))
        assert delta <= atol, f"{path}: |delta| = {delta:.3e} exceeds atol = {atol:.0e}"
        return

    assert committed == regenerated, f"{path}: {committed!r} != {regenerated!r}"


def _assert_document_equivalent(committed: dict, regenerated: dict) -> None:
    assert set(committed) == set(regenerated), "top-level schema changed"
    _assert_equivalent(committed["schema"], regenerated["schema"], "schema", 0.0)

    committed_vectors = {vector["name"]: vector for vector in committed["vectors"]}
    regenerated_vectors = {vector["name"]: vector for vector in regenerated["vectors"]}
    assert set(committed_vectors) == set(regenerated_vectors), (
        "vector set changed "
        f"(missing={sorted(set(committed_vectors) - set(regenerated_vectors))}, "
        f"extra={sorted(set(regenerated_vectors) - set(committed_vectors))})"
    )

    for name, vector in committed_vectors.items():
        regenerated_vector = regenerated_vectors[name]
        # name, module, function, compare and atol are metadata: exact.
        metadata = {k: v for k, v in vector.items() if k not in ("inputs", "expected")}
        regenerated_metadata = {
            k: v for k, v in regenerated_vector.items() if k not in ("inputs", "expected")
        }
        _assert_equivalent(metadata, regenerated_metadata, name, 0.0)

        _assert_equivalent(
            vector["inputs"], regenerated_vector["inputs"], f"{name}.inputs", INPUT_ATOL
        )
        expected_atol = (
            0.0 if vector["compare"] == "exact" else float(vector.get("atol", INPUT_ATOL))
        )
        _assert_equivalent(
            vector["expected"], regenerated_vector["expected"], f"{name}.expected", expected_atol
        )


def test_committed_fixture_matches_regeneration():
    assert FIXTURE.exists(), "missing fixture; run: python -m benchmarks.reference"
    on_disk = json.loads(FIXTURE.read_text(encoding="utf-8"))
    _assert_document_equivalent(on_disk, build_fixture_document())


def test_fixture_covers_the_frozen_chapters():
    document = build_fixture_document()
    assert document["schema"] == 1
    functions = {vector["function"] for vector in document["vectors"]}
    assert {
        "rotation_matrix3D",
        "compose_rotations",
        "quaternion_multiply",
        "compose_transforms",
        "transform_points",
    } <= functions


def test_structural_and_predicate_vectors_compare_exactly():
    document = build_fixture_document()
    exact = {v["name"] for v in document["vectors"] if v["compare"] == "exact"}
    assert any(name.startswith("is_") for name in exact)
    assert "is_transformation_matrix.bad_last_row" in exact
