"""Reference fixtures for the C++ conformance harness must stay in sync.

Fast and deterministic, so this runs in the default suite: it regenerates the
fixture document from the library and asserts it matches the committed file.
Any behaviour change therefore fails here until the fixtures are regenerated
and reviewed — which is the intended signal, not an inconvenience.
"""

from __future__ import annotations

import json
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


def test_committed_fixture_matches_regeneration():
    assert FIXTURE.exists(), "missing fixture; run: python -m benchmarks.reference"
    on_disk = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert on_disk == build_fixture_document()


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
