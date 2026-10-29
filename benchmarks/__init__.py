"""Benchmark harness for the frozen astra-geometry API.

Measures per-call time, peak traced memory and output hashes for the SO(3),
quaternion and SE(3) chapters, and emits a deterministic JSON report. The same
JSON fixtures are meant to feed the planned C++ conformance harness.

Standard library only; NumPy is imported because the library under measurement
uses it. See the project note, section 11, for scope and rationale.
"""
