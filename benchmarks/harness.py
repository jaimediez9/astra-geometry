"""Timing, memory and hashing utilities for the benchmark harness.

Standard library only: ``time``, ``tracemalloc``, ``hashlib``, ``json``,
``platform``, ``subprocess``. NumPy is imported only to detect array outputs.

Measurements are design signals for the planned C++20 core, not predictions of
on-target timing. Nothing here is used by the shipped library.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def git_rev() -> str:
    """Short HEAD revision, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def env_info() -> dict[str, str]:
    """Environment fingerprint so a report is comparable only to its peers."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_rev": git_rev(),
    }


@dataclass(frozen=True)
class Timing:
    median_ns: float
    iqr_ns: float
    min_ns: int
    max_ns: int
    repeats: int
    warmup: int


def time_call(
    fn: Callable[[], Any],
    *,
    repeats: int = 1000,
    warmup: int = 100,
) -> Timing:
    """Time ``fn`` per call and return median/IQR over ``repeats`` samples.

    A warmup pass runs first so the first call's import/cache cost is not in
    the sample. Wall-clock noise is handled by reporting the median and IQR
    rather than the mean; ``min``/``max`` are kept for spread visibility.
    """
    for _ in range(warmup):
        fn()

    samples: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)

    samples.sort()
    n = len(samples)
    median = statistics.median(samples)
    q1 = samples[n // 4]
    q3 = samples[(3 * n) // 4]

    return Timing(
        median_ns=float(median),
        iqr_ns=float(q3 - q1),
        min_ns=int(samples[0]),
        max_ns=int(samples[-1]),
        repeats=repeats,
        warmup=warmup,
    )


def peak_bytes(fn: Callable[[], Any]) -> int:
    """Peak Python-traced allocation (bytes) during a single call.

    Approximate by construction: ``tracemalloc`` does not see NumPy's own array
    allocator unless NumPy is built with the tracemalloc domain, so array
    buffers are undercounted. Reported as a lower-bound proxy, paired with a
    static count of array-shaped temporaries (see ``cases``).
    """
    fn()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def output_checksum(fn: Callable[[], Any]) -> str:
    """Stable hash of one call's exact output bytes (bit-for-bit signal)."""
    out = fn()
    if isinstance(out, np.ndarray):
        payload = np.ascontiguousarray(out).tobytes()
        header = f"{out.dtype}{out.shape}".encode()
    else:
        payload = repr(out).encode()
        header = b""
    return hashlib.sha256(header + payload).hexdigest()[:16]


CALIBRATION_REPEATS = 200


def make_calibration() -> Callable[[], Any]:
    """A fixed machine-state probe with a stable NumPy instruction mix.

    Its cost tracks the machine's current state (CPU frequency, boost, thermal),
    which was observed to swing the same code by 2-4x between runs. Dividing a
    case's time by the probe's time cancels that drift, so ratios stay
    comparable across runs and machines. The absolute ns are still reported for
    the record, but the *normalized* value is the one to compare.
    """
    a = np.arange(1.0, 10.0).reshape(3, 3)
    b = np.eye(3) * 1.5
    eye = np.eye(3)
    return lambda: np.allclose((a @ b).T @ a, eye, atol=1e-6, rtol=0.0)


def calibration_ns() -> float:
    """Median ns of the calibration probe under the machine's current state."""
    return time_call(
        make_calibration(), repeats=CALIBRATION_REPEATS, warmup=50
    ).median_ns


def timing_to_dict(timing: Timing) -> dict[str, Any]:
    return asdict(timing)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write deterministic JSON: sorted keys, stable separators, trailing NL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
