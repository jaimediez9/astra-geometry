"""CLI runner for the astra-geometry benchmark suite.

Usage::

    python -m benchmarks.run                       # all cases, all sizes
    python -m benchmarks.run --select rotation     # substring filter
    python -m benchmarks.run --repeat 200 --out benchmarks/results/bench.json

Emits a deterministic JSON report (sorted keys) containing, per case: per-call
timing (median/IQR/min/max), approximate peak memory, the static temporary
count, a bit-for-bit output checksum, and the scaling sweep for point-set
operations. It also computes the headline validation-cost ratios: the cost of
each ``is_*`` predicate against the operation it guards.

The JSON shape is intended to be consumed unchanged by the planned C++
conformance harness.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from . import cases
from .harness import (
    calibration_ns,
    env_info,
    output_checksum,
    peak_bytes,
    time_call,
    timing_to_dict,
    write_json,
)

SCHEMA_VERSION = 1

# Wall-clock budget spent spinning the selected callables before any timing, to
# let the CPU reach a steady frequency (see the note in ``main``).
WARMUP_BUDGET_S = 5.0


def _measure_case(case: cases.Case, *, repeats: int, warmup: int) -> dict:
    scales = case.sizes if case.sizes else (0,)  # 0 = sentinel: no size dimension

    # Each measurement is bracketed by its OWN calibration pair: for a case that
    # sweeps sizes, the naive once-per-case bracket would average machine states
    # from the smallest and the largest N, mis-normalizing both.
    timings = {}
    normalizers = {}
    for n in scales:
        calib_before = calibration_ns()
        timings[n] = time_call(case.make(n), repeats=repeats, warmup=warmup)
        calib_after = calibration_ns()
        normalizers[n] = 0.5 * (calib_before + calib_after)

    per_call = timings.get(0)
    calibration = normalizers.get(0)

    scaling: dict[str, list[float]] = {}
    for n, timing in timings.items():
        if n == 0:
            continue
        scaling[str(n)] = [
            timing.median_ns,
            timing.median_ns / n,
            timing.median_ns / normalizers[n],
        ]

    # Representative callable for memory/checksum: smallest size if point-set,
    # otherwise the fixed case.
    representative = case.make(scales[0] if case.sizes else 0)

    checksum_a = output_checksum(representative)
    checksum_b = output_checksum(representative)

    return {
        "name": case.name,
        "module": case.module,
        "function": case.function,
        "category": case.category,
        "input_spec": case.input_spec,
        "static_temporaries": case.static_temporaries,
        "timing": timing_to_dict(per_call) if per_call else None,
        "calibration_ns": calibration,
        "normalized": (per_call.median_ns / calibration) if per_call else None,
        "memory": {"peak_bytes": peak_bytes(representative)},
        "scaling": scaling,
        "checksum": checksum_a,
        "reproducible_bitwise": checksum_a == checksum_b,
    }


def _validation_ratios(records: dict[str, dict]) -> dict[str, dict]:
    """Validation cost relative to the operation it guards.

    Uses the calibration-normalized values, so the ratios survive the machine
    state drift that swung raw absolute times by 2-4x between runs.
    """
    ratios: dict[str, dict] = {}
    for validator, operation, size in cases.VALIDATION_PAIRS:
        v = records.get(validator)
        op = records.get(operation)
        if not v or not op or v.get("normalized") is None:
            continue
        if size is None:
            op_norm = op.get("normalized")
        else:
            entry = op["scaling"].get(str(size))
            op_norm = entry[2] if entry else None
        if op_norm is None:
            continue
        key = f"{validator} / {operation}"
        ratios[key] = {
            "validator_normalized": v["normalized"],
            "operation_normalized": op_norm,
            "size": size,
            "ratio": v["normalized"] / op_norm,
        }
    return ratios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", default=None,
                        help="only run cases whose name contains this substring")
    parser.add_argument("--repeat", type=int, default=1000,
                        help="timed repetitions per case (default 1000)")
    parser.add_argument("--warmup", type=int, default=100,
                        help="warmup calls per case (default 100)")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: stdout summary)")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing but the output path")
    args = parser.parse_args(argv)

    selected = cases.select_cases(args.select)
    if not selected:
        parser.error(f"no case matches {args.select!r}")

    # Global warm-up, TIME-BUDGETED: the first case in a fresh process runs
    # ~2x slower than later ones because the CPU has not ramped its frequency
    # yet. A fixed small call count (milliseconds) does not fix it; spinning
    # the selected callables for a wall-clock budget does, so every case is
    # measured on a ramped CPU and case order stops biasing the ratios.
    warm_fns = [case.make(case.sizes[0] if case.sizes else 0) for case in selected]
    deadline = time.perf_counter() + WARMUP_BUDGET_S
    while time.perf_counter() < deadline:
        for warm_fn in warm_fns:
            warm_fn()

    records: dict[str, dict] = {}
    for case in selected:
        record = _measure_case(case, repeats=args.repeat, warmup=args.warmup)
        records[case.name] = record

    report = {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": env_info(),
        "params": {"repeats": args.repeat, "warmup": args.warmup,
                   "select": args.select},
        "cases": [records[c.name] for c in selected],
        "validation_ratios": _validation_ratios(records),
    }

    if args.out:
        write_json(args.out, report)
        if not args.quiet:
            print(f"wrote {args.out}")
    elif not args.quiet:
        import json
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
