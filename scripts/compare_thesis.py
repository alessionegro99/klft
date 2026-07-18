#!/usr/bin/env python3
"""Compare analyzed (1/2) Tr G values with thesis Table 4.4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_result(specification: str) -> tuple[int, Path]:
    """Parse NT=analysis.json."""
    nt_text, separator, path_text = specification.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("result must have form NT=analysis.json")
    return int(nt_text), Path(path_text)


def normalized_trace(analysis: dict[str, object]) -> tuple[float, float]:
    """Return the SU(2) trace normalized to one at the identity."""
    if "mean_half_TrG" in analysis:
        return (
            float(analysis["mean_half_TrG"]),
            float(analysis["mean_half_TrG_error"]),
        )
    return (
        0.5 * float(analysis["mean_TrG"]),
        0.5 * float(analysis["mean_TrG_error"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=parse_result, nargs="+")
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "dirichlet_thesis"
            / "thesis_benchmarks.json"
        ),
    )
    args = parser.parse_args()
    benchmarks = json.loads(args.benchmarks.read_text(encoding="utf-8"))
    print("Nt measured_half_TrG error thesis error difference_z")
    for nt, path in args.result:
        analysis = json.loads(path.read_text(encoding="utf-8"))
        measured, measured_error = normalized_trace(analysis)
        reference, reference_error = benchmarks["values"][str(nt)]
        combined_error = math.hypot(measured_error, reference_error)
        z_score = (measured - reference) / combined_error
        print(
            f"{nt} {measured:.9g} {measured_error:.3g} "
            f"{reference:.9g} {reference_error:.3g} {z_score:.4g}"
        )


if __name__ == "__main__":
    main()
