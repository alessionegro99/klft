#!/usr/bin/env python3
"""Compare analyzed Tr G values with thesis Table 4.4 using combined errors."""

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
    print("Nt measured error thesis error difference_z")
    for nt, path in args.result:
        analysis = json.loads(path.read_text(encoding="utf-8"))
        measured = float(analysis["mean_TrG"])
        measured_error = float(analysis["mean_TrG_error"])
        reference, reference_error = benchmarks["values"][str(nt)]
        combined_error = math.hypot(measured_error, reference_error)
        z_score = (measured - reference) / combined_error
        print(
            f"{nt} {measured:.9g} {measured_error:.3g} "
            f"{reference:.9g} {reference_error:.3g} {z_score:.4g}"
        )


if __name__ == "__main__":
    main()
