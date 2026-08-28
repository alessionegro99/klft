#!/usr/bin/env python3
"""Audit the new mass analysis against accepted fixed-beta/Nt results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def audit(
    prior_points: Path, project_root: Path, audit_results: Path
) -> dict[str, object]:
    """Return the two-point mass-reproduction audit."""
    with prior_points.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    comparisons: list[dict[str, object]] = []
    for row in rows:
        previous_path = project_root / row["source_analysis"] / "analysis.json"
        current_path = audit_results / row["ensemble"] / "analysis.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        current = json.loads(current_path.read_text(encoding="utf-8"))
        previous_mass = previous.get("mass")
        previous_error = previous.get("mass_error")
        current_mass = current.get("mass")
        current_error = current.get("mass_error")
        accepted = all(
            value is not None
            for value in (previous_mass, previous_error, current_mass, current_error)
        )
        combined_error = (
            math.hypot(float(previous_error), float(current_error))
            if accepted
            else None
        )
        z = (
            abs(float(current_mass) - float(previous_mass)) / combined_error
            if accepted and combined_error is not None and combined_error > 0.0
            else None
        )
        passed = bool(accepted and z is not None and z <= 2.0)
        comparisons.append(
            {
                "target_c": float(row["target_c"]),
                "physical_Nt": int(row["physical_Nt"]),
                "Ns": int(row["Ns"]),
                "beta": float(row["beta"]),
                "ensemble": row["ensemble"],
                "source_campaign": row["source_campaign"],
                "previous_result": str(previous_path),
                "audit_result": str(current_path),
                "previous_mass": previous_mass,
                "previous_mass_error": previous_error,
                "audit_mass": current_mass,
                "audit_mass_error": current_error,
                "difference_combined_sigma": z,
                "pass": passed,
            }
        )
    return {
        "criterion": (
            "new correlated-window mass is accepted and differs from the "
            "previously accepted mass by at most two combined standard deviations"
        ),
        "comparisons": comparisons,
        "all_pass": all(bool(comparison["pass"]) for comparison in comparisons),
    }


def write_outputs(result: dict[str, object], output: Path) -> None:
    """Write strict JSON, CSV, and the production audit gate."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "audit_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        columns = (
            "target_c",
            "physical_Nt",
            "Ns",
            "beta",
            "ensemble",
            "source_campaign",
            "previous_mass",
            "previous_mass_error",
            "audit_mass",
            "audit_mass_error",
            "difference_combined_sigma",
            "pass",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for comparison in result["comparisons"]:
            writer.writerow({key: comparison[key] for key in columns})
    status = "PASS" if bool(result["all_pass"]) else "FAIL"
    (output / "audit_gate.tsv").write_text(
        f"gate\t{status}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-points", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prior_points, args.project_root, args.audit_results)
    write_outputs(result, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
