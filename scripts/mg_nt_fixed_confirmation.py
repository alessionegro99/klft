#!/usr/bin/env python3
"""Evaluate the preregistered ``Nt=4,c=1`` confirmation gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate(
    tuning_summary: dict[str, object], confirmation: dict[str, object]
) -> dict[str, object]:
    """Compare the independent confirmation mass with the exact target."""
    tuning = next(
        record
        for record in tuning_summary["groups"]
        if float(record["target_c"]) == 1.0
        and int(record["physical_Nt"]) == 4
    )
    if not bool(tuning["accepted"]):
        raise ValueError("the Nt=4,c=1 pilot tuning was not accepted")
    mass = confirmation.get("mass")
    mass_error = confirmation.get("mass_error")
    if mass is None or mass_error is None:
        measured = None
        measured_error = None
        combined_error = None
        difference = None
        threshold = 0.03
        accepted = False
    else:
        nt = 4
        measured = nt * float(mass)
        measured_error = nt * float(mass_error)
        interval = tuning["beta_star"]["statistical_16_84"]
        beta_sigma = 0.5 * (float(interval[1]) - float(interval[0]))
        slope = abs(float(tuning["fit"]["b"]))
        tuning_prediction_error = slope * beta_sigma
        combined_error = math.hypot(measured_error, tuning_prediction_error)
        difference = abs(measured - 1.0)
        threshold = max(combined_error, 0.03)
        accepted = difference <= threshold
    return {
        "target_c": 1.0,
        "physical_Nt": 4,
        "beta": confirmation.get("beta"),
        "measured_mass_Nt": measured,
        "measured_mass_Nt_error": measured_error,
        "combined_standard_deviation": combined_error,
        "absolute_difference_from_target": difference,
        "three_percent_threshold": 0.03,
        "acceptance_threshold": threshold,
        "criterion": (
            "absolute difference <= max(one combined standard deviation, 3 percent)"
        ),
        "accepted": accepted,
        "next_action": "stop_for_review" if accepted else "reject_tuning_strategy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tuning = json.loads(args.tuning_summary.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8"))
    result = evaluate(tuning, confirmation)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "confirmation_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    status = "PASS" if bool(result["accepted"]) else "FAIL"
    (args.output / "confirmation_gate.tsv").write_text(
        f"gate\t{status}\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
