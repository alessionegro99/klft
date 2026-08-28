#!/usr/bin/env python3
"""Gate fixed-mass statistics and finite-volume replacement results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beta_nt_fixed.pilot_summary import finite_volume_comparison


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(pilot: Path, replacements: Path, output: Path) -> dict[str, object]:
    """Write and return the final-check gate record."""
    old_stats = _load(pilot / "Nt_3_Ns_80_b_4p40/analysis.json")
    new_stats = _load(replacements / "Nt_3_Ns_80_b_4p40/analysis.json")
    relative_error = float(new_stats["mass_error"]) / float(new_stats["mass"])
    stats_pass = relative_error <= 0.01

    volumes: dict[str, object] = {}
    for beta in ("4p95", "5p30"):
        old_path = pilot / f"Nt_4_Ns_48_b_{beta}/analysis.json"
        new_path = replacements / f"Nt_4_Ns_96_b_{beta}/analysis.json"
        old = _load(old_path)
        new = _load(new_path)
        comparison = finite_volume_comparison(old_path, old, new_path, new)
        larger_lower = float(new["Ns_over_xi_2nd_68_percent_interval"][0])
        accepted = bool(
            larger_lower >= 10.0
            and comparison["xi_2nd_consistency_pass"]
            and comparison["full_correlator_pass"]
        )
        volumes[beta] = {
            **comparison,
            "larger_68_percent_lower_Ns_over_xi_2nd": larger_lower,
            "larger_volume_replacement_accepted": accepted,
        }

    summary = {
        "statistics_beta_4p40": {
            "old_sweeps_per_chain": 8000,
            "new_sweeps_per_chain": 80000,
            "old_relative_mass_error": float(old_stats["mass_error"])
            / float(old_stats["mass"]),
            "new_relative_mass_error": relative_error,
            "accepted": stats_pass,
        },
        "finite_volume_replacements": volumes,
        "all_checks_accepted": stats_pass
        and all(item["larger_volume_replacement_accepted"] for item in volumes.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--replacement-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.pilot_results, args.replacement_results, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["all_checks_accepted"]:
        raise SystemExit("final statistics or volume gate failed")


if __name__ == "__main__":
    main()
