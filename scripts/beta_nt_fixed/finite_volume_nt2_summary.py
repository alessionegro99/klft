#!/usr/bin/env python3
"""Summarize the fixed-beta/Nt ``Ns=128`` versus ``Ns=160`` check."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from beta_nt_fixed.pilot_summary import (
    finite_volume_comparison,
    weighted_scaling_fit,
)


def load_result(path: Path) -> dict[str, object]:
    """Load one compact ensemble result."""
    return json.loads(path.read_text(encoding="utf-8"))


def revised_scaling(
    pilot_results: Path, nt2_result: dict[str, object]
) -> dict[str, object]:
    """Fit the fixed-beta/Nt line using the checked ``Nt=2`` point."""
    line = [
        nt2_result,
        load_result(
            pilot_results / "Nt_3_Ns_144_b_5p4" / "analysis.json"
        ),
        load_result(
            pilot_results / "Nt_4_Ns_192_b_7p2" / "analysis.json"
        ),
    ]
    if any(result["xi_2nd_error"] is None for result in line):
        raise ValueError("fixed-beta/Nt scaling needs resolved xi_2nd errors")
    nt = np.asarray(
        [result["physical_Nt"] for result in line], dtype=np.float64
    )
    xi = np.asarray([result["xi_2nd"] for result in line], dtype=np.float64)
    error = np.asarray(
        [result["xi_2nd_error"] for result in line], dtype=np.float64
    )
    return {
        "lambda_squared_tree": float(line[0]["lambda_squared_tree"]),
        "Nt2_spatial_volume": int(line[0]["Ns"]),
        "xi_2nd_over_Nt": weighted_scaling_fit(nt, xi / nt, error / nt),
        "interpretation": "preliminary scaling evidence only",
    }


def _comparison_arrays(
    smaller_path: Path, larger_path: Path
) -> dict[str, np.ndarray]:
    """Return the common-distance correlator difference and covariance."""
    smaller = np.load(smaller_path.with_name("covariance.npz"))
    larger = np.load(larger_path.with_name("covariance.npz"))
    smaller_r = smaller["R"].astype(np.int64)
    larger_r = larger["R"].astype(np.int64)
    common_r = np.intersect1d(smaller_r, larger_r)
    common_r = common_r[common_r > 0]
    smaller_indices = np.searchsorted(smaller_r, common_r)
    larger_indices = np.searchsorted(larger_r, common_r)
    smaller_correlator = smaller["C_G"][smaller_indices]
    larger_correlator = larger["C_G"][larger_indices]
    covariance = smaller["C_G_covariance"][
        np.ix_(smaller_indices, smaller_indices)
    ] + larger["C_G_covariance"][
        np.ix_(larger_indices, larger_indices)
    ]
    return {
        "common_R": common_r,
        "C_G_Ns128": smaller_correlator,
        "C_G_Ns160": larger_correlator,
        "C_G_difference_Ns128_minus_Ns160": (
            smaller_correlator - larger_correlator
        ),
        "C_G_difference_covariance": covariance,
    }


def _write_ensemble_table(
    results: Sequence[dict[str, object]], output: Path
) -> None:
    """Write the three beta=3.6 volume points to CSV and NPZ."""
    rows = []
    for result in results:
        xi_interval = result["xi_2nd_68_percent_interval"]
        ratio_interval = result["Ns_over_xi_2nd_68_percent_interval"]
        rows.append(
            {
                "purpose": result["purpose"],
                "beta": result["beta"],
                "physical_Nt": result["physical_Nt"],
                "stored_L2": result["stored_L2"],
                "Ns": result["Ns"],
                "lambda_squared_tree": result["lambda_squared_tree"],
                "xi_2nd": result["xi_2nd"],
                "xi_2nd_error": result["xi_2nd_error"],
                "xi_2nd_68_percent_lower": xi_interval[0],
                "xi_2nd_68_percent_upper": xi_interval[1],
                "Ns_over_xi_2nd": result["Ns_over_xi_2nd"],
                "Ns_over_xi_2nd_error": result["Ns_over_xi_2nd_error"],
                "Ns_over_xi_2nd_68_percent_lower": ratio_interval[0],
                "Ns_over_xi_2nd_68_percent_upper": ratio_interval[1],
                "xi_exp": result["xi_exp"],
                "xi_exp_error": result["xi_exp_error"],
            }
        )
    with (output / "finite_volume_points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output / "finite_volume_points.npz",
        Ns=np.asarray([row["Ns"] for row in rows], dtype=np.int64),
        xi_2nd=np.asarray([row["xi_2nd"] for row in rows], dtype=np.float64),
        xi_2nd_error=np.asarray(
            [row["xi_2nd_error"] for row in rows], dtype=np.float64
        ),
        Ns_over_xi_2nd=np.asarray(
            [row["Ns_over_xi_2nd"] for row in rows], dtype=np.float64
        ),
    )


def summarize(
    pilot_results: Path, extension_results: Path, output: Path
) -> dict[str, object]:
    """Write compact volume-check products and return the decision record."""
    ns96_path = pilot_results / "Nt_2_Ns_96_b_3p6" / "analysis.json"
    ns128_path = pilot_results / "Nt_2_Ns_128_b_3p6" / "analysis.json"
    ns160_path = extension_results / "Nt_2_Ns_160_b_3p6" / "analysis.json"
    ns96 = load_result(ns96_path)
    ns128 = load_result(ns128_path)
    ns160 = load_result(ns160_path)
    comparison = finite_volume_comparison(
        ns128_path, ns128, ns160_path, ns160
    )
    accepted = bool(comparison["smaller_volume_accepted"])
    scaling = revised_scaling(pilot_results, ns128) if accepted else None
    target_size = 16 * math.ceil(10.0 * float(ns128["xi_2nd"]) / 16.0)
    selected_size = max(int(ns128["Ns"]), target_size) if accepted else None
    scaling_pass = bool(
        scaling is not None
        and float(scaling["xi_2nd_over_Nt"]["p_value"]) >= 0.05
    )
    summary = {
        "finite_volume_comparison_Ns128_vs_Ns160": comparison,
        "revised_constant_lambda_continuum_scaling": scaling,
        "decision": {
            "finite_volume_gate_pass": accepted,
            "selected_Nt2_spatial_volume": selected_size,
            "target_rule_size_from_10_xi_2nd": target_size,
            "constant_lambda_Nt6": (
                {"physical_Nt": 6, "beta": 10.8}
                if accepted and scaling_pass
                else None
            ),
            "interpretation": (
                "Ns=128 is stable against Ns=160 under all preregistered "
                "criteria"
                if accepted
                else "finite-volume stability is not established"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_ensemble_table([ns96, ns128, ns160], output)
    np.savez(
        output / "finite_volume_comparison.npz",
        **_comparison_arrays(ns128_path, ns160_path),
    )
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (output / "finite_volume_summary.json").write_text(text, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--extension-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.pilot_results, args.extension_results, args.output
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
