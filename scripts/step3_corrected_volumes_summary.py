#!/usr/bin/env python3
"""Summarize the corrected ``Nt=3,4`` constant-coupling volumes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from step3_campaign_summary import (
    finite_volume_comparison,
    weighted_scaling_fit,
)


def load_result(path: Path) -> dict[str, object]:
    """Load one compact ensemble result."""
    return json.loads(path.read_text(encoding="utf-8"))


def scaling_fit(results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Fit ``xi_2nd/Nt = q0 + c/Nt`` for the three slab thicknesses."""
    ordered = sorted(results, key=lambda result: int(result["physical_Nt"]))
    if [int(result["physical_Nt"]) for result in ordered] != [2, 3, 4]:
        raise ValueError("constant-lambda scaling requires Nt=2,3,4")
    if any(result["xi_2nd_error"] is None for result in ordered):
        raise ValueError("constant-lambda scaling needs resolved xi_2nd errors")
    nt = np.asarray(
        [result["physical_Nt"] for result in ordered], dtype=np.float64
    )
    xi = np.asarray([result["xi_2nd"] for result in ordered], dtype=np.float64)
    error = np.asarray(
        [result["xi_2nd_error"] for result in ordered], dtype=np.float64
    )
    return weighted_scaling_fit(nt, xi / nt, error / nt)


def comparison_arrays(
    smaller_path: Path, larger_path: Path
) -> dict[str, np.ndarray]:
    """Return common-distance correlators and their difference covariance."""
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
    ] + larger["C_G_covariance"][np.ix_(larger_indices, larger_indices)]
    return {
        "common_R": common_r,
        "C_G_smaller": smaller_correlator,
        "C_G_larger": larger_correlator,
        "C_G_difference_smaller_minus_larger": (
            smaller_correlator - larger_correlator
        ),
        "C_G_difference_covariance": covariance,
    }


def write_points(
    results: Sequence[tuple[str, dict[str, object]]], output: Path
) -> None:
    """Write portable corrected-volume point tables."""
    rows = []
    for source, result in results:
        xi_interval = result["xi_2nd_68_percent_interval"]
        ratio_interval = result["Ns_over_xi_2nd_68_percent_interval"]
        rows.append(
            {
                "source": source,
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
    with (output / "corrected_volume_points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output / "corrected_volume_points.npz",
        source=np.asarray([row["source"] for row in rows]),
        physical_Nt=np.asarray(
            [row["physical_Nt"] for row in rows], dtype=np.int64
        ),
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
    step3_results: Path,
    fv160_results: Path,
    corrected_results: Path,
    output: Path,
) -> dict[str, object]:
    """Compare pilot and corrected volumes and update the scaling fit."""
    nt2_path = step3_results / "Nt_2_Ns_128_b_3p6" / "analysis.json"
    nt3_old_path = step3_results / "Nt_3_Ns_144_b_5p4" / "analysis.json"
    nt4_old_path = step3_results / "Nt_4_Ns_192_b_7p2" / "analysis.json"
    nt3_new_path = (
        corrected_results / "Nt_3_Ns_176_b_5p4" / "analysis.json"
    )
    nt4_new_path = (
        corrected_results / "Nt_4_Ns_256_b_7p2" / "analysis.json"
    )
    paths = [
        nt2_path,
        nt3_old_path,
        nt3_new_path,
        nt4_old_path,
        nt4_new_path,
    ]
    nt2, nt3_old, nt3_new, nt4_old, nt4_new = [
        load_result(path) for path in paths
    ]
    fv_summary = load_result(fv160_results / "finite_volume_summary.json")
    nt3_comparison = finite_volume_comparison(
        nt3_old_path, nt3_old, nt3_new_path, nt3_new
    )
    nt4_comparison = finite_volume_comparison(
        nt4_old_path, nt4_old, nt4_new_path, nt4_new
    )
    pilot_fit = scaling_fit([nt2, nt3_old, nt4_old])
    corrected_fit = scaling_fit([nt2, nt3_new, nt4_new])
    corrected_ratio_pass = {
        "Nt3": (
            float(nt3_new["Ns_over_xi_2nd_68_percent_interval"][0]) >= 8.0
        ),
        "Nt4": (
            float(nt4_new["Ns_over_xi_2nd_68_percent_interval"][0]) >= 8.0
        ),
    }
    summary = {
        "pilot_vs_corrected_volume": {
            "Nt3_Ns144_vs_Ns176": nt3_comparison,
            "Nt4_Ns192_vs_Ns256": nt4_comparison,
        },
        "pilot_constant_lambda_scaling": {
            "lambda_squared_tree": float(nt2["lambda_squared_tree"]),
            "xi_2nd_over_Nt": pilot_fit,
            "interpretation": "preliminary scaling evidence only",
        },
        "corrected_constant_lambda_scaling": {
            "lambda_squared_tree": float(nt2["lambda_squared_tree"]),
            "xi_2nd_over_Nt": corrected_fit,
            "interpretation": "preliminary scaling evidence only",
        },
        "decision": {
            "Nt2_finite_volume_gate_pass": bool(
                fv_summary["decision"]["finite_volume_gate_pass"]
            ),
            "corrected_68_percent_lower_Ns_over_xi_2nd": {
                "Nt3": float(
                    nt3_new["Ns_over_xi_2nd_68_percent_interval"][0]
                ),
                "Nt4": float(
                    nt4_new["Ns_over_xi_2nd_68_percent_interval"][0]
                ),
            },
            "corrected_volume_ratio_at_least_8": corrected_ratio_pass,
            "all_corrected_volume_ratio_gates_pass": all(
                corrected_ratio_pass.values()
            ),
            "corrected_scaling_fit_p_at_least_0p05": (
                float(corrected_fit["p_value"]) >= 0.05
            ),
            "interpretation": (
                "Use the corrected Nt=3,4 points for subsequent scaling; "
                "the fit remains preliminary."
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_points(
        [
            ("Nt2_volume_checked", nt2),
            ("pilot", nt3_old),
            ("corrected", nt3_new),
            ("pilot", nt4_old),
            ("corrected", nt4_new),
        ],
        output,
    )
    np.savez(
        output / "correlator_comparison_Nt3.npz",
        **comparison_arrays(nt3_old_path, nt3_new_path),
    )
    np.savez(
        output / "correlator_comparison_Nt4.npz",
        **comparison_arrays(nt4_old_path, nt4_new_path),
    )
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (output / "corrected_volume_summary.json").write_text(
        text, encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3-results", type=Path, required=True)
    parser.add_argument("--fv160-results", type=Path, required=True)
    parser.add_argument("--corrected-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        args.step3_results,
        args.fv160_results,
        args.corrected_results,
        args.output,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
