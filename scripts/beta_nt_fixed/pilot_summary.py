#!/usr/bin/env python3
"""Aggregate compact fixed-beta/Nt pilot results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats


FloatArray = NDArray[np.float64]


def load_results(root: Path) -> list[tuple[Path, dict[str, object]]]:
    """Load the eight required per-ensemble compact result files."""
    results = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in root.glob("Nt_*_Ns_*_b_*/analysis.json")
    ]
    results.sort(
        key=lambda item: (
            int(item[1]["physical_Nt"]),
            float(item[1]["beta"]),
            int(item[1]["Ns"]),
        )
    )
    if len(results) != 8:
        raise ValueError(f"{root}: expected eight ensemble analyses, found {len(results)}")
    purposes = {str(result["purpose"]) for _, result in results}
    if len(purposes) != 8:
        raise ValueError("pilot result purposes are not unique")
    return results


def generalized_chi_square(
    difference: FloatArray, covariance: FloatArray
) -> dict[str, float | int]:
    """Return chi-square in the numerically resolved covariance subspace."""
    eigenvalues, eigenvectors = linalg.eigh(covariance)
    if len(eigenvalues) == 0 or eigenvalues[-1] <= 0.0:
        raise ValueError("comparison covariance has no positive modes")
    tolerance = (
        np.finfo(np.float64).eps * len(eigenvalues) * float(eigenvalues[-1])
    )
    retained = eigenvalues > tolerance
    rank = int(np.count_nonzero(retained))
    if rank == 0:
        raise ValueError("comparison covariance is numerically singular")
    projected = eigenvectors[:, retained].T @ difference
    chi2 = float(np.sum(projected * projected / eigenvalues[retained]))
    return {
        "chi2": chi2,
        "dof": rank,
        "covariance_rank": rank,
        "p_value": float(stats.chi2.sf(chi2, rank)),
    }


def finite_volume_comparison(
    smaller_path: Path,
    smaller: dict[str, object],
    larger_path: Path,
    larger: dict[str, object],
) -> dict[str, object]:
    """Apply the preregistered criteria to two increasing spatial volumes."""
    if int(smaller["physical_Nt"]) != int(larger["physical_Nt"]):
        raise ValueError("finite-volume ensembles have different Nt")
    if not np.isclose(float(smaller["beta"]), float(larger["beta"])):
        raise ValueError("finite-volume ensembles have different beta")
    if int(smaller["Ns"]) >= int(larger["Ns"]):
        raise ValueError("finite-volume comparison requires increasing Ns")
    if smaller["xi_2nd_error"] is None or larger["xi_2nd_error"] is None:
        raise ValueError("finite-volume comparison needs resolved xi_2nd errors")

    smaller_npz = np.load(smaller_path.with_name("covariance.npz"))
    larger_npz = np.load(larger_path.with_name("covariance.npz"))
    smaller_r = smaller_npz["R"].astype(np.int64)
    larger_r = larger_npz["R"].astype(np.int64)
    common_r = np.intersect1d(smaller_r, larger_r)
    common_r = common_r[common_r > 0]
    smaller_indices = np.searchsorted(smaller_r, common_r)
    larger_indices = np.searchsorted(larger_r, common_r)
    difference = (
        smaller_npz["C_G"][smaller_indices]
        - larger_npz["C_G"][larger_indices]
    )
    covariance = smaller_npz["C_G_covariance"][
        np.ix_(smaller_indices, smaller_indices)
    ] + larger_npz["C_G_covariance"][
        np.ix_(larger_indices, larger_indices)
    ]
    correlator_test = generalized_chi_square(difference, covariance)

    xi_smaller = float(smaller["xi_2nd"])
    xi_larger = float(larger["xi_2nd"])
    error_smaller = float(smaller["xi_2nd_error"])
    error_larger = float(larger["xi_2nd_error"])
    difference_z = abs(xi_smaller - xi_larger) / math.hypot(
        error_smaller, error_larger
    )
    relative_difference = abs(xi_smaller - xi_larger) / xi_larger
    volume_lower = float(smaller["Ns_over_xi_2nd_68_percent_interval"][0])
    volume_pass = volume_lower >= 8.0
    xi_pass = difference_z < 2.0 and relative_difference < 0.05
    correlator_pass = float(correlator_test["p_value"]) >= 0.05
    accepted = volume_pass and xi_pass and correlator_pass
    return {
        "physical_Nt": int(smaller["physical_Nt"]),
        "beta": float(smaller["beta"]),
        "smaller_Ns": int(smaller["Ns"]),
        "larger_Ns": int(larger["Ns"]),
        "smaller_68_percent_lower_Ns_over_xi_2nd": volume_lower,
        "volume_ratio_pass": volume_pass,
        "xi_2nd_difference_z": difference_z,
        "xi_2nd_relative_difference": relative_difference,
        "xi_2nd_consistency_pass": xi_pass,
        "common_positive_R": common_r.tolist(),
        "full_correlator_test": correlator_test,
        "full_correlator_pass": correlator_pass,
        "smaller_volume_accepted": accepted,
    }


def finite_volume_decision(
    anchor_path: Path,
    anchor: dict[str, object],
    check_path: Path,
    check: dict[str, object],
) -> dict[str, object]:
    """Apply all preregistered ``Ns=96`` anchor acceptance criteria."""
    comparison = finite_volume_comparison(
        anchor_path, anchor, check_path, check
    )
    accepted = bool(comparison["smaller_volume_accepted"])
    return {
        "anchor_Ns": comparison["smaller_Ns"],
        "check_Ns": comparison["larger_Ns"],
        "anchor_68_percent_lower_Ns_over_xi_2nd": comparison[
            "smaller_68_percent_lower_Ns_over_xi_2nd"
        ],
        "volume_ratio_pass": comparison["volume_ratio_pass"],
        "xi_2nd_difference_z": comparison["xi_2nd_difference_z"],
        "xi_2nd_relative_difference": comparison[
            "xi_2nd_relative_difference"
        ],
        "xi_2nd_consistency_pass": comparison["xi_2nd_consistency_pass"],
        "common_positive_R": comparison["common_positive_R"],
        "full_correlator_test": comparison["full_correlator_test"],
        "full_correlator_pass": comparison["full_correlator_pass"],
        "anchor_accepted": accepted,
        "required_additional_ensemble": (
            None
            if accepted
            else {"physical_Nt": 2, "beta": 3.6, "Ns": 160, "chains": 4}
        ),
    }


def weighted_scaling_fit(
    nt: FloatArray, quantity: FloatArray, error: FloatArray
) -> dict[str, object]:
    """Fit ``q(Nt)=q0+c/Nt`` with independent Gaussian errors."""
    if len(nt) != 3 or np.any(error <= 0.0):
        raise ValueError("scaling fit needs three positive uncertainties")
    design = np.column_stack((np.ones_like(nt), 1.0 / nt))
    weights = np.diag(1.0 / (error * error))
    covariance = linalg.inv(design.T @ weights @ design)
    parameters = covariance @ design.T @ weights @ quantity
    residual = quantity - design @ parameters
    chi2 = float(residual @ weights @ residual)
    dof = len(nt) - len(parameters)
    return {
        "model": "q(Nt) = q0 + c/Nt",
        "Nt": nt.astype(int).tolist(),
        "q": quantity.tolist(),
        "q_error": error.tolist(),
        "q0": float(parameters[0]),
        "q0_error": float(math.sqrt(covariance[0, 0])),
        "c": float(parameters[1]),
        "c_error": float(math.sqrt(covariance[1, 1])),
        "chi2": chi2,
        "dof": dof,
        "p_value": float(stats.chi2.sf(chi2, dof)),
        "interpretation": "preliminary scaling evidence only",
    }


def continuum_scaling(
    results: Sequence[tuple[Path, dict[str, object]]]
) -> dict[str, object]:
    """Fit the three constant-tree-coupling points in slab units."""
    line = [
        result
        for _, result in results
        if str(result["purpose"]).startswith("constant_lambda_")
        and result["purpose"] != "anchor_finite_volume"
    ]
    line.sort(key=lambda result: int(result["physical_Nt"]))
    if len(line) != 3:
        raise ValueError("fixed-beta/Nt line must contain Nt=2,3,4")
    nt = np.asarray([result["physical_Nt"] for result in line], dtype=np.float64)
    xi = np.asarray([result["xi_2nd"] for result in line], dtype=np.float64)
    xi_error = np.asarray(
        [result["xi_2nd_error"] for result in line], dtype=np.float64
    )
    output: dict[str, object] = {
        "lambda_squared_tree": float(line[0]["lambda_squared_tree"]),
        "xi_2nd_over_Nt": weighted_scaling_fit(nt, xi / nt, xi_error / nt),
    }
    if all(result["xi_exp"] is not None for result in line):
        xi_exp = np.asarray([result["xi_exp"] for result in line], dtype=np.float64)
        xi_exp_error = np.asarray(
            [result["xi_exp_error"] for result in line], dtype=np.float64
        )
        output["xi_exp_over_Nt"] = weighted_scaling_fit(
            nt, xi_exp / nt, xi_exp_error / nt
        )
        output["m_Nt"] = (nt / xi_exp).tolist()
    else:
        output["xi_exp_over_Nt"] = None
        output["m_Nt"] = None
    return output


def production_recommendations(
    results: Sequence[tuple[Path, dict[str, object]]],
    finite_volume: dict[str, object],
    scaling: dict[str, object],
) -> dict[str, object]:
    """Translate measured lengths into the preregistered next-grid rules."""
    sizes = []
    for _, result in results:
        if result["purpose"] == "anchor_finite_volume":
            continue
        target = 16 * math.ceil(10.0 * float(result["xi_2nd"]) / 16.0)
        sizes.append(
            {
                "physical_Nt": int(result["physical_Nt"]),
                "beta": float(result["beta"]),
                "pilot_Ns": int(result["Ns"]),
                "recommended_production_Ns": target,
            }
        )

    nt2 = [
        result
        for _, result in results
        if int(result["physical_Nt"]) == 2
        and result["purpose"] != "anchor_finite_volume"
    ]
    nt2.sort(key=lambda result: float(result["beta"]))
    beta = np.asarray([result["beta"] for result in nt2], dtype=np.float64)
    xi = np.asarray([result["xi_2nd"] for result in nt2], dtype=np.float64)
    bracketed = bool(np.min(xi) <= 5.0 <= np.max(xi))
    extra_coupling: dict[str, object] | None = None
    if not bracketed:
        slope, intercept = np.polyfit(beta, np.log(xi), 1)
        estimated_beta = float((math.log(5.0) - intercept) / slope)
        extra_coupling = {
            "physical_Nt": 2,
            "estimated_beta_for_xi_2nd_approximately_5": estimated_beta,
            "estimation": (
                "interpolation"
                if float(beta[0]) <= estimated_beta <= float(beta[-1])
                else "extrapolation requiring review"
            ),
        }

    scaling_ok = (
        float(scaling["xi_2nd_over_Nt"]["p_value"]) >= 0.05
    )
    add_nt6 = bool(finite_volume["anchor_accepted"] and scaling_ok)
    return {
        "target_rule": "round 10 xi_2nd upward to a multiple of 16",
        "recommended_spatial_sizes": sizes,
        "Nt2_xi_approximately_5_bracketed": bracketed,
        "additional_Nt2_coupling": extra_coupling,
        "constant_lambda_Nt6": (
            {"physical_Nt": 6, "beta": 10.8}
            if add_nt6
            else None
        ),
    }


def write_tables(
    results: Sequence[tuple[Path, dict[str, object]]], output: Path
) -> None:
    """Write portable CSV and NPZ campaign tables."""
    rows = []
    for _, result in results:
        diagnostics = result["thermalization_and_blocking"]
        chains = diagnostics["chains"]
        plateau = diagnostics["blocking_error_plateau"]
        fit_plateau = result["xi_exp_blocking_error_plateau"]
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
                "Ns_over_xi_2nd": result["Ns_over_xi_2nd"],
                "Ns_over_xi_2nd_error": result["Ns_over_xi_2nd_error"],
                "xi_2nd_bootstrap_valid_fraction": result[
                    "xi_2nd_bootstrap_valid_fraction"
                ],
                "xi_exp": result["xi_exp"],
                "xi_exp_error": result["xi_exp_error"],
                "xi_exp_mass_blocking_plateau_pass": (
                    None
                    if fit_plateau is None
                    else fit_plateau[
                        "selected_block_on_xi_exp_and_mass_plateau"
                    ]
                ),
                "blocking_plateau_all_found": plateau[
                    "all_primary_and_secondary_plateaus_found"
                ],
                "blocking_common_length_sweeps": plateau[
                    "selected_common_block_length_sweeps"
                ],
                "blocking_tau_minimum_sweeps": plateau[
                    "minimum_tau_block_length_sweeps"
                ],
                "thermalization_cuts_sweeps": ";".join(
                    str(chain["thermalization_cut_sweeps"]) for chain in chains
                ),
                "block_lengths_sweeps": ";".join(
                    str(chain["block_length_sweeps"]) for chain in chains
                ),
                "complete_blocks": ";".join(
                    str(chain["n_complete_blocks"]) for chain in chains
                ),
            }
        )
    with (output / "campaign_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output / "campaign_summary.npz",
        beta=np.asarray([row["beta"] for row in rows], dtype=np.float64),
        physical_Nt=np.asarray(
            [row["physical_Nt"] for row in rows], dtype=np.int64
        ),
        Ns=np.asarray([row["Ns"] for row in rows], dtype=np.int64),
        lambda_squared_tree=np.asarray(
            [row["lambda_squared_tree"] for row in rows], dtype=np.float64
        ),
        xi_2nd=np.asarray([row["xi_2nd"] for row in rows], dtype=np.float64),
        xi_2nd_error=np.asarray(
            [
                np.nan
                if row["xi_2nd_error"] is None
                else row["xi_2nd_error"]
                for row in rows
            ],
            dtype=np.float64,
        ),
        xi_2nd_bootstrap_valid_fraction=np.asarray(
            [row["xi_2nd_bootstrap_valid_fraction"] for row in rows],
            dtype=np.float64,
        ),
        blocking_common_length_sweeps=np.asarray(
            [row["blocking_common_length_sweeps"] for row in rows],
            dtype=np.int64,
        ),
        xi_exp=np.asarray(
            [np.nan if row["xi_exp"] is None else row["xi_exp"] for row in rows],
            dtype=np.float64,
        ),
    )


def aggregate(root: Path, output: Path) -> dict[str, object]:
    """Aggregate all ensemble results and return the campaign decision record."""
    results = load_results(root)
    by_purpose = {str(result["purpose"]): (path, result) for path, result in results}
    finite_volume = finite_volume_decision(
        *by_purpose["constant_lambda_anchor"],
        *by_purpose["anchor_finite_volume"],
    )
    scaling = continuum_scaling(results)
    recommendations = production_recommendations(results, finite_volume, scaling)
    output.mkdir(parents=True, exist_ok=True)
    write_tables(results, output)
    return {
        "finite_volume_anchor": finite_volume,
        "constant_lambda_continuum_scaling": scaling,
        "production_recommendations": recommendations,
        "interpretation": "pilot results provide preliminary scaling evidence only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.results, args.output)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.output / "campaign_summary.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
