#!/usr/bin/env python3
"""Tune beta on fixed-``m_G N_t`` trajectories from accepted mass samples.

The local model is the preregistered weighted fit
``log(m_G N_t) = a + b beta``. Bootstrap propagation follows B. Efron and
R. Tibshirani, *An Introduction to the Bootstrap* (Chapman & Hall, 1993):
each replica draws once from every independently generated ensemble mass
distribution, refits ``a,b``, and solves for the target coupling.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats


FloatArray = NDArray[np.float64]
BOOTSTRAP_SAMPLES = 1_000
TUNING_SEED = 20_260_724
MINIMUM_BLOCKS = 30


@dataclass(frozen=True)
class Point:
    """One accepted or rejected pilot mass measurement."""

    target_c: float
    nt: int
    ns: int
    beta: float
    ensemble: str
    source_campaign: str
    result_path: Path
    result: dict[str, object]
    mass_bootstrap: FloatArray


def _load_json(path: Path) -> dict[str, object]:
    """Load one strict JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_mass_samples(path: Path) -> FloatArray:
    """Load finite accepted mass replicas from an analysis archive."""
    with np.load(path) as archive:
        if "mass_bootstrap" not in archive:
            return np.empty(0, dtype=np.float64)
        samples = np.asarray(archive["mass_bootstrap"], dtype=np.float64)
    return samples[np.isfinite(samples) & (samples > 0.0)]


def load_points(
    manifest: Path,
    prior_points: Path,
    results: Path,
    audit_results: Path,
    extension_manifest: Path | None = None,
    extension_results: Path | None = None,
) -> list[Point]:
    """Load pilot, optional extension, and explicitly labelled prior points."""
    points = _load_manifest_points(manifest, results, "mg_nt_fixed")
    if (extension_manifest is None) != (extension_results is None):
        raise ValueError("extension manifest and results must be supplied together")
    if extension_manifest is not None and extension_results is not None:
        points.extend(
            _load_manifest_points(
                extension_manifest,
                extension_results,
                "mg_nt_fixed_extension_round1",
            )
        )

    with prior_points.open(encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in prior_rows:
        ensemble = row["ensemble"]
        result_path = audit_results / ensemble / "analysis.json"
        result = _load_json(result_path)
        points.append(
            Point(
                float(row["target_c"]),
                int(row["physical_Nt"]),
                int(row["Ns"]),
                float(row["beta"]),
                ensemble,
                row["source_campaign"],
                result_path,
                result,
                _load_mass_samples(result_path.with_name("covariance.npz")),
            )
        )
    return points


def _load_manifest_points(
    manifest: Path, results: Path, source_campaign: str
) -> list[Point]:
    """Load one four-chain-per-point manifest and its compact masses."""
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        ensemble = Path(row["run"]).parts[-2]
        grouped.setdefault((row["target_c"], ensemble), []).append(row)
    points: list[Point] = []
    for (target_c, ensemble), group in grouped.items():
        if len(group) != 4:
            raise ValueError(f"{ensemble}: expected four manifest chains")
        first = group[0]
        result_path = results / ensemble / "analysis.json"
        result = _load_json(result_path)
        points.append(
            Point(
                float(target_c),
                int(first["physical_Nt"]),
                int(first["Ns"]),
                float(first["beta"]),
                ensemble,
                source_campaign,
                result_path,
                result,
                _load_mass_samples(result_path.with_name("covariance.npz")),
            )
        )

    return points


def weighted_log_fit(
    beta: FloatArray, mass_nt: FloatArray, mass_nt_error: FloatArray
) -> dict[str, float | FloatArray]:
    """Return the two-parameter weighted log-linear fit."""
    if len(beta) != 3 or np.any(mass_nt <= 0.0) or np.any(mass_nt_error <= 0.0):
        raise ValueError("the tuning fit needs three positive measurements")
    y = np.log(mass_nt)
    sigma = mass_nt_error / mass_nt
    design = np.column_stack((np.ones_like(beta), beta))
    weighted_design = design / sigma[:, None]
    weighted_y = y / sigma
    parameters, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    residual = (y - design @ parameters) / sigma
    chi2 = float(residual @ residual)
    dof = len(beta) - 2
    covariance = np.linalg.inv(weighted_design.T @ weighted_design)
    return {
        "a": float(parameters[0]),
        "b": float(parameters[1]),
        "chi2": chi2,
        "dof": float(dof),
        "p_value": float(stats.chi2.sf(chi2, dof)),
        "parameter_covariance": covariance,
        "log_sigma": sigma,
    }


def _complete_blocks(result: dict[str, object]) -> list[int]:
    """Return final complete-block counts for all four replicas."""
    thermalization = result.get("thermalization_and_blocking")
    if not isinstance(thermalization, dict):
        return []
    chains = thermalization.get("chains")
    if not isinstance(chains, list):
        return []
    return [int(chain["n_complete_blocks"]) for chain in chains]


def _two_point_interpolations(
    beta: FloatArray, log_mass_nt: FloatArray, log_target: float
) -> list[dict[str, object]]:
    """Return every target-bracketing two-point log interpolation."""
    results: list[dict[str, object]] = []
    for left in range(len(beta) - 1):
        for right in range(left + 1, len(beta)):
            y_left = float(log_mass_nt[left])
            y_right = float(log_mass_nt[right])
            if (y_left - log_target) * (y_right - log_target) > 0.0:
                continue
            if y_right == y_left:
                continue
            value = float(
                beta[left]
                + (log_target - y_left)
                * (beta[right] - beta[left])
                / (y_right - y_left)
            )
            results.append(
                {
                    "indices": [left, right],
                    "beta_values": [float(beta[left]), float(beta[right])],
                    "beta_star": value,
                }
            )
    return results


def _select_local_triple(points: Sequence[Point]) -> list[Point]:
    """Select the closest contiguous three-point central-value bracket."""
    ordered = sorted(points, key=lambda point: point.beta)
    if len(ordered) < 3:
        raise ValueError("a tuning group needs at least three beta points")
    if len(ordered) == 3:
        return ordered
    target = ordered[0].target_c
    candidates: list[tuple[tuple[float, float, float], list[Point]]] = []
    for first in range(len(ordered) - 2):
        triple = ordered[first : first + 3]
        mass_nt = np.asarray(
            [
                math.nan
                if point.result.get("mass") is None
                else point.nt * float(point.result["mass"])
                for point in triple
            ],
            dtype=np.float64,
        )
        finite_positive = bool(
            np.all(np.isfinite(mass_nt)) and np.all(mass_nt > 0.0)
        )
        bracketed = bool(
            finite_positive and np.min(mass_nt) <= target <= np.max(mass_nt)
        )
        distance = (
            float(np.max(np.abs(np.log(mass_nt / target))))
            if finite_positive
            else math.inf
        )
        score = (
            0.0 if bracketed else 1.0,
            distance,
            triple[-1].beta - triple[0].beta,
        )
        candidates.append((score, triple))
    return min(candidates, key=lambda item: item[0])[1]


def tune_group(
    points: Sequence[Point], bootstrap_seed: int
) -> tuple[dict[str, object], FloatArray]:
    """Apply all preregistered gates and tune one ``(c,Nt)`` group."""
    available = sorted(points, key=lambda point: point.beta)
    ordered = _select_local_triple(available)
    target_c = ordered[0].target_c
    nt = ordered[0].nt
    if any(point.target_c != target_c or point.nt != nt for point in ordered):
        raise ValueError("mixed tuning group")

    reasons: list[str] = []
    masses: list[float] = []
    mass_errors: list[float] = []
    point_records: list[dict[str, object]] = []
    for point in ordered:
        mass = point.result.get("mass")
        mass_error = point.result.get("mass_error")
        blocks = _complete_blocks(point.result)
        accepted = mass is not None and mass_error is not None
        if not accepted:
            reasons.append(f"{point.ensemble}: no accepted asymptotic mass")
        if len(blocks) != 4 or min(blocks, default=0) < MINIMUM_BLOCKS:
            reasons.append(f"{point.ensemble}: fewer than 30 complete blocks")
        if len(point.mass_bootstrap) < 2:
            reasons.append(f"{point.ensemble}: no saved mass bootstrap samples")
        masses.append(float(mass) if mass is not None else math.nan)
        mass_errors.append(float(mass_error) if mass_error is not None else math.nan)
        point_records.append(
            {
                "beta": point.beta,
                "Ns": point.ns,
                "ensemble": point.ensemble,
                "source_campaign": point.source_campaign,
                "result": str(point.result_path),
                "mass": mass,
                "mass_error": mass_error,
                "mass_Nt": None if mass is None else nt * float(mass),
                "mass_Nt_error": (
                    None if mass_error is None else nt * float(mass_error)
                ),
                "mass_bootstrap_samples": len(point.mass_bootstrap),
                "complete_blocks_per_chain": blocks,
            }
        )

    selection_record = {
        "rule": (
            "closest contiguous three-point central-value bracket; minimize "
            "maximum absolute log-distance from target"
        ),
        "available_beta_values": [point.beta for point in available],
        "selected_beta_values": [point.beta for point in ordered],
        "selected_ensembles": [point.ensemble for point in ordered],
    }

    beta = np.asarray([point.beta for point in ordered], dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    mass_error = np.asarray(mass_errors, dtype=np.float64)
    empty = np.empty(0, dtype=np.float64)
    if reasons:
        return (
            {
                "target_c": target_c,
                "physical_Nt": nt,
                "points": point_records,
                "local_point_selection": selection_record,
                "accepted": False,
                "add_bracket_point": True,
                "failure_reasons": sorted(set(reasons)),
                "beta_star": None,
            },
            empty,
        )

    mass_nt = nt * mass
    mass_nt_error = nt * mass_error
    inversion = np.diff(mass_nt)
    inversion_tolerance = 2.0 * np.hypot(mass_nt_error[:-1], mass_nt_error[1:])
    monotonic = bool(np.all(inversion <= inversion_tolerance))
    bracketed = bool(np.min(mass_nt) <= target_c <= np.max(mass_nt))
    if not monotonic:
        reasons.append("masses are nonmonotonic beyond two combined sigma")
    if not bracketed:
        reasons.append("target is not bracketed")

    fit = weighted_log_fit(beta, mass_nt, mass_nt_error)
    slope = float(fit["b"])
    if slope >= 0.0:
        reasons.append("log-linear slope is not negative")
    if float(fit["p_value"]) < 0.05:
        reasons.append("log-linear interpolation has p < 0.05")
    if reasons:
        return (
            {
                "target_c": target_c,
                "physical_Nt": nt,
                "points": point_records,
                "local_point_selection": selection_record,
                "mass_monotonic_within_two_sigma": monotonic,
                "target_bracketed": bracketed,
                "fit": {
                    key: value
                    for key, value in fit.items()
                    if key not in {"parameter_covariance", "log_sigma"}
                },
                "accepted": False,
                "add_bracket_point": True,
                "failure_reasons": sorted(set(reasons)),
                "beta_star": None,
            },
            empty,
        )

    central_beta_star = (math.log(target_c) - float(fit["a"])) / slope
    rng = np.random.default_rng(bootstrap_seed)
    sampled_mass = np.column_stack(
        [rng.choice(point.mass_bootstrap, BOOTSTRAP_SAMPLES) for point in ordered]
    )
    sampled_log_mass_nt = np.log(nt * sampled_mass)
    sigma = np.asarray(fit["log_sigma"], dtype=np.float64)
    design = np.column_stack((np.ones_like(beta), beta))
    weight = np.diag(1.0 / (sigma * sigma))
    projector = np.linalg.inv(design.T @ weight @ design) @ design.T @ weight
    parameters = sampled_log_mass_nt @ projector.T
    valid = np.isfinite(parameters).all(axis=1) & (parameters[:, 1] < 0.0)
    beta_star_samples = (
        math.log(target_c) - parameters[valid, 0]
    ) / parameters[valid, 1]
    valid_fraction = len(beta_star_samples) / BOOTSTRAP_SAMPLES
    if valid_fraction < 0.90:
        reasons.append("fewer than 90 percent valid tuning bootstrap refits")
    if len(beta_star_samples) == 0:
        return (
            {
                "target_c": target_c,
                "physical_Nt": nt,
                "points": point_records,
                "local_point_selection": selection_record,
                "mass_monotonic_within_two_sigma": monotonic,
                "target_bracketed": bracketed,
                "fit": {
                    key: value
                    for key, value in fit.items()
                    if key not in {"parameter_covariance", "log_sigma"}
                },
                "accepted": False,
                "add_bracket_point": True,
                "failure_reasons": sorted(set(reasons)),
                "beta_star": None,
            },
            empty,
        )

    interval = np.quantile(beta_star_samples, [0.16, 0.5, 0.84])
    pair_interpolations = _two_point_interpolations(
        beta, np.log(mass_nt), math.log(target_c)
    )
    shifts = [
        abs(float(entry["beta_star"]) - central_beta_star)
        for entry in pair_interpolations
    ]
    systematic = max(shifts) if shifts else math.nan
    record = {
        "target_c": target_c,
        "physical_Nt": nt,
        "points": point_records,
        "local_point_selection": selection_record,
        "mass_monotonic_within_two_sigma": monotonic,
        "target_bracketed": bracketed,
        "fit": {
            "model": "log(m_G*N_t) = a + b*beta",
            "a": float(fit["a"]),
            "b": slope,
            "chi2": float(fit["chi2"]),
            "dof": int(fit["dof"]),
            "p_value": float(fit["p_value"]),
            "parameter_covariance": np.asarray(
                fit["parameter_covariance"], dtype=np.float64
            ).tolist(),
        },
        "accepted": not reasons,
        "add_bracket_point": bool(reasons),
        "failure_reasons": sorted(set(reasons)),
        "beta_star": {
            "central_three_point": central_beta_star,
            "median": float(interval[1]),
            "statistical_16_84": [float(interval[0]), float(interval[2])],
            "bootstrap_samples_requested": BOOTSTRAP_SAMPLES,
            "bootstrap_samples_valid": len(beta_star_samples),
            "bootstrap_valid_fraction": valid_fraction,
            "bootstrap_seed": bootstrap_seed,
            "two_point_target_bracketing_interpolations": pair_interpolations,
            "interpolation_systematic_max_abs_shift": systematic,
        },
    }
    return record, beta_star_samples


def tune_campaign(points: Sequence[Point]) -> tuple[dict[str, object], dict[str, FloatArray]]:
    """Tune every target and thickness and return strict compact products."""
    groups: dict[tuple[float, int], list[Point]] = {}
    for point in points:
        groups.setdefault((point.target_c, point.nt), []).append(point)
    expected = {(target, nt) for target in (0.5, 1.0, 2.0) for nt in (2, 3, 4)}
    if set(groups) != expected:
        raise ValueError(f"unexpected tuning groups: {sorted(groups)}")

    records: list[dict[str, object]] = []
    samples: dict[str, FloatArray] = {}
    for index, key in enumerate(sorted(groups)):
        record, values = tune_group(groups[key], TUNING_SEED + index)
        records.append(record)
        tag = f"c_{key[0]:g}_Nt_{key[1]}".replace(".", "p")
        samples[tag] = values
    accepted = all(bool(record["accepted"]) for record in records)
    confirmation = next(
        record for record in records
        if float(record["target_c"]) == 1.0 and int(record["physical_Nt"]) == 4
    )
    return (
        {
            "campaign_type": "nonperturbative_mg_nt",
            "trajectory_definition": "c = m_G N_t fixed nonperturbatively",
            "targets": [0.5, 1.0, 2.0],
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "groups": records,
            "all_nine_tunings_accepted": accepted,
            "confirmation_Nt4_c1_beta": (
                confirmation["beta_star"]["median"]
                if bool(confirmation["accepted"])
                else None
            ),
            "next_action": (
                "run_Nt4_c1_confirmation"
                if accepted
                else "add_bracket_points_for_failed_groups"
            ),
        },
        samples,
    )


def write_outputs(
    summary: dict[str, object], samples: dict[str, FloatArray], output: Path
) -> None:
    """Write JSON, CSV, and bootstrap NPZ compact products."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "tuning_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        columns = (
            "target_c",
            "physical_Nt",
            "accepted",
            "beta_star_median",
            "beta_star_16",
            "beta_star_84",
            "interpolation_systematic",
            "fit_p_value",
            "failure_reasons",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in summary["groups"]:
            beta_star = record["beta_star"]
            writer.writerow(
                {
                    "target_c": record["target_c"],
                    "physical_Nt": record["physical_Nt"],
                    "accepted": record["accepted"],
                    "beta_star_median": (
                        "" if beta_star is None else beta_star["median"]
                    ),
                    "beta_star_16": (
                        "" if beta_star is None else beta_star["statistical_16_84"][0]
                    ),
                    "beta_star_84": (
                        "" if beta_star is None else beta_star["statistical_16_84"][1]
                    ),
                    "interpolation_systematic": (
                        ""
                        if beta_star is None
                        else beta_star["interpolation_systematic_max_abs_shift"]
                    ),
                    "fit_p_value": record.get("fit", {}).get("p_value", ""),
                    "failure_reasons": "; ".join(record["failure_reasons"]),
                }
            )
    np.savez(output / "beta_star_bootstrap.npz", **samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-points", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--audit-results", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path)
    parser.add_argument("--extension-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = load_points(
        args.manifest,
        args.prior_points,
        args.results,
        args.audit_results,
        args.extension_manifest,
        args.extension_results,
    )
    summary, samples = tune_campaign(points)
    write_outputs(summary, samples, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
