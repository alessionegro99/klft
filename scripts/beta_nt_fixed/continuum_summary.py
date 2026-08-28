#!/usr/bin/env python3
"""Assemble fixed-``g_3^2 L`` points and linear continuum extrapolations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats


FloatArray = NDArray[np.float64]
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_SAMPLES = 1000


@dataclass(frozen=True)
class Point:
    """One analyzed lattice spacing and its joint-bootstrap observables."""

    metadata: dict[str, object]
    mass_samples: FloatArray
    ratio_samples: FloatArray | None


def weighted_continuum_fit(
    nt: FloatArray, values: FloatArray, errors: FloatArray
) -> tuple[FloatArray, float, int, float]:
    """Fit ``q(Nt)=q_cont+c/Nt`` with independent point errors."""
    if len(nt) < 3 or values.shape != nt.shape or errors.shape != nt.shape:
        raise ValueError("a continuum fit needs at least three matching points")
    if np.any(errors <= 0.0) or np.any(~np.isfinite(errors)):
        raise ValueError("continuum-fit errors must be finite and positive")
    design = np.column_stack((np.ones_like(nt), 1.0 / nt))
    weighted_design = design / errors[:, None]
    weighted_values = values / errors
    parameters, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_values, rcond=None
    )
    if rank != 2:
        raise ValueError("continuum-fit design matrix is rank deficient")
    residual = (values - design @ parameters) / errors
    chi2 = float(np.dot(residual, residual))
    dof = len(nt) - 2
    p_value = float(stats.chi2.sf(chi2, dof))
    return np.asarray(parameters, dtype=np.float64), chi2, dof, p_value


def _bootstrap_fit(
    nt: FloatArray,
    central: FloatArray,
    errors: FloatArray,
    samples: list[FloatArray],
    rng: np.random.Generator,
) -> tuple[dict[str, object], FloatArray]:
    """Fit the central values and exactly 1,000 permuted bootstrap replicas."""
    parameters, chi2, dof, p_value = weighted_continuum_fit(nt, central, errors)
    valid = [sample[np.isfinite(sample)] for sample in samples]
    if any(len(sample) < BOOTSTRAP_SAMPLES for sample in valid):
        raise ValueError(
            "each continuum point needs at least 1000 finite bootstrap replicas"
        )
    aligned = np.column_stack(
        [rng.permutation(sample)[:BOOTSTRAP_SAMPLES] for sample in valid]
    )
    parameter_samples = np.empty((BOOTSTRAP_SAMPLES, 2), dtype=np.float64)
    for sample_index, sample_values in enumerate(aligned):
        parameter_samples[sample_index] = weighted_continuum_fit(
            nt, sample_values, errors
        )[0]
    parameter_errors = np.std(parameter_samples, axis=0, ddof=1)
    result = {
        "exponent": 1,
        "continuum": float(parameters[0]),
        "continuum_error": float(parameter_errors[0]),
        "coefficient": float(parameters[1]),
        "coefficient_error": float(parameter_errors[1]),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "p_value": p_value,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    return result, parameter_samples


def _read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty continuum selection: {path}")
    return rows


def load_points(
    selection: Path, analysis_root: Path, allow_incomplete: bool
) -> tuple[list[Point], list[dict[str, object]]]:
    """Load every available selected result without substituting volumes."""
    points: list[Point] = []
    unavailable: list[dict[str, object]] = []
    for row in _read_selection(selection):
        directory = analysis_root / row["analysis_relative_path"]
        result_path = directory / "analysis.json"
        covariance_path = directory / "covariance.npz"
        identity = {
            "beta_over_Nt": row["beta_over_nt"],
            "physical_Nt": int(row["physical_Nt"]),
            "analysis_directory": str(directory),
        }
        if not result_path.is_file() or not covariance_path.is_file():
            unavailable.append({**identity, "reason": "analysis files missing"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("mass") is None or result.get("xi_exp") is None:
            unavailable.append({**identity, "reason": "no accepted K0 mass"})
            continue
        with np.load(covariance_path) as archive:
            required = {"mass_bootstrap", "xi_exp_bootstrap", "xi_2nd_bootstrap"}
            if not required.issubset(archive.files):
                unavailable.append(
                    {**identity, "reason": "joint bootstrap arrays missing"}
                )
                continue
            mass_samples = (
                np.asarray(archive["mass_bootstrap"], dtype=np.float64)
                * int(row["physical_Nt"])
            )
            xi_exp_samples = np.asarray(
                archive["xi_exp_bootstrap"], dtype=np.float64
            )
            xi_2nd_samples = np.asarray(
                archive["xi_2nd_bootstrap"], dtype=np.float64
            )
        ratio_samples = None
        if xi_exp_samples.shape == xi_2nd_samples.shape:
            ratio_samples = xi_exp_samples / xi_2nd_samples
        nt = int(row["physical_Nt"])
        mass_nt = float(result["mass"]) * nt
        mass_nt_error = float(result["mass_error"]) * nt
        ratio = float(result["xi_exp"]) / float(result["xi_2nd"])
        ratio_error = (
            float(np.std(ratio_samples[np.isfinite(ratio_samples)], ddof=1))
            if ratio_samples is not None
            else None
        )
        metadata: dict[str, object] = {
            **identity,
            "g3_squared_L": float(row["g3_squared_L"]),
            "beta": float(row["beta"]),
            "Ns": int(row["Ns"]),
            "role": row["role"],
            "mass_Nt": mass_nt,
            "mass_Nt_error": mass_nt_error,
            "xi_exp_over_xi_2nd": ratio,
            "xi_exp_over_xi_2nd_error": ratio_error,
        }
        points.append(Point(metadata, mass_samples, ratio_samples))
    if unavailable and not allow_incomplete:
        first = unavailable[0]
        raise FileNotFoundError(
            f"selected point Nt={first['physical_Nt']} on beta/Nt="
            f"{first['beta_over_Nt']}: {first['reason']}"
        )
    return points, unavailable


def summarize(
    selection: Path,
    analysis_root: Path,
    output_npz: Path,
    allow_incomplete: bool,
) -> dict[str, object]:
    """Return all available points and the retained ``p=1`` fits."""
    points, unavailable = load_points(selection, analysis_root, allow_incomplete)
    lines: dict[str, object] = {}
    arrays: dict[str, FloatArray] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for line in sorted({str(point.metadata["beta_over_Nt"]) for point in points}):
        line_points = [
            point
            for point in points
            if point.metadata["beta_over_Nt"] == line
        ]
        continuum_points = [
            point for point in line_points if point.metadata["role"] == "continuum"
        ]
        line_result: dict[str, object] = {
            "g3_squared_L": float(line_points[0].metadata["g3_squared_L"]),
            "points": [point.metadata for point in line_points],
            "mass_Nt_p1_fit": None,
            "xi_exp_over_xi_2nd_p1_fit": None,
        }
        if len(continuum_points) >= 3:
            nt = np.asarray(
                [point.metadata["physical_Nt"] for point in continuum_points],
                dtype=np.float64,
            )
            mass = np.asarray(
                [point.metadata["mass_Nt"] for point in continuum_points],
                dtype=np.float64,
            )
            mass_error = np.asarray(
                [point.metadata["mass_Nt_error"] for point in continuum_points],
                dtype=np.float64,
            )
            fit, samples = _bootstrap_fit(
                nt,
                mass,
                mass_error,
                [point.mass_samples for point in continuum_points],
                rng,
            )
            line_result["mass_Nt_p1_fit"] = fit
            arrays[f"mass_line_{line.replace('.', 'p')}_p1"] = samples
            if all(point.ratio_samples is not None for point in continuum_points):
                ratio = np.asarray(
                    [
                        point.metadata["xi_exp_over_xi_2nd"]
                        for point in continuum_points
                    ],
                    dtype=np.float64,
                )
                ratio_error = np.asarray(
                    [
                        point.metadata["xi_exp_over_xi_2nd_error"]
                        for point in continuum_points
                    ],
                    dtype=np.float64,
                )
                fit, samples = _bootstrap_fit(
                    nt,
                    ratio,
                    ratio_error,
                    [
                        point.ratio_samples
                        for point in continuum_points
                        if point.ratio_samples is not None
                    ],
                    rng,
                )
                line_result["xi_exp_over_xi_2nd_p1_fit"] = fit
                arrays[f"ratio_line_{line.replace('.', 'p')}_p1"] = samples
        lines[line] = line_result
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays)
    return {
        "definition": "g3_squared_L=4*Nt/beta; mass_Nt=(a*m_G)*Nt",
        "continuum_model": "q(Nt)=q_cont+c/Nt; only p=1 is retained",
        "point_covariance": "independent ensembles",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lines": lines,
        "unavailable_points": unavailable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--covariance-output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = summarize(
        args.selection,
        args.analysis_root,
        args.covariance_output,
        args.allow_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
