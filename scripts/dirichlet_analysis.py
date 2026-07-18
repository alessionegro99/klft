#!/usr/bin/env python3
"""Autocorrelation-aware analysis of temporal-Dirichlet measurements.

The automatic autocorrelation window follows U. Wolff, Comput. Phys. Commun.
156 (2004) 143, arXiv:hep-lat/0306017.  The finite-volume K0 form is Eq.
(4.14) of M. Nebbia, *Effects of Dirichlet Boundary Conditions in Yang-Mills
Theories* (2024/2025 MSc thesis). Delete-one-block jackknife errors follow
B. Efron, *The Jackknife, the Bootstrap and Other Resampling Plans*, SIAM
CBMS-NSF 38 (1982), DOI 10.1137/1.9781611970319.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, special, stats


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KeyedChain:
    """One Markov chain arranged as measurement x key x value."""

    path: Path
    steps: FloatArray
    keys: FloatArray
    values: FloatArray


def load_keyed_chain(
    path: Path,
    key_columns: Sequence[int],
    value_columns: Sequence[int],
    discard: int,
) -> KeyedChain:
    """Load a long table and validate its rectangular step/key structure."""
    raw = np.loadtxt(path, comments="#", ndmin=2, dtype=np.float64)
    if raw.size == 0:
        raise ValueError(f"{path}: empty measurement file")
    steps = np.unique(raw[:, 0])
    if discard < 0 or discard >= len(steps):
        raise ValueError(f"{path}: invalid thermalization discard {discard}")
    steps = steps[discard:]
    raw = raw[np.isin(raw[:, 0], steps)]
    if key_columns:
        keys = np.unique(raw[:, list(key_columns)], axis=0)
    else:
        keys = np.zeros((1, 0), dtype=np.float64)
    expected_rows = len(steps) * len(keys)
    if len(raw) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} step/key rows, found {len(raw)}"
        )
    # Sort once by step and then by each key. This avoids an O(n_steps*n_keys^2)
    # scan for long boundary-Wilson tables while retaining explicit grid checks.
    sort_columns = [raw[:, column] for column in reversed(key_columns)]
    sort_columns.append(raw[:, 0])
    ordered = raw[np.lexsort(tuple(sort_columns))]
    step_grid = ordered[:, 0].reshape(len(steps), len(keys))
    if not np.array_equal(step_grid, np.broadcast_to(steps[:, None], step_grid.shape)):
        raise ValueError(f"{path}: incomplete or duplicate measurement steps")
    if key_columns:
        key_grid = ordered[:, list(key_columns)].reshape(
            len(steps), len(keys), len(key_columns)
        )
        if not np.array_equal(
            key_grid, np.broadcast_to(keys[None, :, :], key_grid.shape)
        ):
            raise ValueError(f"{path}: incomplete, duplicate, or inconsistent keys")
    values = ordered[:, list(value_columns)].reshape(
        len(steps), len(keys), len(value_columns)
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite measurements")
    return KeyedChain(path, steps, keys, values)


def require_matching_grids(chains: Sequence[KeyedChain]) -> None:
    """Require identical observable keys across independent chains."""
    if not chains:
        raise ValueError("at least one chain is required")
    reference = chains[0].keys
    for chain in chains[1:]:
        if not np.array_equal(chain.keys, reference):
            raise ValueError(f"observable grid differs in {chain.path}")


def integrated_autocorrelation_time(values: FloatArray, c: float = 5.0) -> float:
    """Estimate tau_int with an FFT autocovariance and automatic window."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or len(series) < 4:
        raise ValueError("autocorrelation analysis needs at least four values")
    centered = series - np.mean(series)
    if not np.any(centered):
        return 0.5
    n = len(centered)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_size
    )[:n]
    autocovariance /= np.arange(n, 0, -1, dtype=np.float64)
    rho = autocovariance / autocovariance[0]
    tau = 0.5
    for lag in range(1, n):
        tau += rho[lag]
        if lag >= c * max(tau, 0.5):
            break
    return max(float(tau), 0.5)


def maximum_tau(chains: Sequence[KeyedChain]) -> float:
    """Return the largest tau_int among every column in the supplied chains."""
    largest = 0.5
    for chain in chains:
        flat = chain.values.reshape(len(chain.steps), -1)
        for column in flat.T:
            largest = max(largest, integrated_autocorrelation_time(column))
    return largest


def block_values(values: FloatArray, block_size: int) -> FloatArray:
    """Return non-overlapping block means, dropping only a final partial block."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    nblocks = len(values) // block_size
    if nblocks < 2:
        raise ValueError("fewer than two complete blocks")
    trimmed = values[: nblocks * block_size]
    return trimmed.reshape(nblocks, block_size, *values.shape[1:]).mean(axis=1)


def concatenate_blocks(
    chains: Sequence[KeyedChain], block_size: int, value_index: int
) -> tuple[FloatArray, list[FloatArray]]:
    """Block each independent chain before combining its block means."""
    per_chain = [
        block_values(chain.values[:, :, value_index], block_size)
        for chain in chains
    ]
    return np.concatenate(per_chain, axis=0), per_chain


def mean_and_covariance(blocks: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the mean and covariance of that mean from equal-size blocks."""
    flat = np.asarray(blocks, dtype=np.float64)
    if flat.ndim != 2 or len(flat) < 2:
        raise ValueError("mean covariance needs a two-dimensional block array")
    covariance = np.atleast_2d(np.cov(flat, rowvar=False, ddof=1)) / len(flat)
    return np.mean(flat, axis=0), covariance


def jackknife_transform(
    blocks: FloatArray, transform: Callable[[FloatArray], FloatArray]
) -> tuple[FloatArray, FloatArray]:
    """Delete-one-block jackknife for an arbitrary vector-valued transform."""
    if len(blocks) < 3:
        raise ValueError("jackknife needs at least three blocks")
    total = np.sum(blocks, axis=0)
    samples = np.asarray(
        [transform((total - block) / (len(blocks) - 1)) for block in blocks],
        dtype=np.float64,
    )
    estimate = transform(total / len(blocks))
    centered = samples - np.mean(samples, axis=0)
    covariance = (len(blocks) - 1) / len(blocks) * centered.T @ centered
    return np.asarray(estimate), np.atleast_2d(covariance)


def maximum_chain_disagreement(per_chain_blocks: Sequence[FloatArray]) -> float:
    """Largest pairwise component z-score between independent chains."""
    maximum = 0.0
    for left_index, left in enumerate(per_chain_blocks):
        left_mean, left_cov = mean_and_covariance(left)
        for right in per_chain_blocks[left_index + 1 :]:
            right_mean, right_cov = mean_and_covariance(right)
            variance = np.diag(left_cov + right_cov)
            valid = variance > 0.0
            if np.any(valid):
                z = np.abs(left_mean[valid] - right_mean[valid]) / np.sqrt(
                    variance[valid]
                )
                maximum = max(maximum, float(np.max(z)))
    return maximum


def maximum_split_disagreement(chains: Sequence[KeyedChain]) -> float:
    """Largest first-half/second-half z-score, used as stationarity diagnostic."""
    maximum = 0.0
    for chain in chains:
        midpoint = len(chain.steps) // 2
        if midpoint < 4:
            continue
        flat = chain.values.reshape(len(chain.steps), -1)
        for column in flat.T:
            halves = (column[:midpoint], column[-midpoint:])
            means = [float(np.mean(half)) for half in halves]
            variances = []
            for half in halves:
                tau = integrated_autocorrelation_time(half)
                variances.append(float(np.var(half, ddof=1) * 2.0 * tau / len(half)))
            if sum(variances) > 0.0:
                maximum = max(
                    maximum,
                    abs(means[0] - means[1]) / math.sqrt(sum(variances)),
                )
    return maximum


def finite_volume_k0(r: FloatArray, amplitude: float, mass: float, ns: int) -> FloatArray:
    """A[K0(mR)+K0(m(Ns-R))] on a periodic spatial lattice."""
    return amplitude * (special.k0(mass * r) + special.k0(mass * (ns - r)))


def fit_k0(
    separations: FloatArray,
    values: FloatArray,
    covariance: FloatArray,
    ns: int,
    correlated: bool,
) -> dict[str, float | bool]:
    """Fit the finite-periodic-volume K0 correlator with A>0 and m>0."""
    if len(separations) < 3 or np.any(separations <= 0):
        raise ValueError("K0 fit needs at least three strictly positive distances")
    if correlated:
        try:
            scale = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError as error:
            raise ValueError("fit covariance is singular or not positive definite") from error

        def whiten(residual: FloatArray) -> FloatArray:
            return linalg.solve_triangular(scale, residual, lower=True)

    else:
        errors = np.sqrt(np.diag(covariance))
        if np.any(errors <= 0.0):
            raise ValueError("fit variances must be positive")

        def whiten(residual: FloatArray) -> FloatArray:
            return residual / errors

    first_mass = max(1.0 / ns, 0.05)
    denominator = finite_volume_k0(
        np.asarray([separations[0]]), 1.0, first_mass, ns
    )[0]
    first_amplitude = max(values[0] / denominator, 1.0e-8)

    def residual(parameters: FloatArray) -> FloatArray:
        model = finite_volume_k0(separations, parameters[0], parameters[1], ns)
        return whiten(values - model)

    fit = optimize.least_squares(
        residual,
        x0=np.asarray([first_amplitude, first_mass]),
        bounds=(np.asarray([0.0, 1.0e-12]), np.asarray([np.inf, np.inf])),
    )
    if not fit.success:
        raise RuntimeError(f"K0 fit failed: {fit.message}")
    chi2 = float(np.dot(fit.fun, fit.fun))
    dof = len(separations) - 2
    parameter_covariance = linalg.inv(fit.jac.T @ fit.jac)
    return {
        "correlated": correlated,
        "amplitude": float(fit.x[0]),
        "amplitude_error": float(math.sqrt(parameter_covariance[0, 0])),
        "mass": float(fit.x[1]),
        "mass_error": float(math.sqrt(parameter_covariance[1, 1])),
        "chi2": chi2,
        "dof": dof,
        "p_value": float(stats.chi2.sf(chi2, dof)) if dof > 0 else math.nan,
    }


def jackknife_k0_fit(
    blocks: FloatArray,
    separations: FloatArray,
    covariance: FloatArray,
    ns: int,
    correlated: bool,
) -> tuple[FloatArray, FloatArray]:
    """Jackknife ``(A, m)`` while keeping the full-sample fit weights fixed."""

    def transform(mean: FloatArray) -> FloatArray:
        fit = fit_k0(separations, mean, covariance, ns, correlated)
        return np.asarray([fit["amplitude"], fit["mass"]], dtype=np.float64)

    return jackknife_transform(blocks, transform)


def json_array(values: FloatArray) -> list[float | None]:
    """Convert an array to strict JSON, mapping non-finite diagnostics to null."""
    return [float(value) if np.isfinite(value) else None for value in values]


def analyse(args: argparse.Namespace) -> dict[str, object]:
    correlators = [
        load_keyed_chain(path, (1,), (2, 3), args.thermalization)
        for path in args.correlator
    ]
    require_matching_grids(correlators)
    separations = correlators[0].keys[:, 0].astype(np.int64)
    if np.any(separations < 0) or np.any(separations > args.ns // 2):
        raise ValueError("correlator contains redundant R > floor(Ns/2) rows")
    if len(np.unique(separations)) != len(separations):
        raise ValueError("duplicate correlator separation")

    summaries = [
        load_keyed_chain(path, (), (1, 2, 3, 4), args.thermalization)
        for path in args.summary
    ]
    profiles = [
        load_keyed_chain(path, (1, 2), (3,), args.thermalization)
        for path in args.profile
    ]
    wilson = [
        load_keyed_chain(path, (1, 2), (3,), args.thermalization)
        for path in args.wilson
    ]
    quality = [
        load_keyed_chain(path, (1,), (2,), args.thermalization)
        for path in args.bulk_polyakov
    ]
    for optional in (summaries, profiles, wilson, quality):
        if optional:
            if len(optional) != len(correlators):
                raise ValueError("each supplied observable needs one file per chain")
            require_matching_grids(optional)

    tau_inputs = correlators + summaries + profiles + wilson + quality
    tau_max = maximum_tau(tau_inputs)
    block_size = args.block_size or max(1, math.ceil(2.0 * tau_max))
    if block_size <= tau_max:
        raise ValueError(
            f"block size {block_size} is not larger than max tau_int {tau_max:.4g}"
        )

    primary_blocks, _ = concatenate_blocks(correlators, block_size, 0)
    double_blocks, _ = concatenate_blocks(correlators, block_size, 1)
    primary_mean, primary_covariance = mean_and_covariance(primary_blocks)
    primary_error = np.sqrt(np.diag(primary_covariance))
    if primary_mean[0] != 1.0 or primary_error[0] != 0.0:
        raise ValueError("C_G(0) is not exactly one configuration by configuration")

    legacy_difference = double_blocks - primary_blocks
    legacy_mean, legacy_covariance = mean_and_covariance(legacy_difference)
    legacy_error = np.sqrt(np.diag(legacy_covariance))
    legacy_valid = legacy_error > 0.0
    legacy_max_z = (
        float(np.max(np.abs(legacy_mean[legacy_valid]) / legacy_error[legacy_valid]))
        if np.any(legacy_valid)
        else 0.0
    )

    observable_groups = [
        group
        for group in (correlators, summaries, profiles, wilson, quality)
        if group
    ]
    chain_max_z = 0.0
    split_max_z = 0.0
    for group in observable_groups:
        per_chain_blocks = [
            block_values(
                chain.values.reshape(len(chain.steps), -1), block_size
            )
            for chain in group
        ]
        chain_max_z = max(
            chain_max_z, maximum_chain_disagreement(per_chain_blocks)
        )
        split_max_z = max(split_max_z, maximum_split_disagreement(group))
    if chain_max_z > args.max_diagnostic_z:
        raise ValueError(
            f"independent chains disagree: max z={chain_max_z:.3g} exceeds "
            f"{args.max_diagnostic_z:g}"
        )
    if split_max_z > args.max_diagnostic_z:
        raise ValueError(
            f"post-cut history is not stationary: max split z={split_max_z:.3g} "
            f"exceeds {args.max_diagnostic_z:g}"
        )
    if legacy_max_z > args.max_diagnostic_z:
        raise ValueError(
            f"global SU(2) orientation is not sampled: max group-average "
            f"z={legacy_max_z:.3g} exceeds {args.max_diagnostic_z:g}"
        )

    def effective_transform(correlator: FloatArray) -> FloatArray:
        result = np.full(len(correlator) - 1, np.nan, dtype=np.float64)
        valid = (correlator[:-1] > 0.0) & (correlator[1:] > 0.0)
        result[valid] = np.log(correlator[:-1][valid] / correlator[1:][valid])
        return result

    effective, effective_covariance = jackknife_transform(
        primary_blocks, effective_transform
    )
    result: dict[str, object] = {
        "n_chains": len(correlators),
        "thermalization_measurements_discarded": args.thermalization,
        "max_tau_int": tau_max,
        "block_size": block_size,
        "n_blocks": len(primary_blocks),
        "max_chain_disagreement_z": chain_max_z,
        "max_split_history_z": split_max_z,
        "max_global_orientation_relation_z": legacy_max_z,
        "R": separations.tolist(),
        "C_G": json_array(primary_mean),
        "C_G_error": json_array(primary_error),
        "C_double_minus_C_G": json_array(legacy_mean),
        "C_double_minus_C_G_error": json_array(legacy_error),
        "m_eff_log": json_array(effective),
        "m_eff_log_error": json_array(np.sqrt(np.diag(effective_covariance))),
    }

    if summaries:
        summary_blocks = np.concatenate(
            [block_values(chain.values[:, 0, :], block_size) for chain in summaries],
            axis=0,
        )
        summary_mean, summary_covariance = mean_and_covariance(summary_blocks)

        def xi_transform(summary: FloatArray) -> FloatArray:
            ratio = summary[2] / summary[3] - 1.0
            xi = (
                math.sqrt(ratio) / (2.0 * math.sin(math.pi / args.ns))
                if ratio >= 0.0 and summary[3] > 0.0
                else math.nan
            )
            return np.asarray([xi])

        xi, xi_covariance = jackknife_transform(summary_blocks, xi_transform)
        result["mean_TrG"] = float(summary_mean[0])
        result["mean_TrG_error"] = float(math.sqrt(summary_covariance[0, 0]))
        # The measurement file retains raw Tr G for the legacy group-averaging
        # identity. The thesis plots a trace normalized to one at G=I.
        result["mean_half_TrG"] = 0.5 * float(summary_mean[0])
        result["mean_half_TrG_error"] = 0.5 * float(
            math.sqrt(summary_covariance[0, 0])
        )
        result["xi_2nd"] = float(xi[0]) if np.isfinite(xi[0]) else None
        result["xi_2nd_error"] = (
            float(math.sqrt(xi_covariance[0, 0]))
            if np.isfinite(xi_covariance[0, 0])
            else None
        )

    fits: list[dict[str, object]] = []
    fit_rmax_values = args.fit_rmax or [int(separations[-1])]
    for fit_rmin in args.fit_rmin:
        for fit_rmax in fit_rmax_values:
            selected = (separations >= fit_rmin) & (separations <= fit_rmax)
            if np.count_nonzero(selected) < 3:
                continue
            fit_entry: dict[str, object] = {
                "r_min": fit_rmin,
                "r_max": fit_rmax,
            }
            for correlated in (True, False):
                label = "correlated" if correlated else "uncorrelated"
                try:
                    fit_result = fit_k0(
                        separations[selected].astype(np.float64),
                        primary_mean[selected],
                        primary_covariance[np.ix_(selected, selected)],
                        args.ns,
                        correlated,
                    )
                except (ValueError, RuntimeError, linalg.LinAlgError) as error:
                    fit_entry[label] = {"error": str(error)}
                    continue
                try:
                    jackknife_parameters, jackknife_covariance = jackknife_k0_fit(
                        primary_blocks[:, selected],
                        separations[selected].astype(np.float64),
                        primary_covariance[np.ix_(selected, selected)],
                        args.ns,
                        correlated,
                    )
                    jackknife_variances = np.diag(jackknife_covariance)
                    if np.any(~np.isfinite(jackknife_variances)) or np.any(
                        jackknife_variances < 0.0
                    ):
                        raise ValueError("non-finite jackknife fit covariance")
                    fit_result["amplitude_curvature_error"] = fit_result[
                        "amplitude_error"
                    ]
                    fit_result["mass_curvature_error"] = fit_result["mass_error"]
                    fit_result["amplitude"] = float(jackknife_parameters[0])
                    fit_result["mass"] = float(jackknife_parameters[1])
                    fit_result["amplitude_error"] = float(
                        math.sqrt(jackknife_variances[0])
                    )
                    fit_result["mass_error"] = float(
                        math.sqrt(jackknife_variances[1])
                    )
                    fit_result["error_method"] = (
                        "delete-one-block jackknife with fixed full-sample weights"
                    )
                except (ValueError, RuntimeError, linalg.LinAlgError) as error:
                    fit_result["jackknife_error"] = str(error)
                fit_entry[label] = fit_result
            fits.append(fit_entry)
    result["K0_fit_scans"] = fits

    if profiles:
        profile_blocks = np.concatenate(
            [block_values(chain.values[:, :, 0], block_size) for chain in profiles],
            axis=0,
        )
        profile_mean, profile_covariance = mean_and_covariance(profile_blocks)
        profile_error = np.sqrt(np.diag(profile_covariance))
        keys = profiles[0].keys
        thickness = int(np.max(keys[keys[:, 0] == 0, 1]))
        reflection_z = []
        for index, (kind, coordinate) in enumerate(keys):
            reflected_coordinate = thickness - coordinate
            matches = np.flatnonzero(
                (keys[:, 0] == kind) & (keys[:, 1] == reflected_coordinate)
            )
            if len(matches) == 1:
                other = int(matches[0])
                variance = (
                    profile_covariance[index, index]
                    + profile_covariance[other, other]
                    - 2.0 * profile_covariance[index, other]
                )
                if variance > 0.0:
                    reflection_z.append(
                        abs(profile_mean[index] - profile_mean[other])
                        / math.sqrt(variance)
                    )
        result["plaquette_profile"] = {
            "kind_t": keys.tolist(),
            "mean": json_array(profile_mean),
            "error": json_array(profile_error),
            "max_reflection_z": max(reflection_z, default=0.0),
        }

    if wilson:
        wilson_blocks = np.concatenate(
            [block_values(chain.values[:, :, 0], block_size) for chain in wilson],
            axis=0,
        )
        wilson_mean, wilson_covariance = mean_and_covariance(wilson_blocks)
        result["boundary_wilson"] = {
            "R_H": wilson[0].keys.tolist(),
            "mean": json_array(wilson_mean),
            "error": json_array(np.sqrt(np.diag(wilson_covariance))),
        }

    if quality:
        quality_blocks = np.concatenate(
            [block_values(chain.values[:, :, 0], block_size) for chain in quality],
            axis=0,
        )
        quality_mean, quality_covariance = mean_and_covariance(quality_blocks)
        result["bulk_horizontal_polyakov"] = {
            "R": quality[0].keys[:, 0].astype(int).tolist(),
            "mean": json_array(quality_mean),
            "error": json_array(np.sqrt(np.diag(quality_covariance))),
        }

    correlation_denominator = np.sqrt(
        np.outer(np.diag(primary_covariance), np.diag(primary_covariance))
    )
    correlation = np.full_like(primary_covariance, np.nan)
    np.divide(
        primary_covariance,
        correlation_denominator,
        out=correlation,
        where=correlation_denominator > 0.0,
    )
    np.savez(
        args.covariance_output,
        R=separations,
        C_G=primary_mean,
        covariance=primary_covariance,
        correlation=correlation,
        fit_R=separations[1:],
        fit_covariance=primary_covariance[1:, 1:],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correlator", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, nargs="*", default=[])
    parser.add_argument("--profile", type=Path, nargs="*", default=[])
    parser.add_argument("--wilson", type=Path, nargs="*", default=[])
    parser.add_argument("--bulk-polyakov", type=Path, nargs="*", default=[])
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument(
        "--thermalization",
        type=int,
        default=500,
        help="number of initial measurements discarded separately per chain",
    )
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument(
        "--max-diagnostic-z",
        type=float,
        default=5.0,
        help="fail rather than combine chains when a consistency diagnostic exceeds this",
    )
    parser.add_argument("--fit-rmin", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--fit-rmax", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("dirichlet_analysis.json"))
    parser.add_argument(
        "--covariance-output", type=Path, default=Path("dirichlet_covariance.npz")
    )
    args = parser.parse_args()
    if args.ns < 2:
        raise ValueError("Ns must be at least two")
    result = analyse(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
