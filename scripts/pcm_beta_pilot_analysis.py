#!/usr/bin/env python3
"""Chain-aware analysis of the exact-``Nt=1`` SU(2) PCM pilot.

Independent histories are never concatenated. Each chain is thermalized and
blocked separately, then a two-level bootstrap resamples chains and complete
within-chain block vectors. This preserves covariance across ``R``. Blocking
before resampling follows H. R. Kunsch, Ann. Stat. 17 (1989) 1217,
DOI 10.1214/aos/1176347265; the nested resampling logic follows the
hierarchical bootstrap described by S. Saravanan, N. Berman, and A. Sober,
Nature Neuroscience 23 (2020) 1647, DOI 10.1038/s41593-020-00709-0.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path

import numpy as np

from dirichlet_analysis import (
    FloatArray,
    block_values,
    integrated_autocorrelation_time,
    jackknife_transform,
    load_keyed_chain,
    maximum_chain_disagreement,
    mean_and_covariance,
    require_matching_grids,
)
from thermalization_diagnostics import split_z_score


def hierarchical_bootstrap(
    per_chain_blocks: Sequence[FloatArray],
    transform: Callable[[FloatArray], FloatArray],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return point estimate, samples, and covariance from nested resampling.

    Every block is a complete observable vector. The outer level resamples
    independent chains; the inner level independently resamples block means
    within each selected chain. Equal chain weights prevent a longer retained
    history from silently dominating the between-run fluctuations.
    """
    if len(per_chain_blocks) < 2:
        raise ValueError("hierarchical bootstrap needs at least two chains")
    if n_resamples < 2:
        raise ValueError("at least two bootstrap resamples are required")
    vector_shape = per_chain_blocks[0].shape[1:]
    if any(blocks.shape[1:] != vector_shape for blocks in per_chain_blocks):
        raise ValueError("chain block vectors have inconsistent shapes")
    if any(len(blocks) < 2 for blocks in per_chain_blocks):
        raise ValueError("every chain needs at least two complete blocks")

    chain_means = np.asarray(
        [np.mean(blocks, axis=0) for blocks in per_chain_blocks],
        dtype=np.float64,
    )
    estimate = np.asarray(transform(np.mean(chain_means, axis=0)), dtype=np.float64)
    samples = np.empty((n_resamples, *estimate.shape), dtype=np.float64)
    n_chains = len(per_chain_blocks)
    for sample_index in range(n_resamples):
        selected_chains = rng.integers(0, n_chains, size=n_chains)
        resampled_means = []
        for chain_index in selected_chains:
            blocks = per_chain_blocks[int(chain_index)]
            selected_blocks = rng.integers(0, len(blocks), size=len(blocks))
            resampled_means.append(np.mean(blocks[selected_blocks], axis=0))
        samples[sample_index] = transform(np.mean(resampled_means, axis=0))
    if not np.all(np.isfinite(samples)):
        raise ValueError("non-finite derived quantity in hierarchical bootstrap")
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    return estimate, samples, covariance


def second_moment_length(summary: FloatArray, ns: int) -> float:
    """Return xi_2nd from ``(Ctilde(0), Ctilde(p_min))`` ensemble means."""
    c_zero = float(summary[2])
    c_minimum = float(summary[3])
    ratio = c_zero / c_minimum - 1.0 if c_minimum > 0.0 else -1.0
    if ratio <= 0.0:
        raise ValueError("second-moment length is unresolved (non-positive ratio)")
    return math.sqrt(ratio) / (2.0 * math.sin(math.pi / ns))


def tau_diagnostics(summary: FloatArray, correlator: FloatArray) -> dict[str, float]:
    """Return chain-specific autocorrelation estimates used for blocking."""
    correlator_taus = [
        integrated_autocorrelation_time(correlator[:, index])
        for index in range(1, correlator.shape[1])
    ]
    return {
        "mean_TrG": integrated_autocorrelation_time(summary[:, 0]),
        "Ctilde_0": integrated_autocorrelation_time(summary[:, 2]),
        "Ctilde_pmin": integrated_autocorrelation_time(summary[:, 3]),
        "C_G_R1": integrated_autocorrelation_time(correlator[:, 1]),
        "max_C_G_R_positive": max(correlator_taus),
    }


def json_floats(values: FloatArray) -> list[float]:
    """Convert finite float64 values to JSON scalars."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot serialize non-finite analysis result")
    return [float(value) for value in array]


def analyse(args: argparse.Namespace) -> dict[str, object]:
    """Analyze one beta using per-chain cuts, blocks, and nested resampling."""
    n_chains = len(args.correlator)
    if n_chains != len(args.summary) or n_chains != len(args.thermalization):
        raise ValueError("supply one correlator, summary, and cut per chain")
    if n_chains < 2:
        raise ValueError("at least two independent chains are required")

    correlators = [
        load_keyed_chain(path, (1,), (2,), cut)
        for path, cut in zip(args.correlator, args.thermalization, strict=True)
    ]
    summaries = [
        load_keyed_chain(path, (), (1, 2, 3, 4), cut)
        for path, cut in zip(args.summary, args.thermalization, strict=True)
    ]
    require_matching_grids(correlators)
    for correlator, summary in zip(correlators, summaries, strict=True):
        if not np.array_equal(correlator.steps, summary.steps):
            raise ValueError(f"measurement grids differ below {summary.path.parent}")

    separations = correlators[0].keys[:, 0].astype(np.int64)
    if not np.array_equal(separations, np.arange(args.ns // 2 + 1)):
        raise ValueError("expected exactly the independent R=0,...,floor(Ns/2) grid")
    if any(not np.all(chain.values[:, 0, 0] == 1.0) for chain in correlators):
        raise ValueError("C_G(0) is not exactly one configuration by configuration")

    per_chain_blocks: list[FloatArray] = []
    chain_results: list[dict[str, object]] = []
    for correlator, summary, cut in zip(
        correlators, summaries, args.thermalization, strict=True
    ):
        summary_values = summary.values[:, 0, :]
        correlator_values = correlator.values[:, :, 0]
        taus = tau_diagnostics(summary_values, correlator_values)
        tau_max = max(taus.values())
        block_size = math.floor(args.block_factor * tau_max) + 1
        combined = np.concatenate((summary_values, correlator_values), axis=1)
        blocks = block_values(combined, block_size)
        if len(blocks) < args.minimum_blocks:
            raise ValueError(
                f"{summary.path.parent}: only {len(blocks)} blocks after cut; "
                f"need at least {args.minimum_blocks}"
            )
        per_chain_blocks.append(blocks)

        def xi_transform(mean: FloatArray) -> FloatArray:
            return np.asarray([second_moment_length(mean[:4], args.ns)])

        xi, xi_covariance = jackknife_transform(blocks, xi_transform)
        block_mean, block_covariance = mean_and_covariance(blocks)
        split_inputs = {
            "mean_TrG": summary_values[:, 0],
            "Ctilde_0": summary_values[:, 2],
            "Ctilde_pmin": summary_values[:, 3],
            "C_G_R1": correlator_values[:, 1],
        }
        chain_results.append(
            {
                "chain": str(summary.path.parent),
                "thermalization_cut": cut,
                "retained_measurements": len(summary.steps),
                "tau_int": taus,
                "block_size": block_size,
                "n_blocks": len(blocks),
                "discarded_partial_measurements": len(summary.steps)
                - len(blocks) * block_size,
                "post_cut_split_z": {
                    name: split_z_score(values)
                    for name, values in split_inputs.items()
                },
                "xi_2nd": float(xi[0]),
                "xi_2nd_jackknife_error": float(math.sqrt(xi_covariance[0, 0])),
                "C_G_R1": float(block_mean[5]),
                "C_G_R1_error": float(math.sqrt(block_covariance[5, 5])),
            }
        )

    def transform(mean: FloatArray) -> FloatArray:
        xi = second_moment_length(mean[:4], args.ns)
        correlator = mean[4:]
        return np.concatenate(
            (correlator, np.asarray([xi, args.ns / xi, mean[0]]))
        )

    estimate, samples, covariance = hierarchical_bootstrap(
        per_chain_blocks,
        transform,
        args.bootstrap_samples,
        np.random.default_rng(args.bootstrap_seed),
    )
    n_r = len(separations)
    correlator_mean = estimate[:n_r]
    xi = float(estimate[n_r])
    volume_ratio = float(estimate[n_r + 1])
    mean_tr_g = float(estimate[n_r + 2])
    errors = np.sqrt(np.diag(covariance))
    lower, upper = np.quantile(samples, [0.16, 0.84], axis=0)
    correlator_covariance = covariance[:n_r, :n_r]
    denominator = np.sqrt(
        np.outer(np.diag(correlator_covariance), np.diag(correlator_covariance))
    )
    correlator_correlation = np.full_like(correlator_covariance, np.nan)
    np.divide(
        correlator_covariance,
        denominator,
        out=correlator_correlation,
        where=denominator > 0.0,
    )
    np.savez(
        args.covariance_output,
        R=separations,
        C_G=correlator_mean,
        covariance=correlator_covariance,
        correlation=correlator_correlation,
        fit_R=separations[1:],
        fit_covariance=correlator_covariance[1:, 1:],
    )

    conservative_criteria = bool(
        lower[n_r] >= args.minimum_xi
        and args.ns / upper[n_r] >= args.minimum_volume_ratio
    )
    return {
        "method": (
            "per-chain thermalization and blocking; hierarchical bootstrap of "
            "independent chains and complete within-chain block vectors"
        ),
        "ns": args.ns,
        "n_chains": n_chains,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "block_factor_times_tau_int": args.block_factor,
        "minimum_blocks_per_chain": args.minimum_blocks,
        "chains": chain_results,
        "max_pairwise_chain_disagreement_z": maximum_chain_disagreement(
            per_chain_blocks
        ),
        "R": separations.tolist(),
        "C_G": json_floats(correlator_mean),
        "C_G_error": json_floats(errors[:n_r]),
        "xi_2nd": xi,
        "xi_2nd_error": float(errors[n_r]),
        "xi_2nd_68_percent_interval": [float(lower[n_r]), float(upper[n_r])],
        "Ns_over_xi_2nd": volume_ratio,
        "Ns_over_xi_2nd_error": float(errors[n_r + 1]),
        "Ns_over_xi_2nd_68_percent_interval": [
            float(lower[n_r + 1]),
            float(upper[n_r + 1]),
        ],
        "mean_TrG": mean_tr_g,
        "mean_TrG_error": float(errors[n_r + 2]),
        "selection_criteria": {
            "minimum_xi": args.minimum_xi,
            "minimum_Ns_over_xi": args.minimum_volume_ratio,
            "central_values_pass": bool(
                xi >= args.minimum_xi and volume_ratio >= args.minimum_volume_ratio
            ),
            "68_percent_conservative_pass": conservative_criteria,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correlator", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    cuts = parser.add_mutually_exclusive_group(required=True)
    cuts.add_argument("--thermalization", type=int, nargs="+")
    cuts.add_argument(
        "--thermalization-diagnostics",
        type=Path,
        help="JSON from thermalization_diagnostics.py; paths and cut order are checked",
    )
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--block-factor", type=float, default=2.0)
    parser.add_argument("--minimum-blocks", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--minimum-xi", type=float, default=5.0)
    parser.add_argument("--minimum-volume-ratio", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=Path("pcm_beta_analysis.json"))
    parser.add_argument(
        "--covariance-output", type=Path, default=Path("pcm_beta_covariance.npz")
    )
    args = parser.parse_args()
    if args.ns < 2:
        raise ValueError("Ns must be at least two")
    if args.block_factor <= 1.0:
        raise ValueError("block factor must exceed one")
    if args.minimum_blocks < 3:
        raise ValueError("at least three blocks per chain are required")
    if args.thermalization_diagnostics is not None:
        diagnostics = json.loads(
            args.thermalization_diagnostics.read_text(encoding="utf-8")
        )
        diagnostic_paths = [Path(entry["chain"]) for entry in diagnostics["chains"]]
        summary_paths = [path.parent for path in args.summary]
        if diagnostic_paths != summary_paths:
            raise ValueError("diagnostic chain order does not match summary paths")
        args.thermalization = diagnostics["recommended_per_chain_cuts"]
    result = analyse(args)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
