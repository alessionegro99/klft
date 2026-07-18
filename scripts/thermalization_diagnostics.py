#!/usr/bin/env python3
"""Quantitative thermalization diagnostics for independent Dirichlet chains.

The candidate cut uses MSER-5 batch means [White, Simulation 69 (1997) 323].
Post-cut integrated autocorrelation times use the automatic-window estimator of
Sokal, hep-lat/9605018, Secs. 3.2 and 6.4. A cut is accepted only after the
reported split-history and independent-chain diagnostics are also inspected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dirichlet_analysis import integrated_autocorrelation_time


FloatArray = NDArray[np.float64]


def mser5_cut(values: FloatArray) -> int:
    """Return the MSER-5 candidate deletion count in measurements."""
    batch_size = 5
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("MSER-5 expects a one-dimensional series")
    n_batches = len(series) // batch_size
    if n_batches < 4:
        return 0
    batches = series[: n_batches * batch_size].reshape(n_batches, batch_size)
    batches = batches.mean(axis=1)
    reverse_sum = np.cumsum(batches[::-1])[::-1]
    reverse_sum2 = np.cumsum((batches * batches)[::-1])[::-1]
    n_remaining = np.arange(n_batches, 0, -1, dtype=np.float64)
    variance = (reverse_sum2 - reverse_sum * reverse_sum / n_remaining) / np.maximum(
        n_remaining - 1.0, 1.0
    )
    variance_of_mean = variance / n_remaining
    search = max(1, n_batches // 2)
    return int(np.argmin(variance_of_mean[:search])) * batch_size


def load_diagnostics(summary_path: Path, correlator_path: Path) -> dict[str, FloatArray]:
    """Load three complementary scalar histories from one chain."""
    summary = np.loadtxt(summary_path, comments="#", ndmin=2, dtype=np.float64)
    correlator = np.loadtxt(
        correlator_path, comments="#", ndmin=2, dtype=np.float64
    )
    if summary.shape[1] != 6 or correlator.shape[1] != 5:
        raise ValueError(f"unexpected observable columns below {summary_path.parent}")
    r_one = correlator[correlator[:, 1] == 1.0]
    if len(r_one) != len(summary) or not np.array_equal(r_one[:, 0], summary[:, 0]):
        raise ValueError(f"mismatched measurement grids below {summary_path.parent}")
    values = {
        "mean_TrG": summary[:, 1],
        "Ctilde_0": summary[:, 3],
        "C_G_R1": r_one[:, 2],
    }
    if any(not np.all(np.isfinite(series)) for series in values.values()):
        raise ValueError(f"non-finite diagnostic below {summary_path.parent}")
    return values


def split_z_score(values: FloatArray) -> float:
    """Compare the post-cut history halves using autocorrelation-aware errors."""
    midpoint = len(values) // 2
    if midpoint < 4:
        return math.nan
    halves = (values[:midpoint], values[-midpoint:])
    means = [float(np.mean(half)) for half in halves]
    variances = [
        float(np.var(half, ddof=1) * 2.0 * integrated_autocorrelation_time(half) / len(half))
        for half in halves
    ]
    return (
        abs(means[0] - means[1]) / math.sqrt(sum(variances))
        if sum(variances) > 0.0
        else 0.0
    )


def analyse(
    summary_paths: Sequence[Path],
    correlator_paths: Sequence[Path],
    minimum_cut: int,
) -> dict[str, object]:
    """Return per-chain MSER cuts and post-cut stationarity diagnostics."""
    if len(summary_paths) != len(correlator_paths) or not summary_paths:
        raise ValueError("supply one summary and correlator file per chain")
    if minimum_cut < 0:
        raise ValueError("minimum cut must be non-negative")
    chains = [
        load_diagnostics(summary, correlator)
        for summary, correlator in zip(summary_paths, correlator_paths, strict=True)
    ]
    lengths = {len(next(iter(chain.values()))) for chain in chains}
    if len(lengths) != 1:
        raise ValueError("independent chains have different history lengths")
    n_measurements = lengths.pop()
    per_chain: list[dict[str, object]] = []
    recommended_cut = minimum_cut
    for summary_path, chain in zip(summary_paths, chains, strict=True):
        cuts = {name: mser5_cut(values) for name, values in chain.items()}
        chain_cut = max([minimum_cut, *cuts.values()])
        recommended_cut = max(recommended_cut, chain_cut)
        per_chain.append(
            {
                "chain": str(summary_path.parent),
                "mser5_cut": cuts,
                "candidate_cut": chain_cut,
            }
        )
    if recommended_cut >= n_measurements - 4:
        raise ValueError("candidate thermalization cut leaves too few measurements")

    for entry, chain in zip(per_chain, chains, strict=True):
        post_cut = {
            name: values[recommended_cut:] for name, values in chain.items()
        }
        entry["post_cut_tau_int"] = {
            name: integrated_autocorrelation_time(values)
            for name, values in post_cut.items()
        }
        entry["post_cut_split_z"] = {
            name: split_z_score(values) for name, values in post_cut.items()
        }

    chain_pair_max_z: dict[str, float] = {}
    for name in chains[0]:
        means = []
        variances = []
        for chain in chains:
            values = chain[name][recommended_cut:]
            tau = integrated_autocorrelation_time(values)
            means.append(float(np.mean(values)))
            variances.append(float(np.var(values, ddof=1) * 2.0 * tau / len(values)))
        maximum = 0.0
        for left in range(len(chains)):
            for right in range(left + 1, len(chains)):
                variance = variances[left] + variances[right]
                if variance > 0.0:
                    maximum = max(
                        maximum,
                        abs(means[left] - means[right]) / math.sqrt(variance),
                    )
        chain_pair_max_z[name] = maximum

    return {
        "method": "MSER-5 candidate plus post-cut Sokal-window diagnostics",
        "n_chains": len(chains),
        "n_measurements_per_chain": n_measurements,
        "minimum_cut": minimum_cut,
        "recommended_common_cut": recommended_cut,
        "chains": per_chain,
        "post_cut_max_pairwise_chain_z": chain_pair_max_z,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--correlator", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum-cut", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path("thermalization_diagnostics.json")
    )
    args = parser.parse_args()
    result = analyse(args.summary, args.correlator, args.minimum_cut)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
