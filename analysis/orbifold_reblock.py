"""Reblock existing chainwise Wilson statistics for uncertainty checks.

Adjacent equal-size blocks are averaged within each chain. The vector
hierarchical bootstrap is the one in orbifold_hmc_stats.py: chains are never
concatenated and cross-loop covariance is retained.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from orbifold_hmc_stats import hierarchical_bootstrap


def combine_blocks(values: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Average adjacent blocks separately in each chain, dropping a short tail."""
    if values.ndim != 3 or factor < 1 or not np.all(np.isfinite(values)):
        raise ValueError("expected finite chain/block/observable data and positive factor")
    chains, blocks, observables = values.shape
    count = blocks // factor
    if chains < 2 or count < 8:
        raise ValueError("require at least two chains and eight new blocks per chain")
    return values[:, :count * factor].reshape(
        chains, count, factor, observables).mean(axis=2)


def reblock(source: Path, output: Path, factor: int, samples: int, seed: int) -> None:
    """Write a new covariance/bootstrap archive without touching source data."""
    if samples < 2:
        raise ValueError("at least two bootstrap samples are required")
    for path in (output, output.with_suffix(".summary.txt")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    with np.load(source) as data:
        pairs = data["pairs"].copy()
        original = np.asarray(data["block_means"], dtype=np.float64)
        steps = data["measurement_trajectories"].copy()
        tau = data["tau_int"].copy()
        rhat = data["split_rhat"].copy()
    if len(steps) % original.shape[1]:
        raise ValueError("measurement count does not match original equal-size blocks")
    original_length = len(steps) // original.shape[1]
    blocks = combine_blocks(original, factor)
    length = factor * original_length
    if length <= np.max(tau):
        raise ValueError("blocks must exceed the measured autocorrelation scale")
    bootstrap = hierarchical_bootstrap(blocks, samples, seed)
    mean = blocks.mean(axis=(0, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, pairs=pairs, mean=mean,
                        bootstrap_means=bootstrap,
                        covariance=np.cov(bootstrap, rowvar=False),
                        block_means=blocks, tau_int=tau, split_rhat=rhat,
                        measurement_trajectories=steps[:blocks.shape[1] * length])
    output.with_suffix(".summary.txt").write_text(
        f"source {source}\nsource_sha256 {hashlib.sha256(source.read_bytes()).hexdigest()}\n"
        f"original_block_length {original_length}\nblock_factor {factor}\n"
        f"block_length {length}\nblocks_per_chain {blocks.shape[1]}\n"
        f"retained_vectors_per_chain {blocks.shape[1] * length}\n"
        f"maximum_loop_tau_int {np.max(tau):.17g}\n"
        f"bootstrap_samples {samples}\nbootstrap_seed {seed}\n"
        "method adjacent_chainwise_reblocking_then_vector_hierarchical_bootstrap\n",
        encoding="utf-8")


def self_test() -> None:
    """Check exact adjacent averaging, chain separation, and short-tail removal."""
    values = np.arange(2 * 19 * 3, dtype=np.float64).reshape(2, 19, 3)
    result = combine_blocks(values, 2)
    expected = (values[:, :18:2] + values[:, 1:18:2]) / 2.0
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(combine_blocks(values, 1), values)
    for factor in (0, 3):
        try:
            combine_blocks(values, factor)
        except ValueError:
            continue
        raise AssertionError("invalid/undersampled reblocking was accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--factor", type=int)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=26091107)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source is None or args.output is None or args.factor is None:
        parser.error("source, --output, and --factor are required")
    if args.output.suffix != ".npz":
        parser.error("output must have .npz suffix")
    reblock(args.source, args.output, args.factor, args.samples, args.seed)


if __name__ == "__main__":
    main()
