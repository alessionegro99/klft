#!/usr/bin/env python3
"""Autocorrelation-aware statistics for independent orbifold HMC chains.

Autocorrelation times use the automatic window of U. Wolff,
Comput. Phys. Commun. 156 (2004) 143, hep-lat/0306017. Split R-hat follows
A. Vehtari et al., Bayesian Anal. 16 (2021) 667, doi:10.1214/20-BA1221.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from partition_benchmark import integrated_autocorrelation_time


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def load_loops(path: Path) -> tuple[IntArray, IntArray, FloatArray]:
    """Load and validate one complete loop vector per recorded trajectory."""
    rows = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[1] != 4 or not np.all(np.isfinite(rows)):
        raise ValueError(f"invalid Wilson-loop data in {path}: {rows.shape}")
    steps, counts = np.unique(rows[:, 0].astype(np.int64), return_counts=True)
    if steps.size == 0 or np.any(counts != counts[0]):
        raise ValueError(f"incomplete Wilson-loop vectors in {path}")
    blocks = rows.reshape(steps.size, counts[0], 4)
    pairs = blocks[0, :, 1:3].astype(np.int64)
    if not np.array_equal(
        blocks[:, :, 0], np.broadcast_to(steps[:, None], blocks[:, :, 0].shape)
    ):
        raise ValueError(f"noncontiguous Wilson-loop vectors in {path}")
    if not np.array_equal(
        blocks[:, :, 1:3].astype(np.int64),
        np.broadcast_to(pairs, blocks[:, :, 1:3].shape),
    ):
        raise ValueError(f"inconsistent Wilson-loop grid in {path}")
    return steps, pairs, blocks[:, :, 3]


def split_rhat(values: FloatArray) -> FloatArray:
    """Return split R-hat for arrays shaped (chain, draw, observable)."""
    half = values.shape[1] // 2
    if half < 2:
        raise ValueError("at least four retained draws are required")
    split = np.concatenate((values[:, :half], values[:, -half:]), axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    numerator = (half - 1) * within / half + between / half
    return np.sqrt(np.divide(numerator, within, out=np.ones_like(within), where=within > 0.0))


def hierarchical_bootstrap(
    blocks: FloatArray, samples: int, seed: int
) -> FloatArray:
    """Resample chains, then vector-valued blocks within selected chains."""
    rng = np.random.default_rng(seed)
    chains, n_blocks, observables = blocks.shape
    output = np.empty((samples, observables), dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(start + 100, samples)
        outer = rng.integers(chains, size=(stop - start, chains))
        inner = rng.integers(n_blocks, size=(stop - start, chains, n_blocks))
        output[start:stop] = blocks[outer[:, :, None], inner].mean(axis=(1, 2))
    return output


def analyze(root: Path, output: Path, discard: int, samples: int, seed: int) -> None:
    """Analyze all ``chain*`` directories below *root*."""
    directories = sorted(path for path in root.glob("chain*") if path.is_dir())
    if len(directories) < 2:
        raise ValueError("at least two chain directories are required")

    loop_series = []
    action_series = []
    acceptances = []
    reference_steps: IntArray | None = None
    reference_pairs: IntArray | None = None
    for directory in directories:
        hmc = np.loadtxt(directory / "hmc.out", comments="#", dtype=np.float64, ndmin=2)
        diagnostic = np.loadtxt(
            directory / "diagnostic.out", comments="#", dtype=np.float64, ndmin=2
        )
        steps, pairs, loops = load_loops(directory / "wilson.out")
        if (
            hmc.shape[1] != 7
            or diagnostic.shape[1] != 4
            or not np.all(np.isfinite(hmc))
            or not np.all(np.isfinite(diagnostic))
        ):
            raise ValueError(f"invalid HMC or diagnostic data in {directory}")
        production_diagnostic = diagnostic[diagnostic[:, 1] > 0]
        if not np.array_equal(production_diagnostic[:, 0].astype(np.int64), steps):
            raise ValueError(f"diagnostics and Wilson loops are misaligned in {directory}")
        if reference_steps is not None and not np.array_equal(steps, reference_steps):
            raise ValueError("chain measurement trajectories differ")
        if reference_pairs is not None and not np.array_equal(pairs, reference_pairs):
            raise ValueError("chain Wilson-loop grids differ")
        reference_steps, reference_pairs = steps, pairs
        loop_series.append(loops)
        action_series.append(production_diagnostic[:, 2])
        acceptances.append(float(np.mean(hmc[hmc[:, 1] > 0, 2])))

    loops = np.asarray(loop_series)[:, discard:]
    actions = np.asarray(action_series)[:, discard:, None]
    if loops.shape[1] < 8:
        raise ValueError("discard leaves too few measurements")
    assert reference_pairs is not None and reference_steps is not None

    tau = np.asarray(
        [
            [
                integrated_autocorrelation_time(chain[:, column])
                for column in range(loops.shape[2])
            ]
            for chain in loops
        ]
    )
    max_tau = float(np.max(tau))
    block_length = int(np.ceil(max_tau)) + 1
    n_blocks = loops.shape[1] // block_length
    if n_blocks < 2:
        raise ValueError("too few measurements for the estimated autocorrelation")
    blocked = loops[:, : n_blocks * block_length].reshape(
        len(directories), n_blocks, block_length, loops.shape[2]
    ).mean(axis=2)
    bootstrap = hierarchical_bootstrap(blocked, samples, seed)
    mean = blocked.mean(axis=(0, 1))
    covariance = np.cov(bootstrap, rowvar=False)
    error = np.sqrt(np.diag(covariance))
    rhat = split_rhat(loops)
    action_rhat = float(split_rhat(actions)[0])
    action_tau = [integrated_autocorrelation_time(chain[:, 0]) for chain in actions]
    w11 = int(np.flatnonzero(np.all(reference_pairs == (1, 1), axis=1))[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((reference_pairs, mean, error, np.max(tau, axis=0), rhat))
    np.savetxt(
        output,
        table,
        header="R T mean hierarchical_bootstrap_se max_chain_tau_int split_rhat",
        fmt=["%d", "%d"] + ["%.17g"] * 4,
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        pairs=reference_pairs,
        measurement_trajectories=reference_steps[
            discard : discard + n_blocks * block_length
        ],
        block_means=blocked,
        bootstrap_means=bootstrap,
        mean=mean,
        covariance=covariance,
        tau_int=tau,
        split_rhat=rhat,
    )
    quarter_size = loops.shape[1] // 4
    quarters = loops[:, : 4 * quarter_size, w11].reshape(
        len(directories), 4, quarter_size
    ).mean(axis=2)
    output.with_suffix(".summary.txt").write_text(
        "\n".join(
            [
                f"chains {len(directories)}",
                f"available_vectors {loops.shape[1] + discard}",
                f"discarded_vectors {discard}",
                f"retained_vectors_per_chain {loops.shape[1]}",
                f"unused_tail_vectors {loops.shape[1] - n_blocks * block_length}",
                f"loops_per_vector {loops.shape[2]}",
                "production_acceptance " + " ".join(f"{x:.17g}" for x in acceptances),
                f"action_split_rhat {action_rhat:.17g}",
                "action_tau_int " + " ".join(f"{x:.17g}" for x in action_tau),
                f"W11_mean {mean[w11]:.17g}",
                f"W11_hierarchical_bootstrap_se {error[w11]:.17g}",
                f"W11_split_rhat {rhat[w11]:.17g}",
                "W11_quarters "
                + " ; ".join(
                    " ".join(f"{x:.17g}" for x in row) for row in quarters
                ),
                f"maximum_loop_split_rhat {np.max(rhat):.17g}",
                f"maximum_loop_tau_int {max_tau:.17g}",
                f"block_length {block_length}",
                f"blocks_per_chain {n_blocks}",
                f"blocking_adequate {str(block_length > max_tau and n_blocks >= 8).lower()}",
                f"bootstrap_samples {samples}",
                f"bootstrap_seed {seed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def self_test() -> None:
    rng = np.random.default_rng(240831)
    values = rng.normal(size=(4, 200, 2))
    assert np.max(split_rhat(values)) < 1.1
    values[0, :, 0] += 2.0
    assert split_rhat(values)[0] > 1.1
    blocks = rng.normal(size=(4, 10, 3))
    assert hierarchical_bootstrap(blocks, 20, 1).shape == (20, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discard-vectors", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.output is None:
        parser.error("root and --output are required")
    if args.discard_vectors < 0 or args.bootstrap_samples < 2:
        parser.error("discard must be non-negative and bootstrap samples must be >= 2")
    analyze(args.root, args.output, args.discard_vectors, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
