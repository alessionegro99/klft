#!/usr/bin/env python3
"""Autocorrelation-aware statistics for independent Wilson-loop chains.

Autocorrelation times use the automatic window of U. Wolff,
Comput. Phys. Commun. 156 (2004) 143, hep-lat/0306017. Split R-hat follows
A. Vehtari et al., Bayesian Anal. 16 (2021) 667, doi:10.1214/20-BA1221.
"""

from __future__ import annotations

import argparse
import tempfile
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


def load_heatbath_monitor(path: Path, steps: IntArray) -> FloatArray:
    """Load the plaquette column aligned with heatbath loop measurements."""
    rows = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[1] != 5 or not np.all(np.isfinite(rows)):
        raise ValueError(f"invalid heatbath plaquette data in {path}: {rows.shape}")
    if not np.array_equal(rows[:, 0].astype(np.int64), steps):
        raise ValueError(f"plaquettes and Wilson loops are misaligned in {path.parent}")
    return rows[:, 1]


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


def analyze(
    root: Path,
    output: Path,
    discard: int,
    samples: int,
    seed: int,
    heatbath: bool,
) -> None:
    """Analyze all ``chain*`` directories below *root*."""
    directories = sorted(path for path in root.glob("chain*") if path.is_dir())
    if len(directories) < 2:
        raise ValueError("at least two chain directories are required")

    loop_series = []
    monitor_series = []
    acceptances = []
    reference_steps: IntArray | None = None
    reference_pairs: IntArray | None = None
    for directory in directories:
        steps, pairs, loops = load_loops(directory / "wilson.out")
        if heatbath:
            monitor = load_heatbath_monitor(directory / "plaquette.out", steps)
        else:
            hmc = np.loadtxt(
                directory / "hmc.out", comments="#", dtype=np.float64, ndmin=2
            )
            diagnostic = np.loadtxt(
                directory / "diagnostic.out",
                comments="#",
                dtype=np.float64,
                ndmin=2,
            )
            if (
                hmc.shape[1] != 7
                or diagnostic.shape[1] != 4
                or not np.all(np.isfinite(hmc))
                or not np.all(np.isfinite(diagnostic))
            ):
                raise ValueError(f"invalid HMC or diagnostic data in {directory}")
            production_diagnostic = diagnostic[diagnostic[:, 1] > 0]
            if not np.array_equal(
                production_diagnostic[:, 0].astype(np.int64), steps
            ):
                raise ValueError(
                    f"diagnostics and Wilson loops are misaligned in {directory}"
                )
            monitor = production_diagnostic[:, 2]
            acceptances.append(float(np.mean(hmc[hmc[:, 1] > 0, 2])))
        if reference_steps is not None and not np.array_equal(steps, reference_steps):
            raise ValueError("chain measurement trajectories differ")
        if reference_pairs is not None and not np.array_equal(pairs, reference_pairs):
            raise ValueError("chain Wilson-loop grids differ")
        reference_steps, reference_pairs = steps, pairs
        loop_series.append(loops)
        monitor_series.append(monitor)

    loops = np.asarray(loop_series)[:, discard:]
    monitors = np.asarray(monitor_series)[:, discard:, None]
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
    monitor_rhat = float(split_rhat(monitors)[0])
    monitor_tau = [
        integrated_autocorrelation_time(chain[:, 0]) for chain in monitors
    ]
    monitor_name = "plaquette" if heatbath else "action"
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
                f"driver {'heatbath' if heatbath else 'orbifold_hmc'}",
                "production_acceptance "
                + ("n/a" if heatbath else " ".join(f"{x:.17g}" for x in acceptances)),
                f"{monitor_name}_split_rhat {monitor_rhat:.17g}",
                f"{monitor_name}_tau_int "
                + " ".join(f"{x:.17g}" for x in monitor_tau),
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
    rows = np.column_stack(
        (np.array([10, 20]), rng.normal(size=(2, 3)), np.array([0.1, 0.2]))
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "plaquette.out"
        np.savetxt(path, rows)
        assert np.array_equal(
            load_heatbath_monitor(path, np.array([10, 20])), rows[:, 1]
        )
        steps = 10 * np.arange(1, 201)
        pairs = np.array([(1, 1), (1, 2)])
        for chain_index in range(2):
            chain = root / f"chain{chain_index + 1:02d}"
            chain.mkdir()
            loops = rng.normal((0.6, 0.4), 0.01, size=(len(steps), 2))
            np.savetxt(
                chain / "wilson.out",
                np.column_stack(
                    (
                        np.repeat(steps, 2),
                        np.tile(pairs, (len(steps), 1)),
                        loops.ravel(),
                    )
                ),
            )
            np.savetxt(
                chain / "plaquette.out",
                np.column_stack((steps, rng.normal(0.6, 0.01, (len(steps), 4)))),
            )
        output = root / "stats.tsv"
        analyze(root, output, 0, 20, 1, True)
        assert output.is_file() and output.with_suffix(".npz").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discard-vectors", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--heatbath",
        action="store_true",
        help="read plaquette.out plus wilson.out instead of orbifold HMC files",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.output is None:
        parser.error("root and --output are required")
    if args.discard_vectors < 0 or args.bootstrap_samples < 2:
        parser.error("discard must be non-negative and bootstrap samples must be >= 2")
    analyze(
        args.root,
        args.output,
        args.discard_vectors,
        args.bootstrap_samples,
        args.seed,
        args.heatbath,
    )


if __name__ == "__main__":
    main()
