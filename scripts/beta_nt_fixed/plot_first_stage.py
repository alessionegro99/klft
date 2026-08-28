#!/usr/bin/env python3
"""Plot first-stage thermalization and blocking diagnostics per ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError("statanalysis-Bonn/plot_utils.py is required")
sys.path.insert(0, str(STATANALYSIS))
sys.path.insert(0, str(SCRIPTS))
import plot_utils as pu  # noqa: E402

from beta_nt_fixed.analysis import fused_replica_jackknife  # noqa: E402
from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402


FloatArray = NDArray[np.float64]
THERMALIZATION_BIN_SWEEPS = 100
MINIMUM_BLOCKS = 30
FALLBACK_BLOCK_LENGTHS = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    320,
    400,
    512,
    640,
    800,
    1000,
    1200,
    1400,
)


pu.setup_mpl()


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty first-stage manifest: {path}")
    return rows


def _diagnostics(result: dict[str, object]) -> dict[str, object]:
    nested = result.get("thermalization_and_blocking")
    return nested if isinstance(nested, dict) else result


def _chain_cut(chain: dict[str, object]) -> int:
    if "thermalization_cut_sweeps" in chain:
        return int(chain["thermalization_cut_sweeps"])
    if "thermalization_cut" in chain:
        return int(chain["thermalization_cut"])
    raise KeyError("chain has no thermalization cut")


def _load_summary(
    chain: dict[str, object], maximum_sweep: int | None
) -> tuple[FloatArray, FloatArray]:
    path = Path(str(chain["chain"])) / "holonomy_summary.out"
    table = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if table.shape[1] < 4:
        raise ValueError(f"{path}: expected step and Ctilde_0 columns")
    finite = np.isfinite(table[:, 0]) & np.isfinite(table[:, 3])
    if maximum_sweep is not None:
        finite &= table[:, 0] <= maximum_sweep
    selected = table[finite]
    if len(selected) < 2:
        raise ValueError(f"{path}: too few usable measurements")
    return selected[:, 0], selected[:, 3]


def binned_means(
    steps: FloatArray, values: FloatArray, width_sweeps: int
) -> tuple[FloatArray, FloatArray]:
    """Return non-overlapping presentation-bin centers and means."""
    if width_sweeps <= 0 or steps.shape != values.shape:
        raise ValueError("invalid thermalization binning input")
    groups = ((steps.astype(np.int64) - 1) // width_sweeps).astype(np.int64)
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    centers = np.bincount(inverse, weights=steps) / counts
    means = np.bincount(inverse, weights=values) / counts
    return centers, means


def _block_series(values: FloatArray, block_length: int) -> FloatArray:
    n_blocks = len(values) // block_length
    if n_blocks < 3:
        raise ValueError("too few complete blocks")
    complete = values[: n_blocks * block_length]
    return np.mean(complete.reshape(n_blocks, block_length), axis=1)[:, None]


def fallback_blocking_curve(
    values_by_chain: list[FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a chain-stratified delete-one-block error curve.

    This uses the same replica-stratified jackknife implementation as the
    campaign gate. It is needed only for the older ``Nt=1`` PCM references,
    whose result schema predates the full whole-history plateau record.
    """
    lengths = np.asarray(
        [
            length
            for length in FALLBACK_BLOCK_LENGTHS
            if min(len(values) // length for values in values_by_chain)
            >= MINIMUM_BLOCKS
        ],
        dtype=np.float64,
    )
    if len(lengths) < 3:
        raise ValueError("too few block lengths for first-stage plot")
    errors = np.empty(len(lengths), dtype=np.float64)
    uncertainties = np.empty(len(lengths), dtype=np.float64)
    identity: Callable[[FloatArray], FloatArray] = lambda value: value
    for index, length_float in enumerate(lengths):
        length = int(length_float)
        blocks = [_block_series(values, length) for values in values_by_chain]
        _, error, degrees_of_freedom = fused_replica_jackknife(blocks, identity)
        errors[index] = float(error[0])
        uncertainties[index] = float(
            error[0] / np.sqrt(2.0 * degrees_of_freedom[0])
        )
    return lengths, errors, uncertainties


def _title(row: dict[str, str], subject: str) -> str:
    status = row["status"].replace("_", " ")
    return (
        rf"{subject}, $\beta/N_t={row['beta_over_nt']},\ "
        rf"N_t={int(row['physical_Nt'])},\ N_s={int(row['Ns'])}$ "
        rf"({status})"
    )


def plot_thermalization(
    row: dict[str, str],
    chains: list[dict[str, object]],
    histories: list[tuple[FloatArray, FloatArray]],
    output: Path,
) -> None:
    """Plot the zero-momentum holonomy structure-factor histories."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for index, (chain, history) in enumerate(zip(chains, histories, strict=True)):
        palette = index + 1
        centers, means = binned_means(
            history[0], history[1], THERMALIZATION_BIN_SWEEPS
        )
        cut = _chain_cut(chain)
        axis.plot(
            centers,
            means,
            label=rf"chain {index + 1}, cut ${cut}$",
            **pu.fit(palette, linestyle="-", linewidth=1.2),
        )
        axis.axvline(cut, **pu.fit(palette, linestyle="--", linewidth=1.0))
    pu.style_axes(
        axis,
        xlabel=r"composite sweep",
        ylabel=r"100-sweep mean of $\widetilde C_G(0)$",
        title=_title(row, "Thermalization history"),
        legend=True,
    )
    save_figure(figure, output / "thermalization.pdf")


def plot_blocking_error(
    row: dict[str, str],
    diagnostics: dict[str, object],
    chains: list[dict[str, object]],
    histories: list[tuple[FloatArray, FloatArray]],
    output: Path,
) -> str:
    """Plot the chain-aware blocking error of ``Ctilde_0``."""
    plateau = diagnostics.get("blocking_error_plateau")
    selected_group = None
    selected_length = None
    if isinstance(plateau, dict):
        record = plateau["observables"]["Ctilde_0"]
        lengths = np.asarray(
            plateau["candidate_block_lengths_sweeps"], dtype=np.float64
        )
        errors = np.asarray(record["error"], dtype=np.float64)
        uncertainties = np.asarray(
            record["error_uncertainty"], dtype=np.float64
        )
        minimum_tau = float(plateau["minimum_tau_block_length_sweeps"])
        selected_group = plateau.get("common_plateau_group_sweeps")
        selected_length = plateau.get("selected_common_block_length_sweeps")
        method = "recorded full-observable gate; Ctilde_0 component shown"
    else:
        post_cut = [
            values[steps > _chain_cut(chain)]
            for chain, (steps, values) in zip(chains, histories, strict=True)
        ]
        lengths, errors, uncertainties = fallback_blocking_curve(post_cut)
        maximum_tau = max(
            max(float(value) for value in chain["tau_int"].values())
            for chain in chains
        )
        minimum_tau = 2.0 * maximum_tau
        selected_length = max(int(chain["block_size"]) for chain in chains)
        method = "PCM reference fallback: chain-stratified jackknife"

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axvline(
        minimum_tau,
        label=rf"autocorrelation bound: ${minimum_tau:g}$ sweeps",
        **pu.fit(0, linestyle="--"),
    )
    if selected_group is not None:
        axis.axvspan(
            float(selected_group[0]),
            float(selected_group[-1]),
            label="selected common plateau",
            **pu.conf_band(2),
        )
    elif selected_length is not None:
        axis.axvline(
            float(selected_length),
            label=rf"selected block: ${int(selected_length)}$ sweeps",
            **pu.fit(2, linestyle=":"),
        )
    axis.errorbar(
        lengths,
        errors,
        uncertainties,
        label=r"$\widetilde C_G(0)$",
        **pu.data(1, linestyle="-", markersize=8),
    )
    axis.set_xscale("log")
    pu.style_axes(
        axis,
        xlabel=r"block length (sweeps)",
        ylabel=r"blocking error of $\widetilde C_G(0)$",
        title=_title(row, "Blocking-error history"),
        legend=True,
    )
    save_figure(figure, output / "blocking_error.pdf")
    return method


def plot_row(row: dict[str, str], output_root: Path) -> None:
    """Write the two first-stage plots and their compact provenance."""
    result_path = Path(row["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    diagnostics = _diagnostics(result)
    chains = list(diagnostics["chains"])
    if len(chains) != 4:
        raise ValueError(f"{result_path}: expected four independent chains")
    maximum = result.get("maximum_sweep_inclusive")
    if maximum is None:
        maximum = diagnostics.get("maximum_sweep_inclusive")
    maximum_sweep = int(maximum) if maximum is not None else None
    histories = [_load_summary(chain, maximum_sweep) for chain in chains]
    output = output_root / row["output_relative_path"]
    output.mkdir(parents=True, exist_ok=True)
    plot_thermalization(row, chains, histories, output)
    blocking_method = plot_blocking_error(
        row, diagnostics, chains, histories, output
    )
    metadata = {
        "beta_over_nt": float(row["beta_over_nt"]),
        "physical_Nt": int(row["physical_Nt"]),
        "Ns": int(row["Ns"]),
        "beta": float(row["beta"]),
        "role": row["role"],
        "status": row["status"],
        "result_path": str(result_path),
        "maximum_sweep_inclusive": maximum_sweep,
        "raw_chain_directories": [str(chain["chain"]) for chain in chains],
        "thermalization_bin_sweeps": THERMALIZATION_BIN_SWEEPS,
        "blocking_observable": "Ctilde_0",
        "blocking_method": blocking_method,
        "bootstrap_operations": 0,
        "plot_files": ["thermalization.pdf", "blocking_error.pdf"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for row in _load_manifest(args.manifest):
        plot_row(row, args.output_root)


if __name__ == "__main__":
    main()
