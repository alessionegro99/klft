#!/usr/bin/env python3
"""Plot selected blocking-error curves from one diagnostics result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError("statanalysis-Bonn/plot_utils.py is required")
sys.path.insert(0, str(STATANALYSIS))
sys.path.insert(0, str(SCRIPTS))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402


pu.setup_mpl()
OBSERVABLES = {
    "Tr_G_real": r"$\mathrm{Re}\,\mathrm{Tr}\,G$",
    "Ctilde_0": r"$\widetilde C_G(0)$",
    "Ctilde_pmin": r"$\widetilde C_G(p_{\min})$",
    "C_G_R1": r"$C_G(1)$",
}


def plot_diagnostics(
    diagnostics: dict[str, object],
    output: Path,
    ensemble_label: str,
    title_prefix: str,
) -> None:
    """Write one blocking-error plot for each selected primary observable."""
    blocking = diagnostics["blocking_error_plateau"]
    block_lengths = np.asarray(
        blocking["candidate_block_lengths_sweeps"], dtype=np.float64
    )
    minimum_tau = float(blocking["minimum_tau_block_length_sweeps"])
    for name, label in OBSERVABLES.items():
        record = blocking["observables"][name]
        error = np.asarray(record["error"], dtype=np.float64)
        uncertainty = np.asarray(record["error_uncertainty"], dtype=np.float64)
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.axvline(
            minimum_tau,
            label=rf"autocorrelation bound: ${minimum_tau:g}$ sweeps",
            **pu.fit(0, linestyle="--"),
        )
        axis.errorbar(
            block_lengths,
            error,
            uncertainty,
            label=(
                "individual plateau found"
                if bool(record["plateau_found"])
                else "no individual plateau"
            ),
            **pu.data(1, linestyle="-"),
        )
        axis.set_xscale("log")
        pu.style_axes(
            axis,
            xlabel=r"block length (sweeps)",
            ylabel=rf"blocking error of {label}",
            title=rf"{title_prefix}: {ensemble_label}",
            legend=True,
        )
        save_figure(figure, output / f"blocking_error_{name}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble-label", required=True)
    parser.add_argument(
        "--title-prefix",
        default="Preliminary rejected blocking audit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    plot_diagnostics(
        diagnostics,
        args.output,
        args.ensemble_label,
        args.title_prefix,
    )


if __name__ == "__main__":
    main()
