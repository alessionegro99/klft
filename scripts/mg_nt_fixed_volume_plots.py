#!/usr/bin/env python3
"""Plot the final fixed-mass Nt=4 finite-volume checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "statanalysis-Bonn"))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402
from mg_nt_fixed_plot import plot_brackets, plot_tuned_beta_over_nt  # noqa: E402


pu.setup_mpl()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_correlators(
    smaller: dict[str, object], larger: dict[str, object], output: Path
) -> None:
    """Compare common separations in two periodic spatial boxes."""
    r = np.arange(1, int(smaller["Ns"]) // 2 + 1)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, result in enumerate((smaller, larger), start=1):
        axis.errorbar(
            r,
            np.asarray(result["C_G"])[r],
            np.asarray(result["C_G_error"])[r],
            label=rf"$N_s={int(result['Ns'])}$",
            **pu.data(palette, linestyle="-"),
        )
    axis.set_yscale("log")
    pu.style_axes(
        axis,
        xlabel=r"$R/a$",
        ylabel=r"$C_G(R)$",
        title=(
            rf"Periodic-image check, $N_t=4$, "
            rf"$\beta={float(smaller['beta']):g}$"
        ),
        legend=True,
    )
    beta = str(smaller["beta"]).replace(".", "p")
    save_figure(figure, output / f"periodic_correlator_volume_beta_{beta}.pdf")


def plot_masses(pairs: list[tuple[dict[str, object], dict[str, object]]], output: Path) -> None:
    """Show the extracted asymptotic mass at both volumes."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, (smaller, larger) in enumerate(pairs, start=1):
        axis.errorbar(
            [smaller["Ns"], larger["Ns"]],
            [smaller["mass"], larger["mass"]],
            [smaller["mass_error"], larger["mass_error"]],
            label=rf"$\beta={float(smaller['beta']):g}$",
            **pu.data(palette, linestyle="-"),
        )
    pu.style_axes(
        axis,
        xlabel=r"$N_s$",
        ylabel=r"$a m_G$",
        title=r"Asymptotic-mass finite-volume check, $N_t=4$",
        legend=True,
    )
    save_figure(figure, output / "mass_volume_stability_Nt_4.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--replacement-results", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tuning = _load(args.tuning_summary)
    plot_brackets(tuning, args.output, preliminary=True)
    plot_tuned_beta_over_nt(tuning, args.output, preliminary=True)
    pairs = []
    for beta in ("4p95", "5p30"):
        smaller = _load(args.pilot_results / f"Nt_4_Ns_48_b_{beta}/analysis.json")
        larger = _load(
            args.replacement_results / f"Nt_4_Ns_96_b_{beta}/analysis.json"
        )
        if int(smaller["Ns"]) != 48 or int(larger["Ns"]) != 96:
            raise ValueError("expected the preregistered Ns=48 and Ns=96 pair")
        pairs.append((smaller, larger))
        plot_correlators(smaller, larger, args.output)
    plot_masses(pairs, args.output)


if __name__ == "__main__":
    main()
