#!/usr/bin/env python3
"""Plot the corrected ``Nt=3,4`` Step-3 volume analyses."""

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
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError("statanalysis-Bonn/plot_utils.py is required")
sys.path.insert(0, str(STATANALYSIS))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402


pu.setup_mpl()


def load_result(path: Path) -> dict[str, object]:
    """Load one JSON result."""
    return json.loads(path.read_text(encoding="utf-8"))


def plot_volume_correlators(
    pilot: dict[str, object], corrected: dict[str, object], output: Path
) -> None:
    """Compare the full correlator before and after increasing ``Ns``."""
    nt = int(corrected["physical_Nt"])
    beta = float(corrected["beta"])
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, result in enumerate((pilot, corrected), start=1):
        axis.errorbar(
            np.asarray(result["R"], dtype=np.float64),
            np.asarray(result["C_G"], dtype=np.float64),
            np.asarray(result["C_G_error"], dtype=np.float64),
            label=rf"$N_s={int(result['Ns'])}$",
            **pu.data(palette, linestyle="-"),
        )
    axis.set_yscale("symlog", linthresh=1.0e-5)
    pu.style_axes(
        axis,
        xlabel=r"$R/a$",
        ylabel=r"$C_G(R)$",
        title=rf"Volume comparison, $N_t={nt}$, $\beta={beta:g}$",
        legend=True,
        grid=True,
    )
    save_figure(
        figure,
        output
        / (
            f"finite_volume_correlator_Nt_{nt}_Ns_"
            f"{int(pilot['Ns'])}_{int(corrected['Ns'])}.pdf"
        ),
    )


def plot_profile(result: dict[str, object], output: Path) -> None:
    """Plot the spatial and temporal plaquette profile."""
    profile = result["plaquette_profile"]
    keys = np.asarray(profile["kind_t"], dtype=np.float64)
    mean = np.asarray(profile["mean"], dtype=np.float64)
    error = np.asarray(profile["error"], dtype=np.float64)
    spatial = keys[:, 0] == 0.0
    temporal = keys[:, 0] == 1.0
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.errorbar(
        keys[spatial, 1],
        mean[spatial],
        error[spatial],
        label=r"$P_s(t)$",
        **pu.data(1, linestyle="-"),
    )
    axis.errorbar(
        keys[temporal, 1],
        mean[temporal],
        error[temporal],
        label=r"$P_t(t+1/2)$",
        **pu.data(2, linestyle="-"),
    )
    nt = int(result["physical_Nt"])
    ns = int(result["Ns"])
    beta = float(result["beta"])
    pu.style_axes(
        axis,
        xlabel=r"$t/a$",
        ylabel=r"$\frac{1}{2}\,\mathrm{Re}\,\mathrm{Tr}\,U_p$",
        title=rf"$N_t={nt},\ N_s={ns},\ \beta={beta:g}$",
        legend=True,
        grid=True,
    )
    save_figure(
        figure,
        output / f"plaquette_profile_Nt_{nt}_Ns_{ns}.pdf",
    )


def plot_scaling(summary: dict[str, object], output: Path) -> None:
    """Plot the corrected constant-coupling scaling fit."""
    fit = summary["corrected_constant_lambda_scaling"]["xi_2nd_over_Nt"]
    nt = np.asarray(fit["Nt"], dtype=np.float64)
    inverse_nt = 1.0 / nt
    quantity = np.asarray(fit["q"], dtype=np.float64)
    error = np.asarray(fit["q_error"], dtype=np.float64)
    x_line = np.linspace(0.0, 1.05 * np.max(inverse_nt), 200)
    y_line = float(fit["q0"]) + float(fit["c"]) * x_line
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(x_line, y_line, **pu.fit(1))
    axis.errorbar(inverse_nt, quantity, error, **pu.data(1))
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=r"$\xi_{\mathrm{2nd}}/N_t$",
        title=r"Constant $\lambda^2_{\rm tree}=20/9$, corrected volumes",
        grid=True,
    )
    save_figure(
        figure, output / "constant_lambda_xi_2nd_over_Nt_corrected.pdf"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3-results", type=Path, required=True)
    parser.add_argument("--corrected-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pilot_paths = [
        args.step3_results / "Nt_3_Ns_144_b_5p4" / "analysis.json",
        args.step3_results / "Nt_4_Ns_192_b_7p2" / "analysis.json",
    ]
    corrected_paths = [
        args.corrected_results / "Nt_3_Ns_176_b_5p4" / "analysis.json",
        args.corrected_results / "Nt_4_Ns_256_b_7p2" / "analysis.json",
    ]
    pilot = [load_result(path) for path in pilot_paths]
    corrected = [load_result(path) for path in corrected_paths]
    for old, new in zip(pilot, corrected, strict=True):
        plot_volume_correlators(old, new, args.output)
        plot_profile(new, args.output)
    summary = load_result(
        args.corrected_results / "corrected_volume_summary.json"
    )
    plot_scaling(summary, args.output)


if __name__ == "__main__":
    main()
