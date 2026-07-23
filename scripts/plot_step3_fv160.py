#!/usr/bin/env python3
"""Plot compact products from the Step-3 ``Ns=160`` volume check."""

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
    """Load one compact ensemble result."""
    return json.loads(path.read_text(encoding="utf-8"))


def plot_correlators(
    results: list[dict[str, object]], output: Path
) -> None:
    """Plot the three beta=3.6 correlators used in volume checks."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, result in enumerate(results, start=1):
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
        title=r"Finite-volume sequence, $N_t=2$, $\beta=3.6$",
        legend=True,
        grid=True,
    )
    save_figure(
        figure, output / "finite_volume_correlator_Ns_96_128_160.pdf"
    )


def plot_lengths(results: list[dict[str, object]], output: Path) -> None:
    """Plot second-moment length versus spatial volume."""
    ns = np.asarray([result["Ns"] for result in results], dtype=np.float64)
    xi = np.asarray([result["xi_2nd"] for result in results], dtype=np.float64)
    interval = np.asarray(
        [result["xi_2nd_68_percent_interval"] for result in results],
        dtype=np.float64,
    )
    error = np.vstack((xi - interval[:, 0], interval[:, 1] - xi))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.errorbar(ns, xi, error, **pu.data(1, linestyle="-"))
    pu.style_axes(
        axis,
        xlabel=r"$N_s$",
        ylabel=r"$\xi_{\mathrm{2nd}}/a$",
        title=r"Finite-volume sequence, $N_t=2$, $\beta=3.6$",
        grid=True,
    )
    save_figure(figure, output / "finite_volume_xi_2nd.pdf")


def plot_profile(result: dict[str, object], output: Path) -> None:
    """Plot the spatial and temporal plaquette profile at ``Ns=160``."""
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
    pu.style_axes(
        axis,
        xlabel=r"$t/a$",
        ylabel=r"$\frac{1}{2}\,\mathrm{Re}\,\mathrm{Tr}\,U_p$",
        title=r"$N_t=2,\ N_s=160,\ \beta=3.6$",
        legend=True,
        grid=True,
    )
    save_figure(
        figure, output / "plaquette_profile_Nt_2_Ns_160_b_3p6.pdf"
    )


def plot_revised_scaling(summary: dict[str, object], output: Path) -> None:
    """Plot the volume-checked constant-coupling scaling fit when available."""
    scaling = summary["revised_constant_lambda_continuum_scaling"]
    if scaling is None:
        return
    fit = scaling["xi_2nd_over_Nt"]
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
        title=r"Constant $\lambda^2_{\rm tree}=20/9$, volume checked",
        grid=True,
    )
    save_figure(
        figure, output / "constant_lambda_xi_2nd_over_Nt_fv_checked.pdf"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3-results", type=Path, required=True)
    parser.add_argument("--extension-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        args.step3_results / "Nt_2_Ns_96_b_3p6" / "analysis.json",
        args.step3_results / "Nt_2_Ns_128_b_3p6" / "analysis.json",
        args.extension_results / "Nt_2_Ns_160_b_3p6" / "analysis.json",
    ]
    results = [load_result(path) for path in paths]
    summary = load_result(args.extension_results / "finite_volume_summary.json")
    plot_correlators(results, args.output)
    plot_lengths(results, args.output)
    plot_profile(results[-1], args.output)
    plot_revised_scaling(summary, args.output)


if __name__ == "__main__":
    main()
