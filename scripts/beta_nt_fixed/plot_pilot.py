#!/usr/bin/env python3
"""Plot compact fixed-beta/Nt pilot products after qbig analysis."""

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
CAMPAIGN_SCRIPTS = Path(__file__).resolve().parent
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError("statanalysis-Bonn/plot_utils.py is required")
sys.path.insert(0, str(STATANALYSIS))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(CAMPAIGN_SCRIPTS))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402
from pilot_summary import load_results  # noqa: E402


pu.setup_mpl()


def xi_interval_error(result: dict[str, object]) -> np.ndarray:
    """Return asymmetric 68% errors around the central ``xi_2nd``."""
    xi = float(result["xi_2nd"])
    interval = np.asarray(
        result["xi_2nd_68_percent_interval"], dtype=np.float64
    )
    return np.asarray([[xi - interval[0]], [interval[1] - xi]])


def ensemble_label(result: dict[str, object]) -> str:
    """Return a concise physics label for one ensemble."""
    return (
        rf"$N_t={int(result['physical_Nt'])},\ N_s={int(result['Ns'])},\ "
        rf"\beta={float(result['beta']):g}$"
    )


def plot_nt2_lengths(
    results: list[tuple[Path, dict[str, object]]], output: Path
) -> None:
    """Plot the fixed-thickness coupling scan and finite-volume duplicate."""
    selected = [result for _, result in results if int(result["physical_Nt"]) == 2]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, result in enumerate(selected, start=1):
        axis.errorbar(
            [float(result["beta"])],
            [float(result["xi_2nd"])],
            xi_interval_error(result),
            label=rf"$N_s={int(result['Ns'])}$",
            **pu.data(palette),
        )
    axis.axhline(5.0, **pu.fit(0, linestyle="dashed"))
    pu.style_axes(
        axis,
        xlabel=r"$\beta$",
        ylabel=r"$\xi_{\mathrm{2nd}}/a$",
        title=r"Fixed-$N_t=2$ pilot scan",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "xi_2nd_vs_beta_Nt_2.pdf")


def plot_fixed_beta(
    results: list[tuple[Path, dict[str, object]]], output: Path
) -> None:
    """Plot the beta=2.1 thickness sequence in slab units."""
    selected = sorted(
        (
            result
            for _, result in results
            if np.isclose(float(result["beta"]), 2.1)
        ),
        key=lambda result: int(result["physical_Nt"]),
    )
    nt = np.asarray([result["physical_Nt"] for result in selected], dtype=np.float64)
    xi = np.asarray([result["xi_2nd"] for result in selected], dtype=np.float64)
    interval = np.asarray(
        [result["xi_2nd_68_percent_interval"] for result in selected],
        dtype=np.float64,
    )
    error = np.vstack((xi - interval[:, 0], interval[:, 1] - xi))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.errorbar(
        nt, xi / nt, error / nt[None, :], **pu.data(1, linestyle="-")
    )
    pu.style_axes(
        axis,
        xlabel=r"$N_t$",
        ylabel=r"$\xi_{\mathrm{2nd}}/N_t$",
        title=r"Fixed $\beta=2.1$",
        grid=True,
    )
    save_figure(figure, output / "fixed_beta_xi_2nd_over_Nt.pdf")


def plot_fixed_beta_over_nt(
    summary: dict[str, object], output: Path, key: str, filename: str
) -> None:
    """Plot one ``q0+c/Nt`` scaling fit."""
    fit = summary["constant_lambda_continuum_scaling"][key]
    if fit is None:
        return
    nt = np.asarray(fit["Nt"], dtype=np.float64)
    inverse_nt = 1.0 / nt
    quantity = np.asarray(fit["q"], dtype=np.float64)
    error = np.asarray(fit["q_error"], dtype=np.float64)
    x_line = np.linspace(0.0, float(1.05 * np.max(inverse_nt)), 200)
    y_line = float(fit["q0"]) + float(fit["c"]) * x_line
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(x_line, y_line, **pu.fit(1))
    axis.errorbar(inverse_nt, quantity, error, **pu.data(1))
    ylabel = (
        r"$\xi_{\mathrm{2nd}}/N_t$"
        if key == "xi_2nd_over_Nt"
        else r"$\xi_{\exp}/N_t$"
    )
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=ylabel,
        title=r"Fixed $\beta/N_t=1.8$ ($g_3^2L=20/9$)",
        grid=True,
    )
    save_figure(figure, output / filename)


def plot_anchor_comparison(
    results: list[tuple[Path, dict[str, object]]], output: Path
) -> None:
    """Plot the two beta=3.6 full correlators used in the volume test."""
    selected = [
        result
        for _, result in results
        if result["purpose"] in {"constant_lambda_anchor", "anchor_finite_volume"}
    ]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette, result in enumerate(selected, start=1):
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
        title=r"Finite-volume check, $N_t=2$, $\beta=3.6$",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "anchor_finite_volume_correlator.pdf")


def plot_profiles(
    results: list[tuple[Path, dict[str, object]]], output: Path
) -> None:
    """Write one spatial/temporal profile plot per ensemble."""
    for _, result in results:
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
            title=ensemble_label(result),
            legend=True,
            grid=True,
        )
        tag = (
            f"Nt_{int(result['physical_Nt'])}_Ns_{int(result['Ns'])}_"
            f"b_{str(result['beta']).replace('.', 'p')}"
        )
        save_figure(figure, output / f"plaquette_profile_{tag}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    summary = json.loads(
        (args.results / "campaign_summary.json").read_text(encoding="utf-8")
    )
    plot_nt2_lengths(results, args.output)
    plot_fixed_beta(results, args.output)
    plot_fixed_beta_over_nt(
        summary,
        args.output,
        "xi_2nd_over_Nt",
        "fixed_beta_over_nt_xi_2nd_over_Nt.pdf",
    )
    plot_fixed_beta_over_nt(
        summary,
        args.output,
        "xi_exp_over_Nt",
        "fixed_beta_over_nt_xi_exp_over_Nt.pdf",
    )
    plot_anchor_comparison(results, args.output)
    plot_profiles(results, args.output)


if __name__ == "__main__":
    main()
