#!/usr/bin/env python3
"""Plot fixed-``g_3^2L`` masses and scaling observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import k0


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
LINE_PALETTES = {"1.8": 1, "2.0": 2, "2.1": 3}


def _tag(value: str) -> str:
    return value.replace(".", "p")


def _line_label(line: str, g3_squared_l: float) -> str:
    return rf"$\beta/N_t={line}$ ($g_3^2L={g3_squared_l:.3f}$)"


def _title(text: str, preliminary: bool) -> str:
    """Prefix a figure title when it uses an incomplete campaign selection."""
    return f"Preliminary: {text}" if preliminary else text


def _axis_value(point: dict[str, object], variable: str) -> float:
    nt = float(point["physical_Nt"])
    if variable == "inverse_nt":
        return 1.0 / nt
    if variable == "inverse_nt_squared":
        return 1.0 / nt**2
    if variable == "inverse_beta":
        return 1.0 / float(point["beta"])
    raise ValueError(f"unknown scaling variable {variable}")


def _line_x(nt: np.ndarray, line: str, variable: str) -> np.ndarray:
    if variable == "inverse_nt":
        return 1.0 / nt
    if variable == "inverse_nt_squared":
        return 1.0 / nt**2
    if variable == "inverse_beta":
        return 1.0 / (float(line) * nt)
    raise ValueError(f"unknown scaling variable {variable}")


def plot_mass_scaling(
    summary: dict[str, object], output: Path, variable: str, preliminary: bool
) -> None:
    """Plot ``m_G Nt`` and the retained linear-in-``1/Nt`` fit."""
    labels = {
        "inverse_nt": r"$1/N_t$",
        "inverse_nt_squared": r"$1/N_t^2$",
        "inverse_beta": r"$1/\beta$",
    }
    names = {
        "inverse_nt": "mass_Nt_vs_inverse_Nt.pdf",
        "inverse_nt_squared": "mass_Nt_vs_inverse_Nt_squared.pdf",
        "inverse_beta": "mass_Nt_vs_inverse_beta.pdf",
    }
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    fit_legend_added = False
    for line, line_result in summary["lines"].items():
        palette = LINE_PALETTES.get(line, 1)
        points = line_result["points"]
        continuum = [point for point in points if point["role"] == "continuum"]
        reference = [point for point in points if point["role"] == "pcm_reference"]
        if continuum:
            axis.errorbar(
                [_axis_value(point, variable) for point in continuum],
                [point["mass_Nt"] for point in continuum],
                [point["mass_Nt_error"] for point in continuum],
                label=_line_label(line, float(line_result["g3_squared_L"])),
                **pu.data(palette),
            )
        if reference:
            axis.errorbar(
                [_axis_value(point, variable) for point in reference],
                [point["mass_Nt"] for point in reference],
                [point["mass_Nt_error"] for point in reference],
                label=rf"exact PCM, $\beta={float(reference[0]['beta']):g}$",
                **pu.results(palette),
            )
        fit = line_result["mass_Nt_p1_fit"]
        if fit is not None and continuum:
            minimum_nt = min(float(point["physical_Nt"]) for point in continuum)
            nt_curve = np.linspace(minimum_nt, 1000.0, 500)
            y_curve = float(fit["continuum"]) + float(
                fit["coefficient"]
            ) / nt_curve
            x_curve = _line_x(nt_curve, line, variable)
            order = np.argsort(x_curve)
            axis.plot(
                x_curve[order],
                y_curve[order],
                label=(
                    r"$p=1$ fit"
                    + (" (rejected)" if float(fit["p_value"]) < 0.05 else "")
                    if not fit_legend_added
                    else None
                ),
                **pu.fit(palette, linestyle="-"),
            )
            fit_legend_added = True
    pu.style_axes(
        axis,
        xlabel=labels[variable],
        ylabel=r"$m_GN_t$",
        title=_title(r"Continuum scaling at fixed $g_3^2L$", preliminary),
        legend=True,
    )
    save_figure(figure, output / names[variable])


def plot_continuum_mass(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot the extrapolated slab mass versus physical plate separation."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    available = [
        line_result
        for line_result in summary["lines"].values()
        if line_result["mass_Nt_p1_fit"] is not None
    ]
    if available:
        axis.errorbar(
            [line["g3_squared_L"] for line in available],
            [line["mass_Nt_p1_fit"]["continuum"] for line in available],
            [
                line["mass_Nt_p1_fit"]["continuum_error"]
                for line in available
            ],
            label=r"$p=1$",
            **pu.data(1),
        )
    pu.style_axes(
        axis,
        xlabel=r"$g_3^2L$",
        ylabel=r"$(m_GL)_{\mathrm{cont}}$",
        title=_title(
            r"Continuum holonomy mass versus plate separation", preliminary
        ),
        legend=True,
    )
    save_figure(figure, output / "continuum_mass_vs_g3_squared_L.pdf")


def plot_length_ratio(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot ``xi_exp/xi_2nd`` versus the lattice spacing proxy."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    fit_legend_added = False
    for line, line_result in summary["lines"].items():
        palette = LINE_PALETTES.get(line, 1)
        points = [
            point
            for point in line_result["points"]
            if point["xi_exp_over_xi_2nd_error"] is not None
        ]
        if not points:
            continue
        continuum = [point for point in points if point["role"] == "continuum"]
        reference = [
            point for point in points if point["role"] == "pcm_reference"
        ]
        if continuum:
            axis.errorbar(
                [1.0 / float(point["physical_Nt"]) for point in continuum],
                [point["xi_exp_over_xi_2nd"] for point in continuum],
                [point["xi_exp_over_xi_2nd_error"] for point in continuum],
                label=_line_label(line, float(line_result["g3_squared_L"])),
                **pu.data(palette),
            )
        if reference:
            axis.errorbar(
                [1.0 / float(point["physical_Nt"]) for point in reference],
                [point["xi_exp_over_xi_2nd"] for point in reference],
                [point["xi_exp_over_xi_2nd_error"] for point in reference],
                label=rf"exact PCM, $\beta={float(reference[0]['beta']):g}$",
                **pu.results(palette),
            )
        fit = line_result["xi_exp_over_xi_2nd_p1_fit"]
        if fit is not None and continuum:
            minimum_nt = min(float(point["physical_Nt"]) for point in continuum)
            nt_curve = np.linspace(minimum_nt, 1000.0, 500)
            axis.plot(
                1.0 / nt_curve,
                float(fit["continuum"])
                + float(fit["coefficient"]) / nt_curve,
                label=r"$p=1$ fit" if not fit_legend_added else None,
                **pu.fit(palette, linestyle="-"),
            )
            fit_legend_added = True
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=r"$\xi_{\exp}/\xi_{\mathrm{2nd}}$",
        title=_title(r"Exponential-to-second-moment length ratio", preliminary),
        legend=True,
    )
    save_figure(figure, output / "xi_exp_over_xi_2nd_vs_inverse_Nt.pdf")


def plot_spatial_volume(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot the selected spatial sizes in second-moment correlation lengths."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axhline(
        10.0,
        label=r"previously rejected scale $N_s/\xi_{\mathrm{2nd}}=10$",
        **pu.fit(0, linestyle="--"),
    )
    for line, line_result in summary["lines"].items():
        points = line_result["points"]
        values = [_load_analysis(point) for point in points]
        axis.errorbar(
            [1.0 / float(point["physical_Nt"]) for point in points],
            [float(result["Ns_over_xi_2nd"]) for result in values],
            [float(result["Ns_over_xi_2nd_error"]) for result in values],
            label=_line_label(line, float(line_result["g3_squared_L"])),
            **pu.data(LINE_PALETTES.get(line, 1), linestyle="-"),
        )
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=r"$N_s/\xi_{\mathrm{2nd}}$",
        title=_title(r"Spatial-volume diagnostic", preliminary),
        legend=True,
    )
    save_figure(figure, output / "Ns_over_xi_2nd_vs_inverse_Nt.pdf")


def plot_chiral_mean(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot the real holonomy trace as a symmetry diagnostic."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axhline(0.0, **pu.fit(0, linestyle="--"))
    for line, line_result in summary["lines"].items():
        points = line_result["points"]
        values = [_load_analysis(point) for point in points]
        means = [
            float(
                result["mean_Tr_G_real"]
                if "mean_Tr_G_real" in result
                else result["mean_TrG"]
            )
            for result in values
        ]
        errors = [
            float(
                result["mean_Tr_G_real_error"]
                if "mean_Tr_G_real_error" in result
                else result["mean_TrG_error"]
            )
            for result in values
        ]
        axis.errorbar(
            [1.0 / float(point["physical_Nt"]) for point in points],
            means,
            errors,
            label=_line_label(line, float(line_result["g3_squared_L"])),
            **pu.data(LINE_PALETTES.get(line, 1), linestyle="-"),
        )
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=r"$\langle\mathrm{Re}\,\mathrm{Tr}\,G\rangle$",
        title=_title(r"Chiral-symmetry diagnostic", preliminary),
        legend=True,
    )
    save_figure(figure, output / "mean_Tr_G_real_vs_inverse_Nt.pdf")


def _load_analysis(point: dict[str, object]) -> dict[str, object]:
    path = Path(str(point["analysis_directory"])) / "analysis.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_k0_fit(result: dict[str, object]) -> dict[str, object]:
    """Return the scan entry that produced the reported central mass."""
    mass = float(result["mass"])
    matches = [
        fit
        for fit in result["K0_fit_scan"]
        if bool(fit.get("stable_patch_anchor", False))
        and np.isclose(float(fit["mass"]), mass, rtol=1.0e-12, atol=0.0)
    ]
    if not matches:
        raise ValueError("reported mass has no matching stable K0 fit")
    return matches[0]


def plot_mass_fit_diagnostics(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot every accepted correlator and its selected periodic-K0 fit."""
    for line, line_result in summary["lines"].items():
        for point in line_result["points"]:
            result = _load_analysis(point)
            fit = _selected_k0_fit(result)
            separation = np.asarray(result["R"], dtype=np.float64)
            correlator = np.asarray(result["C_G"], dtype=np.float64)
            error = np.asarray(result["C_G_error"], dtype=np.float64)
            fit_r = np.linspace(float(fit["r_min"]), float(fit["r_max"]), 500)
            fit_y = float(fit["amplitude"]) * (
                k0(float(fit["mass"]) * fit_r)
                + k0(float(fit["mass"]) * (float(point["Ns"]) - fit_r))
            )
            figure, axis = plt.subplots(figsize=FIGURE_SIZE)
            positive_r = separation > 0.0
            axis.errorbar(
                separation[positive_r],
                correlator[positive_r],
                error[positive_r],
                label=r"$C_G(R)$",
                **pu.data(1),
            )
            axis.plot(
                fit_r,
                fit_y,
                label=(
                    rf"periodic-$K_0$ fit, $R/a={int(fit['r_min'])}$--"
                    rf"${int(fit['r_max'])}$, $p={float(fit['bootstrap_gof_p_value']):.3f}$"
                ),
                **pu.fit(2),
            )
            axis.set_yscale("symlog", linthresh=1.0e-6)
            nt = int(point["physical_Nt"])
            pu.style_axes(
                axis,
                xlabel=r"$R/a$",
                ylabel=r"$C_G(R)$",
                title=_title(
                    rf"Selected mass fit, $\beta/N_t={line},\ N_t={nt},\ "
                    rf"N_s={int(point['Ns'])}$",
                    preliminary,
                ),
                legend=True,
            )
            save_figure(
                figure,
                output
                / (
                    f"C_G_mass_fit_beta_over_Nt_{_tag(line)}_Nt_{nt}.pdf"
                ),
            )


def plot_plaquette_profiles(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot the spatial and temporal plaquette profile at every accepted point."""
    for line, line_result in summary["lines"].items():
        for point in line_result["points"]:
            result = _load_analysis(point)
            profile = result.get("plaquette_profile")
            if profile is None:
                continue
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
            nt = int(point["physical_Nt"])
            pu.style_axes(
                axis,
                xlabel=r"$t/a$",
                ylabel=r"$\frac{1}{2}\,\mathrm{Re}\,\mathrm{Tr}\,U_p$",
                title=_title(
                    rf"Plaquette profile, $\beta/N_t={line},\ N_t={nt},\ "
                    rf"N_s={int(point['Ns'])}$",
                    preliminary,
                ),
                legend=True,
            )
            save_figure(
                figure,
                output
                / (
                    f"plaquette_profile_beta_over_Nt_{_tag(line)}_Nt_{nt}.pdf"
                ),
            )


def plot_profile_collapses(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Compare plaquette profiles across the accepted resolutions."""
    for line, line_result in summary["lines"].items():
        for kind, label, filename in (
            (0.0, r"$P_s(t)$", "spatial"),
            (1.0, r"$P_t(t+1/2)$", "temporal"),
        ):
            figure, axis = plt.subplots(figsize=FIGURE_SIZE)
            plotted = False
            for point_index, point in enumerate(line_result["points"], start=1):
                result = _load_analysis(point)
                profile = result.get("plaquette_profile")
                if profile is None:
                    continue
                keys = np.asarray(profile["kind_t"], dtype=np.float64)
                mean = np.asarray(profile["mean"], dtype=np.float64)
                error = np.asarray(profile["error"], dtype=np.float64)
                selected = keys[:, 0] == kind
                nt = int(point["physical_Nt"])
                role = " PCM" if point["role"] == "pcm_reference" else ""
                axis.errorbar(
                    keys[selected, 1] / nt,
                    mean[selected],
                    error[selected],
                    label=rf"$N_t={nt}$" + role,
                    **pu.data((point_index - 1) % 9 + 1, linestyle="-"),
                )
                plotted = True
            if not plotted:
                plt.close(figure)
                continue
            pu.style_axes(
                axis,
                xlabel=r"$t/N_t$",
                ylabel=label,
                title=_title(
                    rf"Plaquette-profile scaling, $\beta/N_t={line}$",
                    preliminary,
                ),
                legend=True,
            )
            save_figure(
                figure,
                output
                / f"plaquette_{filename}_collapse_beta_over_Nt_{_tag(line)}.pdf",
            )


def plot_full_height_residuals(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot aligned-bootstrap residuals ``W_b(R,Nt)-C_G(R)``."""
    for line, line_result in summary["lines"].items():
        for point in line_result["points"]:
            result = _load_analysis(point)
            if result.get("boundary_wilson") is None:
                continue
            nt = int(point["physical_Nt"])
            archive_path = Path(str(point["analysis_directory"])) / "covariance.npz"
            with np.load(archive_path) as archive:
                separation = np.asarray(archive["R"], dtype=np.int64)
                correlator = np.asarray(archive["C_G"], dtype=np.float64)
                correlator_bootstrap = np.asarray(
                    archive["C_G_bootstrap"], dtype=np.float64
                )
                keys = np.asarray(archive["boundary_wilson_R_H"], dtype=np.int64)
                boundary = np.asarray(
                    archive["boundary_wilson_mean"], dtype=np.float64
                )
                boundary_bootstrap = np.asarray(
                    archive["boundary_wilson_bootstrap"], dtype=np.float64
                )
            full_height_indices = np.flatnonzero(keys[:, 1] == nt)
            cg_index = {int(value): index for index, value in enumerate(separation)}
            matched_cg = np.asarray(
                [cg_index[int(keys[index, 0])] for index in full_height_indices],
                dtype=np.int64,
            )
            difference = boundary[full_height_indices] - correlator[matched_cg]
            difference_bootstrap = (
                boundary_bootstrap[:, full_height_indices]
                - correlator_bootstrap[:, matched_cg]
            )
            difference_error = np.std(difference_bootstrap, axis=0, ddof=1)
            x = float(result["mass"]) * keys[full_height_indices, 0]
            figure, axis = plt.subplots(figsize=FIGURE_SIZE)
            axis.axhline(0.0, **pu.fit(0, linestyle="--"))
            axis.errorbar(x, difference, difference_error, **pu.data(1))
            pu.style_axes(
                axis,
                xlabel=r"$m_GR$",
                ylabel=r"$W_b(R,N_t)-C_G(R)$",
                title=_title(
                    rf"Full-height identity, $\beta/N_t={line},\ N_t={nt},\ "
                    rf"N_s={int(point['Ns'])}$",
                    preliminary,
                ),
            )
            save_figure(
                figure,
                output
                / (
                    f"W_b_full_height_minus_C_G_beta_over_Nt_{_tag(line)}_"
                    f"Nt_{nt}.pdf"
                ),
            )


def plot_correlator_collapse(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot normalized correlators against the fitted physical distance."""
    for line, line_result in summary["lines"].items():
        points = line_result["points"]
        if not points:
            continue
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for point_index, point in enumerate(points, start=1):
            result = _load_analysis(point)
            mass = float(result["mass"])
            separation = np.asarray(result["R"], dtype=np.float64)
            correlator = np.asarray(result["C_G"], dtype=np.float64)
            error = np.asarray(result["C_G_error"], dtype=np.float64)
            normalization = correlator[0]
            role = " PCM" if point["role"] == "pcm_reference" else ""
            axis.errorbar(
                mass * separation,
                correlator / normalization,
                error / normalization,
                label=rf"$N_t={int(point['physical_Nt'])}$" + role,
                **pu.data((point_index - 1) % 9 + 1, linestyle="-"),
            )
        axis.set_yscale("symlog", linthresh=1.0e-5)
        pu.style_axes(
            axis,
            xlabel=r"$m_GR$",
            ylabel=r"$C_G(R)/C_G(0)$",
            title=_title(
                _line_label(line, float(line_result["g3_squared_L"])), preliminary
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"C_G_over_C_G0_vs_mG_R_beta_over_Nt_{_tag(line)}.pdf",
        )


def plot_boundary_surfaces(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Interpolate each measured boundary-loop surface on physical axes."""
    colormap = LinearSegmentedColormap.from_list(
        "project_boundary",
        [pu.bright_color_dict[2], pu.color_dict[0], pu.bright_color_dict[1]],
    )
    for line, line_result in summary["lines"].items():
        for point in line_result["points"]:
            result = _load_analysis(point)
            boundary = result.get("boundary_wilson")
            if boundary is None:
                continue
            nt = int(point["physical_Nt"])
            mass = float(result["mass"])
            keys = np.asarray(boundary["R_H"], dtype=np.int64)
            values = np.asarray(boundary["mean"], dtype=np.float64)
            r_values = np.unique(keys[:, 0])
            h_values = np.unique(keys[:, 1])
            grid = np.full((len(h_values), len(r_values)), np.nan)
            r_index = {value: index for index, value in enumerate(r_values)}
            h_index = {value: index for index, value in enumerate(h_values)}
            for key, value in zip(keys, values, strict=True):
                grid[h_index[key[1]], r_index[key[0]]] = value
            if np.any(~np.isfinite(grid)):
                raise ValueError("boundary-loop grid is incomplete")
            x = mass * r_values.astype(np.float64)
            y = h_values.astype(np.float64) / nt
            x_fine = np.linspace(x[0], x[-1], max(300, 2 * len(x)))
            y_fine = np.linspace(y[0], y[-1], 200)
            interpolation = RegularGridInterpolator((y, x), grid)
            yy, xx = np.meshgrid(y_fine, x_fine, indexing="ij")
            fine = interpolation(np.column_stack((yy.ravel(), xx.ravel()))).reshape(
                yy.shape
            )
            figure, axis = plt.subplots(figsize=FIGURE_SIZE)
            image = axis.pcolormesh(
                x_fine,
                y_fine,
                fine,
                shading="auto",
                cmap=colormap,
                vmin=-1.0,
                vmax=1.0,
            )
            figure.colorbar(image, ax=axis, label=r"$W_b(R,H)$")
            pu.style_axes(
                axis,
                xlabel=r"$m_GR$",
                ylabel=r"$H/N_t$",
                title=_title(
                    rf"$\beta/N_t={line},\ N_t={nt},\ "
                    rf"N_s={int(point['Ns'])}$",
                    preliminary,
                ),
            )
            save_figure(
                figure,
                output
                / (
                    f"W_b_vs_mG_R_H_over_Nt_beta_over_Nt_{_tag(line)}_"
                    f"Nt_{nt}.pdf"
                ),
            )


def plot_boundary_height_curves(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot every measured boundary-loop height with bootstrap errors."""
    for line, line_result in summary["lines"].items():
        for point in line_result["points"]:
            result = _load_analysis(point)
            boundary = result.get("boundary_wilson")
            if boundary is None:
                continue
            nt = int(point["physical_Nt"])
            mass = float(result["mass"])
            keys = np.asarray(boundary["R_H"], dtype=np.int64)
            values = np.asarray(boundary["mean"], dtype=np.float64)
            errors = np.asarray(boundary["error"], dtype=np.float64)
            figure, axis = plt.subplots(figsize=FIGURE_SIZE)
            for height in np.unique(keys[:, 1]):
                selected = keys[:, 1] == height
                order = np.argsort(keys[selected, 0])
                separation = keys[selected, 0][order].astype(np.float64)
                axis.errorbar(
                    mass * separation,
                    values[selected][order],
                    errors[selected][order],
                    label=rf"$H/N_t={height / nt:g}$",
                    **pu.data((int(height) - 1) % 9 + 1, linestyle="-"),
                )
            pu.style_axes(
                axis,
                xlabel=r"$m_GR$",
                ylabel=r"$W_b(R,H)$",
                title=_title(
                    rf"$\beta/N_t={line},\ N_t={nt},\ "
                    rf"N_s={int(point['Ns'])}$",
                    preliminary,
                ),
                legend=True,
            )
            save_figure(
                figure,
                output
                / (
                    f"W_b_height_curves_beta_over_Nt_{_tag(line)}_"
                    f"Nt_{nt}.pdf"
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preliminary", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for variable in ("inverse_nt", "inverse_nt_squared", "inverse_beta"):
        plot_mass_scaling(summary, args.output, variable, args.preliminary)
    plot_continuum_mass(summary, args.output, args.preliminary)
    plot_length_ratio(summary, args.output, args.preliminary)
    plot_spatial_volume(summary, args.output, args.preliminary)
    plot_chiral_mean(summary, args.output, args.preliminary)
    plot_correlator_collapse(summary, args.output, args.preliminary)
    plot_mass_fit_diagnostics(summary, args.output, args.preliminary)
    plot_plaquette_profiles(summary, args.output, args.preliminary)
    plot_profile_collapses(summary, args.output, args.preliminary)
    plot_boundary_surfaces(summary, args.output, args.preliminary)
    plot_boundary_height_curves(summary, args.output, args.preliminary)
    plot_full_height_residuals(summary, args.output, args.preliminary)


if __name__ == "__main__":
    main()
