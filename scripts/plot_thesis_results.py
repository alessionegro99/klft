#!/usr/bin/env python3
"""Plot compact Dirichlet results after raw-history analysis on qbig."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError(
        "statanalysis-Bonn/plot_utils.py is required for the project plot style"
    )
sys.path.insert(0, str(STATANALYSIS))
import plot_utils as pu  # noqa: E402


FIGURE_SIZE = (16.0, 9.0)
PALETTE_INDICES = (1, 2, 3)


pu.setup_mpl()


def load_results(root: Path) -> dict[int, dict[str, object]]:
    """Load one ``cut_500/analysis.json`` file per available thickness."""
    results: dict[int, dict[str, object]] = {}
    for path in root.glob("Nt_*_Ns_64_b_12p15266/cut_500/analysis.json"):
        nt = int(path.parents[1].name.split("_")[1])
        results[nt] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise FileNotFoundError(f"no analyzed ensembles below {root}")
    return dict(sorted(results.items()))


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save one 16:9 plot as a single-page PDF."""
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"figure output must be PDF, got {path}")
    width, height = figure.get_size_inches()
    if not np.allclose((width, height), FIGURE_SIZE):
        raise ValueError(
            f"figure must use figsize={FIGURE_SIZE}, got {(width, height)}"
        )
    plot_axes = [axis for axis in figure.axes if not hasattr(axis, "_colorbar")]
    if len(plot_axes) != 1:
        raise ValueError(f"figure must contain exactly one plot, got {len(plot_axes)}")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_profiles(results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot spatial and temporal plaquette profiles for each thickness."""
    visible_values: list[np.ndarray] = []
    visible_errors: list[np.ndarray] = []
    for nt, result in results.items():
        profile = result["plaquette_profile"]
        keys = np.asarray(profile["kind_t"], dtype=np.float64)
        mean = np.asarray(profile["mean"], dtype=np.float64)
        error = np.asarray(profile["error"], dtype=np.float64)
        visible = (keys[:, 0] == 1.0) | (
            (keys[:, 0] == 0.0) & (keys[:, 1] > 0.0) & (keys[:, 1] < nt)
        )
        visible_values.append(mean[visible])
        visible_errors.append(error[visible])
    all_values = np.concatenate(visible_values)
    all_errors = np.concatenate(visible_errors)
    lower = float(np.min(all_values - all_errors))
    upper = float(np.max(all_values + all_errors))
    padding = 0.05 * (upper - lower)

    for nt, result in results.items():
        profile = result["plaquette_profile"]
        keys = np.asarray(profile["kind_t"], dtype=np.float64)
        mean = np.asarray(profile["mean"], dtype=np.float64)
        error = np.asarray(profile["error"], dtype=np.float64)
        spatial = (keys[:, 0] == 0.0) & (keys[:, 1] > 0.0) & (keys[:, 1] < nt)
        temporal = keys[:, 0] == 1.0
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.errorbar(
            keys[spatial, 1],
            mean[spatial],
            error[spatial],
            label=r"$P_s$",
            **pu.data(1, linestyle="-"),
        )
        axis.errorbar(
            keys[temporal, 1],
            mean[temporal],
            error[temporal],
            label=r"$P_t$",
            **pu.data(2, linestyle="-"),
        )
        pu.style_axes(
            axis,
            xlabel=r"$t$",
            ylabel=r"$\frac{1}{2}\,\mathrm{Re}\,\mathrm{Tr}\,U_p$",
            title=rf"Plaquette profile, $N_t={nt}$",
            legend=True,
            grid=True,
            ylim=(lower - padding, upper + padding),
        )
        save_figure(figure, output / f"plaquette_profile_Nt_{nt}.pdf")


def plot_correlators(results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot the invariant slab correlator and the height-one boundary loop."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette_index, (nt, result) in zip(
        PALETTE_INDICES, results.items(), strict=True
    ):
        r = np.asarray(result["R"], dtype=np.float64)
        correlator = np.asarray(result["C_G"], dtype=np.float64)
        error = np.asarray(result["C_G_error"], dtype=np.float64)
        axis.errorbar(
            r,
            correlator,
            error,
            label=rf"$N_t={nt}$",
            **pu.data(palette_index, linestyle="-"),
        )
    pu.style_axes(
        axis,
        ylog=True,
        xlabel=r"$R$",
        ylabel=r"$C_G(R)$",
        title=r"Invariant boundary-to-boundary holonomy correlator",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "holonomy_correlator.pdf")

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette_index, (nt, result) in zip(
        PALETTE_INDICES, results.items(), strict=True
    ):
        wilson = result["boundary_wilson"]
        keys = np.asarray(wilson["R_H"], dtype=np.float64)
        mean = np.asarray(wilson["mean"], dtype=np.float64)
        wilson_error = np.asarray(wilson["error"], dtype=np.float64)
        selected = keys[:, 1] == 1.0
        axis.errorbar(
            keys[selected, 0],
            mean[selected],
            wilson_error[selected],
            label=rf"$N_t={nt}$",
            **pu.data(palette_index, linestyle="-"),
        )
    pu.style_axes(
        axis,
        ylog=True,
        xlabel=r"$R$",
        ylabel=r"$W_b(R,1)$",
        title=r"Boundary-anchored height-one loop",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "boundary_loop_H_1.pdf")


def plot_trace_comparison(
    results: dict[int, dict[str, object]], benchmarks: Path, output: Path
) -> None:
    """Compare hot-start normalized traces with thesis Table 4.4."""
    reference = json.loads(benchmarks.read_text(encoding="utf-8"))["values"]
    thesis_nt = np.asarray(sorted(map(int, reference)), dtype=np.int64)
    thesis_mean = np.asarray([reference[str(nt)][0] for nt in thesis_nt])
    thesis_error = np.asarray([reference[str(nt)][1] for nt in thesis_nt])
    measured_nt = np.asarray(list(results), dtype=np.int64)
    measured = np.asarray(
        [0.5 * float(results[nt]["mean_TrG"]) for nt in measured_nt]
    )
    measured_error = np.asarray(
        [0.5 * float(results[nt]["mean_TrG_error"]) for nt in measured_nt]
    )

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.errorbar(
        thesis_nt,
        thesis_mean,
        thesis_error,
        label="thesis Table 4.4 (cold start)",
        **pu.data(0, linestyle="-"),
    )
    axis.errorbar(
        measured_nt,
        measured,
        measured_error,
        label="this work (hot start)",
        **pu.data(2),
    )
    axis.axhline(0.0, **pu.fit(0, linewidth=1.2))
    pu.style_axes(
        axis,
        xlabel=r"$N_t$",
        ylabel=r"$\left\langle \frac{1}{2}\,\mathrm{Tr}\,G\right\rangle$",
        title="Traced holonomy thermalization diagnostic",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "traced_holonomy_comparison.pdf")


def plot_correlations(results_root: Path, results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot correlation matrices on the independent nonzero-distance range."""
    colormap = LinearSegmentedColormap.from_list(
        "project_diverging",
        [pu.bright_color_dict[1], pu.color_dict[0], pu.bright_color_dict[2]],
    )
    for nt in results:
        path = (
            results_root
            / f"Nt_{nt}_Ns_64_b_12p15266"
            / "cut_500"
            / "covariance.npz"
        )
        with np.load(path) as data:
            covariance = np.asarray(data["covariance"], dtype=np.float64)[1:, 1:]
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        image = axis.imshow(
            correlation,
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
            cmap=colormap,
        )
        pu.style_axes(
            axis,
            xlabel=r"$R$ index",
            ylabel=r"$R$ index",
            title=rf"Holonomy correlation matrix, $N_t={nt}$",
        )
        figure.colorbar(image, ax=axis, label="correlation")
        save_figure(figure, output / f"holonomy_correlation_matrix_Nt_{nt}.pdf")


def write_summary(
    results: dict[int, dict[str, object]], benchmarks: Path, output: Path
) -> None:
    """Write the principal diagnostics and direct thesis comparisons."""
    reference = json.loads(benchmarks.read_text(encoding="utf-8"))
    trace_reference = reference["values"]
    wilson_reference = reference["boundary_wilson_R64_H1"]["values"]
    rows: list[dict[str, float | int | str]] = []
    for nt, result in results.items():
        trace = 0.5 * float(result["mean_TrG"])
        trace_error = 0.5 * float(result["mean_TrG_error"])
        thesis_trace, thesis_trace_error = trace_reference[str(nt)]
        trace_z = (trace - thesis_trace) / math.hypot(
            trace_error, thesis_trace_error
        )

        wilson = result["boundary_wilson"]
        keys = np.asarray(wilson["R_H"], dtype=np.int64)
        selected = (keys[:, 0] == 64) & (keys[:, 1] == 1)
        if np.count_nonzero(selected) != 1:
            raise ValueError(f"Nt={nt}: missing unique W_b(64,1)")
        wilson_mean = float(np.asarray(wilson["mean"])[selected][0])
        wilson_error = float(np.asarray(wilson["error"])[selected][0])
        thesis_wilson = ""
        thesis_wilson_error = ""
        wilson_z: float | str = ""
        if str(nt) in wilson_reference:
            thesis_wilson, thesis_wilson_error = wilson_reference[str(nt)]
            wilson_z = (wilson_mean - thesis_wilson) / math.hypot(
                wilson_error, thesis_wilson_error
            )
        fit = next(
            entry
            for entry in result["K0_fit_scans"]
            if entry["r_min"] == 6 and entry["r_max"] == 12
        )
        correlated_fit = fit["correlated"]
        uncorrelated_fit = fit["uncorrelated"]
        maximum_correlated_p = max(
            float(entry["correlated"].get("p_value", 0.0))
            for entry in result["K0_fit_scans"]
        )
        rows.append(
            {
                "Nt": nt,
                "half_TrG": trace,
                "half_TrG_error": trace_error,
                "thesis_half_TrG": thesis_trace,
                "thesis_half_TrG_error": thesis_trace_error,
                "half_TrG_difference_z": trace_z,
                "xi_2nd": float(result["xi_2nd"]),
                "xi_2nd_error": float(result["xi_2nd_error"]),
                "Wb_64_1": wilson_mean,
                "Wb_64_1_error": wilson_error,
                "thesis_Wb_64_1": thesis_wilson,
                "thesis_Wb_64_1_error": thesis_wilson_error,
                "Wb_64_1_difference_z": wilson_z,
                "max_profile_reflection_z": float(
                    result["plaquette_profile"]["max_reflection_z"]
                ),
                "max_tau_int": float(result["max_tau_int"]),
                "block_size": int(result["block_size"]),
                "n_blocks": int(result["n_blocks"]),
                "max_chain_z": float(result["max_chain_disagreement_z"]),
                "max_split_z": float(result["max_split_history_z"]),
                "max_orientation_z": float(
                    result["max_global_orientation_relation_z"]
                ),
                "K0_R6_12_correlated_mass": float(correlated_fit["mass"]),
                "K0_R6_12_correlated_mass_error": float(
                    correlated_fit["mass_error"]
                ),
                "K0_R6_12_correlated_chi2_dof": float(
                    correlated_fit["chi2"] / correlated_fit["dof"]
                ),
                "K0_R6_12_correlated_p": float(correlated_fit["p_value"]),
                "maximum_correlated_fit_scan_p": maximum_correlated_p,
                "K0_R6_12_uncorrelated_mass": float(uncorrelated_fit["mass"]),
                "K0_R6_12_uncorrelated_mass_error": float(
                    uncorrelated_fit["mass_error"]
                ),
                "K0_R6_12_uncorrelated_chi2_dof": float(
                    uncorrelated_fit["chi2"] / uncorrelated_fit["dof"]
                ),
                "K0_R6_12_uncorrelated_p": float(uncorrelated_fit["p_value"]),
            }
        )
    with (output / "campaign_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    plot_profiles(results, args.output)
    plot_correlators(results, args.output)
    plot_trace_comparison(results, args.benchmarks, args.output)
    plot_correlations(args.results, results, args.output)
    write_summary(results, args.benchmarks, args.output)


if __name__ == "__main__":
    main()
