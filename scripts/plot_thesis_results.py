#!/usr/bin/env python3
"""Plot compact Dirichlet results after raw-history analysis on qbig."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


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
    """Save a compact vector figure."""
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_profiles(results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot spatial and temporal plaquette profiles for each thickness."""
    figure, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.5))
    axes_array = np.atleast_1d(axes)
    for axis, (nt, result) in zip(axes_array, results.items(), strict=True):
        profile = result["plaquette_profile"]
        keys = np.asarray(profile["kind_t"], dtype=np.float64)
        mean = np.asarray(profile["mean"], dtype=np.float64)
        error = np.asarray(profile["error"], dtype=np.float64)
        spatial = (keys[:, 0] == 0.0) & (keys[:, 1] > 0.0) & (keys[:, 1] < nt)
        temporal = keys[:, 0] == 1.0
        axis.errorbar(
            keys[spatial, 1], mean[spatial], error[spatial], fmt="o-", label=r"$P_s$"
        )
        axis.errorbar(
            keys[temporal, 1],
            mean[temporal],
            error[temporal],
            fmt="s-",
            label=r"$P_t$",
        )
        axis.set(title=f"Nt={nt}", xlabel=r"$t$")
        axis.grid(alpha=0.25)
    axes_array[0].set_ylabel(r"$(1/2)\,\mathrm{Re}\,\mathrm{Tr}\,U_p$")
    axes_array[-1].legend()
    figure.suptitle(r"Plaquette profiles; fixed $P_s(0)=P_s(Nt)=1$ omitted")
    save_figure(figure, output / "plaquette_profiles.pdf")


def plot_correlators(results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot the invariant slab correlator and the height-one boundary loop."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for nt, result in results.items():
        r = np.asarray(result["R"], dtype=np.float64)
        correlator = np.asarray(result["C_G"], dtype=np.float64)
        error = np.asarray(result["C_G_error"], dtype=np.float64)
        axes[0].errorbar(r, correlator, error, fmt=".-", label=f"Nt={nt}")

        wilson = result["boundary_wilson"]
        keys = np.asarray(wilson["R_H"], dtype=np.float64)
        mean = np.asarray(wilson["mean"], dtype=np.float64)
        wilson_error = np.asarray(wilson["error"], dtype=np.float64)
        selected = keys[:, 1] == 1.0
        axes[1].errorbar(
            keys[selected, 0],
            mean[selected],
            wilson_error[selected],
            fmt=".-",
            label=f"Nt={nt}",
        )
    axes[0].set(
        xlabel=r"$R$",
        ylabel=r"$C_G(R)$",
        title=r"Invariant full-height holonomy correlator",
        yscale="log",
    )
    axes[1].set(
        xlabel=r"$R$",
        ylabel=r"$W_b(R,1)$",
        title=r"Boundary-anchored height-one loop",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    save_figure(figure, output / "correlators.pdf")


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

    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.errorbar(
        thesis_nt,
        thesis_mean,
        thesis_error,
        fmt="o-",
        color="0.45",
        label="thesis Table 4.4 (cold start)",
    )
    axis.errorbar(
        measured_nt,
        measured,
        measured_error,
        fmt="s",
        color="tab:red",
        label="this work (hot start)",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Nt",
        ylabel=r"$\langle (1/2)\,\mathrm{Tr}\,G\rangle$",
        title="Traced holonomy thermalization diagnostic",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output / "traced_holonomy_comparison.pdf")


def plot_correlations(results_root: Path, results: dict[int, dict[str, object]], output: Path) -> None:
    """Plot correlation matrices on the independent nonzero-distance range."""
    figure, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.5))
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, nt in zip(axes_array, results, strict=True):
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
        image = axis.imshow(correlation, origin="lower", vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set(title=f"Nt={nt}", xlabel=r"$R$ index", ylabel=r"$R$ index")
    if image is not None:
        figure.colorbar(image, ax=axes_array.tolist(), shrink=0.8, label="correlation")
    figure.subplots_adjust(wspace=0.3, right=0.9)
    figure.savefig(output / "holonomy_correlation_matrices.pdf", bbox_inches="tight")
    plt.close(figure)


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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
