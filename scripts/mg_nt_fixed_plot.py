#!/usr/bin/env python3
"""Plot compact nonperturbative fixed-mass tuning products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import k0


ROOT = Path(__file__).resolve().parents[1]
STATANALYSIS = ROOT / "statanalysis-Bonn"
if not (STATANALYSIS / "plot_utils.py").is_file():
    raise FileNotFoundError("statanalysis-Bonn/plot_utils.py is required")
sys.path.insert(0, str(STATANALYSIS))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402


pu.setup_mpl()


def _title(text: str, preliminary: bool) -> str:
    """Prefix a figure title when it uses pilot or incomplete tuning data."""
    return f"Preliminary: {text}" if preliminary else text


def plot_brackets(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Write one mass-versus-beta bracket plot for every target."""
    for target in summary["targets"]:
        selected = [
            record
            for record in summary["groups"]
            if float(record["target_c"]) == float(target)
        ]
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for palette, record in enumerate(selected, start=1):
            points = record["points"]
            contains_prior = any(
                point.get("source_campaign") == "beta_nt_fixed"
                for point in points
            )
            beta = np.asarray([point["beta"] for point in points], dtype=np.float64)
            mass_nt = np.asarray(
                [point["mass_Nt"] for point in points], dtype=np.float64
            )
            error = np.asarray(
                [point["mass_Nt_error"] for point in points], dtype=np.float64
            )
            axis.errorbar(
                beta,
                mass_nt,
                error,
                label=(
                    rf"$N_t={int(record['physical_Nt'])}$"
                    + (" (fit rejected)" if not bool(record["accepted"]) else "")
                    + (
                        r" (prior fixed-$\beta/N_t$)"
                        if contains_prior
                        else ""
                    )
                ),
                **pu.data(palette),
            )
            if record.get("fit") is not None:
                line_beta = np.linspace(float(beta[0]), float(beta[-1]), 200)
                fit = record["fit"]
                axis.plot(
                    line_beta,
                    np.exp(float(fit["a"]) + float(fit["b"]) * line_beta),
                    **pu.fit(
                        palette,
                        linestyle="-" if bool(record["accepted"]) else "--",
                    ),
                )
        axis.axhline(float(target), **pu.fit(0, linestyle="dashed"))
        pu.style_axes(
            axis,
            xlabel=r"$\beta$",
            ylabel=r"$m_G N_t$",
            title=_title(
                rf"Fixed-mass pilot, target $c={float(target):g}$", preliminary
            ),
            legend=True,
        )
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
        )
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        tag = str(target).replace(".", "p")
        save_figure(figure, output / f"mass_brackets_c_{tag}.pdf")


def _accepted_records(
    summary: dict[str, object], target: float
) -> list[dict[str, object]]:
    """Return accepted tuning groups for one target in increasing ``Nt``."""
    return sorted(
        (
            record
            for record in summary["groups"]
            if float(record["target_c"]) == target and bool(record["accepted"])
        ),
        key=lambda record: int(record["physical_Nt"]),
    )


def plot_tuned_beta(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot accepted tuned couplings against temporal resolution."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    plotted = False
    for palette, target in enumerate(summary["targets"], start=1):
        records = _accepted_records(summary, float(target))
        if not records:
            continue
        nt = np.asarray(
            [record["physical_Nt"] for record in records], dtype=np.float64
        )
        beta = np.asarray(
            [record["beta_star"]["median"] for record in records],
            dtype=np.float64,
        )
        interval = np.asarray(
            [record["beta_star"]["statistical_16_84"] for record in records],
            dtype=np.float64,
        )
        error = np.vstack((beta - interval[:, 0], interval[:, 1] - beta))
        contains_prior = any(
            point.get("source_campaign") == "beta_nt_fixed"
            for record in records
            for point in record["points"]
        )
        axis.errorbar(
            nt,
            beta,
            error,
            label=rf"$c={float(target):g}$"
            + (r" (includes prior fixed-$\beta/N_t$ input)" if contains_prior else ""),
            **pu.data(
                palette,
                linestyle="-" if np.all(np.diff(nt) == 1.0) else "none",
            ),
        )
        plotted = True
    if not plotted:
        plt.close(figure)
        return
    pu.style_axes(
        axis,
        xlabel=r"$N_t$",
        ylabel=r"$\beta_\star$",
        title=_title(
            r"Nonperturbatively tuned fixed-$m_G N_t$ trajectories",
            preliminary,
        ),
        legend=True,
    )
    save_figure(figure, output / "tuned_beta_vs_Nt.pdf")


def plot_tuned_beta_over_nt(
    summary: dict[str, object], output: Path, preliminary: bool
) -> None:
    """Plot accepted ``beta_star/Nt`` values against ``1/Nt``."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    plotted = False
    for palette, target in enumerate(summary["targets"], start=1):
        records = _accepted_records(summary, float(target))
        if not records:
            continue
        nt = np.asarray(
            [record["physical_Nt"] for record in records], dtype=np.float64
        )
        beta = np.asarray(
            [record["beta_star"]["median"] for record in records],
            dtype=np.float64,
        )
        interval = np.asarray(
            [record["beta_star"]["statistical_16_84"] for record in records],
            dtype=np.float64,
        )
        ratio = beta / nt
        error = np.vstack(
            ((beta - interval[:, 0]) / nt, (interval[:, 1] - beta) / nt)
        )
        contains_prior = any(
            point.get("source_campaign") == "beta_nt_fixed"
            for record in records
            for point in record["points"]
        )
        axis.errorbar(
            1.0 / nt,
            ratio,
            error,
            label=rf"$c={float(target):g}$"
            + (r" (includes prior fixed-$\beta/N_t$ input)" if contains_prior else ""),
            **pu.data(
                palette,
                linestyle="-" if np.all(np.diff(nt) == 1.0) else "none",
            ),
        )
        plotted = True
    if not plotted:
        plt.close(figure)
        return
    pu.style_axes(
        axis,
        xlabel=r"$1/N_t$",
        ylabel=r"$\beta_\star/N_t$",
        title=_title(r"Tuned coupling per temporal link", preliminary),
        legend=True,
    )
    save_figure(figure, output / "tuned_beta_over_Nt_vs_inverse_Nt.pdf")


def load_campaign_results(
    analysis_root: Path,
) -> list[tuple[Path, dict[str, object]]]:
    """Load pilot, extension, and explicitly labelled prior-audit results."""
    paths: list[Path] = []
    for relative in (Path("pilot"), Path("extensions/round1"), Path("audit")):
        directory = analysis_root / relative
        if directory.is_dir():
            paths.extend(directory.rglob("analysis.json"))
    records = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(set(paths))
    ]
    return sorted(
        records,
        key=lambda item: (
            float(item[1]["target_c"]),
            int(item[1]["physical_Nt"]),
            float(item[1]["beta"]),
        ),
    )


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


def _tag_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _prior_suffix(result: dict[str, object]) -> str:
    if result.get("purpose") == "prior_information_audit":
        return r",\ \mathrm{prior\ fixed}\ \beta/N_t"
    return ""


def plot_mass_fit_diagnostics(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot every available correlator and its accepted periodic-K0 fit."""
    for _, result in results:
        fit = _selected_k0_fit(result)
        separation = np.asarray(result["R"], dtype=np.float64)
        correlator = np.asarray(result["C_G"], dtype=np.float64)
        error = np.asarray(result["C_G_error"], dtype=np.float64)
        fit_r = np.linspace(float(fit["r_min"]), float(fit["r_max"]), 500)
        fit_y = float(fit["amplitude"]) * (
            k0(float(fit["mass"]) * fit_r)
            + k0(float(fit["mass"]) * (float(result["Ns"]) - fit_r))
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
        target = float(result["target_c"])
        nt = int(result["physical_Nt"])
        beta = float(result["beta"])
        pu.style_axes(
            axis,
            xlabel=r"$R/a$",
            ylabel=r"$C_G(R)$",
            title=_title(
                rf"Selected mass fit, $c={target:g},\ N_t={nt},\ "
                rf"\beta={beta:g}{_prior_suffix(result)}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output
            / (
                f"C_G_mass_fit_c_{_tag_number(target)}_Nt_{nt}_"
                f"Ns_{int(result['Ns'])}_b_{_tag_number(beta)}.pdf"
            ),
        )


def plot_plaquette_profiles(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot the plaquette profile at every available pilot point."""
    for _, result in results:
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
        target = float(result["target_c"])
        nt = int(result["physical_Nt"])
        beta = float(result["beta"])
        pu.style_axes(
            axis,
            xlabel=r"$t/a$",
            ylabel=r"$\frac{1}{2}\,\mathrm{Re}\,\mathrm{Tr}\,U_p$",
            title=_title(
                rf"Plaquette profile, $c={target:g},\ N_t={nt},\ "
                rf"\beta={beta:g}{_prior_suffix(result)}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output
            / (
                f"plaquette_profile_c_{_tag_number(target)}_Nt_{nt}_"
                f"Ns_{int(result['Ns'])}_b_{_tag_number(beta)}.pdf"
            ),
        )


def plot_correlator_groups(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Compare normalized correlators across every local tuning bracket."""
    groups = sorted(
        {
            (float(result["target_c"]), int(result["physical_Nt"]))
            for _, result in results
        }
    )
    for target, nt in groups:
        selected = [
            result
            for _, result in results
            if float(result["target_c"]) == target
            and int(result["physical_Nt"]) == nt
        ]
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for palette, result in enumerate(selected, start=1):
            separation = np.asarray(result["R"], dtype=np.float64)
            correlator = np.asarray(result["C_G"], dtype=np.float64)
            error = np.asarray(result["C_G_error"], dtype=np.float64)
            prior = (
                r" (prior fixed-$\beta/N_t$)"
                if result.get("purpose") == "prior_information_audit"
                else ""
            )
            axis.errorbar(
                float(result["mass"]) * separation,
                correlator / correlator[0],
                error / correlator[0],
                label=rf"$\beta={float(result['beta']):g}$" + prior,
                **pu.data((palette - 1) % 9 + 1, linestyle="-"),
            )
        axis.set_yscale("symlog", linthresh=1.0e-5)
        pu.style_axes(
            axis,
            xlabel=r"$m_GR$",
            ylabel=r"$C_G(R)/C_G(0)$",
            title=_title(
                rf"Local bracket correlators, $c={target:g},\ N_t={nt}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"C_G_bracket_c_{_tag_number(target)}_Nt_{nt}.pdf",
        )


def _length_ratio(path: Path, result: dict[str, object]) -> tuple[float, float]:
    """Return ``xi_exp/xi_2nd`` and its aligned-bootstrap uncertainty."""
    with np.load(path.with_name("covariance.npz")) as archive:
        xi_exp = np.asarray(archive["xi_exp_bootstrap"], dtype=np.float64)
        xi_2nd = np.asarray(archive["xi_2nd_bootstrap"], dtype=np.float64)
    replicas = xi_exp / xi_2nd
    finite = replicas[np.isfinite(replicas)]
    if finite.size < 2:
        raise ValueError(f"too few finite length-ratio replicas in {path}")
    return float(result["xi_exp"]) / float(result["xi_2nd"]), float(
        np.std(finite, ddof=1)
    )


def plot_length_ratios(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot the exponential-to-second-moment ratio over each pilot grid."""
    for target in sorted({float(result["target_c"]) for _, result in results}):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        for palette, nt in enumerate(
            sorted(
                {
                    int(result["physical_Nt"])
                    for _, result in results
                    if float(result["target_c"]) == target
                }
            ),
            start=1,
        ):
            selected = [
                (path, result)
                for path, result in results
                if float(result["target_c"]) == target
                and int(result["physical_Nt"]) == nt
            ]
            values = [_length_ratio(path, result) for path, result in selected]
            contains_prior = any(
                result.get("purpose") == "prior_information_audit"
                for _, result in selected
            )
            axis.errorbar(
                [float(result["beta"]) for _, result in selected],
                [value for value, _ in values],
                [error for _, error in values],
                label=rf"$N_t={nt}$"
                + (
                    r" (includes prior fixed-$\beta/N_t$ point)"
                    if contains_prior
                    else ""
                ),
                **pu.data(palette, linestyle="-"),
            )
        pu.style_axes(
            axis,
            xlabel=r"$\beta$",
            ylabel=r"$\xi_{\exp}/\xi_{\mathrm{2nd}}$",
            title=_title(
                rf"Length-estimator comparison, target $c={target:g}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"xi_exp_over_xi_2nd_c_{_tag_number(target)}.pdf",
        )


def plot_volume_safety(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot the nominal spatial-volume measure ``m_G Ns``."""
    for target in sorted({float(result["target_c"]) for _, result in results}):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.axhline(12.0, label=r"nominal target $m_GN_s=12$", **pu.fit(0, linestyle="--"))
        for palette, nt in enumerate(
            sorted(
                {
                    int(result["physical_Nt"])
                    for _, result in results
                    if float(result["target_c"]) == target
                }
            ),
            start=1,
        ):
            selected = [
                result
                for _, result in results
                if float(result["target_c"]) == target
                and int(result["physical_Nt"]) == nt
            ]
            contains_prior = any(
                result.get("purpose") == "prior_information_audit"
                for result in selected
            )
            axis.errorbar(
                [float(result["beta"]) for result in selected],
                [float(result["mass"]) * float(result["Ns"]) for result in selected],
                [
                    float(result["mass_error"]) * float(result["Ns"])
                    for result in selected
                ],
                label=rf"$N_t={nt}$"
                + (
                    r" (includes prior fixed-$\beta/N_t$ point)"
                    if contains_prior
                    else ""
                ),
                **pu.data(palette, linestyle="-"),
            )
        pu.style_axes(
            axis,
            xlabel=r"$\beta$",
            ylabel=r"$m_GN_s$",
            title=_title(
                rf"Spatial-volume diagnostic, target $c={target:g}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"mG_Ns_c_{_tag_number(target)}.pdf",
        )


def plot_second_moment_volume(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot the measured spatial extent in second-moment correlation lengths."""
    for target in sorted({float(result["target_c"]) for _, result in results}):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.axhline(
            10.0,
            label=r"previously rejected scale $N_s/\xi_{\mathrm{2nd}}=10$",
            **pu.fit(0, linestyle="--"),
        )
        for palette, nt in enumerate(
            sorted(
                {
                    int(result["physical_Nt"])
                    for _, result in results
                    if float(result["target_c"]) == target
                }
            ),
            start=1,
        ):
            selected = [
                result
                for _, result in results
                if float(result["target_c"]) == target
                and int(result["physical_Nt"]) == nt
            ]
            contains_prior = any(
                result.get("purpose") == "prior_information_audit"
                for result in selected
            )
            axis.errorbar(
                [float(result["beta"]) for result in selected],
                [float(result["Ns_over_xi_2nd"]) for result in selected],
                [float(result["Ns_over_xi_2nd_error"]) for result in selected],
                label=rf"$N_t={nt}$"
                + (
                    r" (includes prior fixed-$\beta/N_t$ point)"
                    if contains_prior
                    else ""
                ),
                **pu.data(palette, linestyle="-"),
            )
        pu.style_axes(
            axis,
            xlabel=r"$\beta$",
            ylabel=r"$N_s/\xi_{\mathrm{2nd}}$",
            title=_title(
                rf"Second-moment volume diagnostic, target $c={target:g}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"Ns_over_xi_2nd_c_{_tag_number(target)}.pdf",
        )


def plot_chiral_means(
    results: list[tuple[Path, dict[str, object]]],
    output: Path,
    preliminary: bool,
) -> None:
    """Plot the real holonomy trace as a chiral-symmetry diagnostic."""
    for target in sorted({float(result["target_c"]) for _, result in results}):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.axhline(0.0, **pu.fit(0, linestyle="--"))
        for palette, nt in enumerate(
            sorted(
                {
                    int(result["physical_Nt"])
                    for _, result in results
                    if float(result["target_c"]) == target
                }
            ),
            start=1,
        ):
            selected = [
                result
                for _, result in results
                if float(result["target_c"]) == target
                and int(result["physical_Nt"]) == nt
            ]
            contains_prior = any(
                result.get("purpose") == "prior_information_audit"
                for result in selected
            )
            axis.errorbar(
                [float(result["beta"]) for result in selected],
                [float(result["mean_Tr_G_real"]) for result in selected],
                [float(result["mean_Tr_G_real_error"]) for result in selected],
                label=rf"$N_t={nt}$"
                + (
                    r" (includes prior fixed-$\beta/N_t$ point)"
                    if contains_prior
                    else ""
                ),
                **pu.data(palette, linestyle="-"),
            )
        pu.style_axes(
            axis,
            xlabel=r"$\beta$",
            ylabel=r"$\langle\mathrm{Re}\,\mathrm{Tr}\,G\rangle$",
            title=_title(
                rf"Chiral-symmetry diagnostic, target $c={target:g}$",
                preliminary,
            ),
            legend=True,
        )
        save_figure(
            figure,
            output / f"mean_Tr_G_real_c_{_tag_number(target)}.pdf",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis-root", type=Path, default=ROOT / "analysis/mg_nt_fixed"
    )
    parser.add_argument("--preliminary", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot_brackets(summary, args.output, args.preliminary)
    plot_tuned_beta(summary, args.output, args.preliminary)
    plot_tuned_beta_over_nt(summary, args.output, args.preliminary)
    results = load_campaign_results(args.analysis_root)
    plot_mass_fit_diagnostics(results, args.output, args.preliminary)
    plot_plaquette_profiles(results, args.output, args.preliminary)
    plot_correlator_groups(results, args.output, args.preliminary)
    plot_length_ratios(results, args.output, args.preliminary)
    plot_volume_safety(results, args.output, args.preliminary)
    plot_second_moment_volume(results, args.output, args.preliminary)
    plot_chiral_means(results, args.output, args.preliminary)


if __name__ == "__main__":
    main()
