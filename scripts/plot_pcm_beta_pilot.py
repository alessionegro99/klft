#!/usr/bin/env python3
"""Plot compact, chain-aware results from the exact-PCM beta pilot."""

from __future__ import annotations

import argparse
import csv
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
    raise FileNotFoundError(
        "statanalysis-Bonn/plot_utils.py is required for the project plot style"
    )
sys.path.insert(0, str(STATANALYSIS))
import plot_utils as pu  # noqa: E402

from plot_thesis_results import FIGURE_SIZE, save_figure  # noqa: E402


pu.setup_mpl()


def load_results(root: Path) -> dict[float, dict[str, object]]:
    """Load one compact analysis file for each pilot coupling."""
    results: dict[float, dict[str, object]] = {}
    for path in root.glob("Nt_1_Ns_96_b_*/analysis.json"):
        tag = path.parent.name.rsplit("_b_", maxsplit=1)[1]
        beta = float(tag.replace("p", "."))
        results[beta] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise FileNotFoundError(f"no PCM beta analyses below {root}")
    return dict(sorted(results.items()))


def plot_xi(results: dict[float, dict[str, object]], output: Path) -> None:
    """Plot xi_2nd and the predeclared finite-volume acceptance window."""
    beta = np.asarray(list(results), dtype=np.float64)
    xi = np.asarray([result["xi_2nd"] for result in results.values()])
    error = np.asarray([result["xi_2nd_error"] for result in results.values()])
    ns = int(next(iter(results.values()))["ns"])
    minimum_xi = float(
        next(iter(results.values()))["selection_criteria"]["minimum_xi"]
    )
    minimum_ratio = float(
        next(iter(results.values()))["selection_criteria"]["minimum_Ns_over_xi"]
    )
    maximum_xi = ns / minimum_ratio

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.axhspan(minimum_xi, maximum_xi, **pu.conf_band(7, alpha=0.18))
    axis.errorbar(beta, xi, error, **pu.data(1))
    pu.style_axes(
        axis,
        xlabel=r"$\beta$",
        ylabel=r"$\xi_{\mathrm{2nd}}/a$",
        title=rf"Exact SU(2) PCM limit, $N_s={ns}$",
        grid=True,
        xlim=(float(beta[0] - 0.03), float(beta[-1] + 0.03)),
    )
    save_figure(figure, output / "xi_2nd_vs_beta.pdf")


def plot_correlator(results: dict[float, dict[str, object]], output: Path) -> None:
    """Plot the primary invariant correlator for every pilot coupling."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for palette_index, (beta, result) in enumerate(results.items(), start=1):
        separation = np.asarray(result["R"], dtype=np.float64)
        correlator = np.asarray(result["C_G"], dtype=np.float64)
        error = np.asarray(result["C_G_error"], dtype=np.float64)
        axis.errorbar(
            separation,
            correlator,
            error,
            label=rf"$\beta={beta:g}$",
            **pu.data(palette_index, linestyle="-"),
        )
    axis.set_yscale("symlog", linthresh=1.0e-4)
    pu.style_axes(
        axis,
        xlabel=r"$R/a$",
        ylabel=r"$C_G(R)$",
        title=r"Exact SU(2) PCM holonomy correlator",
        legend=True,
        grid=True,
    )
    save_figure(figure, output / "holonomy_correlator.pdf")


def write_summary(results: dict[float, dict[str, object]], output: Path) -> None:
    """Write the selection quantities and chain-aware diagnostics."""
    rows: list[dict[str, object]] = []
    for beta, result in results.items():
        chains = result["chains"]
        rows.append(
            {
                "beta": beta,
                "xi_2nd": result["xi_2nd"],
                "xi_2nd_error": result["xi_2nd_error"],
                "xi_2nd_q16": result["xi_2nd_68_percent_interval"][0],
                "xi_2nd_q84": result["xi_2nd_68_percent_interval"][1],
                "Ns_over_xi_2nd": result["Ns_over_xi_2nd"],
                "Ns_over_xi_2nd_error": result["Ns_over_xi_2nd_error"],
                "conservative_pass": result["selection_criteria"][
                    "68_percent_conservative_pass"
                ],
                "max_chain_z": result["max_pairwise_chain_disagreement_z"],
                "thermalization_cuts": ";".join(
                    str(chain["thermalization_cut"]) for chain in chains
                ),
                "block_sizes": ";".join(str(chain["block_size"]) for chain in chains),
                "n_blocks": ";".join(str(chain["n_blocks"]) for chain in chains),
            }
        )
    with (output / "campaign_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    plot_xi(results, args.output)
    plot_correlator(results, args.output)
    write_summary(results, args.output)


if __name__ == "__main__":
    main()
