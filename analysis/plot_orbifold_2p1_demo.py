"""Plot accepted 2+1D demonstration tables with the shared Bonn style.

No fitting is performed here: the input potential must come from the
complete-covariance fit in orbifold_static_potential.py. Each PDF contains
one plot on an uncropped 16:9 canvas. Requires numpy and matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Plot the potential and one independently inspectable time plateau."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("potential", type=Path)
    parser.add_argument("--style-dir", type=Path, required=True)
    parser.add_argument("--effective-r", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style_dir.resolve()))
    import plot_utils as style

    table = np.loadtxt(args.potential, dtype=np.float64, ndmin=2)
    if table.shape[1] != 12 or not np.all(np.isfinite(table)):
        raise ValueError("expected the finite 12-column potential table")
    if np.any(table[:, 5] <= 0.0):
        raise ValueError("potential uncertainties must be positive")
    rows = table[table[:, 0] == args.effective_r]
    if len(rows) != 1:
        raise ValueError("effective-r must have exactly one accepted plateau")
    potential_pdf = args.potential.with_suffix(".pdf")
    effective_pdf = args.potential.with_name(
        f"effective_potential_R{args.effective_r:02d}.pdf")
    for path in (potential_pdf, effective_pdf):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    title = r"Finite-mass SU(3) orbifold HMC: $2+1$ dimensions, $32^3$"
    caption = (r"$a_s=a_t=0.2,\ g=1,\ m=m_{\mathrm{U(1)}}=40$"
               r"; periodic, full unfixed action; unsmeared loops")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.89)
    ax.errorbar(table[:, 0], table[:, 4], yerr=table[:, 5],
                label="Correlated time-plateau estimates", **style.data(1))
    style.style_axes(ax, xlabel=r"Separation $R=r/a_s$",
                     ylabel=r"Static potential $a_t V(R)$", title=title,
                     legend=True)
    fig.text(0.5, 0.035, caption, ha="center", fontsize=20)
    fig.savefig(potential_pdf)
    plt.close(fig)

    effective = np.loadtxt(args.potential.with_suffix(".effective.tsv"),
                           dtype=np.float64, ndmin=2)
    # Undefined logarithms or bootstrap ratios are not plotted or imputed.
    selected = effective[(effective[:, 0] == args.effective_r)
                         & np.isfinite(effective[:, 4])
                         & np.isfinite(effective[:, 5])
                         & (effective[:, 6] == 1.0)]
    if len(selected) < 2:
        raise ValueError("too few fully positive effective-potential ratios")
    row = rows[0]
    lo, hi = row[2] + 0.5, row[3] + 0.5
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.89)
    ax.errorbar(selected[:, 2], selected[:, 4], yerr=selected[:, 5],
                label=r"$\log[W(R,T)/W(R,T+1)]$", **style.data(1))
    ax.plot([lo, hi], [row[4], row[4]], **style.fit(2),
            label=f"Correlated plateau: T={int(row[2])}--{int(row[3])}")
    ax.fill_between([lo, hi], row[4] - row[5], row[4] + row[5],
                    **style.conf_band(2))
    style.style_axes(ax, xlabel=r"Time midpoint $T+1/2$",
                     ylabel=r"Effective potential $a_t V_{\mathrm{eff}}$",
                     title=title + rf", $R={args.effective_r}$", legend=True)
    fig.text(0.5, 0.035, caption, ha="center", fontsize=20)
    fig.savefig(effective_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
