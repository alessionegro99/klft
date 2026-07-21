# Exact-PCM beta calibration pilot

This compact result set calibrates the physical `Nt=1` Dirichlet slab, which is
exactly the two-dimensional SU(2) Principal Chiral Model. The simulations use
KLFT directly; no O(4) or separate PCM implementation enters the calculation.

## Campaign

- Geometry: physical `Nt=1`, stored temporal extent 2, `Ns=96`
- Couplings: beta 1.9, 2.0, 2.1
- Starts: four independent hot chains per coupling
- Seeds: 32091, 90517, 177013, 515029
- History: 10000 heatbath sweeps per chain, each followed by five
  overrelaxation sweeps; every sweep measured
- Production job: qbig Slurm array `259744_[0-11]`
- KLFT: commit `2871b8abc9c28917c2049243d3483d6dada4276a`
- P100 heatbath SHA-256:
  `5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`

Raw histories remain below `/qbigwork2/negro/dirichlet/data/`; none are stored
here.

## Statistical method

Each chain has its own thermalization diagnostic, autocorrelation estimate,
and blocking operation. A 500-measurement cut is retained independently for
all chains after MSER-5 and post-cut stationarity checks. Complete block vectors
are resampled within each chain, and independent chains are then resampled at a
second level. The 20000-replicate hierarchical bootstrap uses seed 20260720 and
preserves the covariance among all independent `R` values. Independent raw
histories are never concatenated.

The largest post-cut split-history discrepancy is 2.54 standard deviations;
the largest pairwise chain discrepancy over the full analyzed vector is 3.17.
Every chain retains at least 1187 complete blocks. With only four independent
chains, however, the outer-level uncertainty itself is resolved less well than
the within-chain sample sizes might imply.

## Result

| beta | `xi_2nd/a` | 68% interval | `Ns/xi_2nd` | 68% interval |
|---:|---:|---:|---:|---:|
| 1.9 | 6.014(126) | [5.887, 6.139] | 15.963(336) | [15.638, 16.307] |
| 2.0 | 7.609(179) | [7.429, 7.788] | 12.616(298) | [12.327, 12.922] |
| 2.1 | 10.388(125) | [10.262, 10.512] | 9.242(112) | [9.132, 9.355] |

All tested couplings satisfy `xi_2nd/a >= 5` and `Ns/xi_2nd >= 8`, including
the uncertainty-aware criterion. Beta 2.1 is the selected strongest tested
coupling for the finite-thickness pilot.

`plots/xi_2nd_vs_beta.pdf` and `plots/holonomy_correlator.pdf` are cluster-made,
single-page 16:9 PDFs using LaTeX and the pinned `statanalysis-Bonn` palette at
commit `db0086259f1d33a443090ae2bb5f4c0a6ae1373d`. `plots/campaign_summary.csv`
contains the machine-readable selection table; each ensemble directory holds
the diagnostics, numerical result, and full independent-distance covariance.
