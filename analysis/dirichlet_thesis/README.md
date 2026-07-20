# Initial thesis reproduction

This compact dataset summarizes four independent hot-start chains at each
physical slab thickness `Nt=2,6,12`, with `Ns=64`, `beta=12.15266`, 8000
measured sweeps per chain, five overrelaxation sweeps per heatbath sweep, and a
500-measurement thermalization cut. Raw histories remain on qbig below
`/qbigwork2/negro/dirichlet/data/`.

The simulation used KLFT commit `2871b8abc9c28917c2049243d3483d6dada4276a`
and the P100 executable with SHA-256
`5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`.
Production jobs were `259718` and `259723`; final analysis was job `259734`
and its original plot regeneration was job `259738`, both completed with exit
code zero. On 2026-07-20 the plots were regenerated from these unchanged
compact products with the project plotting rules below. This presentation-only
regeneration was local because qbig's SSH port was unreachable; no raw history
was transferred and `campaign_summary.csv` did not change.

## Main comparisons

The thesis values are from Tables 4.4 and 4.7 of [Marco Nebbia's
thesis](../../docs/marconebbia_tesi.pdf). `half_TrG` denotes the SU(2)-normalized
trace `(1/2)Tr G`. Errors below use independent-chain blocking and retain
Markov-chain autocorrelation.

| Nt | `half_TrG` hot | thesis cold | difference | `xi_2nd` | `W_b(64,1)` | thesis |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0227(334) | 0.537(20) | -13.21 sigma | 78.40(17) | 0.08297(22) | -- |
| 6 | 0.0036(106) | 0.076(23) | -2.86 sigma | 33.07(13) | 0.04138(27) | 0.0416(2) |
| 12 | 0.00406(153) | -0.001(4) | +1.18 sigma | 10.27(11) | 0.03922(26) | 0.0389(2) |

The Nt=2 traced holonomy does not reproduce the thesis cold-start value. This
is expected from the thesis's own slow-thermalization diagnosis: the hot result
is consistent with zero and differs by 13.2 combined standard deviations. In
contrast, the gauge-invariant boundary-loop benchmarks at Nt=6 and Nt=12 agree
with Table 4.7 by 0.65 and 0.98 combined standard deviations. This is a useful
quantitative check that does not depend on the global holonomy orientation.

The second-moment length at Nt=2 exceeds the spatial extent. Thus this coupling
and volume cannot support a controlled thin-slab mass-gap extraction, in
agreement with the limitation identified in the thesis conclusion.

## Symmetry and sampling checks

- The maximum reflection-profile discrepancy is 0.60, 1.02, and 1.16 standard
  deviations for Nt=2, 6, and 12. The central profiles approach a plateau near
  0.916 as the slab thickens, while the temporal boundary layer is near 0.924.
- `C_G(0)=1` exactly in every recorded configuration. The full-height boundary
  loop equals `C_G` configuration by configuration within the text-output
  tolerance, and its full-distance form has the expected `R <-> 64-R` echo.
- `W_b(R,1)` is not echo-symmetric and approaches 0.08297(22), 0.04138(27), and
  0.03922(26) at `R=64`, reproducing the qualitative behavior in thesis Sec.
  4.3.4.
- The largest between-chain, split-history, and SU(2) global-orientation
  relation diagnostics are 3.68, 3.03, and 2.18 standard deviations,
  respectively, below the predeclared rejection threshold of five.
- The selected block lengths are 294, 108, and 22 measurements, larger than
  twice the maximum estimated integrated autocorrelation times 146.6, 53.7,
  and 10.7. They leave 100, 276, and 1360 independent-chain blocks.

## K0 fit diagnostic

For the thesis finite-volume form

```text
C_G(R) = A [K0(mR) + K0(m(64-R))],
```

all correlated fit ranges in the scan are rejected. On `R=6..12`, the
correlated chi-squared per degree of freedom is 17.7, 49.6, and 9.90 for Nt=2,
6, and 12. The largest correlated p-value anywhere in each scan is
`1.36e-17`, `1.32e-51`, and `2.64e-8`.

Ignoring covariance gives apparently acceptable `R=6..12` fits:

| Nt | uncorrelated `m` | chi-squared/dof | p-value |
|---:|---:|---:|---:|
| 2 | 0.0005590(39) | 0.726 | 0.604 |
| 6 | 0.020855(64) | 1.890 | 0.092 |
| 12 | 0.09904(36) | 0.887 | 0.489 |

These are diagnostics, not mass determinations: the cross-distance covariance
cannot be discarded, and at Nt=2 the correlation length is larger than the
box. A lower beta and/or larger spatial volume is needed for a controlled mass
study.

## Compact products

Every figure below is a single-page 16:9 PDF with one plot, LaTeX text, and the
palette and styles from `statanalysis-Bonn/plot_utils.py`.

- [campaign summary](thesis_initial_plots/campaign_summary.csv)
- Plaquette profiles: [Nt=2](thesis_initial_plots/plaquette_profile_Nt_2.pdf),
  [Nt=6](thesis_initial_plots/plaquette_profile_Nt_6.pdf), and
  [Nt=12](thesis_initial_plots/plaquette_profile_Nt_12.pdf)
- [holonomy correlator](thesis_initial_plots/holonomy_correlator.pdf)
- [height-one boundary loop](thesis_initial_plots/boundary_loop_H_1.pdf)
- [traced-holonomy comparison](thesis_initial_plots/traced_holonomy_comparison.pdf)
- Holonomy correlation matrices: [Nt=2](thesis_initial_plots/holonomy_correlation_matrix_Nt_2.pdf),
  [Nt=6](thesis_initial_plots/holonomy_correlation_matrix_Nt_6.pdf), and
  [Nt=12](thesis_initial_plots/holonomy_correlation_matrix_Nt_12.pdf)

Each `Nt_*` directory contains the full compact JSON result, thermalization
diagnostics, and the independent-distance covariance matrix. No raw Markov
chain is stored locally.
