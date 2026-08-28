# Fixed-beta/Nt mixed pilot

This directory contains the compact numerical products from the eight-point
pilot in the `(beta, Nt)` plane. Raw Monte Carlo histories remain on qbig.

## Plane covered

Each point has four independent hot-start chains of 10000 composite sweeps.
Holonomy observables were measured every sweep and plaquette profiles every
five sweeps.

| trajectory | `Nt` | beta | `Ns` | `g_3^2 L=4Nt/beta` | `xi_2nd` | 68% interval | `xi_exp` |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed beta endpoint | 1 | 2.1 | 96 | 1.90476 | 10.622(215) | [10.403, 10.840] | 10.394(18) |
| fixed beta | 2 | 2.1 | 48 | 3.80952 | 1.201(114) | [1.083, 1.307] | 1.190(31) |
| fixed beta | 3 | 2.1 | 48 | 5.71429 | 0.580 | [0.276, 0.777] | 0.617(31) |
| fixed `Nt=2` | 2 | 3.0 | 64 | 2.66667 | 4.066(81) | [3.985, 4.146] | 4.087(10) |
| fixed beta/Nt anchor | 2 | 3.6 | 96 | 2.22222 | 9.744(227) | [9.515, 9.966] | -- |
| anchor volume check | 2 | 3.6 | 128 | 2.22222 | 9.561(301) | [9.259, 9.855] | -- |
| fixed beta/Nt | 3 | 5.4 | 144 | 2.22222 | 16.581(631) | [15.935, 17.204] | -- |
| fixed beta/Nt | 4 | 7.2 | 192 | 2.22222 | 25.177(1.044) | [24.118, 26.199] | -- |

For `Nt=3`, beta 2.1, 90.175% of bootstrap samples have positive
`xi_2nd^2`. The quoted interval is the unconditional percentile interval;
the symmetric error is intentionally omitted. A missing `xi_exp` means that
the correlated periodic-K0 fit did not satisfy the preregistered fit-quality
and adjacent-range-stability requirements.

## Conclusions

- At fixed `Nt=2`, `xi_2nd` increases from 1.201(114) at beta 2.1 to
  4.066(81) at beta 3.0 and 9.744(227) at beta 3.6. Thus beta 3.0 and 3.6
  bracket the target `xi/a` near five; no additional interpolated coupling is
  required.
- At fixed beta 2.1, `xi_2nd/Nt` is 10.622(215), 0.601(57), and has central
  value 0.193 at `Nt=1,2,3`. The endpoint is the exact-PCM reference; the
  finite-thickness points rapidly enter the short-correlation-length regime.
- Along `beta/Nt=1.8`, equivalently `g_3^2 L=20/9`, `xi_2nd/Nt` is
  4.872(114), 5.527(210), and
  6.294(261) at `Nt=2,3,4`. The fit `q(Nt)=q0+c/Nt` gives
  `q0=7.383(432)`, `c=-5.056(970)`, and `p=0.288`. With only three lattice
  spacings this is preliminary scaling evidence, not a continuum result.
- The `Ns=96` anchor passes the volume-ratio and `xi_2nd` consistency tests
  against `Ns=128`, but the full-correlator comparison gives
  `chi2=106.78` for 48 resolved modes and `p=2.33e-6`. The anchor is rejected;
  an `Nt=2`, beta 3.6, `Ns=160` four-chain check is required before
  production. The optional `(Nt,beta)=(6,10.8)` point is therefore deferred.
- All primary and secondary observables have a common whole-history blocking
  plateau. Every chain retains at least 30 complete blocks; the `Nt=4` point
  retains 39 and therefore misses only the target of 50.

## Follow-up resolution

The required `Nt=2`, beta 3.6, `Ns=160` ensemble has now been completed. Its
comparison with `Ns=128` passes the volume-ratio, `xi_2nd`, and full-correlator
criteria (`chi2=48.54` for 64 modes, `p=0.924`), so `Ns=128` is accepted.
The compact follow-up analysis and revised scaling fit are in
[`../beta_over_nt_1p8/finite_volume_nt2/`](../beta_over_nt_1p8/finite_volume_nt2/README.md).

## Plots and tables

- [`xi_2nd_vs_beta_Nt_2.pdf`](plots/xi_2nd_vs_beta_Nt_2.pdf): fixed-`Nt=2`
  beta scan and the duplicate-volume anchor.
- [`fixed_beta_xi_2nd_over_Nt.pdf`](plots/fixed_beta_xi_2nd_over_Nt.pdf):
  fixed-beta thickness scan.
- [`fixed_beta_over_nt_xi_2nd_over_Nt.pdf`](plots/fixed_beta_over_nt_xi_2nd_over_Nt.pdf):
  fixed-beta/Nt scaling fit of the second-moment length.
- [`anchor_finite_volume_correlator.pdf`](plots/anchor_finite_volume_correlator.pdf):
  full-correlator finite-volume comparison.
- `plots/plaquette_profile_*.pdf`: one spatial/temporal profile for every
  ensemble.
- [`campaign_summary.csv`](campaign_summary.csv) and
  [`campaign_summary.json`](campaign_summary.json): portable numerical table
  and campaign decisions. Each ensemble directory contains its full
  diagnostics, result, and covariance/bootstrap NPZ file.

All PDFs are single-page 16:9 LaTeX figures produced on qbig. Production array
`259794`, final analysis array `259877`, aggregation job `259886`, and plotting
job `259887` completed with exit code zero. The simulation executable was
built from KLFT commit `7eed1f7d9bb0948621a2d4020706d19c56406001` and has
SHA-256 `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
