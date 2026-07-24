# Step-3 corrected constant-coupling volumes

This directory contains the compact analysis of the corrected
`lambda_tree^2=20/9` points. The physical slab has temporal Dirichlet
boundaries and periodic spatial directions; KLFT stores `L2=Nt+1`. Each
ensemble has four independent hot-start chains of 20000 composite sweeps. A
composite sweep is one heatbath sweep followed by five overrelaxation sweeps.
Holonomy observables were measured every sweep and plaquette profiles every
five sweeps.

Raw histories remain on qbig. Only JSON, CSV, NPZ, and PDF products were
synchronized here after a dry run.

## Results

| `Nt` | beta | `Ns` | `xi_2nd/a` | `Ns/xi_2nd` | 68% lower `Ns/xi_2nd` | `xi_exp/a` |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 5.4 | 176 | 16.855(534) | 10.442(334) | 10.120 | -- |
| 4 | 7.2 | 256 | 24.383(1.126) | 10.499(496) | 10.055 | -- |

Both volume-ratio lower bounds exceed eight. No exponential length passes all
correlated periodic-`K0`, bootstrap-refit, adjacent-range, and blocking-error
plateau gates, so `xi_exp` and the mass are JSON `null` rather than estimates
from a rejected fit.

The pilot and corrected second-moment lengths remain statistically compatible:

| `Nt` | pilot `Ns` | corrected `Ns` | relative `xi_2nd` shift | combined-distance `z` |
|---:|---:|---:|---:|---:|
| 3 | 144 | 176 | 1.63% | 0.332 |
| 4 | 192 | 256 | 3.25% | 0.517 |

The full correlators are not volume-stable. For `Nt=3`, the common 72-mode
comparison gives `chi2=146.32`, `p=5.44e-7`; for `Nt=4`, the common 96-mode
comparison gives `chi2=311.13`, `p=1.59e-24`. The old `Ns=144,192` points are
therefore rejected for full-correlator work. This test shows that the repeats
were necessary; because no volume larger than `Ns=176,256` was simulated, it
does not by itself constitute a second direct infinite-volume check of the new
points.

Using the volume-checked `Nt=2,Ns=128` result and the two corrected points,

| `Nt` | `xi_2nd/Nt` |
|---:|---:|
| 2 | 4.780(151) |
| 3 | 5.618(178) |
| 4 | 6.096(282) |

the preliminary fit `q(Nt)=q0+c/Nt` gives
`q0=7.356(448)`, `c=-5.160(1.076)`, `chi2=0.02197` for one degree of freedom,
and `p=0.882`. This is preliminary scaling evidence, not a continuum
determination.

## Statistical checks and provenance

MSER-5 selected the minimum 500-sweep cut for every chain. At `Nt=3`, the
largest monitored per-chain autocorrelation times are 20.08, 20.26, 20.94,
and 22.05 sweeps. Every primary and secondary blocking error shares the
80/120/160-sweep plateau; 160-sweep blocks leave 121 complete blocks per
chain. The largest split-history and between-chain discrepancies are 2.85 and
2.26 standard deviations.

At `Nt=4`, the corresponding largest autocorrelation times are 49.40, 41.71,
53.44, and 37.97 sweeps. The common plateau is 160/240/320 sweeps; 320-sweep
blocks leave 60 complete blocks per chain. The largest split-history and
between-chain discrepancies are 2.86 and 2.56 standard deviations. No block
crosses a replica boundary. The final covariance and derived quantities use a
seeded 20000-sample hierarchical bootstrap of complete within-chain vectors.

Production array `274308_[0-7]`, diagnostics array `282238_[0-1]`, analysis
array `282240_[0-1]`, summary job `282242`, and plotting job `282243` all
completed with exit code zero and no Slurm dependency. Analysis workflow
commit: `bda1a32`. KLFT commit:
`7eed1f7d9bb0948621a2d4020706d19c56406001`; heatbath SHA-256:
`41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.

## Plots

- [Corrected constant-coupling scaling](plots/constant_lambda_xi_2nd_over_Nt_corrected.pdf)
- [`Nt=3`: `Ns=144` versus 176 correlator](plots/finite_volume_correlator_Nt_3_Ns_144_176.pdf)
- [`Nt=4`: `Ns=192` versus 256 correlator](plots/finite_volume_correlator_Nt_4_Ns_192_256.pdf)
- [`Nt=3,Ns=176` plaquette profile](plots/plaquette_profile_Nt_3_Ns_176.pdf)
- [`Nt=4,Ns=256` plaquette profile](plots/plaquette_profile_Nt_4_Ns_256.pdf)

Every PDF is one 16:9 LaTeX-rendered plot using the shared `plot_utils` style.
