# Fixed-beta/Nt `Ns=160` finite-volume check

This directory contains the compact products from the follow-up
`Nt=2`, beta 3.6 volume check. Raw histories remain on qbig.

## Result

Each volume has four independent hot-start chains of 10000 composite sweeps.
The holonomy cadence is one sweep and the plaquette-profile cadence is five
sweeps.

| `Ns` | `xi_2nd` | 68% interval | `Ns/xi_2nd` | `xi_exp` |
|---:|---:|---:|---:|---:|
| 96 | 9.744(227) | [9.515, 9.966] | 9.85 | -- |
| 128 | 9.561(301) | [9.259, 9.855] | 13.388(424) | -- |
| 160 | 9.529(486) | [9.036, 10.006] | 16.792(877) | -- |

For `Ns=160`, all 20000 bootstrap samples have positive `xi_2nd^2`. No
correlated periodic-K0 fit satisfies all fit-quality, adjacent-range, and
fit-derived blocking-plateau gates, so `xi_exp` remains null.

The preregistered `Ns=128` versus 160 comparison passes every criterion:

- the 68% lower interval for `Ns/xi_2nd` at `Ns=128` is 12.988, above 8;
- the two `xi_2nd` estimates differ by 0.056 combined standard deviations and
  0.34%, below the limits of 2 standard deviations and 5%;
- the full correlator over `R=1,...,64` gives `chi2=48.54` for 64 resolved
  covariance modes and `p=0.924`.

Thus `Ns=128` is accepted for the `Nt=2`, beta 3.6 point. The nominal
`10 xi_2nd` rule rounds to `Ns=96`, but the direct volume test rejects that
smaller lattice, so the empirical lower bound `Ns=128` takes precedence.

Using the accepted `Ns=128` value, the preliminary fixed-beta/Nt fit
`q(Nt)=q0+c/Nt` for `q=xi_2nd/Nt` gives
`q0=7.525(453)`, `c=-5.540(1.075)`, and `p=0.344`. This is preliminary
scaling evidence from three lattice spacings, not a continuum result. The
finite-volume and scaling gates now permit considering the optional
`(Nt,beta)=(6,10.8)` point; it has not been submitted.

## Statistical diagnostics

MSER-5 selects the 500-sweep minimum cut independently for all four chains.
Across the chains, the largest required autocorrelation is
`tau_int[Tr G]=6.33` sweeps. Every primary correlator/profile component and
both secondary observables share the blocking-error plateau at 30, 40, and 60
sweeps. The selected 60-sweep blocks are longer than `2 tau_int` and leave 158
complete blocks in every chain. The largest split-history discrepancy is
2.25 standard deviations; the largest blocked between-chain discrepancy is
2.64 standard deviations.

The full diagnostics, including every observable's blocking curve, are in
[`thermalization_diagnostics.json`](Nt_2_Ns_160_b_3p6/thermalization_diagnostics.json).
The final estimates are in
[`analysis.json`](Nt_2_Ns_160_b_3p6/analysis.json), with covariance and seeded
bootstrap arrays in `covariance.npz`.

## Plots and tables

- [`finite_volume_correlator_Ns_96_128_160.pdf`](plots/finite_volume_correlator_Ns_96_128_160.pdf)
- [`finite_volume_xi_2nd.pdf`](plots/finite_volume_xi_2nd.pdf)
- [`plaquette_profile_Nt_2_Ns_160_b_3p6.pdf`](plots/plaquette_profile_Nt_2_Ns_160_b_3p6.pdf)
- [`fixed_beta_over_nt_xi_2nd_over_Nt_fv_checked.pdf`](plots/fixed_beta_over_nt_xi_2nd_over_Nt_fv_checked.pdf)
- [`finite_volume_points.csv`](finite_volume_points.csv),
  [`finite_volume_summary.json`](finite_volume_summary.json), and the
  corresponding NPZ products

Every PDF is a single-page 16:9 LaTeX figure using the shared plotting style.

## Provenance

Production array `274107_[0-3]`, diagnostics job `274303`, analysis job
`274304`, summary job `274305`, and plotting job `274306` completed with exit
code zero and no Slurm dependency. Analysis used project commit `af5fe72` and
bootstrap seed 20260723. The simulation executable is the sole project P100
build from KLFT commit `7eed1f7d9bb0948621a2d4020706d19c56406001`, SHA-256
`41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
The seeds are 98728066, 856366050, 397310628, and 411907572. Raw output remains
under `/qbigwork2/negro/dirichlet/data/Nt_2_Ns_160_b_3p6/`.
