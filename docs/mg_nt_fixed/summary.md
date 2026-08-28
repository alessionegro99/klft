# Nonperturbative fixed-`m_G N_t` campaign

## Definition and separation

This campaign tunes the bare coupling independently at each temporal
resolution so that

\[
c=m_GN_t\in\{0.5,1.0,2.0\}.
\]

Because \(m_G=1/\xi_{\exp}\), the three targets correspond to
\(\xi_{\exp}=2N_t,N_t,N_t/2\). Here `Nt` is the physical number of temporal
links between the Dirichlet surfaces and KLFT stores `L2=Nt+1`. The boundary
conditions are temporal Dirichlet and periodic spatial.

This is campaign type `nonperturbative_mg_nt`. It is distinct from
`beta_nt_fixed`, which holds \(\beta/N_t\) fixed. Inputs, compact products, and
programs use `data/mg_nt_fixed/`, `analysis/mg_nt_fixed/`, and the
`mg_nt_fixed_` prefix. Fixed-`beta/Nt` results are allowed only as explicitly
labelled prior information.

## Status on 2026-08-28

The 25 pilot ensembles, all 100 chains, all diagnostics, and all 25 mass
analyses are complete. Every pilot point has an accepted asymptotic mass, an
accepted mass/`xi_exp` blocking plateau, at least 30 complete blocks per
chain, and 1000 finite mass replicas. Official tuning job `282495` accepted
seven of nine `(c,Nt)` groups. It rejected `(0.5,3)` because the local
log-linear fit has `p=0.009674`, and `(0.5,4)` because the target is not
bracketed.

The required first extension therefore adds `(Nt,Ns,beta)=(3,80,4.27)` and
`(4,96,5.50)`, four independent chains each. Production array
`282496_[0-7]` completed in 1:22--1:59 and all eight chains passed the
manifest-wide validator. Diagnostics array `282504_[0-1]` passed: the selected
blocks are 40 and 60 sweeps, leaving 187 and 125 blocks per chain. Mass
analysis array `282506_[0-1]` completed. The measured values are
`m_G Nt=0.515906 +/- 0.002014` with fit `p=0.6064` at `Nt=3` and
`0.531824 +/- 0.002275` with `p=0.2567` at `Nt=4`; both pass every mass and
blocking gate with 1000 finite replicas.

Round-1 tuning job `282508` then completed. It accepts `(c,Nt)=(0.5,4)` at
`beta_star=5.58394 [5.57975,5.58790]`, but `(0.5,3)` still fails the local
log-linear gate with `p=0.01350`. Eight of nine tunings now pass. No
confirmation was submitted at that stage.

Round 2 adds `(Nt,Ns,beta)=(3,80,4.290)`. Its four 8000-sweep chains pass
validation and diagnostics with 187 complete 40-sweep blocks per chain. Mass
analysis job `289402` obtains `m_G Nt=0.5051264 +/- 0.0020453`; the selected
`R=5--40` fit has recentered-bootstrap `p=0.8891`, a stable three-by-three
window patch, and 1000 finite positive replicas. Tuning job `289403` then
accepts all nine groups. The new `(c,Nt)=(0.5,3)` result is
`beta_star=4.29462 [4.29213,4.29828]`, fit `p=0.1474`, and interpolation
systematic `0.00513`. At that stage, confirmation remained unsubmitted pending
the listed statistics, volume, and final-bootstrap checks.

Those final checks are now complete. Four fresh 80000-sweep chains reduce the
beta-4.40 relative mass error from 3.14% to 0.204%. Direct `Ns=48` versus 96
checks at beta 4.95 and 5.30 find mass agreement within 0.422 and 0.730
combined standard deviations and `xi_2nd` agreement within 0.480 standard
deviations. The raw correlators separate near the `Ns=48` midpoint because
the two boxes have different periodic images; that comparison is retained as
a diagnostic rather than misused as a volume-independence gate. The larger
volumes have 68% lower bounds `Ns/xi_2nd=17.375` and 14.390.

Final tuning job `289431` accepts all nine groups. Its `c=2,Nt=4` fit remains
borderline but passes with `p=0.05155` after the three input analyses were
repeated with 10000 replicas. The tuned confirmation coupling is
`beta=4.61755576`. Four independent confirmation chains pass validation,
diagnostics, and mass analysis. They give
`m_G Nt=1.006200 +/- 0.006542`, selected-fit `p=0.4266`, and pass the
preregistered 3% agreement gate. The tuning-and-confirmation phase is
complete.

The local campaign provenance is:

- master seed: `20260724`;
- KLFT revision: `7eed1f7d9bb0948621a2d4020706d19c56406001`;
- P100 heatbath SHA-256:
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`;
- `uv.lock` SHA-256:
  `1bee9b5d384309d87510e632ea585807e4e887791bb00ab808b599cfc8bab67d`;
- analysis-code aggregate SHA-256:
  `aed1e3cb8298c26a5fbed5d3a9e1ef670b9d4ffc7c0080327fee3fa761713807`;
- sole P100 build: `KLFT_NDIM=3`, `KLFT_NC=2`, OpenMP and CUDA enabled,
  `Kokkos_ARCH_PASCAL60=ON`, GCC 11.2.0, and CUDA 11.5 at
  `/qbigwork2/negro/dirichlet/build/klft`; the complete record is in
  `data/mg_nt_fixed/pilot/build_provenance.txt`;
- histories per new point: four independent hot starts, 8000 composite sweeps
  each, with one heatbath plus five overrelaxation sweeps;
- cadence: holonomy observables every sweep and plaquette profiles every five
  sweeps; Wilson loops and horizontal Polyakov loops disabled.

## Observable audit

- \(C_G(R)\) is implemented and validated, including \(C_G(0)=1\), spatial
  reflection, chiral invariance, and the exact `Nt=1` PCM reduction.
- \(\xi_{\mathrm{2nd}}\) is computed from ensemble structure factors and has
  synthetic and finite-volume tests.
- The periodic \(K_0\) exponential fit now scans every contiguous window with
  at least five points. `Rmax` runs from `Ns/2` downward by two to the last
  value not below `ceil(Ns/3)`, and every allowed positive `Rmin` is tested.
  Parameters are estimated with diagonal covariance and refitted to joint
  vector bootstrap replicas; the complete covariance is retained for the
  post-fit statistic and its recentered-bootstrap goodness-of-fit p-value.
  Acceptance requires `p >= 0.05`, positive mass, at least 90% valid bootstrap
  refits, relative mass error below 25%, and a stable three-by-three adjacent
  `(Rmin,Rmax)` patch. Pairwise stability retains the earlier
  two-combined-standard-deviation criterion. The earliest `Rmin` patch is
  selected, with the largest `Rmax` breaking ties.
- The existing whole-history blocking-error plateau gate for both mass and
  \(\xi_{\exp}\) remains mandatory. Accepted mass replicas are saved in the
  compact NPZ output for tuning.
- \(W_b(R,H)\) exists for all heights and full height is tested against
  \(C_G\), but intermediate heights and their chain-aware statistics are not
  yet validated for production inference.
- No optimized scalar operator or validated bulk \(0^{++}\) extraction exists.
  Slice plaquette profiles do not by themselves define a bulk glueball mass.

The statistical workflow uses independent per-chain MSER-5 cuts with a
500-sweep minimum, Sokal-window autocorrelation estimates, separate
within-chain blocks, whole-history primary and secondary blocking plateaus,
and a hierarchical bootstrap preserving separation covariance. Every chain
must retain at least 30 complete blocks; 50 is the target.

## Pilot grid

Bold entries are prior information from `beta_nt_fixed`; they are excluded
from the 25 new ensembles and from the new seed manifest.

| target `c` | `Nt` | beta values | `Ns` |
|---:|---:|---|---:|
| 0.5 | 2 | 2.90, **3.00 prior**, 3.10 | 64 |
| 0.5 | 3 | 4.20, 4.40, 4.60 | 80 |
| 0.5 | 4 | 5.60, 5.90, 6.20 | 96 |
| 1.0 | 2 | 2.35, 2.50, 2.65 | 32 |
| 1.0 | 3 | 3.50, 3.70, 3.90 | 48 |
| 1.0 | 4 | 4.60, 4.95, 5.30 | 48 |
| 2.0 | 2 | 1.90, 2.00, **2.10 prior** | 32; prior uses 48 |
| 2.0 | 3 | 2.85, 3.05, 3.25 | 32 |
| 2.0 | 4 | 3.70, 4.00, 4.30 | 32 |

The nominal volume target is \(m_GN_s\ge12\). The prior points are
`Nt_2_Ns_64_b_3p0` and `Nt_2_Ns_48_b_2p1`, with their original seeds and
source analysis recorded in `data/mg_nt_fixed/pilot/prior_points.tsv`.

The gate-required round-1 extension is separate from the original pilot
manifest:

| target `c` | `Nt` | beta | `Ns` | purpose |
|---:|---:|---:|---:|---|
| 0.5 | 3 | 4.27 | 80 | resolve failed three-point log-linearity gate |
| 0.5 | 4 | 5.50 | 96 | add a lower-beta target bracket |

Its manifest SHA-256 is
`c4b4f4c8e0ac1f3186638ecd4463916bb37e9109fe80490d17c37952f8eb059a`.
Seeds are deterministic master-seed draws 105--112 and are recorded in
`data/mg_nt_fixed/extensions/round1/seed_provenance.tsv`.

## Execution and tuning gates

The first GPU job times 100 sweeps at `(Nt,Ns,beta)=(4,96,6.20)`. Production
stops if this projects above two hours for one chain or 12 total P100
GPU-hours. Every GPU task requests one typed Pascal GPU. Arrays have no
concurrency cap and no Slurm dependencies; node availability and GPU occupancy
must be inspected before each manual stage.

For each `(c,Nt)`, the local interpolation is

\[
\log[m_G(\beta,N_t)N_t]=a+b\beta,
\qquad
\beta_\star=\frac{\log c-a}{b}.
\]

Each of 1000 seeded tuning replicas draws from every point's accepted mass
replicas, refits `a,b`, and solves for `beta_star`. The reported statistical
result is the median and 16--84% interval. The interpolation systematic is the
largest absolute shift from a target-bracketing two-point log interpolation.

Another bracket point is required if a mass is absent, a chain has fewer than
30 blocks, the masses have a statistically significant monotonicity inversion,
the target is not bracketed, the three-point fit has `p < 0.05`, or fewer than
90% of tuning bootstrap fits are valid. Extrapolation is forbidden.

After all nine tunings pass, four new independent chains confirm the tuned
`Nt=4,c=1` point. The strategy is accepted only when the measured `m_G Nt`
differs from one by no more than the wider of one combined standard deviation
and 3%. The campaign then stops for review.

## Pilot tuning result

The official 1000-replica output is
`analysis/mg_nt_fixed/tuning/tuning_summary.json`. Statistical intervals are
the 16th--84th percentiles; `sys` is the maximum shift from a
target-bracketing two-point log interpolation.

| `c` | `Nt` | result | `beta_star` or failure | fit `p` | `sys` |
|---:|---:|---|---|---:|---:|
| 0.5 | 2 | pass | `2.98734 [2.98603,2.98843]` | 0.2114 | 0.00183 |
| 0.5 | 3 | fail | log-linear interpolation has `p<0.05` | 0.00967 | -- |
| 0.5 | 4 | fail | target is not bracketed | 0.5552 | -- |
| 1.0 | 2 | pass | `2.50249 [2.50063,2.50425]` | 0.0821 | 0.00389 |
| 1.0 | 3 | pass | `3.59502 [3.59184,3.59797]` | 0.3439 | 0.00199 |
| 1.0 | 4 | pass | `4.61998 [4.61375,4.62520]` | 0.5623 | 0.00086 |
| 2.0 | 2 | pass | `1.96373 [1.96123,1.96596]` | 0.0684 | 0.00735 |
| 2.0 | 3 | pass | `2.93960 [2.93715,2.94191]` | 0.5220 | 0.00141 |
| 2.0 | 4 | pass | `3.76006 [3.75529,3.76513]` | 0.05084 | 0.00387 |

The last row is close to the 5% gate but passes it. The failed rows have no
quoted `beta_star`; extrapolation is not used.

## Round-1 tuning result

The compact result is
`analysis/mg_nt_fixed/tuning_round1/tuning_summary.json`. Only the two groups
affected by the extension change:

| `c` | `Nt` | selected beta points | result | `beta_star` or failure | fit `p` | `sys` |
|---:|---:|---|---|---|---:|---:|
| 0.5 | 3 | 4.20, 4.27, 4.40 | fail | log-linear interpolation has `p<0.05` | 0.01350 | -- |
| 0.5 | 4 | 5.50, 5.60, 5.90 | pass | `5.58394 [5.57975,5.58790]` | 0.07446 | 0.00574 |

The other seven groups retain their accepted pilot values. The next gated
action is another target-bracketing point for `c=0.5,Nt=3`, not an
extrapolation. For simulation placement only, the two-point log interpolation
between beta 4.27 and 4.40 suggests beta approximately 4.290. This is not a
reported tuned beta.

## Round-2 tuning result

The compact result is
`analysis/mg_nt_fixed/tuning_round2/tuning_summary.json`. All nine groups
pass. The only changed group selects beta `4.27,4.29,4.40` and gives
`beta_star=4.29462 [4.29213,4.29828]`, fit `p=0.1474`, and interpolation
systematic `0.00513`. Beta 4.40 remains selected, so its physical statistics
must be increased before final tuning.

## Final tuning and confirmation

The compact final tuning is
`analysis/mg_nt_fixed/tuning_final/tuning_summary.json`; all nine groups pass.
For `(c,Nt)=(1,4)`, it gives
`beta_star=4.61756 [4.61267,4.62241]`, fit `p=0.3066`, and interpolation
systematic `0.00313`. The independent measurement at that coupling gives
`m_G Nt=1.006200 +/- 0.006542`. Its absolute target difference is 0.006200,
below the 0.03 gate threshold.

## Plots

The historical round-2 set contains 84 grid-free, single-page 16:9 PDFs under
`analysis/mg_nt_fixed/preliminary_plots/`. A focused checked set under
`analysis/mg_nt_fixed/final_plots/` contains the three final tuning brackets,
`beta_star/Nt` versus `1/Nt`, two direct periodic-correlator volume
comparisons, and the corresponding extracted-mass volume comparison. The
larger-volume replacements resolve the low-volume limitation in the original
`c=1,Nt=4` interpolation.

## Physics visible in the checked data

- The confirmation gives `xi_exp/xi_2nd=1.0046`, consistent with one dominant
  screening scale and the selected periodic-`K0` description at this point.
- From `Nt=2` to 4, `beta_star/Nt` decreases by 6.5%, 7.7%, and 4.3% for
  `c=0.5`, 1, and 2. These are visible cutoff effects. Three temporal
  resolutions do not justify a controlled continuum extrapolation.
- The extracted masses at beta 4.95 and 5.30 are stable from `Ns=48` to 96.
  The raw correlators separate only near the smaller-box midpoint, as expected
  from the different periodic images.
- The plaquette profiles show the fixed boundary layers and reflection
  symmetry. A quantitative boundary-energy or profile screening scale still
  requires the deferred chain-aware profile covariance analysis and a stated
  subtraction or renormalization prescription.

## Job record

- Timing job `282244` completed on `lnode13` with exit code zero. One hundred
  sweeps at `(Nt,Ns,beta)=(4,96,6.20)` took 1.614251 s, projecting 0.03587224
  GPU-hours for the largest chain and 0.93835035 total P100 GPU-hours. The
  dimension, Dirichlet, cadence, output-count, `C_G(0)=1`, disabled-output, and
  executable/input/dependency hash checks passed.
- Audit diagnostics array `282245_[0-1]` completed with exit code zero. Both
  prior ensembles use four independent 500-sweep cuts and 20-sweep blocks,
  leaving 475 complete blocks per chain.
- Audit analysis array `282247_[0-1]` failed before producing compact results
  because it encountered an incomplete intermediate source update. Array
  `282254_[0-1]` was cancelled after 8:18 without compact results when its
  full-covariance parameter fit was superseded by the repository's required
  diagonal-fit/full-covariance-GOF convention. The logs are retained.
- Corrected dependency-free audit array `282256_[0-1]` completed with exit
  code zero in 36:07 and 1:06:55. At beta 3.0 the new mass is
  `0.2447365(775)`, differing from the prior result by 0.063 combined standard
  deviations; at beta 2.1 it is `0.82720(124)`, differing by 0.535 combined
  standard deviations. Both fits have 1000/1000 successful mass and GOF
  refits, recentered-bootstrap p-values 0.189 and 0.326, accepted mass and
  `xi_exp` blocking plateaus, and 475 blocks per chain. Summary job `282320`
  completed and wrote `gate PASS`.
- Before production, all 100 input hashes, the KLFT commit and executable,
  dependency lock, analysis-code hash, available space, queue allocations,
  and physical GPU occupancy on `lnode13`--`lnode17` were checked. Another
  scheduler array occupied 36 P100s; GPUs 0--3 on `lnode14` were the four
  verified free devices.
- Production array `282323_[0-99]` was submitted without a dependency or
  concurrency cap. It was monitored live for more than one minute. Tasks
  0--3 completed on `lnode14` with exit code zero in 1:02--1:05 while tasks
  4--7 began. The completed task-0 history has exactly 8000 holonomy summaries,
  264000 correlator rows, and 8000 profile rows; its dimensions, seed,
  cadences, hashes, exact `C_G(0)=1`, and disabled-output checks all pass.
- All 100 production tasks completed with exit code zero. The manifest-wide
  read-only validator passed all 100 chains: complete files, exact metadata
  and input hashes, complete measurement grids and counts, exact `C_G(0)=1`,
  and no disabled output. Dependency-free diagnostics array
  `282445_[0-24]` was then submitted; 22 tasks started immediately and three
  remained scheduler-pending.
- All 25 diagnostics tasks completed with exit code zero. Every chain uses a
  500-sweep cut, every primary and secondary observable has an accepted
  three-length blocking plateau, and selected blocks are longer than twice
  the largest required autocorrelation time. Retained counts range from 46
  to 375 blocks per chain. Only `Nt=4,Ns=96,beta=6.20` misses the 50-block
  target, with 46; it remains above the hard floor of 30. Dependency-free
  analysis array `282470_[0-24]` was submitted after this inspection.
- Analysis array `282470_[0-24]` completed with exit code zero. All 25 compact
  results were synchronized and inspected: every mass is accepted, every
  mass/`xi_exp` plateau passes, and each covariance archive contains 1000
  finite mass replicas. Tuning job `282495` then completed in 15 seconds and
  produced the seven-pass/two-fail result above.
- Round-1 extension inputs were generated only after the tuning failures.
  Before submission, all eight input hashes, the unchanged KLFT and heatbath
  hashes, raw-output absence, Slurm allocations, and direct GPU occupancy on
  `lnode13`--`lnode17` were checked. Production array `282496_[0-7]` used one
  typed Pascal GPU per task, no dependency, and no concurrency cap. It was
  monitored live beyond one minute, completed with exit code zero, and passed
  the eight-chain validator. Diagnostics array `282504_[0-1]` completed with
  exit code zero. Analysis array `282506_[0-1]` completed in 47:59 and 1:12:40
  with exit code zero; both compact results passed validation. Tuning job
  `282508` completed in 15 seconds and produced the eight-pass/one-fail result
  above.

## Preliminary-statistics checklist

The current numbers are suitable for tuning decisions, but they are pilot
results rather than final production measurements.

- [x] Analyze only complete immutable histories; keep all chains separate
  through cuts, blocking, and hierarchical bootstrap.
- [x] Require at least 30 complete blocks per chain, accepted primary and
  secondary blocking plateaus, and 1000 finite mass replicas.
- [x] Record the two round-1 extension points as preliminary, despite their
  good 187/125 blocks per chain and sub-percent mass errors.
- [x] Increase independent Monte Carlo statistics at
  `Nt=3,Ns=80,beta=4.40`; four fresh 80000-sweep chains reduce the relative
  mass error from 3.14% to 0.204%.
- [x] Add and analyze the target-bracketing `c=0.5,Nt=3` point at beta 4.290.
  Round-2 tuning passes all nine groups; the resulting beta remains
  preliminary until the remaining checks below pass.
- [ ] Increase statistics for `Nt=4,Ns=96,beta=6.20` if it is used in a final
  result: 46 blocks per chain pass the hard floor but miss the target of 50.
- [x] Directly volume-check and replace the two low-volume `c=1,Nt=4`
  interpolation inputs with accepted `Ns=96` results.
- [x] Repeat all three `c=2,Nt=4` input analyses with 10000 bootstrap
  replicas; the final interpolation passes with `p=0.05155`.
- [x] Run the independent `Nt=4,c=1` confirmation after all nine tunings
  pass; `m_G Nt=1.006200 +/- 0.006542` passes the 3% gate.

Increasing bootstrap replicas only reduces resampling noise; it does not add
physical statistics. The latter requires longer independent histories or new
independent chains.

## Deferred roadmap

The `Nt=6` pilot, Wilson-loop cadence changes, all-height and intermediate
height validation, chain-aware Wilson/profile residual analysis, the explicit
`Nt=4,c=0.5` volume check, scalar-profile covariance fit, and continuum
analysis remain deferred pending a new roadmap decision. `m_0++` remains null
unless an independent bulk glueball extraction is validated.
