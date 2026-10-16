# Numerical validation summary

A four-ensemble matched-beta/Nt diagnostic batch has completed: 16 chains
of 8,000 sweeps, using three A100s on lnode19 and leaving GPU 6 free.
All sixteen chains completed by 06:05 CEST on 16 October: 128,000 sweeps and
0.274520466 A100 GPU-hours, according to the remote completion records.
SSH is restored and compact completion records are downloaded. CPU array
291008 completed all four ensemble analyses. Collection 291015 verified
45 ensembles, 184 chains and 12,721 mass windows, with exactly 500 replicas.
The existing top-level tables contain every ensemble. Regression 291016
passed 50 tests. All 1,267 plots are title-free, footer-free and grid-free,
with single-page 16:9 geometry verified. The updated 22-page logbook passed
visual review and freshness checks. No mg_nt simulation or mass analysis
remains pending.
No new simulations are launched by this analysis.
See the [campaign report](docs/mg_nt_fixed/summary.md)
and [execution record](running_simulations.md).

Fixed-beta/Nt production has 36 of 40 chains complete and eleven accepted slab
masses. The 48,300-sweep `Nt=5,beta=10.5` analysis completed with exactly
1,000 replicas but found no stable asymptotic mass patch; `Nt=4,beta=8.4`
still fails the common blocking plateau. The cleaned comparison retains only
`p=1` fits, 64 grid-free comparison PDFs, and 30 per-ensemble first-stage
PDFs. Direct volume and statistics checks remain necessary. Exact results and
provenance are in
[`docs/beta_nt_fixed/summary.md`](docs/beta_nt_fixed/summary.md).

The separate fixed-`m_G Nt` reanalysis is complete: 38 physical ensembles
(156 chains), exactly 500 bootstrap replicas, 10 flagged preliminary tuning
points and three p=1 continuum comparisons. Omitting Nt=2 shifts the continuum
intercepts by 2.2%, 2.5% and 9.2% for c=0.5, 1 and 2; finer-resolution checks
are required before a precision continuum claim. All 53 targeted tests pass;
6,224 mass windows and 1,017 plot PDFs were verified. No new simulations were
submitted. Results, the cleaned physics tree and provenance are in the
[campaign report](docs/mg_nt_fixed/summary.md) and [PDF logbook](docs/mg_nt_fixed/logbook.pdf).

## Local removal of the auxiliary wrapping sector

All twelve subsequently approved Nt=1 A100 production chains completed:
96,000 sweeps and 3.0381 GPU-hours. Their qbig analysis uses exactly 500
replicas. That batch brought the physics tree to 41 ensembles and 168 chains;
Nt=1 matched-beta diagnostics remain separate from fixed-c continuum inputs.
Ns=2048 has preliminary xi2nd/a = 146.7 ± 33.6 and only 6--11 complete blocks
per chain, requiring more statistics and an independent volume check.
See the [campaign report](docs/mg_nt_fixed/summary.md),
[logbook](docs/mg_nt_fixed/logbook.pdf) and [execution record](running_simulations.md).

An additional user-authorized native A100 test completed successfully on
lnode19 GPU 3 (manager PID `3180516`, exit 0), using the separate
`build/klft-a100` directory, CUDA 12.6/GCC 13.3 and AMPERE80. All six GPU CTests
passed in 16.62 s and CUDA memory checking found zero errors. The eight-sweep
Nt=1 smoke run and 100-sweep tests at Ns=128, 512 and 2048 passed all output
and physical-checkpoint checks. Their measured times project to 3.859 A100
GPU-hours for twelve 8,000-sweep chains (54.18 minutes for the largest chain).
These short hot starts establish execution integrity and timing only; masses,
thermalization and finite-volume effects remain unmeasured. Compact results
are saved in [open_slab_a100_runs.json](data/mg_nt_fixed/open_slab_a100_runs.json).
No production was launched. The P100 executable is unchanged; its driver-loading
failure and validation retry are recorded below and in the
[execution record](running_simulations.md).

KLFT commits `c78efbd1ed6c60fc809db54ac9309fa532854982` (open storage) and
`f47421d895469b72d4af8a9b8479807f2b09a692` (CUDA 11.5 compatibility) are
committed and pushed to `origin/orbifold`. Dirichlet fields now allocate only physical links:
`Nt+1` spatial-link slices and `Nt` temporal-link layers. No closing temporal
link is allocated or evolved. `Nt` counts links and `L2=Nt+1` counts slices.
The shared library's periodic and orbifold modes remain available.

Historical mass results retain their meaning. With fixed tangential links on
both surfaces, the historical action factorizes as `S_old(U,V)=S_slab(U)+S_aux(V)`.
Integrating over the auxiliary links multiplies the partition function and
the numerator of every slab expectation by the same factor. The physical
distribution is therefore unchanged. Tests compare every dynamical staple,
the action, correlators, structure factors, profiles and all-height boundary
loops between open storage and a historical field with arbitrary auxiliary
links. The physical `Nt=1` PCM identity remains exact. Historical aggregate
plaquettes included the auxiliary sector; their original provenance is retained.

Validation on the final local code:

- Six 3D SU(2), OpenMP CTests pass, including statistical hot/cold and bulk
  checks and periodic/orbifold regressions (32.35 s, rerun after the CUDA fix).
- All five non-statistical CTests pass with ASan/UBSan, using
  `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1` (2.46 s, rerun).
- Before the CUDA-only initialization refactor, project Python discovery:
  97 tests, six archived-workflow skips; the
  separate fixed-beta/Nt suite has 39 passing tests. KLFT's two Python
  partition-generation tests pass.
- On `c78efbd`, an eight-sweep `Nt=2,Ns=4,beta=2.2` hot start and two-sweep restart pass.
  Compact checkpoints contain exactly 128 physical links, flag 3, exact fixed
  surfaces and normalized SU(2) matrices. Holonomy/profile/Wilson counts are
  8/4/2, and the full-height loop equals the correlator.

The audit also corrected non-transactional CPU checkpoint loading and added
guards against periodic Wilson/link and periodic-clover flow measurements
in the Dirichlet drivers. The boundary-preserving flow kernel is tested
separately. Legacy rectangular checkpoints need explicit conversion before
use with the open-slab drivers. No converter has been implemented.

The main measured bottleneck was the serial host holonomy calculation and
full-field measurement transfers. Correlators, Fourier modes and profiles now
use device reductions, verified against direct host sums. Empty spatial
update sweeps at `Nt=1` are skipped. Indicative two-thread Debug timings at
`Ns=256,Nt=1` changed from 3.75 to 0.65 seconds per holonomy measurement and
0.138 to 0.066 seconds per profile. These local timings are diagnostic;
P100 performance remains unmeasured. The direct correlator still has cubic
spatial-size scaling. Repeated allocations and per-measurement file opens
remain candidates for profiling if they matter on P100.

Local build: `build/klft`, Debug, `-Wall -Wextra`, 3D SU(2), OpenMP, statistical
tests enabled. Heatbath SHA256:
`9d2c093c2e429d172b9d331acd8526171d07d9bb25856b53c1ffd4882e446520`.
Benchmark entry point: `build/klft/tests/dirichlet_benchmark Ns Nt repeats`.
The [geometry documentation](README.md#geometry-and-action-convention)
describes the conventions.

qbig now has a clean checkout at `/qbigwork2/negro/dirichlet/src/klft`.
After the goal was resumed, a permission-gated SSH transfer supplied the
committed source bundle, SHA256
`a3ed2c98d90890e8636a5e0acd1e7faddd2ef2b45a490c73eb1e77a95837ce24`
for revision `f47421d`.
The old modified checkout is untouched; its build is preserved in
`archive/klft_builds/41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57/`
under the remote project root, with the old executable checksum unchanged.
CPU build job `290993` failed because CUDA 11.5 rejects an extended lambda
inside a constructor. The same kernel now resides in an ordinary helper;
corrected build `290994` completed with exit 0 in 4m24s on `lcpunode02`.
The sole active P100 build's heatbath SHA256 is
`856a79fada4dc2e91ac7bdbf82c347e1392729dfb214eeafd25529eb64f667dd`.
GPU validation `290995` failed on `lnode17` in 13 seconds with
`cudaErrorStubLibrary`: the executable's RUNPATH resolved CUDA's link-time
driver stub. All six processes stopped during CUDA initialization, before
numerical validation. Project commit `b314b01` makes both P100 GPU launchers
preload only the installed driver, preserving the module CUDA/GCC libraries;
a read-only loader check confirms that resolution. The unchanged binary
passed retry `290996` on lnode16 on 15 October (38 seconds): all six CTests,
CUDA memory checking and the benchmark passed. No P100 Nt=1 smoke, timing or
production job was submitted.
The [attempt record](data/mg_nt_fixed/open_slab_validation_attempts.json)
preserves the failure and corrected-script hashes.

Local project commits `8c06326` and `4f92581` add the manual CPU-build and P100-validation
submission scripts; with the `b314b01` runtime correction, five tests pass for
syntax, Slurm-only execution, resource requests, immutable provenance,
revision binding and real-driver selection. The Nt=1 input
generator has seven passing tests covering exact tuned couplings, physical
geometry, cadences, seeds and overwrite refusal. Both Slurm scripts
are transferred. Production timing must use the established five
overrelaxations per heatbath sweep; the two-overrelaxation benchmark above is
only diagnostic. The project repository now uses the same GitHub remote as
KLFT, with the user-approved independent branch `dirichlet-campaign`.
Simulation changes remain on `orbifold`; `master` is unchanged.

Project commit `08a41e4` adds a guarded Nt=1 launcher with twelve passing
execution-check tests; `87b91f8` records the sixteen immutable smoke, timing
and production inputs and the successful build provenance. All are
checksum-verified on qbig. The three exact Nt=2 tuned beta values are reused
at Nt=1, with provisional Ns=2048, 512 and 128. These are matched-beta
diagnostics, with unknown Nt=1 mass and unvalidated finite-volume effects.
They remain separate from fixed-c continuum inputs. Production requires the
manual GPU-validation, smoke and timing stages to pass. See the current
[execution record](running_simulations.md) for seeds, paths and job states.

## Fixed-beta/Nt pilot implementation and execution

KLFT commit `7eed1f7d9bb0948621a2d4020706d19c56406001` adds a separate
Dirichlet-suite cadence. A zero suite interval inherits the base measurement
interval, preserving previous inputs; an explicit interval must be an integer
multiple of the base interval. Deterministic tests verify that steps 1 through
10 yield ten holonomy measurements and profiles only at steps 5 and 10, and
also cover parsing and legacy inheritance. The full local 3D SU(2) CTest suite
passes, including the statistical test. ASan/UBSan deterministic testing
passes with leak detection disabled because LeakSanitizer cannot run under the
workspace ptrace sandbox.

The mixed pilot contains eight grid points and 32 unique hot-start chains,
not 28: all eight explicitly listed ensembles are required to retain both the
`Nt=1` matched-cadence reference and the `Ns=128` finite-volume comparison.
Every input stores `L2=Nt+1`, measures holonomy at cadence one and profiles at
cadence five, disables Wilson and horizontal-Polyakov observables, and has a
manifested SHA-256. qbig stages are submitted and inspected explicitly without
Slurm dependencies; their job IDs are appended to
`/qbigwork2/negro/dirichlet/campaigns/step3_scan/queued_jobs.tsv`.

Remote build jobs `259773` and `259782` failed before producing a binary, so
Slurm canceled both dependent chains. The first forced `nvcc_wrapper`; the
second identified the underlying node configuration: P100 login shells append
CUDA 9.2's `/usr/local/cuda/include` to `CPATH` after the correct CUDA 11.5
module paths. The build now removes only that stale entry and asserts it is
absent before configuration. It uses the normal C++ compiler with Kokkos's
compiler launcher, keeps the sole project build at
`/qbigwork2/negro/dirichlet/build/klft`, and lets all 32 one-GPU production
tasks run concurrently when P100s are available. Corrected build job `259790`
completed both Dirichlet CTests. Its sole project heatbath executable has
SHA-256 `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
Timing job `259793` measured 7.538261 seconds per 100 largest-grid sweeps and
projected 0.20939614 hours for one largest chain and 1.70332127 total P100
GPU-hours. All 32 tasks in production array `259794` started simultaneously
across `lnode13`--`lnode16` and completed with exit code zero.

The fixed-beta/Nt analysis additionally scans blocking-error curves over each full
post-cut history for every primary component and for the secondary `xi_2nd`
and `Ns/xi_2nd`. At each candidate length, every replica is blocked separately
and incomplete tails are dropped separately; only the resulting replica means
are fused. A replica-stratified delete-one-block jackknife must show a
quantitative three-length error plateau above the autocorrelation bound before
the final common block length is accepted. Final diagnostic array `259869`
found one shared plateau for every checked primary and secondary observable.
Common block lengths range from 20 to 240 sweeps and leave 39--475 complete
blocks per replica; the largest `Nt=4,Ns=192` ensemble passes the 30-block
requirement with 39 blocks but misses the 50-block target.

For `Nt=3,Ns=48,beta=2.1`, 1965 of 20000 hierarchical bootstrap samples have
negative `xi_2nd^2`. The 16th percentile remains positive, so the length is
resolved at 68% confidence, but a symmetric standard deviation cannot be
computed without truncation. The compact result therefore retains signed
`xi_2nd^2` samples, records a valid fraction of 0.90175, reports the
unconditional percentile interval, and sets the symmetric `xi_2nd` and
`Ns/xi_2nd` errors to JSON `null`. Array `259860` was superseded after adding
the fit-derived blocking audit, and its obsolete task 2 was cancelled. Final
analysis array `259877` also refits every delete-one-block correlator vector at
each candidate block length; `xi_exp` and `mass` are reported only if the
selected block ends a shared three-length error plateau. Final analysis array
`259877_[0-7]`, aggregation job `259886`, and plotting job `259887` completed
with exit code zero and no Slurm dependencies.

The measured plane is:

| beta | `Nt` | `Ns` | `xi_2nd` | `xi_exp` |
|---:|---:|---:|---:|---:|
| 2.1 | 1 | 96 | 10.622(215) | 10.394(18) |
| 2.1 | 2 | 48 | 1.201(114) | 1.190(31) |
| 2.1 | 3 | 48 | 0.580, 68% interval [0.276, 0.777] | 0.617(31) |
| 3.0 | 2 | 64 | 4.066(81) | 4.087(10) |
| 3.6 | 2 | 96 | 9.744(227) | -- |
| 3.6 | 2 | 128 | 9.561(301) | -- |
| 3.6 | 2 | 160 | 9.529(486) | -- |
| 5.4 | 3 | 144 | 16.581(631) | -- |
| 5.4 | 3 | 176 | 16.855(534) | -- |
| 7.2 | 4 | 192 | 25.177(1.044) | -- |
| 7.2 | 4 | 256 | 24.383(1.126) | -- |

At fixed `Nt=2`, beta 3.0 and 3.6 bracket `xi/a` near five. At fixed beta
2.1, `xi_2nd/Nt` falls sharply from 10.622(215) at the exact-PCM endpoint to
0.601(57) and central value 0.193 at `Nt=2,3`. Along
`beta/Nt=1.8`, equivalently `g_3^2 L=20/9`, using the volume-checked
`Ns=128` value at `Nt=2`,
`xi_2nd/Nt` is 4.780(151), 5.527(210), and 6.294(261). The revised
preliminary fit `q(Nt)=q0+c/Nt` gives `q0=7.525(453)`,
`c=-5.540(1.075)`, and `p=0.344`.

The `Ns=96` anchor passes the volume-ratio and `xi_2nd` consistency tests, but
the full-correlator comparison with `Ns=128` gives `chi2=106.78` for 48
resolved modes and `p=2.33e-6`, so it remains rejected. The follow-up
`Ns=160` ensemble gives `xi_2nd=9.529(486)`. Against `Ns=128`, the length
differs by only 0.056 combined standard deviations and 0.34%, while the full
64-mode correlator test gives `chi2=48.54` and `p=0.924`. The 68% lower
`Ns/xi_2nd` interval at `Ns=128` is 12.988. All criteria therefore pass and
`Ns=128` is accepted.

For the `Ns=160` analysis, independent MSER-5 cuts are 500 sweeps, every
primary and secondary observable shares the 30/40/60-sweep blocking-error
plateau, and the selected 60-sweep blocks leave 158 complete blocks per chain.
Diagnostics job `274303`, analysis job `274304`, summary job `274305`, and
plotting job `274306` completed with exit code zero and no dependencies. The
optional `(Nt,beta)=(6,10.8)` point now passes the preregistered finite-volume
and preliminary-scaling gates but has not been submitted. Compact results and
single-page 16:9 LaTeX PDFs are documented in
`analysis/beta_nt_fixed/beta_over_nt_1p8/validation/finite_volume_nt2/`; raw
histories remain on qbig.

## Fixed-beta/Nt corrected `Nt=3,4` volumes

The corrected `(Nt,beta,Ns)=(3,5.4,176),(4,7.2,256)` campaign used four
independent 20000-sweep hot-start chains per ensemble with the same one/five
measurement cadences. Production array `274308_[0-7]`, diagnostics array
`282238_[0-1]`, analysis array `282240_[0-1]`, summary job `282242`, and plot
job `282243` all completed with exit code zero and no Slurm dependency.

Independent MSER-5 cuts are 500 sweeps throughout. At `Nt=3`, every primary
and secondary observable shares the 80/120/160-sweep blocking-error plateau;
160-sweep blocks leave 121 complete blocks per chain. At `Nt=4`, the common
plateau is 160/240/320 sweeps and 320-sweep blocks leave 60 blocks per chain.
The maximum monitored autocorrelation times are 22.05 and 53.44 sweeps,
respectively, so both selected blocks exceed twice the autocorrelation scale.
The largest split-history/between-chain discrepancies are 2.85/2.26 standard
deviations at `Nt=3` and 2.86/2.56 at `Nt=4`.

A seeded 20000-sample hierarchical bootstrap gives
`xi_2nd=16.855(534)`, `Ns/xi_2nd=10.442(334)` at `Nt=3` and
`xi_2nd=24.383(1.126)`, `Ns/xi_2nd=10.499(496)` at `Nt=4`. The 68% lower
volume-ratio bounds are 10.120 and 10.055. No exponential fit passes all
correlated periodic-`K0`, range-stability, bootstrap-refit, and blocking
plateau gates, so both `xi_exp` values are null.

Although the pilot and corrected `xi_2nd` values differ by only 0.332 and
0.517 combined standard deviations, their full correlators are incompatible:
`chi2/dof=146.32/72`, `p=5.44e-7` at `Nt=3` and
`chi2/dof=311.13/96`, `p=1.59e-24` at `Nt=4`. The pilot volumes are therefore
not suitable for full-correlator conclusions. With the corrected points,
`xi_2nd/Nt=4.780(151),5.618(178),6.096(282)` for `Nt=2,3,4`. The preliminary
fit `q(Nt)=q0+c/Nt` gives `q0=7.356(448)`, `c=-5.160(1.076)`, and `p=0.882`.
Compact products and five one-page 16:9 PDFs are in
`analysis/beta_nt_fixed/beta_over_nt_1p8/validation/corrected_volumes_nt3_nt4/`; raw
histories remain on qbig.

## qbig environment

On 2026-07-17, Slurm reported `lnode13` through `lnode17` idle, each with eight Pascal GPUs. Direct inspection of `lnode13` identified eight idle Tesla P100-PCIE-16GB GPUs. This is an availability snapshot, not a permanent allocation.

The fresh qbig production build uses KLFT commit `2871b8abc9c28917c2049243d3483d6dada4276a`, CUDA 11.5.0, GCC 11.2.0, Kokkos CUDA/OpenMP, `Kokkos_ARCH_PASCAL60=ON`, `KLFT_NDIM=3`, and `KLFT_NC=2`. Its heatbath SHA-256 is `5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`.

## Exact-PCM beta calibration

At physical `Nt=1`, KLFT's Dirichlet slab is exactly the two-dimensional SU(2)
PCM; no separate O(4) or PCM simulator was used. Twelve hot-start chains at
`Ns=96` and beta 1.9, 2.0, 2.1 completed on qbig job `259744`. Each coupling
has four independent 10000-measurement histories, and `C_G(0)=1` holds for
every saved measurement.

Each run was thermalized and analyzed independently. The retained cut is 500
for each chain, selected after per-chain MSER-5, autocorrelation, and
split-history checks. Post-cut `tau_int[Tr G]` ranges from 1.55 to 1.72 at beta
1.9, 2.14 to 2.50 at beta 2.0, and 3.68 to 4.00 at beta 2.1. Chain-specific
block sizes are 4, 5--6, and 8, leaving at least 1187 complete blocks in every
run. The largest full-vector pairwise chain discrepancy is 3.17 standard
deviations.

A 20000-sample hierarchical bootstrap first resamples complete correlator
block vectors within a run and then resamples the four independent runs. It
therefore preserves covariance across `R` and never creates a block crossing a
run boundary. The results are:

| beta | `xi_2nd/a` | 68% interval | `Ns/xi_2nd` | 68% interval |
|---:|---:|---:|---:|---:|
| 1.9 | 6.014(126) | [5.887, 6.139] | 15.963(336) | [15.638, 16.307] |
| 2.0 | 7.609(179) | [7.429, 7.788] | 12.616(298) | [12.327, 12.922] |
| 2.1 | 10.388(125) | [10.262, 10.512] | 9.242(112) | [9.132, 9.355] |

All three pass `xi_2nd/a >= 5` and `Ns/xi_2nd >= 8`, including when the 68%
interval is required to remain inside the window. Beta 2.1 is therefore the
selected strongest tested coupling for the next finite-thickness pilot. The
outer bootstrap has only four independent run units, so its uncertainty is
less precisely determined than the large within-run block counts suggest; a
production determination should add chains rather than merely lengthening
these four.

Numerical JSON, covariance matrices, the CSV table, and one-page 16:9 LaTeX
PDFs are in `analysis/pcm_beta_pilot/`. They were produced on qbig and only the
compact 0.5 MiB result set was synchronized locally. Raw histories remain on
qbig.

## P100 smoke validation

Slurm job `259715` completed on `lnode13` with exit code 0. The deterministic suite passed. The small statistical suite reported hot/cold plaquette agreement, spatial and temporal reflection-profile differences of 0.67 and 2.04 standard deviations, the global-orientation group-averaging relation at 0.40 standard deviations, and long-slab central plaquette recovery at 0.20 standard deviations.

For eight hot-start `Ns=8`, physical `Nt=1` measurements at beta 12.15266, `C_G(0)` was exactly one configuration by configuration. The maximum absolute difference between `W_b(R,1)` and `C_G(R)` over the independent distances was exactly zero in the printed float64 output.

An `Ns=64`, physical `Nt=2` P100 timing run completed 100 heatbath sweeps, each followed by five overrelaxation sweeps and all selected measurements, in 89.42 seconds. Peak resident memory was 161912 KiB. This predicts about 1.99 hours for an 8000-sweep chain at the same thickness, before scheduler startup overhead.

The corresponding physical `Nt=12` timing run, measuring only the required boundary-loop heights `H=1` and `H=Nt`, took 106.01 seconds for 100 sweeps and 161840 KiB peak resident memory. This predicts about 2.36 hours for an 8000-sweep chain.

Twelve production chains at physical `Nt=2,6,12` completed with exit code 0. Each thickness has four independent hot starts and each chain has 8000 measurements. The `Nt=2`, `Nt=6`, and `Nt=12` task ranges were 2:06:30--2:08:29, 2:15:59--2:18:29, and 2:32:25--2:34:46, respectively. Every job's post-run checks confirmed exact `C_G(0)=1` and agreement of the full-height boundary loop with `C_G` within the printed-output tolerance.

MSER-5 thermalization diagnostics recommend retaining the preregistered 500-measurement cut for all three ensembles. Across chains, post-cut `tau_int[Tr G]` is 89--147 at `Nt=2`, 30--54 at `Nt=6`, and 8.6--10.7 at `Nt=12`; the invariant `C_G(1)` has `tau_int` below 0.76 throughout. All selected split-history z-scores are below 1.84 and the largest selected between-chain z-score is 3.18.

The full blocked analysis uses block lengths 294, 108, and 22 at Nt=2, 6, and 12, leaving 100, 276, and 1360 blocks. Its largest all-observable between-chain, split-history, and global-orientation-relation diagnostics are 3.68, 3.03, and 2.18 standard deviations, below the predeclared threshold of five. Plaquette-profile reflection discrepancies are at most 0.60, 1.02, and 1.16 standard deviations.

With the thesis normalization `(1/2)Tr G`, the hot-start traced holonomy is 0.0227(334), 0.0036(106), and 0.00406(153) at Nt=2, 6, and 12. Compared with thesis Table 4.4, the differences are -13.21, -2.86, and +1.18 combined standard deviations. The strong Nt=2 disagreement confirms the thesis's warning that its cold-start value was not thermalized on the 8000-sweep time scale.

The gauge-invariant boundary benchmark reproduces thesis Table 4.7: `W_b(64,1)=0.04138(27)` at Nt=6 and `0.03922(26)` at Nt=12, differing from the thesis by -0.65 and +0.98 combined standard deviations. The full-height boundary loop equals `C_G` configuration by configuration, while the height-one loop has the expected non-symmetric trend.

The second-moment lengths are 78.40(17), 33.07(13), and 10.27(11). At Nt=2 the length exceeds `Ns=64`, so this ensemble cannot yield a controlled mass gap. Every correlated periodic-K0 fit range tested is rejected; uncorrelated fits can look acceptable but are not used for inference because they discard strong cross-distance covariance.

Final analysis job `259734` and plotting jobs `259737` and `259738` completed with exit code 0. Compact JSON, covariance matrices, CSV, and PDFs are in `analysis/dirichlet_thesis/`; the numerical products synchronized after those jobs matched the qbig copies byte for byte. Raw histories remain only on qbig. On 2026-07-20 the PDFs were regenerated locally from the unchanged compact JSON/NPZ files to enforce one 16:9 LaTeX plot per page and the shared plotting palette. qbig was unreachable at port 22 during this presentation-only update, so the new PDFs have not yet been reproduced byte for byte on qbig.

The initial thesis-reproduction uncertainties in the preceding paragraphs use
the older pooled-block analysis. It blocks each chain before pooling, so no
block crosses a run boundary, but it does not perform the run-level
hierarchical resampling now required by this project. Treat those uncertainties
as provisional and recompute them with the chain-aware method before quoting
them in research output.
