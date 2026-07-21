# Running simulations

## Step-3 Dirichlet slab parameter-space pilot

- Status: second retry prepared. Build jobs `259773` and `259782` failed before
  producing an executable, and Slurm canceled their dependent chains. The
  first forced `nvcc_wrapper`; the second exposed the underlying node setup:
  P100 login shells append CUDA 9.2's `/usr/local/cuda/include` to `CPATH` even
  after loading CUDA 11.5. The build now removes only that stale entry and
  asserts it is absent before using the normal C++ compiler with Kokkos's
  compiler launcher. Retry job IDs are written to
  `/qbigwork2/negro/dirichlet/campaigns/step3_scan/queued_jobs.tsv`
- Grid: eight ensembles and four independent hot-start chains per ensemble
  (32 chains): `(Nt,beta,Ns)=(1,2.1,96),(2,2.1,48),(3,2.1,48),`
  `(2,3.0,64),(2,3.6,96),(3,5.4,144),(4,7.2,192),`
  `(2,3.6,128)`
- Histories: 10000 composite sweeps per chain; one heatbath plus five
  overrelaxation sweeps; globally unique explicit seeds in `manifest.tsv`
- Cadence: holonomy correlator, Fourier components, and summary every sweep;
  spatial and temporal plaquette profiles every five sweeps
- Disabled observables: boundary Wilson loops, generic temporal/planar Wilson
  loops, horizontal Polyakov loops, PCM matching, and effective-action output
- Inputs and immutable manifest: local `data/step3_scan/`; remote
  `/qbigwork2/negro/dirichlet/campaigns/step3_scan/`
- Raw output: `/qbigwork2/negro/dirichlet/data/Nt_<Nt>_Ns_<Ns>_b_<beta>/seed_<seed>/`
- Logs: `/qbigwork2/negro/dirichlet/logs/step3-*.out`
- Slurm request: one typed Pascal GPU per build, timing, or production task;
  production array `0-31` with no concurrency cap, allowing Slurm to use all
  available P100s across the eligible nodes; no comma-separated node list
- Timing gate: 100 sweeps at `Nt=4,Ns=192,beta=7.2`; dependent production is
  released only when the projection is at most two hours for one chain and at
  most 12 total P100 GPU-hours
- KLFT: commit `7eed1f7d9bb0948621a2d4020706d19c56406001`, fresh
  CUDA/OpenMP `KLFT_NDIM=3`, `KLFT_NC=2`, Pascal60 build at the single project
  path `/qbigwork2/negro/dirichlet/build/klft`;
  executable checksum is recorded by the build job and in every chain's
  metadata
- Analysis: independent MSER-5 cuts (minimum 500 sweeps), Sokal-window
  autocorrelations in sweep units, whole-history blocking-error curves for all
  primary and secondary observables, replica-stratified fusion only after
  separate blocking, a common error-plateau block longer than `2 tau_int`, at
  least 30 blocks per chain, and a seeded 20000-sample hierarchical bootstrap
- Compact output: `/qbigwork2/negro/dirichlet/analysis/step3_scan/`; raw
  histories are not synchronized

## Exact-PCM beta calibration pilot

- Status: completed and analyzed
- Purpose: choose a coupling in the exact physical-`Nt=1` SU(2) PCM limit with
  `xi_2nd/a >= 5` and `Ns/xi_2nd >= 8`, before repeating the finite-thickness
  thesis scan
- Ensembles: physical `Nt=1`, `Ns=96`, beta = 1.9, 2.0, 2.1, hot starts
- Seeds at each coupling: 32091, 90517, 177013, 515029
- History per chain: 10000 sweeps, five overrelaxation sweeps per
  heatbath sweep, measurement interval 1
- Observables: the normalized single-trace holonomy correlator and its
  zero/minimum-momentum summary; at `Nt=1` these are exactly the SU(2) PCM spin
  correlator and structure factors
- Timing check: Slurm job `259743_0`, beta 1.9, seed 700019, 100 sweeps with
  otherwise identical geometry and measurements; it completed on `lnode13`
  with 100 measurements, exact `C_G(0)=1`, and 1.45 seconds in heatbath
- Slurm request: one typed Pascal GPU per task on `lnode13`--`lnode17`, three
  CPU cores, 4 GiB
- Production: array `259744_[0-11]` completed on `lnode13` and `lnode14` with
  exit code zero in 1:56--2:40 per task. Every chain has exactly 10000 summary
  rows and exact `C_G(0)=1` at every measurement.
- Remote data: `/qbigwork2/negro/dirichlet/data/Nt_1_Ns_96_b_<beta>/seed_<seed>`
- Inputs and manifest: `/qbigwork2/negro/dirichlet/campaigns/pcm_beta_pilot/`
- Logs: `/qbigwork2/negro/dirichlet/logs/pcm-beta-<job>_<task>.out`
- KLFT source/build: commit
  `2871b8abc9c28917c2049243d3483d6dada4276a`,
  `/hadron/negro/build/klft-p100-3D-su2`, heatbath SHA-256
  `5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`
- Analysis rule: select thermalization and estimate `tau_int` independently for
  each chain; block and bootstrap only within each chain; combine runs with a
  hierarchical bootstrap carrying the full `R` vector. Raw histories are never
  concatenated before resampling.
- Thermalization: diagnostic array `259756_[0-2]` completed. MSER-5 candidates
  were at most 65 measurements; the conservative minimum cut of 500 was
  retained independently for every chain. The largest post-cut split-history
  discrepancy is 2.54 standard deviations.
- Analysis: initial array `259759_[0-2]` failed because a scalar/array shape
  check incorrectly rejected exact `C_G(0)` data. Commit `ddafade` fixed the
  check; retry `259762_[0-2]` completed in 41--59 seconds. The 20000-sample
  hierarchical bootstrap uses seed 20260720.
- Blocking: beta 1.9 uses four blocks sizes of 4 (2375 blocks per chain); beta
  2.0 uses sizes 5, 5, 6, 5 (1900, 1900, 1583, 1900 blocks); beta 2.1 uses size
  8 (1187 blocks per chain). These are all longer than twice the corresponding
  maximum selected `tau_int`.
- Plotting: jobs `259765` and `259766` failed respectively because the remote
  `statanalysis-Bonn` checkout and the current shared plot helper were absent.
  The dependency was pinned at commit
  `db0086259f1d33a443090ae2bb5f4c0a6ae1373d`; job `259767` completed, and job
  `259768` completed the final presentation-only clarification. Both PDFs are
  one 1152-by-648-point page.
- Compact outputs: `/qbigwork2/negro/dirichlet/analysis/pcm_beta_pilot/` and
  local `analysis/pcm_beta_pilot/`. The synchronized products total about
  0.5 MiB; the roughly 323 MiB of raw histories remain only on qbig.
- Result: beta 1.9, 2.0, and 2.1 give respectively
  `xi_2nd = 6.014(126), 7.609(179), 10.388(125)` and
  `Ns/xi_2nd = 15.963(336), 12.616(298), 9.242(112)`. All pass the
  uncertainty-aware target; beta 2.1 is the selected strongest tested
  coupling.

## qbig P100 validation

- Status: completed
- Physics: SU(2) Wilson Yang--Mills in 2+1 dimensions, temporal Dirichlet boundaries
- Coupling: beta = 12.15266
- Smoke ensemble: physical `Nt=1`, `Ns=8`, hot seed 12001, 8 sweeps, 5 overrelaxation sweeps, measurement interval 1
- Remote data: `/qbigwork2/negro/dirichlet/data/smoke_Nt_1_Ns_8_b_12p15266`
- Slurm target: one P100 on `lnode13`, three CPU cores, 4 GiB, 10 minutes
- Source: `/hadron/negro/src/klft`, commit `2871b8abc9c28917c2049243d3483d6dada4276a`
- Build: `/hadron/negro/build/klft-p100-3D-su2`
- Build options: Release, `KLFT_NDIM=3`, `KLFT_NC=2`, CUDA/OpenMP enabled, `Kokkos_ARCH_PASCAL60=ON`
- CUDA/compiler: CUDA 11.5.0, GCC 11.2.0
- Heatbath SHA-256: `5e7ad09405829c2a7831b666de16bb0e195391c6b020b4df901134cd632cdaa4`
- Submission: job `259714` exposed a CUDA-only host/device access in the local-action test harness; KLFT commit `2871b8a` fixed the test. Retry job `259715` completed on `lnode13` with exit code 0 in 21 seconds.
- Log: `/qbigwork2/negro/dirichlet/logs/smoke-259715.out`
- Validation: deterministic and statistical Dirichlet tests passed; all eight measured configurations had exact `C_G(0)=1`, and `W_b(R,1)-C_G(R)=0` at every independent separation.

No thesis-scale chains should be submitted until an `Ns=64` timing run passes.

## Ns=64 timing run

- Status: completed
- Ensemble: physical `Nt=2`, `Ns=64`, beta 12.15266, hot seed 12002
- History: 100 sweeps, 5 overrelaxation sweeps per heatbath sweep, measurement interval 1, all Dirichlet observables and both available boundary-loop heights
- Slurm: job `259716` on one P100 requested from `lnode13`
- Remote data: `/qbigwork2/negro/dirichlet/data/timing_Nt_2_Ns_64_b_12p15266`
- Log: `/qbigwork2/negro/dirichlet/logs/timing-259716.out`
- Result: 100 sweeps and measurements completed in 89.42 seconds with 161912 KiB peak resident memory; all output-count and `C_G(0)` checks passed.

## Initial thesis reproduction: Nt=2

- Status: completed
- Ensemble: physical `Nt=2`, `Ns=64`, beta 12.15266, hot starts
- Seeds: 32091, 90517, 177013, 515029
- History per chain: 8000 sweeps, 5 overrelaxation sweeps per heatbath sweep, measurement interval 1
- Observables: holonomy correlators and summary, slice plaquette profiles, `W_b(R,1)` and `W_b(R,Nt)`
- Slurm: array job `259718_[0-3]`; all four tasks started on `lnode13` with distinct scheduler-assigned visible GPUs 0 through 3
- Remote data: `/qbigwork2/negro/dirichlet/data/Nt_2_Ns_64_b_12p15266/seed_<seed>`
- Logs: `/qbigwork2/negro/dirichlet/logs/thesis-259718_<task>.out`
- Provenance: each run has `run_metadata.txt` with job, node, visible GPU, seed, input hash, KLFT commit, and executable hash
- First-output check: all four chains wrote 14 complete measurements after startup; no chain shares a visible GPU
- Completion: all four tasks exited 0 after 2:06:30 to 2:08:29; each chain has exactly 8000 measurements and passed the post-run `C_G(0)` and full-height-loop checks

## Nt=12 timing run

- Status: completed
- Ensemble: physical `Nt=12`, `Ns=64`, beta 12.15266, hot seed 12012
- History: 100 sweeps, 5 overrelaxation sweeps, measurement interval 1, boundary loops at only `H=1` and `H=Nt`
- Slurm: job `259722`, one Pascal GPU
- Remote data: `/qbigwork2/negro/dirichlet/data/timing_Nt_12_Ns_64_b_12p15266`
- Log: `/qbigwork2/negro/dirichlet/logs/timing-nt12-259722.out`
- Result: 100 sweeps and measurements completed in 106.01 seconds with 161840 KiB peak resident memory; all checks passed.

## Initial thesis reproduction: Nt=6 and Nt=12

- Status: completed
- Ensembles: physical `Nt=6` and `Nt=12`, `Ns=64`, beta 12.15266, hot starts
- Seeds at each thickness: 32091, 90517, 177013, 515029
- History per chain: 8000 sweeps, 5 overrelaxation sweeps per heatbath sweep, measurement interval 1
- Observables: holonomy correlators and summary, slice plaquette profiles, `W_b(R,1)` and `W_b(R,Nt)`
- Slurm: array job `259723_[4-11]`, one scheduler-managed Pascal GPU per task
- Remote data: `/qbigwork2/negro/dirichlet/data/Nt_{6,12}_Ns_64_b_12p15266/seed_<seed>`
- Logs: `/qbigwork2/negro/dirichlet/logs/thesis-259723_<task>.out`
- Placement: `Nt=6` uses visible GPUs 4 through 7 on `lnode13`; `Nt=12` uses visible GPUs 0 through 3 on `lnode14`
- First-output check: all eight chains wrote complete observable rows and provenance metadata
- Completion: all eight tasks exited 0. `Nt=6` took 2:15:59 to 2:18:29; `Nt=12` took 2:32:25 to 2:34:46. Every chain has exactly 8000 measurements and passed the post-run exact checks.

## Autocorrelation-aware analysis

- Status: completed
- Thermalization: MSER-5 plus post-cut diagnostics recommends the common thesis cut of 500 for all three thicknesses
- Initial Slurm analysis: CPU array job `259731_[0-2]`; all tasks completed with exit code 0
- Final reproducible analysis after the `scripts/` layout and explicit `(1/2)Tr G` output: job `259734_[0-2]`; Nt=2, 6, and 12 completed in 2:42, 2:20, and 9:07 with exit code 0
- Plotting and campaign table: dependent job `259737` completed with exit code 0 in 30 seconds; job `259738` regenerated the compact products with portable LF CSV output and completed with exit code 0 in 18 seconds
- Presentation-only plot regeneration on 2026-07-20 used the unchanged compact JSON/NPZ products locally because repeated bounded SSH connections to qbig timed out at port 22. No raw history was transferred and the numerical campaign table was unchanged; rerunning the same plotting command on qbig remains an infrastructure-only task.
- Logs: `/qbigwork2/negro/dirichlet/logs/analysis-259734_<task>.out` and `/qbigwork2/negro/dirichlet/logs/plots-{259737,259738}.out`
- Compact outputs: `/qbigwork2/negro/dirichlet/analysis/Nt_<Nt>_Ns_64_b_12p15266/`
- Local compact outputs: `analysis/dirichlet_thesis/`; raw histories were not synchronized
