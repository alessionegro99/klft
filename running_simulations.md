# Running simulations

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
- Plotting and campaign table: dependent job `259737`, completed with exit code 0 in 30 seconds
- Logs: `/qbigwork2/negro/dirichlet/logs/analysis-259734_<task>.out` and `/qbigwork2/negro/dirichlet/logs/plots-259737.out`
- Compact outputs: `/qbigwork2/negro/dirichlet/analysis/Nt_<Nt>_Ns_64_b_12p15266/`
- Local compact outputs: `analysis/dirichlet_thesis/`; raw histories were not synchronized
