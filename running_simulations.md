# Running simulations

As checked on 2026-08-28 at 10:52 CEST, the user queue was empty,
`lnode13`--`lnode17` were Slurm-idle, and direct inspection found no GPU
compute processes on their 40 P100s. This is an availability snapshot, not a
reservation. Fixed-beta/Nt production tasks `282274_[36-39]` remain timed
out. Fixed-mass extension analyses `282506_[0-1]` and round-1 tuning `282508`
remain complete.

## Nonperturbative fixed-`m_G Nt` final checks

- Status: staged byte-for-byte; smoke and timing passed; production array
  `289406_[0-11]` is active on `lnode13--14`.
- Purpose: increase the selected `Nt=3,Ns=80,beta=4.40` statistics, directly
  check the two low-volume `c=1,Nt=4` tuning inputs, and repeat the borderline
  `c=2,Nt=4` mass analyses with 10000 joint-bootstrap replicas before the
  independent confirmation.
- New GPU ensembles: four fresh 80000-sweep chains at
  `(Nt,Ns,beta)=(3,80,4.40)` and four 8000-sweep chains at each of
  `(4,96,4.95)` and `(4,96,5.30)`. All are independent hot starts with one
  heatbath plus five overrelaxation sweeps; holonomy cadence is one and
  profile cadence is five. Wilson and horizontal-Polyakov measurements remain
  disabled.
- Seeds: production draws 119--130 from `np.random.default_rng(20260724)`;
  smoke and timing use draws 131--132. Exact values and input hashes are in
  `data/mg_nt_fixed/final_checks/seed_provenance.tsv` and `manifest.tsv`.
- Existing-data check: reanalyze the three `c=2,Nt=4` pilot ensembles at beta
  3.70, 4.00, and 4.30 with 10000 replicas and seeds `20261020--20261022`;
  no new Markov histories are used for this stage.
- Resources: smoke, timing, and each production task request one typed Pascal
  GPU. Production is an uncapped 12-task array with two-hour walltime.
  Diagnostics and both analysis arrays use four CPUs and no scheduler
  dependencies. Final gates and tuning use two CPUs.
- Inputs: `data/mg_nt_fixed/final_checks/`; immutable manifest SHA-256
  `275a4c408c5334b2e19408fca73e4368c8ca442b2f8ce0c64a09eafd28db7f96`.
  Remote campaign, raw, compact-result, and log roots are respectively
  `/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks/`,
  `/qbigwork2/negro/dirichlet/data/mg_nt_fixed/final_checks/`,
  `/qbigwork2/negro/dirichlet/analysis/mg_nt_fixed/final_checks/`, and
  `/qbigwork2/negro/dirichlet/logs/mgnt-final-*`.
- Execution: at 12:08 CEST the user queue was empty, `lnode13--17` were
  Slurm-idle, and direct checks found no GPU compute processes on all 40
  P100s. Smoke job `289404` completed on `lnode13` in 12 seconds with exit
  code zero and `gate PASS`. Timing job `289405` completed there in 13
  seconds; 100 largest-volume sweeps took 1.69 seconds, projecting 0.375556
  hours for the 80000-sweep chain and 1.80267 total P100 GPU-hours. Production
  array `289406_[0-11]` started all tasks concurrently across `lnode13--14`.
  Representative initial metadata and output from all three ensembles match
  the manifest, including dimensions, beta, purpose, seed, and GPU assignment.

## Nonperturbative fixed-`m_G Nt` round-2 extension

- Status: complete. Smoke, timing, production, diagnostics, mass analysis,
  and round-2 tuning all completed and passed validation. All nine tuning
  groups now pass.
- Purpose: resolve the sole failed tuning group `(c,Nt)=(0.5,3)`, whose
  round-1 local log-linear interpolation has `p=0.01350`.
- Ensemble: `(physical Nt,stored L2,Ns,beta)=(3,4,80,4.290)`. Beta 4.290 is
  only a placement estimate from the target-bracketing beta-4.27 and
  beta-4.40 measurements.
- Production seeds: `767348994`, `1941201624`, `190362085`, and `1526132929`,
  draws 113--116 from `np.random.default_rng(20260724)`. Separate smoke and
  timing seeds are `922096603` and `1947777267`.
- Histories: four independent hot starts, 8000 composite sweeps per chain,
  one heatbath plus five overrelaxation sweeps. Holonomy measurements use
  cadence one and profiles cadence five; boundary Wilson loops and horizontal
  Polyakov loops are disabled.
- Validation: one immutable 25-sweep smoke input followed by one immutable
  100-sweep timing input. Production is forbidden unless both local campaign
  gates pass; the timing limits remain two hours per chain and 12 total P100
  GPU-hours.
- Resources: smoke and timing request one typed Pascal GPU, three CPUs, and
  5 GB each. Production is an uncapped four-task array with the same per-task
  resources and a two-hour walltime. Diagnostics use four CPUs/12 GB/two
  hours; analysis uses four CPUs/16 GB/eight hours; tuning uses two CPUs/8 GB/
  30 minutes. No job has a Slurm dependency.
- Inputs and manifest: `data/mg_nt_fixed/extensions/round2/`; immutable
  manifest SHA-256 `a750d240ef00f3242b26e98ccaffe525dffef8996a1f551c5a2825f9ce3f4e10`.
- KLFT: commit `7eed1f7d9bb0948621a2d4020706d19c56406001`, heatbath SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`,
  sole P100 build `/qbigwork2/negro/dirichlet/build/klft`.
- Remote campaign:
  `/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round2/`;
  raw histories:
  `/qbigwork2/negro/dirichlet/data/mg_nt_fixed/extensions/round2/`; compact
  results:
  `/qbigwork2/negro/dirichlet/analysis/mg_nt_fixed/extensions/round2/`.
- Planned log prefixes: `mgnt-ext2-smoke`, `mgnt-ext2-timing`,
  `mgnt-ext2-prod`, `mgnt-ext2-diag`, `mgnt-ext2-analysis`, and
  `mgnt-ext2-tuning` below `/qbigwork2/negro/dirichlet/logs/`.
- Execution: smoke job `289395` completed on `lnode13` in 12 seconds and wrote
  `gate PASS`; its 25 summary, 1025 correlator, and 35 profile rows match the
  immutable input. Timing job `289396` completed on `lnode13` in 13 seconds;
  100 sweeps took 1.52 seconds, projecting 0.0337778 hours per chain and
  0.135111 total P100 GPU-hours, so its gate passed. Production array
  `289397_[0-3]` completed on `lnode13` in 1:24--1:26 with exit code zero.
  The manifest-wide validator passed all four chains, including hashes,
  metadata, exact 8000-sweep measurement counts, and `run_complete.txt`.
- Diagnostics job `289401` completed on `lnode11` in 21 seconds with exit code
  zero. Every chain uses a 500-sweep cut; the common 20/30/40-sweep plateau
  selects 40-sweep blocks and leaves 187 complete blocks per chain. The
  largest monitored autocorrelation time is 4.421 sweeps, the maximum
  split-history discrepancy is 2.245 standard deviations, and the maximum
  between-chain discrepancy is 2.704 standard deviations. All primary and
  secondary plateau gates pass. Dependency-free mass analysis job `289402`
  completed on `lnode11` in 39:33 with exit code zero, using 1000 replicas
  and bootstrap seed `20261001`. It obtained
  `m_G Nt=0.5051264 +/- 0.0020453`; the selected `R=5--40` fit has a stable
  three-by-three window patch, recentered-bootstrap `p=0.8891`, and 100%
  finite positive mass replicas. Tuning job `289403` completed on `lnode11`
  in 16 seconds with exit code zero. All nine groups pass; for
  `(c,Nt)=(0.5,3)`, the selected beta points are `4.27,4.29,4.40`, the fit
  has `p=0.1474`, and `beta_star=4.29462 [4.29213,4.29828]` with interpolation
  systematic `0.00513`.

## Fixed-beta/Nt production extension

- Status: 36 of 40 production chains and eight of ten ensemble analyses are
  complete and accepted. Tasks `282274_[36-39]`, the four
  `(Nt,Ns,beta)=(5,688,10.5)` chains, all timed out after 95.5 hours. Their
  final holonomy steps are 48399, 48778, 48724, and 48324; none has
  `run_complete.txt`. The dependent final pipeline was cancelled, and no GPU
  production is active. Clean P100 build
  job `282258` failed before configuration because
  its source path used an incorrect 8-character directory tag; it created no
  build or simulation output. Corrected clean-build job `282259` supersedes
  it and completed successfully on `lnode13`. Both deterministic and
  statistical Dirichlet CTests passed. The P100 executable SHA-256 is
  `a22087d10bdc78a9fae501cc0e4e0e91116577ce2bd684345331c4113e038ef8`.
  Build and cadence-20 smoke logs are
  `/qbigwork2/negro/dirichlet/logs/bntf-build-282259.out` and
  `/qbigwork2/negro/dirichlet/logs/bntf-smoke-282265.out`.
- Reanalysis: CPU array `282249_[0-4]` was cancelled after about 35 minutes
  because it redundantly scanned every `Rmax` down to `Ns/3` and had not
  produced compact output. Replacement array `282260_[0-4]` retains the full
  `Rmin` scan and 3-by-3 stability test at the three endpoint choices
  `Ns/2`, `Ns/2-2`, and `Ns/2-4`. It applies diagonal-covariance periodic-
  `K0` fits to the existing `Nt=2` finite-volume sequence and the selected
  `Nt=3,4` volumes. Complete correlator vectors are resampled jointly with
  1000 seeded hierarchical-bootstrap replicas; fit errors come from the
  refits, while the full covariance is used only for post-fit `chi2/dof` and
  a recentered-bootstrap goodness-of-fit test.
- Reanalysis outputs (replacement array):
  `/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/reanalysis_v2/`; logs:
  `/qbigwork2/negro/dirichlet/logs/bntf-reanalysis-282260_<task>.out`.
- Replacement array `282260_[0-4]` completed. The selected `Nt=2,3,4` fits
  give `m_G Nt=0.20736(77),0.17911(57),0.16236(92)`.
- Borderline-fit confirmation array `282321_[3-4]` repeated the selected
  `Nt=3,4` analyses with 10000 joint-bootstrap replicas and independent
  explicit seeds `20260823,20260824`. It writes new immutable output under
  `/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/reanalysis_v3/`. Both
  tasks completed with exit code zero. They give `m_G Nt=0.179114(584)` with
  `p=0.06579` and `0.162349(934)` with `p=0.05179`; both pass, with the `Nt=4`
  fit close to the gate.
- The earlier preliminary jobs `282318/282319` completed before most
  production analyses and are stale. The reproducible current partial summary
  is local under `analysis/beta_nt_fixed/comparison/`; it records missing or
  rejected points explicitly and substitutes none.
- Diagnostic jobs `282509` at sweep 40000 and `282510` at sweep 48300 used a
  block grid with fewer than three candidates above their autocorrelation
  bounds, so their apparent rejections are superseded. The corrected grid adds
  viable 1000- and 1400-sweep candidates. Job `282511` completed in 5:43 with
  exit code zero and accepts `[1000,1200,1400]` sweeps, selecting 1400-sweep
  blocks and leaving `32,32,33,32` blocks after independent cuts
  `2125,2250,1750,3250`. The maximum between-chain discrepancy is 3.102 sigma
  in `Tr_G_real`. The byte-matched compact diagnostic is local under
  `analysis/beta_nt_fixed/beta_over_nt_2p1/Nt_5_Ns_688_b_10p5/`.
- Preliminary mass-analysis job `282512` ran dependency-free with eight CPUs,
  32 GB, a 12-hour limit, exactly 1000 replicas, and bootstrap seed
  `20260910`. It completed in 3:56:46 with exit code zero. Eight individual
  windows pass the pointwise mass gates, but no stable three-by-three patch
  exists; `mass` and `xi_exp` are null, so the snapshot enters no continuum
  fit. Remote compact output remains under
  `/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/preliminary_snapshot_48300_v2/`;
  log: `/qbigwork2/negro/dirichlet/logs/bntf-pre483v2-analysis-282512.out`.
- First-stage plotting job `282513` completed in 1:21 with exit code zero. It
  generated 30 grid-free PDFs: one thermalization and one blocking-error plot
  for each of 13 slab ensembles and two PCM references. Remote compact output:
  `/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/first_stage_clean_20260728/`;
  log: `/qbigwork2/negro/dirichlet/logs/bntf-first-stage-282513.out`.
- The current accepted selection has 64 grid-free preliminary PDFs under
  `analysis/beta_nt_fixed/comparison/plots/`. Its continuum products contain
  only `p=1` fits and exactly 1000 bootstrap replicas per operation.
- Matched exact-PCM references: CPU array `282267_[0-1]` failed before
  analysis because its wrapper directly executed a non-executable script and
  produced no compact output. Corrected array `282272_[0-1]` reanalyzes the
  existing `Nt=1,Ns=96,beta=2.0,2.1` histories with the same diagonal-fit,
  joint-bootstrap, and complete-covariance goodness-of-fit convention. These
  are the `Nt=1` endpoints of the new `beta/Nt=2.0,2.1` lines. Compact output:
  `/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/pcm_reanalysis_v2/`;
  logs: `/qbigwork2/negro/dirichlet/logs/bntf-pcm-fit-282272_<task>.out`.
  Both corrected tasks completed with exit code zero. They give
  `m_G=0.126859(380)` at beta `2.0` and `0.096229(318)` at beta `2.1`.
- Production grid: ten ensembles and four independent hot-start chains each.
  The exact fixed `beta/Nt` lines are `1.8`, `2.0`, and `2.1`, corresponding
  to `g_3^2 L = 20/9`, `2`, and `40/21`. New points are `(Nt,Ns,beta) =
  (5,336,9.0),(6,416,10.8)`; `(2,192,4.0),(3,304,6.0),(4,416,8.0),
  (5,544,10.0)`; and `(2,240,4.2),(3,384,6.3),(4,528,8.4),
  (5,688,10.5)`.
- Histories: 50000 composite sweeps per chain, one heatbath plus five
  overrelaxation sweeps. Holonomy correlators, Fourier components, and
  summaries are measured every sweep; profiles every five sweeps; boundary
  Wilson loops at every height every 25 sweeps in the accepted production
  revision. The largest boundary ranges
  reach at least three expected correlation lengths.
- Inputs and explicit seeds are grouped by line under
  `data/beta_nt_fixed/beta_over_nt_*/inputs/`; the immutable shared manifest is
  `data/beta_nt_fixed/comparison/provenance/cadence25/manifest.tsv`. Staged campaign:
  `/qbigwork2/negro/dirichlet/campaigns/beta_nt_fixed/production/cadence25/`;
  raw root:
  `/qbigwork2/negro/dirichlet/data/beta_nt_fixed/production/`.
- KLFT: clean pinned commit
  `f3928aa958a3aef18d215670faafc817ba322484`, source
  `/qbigwork2/negro/dirichlet/src/klft-f3928aa`, P100 build
  `/qbigwork2/negro/dirichlet/build/klft-f3928aa-p100`. Build log:
  `/qbigwork2/negro/dirichlet/logs/bntf-build-282259.out`.
- Gates: the accepted revision used a 25-sweep `Nt=2,Ns=32,beta=4.0` smoke
  to validate all output counts and the exact `W_b(R,Nt)=C_G(R)` identity,
  followed by a 25-sweep timing run of the largest
  `Nt=5,Ns=688,beta=10.5,Rmax=208` ensemble. Both passed before production.
- Smoke job `282265` completed successfully on `lnode13` in 0.38 s of
  executable wall time. It produced exactly 20 summary rows, 340 correlator
  rows, 20 profile rows, and 18 boundary-loop rows; `C_G(0)=W_b(0,H)=1`, and
  every `W_b(R,2)` agrees with the step-20 `C_G(R)` within `2e-11`.
  Cadence-20 timing job `282266` followed; its log is
  `/qbigwork2/negro/dirichlet/logs/bntf-timing-282266.out`.
- Cadence-20 timing job `282266` completed in 150.875257 s and correctly
  failed its gate: the deliberately conservative 50000-sweep projection is
  104.77448403 hours, above the 93-hour limit. No production task was
  submitted from that revision.
- Superseding immutable revision `cadence25` uses 42 fresh seeds (40
  production plus two validations) and keeps the same physics grid, profiles,
  and all-height boundary ranges while measuring boundary loops every 25
  sweeps. Its manifest SHA-256 is
  `9cab946bd0ab735cc943e62c362ac37de7a1eb52f5d4ab5043cd9b4ac6be03bc`.
  Inputs are locally grouped under `data/beta_nt_fixed/beta_over_nt_*/inputs/`
  and staged at
  `/qbigwork2/negro/dirichlet/campaigns/beta_nt_fixed/production/cadence25/`.
  Replacement smoke job `282269` failed before KLFT execution because its
  wrapper directly executed a non-executable shared script; it created no
  validation output. Corrected smoke job `282270` superseded it and passed in
  0.44 s of executable wall time with all generic count, cadence, and exact
  boundary-identity checks. Logs:
  `/qbigwork2/negro/dirichlet/logs/bntf-smoke25-282269.out`,
  `/qbigwork2/negro/dirichlet/logs/bntf-smoke25-282270.out`, and
  `/qbigwork2/negro/dirichlet/logs/bntf-time25-282271.out`.
- Cadence-25 timing job `282271` passed: 164.568808 s for the 25-sweep
  largest case, conservatively projecting 91.42711556 hours per 50000-sweep
  chain. Immediately before production submission, Slurm reported all five
  Pascal nodes `lnode13`--`lnode17` idle and direct `nvidia-smi` checks found
  no compute process on any of their 40 GPUs.
- Production array `282274_[0-39]` was submitted without a concurrency cap.
  Every task requests one typed Pascal GPU, three CPU cores, 8 GB, and
  `3-23:30:00`. Logs are
  `/qbigwork2/negro/dirichlet/logs/bntf-prod25-282274_<task>.out`.
- First-output inspection: all 40 tasks started simultaneously across
  `lnode13`--`lnode17`, eight typed P100 tasks per node. Sampled tasks 0, 8,
  24, and 36 have the intended dimensions, coupling, seed, three cadences,
  input hash, KLFT commit, executable hash, node, and GPU recorded in
  `run_metadata.txt`. Their holonomy/profile/boundary files grow only on the
  intended grids. Live checks on tasks 8 and 24 find zero violations of
  `C_G(0)=1`, `W_b(0,H)=1`, or `W_b(R,Nt)=C_G(R)` at the latest common
  measurement. Tasks 0--35 now have exit code zero and `run_complete.txt`,
  which is written only after all wrapper validation checks pass.
- Incremental analysis: diagnostics job `282344_[2]` exposed a cadence-grid
  bug before writing a result: the blocking candidates were expressed in
  five-sweep units and none was divisible by the new 25-sweep boundary
  cadence. The corrected grid uses units of the least common multiple of all
  enabled cadences; its new cadence-25 regression test passes. Failed
  dependent analysis job `282345_[2]` produced no output. Replacement
  diagnostics job `282354_[2]` completed with exit code zero. Per-chain cuts
  are `625,2625,750,750` sweeps; the common `300,400,600`-sweep blocking
  plateau selects 600-sweep blocks and leaves `82,78,82,82` blocks. The
  largest between-chain discrepancy is `3.11` standard deviations in
  `W_b(R=2,H=2)`, and all primary and secondary plateau gates pass. Analysis
  job `282355_[2]` completed. The subsequent incremental pairs through
  `282386/282387` also completed; all eight available ensembles have accepted
  masses, accepted mass/`xi_exp` plateaus, at least 30 blocks per chain, and
  1000 finite mass replicas. Diagnostics `282388_8` completed for
  `(Nt,Ns,beta)=(4,528,8.4)`, but analysis `282389_8` failed because no common
  three-length blocking plateau exists across all 924 observables. Jobs
  `282390_9/282391_9` were cancelled after their production dependencies
  timed out. The separate common-cutoff jobs `282511/282512` supersede them
  only for explicitly preliminary analysis.
- Per-ensemble dependencies are recorded remotely in
  `campaigns/beta_nt_fixed/production/cadence25/incremental_analysis_jobs.tsv`:

| index | `(Nt,Ns,beta)` | production tasks | diagnostics | analysis |
|---:|---:|---:|---:|---:|
| 0 | `(5,336,9.0)` | `282274_[0-3]` | `282374_0` | `282375_0` |
| 1 | `(6,416,10.8)` | `282274_[4-7]` | `282376_1` | `282377_1` |
| 2 | `(2,192,4.0)` | `282274_[8-11]` | `282354_2` | `282355_2` |
| 3 | `(3,304,6.0)` | `282274_[12-15]` | `282378_3` | `282379_3` |
| 4 | `(4,416,8.0)` | `282274_[16-19]` | `282380_4` | `282381_4` |
| 5 | `(5,544,10.0)` | `282274_[20-23]` | `282382_5` | `282383_5` |
| 6 | `(2,240,4.2)` | `282274_[24-27]` | `282384_6` | `282385_6` |
| 7 | `(3,384,6.3)` | `282274_[28-31]` | `282386_7` | `282387_7` |
| 8 | `(4,528,8.4)` | `282274_[32-35]` | `282388_8` | `282389_8` |
| 9 | `(5,688,10.5)` | `282274_[36-39]` | `282390_9` | `282391_9` |

Every diagnostics job waited only for its four production chains; its analysis
job waited for that diagnostic. Index 9 was cancelled by the timed-out
dependency and is not a completed production analysis.
- Current partial continuum analysis uses the exact accepted selection in
  `analysis/beta_nt_fixed/comparison/` and retains only `p=1`. On
  `beta/Nt=1.8` (`Nt=2--6`) it gives `m_G L=0.118373(884)`, fit
  `p=0.1328`; on `beta/Nt=2.0` (`Nt=2--5`) it gives `0.073017(764)`, fit
  `p=0.8508`. Line 2.1 has only two accepted resolutions and is not fit.
- Cancelled post-production pipeline: diagnostics array `282314_[0-9]`, full
  1000-replica analysis array `282315_[0-9]`, continuum summary job `282316`,
  and PDF plotting job `282317` were cancelled when production tasks 36--39
  timed out.
  Diagnostics and blocking remain independent-chain operations. The cleaned
  summary uses only fixed `p=1` and records any unavailable mass rather than
  substituting an unaccepted fit. Logs use the corresponding
  `bntf-diag`, `bntf-analysis`, `bntf-summary`, and `bntf-plots` prefixes in
  `/qbigwork2/negro/dirichlet/logs/`.

## Nonperturbative fixed-`m_G Nt`

- Status: the full pilot, all 100 chains, all 25 diagnostics, and all 25 mass
  analyses are complete and validated. The two required extension ensembles
  are also fully analyzed. Round-1 tuning job `282508` passes eight of nine
  targets: `(c,Nt)=(0.5,4)` is now tuned, while `(0.5,3)` still fails the
  local log-linear gate with `p=0.01350`. No confirmation is submitted.
- Campaign type: `nonperturbative_mg_nt`, separate from `beta_nt_fixed`.
- Targets: `m_G Nt = 0.5, 1.0, 2.0`; physical `Nt=2,3,4`.
- Grid: 25 new ensembles and 100 new hot-start chains. The beta 3.0
  `(c,Nt)=(0.5,2)` and beta 2.1 `(c,Nt)=(2,2)` points are explicitly labelled
  prior information and were independently reanalyzed before production.
- Histories: 8000 composite sweeps per new chain, one heatbath plus five
  overrelaxation sweeps; holonomy cadence one and profile cadence five.
- Inputs: `data/mg_nt_fixed/pilot/`; compact output:
  `analysis/mg_nt_fixed/`; remote raw root:
  `/qbigwork2/negro/dirichlet/data/mg_nt_fixed/`.
- Resources: dependency-free arrays with no concurrency cap, one typed Pascal
  GPU per task. The 100-sweep `(Nt,Ns,beta)=(4,96,6.20)` timing gate must stay
  below two hours per chain and 12 total P100 GPU-hours.
- KLFT: reuse the sole verified P100 build only while its revision remains
  `7eed1f7d9bb0948621a2d4020706d19c56406001`; expected heatbath SHA-256 is
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
- Timing job `282244` completed on `lnode13`: 1.614251 s per 100 largest-case
  sweeps, projecting 0.03587224 hours per largest chain and 0.93835035 total
  P100 GPU-hours. Output counts, metadata, `C_G(0)=1`, both cadences, disabled
  outputs, and all recorded hashes passed.
- Audit diagnostics array `282245_[0-1]` completed. Analysis array `282247`
  failed on an incomplete intermediate source; superseded array `282254` was
  cancelled after 8:18 without compact results when the fit convention was
  corrected. Dependency-free array `282256_[0-1]` completed in 36:07 and
  1:06:55. The beta-3.0 and beta-2.1 masses reproduce the prior accepted
  values within 0.063 and 0.535 combined standard deviations; audit summary
  job `282320` wrote `gate PASS`.
- Production preflight found another valid Slurm allocation on 36 P100s and
  four genuinely free devices on `lnode14`; all hashes and output-absence
  checks passed. Array `282323_[0-99]` is dependency-free and uncapped. It was
  monitored for more than one minute; tasks 0--3 completed cleanly in
  1:02--1:05 and task 0 passed its exact output-count, cadence, metadata,
  `C_G(0)=1`, and disabled-output checks.
- All 100 production tasks completed with exit code zero. A manifest-wide
  read-only validation passed every chain's files, metadata, hashes, exact
  measurement grids and counts, `C_G(0)=1`, and disabled-output checks.
  Dependency-free diagnostics array `282445_[0-24]` was then submitted.
- Diagnostics array `282445_[0-24]` completed with exit code zero. All chains
  use 500-sweep cuts and accepted primary/secondary three-length plateaus;
  selected blocks retain 46--375 blocks per chain. The 46-block largest case
  misses the 50-block target but passes the hard floor of 30. Dependency-free
  mass-analysis array `282470_[0-24]` completed with exit code zero. Every
  point has an accepted mass, an accepted mass/`xi_exp` plateau, at least 30
  blocks per chain, and 1000 finite mass replicas.
- Official tuning job `282495` completed in 15 seconds. Accepted beta values
  are `2.98734,2.50249,3.59502,4.61998,1.96373,2.93960,3.76006` for
  `(c,Nt)=(0.5,2),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4)`, respectively. The
  two rejected `c=0.5` rows are not extrapolated.
- Gate-required extension points are `(c,Nt,Ns,beta)=(0.5,3,80,4.27)` and
  `(0.5,4,96,5.50)`, four fresh hot-start chains each. Immutable manifest
  SHA-256 is
  `c4b4f4c8e0ac1f3186638ecd4463916bb37e9109fe80490d17c37952f8eb059a`.
  Production `282496_[0-7]` completed in 1:22--1:59 and all eight chains
  passed validation. Diagnostics `282504_[0-1]` selected 40- and 60-sweep
  blocks, leaving 187 and 125 blocks per chain. Analysis `282506_[0-1]`
  completed with accepted `m_G Nt=0.515906 +/- 0.002014` and
  `0.531824 +/- 0.002275`, with fit p-values 0.6064 and 0.2567. Tuning
  `282508` accepts `c=0.5,Nt=4` at
  `beta_star=5.58394 [5.57975,5.58790]`; `c=0.5,Nt=3` still requires another
  bracket point. The `Nt=4,c=1` confirmation remains unsubmitted until all
  nine tunings pass.
- Preliminary-statistics checklist:
  `analysis/mg_nt_fixed/preliminary_checklist.md`. Highest priorities are a
  new `c=0.5,Nt=3` point near beta 4.290, increased statistics at beta 4.40,
  additional statistics for any final use of the 46-block beta-6.20
  ensemble, and larger-volume checks for bracket inputs with
  `Ns/xi_2nd < 10`. The 84 current grid-free pilot PDFs are under
  `analysis/mg_nt_fixed/preliminary_plots/`. More bootstrap replicas do not
  substitute for new Markov data.
- Logs: `/qbigwork2/negro/dirichlet/logs/mgnt-*.out`; compact audit output:
  `/qbigwork2/negro/dirichlet/analysis/mg_nt_fixed/audit/`.

## Fixed-beta/Nt corrected `Nt=3,4` volumes

- Status: completed, analyzed, plotted, and compact products synchronized.
  Timing job `274307`, production array `274308_[0-7]`, diagnostics array
  `282238_[0-1]`, analysis array `282240_[0-1]`, summary job `282242`, and
  plotting job `282243` all completed with exit code zero. Every stage was
  submitted explicitly with `Dependency=(null)`.
- Purpose: replace the pilot fixed-beta/Nt `Nt=3,4` points with spatial
  sizes satisfying the measured target `Ns/xi_2nd` near 10 before extending
  the line to `Nt=6`
- Ensembles: `(physical Nt,beta,Ns)=(3,5.4,176),(4,7.2,256)`, with stored
  temporal extents `L2=4,5`, `beta/Nt=1.8`, and `g_3^2 L=20/9`
- Seeds: 1139865066, 1786727755, 1777987394, 1986906165 at `Nt=3`; 32943504,
  1111774376, 350847387, 1543348040 at `Nt=4`; all are independent hot starts
  and distinct from earlier campaign stages
- Histories: 20000 composite sweeps per chain; one heatbath plus five
  overrelaxation sweeps
- Cadence: holonomy observables every sweep; plaquette profiles every five
  sweeps; Wilson-loop and horizontal-Polyakov output disabled
- Inputs and manifest: local
  `data/beta_nt_fixed/beta_over_nt_1p8/validation/corrected_volumes_nt3_nt4/`; legacy
  remote campaign
  `/qbigwork2/negro/dirichlet/campaigns/step3_corrected_volumes/`
- Raw output:
  `/qbigwork2/negro/dirichlet/data/Nt_<Nt>_Ns_<Ns>_b_<beta>/seed_<seed>/`
- Logs: timing
  `/qbigwork2/negro/dirichlet/logs/s3cv-timing-274307.out`; production
  `/qbigwork2/negro/dirichlet/logs/s3cv-prod-274308_<task>.out`
- Slurm request: the 100-sweep timing job and the eight-task production array
  were submitted separately without dependencies. Production has no array
  concurrency cap and requests one typed Pascal GPU per task.
- Timing gate: the largest `Nt=4,Ns=256` case took 16.247056 seconds for 100
  sweeps, projecting 0.90261422 hours per 20000-sweep largest chain and a
  conservative 7.22091376 total P100 GPU-hours. This passes the two-hour
  per-chain and 12-GPU-hour total limits.
- Timing validation: exactly 100 holonomy summaries, 12900 correlator rows,
  and 180 profile rows at steps `5,10,...,100`; every measured `C_G(0)` is
  one and all disabled Wilson/Polyakov outputs are absent.
- Live production validation: all eight inputs and metadata have the intended
  physical/stored extents, seeds, and cadences; output is growing in every run.
  All live `C_G(0)` values are one, profile steps are multiples of five, and
  disabled Wilson/Polyakov files are absent.
- Completed production validation: all eight post-run count, `C_G(0)`, profile
  cadence, metadata, and disabled-output checks passed. The `Nt=3` tasks ran
  for 15:13--16:13 and the `Nt=4` tasks for 55:09--56:03.
- Analysis: every chain uses an independently selected 500-sweep MSER-5 cut.
  At `Nt=3`, the shared whole-history blocking plateau is 80/120/160 sweeps;
  160-sweep blocks leave 121 complete blocks per chain. At `Nt=4`, the plateau
  is 160/240/320 sweeps; 320-sweep blocks leave 60 blocks per chain. The final
  seeded hierarchical bootstrap kept the preregistered 20000 samples.
- Results: `xi_2nd=16.855(534)` and `Ns/xi_2nd=10.442(334)` at `Nt=3`;
  `xi_2nd=24.383(1.126)` and `Ns/xi_2nd=10.499(496)` at `Nt=4`. Both 68%
  lower volume-ratio bounds exceed eight. `xi_exp` is null at both points
  because no correlated periodic-`K0` fit passes every gate.
- Volume comparison: pilot versus corrected full-correlator tests give
  `chi2/dof=146.32/72`, `p=5.44e-7` at `Nt=3` and
  `chi2/dof=311.13/96`, `p=1.59e-24` at `Nt=4`; the pilot volumes are rejected
  for full-correlator work. The corrected scaling fit gives
  `q0=7.356(448)`, `c=-5.160(1.076)`, and `p=0.882`.
- Compact output: legacy remote `analysis/step3_corrected_volumes/` and local
  `analysis/beta_nt_fixed/beta_over_nt_1p8/validation/corrected_volumes_nt3_nt4/`
  (about 3 MB). Dry-run and compact-only synchronization completed on
  2026-07-24; raw histories were not synchronized. Analysis workflow commit
  `bda1a32`.
- Bootstrap decision for subsequent new analyses: use 1000 resamples by
  default; increase only a final calculation whose fit-success gate or
  interval is numerically borderline. Existing 20000-sample results remain
  unchanged.
- KLFT: reuse the sole verified P100 build at
  `/qbigwork2/negro/dirichlet/build/klft`, commit
  `7eed1f7d9bb0948621a2d4020706d19c56406001`, heatbath SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`

## Fixed-beta/Nt `Ns=160` finite-volume check

- Status: completed, analyzed, plotted, and compact products synchronized.
  Smoke job `274106`, production array `274107_[0-3]`, diagnostics job
  `274303`, analysis job `274304`, summary job `274305`, and plotting job
  `274306` all completed with exit code zero and no Slurm dependency.
- Purpose: decide whether the `Nt=2`, beta 3.6 correlator is stable between
  `Ns=128` and 160 before extending the fixed-beta/Nt continuum line
- Ensemble: physical `Nt=2`, stored `L2=3`, `Ns=160`, beta 3.6,
  `beta/Nt=1.8` and `g_3^2 L=20/9`
- Seeds: 98728066, 856366050, 397310628, 411907572; four independent hot starts
- Histories: 10000 composite sweeps per chain; one heatbath plus five
  overrelaxation sweeps
- Cadence: holonomy observables every sweep; plaquette profiles every five
  sweeps; Wilson-loop and horizontal-Polyakov output disabled
- Inputs and manifest: local
  `data/beta_nt_fixed/beta_over_nt_1p8/validation/finite_volume_nt2/`; legacy remote
  campaign
  `/qbigwork2/negro/dirichlet/campaigns/step3_fv160/`
- Raw output:
  `/qbigwork2/negro/dirichlet/data/Nt_2_Ns_160_b_3p6/seed_<seed>/`
- Slurm request: one typed Pascal GPU per task; dependency-free four-task array
  with no concurrency cap. `scontrol` reports `Dependency=(null)` for every
  task. Logs are `/qbigwork2/negro/dirichlet/logs/s3fv-prod-274107_<task>.out`.
- Production: all four tasks started simultaneously on separately allocated
  P100s on `lnode13` and completed in 6:14--6:22.
- Smoke validation: 100-sweep job `274106` ran on `lnode13` and completed in
  16 seconds, with 3.91 seconds in the KLFT executable. It produced exactly
  100 holonomy-summary rows, 8100 correlator rows, and 100 profile rows at
  steps `5,10,...,100`; all measured `C_G(0)` values equal one and every
  disabled Wilson/Polyakov output is absent. The kernel timing projects about
  6.5 minutes per 10000-sweep chain. The log is
  `/qbigwork2/negro/dirichlet/logs/s3fv-smoke-274106.out`.
- KLFT: reuse the sole verified P100 build at
  `/qbigwork2/negro/dirichlet/build/klft`, commit
  `7eed1f7d9bb0948621a2d4020706d19c56406001`, heatbath SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`
- Analysis: project commit `af5fe72`, per-chain MSER-5 cut 500 sweeps,
  common whole-history blocking plateau 30/40/60 sweeps, selected block length
  60 sweeps, and 158 complete blocks per chain. A seeded 20000-sample
  hierarchical bootstrap gives `xi_2nd=9.529(486)` and
  `Ns/xi_2nd=16.792(877)`; `xi_exp` is null because no correlated periodic-K0
  fit passes every preregistered gate.
- Finite-volume decision: `Ns=128` versus 160 differs by 0.056 combined
  standard deviations and 0.34%. The 64-mode full-correlator comparison gives
  `chi2=48.54`, `p=0.924`; all criteria pass and `Ns=128` is accepted.
  The revised preliminary fixed-beta/Nt fit gives `q0=7.525(453)`,
  `c=-5.540(1.075)`, and `p=0.344`. The optional `(Nt,beta)=(6,10.8)` point
  may now be considered but has not been submitted.
- Compact output: legacy remote `analysis/step3_fv160/` and local
  `analysis/beta_nt_fixed/beta_over_nt_1p8/validation/finite_volume_nt2/` (about
  1.4 MB). Dry-run and compact-only synchronization completed; raw histories
  were not synchronized.

## Fixed-beta/Nt mixed parameter-space pilot

- Status: completed, analyzed, plotted, and compact products synchronized.
  Final analysis array `259877_[0-7]`, aggregation job `259886`, and plotting
  job `259887` all completed with exit code zero and no Slurm dependency.
  Build jobs `259773`
  and `259782` failed before producing an executable. Corrected build job
  `259790` completed on `lnode13` in 3:29, including both Dirichlet CTests.
  It removes the stale CUDA 9.2 include path added by the node login shell and
  otherwise uses the CUDA 11.5/GCC 11.2 environment.
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
- Inputs and immutable manifest: local
  `data/beta_nt_fixed/comparison/provenance/legacy_pilot_inputs/`; legacy remote
  `/qbigwork2/negro/dirichlet/campaigns/step3_scan/`
- Raw output: `/qbigwork2/negro/dirichlet/data/Nt_<Nt>_Ns_<Ns>_b_<beta>/seed_<seed>/`
- Logs: `/qbigwork2/negro/dirichlet/logs/step3-*.out`
- Slurm request: one typed Pascal GPU per build, timing, or production task;
  production array `0-31` with no concurrency cap, allowing Slurm to use all
  available P100s across the eligible nodes; no comma-separated node list
- Timing gate: dependency-free job `259793` ran 100 sweeps at
  `Nt=4,Ns=192,beta=7.2` in 7.538261 seconds. It projected 0.20939614 hours
  for the largest chain and 1.70332127 total P100 GPU-hours, passing the
  two-hour and 12-GPU-hour limits.
- KLFT: commit `7eed1f7d9bb0948621a2d4020706d19c56406001`, fresh
  CUDA/OpenMP `KLFT_NDIM=3`, `KLFT_NC=2`, Pascal60 build at the single project
  path `/qbigwork2/negro/dirichlet/build/klft`; heatbath SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`
- Production: dependency-free array `259794_[0-31]`; all 32 tasks began at
  2026-07-21 16:06:18 across `lnode13`--`lnode16`, used one scheduler-assigned
  P100 each, and completed with exit code zero in 1:05--13:04.
- Analysis: independent MSER-5 cuts (minimum 500 sweeps), Sokal-window
  autocorrelations in sweep units, whole-history blocking-error curves for all
  primary and secondary observables, replica-stratified fusion only after
  separate blocking, a common error-plateau block longer than `2 tau_int`, at
  least 30 blocks per chain, and a seeded 20000-sample hierarchical bootstrap
- Blocking diagnostics: dependency-free array `259869_[0-7]` completed.
  Every primary correlator/profile component and both secondary observables
  share one three-length plateau. Selected common block lengths are 60,
  20, 20, 20, 120, 160, 240, and 60 sweeps in manifest ensemble order. The
  corresponding per-chain block counts are 158, 475, 475, 475, 79, 59, 39,
  and 158; only `Nt=4,Ns=192` misses the 50-block target, while retaining 39
  blocks above the required floor of 30.
- Analysis retries: array `259850` produced seven results but task 2 stopped
  because 1965 of 20000 bootstrap estimates had negative `xi_2nd^2`. The
  final code preserves those signed squared estimates, reports the 0.90175
  valid fraction and unconditional percentile interval, and does not quote a
  symmetric error from a truncated sample. Array `259860` was superseded when
  the fit-derived plateau audit was added; its obsolete task 2 was cancelled.
  Final array `259877_[0-7]` also refits every delete-one-block sample and
  reports non-null `xi_exp` and `mass` only when the selected block ends a
  shared three-length error plateau for both.
- Final analysis: array `259877_[0-7]` completed in 0:32--32:07. All eight
  compact JSON/NPZ pairs pass metadata, exact `C_G(0)`, covariance,
  20000-sample, blocking, and fit-gate checks. Aggregation job `259886`
  completed in 15 seconds and plotting job `259887` in 38 seconds.
- Scheduling: stages are submitted only after their inputs have been
  inspected. `queue_all.sh STAGE` submits one stage with no `afterok`
  dependency and appends the job ID with dependency `none`.
- Compact output: legacy remote
  `/qbigwork2/negro/dirichlet/analysis/step3_scan/` and local
  `analysis/beta_nt_fixed/pilot/` (7.5 MiB). The dry-run and compact-only
  transfer completed on 2026-07-21; raw histories were not synchronized.

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
