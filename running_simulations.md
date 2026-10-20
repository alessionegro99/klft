# Running simulations

## Nt=1 matched-y extension — all simulations complete, analysis not started

The user approved Nt=1, Ns=128 at beta=1.45, 1.50, 4.6/3 and 1.55,
with four independent 8,000-sweep hot chains per coupling (128,000 sweeps).
Stored L2=2; 1 heatbath plus 5 overrelaxation sweeps; holonomy cadence 1,
profiles 5; Wilson loops, flow and horizontal Polyakov loops disabled.
These matched-beta/Nt diagnostics have target_c=null and reference_c=0.5;
they remain excluded from fixed-c continuum inputs until explicitly tuned.
Exactly 500 replicas will be used in the later analysis on qbig.

P100 nodes 13,14,15,17 were idle with all GPUs reporting zero utilization,
zero memory use and no compute processes. Node16 was excluded because its
direct SSH host-key check failed; no host-key policy was changed. Slurm
reports all five Pascal nodes idle. Free project storage was 29 GB; the
runner requires at least 12 GiB free before each run.

The existing clean KLFT revision is f47421d895469b72d4af8a9b8479807f2b09a692.
P100 executable SHA256:
856a79fada4dc2e91ac7bdbf82c347e1392729dfb214eeafd25529eb64f667dd.
Build, compiler, CUDA and dependency provenance are in
`data/mg_nt_fixed/open_slab_build.txt`; CTest, memory-check and benchmark
validation are in `open_slab_validation.txt`. No rebuild was needed.
Fresh seed inventory covered 317 YAML inputs and 297 reserved seeds, plus
local unsubmitted inputs. All 21 new seeds are unique and disjoint from them.
The generator and execution checks passed 33 tests; source commit `66aaf7c`.

Inputs: `data/mg_nt_fixed/nt1_scaling_manifest.json`, under the canonical
`c_0p5/Nt_1/Ns_128/beta_<beta>/seed_<seed>/` tree. Raw outputs remain under
`/qbigwork2/negro/dirichlet/data/mg_nt_fixed/` at the corresponding paths.
Launcher: `data/mg_nt_fixed/nt1_scaling_run.slurm`; one typed Pascal GPU,
one CPU, 8 GB and two hours per task; no dependency or concurrency cap.
Logs: `/qbigwork2/negro/dirichlet/logs/mgnt-nt1-y-<array>_<task>.out`.
Stages are manual: one 10-sweep Ns=4 smoke, four 100-sweep Ns=128 timings,
then production only if the projected total is <=12 GPU-hours and each
chain <=2 GPU-hours. Overwrite and changed-hash checks are enforced.

Immutable inputs commit: `fa8c432`; manifest SHA256:
`cb6d2f52c9e3e0d6477a84dec7f293412034ba026b1cf16110c1bf1e04a7c4b0`.
All 21 uploaded inputs, metadata and source/environment hashes were checked.
Smoke job `291028` completed on lnode17 with exit 0 in 15 seconds (heatbath
0.365 seconds). Its flag-3 checkpoint has Nt=1, L2=2, 80 physical links,
fixed identity surface links and maximum squared-norm error 2.22e-16.
All 10 holonomy measurements, 30 correlator rows and two profiles passed
cadence, finiteness and C_G(0)=1 checks. No statistical analysis was performed.

Timing array `291029_[0-3]` completed with exit 0 on lnode17 in 15--16
seconds per task. The 100-sweep executable wall times were 1.2211, 1.2681,
1.2695 and 1.1691 seconds in increasing beta order. Every run passed 100
holonomy / 6,500 correlator-row / 20 profile checks and the flag-3 boundary
check. The separately approved timing gate projects 0.43803 total P100
GPU-hours and at most 0.028211 GPU-hours (101.56 seconds) per chain.
All raw-output hashes were rechecked before approval. Gate:
`/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/nt1_scaling_timing_gate.json`.
Immediately before production, the four eligible nodes' GPUs were again
idle, the queue empty and free storage 29 GB.

Production array `291033_[0-15]` was submitted without dependencies or a
concurrency cap. Indices 0--7 run on lnode17 and 8--15 on lnode13, each
with its own allocated GPU. The first history was inspected while running;
the first post-start check recorded 484 holonomy measurements for task 0.
No timing or smoke run will enter the production analysis.

All sixteen chains completed by 20:00:54.775 UTC on 20 October (22:00:55
CEST). First heatbath start: 20:00:06.309 UTC. Inspection at 20:01:35 UTC,
more than one minute after the actual start, confirmed all tasks COMPLETED
with exit 0 and no active array jobs. Heatbath times are 29.12--46.56 seconds
per chain, totaling 0.152497569 P100 GPU-hours. Slurm task durations including
startup and validation are 46--65 seconds. All sixteen node/GPU pairs were
distinct. No production failure or overwrite occurred.

The completion checks passed for all 21 smoke, timing and production runs.
Every production chain has 8,000 holonomy measurements, 520,000 correlator
rows, 1,600 profile measurements, and an Nt=1 flag-3 physical checkpoint.
The total production count is 128,000 sweeps. Every input identity/hash,
manifest hash and recorded output size was rechecked; raw-output hashes
were computed on the allocated compute nodes. Compact start/result records
and the timing gate are synchronized locally. The four recorded raw files
per production chain total 477,119,592 bytes on qbig; free storage is 28 GB.
New mass/thermalization/blocking analysis has not been launched. The current
45-ensemble analyzed tables and plots remain unchanged.

Report-only jobs `291049` and `291050` completed with exit 0 on lnode11
in 40 and 39 seconds. The final rebuild keeps the execution note on the
Nt=1 page without adding an almost-empty page. It verifies the unchanged
45 ensembles, 184 chains, 12,721 mass windows and 1,268 plot PDFs, then
compiles the 23-page logbook with zero bootstrap operations. Remote and
local freshness checks pass; the four affected pages were rendered and
visually inspected. Local Poppler checks cover all existing plot PDFs.
Input and compact-completion commits are `fa8c432` and `a1d9bf1`.

| beta | Production seeds |
|---:|---|
| 1.45 | 1183505828, 1652905867, 1492645710, 1776605214 |
| 1.50 | 2054294665, 1698854720, 1813195288, 1644106385 |
| 1.5333333333333333 | 1633219015, 767348993, 1941201623, 1526132928 |
| 1.55 | 922096602, 1947777266, 1617814432, 1856052414 |

## Nt=1 mass-scaling plot update

On 20 October, CPU job `291023` completed with exit 0 in 21 seconds,
adding the three existing Nt=1 masses to the all-coupling mass plots and
adding `comparison/mass_Nt_vs_beta_over_Nt.pdf`. The saved masses and their
500-replica standard errors are unchanged; no simulation, fit or resampling
was performed. The beta/Nt=1.5 comparison and fixed-c fits are unchanged.
The plotting-only mode requests one CPU, 8 GB and one hour in batch,
excluding lnode13--17, without a GPU or dependency. Command:
`sbatch scripts/mg_nt_fixed_scaling_collect.slurm --mass-plots-only`, from
`/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_scaling`.
Logs: `/qbigwork2/negro/dirichlet/logs/mgnt-scaling-collect-<job>.out`.
Source commit: `a9b08e9`.

Regression job `291024` failed only the new plot-test assertion because
Matplotlib exposed object-typed original coordinates. The corrected test
uses Matplotlib's converted numeric coordinates; two focused local tests
pass. Regression rerun `291025` completed with exit 0 in 30 seconds on
lnode11: all 51 tests passed in 14.826 seconds (two CPUs, same memory/time
limits). Scientific estimates are unchanged by this test correction.

Logbook preflight `291026` failed before analysis validation because an index
sync replaced the snapshot's analysis symlink with a one-file directory.
The canonical result directory remained intact. The replacement directory
was preserved at `/tmp/mgnt-plot-sync.BGnS5L` on qbig, and the original link
to `/qbigwork2/negro/dirichlet/analysis` was restored. Subsequent index syncs
use the canonical destination directly. Report verification and compilation
were rerun after this repair; no scientific products were removed.
Job `291027` completed with exit 0 in 1m04s on lnode11, verifying 45
ensembles, 184 chains, 12,721 windows and 1,268 plot PDFs before building
the 23-page logbook. Remote and local freshness checks pass. The three
changed mass plots and the two affected logbook pages were rendered and
visually inspected. Source/test follow-up commit: `fcd31e8`.

## Matched-y scaling diagnostic — simulations and mass analyses complete

Checked on lnode19 at 07:36 UTC on 16 October: the scheduler finished
successfully at 04:05:22 UTC. All sixteen production execution records report
`complete`, with 128,000 measured sweeps. The last chain finished at
04:05:18 UTC (06:05:18 CEST). Total recorded heatbath time is
0.274520466 A100 GPU-hours; chain times range from 45.23 to 100.95 seconds.
The smoke/timing stage started at 03:58:21 UTC, and all three production
workers launched at 03:59:22 UTC. No scheduler or production worker remains
active. GPU 6 was excluded throughout.

SSH access is restored. Compact execution records were synchronized after a
dry run; all 16 production records pass input, identity, geometry, cadence
and unique-seed checks. Raw histories remain on qbig. The earlier failed
download changed no numerical results.

CPU analysis array `291008_[0-3]` uses the immutable
`data/mg_nt_fixed/scaling_analysis_manifest.json` (SHA256
`39576b223df2e060e0d1734ee7b881bcabf4095e4042f1872040907c5fa56106`).
Indices are Nt3 Ns96 beta4.5; Nt4 Ns160 beta6; Nt4 Ns160 beta6.2;
Nt6 Ns320 beta9. Each task requests 8 CPUs, 24 GB, two hours, batch partition,
excluding P100 nodes, with no dependencies or concurrency cap. Snapshot:
`/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_scaling`; logs:
`/qbigwork2/negro/dirichlet/logs/mgnt-scaling-analysis-291008_<index>.out`.
The unchanged chain-aware estimator uses exactly 500 replicas and exhaustive
window scans, independently checking each raw observable's saved hash.
Code commit `31b0fdf`; metadata commit `35d4fe3`; 29 targeted tests pass.
All four groups retain `target_c=null`, `fixed_c_continuum_input=false`.
No new Monte Carlo simulation is launched by this analysis.

Tasks 291008_0, _1 and _2 completed with exit 0 in 3m32s, 8m57s and 10m49s,
on lcpunode02, lnode03 and lnode04. Their masses are 0.1386481 ± 0.0003485,
0.0941488 ± 0.0001933, and 0.0821284 ± 0.0001862; errors are bootstrap SEs.
Task _3 completed with exit 0 in 1h12m47s on lnode05, with mass
0.05630646 ± 0.00012572. Every new chain has a 500-sweep cut.
Blocks are 80, 60, 80 and 240 sweeps for indices 0--3, retaining 93,125,93,31
complete blocks per chain, respectively. The Nt=6 group needs increased
statistics to reach the 50-block target at its present block length.

Plot-only collection job `291012` uses one CPU, 8 GB and a one-hour limit,
excluding P100 nodes, with no dependency. It renders Lorenzo's supplied
trial-exponent collapse and both second-moment length and squared-length
comparisons using only completed compact results. Pending coverage is
explicit in `comparison/scaling_plot_status.json` and figure footnotes.
Log: `/qbigwork2/negro/dirichlet/logs/mgnt-scaling-collect-291012.out`.
It completed with exit 0 in 43 seconds on lnode11, plotting 44 completed
ensembles and recording Nt6 beta9 as pending.

Regression job `291013` completed with exit 0 in 30 seconds on lnode11:
48 targeted tests passed in the locked environment (two CPUs, 8 GB,
one-hour limit, no GPU or dependency). This includes inventory/role checks,
parallel-fit equivalence, covariance-rank accounting and direct-ratio
xi-squared uncertainty propagation. Scheduler-launch messages in its unit-test
log are mocked tests; no simulation was launched. Log:
`/qbigwork2/negro/dirichlet/logs/mgnt-scaling-collect-291013.out`.

Plot-only refresh `291014` completed with exit 0 in 36 seconds on lnode11,
adding focused Nt>=2 length and squared-length comparisons. All sixteen
simulations were already complete: its pending label referred only to the
Nt6 mass-window analysis. The user subsequently requested title-free,
footer-free figures and complete coverage in the existing top-level tables.
The plotting refresh preserves every saved cut, block choice and fit result.

Final collection `291015` completed with exit 0 in 1m42s on lnode11,
verifying 45 ensembles, 184 chains, 1,776,000 measured sweeps and 12,721
mass windows. All five top-level statistical tables contain the full set,
with row counts checked against the per-ensemble tables. The fixed-c tuning
and p=1 continuum CSV checksums are unchanged. Regression `291016` passed
50 tests in the locked environment (exit 0, 31 seconds, lnode11).

Title-free refresh `291017` requests four CPUs, 16 GB and one hour, with no
GPU or dependency. It regenerates per-ensemble figures from saved estimates
and error curves, loading raw histories only for thermalization display.
No new fit or bootstrap draw is performed. It then repeats product
verification. Logs use `logs/mgnt-scaling-collect-<job>.out` under the same
remote root. Source commits: `e09009b`, `b0b1ba4`.
The refresh completed with exit 0 in 4m47s on lnode11; all 45 ensembles were
replotted and repeated verification passed for 1,267 PDFs and 12,721 windows.
Local Poppler QA passed for all 1,267 single-page 16:9 plot PDFs. Logbook
build `291018` completed with exit 0 in 29 seconds on lnode11; its freshness
check passed. The final visual review adds the Nt6 positive-tail caveat and
removes two nearly empty spill pages. Final logbook build `291019` completed
with exit 0 in 30 seconds on lnode11. The resulting 22-page PDF passed visual
review, has no LaTeX layout warnings, and passes freshness checks on qbig
and locally. No simulation or mass-window analysis remains pending in this
mg_nt register.

### Approved inputs and execution history

The user approved this four-ensemble first batch and use of available A100s,
with one of the four free lnode19 GPUs left unused. Execution uses GPUs 3,4,5
on CPU cores 32,33,34; GPU 6 is excluded. GPUs 0,1,2 belong to other work.
P100 nodes lnode13--17 were fully allocated at the original planning check and
are not used. This batch runs directly on the authorized non-Slurm A100 node.

| Nt (links) | L2 (slices) | Ns | beta | beta/Nt | Production seeds |
|---:|---:|---:|---:|---:|---|
| 3 | 4 | 96 | 4.5 | 1.5 | 1932351770, 1957974461, 1636737046, 781803439 |
| 4 | 5 | 160 | 6.0 | 1.5 | 2119748451, 1999456795, 2072474296, 1813985793 |
| 6 | 7 | 320 | 9.0 | 1.5 | 1876858385, 964494047, 1715733354, 734628251 |
| 4 | 5 | 160 | 6.2 | 1.55 | 1165210146, 1729419008, 1766896381, 2034175857 |

Each ensemble has four independent hot-start chains of 8,000 composite sweeps
(one heatbath plus five overrelaxations): 16 chains, 128,000 production sweeps.
Holonomy cadence is 1; plaquette-profile cadence is 5; Wilson loops, horizontal
Polyakov observables and flow are disabled. Future analysis uses exactly 500
bootstrap replicas with the approved chain-aware mg_nt policy.

These points test matched beta/Nt scaling, with a direct Ns=96 versus 160
comparison at Nt=4,beta=6.2. The canonical `c_0p5` folder is the source grid
reference only: `target_c=null`, `fixed_c_continuum_input=false`, campaign type
`matched_y_scaling`. They enter the combined register through a separate
immutable inventory and remain excluded from fixed-c tuning.
The immutable [manifest](data/mg_nt_fixed/scaling_manifest.json) and
[execution plan](data/mg_nt_fixed/scaling_execution.json) preserve this role.
All 21 new seeds (one smoke, four timings and sixteen production chains) are
distinct from the current qbig and local input inventories; master seed 20260724.

Reuse the validated native A100 build, without rebuilding: KLFT
`f47421d895469b72d4af8a9b8479807f2b09a692`, executable
`/qbigwork2/negro/dirichlet/build/klft-a100/binaries/heatbath`, SHA256
`1d0b2d0bacc008b3b3ca16e6aa6c96dafac31f82f48047af0634f8c51b100cd8`.
Build/validation, input, script and locked-environment hashes are recorded.
47 targeted input/execution/scheduling tests pass; the physical Nt=1 checkpoint limit
and fixed surfaces at Nt=3,4,6 are tested. No new mass analysis is performed.

The user's request to schedule jobs when GPUs become free authorizes automatic
progression through the existing stages. There are no Slurm jobs or dependencies.
First run a 10-sweep Nt=6,Ns=4 smoke and four 100-sweep timings on GPU 3.
Production requires successful output checks and projected cost <=12 GPU-hours
total, <=2 hours per chain and <=2 hours per worker. Each worker also enforces
the two-hour timeout, exclusive execution records, an idle-GPU check before
every chain, input/output overwrite refusal and a 12-GiB free-space reserve.

The one-shot scheduler started on **15 October 2026 at 12:20:52 UTC**:
manager PID `3350720`, Python PID `3350734`, node `lnode19`. It polls every
30 seconds and requires two consecutive process-free GPU checks. Once
preflight passes, the three workers start independently as GPUs 3,4,5 become
free; they run 6,5,5 chains respectively. GPU 6 is never used by this batch.
New starts expire on **17 October at 12:20:52 UTC**; in-flight bounded workers
may finish. Failure stops dispatch and its own active workers, with no automatic
retry of failed or partially written runs. No simulation has started at the
initial activation check: all three assigned GPUs remain occupied. Rechecked
at 12:22:27 UTC after 95 seconds: the detached manager remains alive, the
status heartbeat advanced to 12:22:23 UTC, and no stage-start record exists.

Scheduler launcher: `campaigns/mg_nt_fixed/run_scaling_when_free.sh`.
Stage launcher: `campaigns/mg_nt_fixed/run_scaling_a100.sh <stage> <worker>`.
Scheduler log: `logs/dir-scaling-scheduler.out`; stage logs:
`logs/dir-scaling-scheduled-<preflight|production>-<worker>.out`.
Live state: `campaigns/mg_nt_fixed/scaling_schedule_status.json`.
The [automatic-dispatch contract](data/mg_nt_fixed/scaling_schedule.json)
pins source hashes and the original immutable execution plan. The
[start record](data/mg_nt_fixed/scaling_schedule_start.json) and
[status snapshot](data/mg_nt_fixed/scaling_schedule_status.json) are compact
local copies; the latter is current only at its `updated_at` timestamp.
Raw inputs/output:
`/qbigwork2/negro/dirichlet/data/mg_nt_fixed/c_0p5/Nt_<Nt>/Ns_<Ns>/beta_<beta>/seed_<seed>/`.

The earlier manual preflight manager PID 3348454 exited before creating any execution-start
record or running a heatbath sweep. Its idle-GPU guard found GPU 3 occupied
by PID 3348228. At 12:00:49 UTC on 15 October, user `gruber` occupied GPUs
3,4,5,6 with `meson_new` PIDs 3348228,3348232,3348236,3348240. These jobs had
started after our earlier free-device checks. None was altered or overlapped.
That attempt launched no production chain or Slurm job. Its original log
`logs/dir-scaling-preflight.out` and
[dispatch record](data/mg_nt_fixed/scaling_dispatch.json) are preserved.
All 21 inputs were hash-verified and unused immediately before scheduler
activation; direct GPU occupancy checks found existing compute processes on
GPUs 3--6, which remain untouched. Slurm CLI commands were unavailable in the
qbig login environment on this check; no Slurm submission was attempted.
Launch code `98e1edf`, inputs `30fe913`, and planning record `2d83d68` are
committed and pushed to `dirichlet-campaign`, followed by scheduler code
`2543332` and its authorization `451c21f`. Existing analysis is unchanged.

Validation: `UV_CACHE_DIR=/tmp/dirichlet-uv-cache PYTHONPATH=scripts uv run --frozen
python -m unittest scripts.test_mg_nt_fixed_scaling_schedule
scripts.test_mg_nt_fixed_scaling scripts.test_mg_nt_fixed_nt1_run
scripts.test_mg_nt_fixed_nt1_inputs scripts.test_mg_nt_fixed_nt1_a100`.
All 47 tests pass. `bash -n data/mg_nt_fixed/run_scaling_when_free.sh` and
`uv run --frozen python scripts/mg_nt_fixed_logbook.py --check` also pass.

## Nt=1 matched-beta production — all twelve complete and analyzed

Checked 15 October: all twelve 8,000-sweep production chains completed on
14 October by 18:58:02 CEST, with validated counts, hashes and physical
checkpoints. Total: 96,000 sweeps and 3.0381 A100 GPU-hours. All workers
exited successfully; the longest chain took 44.38 minutes. The completion
records are synchronized locally. The four worker result JSONs and each
seed's `execution_result.json` preserve the actual A100 provenance.

The requested analysis uses `nt1_analysis_manifest.json`, three physical
ensembles, independent chain cuts/blocks and exactly 500 replicas. First-stage
analysis ran on Slurm CPU nodes using
`scripts/mg_nt_fixed_nt1_analysis.slurm --first-stage-only` (8 CPUs, 24 GiB,
two-hour limit, no GPU). Job array 290998 completed all three first-stage
analyses on lcpunode02/lnode03/lnode04. No new simulation is submitted.
The user approved coarse endpoints followed by dense plateau checks on
15 October; the large Ns=512/2048 analyses retain every integer separation
inside each tested window and all 500 refits. Ns=128 retains the exhaustive
scan. Recommendations from the adaptive scan are earliest among tested
patches and are explicitly flagged for review. Raw histories remain on qbig.
Mass-fit array 291001 completed all three tasks with exit 0: Ns=2048 on
lcpunode02 (5m04s), Ns=512 on lnode03 (2m39s), and Ns=128 on lnode04 (7m24s).
Preliminary masses are respectively 0.0067043124 ± 0.0000377466,
0.0287938046 ± 0.0000630431, and 0.1411131263 ± 0.0002194111.
The largest box has only 6--11 complete blocks per chain and unresolved
blocking/chain-consistency diagnostics; these conditional fit errors do not
include equilibration or fit-window bias. Collection/verification/logbook
job 291004 completed with exit 0 in 1m23s. A final display/index rebuild is
job 291005, completed with exit 0 in 1m22s, separately without Slurm dependencies.
All 1,105 plot PDFs passed local single-page 16:9 checks. The 17-page
logbook was visually inspected and its source/output freshness check passed.
The fixed-c tuning and continuum CSV checksums are unchanged by the Nt=1
addition. No further simulation or analysis job remains active for this task.
Report-only job 291006 completed with exit 0 in 27 seconds. Its freshness
manifest tracks the active versioned pipeline and shared dependencies;
unused historical helper scripts no longer invalidate a fresh checkout.
All 33 targeted analysis/provenance/report tests pass. Analysis snapshot:
`/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_nt1/`; logs:
`/qbigwork2/negro/dirichlet/logs/mgnt-nt1-analysis-291001_<index>.out` and
`/qbigwork2/negro/dirichlet/logs/mgnt-nt1-collect-291006.out`.
P100 validation 290996 completed successfully on lnode16 on 15 October,
05:03:35--05:04:13 CEST: six CTests, CUDA memory check and benchmark passed.

The user approved A100 production after the native smoke/timing checks. The
execution plan is [nt1_a100_production.json](data/mg_nt_fixed/nt1_a100_production.json).
It reuses the twelve unused 8,000-sweep inputs and independent seeds from
`nt1_manifest.json`; original P100 planning metadata is retained unchanged.
The actual A100 executable, execution-plan hash, worker PID and GPU UUID are
recorded in each seed directory's `execution_start.json` and `execution_result.json`.

Four direct workers on lnode19 use GPUs 3--6 and CPU cores 32--35, respectively.
Fresh preflight found those GPUs idle; other projects on GPUs 0--2 are left
untouched. Each worker runs its three chains in increasing Ns order, with a
two-hour whole-worker timeout, an exclusive worker lock, per-run overwrite
refusal and a 12-GiB free-disk reserve. All 36 launcher/input tests pass.
There is no Slurm submission or automatic analysis for this A100 execution.

| Reference c at Nt=2 | Exact beta | Nt=1 Ns | Production seeds (workers 0, 1, 2, 3) |
| ---: | ---: | ---: | --- |
| 2 | 1.962386866460073 | 128 | 1178212110, 821564244, 1405694367, 2113319249 |
| 1 | 2.5044558261064154 | 512 | 696671852, 1410713389, 941163981, 1741805734 |
| 0.5 | 2.989543168360997 | 2048 | 1606565234, 586569993, 1641392374, 1927164974 |

All runs use Nt=1 temporal link, L2=2 slices, hot starts, five overrelaxations,
holonomy cadence 1, profile cadence 5 and no Wilson measurements. Actual Nt=1
masses and volume adequacy remain to be measured; `target_c` is null and these
inputs are excluded from fixed-c continuum fits. Raw paths retain the canonical
`data/mg_nt_fixed/c_<reference_c>/Nt_1/Ns_<Ns>/beta_<beta>/seed_<seed>/` tree.

The native binary is the validated `build/klft-a100/binaries/heatbath`, KLFT
`f47421d895469b72d4af8a9b8479807f2b09a692`, SHA256
`1d0b2d0bacc008b3b3ca16e6aa6c96dafac31f82f48047af0634f8c51b100cd8`.
The total timing projection is 3.859 A100 GPU-hours, largest chain 54.18 min.
Launcher: `campaigns/mg_nt_fixed/run_nt1_a100.sh <worker>`; worker logs:
`logs/dir-nt1-a100-worker-<worker>.out`.

Launched at 18:11:32 CEST on 14 October. Timeout-manager PIDs for workers
0--3 are `3196198`, `3196199`, `3196200`, `3196201`; the Python worker PIDs
and exact start times are in `data/mg_nt_fixed/nt1_a100_worker_<worker>_start.json`.
Inspected live at 18:12:40 after more than one minute. At 18:13:48, all eight
Ns=128/512 chains had completed 8,000 sweeps and passed the complete-output,
cadence, C_G(0), exact-boundary and checkpoint-unitarity checks. Process times
were 30.67--31.27 s (Ns=128) and 77.12--78.82 s (Ns=512). All four Ns=2048
chains were running, with 61--68 measurements already written. No failure or
restart occurred; no mass analysis has been performed. Raw histories remain
on qbig; only compact execution records are synchronized.

Production code commit: `45b4fe1`; execution-plan commit: `c1f28a2`. The plan
SHA256 is `31b2ab1c5d9513e4effdcdd4063d9f5839b09305ff1b6e14900e89be562b292b`.
The separate campaign Git branch is `dirichlet-campaign` in
`git@github.com:alessionegro99/klft.git`; simulation source remains on
`orbifold`. Existing unrelated uncommitted campaign work remains local.
P100 validation `290996` subsequently completed as recorded above and submits no production.

## A100 open-slab test — completed successfully

The user authorized a direct-node test on lnode18/19. At preflight, lnode18
was occupied; lnode19 GPUs 3--6 had no compute processes. The test uses only
lnode19 GPU 3, UUID `GPU-439f10b2-3f2f-c6d2-12cc-deca128ca0f2`, with an
occupancy recheck before each GPU stage. Its separate native 3D SU(2) build
is `/qbigwork2/negro/dirichlet/build/klft-a100`, from pushed revision
`f47421d895469b72d4af8a9b8479807f2b09a692`, CUDA 12.6/GCC 13.3 and AMPERE80.
The established environment order is `setup_qbig3.sh`, then
`load_modules_qbig3.sh`. The P100 binary is unchanged; its separate validation
failed during driver loading and was resubmitted as `290996`, as recorded below.

Scope: all six CTests, CUDA memory checking, one eight-sweep Nt=1 smoke test
(Ns=4, seed 1468374202), then 100 sweeps at each exact matched beta, increasing
Ns: 128/seed 1739874403, 512/seed 683869512, 2048/seed 1012559264. Inputs are
in the canonical physics/seed tree and listed in
[a100_test_manifest.json](data/mg_nt_fixed/a100_test_manifest.json). Each uses
five overrelaxations, holonomy cadence 1 and profile cadence 5. All four seeds
are new. No production is authorized by this test; no mass or bootstrap is
computed. A100 timings cannot replace the P100 timing gate.

The direct test is bounded to 45 minutes, one GPU and CPU affinity 32--35,
with four build workers and one OpenMP thread for tests/runs. The launcher
refuses existing test outputs and an occupied GPU. Remote records and log:
`campaigns/mg_nt_fixed/open_slab_a100_*` and `logs/dir-open-a100.out`.

Launched at 17:18:24 CEST on 14 October, manager PID `3180516`, worker
`3180518`. The first live inspection found the native build running normally.
Code commit: `afb6881`; immutable-input commit: `0e03f21`. All five A100 setup
tests and nineteen shared Nt=1 input/execution tests pass. Input manifest SHA256:
`d12394b5a3e8d75d7e50f304ae90d2be0f5de96256d1e2f39e406f752cd2c3cf`.
The remote launcher, runner and manifest hashes match their local originals.

The manager and worker have exited with code 0. All six native GPU CTests
passed (16.62 s total), including the statistical test (13.88 s). CUDA
compute-sanitizer reports zero errors. Its deliberate invalid-input/checkpoint
rejection messages are negative-test output; the deterministic test passed.
All four short runs passed measurement-count, cadence, C_G(0), exact-surface
and physical-checkpoint checks. The largest squared SU(2) norm error was
4.44e-16. GPU 3 was released after completion; unrelated GPU processes on
devices 0--2 remained running. No production was launched.

| Ns | beta | Sweeps | Heatbath process time | Projected time for 8,000 sweeps |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 2.989543168360997 | 8 | 0.616 s | smoke only |
| 128 | 1.962386866460073 | 100 | 0.917 s | 1.22 min |
| 512 | 2.5044558261064154 | 100 | 1.868 s | 2.49 min |
| 2048 | 2.989543168360997 | 100 | 40.634 s | 54.18 min |

The timings include process startup, all measurement output and final
checkpoint writing. Linear projection gives 3.859 A100 GPU-hours for four
8,000-sweep chains at each beta. These are planning estimates from short hot
starts; thermalization, statistics and physical volumes remain unvalidated.
Native heatbath SHA256:
`1d0b2d0bacc008b3b3ca16e6aa6c96dafac31f82f48047af0634f8c51b100cd8`.
Compact [build](data/mg_nt_fixed/open_slab_a100_build.txt),
[validation](data/mg_nt_fixed/open_slab_a100_validation.txt) and
[run/timing records](data/mg_nt_fixed/open_slab_a100_runs.json) are local;
raw histories remain on qbig. The P100 binary checksum is unchanged. Subsequent
production approval and execution are recorded in the section above.

## Open Dirichlet geometry — completed P100 validation and historical preparation

KLFT `f47421d895469b72d4af8a9b8479807f2b09a692` is committed and pushed to
`origin/orbifold`. It uses open storage with no closing temporal layer
(introduced by `c78efbd`), with a CUDA 11.5 initialization compatibility fix.
Six CTests, five sanitizer CTests, project analysis tests and a small
driver/restart smoke test pass; see the
[validation and audit](summary.md#local-removal-of-the-auxiliary-wrapping-sector).

Planned diagnostic simulations use `Nt=1` at the current `Nt=2` tuned medians:
beta 2.989543168360997, 2.5044558261064154 and 1.962386866460073, associated
with reference c values 0.5, 1 and 2. Their actual `mG Nt` values are to be
measured; they must not be inserted as fixed-c continuum points. The working
plan is four independent 8,000-sweep hot chains per beta, five overrelaxations
per heatbath sweep, holonomy every sweep, profiles every five sweeps and
Wilson loops disabled. The provisional spatial sizes are respectively 2048,
512 and 128. They require timing and a subsequent finite-volume assessment.
These were the original preparation settings. All twelve production chains
subsequently completed on A100s; the current status is recorded above.

After the goal was resumed, the committed source was transferred with the
permission-gated SSH route. The verified Git bundle checksum is
`a3ed2c98d90890e8636a5e0acd1e7faddd2ef2b45a490c73eb1e77a95837ce24`
for `klft-f47421d.bundle`.
The clean checkout is `/qbigwork2/negro/dirichlet/src/klft`, with pinned Kokkos
`6e357fa03b1e01454dd0fcac01f7af57e9db20c4` and yaml-cpp
`28f93bdec6387d42332220afa9558060c8016795`. The older modified checkout
`/hadron/negro/src/klft` remains untouched. Its P100 build was moved intact to
`/qbigwork2/negro/dirichlet/archive/klft_builds/41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57/`,
including a saved source diff. Its heatbath executable retains the directory's
SHA256. No raw history was moved or changed.

- Build job `290993`: FAILED with exit 2 after 2m04s on `lcpunode02`, four
  CPUs, 8 GB. CUDA 11.5 rejects an extended device lambda inside a constructor;
  the unchanged initialization kernel was moved into a normal helper in
  `f47421d`. No new
  executable or build-provenance file was produced. Sole active build path:
  `/qbigwork2/negro/dirichlet/build/klft`.
  Script: `/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/build_open_slab.slurm`.
  Log: `/qbigwork2/negro/dirichlet/logs/dir-open-build-290993.out`.
- Build retry `290994`: COMPLETED, exit 0, 4m24s on `lcpunode02`, four
  CPUs, 8 GB, 45-minute limit. New P100 heatbath SHA256:
  `856a79fada4dc2e91ac7bdbf82c347e1392729dfb214eeafd25529eb64f667dd`.
  [Build provenance](data/mg_nt_fixed/open_slab_build.txt) SHA256:
  `ee8749ae838939e7402eb8693a3ec82cf3b94773f79265de5f47238713c837ae`.
  Log: `/qbigwork2/negro/dirichlet/logs/dir-open-build-290994.out`.
- P100 validation `290995`: FAILED, exit 8, on `lnode17`, 17:13:22--17:13:35
  CEST on 14 October. All six test processes failed during CUDA initialization
  with `cudaErrorStubLibrary`, before numerical validation. The CPU build's
  RUNPATH selected `/opt/eb/software/CUDA/11.5.0/lib/stubs/libcuda.so.1`.
  No successful-validation record or simulation history was produced.
  Log: `/qbigwork2/negro/dirichlet/logs/dir-open-check-290995.out`.
- Runtime correction `b314b01` explicitly preloads the installed driver
  `/usr/lib/x86_64-linux-gnu/libcuda.so.1` in both P100 GPU launchers. A read-only
  loader check on `lnode17` resolves the real driver while retaining the module
  CUDA/GCC libraries. The binary and build provenance are unchanged. Five
  launcher safeguards and nineteen Nt=1 input/execution tests pass.
- P100 validation retry `290996`: COMPLETED, exit 0, on lnode16,
  05:03:35--05:04:13 CEST on 15 October. All six CTests, CUDA memory
  checking and the benchmark passed. It requested one typed Pascal GPU,
  one CPU, 8 GB and 45 minutes, with no dependencies or follow-up submissions.
  The corrected script's local/remote SHA256 is
  `797d63c7920ce9f836e7244aaf3942bbb94607f7319f99325ebe7d9be23c6e77`.
  Log: `/qbigwork2/negro/dirichlet/logs/dir-open-check-290996.out`.
  [Attempt record](data/mg_nt_fixed/open_slab_validation_attempts.json).
- P100 Nt=1 smoke, timing and production: not submitted. Its GPU validation
  and production-cadence timing must pass first. Recheck free space before
  production; the post-A100 check found about 41 GB free on qbigwork2.

The manually staged [CPU build](data/mg_nt_fixed/build_open_slab.slurm) and
[P100 validation](data/mg_nt_fixed/validate_open_slab.slurm) scripts are
committed locally as `8c06326`, updated by `4f92581` and `b314b01`, with five passing
safeguard tests. They bind
the pushed KLFT revision, verify executable/test hashes, preserve existing
build provenance and refuse execution outside Slurm. They submit no other
jobs. GPU validation includes CUDA memory checking. The diagnostic benchmark
uses two overrelaxations; a separate
100-sweep measurement-enabled timing run with the production value of five
is still required. Both scripts are transferred and submitted manually.
The KLFT commit is pushed on `origin/orbifold`. The user subsequently approved
publishing this independent project history to the same remote on the separate
`dirichlet-campaign` branch, preserving `master` and `orbifold`.

The committed `scripts/mg_nt_fixed_nt1_inputs.py` has seven passing tests.
It generates one smoke input, three timing inputs and twelve production
inputs only after a P100 executable checksum is available. The seed inventory
(`06338d1`) covers 276 qbig input files and 256 distinct seeds. New seeds are
distinct positive int32 values from master seed 20260724. All sixteen inputs
are committed as `87b91f8` and transferred, with a clean checksum-based rsync
verification. That transfer brought the input count to 292; the separate A100
test added four more inputs. The four pre-existing reused
seeds (32091, 90517, 177013, 515029) are unchanged; all sixteen new seeds occur
once. [Nt=1 manifest](data/mg_nt_fixed/nt1_manifest.json) SHA256:
`0bd0baedbfee731080568edb396a1ee31a287c489f8550bdc4431ba69878cd01`.

The [launcher](data/mg_nt_fixed/nt1_run.slurm) and
[execution checker](scripts/mg_nt_fixed_nt1_run.py) are committed as `08a41e4`
and transferred to `campaigns/mg_nt_fixed/` on qbig. Twelve execution tests pass
(19 together with the input tests). The checker binds the build and input
hashes, refuses existing outputs, checks every measurement's order and cadence,
C_G(0), finite primitive observables, exact boundary links and all SU(2) norms
in the flag-3 checkpoint. It records wall time and raw-file checksums. A
separate manual timing gate requires projected costs <=2 GPU-hours per chain
and <=12 GPU-hours total. No script submits follow-up jobs. All new analyses
will use 500 bootstrap replicas.

## Current fixed-mass status

Canonical reanalysis is complete: 38 physical ensembles, 156 chains,
10 preliminary tuning points and three p=1 continuum comparisons, all with
500-replica bootstrap propagation. Final collection `290908` completed with
exit zero in 1m25s; scientific verification covers 6,224 mass windows. Local
Poppler checks passed for 1,017 single-page 16:9 plot PDFs. All 53 targeted
tests pass. No new simulation was submitted and no campaign analysis job
remains active. See [the current report](docs/mg_nt_fixed/summary.md) and the
canonical reanalysis execution record at the end of this file.

The stage-named checkpoints below are historical. Their original compact
products and inputs are preserved in the checksum-verified qbig archive;
current local products use only the `c/Nt/Ns/beta` tree.

## Fixed-mass completed-result review, 2026-10-14

At the 10:04 CEST check, production `290687_[0-19]`, historical reanalysis
`290658_[0-27]`, and continuation analysis `290707_[0-4]` are all COMPLETED
with exit zero. No fixed-mass production or mass-analysis job remains active.
The longest continuation analysis took 1h40m07s (Ns=128). All 33 compact
analysis products are downloaded; raw histories remain on qbig.

- CPU collection `290835`: COMPLETED, exit 0, 15 s; recomputes the nine
  historical tuning groups using corrected 500-sample mass distributions.
  Script: `data/mg_nt_fixed/bootstrap_correction_20261012/qbig/collect.slurm`.
- Current code snapshot: `/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_review_20261014/`.
  Review/plot submission script: `data/mg_nt_fixed/review_20261014/qbig/review.slurm`.
  One CPU, 4 GB, 15-minute limit, no dependencies; no simulation is launched.
  Review products: `analysis/mg_nt_fixed/review_20261014/`; multi-ensemble
  figures: `analysis/mg_nt_fixed/comparison/review_20261014/`.
- `290836`: review and plotting on lnode11, failed after 3m21s while moving
  temporary comparison PDFs across filesystems. All numerical results and
  per-simulation plots were already saved. A tested `shutil.move` fallback
  fixes the reporting error; recovery regenerates comparisons without bootstrap.
- `290838`: COMPLETED, exit 0, 1m01s; rejection diagnostics, all 396 raw
  checksum rechecks, and one new 500-replica tuning using the existing Ns=96
  near-target point. That refinement passes; six of ten current groups pass.
  Script: `data/mg_nt_fixed/review_20261014/qbig/rejected_fits.slurm`.
- `290839`: report preflight failed (one of 28 tests) because the fresh code
  snapshot lacked `uv.lock`; numerical analysis was unaffected. The original
  `uv.lock` and `pyproject.toml` are now staged, without changing dependencies.
  Report script: `data/mg_nt_fixed/review_20261014/qbig/report.slurm`.
- `290840`: all 28 tests passed, comparison plots and 70-record PDF built;
  final freshness check correctly failed after the plotting source was updated
  during the job. Rerun with unchanged sources before final publication.
- `290841`: COMPLETED, exit 0, 55 s; all 28 regression tests pass, comparison
  plots recovered without resampling, corrected confirmation gate passes,
  20-page logbook with 70 historical/current records built. The final
  `mg_nt_fixed_logbook.py --check` passes and LaTeX reports no overflow warnings.
  Final build provenance: `analysis/mg_nt_fixed/logbook/build_manifest.json`.
  No fixed-mass jobs remain active after this review; no new simulation submitted.

The dated checkpoints below are historical. Current gate decisions are in
`docs/mg_nt_fixed/summary.md` and the PDF logbook.

## Fixed-mass correction and continuation, 2026-10-12

The user approved the fixed-chain block-bootstrap correction and dispatch.
All new bootstrap operations use exactly 500 samples on qbig compute nodes.

- `290656`: corrected-estimator validation, one CPU/8 GB, completed on
  `lnode11` in 44 s, exit 0. All 18 targeted tests, IID and AR(1) variance
  checks pass. Confirmation fixed-window checks at B=30/80/160 remain within
  the 3% target tolerance. This diagnostic does not replace the full gates.
- `290657`: continuation smoke/timing job, one typed P100, two CPUs/5 GB,
  15-minute limit; failed after 18 s on lnode17 in the final awk projection,
  after all four histories and output checks completed successfully.
  `290659` rechecks the completed inputs and corrected projection, passes in
  11 s, exit 0. Gate PASS: slowest 100 sweeps 3.15 s, longest projected
  chain 0.07 GPU-hours, total 1.4 GPU-hours. No history was overwritten.
  Four short runs cover Nt=6,Ns=96 and Nt=4,Ns=128;
  production requires correct counts, C_G(0)=1, hashes, and projections
  below two GPU-hours per chain and 12 total.
- `290687_[0-19]`: all 20 tasks COMPLETED, exit 0, after being observed running
  beyond one minute on nodes 13/14/17. One P100, two CPUs, 5 GB,
  two-hour limit each. No dependencies or concurrency cap.
  Every chain reached 8000 sweeps and passed all output checks. Total
  executable time 0.768017 GPU-hours; allocation time 0.935278 GPU-hours.
  Longest Slurm task 4m30s. Raw histories remain on qbig.
- `290707_[0-4]`: new-ensemble analyses submitted after production completion,
  one CPU/8 GB/four hours each, no dependency. Each writes first-stage
  thermalization and blocking PDFs before the full mass scan.
- `290658_[0-27]`: full corrected mass reanalysis running on CPU nodes
  11, 03, 04 and lcpunode02, one CPU/8 GB/four hours per ensemble.
  P100 nodes excluded from CPU analysis allocation. Logs:
  `/qbigwork2/negro/dirichlet/logs/mgnt-corrected-mass-290658_<index>.out`.
  At 12:07:42 CEST, 11 completed and 17 were running; no failures.
- Immutable inputs: `data/mg_nt_fixed/continuation_20261012/manifest.tsv`:
  Nt=6,c=1,Ns=96,beta=6.40/6.70/7.00 and
  Nt=4,c=0.5,Ns=96/128,beta=5.58394149; four hot 8000-sweep chains each.
  Explicit seeds are master-seed 20260724 draws 133--152 (zero based);
  draws 153--156 are short validation runs. Cadences 1/5; Wilson loops off.
  The matched-volume beta is the historical near-target value; corrected
  target acceptance remains pending. No new tuning is claimed by placement.
- Sole unchanged P100 binary: `build/klft/binaries/heatbath`, SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`,
  KLFT `7eed1f7d9bb0948621a2d4020706d19c56406001`. No dependency edits/rebuild.
- Remote input root `/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/continuation_20261012/`;
  raw root `/qbigwork2/negro/dirichlet/data/mg_nt_fixed/continuation_20261012/`;
  logs `/qbigwork2/negro/dirichlet/logs/mgnt-cont-{validate,prod}-*.out`.
  All jobs are dependency-free; production has no concurrency cap.
  Node 16 is excluded because its direct occupancy check is blocked by host-key
  verification. Fresh checks found eight free P100s each on 13/14/15/17;
  node 15 has no free CPUs. Slurm assigns one typed Pascal GPU per task.
- The 27 selected old tuning inputs plus the independent confirmation are
  catalogued for complete reanalysis under
  `analysis/mg_nt_fixed/bootstrap_correction_20261012/`; historical outputs
  remain unchanged. Code snapshot:
  `/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_corrected_20261012/`.

## Fixed-mass pre-dispatch code audit, 2026-10-12

Historical audit status before the subsequent correction approval:
production had not been submitted; the analysis estimator required review.
The requested continuation remains `Nt=6,c=1,Ns=96,beta=6.4,6.7,7.0`
and a matched `Nt=4,c=0.5,beta=5.58394,Ns=96,128` volume check.
New bootstrap operations use exactly 500 samples, superseding the earlier
1000 policy. All new numerical work runs on qbig.

- `290651`: one CPU, 8 GB, 20-minute limit; completed on `lnode11` in 34 s,
  exit 0. It audits the IID bootstrap limit, refits existing confirmation
  histories at blocks 30/80/160 with the fixed accepted window, and times
  deterministic synthetic fits. No new Monte Carlo history is generated.
- `290652`: one typed Pascal GPU, two CPUs, 5 GB, 10-minute limit,
  single-node request `lnode13`; finished in 22 s, exit 8. Four of five
  CTests pass. `gradient_flow_check` fails with a device-memory access error;
  both Dirichlet and both partitioning tests pass. Gradient flow is disabled
  in the campaign. No rebuild was attempted.
- Report jobs `290653` and `290655`: one CPU, 4 GB, 15-minute limits;
  both completed on `lnode11` (49 s and 32 s), exit 0. All 12
  policy/routing/tuning tests pass. Final job `290655` builds the 14-page
  logbook and grid-free diagnostic plots and passes the source/output
  freshness check. The PDF and compact analysis products were synchronized
  after dry runs. No numerical analysis or plotting ran locally.
- All jobs are dependency-free. Sources and Slurm scripts are recorded under
  `data/mg_nt_fixed/code_audit_20261012/` and remote snapshot
  `/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_20261012/`.
  Compact outputs: `/qbigwork2/negro/dirichlet/analysis/mg_nt_fixed/code_audit_20261012/`.
  Logs: `/qbigwork2/negro/dirichlet/logs/mgnt-code-audit-290651.out` and
  `mgnt-code-ctest-290652.out`. Only the 9-KB JSON/test log was downloaded,
  after a dry run; raw data remains on qbig.
- At submission, `lnode13` was mixed with 12 unallocated CPUs and eight
  P100s without compute processes. Direct checks also found no GPU compute
  processes on nodes 14, 15 and 17; node 15's CPUs were fully allocated.
  Node 16's direct occupancy check was blocked by SSH host-key verification;
  that security check was not bypassed. Availability must be rechecked.
- The sole P100 executable still has SHA-256
  `41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
  Remote KLFT HEAD is `7eed1f7d9bb0948621a2d4020706d19c56406001`, with five
  dirty Wilson-related source/test files, preserved unchanged. Findings and
  the next-step decision are in `docs/mg_nt_fixed/summary.md` and the logbook.

## Fixed-mass logbook check, 2026-10-12

At 10:16 CEST, qbig had no active fixed-mass campaign jobs. Final tuning
`289431`, confirmation production `289432_[0-3]`, diagnostics `289436`,
analysis `289437`, and gate `289438` are `COMPLETED`, exit code zero.
The local and remote final tuning and confirmation-summary SHA-256 hashes
match. Other projects have active jobs; the August idle-GPU snapshot below
is historical.

A two-CPU, 4-GB, 15-minute Slurm task `290650` completed in 19 seconds
with exit code zero. It exported compact
100-sweep presentation means for three logbook examples: the confirmation,
the 80000-sweep beta-4.40 replacement, and the beta-6.20 pilot. It reads
existing raw histories on qbig and performs no bootstrap or simulation.
Script: `data/mg_nt_fixed/logbook/qbig/export_histories.slurm`;
output: `analysis/mg_nt_fixed/logbook/history_examples_20261012.json`;
logs: `/qbigwork2/negro/dirichlet/logs/mgnt-logbook-290650.out`.
The 185-KB JSON was synchronized after a dry run. No raw histories were
transferred and no new GPU simulation was submitted. The logbook is
`docs/mg_nt_fixed/logbook.pdf`; per-simulation examples are under each
result's `plots/first_stage/` directory. Every new bootstrap uses exactly
500 replicas under the subsequent user update; the August 10000-replica
results below are historical.

As checked on 2026-08-28 at 10:52 CEST, the user queue was empty,
`lnode13`--`lnode17` were Slurm-idle, and direct inspection found no GPU
compute processes on their 40 P100s. This is an availability snapshot, not a
reservation. Fixed-beta/Nt production tasks `282274_[36-39]` remain timed
out. Fixed-mass extension analyses `282506_[0-1]` and round-1 tuning `282508`
remain complete.

## Nonperturbative fixed-`m_G Nt` final checks

- Status: complete. Smoke, timing, production, validation, diagnostics,
  analyses, corrected volume/statistics gates, 10000-replica reanalyses, and
  final tuning all passed. The independent confirmation is also complete and
  accepted.
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
- Production completed with exit code zero: the four 80000-sweep chains took
  12:27--12:47 and the eight volume chains took 1:58--2:11. The manifest-wide
  validator passed all 12 chains. Diagnostics array `289424_[0-2]` completed
  with exit code zero. Selected block lengths are 80, 60, and 60 sweeps,
  leaving `992/993/993/993`, `125/125/125/125`, and `125/125/125/125` complete
  blocks. Every primary and secondary blocking plateau passes; the maximum
  split-history discrepancies are 2.731, 2.451, and 2.465 standard deviations,
  and maximum between-chain discrepancies are 3.000, 2.839, and 2.860.
- Initial 10000-replica array `289418_[0-2]` failed in 11 seconds because the
  wrapper required `run_complete.txt`, which predates these accepted pilot
  histories. It produced no compact output. Commit `1919b0d` replaces that
  precondition with the accepted prior diagnostics and analysis plus the raw
  observable files. Corrected array `289421_[0-2]` completed with exit code
  zero in 5:15--8:45. All three selected fits contain exactly 10000 final
  replicas; their recentered-bootstrap fit p-values are 0.9960, 0.8831, and
  0.8387 at beta 3.70, 4.00, and 4.30.
- Mass-analysis array `289427_[0-2]` completed with exit code zero in 41:18,
  2:37:47, and 1:20:31. The beta-4.40 replacement has 0.204% relative mass
  error. At beta 4.95 and 5.30, the `Ns=48` and `Ns=96` masses agree within
  0.422 and 0.730 combined standard deviations; `xi_2nd` agrees within 0.480
  and 0.480 standard deviations, and the larger-volume 68% lower bounds on
  `Ns/xi_2nd` are 17.375 and 14.390.
- Gate job `289430` stopped before tuning because comparing raw periodic
  correlators through the smaller box midpoint confounds volume dependence
  with the known periodic image. The failed diagnostic record is retained as
  `final_checks_summary_failed_raw_periodic_gate.json`. Commit `f6bc4e4`
  gates on the extracted mass and `xi_2nd` while retaining the raw-correlator
  comparison as a diagnostic. Corrected job `289431` completed in 16 seconds
  with exit code zero. All nine final tunings pass. The borderline
  `(c,Nt)=(2,4)` fit has `p=0.05155`; the independent confirmation coupling is
  `beta=4.61755576172643`.

## Nonperturbative fixed-`m_G Nt` confirmation

- Status: complete and accepted; final gate `PASS`.
- Ensemble: `(physical Nt,stored L2,Ns,beta)=(4,5,48,4.61755576)`, target
  `c=1`, four independent 8000-sweep hot-start chains with reserved seeds
  `816415709`, `1698854721`, `738942754`, and `1813195289`.
- Input manifest SHA-256:
  `6e2d150f16934334ecacf20b53a4554c91f8d7f06511bc87f0e199a51cfff0f9`.
  The recorded final tuning-summary SHA-256 is
  `92ccc18ac8fa55c53130eed7c9f9b409b8d256258629bdf7e2e387451ea5fb44`.
- Acceptance: after chain-aware diagnostics and mass analysis, require the
  measured `m_G Nt` to agree with one within the wider of one combined
  standard deviation and 3%. No scheduler dependencies are used.
- Execution: at 15:26 CEST the user queue was empty, `lnode13--17` were idle,
  and direct checks found no GPU compute processes on all 40 P100s.
  Production array `289432_[0-3]` completed on `lnode13` in 1:00--1:01 with
  exit code zero; all four 8000-sweep histories passed exact input-hash and
  measurement-count validation. Diagnostics job `289436` completed in 17
  seconds. All chains use a 500-sweep cut and 30-sweep blocks, leaving 250
  blocks per chain; every primary and secondary plateau passes. Analysis job
  `289437` completed on `lnode11` in 11:48 with exit code zero. Its selected
  `R=4--24` fit has recentered-bootstrap `p=0.4266`, 1000 finite replicas,
  and an accepted mass/`xi_exp` plateau. The result is
  `m_G Nt=1.006200 +/- 0.006542`. Gate job `289438` completed in 12 seconds
  and passed: the 0.006200 absolute target discrepancy is below the 0.03
  acceptance threshold.

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
# Canonical fixed-mG Nt reanalysis

Implementation complete: no new Monte Carlo simulations submitted.
CPU validation job `290845` completed with 13 tests passing, checking fixed-chain weighted pooling and
the unchanged shared bootstrap. Resources: batch, 1 CPU, 4 GB, 20 minutes,
excluding P100 nodes. Log: `/qbigwork2/negro/dirichlet/logs/mgnt-canonical-tests-290845.out`.
Code snapshot: `/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_reorganization`.
Immutable inventory: `data/mg_nt_fixed/manifest.json` (38 ensembles, 156 chains,
1,552,000 sweeps). The four 80,000-sweep chains are explicitly flagged.
Original raw paths and input hashes are retained in that manifest; no raw
history is downloaded or moved. Exactly 500 replicas per bootstrap operation.
Pre-reorganization local products, including uncommitted work, are recoverable
from the verified qbig archive recorded in the manifest (SHA256
`84b0b7ca8e3c6b07f6ea8dc0dd07976e6cb0a57e90dc2499ee459abc4169aef6`).
Smoke analysis `290846_34` completed, exit 0, in 2m03s. It reanalysed the
existing Nt3 Ns32 beta3.25 ensemble with a 20-sweep block and retained its
numerical mass, 0.4748065091 ± 0.0018962661, flagged for between-chain review.
Full canonical reanalysis array `290847` covers indices `0-33,35-37` of the
immutable manifest (37 remaining ensembles). Each task requests one CPU,
8 GB and four hours, excluding P100 nodes, without dependencies or a
concurrency cap. Logs: `/qbigwork2/negro/dirichlet/logs/mgnt-canonical-290847_<index>.out`.
No simulation executable is rebuilt or run; all raw inputs remain unchanged.
Array `290847` was inspected running after more than one minute on CPU nodes
lnode03--06, lnode11 and lcpunode02. The initial input hashes/counts pass,
including the pooled eight-chain case. Regression job `290885` failed on a
logbook string-escaping error, which was corrected before report generation.
Job `290886` then passed 32 active fixed-mass tests and 13 fixed-beta analysis
tests; six historical generator/workflow test modules are skipped when their
archived inputs are absent. Together with the six shared-bootstrap tests in
`290845`, there are 51 distinct passing tests. The retired generators and their
test fixtures remain reproducible in the checksum-verified pre-change archive.
Parallel-scheduler test `290889` passed exact equality of serial/parallel
500-replica parameter arrays and window selections (one additional test).
The serial Ns128 scan required about 17 minutes for its first 118/550 windows.
Tasks `290847_13`, `_24` and `_27` were cancelled to replace their independent
window scans with eight-CPU jobs `290891_[13,24,27]` (8 GB, two hours,
P100 nodes excluded). Their partial diagnostics are recoverable under the
verified archive's `superseded_partial_analyses/` directory. Remaining tasks
continue without modification. Bootstrap inputs and the fit objective are
unchanged. `290887` found no selected-block changes after eliminating numerical
roundoff in exactly fixed observables for the then-completed ensembles.
All 38 ensemble analyses are now complete, exit zero: smoke `290846_34`,
34 completed tasks of `290847`, and three replacements in `290891`.
Parallel replacements completed in 7m52s (Ns128), 5m40s (Nt4 beta4.95),
and 4m25s (Nt6 beta6.7). Campaign/input-identity tests `290894` passed,
including positive-slope flagging without automatic rejection. There are
53 distinct passing tests in total. No new Monte Carlo simulation was run.
Collection `290896` produced 38 masses, 10 tuning points and three continuum
targets, then failed a metadata-parser check: the scalar YAML regex consumed
a section newline. The parser was restricted to horizontal indentation.
Collection `290897` passed the scientific array/input checks but stopped
because qbig has no `pdfinfo`. Statistical verification stays on qbig;
single-page 16:9 PDF metadata and rendered layout are checked locally with
Poppler, without transferring raw histories or performing local resampling.
Collection `290908` completed with exit zero in 1m25s. All 38 ensemble products,
156 chains and 6,224 mass windows passed scientific verification; all 1,017
plot PDFs passed local single-page 16:9 geometry checks. The 14-page logbook
passed its source/result freshness check and was visually reviewed page by
page. Superseded local data/analysis stage entries were moved to recoverable
storage only after archive-checksum and new-product verification. Raw histories
remain unchanged on qbig. The three continuum trajectories are retained as
preliminary: omitting Nt=2 shifts their intercepts well beyond bootstrap errors.
The same superseded compact analysis entries on qbig were moved into the
archive's `retired_remote/` directory; original raw input/output paths remain
unchanged. Post-cleanup inventory/report job `290924` passed in 25 s.
Final layout build `290931` passed in 24 s on lnode11 (one CPU, 4 GB,
10-minute limit, P100 nodes excluded, no dependencies). Its LaTeX log has no
overflow warnings and its freshness check passes. Both canonical trees were
checksum-compared against qbig after synchronization; every file matches.
