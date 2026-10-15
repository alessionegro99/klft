# Fixed mG Nt campaign

The fixed-c reanalysis retains **38 physical ensembles, 156 independent
chains, 10 preliminary tuning points and three p=1 continuum comparisons**.
Every bootstrap operation uses exactly 500 replicas. All raw-history analysis
and plotting runs on qbig compute nodes. No new simulation was submitted.

Twelve subsequently authorized Nt=1 A100 chains have completed, adding three
matched-beta diagnostic ensembles and 96,000 sweeps. Their canonical leaves
use the source Nt=2 reference c; `target_c` is null and
`fixed_c_continuum_input=false`. They do not extend the fixed-c continuum fit.
The combined inventory contains 41 ensembles and 168 chains, with 1,648,000
measured sweeps. Inputs and actual execution records are in the same physics
tree; the original immutable fixed-c manifest is preserved, with a separate
immutable `nt1_analysis_manifest.json` for these untuned diagnostics.

## New Nt=1 results and the Ns=2048 choice

All twelve chains passed complete-output, cadence, boundary and checkpoint
checks, finishing on lnode19 by 18:58:02 CEST on 14 October. Total heatbath
time was 3.0381 A100 GPU-hours. The independent P100 validation 290996 also
completed successfully on 15 October; no new P100 production was launched.

The sizes 128, 512 and 2048 were explicitly supplied candidate inputs, marked
`volume_status=unvalidated`; no recorded quantitative derivation establishes
their necessity. In particular, c=0.5 in the Nt=1 path is the source Nt=2
target, not an imposed Nt=1 mass. At beta=2.989543 the preliminary measured
second-moment length is 146.74 ± 33.62, giving Ns/xi2nd about 14. The box is
not obviously excessive at this coupling. A second volume is required for
volume control.

| Ns | beta | xi2nd/a (bootstrap SE) | selected block (sweeps) | complete blocks per chain |
|---:|---:|---:|---:|---:|
| 128 | 1.962386866460073 | 7.3484 ± 0.2269 | 200 | 37 |
| 512 | 2.5044558261064154 | 40.590 ± 2.941 | 200 | 36–37 |
| 2048 | 2.989543168360997 | 146.744 ± 33.617 | 635 | 6–11 |

Ns=2048 is statistics-limited: cuts are 825–3625 sweeps, the largest estimated
tau is 316.86 sweeps, the common blocking plateau is unresolved, and split/
between-chain diagnostics require review. At B=635, fifty complete blocks
require 31,750 post-cut sweeps per chain, or 35,375 total using the largest
observed cut. This planning estimate must be rechecked with longer histories.

First-stage array 290998 completed all three ensembles. Mass-fit array 291001
uses the user-approved coarse endpoint scan plus dense adjacent checks for
Ns=512/2048, retaining every integer separation and all 500 refits inside
each window. Ns=128 retains the exhaustive scan. The adaptive recommendation
is earliest among tested patches. A full covariance from 500 replicas has
rank at most 499; wider-window chi-square is explicitly undefined, while
numerically defined diagonal-fit masses remain available with flags.

| Ns | a mG (500-replica SE) | selected R window | tested windows |
|---:|---:|---:|---:|
| 128 | 0.1411131 ± 0.0002194 | 4–64 | 550 |
| 512 | 0.02879380 ± 0.00006304 | 7–256 | 161 |
| 2048 | 0.00670431 ± 0.00003775 | 18–1024 | 188 |

These are conditional preliminary mass estimates. In the largest box,
xi_exp/a is approximately 149 and mG Ns approximately 13.7; the unresolved
blocking and equilibration diagnostics remain separate limitations.
Collection job 291004 completed in 1m23s. Numerical verification covers
41 ensembles, 168 unique chains, 1,648,000 sweeps, 7,123 mass windows and
1,105 plot PDFs. The original fixed-c tuning and p=1 continuum values are
retained; new Nt=1 points are shown in dedicated comparison plots.

[Human-readable logbook](logbook.pdf) ·
[Analysis tree](../../analysis/mg_nt_fixed/README.md) ·
[CSV column guide](csv_columns.md) ·
[Immutable inputs](../../data/mg_nt_fixed/README.md).

## Physics and interpretation

The trajectory fixes c = (a mG) Nt = mG L, where L = a Nt is the slab
thickness. It uses the screening mass of the boundary-to-boundary holonomy
correlator C_G(R) = ⟨Re Tr[G(x)G†(x+R)]/2⟩ as the physical ruler. The targets
c = 0.5, 1, 2 correspond to ξexp/L = 2, 1, 1/2. Transverse volume is controlled
separately through (a mG) Ns. The stored temporal extent is Nt+1.

For 3D SU(2) Wilson β = 4/(a g₃,₀²), the nontrivial extrapolated quantity is
Q = mG/g₃,₀² = c βstar/(4 Nt). The only continuum ansatz is Q = Q₀ + d/Nt.
Statistical intervals are conditional on the selected mass windows, coupling
windows and volumes. The logbook gives the action, boundary conditions,
correlator definition, full derivation and references.

Resolution-range sensitivity is substantial. Omitting Nt=2 shifts Q₀ by much
more than the bootstrap error. These are preliminary extrapolations, with
all requested points retained for visual review; a precision continuum mass
requires finer-resolution checks.

| c | Nt retained | Q₀, all Nt (statistical SE) | Q₀, omit Nt=2 (statistical SE) |
|---:|---|---:|---:|
| 0.5 | 2, 3, 4 | 0.163311 ± 0.000124 | 0.159713 ± 0.000331 |
| 1 | 2, 3, 4, 6 | 0.262250 ± 0.000158 | 0.255715 ± 0.000261 |
| 2 | 2, 3, 4 | 0.457335 ± 0.000768 | 0.415095 ± 0.001526 |

The two-point drop-coarsest fits at c=0.5 and 2 have zero residual degrees
of freedom. CSVs also report a separate conservative interpolation-shift
bound. This is not combined with statistical errors.

## Statistics and recoverable simulations

All 38 physical ensembles are retained, including previously gated fits,
unused bracket points and smaller-volume comparisons. Nine short validation
histories are excluded and remain on qbig. Repeated historical analyses of
the same histories are not counted as independent simulations.

| Chain duration | Chains | Measured sweeps | Label |
|---:|---:|---:|---|
| 8,000 | 144 | 1,152,000 | standard |
| 10,000 | 8 | 80,000 | explicitly labelled prior inputs |
| 80,000 | 4 | 320,000 | `length_outlier` |

Total measured sweeps: 1,552,000. Independently thermalized: 1,473,875.
Used in complete within-chain blocks: 1,467,160. These counts are measured
sweeps, not independent-sample counts.

At c=0.5, Nt=3, Ns=80, β=4.4, all four short and four long chains are pooled
with weights proportional to used complete-block measurements. The long-chain
flag describes duration. Short-only, long-only and jointly resampled pooled
comparisons are saved in the same ensemble leaf.

## Tuning

| c | Nt | βstar median | Statistical 16–84% interval | Interpolation shift |
|---:|---:|---:|---|---:|
| 0.5 | 2 | 2.989543 | [2.988803, 2.990298] | 0.002756 |
| 0.5 | 3 | 4.306930 | [4.305876, 4.308139] | 0.000383 |
| 0.5 | 4 | 5.584636 | [5.582117, 5.586848] | 0.000685 |
| 1 | 2 | 2.504456 | [2.503821, 2.505144] | 0.000365 |
| 1 | 3 | 3.598496 | [3.596793, 3.600156] | 0.002012 |
| 1 | 4 | 4.630477 | [4.627923, 4.632584] | 0.003520 |
| 1 | 6 | 6.666379 | [6.663606, 6.669126] | 0.010801 |
| 2 | 2 | 1.962387 | [1.960268, 1.963997] | 0.002686 |
| 2 | 3 | 2.942831 | [2.941005, 2.944249] | 0.001875 |
| 2 | 4 | 3.773020 | [3.770404, 3.775364] | 0.003486 |

These are flagged preliminary results. The interpolation shift is the largest
central shift between the selected three-point log-linear fit and a bracketing
two-point interpolation. The Nt=4,c=1 confirmation ensemble contributes to
the new interpolation; its historical independent-confirmation status applies
to the earlier fit only. Distinct volumes are never pooled.

The c=0.5,Nt=4,β=5.58394149 matched Ns=96/128 masses differ by about 0.49%
(1.37 combined standard errors). Both boxes and their fit-window diagnostics
remain available. This supports the pilot volume comparison, conditional on
the fitted asymptotic model.

## Statistical procedure and review

1. Verify input hashes, dimensions, cadence and complete output counts.
2. Cut each chain independently with MSER-5 and a 500-sweep minimum; estimate
   Sokal-window autocorrelation times for every separation and monitor.
3. Scan whole-history blocking errors using exactly 500 vector-valued replicas
   per block size. Retain chain boundaries and fixed used-measurement weights.
   Show error saturation, the chosen B and remaining block counts.
4. Fit all contiguous periodic-K0 windows with at least five points, scanning
   every positive Rmin and Rmax from Ns/2 down to at least ceil(Ns/3), step two.
   Use the diagonal objective for the central data and all 500 replicas;
   report full-covariance χ² descriptively, without a p-value or χ² veto.
5. Suggest the earliest stable 3×3 mass patch. Keep defined flagged windows,
   store all 500 refits and failed slots, and permit explicit manual choices.
6. Fit every contiguous three-coupling window; use bracketing numerical fits
   for tuning. Propagate aligned replicas through tuning and p=1 continuum fits.
   Show all-resolution and drop-coarsest sensitivity.

Nt=3,Ns=32,β=3.25 is retained: a mG = 0.47480651 ± 0.00189627, with a local
plateau and explicit late-window-drift/between-chain flags. High relative mass
error, low refit success, missing blocking plateaus and small block counts are
flags. Undefined numerical fits and unbracketed extrapolations remain unavailable.
Singular full covariance does not erase a defined diagonal fit; its χ² is null.

The tables `ensembles.csv`, `chains.csv`, `autocorrelations.csv`,
`correlators.csv`, `fit_windows.csv`, `tuning.csv` and `continuum.csv` are at
the analysis root, with filtered tables lower in the tree. Every simulation
has its own plots and first-stage examples. Comparison plots sit at the
nearest common physics level. All figures are grid-free, single-plot,
single-page 16:9 PDFs. Blocking examples remain outside the logbook.

Edit [review.csv](../../analysis/mg_nt_fixed/review.csv) for manual windows
or inclusion decisions, then rerun `mg_nt_fixed_campaign.py` and
`mg_nt_fixed_logbook.py` on qbig. A manual mass-window override explicitly
flags its blocking diagnostic for review; its saved 500-parameter samples
are used immediately in downstream tuning.

## Provenance and verification

The simulation executable is unchanged: KLFT
`7eed1f7d9bb0948621a2d4020706d19c56406001`, P100 build
`/qbigwork2/negro/dirichlet/build/klft`, executable SHA256
`41fc9502a1f74106894ff077a3d9e2905dd6a75f7bf65c96971150a74132ed57`.
Build options/compiler/CUDA and the lock hash are in
`data/mg_nt_fixed/build_provenance.json`. Raw-file, input and analysis-code
hashes and named RNG seeds are recorded per ensemble.

Source snapshot: `/qbigwork2/negro/dirichlet/reviews/mg_nt_fixed_reorganization`.
Numerical jobs: smoke `290846_34`; array `290847` with 34 completed tasks;
parallel replacements `290891_[13,24,27]`. The three serial replacements were
cancelled after preserving partial diagnostics. Parallel and serial fits were
tested to produce identical 500-replica arrays and window choices.
There are 53 distinct passing tests, including fixed-beta regressions.
Six historical generator/workflow test modules require the archived fixtures.
Collection `290908` completed in 1m25s with exit zero. Scientific verification
checks all 38 ensembles, 156 chains and 6,224 mass windows. Local Poppler QA
checks all 1,017 plot PDFs as single-page 16:9; all 14 logbook pages were
visually inspected. No numerical resampling was performed locally.
The post-cleanup input-identity test passed in `290924`; final report build
`290931` passed with no LaTeX overflow warnings and a successful source/result
freshness check. Canonical local data and analysis files match qbig by checksum.

The pre-change archive includes existing uncommitted work and historical
bootstrap products with their original counts. Its verified SHA256 is
`84b0b7ca8e3c6b07f6ea8dc0dd07976e6cb0a57e90dc2499ee459abc4169aef6`; it is stored
at `/qbigwork2/negro/dirichlet/archive/mg_nt_fixed/<sha256>/workspace.tar.gz`.
Original raw paths remain unchanged; `data/mg_nt_fixed/migration.csv` maps
historical inputs to the canonical tree. Earlier outer-chain bootstrap
products and p-value-gated decisions are historical evidence in that archive.
Superseded local stage folders have been retired from the active data/analysis
trees after verification. Recovery paths are recorded in
`generated/local_retirement.json`; the checksum-verified qbig archive remains
the durable recovery copy. Superseded remote compact analysis products are
under the same archive's `retired_remote/` directory, with exact paths in
`generated/remote_retirement.json`.
Current validation/build details are in `generated/` and `running_simulations.md`.

## Next step

Visually review the flagged mass and coupling windows first. Prioritize finer
resolution coverage: Nt=6 for c=0.5 and c=2, followed by independent tuning
confirmation and volume/timing checks. The strong resolution-range sensitivity
currently limits the continuum interpretation more than the reported
conditional statistical errors.

More independent 8,000-sweep chains improve statistics while keeping the
standard duration. At fixed B, 50 blocks per chain require cut+50B sweeps:
10,500 at B=200 or 12,500 at B=240 for a 500-sweep cut. Additional short chains
do not by themselves increase that per-chain count. Approximately twice the
effective statistics gives errors smaller by √2; a factor-two error reduction
requires approximately four times the effective statistics. Model, window,
volume and continuum systematics require their own checks.
