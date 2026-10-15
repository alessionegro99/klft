# Repository Guidelines

## Project Structure & Module Organization

This workspace studies Dirichlet boundary conditions in lattice Yang--Mills theory. Put simulation inputs and campaign metadata in `data/`, analyzed data and plots in `analysis/`, Python analysis programs and their tests in `scripts/`, and supporting documents in `docs/`. `docs/marconebbia_tesi.pdf` is the current physics reference. `klft/` and `statanalysis-Bonn/` are symbolic links to external simulation and analysis repositories; keep them as links and submit changes to those projects separately rather than copying their sources here. Keep raw Monte Carlo histories on qbig; do not download them locally.

## Build, Test, and Development Commands

There is no root build system yet. Build the linked KLFT dependency for the project's three-dimensional SU(2) setup with:

```bash
cmake -S klft -B build/klft -DKLFT_NDIM=3 -DKLFT_NC=2 \
  -DKokkos_ENABLE_OPENMP=ON
cmake --build build/klft -j
ctest --test-dir build/klft --output-on-failure
```

Run KLFT's Python partitioning check with `cd klft && uv run python tools/test_generate_partitioning.py`. Add a root command here when a project-level workflow is introduced; do not document uncommitted local scripts.

## Coding Style & Naming Conventions

Use four spaces in Python, `snake_case` for files and functions, and `PascalCase` for classes. Add type hints to public signatures and NumPy-style docstrings. Use `numpy.float64` by default and `np.random.default_rng(seed)` for stochastic work. Keep reusable functions separate from plotting or exploratory scripts. Name production ensembles `data/Nt_<thickness>_Ns_<size>_b_<beta>/seed_<seed>/`, replacing the decimal point in beta by `p`, for example `data/Nt_2_Ns_64_b_12p15266/seed_12001/`. Here `Nt` is the physical number of temporal links between the Dirichlet surfaces; the KLFT input specifies temporal site extent `L2 = Nt + 1`.

Current KLFT Dirichlet fields use open storage: `Nt+1` spatial-link slices and
`Nt` temporal-link layers. No closing temporal layer is allocated or evolved.
Preserve historical executable/input provenance: older aggregate plaquettes
included the independent wrapping sector. Open-slab restart files use boundary
flag 3 and require explicit conversion of older rectangular checkpoints.
`Nt` continues to count links, with `L2=Nt+1` slices. The periodic-clover flow
measurements are unavailable in the Dirichlet drivers; the boundary-preserving
flow kernel remains tested separately.

## Plotting and Fitting

Save every figure as PDF. A PDF page may contain at most one plot: do not use
subplots, panels, insets, or plot montages. Split related plots into separate
single-page PDF files. Create every figure with `figsize=(16, 9)` and preserve
that 16:9 canvas when saving.

Use LaTeX rendering (`text.usetex = True`) with the serif style from
`statanalysis-Bonn/plot_utils.py`. Use that module's `color_dict`, markers, and
data/fit style helpers; do not choose ad hoc Matplotlib colors or markers.
Use `statanalysis-Bonn/fit_utils.py` for standard fitting and uncertainty
formatting. A project-specific fit implementation is justified only when the
shared utility does not support the required statistical treatment, such as a
full correlated covariance fit; document that reason where the fit is
implemented.

For correlators whose separations are strongly cross-correlated, use the
following fit procedure by default. Thermalize and block every independent
chain separately, then construct vector-valued bootstrap replicas: every
resampling draw must act on the complete vector of separations, with one RNG
seed recorded for the analysis, so the covariance across `R` is preserved.
Estimate the fit parameters from a diagonal-covariance (uncorrelated) fit to
the central data and repeat that same fit for every bootstrap replica. Quote
parameter uncertainties from the bootstrap distribution. After fitting,
evaluate and report

\[
\chi^2_{\rm full}=r^T\operatorname{Cov}^{-1}r,
\qquad \chi^2_{\rm full}/{\rm dof},
\]

using the complete covariance matrix across the fitted separations. Because
the parameters do not minimize `chi2_full`, its analytic chi-square p-value
with `n_points - n_parameters` degrees of freedom is not valid. If a
goodness-of-fit p-value is required, calibrate this statistic with a
recentered joint bootstrap using the same diagonal fit procedure.

## Testing and Reproducibility

Use exactly 500 samples for every new bootstrap operation (user update,
12 October 2026, superseding the previous 1000-sample policy). Preserve the
recorded counts of historical products and label any historical exceptions.

For `mg_nt_fixed`, update `docs/mg_nt_fixed/logbook.pdf` after every new
analyzed result, including failed gates and revised conclusions. Review its
`logbook.tex` narrative and `review.json`, then run on a qbig compute node
`uv run --frozen python scripts/mg_nt_fixed_logbook.py`; verify freshness with
the same command plus `--check`. Keep plots in `analysis/mg_nt_fixed/`:
single-simulation figures belong in that simulation's `plots/` directory,
thermalization and blocking examples in `plots/first_stage/`. Use the canonical
`c_<c>/Nt_<Nt>/Ns_<Ns>/beta_<beta>/` tree in both data and analysis. Figures
combining simulations belong in `comparison/` at their nearest common physics
level. Keep dates, bootstrap counts and historical stages out of active folder
names. Keep blocking-example figures outside the logbook PDF.
All plots are grid-free. Keep the fixed-beta/Nt campaign separate.

No root test framework or coverage threshold is configured. Place future Python tests beside the analysis code as `scripts/test_<module>.py`. Check small-volume or analytically known limits, and test KLFT changes with CTest. Every reported result must record the boundary conditions, lattice dimensions, coupling convention, seed, thermalization, measurement spacing, and dependency revision. Estimate Monte Carlo uncertainty with autocorrelation-aware binning, jackknife, or bootstrap; never use an independent-sample standard error for correlated chains.

## qbig Production Workflow

Use qbig for production GPU simulations and analysis of raw histories. The remote project root is `/qbigwork2/negro/dirichlet`, with raw outputs below `/qbigwork2/negro/dirichlet/data/`. Never run production work on the login node.

Target the Slurm-managed P100 nodes `lnode13` through `lnode17` for this project. Before submission, inspect `sinfo`, `squeue`, and GPU occupancy; do not assume a node or GPU is idle from its nominal assignment. Request one typed Pascal GPU per task with `--gres=gpu:pascal:1`; do not use a comma-separated `--nodelist` as an allowed-node set because this Slurm interprets it as a multi-node request. Use a P100-specific CUDA build. Build a fresh three-dimensional SU(2) executable from the intended clean KLFT revision, and record the KLFT commit, CMake options, compiler, CUDA version, executable checksum, and build path with the campaign.

For every campaign:

1. Generate immutable input files and a checked-in submission script locally. Use independent explicit seeds and separate seed directories. KLFT output files are append-only, so refuse to start when an intended output already exists.
2. Run a small deterministic or short smoke test first. Check the recorded dimensions, Dirichlet flag, boundary convention, first observable rows, and output paths before scaling out.
3. Record planned and active jobs in `running_simulations.md`: physics parameters, seeds, sweeps, measurement interval, job IDs, requested node/GPU resources, remote paths, KLFT revision, and log paths. Update failed, cancelled, and completed jobs explicitly.
4. Inspect the first production output and job log after submission. A successful `sbatch` call alone is not validation.
5. Analyze and plot raw histories on qbig. Use the locked environment below `/qbigwork2/negro/dirichlet/project/` through `/qbigwork2/negro/dirichlet/bin/uv`; qbig's system Python 3.5 stack is too old for this project. Keep independent chains separate through thermalization and consistency checks; estimate integrated autocorrelation times, choose blocks longer than the autocorrelation scale, and preserve cross-distance covariance in derived correlators and fits.
6. Keep raw data on qbig and synchronize only compact analyzed data, metadata, tables, and plots into local `analysis/`. Dry-run synchronization first, use partial transfers, and never use `--delete` for scientific data.
7. Summarize completed numerical checks and provenance in `summary.md`. Do not report a production result until the corresponding job and analysis records are reproducible from those two tracking files.

For the fixed-beta/Nt slab campaign, holonomy correlators, Fourier components,
and holonomy summaries use `measurement_interval: 1`; plaquette profiles use
`dirichlet_suite_measurement_interval: 5`; accepted production boundary
Wilson loops use `boundary_wilson_measurement_interval: 25`. Keep both
secondary intervals
integer multiples of the base interval and report all three cadences. Measure
all boundary-loop heights for the continuum production. Horizontal Polyakov
loops and effective-action reconstruction are disabled for this campaign.

Follow the established gaugebreaking pattern: simulate remotely, analyze raw data where it resides, retain raw histories, and version compact reproducible products. Thermalization selection and uncertainty estimates must be quantitative; history plots are diagnostics, not substitutes for autocorrelation-aware analysis.

### Independent-chain resampling

Never concatenate independent Markov histories before thermalization, blocking,
jackknife, or bootstrap. Such concatenation creates artificial transitions at
run boundaries, permits a block to mix unrelated chains, and hides differences
in equilibration. A visually stable error plateau is not sufficient when only a
few large blocks remain.

Determine the thermalization cut and estimate `tau_int` for each chain
separately. Form blocks and perform within-chain resampling separately, then
combine run-level estimates with equal chain weights. Keep chain identities
fixed and independently resample complete block vectors within every chain;
this preserves joint covariance across separations. Do not add an outer
resampling of chain identities: these Markov replicas estimate the same
equilibrium expectation (correction approved 12 October 2026). Assess chain
consistency separately. Report every chain's cut, `tau_int`, block length, and retained block
count. All quoted uncertainties must follow this chain-aware procedure.

For `mg_nt_fixed`, the approved pooled-with-boundaries policy supersedes equal
chain weights: use fixed weights proportional to the measurements retained in
complete blocks. Preserve independent cuts, autocorrelations and resampling
within each chain. Flag the four 80,000-sweep chains as `length_outlier`; this
is a duration flag. Retain numerically defined mass and interpolation fits as
preliminary, with diagnostic flags and manual review in `review.csv`. Use a
mass-stability window recommendation; full-covariance chi-square is descriptive
and no p-value gate is applied. Save all 500 window refits. Continuum fits use
only p=1 and may include flagged preliminary inputs. Exclude undefined results
and explicitly user-excluded points. Keep the fixed-beta/Nt policy unchanged.
The canonical entry points are `scripts/mg_nt_fixed_analysis.py` (one physical
ensemble), `scripts/mg_nt_fixed_campaign.py` (tables, tuning and p=1 continuum),
`scripts/mg_nt_fixed_verify.py`, and `scripts/mg_nt_fixed_logbook.py`.
`mg_nt_fixed_parallel.py` schedules the same window fits across the CPUs
allocated by Slurm; it uses the same saved ensemble replicas. Older stage
generators and their inputs are preserved in the qbig recovery archive.

The Nt=1 matched-beta diagnostic uses `nt1_analysis_manifest.json` alongside
the unchanged fixed-c manifest. Its c folders identify the source Nt=2
target; keep `target_c: null` and `fixed_c_continuum_input: false` through
all exports. For its Ns=512/2048 boxes, the user approved coarse endpoint
scanning followed by dense adjacent plateau checks (`mg_nt_fixed_adaptive.py`).
Keep every separation inside each window and all 500 refits; flag that the
recommendation is earliest among tested patches. Covariance estimated from
500 centered replicas has rank at most 499: report an undefined full-covariance
chi-square when singular, retaining defined diagonal-fit masses for review.

For the fixed-beta/Nt campaign, also scan the blocking error over the whole
post-cut history for every primary and secondary observable. Block each
replica separately at every candidate size and fuse only replica-level blocked
estimates; no block may
cross a replica boundary. Require and record a quantitative error plateau in
addition to the `tau_int` lower bound before accepting the final block length.

## Commit & Pull Request Guidelines

Use short imperative subjects, such as `Add Dirichlet plaquette profile`. Pull requests should describe the physics and data impact, list exact validation commands, identify input-data provenance, and include plots when numerical outputs change. Keep generated data, dependency edits, and analysis-code changes in separate commits. Commit and push KLFT changes in the KLFT repository; commit project inputs, analysis, metadata, and compact results here.
