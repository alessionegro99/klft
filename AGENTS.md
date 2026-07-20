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

Use four spaces in Python, `snake_case` for files and functions, and `PascalCase` for classes. Add type hints to public signatures and NumPy-style docstrings. Use `numpy.float64` by default and `np.random.default_rng(seed)` for stochastic work. Keep reusable functions separate from plotting or exploratory scripts. Name production ensembles `data/Nt_<thickness>_Ns_<size>_b_<beta>/seed_<seed>/`, replacing the decimal point in beta by `p`, for example `data/Nt_2_Ns_64_b_12p15266/seed_12001/`. Here `Nt` is the physical number of temporal links between the Dirichlet surfaces; the current periodic-wrapper KLFT input therefore has temporal site extent `Nt + 1`.

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

## Testing and Reproducibility

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

Follow the established gaugebreaking pattern: simulate remotely, analyze raw data where it resides, retain raw histories, and version compact reproducible products. Thermalization selection and uncertainty estimates must be quantitative; history plots are diagnostics, not substitutes for autocorrelation-aware analysis.

### Independent-chain resampling

Never concatenate independent Markov histories before thermalization, blocking,
jackknife, or bootstrap. Such concatenation creates artificial transitions at
run boundaries, permits a block to mix unrelated chains, and hides differences
in equilibration. A visually stable error plateau is not sufficient when only a
few large blocks remain.

Determine the thermalization cut and estimate `tau_int` for each chain
separately. Form blocks and perform within-chain resampling separately, then
combine run-level estimates through their independent fluctuations. Use a
hierarchical bootstrap when a joint covariance across correlator separations is
required. Report every chain's cut, `tau_int`, block length, and retained block
count. All quoted uncertainties must follow this chain-aware procedure.

## Commit & Pull Request Guidelines

Use short imperative subjects, such as `Add Dirichlet plaquette profile`. Pull requests should describe the physics and data impact, list exact validation commands, identify input-data provenance, and include plots when numerical outputs change. Keep generated data, dependency edits, and analysis-code changes in separate commits. Commit and push KLFT changes in the KLFT repository; commit project inputs, analysis, metadata, and compact results here.
