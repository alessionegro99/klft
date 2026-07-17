# Repository Guidelines

## Project Structure & Module Organization

This workspace studies Dirichlet boundary conditions in lattice Yang--Mills theory. Put project-specific analysis code in `analysis/`, simulation inputs and results in `data/`, and supporting documents in `docs/`. `docs/marconebbia_tesi.pdf` is the current physics reference. `klft/` and `statanalysis-Bonn/` are symbolic links to external simulation and analysis repositories; keep them as links and submit changes to those projects separately rather than copying their sources here.

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

Use four spaces in Python, `snake_case` for files and functions, and `PascalCase` for classes. Add type hints to public signatures and NumPy-style docstrings. Use `numpy.float64` by default and `np.random.default_rng(seed)` for stochastic work. Keep reusable functions separate from plotting or exploratory scripts. Name output directories with the parameters needed to identify a run, for example `data/beta_6p0_L32_T16_seed_1234/`.

## Testing and Reproducibility

No root test framework or coverage threshold is configured. Place future Python tests beside the analysis as `analysis/test_<module>.py`. Check small-volume or analytically known limits, and test KLFT changes with CTest. Every reported result must record the boundary conditions, lattice dimensions, coupling convention, seed, thermalization, measurement spacing, and dependency revision. Estimate Monte Carlo uncertainty with autocorrelation-aware binning, jackknife, or bootstrap; never use an independent-sample standard error for correlated chains.

## Commit & Pull Request Guidelines

This checkout has no readable root Git history, so no repository-specific message convention can be inferred. Use short imperative subjects, such as `Add Dirichlet plaquette profile`. Pull requests should describe the physics and data impact, list exact validation commands, identify input-data provenance, and include plots when numerical outputs change. Keep generated data, dependency edits, and analysis-code changes in separate commits.
