# Dirichlet Yang--Mills/PCM study

This workspace contains the project-specific analysis and production inputs
for temporal Dirichlet boundaries in KLFT. The physics reference is
[`docs/marconebbia_tesi.pdf`](docs/marconebbia_tesi.pdf).

Simulation inputs and campaign metadata are in `data/`, Python programs are in
`scripts/`, and compact analyzed data and plots are in `analysis/`. Raw Markov
chains remain on qbig and are never synchronized to this workspace.

Current campaign status is documented separately. The
[fixed `beta/Nt`](docs/beta_nt_fixed/summary.md) analysis has `p=1` continuum
fits on the 1.8 and 2.0 lines, 64 comparison PDFs, and 30 per-ensemble
first-stage PDFs; the 2.1 line remains incomplete. The separate
[fixed `m_G Nt`](docs/mg_nt_fixed/summary.md) campaign has been reanalysed:
49 physical ensembles (200 chains) in a `c/Nt/Ns/beta` tree, 500-replica errors,
10 tuning points and three preliminary `p=1` continuum comparisons. Three
Nt=1 matched-beta and eight matched-beta/Nt ensembles are kept separate from fixed-c inputs. The
continuum intercepts remain sensitive to the included resolutions. See its
[PDF logbook](docs/mg_nt_fixed/logbook.pdf) and
[analysis index](analysis/mg_nt_fixed/README.md). All simulations and mass
scans in this register are complete, including the four new Nt=1, Ns=128
ensembles. The top-level tables include every analyzed ensemble.
The [nine scaling comparisons](analysis/mg_nt_fixed/comparison/README.md)
show amG, ξ₂nd/a and (ξ₂nd/a)² with p=0,1,3/2, all versus beta/Nt,
using open markers and Nt-only legends. See the campaign report for provenance.

This implementation is restricted to the pure Wilson action in a 3D SU(2)
build. Set `temporal_dirichlet: true` in either update section. With it false,
the original periodic initialization, checkerboard, action, updates, and
observables are unchanged.

## Geometry and action convention

KLFT's Dirichlet fields use an open temporal geometry. For physical slab
thickness `Nt`, the temporal site extent is

```text
temporal_site_extent = L2 = Nt + 1
lower_boundary_slice = 0
upper_boundary_slice = Nt
```

`Nt` counts **physical temporal links**. There are `Nt+1` time slices,
including both fixed surfaces, and `Nt-1` interior slices. For example,
`Nt=4` has slices `0,1,2,3,4` and four links joining them.

Spatial links `U_0` and `U_1` (KLFT directions 0 and 1) at slices 0 and `Nt`
are exactly the identity. Physical temporal links `U_2(x,t)`, `0 <= t < Nt`,
remain dynamical. Spatial links occupy `Nt+1` slices and temporal links occupy
`Nt` layers. There is no closing temporal link at `t=Nt`. Immutable campaign
inputs retain their meaning: `L2=Nt+1`.

The historical implementation evolved the unused wrapping links. With both
tangential endpoint layers fixed, their Wilson plaquettes form an independent
two-dimensional PCM. Integrating out that factor leaves the same physical
slab distribution. Removing those links therefore preserves the slab action,
holonomy correlators, profiles and boundary loops; fixed-seed trajectories can
change because the random-number consumption changes. The deterministic
checks compare arbitrary legacy wrapping links with genuinely open storage,
including every physical-link staple. The physical boundary construction is
described in Sec. 4.1 of Marco Nebbia, *Effects of Dirichlet Boundary Conditions
in Yang-Mills Theories* (MSc thesis, University of Turin, 2024/2025).

Odd temporal site extents are safe because time is open and temporal links at
adjacent `t` do not share a plaquette.
Odd periodic spatial extents use three independent colors; even lattices keep
the original two-color update order. This permits the thesis quality geometry
with a spatial extent of 9.

## Initialization, updates, and configuration files

Cold, hot, near-identity random, partitioned, and driver restart paths enforce
or validate the fixed links. Metropolis, partitioned Metropolis, SU(2)
heatbath, overrelaxation, reunitarization, and RK3 gradient flow skip or restore
them exactly. All physical temporal links remain dynamical. Periodic
Wilson/Polyakov/link measurements and the driver's periodic-clover flow
measurements are rejected in Dirichlet mode. The boundary-preserving RK3
flow kernel is validated independently.

`configuration_output` writes a versioned binary float64 configuration after a
successful run. Use `start: "restart"` with `configuration_input` to read it.
The reader validates dimensions, group, finiteness, unitarity, and exact
fixed links before changing the target field. New open-slab configurations
carry boundary flag 3 and store physical links only. Older rectangular files
require an explicit conversion and are refused by the open-slab driver. Periodic
checkpoints retain their existing format. The file is native-endian and
intended for restart with the same architecture.

## Observables and normalization

The physical holonomy is temporally ordered as

```text
G(x) = U_2(x,0) U_2(x,1) ... U_2(x,Nt-1).
```

It excludes `U_2(x,Nt)`. The primary correlator is

```text
C_G(R) = (1/2) Re Tr[G(x) G(x+R)^dagger],
```

averaged over origins and both spatial directions and stored only for
`0 <= R <= floor(min(L0,L1)/2)`. Under `G -> L G R^dagger`, the product becomes
`L G(x)G(y)^dagger L^dagger`; cyclicity of the trace proves invariance.
`C_G(0)` is assigned the exact value one after its configuration-level SU(2)
identity has been checked by the tests.

`holonomy_correlator.out` contains `C_G`, the raw legacy product
`Tr G(x) Tr G(y)^dagger`, and the configuration-level logarithmic slope. The
latter is diagnostic: a 2D asymptotic correlator is proportional to `K0(mR)`,
not exactly to an exponential. `holonomy_summary.out` contains raw `Tr G` and
the zero/minimum-momentum structure factors. Analysis output includes raw
`mean_TrG` and thesis-normalized `mean_half_TrG=(1/2)mean_TrG`. The
per-configuration `xi` is also diagnostic; the analysis program forms the
second-moment length from ensemble means with a block jackknife.

The legacy `Tr G` is not invariant under independent left and right rotations.
After correct global-orientation sampling, SU(2) Haar averaging gives

```text
<Tr G(x) Tr G(y)^dagger> = (1/2)<Tr[G(x)G(y)^dagger]> = C_G(R)
```

with the output conventions above. The statistical test checks this relation.

Plaquette profiles contain normalized `(1/2) Re Tr U_p`. Kind 0 is the spatial
plaquette on every stored slice, including the explicitly inspectable fixed
boundary values. Kind 1 averages the two physical temporal planes at
half-integer coordinates and excludes the wrapping sector.

Current aggregate plaquettes describe the physical slab: spatial averages use
`Nt+1` layers including the fixed surfaces, temporal averages use `Nt` layers,
and their total is weighted by the corresponding plaquette counts.
Historical outputs from the old executable included the auxiliary sector in
these aggregate observables; their values must retain that recorded convention.
The dedicated slice profiles and all existing mass-analysis inputs are unchanged.

Boundary loops contain normalized `(1/2) Re Tr W_b(R,H)`. The lower horizontal
side is at `t=0`; either all `1 <= H <= Nt` or only `H=1,Nt` are written. The
tests verify `W_b(R,Nt)=C_G(R)` and, at `Nt=1`, the equality holds together with
the PCM spin correlator configuration by configuration.

For the optional rotated quality check, `bulk_horizontal_polyakov.out` stores
the raw double-trace correlator of loops wrapping direction 0 at two slices
symmetric about the slab midpoint, following thesis Eqs. (4.4)-(4.5).

## Exact `Nt=1` reduction

For one physical temporal link, `G(x)=U_2(x,0)`. Fixed tangential links reduce
each temporal-spatial plaquette to `G(x)G(x+e_i)^dagger`, so

```text
S_YM,slab - S_fixed_spatial
  = -(beta/2) sum_{x,i} Re Tr[G(x)G(x+e_i)^dagger]
  = S_PCM.
```

The standalone slab, periodic, and PCM action evaluators are test utilities.
Random configurations and several sizes are compared to machine precision;
no Monte Carlo average enters this test. `Nt=1` is the physical one-link slab
and remains available.

## Build and tests

```bash
cmake -S klft -B build/klft -DKLFT_NDIM=3 -DKLFT_NC=2 \
  -DKokkos_ENABLE_OPENMP=ON -DKLFT_ENABLE_STATISTICAL_TESTS=ON
cmake --build build/klft -j
OMP_NUM_THREADS=2 ctest --test-dir build/klft --output-on-failure
uv run --frozen python scripts/test_dirichlet_analysis.py
uv run --frozen python scripts/test_thermalization_diagnostics.py
uv run --frozen python scripts/test_compare_thesis.py
```

The fast suite covers boundary initialization and updates, restart I/O, local
staple/action differences, reflection, chiral symmetry, action factorization,
the exact `Nt=1` PCM action and observables, odd and minimal extents, and
periodic regression. The opt-in statistical test estimates autocorrelation
times and adaptively checks hot/cold convergence, reflection profiles, bulk
recovery, global-orientation averaging, and full-distance correlator symmetry.

For host address/undefined-behavior checks, use a separate serial build:

```bash
cmake -S klft -B build/klft-asan -DKLFT_NDIM=3 -DKLFT_NC=2 \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build/klft-asan -j
ctest --test-dir build/klft-asan -L fast --output-on-failure
```

## Thesis reproduction

Generate four independent hot-start chains for each `Nt=2,...,12` at
`beta=12.15266` and `64x64`:

```bash
uv run python data/dirichlet_thesis/generate_inputs.py \
  --output data/dirichlet_thesis/runs
uv run python data/dirichlet_thesis/run_inputs.py \
  build/klft/binaries/heatbath data/dirichlet_thesis/runs
```

The generator defaults to 8000 measured sweeps, five overrelaxation sweeps,
all boundary-loop heights, and separate directories/seeds. Production
thermalization must be assessed from the saved histories; 500 is only the
thesis starting cut, not a universal guarantee.

The qbig campaign uses the nested naming convention and only the required
boundary-loop heights in its first pass:

```bash
uv run --frozen python data/dirichlet_thesis/generate_inputs.py \
  --output data/dirichlet_thesis/qbig/thesis_initial \
  --nt 2 6 12 --ns 64 --sweeps 8000 \
  --seeds 32091 90517 177013 515029 \
  --only-minimum-heights --nested-ensemble-layout
```

Raw histories live below `/qbigwork2/negro/dirichlet/data/`; the locked qbig
analysis environment is below `/qbigwork2/negro/dirichlet/project/`. Diagnose
thermalization before selecting a common cut, then run the full correlated
analysis:

```bash
ssh negro@qbig /qbigwork2/negro/dirichlet/project/data/dirichlet_thesis/qbig/analyze_ensemble.sh diagnose 2
ssh negro@qbig /qbigwork2/negro/dirichlet/project/data/dirichlet_thesis/qbig/analyze_ensemble.sh analyse 2 CUT
data/dirichlet_thesis/qbig/sync_compact.sh --dry-run
data/dirichlet_thesis/qbig/sync_compact.sh
```

MSER-5 only proposes a cut. The post-cut split-history, between-chain,
autocorrelation, and global-orientation checks decide whether the cut and
history length are adequate. The sync script transfers only compact analysis
products into `analysis/dirichlet_thesis/`, excludes duplicate analysis logs,
and never uses `--delete`. Run both the numerical analysis and plotting on
qbig; synchronize only the resulting JSON, CSV, covariance matrices, and PDFs.

Independent raw histories must never be appended before statistical analysis.
Choose the thermalization cut and estimate `tau_int` for each run separately,
block or resample within each run, and only then combine independent run-level
fluctuations. For a correlator, use a hierarchical bootstrap so that each
within-run resample carries the complete vector of independent `R` values and
therefore preserves their covariance. This avoids artificial chain-boundary
transitions and unstable plateaus based on too few pooled blocks.

The program selects a block length greater than twice the largest estimated
autocorrelation time, reports split-history and between-chain diagnostics,
keeps the full independent-distance covariance, jackknifes derived slopes and
`xi_2nd` and fitted parameters, and scans correlated and uncorrelated periodic
`A[K0(mR)+K0(m(Ns-R))]` fits with `chi2`, degrees of freedom, and p-values.
The fit jackknife keeps the full-sample covariance weights fixed and reports
the local-curvature errors separately.
It refuses to combine chains whose default consistency diagnostics exceed five
standard deviations, refuses redundant `R>Ns/2` input, and refuses a singular
correlated fit. Adjust `--max-diagnostic-z` only with a stated statistical
reason.

Compare `(1/2)Tr G` against thesis Table 4.4 with combined uncertainties:

```bash
uv run python scripts/compare_thesis.py \
  2=nt2_analysis.json 3=nt3_analysis.json 4=nt4_analysis.json
```

Generate the optional 16000-sweep rotated `Nt=160`, `9x160` benchmark with
`--quality-benchmark`. Analyze its `bulk_horizontal_polyakov.out` through the
`--bulk-polyakov` option. This benchmark is intentionally not a default test.

### Exact-PCM beta pilot

The first coupling calibration uses only KLFT at physical `Nt=1`, where its
holonomy is exactly the SU(2) PCM spin. Generate the `Ns=96`, beta
`1.9, 2.0, 2.1` pilot with:

```bash
uv run --frozen python data/dirichlet_thesis/generate_inputs.py \
  --output data/dirichlet_thesis/qbig/pcm_beta_pilot \
  --nt 1 --beta 1.9 2.0 2.1 --ns 96 --sweeps 10000 \
  --seeds 32091 90517 177013 515029 \
  --holonomy-only --nested-ensemble-layout
```

On qbig, first submit `submit_pcm_beta_analysis.slurm` with `MODE=diagnose`.
Inspect every chain's proposed cut, post-cut `tau_int`, split-history check, and
between-chain consistency. Then resubmit it with `MODE=analyse`. The numerical
stage enforces at least 30 complete blocks in every chain and uses a seeded
hierarchical bootstrap of full correlator vectors. A coupling is conservatively
accepted only if the 68% interval satisfies `xi_2nd >= 5` and
`Ns/xi_2nd >= 8`.

The completed pilot, selected beta, diagnostics, and compact products are
documented in [`analysis/pcm_beta_pilot/`](analysis/pcm_beta_pilot/README.md).

### Fixed-beta/Nt campaign

At fixed `beta/Nt`, the convention `beta=4/(a g_3^2)` gives
`g_3^2 L=4 Nt/beta`; hence the continuum limit increases `Nt` and beta at a
fixed ratio. The cleaned analysis retains exactly 1,000 bootstrap replicas per
operation and only `p=1` continuum fits. Exact status, failures, results, job
IDs, and provenance are maintained in
[`docs/beta_nt_fixed/summary.md`](docs/beta_nt_fixed/summary.md).

The completed initial Nt=2, 6, 12 campaign, quantitative thesis comparisons,
and cluster-generated plots are documented in
[`analysis/dirichlet_thesis/`](analysis/dirichlet_thesis/README.md).
