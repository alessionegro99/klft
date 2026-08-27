# KLFT

KLFT is a C++17 lattice-gauge simulation library built on
[Kokkos](https://kokkos.org/) with YAML-configured Metropolis and heatbath
drivers. The spacetime dimension and gauge group are selected at configure
time; lattice parameters, updates, measurements, and output files are selected
at run time.

## Quick start

KLFT requires CMake 3.21 or newer, a C++17 compiler, and Git. Kokkos and
yaml-cpp are submodules.

```bash
git clone --recurse-submodules https://github.com/alessionegro99/klft.git
cd klft
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DKLFT_NDIM=4 -DKLFT_NC=3 -DKokkos_ENABLE_OPENMP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For an existing clone, initialize dependencies with:

```bash
git submodule update --init --recursive
```

Choose the Kokkos backend and architecture for the target machine; see the
[Kokkos CMake options](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).
The checked-in presets target specific Zen 3, P100, or A100 machines, so inspect
`CMakePresets.json` before using one.

## Compile-time configuration

One build directory represents one theory. Use separate build directories for
different dimensions, groups, or backends.

| CMake option | Values | Default |
| --- | --- | --- |
| `KLFT_NDIM` | `2`, `3`, or `4` spacetime dimensions | `4` |
| `KLFT_NC` | `1` = U(1), `2` = SU(2), `3` = SU(3) | `3` |
| `BUILD_TESTING` | register deterministic CTest checks | `ON` |
| `KLFT_ENABLE_STATISTICAL_TESTS` | register the slow 3D SU(2) Dirichlet integration check | `OFF` |

Direction `KLFT_NDIM - 1` is Euclidean time. Periodic boundaries are the
default.

## Run a simulation

The two drivers use the same command line:

```text
-f FILE, --filename FILE   read FILE instead of input.yaml
-h, --help                 show KLFT options
--kokkos-help              show Kokkos options
```

Run a driver without arguments in an empty run directory to generate the
complete input file for that driver and compiled theory. The generator never
overwrites an existing `input.yaml`.

```bash
mkdir run-metropolis
cd run-metropolis
../build/binaries/metropolis
../build/binaries/metropolis -f input.yaml
```

Use `heatbath` in place of `metropolis` for heatbath updates with
`nOverrelax` overrelaxation sweeps. Heatbath currently supports `epsilon1` but
rejects nonzero `epsilon2`; finite SU(2) partitionings are Metropolis-only.

The generated YAML is the authoritative list of run-time keys. In both
drivers, `start` is `cold`, `hot`, or `restart`. A restart requires
`configuration_input`; set `configuration_output` to save the final field.
Restart files record their format and compiled theory and are validated when
loaded.

## Measurements and gradient flow

`GaugeObservableParams` controls incremental, space-separated output for:

- plaquettes averaged over all, spatial-spatial, or spatial-temporal planes;
- temporal and arbitrary-plane Wilson loops, with optional multihit;
- Polyakov loops, correlators, and zero/minimum-momentum susceptibilities;
- normalized `Re Tr(U) / Nc` and `Re Tr(U^2) / Nc`;
- nested Wilson actions and the specialized Dirichlet observables below.

Each split plaquette is normalized by its own number of planes. Metropolis
multihit uses local Metropolis link updates; heatbath multihit uses heatbath
plus the configured overrelaxation updates.

The susceptibility output contains raw-loop `G_0 = |A(0)|^2` and `G_pmin`, the
average of `|A(p_min)|^2` over spatial directions. At analysis time,
`U4 = <G_0^2> / <G_0>^2` and
`xi = sqrt(<G_0> / <G_pmin> - 1) / (2 sin(pi/L))`. Raw loops are used here
because a multihit estimator would bias the diagonal self-term.

`GradientFlowParams` enables third-order Runge--Kutta Wilson flow at requested
`t/a^2` values, with clover energy, Wilson-loop, and optional `t0` output.

## Specialized modes

### Finite SU(2) partitionings

An SU(2) Metropolis build can restrict links to an included linear or Fibonacci
point set:

```yaml
PartitioningParams:
  enabled: true
  table_file: "partitionings/fibonacci_N88.yaml"
```

The update follows the point sets and nearest-neighbor proposals of
[Hartung et al., EPJC 82 (2022) 237](https://arxiv.org/abs/2201.09625) and
includes the [Metropolis--Hastings](https://doi.org/10.1093/biomet/57.1.97)
degree ratio. `nHits` is the number of nearest-neighbor proposals per link;
`delta` is ignored. Partition mode requires `epsilon1: 0`, `epsilon2: 0`, and
both multihit counts equal to one.

Generate and test tables with the locked Python environment:

```bash
uv run python tools/generate_partitioning.py linear 3 partitionings/linear_m3.yaml
uv run python tools/generate_partitioning.py fibonacci 88 partitionings/fibonacci_N88.yaml
uv run python tools/test_generate_partitioning.py
```

### Temporal Dirichlet slabs

Temporal Dirichlet mode is restricted to 3D SU(2) pure Wilson builds. For
physical slab thickness `Nt`, store `L2: Nt + 1` sites and set
`temporal_dirichlet: true`. Spatial links in directions 0 and 1 at stored
slices 0 and `Nt` are fixed to the identity; direction-2 links remain
dynamical.

The periodic neighbor table is unchanged. The wrapping link based at `t=Nt`
is an exactly factorized PCM sector and is excluded from the physical
holonomy:

```text
G(x) = U_2(x,0) ... U_2(x,Nt-1).
```

Use the dedicated Dirichlet holonomy, plaquette-profile, boundary-Wilson-loop,
and bulk-Polyakov measurements. Periodic Polyakov observables are rejected in
this mode; the aggregate plaquette still includes the factorized wrapping
sector. Restart loading validates the fixed boundaries and fails rather than
repairing an invalid field.

## Analysis helpers

The scripts in `analysis/` summarize the incremental output. Inspect their
arguments with:

```bash
uv run python analysis/plaquette_stats.py --help
uv run python analysis/polyakov_stats.py --help
uv run python analysis/wtemp_stats.py --help
uv run python analysis/partition_benchmark.py --help
```

The first three require an explicit thermalization cut and block size; choose
blocks longer than the measured autocorrelation scale. The partition benchmark
uses the automatic-window autocorrelation estimate of
[Wolff, CPC 156 (2004) 143](https://arxiv.org/abs/hep-lat/0306017).

## Repository layout

- `include/`: public fields, groups, updates, observables, parameters, and I/O
- `lib/`: compiled Metropolis and heatbath implementations
- `binaries/`: simulation drivers and deterministic gradient-flow check
- `tests/`: configuration-dependent deterministic and statistical checks
- `analysis/`: Python post-processing helpers
- `partitionings/`: checked finite SU(2) point sets
- `thirdparty/`: Kokkos and yaml-cpp submodules

KLFT is distributed under the terms in `LICENSE`.
