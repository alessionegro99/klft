# klft

A library for lattice field theory simulation accelerated using Kokkos

# Installation

use git to clone the repository

```bash
git clone https://github.com/aniketsen/klft.git /path/to/klft
cd /path/to/klft
```

setup `kokkos` and `yaml-cpp` 

```bash
git submodule update --init --recursive
```

build the library

```bash
mkdir /path/to/build
cd /path/to/build

cmake [Kokkos options] /path/to/klft

make -j<number of threads>
```

### Kokkos options

The most important Kokkos options are:

`-DKokkos_ENABLE_CUDA=ON` to enable CUDA support

`-DKokkos_ENABLE_OPENMP=ON` to enable OpenMP support

`-DKokkos_ARCH_<arch>=ON` to enable a specific architecture (e.g. `-DKokkos_ARCH_AMPERE80=ON` for NVIDIA A100 gpus)

see the [Kokkos documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cmake-keywords) for more options

### KLFT options

KLFT is configured in the Bonn style: the dimension and gauge group are chosen at compile time.

`-DKLFT_NDIM=2|3|4` chooses the lattice dimension

`-DKLFT_NC=1|2|3` chooses the gauge group:
- `1` = `U(1)`
- `2` = `SU(2)`
- `3` = `SU(3)`

Example:

```bash
cmake -DKokkos_ENABLE_CUDA=ON -DKLFT_NDIM=4 -DKLFT_NC=3 /path/to/klft
```

## Code layout

The public headers are now grouped by role under `include/`:

- `include/core/` for shared types, compiled-theory settings, and indexing helpers
- `include/groups/` for gauge-group storage and algebra
- `include/fields/` for Kokkos-backed lattice field wrappers
- `include/updates/` for Metropolis and heatbath/overrelaxation kernels
- `include/observables/` for plaquettes, Wilson loops, Retrace(U), and nested Wilson actions
- `include/params/` for runtime parameter structs
- `include/io/` for YAML parsing and driver/sample-input helpers

# Usage

## Metropolis

```bash
binaries/metropolis
  -f <file_name> --filename <file_name>
     Name of the input file.
     Default: input.yaml
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```

Running `binaries/metropolis` with no arguments writes a sample `input.yaml`
and exits.

Observable files are written incrementally and use space-separated columns.
The plaquette file written by `metropolis` starts with `step`, ends with
`acceptance_rate time`, and contains the selected plaquette columns described
below.
Wilson-loop multihit measurements in this executable use Metropolis local-link
updates for the averaged links.

`start` accepts `"cold"` (all links equal the identity) or `"hot"` (random
group matrices generated from `seed`). Omitting it preserves the cold-start
default.

### SU(2) partitionings

An SU(2) build can restrict every link to a finite linear or Fibonacci
partition. The implementation follows the point sets and nearest-neighbor
Metropolis update of [Hartung et al., EPJC 82 (2022)
237](https://arxiv.org/abs/2201.09625). Because neighbor degrees can vary, it
uses the full Metropolis--Hastings proposal ratio of [Hastings, Biometrika 57
(1970) 97](https://doi.org/10.1093/biomet/57.1.97).

```yaml
PartitioningParams:
  enabled: true
  table_file: "partitionings/fibonacci_N88.yaml"
```

`nHits` remains the number of nearest-neighbor proposals per link; the paper
uses `nHits: 10`. `delta` is ignored in partition mode. Partition mode requires
`epsilon1: 0`, `epsilon2: 0`, `wilson_loop_multihit: 1`, and
`polyakov_loop_multihit: 1`. It is rejected by non-SU(2) builds and by the
heatbath driver. Cold starts use the table point nearest the identity; hot
starts sample table points according to their integration weights.

Generate the deterministic tables with NumPy and SciPy through `uv`:

```bash
uv run python tools/generate_partitioning.py linear 3 partitionings/linear_m3.yaml
uv run python tools/generate_partitioning.py fibonacci 88 partitionings/fibonacci_N88.yaml
uv run python tools/test_generate_partitioning.py
```

Linear tables use the analytic weight `(sqrt(2)/M)^3` and a sign-aware unit
transfer graph. Fibonacci tables use uniform weights and convex-hull edges,
which are the spherical Delaunay neighbors. `--weights FILE` overrides the
default weights with a one-column text or `.npy` file. Files use JSON syntax,
which is valid YAML, and are validated again when loaded by KLFT.

Set `measure_retrace_U2: true` and `RetraceU2_filename` to write the normalized
observable `Re Tr(U^2) / Nc`. For a benchmark with plaquettes measured every
sweep, report update and autocorrelation-adjusted sampling rates with:

```bash
uv run python analysis/partition_benchmark.py plaquette.out \
  --volume 4096 --dimensions 4 --hits 10 --thermalization 1000
```

### Example input.yaml

```yaml
# input.yaml
MetropolisParams:
  L0: 8
  L1: 8
  L2: 8
  L3: 8
  nHits: 10
  nSweep: 1000
  seed: 32091
  start: "cold"
  beta: 2.0
  delta: 0.1
  epsilon1: 0.0
  epsilon2: 0.0

PartitioningParams:
  enabled: false
  table_file: "partitionings/fibonacci_N88.yaml"

GaugeObservableParams:
  measurement_interval: 10
  measure_plaquette: true
  measure_plaquette_spatial: false
  measure_plaquette_temporal: false
  measure_wilson_loop_temporal: true
  measure_wilson_loop_mu_nu: true
  measure_polyakov_loop: true
  measure_polyakov_correlator: true
  measure_polyakov_susceptibility: true
  measure_retrace_U: false
  measure_retrace_U2: false
  wilson_loop_multihit: 1
  polyakov_loop_multihit: 1
  polyakov_correlator_max_r: 4
  measure_nested_wilson_action: false
  nested_child_offset: [0, 0, 0, 0]
  W_temp_L_T_pairs:
    - [2, 2]
    - [3, 3]
    - [4, 4]
  W_mu_nu_pairs:
    - [0, 1]
    - [0, 2]
    - [0, 3]
  W_Lmu_Lnu_pairs:
    - [2, 2]
    - [3, 3]
    - [4, 3]
  plaquette_filename: "plaquette.out"
  W_temp_filename: "w_temp.out"
  W_mu_nu_filename: "w_mu_nu.out"
  polyakov_loop_filename: "polyakov_loop.out"
  polyakov_correlator_filename: "polyakov_correlator.out"
  polyakov_susceptibility_filename: "polyakov_susceptibility.out"
  RetraceU_filename: "retrace_u.out"
  RetraceU2_filename: "retrace_u2.out"
  nested_wilson_action_filename: "nested_wilson_action.out"
  write_to_file: true

GradientFlowParams:
  enabled: false
  integrator: "rk3"
  dt: 0.01
  t_values: [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
  measure_energy_clover: true
  measure_wilson_loop_temporal: false
  measure_wilson_loop_mu_nu: false
  extract_t0: false
  t0_target: 0.3
  obs_filename: "gradient_flow_obs.dat"
  W_temp_filename: "gradient_flow_wtemp.dat"
  W_mu_nu_filename: "gradient_flow_w_mu_nu.dat"
  t0_filename: "gradient_flow_t0.dat"
```

## Heatbath

```bash
binaries/heatbath
  -f <file_name> --filename <file_name>
     Name of the input file.
     Default: input.yaml
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```

Running `binaries/heatbath` with no arguments writes a sample `input.yaml` and exits.

Observable files are written incrementally and use space-separated columns.
The plaquette file written by `heatbath` starts with `step`, ends with `time`,
and contains the selected plaquette columns. `measure_plaquette` writes the
existing average over all planes; `measure_plaquette_spatial` and
`measure_plaquette_temporal` independently add averages over spatial-spatial
and spatial-temporal planes. Each component is normalized by its own number of
planes, as in `yang-mills-Bonn`. KLFT uses direction `KLFT_NDIM - 1` as time.
Wilson-loop multihit measurements in this executable use heatbath plus
`nOverrelax` local overrelaxation updates for the averaged links.
Polyakov-loop multihit in this executable uses the same local heatbath plus
overrelaxation updates on every temporal link of the loop.
Polyakov-loop correlators are written as `# step R real imaginary`; `R = 0`
and `R = 1` always use raw Polyakov loops, while only `R >= 2` uses the
configured Polyakov multihit.
The Polyakov-susceptibility file is written as `# step G_0 G_pmin`, from the
spatial Fourier amplitude
`A(p) = 1/[(KLFT_NDIM-1)V_s] sum_x exp(i p.x) P(x)` used by the Bonn-lverzich
comparison: `G_0 = |A(0)|^2`, while `G_pmin` is
`|A(p_min)|^2`, averaged over the spatial directions, with
`p_min = 2*pi/L`. These use raw Polyakov loops, independent of
`polyakov_loop_multihit` (a multihit estimator would bias the diagonal
self-term of `|A(p)|^2`). The finite-size-scaling analysis builds the Binder
cumulant `U4 = <G_0^2>/<G_0>^2` and the second-moment correlation length
`xi = sqrt(<G_0>/<G_pmin> - 1) / (2*sin(pi/L))`.

### Example heatbath input.yaml

```yaml
# input.yaml
HeatbathParams:
  L0: 8
  L1: 8
  L2: 8
  L3: 8
  nSweep: 1000
  nOverrelax: 5
  seed: 32091
  start: "cold"
  beta: 2.0
  epsilon1: 0.0    # supported by heatbath/overrelaxation
  epsilon2: 0.0    # currently not supported by heatbath/overrelaxation

GaugeObservableParams:
  measurement_interval: 10
  measure_plaquette: true
  measure_plaquette_spatial: false
  measure_plaquette_temporal: false
  measure_wilson_loop_temporal: true
  measure_wilson_loop_mu_nu: true
  measure_polyakov_loop: true
  measure_polyakov_correlator: true
  measure_polyakov_susceptibility: true
  measure_retrace_U: false
  wilson_loop_multihit: 1  # H+OR Wilson-loop multihit; 1 disables averaged-link sampling
  polyakov_loop_multihit: 1  # H+OR Polyakov-loop multihit; 1 disables averaged-link sampling
  polyakov_correlator_max_r: 4  # must not exceed half the smallest spatial extent
  measure_nested_wilson_action: false
  nested_child_offset: [0, 0, 0, 0]
  W_temp_L_T_pairs:
    - [2, 2]
    - [3, 3]
    - [4, 4]
  W_mu_nu_pairs:
    - [0, 1]
    - [0, 2]
    - [0, 3]
  W_Lmu_Lnu_pairs:
    - [2, 2]
    - [3, 3]
    - [4, 3]
  plaquette_filename: "plaquette.out"
  W_temp_filename: "w_temp.out"
  W_mu_nu_filename: "w_mu_nu.out"
  polyakov_loop_filename: "polyakov_loop.out"
  polyakov_correlator_filename: "polyakov_correlator.out"
  polyakov_susceptibility_filename: "polyakov_susceptibility.out"
  RetraceU_filename: "retrace_u.out"
  nested_wilson_action_filename: "nested_wilson_action.out"
  write_to_file: true

GradientFlowParams:
  enabled: false
  integrator: "rk3"
  dt: 0.01
  t_values: [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
  measure_energy_clover: true
  measure_wilson_loop_temporal: false
  measure_wilson_loop_mu_nu: false
  extract_t0: false
  t0_target: 0.3
  obs_filename: "gradient_flow_obs.dat"
  W_temp_filename: "gradient_flow_wtemp.dat"
  W_mu_nu_filename: "gradient_flow_w_mu_nu.dat"
  t0_filename: "gradient_flow_t0.dat"
```
