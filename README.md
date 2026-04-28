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
The plaquette file written by `metropolis` stores `step plaquette acceptance_rate time`.
Wilson-loop multihit measurements in this executable use Metropolis local-link
updates for the averaged links.

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
  beta: 2.0
  delta: 0.1
  epsilon1: 0.0
  epsilon2: 0.0

GaugeObservableParams:
  measurement_interval: 10
  measure_plaquette: true
  measure_wilson_loop_temporal: true
  measure_wilson_loop_mu_nu: true
  measure_polyakov_loop: true
  measure_polyakov_correlator: true
  measure_retrace_U: false
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
The plaquette file written by `heatbath` stores `step plaquette time`.
Wilson-loop multihit measurements in this executable use heatbath plus
`nOverrelax` local overrelaxation updates for the averaged links.
Polyakov-loop multihit in this executable uses the same local heatbath plus
overrelaxation updates on every temporal link of the loop.
Polyakov-loop correlators are written as `# step R real imaginary`; `R = 0`
and `R = 1` always use raw Polyakov loops, while only `R >= 2` uses the
configured Polyakov multihit.

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
  beta: 2.0
  epsilon1: 0.0    # supported by heatbath/overrelaxation
  epsilon2: 0.0    # currently not supported by heatbath/overrelaxation

GaugeObservableParams:
  measurement_interval: 10
  measure_plaquette: true
  measure_wilson_loop_temporal: true
  measure_wilson_loop_mu_nu: true
  measure_polyakov_loop: true
  measure_polyakov_correlator: true
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
