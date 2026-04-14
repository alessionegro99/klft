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

### Example input.yaml

```yaml
# input.yaml
MetropolisParams:    # parameters for the Metropolis algorithm
  L0: 8       # lattice extent in 0 direction
  L1: 8       # lattice extent in 1 direction
  L2: 8       # lattice extent in 2 direction
  L3: 8       # lattice extent in 3 direction
  nHits: 10       # number of hits per sweep
  nSweep: 1000      # number of sweeps
  seed: 32091     # random seed
  beta: 2.0       # inverse coupling constant
  delta: 0.1      # step size for the Metropolis algorithm

GaugeObservableParams:
  measurement_interval: 10               # interval for measurements
  measure_plaquette: true                # measure the plaquette
  measure_wilson_loop_temporal: true    # measure the temporal Wilson loop
  measure_wilson_loop_mu_nu: true       # measure the spatial Wilson loop
  wilson_loop_multihit: 1               # Wilson-loop-only multihit; 1 keeps the current observable
  nested_child_offset: [0, 0, 0, 0]     # required if measure_nested_wilson_action is true
  W_temp_L_T_pairs:      # pairs of (L, T) values for the temporal Wilson loop
    - [2, 2]
    - [3, 3]             # keep a non-decreasing order (as much as possible)
    - [4, 4]
  W_mu_nu_pairs:      # pairs of (mu, nu) values for the planar Wilson loop
    - [0, 1]
    - [0, 2]
    - [0, 3]
  W_Lmu_Lnu_pairs:      # pairs of (Lmu, Lnu) values for the lengths of the 
    - [2, 2]            # planar Wilson loop in the mu and nu directions
    - [3, 3]            # again, keep a non-decreasing order (as much as possible)
    - [4, 3]
  plaquette_filename: "plaquette.out"  # filename to output the plaquette
  W_temp_filename: "w_temp.out"        # filename to output the temporal Wilson loop
  W_mu_nu_filename: "w_mu_nu.out"      # filename to output the planar Wilson loop
  write_to_file: true                  # write the measurements to file
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
  delta: 0.1       # only used by Wilson-loop multihit measurements
  epsilon1: 0.0    # supported by heatbath/overrelaxation
  epsilon2: 0.0    # currently not supported by heatbath/overrelaxation

GaugeObservableParams:
  measurement_interval: 10
  measure_plaquette: true
  measure_wilson_loop_temporal: true
  measure_wilson_loop_mu_nu: true
  measure_retrace_U: false
  wilson_loop_multihit: 1
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
  RetraceU_filename: "retrace_u.out"
  nested_wilson_action_filename: "nested_wilson_action.out"
  write_to_file: true
```
