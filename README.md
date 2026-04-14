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

### Example input.yaml

```yaml
# input.yaml
MetropolisParams:    # parameters for the Metropolis algorithm
  Ndims: 4    # number of dimensions [2,3,4]
  Nd: 4       # number of link dimensions (this must be strictly same as Ndims)
  Nc: 1       # number of colors (defines SU(Nc)) [1,2,3]
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
  W_temp_L_T_pairs:      # pairs of (L, T) values for the temporal Wilson loop
    - [2, 2]
    - [3, 4]             # keep a non-decreasing order (as much as possible)
    - [4, 3]             # for maximum efficiency
    - [4, 4]
  W_mu_nu_pairs:      # pairs of (mu, nu) values for the planar Wilson loop
    - [0, 1]
    - [1, 2]
    - [3, 2]
  W_Lmu_Lnu_pairs:      # pairs of (Lmu, Lnu) values for the lengths of the 
    - [2, 2]            # planar Wilson loop in the mu and nu directions
    - [3, 3]            # again, keep a non-decreasing order (as much as possible)
    - [4, 3]
  plaquette_filename: "plaquette.out"  # filename to output the plaquette
  W_temp_filename: "W_temp.out"        # filename to output the temporal Wilson loop
  W_mu_nu_filename: "W_mu_nu.out"      # filename to output the planar Wilson loop
  write_to_file: true                  # write the measurements to file
```

# Environment variables

### KLFT_VERBOSITY
Set the verbosity level of the library.
- 0: silent
- 1: summarize
- 2: verbose
- &gt;=3: debug
- Default: 0

### KLFT_TUNING
Sets whether to tune the Kokkos `MDRangePolicy` for `rank > 1` or not.
- 0: do not tune
- 1: tune
- Default: 0

### KLFT_CACHE_FILE
Sets the file to store the tuning results.
Also sets the file to read the tuning results from.
- Default: None
