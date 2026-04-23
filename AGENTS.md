# Repository Guidelines

## Project Structure & Module Organization

KLFT is a C++17 lattice field theory codebase built around Kokkos and yaml-cpp. Public headers live under `include/`: `core/` for theory settings and indexing, `groups/` for gauge algebra, `fields/` for Kokkos containers, `updates/` for Metropolis/heatbath kernels, `observables/` for measurements, `params/` for runtime structs, and `io/` for YAML and driver helpers. `lib/` builds the static `klft` library; `binaries/` contains the `metropolis` and `heatbath` CLI entry points. `thirdparty/` contains vendored submodules; do not edit them for KLFT changes. If `yang-mills-Bonn/` is present, treat it as reference code, not first-party source.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: fetch Kokkos and yaml-cpp.
- `cmake --preset debug` or `cmake --preset relwithdebinfo`: configure OpenMP/Zen3 builds under `../../Build/`.
- `cmake --build --preset debug -j8`: build the `klft` library and both drivers.
- `cmake -S . -B build -DKLFT_NDIM=4 -DKLFT_NC=3 -DKokkos_ENABLE_OPENMP=ON`: portable local configure when presets do not match the machine.
- `./build/binaries/heatbath` or `./build/binaries/metropolis`: write a sample `input.yaml` with no arguments; run one with `-f input.yaml`.

## Coding Style & Naming Conventions

Use the existing two-space indentation and compact C++ style. Header filenames are generally `snake_case.hpp`; public types use names such as `GaugeObservableParams`, `SUN<Nc>`, and `GaugeField`. Keep device code Kokkos-compatible: avoid host-only APIs inside `KOKKOS_*_FUNCTION` paths, preserve template dispatch over dimension and gauge group, and prefer existing helpers over duplicated algebra or indexing logic. There is no repository formatting config; keep diffs locally consistent.

## Testing Guidelines

There is no first-party test suite or CTest setup. Build the affected configuration and smoke-test the relevant driver. For parser changes, confirm sample generation and `-f input.yaml` parsing. For numerical update or observable changes, record the backend, `KLFT_NDIM`, `KLFT_NC`, lattice size, seed, beta, and output such as plaquette values.

## Commit & Pull Request Guidelines

Recent history uses short, direct subjects such as `Refactor KLFT layout and update algorithms` or `Add nested child offset to sample inputs`. Prefer imperative, specific commit messages and keep unrelated formatting or vendored changes out. Pull requests should describe the physics or configuration impact, list build/run commands used, include relevant output snippets, and call out any changes to YAML fields or observable file formats.

## Configuration Notes

`KLFT_NDIM` accepts `2`, `3`, or `4`; `KLFT_NC` accepts `1`, `2`, or `3`. Keep generated build directories, local workspaces, sample `input.yaml`, and measurement outputs out of commits unless they are intentional documentation fixtures.
