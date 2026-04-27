# Repository Guidelines

## Project Structure & Module Organization

KLFT is a C++17 lattice field theory library built with Kokkos and yaml-cpp. Public headers are under `include/`: `core/` for theory settings and indexing, `groups/` for gauge algebra, `fields/` for Kokkos-backed containers, `updates/` for Metropolis and heatbath kernels, `observables/` for measurements, `params/` for runtime structs, and `io/` for YAML/driver helpers. `lib/` builds the library, `binaries/` contains the `metropolis` and `heatbath` drivers, and `analysis/` holds Python post-processing helpers. Vendored dependencies live in `thirdparty/`; avoid local changes there unless updating submodules intentionally.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: fetch Kokkos and yaml-cpp.
- `cmake --preset debug`: configure a Debug OpenMP/Zen3 build in `../../build/klft-debug`.
- `cmake --build --preset debug -j8`: build the `klft` library and both drivers.
- `cmake --preset relwithdebinfo`: configure an optimized build with debug symbols.
- `cmake -S . -B build -DKLFT_NDIM=4 -DKLFT_NC=3 -DKokkos_ENABLE_OPENMP=ON`: portable local configure when presets do not match the machine.
- `./build/binaries/heatbath` or `./build/binaries/metropolis`: write a sample `input.yaml` with no arguments; run one with `-f input.yaml`.

## Coding Style & Naming Conventions

Use the existing compact C++ style with two-space indentation. Header files use `snake_case.hpp`; public types use `PascalCase` names such as `GaugeObservableParams`, while template/group identifiers may follow established forms like `SUN<Nc>`. Keep device paths Kokkos-compatible: avoid host-only APIs inside `KOKKOS_*_FUNCTION` code and prefer existing indexing, group, and observable helpers. There is no formatter config, so keep formatting locally consistent.

## Testing Guidelines

No first-party CTest or unit test suite is currently present. For changes, build the affected preset or explicit CMake configuration and smoke-test the relevant driver. For parser or sample-input changes, verify sample generation and `-f input.yaml` parsing. For kernels or observables, record backend, `KLFT_NDIM`, `KLFT_NC`, lattice size, seed, beta, and representative output.

## Commit & Pull Request Guidelines

Recent history uses short, direct subjects, including `Updates`, `Added useful preset`, and `Add nested child offset to sample inputs`. Prefer imperative, specific messages and keep unrelated formatting, generated output, and vendored changes separate. Pull requests should summarize physics/configuration impact, list build and run commands, include output snippets, and call out YAML or file-format changes.

## Configuration Notes

`KLFT_NDIM` accepts `2`, `3`, or `4`; `KLFT_NC` accepts `1`, `2`, or `3` (`U(1)`, `SU(2)`, `SU(3)`). Keep local build trees, generated `input.yaml`, and measurement outputs out of commits unless they are deliberate documentation fixtures.
