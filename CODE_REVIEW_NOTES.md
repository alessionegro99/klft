# KLFT Code Review Notes

This file collects possible bugs, Kokkos/GPU performance improvements, and
cleanup ideas from the file-by-file review. It is a persistent checklist, not an
implementation plan. Mark an item `[x]` only after the change has been
implemented, reviewed, and validated, or after we explicitly decide to drop it.
Performance work should preserve Kokkos GPU execution and be evaluated with the
A100/CUDA preset when possible.

## Highest Priority

- [x] `include/updates/gradient_flow.hpp`: Keep current `t0` search range tied
  to `max(t_values)`; this behavior is intentional.
- [x] `include/observables/gauge_observables.hpp`: Fix staged-measurement clearing
  so output buffers are cleared only after successful file appends.

## Core

- [ ] `include/core/common.hpp`: No clear bug. Main possible optimization is
  replacing the global default `MDRangePolicy` alias with a backend/rank-specific
  policy factory after A100 benchmarks. Watch later files for unsafe aliasing
  with `Restrict` views.
- [ ] `include/core/compiled_theory.hpp`: No runtime performance issue. Consider
  adding a generic `make_gauge_field_with`/`IndexArray` factory here to avoid
  duplicated rank dispatch, and optional `static_assert`s for clearer compile
  errors.
- [x] `include/core/indexing.hpp`: Fix `shift_index_minus` for
  `shift >= dimension` using normalized modulo.
- [ ] `include/core/indexing.hpp`: Performance issue: `shift_index_plus/minus`
  copy full `IndexArray`s every step, hot in Wilson loops/clover; consider
  in-place shift helpers or direct coordinate arithmetic in hot loops.
- [ ] `include/core/klft_config.hpp.in`: No clear bug or performance issue. CMake
  validates `KLFT_NDIM`/`KLFT_NC` before generating this file. Possible cleanup:
  add `static_assert`s in the generated header for clearer errors if included
  outside normal CMake flow.

## Field Wrappers

- [ ] `include/fields/complex_field.hpp`: Temporary fields are always
  zero-initialized with a kernel and fence, but observables often overwrite every
  entry before `sum`; add uninitialized allocation path or direct reductions to
  avoid extra kernels. Const accessors returning mutable refs are Kokkos-view-like
  but risky API; document or separate read/write access.
- [ ] `include/fields/gauge_field.hpp`: No obvious staple formula bug. Add an
  uninitialized allocation constructor/tag for temporary gauge fields; current
  copy/workspace paths pay avoidable initialization kernels and fences. Consider
  replacing runtime device asserts with `static_assert(Nd == rank)`. Same Kokkos
  shallow-const accessor caveat as `complex_field.hpp`.
- [ ] `include/fields/scalar_field.hpp`: No clear correctness bug. Current
  Metropolis acceptance counting uses a full scalar field plus `sum`; replace
  with direct Kokkos `parallel_reduce` in the update kernel to avoid
  allocation/init/write/read overhead. Same Kokkos shallow-const accessor caveat
  as other field wrappers.
- [ ] `include/fields/field_type_traits.hpp`: No bug/performance issue.
  Low-priority cleanup: field wrappers carry both rank-specific class names and
  an `Nd` template parameter; enforce `Nd == rank` or remove the redundant
  parameter to prevent accidental misuse.

## Groups

- [x] `include/groups/gauge_group.hpp`: Add clearer `static_assert`/error for
  unsupported `Nc`.
- [ ] `include/groups/gauge_group.hpp`: Keep in mind `index_t=int` limits very
  large indexing.
- [ ] `include/groups/group_ops.hpp`: No obvious group-algebra bug. Optimize
  `SUNMatrix` operators by avoiding `make_zero_sun_matrix` when every entry is
  overwritten, and implement `+=`/`-=`/`*=` in place to reduce
  temporaries/register pressure. Robustness: `restoreSUN(U1/SU2)` divides by
  zero on zero input; add guard if zero links can ever reach projection.

## IO And YAML

- [ ] `include/io/driver_utils.hpp`: Host-side only, no Kokkos concern. Minor
  cleanup: avoid README/sample YAML drift, optionally reject extra positional
  args and reset `getopt` state if `parse_driver_args` is ever unit-tested or
  reused.
- [ ] `include/io/input_parser.hpp`: Host-side only. Improve robustness by catching
  `YAML::Exception` around all `node.as<T>()` parsing, not only `LoadFile`.
  Reduce repeated `YAML::LoadFile` calls by loading once and parsing all param
  structs from the same `YAML::Node`. Consider validating
  `polyakov_correlator_max_r` against lattice extents before runtime measurement
  throws.
- [ ] `include/klft.hpp`: No bug/performance issue; minimal public driver
  declarations only.

## Observables

- [ ] `include/observables/clover_energy.hpp`: Clover orientation looks consistent;
  no obvious physics bug. Performance issue: hot gradient-flow/`t0` observable
  uses 1D `RangePolicy` with `lin -> site` division/modulo and many shifted
  `IndexArray` copies. Rewrite with rank `MDRangePolicy`/direct coordinate
  arithmetic or optimized shift helpers. Do not change normalization without
  explicit Bonn/reference comparison.
- [ ] `include/observables/gauge_observables.hpp`: Host staging vectors are mostly
  size-1 when `write_to_file=true`; direct row append would reduce allocations.
- [x] `include/observables/gauge_observables.hpp`: Add size checks for
  Wilson/retrace flushers too.
- [ ] `include/observables/multihit_links.hpp`: No obvious bug. Metropolis multihit
  can cache old trace/action term per hit. Heatbath multihit reprojects after
  each local update; safe but potentially expensive for SU(3).
- [x] `include/observables/nested_wilson_action.hpp`: Replace blocked-plaquette
  temporary field plus `sum` with direct Kokkos `parallel_reduce`.
- [ ] `include/observables/nested_wilson_action.hpp`: Reduce repeated
  shifted-index/`blocked_link` work.
- [x] `include/observables/plaquette.hpp`: Replace per-site complex field plus
  second reduction with direct Kokkos `parallel_reduce`.
- [ ] `include/observables/plaquette.hpp`: Avoid `shift_index` copies in hot loop.
- [x] `include/observables/wilson_loop.hpp`: Replace per-site complex field plus
  second reduction with direct Kokkos `parallel_reduce`, preserving raw,
  Metropolis-multihit, and heatbath-multihit paths.
- [x] `include/observables/polyakov_loop.hpp`: Single Polyakov loop now uses one
  direct complex reduction instead of a local array plus separate real/imag
  reductions.
- [ ] `include/observables/polyakov_loop.hpp`: `RangePolicy` `lin -> site`
  divisions remain avoidable for Polyakov-loop helpers.
- [x] `include/observables/polyakov_correlator.hpp`: Combine repeated real and
  imaginary correlator reductions into one complex reducer.
- [ ] `include/observables/polyakov_correlator.hpp`: Consider computing multiple
  `R` values in one pass if `max_r` is large.
- [x] `include/observables/retrace.hpp`: Replace linear index division/modulo
  per link with an `MDRangePolicy` site reduction.

## Params

- [x] `include/params/heatbath_params.hpp`: Remove unused `HeatbathParams::delta`.
- [ ] `include/params/*.hpp`: Constructor style can be made consistent.

## Updates

- [ ] `include/updates/heatbath_link_updates.hpp`: No clear bug from inspection. Keep
  `epsilon2` explicitly unsupported for heatbath. Benchmark whether repeated
  `restoreSUN` calls are needed for every multihit/overrelax substep.
- [ ] `include/updates/metropolis.hpp`: Staple reuse across `nHits` is correct
  because staple excludes the updated link. Performance issue: acceptance
  counting uses scalar field plus `sum`; use direct reduction.
- [ ] `include/updates/heatbath.hpp`: Kernel/fence structure is conservative and
  likely correct. Many fences are algorithmically tied to checkerboard
  dependencies; only optimize after benchmarking.
- [ ] `include/updates/gradient_flow.hpp`: Biggest performance target:
  copy/workspace initialization is wasteful, and every RK3 substage fences. Use
  uninitialized workspaces and reduce fences with ordered execution-space
  dispatch.

## Binaries, CMake, And Analysis

- [x] `binaries/gradient_flow_check.cpp`: Wire the check into CTest and remove
  the group name from its banner.
- [x] `CMakeLists.txt`/`CMakePresets.json`: Align preset CMake minimum with
  `CMakeLists.txt` and use `PROJECT_SOURCE_DIR`/`PROJECT_BINARY_DIR` paths.
- [x] `analysis/wtemp_stats.py`: Understand gradient-flow temporal Wilson-loop
  files with `conf_id t_over_a2 L T W_temp` columns.
- [x] `analysis/*.py`: Factor duplicated blocking logic into
  `analysis/stats_common.py`.

## Cross-Cutting Reminders

- [ ] Field wrappers: add uninitialized allocation paths for temporary
  fields/workspaces, avoid unnecessary initialization fences, and document
  Kokkos-style mutable access through const wrappers.
- [ ] Acceptance counting: consider direct Kokkos reductions instead of scalar-field
  temporaries.
- [ ] Group ops: reduce `SUNMatrix` temporaries/in-place ops; guard U(1)/SU(2)
  projection against zero norm.
- [ ] YAML parsing: load once, catch conversion errors, move more runtime validation
  into input validation.
- [ ] Clover energy: optimize index handling; verify any normalization change against
  Bonn/reference.
