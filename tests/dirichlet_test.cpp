#include "core/compiled_theory.hpp"
#include "core/temporal_dirichlet.hpp"
#include "io/gauge_configuration.hpp"
#include "observables/dirichlet_action.hpp"
#include "partitioning/partition_table.hpp"
#include "updates/gradient_flow.hpp"
#include "updates/heatbath.hpp"
#include "updates/metropolis.hpp"
#include "updates/partitioned_metropolis.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace klft;
using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
using Gauge = typename DeviceGaugeFieldType<3, 2>::type;

int failures = 0;

void check(const bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

bool close(const real_t left, const real_t right,
           const real_t relative_tolerance = 2.0e-12) {
  return std::abs(left - right) <=
         relative_tolerance *
             std::max({real_t(1.0), std::abs(left), std::abs(right)});
}

real_t su2_distance2(const SU2 &left, const SU2 &right) {
  real_t result = 0.0;
  for (index_t component = 0; component < 4; ++component) {
    const real_t delta = left.comp[component] - right.comp[component];
    result += delta * delta;
  }
  return result;
}

Gauge make_identity(const index_t nx, const index_t ny,
                    const index_t slab_thickness) {
  return make_identity_gauge_field<3, 2>(nx, ny, slab_thickness + 1, 1);
}

Gauge make_random_valid(const index_t nx, const index_t ny,
                        const index_t slab_thickness, const uint64_t seed) {
  RNG rng(seed);
  auto gauge =
      make_hot_gauge_field<3, 2>(nx, ny, slab_thickness + 1, 1, rng);
  apply_temporal_dirichlet_boundaries<3, 2>(gauge);
  return gauge;
}

SU2 get_link(const Gauge &gauge, const IndexArray<3> &site,
             const index_t mu) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        gauge.field);
  return host(site[0], site[1], site[2], mu);
}

void set_link(Gauge &gauge, const IndexArray<3> &site, const index_t mu,
              const SU2 &link) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  gauge.field);
  host(site[0], site[1], site[2], mu) = link;
  Kokkos::deep_copy(gauge.field, host);
  Kokkos::fence();
}

SU2 get_staple(const Gauge &gauge, const IndexArray<3> &site,
               const index_t mu) {
  Kokkos::View<SU2> result("dirichlet_test_staple");
  Kokkos::parallel_for(
      "compute_dirichlet_test_staple", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const index_t) { result() = gauge.staple(site, mu); });
  Kokkos::fence();
  SU2 host_result;
  Kokkos::deep_copy(host_result, result);
  return host_result;
}

bool fields_equal(const Gauge &left, const Gauge &right,
                  const real_t tolerance = 0.0) {
  if (left.dimensions != right.dimensions) {
    return false;
  }
  const auto left_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), left.field);
  const auto right_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), right.field);
  for (index_t x = 0; x < left.dimensions[0]; ++x) {
    for (index_t y = 0; y < left.dimensions[1]; ++y) {
      for (index_t t = 0; t < left.dimensions[2]; ++t) {
        for (index_t mu = 0; mu < 3; ++mu) {
          if (su2_distance2(left_host(x, y, t, mu),
                            right_host(x, y, t, mu)) >
              tolerance * tolerance) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

Gauge reflected_configuration(const Gauge &source) {
  auto reflected = make_identity(source.dimensions[0], source.dimensions[1],
                                 slab_thickness_in_links<3>(source.dimensions));
  const auto input = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         source.field);
  auto output = Kokkos::create_mirror_view(reflected.field);
  const index_t top = upper_boundary_slice<3>(source.dimensions);
  for (index_t x = 0; x < source.dimensions[0]; ++x) {
    for (index_t y = 0; y < source.dimensions[1]; ++y) {
      for (index_t t = 0; t <= top; ++t) {
        output(x, y, t, 0) = input(x, y, top - t, 0);
        output(x, y, t, 1) = input(x, y, top - t, 1);
        output(x, y, t, 2) =
            t == top ? conj(input(x, y, top, 2))
                     : conj(input(x, y, top - t - 1, 2));
      }
    }
  }
  Kokkos::deep_copy(reflected.field, output);
  Kokkos::fence();
  return reflected;
}

Gauge chiral_transform(const Gauge &source, const SU2 &left,
                       const SU2 &right) {
  auto transformed = copy_gauge_field<3, 2>(source);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  transformed.field);
  const index_t top = upper_boundary_slice<3>(source.dimensions);
  for (index_t x = 0; x < source.dimensions[0]; ++x) {
    for (index_t y = 0; y < source.dimensions[1]; ++y) {
      host(x, y, 0, 2) = left * host(x, y, 0, 2);
      host(x, y, top - 1, 2) = host(x, y, top - 1, 2) * conj(right);
      host(x, y, top, 2) =
          right * host(x, y, top, 2) * conj(left);
    }
  }
  Kokkos::deep_copy(transformed.field, host);
  Kokkos::fence();
  return transformed;
}

real_t direct_spin_correlator(const SlabHolonomyField<2> &spins,
                              const index_t separation) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        spins);
  const index_t nx = static_cast<index_t>(spins.extent(0));
  const index_t ny = static_cast<index_t>(spins.extent(1));
  real_t sum = 0.0;
  for (index_t x = 0; x < nx; ++x) {
    for (index_t y = 0; y < ny; ++y) {
      sum += trace(host(x, y) * conj(host((x + separation) % nx, y))).real() /
             2.0;
      sum += trace(host(x, y) * conj(host(x, (y + separation) % ny))).real() /
             2.0;
    }
  }
  return sum / (2.0 * static_cast<real_t>(nx * ny));
}

void check_odd_spatial_coloring() {
  const IndexArray<3> even_dimensions{4, 6, 8};
  check(lattice_color_total<3>(even_dimensions, false) == 8,
        "legacy even lattice retains eight checkerboard colors");
  for (index_t linear_color = 0; linear_color < 8; ++linear_color) {
    const auto colors =
        decode_lattice_color<3>(linear_color, even_dimensions, false);
    const auto old_colors = oddeven_array<3>(linear_color);
    for (index_t d = 0; d < 3; ++d) {
      check(colors[d] == static_cast<index_t>(old_colors[d]),
            "even-lattice color order is unchanged");
    }
    const IndexArray<3> compressed{1, 1, 1};
    check(index_lattice_color<3, index_t>(compressed, colors,
                                          even_dimensions, false) ==
              index_odd_even<3, index_t>(compressed, old_colors),
          "even-lattice checkerboard site mapping is unchanged");
  }

  const IndexArray<3> dimensions{5, 3, 4};
  const index_t volume = dimensions[0] * dimensions[1] * dimensions[2];
  std::vector<index_t> visits(static_cast<size_t>(volume), 0);
  std::vector<index_t> site_color(static_cast<size_t>(volume), -1);
  const auto linear_index = [&](const IndexArray<3> &site) {
    return site[2] + dimensions[2] *
                         (site[1] + dimensions[1] * site[0]);
  };
  const index_t color_count = lattice_color_total<3>(dimensions, true);
  check(color_count == 18,
        "odd periodic spatial directions use three colors");
  for (index_t linear_color = 0; linear_color < color_count;
       ++linear_color) {
    const auto colors =
        decode_lattice_color<3>(linear_color, dimensions, true);
    IndexArray<3> end;
    for (index_t d = 0; d < 3; ++d) {
      end[d] = lattice_color_extent<3>(dimensions, d, colors[d], true);
    }
    for (index_t i0 = 0; i0 < end[0]; ++i0) {
      for (index_t i1 = 0; i1 < end[1]; ++i1) {
        for (index_t i2 = 0; i2 < end[2]; ++i2) {
          const auto site = index_lattice_color<3, index_t>(
              IndexArray<3>{i0, i1, i2}, colors, dimensions, true);
          const index_t flat = linear_index(site);
          ++visits[flat];
          site_color[flat] = linear_color;
        }
      }
    }
  }
  check(std::all_of(visits.begin(), visits.end(),
                    [](const index_t count) { return count == 1; }),
        "mixed two/three-color decomposition covers every site once");
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t t = 0; t < dimensions[2]; ++t) {
        const IndexArray<3> site{x, y, t};
        for (index_t d = 0; d < 2; ++d) {
          auto neighbor = site;
          neighbor[d] = (neighbor[d] + 1) % dimensions[d];
          check(site_color[linear_index(site)] !=
                    site_color[linear_index(neighbor)],
                "periodic spatial neighbors have distinct update colors");
        }
      }
    }
  }

  auto gauge = make_random_valid(5, 3, 3, 7777);
  HeatbathParams params;
  params.L0 = 5;
  params.L1 = 3;
  params.L2 = 4;
  params.beta = 2.2;
  params.nOverrelax = 1;
  params.temporal_dirichlet = true;
  RNG rng(7778);
  full_heatbath_sweep<3, 2>(gauge, params, rng);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(gauge),
        "odd periodic spatial extents update with exact boundaries");
}

void check_initialization_and_io(const std::string &partition_file) {
  auto cold = make_identity(4, 4, 2);
  apply_temporal_dirichlet_boundaries<3, 2>(cold);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(cold),
        "cold start fixes boundary tangential links");

  RNG hot_rng(1024);
  auto hot = make_hot_gauge_field<3, 2>(4, 4, 3, 1, hot_rng);
  apply_temporal_dirichlet_boundaries<3, 2>(hot);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(hot),
        "hot start fixes boundary tangential links");
  check(!is_exact_identity<2>(get_link(hot, IndexArray<3>{0, 0, 0}, 2)),
        "hot-start normal link at lower boundary remains random");
  check(!is_exact_identity<2>(get_link(hot, IndexArray<3>{0, 0, 2}, 2)),
        "hot-start wrapping normal link remains random");

  RNG local_rng(2048);
  auto local_random = make_random_gauge_field_with<3, 2>(
      IndexArray<3>{4, 4, 3}, local_rng, 0.2);
  apply_temporal_dirichlet_boundaries<3, 2>(local_random);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(local_random),
        "near-identity random initialization fixes boundaries");

  PartitionDeviceTable table;
  check(loadPartitionTable(partition_file, table),
        "load partition table for initialization tests");
  RNG partition_rng(4096);
  auto partition_cold = make_identity(4, 4, 2);
  initializePartitionGaugeField<3>(partition_cold, table, "cold",
                                   partition_rng, true);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(partition_cold),
        "partition cold start fixes boundaries");
  auto partition_hot = make_identity(4, 4, 2);
  initializePartitionGaugeField<3>(partition_hot, table, "hot", partition_rng,
                                   true);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(partition_hot),
        "partition hot start fixes boundaries");

  const std::string filename = "/tmp/klft_dirichlet_test.cfg";
  check(save_gauge_configuration<3, 2>(filename, partition_hot, true),
        "save partitioned Dirichlet configuration");
  auto partition_restart = make_identity(4, 4, 2);
  check(load_gauge_configuration<3, 2>(filename, partition_restart, true),
        "reload partitioned Dirichlet configuration");
  const auto restart_indices = initializePartitionIndicesFromGauge<3>(
      partition_restart, table, true);
  check(restart_indices.extent(0) != 0,
        "reconstruct partition indices for a restart");

  check(save_gauge_configuration<3, 2>(filename, hot, true),
        "save valid Dirichlet configuration");
  auto restored = make_identity(4, 4, 2);
  check(load_gauge_configuration<3, 2>(filename, restored, true),
        "reload valid Dirichlet configuration");
  check(fields_equal(hot, restored),
        "configuration save/reload preserves every link exactly");
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(restored),
        "restarted configuration validates exact boundaries");

  auto invalid = copy_gauge_field<3, 2>(hot);
  set_link(invalid, IndexArray<3>{0, 0, 0}, 0,
           get_link(hot, IndexArray<3>{0, 0, 0}, 2));
  check(!save_gauge_configuration<3, 2>(filename, invalid, true),
        "configuration writer rejects invalid fixed links");
  std::remove(filename.c_str());
}

void check_updates(const std::string &partition_file) {
  auto initial = make_random_valid(4, 4, 4, 8192);

  HeatbathParams heatbath;
  heatbath.L0 = 4;
  heatbath.L1 = 4;
  heatbath.L2 = 5;
  heatbath.beta = 2.2;
  heatbath.nOverrelax = 1;
  heatbath.temporal_dirichlet = true;
  RNG heatbath_rng(8193);
  auto heatbath_field = copy_gauge_field<3, 2>(initial);
  const SU2 adjacent_before =
      get_link(heatbath_field, IndexArray<3>{0, 0, 1}, 0);
  const SU2 bulk_before =
      get_link(heatbath_field, IndexArray<3>{0, 0, 2}, 0);
  const SU2 normal_before =
      get_link(heatbath_field, IndexArray<3>{0, 0, 0}, 2);
  const SU2 wrapping_before =
      get_link(heatbath_field, IndexArray<3>{0, 0, 4}, 2);
  for (index_t sweep = 0; sweep < 3; ++sweep) {
    full_heatbath_sweep<3, 2>(heatbath_field, heatbath, heatbath_rng);
  }
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(heatbath_field),
        "heatbath and overrelaxation preserve fixed links exactly");
  check(su2_distance2(adjacent_before,
                      get_link(heatbath_field, IndexArray<3>{0, 0, 1}, 0)) >
            1.0e-12,
        "boundary-adjacent spatial link changes under update");
  check(su2_distance2(bulk_before,
                      get_link(heatbath_field, IndexArray<3>{0, 0, 2}, 0)) >
            1.0e-12,
        "bulk spatial link changes under update");
  check(su2_distance2(normal_before,
                      get_link(heatbath_field, IndexArray<3>{0, 0, 0}, 2)) >
            1.0e-12,
        "normal link emerging from boundary remains dynamical");
  check(su2_distance2(wrapping_before,
                      get_link(heatbath_field, IndexArray<3>{0, 0, 4}, 2)) >
            1.0e-12,
        "factorized wrapping normal link remains dynamical");

  MetropolisParams metropolis;
  metropolis.L0 = 4;
  metropolis.L1 = 4;
  metropolis.L2 = 5;
  metropolis.beta = 2.2;
  metropolis.delta = 0.25;
  metropolis.nHits = 2;
  metropolis.temporal_dirichlet = true;
  RNG metropolis_rng(8194);
  auto metropolis_field = copy_gauge_field<3, 2>(initial);
  for (index_t sweep = 0; sweep < 3; ++sweep) {
    sweep_Metropolis<3, 2>(metropolis_field, metropolis, metropolis_rng);
  }
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(metropolis_field),
        "Metropolis preserves fixed links exactly");

  auto flowed = copy_gauge_field<3, 2>(initial);
  GradientFlowWorkspace<3, 2> workspace(flowed.dimensions);
  flow_step_rk3<3, 2>(flowed, workspace, 0.01, true);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(flowed),
        "gradient flow preserves fixed links exactly");

  PartitionDeviceTable table;
  check(loadPartitionTable(partition_file, table),
        "load partition table for update test");
  RNG partition_rng(8195);
  auto partition_field = make_identity(4, 4, 4);
  auto indices = initializePartitionGaugeField<3>(
      partition_field, table, "hot", partition_rng, true);
  MetropolisParams partition_params = metropolis;
  partition_params.L2 = 5;
  sweepPartitionedMetropolis<3>(partition_field, indices, table,
                                partition_params, partition_rng);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(partition_field),
        "partitioned Metropolis preserves fixed links exactly");

  // Nt=1 and Nt=2 exercise the smallest stored extents 2 and 3; the latter
  // is the odd temporal extent not supported by the periodic checkerboard.
  for (const index_t thickness : {1, 2}) {
    auto small = make_random_valid(2, 2, thickness, 9000 + thickness);
    HeatbathParams small_params = heatbath;
    small_params.L0 = 2;
    small_params.L1 = 2;
    small_params.L2 = thickness + 1;
    RNG small_rng(9100 + thickness);
    full_heatbath_sweep<3, 2>(small, small_params, small_rng);
    check(temporal_dirichlet_boundaries_are_exact<3, 2>(small),
          "smallest Dirichlet geometry updates without invalid indexing");
  }
}

void check_local_action_differences() {
  constexpr real_t beta = 3.25;
  auto gauge = make_random_valid(4, 4, 4, 16384);
  auto proposals = make_random_valid(4, 4, 4, 16385);
  struct LinkCase {
    IndexArray<3> site;
    index_t direction;
    const char *description;
  };
  const std::vector<LinkCase> cases{
      {IndexArray<3>{0, 0, 2}, 0, "bulk spatial link"},
      {IndexArray<3>{0, 0, 2}, 2, "bulk temporal link"},
      {IndexArray<3>{0, 0, 1}, 0, "boundary-adjacent spatial link"},
      {IndexArray<3>{0, 0, 0}, 2, "temporal link emerging from boundary"}};

  for (const auto &test_case : cases) {
    const SU2 old_link = get_link(gauge, test_case.site, test_case.direction);
    const SU2 new_link =
        get_link(proposals, test_case.site, test_case.direction);
    const SU2 staple = get_staple(gauge, test_case.site, test_case.direction);
    const real_t predicted =
        WilsonLocalActionDifference<2>(old_link, new_link, staple, beta);
    const real_t old_action = DirichletSlabWilsonAction<2>(gauge, beta);
    set_link(gauge, test_case.site, test_case.direction, new_link);
    const real_t new_action = DirichletSlabWilsonAction<2>(gauge, beta);
    check(close(predicted, new_action - old_action, 8.0e-12),
          test_case.description);
    set_link(gauge, test_case.site, test_case.direction, old_link);
  }
}

void check_symmetries_and_factorization() {
  constexpr real_t beta = 2.75;
  auto gauge = make_random_valid(4, 4, 4, 32768);
  const real_t periodic = PeriodicWilsonAction<2>(gauge, beta);
  const real_t slab = DirichletSlabWilsonAction<2>(gauge, beta);
  const real_t wrapping = WrappingPCMAction<2>(gauge, beta);
  check(close(periodic, slab + wrapping, 8.0e-12),
        "periodic storage factorizes into slab and wrapping PCM actions");

  auto reflected = reflected_configuration(gauge);
  check(temporal_dirichlet_boundaries_are_exact<3, 2>(reflected),
        "time reflection preserves fixed boundaries");
  check(close(DirichletSlabWilsonAction<2>(gauge, beta),
              DirichletSlabWilsonAction<2>(reflected, beta), 8.0e-12),
        "slab action is invariant under midpoint reflection");
  check(close(PeriodicWilsonAction<2>(gauge, beta),
              PeriodicWilsonAction<2>(reflected, beta), 8.0e-12),
        "factorized full action is invariant under midpoint reflection");
  const auto profile = MeasureDirichletPlaquetteProfiles<2>(gauge);
  const auto reflected_profile =
      MeasureDirichletPlaquetteProfiles<2>(reflected);
  const index_t thickness = slab_thickness_in_links<3>(gauge.dimensions);
  for (index_t t = 0; t <= thickness; ++t) {
    check(close(profile.spatial[t][1],
                reflected_profile.spatial[thickness - t][1]),
          "spatial profile reflection indexing");
  }
  for (index_t t = 0; t < thickness; ++t) {
    check(close(profile.temporal[t][1],
                reflected_profile.temporal[thickness - 1 - t][1]),
          "temporal profile reflection indexing");
  }

  auto transformations = make_random_valid(2, 2, 2, 32769);
  const SU2 left = get_link(transformations, IndexArray<3>{0, 0, 0}, 2);
  const SU2 right = get_link(transformations, IndexArray<3>{1, 0, 0}, 2);
  auto transformed = chiral_transform(gauge, left, right);
  check(close(PeriodicWilsonAction<2>(gauge, beta),
              PeriodicWilsonAction<2>(transformed, beta), 8.0e-12),
        "independent boundary SU(2) transformations preserve total action");

  const auto original_spins = SlabHolonomy<2>(gauge);
  const auto transformed_spins = SlabHolonomy<2>(transformed);
  const auto original_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), original_spins);
  const auto transformed_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), transformed_spins);
  const real_t original_pair =
      trace(original_host(0, 0) * conj(original_host(1, 0))).real();
  const real_t transformed_pair =
      trace(transformed_host(0, 0) * conj(transformed_host(1, 0))).real();
  check(close(original_pair, transformed_pair),
        "single-trace holonomy correlator is chiral invariant");
  check(std::abs(trace(original_host(0, 0)).real() -
                 trace(transformed_host(0, 0)).real()) >
            1.0e-8,
        "Tr G generally changes under independent boundary rotations");
}

void check_exact_nt1_mapping() {
  constexpr real_t beta = 4.125;
  for (const index_t size : {2, 4, 6}) {
    for (uint64_t sample = 0; sample < 3; ++sample) {
      auto gauge = make_random_valid(size, size, 1,
                                     65536 + 16 * size + sample);
      const auto spins = SlabHolonomy<2>(gauge);
      const real_t slab = DirichletSlabWilsonAction<2>(gauge, beta);
      const real_t constant =
          FixedBoundarySpatialActionConstant<2>(gauge, beta);
      const real_t pcm = PCMAction<2>(spins, beta);
      check(close(slab - constant, pcm, 8.0e-12),
            "Nt=1 Yang-Mills action equals PCM action after fixed constant");

      const auto correlator = MeasureDirichletHolonomy<2>(gauge);
      const auto loops = MeasureBoundaryWilsonLoops<2>(gauge, false,
                                                        size / 2);
      check(correlator.correlator.front().chiral_correlator == 1.0,
            "C_G(0) is exactly one configuration by configuration");
      for (const auto &point : correlator.correlator) {
        const real_t direct =
            direct_spin_correlator(spins, point.separation);
        check(close(point.chiral_correlator, direct),
              "Nt=1 holonomy correlator equals PCM spin correlator");
        const auto loop = std::find_if(
            loops.loops.begin(), loops.loops.end(), [&](const auto &entry) {
              return static_cast<index_t>(entry[0]) == point.separation &&
                     static_cast<index_t>(entry[1]) == 1;
            });
        check(loop != loops.loops.end() &&
                  close((*loop)[2], point.chiral_correlator, 8.0e-12),
              "Nt=1 boundary Wilson loop equals C_G configuration-wise");
      }
    }
  }
}

void check_full_height_loop_and_periodic_regression() {
  auto gauge = make_random_valid(4, 4, 3, 131072);
  const auto correlator = MeasureDirichletHolonomy<2>(gauge);
  const auto loops = MeasureBoundaryWilsonLoops<2>(gauge, false, 2);
  for (const auto &point : correlator.correlator) {
    const auto loop = std::find_if(
        loops.loops.begin(), loops.loops.end(), [&](const auto &entry) {
          return static_cast<index_t>(entry[0]) == point.separation &&
                 static_cast<index_t>(entry[1]) == 3;
        });
    check(loop != loops.loops.end() &&
              close((*loop)[2], point.chiral_correlator, 8.0e-12),
          "full-height boundary Wilson loop equals C_G");
  }

  auto quality_cold = make_identity(9, 4, 4);
  apply_temporal_dirichlet_boundaries<3, 2>(quality_cold);
  const auto horizontal =
      MeasureDirichletBulkPolyakovCorrelator<2>(quality_cold, 4);
  check(horizontal.correlator.size() == 3,
        "bulk horizontal Polyakov correlator keeps symmetric even R");
  for (const auto &point : horizontal.correlator) {
    check(point[1] == 4.0,
          "cold horizontal double-trace Polyakov correlator equals four");
  }

  // Disabled mode retains the original periodic geometry and initialization.
  auto periodic_cold = make_identity_gauge_field<3, 2>(4, 4, 4, 1);
  check(GaugePlaquette<3, 2>(periodic_cold) == 1.0,
        "periodic cold plaquette regression");
  check(PeriodicWilsonAction<2>(periodic_cold, 1.5) ==
            -1.5 * 4.0 * 4.0 * 4.0 * 3.0,
        "periodic cold action regression");
  MetropolisParams defaults;
  HeatbathParams heatbath_defaults;
  check(!defaults.temporal_dirichlet && !heatbath_defaults.temporal_dirichlet,
        "Dirichlet mode is disabled by default");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::printf("usage: dirichlet_test PARTITION_TABLE\n");
    return 2;
  }
  Kokkos::initialize(argc, argv);
  {
    check_initialization_and_io(argv[1]);
    check_odd_spatial_coloring();
    check_updates(argv[1]);
    check_local_action_differences();
    check_symmetries_and_factorization();
    check_exact_nt1_mapping();
    check_full_height_loop_and_periodic_regression();
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("All temporal Dirichlet deterministic checks passed.\n");
  }
  return failures == 0 ? 0 : 1;
}
