#include "io/orbifold_configuration.hpp"
#include "orbifold.hpp"
#include "observables/plaquette.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <string>

namespace {

using namespace klft;

int failures = 0;

void check(const bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

bool close(const real_t left, const real_t right,
           const real_t relative_tolerance = 1.0e-10,
           const real_t absolute_tolerance = 1.0e-12) {
  return std::abs(left - right) <=
         absolute_tolerance +
             relative_tolerance * std::max(std::abs(left), std::abs(right));
}

OrbifoldDimensions test_dimensions(const index_t l0, const index_t l1,
                                    const index_t l2, const index_t lt) {
  OrbifoldDimensions result{};
  result[0] = l0;
  result[1] = l1;
  result[orbifold_time_direction] = lt;
  if constexpr (compiled_rank == 4) {
    result[2] = l2;
  }
  return result;
}

OrbifoldDimensions test_site(const index_t x, const index_t y,
                             const index_t z, const index_t t) {
  return test_dimensions(x, y, z, t);
}

real_t site_value(const OrbifoldDimensions &site) {
  constexpr std::array<real_t, 4> weights{1.0, 2.0, 3.0, 5.0};
  real_t result = 1.0;
  for (size_t d = 0; d < compiled_rank; ++d) {
    result += weights[d] * site[d];
  }
  return result;
}

real_t matrix_distance_squared(const OrbifoldMatrix &left, const OrbifoldMatrix &right) {
  return orbifold_matrix_norm_squared(left - right);
}

real_t field_distance(const OrbifoldField &left, const OrbifoldField &right) {
  const auto lz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       left.spatial);
  const auto rz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       right.spatial);
  const auto lu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       left.temporal);
  const auto ru = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       right.temporal);
  real_t maximum = 0.0;
  const size_t sites = wilson_site_count<compiled_rank>(left.dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, left.dimensions);
    maximum = std::max(
        maximum,
        std::sqrt(matrix_distance_squared(orbifold_temporal_ref(lu, site),
                                          orbifold_temporal_ref(ru, site))));
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      maximum = std::max(
          maximum,
          std::sqrt(matrix_distance_squared(
              orbifold_spatial_ref(lz, site, j),
              orbifold_spatial_ref(rz, site, j))));
    }
  }
  return maximum;
}

real_t force_norm(const OrbifoldField &force) {
  const auto z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      force.spatial);
  const auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                      force.temporal);
  real_t result = 0.0;
  const size_t sites = wilson_site_count<compiled_rank>(force.dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, force.dimensions);
    result += orbifold_matrix_norm_squared(orbifold_temporal_ref(u, site));
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      result +=
          orbifold_matrix_norm_squared(orbifold_spatial_ref(z, site, j));
    }
  }
  return std::sqrt(result);
}

OrbifoldMatrix diagonal_gauge_matrix(const real_t angle) {
  OrbifoldMatrix result = orbifold_identity();
  matrix_ref(result, 0, 0) =
      complex_t(std::cos(angle), std::sin(angle));
  if constexpr (compiled_nc > 1) {
    matrix_ref(result, 1, 1) = complex_t(std::cos(angle), -std::sin(angle));
  }
  return result;
}

OrbifoldMatrix local_gauge_matrix(const real_t scale) {
  Kokkos::Array<real_t, orbifold_algebra_dimensions> coefficients{};
  coefficients[0] = 0.07 * scale;
  if constexpr (compiled_nc == 3) {
    coefficients[4] = -0.04 * scale;
    coefficients[7] = 0.03 * scale;
  } else if constexpr (compiled_nc == 2) {
    coefficients[2] = 0.03 * scale;
  }
  return orbifold_exp_group(orbifold_algebra(coefficients));
}

void check_polar_projection() {
  OrbifoldMatrix positive = orbifold_identity();
  matrix_ref(positive, 0, 0) = 1.2;
  if constexpr (compiled_nc >= 2) {
    matrix_ref(positive, 1, 1) = 0.9;
    matrix_ref(positive, 0, 1) = 0.15;
    matrix_ref(positive, 1, 0) = 0.15;
  }
  if constexpr (compiled_nc == 3) {
    matrix_ref(positive, 2, 2) = 1.1;
  }
  const OrbifoldMatrix unitary = local_gauge_matrix(1.3);
  bool converged = false;
  const OrbifoldMatrix projected =
      orbifold_polar_unitary(positive * unitary, converged);
  check(converged && matrix_distance_squared(projected, unitary) < 1.0e-24,
        "orbifold polar projection recovers a known unitary factor");
}

void check_algebra_normalization() {
  for (index_t a = 0; a < orbifold_algebra_dimensions; ++a) {
    Kokkos::Array<real_t, orbifold_algebra_dimensions> ca{};
    ca[a] = 1.0;
    const auto ta = orbifold_algebra(ca);
    check(orbifold_matrix_norm_squared(ta + conj(ta)) < 1.0e-28,
          "momentum generators are antihermitian");
    for (index_t b = 0; b < orbifold_algebra_dimensions; ++b) {
      Kokkos::Array<real_t, orbifold_algebra_dimensions> cb{};
      cb[b] = 1.0;
      check(close(-2.0 * trace(ta * orbifold_algebra(cb)).real(), a == b ? 1.0 : 0.0),
            "momentum generators have the Gaussian kinetic normalization");
    }
  }
  if constexpr (compiled_nc == 1) {
    OrbifoldActionParams invalid;
    invalid.u1_mass = 1.0;
    bool rejected = false;
    try { invalid.validate(); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "U(1) rejects gauge-breaking determinant pinning");
  }
}

void check_polar_wilson_phase() {
  const auto dimensions = test_dimensions(4, 4, 4, 4);
  OrbifoldField field(dimensions);
  auto z = Kokkos::create_mirror_view(field.spatial);
  constexpr std::array<real_t, 4> angles{0.0, 0.2, -0.3, 0.1};
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site = wilson_linear_to_site<compiled_rank>(linear, dimensions);
    const real_t angle = angles[site[orbifold_time_direction]];
    const auto value = orbifold_matrix_scale(orbifold_identity(),
        0.7 * complex_t(std::cos(angle), std::sin(angle)));
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      orbifold_spatial_ref(z, site, j) = value;
    }
  }
  Kokkos::deep_copy(field.spatial, z);
  const auto projected = orbifold_projected_gauge_field(field);
  const auto loops = orbifold_wilson_loops(field, 2, 2);
  for (const auto &loop : loops) {
    const index_t r = static_cast<index_t>(loop[0]);
    const index_t t = static_cast<index_t>(loop[1]);
    real_t expected = 0.0;
    for (size_t time = 0; time < angles.size(); ++time) {
      expected += std::cos(r * (angles[time] - angles[(time + t) % angles.size()])) / 4.0;
    }
    check(close(loop[2], expected, 2.0e-12),
          "polar Wilson loops retain the U(Nc) determinant phase");
    real_t direct = 0.0;
    Kokkos::parallel_reduce("direct_polar_loop", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear, real_t &sum) {
          const auto site = wilson_linear_to_site<compiled_rank>(linear, dimensions);
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            sum += WilsonLoopRawAtSite<compiled_rank, compiled_nc>(
                projected, site, j, orbifold_time_direction, r, t, dimensions,
                orbifold_identity()).real();
          }
        }, direct);
    check(close(loop[2], direct / (sites * orbifold_spatial_directions * orbifold_colors),
                2.0e-12), "fused polar loops agree with direct rectangular products");
  }
}

OrbifoldField deterministic_field(const OrbifoldDimensions &dimensions,
                                   const OrbifoldActionParams &params,
                                   const char *label) {
  const real_t vacuum = std::sqrt(params.vacuum_scale_squared());
  OrbifoldField field(dimensions, orbifold_identity() * vacuum,
                       orbifold_identity(), label);
  auto z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                field.spatial);
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                field.temporal);
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    const real_t value_at_site = site_value(site);
    Kokkos::Array<real_t, orbifold_algebra_dimensions> coefficients{};
    coefficients[linear % orbifold_algebra_dimensions] = 0.025 * value_at_site;
    orbifold_temporal_ref(u, site) =
        orbifold_exp_group(orbifold_algebra(coefficients));
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      OrbifoldMatrix value = orbifold_identity() * vacuum;
      for (index_t row = 0; row < orbifold_colors; ++row) {
        for (index_t col = 0; col < orbifold_colors; ++col) {
          const real_t component =
              value_at_site + 7.0 * j + 3.0 * row + col;
          matrix_ref(value, row, col) +=
              complex_t(0.002 * component,
                        0.001 * (component + row - col));
        }
      }
      orbifold_spatial_ref(z, site, j) = value;
    }
  }
  Kokkos::deep_copy(field.spatial, z);
  Kokkos::deep_copy(field.temporal, u);
  Kokkos::fence();
  return field;
}

void perturb_spatial(OrbifoldField &field, const OrbifoldDimensions &site,
                     const index_t j, const index_t row, const index_t col,
                     const complex_t change) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   field.spatial);
  matrix_ref(orbifold_spatial_ref(host, site, j), row, col) += change;
  Kokkos::deep_copy(field.spatial, host);
  Kokkos::fence();
}

void perturb_temporal(OrbifoldField &field, const OrbifoldDimensions &site,
                      const OrbifoldMatrix &generator, const real_t step) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   field.temporal);
  orbifold_temporal_ref(host, site) =
      orbifold_exp_group(generator * step) * orbifold_temporal_ref(host, site);
  Kokkos::deep_copy(field.temporal, host);
  Kokkos::fence();
}

void check_vacuum_and_temporal_normalization() {
  const auto dimensions = test_dimensions(2, 2, 2, 4);
  OrbifoldActionParams params;
  params.spatial_spacing = 0.7;
  params.temporal_spacing = 0.2;
  params.coupling = 1.3;
  params.scalar_mass = 0.4;
  params.u1_mass = compiled_nc == 1 ? 0.0 : 0.6;
  const OrbifoldMatrix vacuum =
      orbifold_identity() * std::sqrt(params.vacuum_scale_squared());
  OrbifoldField cold(dimensions, vacuum, orbifold_identity(), "orbifold_cold");
  OrbifoldField force(dimensions, orbifold_zero(), orbifold_zero(),
                       "orbifold_cold_force");
  check(std::abs(orbifold_action(cold, params)) < 1.0e-25,
        "orbifold vacuum action vanishes");
  orbifold_force(cold, params, force);
  check(force_norm(force) < 1.0e-12, "orbifold vacuum force vanishes");
  const auto cold_loops = orbifold_wilson_loops(cold, 1, 1);
  check(cold_loops.size() == 1 && close(cold_loops[0][2], 1.0),
        "orbifold vacuum Wilson loop equals one");

  OrbifoldActionParams temporal_params = params;
  temporal_params.scalar_mass = 0.0;
  temporal_params.u1_mass = 0.0;
  OrbifoldField temporal_field(dimensions, orbifold_zero(), orbifold_identity(),
                                "orbifold_temporal_normalization");
  auto z = Kokkos::create_mirror_view(temporal_field.spatial);
  const std::array<real_t, 4> q{0.7, 1.1, 0.9, 1.4};
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      orbifold_spatial_ref(z, site, j) =
          orbifold_identity() * q[site[orbifold_time_direction]];
    }
  }
  Kokkos::deep_copy(temporal_field.spatial, z);
  real_t time_difference = 0.0;
  const index_t nt = dimensions[orbifold_time_direction];
  for (index_t t = 0; t < nt; ++t) {
    const real_t delta = q[(t + 1) % nt] - q[t];
    time_difference += delta * delta;
  }
  const real_t expected =
      static_cast<real_t>(sites / nt) * orbifold_spatial_directions * orbifold_colors *
      time_difference / temporal_params.temporal_spacing;
  check(close(orbifold_action(temporal_field, temporal_params), expected,
              2.0e-13),
        "temporal term matches the notebook normalization");

  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                cold.temporal);
  const OrbifoldMatrix holonomy = diagonal_gauge_matrix(0.4);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    if (site[orbifold_time_direction] == 0) {
      orbifold_temporal_ref(u, site) = holonomy;
    }
  }
  Kokkos::deep_copy(cold.temporal, u);
  check(std::abs(orbifold_action(cold, params)) < 1.0e-24,
        "commuting nontrivial temporal holonomy has zero vacuum action");
  check(std::abs(trace(holonomy).real() / orbifold_colors - 1.0) > 1.0e-3,
        "full action retains a nontrivial Polyakov holonomy");
}

void check_spatial_normalization() {
  const auto dimensions = test_dimensions(3, 2, 2, 2);
  OrbifoldActionParams params;
  params.spatial_spacing = 0.7;
  params.temporal_spacing = 0.3;
  params.coupling = 1.2;
  const real_t as = params.spatial_spacing;
  const real_t at = params.temporal_spacing;
  const real_t g2 = params.coupling * params.coupling;
  const real_t spatial_volume =
      std::pow(as, static_cast<real_t>(orbifold_spatial_directions));
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  const real_t volume = static_cast<real_t>(sites);

  OrbifoldField d_field(dimensions, orbifold_zero(), orbifold_identity(),
                         "orbifold_d_normalization");
  auto d_host = Kokkos::create_mirror_view(d_field.spatial);
  const std::array<real_t, 3> q{0.4, 0.8, 1.1};
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    orbifold_spatial_ref(d_host, site, 0) = orbifold_identity() * q[site[0]];
    for (index_t j = 1; j < orbifold_spatial_directions; ++j) {
      orbifold_spatial_ref(d_host, site, j) = orbifold_zero();
    }
  }
  Kokkos::deep_copy(d_field.spatial, d_host);
  real_t d_sum = 0.0;
  for (index_t x = 0; x < dimensions[0]; ++x) {
    const real_t difference =
        q[x] * q[x] - q[(x + dimensions[0] - 1) % dimensions[0]] *
                            q[(x + dimensions[0] - 1) % dimensions[0]];
    d_sum += difference * difference;
  }
  const real_t d_expected =
      at * g2 / (2.0 * spatial_volume) * orbifold_colors *
      (volume / dimensions[0]) * d_sum;
  check(close(orbifold_action(d_field, params), d_expected, 2.0e-13),
        "D-term normalization matches an analytic scalar-link case");

  OrbifoldField f_field(dimensions, orbifold_zero(), orbifold_identity(),
                         "orbifold_f_normalization");
  auto f_host = Kokkos::create_mirror_view(f_field.spatial);
  OrbifoldMatrix z0 = orbifold_zero();
  OrbifoldMatrix z1 = orbifold_zero();
  constexpr real_t a = 0.6;
  constexpr real_t b = 0.9;
  if constexpr (compiled_nc >= 2) {
    matrix_ref(z0, 0, 1) = a;
    matrix_ref(z0, 1, 0) = a;
    matrix_ref(z1, 0, 1) = complex_t(0.0, -b);
    matrix_ref(z1, 1, 0) = complex_t(0.0, b);
  } else {
    z0 = orbifold_identity() * a;
  }
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    orbifold_spatial_ref(f_host, site, 0) = z0;
    if constexpr (orbifold_spatial_directions >= 2) {
      orbifold_spatial_ref(f_host, site, 1) = compiled_nc == 1 ?
          orbifold_identity() * q[site[0]] : z1;
    }
    for (index_t j = 2; j < orbifold_spatial_directions; ++j) {
      orbifold_spatial_ref(f_host, site, j) = orbifold_zero();
    }
  }
  Kokkos::deep_copy(f_field.spatial, f_host);
  real_t f_norm = volume * 8.0 * a * a * b * b;
  if constexpr (compiled_nc == 1) {
    f_norm = 0.0;
    for (size_t x = 0; x < q.size(); ++x) {
      const real_t delta = q[(x + 1) % q.size()] - q[x];
      f_norm += volume / q.size() * a * a * delta * delta;
    }
  }
  const real_t f_expected = orbifold_spatial_directions >= 2 ?
      at * 2.0 * g2 / spatial_volume * f_norm : 0.0;
  check(close(orbifold_action(f_field, params), f_expected, 2.0e-13),
        "F-term normalization matches an analytic commutator case");

  params.scalar_mass = 0.5;
  params.u1_mass = compiled_nc == 1 ? 0.0 : 0.4;
  constexpr real_t uniform_q = 0.75;
  OrbifoldField potential_field(dimensions,
                                 orbifold_identity() * uniform_q,
                                 orbifold_identity(),
                                 "orbifold_potential_normalization");
  const real_t c = params.vacuum_scale_squared();
  const real_t factor_mass =
      params.scalar_mass * params.scalar_mass * g2 /
      (2.0 * std::pow(as, orbifold_spatial_directions - 2));
  const real_t factor_det = params.u1_mass * params.u1_mass * c;
  const real_t radial = uniform_q * uniform_q - c;
  const real_t determinant =
      std::pow(uniform_q / std::sqrt(c), orbifold_colors) - 1.0;
  const real_t potential_expected =
      at * volume * orbifold_spatial_directions *
      (factor_mass * orbifold_colors * radial * radial +
       factor_det * determinant * determinant);
  check(close(orbifold_action(potential_field, params), potential_expected,
              2.0e-13),
        "radial and determinant normalizations match a uniform-link case");
}

void check_gauge_invariance(const OrbifoldActionParams &params) {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  const auto source = deterministic_field(dimensions, params, "orbifold_gauge");
  OrbifoldField transformed(dimensions, orbifold_zero(), orbifold_identity(),
                             "orbifold_gauge_transformed");
  const auto input_z = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), source.spatial);
  const auto input_u = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), source.temporal);
  auto output_z = Kokkos::create_mirror_view(transformed.spatial);
  auto output_u = Kokkos::create_mirror_view(transformed.temporal);
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  for (size_t linear = 0; linear < sites; ++linear) {
    const auto site =
        wilson_linear_to_site<compiled_rank>(linear, dimensions);
    const OrbifoldMatrix here = local_gauge_matrix(site_value(site));
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      const auto plus_j = shift_index_plus(site, j, 1, dimensions);
      const OrbifoldMatrix there = local_gauge_matrix(site_value(plus_j));
      orbifold_spatial_ref(output_z, site, j) =
          here * orbifold_spatial_ref(input_z, site, j) * conj(there);
    }
    const auto plus_t =
        shift_index_plus(site, orbifold_time_direction, 1, dimensions);
    const OrbifoldMatrix later = local_gauge_matrix(site_value(plus_t));
    orbifold_temporal_ref(output_u, site) =
        here * orbifold_temporal_ref(input_u, site) * conj(later);
  }
  Kokkos::deep_copy(transformed.spatial, output_z);
  Kokkos::deep_copy(transformed.temporal, output_u);
  check(close(orbifold_action(source, params),
              orbifold_action(transformed, params), 2.0e-12),
        "full orbifold action is gauge invariant");
  const auto source_loops = orbifold_wilson_loops(source, 1, 1);
  const auto transformed_loops = orbifold_wilson_loops(transformed, 1, 1);
  check(source_loops.size() == transformed_loops.size() &&
            close(source_loops[0][2], transformed_loops[0][2], 2.0e-11),
        "orbifold Wilson loop is gauge invariant");
}

void check_forces(const OrbifoldActionParams &params) {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  const auto field = deterministic_field(dimensions, params, "orbifold_fd");
  OrbifoldField force(dimensions, orbifold_zero(), orbifold_zero(),
                       "orbifold_fd_force");
  orbifold_force(field, params, force);
  const auto force_z = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), force.spatial);
  const auto force_u = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), force.temporal);
  const auto site = test_site(0, 1, 0, 1);
  constexpr index_t j = orbifold_spatial_directions - 1;
  constexpr index_t row = 0;
  constexpr index_t col = orbifold_colors - 1;
  constexpr real_t h = 2.0e-6;

  auto plus = copy_orbifold_field(field, "orbifold_fd_plus_real");
  auto minus = copy_orbifold_field(field, "orbifold_fd_minus_real");
  perturb_spatial(plus, site, j, row, col, complex_t(h, 0.0));
  perturb_spatial(minus, site, j, row, col, complex_t(-h, 0.0));
  const real_t real_fd =
      (orbifold_action(plus, params) - orbifold_action(minus, params)) /
      (2.0 * h);
  check(close(real_fd,
              matrix_ref(orbifold_spatial_ref(force_z, site, j), row, col)
                  .real(),
              3.0e-6, 3.0e-8),
        "spatial real force matches a central difference");

  plus = copy_orbifold_field(field, "orbifold_fd_plus_imag");
  minus = copy_orbifold_field(field, "orbifold_fd_minus_imag");
  perturb_spatial(plus, site, j, row, col, complex_t(0.0, h));
  perturb_spatial(minus, site, j, row, col, complex_t(0.0, -h));
  const real_t imag_fd =
      (orbifold_action(plus, params) - orbifold_action(minus, params)) /
      (2.0 * h);
  check(close(imag_fd,
              matrix_ref(orbifold_spatial_ref(force_z, site, j), row, col)
                  .imag(),
              3.0e-6, 3.0e-8),
        "spatial imaginary force matches a central difference");

  Kokkos::Array<real_t, orbifold_algebra_dimensions> coefficients{};
  coefficients[0] = 0.7;
  coefficients[orbifold_algebra_dimensions - 1] -= 0.2;
  const OrbifoldMatrix generator = orbifold_algebra(coefficients);
  plus = copy_orbifold_field(field, "orbifold_fd_plus_group");
  minus = copy_orbifold_field(field, "orbifold_fd_minus_group");
  perturb_temporal(plus, site, generator, h);
  perturb_temporal(minus, site, generator, -h);
  const real_t group_fd =
      (orbifold_action(plus, params) - orbifold_action(minus, params)) /
      (2.0 * h);
  const real_t group_expected =
      2.0 *
      trace(orbifold_temporal_ref(force_u, site) * generator).real();
  if (!close(group_fd, group_expected, 3.0e-6, 3.0e-8)) {
    std::printf("group force: finite difference %.16e, analytic %.16e\n",
                group_fd, group_expected);
  }
  check(close(group_fd, group_expected, 3.0e-6, 3.0e-8),
        "temporal force matches a group central difference");
}

void check_hmc(const OrbifoldActionParams &params) {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  auto field = deterministic_field(dimensions, params, "orbifold_hmc_field");
  const auto initial = copy_orbifold_field(field, "orbifold_hmc_initial");
  OrbifoldHMCParams hmc_params;
  hmc_params.step_size = 2.0e-4;
  hmc_params.steps = 4;
  OrbifoldHMC hmc(field, params, hmc_params, 240827);
  hmc.randomize_momenta();
  const real_t initial_hamiltonian =
      orbifold_action(field, params) + hmc.kinetic_energy();
  hmc.integrate();
  const real_t integration_error =
      orbifold_action(field, params) + hmc.kinetic_energy() -
      initial_hamiltonian;
  auto fine_field = copy_orbifold_field(initial, "orbifold_hmc_fine");
  OrbifoldHMCParams fine_params = hmc_params;
  fine_params.step_size *= 0.5;
  fine_params.steps *= 2;
  OrbifoldHMC fine_hmc(fine_field, params, fine_params, 240827);
  fine_hmc.randomize_momenta();
  const real_t fine_initial =
      orbifold_action(fine_field, params) + fine_hmc.kinetic_energy();
  fine_hmc.integrate();
  const real_t fine_error =
      orbifold_action(fine_field, params) + fine_hmc.kinetic_energy() -
      fine_initial;
  if (!(std::abs(fine_error) < 0.35 * std::abs(integration_error))) {
    std::printf("leapfrog Delta H: coarse %.16e, fine %.16e\n",
                integration_error, fine_error);
  }
  check(std::abs(fine_error) < 0.35 * std::abs(integration_error),
        "orbifold leapfrog Delta H converges quadratically");
  hmc.negate_momenta();
  hmc.integrate();
  check(field_distance(field, initial) < 2.0e-10,
        "orbifold leapfrog is reversible");
  auto errors = orbifold_temporal_group_errors(field);
  check(errors.unitarity < 2.0e-11 && errors.determinant < 2.0e-11,
        "temporal links remain in the gauge group under leapfrog");

  const OrbifoldHMCResult result = hmc.step();
  check(std::isfinite(result.initial_hamiltonian) &&
            std::isfinite(result.final_hamiltonian) &&
            std::isfinite(result.delta_hamiltonian),
        "one orbifold HMC step has finite Hamiltonians");
  errors = orbifold_temporal_group_errors(field);
  check(errors.unitarity < 2.0e-10 && errors.determinant < 2.0e-10,
        "one orbifold HMC step preserves the gauge group");
}

void check_hot_start(const OrbifoldActionParams &params) {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  const OrbifoldMatrix vacuum =
      orbifold_identity() * std::sqrt(params.vacuum_scale_squared());
  OrbifoldField cold(dimensions, vacuum, orbifold_identity(),
                      "orbifold_hot_reference");
  auto hot = copy_orbifold_field(cold, "orbifold_hot_field");
  Kokkos::Random_XorShift64_Pool<> rng(240831);
  initialize_hot_orbifold_field(hot, params, 0.01, rng);

  check(field_distance(hot, cold) > 0.1,
        "orbifold hot start differs from the cold vacuum");
  const auto errors = orbifold_temporal_group_errors(hot);
  check(errors.unitarity < 1.0e-12 && errors.determinant < 1.0e-12,
        "orbifold hot start has group-valued temporal links");
  check(std::isfinite(orbifold_action(hot, params)),
        "orbifold hot start has finite action");
  const auto loops = orbifold_wilson_loops(hot, 1, 1);
  check(loops.size() == 1 && std::isfinite(loops[0][2]),
        "orbifold hot start has a finite Wilson loop");
}

void check_configuration_io(const OrbifoldActionParams &params) {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  const auto source =
      deterministic_field(dimensions, params, "orbifold_checkpoint_source");
  const std::string filename = "klft_orbifold_test_" +
                               std::to_string(compiled_rank) + "_" +
                               std::to_string(compiled_nc) + ".cfg";
  std::remove(filename.c_str());
  check(save_orbifold_configuration_atomic(filename, source, params),
        "save orbifold checkpoint atomically");

  OrbifoldField restored(dimensions, orbifold_zero(), orbifold_identity(),
                          "orbifold_checkpoint_restored");
  check(load_orbifold_configuration(filename, restored, params),
        "load orbifold checkpoint");
  check(field_distance(source, restored) == 0.0,
        "orbifold checkpoint preserves every matrix exactly");
  check(orbifold_action(source, params) == orbifold_action(restored, params),
        "orbifold checkpoint preserves the action exactly");
  const auto source_loop = orbifold_wilson_loops(source, 1, 1);
  const auto restored_loop = orbifold_wilson_loops(restored, 1, 1);
  // Fields above must round-trip bitwise. Fused Wilson loops use atomic
  // summation, whose order can change between calls even on the same field.
  const real_t reduction_roundoff = 128.0 * std::numeric_limits<real_t>::epsilon();
  std::printf("Checkpoint Wilson-loop reduction delta %.3e\n",
              source_loop[0][2] - restored_loop[0][2]);
  check(close(source_loop[0][2], restored_loop[0][2],
              reduction_roundoff, reduction_roundoff),
        "orbifold checkpoint preserves Wilson loops within reduction roundoff");

  auto wrong_params = params;
  wrong_params.scalar_mass += 1.0;
  check(!load_orbifold_configuration(filename, restored, wrong_params),
        "orbifold checkpoint rejects mismatched action parameters");

  // Version 1 had no rank field and was exclusively 4D SU(3).
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes{std::istreambuf_iterator<char>(input),
                          std::istreambuf_iterator<char>()};
  input.close();
  if (bytes.size() < 24) {
    check(false, "checkpoint is too short for its fixed header");
    return;
  }
  const std::string truncated_filename = filename + ".truncated";
  {
    std::ofstream truncated(truncated_filename, std::ios::binary);
    truncated.write(bytes.data(), bytes.size() - 1);
  }
  OrbifoldField untouched(dimensions, orbifold_zero(), orbifold_identity(),
                           "checkpoint_failed_load");
  const auto before_failure = copy_orbifold_field(untouched, "before_failed_load");
  check(!load_orbifold_configuration(truncated_filename, untouched, params),
        "orbifold checkpoint rejects truncated matrix data");
  check(field_distance(untouched, before_failure) == 0.0,
        "failed checkpoint load leaves the destination field unchanged");
  std::remove(truncated_filename.c_str());
  auto v2_bytes = bytes;
  const std::uint32_t version2 = 2;
  std::memcpy(v2_bytes.data() + 8, &version2, sizeof(version2));
  v2_bytes.erase(v2_bytes.begin() + 20, v2_bytes.begin() + 24);
  const std::string v2_filename = filename + ".v2";
  {
    std::ofstream v2(v2_filename, std::ios::binary);
    v2.write(v2_bytes.data(), v2_bytes.size());
  }
  check(load_orbifold_configuration(v2_filename, restored, params) == (compiled_nc == 3),
        "version-2 checkpoints remain readable only as SU(3)");
  std::remove(v2_filename.c_str());
  const std::uint32_t legacy_version = 1;
  std::memcpy(bytes.data() + 8, &legacy_version, sizeof(legacy_version));
  bytes.erase(bytes.begin() + 16, bytes.begin() + 24);
  const std::string legacy_filename = filename + ".legacy";
  {
    std::ofstream legacy(legacy_filename, std::ios::binary);
    legacy.write(bytes.data(), bytes.size());
  }
  check(load_orbifold_configuration(legacy_filename, restored, params) ==
            (compiled_rank == 4 && compiled_nc == 3),
        "legacy checkpoint is accepted only in its original 4D theory");
  std::remove(legacy_filename.c_str());

  {
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::binary);
    const std::uint32_t wrong_rank = compiled_rank == 3 ? 4 : 3;
    file.seekp(16);
    file.write(reinterpret_cast<const char *>(&wrong_rank), sizeof(wrong_rank));
  }
  check(!load_orbifold_configuration(filename, restored, params),
        "orbifold checkpoint rejects mismatched spacetime rank");
  {
    std::fstream file(filename, std::ios::in | std::ios::out | std::ios::binary);
    const std::uint32_t rank = compiled_rank;
    const std::uint32_t wrong_nc = compiled_nc == 3 ? 2 : 3;
    file.seekp(16);
    file.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
    file.write(reinterpret_cast<const char *>(&wrong_nc), sizeof(wrong_nc));
  }
  check(!load_orbifold_configuration(filename, restored, params),
        "orbifold checkpoint rejects mismatched gauge group");
  std::remove(filename.c_str());
}

void check_constrained_wilson_limit() {
  const auto dimensions = test_dimensions(2, 2, 2, 2);
  Kokkos::Random_XorShift64_Pool<> rng(310831);
  auto gauge = make_hot_gauge_field<compiled_rank, compiled_nc>(2, 2, 2, 2, rng);
  OrbifoldActionParams params;
  params.spatial_spacing = 0.3;
  params.temporal_spacing = 0.2;
  params.coupling = 1.2;
  params.scalar_mass = 100.0;
  params.u1_mass = compiled_nc == 1 ? 0.0 : 100.0;
  OrbifoldField field(dimensions, orbifold_zero(), orbifold_identity(),
                       "orbifold_wilson_limit");
  initialize_orbifold_from_gauge(field, gauge, params);

  const auto plaquettes = GaugePlaquettes<compiled_rank, compiled_nc>(gauge, false);
  const real_t sites = wilson_site_count<compiled_rank>(dimensions);
  constexpr real_t d = orbifold_spatial_directions;
  const real_t g2 = params.coupling * params.coupling;
  // Substitute Z = sqrt(a_s^(d-2)/(2g^2)) U into the full action.
  // Each squared plaquette difference is 2(Nc - ReTr U_p).
  const real_t expected =
      params.temporal_spacing * std::pow(params.spatial_spacing, d - 4) / g2 *
          (orbifold_colors * sites * d * (d - 1) / 2 - plaquettes.spatial) +
      std::pow(params.spatial_spacing, d - 2) /
          (g2 * params.temporal_spacing) *
          (orbifold_colors * sites * d - plaquettes.temporal);
  check(close(orbifold_action(field, params), expected, 2.0e-12, 2.0e-11),
        "constrained orbifold action equals anisotropic Wilson action");
  std::vector<Kokkos::Array<index_t, 2>> pairs{{1, 1}};
  std::vector<Kokkos::Array<real_t, 3>> compact_loops;
  WilsonLoop_temporal_raw_fused<compiled_rank, compiled_nc>(gauge, pairs, compact_loops);
  const auto orbifold_loops = orbifold_wilson_loops(field, 1, 1);
  check(close(orbifold_loops[0][2], compact_loops[0][2], 2.0e-12),
        "constrained polar Wilson loops agree with the compact implementation");
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  try {
    OrbifoldActionParams params;
    params.spatial_spacing = 0.8;
    params.temporal_spacing = 0.25;
    params.coupling = 1.1;
    params.scalar_mass = 0.35;
    params.u1_mass = compiled_nc == 1 ? 0.0 : 0.45;
    check_algebra_normalization();
    check_polar_projection();
    check_polar_wilson_phase();
    check_vacuum_and_temporal_normalization();
    check_spatial_normalization();
    check_gauge_invariance(params);
    check_forces(params);
    check_hot_start(params);
    check_configuration_io(params);
    check_constrained_wilson_limit();
    check_hmc(params);
  } catch (const std::exception &error) {
    std::printf("FAIL: unexpected exception: %s\n", error.what());
    ++failures;
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("orbifold checks passed\n");
  }
  return failures == 0 ? 0 : 1;
}
