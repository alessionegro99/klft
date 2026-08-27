#include "orbifold.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>

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

real_t matrix_distance_squared(const SUN<3> &left, const SUN<3> &right) {
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
  for (index_t x = 0; x < left.dimensions[0]; ++x) {
    for (index_t y = 0; y < left.dimensions[1]; ++y) {
      for (index_t z = 0; z < left.dimensions[2]; ++z) {
        for (index_t t = 0; t < left.dimensions[3]; ++t) {
          maximum = std::max(
              maximum,
              std::sqrt(matrix_distance_squared(lu(x, y, z, t),
                                                ru(x, y, z, t))));
          for (index_t j = 0; j < 3; ++j) {
            maximum = std::max(
                maximum,
                std::sqrt(matrix_distance_squared(lz(x, y, z, t, j),
                                                  rz(x, y, z, t, j))));
          }
        }
      }
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
  for (index_t x = 0; x < force.dimensions[0]; ++x) {
    for (index_t y = 0; y < force.dimensions[1]; ++y) {
      for (index_t zz = 0; zz < force.dimensions[2]; ++zz) {
        for (index_t t = 0; t < force.dimensions[3]; ++t) {
          result += orbifold_matrix_norm_squared(u(x, y, zz, t));
          for (index_t j = 0; j < 3; ++j) {
            result += orbifold_matrix_norm_squared(z(x, y, zz, t, j));
          }
        }
      }
    }
  }
  return std::sqrt(result);
}

SUN<3> diagonal_gauge_matrix(const real_t angle) {
  SUN<3> result = identitySUN<3>();
  matrix_ref(result, 0, 0) =
      complex_t(std::cos(angle), std::sin(angle));
  matrix_ref(result, 1, 1) =
      complex_t(std::cos(angle), -std::sin(angle));
  return result;
}

OrbifoldField deterministic_field(const IndexArray<4> &dimensions,
                                   const OrbifoldActionParams &params,
                                   const char *label) {
  const real_t vacuum = std::sqrt(params.vacuum_scale_squared());
  OrbifoldField field(dimensions, identitySUN<3>() * vacuum,
                       identitySUN<3>(), label);
  auto z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                field.spatial);
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                field.temporal);
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t zz = 0; zz < dimensions[2]; ++zz) {
        for (index_t t = 0; t < dimensions[3]; ++t) {
          const real_t site_value =
              1.0 + x + 2.0 * y + 3.0 * zz + 5.0 * t;
          Kokkos::Array<real_t, 8> coefficients{};
          coefficients[(x + y + zz + t) % 8] = 0.025 * site_value;
          u(x, y, zz, t) = orbifold_exp_su3(
              orbifold_su3_algebra(coefficients));
          for (index_t j = 0; j < 3; ++j) {
            SUN<3> value = identitySUN<3>() * vacuum;
            for (index_t row = 0; row < 3; ++row) {
              for (index_t col = 0; col < 3; ++col) {
                const real_t component =
                    site_value + 7.0 * j + 3.0 * row + col;
                matrix_ref(value, row, col) +=
                    complex_t(0.002 * component,
                              0.001 * (component + row - col));
              }
            }
            z(x, y, zz, t, j) = value;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(field.spatial, z);
  Kokkos::deep_copy(field.temporal, u);
  Kokkos::fence();
  return field;
}

void perturb_spatial(OrbifoldField &field, const IndexArray<4> &site,
                     const index_t j, const index_t row, const index_t col,
                     const complex_t change) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   field.spatial);
  matrix_ref(host(site[0], site[1], site[2], site[3], j), row, col) += change;
  Kokkos::deep_copy(field.spatial, host);
  Kokkos::fence();
}

void perturb_temporal(OrbifoldField &field, const IndexArray<4> &site,
                      const SUN<3> &generator, const real_t step) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   field.temporal);
  host(site[0], site[1], site[2], site[3]) =
      orbifold_exp_su3(generator * step) *
      host(site[0], site[1], site[2], site[3]);
  Kokkos::deep_copy(field.temporal, host);
  Kokkos::fence();
}

void check_vacuum_and_temporal_normalization() {
  const IndexArray<4> dimensions{2, 2, 2, 4};
  OrbifoldActionParams params;
  params.spatial_spacing = 0.7;
  params.temporal_spacing = 0.2;
  params.coupling = 1.3;
  params.scalar_mass = 0.4;
  params.u1_mass = 0.6;
  const SUN<3> vacuum =
      identitySUN<3>() * std::sqrt(params.vacuum_scale_squared());
  OrbifoldField cold(dimensions, vacuum, identitySUN<3>(), "orbifold_cold");
  OrbifoldField force(dimensions, zeroSUN<3>(), zeroSUN<3>(),
                       "orbifold_cold_force");
  check(std::abs(orbifold_action(cold, params)) < 1.0e-25,
        "orbifold vacuum action vanishes");
  orbifold_force(cold, params, force);
  check(force_norm(force) < 1.0e-12, "orbifold vacuum force vanishes");

  OrbifoldActionParams temporal_params = params;
  temporal_params.scalar_mass = 0.0;
  temporal_params.u1_mass = 0.0;
  OrbifoldField temporal_field(dimensions, zeroSUN<3>(), identitySUN<3>(),
                                "orbifold_temporal_normalization");
  auto z = Kokkos::create_mirror_view(temporal_field.spatial);
  const std::array<real_t, 4> q{0.7, 1.1, 0.9, 1.4};
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t zz = 0; zz < dimensions[2]; ++zz) {
        for (index_t t = 0; t < dimensions[3]; ++t) {
          for (index_t j = 0; j < 3; ++j) {
            z(x, y, zz, t, j) = identitySUN<3>() * q[t];
          }
        }
      }
    }
  }
  Kokkos::deep_copy(temporal_field.spatial, z);
  real_t time_difference = 0.0;
  for (index_t t = 0; t < dimensions[3]; ++t) {
    const real_t delta = q[(t + 1) % dimensions[3]] - q[t];
    time_difference += delta * delta;
  }
  const real_t expected =
      dimensions[0] * dimensions[1] * dimensions[2] * 3.0 * 3.0 *
      time_difference / temporal_params.temporal_spacing;
  check(close(orbifold_action(temporal_field, temporal_params), expected,
              2.0e-13),
        "temporal term matches the notebook normalization");

  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                cold.temporal);
  const SUN<3> holonomy = diagonal_gauge_matrix(0.4);
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t zz = 0; zz < dimensions[2]; ++zz) {
        u(x, y, zz, 0) = holonomy;
      }
    }
  }
  Kokkos::deep_copy(cold.temporal, u);
  check(std::abs(orbifold_action(cold, params)) < 1.0e-24,
        "commuting nontrivial temporal holonomy has zero vacuum action");
  check(std::abs(trace(holonomy).real() / 3.0 - 1.0) > 1.0e-3,
        "full action retains a nontrivial Polyakov holonomy");
}

void check_spatial_normalization() {
  const IndexArray<4> dimensions{3, 2, 2, 2};
  OrbifoldActionParams params;
  params.spatial_spacing = 0.7;
  params.temporal_spacing = 0.3;
  params.coupling = 1.2;
  const real_t as = params.spatial_spacing;
  const real_t at = params.temporal_spacing;
  const real_t g2 = params.coupling * params.coupling;
  const real_t volume = dimensions[0] * dimensions[1] * dimensions[2] *
                        dimensions[3];

  OrbifoldField d_field(dimensions, zeroSUN<3>(), identitySUN<3>(),
                         "orbifold_d_normalization");
  auto d_host = Kokkos::create_mirror_view(d_field.spatial);
  const std::array<real_t, 3> q{0.4, 0.8, 1.1};
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t z = 0; z < dimensions[2]; ++z) {
        for (index_t t = 0; t < dimensions[3]; ++t) {
          d_host(x, y, z, t, 0) = identitySUN<3>() * q[x];
          d_host(x, y, z, t, 1) = zeroSUN<3>();
          d_host(x, y, z, t, 2) = zeroSUN<3>();
        }
      }
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
      at * g2 / (2.0 * as * as * as) * 3.0 * dimensions[1] *
      dimensions[2] * dimensions[3] * d_sum;
  check(close(orbifold_action(d_field, params), d_expected, 2.0e-13),
        "D-term normalization matches an analytic scalar-link case");

  OrbifoldField f_field(dimensions, zeroSUN<3>(), identitySUN<3>(),
                         "orbifold_f_normalization");
  auto f_host = Kokkos::create_mirror_view(f_field.spatial);
  SUN<3> z0 = zeroSUN<3>();
  SUN<3> z1 = zeroSUN<3>();
  constexpr real_t a = 0.6;
  constexpr real_t b = 0.9;
  matrix_ref(z0, 0, 1) = a;
  matrix_ref(z0, 1, 0) = a;
  matrix_ref(z1, 0, 1) = complex_t(0.0, -b);
  matrix_ref(z1, 1, 0) = complex_t(0.0, b);
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t z = 0; z < dimensions[2]; ++z) {
        for (index_t t = 0; t < dimensions[3]; ++t) {
          f_host(x, y, z, t, 0) = z0;
          f_host(x, y, z, t, 1) = z1;
          f_host(x, y, z, t, 2) = zeroSUN<3>();
        }
      }
    }
  }
  Kokkos::deep_copy(f_field.spatial, f_host);
  const real_t f_expected =
      at * 2.0 * g2 / (as * as * as) * volume * 8.0 * a * a * b * b;
  check(close(orbifold_action(f_field, params), f_expected, 2.0e-13),
        "F-term normalization matches an analytic commutator case");

  params.scalar_mass = 0.5;
  params.u1_mass = 0.4;
  constexpr real_t uniform_q = 0.75;
  OrbifoldField potential_field(dimensions,
                                 identitySUN<3>() * uniform_q,
                                 identitySUN<3>(),
                                 "orbifold_potential_normalization");
  const real_t c = params.vacuum_scale_squared();
  const real_t factor_mass =
      params.scalar_mass * params.scalar_mass * g2 / (2.0 * as);
  const real_t factor_det = params.u1_mass * params.u1_mass * c;
  const real_t radial = uniform_q * uniform_q - c;
  const real_t determinant =
      uniform_q * uniform_q * uniform_q / std::sqrt(c * c * c) - 1.0;
  const real_t potential_expected =
      at * volume * 3.0 *
      (factor_mass * 3.0 * radial * radial +
       factor_det * determinant * determinant);
  check(close(orbifold_action(potential_field, params), potential_expected,
              2.0e-13),
        "radial and determinant normalizations match a uniform-link case");
}

void check_gauge_invariance(const OrbifoldActionParams &params) {
  const IndexArray<4> dimensions{2, 2, 2, 2};
  const auto source = deterministic_field(dimensions, params, "orbifold_gauge");
  OrbifoldField transformed(dimensions, zeroSUN<3>(), identitySUN<3>(),
                             "orbifold_gauge_transformed");
  const auto input_z = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), source.spatial);
  const auto input_u = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), source.temporal);
  auto output_z = Kokkos::create_mirror_view(transformed.spatial);
  auto output_u = Kokkos::create_mirror_view(transformed.temporal);
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t z = 0; z < dimensions[2]; ++z) {
        for (index_t t = 0; t < dimensions[3]; ++t) {
          const IndexArray<4> site{x, y, z, t};
          const SUN<3> here = diagonal_gauge_matrix(
              0.13 * (1.0 + x + 2.0 * y + 3.0 * z + 5.0 * t));
          for (index_t j = 0; j < 3; ++j) {
            const auto plus_j = shift_index_plus(site, j, 1, dimensions);
            const SUN<3> there = diagonal_gauge_matrix(
                0.13 * (1.0 + plus_j[0] + 2.0 * plus_j[1] +
                        3.0 * plus_j[2] + 5.0 * plus_j[3]));
            output_z(x, y, z, t, j) =
                here * input_z(x, y, z, t, j) * conj(there);
          }
          const auto plus_t = shift_index_plus(site, 3, 1, dimensions);
          const SUN<3> later = diagonal_gauge_matrix(
              0.13 * (1.0 + plus_t[0] + 2.0 * plus_t[1] +
                      3.0 * plus_t[2] + 5.0 * plus_t[3]));
          output_u(x, y, z, t) =
              here * input_u(x, y, z, t) * conj(later);
        }
      }
    }
  }
  Kokkos::deep_copy(transformed.spatial, output_z);
  Kokkos::deep_copy(transformed.temporal, output_u);
  check(close(orbifold_action(source, params),
              orbifold_action(transformed, params), 2.0e-12),
        "full orbifold action is gauge invariant");
}

void check_forces(const OrbifoldActionParams &params) {
  const IndexArray<4> dimensions{2, 2, 2, 2};
  const auto field = deterministic_field(dimensions, params, "orbifold_fd");
  OrbifoldField force(dimensions, zeroSUN<3>(), zeroSUN<3>(),
                       "orbifold_fd_force");
  orbifold_force(field, params, force);
  const auto force_z = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), force.spatial);
  const auto force_u = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), force.temporal);
  const IndexArray<4> site{0, 1, 0, 1};
  constexpr index_t j = 1;
  constexpr index_t row = 0;
  constexpr index_t col = 2;
  constexpr real_t h = 2.0e-6;

  auto plus = copy_orbifold_field(field, "orbifold_fd_plus_real");
  auto minus = copy_orbifold_field(field, "orbifold_fd_minus_real");
  perturb_spatial(plus, site, j, row, col, complex_t(h, 0.0));
  perturb_spatial(minus, site, j, row, col, complex_t(-h, 0.0));
  const real_t real_fd =
      (orbifold_action(plus, params) - orbifold_action(minus, params)) /
      (2.0 * h);
  check(close(real_fd,
              matrix_ref(force_z(site[0], site[1], site[2], site[3], j), row,
                         col)
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
              matrix_ref(force_z(site[0], site[1], site[2], site[3], j), row,
                         col)
                  .imag(),
              3.0e-6, 3.0e-8),
        "spatial imaginary force matches a central difference");

  Kokkos::Array<real_t, 8> coefficients{};
  coefficients[3] = 0.7;
  coefficients[7] = -0.2;
  const SUN<3> generator = orbifold_su3_algebra(coefficients);
  plus = copy_orbifold_field(field, "orbifold_fd_plus_group");
  minus = copy_orbifold_field(field, "orbifold_fd_minus_group");
  perturb_temporal(plus, site, generator, h);
  perturb_temporal(minus, site, generator, -h);
  const real_t group_fd =
      (orbifold_action(plus, params) - orbifold_action(minus, params)) /
      (2.0 * h);
  const real_t group_expected =
      2.0 *
      trace(force_u(site[0], site[1], site[2], site[3]) * generator).real();
  if (!close(group_fd, group_expected, 3.0e-6, 3.0e-8)) {
    std::printf("group force: finite difference %.16e, analytic %.16e\n",
                group_fd, group_expected);
  }
  check(close(group_fd, group_expected, 3.0e-6, 3.0e-8),
        "temporal SU(3) force matches a group central difference");
}

void check_hmc(const OrbifoldActionParams &params) {
  const IndexArray<4> dimensions{2, 2, 2, 2};
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
        "temporal links remain in SU(3) under leapfrog");

  const OrbifoldHMCResult result = hmc.step();
  check(std::isfinite(result.initial_hamiltonian) &&
            std::isfinite(result.final_hamiltonian) &&
            std::isfinite(result.delta_hamiltonian),
        "one orbifold HMC step has finite Hamiltonians");
  errors = orbifold_temporal_group_errors(field);
  check(errors.unitarity < 2.0e-10 && errors.determinant < 2.0e-10,
        "one orbifold HMC step preserves SU(3)");
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
    params.u1_mass = 0.45;
    check_vacuum_and_temporal_normalization();
    check_spatial_normalization();
    check_gauge_invariance(params);
    check_forces(params);
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
