#pragma once

#include "core/compiled_theory.hpp"
#include "core/indexing.hpp"
#include "groups/group_ops.hpp"
#include "observables/wilson_loop.hpp"

#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace klft {

// The orbifold variables are arbitrary complex 3x3 spatial links Z_j and
// compact SU(3) temporal links U_0. The d-dimensional normalization follows
// Eqs. (14)--(15) of Bergner, Hanada, and Mendicelli, arXiv:2506.00755.
// Keeping U_0 explicit retains the periodic holonomy absent after U_0 = 1.
static_assert(compiled_rank >= 2 && compiled_rank <= 4,
              "The orbifold action supports 1+1D, 2+1D, and 3+1D builds.");
inline constexpr index_t orbifold_spatial_directions =
    static_cast<index_t>(compiled_rank - 1);
inline constexpr index_t orbifold_time_direction =
    orbifold_spatial_directions;
using OrbifoldDimensions = IndexArray<compiled_rank>;
using OrbifoldSpatialView = std::conditional_t<
    compiled_rank == 4,
    Kokkos::View<SUN<3> ****[3], Kokkos::MemoryTraits<Kokkos::Restrict>>,
    std::conditional_t<compiled_rank == 3,
      Kokkos::View<SUN<3> ***[2], Kokkos::MemoryTraits<Kokkos::Restrict>>,
      Kokkos::View<SUN<3> **[1], Kokkos::MemoryTraits<Kokkos::Restrict>>>>;
using OrbifoldTemporalView = std::conditional_t<
    compiled_rank == 4,
    Kokkos::View<SUN<3> ****, Kokkos::MemoryTraits<Kokkos::Restrict>>,
    std::conditional_t<compiled_rank == 3,
      Kokkos::View<SUN<3> ***, Kokkos::MemoryTraits<Kokkos::Restrict>>,
      Kokkos::View<SUN<3> **, Kokkos::MemoryTraits<Kokkos::Restrict>>>>;

template <class View>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto)
orbifold_spatial_ref(const View &z, const OrbifoldDimensions &site,
                     const index_t j) {
  if constexpr (compiled_rank == 4) {
    return z(site[0], site[1], site[2], site[3], j);
  } else if constexpr (compiled_rank == 3) {
    return z(site[0], site[1], site[2], j);
  } else {
    return z(site[0], site[1], j);
  }
}

template <class View>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto)
orbifold_temporal_ref(const View &u, const OrbifoldDimensions &site) {
  if constexpr (compiled_rank == 4) {
    return u(site[0], site[1], site[2], site[3]);
  } else if constexpr (compiled_rank == 3) {
    return u(site[0], site[1], site[2]);
  } else {
    return u(site[0], site[1]);
  }
}

struct OrbifoldField {
  OrbifoldSpatialView spatial;
  OrbifoldTemporalView temporal;
  OrbifoldDimensions dimensions;

  explicit OrbifoldField(
      const OrbifoldDimensions &dims,
      const SUN<3> &spatial_init = zeroSUN<3>(),
      const SUN<3> &temporal_init = identitySUN<3>(),
      const std::string &label = "orbifold")
      : dimensions(dims) {
    for (const index_t extent : dimensions) {
      if (extent <= 0) {
        throw std::invalid_argument(
            "OrbifoldField requires positive lattice extents.");
      }
    }
    if constexpr (compiled_rank == 4) {
      spatial = OrbifoldSpatialView(label + "_spatial", dimensions[0],
                                    dimensions[1], dimensions[2], dimensions[3]);
      temporal = OrbifoldTemporalView(label + "_temporal", dimensions[0],
                                      dimensions[1], dimensions[2],
                                      dimensions[3]);
    } else if constexpr (compiled_rank == 3) {
      spatial = OrbifoldSpatialView(label + "_spatial", dimensions[0],
                                    dimensions[1], dimensions[2]);
      temporal = OrbifoldTemporalView(label + "_temporal", dimensions[0],
                                      dimensions[1], dimensions[2]);
    } else {
      spatial = OrbifoldSpatialView(label + "_spatial", dimensions[0], dimensions[1]);
      temporal = OrbifoldTemporalView(label + "_temporal", dimensions[0], dimensions[1]);
    }
    initialize(spatial_init, temporal_init, label);
  }

  void initialize(const SUN<3> &spatial_init, const SUN<3> &temporal_init,
                  const std::string &label) {
    const auto z = spatial;
    const auto u = temporal;
    const auto dims = dimensions;
    const size_t sites = wilson_site_count<compiled_rank>(dims);
    Kokkos::parallel_for(
        label + "_initialize", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear) {
          const auto site = wilson_linear_to_site<compiled_rank>(linear, dims);
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            orbifold_spatial_ref(z, site, j) = spatial_init;
          }
          orbifold_temporal_ref(u, site) = temporal_init;
        });
    Kokkos::fence();
  }
};

inline OrbifoldField copy_orbifold_field(const OrbifoldField &source,
                                         const std::string &label) {
  OrbifoldField copy(source.dimensions, zeroSUN<3>(), zeroSUN<3>(), label);
  Kokkos::deep_copy(copy.spatial, source.spatial);
  Kokkos::deep_copy(copy.temporal, source.temporal);
  Kokkos::fence();
  return copy;
}

struct OrbifoldActionParams {
  real_t spatial_spacing = 1.0;
  real_t temporal_spacing = 1.0;
  real_t coupling = 1.0;
  real_t scalar_mass = 0.0;
  real_t u1_mass = 0.0;

  void validate() const {
    if (!(spatial_spacing > 0.0) || !(temporal_spacing > 0.0) ||
        !(coupling > 0.0) || scalar_mass < 0.0 || u1_mass < 0.0 ||
        !std::isfinite(spatial_spacing) || !std::isfinite(temporal_spacing) ||
        !std::isfinite(coupling) || !std::isfinite(scalar_mass) ||
        !std::isfinite(u1_mass)) {
      throw std::invalid_argument(
          "Orbifold spacings and coupling must be positive; masses must be "
          "non-negative.");
    }
  }

  real_t vacuum_scale_squared() const {
    return std::pow(spatial_spacing, orbifold_spatial_directions - 2) /
           (2.0 * coupling * coupling);
  }
};

template <class RNG>
inline void initialize_hot_orbifold_field(OrbifoldField &field,
                                          const OrbifoldActionParams &params,
                                          const real_t noise, RNG &rng) {
  params.validate();
  if (noise < 0.0) {
    throw std::invalid_argument(
        "Orbifold initialization noise must be non-negative.");
  }
  const auto z = field.spatial;
  const auto u = field.temporal;
  const auto dimensions = field.dimensions;
  const real_t vacuum_scale = Kokkos::sqrt(params.vacuum_scale_squared());
  const real_t complex_noise = noise / Kokkos::sqrt(2.0);
  auto pool = rng;
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  Kokkos::parallel_for(
      "orbifold_hot_start", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
        auto generator = pool.get_state();
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          SUN<3> value;
          rand_matrix(value, generator);
          value *= vacuum_scale;
#pragma unroll
          for (index_t row = 0; row < 3; ++row) {
#pragma unroll
            for (index_t col = 0; col < 3; ++col) {
              matrix_ref(value, row, col) +=
                  complex_t(generator.normal(0.0, complex_noise),
                            generator.normal(0.0, complex_noise));
            }
          }
          orbifold_spatial_ref(z, site, j) = value;
        }
        SUN<3> temporal;
        rand_matrix(temporal, generator);
        orbifold_temporal_ref(u, site) = temporal;
        pool.free_state(generator);
      });
  Kokkos::fence();
}

inline void initialize_orbifold_from_gauge(
    OrbifoldField &field,
    const typename DeviceGaugeFieldType<compiled_rank, 3>::type &gauge,
    const OrbifoldActionParams &params) {
  params.validate();
  if (field.dimensions != gauge.dimensions) {
    throw std::invalid_argument(
        "Orbifold and compact-gauge lattice extents must match.");
  }
  const auto links = gauge;
  const auto z = field.spatial;
  const auto u = field.temporal;
  const auto dimensions = field.dimensions;
  const real_t scale = Kokkos::sqrt(params.vacuum_scale_squared());
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  Kokkos::parallel_for(
      "orbifold_from_compact_gauge", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          orbifold_spatial_ref(z, site, j) = links(site, j) * scale;
        }
        orbifold_temporal_ref(u, site) =
            links(site, orbifold_time_direction);
      });
  Kokkos::fence();
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_spatial_at(const OrbifoldSpatialView &z,
                    const OrbifoldDimensions &site,
                    const index_t j) {
  return orbifold_spatial_ref(z, site, j);
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_temporal_at(const OrbifoldTemporalView &u,
                     const OrbifoldDimensions &site) {
  return orbifold_temporal_ref(u, site);
}

KOKKOS_FORCEINLINE_FUNCTION real_t orbifold_matrix_norm_squared(
    const SUN<3> &a) {
  real_t result = 0.0;
#pragma unroll
  for (index_t row = 0; row < 3; ++row) {
#pragma unroll
    for (index_t col = 0; col < 3; ++col) {
      const complex_t value = matrix_ref(a, row, col);
      result += value.real() * value.real() + value.imag() * value.imag();
    }
  }
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_matrix_scale(const SUN<3> &a, const complex_t scale) {
  SUN<3> result = zeroSUN<3>();
#pragma unroll
  for (index_t i = 0; i < 9; ++i) {
    result.comp[i] = a.comp[i] * scale;
  }
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION complex_t
orbifold_determinant(const SUN<3> &a) {
  return matrix_ref(a, 0, 0) * matrix_ref(a, 1, 1) * matrix_ref(a, 2, 2) +
         matrix_ref(a, 0, 1) * matrix_ref(a, 1, 2) * matrix_ref(a, 2, 0) +
         matrix_ref(a, 0, 2) * matrix_ref(a, 1, 0) * matrix_ref(a, 2, 1) -
         matrix_ref(a, 0, 2) * matrix_ref(a, 1, 1) * matrix_ref(a, 2, 0) -
         matrix_ref(a, 0, 1) * matrix_ref(a, 1, 0) * matrix_ref(a, 2, 2) -
         matrix_ref(a, 0, 0) * matrix_ref(a, 1, 2) * matrix_ref(a, 2, 1);
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_conjugate_cofactor(const SUN<3> &a) {
  SUN<3> result = zeroSUN<3>();
#pragma unroll
  for (index_t row = 0; row < 3; ++row) {
#pragma unroll
    for (index_t col = 0; col < 3; ++col) {
      IndexArray<2> rows{};
      IndexArray<2> cols{};
      index_t ri = 0;
      index_t ci = 0;
#pragma unroll
      for (index_t r = 0; r < 3; ++r) {
        if (r != row) {
          rows[ri++] = r;
        }
      }
#pragma unroll
      for (index_t c = 0; c < 3; ++c) {
        if (c != col) {
          cols[ci++] = c;
        }
      }
      complex_t minor =
          Kokkos::conj(matrix_ref(a, rows[0], cols[0])) *
              Kokkos::conj(matrix_ref(a, rows[1], cols[1])) -
          Kokkos::conj(matrix_ref(a, rows[0], cols[1])) *
              Kokkos::conj(matrix_ref(a, rows[1], cols[0]));
      if ((row + col) % 2 != 0) {
        minor *= -1.0;
      }
      matrix_ref(result, row, col) = minor;
    }
  }
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_d_term(const OrbifoldSpatialView &z,
                 const OrbifoldDimensions &site,
                 const OrbifoldDimensions &dimensions) {
  SUN<3> result = zeroSUN<3>();
#pragma unroll
  for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
    const auto minus_j = shift_index_minus(site, j, 1, dimensions);
    const SUN<3> here = orbifold_spatial_at(z, site, j);
    const SUN<3> behind = orbifold_spatial_at(z, minus_j, j);
    result += here * conj(here) - conj(behind) * behind;
  }
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_f_term(const OrbifoldSpatialView &z,
                 const OrbifoldDimensions &site,
                 const index_t j, const index_t k,
                 const OrbifoldDimensions &dimensions) {
  const auto plus_j = shift_index_plus(site, j, 1, dimensions);
  const auto plus_k = shift_index_plus(site, k, 1, dimensions);
  return orbifold_spatial_at(z, site, j) *
             orbifold_spatial_at(z, plus_j, k) -
         orbifold_spatial_at(z, site, k) *
             orbifold_spatial_at(z, plus_k, j);
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3> orbifold_temporal_difference(
    const OrbifoldSpatialView &z, const OrbifoldTemporalView &u,
    const OrbifoldDimensions &site, const index_t j,
    const OrbifoldDimensions &dimensions) {
  const auto plus_t =
      shift_index_plus(site, orbifold_time_direction, 1, dimensions);
  const auto plus_j = shift_index_plus(site, j, 1, dimensions);
  return orbifold_temporal_at(u, site) * orbifold_spatial_at(z, plus_t, j) *
             conj(orbifold_temporal_at(u, plus_j)) -
         orbifold_spatial_at(z, site, j);
}

inline real_t orbifold_action(const OrbifoldField &field,
                              const OrbifoldActionParams &params) {
  params.validate();
  const auto z = field.spatial;
  const auto u = field.temporal;
  const auto dimensions = field.dimensions;
  const real_t as = params.spatial_spacing;
  const real_t at = params.temporal_spacing;
  const real_t g2 = params.coupling * params.coupling;
  const real_t c = params.vacuum_scale_squared();
  const real_t spatial_volume =
      std::pow(as, static_cast<int>(orbifold_spatial_directions));
  const real_t radial_spacing = std::pow(
      as, static_cast<int>(orbifold_spatial_directions - 2));
  const real_t factor_d = g2 / (2.0 * spatial_volume);
  const real_t factor_f = 2.0 * g2 / spatial_volume;
  const real_t factor_mass =
      params.scalar_mass * params.scalar_mass * g2 /
      (2.0 * radial_spacing);
  const real_t factor_det = params.u1_mass * params.u1_mass * c;
  const real_t determinant_scale = 1.0 / Kokkos::sqrt(c * c * c);
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);

  real_t result = 0.0;
  Kokkos::parallel_reduce(
      "orbifold_action", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear, real_t &local) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
        real_t temporal = 0.0;
        real_t spatial = factor_d *
                         orbifold_matrix_norm_squared(
                             orbifold_d_term(z, site, dimensions));
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          temporal += orbifold_matrix_norm_squared(
              orbifold_temporal_difference(z, u, site, j, dimensions));
          const SUN<3> zj = orbifold_spatial_at(z, site, j);
          const SUN<3> radial =
              zj * conj(zj) - identitySUN<3>() * c;
          spatial += factor_mass * orbifold_matrix_norm_squared(radial);
          const complex_t determinant_constraint =
              orbifold_determinant(zj) * determinant_scale -
              complex_t(1.0, 0.0);
          spatial +=
              factor_det *
              (determinant_constraint.real() * determinant_constraint.real() +
               determinant_constraint.imag() * determinant_constraint.imag());
#pragma unroll
          for (index_t k = j + 1; k < orbifold_spatial_directions; ++k) {
            spatial += factor_f * orbifold_matrix_norm_squared(
                                      orbifold_f_term(z, site, j, k,
                                                       dimensions));
          }
        }
        local += temporal / at + at * spatial;
      },
      result);
  Kokkos::fence();
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_antihermitian_traceless(const SUN<3> &a) {
  SUN<3> result = (a - conj(a)) * 0.5;
  const complex_t mean_trace = trace(result) / 3.0;
#pragma unroll
  for (index_t i = 0; i < 3; ++i) {
    matrix_ref(result, i, i) -= mean_trace;
  }
  return result;
}

inline void orbifold_force(const OrbifoldField &field,
                           const OrbifoldActionParams &params,
                           OrbifoldField &force) {
  params.validate();
  if (field.dimensions != force.dimensions) {
    throw std::invalid_argument("Orbifold force dimensions do not match.");
  }
  const auto z = field.spatial;
  const auto u = field.temporal;
  const auto gz = force.spatial;
  const auto gu = force.temporal;
  const auto dimensions = field.dimensions;
  const real_t as = params.spatial_spacing;
  const real_t at = params.temporal_spacing;
  const real_t g2 = params.coupling * params.coupling;
  const real_t c = params.vacuum_scale_squared();
  const real_t spatial_volume =
      std::pow(as, static_cast<int>(orbifold_spatial_directions));
  const real_t radial_spacing = std::pow(
      as, static_cast<int>(orbifold_spatial_directions - 2));
  const real_t factor_d = g2 / (2.0 * spatial_volume);
  const real_t factor_f = 2.0 * g2 / spatial_volume;
  const real_t factor_mass =
      params.scalar_mass * params.scalar_mass * g2 /
      (2.0 * radial_spacing);
  const real_t factor_det = params.u1_mass * params.u1_mass * c;
  const real_t determinant_scale = 1.0 / Kokkos::sqrt(c * c * c);
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);

  Kokkos::parallel_for(
      "orbifold_force", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
        const auto minus_t =
            shift_index_minus(site, orbifold_time_direction, 1, dimensions);
        const SUN<3> d_here = orbifold_d_term(z, site, dimensions);
        Kokkos::Array<SUN<3>, compiled_rank - 1> dzstar;
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          const auto plus_j = shift_index_plus(site, j, 1, dimensions);
          const auto minus_t_plus_j =
              shift_index_plus(minus_t, j, 1, dimensions);
          const SUN<3> zj = orbifold_spatial_at(z, site, j);
          const SUN<3> b_here =
              orbifold_temporal_difference(z, u, site, j, dimensions);
          const SUN<3> b_behind =
              orbifold_temporal_difference(z, u, minus_t, j, dimensions);
          const SUN<3> temporal_gradient =
              (zeroSUN<3>() - b_here +
               conj(orbifold_temporal_at(u, minus_t)) * b_behind *
                   orbifold_temporal_at(u, minus_t_plus_j)) *
              (2.0 / at);

          dzstar[j] =
              (d_here * zj -
               zj * orbifold_d_term(z, plus_j, dimensions)) *
              (2.0 * factor_d);
          const SUN<3> radial =
              zj * conj(zj) - identitySUN<3>() * c;
          dzstar[j] += radial * zj * (2.0 * factor_mass);
          const complex_t determinant_constraint =
              orbifold_determinant(zj) * determinant_scale -
              complex_t(1.0, 0.0);
          dzstar[j] += orbifold_matrix_scale(
              orbifold_conjugate_cofactor(zj),
              factor_det * determinant_scale * determinant_constraint);
          orbifold_spatial_ref(gz, site, j) = temporal_gradient;
        }

#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
#pragma unroll
          for (index_t k = j + 1; k < orbifold_spatial_directions; ++k) {
            const auto plus_j = shift_index_plus(site, j, 1, dimensions);
            const auto plus_k = shift_index_plus(site, k, 1, dimensions);
            const auto minus_j = shift_index_minus(site, j, 1, dimensions);
            const auto minus_k = shift_index_minus(site, k, 1, dimensions);
            const SUN<3> f_here =
                orbifold_f_term(z, site, j, k, dimensions);
            dzstar[j] +=
                (f_here * conj(orbifold_spatial_at(z, plus_j, k)) -
                 conj(orbifold_spatial_at(z, minus_k, k)) *
                     orbifold_f_term(z, minus_k, j, k, dimensions)) *
                factor_f;
            dzstar[k] +=
                ((zeroSUN<3>() - f_here) *
                     conj(orbifold_spatial_at(z, plus_k, j)) +
                 conj(orbifold_spatial_at(z, minus_j, j)) *
                     orbifold_f_term(z, minus_j, j, k, dimensions)) *
                factor_f;
          }
        }
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          const SUN<3> current = orbifold_spatial_ref(gz, site, j);
          orbifold_spatial_ref(gz, site, j) =
              current + dzstar[j] * (2.0 * at);
        }

        SUN<3> temporal_force = zeroSUN<3>();
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          const auto plus_t =
              shift_index_plus(site, orbifold_time_direction, 1, dimensions);
          const auto plus_j = shift_index_plus(site, j, 1, dimensions);
          const auto minus_j = shift_index_minus(site, j, 1, dimensions);
          const auto minus_j_plus_t =
              shift_index_plus(minus_j, orbifold_time_direction, 1,
                               dimensions);
          const SUN<3> transported =
              orbifold_temporal_at(u, site) *
              orbifold_spatial_at(z, plus_t, j) *
              conj(orbifold_temporal_at(u, plus_j));
          const SUN<3> difference =
              transported - orbifold_spatial_at(z, site, j);
          const SUN<3> transported_behind =
              orbifold_temporal_at(u, minus_j) *
              orbifold_spatial_at(z, minus_j_plus_t, j) *
              conj(orbifold_temporal_at(u, site));
          const SUN<3> difference_behind =
              transported_behind - orbifold_spatial_at(z, minus_j, j);
          temporal_force += transported * conj(difference) -
                            conj(difference_behind) * transported_behind;
        }
        orbifold_temporal_ref(gu, site) =
            orbifold_antihermitian_traceless(temporal_force) * (1.0 / at);
      });
  Kokkos::fence();
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_su3_algebra(const Kokkos::Array<real_t, 8> &a) {
  constexpr real_t sqrt3_inverse = 0.57735026918962576451;
  SUN<3> result = zeroSUN<3>();
  matrix_ref(result, 0, 0) =
      complex_t(0.0, 0.5 * (a[2] + sqrt3_inverse * a[7]));
  matrix_ref(result, 0, 1) = complex_t(0.5 * a[1], 0.5 * a[0]);
  matrix_ref(result, 0, 2) = complex_t(0.5 * a[4], 0.5 * a[3]);
  matrix_ref(result, 1, 0) = complex_t(-0.5 * a[1], 0.5 * a[0]);
  matrix_ref(result, 1, 1) =
      complex_t(0.0, 0.5 * (-a[2] + sqrt3_inverse * a[7]));
  matrix_ref(result, 1, 2) = complex_t(0.5 * a[6], 0.5 * a[5]);
  matrix_ref(result, 2, 0) = complex_t(-0.5 * a[4], 0.5 * a[3]);
  matrix_ref(result, 2, 1) = complex_t(-0.5 * a[6], 0.5 * a[5]);
  matrix_ref(result, 2, 2) = complex_t(0.0, -sqrt3_inverse * a[7]);
  return result;
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3> orbifold_exp_su3(const SUN<3> &a) {
  // Scaling and squaring with a Taylor polynomial; N. J. Higham,
  // SIAM J. Matrix Anal. Appl. 26 (2005) 1179, doi:10.1137/04061101X.
  real_t norm = Kokkos::sqrt(orbifold_matrix_norm_squared(a));
  SUN<3> x = a;
  index_t squarings = 0;
  while (norm > 0.5 && squarings < 60) {
    x *= 0.5;
    norm *= 0.5;
    ++squarings;
  }
  SUN<3> term = identitySUN<3>();
  SUN<3> result = term;
#pragma unroll
  for (index_t order = 1; order <= 24; ++order) {
    term = term * x * (1.0 / static_cast<real_t>(order));
    result += term;
  }
  for (index_t i = 0; i < squarings; ++i) {
    result *= result;
  }
  return result;
}

struct OrbifoldHMCParams {
  real_t step_size = 0.01;
  index_t steps = 10;

  void validate() const {
    if (!(step_size > 0.0) || !std::isfinite(step_size) || steps <= 0) {
      throw std::invalid_argument(
          "Orbifold HMC step size and step count must be positive.");
    }
  }
};

struct OrbifoldHMCResult {
  bool accepted;
  real_t initial_hamiltonian;
  real_t final_hamiltonian;
  real_t delta_hamiltonian;
};

struct OrbifoldGroupErrors {
  real_t unitarity;
  real_t determinant;
};

inline OrbifoldGroupErrors
orbifold_temporal_group_errors(const OrbifoldField &field) {
  const auto u = field.temporal;
  const auto dimensions = field.dimensions;
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  real_t unitarity = 0.0;
  Kokkos::parallel_reduce(
      "orbifold_temporal_unitarity", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear, real_t &maximum) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
        const SUN<3> link = orbifold_temporal_at(u, site);
        const SUN<3> check = conj(link) * link - identitySUN<3>();
        maximum = Kokkos::max(maximum,
                              Kokkos::sqrt(orbifold_matrix_norm_squared(check)));
      },
      Kokkos::Max<real_t>(unitarity));
  real_t determinant = 0.0;
  Kokkos::parallel_reduce(
      "orbifold_temporal_determinant", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear, real_t &maximum) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
        const complex_t error =
            orbifold_determinant(orbifold_temporal_at(u, site)) -
                                complex_t(1.0, 0.0);
        maximum = Kokkos::max(
            maximum,
            Kokkos::sqrt(error.real() * error.real() +
                         error.imag() * error.imag()));
      },
      Kokkos::Max<real_t>(determinant));
  Kokkos::fence();
  return OrbifoldGroupErrors{unitarity, determinant};
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3>
orbifold_polar_unitary(const SUN<3> &z, bool &converged) {
  // Newton's polar iteration computes the unitary U in Z = H U.  Unlike
  // Gram--Schmidt projection, it is equivariant under both endpoint gauge
  // transformations, as required for Wilson loops; see Higham (1986),
  // doi:10.1137/0907079.
  constexpr real_t tolerance_squared = 1.0e-24;
  constexpr real_t singular_tolerance = 1.0e-30;
  converged = false;
  const real_t norm_squared = orbifold_matrix_norm_squared(z);
  if (!(norm_squared > singular_tolerance)) {
    return zeroSUN<3>();
  }

  SUN<3> x = z * Kokkos::sqrt(3.0 / norm_squared);
  for (index_t iteration = 0; iteration < 50; ++iteration) {
    const complex_t determinant = orbifold_determinant(x);
    const real_t determinant_norm_squared =
        determinant.real() * determinant.real() +
        determinant.imag() * determinant.imag();
    if (!(determinant_norm_squared > singular_tolerance)) {
      return zeroSUN<3>();
    }
    const SUN<3> inverse_dagger = orbifold_matrix_scale(
        orbifold_conjugate_cofactor(x),
        complex_t(1.0, 0.0) / Kokkos::conj(determinant));
    x = (x + inverse_dagger) * 0.5;
    const SUN<3> residual = conj(x) * x - identitySUN<3>();
    if (orbifold_matrix_norm_squared(residual) <= tolerance_squared) {
      converged = true;
      return x;
    }
  }
  return x;
}

inline typename DeviceGaugeFieldType<compiled_rank, 3>::type
orbifold_projected_gauge_field(const OrbifoldField &field) {
  const auto dimensions = field.dimensions;
  index_t l2 = 1;
  index_t l3 = 1;
  if constexpr (compiled_rank >= 3) {
    l2 = dimensions[2];
  }
  if constexpr (compiled_rank == 4) {
    l3 = dimensions[3];
  }
  auto projected = make_identity_gauge_field<compiled_rank, 3>(
      dimensions[0], dimensions[1], l2, l3);
  const auto z = field.spatial;
  const auto u = field.temporal;
  const auto links = projected;
  const size_t sites = wilson_site_count<compiled_rank>(dimensions);
  size_t failures = 0;
  Kokkos::parallel_reduce(
      "orbifold_polar_projection", Kokkos::RangePolicy<>(0, sites),
      KOKKOS_LAMBDA(const size_t linear, size_t &local_failures) {
        const auto site =
            wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
        for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
          bool converged = false;
          links(site, j) =
              orbifold_polar_unitary(orbifold_spatial_at(z, site, j),
                                     converged);
          if (!converged) {
            ++local_failures;
          }
        }
        links(site, orbifold_time_direction) =
            orbifold_temporal_at(u, site);
      },
      failures);
  Kokkos::fence();
  if (failures != 0) {
    throw std::runtime_error("Polar decomposition failed for " +
                             std::to_string(failures) +
                             " orbifold spatial links.");
  }
  return projected;
}

inline std::vector<Kokkos::Array<real_t, 3>>
orbifold_wilson_loops(const OrbifoldField &field, const index_t max_r,
                      const index_t max_t) {
  index_t min_spatial_extent = field.dimensions[0];
  for (index_t j = 1; j < orbifold_spatial_directions; ++j) {
    min_spatial_extent = std::min(min_spatial_extent, field.dimensions[j]);
  }
  if (max_r <= 0 || max_t <= 0 || max_r > min_spatial_extent / 2 ||
      max_t > field.dimensions[orbifold_time_direction] / 2) {
    throw std::invalid_argument(
        "Orbifold Wilson-loop extents must be positive and no larger than "
        "half the corresponding periodic lattice extent.");
  }

  std::vector<Kokkos::Array<index_t, 2>> pairs;
  pairs.reserve(static_cast<size_t>(max_r) * static_cast<size_t>(max_t));
  for (index_t r = 1; r <= max_r; ++r) {
    for (index_t t = 1; t <= max_t; ++t) {
      pairs.push_back(Kokkos::Array<index_t, 2>{r, t});
    }
  }

  const auto projected = orbifold_projected_gauge_field(field);
  std::vector<Kokkos::Array<real_t, 3>> loops;
  loops.reserve(pairs.size());
  WilsonLoop_temporal_raw_fused<compiled_rank, 3>(projected, pairs, loops);
  return loops;
}

// Gaussian momenta, leapfrog, and Metropolis acceptance follow Duane et al.,
// Phys. Lett. B 195 (1987) 216, doi:10.1016/0370-2693(87)91197-X, using the
// same orchestration as klft-asen HMC.hpp at revision ae3d9c9.
class OrbifoldHMC {
public:
  OrbifoldHMC(OrbifoldField &field, const OrbifoldActionParams &action_params,
              const OrbifoldHMCParams &hmc_params, const uint64_t seed)
      : field_(field), action_params_(action_params), hmc_params_(hmc_params),
        momentum_(field.dimensions, zeroSUN<3>(), zeroSUN<3>(),
                  "orbifold_momentum"),
        force_(field.dimensions, zeroSUN<3>(), zeroSUN<3>(),
               "orbifold_force_buffer"),
        backup_(field.dimensions, zeroSUN<3>(), zeroSUN<3>(),
                "orbifold_backup"),
        momentum_rng_(seed), host_rng_(seed), uniform_(0.0, 1.0) {
    action_params_.validate();
    hmc_params_.validate();
  }

  void randomize_momenta() {
    const auto pz = momentum_.spatial;
    const auto pu = momentum_.temporal;
    const auto dimensions = field_.dimensions;
    auto rng = momentum_rng_;
    const size_t sites = wilson_site_count<compiled_rank>(dimensions);
    Kokkos::parallel_for(
        "orbifold_randomize_momenta", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear) {
          const auto site =
              wilson_linear_to_site<compiled_rank>(linear, dimensions);
          auto generator = rng.get_state();
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            SUN<3> value = zeroSUN<3>();
#pragma unroll
            for (index_t row = 0; row < 3; ++row) {
#pragma unroll
              for (index_t col = 0; col < 3; ++col) {
                matrix_ref(value, row, col) =
                    complex_t(generator.normal(0.0, 1.0),
                              generator.normal(0.0, 1.0));
              }
            }
            orbifold_spatial_ref(pz, site, j) = value;
          }
          Kokkos::Array<real_t, 8> coefficients;
#pragma unroll
          for (index_t a = 0; a < 8; ++a) {
            coefficients[a] = generator.normal(0.0, 1.0);
          }
          orbifold_temporal_ref(pu, site) =
              orbifold_su3_algebra(coefficients);
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  real_t kinetic_energy() const {
    const auto pz = momentum_.spatial;
    const auto pu = momentum_.temporal;
    const auto dimensions = field_.dimensions;
    const size_t sites = wilson_site_count<compiled_rank>(dimensions);
    real_t result = 0.0;
    Kokkos::parallel_reduce(
        "orbifold_kinetic_energy", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear, real_t &local) {
          const auto site =
              wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            local += 0.5 * orbifold_matrix_norm_squared(
                               orbifold_spatial_ref(pz, site, j));
          }
          const SUN<3> p = orbifold_temporal_ref(pu, site);
          local -= trace(p * p).real();
        },
        result);
    Kokkos::fence();
    return result;
  }

  void integrate() {
    update_momenta(0.5 * hmc_params_.step_size);
    for (index_t step = 0; step < hmc_params_.steps; ++step) {
      update_positions(hmc_params_.step_size);
      if (step + 1 < hmc_params_.steps) {
        update_momenta(hmc_params_.step_size);
      }
    }
    update_momenta(0.5 * hmc_params_.step_size);
  }

  void negate_momenta() {
    const auto pz = momentum_.spatial;
    const auto pu = momentum_.temporal;
    const auto dimensions = field_.dimensions;
    const size_t sites = wilson_site_count<compiled_rank>(dimensions);
    Kokkos::parallel_for(
        "orbifold_negate_momenta", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear) {
          const auto site =
              wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            orbifold_spatial_ref(pz, site, j) *= -1.0;
          }
          orbifold_temporal_ref(pu, site) *= -1.0;
        });
    Kokkos::fence();
  }

  OrbifoldHMCResult step() {
    Kokkos::deep_copy(backup_.spatial, field_.spatial);
    Kokkos::deep_copy(backup_.temporal, field_.temporal);
    randomize_momenta();
    const real_t initial = orbifold_action(field_, action_params_) +
                           kinetic_energy();
    integrate();
    const real_t final =
        orbifold_action(field_, action_params_) + kinetic_energy();
    const real_t delta = final - initial;
    const real_t random =
        std::max(uniform_(host_rng_), std::numeric_limits<real_t>::min());
    const bool accepted = std::isfinite(delta) && std::log(random) < -delta;
    if (!accepted) {
      Kokkos::deep_copy(field_.spatial, backup_.spatial);
      Kokkos::deep_copy(field_.temporal, backup_.temporal);
      Kokkos::fence();
    }
    return OrbifoldHMCResult{accepted, initial, final, delta};
  }

  void update_momenta(const real_t step) {
    orbifold_force(field_, action_params_, force_);
    const auto pz = momentum_.spatial;
    const auto pu = momentum_.temporal;
    const auto gz = force_.spatial;
    const auto gu = force_.temporal;
    const auto dimensions = field_.dimensions;
    const size_t sites = wilson_site_count<compiled_rank>(dimensions);
    Kokkos::parallel_for(
        "orbifold_update_momenta", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear) {
          const auto site =
              wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            orbifold_spatial_ref(pz, site, j) -=
                orbifold_spatial_ref(gz, site, j) * step;
          }
          orbifold_temporal_ref(pu, site) +=
              orbifold_temporal_ref(gu, site) * step;
        });
    Kokkos::fence();
  }

  void update_positions(const real_t step) {
    const auto z = field_.spatial;
    const auto u = field_.temporal;
    const auto pz = momentum_.spatial;
    const auto pu = momentum_.temporal;
    const auto dimensions = field_.dimensions;
    const size_t sites = wilson_site_count<compiled_rank>(dimensions);
    Kokkos::parallel_for(
        "orbifold_update_positions", Kokkos::RangePolicy<>(0, sites),
        KOKKOS_LAMBDA(const size_t linear) {
          const auto site =
              wilson_linear_to_site<compiled_rank>(linear, dimensions);
#pragma unroll
          for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
            orbifold_spatial_ref(z, site, j) +=
                orbifold_spatial_ref(pz, site, j) * step;
          }
          orbifold_temporal_ref(u, site) =
              orbifold_exp_su3(orbifold_temporal_ref(pu, site) * step) *
              orbifold_temporal_ref(u, site);
        });
    Kokkos::fence();
  }

private:
  OrbifoldField field_;
  OrbifoldActionParams action_params_;
  OrbifoldHMCParams hmc_params_;
  OrbifoldField momentum_;
  OrbifoldField force_;
  OrbifoldField backup_;
  Kokkos::Random_XorShift64_Pool<> momentum_rng_;
  std::mt19937_64 host_rng_;
  std::uniform_real_distribution<real_t> uniform_;
};

} // namespace klft
