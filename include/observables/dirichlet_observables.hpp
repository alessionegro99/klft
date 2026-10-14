#pragma once

#include "core/temporal_dirichlet.hpp"
#include "groups/group_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace klft {

template <size_t Nc>
using SlabHolonomyField =
    Kokkos::View<SUN<Nc> **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc, class GaugeFieldType> struct BuildSlabHolonomy3D {
  GaugeFieldType gauge;
  SlabHolonomyField<Nc> holonomy;
  const index_t slab_thickness;

  BuildSlabHolonomy3D(const GaugeFieldType &gauge,
                      const SlabHolonomyField<Nc> &holonomy)
      : gauge(gauge), holonomy(holonomy),
        slab_thickness(slab_thickness_in_links<3>(gauge.dimensions)) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t x,
                                              const index_t y) const {
    SUN<Nc> product = identitySUN<Nc>();
    for (index_t t = 0; t < slab_thickness; ++t) {
      product *= gauge(IndexArray<3>{x, y, t}, temporal_direction<3>());
    }
    holonomy(x, y) = product;
  }
};

// G(x) contains exactly the Nt temporal links between the two fixed surfaces.
template <size_t Nc>
SlabHolonomyField<Nc> SlabHolonomy(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge) {
  const index_t nx = gauge.dimensions[0];
  const index_t ny = gauge.dimensions[1];
  SlabHolonomyField<Nc> holonomy("slab_holonomy", nx, ny);
  Kokkos::parallel_for("build_slab_holonomy",
                       Policy<2>(IndexArray<2>{0, 0},
                                 IndexArray<2>{nx, ny}),
                       BuildSlabHolonomy3D<Nc,
                                            typename DeviceGaugeFieldType<
                                                3, Nc>::type>(gauge,
                                                              holonomy));
  Kokkos::fence();
  return holonomy;
}

struct DirichletCorrelatorPoint {
  index_t separation;
  real_t chiral_correlator;
  real_t legacy_double_trace;
  real_t effective_log_slope;
};

struct DirichletHolonomyMeasurement {
  std::vector<DirichletCorrelatorPoint> correlator;
  real_t mean_trace_real;
  real_t mean_trace_imag;
  real_t fourier_zero;
  real_t fourier_pmin;
  real_t second_moment_xi;
};

struct DirichletPlaquetteProfileMeasurement {
  // Entries are {slice coordinate, normalized Re Tr plaquette / Nc}.
  std::vector<Kokkos::Array<real_t, 2>> spatial;
  std::vector<Kokkos::Array<real_t, 2>> temporal;
};

struct BoundaryWilsonLoopMeasurement {
  // Entries are {R, H, normalized Re Tr W_b(R,H) / Nc}.
  std::vector<Kokkos::Array<real_t, 3>> loops;
};

struct DirichletBulkPolyakovMeasurement {
  // Entries are {temporal separation R, raw double-trace correlator}.
  std::vector<Kokkos::Array<real_t, 2>> correlator;
};

namespace dirichlet_observable_detail {

KOKKOS_FORCEINLINE_FUNCTION index_t wrap_spatial(const index_t x,
                                                 const index_t extent) {
  const index_t result = x % extent;
  return result < 0 ? result + extent : result;
}

template <size_t Nc>
real_t normalized_pair(const SUN<Nc> &left, const SUN<Nc> &right) {
  return trace(left * conj(right)).real() / static_cast<real_t>(Nc);
}

template <size_t Nc>
real_t legacy_pair(const SUN<Nc> &left, const SUN<Nc> &right) {
  const complex_t left_trace = trace(left);
  const complex_t right_trace = trace(right);
  return (left_trace * Kokkos::conj(right_trace)).real();
}

template <size_t Nc, class HostGauge>
SUN<Nc> boundary_wilson_at(const HostGauge &gauge, const index_t x,
                           const index_t y, const index_t spatial_direction,
                           const index_t separation, const index_t height,
                           const IndexArray<3> &dimensions) {
  IndexArray<3> site{x, y, 0};
  SUN<Nc> loop = identitySUN<Nc>();

  for (index_t r = 0; r < separation; ++r) {
    loop *= gauge(site[0], site[1], site[2], spatial_direction);
    site[spatial_direction] = wrap_spatial(
        site[spatial_direction] + 1, dimensions[spatial_direction]);
  }
  for (index_t h = 0; h < height; ++h) {
    loop *= gauge(site[0], site[1], site[2], temporal_direction<3>());
    ++site[temporal_direction<3>()];
  }
  for (index_t r = 0; r < separation; ++r) {
    site[spatial_direction] = wrap_spatial(
        site[spatial_direction] - 1, dimensions[spatial_direction]);
    loop *= conj(gauge(site[0], site[1], site[2], spatial_direction));
  }
  for (index_t h = 0; h < height; ++h) {
    --site[temporal_direction<3>()];
    loop *=
        conj(gauge(site[0], site[1], site[2], temporal_direction<3>()));
  }
  return loop;
}

template <size_t Nc, class HostGauge>
KOKKOS_INLINE_FUNCTION real_t plaquette_at(const HostGauge &gauge, const IndexArray<3> &site,
                    const index_t mu, const index_t nu,
                    const IndexArray<3> &dimensions) {
  auto site_plus_mu = site;
  auto site_plus_nu = site;
  site_plus_mu[mu] = (site_plus_mu[mu] + 1) % dimensions[mu];
  site_plus_nu[nu] = (site_plus_nu[nu] + 1) % dimensions[nu];
  const SUN<Nc> left =
      gauge(site[0], site[1], site[2], mu) *
      gauge(site_plus_mu[0], site_plus_mu[1], site_plus_mu[2], nu);
  const SUN<Nc> right =
      gauge(site[0], site[1], site[2], nu) *
      gauge(site_plus_nu[0], site_plus_nu[1], site_plus_nu[2], mu);
  return trace(left * conj(right)).real() / static_cast<real_t>(Nc);
}

} // namespace dirichlet_observable_detail

// Under G(x) -> L G(x) R^dagger,
// G(x)G(y)^dagger -> L G(x)G(y)^dagger L^dagger; cyclicity makes its trace
// invariant. Tr G alone has no such invariance. Only 0 <= R <= floor(Ns/2)
// is returned, avoiding the exactly redundant R and Ns-R covariance rows.
template <size_t Nc>
DirichletHolonomyMeasurement MeasureDirichletHolonomy(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const index_t requested_max_r = 0) {
  static_assert(Nc == 2,
                "Dirichlet PCM observables are currently defined for SU(2).");
  const index_t nx = gauge.dimensions[0];
  const index_t ny = gauge.dimensions[1];
  const index_t independent_max = std::min(nx, ny) / 2;
  const index_t max_r =
      requested_max_r == 0 ? independent_max : requested_max_r;
  if (max_r < 0 || max_r > independent_max) {
    throw std::runtime_error(
        "dirichlet_correlator_max_r exceeds the independent half-range.");
  }

  const auto holonomy = SlabHolonomy<Nc>(gauge);
  const real_t volume = static_cast<real_t>(nx) * static_cast<real_t>(ny);

  // SU(2): Re Tr(A B^dagger)/2 = sum_{a=0}^3 q_a(A) q_a(B),
  // and Tr(A) Tr(B)^* = 4 q_0(A) q_0(B). Reduce on the device and transfer
  // O(R_max) scalars, instead of copying the field for a serial O(Ns^3) loop.
  // Hierarchical reduction follows Kokkos' Hierarchical_Parallelism tutorial.
  using Team = Kokkos::TeamPolicy<>::member_type;
  constexpr bool x_fast = std::is_same_v<
      typename SlabHolonomyField<Nc>::array_layout, Kokkos::LayoutLeft>;
  Kokkos::View<real_t *[2]> sums("slab_correlator_sums", max_r + 1);
  Kokkos::parallel_for("slab_correlator",
      Kokkos::TeamPolicy<>(2 * (max_r + 1), Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const index_t r = team.league_rank() / 2;
        const index_t kind = team.league_rank() % 2;
        real_t sum = 0.;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nx * ny),
            [&](const index_t i, real_t &value) {
              const index_t x = x_fast ? i % nx : i / ny;
              const index_t y = x_fast ? i / nx : i % ny;
              const SU2 a = holonomy(x, y);
              const SU2 b = holonomy((x + r) % nx, y);
              const SU2 c = holonomy(x, (y + r) % ny);
              if (kind == 0) {
                for (index_t k = 0; k < 4; ++k)
                  value += a.comp[k] * (b.comp[k] + c.comp[k]);
              } else {
                value += 4. * a.comp[0] * (b.comp[0] + c.comp[0]);
              }
            }, sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { sums(r, kind) = sum; });
      });
  // Four zero-momentum sums and 2 directions x 4 components x (cos,sin).
  Kokkos::View<real_t *> modes("slab_fourier_modes", 20);
  Kokkos::parallel_for("slab_fourier",
      Kokkos::TeamPolicy<>(20, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Team &team) {
        const index_t mode = team.league_rank();
        const index_t component = mode < 4 ? mode : ((mode - 4) % 8) / 2;
        const index_t direction = (mode - 4) / 8;
        real_t sum = 0.;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nx * ny),
            [&](const index_t i, real_t &value) {
              const index_t x = x_fast ? i % nx : i / ny;
              const index_t y = x_fast ? i / nx : i % ny;
              real_t factor = 1.;
              if (mode >= 4) {
                const real_t angle = 2. * Kokkos::numbers::pi_v<real_t> *
                    (direction == 0 ? real_t(x) / nx : real_t(y) / ny);
                factor = mode % 2 == 0 ? Kokkos::cos(angle) : Kokkos::sin(angle);
              }
              value += factor * holonomy(x, y).comp[component];
            }, sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { modes(mode) = sum; });
      });
  const auto host_sums = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums);
  const auto host_modes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), modes);

  DirichletHolonomyMeasurement result;
  result.mean_trace_real = 2. * host_modes(0) / volume;
  result.mean_trace_imag = 0.;
  result.correlator.reserve(static_cast<size_t>(max_r + 1));
  for (index_t separation = 0; separation <= max_r; ++separation) {
    const real_t normalization = 2.0 * volume;
    const real_t primary =
        separation == 0 ? 1.0 : host_sums(separation, 0) / normalization;
    result.correlator.push_back(DirichletCorrelatorPoint{
        separation, primary, host_sums(separation, 1) / normalization,
        std::numeric_limits<real_t>::quiet_NaN()});
  }
  for (size_t r = 0; r + 1 < result.correlator.size(); ++r) {
    const real_t current = result.correlator[r].chiral_correlator;
    const real_t next = result.correlator[r + 1].chiral_correlator;
    if (current > 0.0 && next > 0.0) {
      result.correlator[r].effective_log_slope = std::log(current / next);
    }
  }

  // For SU(2), (1/2) Re Tr[G(x)G(y)^dagger] is the Euclidean dot product of
  // the four real quaternion components. Its Fourier transform therefore has
  // the positive structure-factor representation used here.
  real_t zero = 0.0;
  for (index_t component = 0; component < 4; ++component)
    zero += host_modes(component) * host_modes(component);
  zero /= volume;

  real_t pmin_sum = 0.0;
  for (index_t direction = 0; direction < 2; ++direction) {
    real_t direction_value = 0.0;
    for (index_t component = 0; component < 4; ++component) {
      const index_t mode = 4 + 8 * direction + 2 * component;
      const real_t real_sum = host_modes(mode);
      const real_t imag_sum = host_modes(mode + 1);
      direction_value += real_sum * real_sum + imag_sum * imag_sum;
    }
    pmin_sum += direction_value / volume;
  }
  result.fourier_zero = zero;
  result.fourier_pmin = pmin_sum / 2.0;
  result.second_moment_xi = std::numeric_limits<real_t>::quiet_NaN();
  if (nx == ny && result.fourier_pmin > 0.0 &&
      result.fourier_zero >= result.fourier_pmin) {
    result.second_moment_xi =
        std::sqrt(result.fourier_zero / result.fourier_pmin - 1.0) /
        (2.0 * std::sin(Kokkos::numbers::pi_v<real_t> /
                        static_cast<real_t>(nx)));
  }
  return result;
}

template <size_t Nc>
DirichletPlaquetteProfileMeasurement MeasureDirichletPlaquetteProfiles(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge) {
  const auto dimensions = gauge.dimensions;
  const index_t nx = dimensions[0];
  const index_t ny = dimensions[1];
  const index_t slab_thickness = slab_thickness_in_links<3>(dimensions);
  const real_t spatial_volume =
      static_cast<real_t>(nx) * static_cast<real_t>(ny);

  DirichletPlaquetteProfileMeasurement result;
  result.spatial.reserve(static_cast<size_t>(slab_thickness + 1));
  result.temporal.reserve(static_cast<size_t>(slab_thickness));
  for (index_t t = 0; t <= slab_thickness; ++t) {
    real_t sum = 0.0;
    Kokkos::parallel_reduce("slab_spatial_profile",
        Policy<2>(IndexArray<2>{}, IndexArray<2>{nx, ny}),
        KOKKOS_LAMBDA(const index_t x, const index_t y, real_t &value) {
          value += dirichlet_observable_detail::plaquette_at<Nc>(
              gauge, IndexArray<3>{x, y, t}, 0, 1, dimensions);
        }, sum);
    result.spatial.push_back(
        Kokkos::Array<real_t, 2>{static_cast<real_t>(t),
                                 sum / spatial_volume});
  }
  for (index_t t = 0; t < slab_thickness; ++t) {
    real_t sum = 0.0;
    Kokkos::parallel_reduce("slab_temporal_profile",
        Policy<2>(IndexArray<2>{}, IndexArray<2>{nx, ny}),
        KOKKOS_LAMBDA(const index_t x, const index_t y, real_t &value) {
        for (index_t spatial_direction = 0; spatial_direction < 2;
             ++spatial_direction) {
          value += dirichlet_observable_detail::plaquette_at<Nc>(
              gauge, IndexArray<3>{x, y, t}, spatial_direction,
              temporal_direction<3>(), dimensions);
        }
        }, sum);
    result.temporal.push_back(Kokkos::Array<real_t, 2>{
        static_cast<real_t>(t) + 0.5, sum / (2.0 * spatial_volume)});
  }
  return result;
}

template <size_t Nc>
BoundaryWilsonLoopMeasurement MeasureBoundaryWilsonLoops(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const bool all_heights, const index_t requested_max_r = 0) {
  const auto dimensions = gauge.dimensions;
  const index_t nx = dimensions[0];
  const index_t ny = dimensions[1];
  const index_t slab_thickness = slab_thickness_in_links<3>(dimensions);
  const index_t max_r =
      requested_max_r == 0 ? std::min(nx, ny) : requested_max_r;
  if (max_r < 0 || max_r > std::min(nx, ny)) {
    throw std::runtime_error(
        "boundary_wilson_max_r exceeds the shortest spatial extent.");
  }

  std::vector<index_t> heights;
  if (all_heights) {
    for (index_t height = 1; height <= slab_thickness; ++height) {
      heights.push_back(height);
    }
  } else {
    heights.push_back(1);
    if (slab_thickness != 1) {
      heights.push_back(slab_thickness);
    }
  }

  const auto host = copy_gauge_to_host(gauge);
  const real_t volume = static_cast<real_t>(nx) * static_cast<real_t>(ny);
  std::vector<real_t> sums(
      heights.size() * static_cast<size_t>(max_r + 1), 0.0);

  // For a fixed origin, direction, and height, extend the lower and upper
  // spatial transporters by one link when R increases. This evaluates all
  // rectangles in O(R_max), instead of rebuilding each perimeter in
  // O(R_max^2). The temporal transporters are short (at most Nt links).
  for (index_t x = 0; x < nx; ++x) {
    for (index_t y = 0; y < ny; ++y) {
      for (index_t direction = 0; direction < 2; ++direction) {
        for (size_t height_index = 0; height_index < heights.size();
             ++height_index) {
          const index_t height = heights[height_index];
          SUN<Nc> left_vertical = identitySUN<Nc>();
          for (index_t h = 0; h < height; ++h) {
            left_vertical *= host(x, y, h, temporal_direction<3>());
          }

          SUN<Nc> lower = identitySUN<Nc>();
          SUN<Nc> upper_reverse = identitySUN<Nc>();
          for (index_t separation = 0; separation <= max_r; ++separation) {
            IndexArray<3> endpoint{x, y, 0};
            endpoint[direction] = dirichlet_observable_detail::wrap_spatial(
                endpoint[direction] + separation,
                dimensions[direction]);
            SUN<Nc> right_vertical = identitySUN<Nc>();
            for (index_t h = 0; h < height; ++h) {
              endpoint[temporal_direction<3>()] = h;
              right_vertical *=
                  host(endpoint[0], endpoint[1], endpoint[2],
                       temporal_direction<3>());
            }
            const SUN<Nc> loop = lower * right_vertical * upper_reverse *
                                 conj(left_vertical);
            sums[height_index * static_cast<size_t>(max_r + 1) +
                 static_cast<size_t>(separation)] +=
                trace(loop).real() / static_cast<real_t>(Nc);

            if (separation == max_r) {
              continue;
            }
            IndexArray<3> lower_link{x, y, 0};
            lower_link[direction] =
                dirichlet_observable_detail::wrap_spatial(
                lower_link[direction] + separation,
                dimensions[direction]);
            lower *= host(lower_link[0], lower_link[1], lower_link[2],
                          direction);
            auto upper_link = lower_link;
            upper_link[temporal_direction<3>()] = height;
            upper_reverse =
                conj(host(upper_link[0], upper_link[1], upper_link[2],
                          direction)) *
                upper_reverse;
          }
        }
      }
    }
  }

  BoundaryWilsonLoopMeasurement result;
  result.loops.reserve(heights.size() * static_cast<size_t>(max_r + 1));
  for (size_t height_index = 0; height_index < heights.size();
       ++height_index) {
    const index_t height = heights[height_index];
    for (index_t separation = 0; separation <= max_r; ++separation) {
      result.loops.push_back(Kokkos::Array<real_t, 3>{
          static_cast<real_t>(separation), static_cast<real_t>(height),
          sums[height_index * static_cast<size_t>(max_r + 1) +
               static_cast<size_t>(separation)] /
              (2.0 * volume)});
    }
  }
  return result;
}

// Thesis quality check, Eqs. (4.4)-(4.5): a Polyakov loop wraps spatial
// direction 0 and the two loops are placed symmetrically about the slab
// midpoint.  Only separations with the same parity as Nt have integer slice
// coordinates.  The raw product of traces is retained for thesis comparison.
template <size_t Nc>
DirichletBulkPolyakovMeasurement MeasureDirichletBulkPolyakovCorrelator(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const index_t requested_max_r = 0) {
  const auto dimensions = gauge.dimensions;
  const index_t nx = dimensions[0];
  const index_t ny = dimensions[1];
  const index_t thickness = slab_thickness_in_links<3>(dimensions);
  const index_t max_r = requested_max_r == 0 ? thickness : requested_max_r;
  if (max_r < 0 || max_r > thickness) {
    throw std::runtime_error(
        "dirichlet_bulk_polyakov_max_r exceeds the slab thickness.");
  }

  const auto host = copy_gauge_to_host(gauge);
  DirichletBulkPolyakovMeasurement result;
  for (index_t separation = thickness % 2; separation <= max_r;
       separation += 2) {
    const index_t lower_t = (thickness - separation) / 2;
    const index_t upper_t = lower_t + separation;
    real_t sum = 0.0;
    for (index_t y = 0; y < ny; ++y) {
      SUN<Nc> lower = identitySUN<Nc>();
      SUN<Nc> upper = identitySUN<Nc>();
      for (index_t x = 0; x < nx; ++x) {
        lower *= host(x, y, lower_t, 0);
        upper *= host(x, y, upper_t, 0);
      }
      sum += (trace(lower) * Kokkos::conj(trace(upper))).real();
    }
    result.correlator.push_back(Kokkos::Array<real_t, 2>{
        static_cast<real_t>(separation), sum / static_cast<real_t>(ny)});
  }
  return result;
}

} // namespace klft
