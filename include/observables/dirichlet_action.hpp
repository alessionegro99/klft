#pragma once

#include "observables/dirichlet_observables.hpp"
#include "observables/plaquette.hpp"

namespace klft {

// Wilson convention used by KLFT updates:
// S = -(beta/Nc) sum_p Re Tr U_p. Additive constants are retained here so
// factorization against the stored periodic field can be tested directly.
template <size_t Nc>
real_t DirichletSlabWilsonAction(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const real_t beta) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        gauge.field);
  const auto dimensions = gauge.dimensions;
  const index_t slab_thickness = slab_thickness_in_links<3>(dimensions);
  real_t normalized_plaquette_sum = 0.0;

  for (index_t t = 0; t <= slab_thickness; ++t) {
    for (index_t x = 0; x < dimensions[0]; ++x) {
      for (index_t y = 0; y < dimensions[1]; ++y) {
        const IndexArray<3> site{x, y, t};
        normalized_plaquette_sum +=
            dirichlet_observable_detail::plaquette_at<Nc>(
                host, site, 0, 1, dimensions);
        if (t < slab_thickness) {
          normalized_plaquette_sum +=
              dirichlet_observable_detail::plaquette_at<Nc>(
                  host, site, 0, temporal_direction<3>(), dimensions);
          normalized_plaquette_sum +=
              dirichlet_observable_detail::plaquette_at<Nc>(
                  host, site, 1, temporal_direction<3>(), dimensions);
        }
      }
    }
  }
  return -beta * normalized_plaquette_sum;
}

template <size_t Nc>
real_t PeriodicWilsonAction(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const real_t beta) {
  return -(beta / static_cast<real_t>(Nc)) *
         GaugePlaquette<3, Nc>(gauge, false);
}

// The temporal links based at t=Nt close the periodic storage from the upper
// to the lower fixed surface. Their plaquettes form a separate 2D PCM.
template <size_t Nc>
real_t WrappingPCMAction(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const real_t beta) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        gauge.field);
  const auto dimensions = gauge.dimensions;
  const index_t top = upper_boundary_slice<3>(dimensions);
  real_t sum = 0.0;
  for (index_t x = 0; x < dimensions[0]; ++x) {
    for (index_t y = 0; y < dimensions[1]; ++y) {
      for (index_t direction = 0; direction < 2; ++direction) {
        IndexArray<3> neighbor{x, y, top};
        neighbor[direction] =
            (neighbor[direction] + 1) % dimensions[direction];
        const SUN<Nc> left = host(x, y, top, temporal_direction<3>());
        const SUN<Nc> right = host(neighbor[0], neighbor[1], neighbor[2],
                                   temporal_direction<3>());
        sum += trace(left * conj(right)).real() /
               static_cast<real_t>(Nc);
      }
    }
  }
  return -beta * sum;
}

template <size_t Nc>
real_t PCMAction(const SlabHolonomyField<Nc> &spins, const real_t beta) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        spins);
  const index_t nx = static_cast<index_t>(spins.extent(0));
  const index_t ny = static_cast<index_t>(spins.extent(1));
  real_t sum = 0.0;
  for (index_t x = 0; x < nx; ++x) {
    for (index_t y = 0; y < ny; ++y) {
      sum += trace(host(x, y) * conj(host((x + 1) % nx, y))).real() /
             static_cast<real_t>(Nc);
      sum += trace(host(x, y) * conj(host(x, (y + 1) % ny))).real() /
             static_cast<real_t>(Nc);
    }
  }
  return -beta * sum;
}

template <size_t Nc>
real_t FixedBoundarySpatialActionConstant(
    const typename DeviceGaugeFieldType<3, Nc>::type &gauge,
    const real_t beta) {
  const real_t spatial_volume = static_cast<real_t>(gauge.dimensions[0]) *
                                static_cast<real_t>(gauge.dimensions[1]);
  return -2.0 * beta * spatial_volume;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t WilsonLocalActionDifference(
    const SUN<Nc> &old_link, const SUN<Nc> &new_link,
    const SUN<Nc> &staple, const real_t beta) {
  return -(beta / static_cast<real_t>(Nc)) *
         (trace(new_link * staple).real() -
          trace(old_link * staple).real());
}

} // namespace klft
