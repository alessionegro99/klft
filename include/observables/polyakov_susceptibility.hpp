#pragma once

#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "observables/polyakov_loop.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"

#include <Kokkos_Core.hpp>

namespace klft {

// Fourier amplitude of the per-site Polyakov-loop field at spatial momentum p:
//
//   A(p) = (1 / V_s) * sum_x exp(i p.x) P(x),
//
// following Bonati's polyakov_FT in yang-mills-Bonn (lib/gauge_conf_meas.c).
// The sum runs over the spatial sites x (the rank-1 spatial components of the
// loop origin); the temporal direction is rank-1 and is already wrapped inside
// each Polyakov loop, so it is not summed here.
template <size_t rank>
complex_t PolyakovFTAmplitude(
    const Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &local_polyakov,
    const Kokkos::Array<real_t, rank - 1> &momentum,
    const IndexArray<rank> &dimensions) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  complex_t total(0.0, 0.0);

  Kokkos::parallel_reduce(
      "PolyakovFTAmplitude", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t i, complex_t &lsum) {
        const auto site = linear_to_polyakov_origin<rank>(i, dimensions);
        real_t phase = 0.0;
#pragma unroll
        for (index_t d = 0; d < rank - 1; ++d) {
          phase += momentum[d] * static_cast<real_t>(site[d]);
        }
        const complex_t weight(Kokkos::cos(phase), Kokkos::sin(phase));
        lsum += weight * local_polyakov(i);
      },
      Kokkos::Sum<complex_t>(total));

  if (nSpatial > 0) {
    total *= 1.0 / static_cast<real_t>(nSpatial);
  }
  return total;
}

// Per-configuration primaries for the finite-size-scaling analysis of the
// deconfinement transition, from the spatial Fourier modes of the raw
// Polyakov-loop field:
//
//   G_0   = |A(0)|^2 = |P_bar|^2,            P_bar = (1/V_s) sum_x P(x);
//   G_cos = mean over the rank-1 spatial directions i of (Re A(p_min e_i))^2;
//   G_sin = mean over the rank-1 spatial directions i of (Im A(p_min e_i))^2;
//
// with p_min = 2 pi / L_i. For p != 0 the amplitude is complex,
//   A(p) = (1/V_s) [ sum_x cos(p.x) P(x) + i sum_x sin(p.x) P(x) ],
// so G_cos and G_sin are the cosine- and sine-transform structure factors and
// |A(p)|^2 = G_cos + G_sin per configuration (no cross term). Emitting them
// separately is reversible, so the analysis can pick the minimal-momentum
// convention without re-running:
//   - cosine structure factor  G_den = G_cos          (the reference convention);
//   - full power               G_den = G_cos + G_sin  (literal FT of G(x)).
// At p = 0 the sine part vanishes, so G_0 is already the cosine-only value.
//
// From the Monte Carlo ensemble averages the analysis stage forms:
//   - Binder cumulant of the Polyakov loop  U4 = <G_0^2> / <G_0>^2;
//   - second-moment correlation length
//     xi = sqrt(<G_0>/<G_den> - 1) / (2 sin(pi/L))   (equal spatial extents).
// By translation invariance <G_cos> = <G_sin> in equilibrium, so the ratio
// <G_sin>/<G_cos> is a free thermalization / momentum-grid check (-> 1).
// See the commented reference block in perform_measures_localobs_with_tracedef
// of yang-mills-Bonn (lib/gauge_conf_meas.c).
//
// Raw (un-multihit) Polyakov loops are used on purpose: |A(p)|^2 contains the
// diagonal x = y self-term, and a multihit estimator would bias it (loops at
// distinct spatial sites share no links, so only the diagonal is affected).
// This mirrors Bonati's raw polyvec feeding the momentum-space correlator.
template <size_t rank, size_t Nc, class RNG>
Kokkos::Array<real_t, 3> PolyakovSusceptibility(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const RNG &rng) {
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  LocalFieldType local_polyakov("susceptibility_polyakov", nSpatial);
  LocalPolyakovLoop<rank, Nc>(g_in, local_polyakov, rng);

  Kokkos::Array<real_t, rank - 1> zero_momentum;
  for (index_t d = 0; d < static_cast<index_t>(rank - 1); ++d) {
    zero_momentum[d] = 0.0;
  }
  const complex_t A0 =
      PolyakovFTAmplitude<rank>(local_polyakov, zero_momentum, dimensions);
  const real_t G0 = A0.real() * A0.real() + A0.imag() * A0.imag();

  const real_t twopi = 2.0 * Kokkos::numbers::pi_v<real_t>;
  real_t Gcos = 0.0;
  real_t Gsin = 0.0;
  for (index_t i = 0; i < static_cast<index_t>(rank - 1); ++i) {
    Kokkos::Array<real_t, rank - 1> momentum;
    for (index_t d = 0; d < static_cast<index_t>(rank - 1); ++d) {
      momentum[d] = 0.0;
    }
    momentum[i] = twopi / static_cast<real_t>(dimensions[i]);
    const complex_t Ai =
        PolyakovFTAmplitude<rank>(local_polyakov, momentum, dimensions);
    Gcos += Ai.real() * Ai.real();
    Gsin += Ai.imag() * Ai.imag();
  }
  if (rank > 1) {
    Gcos /= static_cast<real_t>(rank - 1);
    Gsin /= static_cast<real_t>(rank - 1);
  }

  return Kokkos::Array<real_t, 3>{G0, Gcos, Gsin};
}

} // namespace klft
