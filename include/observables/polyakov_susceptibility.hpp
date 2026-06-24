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
//   A(p) = [1 / ((rank - 1) V_s)] * sum_x exp(i p.x) P(x),
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
  // Match the yang-mills-Bonn-lverzich normalization, which divides the Fourier
  // amplitude by the number of spatial directions (STDIM-1 = rank-1) in
  // polyakov_FT (lib/gauge_conf_meas.c: `* d_inv_space_vol / (STDIM-1)`).
  // This factor cancels in xi and the Binder cumulant (pure ratios) but is
  // needed for G_0, G_pmin and the susceptibility chi to match lverzich
  // value-by-value. (Upstream yang-mills-Bonn leaves it out: `// /(STDIM-1) ?`.)
  if (rank > 1) {
    total *= 1.0 / static_cast<real_t>(rank - 1);
  }
  return total;
}

// Zero- and minimal-momentum Polyakov-loop correlators G(p) = |A(p)|^2, the two
// per-configuration primaries that feed the finite-size-scaling analysis of the
// deconfinement transition:
//
//   G_0    = G(p = 0) = |P_bar|^2 / (rank - 1)^2,
//            P_bar = (1/V_s) sum_x P(x);
//   G_pmin = mean over the rank-1 spatial directions i of G(p_min e_i),
//            with p_min = 2 pi / L_i.
//
// From their Monte Carlo ensemble averages the analysis stage forms:
//   - Binder cumulant of the Polyakov loop  U4 = <G_0^2> / <G_0>^2
//     (= <m^4>/<m^2>^2 with m^2 = G_0);
//   - second-moment correlation length
//     xi = sqrt(<G_0>/<G_pmin> - 1) / (2 sin(pi/L))   (equal spatial extents).
// See the commented reference block in perform_measures_localobs_with_tracedef
// of yang-mills-Bonn (lib/gauge_conf_meas.c).
//
// Raw (un-multihit) Polyakov loops are used on purpose: |A(p)|^2 contains the
// diagonal x = y self-term, and a multihit estimator would bias it (loops at
// distinct spatial sites share no links, so only the diagonal is affected),
// shifting both G_0 and G_pmin. This mirrors Bonati's raw polyvec feeding the
// momentum-space correlator, whereas the real-space correlator multihits only
// the well-separated R >= 2 contributions.
//
// The xi prefactor 2 sin(pi/L) assumes equal spatial extents (cubic spatial
// volume), as used in the FSS study; G_pmin is then a clean average over the
// equivalent spatial directions.
template <size_t rank, size_t Nc, class RNG>
Kokkos::Array<real_t, 2> PolyakovSusceptibility(
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
  real_t Gpmin = 0.0;
  for (index_t i = 0; i < static_cast<index_t>(rank - 1); ++i) {
    Kokkos::Array<real_t, rank - 1> momentum;
    for (index_t d = 0; d < static_cast<index_t>(rank - 1); ++d) {
      momentum[d] = 0.0;
    }
    momentum[i] = twopi / static_cast<real_t>(dimensions[i]);
    const complex_t Ai =
        PolyakovFTAmplitude<rank>(local_polyakov, momentum, dimensions);
    Gpmin += Ai.real() * Ai.real() + Ai.imag() * Ai.imag();
  }
  if (rank > 1) {
    Gpmin /= static_cast<real_t>(rank - 1);
  }

  return Kokkos::Array<real_t, 2>{G0, Gpmin};
}

} // namespace klft
