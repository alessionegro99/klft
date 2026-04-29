#pragma once
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/plaquette.hpp"

#include <stdexcept>

namespace klft {

// ------------------------------------------------------------
// Utility: number of plaquettes on a rank-dimensional lattice
// ------------------------------------------------------------
template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION real_t
NumPlaquettes(const IndexArray<rank> &dimensions) {
  real_t nsites = 1.0;
#pragma unroll
  for (index_t d = 0; d < rank; ++d) {
    nsites *= static_cast<real_t>(dimensions[d]);
  }
  return nsites * static_cast<real_t>(rank * (rank - 1) / 2);
}

// ------------------------------------------------------------
// Reduced Wilson action from normalized average plaquette
//
// GaugePlaquette(..., true) returns
//   < Re Tr U_p / Nc >
// averaged over sites and plaquette orientations.
//
// This returns the beta-independent reduced action
//   E = Np * (1 - <Re Tr U_p / Nc>)
// ------------------------------------------------------------
template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION real_t ReducedWilsonActionFromAvgPlaquette(
    const real_t avg_plaq, const IndexArray<rank> &dimensions) {
  return NumPlaquettes(dimensions) * (1.0 - avg_plaq);
}

// ------------------------------------------------------------
// Validate one-level dyadic child offset
//
// For one-step blocking, each offset entry must be 0 or 1.
// ------------------------------------------------------------
template <size_t rank>
inline void ValidateDyadicChildOffset(const IndexArray<rank> &child_offset) {
  for (index_t d = 0; d < rank; ++d) {
    if (child_offset[d] != 0 && child_offset[d] != 1) {
      throw std::runtime_error(
          "child_offset entries must be 0 or 1 for one-level dyadic blocking.");
    }
  }
}

// ------------------------------------------------------------
// Functor: blocked plaquette on one dyadic child grid
//
// child_offset[d] should be 0 or 1.
// The coarse site X is mapped to fine site x = 2X + child_offset.
//
// Blocked links:
//   U'_mu(X) = U_mu(x) U_mu(x + mu)
// and then the blocked plaquette is built from U'.
// ------------------------------------------------------------
template <size_t rank, size_t Nc> struct BlockedGaugePlaqOneLevel {
  constexpr static const size_t Nd = rank;

  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  const GaugeFieldType g_in;
  const IndexArray<rank> fine_dimensions;
  const IndexArray<rank> child_offset;

  BlockedGaugePlaqOneLevel(const GaugeFieldType &g_in,
                           const IndexArray<rank> &fine_dimensions,
                           const IndexArray<rank> &child_offset)
      : g_in(g_in), fine_dimensions(fine_dimensions),
        child_offset(child_offset) {}

  KOKKOS_FORCEINLINE_FUNCTION
  SUN<Nc> blocked_link(const IndexArray<rank> &x, const index_t mu) const {
    auto x_plus_mu = shift_index_plus<rank, index_t>(x, mu, 1, fine_dimensions);
    return g_in(x, mu) * g_in(x_plus_mu, mu);
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &coarse_site,
             complex_t &lsum) const {
    SUN<Nc> lmu, lnu;
    complex_t tmunu(0.0, 0.0);

    // corresponding fine coordinates x = 2X + offset
    IndexArray<rank> x;
#pragma unroll
    for (index_t d = 0; d < rank; ++d) {
      x[d] = 2 * coarse_site[d] + child_offset[d];
    }

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          auto x_plus_2mu =
              shift_index_plus<rank, index_t>(x, mu, 2, fine_dimensions);
          auto x_plus_2nu =
              shift_index_plus<rank, index_t>(x, nu, 2, fine_dimensions);

          // lmu = U'_mu(X) U'_nu(X + 2 mu)
          lmu = blocked_link(x, mu) * blocked_link(x_plus_2mu, nu);

          // lnu = U'_nu(X) U'_mu(X + 2 nu)
          lnu = blocked_link(x, nu) * blocked_link(x_plus_2nu, mu);

          tmunu += trace(lmu * conj(lnu));
        }
      }
    }

    lsum += tmunu;
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              complex_t &lsum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              complex_t &lsum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              complex_t &lsum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2, i3}, lsum);
  }
};

// ------------------------------------------------------------
// Average blocked plaquette on one dyadic child grid
// ------------------------------------------------------------
template <size_t rank, size_t Nc>
real_t BlockedGaugePlaquetteOneLevel(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const IndexArray<rank> &child_offset, const bool normalize = true) {

  constexpr static const size_t Nd = rank;

  ValidateDyadicChildOffset(child_offset);

  const auto &fine_dims = g_in.field.layout().dimension;

  IndexArray<rank> coarse_start;
  IndexArray<rank> coarse_end;
  IndexArray<rank> fine_dimensions;
  size_t nCoarseSites = 1;

  for (index_t d = 0; d < rank; ++d) {
    fine_dimensions[d] = fine_dims[d];
    coarse_start[d] = 0;

    if (fine_dims[d] % 2 != 0) {
      throw std::runtime_error(
          "BlockedGaugePlaquetteOneLevel requires even lattice extents.");
    }
    coarse_end[d] = fine_dims[d] / 2;
    nCoarseSites *= static_cast<size_t>(coarse_end[d]);
  }

  complex_t plaq(0.0, 0.0);
  Kokkos::parallel_reduce(
      "BlockedGaugePlaquetteOneLevel", Policy<rank>(coarse_start, coarse_end),
      BlockedGaugePlaqOneLevel<rank, Nc>(g_in, fine_dimensions, child_offset),
      Kokkos::Sum<complex_t>(plaq));

  if (normalize) {
    real_t norm = static_cast<real_t>(nCoarseSites);
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}

// ------------------------------------------------------------
// Result bundle
// ------------------------------------------------------------
template <size_t rank> struct NestedWilsonActionResult {
  real_t plaq_V;     // avg plaquette on full lattice
  real_t plaq_child; // avg plaquette on blocked child lattice
  real_t E_V;        // reduced action on full lattice
  real_t E_child;    // reduced action on blocked child lattice
};

// ------------------------------------------------------------
// Measure full-grid and one-level blocked-child reduced actions
// ------------------------------------------------------------
template <size_t rank, size_t Nc>
NestedWilsonActionResult<rank> MeasureNestedWilsonActionsOneLevel(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const IndexArray<rank> &child_offset) {

  ValidateDyadicChildOffset(child_offset);

  NestedWilsonActionResult<rank> out{};

  const auto &dims_raw = g_in.field.layout().dimension;
  IndexArray<rank> dims;
  IndexArray<rank> child_dims;

  for (index_t d = 0; d < rank; ++d) {
    dims[d] = dims_raw[d];
    if (dims[d] % 2 != 0) {
      throw std::runtime_error(
          "MeasureNestedWilsonActionsOneLevel requires even lattice extents.");
    }
    child_dims[d] = dims[d] / 2;
  }

  out.plaq_V = GaugePlaquette<rank, Nc>(g_in, true);
  out.plaq_child =
      BlockedGaugePlaquetteOneLevel<rank, Nc>(g_in, child_offset, true);

  out.E_V = ReducedWilsonActionFromAvgPlaquette(out.plaq_V, dims);
  out.E_child = ReducedWilsonActionFromAvgPlaquette(out.plaq_child, child_dims);

  return out;
}

} // namespace klft
