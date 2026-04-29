#pragma once
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"

namespace klft {

// Accumulate local plaquettes over one lattice site.
template <size_t rank, size_t Nc> struct GaugePlaq {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  const IndexArray<rank> dimensions;
  GaugePlaq(const GaugeFieldType &g_in, const IndexArray<rank> &dimensions)
      : g_in(g_in), dimensions(dimensions) {}

  KOKKOS_FORCEINLINE_FUNCTION
  complex_t plaquette_at_site(const Kokkos::Array<index_t, rank> &site) const {
    SUN<Nc> lmu, lnu;
    complex_t tmunu(0.0, 0.0);

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          lmu = g_in(site, mu) *
                g_in(shift_index_plus<rank, index_t>(site, mu, 1, dimensions),
                     nu);
          lnu = g_in(site, nu) *
                g_in(shift_index_plus<rank, index_t>(site, nu, 1, dimensions),
                     mu);
          tmunu += trace(lmu * conj(lnu));
        }
      }
    }
    return tmunu;
  }

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &site,
             complex_t &lsum) const {
    lsum += plaquette_at_site(site);
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

template <size_t rank, size_t Nc>
real_t GaugePlaquette(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                      const bool normalize = true) {
  constexpr static const size_t Nd = rank;
  complex_t plaq = 0.0;
  const auto dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  size_t nSites = 1;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
    nSites *= static_cast<size_t>(dimensions[i]);
  }

  Kokkos::parallel_reduce(
      "GaugePlaquette", Policy<rank>(start, end),
      GaugePlaq<rank, Nc>(g_in, dimensions), Kokkos::Sum<complex_t>(plaq));

  // normalization
  if (normalize) {
    real_t norm = static_cast<real_t>(nSites);
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}
} // namespace klft
