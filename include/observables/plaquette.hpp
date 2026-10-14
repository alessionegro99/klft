#pragma once
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"

namespace klft {

struct GaugePlaquetteComponents {
  real_t spatial;
  real_t temporal;
  // Fraction of stored sites supporting a temporal plaquette. Spatial
  // plaquettes include both fixed boundary planes; temporal ones span Nt links.
  real_t temporal_site_fraction = 1.0;
};

// Accumulate spatial-spatial and spatial-temporal plaquettes separately over
// one lattice site. KLFT consistently uses rank - 1 as the temporal direction.
template <size_t rank, size_t Nc> struct GaugePlaq {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  const IndexArray<rank> dimensions;
  const bool temporal_dirichlet;
  GaugePlaq(const GaugeFieldType &g_in, const IndexArray<rank> &dimensions,
            const bool temporal_dirichlet = false)
      : g_in(g_in), dimensions(dimensions),
        temporal_dirichlet(temporal_dirichlet) {}

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &site, real_t &spatial_sum,
             real_t &temporal_sum) const {
    SUN<Nc> lmu, lnu;

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = mu + 1; nu < Nd; ++nu) {
        if (temporal_dirichlet && nu == static_cast<index_t>(rank - 1) &&
            site[rank - 1] == dimensions[rank - 1] - 1) {
          continue;
        }
        lmu = g_in(site, mu) *
              g_in(shift_index_plus<rank, index_t>(site, mu, 1, dimensions),
                   nu);
        lnu = g_in(site, nu) *
              g_in(shift_index_plus<rank, index_t>(site, nu, 1, dimensions),
                   mu);
        const real_t value = trace(lmu * conj(lnu)).real();
        if (nu == static_cast<index_t>(rank - 1)) {
          temporal_sum += value;
        } else {
          spatial_sum += value;
        }
      }
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              real_t &spatial_sum,
                                              real_t &temporal_sum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1}, spatial_sum,
               temporal_sum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              real_t &spatial_sum,
                                              real_t &temporal_sum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2}, spatial_sum,
               temporal_sum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              real_t &spatial_sum,
                                              real_t &temporal_sum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2, i3}, spatial_sum,
               temporal_sum);
  }
};

template <size_t rank, size_t Nc>
GaugePlaquetteComponents GaugePlaquettes(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const bool normalize = true, const bool temporal_dirichlet = false) {
  real_t spatial = 0.0;
  real_t temporal = 0.0;
  const auto dimensions = g_in.dimensions;
  const bool open_time = temporal_dirichlet || gauge_uses_open_time(g_in);
  IndexArray<rank> start;
  IndexArray<rank> end;
  size_t nSites = 1;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
    nSites *= static_cast<size_t>(dimensions[i]);
  }

  Kokkos::parallel_reduce(
      "GaugePlaquettes", Policy<rank>(start, end),
      GaugePlaq<rank, Nc>(g_in, dimensions, open_time), spatial, temporal);

  const size_t nTemporalSites = open_time
      ? nSites - nSites / static_cast<size_t>(dimensions[rank - 1]) : nSites;

  if (normalize) {
    constexpr size_t nSpatialPlanes = (rank - 1) * (rank - 2) / 2;
    constexpr size_t nTemporalPlanes = rank - 1;
    if constexpr (nSpatialPlanes > 0) {
      spatial /= static_cast<real_t>(nSites * nSpatialPlanes * Nc);
    } else {
      spatial = 0.0;
    }
    temporal /= static_cast<real_t>(nTemporalSites * nTemporalPlanes * Nc);
  }

  return GaugePlaquetteComponents{spatial, temporal,
      static_cast<real_t>(nTemporalSites) / static_cast<real_t>(nSites)};
}

// Recombine normalized components using the numbers of physical plaquettes.
// In a slab, temporal planes have Nt layers and spatial planes Nt + 1.
template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION real_t
AverageGaugePlaquette(const GaugePlaquetteComponents &components) {
  constexpr size_t nSpatialPlanes = (rank - 1) * (rank - 2) / 2;
  constexpr size_t nTemporalPlanes = rank - 1;
  const real_t temporal_weight =
      static_cast<real_t>(nTemporalPlanes) * components.temporal_site_fraction;
  return (static_cast<real_t>(nSpatialPlanes) * components.spatial +
          temporal_weight * components.temporal) /
         (static_cast<real_t>(nSpatialPlanes) + temporal_weight);
}

template <size_t rank, size_t Nc>
real_t GaugePlaquette(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                      const bool normalize = true,
                      const bool temporal_dirichlet = false) {
  const auto components = GaugePlaquettes<rank, Nc>(g_in, normalize,
                                                   temporal_dirichlet);
  if (normalize) {
    return AverageGaugePlaquette<rank>(components);
  }
  return components.spatial + components.temporal;
}
} // namespace klft
