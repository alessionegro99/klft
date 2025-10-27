//*******************************************************************************
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
// See the top-level project for licensing details (GPLv3 or later).
//
//*******************************************************************************

#pragma once
#include <Kokkos_Core.hpp>
#include "GLOBAL.hpp"
#include "FieldTypeHelper.hpp"   // for DeviceGaugeFieldType<rank,Nc>

namespace klft {

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
real_t real_part(const real_t& x) { return x; }

template <class T>
KOKKOS_INLINE_FUNCTION
real_t real_part(const std::complex<T>& z) { return static_cast<real_t>(z.real()); }

template <class T>
KOKKOS_INLINE_FUNCTION
real_t real_part(const Kokkos::complex<T>& z) { return static_cast<real_t>(z.real()); }

// linear -> multi-index (row-major) for arbitrary rank
template <size_t rank>
KOKKOS_INLINE_FUNCTION
IndexArray<rank> linear_to_multi(size_t lin, const IndexArray<rank>& dims) {
  IndexArray<rank> idx;
  #pragma unroll
  for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
    const size_t d = static_cast<size_t>(dims[r]);
    idx[r] = static_cast<index_t>(lin % d);
    lin /= d;
  }
  return idx;
}

// get_dims overloads for our wrappers (they hold an IndexArray<rank> named 'dimensions')
template <size_t Nc>
KOKKOS_INLINE_FUNCTION
IndexArray<2> get_dims(const typename DeviceGaugeFieldType<2, Nc>::type& g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION
IndexArray<3> get_dims(const typename DeviceGaugeFieldType<3, Nc>::type& g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION
IndexArray<4> get_dims(const typename DeviceGaugeFieldType<4, Nc>::type& g) {
  return g.dimensions;
}

//------------------------------------------------------------------------------
// Retrace_links_avg
//   Average over all links of Re(Tr U_mu(x))/Nc
//------------------------------------------------------------------------------

template <size_t rank, size_t Nc>
KOKKOS_INLINE_FUNCTION
real_t Retrace_at(const typename DeviceGaugeFieldType<rank, Nc>::type& g,
                  const IndexArray<rank>& site,
                  const index_t mu) {
  const auto& U = g(site, mu); // SUN<Nc> = Kokkos::Array<Kokkos::Array<complex_t,Nc>,Nc>
  real_t trr = 0.0;
  #pragma unroll
  for (index_t a = 0; a < static_cast<index_t>(Nc); ++a) {
    trr += real_part(U[a][a]);
  }
  return trr / static_cast<real_t>(Nc);
}

template <size_t rank, size_t Nc>
real_t Retrace_links_avg(const typename DeviceGaugeFieldType<rank, Nc>::type& g)
{
  static_assert(rank == 2 || rank == 3 || rank == 4, "Retrace_links_avg: rank must be 2, 3, or 4.");

  using Exec = Kokkos::DefaultExecutionSpace;

  const auto dims = get_dims<Nc>(g);

  // number of sites = product of dims
  size_t nSites = 1;
  #pragma unroll
  for (size_t r = 0; r < rank; ++r) nSites *= static_cast<size_t>(dims[r]);

  const size_t nLinks = nSites * static_cast<size_t>(rank);

  real_t total = 0.0;

  Kokkos::parallel_reduce(
    "Retrace_links_avg",
    Kokkos::RangePolicy<Exec>(0, nLinks),
    KOKKOS_LAMBDA(const size_t i, real_t& lsum) {
      const index_t mu = static_cast<index_t>(i % rank);
      const size_t s   = i / rank; // linear site index
      const auto site  = linear_to_multi<rank>(s, dims);
      lsum += Retrace_at<rank, Nc>(g, site, mu);
    },
    total
  );

  // average over all links
  return total / static_cast<real_t>(nLinks);
}

} // namespace klft
