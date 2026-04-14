#pragma once
#include "core/common.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include <Kokkos_Core.hpp>

namespace klft {

// Convert a linear site index into a rank-dimensional lattice index.
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank>
linear_to_multi(size_t lin, const IndexArray<rank> &dims) {
  IndexArray<rank> idx;
#pragma unroll
  for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
    const size_t d = static_cast<size_t>(dims[r]);
    idx[r] = static_cast<index_t>(lin % d);
    lin /= d;
  }
  return idx;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<2>
get_dims(const typename DeviceGaugeFieldType<2, Nc>::type &g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<3>
get_dims(const typename DeviceGaugeFieldType<3, Nc>::type &g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<4>
get_dims(const typename DeviceGaugeFieldType<4, Nc>::type &g) {
  return g.dimensions;
}

// Average Re Tr U over all links.
template <size_t rank, size_t Nc>
KOKKOS_INLINE_FUNCTION real_t
Retrace_at(const typename DeviceGaugeFieldType<rank, Nc>::type &g,
           const IndexArray<rank> &site, const index_t mu) {
  const auto &U = g(site, mu);
  return trace(U).real() / static_cast<real_t>(Nc);
}

template <size_t rank, size_t Nc>
real_t
Retrace_links_avg(const typename DeviceGaugeFieldType<rank, Nc>::type &g) {
  static_assert(rank == 2 || rank == 3 || rank == 4,
                "Retrace_links_avg: rank must be 2, 3, or 4.");

  using Exec = Kokkos::DefaultExecutionSpace;

  const auto dims = get_dims<Nc>(g);

  size_t nSites = 1;
#pragma unroll
  for (size_t r = 0; r < rank; ++r)
    nSites *= static_cast<size_t>(dims[r]);

  const size_t nLinks = nSites * static_cast<size_t>(rank);

  real_t total = 0.0;

  Kokkos::parallel_reduce(
      "Retrace_links_avg", Kokkos::RangePolicy<Exec>(0, nLinks),
      KOKKOS_LAMBDA(const size_t i, real_t &lsum) {
        const index_t mu = static_cast<index_t>(i % rank);
        const size_t s = i / rank; // linear site index
        const auto site = linear_to_multi<rank>(s, dims);
        lsum += Retrace_at<rank, Nc>(g, site, mu);
      },
      total);

  return total / static_cast<real_t>(nLinks);
}

} // namespace klft
