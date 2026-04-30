#pragma once
#include "core/common.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include <Kokkos_Core.hpp>

namespace klft {

// Average Re Tr U over all links.
template <size_t rank, size_t Nc>
KOKKOS_INLINE_FUNCTION real_t
Retrace_at(const typename DeviceGaugeFieldType<rank, Nc>::type &g,
           const IndexArray<rank> &site, const index_t mu) {
  const auto &U = g(site, mu);
  return trace(U).real() / static_cast<real_t>(Nc);
}

template <size_t rank, size_t Nc> struct RetraceLinks {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType g;

  explicit RetraceLinks(const GaugeFieldType &g) : g(g) {}

  KOKKOS_FORCEINLINE_FUNCTION
  real_t retrace_at_site(const IndexArray<rank> &site) const {
    real_t local = 0.0;
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      local += Retrace_at<rank, Nc>(g, site, mu);
    }
    return local;
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              real_t &sum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    sum += retrace_at_site(IndexArray<rank>{i0, i1});
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              real_t &sum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    sum += retrace_at_site(IndexArray<rank>{i0, i1, i2});
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              real_t &sum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    sum += retrace_at_site(IndexArray<rank>{i0, i1, i2, i3});
  }
};

template <size_t rank, size_t Nc>
real_t
Retrace_links_avg(const typename DeviceGaugeFieldType<rank, Nc>::type &g) {
  static_assert(rank == 2 || rank == 3 || rank == 4,
                "Retrace_links_avg: rank must be 2, 3, or 4.");

  const auto dims = g.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  size_t nSites = 1;
  for (index_t r = 0; r < static_cast<index_t>(rank); ++r) {
    start[r] = 0;
    end[r] = dims[r];
    nSites *= static_cast<size_t>(dims[r]);
  }

  const size_t nLinks = nSites * static_cast<size_t>(rank);
  real_t total = 0.0;

  Kokkos::parallel_reduce("Retrace_links_avg", Policy<rank>(start, end),
                          RetraceLinks<rank, Nc>(g), total);

  return total / static_cast<real_t>(nLinks);
}

} // namespace klft
