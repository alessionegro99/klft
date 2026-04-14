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
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site;
  const IndexArray<rank> dimensions;
  GaugePlaq(const GaugeFieldType &g_in, FieldType &plaq_per_site,
            const IndexArray<rank> &dimensions)
      : g_in(g_in), plaq_per_site(plaq_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    SUN<Nc> lmu, lnu;
    complex_t tmunu(0.0, 0.0);

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          lmu =
              g_in(Idcs..., mu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions),
                   nu);
          lnu =
              g_in(Idcs..., nu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, nu, 1, dimensions),
                   mu);
          tmunu += trace(lmu * conj(lnu));
        }
      }
    }
    plaq_per_site(Idcs...) = tmunu;
  }
};

template <size_t rank, size_t Nc>
real_t GaugePlaquette(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                      const bool normalize = true) {
  constexpr static const size_t Nd = rank;
  complex_t plaq = 0.0;
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site(end, complex_t(0.0, 0.0));

  GaugePlaq<rank, Nc> gaugePlaquette(g_in, plaq_per_site, end);

  Kokkos::parallel_for(Policy<rank>(start, end), gaugePlaquette);
  Kokkos::fence();

  // sum over all sites
  plaq = plaq_per_site.sum();
  Kokkos::fence();

  // normalization
  if (normalize) {
    real_t norm = 1.0;
    for (index_t i = 0; i < rank; ++i) {
      norm *= static_cast<real_t>(end[i]);
    }
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}
} // namespace klft
