#pragma once

#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "params/heatbath_params.hpp"
#include "updates/heatbath_link_updates.hpp"

#include <Kokkos_Random.hpp>

namespace klft {

template <size_t rank, size_t Nc, class RNG, bool Overrelax>
struct RestrictedHeatbathGaugeField {
  constexpr static const size_t Nd = rank;
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  const RNG rng;
  const HeatbathParams params;
  const index_t dir;
  const index_t slab_links;
  const Kokkos::Array<bool, rank> oddeven;

  RestrictedHeatbathGaugeField(const GaugeFieldType &g_in,
                               const HeatbathParams &params,
                               const index_t dir, const index_t slab_links,
                               const Kokkos::Array<bool, rank> &oddeven,
                               const RNG &rng)
      : g_in(g_in), rng(rng), params(params), dir(dir),
        slab_links(slab_links), oddeven(oddeven) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site = index_odd_even<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, oddeven);
    if (dir != static_cast<index_t>(time_dir) &&
        site[time_dir] % slab_links == 0) {
      return;
    }

    auto generator = rng.get_state();
    auto link = g_in(site, dir);
    const auto matrix = effective_local_matrix<Nc>(g_in.staple(site, dir),
                                                   params.beta,
                                                   params.epsilon1);
    if constexpr (Overrelax) {
      overrelax_link(link, matrix, generator);
    } else {
      heatbath_link(link, matrix, generator);
    }
    restoreSUN(link);
    g_in(site, dir) = link;
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc> struct RestrictedUnitarizeGaugeField {
  constexpr static const size_t Nd = rank;
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  const index_t slab_links;
  RestrictedUnitarizeGaugeField(const GaugeFieldType &g_in,
                                const index_t slab_links)
      : g_in(g_in), slab_links(slab_links) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(Idcs)...};
#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
      if (mu != static_cast<index_t>(time_dir) &&
          site[time_dir] % slab_links == 0) {
        continue;
      }
      auto link = g_in(site, mu);
      restoreSUN(link);
      g_in(site, mu) = link;
    }
  }
};

template <size_t rank, size_t Nc, class RNG>
void restricted_heatbath_sweep(
    typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const HeatbathParams &params, const index_t slab_links, const RNG &rng) {
  constexpr static const size_t Nd = rank;
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  IndexArray<rank> full_start;
  IndexArray<rank> full_end;
  for (index_t i = 0; i < static_cast<index_t>(Nd); ++i) {
    start[i] = 0;
    end[i] = static_cast<index_t>(dimensions[i] / 2);
    full_start[i] = 0;
    full_end[i] = dimensions[i];
  }

  for (index_t dir = 0; dir < static_cast<index_t>(Nd); ++dir) {
    for (index_t i = 0; i < (1 << rank); ++i) {
      Kokkos::parallel_for(
          Policy<rank>(start, end),
          RestrictedHeatbathGaugeField<rank, Nc, RNG, false>(
              g_in, params, dir, slab_links, oddeven_array<rank>(i), rng));
      Kokkos::fence();
    }
  }

  for (index_t dir = 0; dir < static_cast<index_t>(Nd); ++dir) {
    for (index_t j = 0; j < params.nOverrelax; ++j) {
      for (index_t i = 0; i < (1 << rank); ++i) {
        Kokkos::parallel_for(
            Policy<rank>(start, end),
            RestrictedHeatbathGaugeField<rank, Nc, RNG, true>(
                g_in, params, dir, slab_links, oddeven_array<rank>(i), rng));
        Kokkos::fence();
      }
    }
  }

  Kokkos::parallel_for(Policy<rank>(full_start, full_end),
                       RestrictedUnitarizeGaugeField<rank, Nc>(g_in,
                                                               slab_links));
  Kokkos::fence();
}

} // namespace klft
