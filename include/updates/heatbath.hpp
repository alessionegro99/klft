#pragma once
#include "core/indexing.hpp"
#include "core/temporal_dirichlet.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"
#include "params/heatbath_params.hpp"
#include "updates/gradient_flow.hpp"
#include "updates/heatbath_link_updates.hpp"

#include <Kokkos_Random.hpp>

namespace klft {

// Update one odd/even sublattice with either heatbath or overrelaxation.
template <size_t rank, size_t Nc, class RNG, bool Overrelax>
struct HeatbathGaugeField {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  const RNG rng;
  const HeatbathParams params;
  const index_t dir;
  const Kokkos::Array<index_t, rank> colors;

  HeatbathGaugeField(const GaugeFieldType &g_in, const HeatbathParams &params,
                     const index_t dir,
                     const Kokkos::Array<index_t, rank> &colors,
                     const RNG &rng)
      : g_in(g_in), rng(rng), params(params), dir(dir), colors(colors) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site = index_lattice_color<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, colors, g_in.dimensions,
        params.temporal_dirichlet);
    if (params.temporal_dirichlet &&
        is_temporal_dirichlet_link<rank>(site, dir, g_in.dimensions)) {
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

// Restore exact group projection after each full sweep.
template <size_t rank, size_t Nc> struct UnitarizeGaugeField {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  const bool temporal_dirichlet;
  UnitarizeGaugeField(const GaugeFieldType &g_in,
                      const bool temporal_dirichlet)
      : g_in(g_in), temporal_dirichlet(temporal_dirichlet) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(Idcs)...};
#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
      if (temporal_dirichlet &&
          is_temporal_dirichlet_link<rank>(site, mu, g_in.dimensions)) {
        g_in(site, mu) = identitySUN<Nc>();
        continue;
      }
      auto link = g_in(Idcs..., mu);
      restoreSUN(link);
      g_in(Idcs..., mu) = link;
    }
  }
};

// Perform one heatbath sweep plus the requested overrelaxation sweeps.
template <size_t rank, size_t Nc, class RNG>
void full_heatbath_sweep(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                         const HeatbathParams &params, const RNG &rng) {
  constexpr static const size_t Nd = rank;
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  IndexArray<rank> full_start;
  IndexArray<rank> full_end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = lattice_color_extent<rank>(g_in.dimensions, i, 0,
                                        params.temporal_dirichlet);
    full_start[i] = 0;
    full_end[i] = dimensions[i];
  }
  const index_t color_count =
      lattice_color_total<rank>(g_in.dimensions, params.temporal_dirichlet);

  for (index_t dir = 0; dir < Nd; ++dir) {
    for (index_t i = 0; i < color_count; ++i) {
      const auto colors = decode_lattice_color<rank>(
          i, g_in.dimensions, params.temporal_dirichlet);
      auto color_end = end;
      for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
        color_end[d] = lattice_color_extent<rank>(
            g_in.dimensions, d, colors[d], params.temporal_dirichlet);
      }
      Kokkos::parallel_for(
          Policy<rank>(start, color_end),
          HeatbathGaugeField<rank, Nc, RNG, false>(g_in, params, dir,
                                                   colors, rng));
      Kokkos::fence();
    }
  }

  for (index_t dir = 0; dir < Nd; ++dir) {
    for (index_t j = 0; j < params.nOverrelax; ++j) {
      for (index_t i = 0; i < color_count; ++i) {
        const auto colors = decode_lattice_color<rank>(
            i, g_in.dimensions, params.temporal_dirichlet);
        auto color_end = end;
        for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
          color_end[d] = lattice_color_extent<rank>(
              g_in.dimensions, d, colors[d], params.temporal_dirichlet);
        }
        Kokkos::parallel_for(
            Policy<rank>(start, color_end),
            HeatbathGaugeField<rank, Nc, RNG, true>(
                g_in, params, dir, colors, rng));
        Kokkos::fence();
      }
    }
  }

  Kokkos::parallel_for(Policy<rank>(full_start, full_end),
                       UnitarizeGaugeField<rank, Nc>(
                           g_in, params.temporal_dirichlet));
  Kokkos::fence();
}

// Execute the requested number of heatbath sweeps and measurements.
template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_heatbath(GaugeFieldType &g_in, const HeatbathParams &heatbathParams,
                 GaugeObservableParams &gaugeObsParams,
                 const GradientFlowParams &gradientFlowParams,
                 const RNG &rng) {
  const auto &dimensions = g_in.dimensions;
  if (heatbathParams.temporal_dirichlet) {
    validate_temporal_dirichlet_extents<rank>(dimensions, "Heatbath");
  } else {
    validate_even_extents<rank>(dimensions, "Heatbath");
  }
  gaugeObsParams.include_acceptance_rate = false;

  assert(heatbathParams.L0 == dimensions[0]);
  assert(heatbathParams.L1 == dimensions[1]);
  if constexpr (rank > 2) {
    assert(heatbathParams.L2 == dimensions[2]);
  }
  if constexpr (rank > 3) {
    assert(heatbathParams.L3 == dimensions[3]);
  }

  Kokkos::Timer timer;
  for (size_t step = 0; step < heatbathParams.nSweep; ++step) {
    timer.reset();
    full_heatbath_sweep<rank, Nc>(g_in, heatbathParams, rng);
    const real_t time = timer.seconds();

    measureGaugeObservables<rank, Nc>(g_in, heatbathParams, gaugeObsParams,
                                      step + 1, 0.0, time, rng);
    runGradientFlowMeasurements<rank, Nc>(
        g_in, heatbathParams, gaugeObsParams, gradientFlowParams, step + 1,
        rng);
  }
  return 0;
}

} // namespace klft
