#pragma once
#include "core/indexing.hpp"
#include "core/temporal_dirichlet.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"
#include "params/metropolis_params.hpp"
#include "updates/gradient_flow.hpp"

#include <Kokkos_Random.hpp>

namespace klft {

// Update one odd/even sublattice with local Metropolis proposals.
template <size_t rank, size_t Nc, class RNG> struct MetropolisGaugeField {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType g_in;
  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted;
  const RNG rng;
  const MetropolisParams params;
  const Kokkos::Array<index_t, rank> colors;
  MetropolisGaugeField(const GaugeFieldType &g_in,
                       const MetropolisParams &params,
                       const ScalarFieldType &nAccepted,
                       const Kokkos::Array<index_t, rank> &colors,
                       const RNG &rng)
      : g_in(g_in), nAccepted(nAccepted), rng(rng), params(params),
        colors(colors) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    index_t nAcc_per_site = 0;
    auto generator = rng.get_state();
    const IndexArray<rank> site = index_lattice_color<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, colors, g_in.dimensions,
        params.temporal_dirichlet);
    for (index_t mu = 0; mu < Nd; ++mu) {
      if (params.temporal_dirichlet &&
          is_temporal_dirichlet_link<rank>(site, mu, g_in.dimensions)) {
        continue;
      }
      const SUN<Nc> staple = g_in.staple(site, mu);
      for (index_t hit = 0; hit < params.nHits; ++hit) {
        const SUN<Nc> U_old = g_in(site, mu);
        const SUN<Nc> U_new =
            apply_metropolis_proposal<Nc>(U_old, params.delta, generator);
        real_t dS =
            -(params.beta / static_cast<real_t>(Nc)) *
            (trace(U_new * staple).real() - trace(U_old * staple).real());

        if (params.epsilon1 != 0.0) {
          dS += -params.epsilon1 * static_cast<real_t>(0.5) *
                (trace(U_new).real() - trace(U_old).real());
        }

        if (params.epsilon2 != 0.0) {
          auto retr_U_new = trace(U_new).real();
          auto retr_U_old = trace(U_old).real();

          dS += -params.epsilon2 *
                (retr_U_new * retr_U_new - retr_U_old * retr_U_old);
        }

        bool accept = dS < 0.0;
        if (!accept) {
          accept = (generator.drand(0.0, 1.0) < Kokkos::exp(-dS));
        }
        if (accept) {
          g_in(site, mu) = restoreSUN(U_new);
          nAcc_per_site++;
        }
      }
    }

    nAccepted(Idcs...) += static_cast<real_t>(nAcc_per_site);

    rng.free_state(generator);
  }
};

// Sweep the full lattice once and return the acceptance rate.
template <size_t rank, size_t Nc, class RNG>
real_t sweep_Metropolis(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                        const MetropolisParams &params, const RNG &rng) {
  constexpr static const size_t Nd = rank;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = lattice_color_extent<rank>(g_in.dimensions, i, 0,
                                        params.temporal_dirichlet);
  }

  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted(end, 0.0);

  const index_t color_count =
      lattice_color_total<rank>(g_in.dimensions, params.temporal_dirichlet);
  for (index_t i = 0; i < color_count; ++i) {
    const auto colors = decode_lattice_color<rank>(
        i, g_in.dimensions, params.temporal_dirichlet);
    auto color_end = end;
    for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
      color_end[d] = lattice_color_extent<rank>(
          g_in.dimensions, d, colors[d], params.temporal_dirichlet);
    }
    MetropolisGaugeField<rank, Nc, RNG> metropolis(g_in, params, nAccepted,
                                                   colors, rng);
    Kokkos::parallel_for(Policy<rank>(start, color_end), metropolis);
    Kokkos::fence();
  }
  real_t nAcc_total = nAccepted.sum();
  Kokkos::fence();
  const real_t norm = static_cast<real_t>(
      dynamical_link_count<rank>(g_in.dimensions, params.temporal_dirichlet) *
      static_cast<size_t>(params.nHits));
  nAcc_total /= norm;
  return nAcc_total;
}

// Execute the requested number of Metropolis sweeps and measurements.
template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_metropolis(GaugeFieldType &g_in,
                   const MetropolisParams &metropolisParams,
                   GaugeObservableParams &gaugeObsParams,
                   const GradientFlowParams &gradientFlowParams,
                   const RNG &rng) {
  const auto &dimensions = g_in.dimensions;
  if (metropolisParams.temporal_dirichlet) {
    validate_temporal_dirichlet_extents<rank>(dimensions, "Metropolis");
  } else {
    validate_even_extents<rank>(dimensions, "Metropolis");
  }
  gaugeObsParams.include_acceptance_rate = true;
  assert(metropolisParams.L0 == dimensions[0]);
  assert(metropolisParams.L1 == dimensions[1]);
  if constexpr (rank > 2) {
    assert(metropolisParams.L2 == dimensions[2]);
  }
  if constexpr (rank > 3) {
    assert(metropolisParams.L3 == dimensions[3]);
  }
  Kokkos::Timer timer;
  for (size_t step = 0; step < metropolisParams.nSweep; ++step) {
    timer.reset();
    const real_t acc_rate =
        sweep_Metropolis<rank, Nc>(g_in, metropolisParams, rng);
    const real_t time = timer.seconds();
    measureGaugeObservables<rank, Nc>(g_in, metropolisParams, gaugeObsParams,
                                      step + 1, acc_rate, time, rng);
    runGradientFlowMeasurements<rank, Nc>(
        g_in, metropolisParams, gaugeObsParams, gradientFlowParams, step + 1,
        rng);
  }
  return 0;
}

} // namespace klft
