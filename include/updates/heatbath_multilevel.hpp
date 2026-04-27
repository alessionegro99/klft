#pragma once

#include "observables/gauge_observables.hpp"
#include "observables/polyakov_multilevel.hpp"
#include "params/multilevel_params.hpp"
#include "updates/heatbath.hpp"

namespace klft {

template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_heatbath_multilevel(GaugeFieldType &g_in,
                            const HeatbathParams &heatbathParams,
                            const MultilevelParams &multilevelParams,
                            GaugeObservableParams &gaugeObsParams,
                            const RNG &rng) {
  const auto &dimensions = g_in.dimensions;
  validate_even_extents<rank>(dimensions, "Heatbath multilevel");
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

    if ((gaugeObsParams.measurement_interval == 0) ||
        ((step + 1) % gaugeObsParams.measurement_interval != 0)) {
      continue;
    }

    if (gaugeObsParams.measure_polyakov_loop) {
      const auto P = PolyakovLoopMultilevel<rank, Nc>(
          g_in, heatbathParams, multilevelParams,
          gaugeObsParams.polyakov_loop_multihit, rng);
      gaugeObsParams.polyakov_measurements.push_back(P);
    }

    if (gaugeObsParams.measure_polyakov_correlator) {
      std::vector<Kokkos::Array<real_t, 3>> corr_measurements;
      PolyakovCorrelatorMultilevel<rank, Nc>(
          g_in, gaugeObsParams.polyakov_correlator_max_r,
          gaugeObsParams.polyakov_loop_multihit, corr_measurements,
          heatbathParams, multilevelParams, rng);
      gaugeObsParams.polyakov_correlator_measurements.push_back(
          corr_measurements);
    }

    gaugeObsParams.measurement_steps.push_back(step + 1);
    gaugeObsParams.measurement_times.push_back(time);

    if (gaugeObsParams.write_to_file) {
      appendLatestMultilevelGaugeObservables(gaugeObsParams);
      clearAllGaugeObservables(gaugeObsParams);
    }
  }
  return 0;
}

} // namespace klft
