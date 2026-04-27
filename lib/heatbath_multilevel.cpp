#include "core/compiled_theory.hpp"
#include "io/input_parser.hpp"
#include "updates/heatbath_multilevel.hpp"

#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int HeatbathMultilevel(const std::string &input_file) {
  HeatbathParams heatbathParams;
  MultilevelParams multilevelParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, heatbathParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, multilevelParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams, true)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!validateMultilevelParams(multilevelParams, heatbathParams)) {
    printf("Error validating multilevel parameters\n");
    return -1;
  }

  if (!gaugeObsParams.measure_polyakov_loop &&
      !gaugeObsParams.measure_polyakov_correlator) {
    printf("Error: heatbath_multilevel requires at least one Polyakov "
           "observable\n");
    return -1;
  }
  if (heatbathParams.epsilon2 != 0.0) {
    printf("Error: heatbath/overrelaxation currently supports epsilon1 only; "
           "epsilon2 requires a different local update.\n");
    return -1;
  }

  RNGType rng(heatbathParams.seed);
  auto gauge_field = make_identity_gauge_field<compiled_rank, compiled_nc>(
      heatbathParams.L0, heatbathParams.L1, heatbathParams.L2,
      heatbathParams.L3);
  run_heatbath_multilevel<compiled_rank, compiled_nc>(
      gauge_field, heatbathParams, multilevelParams, gaugeObsParams, rng);

  return 0;
}

} // namespace klft
