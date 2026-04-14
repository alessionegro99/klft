#include "core/compiled_theory.hpp"
#include "io/input_parser.hpp"
#include "updates/heatbath.hpp"

#include <cstdlib>
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

// Run heatbath plus overrelaxation for the theory compiled into the binary.
int Heatbath(const std::string &input_file) {
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  HeatbathParams heatbathParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, heatbathParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }

  if (heatbathParams.epsilon2 != 0.0) {
    printf("Error: heatbath/overrelaxation currently supports epsilon1 only; "
           "epsilon2 requires a different local update.\n");
    return -1;
  }

  heatbathParams.print();
  print_compiled_theory();
  RNGType rng(heatbathParams.seed);

  auto gauge_field = make_identity_gauge_field<compiled_rank, compiled_nc>(
      heatbathParams.L0, heatbathParams.L1, heatbathParams.L2,
      heatbathParams.L3);
  run_heatbath<compiled_rank, compiled_nc>(gauge_field, heatbathParams,
                                           gaugeObsParams, rng);

  return 0;
}

} // namespace klft
