#include "CompiledTheory.hpp"
#include "Metropolis.hpp"
#include "InputParser.hpp"
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

// Run Metropolis for the theory compiled into the binary.
int Metropolis(const std::string &input_file) {
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  MetropolisParams metropolisParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, metropolisParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  metropolisParams.print();
  print_compiled_theory();
  RNGType rng(metropolisParams.seed);
  auto gauge_field = make_identity_gauge_field<compiled_rank, compiled_nc>(
      metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
      metropolisParams.L3);
  run_metropolis<compiled_rank, compiled_nc>(gauge_field, metropolisParams,
                                             gaugeObsParams, rng);
  return 0;
}

} // namespace klft
