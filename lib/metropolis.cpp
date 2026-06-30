#include "core/compiled_theory.hpp"
#include "io/input_parser.hpp"
#include "updates/metropolis.hpp"
#include "updates/partitioned_metropolis.hpp"

#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

// Run Metropolis for the theory compiled into the binary.
int Metropolis(const std::string &input_file) {
  MetropolisParams metropolisParams;
  GaugeObservableParams gaugeObsParams;
  GradientFlowParams gradientFlowParams;
  PartitioningParams partitioningParams;
  if (!parseInputFile(input_file, metropolisParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gradientFlowParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, partitioningParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!validateGradientFlowParams(gradientFlowParams, gaugeObsParams)) {
    printf("Error validating gradient-flow input\n");
    return -1;
  }
  if (!validatePartitioningParams(partitioningParams, metropolisParams,
                                  gaugeObsParams)) {
    printf("Error validating partitioning input\n");
    return -1;
  }
  RNGType rng(metropolisParams.seed);
  if (partitioningParams.enabled) {
    if constexpr (compiled_nc == 2) {
      PartitionDeviceTable table;
      if (!loadPartitionTable(partitioningParams.table_file, table)) {
        return -1;
      }
      auto gauge_field = make_identity_gauge_field<compiled_rank, 2>(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3);
      auto partition_indices = initializePartitionGaugeField<compiled_rank>(
          gauge_field, table, metropolisParams.start, rng);
      return runPartitionedMetropolis<compiled_rank>(
          gauge_field, partition_indices, table, metropolisParams,
          gaugeObsParams, gradientFlowParams, rng);
    }
  }
  auto gauge_field =
      metropolisParams.start == "hot"
          ? make_hot_gauge_field<compiled_rank, compiled_nc>(
                metropolisParams.L0, metropolisParams.L1,
                metropolisParams.L2, metropolisParams.L3, rng)
          : make_identity_gauge_field<compiled_rank, compiled_nc>(
                metropolisParams.L0, metropolisParams.L1,
                metropolisParams.L2, metropolisParams.L3);
  run_metropolis<compiled_rank, compiled_nc>(gauge_field, metropolisParams,
                                             gaugeObsParams,
                                             gradientFlowParams, rng);
  return 0;
}

} // namespace klft
