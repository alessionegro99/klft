#include "core/compiled_theory.hpp"
#include "core/temporal_dirichlet.hpp"
#include "io/gauge_configuration.hpp"
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
  if (!validateTemporalDirichletParams(metropolisParams, gaugeObsParams)) {
    printf("Error validating temporal Dirichlet input\n");
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
      PartitionIndexField partition_indices;
      if (metropolisParams.start == "restart") {
        if (!load_gauge_configuration<compiled_rank, 2>(
                metropolisParams.configuration_input, gauge_field,
                metropolisParams.temporal_dirichlet)) {
          return -1;
        }
        partition_indices = initializePartitionIndicesFromGauge<compiled_rank>(
            gauge_field, table, metropolisParams.temporal_dirichlet);
        if (partition_indices.extent(0) == 0) {
          return -1;
        }
      } else {
        partition_indices = initializePartitionGaugeField<compiled_rank>(
            gauge_field, table, metropolisParams.start, rng,
            metropolisParams.temporal_dirichlet);
      }
      const int run_status = runPartitionedMetropolis<compiled_rank>(
          gauge_field, partition_indices, table, metropolisParams,
          gaugeObsParams, gradientFlowParams, rng);
      if (run_status == 0 && !metropolisParams.configuration_output.empty() &&
          !save_gauge_configuration<compiled_rank, 2>(
              metropolisParams.configuration_output, gauge_field,
              metropolisParams.temporal_dirichlet)) {
        return -1;
      }
      return run_status;
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
  if (metropolisParams.start == "restart") {
    if (!load_gauge_configuration<compiled_rank, compiled_nc>(
            metropolisParams.configuration_input, gauge_field,
            metropolisParams.temporal_dirichlet)) {
      return -1;
    }
  }
  if (metropolisParams.temporal_dirichlet &&
      metropolisParams.start != "restart") {
    apply_temporal_dirichlet_boundaries<compiled_rank, compiled_nc>(
        gauge_field);
  }
  const int run_status = run_metropolis<compiled_rank, compiled_nc>(
      gauge_field, metropolisParams, gaugeObsParams, gradientFlowParams, rng);
  if (run_status == 0 && !metropolisParams.configuration_output.empty() &&
      !save_gauge_configuration<compiled_rank, compiled_nc>(
          metropolisParams.configuration_output, gauge_field,
          metropolisParams.temporal_dirichlet)) {
    return -1;
  }
  return run_status;
}

} // namespace klft
