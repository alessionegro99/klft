#pragma once

#include "core/klft_config.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <string>
#include <utility>
#include <vector>

namespace klft {

inline std::vector<std::pair<int, int>> sample_plane_pairs() {
  std::vector<std::pair<int, int>> pairs;
  for (size_t mu = 0; mu < compiled_rank; ++mu) {
    for (size_t nu = mu + 1; nu < compiled_rank; ++nu) {
      pairs.emplace_back(static_cast<int>(mu), static_cast<int>(nu));
      if (pairs.size() == 3) {
        return pairs;
      }
    }
  }
  return pairs;
}

inline void write_sample_nested_child_offset(std::ofstream &file) {
  file << "  nested_child_offset: [";
  for (size_t d = 0; d < compiled_rank; ++d) {
    if (d != 0) {
      file << ", ";
    }
    file << "0";
  }
  file << "]\n";
}

inline void write_common_observable_sample(std::ofstream &file) {
  file << "GaugeObservableParams:\n"
       << "  measurement_interval: 10\n"
       << "  measure_plaquette: true\n"
       << "  measure_wilson_loop_temporal: true\n"
       << "  measure_wilson_loop_mu_nu: true\n"
       << "  measure_polyakov_loop: true\n"
       << "  measure_polyakov_correlator: true\n"
       << "  measure_retrace_U: false\n"
       << "  wilson_loop_multihit: 1\n"
       << "  polyakov_loop_multihit: 1\n"
       << "  polyakov_correlator_max_r: 4\n"
       << "  measure_nested_wilson_action: false\n";
  write_sample_nested_child_offset(file);
  file << "  W_temp_L_T_pairs:\n"
       << "    - [2, 2]\n"
       << "    - [3, 3]\n"
       << "    - [4, 4]\n"
       << "  W_mu_nu_pairs:\n";

  for (const auto &[mu, nu] : sample_plane_pairs()) {
    file << "    - [" << mu << ", " << nu << "]\n";
  }

  file << "  W_Lmu_Lnu_pairs:\n"
       << "    - [2, 2]\n"
       << "    - [3, 3]\n"
       << "    - [4, 3]\n"
       << "  plaquette_filename: \"plaquette.out\"\n"
       << "  W_temp_filename: \"w_temp.out\"\n"
       << "  W_mu_nu_filename: \"w_mu_nu.out\"\n"
       << "  polyakov_loop_filename: \"polyakov_loop.out\"\n"
       << "  polyakov_correlator_filename: \"polyakov_correlator.out\"\n"
       << "  RetraceU_filename: \"retrace_u.out\"\n"
       << "  nested_wilson_action_filename: \"nested_wilson_action.out\"\n"
       << "  write_to_file: true\n";
}

inline void write_gradient_flow_sample(std::ofstream &file) {
  file << "\n"
       << "GradientFlowParams:\n"
       << "  enabled: false\n"
       << "  integrator: \"rk3\"\n"
       << "  dt: 0.01\n"
       << "  t_values: [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]\n"
       << "  measure_energy_clover: true\n"
       << "  measure_wilson_loop_temporal: false\n"
       << "  measure_wilson_loop_mu_nu: false\n"
       << "  extract_t0: false\n"
       << "  t0_target: 0.3\n"
       << "  obs_filename: \"gradient_flow_obs.dat\"\n"
       << "  W_temp_filename: \"gradient_flow_wtemp.dat\"\n"
       << "  W_mu_nu_filename: \"gradient_flow_w_mu_nu.dat\"\n"
       << "  t0_filename: \"gradient_flow_t0.dat\"\n";
}

inline int write_sample_metropolis_input_file(const std::string &filename) {
  namespace fs = std::filesystem;
  if (fs::exists(filename)) {
    printf("Sample input file already exists: %s\n", filename.c_str());
    return 0;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    printf("Error: could not create sample input file: %s\n", filename.c_str());
    return -1;
  }

  file << "# input.yaml\n"
       << "MetropolisParams:\n"
       << "  L0: 8\n"
       << "  L1: 8\n"
       << "  L2: " << (compiled_rank > 2 ? 8 : 4) << "\n"
       << "  L3: " << (compiled_rank > 3 ? 8 : 4) << "\n"
       << "  nHits: 10\n"
       << "  nSweep: 1000\n"
       << "  seed: 32091\n"
       << "  beta: 2.0\n"
       << "  delta: 0.1\n"
       << "  epsilon1: 0.0\n"
       << "  epsilon2: 0.0\n"
       << "\n";
  write_common_observable_sample(file);
  write_gradient_flow_sample(file);

  printf("Wrote sample input file: %s\n", filename.c_str());
  return 0;
}

inline int write_sample_heatbath_input_file(const std::string &filename) {
  namespace fs = std::filesystem;
  if (fs::exists(filename)) {
    printf("Sample input file already exists: %s\n", filename.c_str());
    return 0;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    printf("Error: could not create sample input file: %s\n", filename.c_str());
    return -1;
  }

  file << "# input.yaml\n"
       << "HeatbathParams:\n"
       << "  L0: 8\n"
       << "  L1: 8\n"
       << "  L2: " << (compiled_rank > 2 ? 8 : 4) << "\n"
       << "  L3: " << (compiled_rank > 3 ? 8 : 4) << "\n"
       << "  nSweep: 1000\n"
       << "  nOverrelax: 5\n"
       << "  seed: 32091\n"
       << "  beta: 2.0\n"
       << "  epsilon1: 0.0\n"
       << "  epsilon2: 0.0\n"
       << "\n";
  write_common_observable_sample(file);
  write_gradient_flow_sample(file);

  printf("Wrote sample input file: %s\n", filename.c_str());
  return 0;
}

inline int parse_driver_args(int argc, char **argv, std::string &input_file) {
  input_file = "input.yaml";
  if (argc == 1) {
    return 1;
  }

  const std::string help_string =
      "  -f <file_name> --filename <file_name>\n"
      "     Name of the input file.\n"
      "     Default: input.yaml\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     This binary is compiled for " +
      std::to_string(compiled_rank) + "D " + compiled_group_name() + ".\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"filename", required_argument, nullptr, 'f'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int c = 0;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "f:h", long_options, &option_index)) !=
         -1) {
    switch (c) {
    case 'f':
      input_file = optarg;
      break;
    case 'h':
      printf("%s", help_string.c_str());
      return -2;
    case 0:
      break;
    default:
      printf("%s", help_string.c_str());
      return -1;
    }
  }
  return 0;
}

} // namespace klft
