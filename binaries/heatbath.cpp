#include "KLFTConfig.hpp"
#include "klft.hpp"
#include <filesystem>
#include <fstream>
#include <getopt.h>

using namespace klft;

#define HLINE                                                                  \
  "====================================================================\n"

// Emit a starter input file for the compiled theory.
int write_sample_input_file(const std::string &filename) {
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
       << "  delta: 0.1\n"
       << "  epsilon1: 0.0\n"
       << "  epsilon2: 0.0\n"
       << "\n"
       << "GaugeObservableParams:\n"
       << "  measurement_interval: 10\n"
       << "  measure_plaquette: true\n"
       << "  measure_wilson_loop_temporal: true\n"
       << "  measure_wilson_loop_mu_nu: true\n"
       << "  measure_retrace_U: false\n"
       << "  wilson_loop_multihit: 1\n"
       << "  measure_nested_wilson_action: false\n"
       << "  W_temp_L_T_pairs:\n"
       << "    - [2, 2]\n"
       << "    - [3, 4]\n"
       << "    - [4, 3]\n"
       << "    - [4, 4]\n"
       << "  W_mu_nu_pairs:\n"
       << "    - [0, 1]\n"
       << "    - [1, 2]\n"
       << "    - [3, 2]\n"
       << "  W_Lmu_Lnu_pairs:\n"
       << "    - [2, 2]\n"
       << "    - [3, 3]\n"
       << "    - [4, 3]\n"
       << "  plaquette_filename: \"plaquette.out\"\n"
       << "  W_temp_filename: \"W_temp.out\"\n"
       << "  W_mu_nu_filename: \"W_mu_nu.out\"\n"
       << "  RetraceU_filename: \"RetraceU.out\"\n"
       << "  nested_wilson_action_filename: \"nested_wilson_action.out\"\n"
       << "  write_to_file: true\n";

  file.close();
  printf("Wrote sample input file: %s\n", filename.c_str());
  return 0;
}

// Parse the standalone driver CLI.
int parse_args(int argc, char **argv, std::string &input_file) {
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
      "     This binary is compiled for "
      + std::to_string(compiled_rank) + "D " + compiled_group_name() + ".\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"filename", required_argument, NULL, 'f'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
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

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Heatbath + overrelaxation for %s gauge fields in %zuD\n",
         compiled_group_name(), compiled_rank);
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  std::string input_file;
  rc = parse_args(argc, argv, input_file);
  if (rc == 0) {
    rc = Heatbath(input_file);
  } else if (rc == 1) {
    rc = write_sample_input_file(input_file);
  } else if (rc == -2) {
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
