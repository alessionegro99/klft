#pragma once
#include "GaugeObservable.hpp"
#include "Heatbath_Params.hpp"
#include "Metropolis_Params.hpp"
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace klft {

// Expand YAML scalars like `4` or `2:6` into explicit index lists.
inline std::vector<index_t> parseIndexRange(const YAML::Node &node) {
  std::vector<index_t> values;
  if (node.IsScalar()) {
    std::string s = node.as<std::string>();
    size_t colon = s.find(':');
    if (colon != std::string::npos) {
      try {
        index_t start = std::stoi(s.substr(0, colon));
        index_t end = std::stoi(s.substr(colon + 1));
        if (start <= end) {
          for (index_t i = start; i <= end; ++i) {
            values.push_back(i);
          }
        }
      } catch (...) {
        printf("Warning: Failed to parse range '%s'\n", s.c_str());
      }
    } else {
      try {
        values.push_back(node.as<index_t>());
      } catch (...) {
        printf("Warning: Failed to parse index '%s'\n", s.c_str());
      }
    }
  }
  return values;
}

// Parse the Metropolis section from the input file.
inline bool parseInputFile(const std::string &filename,
                           MetropolisParams &metropolisParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    if (config["MetropolisParams"]) {
      const auto &mp = config["MetropolisParams"];
      metropolisParams.L0 = mp["L0"].as<index_t>(32);
      metropolisParams.L1 = mp["L1"].as<index_t>(32);
      metropolisParams.L2 = mp["L2"].as<index_t>(32);
      metropolisParams.L3 = mp["L3"].as<index_t>(32);
      metropolisParams.nHits = mp["nHits"].as<index_t>(10);
      metropolisParams.nSweep = mp["nSweep"].as<index_t>(1000);
      metropolisParams.seed = mp["seed"].as<index_t>(1234);

      metropolisParams.beta = mp["beta"].as<double>(1.0);
      metropolisParams.delta = mp["delta"].as<double>(0.1);

      metropolisParams.epsilon1 = mp["epsilon1"].as<real_t>(0.0);
      metropolisParams.epsilon2 = mp["epsilon2"].as<real_t>(0.0);
    } else {
      printf("Error: MetropolisParams not found in input file\n");
      return false;
    }

    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing input file: %s\n", e.what());
    return false;
  }
}

// Parse the heatbath section from the input file.
inline bool parseInputFile(const std::string &filename,
                           HeatbathParams &heatbathParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    if (config["HeatbathParams"]) {
      const auto &hp = config["HeatbathParams"];
      heatbathParams.L0 = hp["L0"].as<index_t>(32);
      heatbathParams.L1 = hp["L1"].as<index_t>(32);
      heatbathParams.L2 = hp["L2"].as<index_t>(32);
      heatbathParams.L3 = hp["L3"].as<index_t>(32);
      heatbathParams.nSweep = hp["nSweep"].as<index_t>(1000);
      heatbathParams.nOverrelax = hp["nOverrelax"].as<index_t>(5);
      heatbathParams.seed = hp["seed"].as<index_t>(1234);

      heatbathParams.beta = hp["beta"].as<double>(1.0);
      heatbathParams.delta = hp["delta"].as<double>(0.1);
      heatbathParams.epsilon1 = hp["epsilon1"].as<real_t>(0.0);
      heatbathParams.epsilon2 = hp["epsilon2"].as<real_t>(0.0);
    } else {
      printf("Error: HeatbathParams not found in input file\n");
      return false;
    }

    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing input file: %s\n", e.what());
    return false;
  }
}

// Parse observable settings and loop grids from the input file.
inline bool parseInputFile(const std::string &filename,
                           GaugeObservableParams &gaugeObservableParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    if (config["GaugeObservableParams"]) {
      const auto &gp = config["GaugeObservableParams"];

      gaugeObservableParams.W_temp_L_T_pairs.clear();
      gaugeObservableParams.W_mu_nu_pairs.clear();
      gaugeObservableParams.W_Lmu_Lnu_pairs.clear();
      gaugeObservableParams.nested_child_offset.clear();

      gaugeObservableParams.measurement_interval =
          gp["measurement_interval"].as<size_t>(0);
      gaugeObservableParams.measure_plaquette =
          gp["measure_plaquette"].as<bool>(false);
      gaugeObservableParams.measure_wilson_loop_temporal =
          gp["measure_wilson_loop_temporal"].as<bool>(false);
      gaugeObservableParams.measure_wilson_loop_mu_nu =
          gp["measure_wilson_loop_mu_nu"].as<bool>(false);
      gaugeObservableParams.measure_retrace_U =
          gp["measure_retrace_U"].as<bool>(false);

      gaugeObservableParams.wilson_loop_multihit =
          gp["wilson_loop_multihit"].as<index_t>(1);

      gaugeObservableParams.measure_nested_wilson_action =
          gp["measure_nested_wilson_action"].as<bool>(false);

      if (gp["nested_child_offset"]) {
        if (!gp["nested_child_offset"].IsSequence()) {
          printf("Error: nested_child_offset must be a YAML sequence\n");
          return false;
        }

        for (const auto &x : gp["nested_child_offset"]) {
          const index_t val = x.as<index_t>();
          if (val != 0 && val != 1) {
            printf("Error: nested_child_offset entries must be 0 or 1\n");
            return false;
          }
          gaugeObservableParams.nested_child_offset.push_back(val);
        }
      }

      if (gp["W_temp_L_T_pairs"]) {
        for (const auto &pair : gp["W_temp_L_T_pairs"]) {
          auto v1 = parseIndexRange(pair[0]);
          auto v2 = parseIndexRange(pair[1]);
          for (auto i1 : v1) {
            for (auto i2 : v2) {
              gaugeObservableParams.W_temp_L_T_pairs.push_back(
                  IndexArray<2>({i1, i2}));
            }
          }
        }
      }

      if (gp["W_mu_nu_pairs"]) {
        for (const auto &pair : gp["W_mu_nu_pairs"]) {
          auto v1 = parseIndexRange(pair[0]);
          auto v2 = parseIndexRange(pair[1]);
          for (auto i1 : v1) {
            for (auto i2 : v2) {
              gaugeObservableParams.W_mu_nu_pairs.push_back(
                  IndexArray<2>({i1, i2}));
            }
          }
        }
      }

      if (gp["W_Lmu_Lnu_pairs"]) {
        for (const auto &pair : gp["W_Lmu_Lnu_pairs"]) {
          auto v1 = parseIndexRange(pair[0]);
          auto v2 = parseIndexRange(pair[1]);
          for (auto i1 : v1) {
            for (auto i2 : v2) {
              gaugeObservableParams.W_Lmu_Lnu_pairs.push_back(
                  IndexArray<2>({i1, i2}));
            }
          }
        }
      }

      gaugeObservableParams.plaquette_filename =
          gp["plaquette_filename"].as<std::string>("");
      gaugeObservableParams.W_temp_filename =
          gp["W_temp_filename"].as<std::string>("");
      gaugeObservableParams.W_mu_nu_filename =
          gp["W_mu_nu_filename"].as<std::string>("");
      gaugeObservableParams.RetraceU_filename =
          gp["RetraceU_filename"].as<std::string>("");
      gaugeObservableParams.nested_wilson_action_filename =
          gp["nested_wilson_action_filename"].as<std::string>("");

      // whether to write to file
      gaugeObservableParams.write_to_file = gp["write_to_file"].as<bool>(false);

      // validation for nested observable
      if (gaugeObservableParams.wilson_loop_multihit < 1) {
        printf("Error: wilson_loop_multihit must be >= 1\n");
        return false;
      }

      if (gaugeObservableParams.measure_nested_wilson_action &&
          !gp["nested_child_offset"]) {
        printf("Error: measure_nested_wilson_action is true but "
               "nested_child_offset is missing\n");
        return false;
      }

      if (gaugeObservableParams.measure_nested_wilson_action &&
          gaugeObservableParams.write_to_file &&
          gaugeObservableParams.nested_wilson_action_filename.empty()) {
        printf("Error: measure_nested_wilson_action is true but "
               "nested_wilson_action_filename is empty\n");
        return false;
      }

      // ...
      // add more parameters above this line as needed
    } else {
      printf("Error: GaugeObservableParams not found in input file\n");
      return false;
    }

    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing input file: %s\n", e.what());
    return false;
  }
}

} // namespace klft
