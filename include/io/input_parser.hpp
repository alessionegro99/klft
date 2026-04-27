#pragma once

#include "core/klft_config.hpp"
#include "observables/gauge_observables.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"
#include "params/multilevel_params.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace klft {

inline bool loadInputConfig(const std::string &filename, YAML::Node &config) {
  try {
    config = YAML::LoadFile(filename);
    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing input file '%s': %s\n", filename.c_str(), e.what());
    return false;
  }
}

inline std::vector<index_t> parseIndexRange(const YAML::Node &node, bool &ok) {
  ok = true;
  std::vector<index_t> values;
  if (!node || !node.IsScalar()) {
    ok = false;
    return values;
  }

  const std::string token = node.as<std::string>();
  const size_t colon = token.find(':');
  if (colon == std::string::npos) {
    try {
      values.push_back(node.as<index_t>());
      return values;
    } catch (...) {
      ok = false;
      return values;
    }
  }

  try {
    const index_t start = std::stoi(token.substr(0, colon));
    const index_t end = std::stoi(token.substr(colon + 1));
    if (start > end) {
      ok = false;
      return values;
    }
    for (index_t i = start; i <= end; ++i) {
      values.push_back(i);
    }
    return values;
  } catch (...) {
    ok = false;
    return values;
  }
}

inline bool validateLatticeExtents(const index_t L0, const index_t L1,
                                   const index_t L2, const index_t L3,
                                   const char *section_name) {
  const index_t dims[4] = {L0, L1, L2, L3};
  for (size_t d = 0; d < compiled_rank; ++d) {
    if (dims[d] <= 0 || dims[d] % 2 != 0) {
      printf("Error: %s requires positive even lattice extents; "
             "dimension %zu is %d\n",
             section_name, d, dims[d]);
      return false;
    }
  }
  return true;
}

inline bool validateObservableFilenames(const GaugeObservableParams &params,
                                        const bool multilevel_output = false) {
  if (multilevel_output && params.measure_plaquette) {
    printf("Error: heatbath_multilevel does not support plaquette output\n");
    return false;
  }
  if (multilevel_output && params.measure_wilson_loop_temporal) {
    printf("Error: heatbath_multilevel does not support temporal Wilson loops\n");
    return false;
  }
  if (multilevel_output && params.measure_wilson_loop_mu_nu) {
    printf("Error: heatbath_multilevel does not support planar Wilson loops\n");
    return false;
  }
  if (multilevel_output && params.measure_retrace_U) {
    printf("Error: heatbath_multilevel does not support Retrace(U)\n");
    return false;
  }
  if (multilevel_output && params.measure_nested_wilson_action) {
    printf("Error: heatbath_multilevel does not support nested Wilson action\n");
    return false;
  }
  if (!params.write_to_file) {
    return true;
  }
  if (!multilevel_output && params.measure_plaquette &&
      params.plaquette_filename.empty()) {
    printf("Error: measure_plaquette is enabled but plaquette_filename is "
           "empty\n");
    return false;
  }
  if (!multilevel_output && params.measure_wilson_loop_temporal &&
      params.W_temp_filename.empty()) {
    printf("Error: measure_wilson_loop_temporal is enabled but W_temp_filename "
           "is empty\n");
    return false;
  }
  if (!multilevel_output && params.measure_wilson_loop_mu_nu &&
      params.W_mu_nu_filename.empty()) {
    printf("Error: measure_wilson_loop_mu_nu is enabled but W_mu_nu_filename "
           "is empty\n");
    return false;
  }
  if (params.measure_polyakov_loop && !multilevel_output &&
      params.polyakov_loop_filename.empty()) {
    printf("Error: measure_polyakov_loop is enabled but polyakov_loop_filename "
           "is empty\n");
    return false;
  }
  if (params.measure_polyakov_loop && multilevel_output &&
      params.polyakov_loop_multilevel_filename.empty()) {
    printf("Error: measure_polyakov_loop is enabled but "
           "polyakov_loop_multilevel_filename is empty\n");
    return false;
  }
  if (params.measure_polyakov_correlator && !multilevel_output &&
      params.polyakov_correlator_filename.empty()) {
    printf("Error: measure_polyakov_correlator is enabled but "
           "polyakov_correlator_filename is empty\n");
    return false;
  }
  if (params.measure_polyakov_correlator && multilevel_output &&
      params.polyakov_correlator_multilevel_filename.empty()) {
    printf("Error: measure_polyakov_correlator is enabled but "
           "polyakov_correlator_multilevel_filename is empty\n");
    return false;
  }
  if (!multilevel_output && params.measure_retrace_U &&
      params.RetraceU_filename.empty()) {
    printf("Error: measure_retrace_U is enabled but RetraceU_filename is "
           "empty\n");
    return false;
  }
  if (!multilevel_output && params.measure_nested_wilson_action &&
      params.nested_wilson_action_filename.empty()) {
    printf("Error: measure_nested_wilson_action is enabled but "
           "nested_wilson_action_filename is empty\n");
    return false;
  }
  return true;
}

inline bool parseInputFile(const std::string &filename,
                           MetropolisParams &metropolisParams) {
  YAML::Node config;
  if (!loadInputConfig(filename, config)) {
    return false;
  }
  if (!config["MetropolisParams"]) {
    printf("Error: MetropolisParams not found in input file\n");
    return false;
  }

  metropolisParams = MetropolisParams{};
  const auto &mp = config["MetropolisParams"];
  metropolisParams.L0 = mp["L0"].as<index_t>(32);
  metropolisParams.L1 = mp["L1"].as<index_t>(32);
  metropolisParams.L2 = mp["L2"].as<index_t>(32);
  metropolisParams.L3 = mp["L3"].as<index_t>(32);
  metropolisParams.nHits = mp["nHits"].as<index_t>(10);
  metropolisParams.nSweep = mp["nSweep"].as<index_t>(1000);
  metropolisParams.seed = mp["seed"].as<index_t>(1234);
  metropolisParams.beta = mp["beta"].as<real_t>(1.0);
  metropolisParams.delta = mp["delta"].as<real_t>(0.1);
  metropolisParams.epsilon1 = mp["epsilon1"].as<real_t>(0.0);
  metropolisParams.epsilon2 = mp["epsilon2"].as<real_t>(0.0);

  if (!validateLatticeExtents(metropolisParams.L0, metropolisParams.L1,
                              metropolisParams.L2, metropolisParams.L3,
                              "MetropolisParams")) {
    return false;
  }
  if (metropolisParams.nHits < 1) {
    printf("Error: nHits must be >= 1\n");
    return false;
  }
  if (metropolisParams.nSweep < 0) {
    printf("Error: nSweep must be >= 0\n");
    return false;
  }
  if (metropolisParams.delta <= 0.0) {
    printf("Error: delta must be > 0\n");
    return false;
  }

  return true;
}

inline bool parseInputFile(const std::string &filename,
                           HeatbathParams &heatbathParams) {
  YAML::Node config;
  if (!loadInputConfig(filename, config)) {
    return false;
  }
  if (!config["HeatbathParams"]) {
    printf("Error: HeatbathParams not found in input file\n");
    return false;
  }

  heatbathParams = HeatbathParams{};
  const auto &hp = config["HeatbathParams"];
  heatbathParams.L0 = hp["L0"].as<index_t>(32);
  heatbathParams.L1 = hp["L1"].as<index_t>(32);
  heatbathParams.L2 = hp["L2"].as<index_t>(32);
  heatbathParams.L3 = hp["L3"].as<index_t>(32);
  heatbathParams.nSweep = hp["nSweep"].as<index_t>(1000);
  heatbathParams.nOverrelax = hp["nOverrelax"].as<index_t>(5);
  heatbathParams.seed = hp["seed"].as<index_t>(1234);
  heatbathParams.beta = hp["beta"].as<real_t>(1.0);
  heatbathParams.delta = hp["delta"].as<real_t>(0.1);
  heatbathParams.epsilon1 = hp["epsilon1"].as<real_t>(0.0);
  heatbathParams.epsilon2 = hp["epsilon2"].as<real_t>(0.0);

  if (!validateLatticeExtents(heatbathParams.L0, heatbathParams.L1,
                              heatbathParams.L2, heatbathParams.L3,
                              "HeatbathParams")) {
    return false;
  }
  if (heatbathParams.nSweep < 0) {
    printf("Error: nSweep must be >= 0\n");
    return false;
  }
  if (heatbathParams.nOverrelax < 0) {
    printf("Error: nOverrelax must be >= 0\n");
    return false;
  }
  return true;
}

inline bool parseInputFile(const std::string &filename,
                           MultilevelParams &multilevelParams) {
  YAML::Node config;
  if (!loadInputConfig(filename, config)) {
    return false;
  }
  if (!config["MultilevelParams"]) {
    printf("Error: MultilevelParams not found in input file\n");
    return false;
  }

  multilevelParams = MultilevelParams{};
  const auto &mp = config["MultilevelParams"];
  const auto levels = mp["levels"];
  if (!levels || !levels.IsSequence() || levels.size() == 0) {
    printf("Error: MultilevelParams.levels must be a non-empty YAML sequence\n");
    return false;
  }

  for (const auto &level : levels) {
    if (!level.IsMap()) {
      printf("Error: each MultilevelParams.levels entry must be a map\n");
      return false;
    }
    const index_t slab_links = level["slab_links"].as<index_t>(0);
    const index_t updates = level["updates"].as<index_t>(0);
    if (slab_links < 2) {
      printf("Error: each multilevel slab_links must be >= 2\n");
      return false;
    }
    if (updates < 1) {
      printf("Error: each multilevel updates value must be >= 1\n");
      return false;
    }
    multilevelParams.levels.emplace_back(slab_links, updates);
  }

  for (size_t i = 1; i < multilevelParams.levels.size(); ++i) {
    const index_t prev = multilevelParams.levels[i - 1].slab_links;
    const index_t cur = multilevelParams.levels[i].slab_links;
    if (cur <= prev) {
      printf("Error: multilevel slab_links must be strictly ascending\n");
      return false;
    }
    if (cur % prev != 0) {
      printf("Error: each multilevel slab_links value must be divisible by the "
             "previous one\n");
      return false;
    }
  }

  return true;
}

inline bool validateMultilevelParams(const MultilevelParams &multilevelParams,
                                     const HeatbathParams &heatbathParams) {
  if (multilevelParams.levels.empty()) {
    printf("Error: MultilevelParams.levels must not be empty\n");
    return false;
  }

  const index_t dims[4] = {heatbathParams.L0, heatbathParams.L1,
                           heatbathParams.L2, heatbathParams.L3};
  const index_t Nt = dims[compiled_rank - 1];
  const index_t largest = multilevelParams.levels.back().slab_links;
  if (Nt % largest != 0) {
    printf("Error: the largest multilevel slab_links value must divide Nt\n");
    return false;
  }

  return true;
}

inline bool parseLoopLengthPairs(
    const YAML::Node &node, const char *field_name,
    std::vector<Kokkos::Array<index_t, 2>> &output) {
  if (!node) {
    return true;
  }
  if (!node.IsSequence()) {
    printf("Error: %s must be a YAML sequence\n", field_name);
    return false;
  }

  for (const auto &pair : node) {
    if (!pair.IsSequence() || pair.size() != 2) {
      printf("Error: each %s entry must have exactly two elements\n",
             field_name);
      return false;
    }
    bool ok0 = false;
    bool ok1 = false;
    const auto first = parseIndexRange(pair[0], ok0);
    const auto second = parseIndexRange(pair[1], ok1);
    if (!ok0 || !ok1) {
      printf("Error: failed to parse %s entry\n", field_name);
      return false;
    }
    for (const auto a : first) {
      for (const auto b : second) {
        if (a < 1 || b < 1) {
          printf("Error: %s entries must be positive\n", field_name);
          return false;
        }
        output.push_back(Kokkos::Array<index_t, 2>{a, b});
      }
    }
  }
  return true;
}

inline bool parsePlanePairs(
    const YAML::Node &node,
    std::vector<Kokkos::Array<index_t, 2>> &output) {
  if (!node) {
    return true;
  }
  if (!node.IsSequence()) {
    printf("Error: W_mu_nu_pairs must be a YAML sequence\n");
    return false;
  }

  for (const auto &pair : node) {
    if (!pair.IsSequence() || pair.size() != 2) {
      printf("Error: each W_mu_nu_pairs entry must have exactly two elements\n");
      return false;
    }
    bool ok0 = false;
    bool ok1 = false;
    const auto first = parseIndexRange(pair[0], ok0);
    const auto second = parseIndexRange(pair[1], ok1);
    if (!ok0 || !ok1) {
      printf("Error: failed to parse W_mu_nu_pairs entry\n");
      return false;
    }
    for (const auto mu : first) {
      for (const auto nu : second) {
        if (mu < 0 || nu < 0 || mu >= static_cast<index_t>(compiled_rank) ||
            nu >= static_cast<index_t>(compiled_rank)) {
          printf("Error: W_mu_nu_pairs entries must be between 0 and %zu\n",
                 compiled_rank - 1);
          return false;
        }
        if (mu == nu) {
          printf("Error: W_mu_nu_pairs entries must use distinct directions\n");
          return false;
        }
        output.push_back(Kokkos::Array<index_t, 2>{mu, nu});
      }
    }
  }
  return true;
}

inline bool parseInputFile(const std::string &filename,
                           GaugeObservableParams &gaugeObservableParams,
                           const bool multilevel_output = false) {
  YAML::Node config;
  if (!loadInputConfig(filename, config)) {
    return false;
  }
  if (!config["GaugeObservableParams"]) {
    printf("Error: GaugeObservableParams not found in input file\n");
    return false;
  }

  gaugeObservableParams = GaugeObservableParams{};
  const auto &gp = config["GaugeObservableParams"];

  gaugeObservableParams.measurement_interval =
      gp["measurement_interval"].as<size_t>(0);
  gaugeObservableParams.measure_plaquette =
      gp["measure_plaquette"].as<bool>(false);
  gaugeObservableParams.measure_wilson_loop_temporal =
      gp["measure_wilson_loop_temporal"].as<bool>(false);
  gaugeObservableParams.measure_wilson_loop_mu_nu =
      gp["measure_wilson_loop_mu_nu"].as<bool>(false);
  gaugeObservableParams.measure_polyakov_loop =
      gp["measure_polyakov_loop"].as<bool>(false);
  gaugeObservableParams.measure_polyakov_correlator =
      gp["measure_polyakov_correlator"].as<bool>(false);
  gaugeObservableParams.measure_retrace_U =
      gp["measure_retrace_U"].as<bool>(false);
  gaugeObservableParams.wilson_loop_multihit =
      gp["wilson_loop_multihit"].as<index_t>(1);
  gaugeObservableParams.polyakov_loop_multihit =
      gp["polyakov_loop_multihit"].as<index_t>(1);
  gaugeObservableParams.polyakov_correlator_max_r =
      gp["polyakov_correlator_max_r"].as<index_t>(0);
  gaugeObservableParams.measure_nested_wilson_action =
      gp["measure_nested_wilson_action"].as<bool>(false);
  gaugeObservableParams.write_to_file = gp["write_to_file"].as<bool>(false);

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

  if (!parseLoopLengthPairs(gp["W_temp_L_T_pairs"], "W_temp_L_T_pairs",
                            gaugeObservableParams.W_temp_L_T_pairs)) {
    return false;
  }
  if (!parsePlanePairs(gp["W_mu_nu_pairs"],
                       gaugeObservableParams.W_mu_nu_pairs)) {
    return false;
  }
  if (!parseLoopLengthPairs(gp["W_Lmu_Lnu_pairs"], "W_Lmu_Lnu_pairs",
                            gaugeObservableParams.W_Lmu_Lnu_pairs)) {
    return false;
  }

  gaugeObservableParams.plaquette_filename =
      gp["plaquette_filename"].as<std::string>("");
  gaugeObservableParams.W_temp_filename =
      gp["W_temp_filename"].as<std::string>("");
  gaugeObservableParams.W_mu_nu_filename =
      gp["W_mu_nu_filename"].as<std::string>("");
  gaugeObservableParams.polyakov_loop_filename =
      gp["polyakov_loop_filename"].as<std::string>("");
  gaugeObservableParams.polyakov_correlator_filename =
      gp["polyakov_correlator_filename"].as<std::string>("");
  gaugeObservableParams.polyakov_loop_multilevel_filename =
      gp["polyakov_loop_multilevel_filename"].as<std::string>("");
  gaugeObservableParams.polyakov_correlator_multilevel_filename =
      gp["polyakov_correlator_multilevel_filename"].as<std::string>("");
  gaugeObservableParams.RetraceU_filename =
      gp["RetraceU_filename"].as<std::string>("");
  gaugeObservableParams.nested_wilson_action_filename =
      gp["nested_wilson_action_filename"].as<std::string>("");

  const bool any_measurement_enabled =
      gaugeObservableParams.measure_plaquette ||
      gaugeObservableParams.measure_wilson_loop_temporal ||
      gaugeObservableParams.measure_wilson_loop_mu_nu ||
      gaugeObservableParams.measure_polyakov_loop ||
      gaugeObservableParams.measure_polyakov_correlator ||
      gaugeObservableParams.measure_retrace_U ||
      gaugeObservableParams.measure_nested_wilson_action;

  if (any_measurement_enabled && gaugeObservableParams.measurement_interval == 0) {
    printf("Error: measurement_interval must be > 0 when observables are "
           "enabled\n");
    return false;
  }
  if (gaugeObservableParams.wilson_loop_multihit < 1) {
    printf("Error: wilson_loop_multihit must be >= 1\n");
    return false;
  }
  if (gaugeObservableParams.polyakov_loop_multihit < 1) {
    printf("Error: polyakov_loop_multihit must be >= 1\n");
    return false;
  }
  if (gaugeObservableParams.polyakov_correlator_max_r < 0) {
    printf("Error: polyakov_correlator_max_r must be >= 0\n");
    return false;
  }
  if (gaugeObservableParams.measure_wilson_loop_temporal &&
      gaugeObservableParams.W_temp_L_T_pairs.empty()) {
    printf("Error: temporal Wilson loop measurement requires W_temp_L_T_pairs\n");
    return false;
  }
  if (gaugeObservableParams.measure_wilson_loop_mu_nu &&
      (gaugeObservableParams.W_mu_nu_pairs.empty() ||
       gaugeObservableParams.W_Lmu_Lnu_pairs.empty())) {
    printf("Error: planar Wilson loop measurement requires both W_mu_nu_pairs "
           "and W_Lmu_Lnu_pairs\n");
    return false;
  }
  if (gaugeObservableParams.measure_nested_wilson_action) {
    if (gaugeObservableParams.nested_child_offset.size() != compiled_rank) {
      printf("Error: nested_child_offset must have exactly %zu entries\n",
             compiled_rank);
      return false;
    }
  }
  if (!validateObservableFilenames(gaugeObservableParams, multilevel_output)) {
    return false;
  }

  return true;
}

} // namespace klft
