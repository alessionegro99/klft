#pragma once

#include "core/klft_config.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"

#include <algorithm>
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

inline bool validateObservableFilenames(const GaugeObservableParams &params) {
  if (!params.write_to_file) {
    return true;
  }
  if (params.measure_plaquette && params.plaquette_filename.empty()) {
    printf("Error: measure_plaquette is enabled but plaquette_filename is "
           "empty\n");
    return false;
  }
  if (params.measure_wilson_loop_temporal && params.W_temp_filename.empty()) {
    printf("Error: measure_wilson_loop_temporal is enabled but W_temp_filename "
           "is empty\n");
    return false;
  }
  if (params.measure_wilson_loop_mu_nu && params.W_mu_nu_filename.empty()) {
    printf("Error: measure_wilson_loop_mu_nu is enabled but W_mu_nu_filename "
           "is empty\n");
    return false;
  }
  if (params.measure_polyakov_loop && params.polyakov_loop_filename.empty()) {
    printf("Error: measure_polyakov_loop is enabled but polyakov_loop_filename "
           "is empty\n");
    return false;
  }
  if (params.measure_polyakov_correlator &&
      params.polyakov_correlator_filename.empty()) {
    printf("Error: measure_polyakov_correlator is enabled but "
           "polyakov_correlator_filename is empty\n");
    return false;
  }
  if (params.measure_retrace_U && params.RetraceU_filename.empty()) {
    printf("Error: measure_retrace_U is enabled but RetraceU_filename is "
           "empty\n");
    return false;
  }
  if (params.measure_nested_wilson_action &&
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
                           GaugeObservableParams &gaugeObservableParams) {
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
  if (!validateObservableFilenames(gaugeObservableParams)) {
    return false;
  }

  return true;
}

inline void normalizeGradientFlowTValues(std::vector<real_t> &t_values) {
  t_values.push_back(0.0);
  std::sort(t_values.begin(), t_values.end());
  t_values.erase(std::unique(t_values.begin(), t_values.end(),
                              [](const real_t a, const real_t b) {
                                return Kokkos::abs(a - b) < 1.0e-14;
                              }),
                  t_values.end());
}

inline bool parseInputFile(const std::string &filename,
                           GradientFlowParams &gradientFlowParams) {
  YAML::Node config;
  if (!loadInputConfig(filename, config)) {
    return false;
  }

  gradientFlowParams = GradientFlowParams{};
  if (!config["GradientFlowParams"]) {
    return true;
  }

  const auto &fp = config["GradientFlowParams"];
  if (fp["time_definition"]) {
    const std::string time_definition = fp["time_definition"].as<std::string>();
    if (time_definition != "fixed_t") {
      printf("Error: GradientFlowParams.time_definition must be fixed_t; "
             "flow times are t/a^2\n");
      return false;
    }
  }
  const std::string integrator = fp["integrator"].as<std::string>("rk3");
  if (integrator != "rk3") {
    printf("Error: GradientFlowParams.integrator must be rk3\n");
    return false;
  }

  gradientFlowParams.enabled = fp["enabled"].as<bool>(false);
  if (fp["epsilon"]) {
    printf("Error: GradientFlowParams.epsilon was renamed to dt\n");
    return false;
  }
  gradientFlowParams.dt = fp["dt"].as<real_t>(0.01);
  gradientFlowParams.measure_energy_clover =
      fp["measure_energy_clover"].as<bool>(true);
  gradientFlowParams.measure_wilson_loop_temporal =
      fp["measure_wilson_loop_temporal"].as<bool>(false);
  gradientFlowParams.measure_wilson_loop_mu_nu =
      fp["measure_wilson_loop_mu_nu"].as<bool>(false);
  gradientFlowParams.extract_t0 = fp["extract_t0"].as<bool>(false);
  gradientFlowParams.t0_target = fp["t0_target"].as<real_t>(0.3);
  gradientFlowParams.obs_filename =
      fp["obs_filename"].as<std::string>("gradient_flow_obs.dat");
  gradientFlowParams.W_temp_filename =
      fp["W_temp_filename"].as<std::string>("gradient_flow_wtemp.dat");
  gradientFlowParams.W_mu_nu_filename =
      fp["W_mu_nu_filename"].as<std::string>("gradient_flow_w_mu_nu.dat");
  gradientFlowParams.t0_filename =
      fp["t0_filename"].as<std::string>("gradient_flow_t0.dat");

  if (fp["times_tau"]) {
    printf("Error: GradientFlowParams.times_tau was renamed to t_values\n");
    return false;
  }
  if (fp["tau_values"]) {
    printf("Error: GradientFlowParams.tau_values was renamed to t_values\n");
    return false;
  }
  gradientFlowParams.t_values.clear();
  if (fp["t_values"]) {
    if (!fp["t_values"].IsSequence()) {
      printf("Error: GradientFlowParams.t_values must be a YAML sequence\n");
      return false;
    }
    for (const auto &entry : fp["t_values"]) {
      const real_t t = entry.as<real_t>();
      if (t < 0.0) {
        printf("Error: GradientFlowParams.t_values entries must be >= 0\n");
        return false;
      }
      gradientFlowParams.t_values.push_back(t);
    }
  }
  normalizeGradientFlowTValues(gradientFlowParams.t_values);

  if (gradientFlowParams.dt <= 0.0) {
    printf("Error: GradientFlowParams.dt must be > 0\n");
    return false;
  }
  if (gradientFlowParams.t0_target <= 0.0) {
    printf("Error: GradientFlowParams.t0_target must be > 0\n");
    return false;
  }

  return true;
}

inline bool
validateGradientFlowParams(const GradientFlowParams &gradientFlowParams,
                           const GaugeObservableParams &gaugeObservableParams) {
  if (!gradientFlowParams.enabled) {
    return true;
  }

  if (gaugeObservableParams.measurement_interval == 0) {
    printf("Error: GradientFlowParams.enabled requires "
           "GaugeObservableParams.measurement_interval > 0\n");
    return false;
  }
  if (gradientFlowParams.measure_energy_clover &&
      gradientFlowParams.obs_filename.empty()) {
    printf("Error: gradient-flow observable output requires "
           "obs_filename\n");
    return false;
  }
  if (gradientFlowParams.extract_t0 &&
      gradientFlowParams.t0_filename.empty()) {
    printf("Error: GradientFlowParams.extract_t0 requires t0_filename\n");
    return false;
  }
  if (gradientFlowParams.measure_wilson_loop_temporal) {
    if (gradientFlowParams.W_temp_filename.empty()) {
      printf("Error: gradient-flow temporal Wilson loops require "
             "W_temp_filename\n");
      return false;
    }
    if (gaugeObservableParams.W_temp_L_T_pairs.empty()) {
      printf("Error: gradient-flow temporal Wilson loops require "
             "GaugeObservableParams.W_temp_L_T_pairs\n");
      return false;
    }
  }
  if (gradientFlowParams.measure_wilson_loop_mu_nu) {
    if (gradientFlowParams.W_mu_nu_filename.empty()) {
      printf("Error: gradient-flow planar Wilson loops require "
             "W_mu_nu_filename\n");
      return false;
    }
    if (gaugeObservableParams.W_mu_nu_pairs.empty() ||
        gaugeObservableParams.W_Lmu_Lnu_pairs.empty()) {
      printf("Error: gradient-flow planar Wilson loops require "
             "GaugeObservableParams.W_mu_nu_pairs and W_Lmu_Lnu_pairs\n");
      return false;
    }
  }
  return true;
}

} // namespace klft
