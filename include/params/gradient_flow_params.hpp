#pragma once

#include "core/common.hpp"

#include <string>
#include <vector>

namespace klft {

struct GradientFlowParams {
  bool enabled;
  real_t dt;
  std::vector<real_t> t_values;

  bool measure_energy_clover;
  bool measure_wilson_loop_temporal;
  bool measure_wilson_loop_mu_nu;

  bool extract_t0;
  real_t t0_target;

  std::string obs_filename;
  std::string W_temp_filename;
  std::string W_mu_nu_filename;
  std::string t0_filename;

  GradientFlowParams()
      : enabled(false), dt(0.01), t_values{0.0},
        measure_energy_clover(true),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        extract_t0(false), t0_target(0.3),
        obs_filename("gradient_flow_obs.dat"),
        W_temp_filename("gradient_flow_wtemp.dat"),
        W_mu_nu_filename("gradient_flow_w_mu_nu.dat"),
        t0_filename("gradient_flow_t0.dat") {}
};

} // namespace klft
