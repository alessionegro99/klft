#pragma once

#include "core/common.hpp"

#include <string>
#include <vector>

namespace klft {

struct GradientFlowParams {
  bool enabled;
  real_t epsilon;
  std::vector<real_t> times_tau;

  bool measure_plaquette;
  bool measure_action;
  bool measure_wilson_loop_temporal;
  bool measure_wilson_loop_mu_nu;

  bool check_action_monotonicity;
  bool check_group_properties;
  bool reunitarize;

  std::string obs_filename;
  std::string W_temp_filename;
  std::string W_mu_nu_filename;

  GradientFlowParams()
      : enabled(false), epsilon(0.01), times_tau{0.0},
        measure_plaquette(true), measure_action(true),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        check_action_monotonicity(true), check_group_properties(true),
        reunitarize(false), obs_filename("gradient_flow_obs.dat"),
        W_temp_filename("gradient_flow_wtemp.dat"),
        W_mu_nu_filename("gradient_flow_w_mu_nu.dat") {}
};

} // namespace klft
