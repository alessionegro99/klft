#include "core/compiled_theory.hpp"
#include "io/driver_utils.hpp"
#include "io/gauge_configuration.hpp"
#include "io/orbifold_configuration.hpp"
#include "orbifold.hpp"

#include <Kokkos_Core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace klft;

struct RunParams {
  IndexArray<4> dimensions;
  uint64_t seed;
  size_t thermalization_trajectories;
  size_t production_trajectories;
  size_t measurement_interval;
  size_t tuning_trajectories;
  size_t maximum_tuning_rounds;
  real_t target_acceptance_min;
  real_t target_acceptance_max;
  std::string start;
  real_t initialization_noise;
  std::string configuration_input;
  std::string configuration_output;
  size_t checkpoint_interval;
  size_t diagnostic_interval;
};

struct ObservableParams {
  index_t max_r;
  index_t max_t;
  std::string hmc_filename;
  std::string wilson_loop_filename;
  std::string diagnostic_filename;
};

struct InputParams {
  RunParams run;
  OrbifoldActionParams action;
  OrbifoldHMCParams hmc;
  ObservableParams observables;
};

YAML::Node required_section(const YAML::Node &config, const char *name) {
  const YAML::Node section = config[name];
  if (!section || !section.IsMap()) {
    throw std::runtime_error(std::string("Missing YAML map: ") + name);
  }
  return section;
}

template <class T>
T required_value(const YAML::Node &section, const char *section_name,
                 const char *key) {
  const YAML::Node value = section[key];
  if (!value) {
    throw std::runtime_error(std::string("Missing YAML key: ") +
                             section_name + "." + key);
  }
  try {
    return value.as<T>();
  } catch (const YAML::Exception &error) {
    throw std::runtime_error(std::string("Invalid YAML value for ") +
                             section_name + "." + key + ": " +
                             error.what());
  }
}

InputParams parse_input(const std::string &filename) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(filename);
  } catch (const YAML::Exception &error) {
    throw std::runtime_error("Could not parse '" + filename + "': " +
                             error.what());
  }

  InputParams params;
  const auto input = required_section(config, "OrbifoldHMCParams");
  const char *section = "OrbifoldHMCParams";
  constexpr const char *extent_keys[4] = {"L0", "L1", "L2", "L3"};
  for (size_t d = 0; d < 4; ++d) {
    params.run.dimensions[d] =
        required_value<index_t>(input, section, extent_keys[d]);
    if (params.run.dimensions[d] <= 0) {
      throw std::runtime_error("Lattice extents must be positive.");
    }
  }
  params.run.seed = required_value<uint64_t>(input, section, "seed");
  params.run.thermalization_trajectories = required_value<size_t>(
      input, section, "thermalization_trajectories");
  params.run.production_trajectories = required_value<size_t>(
      input, section, "production_trajectories");
  params.run.measurement_interval = required_value<size_t>(
      input, section, "measure_every");
  if (params.run.production_trajectories > 0 &&
      params.run.measurement_interval == 0) {
    throw std::runtime_error(
        "OrbifoldHMCParams.measure_every must be positive.");
  }
  params.run.tuning_trajectories =
      required_value<size_t>(input, section, "tuning_trajectories");
  params.run.maximum_tuning_rounds =
      required_value<size_t>(input, section, "maximum_tuning_rounds");
  params.run.target_acceptance_min =
      required_value<real_t>(input, section, "target_acceptance_min");
  params.run.target_acceptance_max =
      required_value<real_t>(input, section, "target_acceptance_max");
  if (params.run.tuning_trajectories != 0) {
    throw std::runtime_error(
        "OrbifoldHMCParams.tuning_trajectories must be zero; this driver "
        "does not tune the integrator automatically.");
  }
  if (params.run.maximum_tuning_rounds == 0 ||
      params.run.target_acceptance_min < 0.0 ||
      params.run.target_acceptance_max > 1.0 ||
      params.run.target_acceptance_min >=
          params.run.target_acceptance_max) {
    throw std::runtime_error("Invalid HMC tuning bounds.");
  }
  if (input["start"]) {
    params.run.start = required_value<std::string>(input, section, "start");
  } else {
    params.run.start = required_value<bool>(input, section, "hot_start")
                           ? "hot"
                           : "cold";
  }
  if (params.run.start != "cold" && params.run.start != "hot" &&
      params.run.start != "restart" && params.run.start != "compact") {
    throw std::runtime_error(
        "OrbifoldHMCParams.start must be cold, hot, restart, or compact.");
  }
  params.run.initialization_noise =
      required_value<real_t>(input, section, "initialization_noise");
  if (params.run.initialization_noise < 0.0) {
    throw std::runtime_error(
        "OrbifoldHMCParams.initialization_noise must be non-negative.");
  }
  params.run.configuration_input =
      input["configuration_input"].as<std::string>("");
  params.run.configuration_output =
      input["configuration_output"].as<std::string>("");
  params.run.checkpoint_interval =
      input["checkpoint_every"].as<size_t>(0);
  params.run.diagnostic_interval =
      input["diagnostic_every"].as<size_t>(0);
  if ((params.run.start == "restart" || params.run.start == "compact") &&
      params.run.configuration_input.empty()) {
    throw std::runtime_error(
        "OrbifoldHMCParams.configuration_input is required for restart and "
        "compact starts.");
  }
  if (params.run.checkpoint_interval > 0 &&
      params.run.configuration_output.empty()) {
    throw std::runtime_error(
        "OrbifoldHMCParams.configuration_output is required for "
        "checkpointing.");
  }
  if (!params.run.configuration_input.empty() &&
      params.run.configuration_input == params.run.configuration_output) {
    throw std::runtime_error(
        "Orbifold input and output configurations must be distinct.");
  }

  params.action.spatial_spacing = required_value<real_t>(
      input, section, "a_s");
  params.action.temporal_spacing = required_value<real_t>(
      input, section, "a_t");
  params.action.coupling = required_value<real_t>(input, section, "g");
  params.action.scalar_mass =
      required_value<real_t>(input, section, "mass");
  params.action.u1_mass =
      required_value<real_t>(input, section, "u1_mass");
  params.action.validate();

  const real_t trajectory_length =
      required_value<real_t>(input, section, "trajectory_length");
  const real_t initial_step_size =
      required_value<real_t>(input, section, "initial_step_size");
  if (!(trajectory_length > 0.0) || !(initial_step_size > 0.0)) {
    throw std::runtime_error(
        "Orbifold trajectory length and step size must be positive.");
  }
  const real_t requested_steps = trajectory_length / initial_step_size;
  if (requested_steps >
      static_cast<real_t>(std::numeric_limits<index_t>::max())) {
    throw std::runtime_error("Orbifold HMC step count is too large.");
  }
  params.hmc.steps =
      std::max<index_t>(1, static_cast<index_t>(std::llround(requested_steps)));
  params.hmc.step_size = trajectory_length / params.hmc.steps;
  params.hmc.validate();

  params.observables.max_r = required_value<index_t>(
      input, section, "max_spatial_separation");
  params.observables.max_t = required_value<index_t>(
      input, section, "max_temporal_extent");
  params.observables.hmc_filename = required_value<std::string>(
      input, section, "hmc_output");
  params.observables.wilson_loop_filename = required_value<std::string>(
      input, section, "wilson_loop_output");
  params.observables.diagnostic_filename =
      input["diagnostic_output"].as<std::string>("");

  const index_t min_spatial_extent =
      std::min({params.run.dimensions[0], params.run.dimensions[1],
                params.run.dimensions[2]});
  if (params.observables.max_r <= 0 || params.observables.max_t <= 0 ||
      params.observables.max_r > min_spatial_extent / 2 ||
      params.observables.max_t > params.run.dimensions[3] / 2) {
    throw std::runtime_error(
        "Wilson-loop extents must be positive and no larger than half the "
        "corresponding periodic lattice extent.");
  }
  if (params.observables.hmc_filename.empty() ||
      params.observables.wilson_loop_filename.empty() ||
      params.observables.hmc_filename ==
          params.observables.wilson_loop_filename) {
    throw std::runtime_error(
        "Output filenames must be nonempty and distinct.");
  }
  if (params.run.diagnostic_interval > 0 &&
      params.observables.diagnostic_filename.empty()) {
    throw std::runtime_error(
        "OrbifoldHMCParams.diagnostic_output is required when "
        "diagnostic_every is positive.");
  }
  if (!params.observables.diagnostic_filename.empty() &&
      (params.observables.diagnostic_filename ==
           params.observables.hmc_filename ||
       params.observables.diagnostic_filename ==
           params.observables.wilson_loop_filename)) {
    throw std::runtime_error("Orbifold output filenames must be distinct.");
  }
  return params;
}

void refuse_existing_output(const std::string &filename) {
  if (std::filesystem::exists(filename)) {
    throw std::runtime_error("Refusing to overwrite existing output: " +
                             filename);
  }
}

void write_metadata(std::ostream &output, const InputParams &params) {
  output << std::setprecision(17)
         << "# theory SU(3) orbifold, 3+1D, periodic, time_direction=3\n"
         << "# dimensions " << params.run.dimensions[0] << " "
         << params.run.dimensions[1] << " " << params.run.dimensions[2]
         << " " << params.run.dimensions[3] << "\n"
         << "# spatial_spacing " << params.action.spatial_spacing << "\n"
         << "# temporal_spacing " << params.action.temporal_spacing << "\n"
         << "# coupling " << params.action.coupling << "\n"
         << "# scalar_mass " << params.action.scalar_mass << "\n"
         << "# u1_mass " << params.action.u1_mass << "\n"
         << "# seed " << params.run.seed << "\n"
         << "# start " << params.run.start << "\n"
         << "# initialization_noise " << params.run.initialization_noise
         << "\n"
         << "# hmc_seed "
         << (params.run.start == "hot" ? params.run.seed + 1
                                         : params.run.seed)
         << "\n"
         << "# hmc_step_size " << params.hmc.step_size << "\n"
         << "# hmc_steps " << params.hmc.steps << "\n"
         << "# hmc_trajectory_length "
         << params.hmc.step_size * params.hmc.steps << "\n"
         << "# thermalization_trajectories "
         << params.run.thermalization_trajectories << "\n"
         << "# production_trajectories "
         << params.run.production_trajectories << "\n"
         << "# measurement_interval " << params.run.measurement_interval
         << "\n"
         << "# diagnostic_interval " << params.run.diagnostic_interval
         << "\n"
         << "# checkpoint_interval " << params.run.checkpoint_interval
         << "\n"
         << "# configuration_input " << params.run.configuration_input
         << "\n"
         << "# configuration_output " << params.run.configuration_output
         << "\n";
}

int run(const std::string &filename) {
  const InputParams params = parse_input(filename);
  refuse_existing_output(params.observables.hmc_filename);
  refuse_existing_output(params.observables.wilson_loop_filename);
  if (!params.observables.diagnostic_filename.empty()) {
    refuse_existing_output(params.observables.diagnostic_filename);
  }
  if (!params.run.configuration_output.empty()) {
    refuse_existing_output(params.run.configuration_output);
  }

  std::ofstream hmc_output(params.observables.hmc_filename);
  std::ofstream wilson_output(params.observables.wilson_loop_filename);
  std::ofstream diagnostic_output;
  if (!params.observables.diagnostic_filename.empty()) {
    diagnostic_output.open(params.observables.diagnostic_filename);
  }
  if (!hmc_output || !wilson_output ||
      (!params.observables.diagnostic_filename.empty() &&
       !diagnostic_output)) {
    throw std::runtime_error("Could not create orbifold output files.");
  }
  write_metadata(hmc_output, params);
  hmc_output << "# trajectory production_trajectory accepted delta_h action "
                "initial_hamiltonian final_hamiltonian\n";
  write_metadata(wilson_output, params);
  wilson_output << "# Wilson loops use the unitary polar part of Z_j and the "
                   "explicit temporal links U_0\n"
                << "# trajectory R T ReTrW_over_Nc\n";
  if (diagnostic_output) {
    write_metadata(diagnostic_output, params);
    diagnostic_output
        << "# trajectory production_trajectory action W11\n";
  }

  const SUN<3> vacuum = identitySUN<3>() *
                        std::sqrt(params.action.vacuum_scale_squared());
  OrbifoldField field(params.run.dimensions, vacuum, identitySUN<3>(),
                       "orbifold_chain");
  if (params.run.start == "hot") {
    Kokkos::Random_XorShift64_Pool<> initialization_rng(params.run.seed);
    initialize_hot_orbifold_field(field, params.action,
                                  params.run.initialization_noise,
                                  initialization_rng);
  } else if (params.run.start == "restart") {
    if (!load_orbifold_configuration(params.run.configuration_input, field,
                                      params.action)) {
      throw std::runtime_error("Could not load orbifold restart.");
    }
  } else if (params.run.start == "compact") {
    auto gauge = make_identity_gauge_field<4, 3>(
        params.run.dimensions[0], params.run.dimensions[1],
        params.run.dimensions[2], params.run.dimensions[3]);
    if (!load_gauge_configuration<4, 3>(params.run.configuration_input, gauge,
                                        false)) {
      throw std::runtime_error("Could not load compact SU(3) start.");
    }
    initialize_orbifold_from_gauge(field, gauge, params.action);
  }
  const uint64_t hmc_seed =
      params.run.start == "hot" ? params.run.seed + 1 : params.run.seed;
  OrbifoldHMC hmc(field, params.action, params.hmc, hmc_seed);

  size_t thermalization_accepts = 0;
  size_t production_accepts = 0;
  const size_t total_trajectories = params.run.thermalization_trajectories +
                                    params.run.production_trajectories;
  for (size_t trajectory = 1; trajectory <= total_trajectories;
       ++trajectory) {
    const bool thermalizing =
        trajectory <= params.run.thermalization_trajectories;
    const size_t production_trajectory =
        thermalizing ? 0
                     : trajectory - params.run.thermalization_trajectories;
    const OrbifoldHMCResult result = hmc.step();
    const real_t action = orbifold_action(field, params.action);
    if (!std::isfinite(result.initial_hamiltonian) ||
        !std::isfinite(result.final_hamiltonian) ||
        !std::isfinite(result.delta_hamiltonian) || !std::isfinite(action)) {
      throw std::runtime_error(
          "Non-finite HMC value at trajectory " +
          std::to_string(trajectory) + "; reduce the step size.");
    }

    if (thermalizing) {
      thermalization_accepts += result.accepted;
    } else {
      production_accepts += result.accepted;
    }
    hmc_output << trajectory << " " << production_trajectory << " "
               << static_cast<int>(result.accepted) << " "
               << result.delta_hamiltonian << " " << action << " "
               << result.initial_hamiltonian << " "
               << result.final_hamiltonian << "\n";
    hmc_output.flush();
    if (!hmc_output) {
      throw std::runtime_error("Failed writing HMC output.");
    }

    if (params.run.diagnostic_interval > 0 &&
        trajectory % params.run.diagnostic_interval == 0) {
      const auto loop = orbifold_wilson_loops(field, 1, 1);
      if (loop.size() != 1 || !std::isfinite(loop[0][2])) {
        throw std::runtime_error("Invalid W(1,1) diagnostic at trajectory " +
                                 std::to_string(trajectory) + ".");
      }
      diagnostic_output << trajectory << " " << production_trajectory << " "
                        << action << " " << loop[0][2] << "\n";
      diagnostic_output.flush();
      if (!diagnostic_output) {
        throw std::runtime_error("Failed writing orbifold diagnostics.");
      }
    }

    if (!thermalizing &&
        production_trajectory % params.run.measurement_interval == 0) {
      const auto loops = orbifold_wilson_loops(
          field, params.observables.max_r, params.observables.max_t);
      for (const auto &loop : loops) {
        if (!std::isfinite(loop[2])) {
          throw std::runtime_error("Non-finite Wilson loop at trajectory " +
                                   std::to_string(trajectory) + ".");
        }
        wilson_output << trajectory << " " << loop[0] << " " << loop[1]
                      << " " << loop[2] << "\n";
      }
      wilson_output.flush();
      if (!wilson_output) {
        throw std::runtime_error("Failed writing Wilson-loop output.");
      }
    }
    if (!params.run.configuration_output.empty() &&
        params.run.checkpoint_interval > 0 &&
        trajectory % params.run.checkpoint_interval == 0 &&
        !save_orbifold_configuration_atomic(
            params.run.configuration_output, field, params.action)) {
      throw std::runtime_error("Could not save orbifold checkpoint.");
    }
  }

  if (!params.run.configuration_output.empty() &&
      (params.run.checkpoint_interval == 0 || total_trajectories == 0 ||
       total_trajectories % params.run.checkpoint_interval != 0) &&
      !save_orbifold_configuration_atomic(
          params.run.configuration_output, field, params.action)) {
    throw std::runtime_error("Could not save final orbifold checkpoint.");
  }

  const real_t thermalization_acceptance =
      params.run.thermalization_trajectories == 0
          ? 0.0
          : static_cast<real_t>(thermalization_accepts) /
                params.run.thermalization_trajectories;
  const real_t production_acceptance =
      params.run.production_trajectories == 0
          ? 0.0
          : static_cast<real_t>(production_accepts) /
                params.run.production_trajectories;
  std::printf("Orbifold HMC complete: thermalization acceptance %.6f, "
              "production acceptance %.6f\n",
              thermalization_acceptance, production_acceptance);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    std::string input_file;
    const int parse_result = klft::parse_driver_args(argc, argv, input_file);
    if (parse_result == 0) {
      result = run(input_file);
    } else if (parse_result == 1) {
      std::printf("Use -f FILE to run the orbifold HMC driver.\n");
      result = 1;
    } else if (parse_result == -2) {
      result = 0;
    } else {
      result = 1;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Error: %s\n", error.what());
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
