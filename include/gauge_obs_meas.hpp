#pragma once
#include "gauge_obs_def.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace klft {

struct GaugeObservableParams {
  size_t measurement_interval;

  bool measure_plaquette;
  bool measure_wilson_loop_temporal;
  bool measure_wilson_loop_mu_nu;
  bool measure_retrace_U;
  bool measure_nested_wilson_action;

  std::vector<index_t> nested_child_offset;

  std::vector<Kokkos::Array<index_t, 2>> W_temp_L_T_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;

  std::vector<size_t> measurement_steps;
  std::vector<real_t> measurement_acceptance_rates;
  std::vector<real_t> measurement_times;

  std::vector<real_t> plaquette_measurements;
  std::vector<real_t> retraceU_measurements;
  std::vector<std::vector<WilsonLoopTemporalMeasurement>> W_temp_measurements;
  std::vector<std::vector<WilsonLoopMuNuMeasurement>> W_mu_nu_measurements;
  std::vector<real_t> nested_plaq_V_measurements;
  std::vector<real_t> nested_plaq_child_measurements;
  std::vector<real_t> nested_E_V_measurements;
  std::vector<real_t> nested_E_child_measurements;

  std::string plaquette_filename;
  std::string W_temp_filename;
  std::string W_mu_nu_filename;
  std::string RetraceU_filename;
  std::string nested_wilson_action_filename;

  bool write_to_file;

  GaugeObservableParams()
      : measurement_interval(0), measure_plaquette(false),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        measure_retrace_U(false), measure_nested_wilson_action(false),
        write_to_file(false) {}
};

inline void requireOpen(const std::ofstream &file) {
  if (!file.is_open()) {
    throw std::runtime_error("Output file is not open.");
  }
}

inline void requireEnabled(const bool enabled, const char *what) {
  if (!enabled) {
    throw std::runtime_error(
        std::string("Requested flush for disabled observable: ") + what);
  }
}

inline void requireSameSize(const size_t a, const size_t b, const char *what) {
  if (a != b) {
    throw std::runtime_error(std::string("Inconsistent sizes for ") + what);
  }
}

template <size_t rank, size_t Nc>
void measureGaugeObservables(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    GaugeObservableParams &params, const size_t step, const real_t acc_rate,
    const real_t time) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }

  if (params.measure_plaquette) {
    const real_t P = GaugePlaquette<rank, Nc>(g_in);
    params.plaquette_measurements.push_back(P);
  }

  if (params.measure_wilson_loop_temporal) {
    std::vector<WilsonLoopTemporalMeasurement> temp_measurements;
    WilsonLoop_temporal<rank, Nc>(g_in, params.W_temp_L_T_pairs,
                                  temp_measurements);

    params.W_temp_measurements.push_back(std::move(temp_measurements));
  }

  if (params.measure_wilson_loop_mu_nu) {
    std::vector<WilsonLoopMuNuMeasurement> temp_measurements;

    for (const auto &pair_mu_nu : params.W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<rank, Nc>(g_in, mu, nu, params.W_Lmu_Lnu_pairs,
                                 temp_measurements);
    }

    params.W_mu_nu_measurements.push_back(std::move(temp_measurements));
  }

  if (params.measure_retrace_U) {
    const real_t R = Retrace_links_avg<rank, Nc>(g_in);
    params.retraceU_measurements.push_back(R);
  }

  if (params.measure_nested_wilson_action) {
    if (params.nested_child_offset.size() != rank) {
      throw std::runtime_error(
          "nested_child_offset must have size equal to rank.");
    }

    IndexArray<rank> child_offset;
    for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
      child_offset[d] = params.nested_child_offset[d];
    }

    const auto res =
        MeasureNestedWilsonActionsOneLevel<rank, Nc>(g_in, child_offset);

    params.nested_plaq_V_measurements.push_back(res.plaq_V);
    params.nested_plaq_child_measurements.push_back(res.plaq_child);
    params.nested_E_V_measurements.push_back(res.E_V);
    params.nested_E_child_measurements.push_back(res.E_child);
  }

  params.measurement_steps.push_back(step);
  params.measurement_acceptance_rates.push_back(acc_rate);
  params.measurement_times.push_back(time);
}

inline void flushPlaquette(std::ofstream &file,
                           const GaugeObservableParams &params,
                           const bool HEADER = true) {
  requireOpen(file);
  requireEnabled(params.measure_plaquette, "plaquette");

  requireSameSize(params.measurement_steps.size(),
                  params.plaquette_measurements.size(),
                  "plaquette measurements");
  requireSameSize(params.measurement_steps.size(),
                  params.measurement_acceptance_rates.size(),
                  "plaquette acceptance rates");
  requireSameSize(params.measurement_steps.size(),
                  params.measurement_times.size(), "plaquette times");

  if (HEADER) {
    file << "# step, plaquette, acceptance_rate, time\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.plaquette_measurements[i] << ", "
         << params.measurement_acceptance_rates[i] << ", "
         << params.measurement_times[i] << "\n";
  }
}

inline void flushWilsonLoopTemporal(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  requireOpen(file);
  requireEnabled(params.measure_wilson_loop_temporal, "temporal Wilson loop");

  requireSameSize(params.measurement_steps.size(),
                  params.W_temp_measurements.size(),
                  "temporal Wilson loop measurements");

  if (HEADER) {
    file << "# step, L, T, W_temp\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement.L << ", "
           << measurement.T << ", " << measurement.value << "\n";
    }
  }
}

inline void flushWilsonLoopMuNu(std::ofstream &file,
                                const GaugeObservableParams &params,
                                const bool HEADER = true) {
  requireOpen(file);
  requireEnabled(params.measure_wilson_loop_mu_nu, "mu-nu Wilson loop");

  requireSameSize(params.measurement_steps.size(),
                  params.W_mu_nu_measurements.size(),
                  "mu-nu Wilson loop measurements");

  if (HEADER) {
    file << "# step, mu, nu, Lmu, Lnu, W_mu_nu\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_mu_nu_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement.mu << ", "
           << measurement.nu << ", " << measurement.Lmu << ", "
           << measurement.Lnu << ", " << measurement.value << "\n";
    }
  }
}

inline void flushRetraceU(std::ofstream &file,
                          const GaugeObservableParams &params,
                          const bool HEADER = true) {
  requireOpen(file);
  requireEnabled(params.measure_retrace_U, "Retrace(U)");

  requireSameSize(params.measurement_steps.size(),
                  params.retraceU_measurements.size(),
                  "Retrace(U) measurements");

  if (HEADER) {
    file << "# step, Retrace(U)\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.retraceU_measurements[i] << "\n";
  }
}

inline void flushNestedWilsonAction(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  requireOpen(file);
  requireEnabled(params.measure_nested_wilson_action, "nested Wilson action");

  const size_t n = params.measurement_steps.size();
  requireSameSize(n, params.nested_plaq_V_measurements.size(),
                  "nested plaq_V measurements");
  requireSameSize(n, params.nested_plaq_child_measurements.size(),
                  "nested plaq_child measurements");
  requireSameSize(n, params.nested_E_V_measurements.size(),
                  "nested E_V measurements");
  requireSameSize(n, params.nested_E_child_measurements.size(),
                  "nested E_child measurements");

  if (HEADER) {
    file << "# step, plaq_V, plaq_child, E_V, E_child\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < n; ++i) {
    file << params.measurement_steps[i] << ", "
         << params.nested_plaq_V_measurements[i] << ", "
         << params.nested_plaq_child_measurements[i] << ", "
         << params.nested_E_V_measurements[i] << ", "
         << params.nested_E_child_measurements[i] << "\n";
  }
}

inline void flushAllGaugeObservables(const GaugeObservableParams &params,
                                     const bool HEADER = true) {
  if (!params.write_to_file) {
    return;
  }

  if (!params.plaquette_filename.empty()) {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    flushPlaquette(file, params, HEADER);
  }

  if (!params.W_temp_filename.empty()) {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    flushWilsonLoopTemporal(file, params, HEADER);
  }

  if (!params.W_mu_nu_filename.empty()) {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    flushWilsonLoopMuNu(file, params, HEADER);
  }

  if (!params.RetraceU_filename.empty()) {
    std::ofstream file(params.RetraceU_filename, std::ios::app);
    flushRetraceU(file, params, HEADER);
  }

  if (!params.nested_wilson_action_filename.empty()) {
    std::ofstream file(params.nested_wilson_action_filename, std::ios::app);
    flushNestedWilsonAction(file, params, HEADER);
  }
}

inline void clearAllGaugeObservables(GaugeObservableParams &params) {
  params.measurement_steps.clear();
  params.measurement_acceptance_rates.clear();
  params.measurement_times.clear();

  params.plaquette_measurements.clear();
  params.W_temp_measurements.clear();
  params.W_mu_nu_measurements.clear();
  params.retraceU_measurements.clear();

  params.nested_plaq_V_measurements.clear();
  params.nested_plaq_child_measurements.clear();
  params.nested_E_V_measurements.clear();
  params.nested_E_child_measurements.clear();
}

} // namespace klft
