#pragma once
#include "GaugeEnergy.hpp"
#include "GaugePlaquette.hpp"
#include "GaugeRetrace.hpp"
#include "Metropolis_Params.hpp"
#include "WilsonLoop.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace klft {

struct GaugeObservableParams {
  size_t measurement_interval;
  bool measure_plaquette;
  bool measure_wilson_loop_temporal;
  bool measure_wilson_loop_mu_nu;
  bool measure_retrace_U;
  index_t wilson_loop_multihit;

  // nested Wilson-action observable
  bool measure_nested_wilson_action;
  std::vector<index_t> nested_child_offset;

  std::vector<Kokkos::Array<index_t, 2>> W_temp_L_T_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;

  std::vector<size_t> measurement_steps;
  std::vector<real_t> measurement_acceptance_rates;
  std::vector<real_t> measurement_times;

  std::vector<real_t> plaquette_measurements;
  std::vector<std::vector<Kokkos::Array<real_t, 3>>> W_temp_measurements;
  std::vector<std::vector<Kokkos::Array<real_t, 5>>> W_mu_nu_measurements;
  std::vector<real_t> retraceU_measurements;

  // nested Wilson-action measurements
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
        measure_retrace_U(false), wilson_loop_multihit(1),
        measure_nested_wilson_action(false), write_to_file(false) {}
};

inline void appendLatestGaugeObservables(const GaugeObservableParams &params);
inline void clearAllGaugeObservables(GaugeObservableParams &params);

template <size_t rank, size_t Nc, class RNG>
void measureGaugeObservables(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const MetropolisParams &metropolisParams, GaugeObservableParams &params,
    const size_t step, const real_t acc_rate, const real_t time,
    const RNG &rng) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }

  if (KLFT_VERBOSITY > 0) {
    printf("Measurement of Gauge Observables\n");
    printf("step: %zu\n", step);
  }

  if (params.measure_plaquette) {
    const real_t P = GaugePlaquette<rank, Nc>(g_in);
    params.plaquette_measurements.push_back(P);
    if (KLFT_VERBOSITY > 0) {
      printf("plaquette: %.12f\n", P);
    }
  }

  if (params.measure_wilson_loop_temporal) {
    if (KLFT_VERBOSITY > 0) {
      printf("temporal Wilson loop:\n");
      printf("L, T, W_temp\n");
    }

    std::vector<Kokkos::Array<real_t, 3>> temp_measurements;
    WilsonLoop_temporal<rank, Nc>(g_in, params.W_temp_L_T_pairs,
                                  temp_measurements,
                                  params.wilson_loop_multihit,
                                  metropolisParams.beta,
                                  metropolisParams.delta, rng);

    if (KLFT_VERBOSITY > 0) {
      for (const auto &measure : temp_measurements) {
        printf("%d, %d, %.12f\n", static_cast<index_t>(measure[0]),
               static_cast<index_t>(measure[1]), measure[2]);
      }
    }

    params.W_temp_measurements.push_back(temp_measurements);
  }

  if (params.measure_wilson_loop_mu_nu) {
    if (KLFT_VERBOSITY > 0) {
      printf("Wilson loop in the mu-nu plane:\n");
      printf("mu, nu, Lmu, Lnu, W_mu_nu\n");
    }

    std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
    for (const auto &pair_mu_nu : params.W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<rank, Nc>(g_in, mu, nu, params.W_Lmu_Lnu_pairs,
                                 temp_measurements,
                                 params.wilson_loop_multihit,
                                 metropolisParams.beta,
                                 metropolisParams.delta, rng);
    }

    if (KLFT_VERBOSITY > 0) {
      for (const auto &measure : temp_measurements) {
        printf("%d, %d, %d, %d, %.12f\n", static_cast<index_t>(measure[0]),
               static_cast<index_t>(measure[1]),
               static_cast<index_t>(measure[2]),
               static_cast<index_t>(measure[3]), measure[4]);
      }
    }

    params.W_mu_nu_measurements.push_back(temp_measurements);
  }

  if (params.measure_retrace_U) {
    const real_t R = Retrace_links_avg<rank, Nc>(g_in);
    params.retraceU_measurements.push_back(R);
    if (KLFT_VERBOSITY > 0) {
      printf("Retrace(U): %.12f\n", R);
    }
  }

  if (params.measure_nested_wilson_action) {
    if (params.nested_child_offset.size() != rank) {
      throw std::runtime_error(
          "nested_child_offset must have size equal to rank.");
    }

    IndexArray<rank> child_offset;
    for (index_t d = 0; d < rank; ++d) {
      child_offset[d] = params.nested_child_offset[d];
    }

    const auto res =
        MeasureNestedWilsonActionsOneLevel<rank, Nc>(g_in, child_offset);

    params.nested_plaq_V_measurements.push_back(res.plaq_V);
    params.nested_plaq_child_measurements.push_back(res.plaq_child);
    params.nested_E_V_measurements.push_back(res.E_V);
    params.nested_E_child_measurements.push_back(res.E_child);

    if (KLFT_VERBOSITY > 0) {
      printf("nested Wilson action:\n");
      printf("plaq(V): %.12f\n", res.plaq_V);
      printf("plaq(child): %.12f\n", res.plaq_child);
      printf("E(V): %.12f\n", res.E_V);
      printf("E(child): %.12f\n", res.E_child);
    }
  }

  params.measurement_steps.push_back(step);
  params.measurement_acceptance_rates.push_back(acc_rate);
  params.measurement_times.push_back(time);

  if (params.write_to_file) {
    appendLatestGaugeObservables(params);
    clearAllGaugeObservables(params);
  }
}

inline bool fileNeedsHeader(const std::string &filename) {
  if (filename.empty()) {
    return false;
  }

  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::exists(filename, ec)) {
    return true;
  }
  return fs::file_size(filename, ec) == 0;
}

inline void flushPlaquette(std::ofstream &file,
                           const GaugeObservableParams &params,
                           const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  if (!params.measure_plaquette) {
    printf("Error: no plaquette measurements available\n");
    return;
  }

  if (params.measurement_steps.size() != params.plaquette_measurements.size() ||
      params.measurement_steps.size() !=
          params.measurement_acceptance_rates.size() ||
      params.measurement_steps.size() != params.measurement_times.size()) {
    printf("Error: inconsistent plaquette metadata sizes\n");
    return;
  }

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
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  if (!params.measure_wilson_loop_temporal) {
    printf("Error: no temporal Wilson loop measurements available\n");
    return;
  }

  if (HEADER) {
    file << "# step, L, T, W_temp\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << "\n";
    }
  }
}

inline void flushWilsonLoopMuNu(std::ofstream &file,
                                const GaugeObservableParams &params,
                                const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  if (!params.measure_wilson_loop_mu_nu) {
    printf("Error: no mu-nu Wilson loop measurements available\n");
    return;
  }

  if (HEADER) {
    file << "# step, mu, nu, Lmu, Lnu, W_mu_nu\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_mu_nu_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << ", " << measurement[3]
           << ", " << measurement[4] << "\n";
    }
  }
}

inline void flushRetraceU(std::ofstream &file,
                          const GaugeObservableParams &params,
                          const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  if (!params.measure_retrace_U) {
    printf("Error: no Retrace(U) measurements available\n");
    return;
  }

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
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  if (!params.measure_nested_wilson_action) {
    printf("Error: no nested Wilson action measurements available\n");
    return;
  }

  const size_t n = params.measurement_steps.size();
  if (n != params.nested_plaq_V_measurements.size() ||
      n != params.nested_plaq_child_measurements.size() ||
      n != params.nested_E_V_measurements.size() ||
      n != params.nested_E_child_measurements.size()) {
    printf("Error: inconsistent nested Wilson action metadata sizes\n");
    return;
  }

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

inline void appendLatestGaugeObservables(const GaugeObservableParams &params) {
  if (!params.write_to_file) {
    return;
  }

  if (params.plaquette_filename != "") {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    flushPlaquette(file, params, fileNeedsHeader(params.plaquette_filename));
    file.flush();
    file.close();
  }

  if (params.W_temp_filename != "") {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    flushWilsonLoopTemporal(file, params, fileNeedsHeader(params.W_temp_filename));
    file.flush();
    file.close();
  }

  if (params.W_mu_nu_filename != "") {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    flushWilsonLoopMuNu(file, params, fileNeedsHeader(params.W_mu_nu_filename));
    file.flush();
    file.close();
  }

  if (params.RetraceU_filename != "") {
    std::ofstream file(params.RetraceU_filename, std::ios::app);
    flushRetraceU(file, params, fileNeedsHeader(params.RetraceU_filename));
    file.flush();
    file.close();
  }

  if (params.nested_wilson_action_filename != "") {
    std::ofstream file(params.nested_wilson_action_filename, std::ios::app);
    flushNestedWilsonAction(
        file, params, fileNeedsHeader(params.nested_wilson_action_filename));
    file.flush();
    file.close();
  }
}

} // namespace klft
