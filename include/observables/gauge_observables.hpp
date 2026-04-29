#pragma once
#include "observables/nested_wilson_action.hpp"
#include "observables/plaquette.hpp"
#include "observables/polyakov_correlator.hpp"
#include "observables/polyakov_loop.hpp"
#include "observables/retrace.hpp"
#include "observables/wilson_loop.hpp"
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
  bool measure_polyakov_loop;
  bool measure_polyakov_correlator;
  bool measure_retrace_U;
  index_t wilson_loop_multihit;
  index_t polyakov_loop_multihit;
  index_t polyakov_correlator_max_r;

  bool measure_nested_wilson_action;
  std::vector<index_t> nested_child_offset;

  std::vector<Kokkos::Array<index_t, 2>> W_temp_L_T_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_mu_nu_pairs;
  std::vector<Kokkos::Array<index_t, 2>> W_Lmu_Lnu_pairs;
  bool include_acceptance_rate;

  std::vector<size_t> measurement_steps;
  std::vector<real_t> measurement_acceptance_rates;
  std::vector<real_t> measurement_times;

  std::vector<real_t> plaquette_measurements;
  std::vector<std::vector<Kokkos::Array<real_t, 3>>> W_temp_measurements;
  std::vector<std::vector<Kokkos::Array<real_t, 5>>> W_mu_nu_measurements;
  std::vector<Kokkos::Array<real_t, 2>> polyakov_measurements;
  std::vector<std::vector<Kokkos::Array<real_t, 3>>>
      polyakov_correlator_measurements;
  std::vector<real_t> retraceU_measurements;

  std::vector<real_t> nested_plaq_V_measurements;
  std::vector<real_t> nested_plaq_child_measurements;
  std::vector<real_t> nested_E_V_measurements;
  std::vector<real_t> nested_E_child_measurements;

  std::string plaquette_filename;
  std::string W_temp_filename;
  std::string W_mu_nu_filename;
  std::string polyakov_loop_filename;
  std::string polyakov_correlator_filename;
  std::string RetraceU_filename;
  std::string nested_wilson_action_filename;

  bool write_to_file;

  GaugeObservableParams()
      : measurement_interval(0), measure_plaquette(false),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        measure_polyakov_loop(false), measure_polyakov_correlator(false),
        measure_retrace_U(false), wilson_loop_multihit(1),
        polyakov_loop_multihit(1), polyakov_correlator_max_r(0),
        measure_nested_wilson_action(false), include_acceptance_rate(false),
        write_to_file(false) {}
};

inline bool appendLatestGaugeObservables(const GaugeObservableParams &params);
inline void clearAllGaugeObservables(GaugeObservableParams &params);

// Measure the requested observables and stage the results for optional output.
template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void measureGaugeObservables(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const UpdateParams &updateParams, GaugeObservableParams &params,
    const size_t step, const real_t acc_rate, const real_t time,
    const RNG &rng) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0)) {
    return;
  }

  if (params.measure_plaquette) {
    const real_t P = GaugePlaquette<rank, Nc>(g_in);
    params.plaquette_measurements.push_back(P);
  }

  if (params.measure_wilson_loop_temporal) {
    std::vector<Kokkos::Array<real_t, 3>> temp_measurements;
    WilsonLoop_temporal<rank, Nc>(g_in, params.W_temp_L_T_pairs,
                                  temp_measurements,
                                  params.wilson_loop_multihit, updateParams,
                                  rng);

    params.W_temp_measurements.push_back(temp_measurements);
  }

  if (params.measure_wilson_loop_mu_nu) {
    std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
    for (const auto &pair_mu_nu : params.W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<rank, Nc>(g_in, mu, nu, params.W_Lmu_Lnu_pairs,
                                 temp_measurements, params.wilson_loop_multihit,
                                 updateParams, rng);
    }

    params.W_mu_nu_measurements.push_back(temp_measurements);
  }

  if (params.measure_polyakov_loop) {
    const auto P = PolyakovLoop<rank, Nc>(
        g_in, params.polyakov_loop_multihit, updateParams, rng);
    params.polyakov_measurements.push_back(P);
  }

  if (params.measure_polyakov_correlator) {
    std::vector<Kokkos::Array<real_t, 3>> corr_measurements;
    PolyakovCorrelator<rank, Nc>(g_in, params.polyakov_correlator_max_r,
                                 params.polyakov_loop_multihit,
                                 corr_measurements, updateParams, rng);
    params.polyakov_correlator_measurements.push_back(corr_measurements);
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
    for (index_t d = 0; d < rank; ++d) {
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
  if (params.include_acceptance_rate) {
    params.measurement_acceptance_rates.push_back(acc_rate);
  }
  params.measurement_times.push_back(time);

  if (params.write_to_file) {
    if (appendLatestGaugeObservables(params)) {
      clearAllGaugeObservables(params);
    }
  }
}

// Return whether the output file still needs a header row.
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

// Append the staged plaquette rows to disk.
inline bool flushPlaquette(std::ofstream &file,
                           const GaugeObservableParams &params,
                           const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_plaquette) {
    printf("Error: no plaquette measurements available\n");
    return false;
  }

  if (params.measurement_steps.size() != params.plaquette_measurements.size() ||
      params.measurement_steps.size() != params.measurement_times.size()) {
    printf("Error: inconsistent plaquette metadata sizes\n");
    return false;
  }
  if (params.include_acceptance_rate &&
      params.measurement_steps.size() != params.measurement_acceptance_rates.size()) {
    printf("Error: inconsistent plaquette acceptance metadata sizes\n");
    return false;
  }

  if (HEADER) {
    if (params.include_acceptance_rate) {
      file << "# step plaquette acceptance_rate time\n";
    } else {
      file << "# step plaquette time\n";
    }
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << " "
         << params.plaquette_measurements[i];
    if (params.include_acceptance_rate) {
      file << " " << params.measurement_acceptance_rates[i];
    }
    file << " " << params.measurement_times[i] << "\n";
  }
  return static_cast<bool>(file);
}

// Append the staged temporal Wilson-loop rows to disk.
inline bool flushWilsonLoopTemporal(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_wilson_loop_temporal) {
    printf("Error: no temporal Wilson loop measurements available\n");
    return false;
  }
  if (params.measurement_steps.size() != params.W_temp_measurements.size()) {
    printf("Error: inconsistent temporal Wilson-loop metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step L T W_temp\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << " " << measurement[0] << " "
           << measurement[1] << " " << measurement[2] << "\n";
    }
  }
  return static_cast<bool>(file);
}

// Append the staged planar Wilson-loop rows to disk.
inline bool flushWilsonLoopMuNu(std::ofstream &file,
                                const GaugeObservableParams &params,
                                const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_wilson_loop_mu_nu) {
    printf("Error: no mu-nu Wilson loop measurements available\n");
    return false;
  }
  if (params.measurement_steps.size() != params.W_mu_nu_measurements.size()) {
    printf("Error: inconsistent mu-nu Wilson-loop metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step mu nu Lmu Lnu W_mu_nu\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_mu_nu_measurements[i]) {
      file << params.measurement_steps[i] << " " << measurement[0] << " "
           << measurement[1] << " " << measurement[2] << " " << measurement[3]
           << " " << measurement[4] << "\n";
    }
  }
  return static_cast<bool>(file);
}

// Append the staged Polyakov-loop rows to disk.
inline bool flushPolyakovLoop(std::ofstream &file,
                              const GaugeObservableParams &params,
                              const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_polyakov_loop) {
    printf("Error: no Polyakov-loop measurements available\n");
    return false;
  }

  if (params.measurement_steps.size() != params.polyakov_measurements.size()) {
    printf("Error: inconsistent Polyakov-loop metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step RePolyakov ImPolyakov\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << " "
         << params.polyakov_measurements[i][0] << " "
         << params.polyakov_measurements[i][1] << "\n";
  }
  return static_cast<bool>(file);
}

// Append the staged Polyakov-correlator rows to disk.
inline bool flushPolyakovCorrelator(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_polyakov_correlator) {
    printf("Error: no Polyakov-correlator measurements available\n");
    return false;
  }

  if (params.measurement_steps.size() !=
      params.polyakov_correlator_measurements.size()) {
    printf("Error: inconsistent Polyakov-correlator metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step R real imaginary\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.polyakov_correlator_measurements[i]) {
      file << params.measurement_steps[i] << " " << measurement[0] << " "
           << measurement[1] << " " << measurement[2] << "\n";
    }
  }
  return static_cast<bool>(file);
}

// Append the staged Retrace(U) rows to disk.
inline bool flushRetraceU(std::ofstream &file,
                          const GaugeObservableParams &params,
                          const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_retrace_U) {
    printf("Error: no Retrace(U) measurements available\n");
    return false;
  }
  if (params.measurement_steps.size() != params.retraceU_measurements.size()) {
    printf("Error: inconsistent Retrace(U) metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step RetraceU\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << " "
         << params.retraceU_measurements[i] << "\n";
  }
  return static_cast<bool>(file);
}

// Append the staged nested-action rows to disk.
inline bool flushNestedWilsonAction(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return false;
  }
  if (!params.measure_nested_wilson_action) {
    printf("Error: no nested Wilson action measurements available\n");
    return false;
  }

  const size_t n = params.measurement_steps.size();
  if (n != params.nested_plaq_V_measurements.size() ||
      n != params.nested_plaq_child_measurements.size() ||
      n != params.nested_E_V_measurements.size() ||
      n != params.nested_E_child_measurements.size()) {
    printf("Error: inconsistent nested Wilson action metadata sizes\n");
    return false;
  }

  if (HEADER) {
    file << "# step plaq_V plaq_child E_V E_child\n";
  }

  file << std::setprecision(12);

  for (size_t i = 0; i < n; ++i) {
    file << params.measurement_steps[i] << " "
         << params.nested_plaq_V_measurements[i] << " "
         << params.nested_plaq_child_measurements[i] << " "
         << params.nested_E_V_measurements[i] << " "
         << params.nested_E_child_measurements[i] << "\n";
  }
  return static_cast<bool>(file);
}

// Drop the current staging buffers after a successful append.
inline void clearAllGaugeObservables(GaugeObservableParams &params) {
  params.measurement_steps.clear();
  params.measurement_acceptance_rates.clear();
  params.measurement_times.clear();

  params.plaquette_measurements.clear();
  params.W_temp_measurements.clear();
  params.W_mu_nu_measurements.clear();
  params.polyakov_measurements.clear();
  params.polyakov_correlator_measurements.clear();
  params.retraceU_measurements.clear();

  params.nested_plaq_V_measurements.clear();
  params.nested_plaq_child_measurements.clear();
  params.nested_E_V_measurements.clear();
  params.nested_E_child_measurements.clear();
}

inline bool canOpenObservableOutputFile(const std::string &filename) {
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open observable file '%s'\n", filename.c_str());
    return false;
  }
  return true;
}

inline bool closeObservableOutputFile(std::ofstream &file,
                                      const std::string &filename) {
  file.flush();
  const bool ok = static_cast<bool>(file);
  file.close();
  if (!ok) {
    printf("Error: failed while writing observable file '%s'\n",
           filename.c_str());
  }
  return ok;
}

// Append one measurement batch to each enabled output file.
inline bool appendLatestGaugeObservables(const GaugeObservableParams &params) {
  if (!params.write_to_file) {
    return true;
  }

  bool can_open_all = true;
  if (params.measure_plaquette && params.plaquette_filename != "") {
    can_open_all &= canOpenObservableOutputFile(params.plaquette_filename);
  }
  if (params.measure_wilson_loop_temporal && params.W_temp_filename != "") {
    can_open_all &= canOpenObservableOutputFile(params.W_temp_filename);
  }
  if (params.measure_wilson_loop_mu_nu && params.W_mu_nu_filename != "") {
    can_open_all &= canOpenObservableOutputFile(params.W_mu_nu_filename);
  }
  if (params.measure_polyakov_loop && params.polyakov_loop_filename != "") {
    can_open_all &= canOpenObservableOutputFile(params.polyakov_loop_filename);
  }
  if (params.measure_polyakov_correlator &&
      params.polyakov_correlator_filename != "") {
    can_open_all &=
        canOpenObservableOutputFile(params.polyakov_correlator_filename);
  }
  if (params.measure_retrace_U && params.RetraceU_filename != "") {
    can_open_all &= canOpenObservableOutputFile(params.RetraceU_filename);
  }
  if (params.measure_nested_wilson_action &&
      params.nested_wilson_action_filename != "") {
    can_open_all &=
        canOpenObservableOutputFile(params.nested_wilson_action_filename);
  }
  if (!can_open_all) {
    return false;
  }

  bool ok = true;
  if (params.measure_plaquette && params.plaquette_filename != "") {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    ok &= flushPlaquette(file, params, fileNeedsHeader(params.plaquette_filename));
    ok &= closeObservableOutputFile(file, params.plaquette_filename);
  }

  if (params.measure_wilson_loop_temporal && params.W_temp_filename != "") {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    ok &= flushWilsonLoopTemporal(file, params,
                                  fileNeedsHeader(params.W_temp_filename));
    ok &= closeObservableOutputFile(file, params.W_temp_filename);
  }

  if (params.measure_wilson_loop_mu_nu && params.W_mu_nu_filename != "") {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    ok &=
        flushWilsonLoopMuNu(file, params, fileNeedsHeader(params.W_mu_nu_filename));
    ok &= closeObservableOutputFile(file, params.W_mu_nu_filename);
  }

  if (params.measure_polyakov_loop && params.polyakov_loop_filename != "") {
    std::ofstream file(params.polyakov_loop_filename, std::ios::app);
    ok &= flushPolyakovLoop(file, params,
                            fileNeedsHeader(params.polyakov_loop_filename));
    ok &= closeObservableOutputFile(file, params.polyakov_loop_filename);
  }

  if (params.measure_polyakov_correlator &&
      params.polyakov_correlator_filename != "") {
    std::ofstream file(params.polyakov_correlator_filename, std::ios::app);
    ok &= flushPolyakovCorrelator(
        file, params, fileNeedsHeader(params.polyakov_correlator_filename));
    ok &= closeObservableOutputFile(file, params.polyakov_correlator_filename);
  }

  if (params.measure_retrace_U && params.RetraceU_filename != "") {
    std::ofstream file(params.RetraceU_filename, std::ios::app);
    ok &= flushRetraceU(file, params, fileNeedsHeader(params.RetraceU_filename));
    ok &= closeObservableOutputFile(file, params.RetraceU_filename);
  }

  if (params.measure_nested_wilson_action &&
      params.nested_wilson_action_filename != "") {
    std::ofstream file(params.nested_wilson_action_filename, std::ios::app);
    ok &= flushNestedWilsonAction(
        file, params, fileNeedsHeader(params.nested_wilson_action_filename));
    ok &= closeObservableOutputFile(file, params.nested_wilson_action_filename);
  }
  return ok;
}

} // namespace klft
