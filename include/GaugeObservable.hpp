//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/

// this file defines the necessary functions to calculate
// different gauge observables to be measured during simulations

#pragma once
#include "GaugePlaquette.hpp"
#include "GaugeRetrace.hpp"
#include "WilsonLoop.hpp"
#include <fstream>
#include <iomanip>

namespace klft {

struct GaugeObservableParams {
  size_t measurement_interval;
  bool measure_plaquette;
  bool measure_wilson_loop_temporal;
  bool measure_wilson_loop_mu_nu;
  bool measure_retrace_U;

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

  std::string plaquette_filename;
  std::string W_temp_filename;
  std::string W_mu_nu_filename;
  std::string RetraceU_filename;

  bool write_to_file;

  GaugeObservableParams()
      : measurement_interval(0), measure_plaquette(false),
        measure_wilson_loop_temporal(false), measure_wilson_loop_mu_nu(false),
        measure_retrace_U(false), write_to_file(false) {}
};

template <size_t rank, size_t Nc>
void measureGaugeObservables(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    GaugeObservableParams &params, const size_t step, const real_t acc_rate,
    const real_t time) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }

  if (KLFT_VERBOSITY > 0) {
    printf("Measurement of Gauge Observables\n");
    printf("step: %zu\n", step);
  }

  if (params.measure_plaquette) {
    real_t P = GaugePlaquette<rank, Nc>(g_in);
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
                                  temp_measurements);
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
                                 temp_measurements);
      if (KLFT_VERBOSITY > 0) {
        for (const auto &measure : temp_measurements) {
          printf("%d, %d, %d, %d, %.12f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]),
                 static_cast<index_t>(measure[2]),
                 static_cast<index_t>(measure[3]), measure[4]);
        }
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

  params.measurement_steps.push_back(step);
  params.measurement_acceptance_rates.push_back(acc_rate);
  params.measurement_times.push_back(time);
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

  if (HEADER)
    file << "# step, plaquette, acceptance_rate, time\n";

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.plaquette_measurements[i] << ", "
         << params.measurement_acceptance_rates[i] << ", "
         << params.measurement_times[i] << "\n";
  }
}

// flush the temporal Wilson loop measurements to disk
inline void flushWilsonLoopTemporal(std::ofstream &file,
                                    const GaugeObservableParams &params,
                                    const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if temporal Wilson loop measurements are available
  if (!params.measure_wilson_loop_temporal) {
    printf("Error: no temporal Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, L, T, W_temp\n";

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto &measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << "\n";
    }
  }
}

// flush the mu-nu Wilson loop measurements to disk
inline void flushWilsonLoopMuNu(std::ofstream &file,
                                const GaugeObservableParams &params,
                                const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if mu-nu Wilson loop measurements are available
  if (!params.measure_wilson_loop_mu_nu) {
    printf("Error: no mu-nu Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, mu, nu, Lmu, Lnu, W_mu_nu\n";

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
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if Retrace(U) measurements are available
  if (!params.measure_retrace_U) {
    printf("Error: no Retrace(U) measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, Retrace(U)\n";

  file << std::setprecision(12);

  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.retraceU_measurements[i] << "\n";
  }
}

// define a global function to flush all measurements
inline void flushAllGaugeObservables(const GaugeObservableParams &params,
                                     const bool HEADER = true) {
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    printf("write_to_file is not enabled\n");
    return;
  }
  // flush plaquette measurements
  if (params.plaquette_filename != "") {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    flushPlaquette(file, params, HEADER);
    file.close();
  }
  // flush temporal Wilson loop measurements
  if (params.W_temp_filename != "") {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    flushWilsonLoopTemporal(file, params, HEADER);
    file.close();
  }
  // flush mu-nu Wilson loop measurements
  if (params.W_mu_nu_filename != "") {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    flushWilsonLoopMuNu(file, params, HEADER);
    file.close();
  }
  if (params.RetraceU_filename != "") {
    std::ofstream file(params.RetraceU_filename, std::ios::app);
    flushRetraceU(file, params, HEADER);
    file.close();
  }
  // ...
  // add more flush functions for other observables here
}

// function to clear all measurements
inline void clearAllGaugeObservables(GaugeObservableParams &params) {
  params.measurement_steps.clear();
  params.measurement_acceptance_rates.clear();
  params.measurement_times.clear();
  params.plaquette_measurements.clear();
  params.W_temp_measurements.clear();
  params.W_mu_nu_measurements.clear();
  params.retraceU_measurements.clear();
  // ...
  // add more clear functions for other observables here
}

} // namespace klft
