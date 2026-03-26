#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "gauge_obs_meas.hpp"
#include "klft_params.hpp"
#include <Kokkos_Random.hpp>
#include <vector>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <size_t rank, size_t Nc> struct NEMCBranchResult {
  size_t spawn_step = 0;
  std::vector<real_t> beta_schedule;
  std::vector<real_t> acceptance_rates;
  std::vector<real_t> plaquettes;

  std::vector<real_t> works;
  real_t work = 0.0; // exp(-work);
};

template <size_t rank, size_t Nc>
inline void
nemc_write_to_file(const std::string &filename,
                   const std::vector<NEMCBranchResult<rank, Nc>> &results,
                   const bool header = true) {
  if (results.empty()) {
    return;
  }

  const size_t nbranches = results.size();
  const size_t nsteps = results[0].beta_schedule.size();

  for (size_t b = 0; b < nbranches; ++b) {
    if (results[b].beta_schedule.size() != nsteps ||
        results[b].plaquettes.size() != nsteps ||
        results[b].works.size() != nsteps) {
      throw std::runtime_error("Inconsistent NEMC branch sizes.");
    }
  }

  if (results[0].acceptance_rates.size() != nsteps) {
    throw std::runtime_error(
        "First NEMC branch has inconsistent acceptance-rate size.");
  }

  for (size_t b = 1; b < nbranches; ++b) {
    for (size_t k = 0; k < nsteps; ++k) {
      if (results[b].beta_schedule[k] != results[0].beta_schedule[k]) {
        throw std::runtime_error("Wide NEMC output requires identical beta "
                                 "schedules across branches.");
      }
    }
  }

  std::ofstream file(filename, std::ios::out);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open NEMC output file.");
  }

  file << std::setprecision(12);

  if (header) {
    file << "# beta, acc_" << results[0].spawn_step;
    for (size_t b = 0; b < nbranches; ++b) {
      file << ", plaq_" << results[b].spawn_step << ", work_"
           << results[b].spawn_step;
    }
    file << "\n";
  }

  for (size_t k = 0; k < nsteps; ++k) {
    file << results[0].beta_schedule[k] << ", "
         << results[0].acceptance_rates[k];

    for (size_t b = 0; b < nbranches; ++b) {
      file << ", " << results[b].plaquettes[k] << ", " << results[b].works[k];
    }
    file << "\n";
  }
}

// deep-copy helpers for spawned nonequilibrium branches
template <size_t Nd, size_t Nc>
deviceGaugeField<Nd, Nc>
clone_gauge_field(const deviceGaugeField<Nd, Nc> &src) {
  deviceGaugeField<Nd, Nc> dst(src.dimensions[0], src.dimensions[1],
                               src.dimensions[2], src.dimensions[3],
                               identitySUN<Nc>());
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::fence();
  return dst;
}

template <size_t Nd, size_t Nc>
deviceGaugeField3D<Nd, Nc>
clone_gauge_field(const deviceGaugeField3D<Nd, Nc> &src) {
  deviceGaugeField3D<Nd, Nc> dst(src.dimensions[0], src.dimensions[1],
                                 src.dimensions[2], identitySUN<Nc>());
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::fence();
  return dst;
}

template <size_t Nd, size_t Nc>
deviceGaugeField2D<Nd, Nc>
clone_gauge_field(const deviceGaugeField2D<Nd, Nc> &src) {
  deviceGaugeField2D<Nd, Nc> dst(src.dimensions[0], src.dimensions[1],
                                 identitySUN<Nc>());
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::fence();
  return dst;
}

template <size_t rank, size_t Nc, class RNG> struct MetropolisUpdateFunctor {
  constexpr static const size_t Nd = rank;

  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType g_in;

  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType acc;

  const RNG rng;
  const MetropolisParams params;
  const IndexArray<rank> dimensions;
  const Kokkos::Array<bool, rank> oddeven;

  MetropolisUpdateFunctor(const GaugeFieldType &g_in,
                          const MetropolisParams &params,
                          const IndexArray<rank> &dimensions,
                          const ScalarFieldType &acc,
                          const Kokkos::Array<bool, rank> &oddeven,
                          const RNG &rng)
      : g_in(g_in), params(params), oddeven(oddeven), rng(rng),
        dimensions(dimensions), acc(acc) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    index_t acc_per_site = 0;

    auto generator = rng.get_state();
    SUN<Nc> r;

    const IndexArray<rank> site = index_odd_even<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, oddeven);

    for (index_t mu = 0; mu < Nd; ++mu) {
      const SUN<Nc> staple = g_in.staple(site, mu);

      for (index_t hit = 0; hit < params.nHits; ++hit) {
        randSUN(r, generator, params.delta);

        const SUN<Nc> U_old = g_in(site, mu);
        const SUN<Nc> U_new = U_old * r;

        real_t dS =
            -(params.beta / static_cast<real_t>(Nc)) *
            (trace(U_new * staple).real() - trace(U_old * staple).real());

        if (params.epsilon1 != 0.0) {
          dS += -params.epsilon1 * static_cast<real_t>(0.5) *
                (trace(U_new).real() - trace(U_old).real());
        }

        // if (params.epsilon2 != 0.0) {
        //   const real_t retr_U_new = trace(U_new).real();
        //   const real_t retr_U_old = trace(U_old).real();
        //   dS += -params.epsilon2 *
        //         (retr_U_new * retr_U_new - retr_U_old * retr_U_old);
        // }

        bool accept = dS < 0.0;

        if (!accept) {
          accept = (generator.drand(0.0, 1.0) < Kokkos::exp(-dS));
        }

        if (accept) {
          g_in(site, mu) = restoreSUN(U_new);
          acc_per_site++;
        }
      }
    }

    acc(Idcs...) += static_cast<real_t>(acc_per_site);
    rng.free_state(generator);
  }
};

// returns acceptance rate
template <size_t rank, size_t Nc, class RNG>
real_t sweep_metropolis(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                        const MetropolisParams &params, const RNG &rng) {
  constexpr static const size_t Nd = rank;

  const auto &dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;

  for (index_t i = 0; i < Nd; ++i) {
    if ((dimensions[i] % 2) != 0) {
      throw std::runtime_error(
          "sweep_metropolis requires even lattice extents in every direction.");
    }
    start[i] = 0;
    end[i] = dimensions[i] / 2;
  }

  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType acc(end, 0.0);

  for (index_t i = 0; i < static_cast<index_t>(1u << rank); ++i) {
    MetropolisUpdateFunctor<rank, Nc, RNG> metropolis(
        g_in, params, end, acc, oddeven_array<rank>(i), rng);

    Kokkos::parallel_for("sweep_metropolis_GaugeField_sublat_" +
                             std::to_string(i),
                         Policy<rank>(start, end), metropolis);

    Kokkos::fence();
  }

  real_t acc_sweep = acc.sum();
  Kokkos::fence();

  real_t norm = 1.0;
  for (index_t i = 0; i < rank; ++i) {
    norm *= static_cast<real_t>(dimensions[i]);
  }
  norm *= static_cast<real_t>(Nd);
  norm *= static_cast<real_t>(params.nHits);
  acc_sweep /= norm;

  return acc_sweep;
}

template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_metropolis(GaugeFieldType &g_in,
                   const MetropolisParams &metropolisParams,
                   GaugeObservableParams &gaugeObsParams, const RNG &rng) {
  constexpr const size_t Nd = rank;
  const auto &dimensions = g_in.dimensions;

  assert(metropolisParams.Ndims == Nd);
  assert(metropolisParams.Nd == Nd);
  assert(metropolisParams.Nc == Nc);
  assert(metropolisParams.L0 == dimensions[0]);
  assert(metropolisParams.L1 == dimensions[1]);
  if constexpr (Nd > 2) {
    assert(metropolisParams.L2 == dimensions[2]);
  }
  if constexpr (Nd > 3) {
    assert(metropolisParams.L3 == dimensions[3]);
  }

  Kokkos::Timer timer;
  for (size_t step = 0; step < static_cast<size_t>(metropolisParams.nSweep);
       ++step) {
    timer.reset();

    const real_t acc_rate =
        sweep_metropolis<rank, Nc>(g_in, metropolisParams, rng);

    const real_t time = timer.seconds();

    const size_t main_sweep = step + 1;
    measureGaugeObservables<rank, Nc>(g_in, gaugeObsParams, main_sweep,
                                      acc_rate, time);
  }

  flushAllGaugeObservables(gaugeObsParams);
  return 0;
}

template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
NEMCBranchResult<rank, Nc> run_nemc_branch(const GaugeFieldType &u_start,
                                           const MetropolisParams &base_params,
                                           const size_t spawn_step) {
  NEMCBranchResult<rank, Nc> out;
  out.spawn_step = spawn_step;

  GaugeFieldType branch_field = clone_gauge_field(u_start);
  MetropolisParams branch_params = base_params;

  RNG branch_rng(static_cast<uint64_t>(base_params.seed) +
                 static_cast<uint64_t>(104729) *
                     static_cast<uint64_t>(spawn_step + 1));

  real_t beta_old = base_params.beta;

  for (index_t k = 1; k <= branch_params.nemc_nsteps; ++k) {
    const real_t beta_new =
        base_params.beta + static_cast<real_t>(k) * branch_params.nemc_dbeta;

    // work increment ΔW_k = S_{beta_new}(U_k) - S_{beta_old}(U_k)
    // here U_k is the configuration BEFORE the sweep at beta_new
    const real_t plaq_before = GaugePlaquette<rank, Nc>(branch_field, true);
    const real_t action_before = ReducedWilsonActionFromAvgPlaquette<rank>(
        plaq_before, branch_field.dimensions);
    const real_t dW = (beta_new - beta_old) * action_before;

    branch_params.beta = beta_new;

    const real_t acc_rate =
        sweep_metropolis<rank, Nc>(branch_field, branch_params, branch_rng);

    const real_t plaq_after = GaugePlaquette<rank, Nc>(branch_field, true);

    out.beta_schedule.push_back(beta_new);
    out.acceptance_rates.push_back(acc_rate);
    out.plaquettes.push_back(plaq_after);
    out.works.push_back(dW);

    out.work += dW;
    beta_old = beta_new;
  }

  return out;
}

template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_metropolis_nemc(GaugeFieldType &g_in,
                        const MetropolisParams &metropolisParams,
                        GaugeObservableParams &gaugeObsParams,
                        std::vector<NEMCBranchResult<rank, Nc>> &nemc_results,
                        const RNG &rng) {
  constexpr const size_t Nd = rank;
  const auto &dimensions = g_in.dimensions;

  assert(metropolisParams.Ndims == Nd);
  assert(metropolisParams.Nd == Nd);
  assert(metropolisParams.Nc == Nc);
  assert(metropolisParams.L0 == dimensions[0]);
  assert(metropolisParams.L1 == dimensions[1]);
  if constexpr (Nd > 2) {
    assert(metropolisParams.L2 == dimensions[2]);
  }
  if constexpr (Nd > 3) {
    assert(metropolisParams.L3 == dimensions[3]);
  }

  Kokkos::Timer timer;
  for (size_t step = 0; step < static_cast<size_t>(metropolisParams.nSweep);
       ++step) {
    timer.reset();

    const real_t acc_rate =
        sweep_metropolis<rank, Nc>(g_in, metropolisParams, rng);

    const real_t time = timer.seconds();

    const size_t main_sweep = step + 1;

    measureGaugeObservables<rank, Nc>(g_in, gaugeObsParams, main_sweep,
                                      acc_rate, time);

    if (metropolisParams.enable_nemc && metropolisParams.nemc_stride > 0 &&
        main_sweep > static_cast<size_t>(metropolisParams.nemc_ntherm) &&
        ((main_sweep - static_cast<size_t>(metropolisParams.nemc_ntherm)) %
             static_cast<size_t>(metropolisParams.nemc_stride) ==
         0)) {
      nemc_results.push_back(
          run_nemc_branch<rank, Nc, RNG>(g_in, metropolisParams, main_sweep));
    }
  }

  flushAllGaugeObservables(gaugeObsParams);
  nemc_write_to_file<rank, Nc>(metropolisParams.nemc_filename, nemc_results);
  return 0;
}

// define for all dimensionalities and gauge groups
// 2D U(1)
template int run_metropolis<2, 1>(deviceGaugeField2D<2, 1> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 2D SU(2)
template int run_metropolis<2, 2>(deviceGaugeField2D<2, 2> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 2D SU(3)
template int run_metropolis<2, 3>(deviceGaugeField2D<2, 3> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 3D U(1)
template int run_metropolis<3, 1>(deviceGaugeField3D<3, 1> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 3D SU(2)
template int run_metropolis<3, 2>(deviceGaugeField3D<3, 2> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 3D SU(3)
template int run_metropolis<3, 3>(deviceGaugeField3D<3, 3> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 4D U(1)
template int run_metropolis<4, 1>(deviceGaugeField<4, 1> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 4D SU(2)
template int run_metropolis<4, 2>(deviceGaugeField<4, 2> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);
// 4D SU(3)
template int run_metropolis<4, 3>(deviceGaugeField<4, 3> &g_in,
                                  const MetropolisParams &metropolisParams,
                                  GaugeObservableParams &gaugeObsParams,
                                  const RNGType &rng);

// NEMC variants
template int run_metropolis_nemc<2, 1>(
    deviceGaugeField2D<2, 1> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<2, 1>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<2, 2>(
    deviceGaugeField2D<2, 2> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<2, 2>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<2, 3>(
    deviceGaugeField2D<2, 3> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<2, 3>> &nemc_results, const RNGType &rng);

template int run_metropolis_nemc<3, 1>(
    deviceGaugeField3D<3, 1> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<3, 1>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<3, 2>(
    deviceGaugeField3D<3, 2> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<3, 2>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<3, 3>(
    deviceGaugeField3D<3, 3> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<3, 3>> &nemc_results, const RNGType &rng);

template int run_metropolis_nemc<4, 1>(
    deviceGaugeField<4, 1> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<4, 1>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<4, 2>(
    deviceGaugeField<4, 2> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<4, 2>> &nemc_results, const RNGType &rng);
template int run_metropolis_nemc<4, 3>(
    deviceGaugeField<4, 3> &g_in, const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    std::vector<NEMCBranchResult<4, 3>> &nemc_results, const RNGType &rng);

} // namespace klft
