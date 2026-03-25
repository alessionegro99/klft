#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "Metropolis_Params.hpp"
#include "gauge_obs_meas.hpp"
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <size_t rank, size_t Nc, class RNG> struct MetropolisGaugeField {
  constexpr static const size_t Nd = rank;

  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType g_in;
  // define the scalar field type
  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted;
  // define the rng
  const RNG rng;
  // define the parameters
  const MetropolisParams params;
  // lattice dimensions
  const IndexArray<rank> dimensions;
  // sublattice definitions
  const Kokkos::Array<bool, rank> oddeven;
  // constructor
  MetropolisGaugeField(const GaugeFieldType &g_in,
                       const MetropolisParams &params,
                       const IndexArray<rank> &dimensions,
                       const ScalarFieldType &nAccepted,
                       const Kokkos::Array<bool, rank> &oddeven, const RNG &rng)
      : g_in(g_in), params(params), oddeven(oddeven), rng(rng),
        dimensions(dimensions), nAccepted(nAccepted) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    index_t nAcc_per_site = 0;
    // get the rng state
    auto generator = rng.get_state();
    SUN<Nc> r;
    // get the site based on odd/even
    const IndexArray<rank> site = index_odd_even<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, oddeven);
    // iterate over mu (sequentially)
    for (index_t mu = 0; mu < Nd; ++mu) {
      // get the staple
      const SUN<Nc> staple = g_in.staple(site, mu);
      // do number of hits
      for (index_t hit = 0; hit < params.nHits; ++hit) {
        // generate a random SUN matrix
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

        bool accept = dS < 0.0;

        if (!accept) {
          accept = (generator.drand(0.0, 1.0) < Kokkos::exp(-dS));
        }

        if (accept) {
          g_in(site, mu) = restoreSUN(U_new);
          nAcc_per_site++;
        }
      }
    }

    nAccepted(Idcs...) += static_cast<real_t>(nAcc_per_site);

    rng.free_state(generator);
  }
};

// returns acceptance rate
template <size_t rank, size_t Nc, class RNG>
real_t sweep_Metropolis(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                        const MetropolisParams &params, const RNG &rng) {
  constexpr static const size_t Nd = rank;

  const auto &dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < Nd; ++i) {
    if ((dimensions[i] % 2) != 0) {
      throw std::runtime_error(
          "sweep_Metropolis requires even lattice extents in every direction.");
    }
    start[i] = 0;
    end[i] = dimensions[i] / 2;
  }

  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted(end, 0.0);

  // into odd and even sites and launch the kernel for each sublattice
  // this results in 2^N sublattices
  for (index_t i = 0; i < static_cast<index_t>(1u << rank); ++i) {
    // define metropolis functor for this sublattice
    MetropolisGaugeField<rank, Nc, RNG> metropolis(g_in, params, end, nAccepted,
                                                   oddeven_array<rank>(i), rng);
    // launch the kernel
    tune_and_launch_for<rank>("sweep_Metropolis_GaugeField_sublat_" +
                                  std::to_string(i),
                              start, end, metropolis);
    Kokkos::fence();
  }

  real_t nAcc_total = nAccepted.sum();

  // TODO: remove(?)
  Kokkos::fence();

  real_t norm = 1.0;
  for (index_t i = 0; i < rank; ++i) {
    norm *= static_cast<real_t>(dimensions[i]);
  }
  norm *= static_cast<real_t>(Nd);
  norm *= static_cast<real_t>(params.nHits);
  nAcc_total /= norm;

  return nAcc_total;
}

template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_metropolis(GaugeFieldType &g_in,
                   const MetropolisParams &metropolisParams,
                   GaugeObservableParams &gaugeObsParams, const RNG &rng) {
  // this algorithm is strictly for Nd = rank
  constexpr const size_t Nd = rank;
  // get the dimensions
  const auto &dimensions = g_in.dimensions;
  // first we check that all the parameters are correct
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
  // timer to measure the time taken per sweep
  Kokkos::Timer timer;
  // metropolis loop
  for (size_t step = 0; step < metropolisParams.nSweep; ++step) {
    timer.reset();
    // sweep
    const real_t acc_rate =
        sweep_Metropolis<rank, Nc>(g_in, metropolisParams, rng);
    // get the time taken for the sweep
    const real_t time = timer.seconds();

    measureGaugeObservables<rank, Nc>(g_in, gaugeObsParams, step, acc_rate,
                                      time);
  }
  // flush the measurements to the files
  flushAllGaugeObservables(gaugeObsParams);

  return 0;
}

// definefor all dimensionalities and gauge groups
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

} // namespace klft
