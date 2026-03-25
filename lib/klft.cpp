// this file defines the main function to run the metropolis
// for 2D, 3D and 4D SU(N) gauge fields

#include "InputParser.hpp"
#include "Tuner.hpp"
#include "gauge_conf_upd.hpp"

// we are hard coding the RNG now to use Kokkos::Random_XorShift64_Pool
// we might want to use our own RNG or allow the user to choose from
// different RNGs in the future
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int Metropolis(const std::string &input_file) {
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);

  if (tuning) {
    const char *cache_file = std::getenv("KLFT_CACHE_FILE");
    if (cache_file) {
      if (KLFT_VERBOSITY > 0) {
        printf("Reading cache file: %s\n", cache_file);
      }
      readTuneCache(cache_file);
    }
  }

  MetropolisParams metropolisParams;
  GaugeObservableParams gaugeObsParams;

  if (!parseInputFile(input_file, metropolisParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }

  metropolisParams.print();

  RNGType rng(metropolisParams.seed);

  if (metropolisParams.Ndims == 4) {
    if (metropolisParams.Nc == 1) {
      deviceGaugeField<4, 1> dev_g_U1_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<1>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<4, 1>> nemc_results;
        run_metropolis_nemc<4, 1>(dev_g_U1_4D, metropolisParams, gaugeObsParams,
                                  nemc_results, rng);
      } else {
        run_metropolis<4, 1>(dev_g_U1_4D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 2) {
      deviceGaugeField<4, 2> dev_g_SU2_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<2>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<4, 2>> nemc_results;
        run_metropolis_nemc<4, 2>(dev_g_SU2_4D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<4, 2>(dev_g_SU2_4D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 3) {
      deviceGaugeField<4, 3> dev_g_SU3_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<3>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<4, 3>> nemc_results;
        run_metropolis_nemc<4, 3>(dev_g_SU3_4D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<4, 3>(dev_g_SU3_4D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  } else if (metropolisParams.Ndims == 3) {
    if (metropolisParams.Nc == 1) {
      deviceGaugeField3D<3, 1> dev_g_U1_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<1>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<3, 1>> nemc_results;
        run_metropolis_nemc<3, 1>(dev_g_U1_3D, metropolisParams, gaugeObsParams,
                                  nemc_results, rng);
      } else {
        run_metropolis<3, 1>(dev_g_U1_3D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 2) {
      deviceGaugeField3D<3, 2> dev_g_SU2_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<2>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<3, 2>> nemc_results;
        run_metropolis_nemc<3, 2>(dev_g_SU2_3D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<3, 2>(dev_g_SU2_3D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 3) {
      deviceGaugeField3D<3, 3> dev_g_SU3_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<3>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<3, 3>> nemc_results;
        run_metropolis_nemc<3, 3>(dev_g_SU3_3D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<3, 3>(dev_g_SU3_3D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  } else if (metropolisParams.Ndims == 2) {
    if (metropolisParams.Nc == 1) {
      deviceGaugeField2D<2, 1> dev_g_U1_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<1>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<2, 1>> nemc_results;
        run_metropolis_nemc<2, 1>(dev_g_U1_2D, metropolisParams, gaugeObsParams,
                                  nemc_results, rng);
      } else {
        run_metropolis<2, 1>(dev_g_U1_2D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 2) {
      deviceGaugeField2D<2, 2> dev_g_SU2_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<2>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<2, 2>> nemc_results;
        run_metropolis_nemc<2, 2>(dev_g_SU2_2D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<2, 2>(dev_g_SU2_2D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else if (metropolisParams.Nc == 3) {
      deviceGaugeField2D<2, 3> dev_g_SU3_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<3>());

      if (metropolisParams.enable_nemc) {
        std::vector<NEMCBranchResult<2, 3>> nemc_results;
        run_metropolis_nemc<2, 3>(dev_g_SU3_2D, metropolisParams,
                                  gaugeObsParams, nemc_results, rng);
      } else {
        run_metropolis<2, 3>(dev_g_SU3_2D, metropolisParams, gaugeObsParams,
                             rng);
      }
    } else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  }

  if (KLFT_TUNING) {
    const char *cache_file = std::getenv("KLFT_CACHE_FILE");
    if (cache_file) {
      writeTuneCache(cache_file);
    } else {
      printf("KLFT_CACHE_FILE not set\n");
    }
  }

  return 0;
}

} // namespace klft
