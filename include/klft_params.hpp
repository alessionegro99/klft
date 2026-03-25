#pragma once
#include "GLOBAL.hpp"

namespace klft {

struct MetropolisParams {

  // geometry
  // L{Ndims-1} is time
  index_t Ndims;
  index_t L0;
  index_t L1;
  index_t L2;
  index_t L3;

  // update
  index_t nHits;
  index_t nSweep;
  index_t seed;

  // gauge field
  size_t Nd; // number of mu degrees of freedom
  size_t Nc; // number of color degrees of freedom

  // action
  real_t beta;     // inverse coupling constant
  real_t delta;    // step size for the metropolis update
  real_t epsilon1; // gauge breaking parameter, if applicable
  // real_t epsilon2;

  // NEMC
  bool enable_nemc = false;
  index_t nemc_stride = 0;
  index_t nemc_nsteps = 0;
  real_t nemc_dbeta = 0.0;
  index_t nemc_ntherm = 0;
  std::string nemc_filename;

  // add more parameters above this line as needed
  // ...
  MetropolisParams() {
    Ndims = 4;
    L0 = 4;
    L1 = 4;
    L2 = 4;
    L3 = 4;

    nHits = 2;
    nSweep = 1000;
    seed = 1234;

    Nd = 4;
    Nc = 2;

    beta = 1.0;
    delta = 0.1;
    epsilon1 = 0.0;
    // epsilon2 = 0.0;

    enable_nemc = false;
    nemc_ntherm = 0;
    nemc_stride = 5;
    nemc_nsteps = 10;
    nemc_dbeta = 0.01;
    nemc_filename = "nemc_results.out";
  }

  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Metropolis Parameters:\n");
      printf("General Parameters:\n");
      printf("Ndims: %lld\n", static_cast<long long>(Ndims));
      printf("L0: %lld\n", static_cast<long long>(L0));
      printf("L1: %lld\n", static_cast<long long>(L1));
      printf("L2: %lld\n", static_cast<long long>(L2));
      printf("L3: %lld\n", static_cast<long long>(L3));
      printf("nHits: %lld\n", static_cast<long long>(nHits));
      printf("nSweep: %lld\n", static_cast<long long>(nSweep));
      printf("seed: %lld\n", static_cast<long long>(seed));

      printf("GaugeField Parameters:\n");
      printf("Nd: %zu\n", Nd);
      printf("Nc: %zu\n", Nc);

      printf("Wilson Action Parameters:\n");
      printf("beta: %.8f\n", beta);
      printf("delta: %.8f\n", delta);
      printf("epsilon1: %.8f\n", epsilon1);
      // printf("epsilon2: %.8f\n", epsilon2);

      printf("NEMC Parameters:\n");
      printf("enable_nemc: %s\n", enable_nemc ? "true" : "false");
      printf("nemc_stride: %lld\n", static_cast<long long>(nemc_stride));
      printf("nemc_nsteps: %lld\n", static_cast<long long>(nemc_nsteps));
      printf("nemc_dbeta: %.8f\n", nemc_dbeta);
      printf("nemc_ntherm: %lld\n", static_cast<long long>(nemc_ntherm));
      printf("nemc_filename: %s\n", nemc_filename.c_str());
    }
  }
};

} // namespace klft
