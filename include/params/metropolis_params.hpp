#pragma once
#include "core/common.hpp"
#include "core/klft_config.hpp"

namespace klft {

struct MetropolisParams {
  index_t L0;
  index_t L1;
  index_t L2;
  index_t L3;
  index_t nHits;
  index_t nSweep;
  index_t seed;
  real_t beta;
  real_t delta;
  real_t epsilon1;
  real_t epsilon2;

  MetropolisParams() {
    L0 = 4;
    L1 = 4;
    L2 = 4;
    L3 = 4;
    nHits = 10;
    nSweep = 1000;
    seed = 1234;

    beta = 1.0;
    delta = 0.1;
    epsilon1 = 0.0;
    epsilon2 = 0.0;
  }

  // Print the runtime parameters for the compiled theory.
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Metropolis Parameters:\n");
      printf("General Parameters:\n");
      printf("Compiled dimension: %zu\n", compiled_rank);
      printf("L0: %d\n", L0);
      printf("L1: %d\n", L1);
      printf("L2: %d\n", L2);
      printf("L3: %d\n", L3);
      printf("nHits: %d\n", nHits);
      printf("nSweep: %d\n", nSweep);
      printf("seed: %d\n", seed);
      printf("GaugeField Parameters:\n");
      printf("Gauge group: %s\n", compiled_group_name());
      printf("Wilson Action Parameters:\n");
      printf("beta: %.4f\n", beta);
      printf("delta: %.4f\n", delta);
    }
  }
};

} // namespace klft
