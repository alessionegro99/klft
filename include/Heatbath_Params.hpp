#pragma once
#include "GLOBAL.hpp"
#include "KLFTConfig.hpp"

namespace klft {

struct HeatbathParams {
  index_t L0;
  index_t L1;
  index_t L2;
  index_t L3;
  index_t nSweep;
  index_t nOverrelax;
  index_t seed;

  real_t beta;
  real_t delta;
  real_t epsilon1;
  real_t epsilon2;

  HeatbathParams()
      : L0(4), L1(4), L2(4), L3(4), nSweep(1000), nOverrelax(5),
        seed(1234), beta(1.0), delta(0.1), epsilon1(0.0),
        epsilon2(0.0) {}

  // Print the runtime parameters for the compiled theory.
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Heatbath Parameters:\n");
      printf("General Parameters:\n");
      printf("Compiled dimension: %zu\n", compiled_rank);
      printf("L0: %d\n", L0);
      printf("L1: %d\n", L1);
      printf("L2: %d\n", L2);
      printf("L3: %d\n", L3);
      printf("nSweep: %d\n", nSweep);
      printf("nOverrelax: %d\n", nOverrelax);
      printf("seed: %d\n", seed);
      printf("GaugeField Parameters:\n");
      printf("Gauge group: %s\n", compiled_group_name());
      printf("Wilson Action Parameters:\n");
      printf("beta: %.4f\n", beta);
      printf("delta: %.4f\n", delta);
      printf("epsilon1: %.4f\n", epsilon1);
      printf("epsilon2: %.4f\n", epsilon2);
    }
  }
};

} // namespace klft
