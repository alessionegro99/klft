#pragma once
#include "core/common.hpp"

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
};

} // namespace klft
