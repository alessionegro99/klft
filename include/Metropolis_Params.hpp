#pragma once
#include "GLOBAL.hpp"

namespace klft {

struct MetropolisParams {

  // general parameters
  index_t Ndims; // number of dimensions of the simulated system
  index_t L0;    // length of the first dimension
  index_t L1;    // length of the second dimension
  index_t L2;    // length of the third dimension
  index_t L3;    // length of the fourth dimension
  // hard cutoff at 4 for now, since we have not
  // implemented any 5D cases yet
  // more dimensions can and will be added when necessary
  index_t nHits;  // number of hits per site in each sweep
  index_t nSweep; // number of sweeps
  index_t seed;   // seed for the random number generator

  // parameters specific to the GaugeField
  size_t Nd; // number of mu degrees of freedom
  size_t Nc; // number of color degrees of freedom

  // parameters specific to the Wilson action
  real_t beta;  // inverse coupling constant
  real_t delta; // step size for the metropolis update

  real_t epsilon1; // gauge breaking parameter, if applicable
  real_t epsilon2;

  // add more parameters above this line as needed
  // ...

  // default parameters
  // in practice, user should set these
  // parameters after constructing the object
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

    // set default values to newly added parameters
    // above this line
    // ...
  }

  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Metropolis Parameters:\n");
      printf("General Parameters:\n");
      printf("Ndims: %d\n", Ndims);
      printf("L0: %d\n", L0);
      printf("L1: %d\n", L1);
      printf("L2: %d\n", L2);
      printf("L3: %d\n", L3);
      printf("nHits: %d\n", nHits);
      printf("nSweep: %d\n", nSweep);
      printf("seed: %d\n", seed);
      printf("GaugeField Parameters:\n");
      printf("Nd: %zu\n", Nd);
      printf("Nc: %zu\n", Nc);
      printf("Wilson Action Parameters:\n");
      printf("beta: %.4f\n", beta);
      printf("delta: %.4f\n", delta);
    }
  }
};

} // namespace klft
