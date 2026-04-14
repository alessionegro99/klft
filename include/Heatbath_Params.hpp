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
