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

#include "CompiledTheory.hpp"
#include "Heatbath.hpp"
#include "InputParser.hpp"

#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int Heatbath(const std::string &input_file) {
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  HeatbathParams heatbathParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, heatbathParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }

  if (heatbathParams.epsilon2 != 0.0) {
    printf("Error: heatbath/overrelaxation currently supports epsilon1 only; "
           "epsilon2 requires a different local update.\n");
    return -1;
  }

  heatbathParams.print();
  print_compiled_theory();
  RNGType rng(heatbathParams.seed);

  auto gauge_field = make_identity_gauge_field<compiled_rank, compiled_nc>(
      heatbathParams.L0, heatbathParams.L1, heatbathParams.L2,
      heatbathParams.L3);
  run_heatbath<compiled_rank, compiled_nc>(gauge_field, heatbathParams,
                                           gaugeObsParams, rng);

  return 0;
}

} // namespace klft
