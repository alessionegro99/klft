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

// utility functions for gauge fields

#include "GaugeField.hpp"

namespace klft {
// calculate staple per site and store in another gauge field
template <size_t Nd, size_t Nc>
const constGaugeField<Nd, Nc> stapleField(const deviceGaugeField<Nd, Nc> g_in) {
  // initialize the output field
  deviceGaugeField<Nd, Nc> g_out(g_in.field.extent(0), g_in.field.extent(1),
                                 g_in.field.extent(2), g_in.field.extent(3),
                                 complex_t(0.0, 0.0));
  // get the start and end indices
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<Nd> start;
  IndexArray<Nd> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  // store the field in a const gauge field
  const constGaugeField<Nd, Nc> g(g_in.field);

  // tune and launch the kernel
  tune_and_launch_for<Nd>(
      "stapleField_GaugeField", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                    const index_t i3) {
// iterate over mu, store staple for each mu
#pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu) { // loop over mu
          // temp SUN matrices to store products
          SUN<Nc> temp = zeroSUN<Nc>();
          // get the x + mu indices
          const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
          const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
          const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
          const index_t i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
// positive directions
#pragma unroll
          for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
            // do nothing for mu = nu
            if (nu == mu)
              continue;
            // get the x + nu indices
            const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
            const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
            const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
            const index_t i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
            // get the staple
            temp += g(i0pmu, i1pmu, i2pmu, i3pmu, nu) *
                    conj(g(i0pnu, i1pnu, i2pnu, i3pnu, mu)) *
                    conj(g(i0, i1, i2, i3, nu));
          } // loop over nu
// negative directions
#pragma unroll
          for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
            // do nothing for mu = nu
            if (nu == mu)
              continue;
            // get the x + mu - nu indices
            const index_t i0pmu_mnu =
                nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
            const index_t i1pmu_mnu =
                nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
            const index_t i2pmu_mnu =
                nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;
            const index_t i3pmu_mnu =
                nu == 3 ? (i3pmu - 1 + dimensions[3]) % dimensions[3] : i3pmu;
            // get the x - nu indices
            const index_t i0mnu =
                nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
            const index_t i1mnu =
                nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
            const index_t i2mnu =
                nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;
            const index_t i3mnu =
                nu == 3 ? (i3 - 1 + dimensions[3]) % dimensions[3] : i3;
            // get the staple
            temp += conj(g(i0pmu_mnu, i1pmu_mnu, i2pmu_mnu, i3pmu_mnu, nu)) *
                    conj(g(i0mnu, i1mnu, i2mnu, i3mnu, mu)) *
                    g(i0mnu, i1mnu, i2mnu, i3mnu, nu);
          } // loop over nu
          // store the staple in the output field
          g_out(i0, i1, i2, i3, mu) = temp;
        } // loop  over mu
      });
  Kokkos::fence();
  // return the output field
  return constGaugeField<Nd, Nc>(g_out.field);
}

} // namespace klft
