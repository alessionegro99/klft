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

// define plaquette functions for different gauge fields

#pragma once
#include "FieldTypeHelper.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"

namespace klft {

// define a function to calculate the gauge plaquette
// U_{mu nu} (x) = Tr[ U_mu(x) U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x) ]
// for SU(N) gauge group

// first define the necessary functor
template <size_t rank, size_t Nc> struct GaugePlaq {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd = rank;
  // define the gauge field type
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  // define the field type
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site;
  const IndexArray<rank> dimensions;
  GaugePlaq(const GaugeFieldType &g_in, FieldType &plaq_per_site,
            const IndexArray<rank> &dimensions)
      : g_in(g_in), plaq_per_site(plaq_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // temp SUN matrices to store products
    SUN<Nc> lmu, lnu;
    // reduction variable for all mu and nu
    complex_t tmunu(0.0, 0.0);

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          // build plaquette in two halves
          // U_{mu nu} (x) = Tr[ lmu * lnu^dagger ]
          // lmu = U_mu(x) * U_nu(x+mu)
          lmu =
              g_in(Idcs..., mu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions),
                   nu);
          // lnu = U_nu(x) * U_mu(x+nu)
          lnu =
              g_in(Idcs..., nu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, nu, 1, dimensions),
                   mu);
// multiply the 2 half plaquettes
// lmu * lnu^dagger
// take the trace
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < Nc; ++c2) {
              tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
            }
          }
        }
      }
    }
    // store the result in the temporary field
    plaq_per_site(Idcs...) = tmunu;
  }
};

template <size_t rank, size_t Nc>
real_t GaugePlaquette(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                      const bool normalize = true) {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd = rank;
  // final return variable
  complex_t plaq = 0.0;
  // get the start and end indices
  // this is temporary solution
  // ideally, we want to have a policy factory
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site(end, complex_t(0.0, 0.0));

  // define the functor
  GaugePlaq<rank, Nc> gaugePlaquette(g_in, plaq_per_site, end);

  Kokkos::parallel_for(Policy<rank>(start, end), gaugePlaquette);
  Kokkos::fence();

  // sum over all sites
  plaq = plaq_per_site.sum();
  Kokkos::fence();

  // normalization
  if (normalize) {
    real_t norm = 1.0;
    for (index_t i = 0; i < rank; ++i) {
      norm *= static_cast<real_t>(end[i]);
    }
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}
} // namespace klft
