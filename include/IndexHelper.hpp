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

// this file defines helper functions for index manipulation

#pragma once
#include "GLOBAL.hpp"

namespace klft {

// return x + shift  mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_plus(const Kokkos::Array<indexType, rank> &idx, const index_t mu,
                 const index_t shift, const IndexArray<rank> &dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] + shift) % dimensions[mu];
  return new_idx;
}

// return x - shift mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_minus(const Kokkos::Array<indexType, rank> &idx, const index_t mu,
                  const index_t shift, const IndexArray<rank> &dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] - shift + dimensions[mu]) % dimensions[mu];
  return new_idx;
}

// return index based on odd/even sublattice
// this does not check if the index is valid
// it is assumed that all of idx is
// less than half of the dimensional extents
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
index_odd_even(const Kokkos::Array<indexType, rank> &idx,
               const Kokkos::Array<bool, rank> &oddeven) {
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = oddeven[i] ? static_cast<index_t>(2 * idx[i] + 1)
                            : static_cast<index_t>(2 * idx[i]);
  }
  return new_idx;
}

// return an array of boolean values
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<bool, rank>
oddeven_array(const indexType &val) {
  Kokkos::Array<bool, rank> oddeven;
  for (index_t i = 0; i < rank; ++i) {
    oddeven[rank - 1 - i] = (val & (1 << i)) != 0;
  }
  return oddeven;
}

} // namespace klft
