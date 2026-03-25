#pragma once
#include "GLOBAL.hpp"

namespace klft {

// x -> x + mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_plus(const Kokkos::Array<indexType, rank> &idx, const index_t mu,
                 const index_t shift, const IndexArray<rank> &dimensions) {
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] + shift) % dimensions[mu];
  return new_idx;
}

// x -> x - mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_minus(const Kokkos::Array<indexType, rank> &idx, const index_t mu,
                  const index_t shift, const IndexArray<rank> &dimensions) {
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  // TODO: check that we are shifting maximum by one
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
    // TODO: check that val is less than 2^rank
    oddeven[rank - 1 - i] = (val & (1 << i)) != 0;
  }
  return oddeven;
}

} // namespace klft
