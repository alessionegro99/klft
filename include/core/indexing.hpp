#pragma once
#include "core/common.hpp"

#include <string>
#include <stdexcept>

namespace klft {

// Shift a lattice index forward in direction `mu` with periodic wrapping.
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
  const index_t extent = dimensions[mu];
  const index_t normalized_shift = shift % extent;
  new_idx[mu] =
      (static_cast<index_t>(idx[mu]) + normalized_shift) % extent;
  return new_idx;
}

// Shift a lattice index backward in direction `mu` with periodic wrapping.
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
  const index_t extent = dimensions[mu];
  const index_t normalized_shift = shift % extent;
  new_idx[mu] =
      (static_cast<index_t>(idx[mu]) - normalized_shift + extent) % extent;
  return new_idx;
}

// Map a half-volume index onto one odd/even sublattice.
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

// Decode a sublattice bit mask into per-direction odd/even flags.
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<bool, rank>
oddeven_array(const indexType &val) {
  Kokkos::Array<bool, rank> oddeven;
  for (index_t i = 0; i < rank; ++i) {
    oddeven[rank - 1 - i] = (val & (1 << i)) != 0;
  }
  return oddeven;
}

template <size_t rank>
inline void validate_even_extents(const IndexArray<rank> &dimensions,
                                  const char *algorithm_name) {
  for (index_t d = 0; d < rank; ++d) {
    if (dimensions[d] <= 0 || dimensions[d] % 2 != 0) {
      throw std::runtime_error(std::string(algorithm_name) +
                               " requires positive even lattice extents in "
                               "every updated direction.");
    }
  }
}

} // namespace klft
