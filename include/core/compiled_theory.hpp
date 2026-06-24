#pragma once

#include "core/klft_config.hpp"
#include "fields/field_type_traits.hpp"
namespace klft {

using CompiledGaugeField =
    typename DeviceGaugeFieldType<compiled_rank, compiled_nc>::type;

template <size_t rank, size_t Nc>
typename DeviceGaugeFieldType<rank, Nc>::type
make_identity_gauge_field(const index_t L0, const index_t L1, const index_t L2,
                          const index_t L3) {
  if constexpr (rank == 4) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1, L2, L3,
                                                         identitySUN<Nc>());
  } else if constexpr (rank == 3) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1, L2,
                                                         identitySUN<Nc>());
  } else {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1,
                                                         identitySUN<Nc>());
  }
}

// Construct a hot configuration using the seeded Kokkos RNG pool supplied by
// the driver. The field wrappers use the same random group-matrix construction
// as the update code (and as yang-mills-Bonn's random start).
template <size_t rank, size_t Nc, class RNG>
typename DeviceGaugeFieldType<rank, Nc>::type
make_hot_gauge_field(const index_t L0, const index_t L1, const index_t L2,
                     const index_t L3, RNG &rng) {
  if constexpr (rank == 4) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1, L2, L3, rng);
  } else if constexpr (rank == 3) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1, L2, rng);
  } else {
    return typename DeviceGaugeFieldType<rank, Nc>::type(L0, L1, rng);
  }
}

} // namespace klft
