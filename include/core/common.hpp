#pragma once
#include "groups/gauge_group.hpp"

#include <Kokkos_Core.hpp>

namespace klft {

template <size_t rank> using IndexArray = Kokkos::Array<index_t, rank>;

template <size_t Nd, size_t Nc>
using GaugeField =
    Kokkos::View<SUN<Nc> ****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField3D =
    Kokkos::View<SUN<Nc> ***[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField2D =
    Kokkos::View<SUN<Nc> **[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field =
    Kokkos::View<complex_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field3D =
    Kokkos::View<complex_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field2D =
    Kokkos::View<complex_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField =
    Kokkos::View<real_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField3D =
    Kokkos::View<real_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField2D =
    Kokkos::View<real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t rank> using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <size_t Nc> constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> zeroSUN();

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<1> zeroSUN<1>() {
  return make_u1(complex_t(0.0, 0.0));
}

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<2> zeroSUN<2>() {
  return make_su2(0.0, 0.0, 0.0, 0.0);
}

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<3> zeroSUN<3>() {
  return make_zero_sun_matrix<3>();
}

template <size_t Nc>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> identitySUN();

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<1> identitySUN<1>() {
  return make_u1(complex_t(1.0, 0.0));
}

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<2> identitySUN<2>() {
  return make_su2(1.0, 0.0, 0.0, 0.0);
}

template <>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<3> identitySUN<3>() {
  auto id = zeroSUN<3>();
#pragma unroll
  for (index_t c = 0; c < 3; ++c) {
    matrix_ref(id, c, c) = complex_t(1.0, 0.0);
  }
  return id;
}

} // namespace klft
