#pragma once

#include <Kokkos_Core.hpp>

namespace klft {

using real_t = double;
using complex_t = Kokkos::complex<real_t>;
using index_t = int;

struct U1 {
  complex_t comp;
};

struct SU2 {
  Kokkos::Array<real_t, 4> comp;
};

template <size_t Nc> struct SUNMatrix {
  Kokkos::Array<complex_t, Nc * Nc> comp;
};

template <size_t Nc> struct GaugeGroupSelector {
  static_assert(Nc >= 1 && Nc <= 3,
                "KLFT supports only Nc = 1, 2, or 3.");
};

template <> struct GaugeGroupSelector<1> {
  using type = U1;
};

template <> struct GaugeGroupSelector<2> {
  using type = SU2;
};

template <> struct GaugeGroupSelector<3> {
  using type = SUNMatrix<3>;
};

template <size_t Nc> using SUN = typename GaugeGroupSelector<Nc>::type;

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr index_t flat_index(const index_t row,
                                                         const index_t col) {
  return row * static_cast<index_t>(Nc) + col;
}

KOKKOS_FORCEINLINE_FUNCTION constexpr U1 make_u1(const complex_t value) {
  return U1{value};
}

KOKKOS_FORCEINLINE_FUNCTION constexpr SU2 make_su2(const real_t p0,
                                                   const real_t p1,
                                                   const real_t p2,
                                                   const real_t p3) {
  return SU2{{p0, p1, p2, p3}};
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr SUNMatrix<Nc> make_zero_sun_matrix() {
  SUNMatrix<Nc> out{};
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc); ++i) {
    out.comp[i] = complex_t(0.0, 0.0);
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr complex_t
matrix_element(const SUNMatrix<Nc> &a, const index_t row, const index_t col) {
  return a.comp[flat_index<Nc>(row, col)];
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr complex_t &
matrix_ref(SUNMatrix<Nc> &a, const index_t row, const index_t col) {
  return a.comp[flat_index<Nc>(row, col)];
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr const complex_t &
matrix_ref(const SUNMatrix<Nc> &a, const index_t row, const index_t col) {
  return a.comp[flat_index<Nc>(row, col)];
}

KOKKOS_FORCEINLINE_FUNCTION constexpr complex_t
matrix_element(const U1 &a, const index_t row, const index_t col) {
  return (row == 0 && col == 0) ? a.comp : complex_t(0.0, 0.0);
}

KOKKOS_FORCEINLINE_FUNCTION constexpr complex_t
matrix_element(const SU2 &a, const index_t row, const index_t col) {
  const real_t p0 = a.comp[0];
  const real_t p1 = a.comp[1];
  const real_t p2 = a.comp[2];
  const real_t p3 = a.comp[3];

  if (row == 0 && col == 0) {
    return complex_t(p0, p3);
  }
  if (row == 0 && col == 1) {
    return complex_t(p2, p1);
  }
  if (row == 1 && col == 0) {
    return complex_t(-p2, p1);
  }
  return complex_t(p0, -p3);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr complex_t
matrix_element(const SUN<Nc> &a, const index_t row, const index_t col) {
  if constexpr (Nc == 1) {
    return matrix_element(static_cast<const U1 &>(a), row, col);
  } else if constexpr (Nc == 2) {
    return matrix_element(static_cast<const SU2 &>(a), row, col);
  } else {
    return matrix_element(static_cast<const SUNMatrix<Nc> &>(a), row, col);
  }
}

} // namespace klft
