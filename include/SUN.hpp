#pragma once

#include "GLOBAL.hpp"

namespace klft {

KOKKOS_FORCEINLINE_FUNCTION U1 operator*(const U1 &a, const U1 &b) {
  return make_u1(a.comp * b.comp);
}

KOKKOS_FORCEINLINE_FUNCTION SU2 operator*(const SU2 &a, const SU2 &b) {
  return make_su2(
      a.comp[0] * b.comp[0] - a.comp[1] * b.comp[1] - a.comp[2] * b.comp[2] -
          a.comp[3] * b.comp[3],
      a.comp[0] * b.comp[1] + a.comp[1] * b.comp[0] - a.comp[2] * b.comp[3] +
          a.comp[3] * b.comp[2],
      a.comp[0] * b.comp[2] + a.comp[2] * b.comp[0] + a.comp[1] * b.comp[3] -
          a.comp[3] * b.comp[1],
      a.comp[0] * b.comp[3] + a.comp[3] * b.comp[0] - a.comp[1] * b.comp[2] +
          a.comp[2] * b.comp[1]);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> operator*(const SUNMatrix<Nc> &a,
                                                    const SUNMatrix<Nc> &b) {
  SUNMatrix<Nc> c = make_zero_sun_matrix<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t j = 0; j < static_cast<index_t>(Nc); ++j) {
      complex_t sum(0.0, 0.0);
#pragma unroll
      for (index_t k = 0; k < static_cast<index_t>(Nc); ++k) {
        sum += matrix_ref(a, i, k) * matrix_ref(b, k, j);
      }
      matrix_ref(c, i, j) = sum;
    }
  }
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION U1 &operator*=(U1 &a, const U1 &b) {
  a = a * b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION SU2 &operator*=(SU2 &a, const SU2 &b) {
  a = a * b;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> &operator*=(SUNMatrix<Nc> &a,
                                                      const SUNMatrix<Nc> &b) {
  a = a * b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION U1 operator+(const U1 &a, const U1 &b) {
  return make_u1(a.comp + b.comp);
}

KOKKOS_FORCEINLINE_FUNCTION SU2 operator+(const SU2 &a, const SU2 &b) {
  return make_su2(a.comp[0] + b.comp[0], a.comp[1] + b.comp[1],
                  a.comp[2] + b.comp[2], a.comp[3] + b.comp[3]);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> operator+(const SUNMatrix<Nc> &a,
                                                    const SUNMatrix<Nc> &b) {
  SUNMatrix<Nc> c = make_zero_sun_matrix<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc); ++i) {
    c.comp[i] = a.comp[i] + b.comp[i];
  }
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION U1 &operator+=(U1 &a, const U1 &b) {
  a = a + b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION SU2 &operator+=(SU2 &a, const SU2 &b) {
  a = a + b;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> &operator+=(SUNMatrix<Nc> &a,
                                                      const SUNMatrix<Nc> &b) {
  a = a + b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION U1 operator-(const U1 &a, const U1 &b) {
  return make_u1(a.comp - b.comp);
}

KOKKOS_FORCEINLINE_FUNCTION SU2 operator-(const SU2 &a, const SU2 &b) {
  return make_su2(a.comp[0] - b.comp[0], a.comp[1] - b.comp[1],
                  a.comp[2] - b.comp[2], a.comp[3] - b.comp[3]);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> operator-(const SUNMatrix<Nc> &a,
                                                    const SUNMatrix<Nc> &b) {
  SUNMatrix<Nc> c = make_zero_sun_matrix<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc); ++i) {
    c.comp[i] = a.comp[i] - b.comp[i];
  }
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION U1 &operator-=(U1 &a, const U1 &b) {
  a = a - b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION SU2 &operator-=(SU2 &a, const SU2 &b) {
  a = a - b;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> &operator-=(SUNMatrix<Nc> &a,
                                                      const SUNMatrix<Nc> &b) {
  a = a - b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION U1 operator*(const U1 &a, const real_t &b) {
  return make_u1(a.comp * b);
}

KOKKOS_FORCEINLINE_FUNCTION SU2 operator*(const SU2 &a, const real_t &b) {
  return make_su2(a.comp[0] * b, a.comp[1] * b, a.comp[2] * b, a.comp[3] * b);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> operator*(const SUNMatrix<Nc> &a,
                                                    const real_t &b) {
  SUNMatrix<Nc> c = make_zero_sun_matrix<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc); ++i) {
    c.comp[i] = a.comp[i] * b;
  }
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION U1 operator*(const real_t &b, const U1 &a) {
  return a * b;
}

KOKKOS_FORCEINLINE_FUNCTION SU2 operator*(const real_t &b, const SU2 &a) {
  return a * b;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> operator*(const real_t &b,
                                                    const SUNMatrix<Nc> &a) {
  return a * b;
}

KOKKOS_FORCEINLINE_FUNCTION U1 &operator*=(U1 &a, const real_t &b) {
  a = a * b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION SU2 &operator*=(SU2 &a, const real_t &b) {
  a = a * b;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> &operator*=(SUNMatrix<Nc> &a,
                                                      const real_t &b) {
  a = a * b;
  return a;
}

KOKKOS_FORCEINLINE_FUNCTION U1 conj(const U1 &a) {
  return make_u1(Kokkos::conj(a.comp));
}

KOKKOS_FORCEINLINE_FUNCTION SU2 conj(const SU2 &a) {
  return make_su2(a.comp[0], -a.comp[1], -a.comp[2], -a.comp[3]);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNMatrix<Nc> conj(const SUNMatrix<Nc> &a) {
  SUNMatrix<Nc> c = make_zero_sun_matrix<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t j = 0; j < static_cast<index_t>(Nc); ++j) {
      matrix_ref(c, i, j) = Kokkos::conj(matrix_ref(a, j, i));
    }
  }
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION complex_t trace(const U1 &a) { return a.comp; }

KOKKOS_FORCEINLINE_FUNCTION complex_t trace(const SU2 &a) {
  return complex_t(2.0 * a.comp[0], 0.0);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION complex_t trace(const SUNMatrix<Nc> &a) {
  complex_t out(0.0, 0.0);
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
    out += matrix_ref(a, i, i);
  }
  return out;
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<1> &r, RNG &generator,
                                         const real_t delta) {
  r = make_u1(Kokkos::exp(complex_t(
      0.0, generator.drand(-delta * Kokkos::numbers::pi_v<real_t>,
                           delta * Kokkos::numbers::pi_v<real_t>))));
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<2> &r, RNG &generator,
                                         const real_t delta) {
  const real_t alpha =
      generator.drand(0.0, delta * 2.0 * Kokkos::numbers::pi_v<real_t>);
  const real_t u = generator.drand(-1.0, 1.0);
  const real_t theta =
      generator.drand(0.0, 2.0 * Kokkos::numbers::pi_v<real_t>);
  const real_t salpha = Kokkos::sin(alpha);
  const real_t radius = Kokkos::sqrt(1.0 - u * u);
  r = make_su2(Kokkos::cos(alpha), radius * Kokkos::sin(theta) * salpha,
               radius * Kokkos::cos(theta) * salpha, u * salpha);
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<3> &r, RNG &generator,
                                         const real_t delta) {
  real_t r1[6], r2[6], norm, fact;
  complex_t z1[3], z2[3], z3[3], z;
  while (true) {
    for (int i = 0; i < 6; ++i) {
      r1[i] = generator.drand(0.0, delta);
    }
    norm = Kokkos::sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2] +
                        r1[3] * r1[3] + r1[4] * r1[4] + r1[5] * r1[5]);
    if (1.0 != (1.0 + norm)) {
      break;
    }
  }

  fact = 1.0 / norm;
  z1[0] = fact * complex_t(r1[0], r1[1]);
  z1[1] = fact * complex_t(r1[2], r1[3]);
  z1[2] = fact * complex_t(r1[4], r1[5]);

  while (true) {
    while (true) {
      for (int i = 0; i < 6; ++i) {
        r2[i] = generator.drand(0.0, delta);
      }
      norm = Kokkos::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2] +
                          r2[3] * r2[3] + r2[4] * r2[4] + r2[5] * r2[5]);
      if (1.0 != (1.0 + norm)) {
        break;
      }
    }

    fact = 1.0 / norm;
    z2[0] = fact * complex_t(r2[0], r2[1]);
    z2[1] = fact * complex_t(r2[2], r2[3]);
    z2[2] = fact * complex_t(r2[4], r2[5]);
    z = Kokkos::conj(z1[0]) * z2[0] + Kokkos::conj(z1[1]) * z2[1] +
        Kokkos::conj(z1[2]) * z2[2];
    z2[0] -= z * z1[0];
    z2[1] -= z * z1[1];
    z2[2] -= z * z1[2];
    norm =
        Kokkos::sqrt(z2[0].real() * z2[0].real() + z2[0].imag() * z2[0].imag() +
                     z2[1].real() * z2[1].real() + z2[1].imag() * z2[1].imag() +
                     z2[2].real() * z2[2].real() + z2[2].imag() * z2[2].imag());
    if (1.0 != (1.0 + norm)) {
      break;
    }
  }

  fact = 1.0 / norm;
  z2[0] *= fact;
  z2[1] *= fact;
  z2[2] *= fact;
  z3[0] = Kokkos::conj((z1[1] * z2[2]) - (z1[2] * z2[1]));
  z3[1] = Kokkos::conj((z1[2] * z2[0]) - (z1[0] * z2[2]));
  z3[2] = Kokkos::conj((z1[0] * z2[1]) - (z1[1] * z2[0]));

  r = zeroSUN<3>();
  matrix_ref(r, 0, 0) = z1[0];
  matrix_ref(r, 0, 1) = z1[1];
  matrix_ref(r, 0, 2) = z1[2];
  matrix_ref(r, 1, 0) = z2[0];
  matrix_ref(r, 1, 1) = z2[1];
  matrix_ref(r, 1, 2) = z2[2];
  matrix_ref(r, 2, 0) = z3[0];
  matrix_ref(r, 2, 1) = z3[1];
  matrix_ref(r, 2, 2) = z3[2];
}

KOKKOS_FORCEINLINE_FUNCTION SUN<1> restoreSUN(const SUN<1> &a) {
  const real_t norm =
      Kokkos::sqrt(a.comp.real() * a.comp.real() + a.comp.imag() * a.comp.imag());
  return make_u1(a.comp / norm);
}

KOKKOS_FORCEINLINE_FUNCTION void restoreSUN(SUN<1> &a) {
  a = restoreSUN(static_cast<const SUN<1> &>(a));
}

KOKKOS_FORCEINLINE_FUNCTION SUN<2> restoreSUN(const SUN<2> &a) {
  const real_t norm =
      Kokkos::sqrt(a.comp[0] * a.comp[0] + a.comp[1] * a.comp[1] +
                   a.comp[2] * a.comp[2] + a.comp[3] * a.comp[3]);
  return make_su2(a.comp[0] / norm, a.comp[1] / norm, a.comp[2] / norm,
                  a.comp[3] / norm);
}

KOKKOS_FORCEINLINE_FUNCTION void restoreSUN(SUN<2> &a) {
  a = restoreSUN(static_cast<const SUN<2> &>(a));
}

KOKKOS_FORCEINLINE_FUNCTION SUN<3> restoreSUN(const SUN<3> &a) {
  SUN<3> c = a;
  const real_t norm0 = Kokkos::sqrt(
      (Kokkos::conj(matrix_ref(c, 0, 0)) * matrix_ref(c, 0, 0) +
       Kokkos::conj(matrix_ref(c, 0, 1)) * matrix_ref(c, 0, 1) +
       Kokkos::conj(matrix_ref(c, 0, 2)) * matrix_ref(c, 0, 2))
          .real());
  const real_t norm1 = Kokkos::sqrt(
      (Kokkos::conj(matrix_ref(c, 1, 0)) * matrix_ref(c, 1, 0) +
       Kokkos::conj(matrix_ref(c, 1, 1)) * matrix_ref(c, 1, 1) +
       Kokkos::conj(matrix_ref(c, 1, 2)) * matrix_ref(c, 1, 2))
          .real());

  matrix_ref(c, 0, 0) /= norm0;
  matrix_ref(c, 0, 1) /= norm0;
  matrix_ref(c, 0, 2) /= norm0;
  matrix_ref(c, 1, 0) /= norm1;
  matrix_ref(c, 1, 1) /= norm1;
  matrix_ref(c, 1, 2) /= norm1;
  matrix_ref(c, 2, 0) = Kokkos::conj(matrix_ref(c, 0, 1) * matrix_ref(c, 1, 2) -
                                     matrix_ref(c, 0, 2) * matrix_ref(c, 1, 1));
  matrix_ref(c, 2, 1) = Kokkos::conj(matrix_ref(c, 0, 2) * matrix_ref(c, 1, 0) -
                                     matrix_ref(c, 0, 0) * matrix_ref(c, 1, 2));
  matrix_ref(c, 2, 2) = Kokkos::conj(matrix_ref(c, 0, 0) * matrix_ref(c, 1, 1) -
                                     matrix_ref(c, 0, 1) * matrix_ref(c, 1, 0));
  matrix_ref(c, 1, 0) = Kokkos::conj(matrix_ref(c, 2, 1) * matrix_ref(c, 0, 2) -
                                     matrix_ref(c, 2, 2) * matrix_ref(c, 0, 1));
  matrix_ref(c, 1, 1) = Kokkos::conj(matrix_ref(c, 2, 2) * matrix_ref(c, 0, 0) -
                                     matrix_ref(c, 2, 0) * matrix_ref(c, 0, 2));
  matrix_ref(c, 1, 2) = Kokkos::conj(matrix_ref(c, 2, 0) * matrix_ref(c, 0, 1) -
                                     matrix_ref(c, 2, 1) * matrix_ref(c, 0, 0));
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION void restoreSUN(SUN<3> &a) {
  a = restoreSUN(static_cast<const SUN<3> &>(a));
}

} // namespace klft
