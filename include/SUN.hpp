// define SUN operations

#pragma once
#include "GLOBAL.hpp"

namespace klft {

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][0] * b[0][j];
#pragma unroll
      for (size_t k = 1; k < Nc; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator*=(SUN<Nc> &a, const SUN<Nc> &b) {
  SUN<Nc> c = a * b;
  a = c;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator+(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator+=(SUN<Nc> &a, const SUN<Nc> &b) {
  SUN<Nc> c = a + b;
  a = c;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator-(const SUN<Nc> &a,
                                              const SUN<Nc> &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator-=(SUN<Nc> &a, const SUN<Nc> &b) {
  SUN<Nc> c = a - b;
  a = c;
  return a;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator*(const SUN<Nc> &a,
                                              const real_t &b) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][j] * b;
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator*=(SUN<Nc> &a, const real_t &b) {
  SUN<Nc> c = a * b;
  a = c;
  return a;
}

// TODO: rename to Hermitian conjugate (dagger)
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> conj(const SUN<Nc> &a) {
  SUN<Nc> c;
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = Kokkos::conj(a[j][i]);
    }
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION complex_t trace(const SUN<Nc> &a) {
  complex_t c(0.0, 0.0);
#pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
    c += a[i][i];
  }
  return c;
}

// random SUN matrix generator
// need to be defined for each Nc

// template <size_t N = Nc, typename std::enable_if<N == 1, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<1> &r, RNG &generator,
                                         real_t delta) {
  // SUN<1> r;
  r[0][0] = Kokkos::exp(
      complex_t(0.0, generator.drand(-delta * Kokkos::numbers::pi_v<real_t>,
                                     delta * Kokkos::numbers::pi_v<real_t>)));
  // return r;
}

// template <size_t N = Nc, typename std::enable_if<N == 2, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<2> &r, RNG &generator,
                                         real_t delta) {
  // SUN<2> r;
  real_t alpha =
      generator.drand(0.0, delta * 2 * Kokkos::numbers::pi_v<real_t>);
  real_t u = generator.drand(-1.0, 1.0);
  real_t theta = generator.drand(0.0, 2.0 * Kokkos::numbers::pi_v<real_t>);
  real_t salpha = Kokkos::sin(alpha);
  real_t n1 = Kokkos::sqrt(1.0 - u * u) * Kokkos::sin(theta);
  real_t n2 = Kokkos::sqrt(1.0 - u * u) * Kokkos::cos(theta);
  r[0][0] = complex_t(Kokkos::cos(alpha), u * salpha);
  r[0][1] = complex_t(n1 * salpha, n2 * salpha);
  r[1][0] = complex_t(-r[0][1].real(), r[0][1].imag());
  r[1][1] = complex_t(r[0][0].real(), -r[0][0].imag());
  // return r;
}

// TODO: suspicious
// template <size_t N = Nc, typename std::enable_if<N == 3, int>::type = 0,
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUN(SUN<3> &r, RNG &generator,
                                         real_t delta) {
  // SUN<3> r;
  real_t r1[6], r2[6], norm, fact;
  complex_t z1[3], z2[3], z3[3], z;
  while (1) {
    for (int i = 0; i < 6; i++) {
      r1[i] = generator.drand(0, delta);
    }
    norm = Kokkos::sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2] +
                        r1[3] * r1[3] + r1[4] * r1[4] + r1[5] * r1[5]);
    if (1.0 != (1.0 + norm))
      break;
  }
  fact = 1.0 / norm;
  z1[0] = fact * complex_t(r1[0], r1[1]);
  z1[1] = fact * complex_t(r1[2], r1[3]);
  z1[2] = fact * complex_t(r1[4], r1[5]);
  while (1) {
    while (1) {
      for (int i = 0; i < 6; i++) {
        r2[i] = generator.drand(0, delta);
      }
      norm = Kokkos::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2] +
                          r2[3] * r2[3] + r2[4] * r2[4] + r2[5] * r2[5]);
      if (1.0 != (1.0 + norm))
        break;
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
    if (1.0 != (1.0 + norm))
      break;
  }
  fact = 1.0 / norm;
  z2[0] *= fact;
  z2[1] *= fact;
  z2[2] *= fact;
  z3[0] = Kokkos::conj((z1[1] * z2[2]) - (z1[2] * z2[1]));
  z3[1] = Kokkos::conj((z1[2] * z2[0]) - (z1[0] * z2[2]));
  z3[2] = Kokkos::conj((z1[0] * z2[1]) - (z1[1] * z2[0]));
  r[0][0] = z1[0];
  r[0][1] = z1[1];
  r[0][2] = z1[2];
  r[1][0] = z2[0];
  r[1][1] = z2[1];
  r[1][2] = z2[2];
  r[2][0] = z3[0];
  r[2][1] = z3[1];
  r[2][2] = z3[2];
  // return r;
}

// restore the gauge symmetry
// this also must be defined for each Nc

KOKKOS_FORCEINLINE_FUNCTION
SUN<1> restoreSUN(const SUN<1> &a) {
  SUN<1> c;
  c[0][0] = a[0][0] / Kokkos::sqrt(a[0][0].real() * a[0][0].real() +
                                   a[0][0].imag() * a[0][0].imag());
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<1> &a) {
  a[0][0] /= Kokkos::sqrt(a[0][0].real() * a[0][0].real() +
                          a[0][0].imag() * a[0][0].imag());
}

KOKKOS_FORCEINLINE_FUNCTION
SUN<2> restoreSUN(const SUN<2> &a) {
  SUN<2> c;
  real_t norm = Kokkos::sqrt(
      a[0][0].real() * a[0][0].real() + a[0][0].imag() * a[0][0].imag() +
      a[0][1].real() * a[0][1].real() + a[0][1].imag() * a[0][1].imag());
  c[0][0] = a[0][0] / norm;
  c[0][1] = a[0][1] / norm;
  c[1][0] = a[1][0] / norm;
  c[1][1] = a[1][1] / norm;
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<2> &a) {
  real_t norm = Kokkos::sqrt(
      a[0][0].real() * a[0][0].real() + a[0][0].imag() * a[0][0].imag() +
      a[0][1].real() * a[0][1].real() + a[0][1].imag() * a[0][1].imag());
  a[0][0] /= norm;
  a[0][1] /= norm;
  a[1][0] /= norm;
  a[1][1] /= norm;
}

KOKKOS_FORCEINLINE_FUNCTION
SUN<3> restoreSUN(const SUN<3> &a) {
  SUN<3> c;
  real_t norm0 = Kokkos::sqrt((Kokkos::conj(a[0][0]) * a[0][0] +
                               Kokkos::conj(a[0][1]) * a[0][1] +
                               Kokkos::conj(a[0][2]) * a[0][2])
                                  .real());
  real_t norm1 = Kokkos::sqrt((Kokkos::conj(a[1][0]) * a[1][0] +
                               Kokkos::conj(a[1][1]) * a[1][1] +
                               Kokkos::conj(a[1][2]) * a[1][2])
                                  .real());
  c[0][0] = a[0][0] / norm0;
  c[0][1] = a[0][1] / norm0;
  c[0][2] = a[0][2] / norm0;
  c[1][0] = a[1][0] / norm1;
  c[1][1] = a[1][1] / norm1;
  c[1][2] = a[1][2] / norm1;
  c[2][0] = Kokkos::conj((c[0][1] * c[1][2]) - (c[0][2] * c[1][1]));
  c[2][1] = Kokkos::conj((c[0][2] * c[1][0]) - (c[0][0] * c[1][2]));
  c[2][2] = Kokkos::conj((c[0][0] * c[1][1]) - (c[0][1] * c[1][0]));
  c[1][0] = Kokkos::conj((c[2][1] * c[0][2]) - (c[2][2] * c[0][1]));
  c[1][1] = Kokkos::conj((c[2][2] * c[0][0]) - (c[2][0] * c[0][2]));
  c[1][2] = Kokkos::conj((c[2][0] * c[0][1]) - (c[2][1] * c[0][0]));
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
void restoreSUN(SUN<3> &a) {
  real_t norm0 = Kokkos::sqrt((Kokkos::conj(a[0][0]) * a[0][0] +
                               Kokkos::conj(a[0][1]) * a[0][1] +
                               Kokkos::conj(a[0][2]) * a[0][2])
                                  .real());
  real_t norm1 = Kokkos::sqrt((Kokkos::conj(a[1][0]) * a[1][0] +
                               Kokkos::conj(a[1][1]) * a[1][1] +
                               Kokkos::conj(a[1][2]) * a[1][2])
                                  .real());
  a[0][0] /= norm0;
  a[0][1] /= norm0;
  a[0][2] /= norm0;
  a[1][0] /= norm1;
  a[1][1] /= norm1;
  a[1][2] /= norm1;
  a[2][0] = Kokkos::conj((a[0][1] * a[1][2]) - (a[0][2] * a[1][1]));
  a[2][1] = Kokkos::conj((a[0][2] * a[1][0]) - (a[0][0] * a[1][2]));
  a[2][2] = Kokkos::conj((a[0][0] * a[1][1]) - (a[0][1] * a[1][0]));
  a[1][0] = Kokkos::conj((a[2][1] * a[0][2]) - (a[2][2] * a[0][1]));
  a[1][1] = Kokkos::conj((a[2][2] * a[0][0]) - (a[2][0] * a[0][2]));
  a[1][2] = Kokkos::conj((a[2][0] * a[0][1]) - (a[2][1] * a[0][0]));
}

} // namespace klft
