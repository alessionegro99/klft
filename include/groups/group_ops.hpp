#pragma once

#include "core/common.hpp"

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

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void right_multiply_embedded_su2(
    SUNMatrix<Nc> &mat, const index_t i, const index_t j, const SU2 &sub) {
  const complex_t fii = matrix_element(sub, 0, 0);
  const complex_t fij = matrix_element(sub, 0, 1);
  const complex_t fji = matrix_element(sub, 1, 0);
  const complex_t fjj = matrix_element(sub, 1, 1);

#pragma unroll
  for (index_t k = 0; k < static_cast<index_t>(Nc); ++k) {
    const complex_t temp0 =
        matrix_ref(mat, k, i) * fii + matrix_ref(mat, k, j) * fji;
    const complex_t temp1 =
        matrix_ref(mat, k, i) * fij + matrix_ref(mat, k, j) * fjj;
    matrix_ref(mat, k, i) = temp0;
    matrix_ref(mat, k, j) = temp1;
  }
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void rand_matrix(U1 &r, RNG &generator) {
  real_t p0 = 0.0;
  real_t p1 = 0.0;
  real_t p = 0.0;

  do {
    p0 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p1 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p = Kokkos::sqrt(p0 * p0 + p1 * p1);
  } while (p <= 1.0e-13);

  r = make_u1(complex_t(p0 / p, p1 / p));
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void rand_matrix(SU2 &r, RNG &generator) {
  real_t p = 2.0;
  real_t p0 = 0.0;
  real_t p1 = 0.0;
  real_t p2 = 0.0;
  real_t p3 = 0.0;

  while (p > 1.0 || p == 0.0) {
    p0 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p1 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p2 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p3 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p = Kokkos::sqrt(p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3);
  }

  r = make_su2(p0 / p, p1 / p, p2 / p, p3 / p);
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void rand_matrix_p0_su2(const real_t p0, SU2 &r,
                                                    RNG &generator) {
  real_t p = 2.0;
  real_t p1 = 0.0;
  real_t p2 = 0.0;
  real_t p3 = 0.0;

  while (p > 1.0 || p == 0.0) {
    p1 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p2 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p3 = 1.0 - 2.0 * generator.drand(0.0, 1.0);
    p = p1 * p1 + p2 * p2 + p3 * p3;
  }

  const real_t scale = Kokkos::sqrt((1.0 - p0 * p0) / p);
  r = make_su2(p0, p1 * scale, p2 * scale, p3 * scale);
}

template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void rand_matrix(SUNMatrix<Nc> &r, RNG &generator) {
  r = identitySUN<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc) - 1; ++i) {
#pragma unroll
    for (index_t j = i + 1; j < static_cast<index_t>(Nc); ++j) {
      SU2 sub;
      rand_matrix(sub, generator);
      right_multiply_embedded_su2(r, i, j, sub);
    }
  }
}

template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
make_metropolis_matrix(const real_t delta, RNG &generator) {
  SUN<Nc> proposal = identitySUN<Nc>();
  SUN<Nc> random_matrix;
  rand_matrix(random_matrix, generator);
  proposal += random_matrix * delta;
  restoreSUN(proposal);
  return proposal;
}

template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
apply_metropolis_proposal(const SUN<Nc> &link, const real_t delta,
                          RNG &generator) {
  const SUN<Nc> proposal = make_metropolis_matrix<Nc>(delta, generator);
  if (generator.drand(0.0, 1.0) < 0.5) {
    return proposal * link;
  }
  return conj(proposal) * link;
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
  real_t norm0 = Kokkos::sqrt(
      (Kokkos::conj(matrix_ref(c, 0, 0)) * matrix_ref(c, 0, 0) +
       Kokkos::conj(matrix_ref(c, 0, 1)) * matrix_ref(c, 0, 1) +
       Kokkos::conj(matrix_ref(c, 0, 2)) * matrix_ref(c, 0, 2))
          .real());

  if (norm0 <= 1.0e-30) {
    return identitySUN<3>();
  }

#pragma unroll
  for (index_t col = 0; col < 3; ++col) {
    matrix_ref(c, 0, col) /= norm0;
  }

  complex_t row10_overlap(0.0, 0.0);
#pragma unroll
  for (index_t col = 0; col < 3; ++col) {
    row10_overlap += matrix_ref(c, 1, col) *
                     Kokkos::conj(matrix_ref(c, 0, col));
  }
#pragma unroll
  for (index_t col = 0; col < 3; ++col) {
    matrix_ref(c, 1, col) -= row10_overlap * matrix_ref(c, 0, col);
  }

  real_t norm1 = Kokkos::sqrt(
      (Kokkos::conj(matrix_ref(c, 1, 0)) * matrix_ref(c, 1, 0) +
       Kokkos::conj(matrix_ref(c, 1, 1)) * matrix_ref(c, 1, 1) +
       Kokkos::conj(matrix_ref(c, 1, 2)) * matrix_ref(c, 1, 2))
          .real());

  if (norm1 <= 1.0e-30) {
    const real_t row0_norm_0 =
        (Kokkos::conj(matrix_ref(c, 0, 0)) * matrix_ref(c, 0, 0)).real();
    const real_t row0_norm_1 =
        (Kokkos::conj(matrix_ref(c, 0, 1)) * matrix_ref(c, 0, 1)).real();
    const real_t row0_norm_2 =
        (Kokkos::conj(matrix_ref(c, 0, 2)) * matrix_ref(c, 0, 2)).real();
    const index_t basis =
        (row0_norm_0 <= row0_norm_1 && row0_norm_0 <= row0_norm_2)
            ? 0
            : ((row0_norm_1 <= row0_norm_2) ? 1 : 2);
#pragma unroll
    for (index_t col = 0; col < 3; ++col) {
      matrix_ref(c, 1, col) = complex_t(col == basis ? 1.0 : 0.0, 0.0);
    }
    row10_overlap = complex_t(0.0, 0.0);
#pragma unroll
    for (index_t col = 0; col < 3; ++col) {
      row10_overlap += matrix_ref(c, 1, col) *
                       Kokkos::conj(matrix_ref(c, 0, col));
    }
#pragma unroll
    for (index_t col = 0; col < 3; ++col) {
      matrix_ref(c, 1, col) -= row10_overlap * matrix_ref(c, 0, col);
    }
    norm1 = Kokkos::sqrt(
        (Kokkos::conj(matrix_ref(c, 1, 0)) * matrix_ref(c, 1, 0) +
         Kokkos::conj(matrix_ref(c, 1, 1)) * matrix_ref(c, 1, 1) +
         Kokkos::conj(matrix_ref(c, 1, 2)) * matrix_ref(c, 1, 2))
            .real());
  }

#pragma unroll
  for (index_t col = 0; col < 3; ++col) {
    matrix_ref(c, 1, col) /= norm1;
  }

  matrix_ref(c, 2, 0) = Kokkos::conj(matrix_ref(c, 0, 1) * matrix_ref(c, 1, 2) -
                                     matrix_ref(c, 0, 2) * matrix_ref(c, 1, 1));
  matrix_ref(c, 2, 1) = Kokkos::conj(matrix_ref(c, 0, 2) * matrix_ref(c, 1, 0) -
                                     matrix_ref(c, 0, 0) * matrix_ref(c, 1, 2));
  matrix_ref(c, 2, 2) = Kokkos::conj(matrix_ref(c, 0, 0) * matrix_ref(c, 1, 1) -
                                     matrix_ref(c, 0, 1) * matrix_ref(c, 1, 0));
  const real_t norm2 = Kokkos::sqrt(
      (Kokkos::conj(matrix_ref(c, 2, 0)) * matrix_ref(c, 2, 0) +
       Kokkos::conj(matrix_ref(c, 2, 1)) * matrix_ref(c, 2, 1) +
       Kokkos::conj(matrix_ref(c, 2, 2)) * matrix_ref(c, 2, 2))
          .real());
#pragma unroll
  for (index_t col = 0; col < 3; ++col) {
    matrix_ref(c, 2, col) /= norm2;
  }

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
