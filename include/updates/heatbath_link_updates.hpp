#pragma once
#include "core/indexing.hpp"
#include "groups/group_ops.hpp"

namespace klft {

// Kennedy-Pendleton / Creutz SU(2) heatbath kernel.
template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION real_t randheat_SU2(const real_t k,
                                                RNG &generator) {
  real_t r, r1, r2, c, r3;

  if (k > 1.6847) {
    while (true) {
      r = generator.drand(0.0, 1.0);
      r1 = generator.drand(0.0, 1.0);

      r = -Kokkos::log(r) / k;
      r1 = -Kokkos::log(r1) / k;

      r2 = generator.drand(0.0, 1.0);
      c = Kokkos::cos(r2 * 2.0 * Kokkos::numbers::pi_v<real_t>);
      c *= c;

      r *= c;
      r1 += r;

      r3 = generator.drand(0.0, 1.0);
      r = 1.0 - 0.5 * r1 - r3 * r3;
      if (r > 0.0) {
        return 1.0 - r1;
      }
    }
  }

  r1 = Kokkos::exp(-2.0 * k);
  while (true) {
    do {
      r = generator.drand(0.0, 1.0);
    } while (r < r1);
    r = 1.0 + Kokkos::log(r) / k;

    r2 = generator.drand(0.0, 1.0);
    if (r * r < r2 * (2.0 - r2)) {
      return r;
    }
  }
}

KOKKOS_FORCEINLINE_FUNCTION real_t sqrtdet_SU2_like(const SUN<2> &a) {
  return Kokkos::sqrt(a.comp[0] * a.comp[0] + a.comp[1] * a.comp[1] +
                      a.comp[2] * a.comp[2] + a.comp[3] * a.comp[3]);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void extract_embedded_SU2(
    const SUNMatrix<Nc> &in, const index_t i, const index_t j, real_t &xi,
    SUN<2> &u) {
  const real_t a00 =
      matrix_element(in, i, i).real() + matrix_element(in, j, j).real();
  const real_t b00 =
      matrix_element(in, i, i).imag() - matrix_element(in, j, j).imag();
  const real_t a01 =
      matrix_element(in, i, j).real() - matrix_element(in, j, i).real();
  const real_t b01 =
      matrix_element(in, i, j).imag() + matrix_element(in, j, i).imag();

  const real_t p = Kokkos::sqrt(a00 * a00 + b00 * b00 + a01 * a01 + b01 * b01);
  xi = 0.5 * p;

  const real_t invp = p > 1.0e-13 ? 1.0 / p : 0.0;
  u = make_su2(a00 * invp, b01 * invp, a01 * invp, b00 * invp);
}

// Fold the Wilson action and linear breaking term into one local matrix.
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> effective_local_matrix(
    const SUN<Nc> &staple, const real_t beta, const real_t epsilon1) {
  // The heatbath kernels follow the Bonn convention: they update with a
  // weight proportional to exp[(1/Nc) ReTr(U M)], so the Wilson coupling
  // enters as M = beta * staple.
  SUN<Nc> out = staple * beta;
  if (epsilon1 != 0.0) {
    out += identitySUN<Nc>() * (static_cast<real_t>(Nc) * 0.5 * epsilon1);
  }
  return out;
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void heatbath_link(SUN<1> &link,
                                               const SUN<1> &matrix,
                                               RNG &generator) {
  const complex_t staple = matrix.comp;
  const real_t k = Kokkos::sqrt(staple.real() * staple.real() +
                                staple.imag() * staple.imag());
  if (k > 1.0e-13) {
    const complex_t aux = link.comp * staple;
    const real_t xold = Kokkos::atan2(aux.imag(), aux.real());
    const real_t y1 = generator.drand(0.0, 1.0);
    const real_t y2 = generator.drand(0.0, 1.0);

    const real_t xnew =
        Kokkos::sqrt(-2.0 / k *
                     Kokkos::log(1.0 - y1 * (1.0 - Kokkos::exp(-k *
                                     Kokkos::numbers::pi_v<real_t> *
                                     Kokkos::numbers::pi_v<real_t> / 2.0)))) *
        Kokkos::cos(2.0 * Kokkos::numbers::pi_v<real_t> * (y2 - 0.5));
    const real_t prob =
        Kokkos::exp(k * (Kokkos::cos(xnew) - xold * xold / 2.0 -
                         Kokkos::cos(xold) + xnew * xnew / 2.0));

    if (generator.drand(0.0, 1.0) < prob) {
      link = make_u1(complex_t(Kokkos::cos(xnew), Kokkos::sin(xnew)) *
                     Kokkos::conj(staple) / k);
    }
  } else {
    rand_matrix(link, generator);
  }
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void overrelax_link(SUN<1> &link,
                                                const SUN<1> &matrix,
                                                RNG &generator) {
  const complex_t staple = matrix.comp;
  const real_t k = Kokkos::sqrt(staple.real() * staple.real() +
                                staple.imag() * staple.imag());
  if (k > 1.0e-13) {
    const complex_t helper = staple / k;
    link = make_u1(Kokkos::conj(helper) * Kokkos::conj(link.comp) *
                   Kokkos::conj(helper));
  } else {
    rand_matrix(link, generator);
  }
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void heatbath_link(SUN<2> &link,
                                               const SUN<2> &matrix,
                                               RNG &generator) {
  SUN<2> matrix1 = matrix;
  const real_t p = sqrtdet_SU2_like(matrix1);
  if (p > 1.0e-13) {
    matrix1 *= 1.0 / p;
    const SUN<2> matrix2 = conj(matrix1);
    const real_t p0 = randheat_SU2(p, generator);
    rand_matrix_p0_su2(p0, link, generator);
    link *= matrix2;
  } else {
    rand_matrix(link, generator);
  }
}

template <class RNG>
KOKKOS_FORCEINLINE_FUNCTION void overrelax_link(SUN<2> &link,
                                                const SUN<2> &matrix,
                                                RNG &generator) {
  SUN<2> matrix1 = matrix;
  const real_t p = sqrtdet_SU2_like(matrix1);
  if (p > 1.0e-13) {
    matrix1 *= 1.0 / p;
    const SUN<2> matrix2 = conj(matrix1);
    matrix1 = matrix2 * conj(link);
    link = matrix1 * matrix2;
  } else {
    rand_matrix(link, generator);
  }
}

template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void heatbath_link(SUNMatrix<Nc> &link,
                                               const SUNMatrix<Nc> &matrix,
                                               RNG &generator) {
  SUNMatrix<Nc> aux = matrix * link;
  SUN<2> u, v, w;
  real_t xi = 0.0;

#pragma unroll
  for (index_t i = 0; i < Nc - 1; ++i) {
#pragma unroll
    for (index_t j = i + 1; j < Nc; ++j) {
      extract_embedded_SU2(aux, i, j, xi, u);
      xi *= 2.0 / static_cast<real_t>(Nc);

      if (xi > 1.0e-13) {
        const real_t p0 = randheat_SU2(xi, generator);
        w = conj(u);
        rand_matrix_p0_su2(p0, v, generator);
        w *= v;
      } else {
        rand_matrix(w, generator);
      }

      right_multiply_embedded_su2(link, i, j, w);
      right_multiply_embedded_su2(aux, i, j, w);
    }
  }
}

template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void overrelax_link(SUNMatrix<Nc> &link,
                                                const SUNMatrix<Nc> &matrix,
                                                RNG &generator) {
  SUNMatrix<Nc> aux = matrix * link;
  SUN<2> u, v;
  real_t xi = 0.0;

#pragma unroll
  for (index_t i = 0; i < Nc - 1; ++i) {
#pragma unroll
    for (index_t j = i + 1; j < Nc; ++j) {
      extract_embedded_SU2(aux, i, j, xi, u);

      if (xi > 1.0e-13) {
        v = conj(u);
        v *= conj(u);
      } else {
        rand_matrix(v, generator);
      }

      right_multiply_embedded_su2(link, i, j, v);
      right_multiply_embedded_su2(aux, i, j, v);
    }
  }
}

} // namespace klft
