#pragma once
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/gauge_observables.hpp"
#include "params/heatbath_params.hpp"

#include <Kokkos_Random.hpp>

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
  SUN<Nc> out = staple * (beta / static_cast<real_t>(Nc));
  if (epsilon1 != 0.0) {
    out += identitySUN<Nc>() * (0.5 * epsilon1);
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

// Update one odd/even sublattice with either heatbath or overrelaxation.
template <size_t rank, size_t Nc, class RNG, bool Overrelax>
struct HeatbathGaugeField {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  const RNG rng;
  const HeatbathParams params;
  const index_t dir;
  const Kokkos::Array<bool, rank> oddeven;

  HeatbathGaugeField(const GaugeFieldType &g_in, const HeatbathParams &params,
                     const index_t dir, const Kokkos::Array<bool, rank> &oddeven,
                     const RNG &rng)
      : g_in(g_in), rng(rng), params(params), dir(dir), oddeven(oddeven) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    auto generator = rng.get_state();
    const IndexArray<rank> site = index_odd_even<rank, size_t>(
        Kokkos::Array<size_t, rank>{Idcs...}, oddeven);

    auto link = g_in(site, dir);
    const auto matrix = effective_local_matrix<Nc>(g_in.staple(site, dir),
                                                   params.beta,
                                                   params.epsilon1);
    if constexpr (Overrelax) {
      overrelax_link(link, matrix, generator);
    } else {
      heatbath_link(link, matrix, generator);
    }
    restoreSUN(link);
    g_in(site, dir) = link;
    rng.free_state(generator);
  }
};

// Restore exact group projection after each full sweep.
template <size_t rank, size_t Nc> struct UnitarizeGaugeField {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType g_in;
  UnitarizeGaugeField(const GaugeFieldType &g_in) : g_in(g_in) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
      auto link = g_in(Idcs..., mu);
      restoreSUN(link);
      g_in(Idcs..., mu) = link;
    }
  }
};

// Perform one heatbath sweep plus the requested overrelaxation sweeps.
template <size_t rank, size_t Nc, class RNG>
void full_heatbath_sweep(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                         const HeatbathParams &params, const RNG &rng) {
  constexpr static const size_t Nd = rank;
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  IndexArray<rank> full_start;
  IndexArray<rank> full_end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = static_cast<index_t>(dimensions[i] / 2);
    full_start[i] = 0;
    full_end[i] = dimensions[i];
  }

  for (index_t dir = 0; dir < Nd; ++dir) {
    for (index_t i = 0; i < (1 << rank); ++i) {
      Kokkos::parallel_for(
          Policy<rank>(start, end),
          HeatbathGaugeField<rank, Nc, RNG, false>(g_in, params, dir,
                                                   oddeven_array<rank>(i), rng));
      Kokkos::fence();
    }
  }

  for (index_t dir = 0; dir < Nd; ++dir) {
    for (index_t j = 0; j < params.nOverrelax; ++j) {
      for (index_t i = 0; i < (1 << rank); ++i) {
        Kokkos::parallel_for(
            Policy<rank>(start, end),
            HeatbathGaugeField<rank, Nc, RNG, true>(
                g_in, params, dir, oddeven_array<rank>(i), rng));
        Kokkos::fence();
      }
    }
  }

  Kokkos::parallel_for(Policy<rank>(full_start, full_end),
                       UnitarizeGaugeField<rank, Nc>(g_in));
  Kokkos::fence();
}

// Execute the requested number of heatbath sweeps and measurements.
template <size_t rank, size_t Nc, class RNG, class GaugeFieldType>
int run_heatbath(GaugeFieldType &g_in, const HeatbathParams &heatbathParams,
                 GaugeObservableParams &gaugeObsParams, const RNG &rng) {
  const auto &dimensions = g_in.dimensions;
  validate_even_extents<rank>(dimensions, "Heatbath");
  gaugeObsParams.include_acceptance_rate = false;

  assert(heatbathParams.L0 == dimensions[0]);
  assert(heatbathParams.L1 == dimensions[1]);
  if constexpr (rank > 2) {
    assert(heatbathParams.L2 == dimensions[2]);
  }
  if constexpr (rank > 3) {
    assert(heatbathParams.L3 == dimensions[3]);
  }

  Kokkos::Timer timer;
  for (size_t step = 0; step < heatbathParams.nSweep; ++step) {
    timer.reset();
    full_heatbath_sweep<rank, Nc>(g_in, heatbathParams, rng);
    const real_t time = timer.seconds();

    measureGaugeObservables<rank, Nc>(g_in, heatbathParams, gaugeObsParams,
                                      step + 1, 0.0, time, rng);
  }
  return 0;
}

} // namespace klft
