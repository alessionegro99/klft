#pragma once

#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"

namespace klft {

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION IndexArray<rank>
clover_linear_to_site(size_t lin, const IndexArray<rank> &dims) {
  IndexArray<rank> site;
#pragma unroll
  for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
    const size_t extent = static_cast<size_t>(dims[d]);
    site[d] = static_cast<index_t>(lin % extent);
    lin /= extent;
  }
  return site;
}

template <size_t rank>
inline size_t clover_site_count(const IndexArray<rank> &dims) {
  size_t nsites = 1;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    nsites *= static_cast<size_t>(dims[d]);
  }
  return nsites;
}

template <size_t rank, size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> clover_q_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V,
    const IndexArray<rank> &site, const index_t mu, const index_t nu) {
  const auto x_p_mu = shift_index_plus<rank>(site, mu, 1, V.dimensions);
  const auto x_p_nu = shift_index_plus<rank>(site, nu, 1, V.dimensions);
  const auto x_m_mu = shift_index_minus<rank>(site, mu, 1, V.dimensions);
  const auto x_m_nu = shift_index_minus<rank>(site, nu, 1, V.dimensions);
  const auto x_m_mu_p_nu =
      shift_index_plus<rank>(x_m_mu, nu, 1, V.dimensions);
  const auto x_p_mu_m_nu =
      shift_index_minus<rank>(x_p_mu, nu, 1, V.dimensions);
  const auto x_m_mu_m_nu =
      shift_index_minus<rank>(x_m_mu, nu, 1, V.dimensions);

  const SUN<Nc> p1 = V(site, mu) * V(x_p_mu, nu) * conj(V(x_p_nu, mu)) *
                     conj(V(site, nu));
  const SUN<Nc> p2 = V(site, nu) * conj(V(x_m_mu_p_nu, mu)) *
                     conj(V(x_m_mu, nu)) * V(x_m_mu, mu);
  const SUN<Nc> p3 = conj(V(x_m_mu, mu)) * conj(V(x_m_mu_m_nu, nu)) *
                     V(x_m_mu_m_nu, mu) * V(x_m_nu, nu);
  const SUN<Nc> p4 = conj(V(x_m_nu, nu)) * V(x_m_nu, mu) *
                     V(x_p_mu_m_nu, nu) * conj(V(site, mu));

  return p1 + p2 + p3 + p4;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> clover_ta(const SUN<Nc> &q) {
  if constexpr (Nc == 1) {
    return make_u1(complex_t(0.0, q.comp.imag()));
  } else if constexpr (Nc == 2) {
    return make_su2(0.0, q.comp[1], q.comp[2], q.comp[3]);
  } else {
    SUN<Nc> ta = (q - conj(q)) * 0.5;
    const complex_t tr = trace(ta) / static_cast<real_t>(Nc);
#pragma unroll
    for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
      matrix_ref(ta, i, i) -= tr;
    }
    return ta;
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t clover_energy_from_q(const SUN<Nc> &q) {
  const SUN<Nc> fhat = clover_ta<Nc>(q) * 0.25;
  return -Kokkos::real(trace(fhat * fhat));
}

template <size_t rank, size_t Nc> struct CloverEnergyDensity {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType V;

  explicit CloverEnergyDensity(const GaugeFieldType &V) : V(V) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin,
                                              real_t &sum) const {
    const auto site = clover_linear_to_site<rank>(lin, V.dimensions);
    real_t local = 0.0;
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
#pragma unroll
      for (index_t nu = mu + 1; nu < static_cast<index_t>(rank); ++nu) {
        const SUN<Nc> q = clover_q_mu_nu<rank, Nc>(V, site, mu, nu);
        local += clover_energy_from_q<Nc>(q);
      }
    }
    sum += local;
  }
};

template <size_t rank, size_t Nc>
real_t measure_clover_energy_density(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V) {
  real_t total = 0.0;
  const size_t nsites = clover_site_count<rank>(V.dimensions);
  Kokkos::parallel_reduce("CloverEnergyDensity",
                          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                              0, nsites),
                          CloverEnergyDensity<rank, Nc>(V), total);
  Kokkos::fence();
  return total / static_cast<real_t>(nsites);
}

} // namespace klft
