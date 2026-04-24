#pragma once
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/multihit_links.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace klft {

template <size_t rank, size_t Nc, class RNG> struct WLoop_munu_metropolis {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  const GaugeFieldType g_in;
  FieldType Wmunu_per_site;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const index_t multihit;
  const real_t beta;
  const real_t delta;
  const real_t epsilon1;
  const real_t epsilon2;
  const RNG rng;
  const IndexArray<rank> dimensions;

  WLoop_munu_metropolis(const GaugeFieldType &g_in, FieldType &Wmunu_per_site,
                        const index_t mu, const index_t nu, const index_t Lmu,
                        const index_t Lnu, const index_t multihit,
                        const real_t beta, const real_t delta,
                        const real_t epsilon1, const real_t epsilon2,
                        const RNG &rng, const IndexArray<rank> &dimensions)
      : g_in(g_in), Wmunu_per_site(Wmunu_per_site), mu(mu), nu(nu), Lmu(Lmu),
        Lnu(Lnu), multihit(multihit), beta(beta), delta(delta),
        epsilon1(epsilon1), epsilon2(epsilon2), rng(rng),
        dimensions(dimensions) {}

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  multihit_link(const Kokkos::Array<index_t, rank> &site, const index_t dir,
                Generator &generator) const {
    return multihit_link_metropolis<Nc>(g_in(site, dir), g_in.staple(site, dir),
                                        multihit, beta, delta, epsilon1,
                                        epsilon2, generator);
  }

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION complex_t loop_at_site(
      const Kokkos::Array<index_t, rank> &origin, Generator &generator) const {
    Kokkos::Array<index_t, rank> site = origin;
    SUN<Nc> loop = identitySUN<Nc>();

    for (index_t i = 0; i < Lmu; ++i) {
      loop *= (multihit > 1 && i > 0) ? multihit_link(site, mu, generator)
                                      : g_in(site, mu);
      site = shift_index_plus<rank>(site, mu, 1, dimensions);
    }

    for (index_t i = 0; i < Lnu; ++i) {
      loop *= (multihit > 1 && i > 0) ? multihit_link(site, nu, generator)
                                      : g_in(site, nu);
      site = shift_index_plus<rank>(site, nu, 1, dimensions);
    }

    for (index_t i = 0; i < Lmu; ++i) {
      site = shift_index_minus<rank>(site, mu, 1, dimensions);
      loop *= conj((multihit > 1 && i > 0) ? multihit_link(site, mu, generator)
                                           : g_in(site, mu));
    }

    for (index_t i = 0; i < Lnu; ++i) {
      site = shift_index_minus<rank>(site, nu, 1, dimensions);
      loop *= conj((multihit > 1 && i > 0) ? multihit_link(site, nu, generator)
                                           : g_in(site, nu));
    }

    return trace(loop);
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const Kokkos::Array<index_t, rank> site{static_cast<index_t>(Idcs)...};
    auto generator = rng.get_state();
    Wmunu_per_site(Idcs...) = loop_at_site(site, generator);
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc, class RNG> struct WLoop_munu_heatbath {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  const GaugeFieldType g_in;
  FieldType Wmunu_per_site;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const index_t multihit;
  const index_t nOverrelax;
  const real_t beta;
  const real_t epsilon1;
  const RNG rng;
  const IndexArray<rank> dimensions;

  WLoop_munu_heatbath(const GaugeFieldType &g_in, FieldType &Wmunu_per_site,
                      const index_t mu, const index_t nu, const index_t Lmu,
                      const index_t Lnu, const index_t multihit,
                      const index_t nOverrelax, const real_t beta,
                      const real_t epsilon1, const RNG &rng,
                      const IndexArray<rank> &dimensions)
      : g_in(g_in), Wmunu_per_site(Wmunu_per_site), mu(mu), nu(nu), Lmu(Lmu),
        Lnu(Lnu), multihit(multihit), nOverrelax(nOverrelax), beta(beta),
        epsilon1(epsilon1), rng(rng), dimensions(dimensions) {}

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  multihit_link(const Kokkos::Array<index_t, rank> &site, const index_t dir,
                Generator &generator) const {
    return multihit_link_heatbath<Nc>(g_in(site, dir), g_in.staple(site, dir),
                                      multihit, nOverrelax, beta, epsilon1,
                                      generator);
  }

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION complex_t loop_at_site(
      const Kokkos::Array<index_t, rank> &origin, Generator &generator) const {
    Kokkos::Array<index_t, rank> site = origin;
    SUN<Nc> loop = identitySUN<Nc>();

    for (index_t i = 0; i < Lmu; ++i) {
      loop *= (multihit > 1 && i > 0) ? multihit_link(site, mu, generator)
                                      : g_in(site, mu);
      site = shift_index_plus<rank>(site, mu, 1, dimensions);
    }

    for (index_t i = 0; i < Lnu; ++i) {
      loop *= (multihit > 1 && i > 0) ? multihit_link(site, nu, generator)
                                      : g_in(site, nu);
      site = shift_index_plus<rank>(site, nu, 1, dimensions);
    }

    for (index_t i = 0; i < Lmu; ++i) {
      site = shift_index_minus<rank>(site, mu, 1, dimensions);
      loop *= conj((multihit > 1 && i > 0) ? multihit_link(site, mu, generator)
                                           : g_in(site, mu));
    }

    for (index_t i = 0; i < Lnu; ++i) {
      site = shift_index_minus<rank>(site, nu, 1, dimensions);
      loop *= conj((multihit > 1 && i > 0) ? multihit_link(site, nu, generator)
                                           : g_in(site, nu));
    }

    return trace(loop);
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const Kokkos::Array<index_t, rank> site{static_cast<index_t>(Idcs)...};
    auto generator = rng.get_state();
    Wmunu_per_site(Idcs...) = loop_at_site(site, generator);
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION complex_t
WilsonLoopRawAtSite(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                    const Kokkos::Array<index_t, rank> &origin,
                    const index_t mu, const index_t nu, const index_t Lmu,
                    const index_t Lnu, const IndexArray<rank> &dimensions) {
  Kokkos::Array<index_t, rank> site = origin;
  SUN<Nc> loop = identitySUN<Nc>();

  for (index_t i = 0; i < Lmu; ++i) {
    loop *= g_in(site, mu);
    site = shift_index_plus<rank>(site, mu, 1, dimensions);
  }
  for (index_t i = 0; i < Lnu; ++i) {
    loop *= g_in(site, nu);
    site = shift_index_plus<rank>(site, nu, 1, dimensions);
  }
  for (index_t i = 0; i < Lmu; ++i) {
    site = shift_index_minus<rank>(site, mu, 1, dimensions);
    loop *= conj(g_in(site, mu));
  }
  for (index_t i = 0; i < Lnu; ++i) {
    site = shift_index_minus<rank>(site, nu, 1, dimensions);
    loop *= conj(g_in(site, nu));
  }

  return trace(loop);
}

template <size_t rank, size_t Nc> struct WLoop_munu_raw {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  const GaugeFieldType g_in;
  FieldType Wmunu_per_site;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const IndexArray<rank> dimensions;

  WLoop_munu_raw(const GaugeFieldType &g_in, FieldType &Wmunu_per_site,
                 const index_t mu, const index_t nu, const index_t Lmu,
                 const index_t Lnu, const IndexArray<rank> &dimensions)
      : g_in(g_in), Wmunu_per_site(Wmunu_per_site), mu(mu), nu(nu), Lmu(Lmu),
        Lnu(Lnu), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const Kokkos::Array<index_t, rank> site{static_cast<index_t>(Idcs)...};
    Wmunu_per_site(Idcs...) =
        WilsonLoopRawAtSite<rank, Nc>(g_in, site, mu, nu, Lmu, Lnu, dimensions);
  }
};

template <size_t rank, size_t Nc, class RNG>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>> &Lmu_nu_pairs,
    std::vector<Kokkos::Array<real_t, 5>> &Wmunu_vals, const index_t multihit,
    const real_t beta, const real_t delta, const real_t epsilon1,
    const real_t epsilon2, const RNG &rng, const bool normalize = true) {
  constexpr static const size_t Nd = rank;
  complex_t Wmunu;

  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType Wmunu_per_site(end, complex_t(0.0, 0.0));

  for (const auto &Lmu_nu : Lmu_nu_pairs) {
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];

    if (multihit > 1) {
      Kokkos::parallel_for(Policy<rank>(start, end),
                           WLoop_munu_metropolis<rank, Nc, RNG>(
                               g_in, Wmunu_per_site, mu, nu, Lmu, Lnu, multihit,
                               beta, delta, epsilon1, epsilon2, rng, end));
    } else {
      Kokkos::parallel_for(Policy<rank>(start, end),
                           WLoop_munu_raw<rank, Nc>(g_in, Wmunu_per_site, mu,
                                                    nu, Lmu, Lnu, end));
    }
    Kokkos::fence();

    Wmunu = Wmunu_per_site.sum();
    Kokkos::fence();

    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < rank; ++i) {
        Wmunu /= static_cast<real_t>(end[i]);
      }
      Wmunu /= static_cast<real_t>(Nc);
    }

    Wmunu_vals.push_back(Kokkos::Array<real_t, 5>{
        static_cast<real_t>(mu), static_cast<real_t>(nu),
        static_cast<real_t>(Lmu), static_cast<real_t>(Lnu), Wmunu.real()});
  }
}

template <size_t rank, size_t Nc, class RNG>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>> &Lmu_nu_pairs,
    std::vector<Kokkos::Array<real_t, 5>> &Wmunu_vals, const index_t multihit,
    const MetropolisParams &updateParams, const RNG &rng,
    const bool normalize = true) {
  WilsonLoop_mu_nu<rank, Nc>(g_in, mu, nu, Lmu_nu_pairs, Wmunu_vals, multihit,
                             updateParams.beta, updateParams.delta,
                             updateParams.epsilon1, updateParams.epsilon2, rng,
                             normalize);
}

template <size_t rank, size_t Nc, class RNG>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>> &Lmu_nu_pairs,
    std::vector<Kokkos::Array<real_t, 5>> &Wmunu_vals, const index_t multihit,
    const HeatbathParams &updateParams, const RNG &rng,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;
  complex_t Wmunu;

  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType Wmunu_per_site(end, complex_t(0.0, 0.0));

  for (const auto &Lmu_nu : Lmu_nu_pairs) {
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];

    if (multihit > 1) {
      Kokkos::parallel_for(Policy<rank>(start, end),
                           WLoop_munu_heatbath<rank, Nc, RNG>(
                               g_in, Wmunu_per_site, mu, nu, Lmu, Lnu, multihit,
                               updateParams.nOverrelax, updateParams.beta,
                               updateParams.epsilon1, rng, end));
    } else {
      Kokkos::parallel_for(Policy<rank>(start, end),
                           WLoop_munu_raw<rank, Nc>(g_in, Wmunu_per_site, mu,
                                                    nu, Lmu, Lnu, end));
    }
    Kokkos::fence();

    Wmunu = Wmunu_per_site.sum();
    Kokkos::fence();

    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < rank; ++i) {
        Wmunu /= static_cast<real_t>(end[i]);
      }
      Wmunu /= static_cast<real_t>(Nc);
    }

    Wmunu_vals.push_back(Kokkos::Array<real_t, 5>{
        static_cast<real_t>(mu), static_cast<real_t>(nu),
        static_cast<real_t>(Lmu), static_cast<real_t>(Lnu), Wmunu.real()});
  }
}

template <size_t rank, size_t Nc, class RNG>
void WilsonLoop_temporal(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const std::vector<Kokkos::Array<index_t, 2>> &L_T_pairs,
    std::vector<Kokkos::Array<real_t, 3>> &Wtemporal_vals,
    const index_t multihit, const real_t beta, const real_t delta,
    const real_t epsilon1, const real_t epsilon2, const RNG &rng,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;

  std::vector<Kokkos::Array<real_t, 5>> Wmunu_vals;
  WilsonLoop_mu_nu<rank, Nc>(g_in, 0, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                             beta, delta, epsilon1, epsilon2, rng, normalize);

  for (const auto &Wmunu : Wmunu_vals) {
    Wtemporal_vals.push_back(
        Kokkos::Array<real_t, 3>{Wmunu[2], Wmunu[3], Wmunu[4]});
  }

  if constexpr (Nd > 2) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 1, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                               beta, delta, epsilon1, epsilon2, rng, normalize);
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }
  if constexpr (Nd > 3) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 2, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                               beta, delta, epsilon1, epsilon2, rng, normalize);
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }

  for (index_t i = 0; i < Wtemporal_vals.size(); ++i) {
    Wtemporal_vals[i][2] /= static_cast<real_t>(Nd - 1);
  }
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void WilsonLoop_temporal(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const std::vector<Kokkos::Array<index_t, 2>> &L_T_pairs,
    std::vector<Kokkos::Array<real_t, 3>> &Wtemporal_vals,
    const index_t multihit, const UpdateParams &updateParams, const RNG &rng,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;

  std::vector<Kokkos::Array<real_t, 5>> Wmunu_vals;
  WilsonLoop_mu_nu<rank, Nc>(g_in, 0, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                             updateParams, rng, normalize);

  for (const auto &Wmunu : Wmunu_vals) {
    Wtemporal_vals.push_back(
        Kokkos::Array<real_t, 3>{Wmunu[2], Wmunu[3], Wmunu[4]});
  }

  if constexpr (Nd > 2) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 1, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                               updateParams, rng, normalize);
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }
  if constexpr (Nd > 3) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 2, Nd - 1, L_T_pairs, Wmunu_vals, multihit,
                               updateParams, rng, normalize);
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }

  for (index_t i = 0; i < Wtemporal_vals.size(); ++i) {
    Wtemporal_vals[i][2] /= static_cast<real_t>(Nd - 1);
  }
}

} // namespace klft
