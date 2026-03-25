#pragma once
#include "FieldTypeHelper.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"
#include "Tuner.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace klft {

struct WilsonLoopTemporalMeasurement {
  index_t L;
  index_t T;
  real_t value;
};

struct WilsonLoopMuNuMeasurement {
  index_t mu;
  index_t nu;
  index_t Lmu;
  index_t Lnu;
  real_t value;
};

// build Wilson lines along the mu and nu directions
template <size_t rank, size_t Nc> struct WLmunu {
  constexpr static const size_t Nd = rank;

  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;

  const GaugeFieldType g_in;
  SUNFieldType WLmu, WLnu;

  const index_t mu, nu;

  index_t Lmu_old, Lnu_old;
  index_t Lmu, Lnu;

  const IndexArray<rank> dimensions;

  WLmunu(const GaugeFieldType &g_in, const index_t mu, const index_t nu,
         const index_t Lmu, const index_t Lnu, SUNFieldType &WLmu,
         SUNFieldType &WLnu, const IndexArray<rank> &dimensions)
      : g_in(g_in), WLmu(WLmu), WLnu(WLnu), mu(mu), nu(nu), Lmu_old(0),
        Lnu_old(0), Lmu(Lmu), Lnu(Lnu), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const Kokkos::Array<index_t, rank> site{static_cast<index_t>(Idcs)...};

    SUN<Nc> lmu = WLmu(Idcs...);
    SUN<Nc> lnu = WLnu(Idcs...);

    for (index_t i = Lmu_old; i < Lmu; ++i) {
      lmu *= g_in(shift_index_plus<rank, index_t>(site, mu, i, dimensions), mu);
    }

    for (index_t i = Lnu_old; i < Lnu; ++i) {
      lnu *= g_in(shift_index_plus<rank, index_t>(site, nu, i, dimensions), nu);
    }

    WLmu(Idcs...) = lmu;
    WLnu(Idcs...) = lnu;
  }

  void update_Lmu_Lnu(const index_t Lmu_new, const index_t Lnu_new) {
    Lmu_old = Lmu;
    Lnu_old = Lnu;
    Lmu = Lmu_new;
    Lnu = Lnu_new;
  }

  void reset_Lmu_Lnu() {
    Lmu_old = 0;
    Lnu_old = 0;
    Lmu = 0;
    Lnu = 0;

    Kokkos::deep_copy(WLmu.field,
                      SUNFieldType(dimensions, identitySUN<Nc>()).field);
    Kokkos::deep_copy(WLnu.field,
                      SUNFieldType(dimensions, identitySUN<Nc>()).field);
  }
};

// build the Wilson loop out of WLmu and WLnu
template <size_t rank, size_t Nc> struct WLoop_munu {
  constexpr static const size_t Nd = rank;

  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  const SUNFieldType WLmu, WLnu;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  FieldType Wmunu_per_site;
  const IndexArray<rank> dimensions;

  WLoop_munu(const SUNFieldType &WLmu, const SUNFieldType &WLnu,
             const index_t mu, const index_t nu, const index_t Lmu,
             const index_t Lnu, FieldType &Wmunu_per_site,
             const IndexArray<rank> &dimensions)
      : WLmu(WLmu), WLnu(WLnu), mu(mu), nu(nu), Lmu(Lmu), Lnu(Lnu),
        Wmunu_per_site(Wmunu_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const Kokkos::Array<index_t, rank> site{static_cast<index_t>(Idcs)...};

    SUN<Nc> lmu =
        WLmu(Idcs...) *
        WLnu(shift_index_plus<rank, index_t>(site, mu, Lmu, dimensions));

    SUN<Nc> lnu =
        WLnu(Idcs...) *
        WLmu(shift_index_plus<rank, index_t>(site, nu, Lnu, dimensions));

    complex_t tmunu(0.0, 0.0);
#pragma unroll
    for (index_t c1 = 0; c1 < static_cast<index_t>(Nc); ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < static_cast<index_t>(Nc); ++c2) {
        tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
      }
    }

    Wmunu_per_site(Idcs...) = tmunu;
  }
};

// calculate Wilson loops in the mu-nu plane for a set of (Lmu, Lnu) pairs
template <size_t rank, size_t Nc>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>> &Lmu_nu_pairs,
    std::vector<WilsonLoopMuNuMeasurement> &Wmunu_vals,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;

  complex_t Wmunu(0.0, 0.0);

  const auto &dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < static_cast<index_t>(Nd); ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;
  SUNFieldType WLmu(end, identitySUN<Nc>());
  SUNFieldType WLnu(end, identitySUN<Nc>());

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType Wmunu_per_site(end, complex_t(0.0, 0.0));

  WLmunu<rank, Nc> wlmunu(g_in, mu, nu, 0, 0, WLmu, WLnu, end);

  for (const auto &Lmu_nu : Lmu_nu_pairs) {
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];

    if (Lmu < wlmunu.Lmu || Lnu < wlmunu.Lnu) {
      wlmunu.reset_Lmu_Lnu();
    }

    assert(Lmu >= wlmunu.Lmu);
    assert(Lnu >= wlmunu.Lnu);

    wlmunu.update_Lmu_Lnu(Lmu, Lnu);

    tune_and_launch_for<rank>("WilsonLoop_GaugeField_WLmunu", start, end,
                              wlmunu);
    Kokkos::fence();

    tune_and_launch_for<rank>("WilsonLoop_GaugeField_WLoop_munu", start, end,
                              WLoop_munu<rank, Nc>(WLmu, WLnu, mu, nu, Lmu, Lnu,
                                                   Wmunu_per_site, end));
    Kokkos::fence();

    Wmunu = Wmunu_per_site.sum();
    Kokkos::fence();

    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < static_cast<index_t>(rank); ++i) {
        Wmunu /= static_cast<real_t>(end[i]);
      }
      Wmunu /= static_cast<real_t>(Nc);
    }

    Wmunu_vals.push_back(
        WilsonLoopMuNuMeasurement{mu, nu, Lmu, Lnu, Wmunu.real()});
  }
}

// calculate temporal Wilson loops and average over spatial directions
template <size_t rank, size_t Nc>
void WilsonLoop_temporal(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const std::vector<Kokkos::Array<index_t, 2>> &L_T_pairs,
    std::vector<WilsonLoopTemporalMeasurement> &Wtemporal_vals,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;

  std::vector<WilsonLoopMuNuMeasurement> Wmunu_vals;

  // first spatial direction
  WilsonLoop_mu_nu<rank, Nc>(g_in, 0, static_cast<index_t>(Nd - 1), L_T_pairs,
                             Wmunu_vals, normalize);

  for (const auto &Wmunu : Wmunu_vals) {
    Wtemporal_vals.push_back(
        WilsonLoopTemporalMeasurement{Wmunu.Lmu, Wmunu.Lnu, Wmunu.value});
  }

  // second spatial direction
  if constexpr (Nd > 2) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 1, static_cast<index_t>(Nd - 1), L_T_pairs,
                               Wmunu_vals, normalize);

    for (index_t i = 0; i < static_cast<index_t>(Wmunu_vals.size()); ++i) {
      if (Wmunu_vals[i].Lmu == Wtemporal_vals[i].L &&
          Wmunu_vals[i].Lnu == Wtemporal_vals[i].T) {
        Wtemporal_vals[i].value += Wmunu_vals[i].value;
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }

  // third spatial direction
  if constexpr (Nd > 3) {
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc>(g_in, 2, static_cast<index_t>(Nd - 1), L_T_pairs,
                               Wmunu_vals, normalize);

    for (index_t i = 0; i < static_cast<index_t>(Wmunu_vals.size()); ++i) {
      if (Wmunu_vals[i].Lmu == Wtemporal_vals[i].L &&
          Wmunu_vals[i].Lnu == Wtemporal_vals[i].T) {
        Wtemporal_vals[i].value += Wmunu_vals[i].value;
      } else {
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }

  for (index_t i = 0; i < static_cast<index_t>(Wtemporal_vals.size()); ++i) {
    Wtemporal_vals[i].value /= static_cast<real_t>(Nd - 1);
  }
}

} // namespace klft
