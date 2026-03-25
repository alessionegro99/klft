#pragma once

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace klft {

// plaquette (real trace of)
template <size_t rank, size_t Nc> struct GaugePlaq {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site;
  const IndexArray<rank> dimensions;

  GaugePlaq(const GaugeFieldType &g_in, FieldType &plaq_per_site,
            const IndexArray<rank> &dimensions)
      : g_in(g_in), plaq_per_site(plaq_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    SUN<Nc> lmu, lnu;
    complex_t tmunu(0.0, 0.0);

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          lmu =
              g_in(Idcs..., mu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions),
                   nu);

          lnu =
              g_in(Idcs..., nu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, nu, 1, dimensions),
                   mu);

#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < Nc; ++c2) {
              tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
            }
          }
        }
      }
    }
    plaq_per_site(Idcs...) = tmunu;
  }
};

template <size_t rank, size_t Nc>
real_t GaugePlaquette(const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                      const bool normalize = true) {
  constexpr static const size_t Nd = rank;
  complex_t plaq = 0.0;

  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site(end, complex_t(0.0, 0.0));

  GaugePlaq<rank, Nc> gaugePlaquette(g_in, plaq_per_site, end);

  Kokkos::parallel_for("GaugePlaquette_GaugeField", Policy<rank>(start, end),
                       gaugePlaquette);
  Kokkos::fence();

  plaq = plaq_per_site.sum();
  Kokkos::fence();

  if (normalize) {
    real_t norm = 1.0;
    for (index_t i = 0; i < rank; ++i) {
      norm *= static_cast<real_t>(end[i]);
    }
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}

// \sum_{x,mu} Re Tr[U_mu(x)]
KOKKOS_INLINE_FUNCTION
real_t real_part(const real_t &x) { return x; }

template <class T>
KOKKOS_INLINE_FUNCTION real_t real_part(const std::complex<T> &z) {
  return static_cast<real_t>(z.real());
}

template <class T>
KOKKOS_INLINE_FUNCTION real_t real_part(const Kokkos::complex<T> &z) {
  return static_cast<real_t>(z.real());
}

template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank>
linear_to_multi(size_t lin, const IndexArray<rank> &dims) {
  IndexArray<rank> idx;
#pragma unroll
  for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
    const size_t d = static_cast<size_t>(dims[r]);
    idx[r] = static_cast<index_t>(lin % d);
    lin /= d;
  }
  return idx;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<2>
get_dims(const typename DeviceGaugeFieldType<2, Nc>::type &g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<3>
get_dims(const typename DeviceGaugeFieldType<3, Nc>::type &g) {
  return g.dimensions;
}

template <size_t Nc>
KOKKOS_INLINE_FUNCTION IndexArray<4>
get_dims(const typename DeviceGaugeFieldType<4, Nc>::type &g) {
  return g.dimensions;
}

template <size_t rank, size_t Nc>
KOKKOS_INLINE_FUNCTION real_t
Retrace_at(const typename DeviceGaugeFieldType<rank, Nc>::type &g,
           const IndexArray<rank> &site, const index_t mu) {
  const auto &U = g(site, mu);
  real_t trr = 0.0;
#pragma unroll
  for (index_t a = 0; a < static_cast<index_t>(Nc); ++a) {
    trr += real_part(U[a][a]);
  }
  return trr / static_cast<real_t>(Nc);
}

template <size_t rank, size_t Nc>
real_t
Retrace_links_avg(const typename DeviceGaugeFieldType<rank, Nc>::type &g) {
  static_assert(rank == 2 || rank == 3 || rank == 4,
                "Retrace_links_avg: rank must be 2, 3, or 4.");

  using Exec = Kokkos::DefaultExecutionSpace;

  const auto dims = get_dims<Nc>(g);

  size_t nSites = 1;
#pragma unroll
  for (size_t r = 0; r < rank; ++r)
    nSites *= static_cast<size_t>(dims[r]);

  const size_t nLinks = nSites * static_cast<size_t>(rank);

  real_t total = 0.0;

  Kokkos::parallel_reduce(
      "Retrace_links_avg", Kokkos::RangePolicy<Exec>(0, nLinks),
      KOKKOS_LAMBDA(const size_t i, real_t &lsum) {
        const index_t mu = static_cast<index_t>(i % rank);
        const size_t s = i / rank;
        const auto site = linear_to_multi<rank>(s, dims);
        lsum += Retrace_at<rank, Nc>(g, site, mu);
      },
      total);

  return total / static_cast<real_t>(nLinks);
}

// energies
template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION real_t
NumPlaquettes(const IndexArray<rank> &dimensions) {
  real_t nsites = 1.0;
#pragma unroll
  for (index_t d = 0; d < rank; ++d) {
    nsites *= static_cast<real_t>(dimensions[d]);
  }
  return nsites * static_cast<real_t>(rank * (rank - 1) / 2);
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION real_t ReducedWilsonActionFromAvgPlaquette(
    const real_t avg_plaq, const IndexArray<rank> &dimensions) {
  return NumPlaquettes(dimensions) * (1.0 - avg_plaq);
}

template <size_t rank>
inline void ValidateDyadicChildOffset(const IndexArray<rank> &child_offset) {
  for (index_t d = 0; d < rank; ++d) {
    if (child_offset[d] != 0 && child_offset[d] != 1) {
      throw std::runtime_error(
          "child_offset entries must be 0 or 1 for one-level dyadic blocking.");
    }
  }
}

template <size_t rank, size_t Nc> struct BlockedGaugePlaqOneLevel {
  constexpr static const size_t Nd = rank;

  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  const GaugeFieldType g_in;
  FieldType plaq_per_site;
  const IndexArray<rank> fine_dimensions;
  const IndexArray<rank> child_offset;

  BlockedGaugePlaqOneLevel(const GaugeFieldType &g_in, FieldType &plaq_per_site,
                           const IndexArray<rank> &fine_dimensions,
                           const IndexArray<rank> &child_offset)
      : g_in(g_in), plaq_per_site(plaq_per_site),
        fine_dimensions(fine_dimensions), child_offset(child_offset) {}

  KOKKOS_FORCEINLINE_FUNCTION
  SUN<Nc> blocked_link(const IndexArray<rank> &x, const index_t mu) const {
    auto x_plus_mu = shift_index_plus<rank, index_t>(x, mu, 1, fine_dimensions);
    return g_in(x, mu) * g_in(x_plus_mu, mu);
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    SUN<Nc> lmu, lnu;
    complex_t tmunu(0.0, 0.0);

    const index_t coarse_idx[rank] = {static_cast<index_t>(Idcs)...};

    IndexArray<rank> x;
#pragma unroll
    for (index_t d = 0; d < rank; ++d) {
      x[d] = 2 * coarse_idx[d] + child_offset[d];
    }

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          auto x_plus_2mu =
              shift_index_plus<rank, index_t>(x, mu, 2, fine_dimensions);
          auto x_plus_2nu =
              shift_index_plus<rank, index_t>(x, nu, 2, fine_dimensions);

          lmu = blocked_link(x, mu) * blocked_link(x_plus_2mu, nu);
          lnu = blocked_link(x, nu) * blocked_link(x_plus_2nu, mu);

#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < Nc; ++c2) {
              tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
            }
          }
        }
      }
    }

    plaq_per_site(Idcs...) = tmunu;
  }
};

template <size_t rank, size_t Nc>
real_t BlockedGaugePlaquetteOneLevel(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const IndexArray<rank> &child_offset, const bool normalize = true) {

  constexpr static const size_t Nd = rank;

  ValidateDyadicChildOffset(child_offset);

  const auto &fine_dims = g_in.field.layout().dimension;

  IndexArray<rank> coarse_start;
  IndexArray<rank> coarse_end;
  IndexArray<rank> fine_dimensions;

  for (index_t d = 0; d < rank; ++d) {
    fine_dimensions[d] = fine_dims[d];
    coarse_start[d] = 0;

    if (fine_dims[d] % 2 != 0) {
      throw std::runtime_error(
          "BlockedGaugePlaquetteOneLevel requires even lattice extents.");
    }
    coarse_end[d] = fine_dims[d] / 2;
  }

  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site(coarse_end, complex_t(0.0, 0.0));

  BlockedGaugePlaqOneLevel<rank, Nc> blockedPlaq(g_in, plaq_per_site,
                                                 fine_dimensions, child_offset);

  Kokkos::parallel_for("BlockedGaugePlaquetteOneLevel_GaugeField",
                       Policy<rank>(coarse_start, coarse_end), blockedPlaq);
  Kokkos::fence();

  complex_t plaq = plaq_per_site.sum();
  Kokkos::fence();

  if (normalize) {
    real_t norm = 1.0;
#pragma unroll
    for (index_t d = 0; d < rank; ++d) {
      norm *= static_cast<real_t>(coarse_end[d]);
    }
    norm *= static_cast<real_t>((Nd * (Nd - 1) / 2) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}

template <size_t rank> struct NestedWilsonActionResult {
  real_t plaq_V;
  real_t plaq_child;
  real_t E_V;
  real_t E_child;
};

template <size_t rank, size_t Nc>
NestedWilsonActionResult<rank> MeasureNestedWilsonActionsOneLevel(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const IndexArray<rank> &child_offset) {

  ValidateDyadicChildOffset(child_offset);

  NestedWilsonActionResult<rank> out{};

  const auto &dims_raw = g_in.field.layout().dimension;
  IndexArray<rank> dims;
  IndexArray<rank> child_dims;

  for (index_t d = 0; d < rank; ++d) {
    dims[d] = dims_raw[d];
    if (dims[d] % 2 != 0) {
      throw std::runtime_error(
          "MeasureNestedWilsonActionsOneLevel requires even lattice extents.");
    }
    child_dims[d] = dims[d] / 2;
  }

  out.plaq_V = GaugePlaquette<rank, Nc>(g_in, true);
  out.plaq_child =
      BlockedGaugePlaquetteOneLevel<rank, Nc>(g_in, child_offset, true);

  out.E_V = ReducedWilsonActionFromAvgPlaquette(out.plaq_V, dims);
  out.E_child = ReducedWilsonActionFromAvgPlaquette(out.plaq_child, child_dims);

  return out;
}

// WilsonLoop
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

    Kokkos::parallel_for("WilsonLoop_GaugeField_WLmunu",
                         Policy<rank>(start, end), wlmunu);
    Kokkos::fence();

    Kokkos::parallel_for("WilsonLoop_GaugeField_WLoop_munu",
                         Policy<rank>(start, end),
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

template <size_t rank, size_t Nc>
void WilsonLoop_temporal(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const std::vector<Kokkos::Array<index_t, 2>> &L_T_pairs,
    std::vector<WilsonLoopTemporalMeasurement> &Wtemporal_vals,
    const bool normalize = true) {
  constexpr static const size_t Nd = rank;

  std::vector<WilsonLoopMuNuMeasurement> Wmunu_vals;

  WilsonLoop_mu_nu<rank, Nc>(g_in, 0, static_cast<index_t>(Nd - 1), L_T_pairs,
                             Wmunu_vals, normalize);

  for (const auto &Wmunu : Wmunu_vals) {
    Wtemporal_vals.push_back(
        WilsonLoopTemporalMeasurement{Wmunu.Lmu, Wmunu.Lnu, Wmunu.value});
  }

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
