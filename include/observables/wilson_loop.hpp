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

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
wilson_linear_to_site(size_t lin, const IndexArray<rank> &dimensions) {
  Kokkos::Array<index_t, rank> site;
  for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
    const size_t extent = static_cast<size_t>(dimensions[d]);
    site[d] = static_cast<index_t>(lin % extent);
    lin /= extent;
  }
  return site;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION size_t
wilson_site_to_linear(const Kokkos::Array<index_t, rank> &site,
                      const IndexArray<rank> &dimensions) {
  size_t lin = 0;
#pragma unroll
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    lin = lin * static_cast<size_t>(dimensions[d]) +
          static_cast<size_t>(site[d]);
  }
  return lin;
}

template <size_t rank>
KOKKOS_INLINE_FUNCTION size_t
wilson_site_count(const IndexArray<rank> &dimensions) {
  size_t nSites = 1;
#pragma unroll
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    nSites *= static_cast<size_t>(dimensions[d]);
  }
  return nSites;
}

template <size_t rank, size_t Nc, class RNG> struct WLoop_munu_metropolis {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  const GaugeFieldType g_in;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const index_t multihit;
  const real_t beta;
  const real_t delta;
  const real_t epsilon1;
  const real_t epsilon2;
  const RNG rng;
  const IndexArray<rank> dimensions;

  WLoop_munu_metropolis(const GaugeFieldType &g_in, const index_t mu,
                        const index_t nu, const index_t Lmu,
                        const index_t Lnu, const index_t multihit,
                        const real_t beta, const real_t delta,
                        const real_t epsilon1, const real_t epsilon2,
                        const RNG &rng, const IndexArray<rank> &dimensions)
      : g_in(g_in), mu(mu), nu(nu), Lmu(Lmu), Lnu(Lnu),
        multihit(multihit), beta(beta), delta(delta), epsilon1(epsilon1),
        epsilon2(epsilon2), rng(rng), dimensions(dimensions) {}

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

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &site,
             complex_t &lsum) const {
    auto generator = rng.get_state();
    lsum += loop_at_site(site, generator);
    rng.free_state(generator);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              complex_t &lsum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              complex_t &lsum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              complex_t &lsum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2, i3}, lsum);
  }
};

template <size_t rank, size_t Nc, class RNG> struct WLoop_munu_heatbath {
  constexpr static const size_t Nd = rank;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  const GaugeFieldType g_in;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const index_t multihit;
  const index_t nOverrelax;
  const real_t beta;
  const real_t epsilon1;
  const RNG rng;
  const IndexArray<rank> dimensions;

  WLoop_munu_heatbath(const GaugeFieldType &g_in, const index_t mu,
                      const index_t nu, const index_t Lmu,
                      const index_t Lnu, const index_t multihit,
                      const index_t nOverrelax, const real_t beta,
                      const real_t epsilon1, const RNG &rng,
                      const IndexArray<rank> &dimensions)
      : g_in(g_in), mu(mu), nu(nu), Lmu(Lmu), Lnu(Lnu),
        multihit(multihit), nOverrelax(nOverrelax), beta(beta),
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

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &site,
             complex_t &lsum) const {
    auto generator = rng.get_state();
    lsum += loop_at_site(site, generator);
    rng.free_state(generator);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              complex_t &lsum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              complex_t &lsum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              complex_t &lsum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2, i3}, lsum);
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

  const GaugeFieldType g_in;
  const index_t mu, nu;
  const index_t Lmu, Lnu;
  const IndexArray<rank> dimensions;

  WLoop_munu_raw(const GaugeFieldType &g_in, const index_t mu,
                 const index_t nu, const index_t Lmu,
                 const index_t Lnu, const IndexArray<rank> &dimensions)
      : g_in(g_in), mu(mu), nu(nu), Lmu(Lmu), Lnu(Lnu),
        dimensions(dimensions) {}

  KOKKOS_FORCEINLINE_FUNCTION void
  contribute(const Kokkos::Array<index_t, rank> &site,
             complex_t &lsum) const {
    lsum +=
        WilsonLoopRawAtSite<rank, Nc>(g_in, site, mu, nu, Lmu, Lnu, dimensions);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              complex_t &lsum) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              complex_t &lsum) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2}, lsum);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              complex_t &lsum) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(Kokkos::Array<index_t, rank>{i0, i1, i2, i3}, lsum);
  }
};

template <size_t rank, size_t Nc> struct TemporalWilsonSpatialLines {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using TransporterView =
      Kokkos::View<SUN<Nc> ***,
                   Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  TransporterView spatial_lines;
  const IndexArray<rank> dimensions;
  const index_t Rmax;
  const size_t nSites;

  TemporalWilsonSpatialLines(const GaugeFieldType &g_in,
                             TransporterView &spatial_lines,
                             const IndexArray<rank> &dimensions,
                             const index_t Rmax, const size_t nSites)
      : g_in(g_in), spatial_lines(spatial_lines), dimensions(dimensions),
        Rmax(Rmax), nSites(nSites) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t work) const {
    const index_t idir = static_cast<index_t>(work / nSites);
    const size_t lin = work % nSites;
    auto site = wilson_linear_to_site<rank>(lin, dimensions);
    auto shifted = site;
    SUN<Nc> line = identitySUN<Nc>();

    spatial_lines(idir, 0, lin) = line;
    for (index_t r = 1; r <= Rmax; ++r) {
      line *= g_in(shifted, idir);
      spatial_lines(idir, r, lin) = line;
      shifted = shift_index_plus<rank>(shifted, idir, 1, dimensions);
    }
  }
};

template <size_t rank, size_t Nc> struct TemporalWilsonInitTransporter {
  using TransporterView =
      Kokkos::View<SUN<Nc> *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  TransporterView Tcurr;

  TemporalWilsonInitTransporter(TransporterView &Tcurr) : Tcurr(Tcurr) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    Tcurr(lin) = identitySUN<Nc>();
  }
};

template <size_t rank, size_t Nc> struct TemporalWilsonUpdateTransporter {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using TransporterView =
      Kokkos::View<SUN<Nc> *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  TransporterView Tcurr;
  const IndexArray<rank> dimensions;
  const index_t t_minus_one;

  TemporalWilsonUpdateTransporter(const GaugeFieldType &g_in,
                                  TransporterView &Tcurr,
                                  const IndexArray<rank> &dimensions,
                                  const index_t t_minus_one)
      : g_in(g_in), Tcurr(Tcurr), dimensions(dimensions),
        t_minus_one(t_minus_one) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    constexpr index_t time_dir = static_cast<index_t>(rank - 1);
    const auto site = wilson_linear_to_site<rank>(lin, dimensions);
    const auto shifted =
        shift_index_plus<rank>(site, time_dir, t_minus_one, dimensions);
    Tcurr(lin) *= g_in(shifted, time_dir);
  }
};

template <size_t rank, size_t Nc> struct TemporalWilsonMeasureFixedT {
  using SpatialLinesView =
      Kokkos::View<SUN<Nc> ***,
                   Kokkos::MemoryTraits<Kokkos::Restrict>>;
  using TransporterView =
      Kokkos::View<SUN<Nc> *, Kokkos::MemoryTraits<Kokkos::Restrict>>;
  using AccumView =
      Kokkos::View<real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  SpatialLinesView spatial_lines;
  TransporterView Tcurr;
  AccumView W_accum;
  const IndexArray<rank> dimensions;
  const index_t t;

  TemporalWilsonMeasureFixedT(SpatialLinesView &spatial_lines,
                              TransporterView &Tcurr, AccumView &W_accum,
                              const IndexArray<rank> &dimensions,
                              const index_t t)
      : spatial_lines(spatial_lines), Tcurr(Tcurr), W_accum(W_accum),
        dimensions(dimensions), t(t) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t r,
                                              const index_t idir,
                                              const index_t site_lin) const {
    constexpr index_t time_dir = static_cast<index_t>(rank - 1);
    const auto site =
        wilson_linear_to_site<rank>(static_cast<size_t>(site_lin), dimensions);
    const auto site_r = shift_index_plus<rank>(site, idir, r, dimensions);
    const auto site_t =
        shift_index_plus<rank>(site, time_dir, t, dimensions);
    const size_t lin_r = wilson_site_to_linear<rank>(site_r, dimensions);
    const size_t lin_t = wilson_site_to_linear<rank>(site_t, dimensions);

    const SUN<Nc> loop = spatial_lines(idir, r, site_lin) * Tcurr(lin_r) *
                         conj(spatial_lines(idir, r, lin_t)) *
                         conj(Tcurr(site_lin));
    const real_t contribution = trace(loop).real();
    Kokkos::atomic_add(&W_accum(r, t), contribution);
  }
};

template <size_t rank> struct TemporalWilsonNormalize {
  using AccumView =
      Kokkos::View<real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  AccumView W_accum;
  const real_t norm;

  TemporalWilsonNormalize(AccumView &W_accum, const real_t norm)
      : W_accum(W_accum), norm(norm) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t r,
                                              const index_t t) const {
    W_accum(r, t) /= norm;
  }
};

template <size_t rank, size_t Nc>
void WilsonLoop_temporal_raw_fused(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const std::vector<Kokkos::Array<index_t, 2>> &L_T_pairs,
    std::vector<Kokkos::Array<real_t, 3>> &Wtemporal_vals,
    const bool normalize = true) {
  if (L_T_pairs.empty()) {
    return;
  }

  constexpr index_t spatial_dirs = static_cast<index_t>(rank - 1);
  const auto dimensions = g_in.dimensions;
  const size_t nSites = wilson_site_count<rank>(dimensions);

  index_t Rmax = 0;
  index_t Tmax = 0;
  for (const auto &pair : L_T_pairs) {
    Rmax = pair[0] > Rmax ? pair[0] : Rmax;
    Tmax = pair[1] > Tmax ? pair[1] : Tmax;
  }

  using SpatialLinesView =
      Kokkos::View<SUN<Nc> ***,
                   Kokkos::MemoryTraits<Kokkos::Restrict>>;
  using TransporterView =
      Kokkos::View<SUN<Nc> *, Kokkos::MemoryTraits<Kokkos::Restrict>>;
  using AccumView =
      Kokkos::View<real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  SpatialLinesView spatial_lines("temporal_wilson_spatial_lines",
                                 spatial_dirs, Rmax + 1, nSites);
  TransporterView Tcurr("temporal_wilson_Tcurr", nSites);
  AccumView W_accum("temporal_wilson_W_accum", Rmax + 1, Tmax + 1);

  Kokkos::parallel_for(
      "TemporalWilsonSpatialLines",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0, static_cast<size_t>(spatial_dirs) * nSites),
      TemporalWilsonSpatialLines<rank, Nc>(g_in, spatial_lines, dimensions,
                                           Rmax, nSites));

  Kokkos::parallel_for(
      "TemporalWilsonInitTransporter",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nSites),
      TemporalWilsonInitTransporter<rank, Nc>(Tcurr));

  Kokkos::deep_copy(W_accum, real_t(0.0));

  const index_t nSitesIndex = static_cast<index_t>(nSites);
  for (index_t t = 1; t <= Tmax; ++t) {
    Kokkos::parallel_for(
        "TemporalWilsonUpdateTransporter",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nSites),
        TemporalWilsonUpdateTransporter<rank, Nc>(g_in, Tcurr, dimensions,
                                                  t - 1));

    Kokkos::parallel_for(
        "TemporalWilsonMeasureFixedT",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {1, 0, 0}, {Rmax + 1, spatial_dirs, nSitesIndex}),
        TemporalWilsonMeasureFixedT<rank, Nc>(spatial_lines, Tcurr, W_accum,
                                              dimensions, t));
  }

  const real_t norm =
      normalize ? static_cast<real_t>(spatial_dirs) *
                      static_cast<real_t>(nSites) * static_cast<real_t>(Nc)
                : static_cast<real_t>(spatial_dirs);
  Kokkos::parallel_for(
      "TemporalWilsonNormalize",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {Rmax + 1, Tmax + 1}),
      TemporalWilsonNormalize<rank>(W_accum, norm));

  auto W_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   W_accum);
  for (const auto &pair : L_T_pairs) {
    Wtemporal_vals.push_back(
        Kokkos::Array<real_t, 3>{static_cast<real_t>(pair[0]),
                                 static_cast<real_t>(pair[1]),
                                 W_host(pair[0], pair[1])});
  }
}

template <size_t rank, size_t Nc, class RNG>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in, const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>> &Lmu_nu_pairs,
    std::vector<Kokkos::Array<real_t, 5>> &Wmunu_vals, const index_t multihit,
    const real_t beta, const real_t delta, const real_t epsilon1,
    const real_t epsilon2, const RNG &rng, const bool normalize = true) {
  const auto dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  for (const auto &Lmu_nu : Lmu_nu_pairs) {
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];
    complex_t Wmunu(0.0, 0.0);

    if (multihit > 1) {
      Kokkos::parallel_reduce(
          "WilsonLoopMuNuMetropolis", Policy<rank>(start, end),
          WLoop_munu_metropolis<rank, Nc, RNG>(
              g_in, mu, nu, Lmu, Lnu, multihit, beta, delta, epsilon1,
              epsilon2, rng, dimensions),
          Kokkos::Sum<complex_t>(Wmunu));
    } else {
      Kokkos::parallel_reduce(
          "WilsonLoopMuNuRaw", Policy<rank>(start, end),
          WLoop_munu_raw<rank, Nc>(g_in, mu, nu, Lmu, Lnu, dimensions),
          Kokkos::Sum<complex_t>(Wmunu));
    }

    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < rank; ++i) {
        Wmunu /= static_cast<real_t>(dimensions[i]);
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
  const auto dimensions = g_in.dimensions;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  for (const auto &Lmu_nu : Lmu_nu_pairs) {
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];
    complex_t Wmunu(0.0, 0.0);

    if (multihit > 1) {
      Kokkos::parallel_reduce(
          "WilsonLoopMuNuHeatbath", Policy<rank>(start, end),
          WLoop_munu_heatbath<rank, Nc, RNG>(
              g_in, mu, nu, Lmu, Lnu, multihit, updateParams.nOverrelax,
              updateParams.beta, updateParams.epsilon1, rng, dimensions),
          Kokkos::Sum<complex_t>(Wmunu));
    } else {
      Kokkos::parallel_reduce(
          "WilsonLoopMuNuRaw", Policy<rank>(start, end),
          WLoop_munu_raw<rank, Nc>(g_in, mu, nu, Lmu, Lnu, dimensions),
          Kokkos::Sum<complex_t>(Wmunu));
    }

    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < rank; ++i) {
        Wmunu /= static_cast<real_t>(dimensions[i]);
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
  if (multihit <= 1) {
    WilsonLoop_temporal_raw_fused<rank, Nc>(g_in, L_T_pairs, Wtemporal_vals,
                                            normalize);
    return;
  }

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
  if (multihit <= 1) {
    WilsonLoop_temporal_raw_fused<rank, Nc>(g_in, L_T_pairs, Wtemporal_vals,
                                            normalize);
    return;
  }

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
