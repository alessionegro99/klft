#pragma once

#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/multihit_links.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"

namespace klft {

// For a single Polyakov loop, the temporal links can be multihit-averaged
// independently. For correlators, KLFT follows the Bonn separation rule:
// R = 0 and R = 1 stay raw, while only R >= 2 can use the multihit
// Polyakov loops.

template <size_t rank>
KOKKOS_INLINE_FUNCTION size_t spatial_volume(const IndexArray<rank> &dimensions) {
  size_t volume = 1;
#pragma unroll
  for (index_t d = 0; d < rank - 1; ++d) {
    volume *= static_cast<size_t>(dimensions[d]);
  }
  return volume;
}

template <size_t rank>
KOKKOS_INLINE_FUNCTION Kokkos::Array<index_t, rank>
linear_to_polyakov_origin(size_t lin, const IndexArray<rank> &dimensions) {
  Kokkos::Array<index_t, rank> site;
  site[rank - 1] = 0;
  for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
    const size_t dim = static_cast<size_t>(dimensions[d]);
    site[d] = static_cast<index_t>(lin % dim);
    lin /= dim;
  }
  return site;
}

template <size_t rank, size_t Nc> struct LocalPolyakovLoopRaw {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  LocalFieldType poly_per_site;
  const IndexArray<rank> dimensions;

  LocalPolyakovLoopRaw(const GaugeFieldType &g_in, LocalFieldType &poly_per_site,
                       const IndexArray<rank> &dimensions)
      : g_in(g_in), poly_per_site(poly_per_site), dimensions(dimensions) {}

  KOKKOS_FORCEINLINE_FUNCTION complex_t
  polyakov_at_site(const Kokkos::Array<index_t, rank> &origin) const {
    auto site = origin;
    SUN<Nc> loop = identitySUN<Nc>();

    for (index_t t = 0; t < dimensions[time_dir]; ++t) {
      loop *= g_in(site, time_dir);
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }

    return trace(loop) * (1.0 / static_cast<real_t>(Nc));
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    poly_per_site(lin) =
        polyakov_at_site(linear_to_polyakov_origin<rank>(lin, dimensions));
  }
};

template <size_t rank, size_t Nc, class RNG> struct LocalPolyakovLoopMetropolis {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  LocalFieldType poly_per_site;
  const index_t multihit;
  const real_t beta;
  const real_t delta;
  const real_t epsilon1;
  const real_t epsilon2;
  const RNG rng;
  const IndexArray<rank> dimensions;

  LocalPolyakovLoopMetropolis(const GaugeFieldType &g_in,
                              LocalFieldType &poly_per_site,
                              const index_t multihit, const real_t beta,
                              const real_t delta, const real_t epsilon1,
                              const real_t epsilon2, const RNG &rng,
                              const IndexArray<rank> &dimensions)
      : g_in(g_in), poly_per_site(poly_per_site), multihit(multihit),
        beta(beta), delta(delta), epsilon1(epsilon1), epsilon2(epsilon2),
        rng(rng), dimensions(dimensions) {}

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION complex_t
  polyakov_at_site(const Kokkos::Array<index_t, rank> &origin,
                   Generator &generator) const {
    auto site = origin;
    SUN<Nc> loop = identitySUN<Nc>();

    for (index_t t = 0; t < dimensions[time_dir]; ++t) {
      loop *= (multihit > 1)
                  ? multihit_link_metropolis<Nc>(
                        g_in(site, time_dir), g_in.staple(site, time_dir),
                        multihit, beta, delta, epsilon1, epsilon2, generator)
                  : g_in(site, time_dir);
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }

    return trace(loop) * (1.0 / static_cast<real_t>(Nc));
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    auto generator = rng.get_state();
    poly_per_site(lin) =
        polyakov_at_site(linear_to_polyakov_origin<rank>(lin, dimensions),
                         generator);
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc, class RNG> struct LocalPolyakovLoopHeatbath {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  LocalFieldType poly_per_site;
  const index_t multihit;
  const index_t nOverrelax;
  const real_t beta;
  const real_t epsilon1;
  const RNG rng;
  const IndexArray<rank> dimensions;

  LocalPolyakovLoopHeatbath(const GaugeFieldType &g_in,
                            LocalFieldType &poly_per_site,
                            const index_t multihit, const index_t nOverrelax,
                            const real_t beta, const real_t epsilon1,
                            const RNG &rng, const IndexArray<rank> &dimensions)
      : g_in(g_in), poly_per_site(poly_per_site), multihit(multihit),
        nOverrelax(nOverrelax), beta(beta), epsilon1(epsilon1), rng(rng),
        dimensions(dimensions) {}

  template <class Generator>
  KOKKOS_FORCEINLINE_FUNCTION complex_t
  polyakov_at_site(const Kokkos::Array<index_t, rank> &origin,
                   Generator &generator) const {
    auto site = origin;
    SUN<Nc> loop = identitySUN<Nc>();

    for (index_t t = 0; t < dimensions[time_dir]; ++t) {
      loop *=
          (multihit > 1)
              ? multihit_link_heatbath<Nc>(
                    g_in(site, time_dir), g_in.staple(site, time_dir),
                    multihit, nOverrelax, beta, epsilon1, generator)
              : g_in(site, time_dir);
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }

    return trace(loop) * (1.0 / static_cast<real_t>(Nc));
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    auto generator = rng.get_state();
    poly_per_site(lin) =
        polyakov_at_site(linear_to_polyakov_origin<rank>(lin, dimensions),
                         generator);
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc, class RNG>
struct LocalPolyakovLoopPairMetropolis {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  LocalFieldType raw_poly_per_site;
  LocalFieldType multihit_poly_per_site;
  const index_t multihit;
  const real_t beta;
  const real_t delta;
  const real_t epsilon1;
  const real_t epsilon2;
  const RNG rng;
  const IndexArray<rank> dimensions;

  LocalPolyakovLoopPairMetropolis(const GaugeFieldType &g_in,
                                  LocalFieldType &raw_poly_per_site,
                                  LocalFieldType &multihit_poly_per_site,
                                  const index_t multihit, const real_t beta,
                                  const real_t delta, const real_t epsilon1,
                                  const real_t epsilon2, const RNG &rng,
                                  const IndexArray<rank> &dimensions)
      : g_in(g_in), raw_poly_per_site(raw_poly_per_site),
        multihit_poly_per_site(multihit_poly_per_site), multihit(multihit),
        beta(beta), delta(delta), epsilon1(epsilon1), epsilon2(epsilon2),
        rng(rng), dimensions(dimensions) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    auto generator = rng.get_state();
    auto site = linear_to_polyakov_origin<rank>(lin, dimensions);
    SUN<Nc> raw_loop = identitySUN<Nc>();
    SUN<Nc> multihit_loop = identitySUN<Nc>();

    for (index_t t = 0; t < dimensions[time_dir]; ++t) {
      const SUN<Nc> raw_link = g_in(site, time_dir);
      raw_loop *= raw_link;
      multihit_loop *=
          (multihit > 1)
              ? multihit_link_metropolis<Nc>(
                    raw_link, g_in.staple(site, time_dir), multihit, beta,
                    delta, epsilon1, epsilon2, generator)
              : raw_link;
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }

    const real_t norm = 1.0 / static_cast<real_t>(Nc);
    raw_poly_per_site(lin) = trace(raw_loop) * norm;
    multihit_poly_per_site(lin) = trace(multihit_loop) * norm;
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc, class RNG>
struct LocalPolyakovLoopPairHeatbath {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const GaugeFieldType g_in;
  LocalFieldType raw_poly_per_site;
  LocalFieldType multihit_poly_per_site;
  const index_t multihit;
  const index_t nOverrelax;
  const real_t beta;
  const real_t epsilon1;
  const RNG rng;
  const IndexArray<rank> dimensions;

  LocalPolyakovLoopPairHeatbath(const GaugeFieldType &g_in,
                                LocalFieldType &raw_poly_per_site,
                                LocalFieldType &multihit_poly_per_site,
                                const index_t multihit,
                                const index_t nOverrelax, const real_t beta,
                                const real_t epsilon1, const RNG &rng,
                                const IndexArray<rank> &dimensions)
      : g_in(g_in), raw_poly_per_site(raw_poly_per_site),
        multihit_poly_per_site(multihit_poly_per_site), multihit(multihit),
        nOverrelax(nOverrelax), beta(beta), epsilon1(epsilon1), rng(rng),
        dimensions(dimensions) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t lin) const {
    auto generator = rng.get_state();
    auto site = linear_to_polyakov_origin<rank>(lin, dimensions);
    SUN<Nc> raw_loop = identitySUN<Nc>();
    SUN<Nc> multihit_loop = identitySUN<Nc>();

    for (index_t t = 0; t < dimensions[time_dir]; ++t) {
      const SUN<Nc> raw_link = g_in(site, time_dir);
      raw_loop *= raw_link;
      multihit_loop *=
          (multihit > 1)
              ? multihit_link_heatbath<Nc>(
                    raw_link, g_in.staple(site, time_dir), multihit,
                    nOverrelax, beta, epsilon1, generator)
              : raw_link;
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }

    const real_t norm = 1.0 / static_cast<real_t>(Nc);
    raw_poly_per_site(lin) = trace(raw_loop) * norm;
    multihit_poly_per_site(lin) = trace(multihit_loop) * norm;
    rng.free_state(generator);
  }
};

template <size_t rank, size_t Nc, class RNG>
void LocalPolyakovLoop(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &local_polyakov,
    const index_t multihit, const MetropolisParams &updateParams,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);

  Kokkos::parallel_for(
      "LocalPolyakovLoopMetropolis", Kokkos::RangePolicy<Exec>(0, nSpatial),
      LocalPolyakovLoopMetropolis<rank, Nc, RNG>(
          g_in, local_polyakov, multihit, updateParams.beta, updateParams.delta,
          updateParams.epsilon1, updateParams.epsilon2, rng, dimensions));
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class RNG>
void LocalPolyakovLoopPair(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &raw_polyakov,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &multihit_polyakov,
    const index_t multihit, const MetropolisParams &updateParams,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);

  Kokkos::parallel_for(
      "LocalPolyakovLoopPairMetropolis",
      Kokkos::RangePolicy<Exec>(0, nSpatial),
      LocalPolyakovLoopPairMetropolis<rank, Nc, RNG>(
          g_in, raw_polyakov, multihit_polyakov, multihit, updateParams.beta,
          updateParams.delta, updateParams.epsilon1, updateParams.epsilon2, rng,
          dimensions));
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class RNG>
void LocalPolyakovLoopPair(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &raw_polyakov,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &multihit_polyakov,
    const index_t multihit, const HeatbathParams &updateParams,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);

  Kokkos::parallel_for(
      "LocalPolyakovLoopPairHeatbath", Kokkos::RangePolicy<Exec>(0, nSpatial),
      LocalPolyakovLoopPairHeatbath<rank, Nc, RNG>(
          g_in, raw_polyakov, multihit_polyakov, multihit,
          updateParams.nOverrelax, updateParams.beta, updateParams.epsilon1, rng,
          dimensions));
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class RNG>
void LocalPolyakovLoop(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &local_polyakov,
    const index_t multihit, const HeatbathParams &updateParams,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);

  Kokkos::parallel_for(
      "LocalPolyakovLoopHeatbath", Kokkos::RangePolicy<Exec>(0, nSpatial),
      LocalPolyakovLoopHeatbath<rank, Nc, RNG>(
          g_in, local_polyakov, multihit, updateParams.nOverrelax,
          updateParams.beta, updateParams.epsilon1, rng, dimensions));
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class RNG>
void LocalPolyakovLoop(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &local_polyakov,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  (void)rng;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);

  Kokkos::parallel_for("LocalPolyakovLoopRaw",
                       Kokkos::RangePolicy<Exec>(0, nSpatial),
                       LocalPolyakovLoopRaw<rank, Nc>(g_in, local_polyakov,
                                                      dimensions));
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
Kokkos::Array<real_t, 2> PolyakovLoop(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const index_t multihit, const UpdateParams &updateParams, const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  LocalFieldType local_polyakov("local_polyakov", nSpatial);

  LocalPolyakovLoop<rank, Nc>(g_in, local_polyakov, multihit, updateParams,
                              rng);

  real_t rep = 0.0;
  real_t imp = 0.0;
  Kokkos::parallel_reduce(
      "PolyakovLoopReal", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t i, real_t &lsum) {
        lsum += local_polyakov(i).real();
      },
      rep);
  Kokkos::parallel_reduce(
      "PolyakovLoopImag", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t i, real_t &lsum) {
        lsum += local_polyakov(i).imag();
      },
      imp);

  if (nSpatial > 0) {
    const real_t invSpatial = 1.0 / static_cast<real_t>(nSpatial);
    rep *= invSpatial;
    imp *= invSpatial;
  }

  return Kokkos::Array<real_t, 2>{rep, imp};
}

} // namespace klft
