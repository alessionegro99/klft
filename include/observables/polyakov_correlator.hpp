#pragma once

#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "observables/polyakov_loop.hpp"
#include "params/heatbath_params.hpp"
#include "params/metropolis_params.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace klft {

template <size_t rank>
KOKKOS_INLINE_FUNCTION size_t
polyakov_origin_to_linear(const Kokkos::Array<index_t, rank> &site,
                          const IndexArray<rank> &dimensions) {
  size_t lin = 0;
#pragma unroll
  for (index_t d = 0; d < rank - 1; ++d) {
    lin = lin * static_cast<size_t>(dimensions[d]) +
          static_cast<size_t>(site[d]);
  }
  return lin;
}

template <size_t rank>
index_t max_polyakov_correlator_r(const IndexArray<rank> &dimensions) {
  index_t min_spatial = dimensions[0];
  for (index_t d = 1; d < rank - 1; ++d) {
    min_spatial = std::min(min_spatial, dimensions[d]);
  }
  return min_spatial / 2;
}

template <size_t rank>
Kokkos::Array<real_t, 3> PolyakovCorrelatorAtR(
    const Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>
        &local_polyakov,
    const index_t R, const IndexArray<rank> &dimensions) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const real_t spatial_dirs = static_cast<real_t>(rank - 1);

  complex_t total(0.0, 0.0);

  Kokkos::parallel_reduce(
      "PolyakovCorrelator", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t i, complex_t &lsum) {
        const auto site = linear_to_polyakov_origin<rank>(i, dimensions);
        const complex_t p0 = local_polyakov(i);
        complex_t local(0.0, 0.0);

#pragma unroll
        for (index_t mu = 0; mu < rank - 1; ++mu) {
          const auto shifted = shift_index_plus<rank>(site, mu, R, dimensions);
          const size_t j = polyakov_origin_to_linear<rank>(shifted, dimensions);
          local += Kokkos::conj(p0) * local_polyakov(j);
        }

        lsum += local;
      },
      Kokkos::Sum<complex_t>(total));

  if (nSpatial > 0) {
    const real_t norm = 1.0 / (static_cast<real_t>(nSpatial) * spatial_dirs);
    total *= norm;
  }

  return Kokkos::Array<real_t, 3>{static_cast<real_t>(R), total.real(),
                                  total.imag()};
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void PolyakovCorrelator(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const index_t max_r, const index_t multihit,
    std::vector<Kokkos::Array<real_t, 3>> &corr_values,
    const UpdateParams &updateParams, const RNG &rng) {
  using LocalFieldType =
      Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  const auto dimensions = g_in.dimensions;
  if (max_r < 0) {
    throw std::runtime_error("polyakov_correlator_max_r must be >= 0");
  }
  if (max_r > max_polyakov_correlator_r<rank>(dimensions)) {
    throw std::runtime_error(
        "polyakov_correlator_max_r exceeds the unique periodic range");
  }

  const size_t nSpatial = spatial_volume<rank>(dimensions);
  LocalFieldType raw_polyakov("raw_polyakov", nSpatial);
  const bool need_multihit = multihit > 1 && max_r >= 2;
  LocalFieldType multihit_polyakov;
  if (need_multihit) {
    multihit_polyakov = LocalFieldType("multihit_polyakov", nSpatial);
    LocalPolyakovLoopPair<rank, Nc>(g_in, raw_polyakov, multihit_polyakov,
                                    multihit, updateParams, rng);
  } else {
    LocalPolyakovLoop<rank, Nc>(g_in, raw_polyakov, 1, updateParams, rng);
  }

  corr_values.clear();
  corr_values.reserve(static_cast<size_t>(max_r + 1));
  for (index_t R = 0; R <= max_r; ++R) {
    if (R < 2 || !need_multihit) {
      corr_values.push_back(
          PolyakovCorrelatorAtR<rank>(raw_polyakov, R, dimensions));
    } else {
      corr_values.push_back(
          PolyakovCorrelatorAtR<rank>(multihit_polyakov, R, dimensions));
    }
  }
}

} // namespace klft
