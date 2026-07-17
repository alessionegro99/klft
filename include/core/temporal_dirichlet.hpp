#pragma once

#include "fields/field_type_traits.hpp"

#include <stdexcept>
#include <string>

namespace klft {

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION constexpr index_t temporal_direction() {
  return static_cast<index_t>(rank - 1);
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION constexpr index_t lower_boundary_slice() {
  return 0;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION index_t
upper_boundary_slice(const IndexArray<rank> &dimensions) {
  return dimensions[temporal_direction<rank>()] - 1;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION index_t
slab_thickness_in_links(const IndexArray<rank> &dimensions) {
  return upper_boundary_slice<rank>(dimensions);
}

// The periodic storage has Nt + 1 temporal sites. Spatial links at t=0 and
// t=Nt are fixed; the temporal link at t=Nt is the explicitly retained,
// factorized wrapping-PCM link and is therefore dynamical.
template <size_t rank, typename indexType>
KOKKOS_FORCEINLINE_FUNCTION bool is_temporal_dirichlet_link(
    const Kokkos::Array<indexType, rank> &site, const index_t mu,
    const IndexArray<rank> &dimensions) {
  const index_t time_dir = temporal_direction<rank>();
  const index_t t = static_cast<index_t>(site[time_dir]);
  return mu != time_dir &&
         (t == lower_boundary_slice<rank>() ||
          t == upper_boundary_slice<rank>(dimensions));
}

template <size_t rank, typename indexType>
KOKKOS_FORCEINLINE_FUNCTION bool is_wrapping_temporal_link(
    const Kokkos::Array<indexType, rank> &site, const index_t mu,
    const IndexArray<rank> &dimensions) {
  return mu == temporal_direction<rank>() &&
         static_cast<index_t>(site[temporal_direction<rank>()]) ==
             upper_boundary_slice<rank>(dimensions);
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION bool is_exact_identity(const SUN<Nc> &link) {
  if constexpr (Nc == 1) {
    return link.comp.real() == 1.0 && link.comp.imag() == 0.0;
  } else if constexpr (Nc == 2) {
    return link.comp[0] == 1.0 && link.comp[1] == 0.0 &&
           link.comp[2] == 0.0 && link.comp[3] == 0.0;
  } else {
#pragma unroll
    for (index_t row = 0; row < static_cast<index_t>(Nc); ++row) {
#pragma unroll
      for (index_t col = 0; col < static_cast<index_t>(Nc); ++col) {
        const complex_t expected(row == col ? 1.0 : 0.0, 0.0);
        if (matrix_ref(link, row, col) != expected) {
          return false;
        }
      }
    }
    return true;
  }
}

template <size_t rank, size_t Nc> struct SetTemporalDirichletLinks {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType gauge;

  explicit SetTemporalDirichletLinks(const GaugeFieldType &gauge)
      : gauge(gauge) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... indices) const {
    const IndexArray<rank> site{static_cast<index_t>(indices)...};
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      if (is_temporal_dirichlet_link<rank>(site, mu, gauge.dimensions)) {
        gauge(site, mu) = identitySUN<Nc>();
      }
    }
  }
};

template <size_t rank, size_t Nc>
void apply_temporal_dirichlet_boundaries(
    typename DeviceGaugeFieldType<rank, Nc>::type &gauge) {
  Kokkos::parallel_for("set_temporal_dirichlet_links",
                       Policy<rank>(IndexArray<rank>{}, gauge.dimensions),
                       SetTemporalDirichletLinks<rank, Nc>(gauge));
  Kokkos::fence();
}

template <size_t rank, size_t Nc> struct CountInvalidTemporalDirichletLinks {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType gauge;

  explicit CountInvalidTemporalDirichletLinks(const GaugeFieldType &gauge)
      : gauge(gauge) {}

  KOKKOS_FORCEINLINE_FUNCTION void contribute(const IndexArray<rank> &site,
                                              size_t &invalid) const {
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      if (is_temporal_dirichlet_link<rank>(site, mu, gauge.dimensions) &&
          !is_exact_identity<Nc>(gauge(site, mu))) {
        ++invalid;
      }
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              size_t &invalid) const {
    static_assert(rank == 2, "2-index overload requires rank 2.");
    contribute(IndexArray<rank>{i0, i1}, invalid);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              size_t &invalid) const {
    static_assert(rank == 3, "3-index overload requires rank 3.");
    contribute(IndexArray<rank>{i0, i1, i2}, invalid);
  }

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const index_t i0,
                                              const index_t i1,
                                              const index_t i2,
                                              const index_t i3,
                                              size_t &invalid) const {
    static_assert(rank == 4, "4-index overload requires rank 4.");
    contribute(IndexArray<rank>{i0, i1, i2, i3}, invalid);
  }
};

template <size_t rank, size_t Nc>
bool temporal_dirichlet_boundaries_are_exact(
    const typename DeviceGaugeFieldType<rank, Nc>::type &gauge) {
  size_t invalid = 0;
  Kokkos::parallel_reduce(
      "validate_temporal_dirichlet_links",
      Policy<rank>(IndexArray<rank>{}, gauge.dimensions),
      CountInvalidTemporalDirichletLinks<rank, Nc>(gauge), invalid);
  Kokkos::fence();
  return invalid == 0;
}

template <size_t rank>
size_t dynamical_link_count(const IndexArray<rank> &dimensions,
                            const bool temporal_dirichlet) {
  size_t volume = 1;
  size_t spatial_volume = 1;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    volume *= static_cast<size_t>(dimensions[d]);
    if (d != temporal_direction<rank>()) {
      spatial_volume *= static_cast<size_t>(dimensions[d]);
    }
  }
  const size_t all_links = volume * rank;
  return temporal_dirichlet
             ? all_links - 2 * spatial_volume * (rank - 1)
             : all_links;
}

// Even periodic directions use the existing two checkerboard colors.  An odd
// periodic cycle is not bipartite, so split its last site into a third color:
// {0,2,...,L-3}, {1,3,...,L-2}, {L-1}.  The odd stored temporal extent in
// Dirichlet mode still needs only two colors: the only wrapping-neighbor pair
// of tangential links lies on the two fixed surfaces, while temporal links at
// adjacent t do not share a plaquette.
template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION index_t lattice_color_count(
    const IndexArray<rank> &dimensions, const index_t direction,
    const bool temporal_dirichlet) {
  const bool broken_temporal_cycle =
      temporal_dirichlet && direction == temporal_direction<rank>();
  return dimensions[direction] % 2 == 0 || broken_temporal_cycle ? 2 : 3;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION index_t lattice_color_extent(
    const IndexArray<rank> &dimensions, const index_t direction,
    const index_t color, const bool temporal_dirichlet) {
  const index_t extent = dimensions[direction];
  if (lattice_color_count<rank>(dimensions, direction,
                                temporal_dirichlet) == 2) {
    return (extent + (color == 0 ? 1 : 0)) / 2;
  }
  return color == 2 ? 1 : extent / 2;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION index_t lattice_color_total(
    const IndexArray<rank> &dimensions, const bool temporal_dirichlet) {
  index_t total = 1;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    total *= lattice_color_count<rank>(dimensions, d, temporal_dirichlet);
  }
  return total;
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
decode_lattice_color(const index_t linear_color,
                     const IndexArray<rank> &dimensions,
                     const bool temporal_dirichlet) {
  Kokkos::Array<index_t, rank> colors;
  index_t remaining = linear_color;
  for (index_t d = static_cast<index_t>(rank) - 1; d >= 0; --d) {
    const index_t count =
        lattice_color_count<rank>(dimensions, d, temporal_dirichlet);
    colors[d] = remaining % count;
    remaining /= count;
  }
  return colors;
}

template <size_t rank, typename indexType>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
index_lattice_color(const Kokkos::Array<indexType, rank> &compressed_site,
                    const Kokkos::Array<index_t, rank> &colors,
                    const IndexArray<rank> &dimensions,
                    const bool temporal_dirichlet) {
  Kokkos::Array<index_t, rank> site;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    if (lattice_color_count<rank>(dimensions, d, temporal_dirichlet) == 3 &&
        colors[d] == 2) {
      site[d] = dimensions[d] - 1;
    } else {
      site[d] = static_cast<index_t>(2 * compressed_site[d] + colors[d]);
    }
  }
  return site;
}

template <size_t rank>
void validate_temporal_dirichlet_extents(
    const IndexArray<rank> &dimensions, const char *algorithm_name) {
  for (index_t d = 0; d < temporal_direction<rank>(); ++d) {
    if (dimensions[d] <= 0) {
      throw std::runtime_error(std::string(algorithm_name) +
                               " requires positive spatial extents.");
    }
  }
  if (dimensions[temporal_direction<rank>()] < 2) {
    throw std::runtime_error(std::string(algorithm_name) +
                             " requires temporal_site_extent >= 2 in "
                             "Dirichlet mode.");
  }
}

} // namespace klft
