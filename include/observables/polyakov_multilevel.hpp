#pragma once

#include "observables/polyakov_correlator.hpp"
#include "params/heatbath_params.hpp"
#include "params/multilevel_params.hpp"
#include "updates/restricted_heatbath.hpp"

#include <stdexcept>
#include <vector>

namespace klft {

template <size_t Nc> struct TensorProduct {
  Kokkos::Array<complex_t, Nc * Nc * Nc * Nc> comp;
};

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION constexpr index_t
tensor_index(const index_t i, const index_t j, const index_t k,
             const index_t l) {
  return (((i * static_cast<index_t>(Nc)) + j) * static_cast<index_t>(Nc) + k) *
             static_cast<index_t>(Nc) +
         l;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION TensorProduct<Nc> zero_tensor_product() {
  TensorProduct<Nc> out{};
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc * Nc * Nc); ++i) {
    out.comp[i] = complex_t(0.0, 0.0);
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION TensorProduct<Nc> identity_tensor_product() {
  TensorProduct<Nc> out = zero_tensor_product<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t k = 0; k < static_cast<index_t>(Nc); ++k) {
      out.comp[tensor_index<Nc>(i, i, k, k)] = complex_t(1.0, 0.0);
    }
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION TensorProduct<Nc>
tensor_product_from_matrices(const SUN<Nc> &a, const SUN<Nc> &b) {
  TensorProduct<Nc> out = zero_tensor_product<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t j = 0; j < static_cast<index_t>(Nc); ++j) {
#pragma unroll
      for (index_t k = 0; k < static_cast<index_t>(Nc); ++k) {
#pragma unroll
        for (index_t l = 0; l < static_cast<index_t>(Nc); ++l) {
          out.comp[tensor_index<Nc>(i, j, k, l)] =
              Kokkos::conj(matrix_element(a, i, j)) *
              matrix_element(b, k, l);
        }
      }
    }
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION TensorProduct<Nc>
tensor_multiply(const TensorProduct<Nc> &a, const TensorProduct<Nc> &b) {
  TensorProduct<Nc> out = zero_tensor_product<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t j = 0; j < static_cast<index_t>(Nc); ++j) {
#pragma unroll
      for (index_t k = 0; k < static_cast<index_t>(Nc); ++k) {
#pragma unroll
        for (index_t l = 0; l < static_cast<index_t>(Nc); ++l) {
          complex_t sum(0.0, 0.0);
#pragma unroll
          for (index_t aidx = 0; aidx < static_cast<index_t>(Nc); ++aidx) {
#pragma unroll
            for (index_t bidx = 0; bidx < static_cast<index_t>(Nc); ++bidx) {
              sum += a.comp[tensor_index<Nc>(i, aidx, k, bidx)] *
                     b.comp[tensor_index<Nc>(aidx, j, bidx, l)];
            }
          }
          out.comp[tensor_index<Nc>(i, j, k, l)] = sum;
        }
      }
    }
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION TensorProduct<Nc>
tensor_scale(const TensorProduct<Nc> &a, const real_t scale) {
  TensorProduct<Nc> out = zero_tensor_product<Nc>();
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc * Nc * Nc); ++i) {
    out.comp[i] = a.comp[i] * scale;
  }
  return out;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void
tensor_add_inplace(TensorProduct<Nc> &a, const TensorProduct<Nc> &b) {
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc * Nc * Nc * Nc); ++i) {
    a.comp[i] += b.comp[i];
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION complex_t
tensor_trace_normalized(const TensorProduct<Nc> &a) {
  complex_t out(0.0, 0.0);
#pragma unroll
  for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
#pragma unroll
    for (index_t j = 0; j < static_cast<index_t>(Nc); ++j) {
      out += a.comp[tensor_index<Nc>(i, i, j, j)];
    }
  }
  return out * (1.0 / static_cast<real_t>(Nc * Nc));
}

template <size_t Nc>
using MatrixSlabView =
    Kokkos::View<SUN<Nc> *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using TensorSlabView =
    Kokkos::View<TensorProduct<Nc> *,
                 Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
void zero_matrix_view(const MatrixSlabView<Nc> &view) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "ZeroMatrixSlabs", Kokkos::RangePolicy<Exec>(0, view.extent(0)),
      KOKKOS_LAMBDA(const size_t i) { view(i) = zeroSUN<Nc>(); });
  Kokkos::fence();
}

template <size_t Nc>
void zero_tensor_view(const TensorSlabView<Nc> &view) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "ZeroTensorSlabs", Kokkos::RangePolicy<Exec>(0, view.extent(0)),
      KOKKOS_LAMBDA(const size_t i) {
        view(i) = zero_tensor_product<Nc>();
      });
  Kokkos::fence();
}

template <size_t Nc>
void add_matrix_views(const MatrixSlabView<Nc> &accum,
                      const MatrixSlabView<Nc> &sample) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "AddMatrixSlabs", Kokkos::RangePolicy<Exec>(0, accum.extent(0)),
      KOKKOS_LAMBDA(const size_t i) { accum(i) += sample(i); });
  Kokkos::fence();
}

template <size_t Nc>
void add_tensor_views(const TensorSlabView<Nc> &accum,
                      const TensorSlabView<Nc> &sample) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "AddTensorSlabs", Kokkos::RangePolicy<Exec>(0, accum.extent(0)),
      KOKKOS_LAMBDA(const size_t i) {
        auto val = accum(i);
        tensor_add_inplace<Nc>(val, sample(i));
        accum(i) = val;
      });
  Kokkos::fence();
}

template <size_t Nc>
void scale_matrix_view(const MatrixSlabView<Nc> &view, const real_t scale) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "ScaleMatrixSlabs", Kokkos::RangePolicy<Exec>(0, view.extent(0)),
      KOKKOS_LAMBDA(const size_t i) { view(i) *= scale; });
  Kokkos::fence();
}

template <size_t Nc>
void scale_tensor_view(const TensorSlabView<Nc> &view, const real_t scale) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "ScaleTensorSlabs", Kokkos::RangePolicy<Exec>(0, view.extent(0)),
      KOKKOS_LAMBDA(const size_t i) {
        view(i) = tensor_scale<Nc>(view(i), scale);
      });
  Kokkos::fence();
}

template <size_t rank, size_t Nc, class RNG> struct BuildBaseSlabMatrices {
  constexpr static const size_t time_dir = rank - 1;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  const GaugeFieldType g_in;
  MatrixSlabView<Nc> slabs;
  const HeatbathParams params;
  const index_t slab_links;
  const index_t multihit;
  const RNG rng;
  const IndexArray<rank> dimensions;
  const size_t nSpatial;

  BuildBaseSlabMatrices(const GaugeFieldType &g_in,
                        const MatrixSlabView<Nc> &slabs,
                        const HeatbathParams &params,
                        const index_t slab_links, const index_t multihit,
                        const RNG &rng, const IndexArray<rank> &dimensions)
      : g_in(g_in), slabs(slabs), params(params), slab_links(slab_links),
        multihit(multihit), rng(rng), dimensions(dimensions),
        nSpatial(spatial_volume<rank>(dimensions)) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t idx) const {
    const size_t slice = idx / nSpatial;
    const size_t spatial_lin = idx % nSpatial;
    auto site = linear_to_polyakov_origin<rank>(spatial_lin, dimensions);
    site[time_dir] = static_cast<index_t>(slice) * slab_links;

    SUN<Nc> slab = identitySUN<Nc>();
    auto generator = rng.get_state();
    for (index_t t = 0; t < slab_links; ++t) {
      const SUN<Nc> raw_link = g_in(site, time_dir);
      const SUN<Nc> link =
          (multihit > 1)
              ? multihit_link_heatbath<Nc>(
                    raw_link, g_in.staple(site, time_dir), multihit,
                    params.nOverrelax, params.beta, params.epsilon1, generator)
              : raw_link;
      slab *= link;
      site = shift_index_plus<rank>(site, time_dir, 1, dimensions);
    }
    rng.free_state(generator);
    slabs(idx) = slab;
  }
};

template <size_t rank, size_t Nc, class RNG>
MatrixSlabView<Nc> compute_base_slab_matrices(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const HeatbathParams &params, const index_t slab_links,
    const index_t multihit, const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const auto dimensions = g_in.dimensions;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nSlices =
      static_cast<size_t>(dimensions[rank - 1] / slab_links);
  MatrixSlabView<Nc> slabs("polyakov_multilevel_matrix_slabs",
                           nSlices * nSpatial);
  Kokkos::parallel_for(
      "BuildBaseSlabMatrices",
      Kokkos::RangePolicy<Exec>(0, nSlices * nSpatial),
      BuildBaseSlabMatrices<rank, Nc, RNG>(
          g_in, slabs, params, slab_links, multihit, rng, dimensions));
  Kokkos::fence();
  return slabs;
}

template <size_t rank, size_t Nc>
MatrixSlabView<Nc>
compose_matrix_slabs(const MatrixSlabView<Nc> &child_slabs,
                     const IndexArray<rank> &dimensions,
                     const index_t child_slab_links,
                     const index_t parent_slab_links) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nParentSlices =
      static_cast<size_t>(dimensions[rank - 1] / parent_slab_links);
  const size_t nChildSlices =
      static_cast<size_t>(dimensions[rank - 1] / child_slab_links);
  const size_t ratio =
      static_cast<size_t>(parent_slab_links / child_slab_links);
  MatrixSlabView<Nc> parent_slabs("polyakov_multilevel_matrix_parent",
                                  nParentSlices * nSpatial);

  Kokkos::parallel_for(
      "ComposeMatrixSlabs",
      Kokkos::RangePolicy<Exec>(0, nParentSlices * nSpatial),
      KOKKOS_LAMBDA(const size_t idx) {
        const size_t parent_slice = idx / nSpatial;
        const size_t spatial_lin = idx % nSpatial;
        SUN<Nc> slab = identitySUN<Nc>();
        for (size_t child = 0; child < ratio; ++child) {
          const size_t child_slice = parent_slice * ratio + child;
          slab *= child_slabs(child_slice * nSpatial + spatial_lin);
        }
        parent_slabs(idx) = slab;
      });
  Kokkos::fence();
  (void)nChildSlices;
  return parent_slabs;
}

template <size_t rank, size_t Nc>
TensorSlabView<Nc>
compute_tensors_from_slab_matrices(const MatrixSlabView<Nc> &slabs,
                                   const IndexArray<rank> &dimensions,
                                   const index_t slab_links,
                                   const index_t max_r) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (max_r < 2) {
    return TensorSlabView<Nc>("polyakov_multilevel_empty_tensors", 0);
  }

  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nSlices =
      static_cast<size_t>(dimensions[rank - 1] / slab_links);
  const size_t nCorr =
      static_cast<size_t>(max_r - 1) * static_cast<size_t>(rank - 1);
  TensorSlabView<Nc> tensors("polyakov_multilevel_base_tensors",
                             nCorr * nSlices * nSpatial);

  Kokkos::parallel_for(
      "BuildBaseSlabTensors",
      Kokkos::RangePolicy<Exec>(0, nCorr * nSlices * nSpatial),
      KOKKOS_LAMBDA(const size_t idx) {
        const size_t spatial_lin = idx % nSpatial;
        const size_t slice = (idx / nSpatial) % nSlices;
        const size_t corr_idx = idx / (nSpatial * nSlices);
        const index_t mu = static_cast<index_t>(corr_idx % (rank - 1));
        const index_t R = static_cast<index_t>(corr_idx / (rank - 1)) + 2;
        auto site = linear_to_polyakov_origin<rank>(spatial_lin, dimensions);
        const auto shifted = shift_index_plus<rank>(site, mu, R, dimensions);
        const size_t shifted_lin =
            polyakov_origin_to_linear<rank>(shifted, dimensions);
        tensors(idx) = tensor_product_from_matrices<Nc>(
            slabs(slice * nSpatial + spatial_lin),
            slabs(slice * nSpatial + shifted_lin));
      });
  Kokkos::fence();
  return tensors;
}

template <size_t rank, size_t Nc>
TensorSlabView<Nc>
compose_tensor_slabs(const TensorSlabView<Nc> &child_slabs,
                     const IndexArray<rank> &dimensions,
                     const index_t child_slab_links,
                     const index_t parent_slab_links,
                     const index_t max_r) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (max_r < 2) {
    return TensorSlabView<Nc>("polyakov_multilevel_empty_parent_tensors", 0);
  }

  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nParentSlices =
      static_cast<size_t>(dimensions[rank - 1] / parent_slab_links);
  const size_t nChildSlices =
      static_cast<size_t>(dimensions[rank - 1] / child_slab_links);
  const size_t ratio =
      static_cast<size_t>(parent_slab_links / child_slab_links);
  const size_t nCorr =
      static_cast<size_t>(max_r - 1) * static_cast<size_t>(rank - 1);
  TensorSlabView<Nc> parent_slabs("polyakov_multilevel_tensor_parent",
                                  nCorr * nParentSlices * nSpatial);

  Kokkos::parallel_for(
      "ComposeTensorSlabs",
      Kokkos::RangePolicy<Exec>(0, nCorr * nParentSlices * nSpatial),
      KOKKOS_LAMBDA(const size_t idx) {
        const size_t spatial_lin = idx % nSpatial;
        const size_t parent_slice = (idx / nSpatial) % nParentSlices;
        const size_t corr_idx = idx / (nSpatial * nParentSlices);
        TensorProduct<Nc> slab = identity_tensor_product<Nc>();
        for (size_t child = 0; child < ratio; ++child) {
          const size_t child_slice = parent_slice * ratio + child;
          const size_t child_idx =
              (corr_idx * nChildSlices + child_slice) * nSpatial +
              spatial_lin;
          slab = tensor_multiply<Nc>(slab, child_slabs(child_idx));
        }
        parent_slabs(idx) = slab;
      });
  Kokkos::fence();
  return parent_slabs;
}

template <size_t rank, size_t Nc, class RNG>
MatrixSlabView<Nc> PolyakovMultilevelMatrixEstimator(
    typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const HeatbathParams &heatbathParams,
    const MultilevelParams &multilevelParams, const size_t level_index,
    const index_t multihit, const RNG &rng) {
  const auto dimensions = g_in.dimensions;
  const auto level = multilevelParams.levels[level_index];
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nSlices =
      static_cast<size_t>(dimensions[rank - 1] / level.slab_links);
  MatrixSlabView<Nc> accum("polyakov_multilevel_matrix_accum",
                           nSlices * nSpatial);
  zero_matrix_view<Nc>(accum);

  for (index_t hit = 0; hit < level.updates; ++hit) {
    restricted_heatbath_sweep<rank, Nc>(g_in, heatbathParams,
                                        level.slab_links, rng);
    MatrixSlabView<Nc> sample;
    if (level_index == 0) {
      sample = compute_base_slab_matrices<rank, Nc>(
          g_in, heatbathParams, level.slab_links, multihit, rng);
    } else {
      const auto child = multilevelParams.levels[level_index - 1];
      const MatrixSlabView<Nc> child_sample =
          PolyakovMultilevelMatrixEstimator<rank, Nc>(
              g_in, heatbathParams, multilevelParams, level_index - 1,
              multihit, rng);
      sample = compose_matrix_slabs<rank, Nc>(
          child_sample, dimensions, child.slab_links, level.slab_links);
    }
    add_matrix_views<Nc>(accum, sample);
  }

  scale_matrix_view<Nc>(accum, 1.0 / static_cast<real_t>(level.updates));
  return accum;
}

template <size_t rank, size_t Nc, class RNG>
TensorSlabView<Nc> PolyakovMultilevelTensorEstimator(
    typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const HeatbathParams &heatbathParams,
    const MultilevelParams &multilevelParams, const size_t level_index,
    const index_t multihit, const index_t max_r, const RNG &rng) {
  const auto dimensions = g_in.dimensions;
  const auto level = multilevelParams.levels[level_index];
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nSlices =
      static_cast<size_t>(dimensions[rank - 1] / level.slab_links);
  const size_t nCorr =
      static_cast<size_t>(max_r - 1) * static_cast<size_t>(rank - 1);
  TensorSlabView<Nc> accum("polyakov_multilevel_tensor_accum",
                           nCorr * nSlices * nSpatial);
  zero_tensor_view<Nc>(accum);

  for (index_t hit = 0; hit < level.updates; ++hit) {
    restricted_heatbath_sweep<rank, Nc>(g_in, heatbathParams,
                                        level.slab_links, rng);
    TensorSlabView<Nc> sample;
    if (level_index == 0) {
      const MatrixSlabView<Nc> matrix_sample =
          compute_base_slab_matrices<rank, Nc>(
              g_in, heatbathParams, level.slab_links, multihit, rng);
      sample = compute_tensors_from_slab_matrices<rank, Nc>(
          matrix_sample, dimensions, level.slab_links, max_r);
    } else {
      const auto child = multilevelParams.levels[level_index - 1];
      const TensorSlabView<Nc> child_sample =
          PolyakovMultilevelTensorEstimator<rank, Nc>(
              g_in, heatbathParams, multilevelParams, level_index - 1,
              multihit, max_r, rng);
      sample = compose_tensor_slabs<rank, Nc>(
          child_sample, dimensions, child.slab_links, level.slab_links, max_r);
    }
    add_tensor_views<Nc>(accum, sample);
  }

  scale_tensor_view<Nc>(accum, 1.0 / static_cast<real_t>(level.updates));
  return accum;
}

template <size_t rank, size_t Nc, class RNG>
Kokkos::Array<real_t, 2> PolyakovLoopMultilevel(
    typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const HeatbathParams &heatbathParams,
    const MultilevelParams &multilevelParams, const index_t multihit,
    const RNG &rng) {
  using Exec = Kokkos::DefaultExecutionSpace;
  if (multilevelParams.levels.empty()) {
    throw std::runtime_error("MultilevelParams.levels must not be empty");
  }

  const auto dimensions = g_in.dimensions;
  const size_t top_index = multilevelParams.levels.size() - 1;
  const index_t top_slab_links =
      multilevelParams.levels[top_index].slab_links;
  const MatrixSlabView<Nc> top_slabs =
      PolyakovMultilevelMatrixEstimator<rank, Nc>(
          g_in, heatbathParams, multilevelParams, top_index, multihit, rng);
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nTopSlices =
      static_cast<size_t>(dimensions[rank - 1] / top_slab_links);

  real_t rep = 0.0;
  real_t imp = 0.0;
  Kokkos::parallel_reduce(
      "PolyakovLoopMultilevelReal", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t spatial_lin, real_t &lsum) {
        SUN<Nc> loop = identitySUN<Nc>();
        for (size_t slice = 0; slice < nTopSlices; ++slice) {
          loop *= top_slabs(slice * nSpatial + spatial_lin);
        }
        lsum += (trace(loop) * (1.0 / static_cast<real_t>(Nc))).real();
      },
      rep);
  Kokkos::parallel_reduce(
      "PolyakovLoopMultilevelImag", Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t spatial_lin, real_t &lsum) {
        SUN<Nc> loop = identitySUN<Nc>();
        for (size_t slice = 0; slice < nTopSlices; ++slice) {
          loop *= top_slabs(slice * nSpatial + spatial_lin);
        }
        lsum += (trace(loop) * (1.0 / static_cast<real_t>(Nc))).imag();
      },
      imp);
  if (nSpatial > 0) {
    const real_t norm = 1.0 / static_cast<real_t>(nSpatial);
    rep *= norm;
    imp *= norm;
  }
  return Kokkos::Array<real_t, 2>{rep, imp};
}

template <size_t rank, size_t Nc>
Kokkos::Array<real_t, 3> PolyakovCorrelatorMultilevelAtR(
    const TensorSlabView<Nc> &top_tensors, const index_t R,
    const index_t top_slab_links, const IndexArray<rank> &dimensions) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const size_t nSpatial = spatial_volume<rank>(dimensions);
  const size_t nTopSlices =
      static_cast<size_t>(dimensions[rank - 1] / top_slab_links);
  const real_t spatial_dirs = static_cast<real_t>(rank - 1);

  real_t rep = 0.0;
  real_t imp = 0.0;
  Kokkos::parallel_reduce(
      "PolyakovCorrelatorMultilevelReal",
      Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t spatial_lin, real_t &lsum) {
        real_t local = 0.0;
        for (index_t mu = 0; mu < static_cast<index_t>(rank - 1); ++mu) {
          const size_t corr_idx =
              static_cast<size_t>(R - 2) * static_cast<size_t>(rank - 1) +
              static_cast<size_t>(mu);
          TensorProduct<Nc> loop = identity_tensor_product<Nc>();
          for (size_t slice = 0; slice < nTopSlices; ++slice) {
            const size_t idx =
                (corr_idx * nTopSlices + slice) * nSpatial + spatial_lin;
            loop = tensor_multiply<Nc>(loop, top_tensors(idx));
          }
          local += tensor_trace_normalized<Nc>(loop).real();
        }
        lsum += local;
      },
      rep);
  Kokkos::parallel_reduce(
      "PolyakovCorrelatorMultilevelImag",
      Kokkos::RangePolicy<Exec>(0, nSpatial),
      KOKKOS_LAMBDA(const size_t spatial_lin, real_t &lsum) {
        real_t local = 0.0;
        for (index_t mu = 0; mu < static_cast<index_t>(rank - 1); ++mu) {
          const size_t corr_idx =
              static_cast<size_t>(R - 2) * static_cast<size_t>(rank - 1) +
              static_cast<size_t>(mu);
          TensorProduct<Nc> loop = identity_tensor_product<Nc>();
          for (size_t slice = 0; slice < nTopSlices; ++slice) {
            const size_t idx =
                (corr_idx * nTopSlices + slice) * nSpatial + spatial_lin;
            loop = tensor_multiply<Nc>(loop, top_tensors(idx));
          }
          local += tensor_trace_normalized<Nc>(loop).imag();
        }
        lsum += local;
      },
      imp);

  if (nSpatial > 0) {
    const real_t norm = 1.0 / (static_cast<real_t>(nSpatial) * spatial_dirs);
    rep *= norm;
    imp *= norm;
  }
  return Kokkos::Array<real_t, 3>{static_cast<real_t>(R), rep, imp};
}

template <size_t rank, size_t Nc, class RNG>
void PolyakovCorrelatorMultilevel(
    typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
    const index_t max_r, const index_t multihit,
    std::vector<Kokkos::Array<real_t, 3>> &corr_values,
    const HeatbathParams &heatbathParams,
    const MultilevelParams &multilevelParams, const RNG &rng) {
  if (max_r < 0) {
    throw std::runtime_error("polyakov_correlator_max_r must be >= 0");
  }
  const auto dimensions = g_in.dimensions;
  if (max_r > max_polyakov_correlator_r<rank>(dimensions)) {
    throw std::runtime_error(
        "polyakov_correlator_max_r exceeds the unique periodic range");
  }
  if (multilevelParams.levels.empty()) {
    throw std::runtime_error("MultilevelParams.levels must not be empty");
  }

  corr_values.clear();
  corr_values.reserve(static_cast<size_t>(max_r + 1));

  if (max_r < 2) {
    PolyakovCorrelator<rank, Nc>(g_in, max_r, 1, corr_values, heatbathParams,
                                 rng);
    return;
  }

  std::vector<Kokkos::Array<real_t, 3>> raw_values;
  PolyakovCorrelator<rank, Nc>(g_in, 1, 1, raw_values, heatbathParams, rng);
  corr_values.insert(corr_values.end(), raw_values.begin(), raw_values.end());

  const size_t top_index = multilevelParams.levels.size() - 1;
  const index_t top_slab_links =
      multilevelParams.levels[top_index].slab_links;
  const TensorSlabView<Nc> top_tensors =
      PolyakovMultilevelTensorEstimator<rank, Nc>(
          g_in, heatbathParams, multilevelParams, top_index, multihit, max_r,
          rng);
  for (index_t R = 2; R <= max_r; ++R) {
    corr_values.push_back(PolyakovCorrelatorMultilevelAtR<rank, Nc>(
        top_tensors, R, top_slab_links, dimensions));
  }
}

} // namespace klft
