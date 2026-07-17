#pragma once

#include "core/indexing.hpp"
#include "core/temporal_dirichlet.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"
#include "params/metropolis_params.hpp"
#include "partitioning/partition_table.hpp"
#include "updates/gradient_flow.hpp"

#include <Kokkos_Random.hpp>

namespace klft {

using PartitionIndexField = Kokkos::View<index_t *>;

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION size_t
partitionLinkIndex(const IndexArray<rank> &site, const index_t mu,
                   const IndexArray<rank> &dimensions) {
  size_t linear = static_cast<size_t>(site[0]);
#pragma unroll
  for (index_t d = 1; d < static_cast<index_t>(rank); ++d) {
    linear = linear * static_cast<size_t>(dimensions[d]) +
             static_cast<size_t>(site[d]);
  }
  return linear * rank + static_cast<size_t>(mu);
}

KOKKOS_FORCEINLINE_FUNCTION real_t
partitionLogAcceptance(const real_t dS, const real_t log_weight_old,
                       const real_t log_weight_new, const index_t degree_old,
                       const index_t degree_new) {
  // Hastings, Biometrika 57 (1970) 97: include q(new->old)/q(old->new).
  return -dS + log_weight_new - log_weight_old +
         Kokkos::log(static_cast<real_t>(degree_old)) -
         Kokkos::log(static_cast<real_t>(degree_new));
}

template <size_t rank, class RNG> struct InitializePartitionGaugeField {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, 2>::type;
  GaugeFieldType gauge;
  PartitionIndexField indices;
  PartitionDeviceTable table;
  RNG rng;
  bool hot;

  InitializePartitionGaugeField(const GaugeFieldType &gauge,
                                const PartitionIndexField &indices,
                                const PartitionDeviceTable &table,
                                const RNG &rng, const bool hot)
      : gauge(gauge), indices(indices), table(table), rng(rng), hot(hot) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(idcs)...};
    if (hot) {
      auto generator = rng.get_state();
      for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
        const real_t sample = generator.drand(0.0, 1.0);
        index_t lower = 0;
        index_t upper = table.size() - 1;
        while (lower < upper) {
          const index_t middle = lower + (upper - lower) / 2;
          if (sample <= table.cumulative_weights(middle)) {
            upper = middle;
          } else {
            lower = middle + 1;
          }
        }
        gauge(site, mu) = table.points(lower);
        indices(partitionLinkIndex<rank>(site, mu, gauge.dimensions)) = lower;
      }
      rng.free_state(generator);
    } else {
      for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
        gauge(site, mu) = table.points(table.cold_index);
        indices(partitionLinkIndex<rank>(site, mu, gauge.dimensions)) =
            table.cold_index;
      }
    }
  }
};

template <size_t rank, class RNG>
PartitionIndexField initializePartitionGaugeField(
    typename DeviceGaugeFieldType<rank, 2>::type &gauge,
    const PartitionDeviceTable &table, const std::string &start,
    const RNG &rng, const bool temporal_dirichlet = false) {
  size_t link_count = rank;
  IndexArray<rank> begin;
  IndexArray<rank> end;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    begin[d] = 0;
    end[d] = gauge.dimensions[d];
    link_count *= static_cast<size_t>(gauge.dimensions[d]);
  }
  PartitionIndexField indices("partition_indices", link_count);
  Kokkos::parallel_for(
      "initialize_partition_gauge_field", Policy<rank>(begin, end),
      InitializePartitionGaugeField<rank, RNG>(gauge, indices, table, rng,
                                                start == "hot"));
  Kokkos::fence();
  if (temporal_dirichlet) {
    apply_temporal_dirichlet_boundaries<rank, 2>(gauge);
  }
  return indices;
}

template <size_t rank> struct InitializePartitionIndicesFromGauge {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, 2>::type;
  GaugeFieldType gauge;
  PartitionIndexField indices;
  PartitionDeviceTable table;
  Kokkos::View<index_t> invalid_count;
  const bool temporal_dirichlet;

  InitializePartitionIndicesFromGauge(const GaugeFieldType &gauge,
                                      const PartitionIndexField &indices,
                                      const PartitionDeviceTable &table,
                                      const Kokkos::View<index_t> &invalid_count,
                                      const bool temporal_dirichlet)
      : gauge(gauge), indices(indices), table(table),
        invalid_count(invalid_count),
        temporal_dirichlet(temporal_dirichlet) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(idcs)...};
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      index_t best = table.cold_index;
      real_t best_distance = 1.0e300;
      for (index_t point = 0; point < table.size(); ++point) {
        real_t distance = 0.0;
        for (index_t component = 0; component < 4; ++component) {
          const real_t delta =
              gauge(site, mu).comp[component] - table.points(point).comp[component];
          distance += delta * delta;
        }
        if (distance < best_distance) {
          best_distance = distance;
          best = point;
        }
      }
      if (!(temporal_dirichlet && is_temporal_dirichlet_link<rank>(
                                       site, mu, gauge.dimensions)) &&
          best_distance >= 1.0e-20) {
        Kokkos::atomic_inc(&invalid_count());
      }
      indices(partitionLinkIndex<rank>(site, mu, gauge.dimensions)) = best;
    }
  }
};

template <size_t rank>
PartitionIndexField initializePartitionIndicesFromGauge(
    const typename DeviceGaugeFieldType<rank, 2>::type &gauge,
    const PartitionDeviceTable &table, const bool temporal_dirichlet) {
  size_t link_count = rank;
  IndexArray<rank> begin;
  IndexArray<rank> end;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    begin[d] = 0;
    end[d] = gauge.dimensions[d];
    link_count *= static_cast<size_t>(gauge.dimensions[d]);
  }
  PartitionIndexField indices("partition_indices", link_count);
  Kokkos::View<index_t> invalid_count("invalid_partition_restart_links");
  Kokkos::deep_copy(invalid_count, 0);
  Kokkos::parallel_for(
      "initialize_partition_indices_from_gauge", Policy<rank>(begin, end),
      InitializePartitionIndicesFromGauge<rank>(gauge, indices, table,
                                                 invalid_count,
                                                 temporal_dirichlet));
  Kokkos::fence();
  index_t host_invalid_count = 0;
  Kokkos::deep_copy(host_invalid_count, invalid_count);
  if (host_invalid_count != 0) {
    printf("Error: restarted partition configuration contains %d dynamical "
           "links outside the partition table\n",
           host_invalid_count);
    return PartitionIndexField{};
  }
  return indices;
}

template <size_t rank, class RNG> struct PartitionedMetropolisGaugeField {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, 2>::type;
  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  GaugeFieldType gauge;
  PartitionIndexField indices;
  PartitionDeviceTable table;
  ScalarFieldType nAccepted;
  RNG rng;
  MetropolisParams params;
  Kokkos::Array<index_t, rank> colors;

  PartitionedMetropolisGaugeField(
      const GaugeFieldType &gauge, const PartitionIndexField &indices,
      const PartitionDeviceTable &table, const MetropolisParams &params,
      const ScalarFieldType &nAccepted,
      const Kokkos::Array<index_t, rank> &colors, const RNG &rng)
      : gauge(gauge), indices(indices), table(table), nAccepted(nAccepted),
        rng(rng), params(params), colors(colors) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... idcs) const {
    index_t accepted_at_site = 0;
    auto generator = rng.get_state();
    const IndexArray<rank> site = index_lattice_color<rank, size_t>(
        Kokkos::Array<size_t, rank>{static_cast<size_t>(idcs)...}, colors,
        gauge.dimensions, params.temporal_dirichlet);
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      if (params.temporal_dirichlet &&
          is_temporal_dirichlet_link<rank>(site, mu, gauge.dimensions)) {
        continue;
      }
      const SU2 staple = gauge.staple(site, mu);
      const size_t link_index =
          partitionLinkIndex<rank>(site, mu, gauge.dimensions);
      for (index_t hit = 0; hit < params.nHits; ++hit) {
        const index_t old_index = indices(link_index);
        const index_t old_begin = table.offsets(old_index);
        const index_t degree_old = table.offsets(old_index + 1) - old_begin;
        index_t neighbor_offset = static_cast<index_t>(
            generator.drand(0.0, 1.0) * static_cast<real_t>(degree_old));
        if (neighbor_offset == degree_old) {
          neighbor_offset = degree_old - 1;
        }
        const index_t new_index = table.neighbors(old_begin + neighbor_offset);
        const index_t degree_new =
            table.offsets(new_index + 1) - table.offsets(new_index);
        const SU2 old_link = table.points(old_index);
        const SU2 new_link = table.points(new_index);
        const real_t dS = -(params.beta / 2.0) *
                          (trace(new_link * staple).real() -
                           trace(old_link * staple).real());
        const real_t log_alpha = partitionLogAcceptance(
            dS, table.log_weights(old_index), table.log_weights(new_index),
            degree_old, degree_new);
        const bool accept =
            log_alpha >= 0.0 ||
            generator.drand(0.0, 1.0) < Kokkos::exp(log_alpha);
        if (accept) {
          gauge(site, mu) = new_link;
          indices(link_index) = new_index;
          ++accepted_at_site;
        }
      }
    }
    nAccepted(idcs...) += static_cast<real_t>(accepted_at_site);
    rng.free_state(generator);
  }
};

template <size_t rank, class RNG>
real_t sweepPartitionedMetropolis(
    typename DeviceGaugeFieldType<rank, 2>::type &gauge,
    const PartitionIndexField &indices, const PartitionDeviceTable &table,
    const MetropolisParams &params, const RNG &rng) {
  IndexArray<rank> begin;
  IndexArray<rank> end;
  real_t proposal_count = static_cast<real_t>(rank * params.nHits);
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    begin[d] = 0;
    end[d] = lattice_color_extent<rank>(gauge.dimensions, d, 0,
                                        params.temporal_dirichlet);
    proposal_count *= static_cast<real_t>(gauge.dimensions[d]);
  }
  if (params.temporal_dirichlet) {
    proposal_count = static_cast<real_t>(
        dynamical_link_count<rank>(gauge.dimensions, true) *
        static_cast<size_t>(params.nHits));
  }
  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted(end, 0.0);
  const index_t color_count =
      lattice_color_total<rank>(gauge.dimensions, params.temporal_dirichlet);
  for (index_t color = 0; color < color_count; ++color) {
    const auto colors = decode_lattice_color<rank>(
        color, gauge.dimensions, params.temporal_dirichlet);
    auto color_end = end;
    for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
      color_end[d] = lattice_color_extent<rank>(
          gauge.dimensions, d, colors[d], params.temporal_dirichlet);
    }
    Kokkos::parallel_for(
        "partitioned_metropolis", Policy<rank>(begin, color_end),
        PartitionedMetropolisGaugeField<rank, RNG>(
            gauge, indices, table, params, nAccepted,
            colors, rng));
    Kokkos::fence();
  }
  const real_t accepted = nAccepted.sum();
  Kokkos::fence();
  return accepted / proposal_count;
}

template <size_t rank, class RNG, class GaugeFieldType>
int runPartitionedMetropolis(
    GaugeFieldType &gauge, const PartitionIndexField &indices,
    const PartitionDeviceTable &table,
    const MetropolisParams &metropolisParams,
    GaugeObservableParams &gaugeObsParams,
    const GradientFlowParams &gradientFlowParams, const RNG &rng) {
  if (metropolisParams.temporal_dirichlet) {
    validate_temporal_dirichlet_extents<rank>(gauge.dimensions,
                                               "partitioned Metropolis");
  } else {
    validate_even_extents<rank>(gauge.dimensions, "partitioned Metropolis");
  }
  gaugeObsParams.include_acceptance_rate = true;
  Kokkos::Timer timer;
  for (size_t step = 0; step < static_cast<size_t>(metropolisParams.nSweep);
       ++step) {
    timer.reset();
    const real_t acceptance = sweepPartitionedMetropolis<rank>(
        gauge, indices, table, metropolisParams, rng);
    const real_t time = timer.seconds();
    measureGaugeObservables<rank, 2>(
        gauge, metropolisParams, gaugeObsParams, step + 1, acceptance, time, rng);
    runGradientFlowMeasurements<rank, 2>(
        gauge, metropolisParams, gaugeObsParams, gradientFlowParams, step + 1,
        rng);
  }
  return 0;
}

} // namespace klft
