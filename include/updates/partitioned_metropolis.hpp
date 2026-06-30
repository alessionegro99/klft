#pragma once

#include "core/indexing.hpp"
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
    const RNG &rng) {
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
  Kokkos::Array<bool, rank> oddeven;

  PartitionedMetropolisGaugeField(
      const GaugeFieldType &gauge, const PartitionIndexField &indices,
      const PartitionDeviceTable &table, const MetropolisParams &params,
      const ScalarFieldType &nAccepted,
      const Kokkos::Array<bool, rank> &oddeven, const RNG &rng)
      : gauge(gauge), indices(indices), table(table), nAccepted(nAccepted),
        rng(rng), params(params), oddeven(oddeven) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... idcs) const {
    index_t accepted_at_site = 0;
    auto generator = rng.get_state();
    const IndexArray<rank> site = index_odd_even<rank, size_t>(
        Kokkos::Array<size_t, rank>{static_cast<size_t>(idcs)...}, oddeven);
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
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
    end[d] = gauge.dimensions[d] / 2;
    proposal_count *= static_cast<real_t>(gauge.dimensions[d]);
  }
  using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
  ScalarFieldType nAccepted(end, 0.0);
  for (index_t color = 0; color < (1 << rank); ++color) {
    Kokkos::parallel_for(
        "partitioned_metropolis", Policy<rank>(begin, end),
        PartitionedMetropolisGaugeField<rank, RNG>(
            gauge, indices, table, params, nAccepted,
            oddeven_array<rank>(color), rng));
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
  validate_even_extents<rank>(gauge.dimensions, "partitioned Metropolis");
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
