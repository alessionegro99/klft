#include "core/compiled_theory.hpp"
#include "observables/retrace.hpp"
#include "partitioning/partition_table.hpp"
#include "updates/partitioned_metropolis.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace {

template <size_t rank, class GaugeFieldType>
bool linksMatchTable(const GaugeFieldType &gauge,
                     const klft::PartitionIndexField &indices,
                     const klft::PartitionDeviceTable &table) {
  const auto host_field = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gauge.field);
  const auto host_indices = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), indices);
  const auto host_points = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), table.points);
  size_t link = 0;
  auto equal_point = [&](const klft::SU2 &point, const klft::index_t index) {
    for (klft::index_t c = 0; c < 4; ++c) {
      if (point.comp[c] != host_points(index).comp[c]) {
        return false;
      }
    }
    return true;
  };
  if constexpr (rank == 2) {
    for (klft::index_t i0 = 0; i0 < gauge.dimensions[0]; ++i0)
      for (klft::index_t i1 = 0; i1 < gauge.dimensions[1]; ++i1)
        for (klft::index_t mu = 0; mu < 2; ++mu, ++link)
          if (!equal_point(host_field(i0, i1, mu), host_indices(link)))
            return false;
  } else if constexpr (rank == 3) {
    for (klft::index_t i0 = 0; i0 < gauge.dimensions[0]; ++i0)
      for (klft::index_t i1 = 0; i1 < gauge.dimensions[1]; ++i1)
        for (klft::index_t i2 = 0; i2 < gauge.dimensions[2]; ++i2)
          for (klft::index_t mu = 0; mu < 3; ++mu, ++link)
            if (!equal_point(host_field(i0, i1, i2, mu), host_indices(link)))
              return false;
  } else {
    for (klft::index_t i0 = 0; i0 < gauge.dimensions[0]; ++i0)
      for (klft::index_t i1 = 0; i1 < gauge.dimensions[1]; ++i1)
        for (klft::index_t i2 = 0; i2 < gauge.dimensions[2]; ++i2)
          for (klft::index_t i3 = 0; i3 < gauge.dimensions[3]; ++i3)
            for (klft::index_t mu = 0; mu < 4; ++mu, ++link)
              if (!equal_point(host_field(i0, i1, i2, i3, mu),
                               host_indices(link)))
                return false;
  }
  return true;
}

bool checkDetailedBalance(const klft::PartitionDeviceTable &table) {
  const auto log_weights = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), table.log_weights);
  const auto offsets = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), table.offsets);
  const auto neighbors = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), table.neighbors);
  for (klft::index_t i = 0; i < table.size(); ++i) {
    const klft::index_t degree_i = offsets(i + 1) - offsets(i);
    const double energy_i = 0.013 * static_cast<double>(i);
    for (klft::index_t k = offsets(i); k < offsets(i + 1); ++k) {
      const klft::index_t j = neighbors(k);
      const klft::index_t degree_j = offsets(j + 1) - offsets(j);
      const double energy_j = 0.013 * static_cast<double>(j);
      const double log_alpha_ij = klft::partitionLogAcceptance(
          energy_j - energy_i, log_weights(i), log_weights(j), degree_i,
          degree_j);
      const double log_alpha_ji = klft::partitionLogAcceptance(
          energy_i - energy_j, log_weights(j), log_weights(i), degree_j,
          degree_i);
      const double flux_ij = std::exp(-energy_i + log_weights(i)) /
                             degree_i * std::min(1.0, std::exp(log_alpha_ij));
      const double flux_ji = std::exp(-energy_j + log_weights(j)) /
                             degree_j * std::min(1.0, std::exp(log_alpha_ji));
      if (std::abs(flux_ij - flux_ji) >
          2.0e-13 * std::max(flux_ij, flux_ji)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  Kokkos::initialize(argc, argv);
  int result = 0;
  {
    const auto convention_test = klft::make_su2(0.1, 0.4, 0.3, 0.2);
    const auto a = klft::matrix_element(convention_test, 0, 0);
    const auto b = klft::matrix_element(convention_test, 0, 1);
    if (a.real() != 0.1 || a.imag() != 0.2 || b.real() != 0.3 ||
        b.imag() != 0.4) {
      printf("SU(2) coordinate-convention check failed\n");
      result = 1;
    }
    klft::PartitionDeviceTable table;
    if (!klft::loadPartitionTable(argv[1], table)) {
      result = 1;
    } else if (!checkDetailedBalance(table)) {
      printf("Detailed-balance check failed\n");
      result = 1;
    } else {
      RNG rng(12345);
      auto gauge = klft::make_identity_gauge_field<klft::compiled_rank, 2>(
          2, 2, 2, 2);
      auto indices = klft::initializePartitionGaugeField<klft::compiled_rank>(
          gauge, table, "cold", rng);
      const double retrace_u2 =
          klft::RetraceU2_links_avg<klft::compiled_rank, 2>(gauge);
      if (std::abs(retrace_u2 - 1.0) > 2.0e-14 ||
          !linksMatchTable<klft::compiled_rank>(gauge, indices, table)) {
        printf("Cold initialization check failed\n");
        result = 1;
      }
      klft::MetropolisParams params;
      params.beta = 0.7;
      params.nHits = 3;
      if (klft::sweepPartitionedMetropolis<klft::compiled_rank>(
              gauge, indices, table, params, rng) < 0.0 ||
          !linksMatchTable<klft::compiled_rank>(gauge, indices, table)) {
        printf("Partition-invariance check failed\n");
        result = 1;
      }
    }
  }
  Kokkos::finalize();
  return result;
}
