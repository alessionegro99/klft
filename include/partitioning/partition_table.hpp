#pragma once

#include "groups/gauge_group.hpp"

#include <Kokkos_Core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <vector>

namespace klft {

struct PartitionDeviceTable {
  Kokkos::View<SU2 *> points;
  Kokkos::View<real_t *> log_weights;
  Kokkos::View<real_t *> cumulative_weights;
  Kokkos::View<index_t *> offsets;
  Kokkos::View<index_t *> neighbors;
  index_t cold_index = 0;

  KOKKOS_INLINE_FUNCTION index_t size() const {
    return static_cast<index_t>(points.extent(0));
  }
};

inline bool loadPartitionTable(const std::string &filename,
                               PartitionDeviceTable &table) {
  try {
    const YAML::Node root = YAML::LoadFile(filename);
    if (root["format_version"].as<index_t>(0) != 1) {
      printf("Error: unsupported partition table format_version\n");
      return false;
    }
    if (root["coordinate_convention"].as<std::string>("") !=
        "klft_su2_comp_p0_p1_p2_p3") {
      printf("Error: unsupported partition coordinate convention\n");
      return false;
    }
    const auto point_nodes = root["points"];
    const auto weight_nodes = root["weights"];
    const auto graph = root["neighbors"];
    const auto offset_nodes = graph["offsets"];
    const auto neighbor_nodes = graph["indices"];
    if (!point_nodes.IsSequence() || point_nodes.size() == 0 ||
        !weight_nodes.IsSequence() || weight_nodes.size() != point_nodes.size() ||
        !offset_nodes.IsSequence() || offset_nodes.size() != point_nodes.size() + 1 ||
        !neighbor_nodes.IsSequence()) {
      printf("Error: malformed partition table arrays\n");
      return false;
    }

    const index_t n = static_cast<index_t>(point_nodes.size());
    std::vector<SU2> points(n);
    std::vector<real_t> weights(n);
    std::vector<index_t> offsets(n + 1);
    std::vector<index_t> neighbors(neighbor_nodes.size());
    std::map<std::array<long long, 4>, index_t> quantized_points;
    constexpr real_t point_tolerance = 1.0e-10;
    real_t weight_sum = 0.0;
    index_t cold_index = 0;

    for (index_t i = 0; i < n; ++i) {
      const auto node = point_nodes[i];
      if (!node.IsSequence() || node.size() != 4) {
        printf("Error: partition point %d does not have four components\n", i);
        return false;
      }
      real_t norm2 = 0.0;
      std::array<long long, 4> key{};
      for (index_t c = 0; c < 4; ++c) {
        points[i].comp[c] = node[c].as<real_t>();
        if (!std::isfinite(points[i].comp[c])) {
          printf("Error: partition point %d is not finite\n", i);
          return false;
        }
        norm2 += points[i].comp[c] * points[i].comp[c];
        key[c] = std::llround(points[i].comp[c] / point_tolerance);
      }
      if (std::abs(norm2 - 1.0) > point_tolerance) {
        printf("Error: partition point %d is not unit normalized\n", i);
        return false;
      }
      if (!quantized_points.emplace(key, i).second) {
        printf("Error: duplicate partition point %d\n", i);
        return false;
      }
      weights[i] = weight_nodes[i].as<real_t>();
      if (!std::isfinite(weights[i]) || weights[i] <= 0.0) {
        printf("Error: partition weight %d is not finite and positive\n", i);
        return false;
      }
      weight_sum += weights[i];
      if (points[i].comp[0] > points[cold_index].comp[0]) {
        cold_index = i;
      }
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      printf("Error: invalid partition weight sum\n");
      return false;
    }

    for (index_t i = 0; i <= n; ++i) {
      offsets[i] = offset_nodes[i].as<index_t>();
    }
    for (size_t k = 0; k < neighbors.size(); ++k) {
      neighbors[k] = neighbor_nodes[k].as<index_t>();
    }
    if (offsets[0] != 0 || offsets[n] != static_cast<index_t>(neighbors.size())) {
      printf("Error: invalid partition CSR endpoints\n");
      return false;
    }
    for (index_t i = 0; i < n; ++i) {
      if (offsets[i] < 0 || offsets[i] >= offsets[i + 1] ||
          offsets[i + 1] > static_cast<index_t>(neighbors.size())) {
        printf("Error: partition vertex %d has an invalid or empty neighbor row\n", i);
        return false;
      }
      index_t previous = -1;
      for (index_t k = offsets[i]; k < offsets[i + 1]; ++k) {
        const index_t j = neighbors[k];
        if (j < 0 || j >= n || j == i || j <= previous) {
          printf("Error: invalid neighbor in partition row %d\n", i);
          return false;
        }
        previous = j;
      }
    }
    for (index_t i = 0; i < n; ++i) {
      for (index_t k = offsets[i]; k < offsets[i + 1]; ++k) {
        const index_t j = neighbors[k];
        if (!std::binary_search(neighbors.begin() + offsets[j],
                                neighbors.begin() + offsets[j + 1], i)) {
          printf("Error: partition edge %d -> %d is not reciprocal\n", i, j);
          return false;
        }
      }
    }
    std::vector<bool> visited(n, false);
    std::queue<index_t> pending;
    visited[0] = true;
    pending.push(0);
    index_t visited_count = 0;
    while (!pending.empty()) {
      const index_t i = pending.front();
      pending.pop();
      ++visited_count;
      for (index_t k = offsets[i]; k < offsets[i + 1]; ++k) {
        const index_t j = neighbors[k];
        if (!visited[j]) {
          visited[j] = true;
          pending.push(j);
        }
      }
    }
    if (visited_count != n) {
      printf("Error: partition neighbor graph is disconnected\n");
      return false;
    }

    const std::string kind = root["kind"].as<std::string>("");
    const index_t parameter = root["parameter"].as<index_t>(0);
    if (kind != "linear" && kind != "fibonacci") {
      printf("Error: partition kind must be linear or fibonacci\n");
      return false;
    }
    if ((kind == "linear" &&
         n != 8 * (parameter * parameter * parameter + 2 * parameter) / 3) ||
        (kind == "fibonacci" && n != parameter)) {
      printf("Error: partition table point count does not match parameter\n");
      return false;
    }
    if (kind == "linear") {
      for (index_t i = 0; i < n; ++i) {
        std::array<long long, 4> antipode{};
        for (index_t c = 0; c < 4; ++c) {
          antipode[c] = std::llround(-points[i].comp[c] / point_tolerance);
        }
        const auto partner_it = quantized_points.find(antipode);
        if (partner_it == quantized_points.end()) {
          printf("Error: linear partition table is not center symmetric\n");
          return false;
        }
        const index_t partner = partner_it->second;
        if (std::abs(weights[i] - weights[partner]) >
            point_tolerance * std::max(weights[i], weights[partner])) {
          printf("Error: antipodal linear points have different weights\n");
          return false;
        }
        for (index_t k = offsets[i]; k < offsets[i + 1]; ++k) {
          std::array<long long, 4> neighbor_antipode{};
          for (index_t c = 0; c < 4; ++c) {
            neighbor_antipode[c] = std::llround(
                -points[neighbors[k]].comp[c] / point_tolerance);
          }
          const index_t partner_neighbor =
              quantized_points.find(neighbor_antipode)->second;
          if (!std::binary_search(neighbors.begin() + offsets[partner],
                                  neighbors.begin() + offsets[partner + 1],
                                  partner_neighbor)) {
            printf("Error: linear neighbor graph violates center symmetry\n");
            return false;
          }
        }
      }
    }

    table.points = Kokkos::View<SU2 *>("partition_points", n);
    table.log_weights = Kokkos::View<real_t *>("partition_log_weights", n);
    table.cumulative_weights =
        Kokkos::View<real_t *>("partition_cumulative_weights", n);
    table.offsets = Kokkos::View<index_t *>("partition_offsets", n + 1);
    table.neighbors =
        Kokkos::View<index_t *>("partition_neighbors", neighbors.size());
    table.cold_index = cold_index;

    auto h_points = Kokkos::create_mirror_view(table.points);
    auto h_log_weights = Kokkos::create_mirror_view(table.log_weights);
    auto h_cumulative = Kokkos::create_mirror_view(table.cumulative_weights);
    auto h_offsets = Kokkos::create_mirror_view(table.offsets);
    auto h_neighbors = Kokkos::create_mirror_view(table.neighbors);
    real_t cumulative = 0.0;
    for (index_t i = 0; i < n; ++i) {
      h_points(i) = points[i];
      h_log_weights(i) = std::log(weights[i]);
      cumulative += weights[i] / weight_sum;
      h_cumulative(i) = i + 1 == n ? 1.0 : cumulative;
      h_offsets(i) = offsets[i];
    }
    h_offsets(n) = offsets[n];
    for (size_t k = 0; k < neighbors.size(); ++k) {
      h_neighbors(k) = neighbors[k];
    }
    Kokkos::deep_copy(table.points, h_points);
    Kokkos::deep_copy(table.log_weights, h_log_weights);
    Kokkos::deep_copy(table.cumulative_weights, h_cumulative);
    Kokkos::deep_copy(table.offsets, h_offsets);
    Kokkos::deep_copy(table.neighbors, h_neighbors);
    Kokkos::fence();
    return true;
  } catch (const YAML::Exception &e) {
    printf("Error parsing partition table '%s': %s\n", filename.c_str(), e.what());
    return false;
  } catch (const std::exception &e) {
    printf("Error loading partition table '%s': %s\n", filename.c_str(), e.what());
    return false;
  }
}

} // namespace klft
