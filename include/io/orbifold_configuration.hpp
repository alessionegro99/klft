#pragma once

#include "io/gauge_configuration.hpp"
#include "orbifold.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace klft {

namespace orbifold_configuration_detail {

constexpr std::array<char, 8> magic{{'K', 'L', 'F', 'T', 'O', 'R', 'B', '1'}};
constexpr std::uint32_t format_version = 2;

inline bool finite_matrix(const SUN<3> &matrix) {
  for (index_t row = 0; row < 3; ++row) {
    for (index_t col = 0; col < 3; ++col) {
      const complex_t value = matrix_ref(matrix, row, col);
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
        return false;
      }
    }
  }
  return true;
}

inline bool valid_temporal_link(const SUN<3> &link) {
  const complex_t determinant =
      orbifold_determinant(link) - complex_t(1.0, 0.0);
  const real_t determinant_error = std::sqrt(
      determinant.real() * determinant.real() +
      determinant.imag() * determinant.imag());
  return gauge_configuration_detail::link_is_finite_and_unitary<3>(link) &&
         determinant_error <= 1.0e-10;
}

inline bool write_action(std::ofstream &file,
                         const OrbifoldActionParams &params) {
  return gauge_configuration_detail::write_scalar(file,
                                                   params.spatial_spacing) &&
         gauge_configuration_detail::write_scalar(file,
                                                   params.temporal_spacing) &&
         gauge_configuration_detail::write_scalar(file, params.coupling) &&
         gauge_configuration_detail::write_scalar(file,
                                                   params.scalar_mass) &&
         gauge_configuration_detail::write_scalar(file, params.u1_mass);
}

inline bool read_matching_action(std::ifstream &file,
                                 const OrbifoldActionParams &params) {
  std::array<real_t, 5> values{};
  for (real_t &value : values) {
    if (!gauge_configuration_detail::read_scalar(file, value)) {
      return false;
    }
  }
  return values[0] == params.spatial_spacing &&
         values[1] == params.temporal_spacing &&
         values[2] == params.coupling && values[3] == params.scalar_mass &&
         values[4] == params.u1_mass;
}

} // namespace orbifold_configuration_detail

// Versioned native-endian format matching gauge_configuration.hpp. Spatial
// matrices are unconstrained; temporal matrices are checked to be SU(3).
inline bool save_orbifold_configuration(const std::string &filename,
                                         const OrbifoldField &field,
                                         const OrbifoldActionParams &params) {
  params.validate();
  std::ofstream file(filename, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    std::printf("Error: could not open orbifold checkpoint '%s'\n",
                filename.c_str());
    return false;
  }

  file.write(orbifold_configuration_detail::magic.data(),
             orbifold_configuration_detail::magic.size());
  const std::uint32_t scalar_bytes = sizeof(real_t);
  const std::uint32_t rank = compiled_rank;
  bool ok = gauge_configuration_detail::write_scalar(
                file, orbifold_configuration_detail::format_version) &&
            gauge_configuration_detail::write_scalar(file, scalar_bytes) &&
            gauge_configuration_detail::write_scalar(file, rank);
  for (size_t d = 0; ok && d < compiled_rank; ++d) {
    const std::int64_t extent = field.dimensions[d];
    ok &= gauge_configuration_detail::write_scalar(file, extent);
  }
  ok &= orbifold_configuration_detail::write_action(file, params);
  const std::uint64_t nsites =
      gauge_configuration_detail::site_count<compiled_rank>(field.dimensions);
  ok &= gauge_configuration_detail::write_scalar(file, nsites);

  const auto spatial = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), field.spatial);
  const auto temporal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), field.temporal);
  for (size_t linear = 0; ok && linear < nsites; ++linear) {
    const auto site = gauge_configuration_detail::linear_to_site<compiled_rank>(
        linear, field.dimensions);
    for (index_t j = 0; ok && j < orbifold_spatial_directions; ++j) {
      const SUN<3> &link = orbifold_spatial_ref(spatial, site, j);
      ok = orbifold_configuration_detail::finite_matrix(link) &&
           gauge_configuration_detail::write_link<3>(file, link);
    }
    const SUN<3> &link = orbifold_temporal_ref(temporal, site);
    ok = ok && orbifold_configuration_detail::valid_temporal_link(link) &&
         gauge_configuration_detail::write_link<3>(file, link);
  }
  file.flush();
  ok &= static_cast<bool>(file);
  if (!ok) {
    std::printf("Error: failed writing orbifold checkpoint '%s'\n",
                filename.c_str());
  }
  return ok;
}

inline bool save_orbifold_configuration_atomic(
    const std::string &filename, const OrbifoldField &field,
    const OrbifoldActionParams &params) {
  const std::filesystem::path temporary = filename + ".tmp";
  if (!save_orbifold_configuration(temporary.string(), field, params)) {
    return false;
  }
  std::error_code error;
  std::filesystem::rename(temporary, filename, error);
  if (error) {
    std::printf("Error: could not install orbifold checkpoint '%s': %s\n",
                filename.c_str(), error.message().c_str());
    std::filesystem::remove(temporary, error);
    return false;
  }
  return true;
}

inline bool load_orbifold_configuration(const std::string &filename,
                                         OrbifoldField &field,
                                         const OrbifoldActionParams &params) {
  params.validate();
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::printf("Error: could not open orbifold checkpoint '%s'\n",
                filename.c_str());
    return false;
  }

  std::array<char, 8> magic{};
  file.read(magic.data(), magic.size());
  std::uint32_t version = 0;
  std::uint32_t scalar_bytes = 0;
  bool ok = static_cast<bool>(file) &&
            gauge_configuration_detail::read_scalar(file, version) &&
            gauge_configuration_detail::read_scalar(file, scalar_bytes);
  if (!ok || magic != orbifold_configuration_detail::magic ||
      (version != 1 && version != orbifold_configuration_detail::format_version) ||
      scalar_bytes != sizeof(real_t)) {
    std::printf("Error: incompatible orbifold checkpoint header in '%s'\n",
                filename.c_str());
    return false;
  }
  if (version == 1 && compiled_rank != 4) {
    std::printf("Error: legacy orbifold checkpoints are 4D only: '%s'\n",
                filename.c_str());
    return false;
  }
  if (version == orbifold_configuration_detail::format_version) {
    std::uint32_t rank = 0;
    if (!gauge_configuration_detail::read_scalar(file, rank) ||
        rank != compiled_rank) {
      std::printf("Error: incompatible orbifold checkpoint rank in '%s'\n",
                  filename.c_str());
      return false;
    }
  }
  for (size_t d = 0; d < compiled_rank; ++d) {
    std::int64_t extent = 0;
    if (!gauge_configuration_detail::read_scalar(file, extent) ||
        extent != field.dimensions[d]) {
      std::printf("Error: incompatible orbifold checkpoint extent in '%s'\n",
                  filename.c_str());
      return false;
    }
  }
  if (!orbifold_configuration_detail::read_matching_action(file, params)) {
    std::printf("Error: incompatible orbifold action parameters in '%s'\n",
                filename.c_str());
    return false;
  }
  const size_t expected_sites =
      gauge_configuration_detail::site_count<compiled_rank>(field.dimensions);
  std::uint64_t nsites = 0;
  if (!gauge_configuration_detail::read_scalar(file, nsites) ||
      nsites != expected_sites) {
    std::printf("Error: invalid orbifold checkpoint site count in '%s'\n",
                filename.c_str());
    return false;
  }

  auto spatial = Kokkos::create_mirror_view(field.spatial);
  auto temporal = Kokkos::create_mirror_view(field.temporal);
  for (size_t linear = 0; linear < nsites; ++linear) {
    const auto site = gauge_configuration_detail::linear_to_site<compiled_rank>(
        linear, field.dimensions);
    for (index_t j = 0; j < orbifold_spatial_directions; ++j) {
      SUN<3> &link = orbifold_spatial_ref(spatial, site, j);
      if (!gauge_configuration_detail::read_link<3>(file, link) ||
          !orbifold_configuration_detail::finite_matrix(link)) {
        std::printf("Error: invalid spatial link in checkpoint '%s'\n",
                    filename.c_str());
        return false;
      }
    }
    SUN<3> &link = orbifold_temporal_ref(temporal, site);
    if (!gauge_configuration_detail::read_link<3>(file, link) ||
        !orbifold_configuration_detail::valid_temporal_link(link)) {
      std::printf("Error: invalid temporal link in checkpoint '%s'\n",
                  filename.c_str());
      return false;
    }
  }
  char trailing = 0;
  if (file.read(&trailing, 1)) {
    std::printf("Error: trailing data in orbifold checkpoint '%s'\n",
                filename.c_str());
    return false;
  }
  Kokkos::deep_copy(field.spatial, spatial);
  Kokkos::deep_copy(field.temporal, temporal);
  Kokkos::fence();
  return true;
}

} // namespace klft
