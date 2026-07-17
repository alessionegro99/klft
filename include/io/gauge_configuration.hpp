#pragma once

#include "core/temporal_dirichlet.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

namespace klft {

namespace gauge_configuration_detail {

constexpr std::array<char, 8> magic{{'K', 'L', 'F', 'T', 'C', 'F', 'G', '1'}};
constexpr std::uint32_t format_version = 1;

template <class T>
bool write_scalar(std::ofstream &file, const T &value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(T));
  return static_cast<bool>(file);
}

template <class T> bool read_scalar(std::ifstream &file, T &value) {
  file.read(reinterpret_cast<char *>(&value), sizeof(T));
  return static_cast<bool>(file);
}

template <size_t rank>
IndexArray<rank> linear_to_site(size_t linear,
                                const IndexArray<rank> &dimensions) {
  IndexArray<rank> site;
  for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
    const size_t extent = static_cast<size_t>(dimensions[d]);
    site[d] = static_cast<index_t>(linear % extent);
    linear /= extent;
  }
  return site;
}

template <size_t rank>
size_t site_count(const IndexArray<rank> &dimensions) {
  size_t count = 1;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    count *= static_cast<size_t>(dimensions[d]);
  }
  return count;
}

template <size_t rank, class View>
decltype(auto) host_link(View &view, const IndexArray<rank> &site,
                         const index_t mu) {
  if constexpr (rank == 2) {
    return view(site[0], site[1], mu);
  } else if constexpr (rank == 3) {
    return view(site[0], site[1], site[2], mu);
  } else {
    return view(site[0], site[1], site[2], site[3], mu);
  }
}

template <size_t Nc>
bool write_link(std::ofstream &file, const SUN<Nc> &link) {
  if constexpr (Nc == 1) {
    const real_t components[2] = {link.comp.real(), link.comp.imag()};
    file.write(reinterpret_cast<const char *>(components), sizeof(components));
  } else if constexpr (Nc == 2) {
    file.write(reinterpret_cast<const char *>(link.comp.data()),
               4 * sizeof(real_t));
  } else {
    for (index_t row = 0; row < 3; ++row) {
      for (index_t col = 0; col < 3; ++col) {
        const complex_t z = matrix_ref(link, row, col);
        const real_t components[2] = {z.real(), z.imag()};
        file.write(reinterpret_cast<const char *>(components),
                   sizeof(components));
      }
    }
  }
  return static_cast<bool>(file);
}

template <size_t Nc>
bool read_link(std::ifstream &file, SUN<Nc> &link) {
  if constexpr (Nc == 1) {
    real_t components[2];
    file.read(reinterpret_cast<char *>(components), sizeof(components));
    link = make_u1(complex_t(components[0], components[1]));
  } else if constexpr (Nc == 2) {
    file.read(reinterpret_cast<char *>(link.comp.data()),
              4 * sizeof(real_t));
  } else {
    for (index_t row = 0; row < 3; ++row) {
      for (index_t col = 0; col < 3; ++col) {
        real_t components[2];
        file.read(reinterpret_cast<char *>(components), sizeof(components));
        matrix_ref(link, row, col) = complex_t(components[0], components[1]);
      }
    }
  }
  return static_cast<bool>(file);
}

template <size_t Nc> bool link_is_finite_and_unitary(const SUN<Nc> &link) {
  constexpr real_t tolerance = 1.0e-10;
  if constexpr (Nc == 1) {
    const real_t re = link.comp.real();
    const real_t im = link.comp.imag();
    return std::isfinite(re) && std::isfinite(im) &&
           std::abs(re * re + im * im - 1.0) <= tolerance;
  } else if constexpr (Nc == 2) {
    real_t norm2 = 0.0;
    for (const real_t component : link.comp) {
      if (!std::isfinite(component)) {
        return false;
      }
      norm2 += component * component;
    }
    return std::abs(norm2 - 1.0) <= tolerance;
  } else {
    for (index_t row = 0; row < 3; ++row) {
      for (index_t col = 0; col < 3; ++col) {
        const complex_t z = matrix_ref(link, row, col);
        if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
          return false;
        }
      }
    }
    const SUN<3> unitary_check = conj(link) * link;
    real_t error2 = 0.0;
    for (index_t row = 0; row < 3; ++row) {
      for (index_t col = 0; col < 3; ++col) {
        const complex_t expected(row == col ? 1.0 : 0.0, 0.0);
        const complex_t delta = matrix_ref(unitary_check, row, col) - expected;
        error2 += delta.real() * delta.real() + delta.imag() * delta.imag();
      }
    }
    return std::sqrt(error2) <= tolerance;
  }
}

} // namespace gauge_configuration_detail

// Versioned native-endian binary format. Link components are stored explicitly
// as float64 values rather than by dumping implementation-defined structs.
template <size_t rank, size_t Nc>
bool save_gauge_configuration(
    const std::string &filename,
    const typename DeviceGaugeFieldType<rank, Nc>::type &gauge,
    const bool temporal_dirichlet) {
  if (temporal_dirichlet &&
      !temporal_dirichlet_boundaries_are_exact<rank, Nc>(gauge)) {
    printf("Error: refusing to save '%s': temporal Dirichlet links are not "
           "exact identities\n",
           filename.c_str());
    return false;
  }

  std::ofstream file(filename, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    printf("Error: could not open configuration output '%s'\n",
           filename.c_str());
    return false;
  }

  file.write(gauge_configuration_detail::magic.data(),
             gauge_configuration_detail::magic.size());
  const std::uint32_t file_rank = rank;
  const std::uint32_t file_nc = Nc;
  const std::uint32_t scalar_bytes = sizeof(real_t);
  const std::uint32_t boundary_flag = temporal_dirichlet ? 1U : 0U;
  bool ok = gauge_configuration_detail::write_scalar(
                file, gauge_configuration_detail::format_version) &&
            gauge_configuration_detail::write_scalar(file, file_rank) &&
            gauge_configuration_detail::write_scalar(file, file_nc) &&
            gauge_configuration_detail::write_scalar(file, scalar_bytes) &&
            gauge_configuration_detail::write_scalar(file, boundary_flag);
  for (index_t d = 0; ok && d < static_cast<index_t>(rank); ++d) {
    const std::int64_t extent = gauge.dimensions[d];
    ok &= gauge_configuration_detail::write_scalar(file, extent);
  }

  const size_t nsites =
      gauge_configuration_detail::site_count<rank>(gauge.dimensions);
  const std::uint64_t nlinks = nsites * rank;
  ok &= gauge_configuration_detail::write_scalar(file, nlinks);

  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gauge.field);
  for (size_t linear = 0; ok && linear < nsites; ++linear) {
    const auto site = gauge_configuration_detail::linear_to_site<rank>(
        linear, gauge.dimensions);
    for (index_t mu = 0; ok && mu < static_cast<index_t>(rank); ++mu) {
      ok &= gauge_configuration_detail::write_link<Nc>(
          file, gauge_configuration_detail::host_link<rank>(host, site, mu));
    }
  }
  file.flush();
  ok &= static_cast<bool>(file);
  if (!ok) {
    printf("Error: failed while writing configuration '%s'\n",
           filename.c_str());
  }
  return ok;
}

template <size_t rank, size_t Nc>
bool load_gauge_configuration(
    const std::string &filename,
    typename DeviceGaugeFieldType<rank, Nc>::type &gauge,
    const bool temporal_dirichlet) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    printf("Error: could not open configuration input '%s'\n",
           filename.c_str());
    return false;
  }

  std::array<char, 8> magic{};
  file.read(magic.data(), magic.size());
  std::uint32_t version = 0;
  std::uint32_t file_rank = 0;
  std::uint32_t file_nc = 0;
  std::uint32_t scalar_bytes = 0;
  std::uint32_t boundary_flag = 0;
  bool ok = static_cast<bool>(file) &&
            gauge_configuration_detail::read_scalar(file, version) &&
            gauge_configuration_detail::read_scalar(file, file_rank) &&
            gauge_configuration_detail::read_scalar(file, file_nc) &&
            gauge_configuration_detail::read_scalar(file, scalar_bytes) &&
            gauge_configuration_detail::read_scalar(file, boundary_flag);
  if (!ok || magic != gauge_configuration_detail::magic ||
      version != gauge_configuration_detail::format_version ||
      file_rank != rank || file_nc != Nc || scalar_bytes != sizeof(real_t) ||
      boundary_flag != (temporal_dirichlet ? 1U : 0U)) {
    printf("Error: incompatible or corrupt configuration header in '%s'\n",
           filename.c_str());
    return false;
  }

  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    std::int64_t extent = 0;
    if (!gauge_configuration_detail::read_scalar(file, extent) ||
        extent != gauge.dimensions[d]) {
      printf("Error: configuration '%s' has incompatible lattice extents\n",
             filename.c_str());
      return false;
    }
  }

  const size_t nsites =
      gauge_configuration_detail::site_count<rank>(gauge.dimensions);
  std::uint64_t nlinks = 0;
  if (!gauge_configuration_detail::read_scalar(file, nlinks) ||
      nlinks != nsites * rank) {
    printf("Error: configuration '%s' has an invalid link count\n",
           filename.c_str());
    return false;
  }

  auto host = Kokkos::create_mirror_view(gauge.field);
  for (size_t linear = 0; linear < nsites; ++linear) {
    const auto site = gauge_configuration_detail::linear_to_site<rank>(
        linear, gauge.dimensions);
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      auto &link = gauge_configuration_detail::host_link<rank>(host, site, mu);
      if (!gauge_configuration_detail::read_link<Nc>(file, link) ||
          !gauge_configuration_detail::link_is_finite_and_unitary<Nc>(link)) {
        printf("Error: invalid link data in configuration '%s'\n",
               filename.c_str());
        return false;
      }
    }
  }
  char trailing = 0;
  if (file.read(&trailing, 1)) {
    printf("Error: configuration '%s' contains trailing data\n",
           filename.c_str());
    return false;
  }

  Kokkos::deep_copy(gauge.field, host);
  Kokkos::fence();
  if (temporal_dirichlet &&
      !temporal_dirichlet_boundaries_are_exact<rank, Nc>(gauge)) {
    printf("Error: configuration '%s' violates temporal Dirichlet boundary "
           "conditions\n",
           filename.c_str());
    return false;
  }
  return true;
}

} // namespace klft
