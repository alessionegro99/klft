#pragma once

#include "core/compiled_theory.hpp"
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/clover_energy.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace klft {

template <size_t Nc> constexpr const char *gradient_flow_group_name() {
  if constexpr (Nc == 1) {
    return "U(1)";
  } else if constexpr (Nc == 2) {
    return "SU(2)";
  } else {
    return "SU(3)";
  }
}

template <size_t Nc> constexpr real_t gradient_flow_group_tolerance() {
  if constexpr (Nc == 3) {
    return 1.0e-8;
  } else {
    return 1.0e-10;
  }
}

KOKKOS_FORCEINLINE_FUNCTION real_t complex_abs(const complex_t &z) {
  return Kokkos::sqrt(z.real() * z.real() + z.imag() * z.imag());
}

template <size_t rank, size_t Nc>
typename DeviceGaugeFieldType<rank, Nc>::type
make_gauge_field_with(const IndexArray<rank> &dimensions,
                      const SUN<Nc> &init) {
  if constexpr (rank == 4) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(
        dimensions[0], dimensions[1], dimensions[2], dimensions[3], init);
  } else if constexpr (rank == 3) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(
        dimensions[0], dimensions[1], dimensions[2], init);
  } else {
    return typename DeviceGaugeFieldType<rank, Nc>::type(dimensions[0],
                                                         dimensions[1], init);
  }
}

template <size_t rank, size_t Nc, class RNG>
typename DeviceGaugeFieldType<rank, Nc>::type
make_random_gauge_field_with(const IndexArray<rank> &dimensions, RNG &rng,
                             const real_t delta) {
  if constexpr (rank == 4) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(
        dimensions[0], dimensions[1], dimensions[2], dimensions[3], rng,
        delta);
  } else if constexpr (rank == 3) {
    return typename DeviceGaugeFieldType<rank, Nc>::type(
        dimensions[0], dimensions[1], dimensions[2], rng, delta);
  } else {
    return typename DeviceGaugeFieldType<rank, Nc>::type(dimensions[0],
                                                         dimensions[1], rng,
                                                         delta);
  }
}

template <size_t rank, size_t Nc>
typename DeviceGaugeFieldType<rank, Nc>::type copy_gauge_field(
    const typename DeviceGaugeFieldType<rank, Nc>::type &g_in) {
  auto out = make_gauge_field_with<rank, Nc>(g_in.dimensions, identitySUN<Nc>());
  Kokkos::deep_copy(out.field, g_in.field);
  Kokkos::fence();
  return out;
}

template <size_t rank, size_t Nc> struct GradientFlowWorkspace {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

  GaugeFieldType W1;
  GaugeFieldType W2;
  GaugeFieldType Z0;
  GaugeFieldType Z1;
  GaugeFieldType Z2;

  explicit GradientFlowWorkspace(const IndexArray<rank> &dimensions)
      : W1(make_gauge_field_with<rank, Nc>(dimensions, identitySUN<Nc>())),
        W2(make_gauge_field_with<rank, Nc>(dimensions, identitySUN<Nc>())),
        Z0(make_gauge_field_with<rank, Nc>(dimensions, zeroSUN<Nc>())),
        Z1(make_gauge_field_with<rank, Nc>(dimensions, zeroSUN<Nc>())),
        Z2(make_gauge_field_with<rank, Nc>(dimensions, zeroSUN<Nc>())) {}
};

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
flow_force_from_link_staple_product(const SUN<Nc> &m) {
  // The existing update convention uses Re Tr[U * staple]. Matching Bonn's
  // Wilson flow therefore uses Z = 0.5 * (M^\dagger - M), with M = U * staple.
  if constexpr (Nc == 1) {
    return make_u1(complex_t(0.0, -m.comp.imag()));
  } else {
    SUN<Nc> z = (conj(m) - m) * 0.5;
    if constexpr (Nc > 1) {
      const complex_t tr = trace(z) / static_cast<real_t>(Nc);
#pragma unroll
      for (index_t i = 0; i < static_cast<index_t>(Nc); ++i) {
        if constexpr (Nc == 2) {
          z.comp[0] = 0.0;
        } else {
          matrix_ref(z, i, i) -= tr;
        }
      }
    }
    return z;
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> exp_algebra(const SUN<Nc> &a) {
  if constexpr (Nc == 1) {
    const real_t alpha = a.comp.imag();
    return make_u1(complex_t(Kokkos::cos(alpha), Kokkos::sin(alpha)));
  } else if constexpr (Nc == 2) {
    const real_t norm2 = a.comp[1] * a.comp[1] + a.comp[2] * a.comp[2] +
                         a.comp[3] * a.comp[3];
    const real_t alpha = Kokkos::sqrt(norm2);
    real_t s_over_alpha = 1.0;
    if (alpha < 1.0e-8) {
      const real_t alpha2 = alpha * alpha;
      s_over_alpha = 1.0 - alpha2 / 6.0 + alpha2 * alpha2 / 120.0;
    } else {
      s_over_alpha = Kokkos::sin(alpha) / alpha;
    }
    return make_su2(Kokkos::cos(alpha), a.comp[1] * s_over_alpha,
                    a.comp[2] * s_over_alpha, a.comp[3] * s_over_alpha);
  } else {
    const SUN<3> id = identitySUN<3>();
    const SUN<3> x = a;
    SUN<3> ris = x * 0.2 + id;
    ris = (ris * x) * 0.25 + id;
    ris = (ris * x) * (1.0 / 3.0) + id;
    ris = (ris * x) * 0.5 + id;
    ris = ris * x + id;
    return restoreSUN(static_cast<const SUN<3> &>(ris));
  }
}

template <> KOKKOS_FORCEINLINE_FUNCTION SUN<3> exp_algebra<3>(const SUN<3> &a) {
  const SUN<3> id = identitySUN<3>();
  SUN<3> x = a;
  SUN<3> ris = x * 0.2 + id;
  ris = (ris * x) * 0.25 + id;
  ris = (ris * x) * (1.0 / 3.0) + id;
  ris = (ris * x) * 0.5 + id;
  ris = ris * x + id;
  return restoreSUN(static_cast<const SUN<3> &>(ris));
}

template <size_t rank, size_t Nc> struct ComputeFlowForce {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType V;
  GaugeFieldType Z;
  const real_t dt;

  ComputeFlowForce(const GaugeFieldType &V, const GaugeFieldType &Z,
                   const real_t dt)
      : V(V), Z(Z), dt(dt) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
      const IndexArray<rank> site{static_cast<index_t>(Idcs)...};
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      const SUN<Nc> m = V(site, mu) * V.staple(site, mu);
      Z(site, mu) = flow_force_from_link_staple_product<Nc>(m) * dt;
    }
  }
};

template <size_t rank, size_t Nc>
void compute_flow_force(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V,
    typename DeviceGaugeFieldType<rank, Nc>::type &Z, const real_t dt) {
  Kokkos::parallel_for(Policy<rank>(IndexArray<rank>{}, V.dimensions),
                       ComputeFlowForce<rank, Nc>(V, Z, dt));
  Kokkos::fence();
}

template <size_t rank, size_t Nc> struct ApplyFlowStage {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType src;
  GaugeFieldType dst;
  GaugeFieldType Z0;
  GaugeFieldType Z1;
  GaugeFieldType Z2;
  const real_t c0;
  const real_t c1;
  const real_t c2;

  ApplyFlowStage(const GaugeFieldType &src, const GaugeFieldType &dst,
                 const GaugeFieldType &Z0, const GaugeFieldType &Z1,
                 const GaugeFieldType &Z2, const real_t c0, const real_t c1,
                 const real_t c2)
      : src(src), dst(dst), Z0(Z0), Z1(Z1), Z2(Z2), c0(c0), c1(c1), c2(c2) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(Idcs)...};
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      SUN<Nc> a = zeroSUN<Nc>();
      if (c0 != 0.0) {
        a += Z0(site, mu) * c0;
      }
      if (c1 != 0.0) {
        a += Z1(site, mu) * c1;
      }
      if (c2 != 0.0) {
        a += Z2(site, mu) * c2;
      }
      dst(site, mu) = exp_algebra<Nc>(a) * src(site, mu);
    }
  }
};

template <size_t rank, size_t Nc> struct ReunitarizeFlowField {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType V;
  explicit ReunitarizeFlowField(const GaugeFieldType &V) : V(V) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      auto link = V(Idcs..., mu);
      restoreSUN(link);
      V(Idcs..., mu) = link;
    }
  }
};

template <size_t rank, size_t Nc>
void reunitarize_flow_field(
    typename DeviceGaugeFieldType<rank, Nc>::type &V) {
  Kokkos::parallel_for(Policy<rank>(IndexArray<rank>{}, V.dimensions),
                       ReunitarizeFlowField<rank, Nc>(V));
  Kokkos::fence();
}

template <size_t rank, size_t Nc>
void apply_flow_stage(
    const typename DeviceGaugeFieldType<rank, Nc>::type &src,
    typename DeviceGaugeFieldType<rank, Nc>::type &dst,
    const typename DeviceGaugeFieldType<rank, Nc>::type &Z0,
    const typename DeviceGaugeFieldType<rank, Nc>::type &Z1,
    const typename DeviceGaugeFieldType<rank, Nc>::type &Z2, const real_t c0,
    const real_t c1, const real_t c2) {
  Kokkos::parallel_for(
      Policy<rank>(IndexArray<rank>{}, src.dimensions),
      ApplyFlowStage<rank, Nc>(src, dst, Z0, Z1, Z2, c0, c1, c2));
  Kokkos::fence();
}

template <size_t rank, size_t Nc>
void flow_step_rk3(typename DeviceGaugeFieldType<rank, Nc>::type &V,
                   GradientFlowWorkspace<rank, Nc> &workspace,
                   const real_t dt) {
  compute_flow_force<rank, Nc>(V, workspace.Z0, dt);
  apply_flow_stage<rank, Nc>(V, workspace.W1, workspace.Z0, workspace.Z1,
                             workspace.Z2, 0.25, 0.0, 0.0);

  compute_flow_force<rank, Nc>(workspace.W1, workspace.Z1, dt);
  apply_flow_stage<rank, Nc>(workspace.W1, workspace.W2, workspace.Z0,
                             workspace.Z1, workspace.Z2, -17.0 / 36.0,
                             8.0 / 9.0, 0.0);

  compute_flow_force<rank, Nc>(workspace.W2, workspace.Z2, dt);
  apply_flow_stage<rank, Nc>(workspace.W2, V, workspace.Z0, workspace.Z1,
                             workspace.Z2, 17.0 / 36.0, -8.0 / 9.0, 0.75);

  reunitarize_flow_field<rank, Nc>(V);
}

template <size_t rank, size_t Nc>
void flow_to_target_time(typename DeviceGaugeFieldType<rank, Nc>::type &V,
                         GradientFlowWorkspace<rank, Nc> &workspace,
                         real_t &current_t, const real_t target_t,
                         const real_t dt) {
  while (current_t + 1.0e-14 < target_t) {
    const real_t step = std::min(dt, target_t - current_t);
    flow_step_rk3<rank, Nc>(V, workspace, step);
    current_t += step;
  }
  current_t = target_t;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t algebra_link_norm(const SUN<Nc> &a) {
  if constexpr (Nc == 1) {
    return Kokkos::abs(a.comp.imag());
  } else if constexpr (Nc == 2) {
    return Kokkos::sqrt(a.comp[1] * a.comp[1] + a.comp[2] * a.comp[2] +
                        a.comp[3] * a.comp[3]);
  } else {
    real_t sum = 0.0;
#pragma unroll
    for (index_t i = 0; i < 9; ++i) {
      sum += matrix_ref(a, i / 3, i % 3).real() *
                 matrix_ref(a, i / 3, i % 3).real() +
             matrix_ref(a, i / 3, i % 3).imag() *
                 matrix_ref(a, i / 3, i % 3).imag();
    }
    return Kokkos::sqrt(sum);
  }
}

template <size_t rank>
KOKKOS_FORCEINLINE_FUNCTION IndexArray<rank>
gradient_flow_linear_to_site(size_t lin, const IndexArray<rank> &dims) {
  IndexArray<rank> site;
#pragma unroll
  for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
    const size_t extent = static_cast<size_t>(dims[d]);
    site[d] = static_cast<index_t>(lin % extent);
    lin /= extent;
  }
  return site;
}

template <size_t rank>
inline size_t gradient_flow_site_count(const IndexArray<rank> &dims) {
  size_t nsites = 1;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    nsites *= static_cast<size_t>(dims[d]);
  }
  return nsites;
}

template <size_t rank, size_t Nc> struct MaxAlgebraNorm {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType Z;
  explicit MaxAlgebraNorm(const GaugeFieldType &Z) : Z(Z) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t i,
                                              real_t &max_norm) const {
    const index_t mu = static_cast<index_t>(i % rank);
    const size_t site_index = i / rank;
    const auto site =
        gradient_flow_linear_to_site<rank>(site_index, Z.dimensions);
    const real_t norm = algebra_link_norm<Nc>(Z(site, mu));
    if (norm > max_norm) {
      max_norm = norm;
    }
  }
};

template <size_t rank, size_t Nc>
real_t max_algebra_norm(
    const typename DeviceGaugeFieldType<rank, Nc>::type &Z) {
  real_t max_norm = 0.0;
  const size_t nlinks = gradient_flow_site_count<rank>(Z.dimensions) * rank;
  Kokkos::parallel_reduce("MaxAlgebraNorm",
                          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                              0, nlinks),
                          MaxAlgebraNorm<rank, Nc>(Z),
                          Kokkos::Max<real_t>(max_norm));
  Kokkos::fence();
  return max_norm;
}

KOKKOS_FORCEINLINE_FUNCTION complex_t determinant_su3(const SUN<3> &u) {
  complex_t det(0.0, 0.0);
  det += matrix_ref(u, 0, 0) * matrix_ref(u, 1, 1) * matrix_ref(u, 2, 2);
  det += matrix_ref(u, 1, 0) * matrix_ref(u, 2, 1) * matrix_ref(u, 0, 2);
  det += matrix_ref(u, 2, 0) * matrix_ref(u, 0, 1) * matrix_ref(u, 1, 2);
  det -= matrix_ref(u, 2, 0) * matrix_ref(u, 1, 1) * matrix_ref(u, 0, 2);
  det -= matrix_ref(u, 1, 0) * matrix_ref(u, 0, 1) * matrix_ref(u, 2, 2);
  det -= matrix_ref(u, 0, 0) * matrix_ref(u, 2, 1) * matrix_ref(u, 1, 2);
  return det;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t link_unitarity_error(const SUN<Nc> &u) {
  if constexpr (Nc == 1) {
    return Kokkos::abs(complex_abs(u.comp) - 1.0);
  } else if constexpr (Nc == 2) {
    const real_t norm2 = u.comp[0] * u.comp[0] + u.comp[1] * u.comp[1] +
                         u.comp[2] * u.comp[2] + u.comp[3] * u.comp[3];
    return Kokkos::abs(norm2 - 1.0);
  } else {
    const SUN<3> check = conj(u) * u;
    real_t sum = 0.0;
#pragma unroll
    for (index_t row = 0; row < 3; ++row) {
#pragma unroll
      for (index_t col = 0; col < 3; ++col) {
        const complex_t target(row == col ? 1.0 : 0.0, 0.0);
        const complex_t diff = matrix_ref(check, row, col) - target;
        sum += diff.real() * diff.real() + diff.imag() * diff.imag();
      }
    }
    return Kokkos::sqrt(sum);
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t link_determinant_error(const SUN<Nc> &u) {
  if constexpr (Nc == 1) {
    return 0.0;
  } else if constexpr (Nc == 2) {
    const real_t det = u.comp[0] * u.comp[0] + u.comp[1] * u.comp[1] +
                       u.comp[2] * u.comp[2] + u.comp[3] * u.comp[3];
    return Kokkos::abs(det - 1.0);
  } else {
    return complex_abs(determinant_su3(u) - complex_t(1.0, 0.0));
  }
}

struct GradientFlowGroupErrors {
  real_t group_error_1;
  real_t group_error_2;
};

template <size_t rank, size_t Nc> struct MaxGroupErrors {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType V;
  explicit MaxGroupErrors(const GaugeFieldType &V) : V(V) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t i,
                                              real_t &max_error) const {
    const index_t mu = static_cast<index_t>(i % rank);
    const size_t site_index = i / rank;
    const auto site =
        gradient_flow_linear_to_site<rank>(site_index, V.dimensions);
    const real_t e1 = link_unitarity_error<Nc>(V(site, mu));
    if (e1 > max_error) {
      max_error = e1;
    }
  }
};

template <size_t rank, size_t Nc> struct MaxDeterminantError {
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  GaugeFieldType V;
  explicit MaxDeterminantError(const GaugeFieldType &V) : V(V) {}

  KOKKOS_FORCEINLINE_FUNCTION void operator()(const size_t i,
                                              real_t &max_error) const {
    const index_t mu = static_cast<index_t>(i % rank);
    const size_t site_index = i / rank;
    const auto site =
        gradient_flow_linear_to_site<rank>(site_index, V.dimensions);
    const real_t e2 = link_determinant_error<Nc>(V(site, mu));
    if (e2 > max_error) {
      max_error = e2;
    }
  }
};

template <size_t rank, size_t Nc>
GradientFlowGroupErrors measure_group_errors(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V) {
  const size_t nlinks = gradient_flow_site_count<rank>(V.dimensions) * rank;
  real_t max_error = 0.0;
  Kokkos::parallel_reduce("MaxGroupErrors",
                          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                              0, nlinks),
                          MaxGroupErrors<rank, Nc>(V),
                          Kokkos::Max<real_t>(max_error));
  Kokkos::fence();

  if constexpr (Nc == 1) {
    return GradientFlowGroupErrors{max_error, 0.0};
  } else {
    real_t max_det = 0.0;
    Kokkos::parallel_reduce("MaxDeterminantError",
                            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                                0, nlinks),
                            MaxDeterminantError<rank, Nc>(V),
                            Kokkos::Max<real_t>(max_det));
    Kokkos::fence();
    return GradientFlowGroupErrors{max_error, max_det};
  }
}

inline real_t gradient_flow_t2e(const real_t t_over_a2, const real_t ehat) {
  return t_over_a2 * t_over_a2 * ehat;
}

inline real_t gradient_flow_nan() {
  return std::numeric_limits<real_t>::quiet_NaN();
}

inline void write_gradient_flow_obs_header(std::ofstream &file) {
  file << "# t_over_a2 = t / a^2\n"
       << "# smoothing radius sqrt(8t) / a = sqrt(8 * t_over_a2)\n"
       << "# Ehat_clover = a^4 E_clover\n"
       << "# t2E_clover = (t/a^2)^2 Ehat_clover\n"
       << "# clover E(t) measured on flowed links V_mu(x,t)\n"
       << "# conf_id t_over_a2 Ehat_clover t2E_clover\n";
}

struct GradientFlowObsRow {
  size_t conf_id;
  real_t t_over_a2;
  real_t ehat_clover;
  real_t t2e_clover;
};

inline void append_gradient_flow_obs_row(const GradientFlowParams &params,
                                         const GradientFlowObsRow &row) {
  std::ofstream file(params.obs_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow file '%s'\n",
           params.obs_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.obs_filename)) {
    write_gradient_flow_obs_header(file);
  }
  file << std::setprecision(12) << row.conf_id << " " << row.t_over_a2 << " "
       << row.ehat_clover << " " << row.t2e_clover << "\n";
  file.flush();
}

inline void write_gradient_flow_t0_header(std::ofstream &file) {
  file << "# t0 solves t^2 E_clover(t) = t0_target\n"
       << "# t0_over_a2 = t0 / a^2\n"
       << "# conf_id t0_target t0_over_a2 lower_t_over_a2 upper_t_over_a2 "
          "lower_t2E_clover upper_t2E_clover\n";
}

inline void append_gradient_flow_t0_row(
    const GradientFlowParams &params, const size_t conf_id,
    const real_t lower_t_over_a2, const real_t upper_t_over_a2,
    const real_t lower_t2e_clover, const real_t upper_t2e_clover,
    const real_t t0_over_a2) {
  std::ofstream file(params.t0_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow t0 file '%s'\n",
           params.t0_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.t0_filename)) {
    write_gradient_flow_t0_header(file);
  }
  file << std::setprecision(12) << conf_id << " " << params.t0_target << " "
       << t0_over_a2 << " " << lower_t_over_a2 << " "
       << upper_t_over_a2 << " " << lower_t2e_clover << " " << upper_t2e_clover
       << "\n";
  file.flush();
}

inline bool gradient_flow_interpolate_t0(
    const real_t lower_t_over_a2, const real_t upper_t_over_a2,
    const real_t lower_t2e_clover, const real_t upper_t2e_clover,
    const real_t target, real_t &t0_over_a2) {
  const real_t lower_delta = lower_t2e_clover - target;
  const real_t upper_delta = upper_t2e_clover - target;
  const bool crossed = (lower_delta <= 0.0 && upper_delta >= 0.0) ||
                       (lower_delta >= 0.0 && upper_delta <= 0.0);
  if (!crossed) {
    return false;
  }

  const real_t denom = upper_t2e_clover - lower_t2e_clover;
  t0_over_a2 = upper_t_over_a2;
  if (Kokkos::abs(denom) > 1.0e-14) {
    t0_over_a2 = lower_t_over_a2 +
                 (target - lower_t2e_clover) *
                     (upper_t_over_a2 - lower_t_over_a2) / denom;
  }
  return true;
}

inline bool gradient_flow_is_initial_time(const real_t t_over_a2) {
  return Kokkos::abs(t_over_a2) <= 1.0e-14;
}

inline index_t gradient_flow_wilson_loop_multihit(
    const GaugeObservableParams &params, const real_t t_over_a2) {
  return gradient_flow_is_initial_time(t_over_a2)
             ? params.wilson_loop_multihit
             : static_cast<index_t>(1);
}

inline void append_gradient_flow_temporal_wloops(
    const GradientFlowParams &params, const size_t conf_id,
    const real_t t_over_a2,
    const std::vector<Kokkos::Array<real_t, 3>> &measurements) {
  std::ofstream file(params.W_temp_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow temporal Wilson-loop file "
           "'%s'\n",
           params.W_temp_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.W_temp_filename)) {
    file << "# conf_id t_over_a2 L T W_temp\n";
  }
  file << std::setprecision(12);
  for (const auto &measurement : measurements) {
    file << conf_id << " " << t_over_a2 << " " << measurement[0] << " "
         << measurement[1] << " " << measurement[2] << "\n";
  }
  file.flush();
}

inline void append_gradient_flow_planar_wloops(
    const GradientFlowParams &params, const size_t conf_id,
    const real_t t_over_a2,
    const std::vector<Kokkos::Array<real_t, 5>> &measurements) {
  std::ofstream file(params.W_mu_nu_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow planar Wilson-loop file '%s'\n",
           params.W_mu_nu_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.W_mu_nu_filename)) {
    file << "# conf_id t_over_a2 mu nu Lmu Lnu W_mu_nu\n";
  }
  file << std::setprecision(12);
  for (const auto &measurement : measurements) {
    file << conf_id << " " << t_over_a2 << " " << measurement[0] << " "
         << measurement[1] << " " << measurement[2] << " " << measurement[3]
         << " " << measurement[4] << "\n";
  }
  file.flush();
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void measure_flowed_wilson_loops(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V,
    const UpdateParams &updateParams, const GaugeObservableParams &gaugeParams,
    const GradientFlowParams &flowParams, const size_t conf_id,
    const real_t t_over_a2, const RNG &rng) {
  const index_t multihit =
      gradient_flow_wilson_loop_multihit(gaugeParams, t_over_a2);
  if (flowParams.measure_wilson_loop_temporal) {
    std::vector<Kokkos::Array<real_t, 3>> measurements;
    WilsonLoop_temporal<rank, Nc>(V, gaugeParams.W_temp_L_T_pairs,
                                  measurements, multihit, updateParams, rng);
    append_gradient_flow_temporal_wloops(flowParams, conf_id, t_over_a2,
                                         measurements);
  }

  if (flowParams.measure_wilson_loop_mu_nu) {
    std::vector<Kokkos::Array<real_t, 5>> measurements;
    for (const auto &pair_mu_nu : gaugeParams.W_mu_nu_pairs) {
      WilsonLoop_mu_nu<rank, Nc>(
          V, pair_mu_nu[0], pair_mu_nu[1], gaugeParams.W_Lmu_Lnu_pairs,
          measurements, multihit, updateParams, rng);
    }
    append_gradient_flow_planar_wloops(flowParams, conf_id, t_over_a2,
                                       measurements);
  }
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void runGradientFlowMeasurements(
    const typename DeviceGaugeFieldType<rank, Nc>::type &U,
    const UpdateParams &updateParams, const GaugeObservableParams &gaugeParams,
    const GradientFlowParams &flowParams, const size_t step, const RNG &rng) {
  if (!flowParams.enabled || gaugeParams.measurement_interval == 0 ||
      (step % gaugeParams.measurement_interval != 0)) {
    return;
  }

  auto V = copy_gauge_field<rank, Nc>(U);
  GradientFlowWorkspace<rank, Nc> workspace(V.dimensions);

  const bool write_obs = flowParams.measure_energy_clover;
  constexpr real_t time_tol = 1.0e-14;
  real_t current_t = 0.0;
  real_t previous_t0_t = 0.0;
  real_t previous_t2e_clover = 0.0;
  bool found_t0 = false;
  size_t output_index = 0;
  const real_t max_t = flowParams.t_values.back();

  auto emit_output = [&](const real_t t_over_a2, real_t ehat_clover,
                         real_t t2e_clover, bool have_clover) {
    if (write_obs) {
      if (!have_clover) {
        ehat_clover = measure_clover_energy_density<rank, Nc>(V);
        t2e_clover = gradient_flow_t2e(t_over_a2, ehat_clover);
      }
      append_gradient_flow_obs_row(
          flowParams,
          GradientFlowObsRow{step, t_over_a2, ehat_clover, t2e_clover});
    }

    measure_flowed_wilson_loops<rank, Nc>(V, updateParams, gaugeParams,
                                          flowParams, step, t_over_a2, rng);
  };

  while (output_index < flowParams.t_values.size() &&
         flowParams.t_values[output_index] <= time_tol) {
    emit_output(flowParams.t_values[output_index], gradient_flow_nan(),
                gradient_flow_nan(), false);
    ++output_index;
  }

  while (current_t + time_tol < max_t) {
    const real_t next_output_t =
        output_index < flowParams.t_values.size()
            ? flowParams.t_values[output_index]
            : max_t;
    const real_t next_stop_t = std::min(max_t, next_output_t);
    const real_t step_t = std::min(flowParams.dt, next_stop_t - current_t);

    flow_step_rk3<rank, Nc>(V, workspace, step_t);
    current_t += step_t;
    if (Kokkos::abs(current_t - next_stop_t) < time_tol) {
      current_t = next_stop_t;
    }

    real_t ehat_clover = gradient_flow_nan();
    real_t t2e_clover = gradient_flow_nan();
    bool have_clover = false;

    if (flowParams.extract_t0 && !found_t0) {
      ehat_clover = measure_clover_energy_density<rank, Nc>(V);
      t2e_clover = gradient_flow_t2e(current_t, ehat_clover);
      have_clover = true;

      real_t t0_over_a2 = 0.0;
      if (gradient_flow_interpolate_t0(
              previous_t0_t, current_t, previous_t2e_clover, t2e_clover,
              flowParams.t0_target, t0_over_a2)) {
        append_gradient_flow_t0_row(flowParams, step, previous_t0_t,
                                    current_t, previous_t2e_clover,
                                    t2e_clover, t0_over_a2);
        found_t0 = true;
      }

      previous_t0_t = current_t;
      previous_t2e_clover = t2e_clover;
    }

    while (output_index < flowParams.t_values.size() &&
           flowParams.t_values[output_index] <= current_t + time_tol) {
      const real_t target_t = flowParams.t_values[output_index];
      const bool output_has_clover =
          have_clover && Kokkos::abs(target_t - current_t) < time_tol;
      emit_output(target_t, ehat_clover, t2e_clover, output_has_clover);
      ++output_index;
    }
  }

  if (flowParams.extract_t0 && !found_t0) {
    printf("Warning: gradient-flow t0 target %.12g not reached at step %zu "
           "up to t/a^2 %.12g\n",
           flowParams.t0_target, step, flowParams.t_values.back());
  }
}

} // namespace klft
