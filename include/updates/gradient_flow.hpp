#pragma once

#include "core/compiled_theory.hpp"
#include "core/indexing.hpp"
#include "fields/field_type_traits.hpp"
#include "groups/group_ops.hpp"
#include "observables/gauge_observables.hpp"
#include "params/gradient_flow_params.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace klft {

inline constexpr real_t gradient_flow_action_tolerance = 1.0e-10;

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
  const real_t epsilon;

  ComputeFlowForce(const GaugeFieldType &V, const GaugeFieldType &Z,
                   const real_t epsilon)
      : V(V), Z(Z), epsilon(epsilon) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    const IndexArray<rank> site{static_cast<index_t>(Idcs)...};
#pragma unroll
    for (index_t mu = 0; mu < static_cast<index_t>(rank); ++mu) {
      const SUN<Nc> m = V(site, mu) * V.staple(site, mu);
      Z(site, mu) = flow_force_from_link_staple_product<Nc>(m) * epsilon;
    }
  }
};

template <size_t rank, size_t Nc>
void compute_flow_force(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V,
    typename DeviceGaugeFieldType<rank, Nc>::type &Z, const real_t epsilon) {
  Kokkos::parallel_for(Policy<rank>(IndexArray<rank>{}, V.dimensions),
                       ComputeFlowForce<rank, Nc>(V, Z, epsilon));
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
                   const real_t epsilon, const bool reunitarize = false) {
  compute_flow_force<rank, Nc>(V, workspace.Z0, epsilon);
  apply_flow_stage<rank, Nc>(V, workspace.W1, workspace.Z0, workspace.Z1,
                             workspace.Z2, 0.25, 0.0, 0.0);

  compute_flow_force<rank, Nc>(workspace.W1, workspace.Z1, epsilon);
  apply_flow_stage<rank, Nc>(workspace.W1, workspace.W2, workspace.Z0,
                             workspace.Z1, workspace.Z2, -17.0 / 36.0,
                             8.0 / 9.0, 0.0);

  compute_flow_force<rank, Nc>(workspace.W2, workspace.Z2, epsilon);
  apply_flow_stage<rank, Nc>(workspace.W2, V, workspace.Z0, workspace.Z1,
                             workspace.Z2, 17.0 / 36.0, -8.0 / 9.0, 0.75);

  if (reunitarize) {
    reunitarize_flow_field<rank, Nc>(V);
  }
}

template <size_t rank, size_t Nc>
void flow_to_target_time(typename DeviceGaugeFieldType<rank, Nc>::type &V,
                         GradientFlowWorkspace<rank, Nc> &workspace,
                         real_t &current_t, const real_t target_t,
                         const real_t epsilon, const bool reunitarize) {
  while (current_t + 1.0e-14 < target_t) {
    const real_t step = std::min(epsilon, target_t - current_t);
    flow_step_rk3<rank, Nc>(V, workspace, step, reunitarize);
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

inline real_t gradient_flow_action_density(const real_t plaquette) {
  return 1.0 - plaquette;
}

inline void write_gradient_flow_obs_header(std::ofstream &file) {
  file << "# tau = 8 t / a^2\n"
       << "# fixed tau => smoothing radius sqrt(8t) = a sqrt(tau)\n"
       << "# conf_id group beta tau t_over_a2 plaquette action_density "
          "delta_action_from_tau0 delta_action_from_previous_tau "
          "group_error_1 group_error_2\n";
}

inline void append_gradient_flow_obs_row(const GradientFlowParams &params,
                                         const size_t conf_id,
                                         const char *group_name,
                                         const real_t beta, const real_t tau,
                                         const real_t t_over_a2,
                                         const real_t plaquette,
                                         const real_t action_density,
                                         const real_t delta_tau0,
                                         const real_t delta_previous,
                                         const GradientFlowGroupErrors errors) {
  std::ofstream file(params.obs_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow file '%s'\n",
           params.obs_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.obs_filename)) {
    write_gradient_flow_obs_header(file);
  }
  file << std::setprecision(12) << conf_id << " " << group_name << " " << beta
       << " " << tau << " " << t_over_a2 << " " << plaquette << " "
       << action_density << " " << delta_tau0 << " " << delta_previous << " "
       << errors.group_error_1 << " " << errors.group_error_2 << "\n";
  file.flush();
}

inline void append_gradient_flow_temporal_wloops(
    const GradientFlowParams &params, const size_t conf_id,
    const char *group_name, const real_t beta, const real_t tau,
    const std::vector<Kokkos::Array<real_t, 3>> &measurements) {
  std::ofstream file(params.W_temp_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow temporal Wilson-loop file "
           "'%s'\n",
           params.W_temp_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.W_temp_filename)) {
    file << "# conf_id group beta tau L T W_temp\n";
  }
  file << std::setprecision(12);
  for (const auto &measurement : measurements) {
    file << conf_id << " " << group_name << " " << beta << " " << tau << " "
         << measurement[0] << " " << measurement[1] << " " << measurement[2]
         << "\n";
  }
  file.flush();
}

inline void append_gradient_flow_planar_wloops(
    const GradientFlowParams &params, const size_t conf_id,
    const char *group_name, const real_t beta, const real_t tau,
    const std::vector<Kokkos::Array<real_t, 5>> &measurements) {
  std::ofstream file(params.W_mu_nu_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open gradient-flow planar Wilson-loop file '%s'\n",
           params.W_mu_nu_filename.c_str());
    return;
  }
  if (fileNeedsHeader(params.W_mu_nu_filename)) {
    file << "# conf_id group beta tau mu nu Lmu Lnu W_mu_nu\n";
  }
  file << std::setprecision(12);
  for (const auto &measurement : measurements) {
    file << conf_id << " " << group_name << " " << beta << " " << tau << " "
         << measurement[0] << " " << measurement[1] << " " << measurement[2]
         << " " << measurement[3] << " " << measurement[4] << "\n";
  }
  file.flush();
}

template <size_t rank, size_t Nc, class UpdateParams, class RNG>
void measure_flowed_wilson_loops(
    const typename DeviceGaugeFieldType<rank, Nc>::type &V,
    const UpdateParams &updateParams, const GaugeObservableParams &gaugeParams,
    const GradientFlowParams &flowParams, const size_t conf_id,
    const real_t tau, const RNG &rng) {
  if (flowParams.measure_wilson_loop_temporal) {
    std::vector<Kokkos::Array<real_t, 3>> measurements;
    WilsonLoop_temporal<rank, Nc>(V, gaugeParams.W_temp_L_T_pairs,
                                  measurements,
                                  gaugeParams.wilson_loop_multihit,
                                  updateParams, rng);
    append_gradient_flow_temporal_wloops(
        flowParams, conf_id, gradient_flow_group_name<Nc>(), updateParams.beta,
        tau, measurements);
  }

  if (flowParams.measure_wilson_loop_mu_nu) {
    std::vector<Kokkos::Array<real_t, 5>> measurements;
    for (const auto &pair_mu_nu : gaugeParams.W_mu_nu_pairs) {
      WilsonLoop_mu_nu<rank, Nc>(
          V, pair_mu_nu[0], pair_mu_nu[1], gaugeParams.W_Lmu_Lnu_pairs,
          measurements, gaugeParams.wilson_loop_multihit, updateParams, rng);
    }
    append_gradient_flow_planar_wloops(
        flowParams, conf_id, gradient_flow_group_name<Nc>(), updateParams.beta,
        tau, measurements);
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

  real_t current_t = 0.0;
  real_t action_tau0 = 0.0;
  real_t previous_action = 0.0;
  bool have_previous = false;

  for (const real_t tau : flowParams.times_tau) {
    const real_t target_t = tau / 8.0;
    flow_to_target_time<rank, Nc>(V, workspace, current_t, target_t,
                                  flowParams.epsilon,
                                  flowParams.reunitarize);

    const real_t plaquette = GaugePlaquette<rank, Nc>(V);
    const real_t action_density = gradient_flow_action_density(plaquette);
    if (!have_previous) {
      action_tau0 = action_density;
    } else if (flowParams.check_action_monotonicity &&
               action_density > previous_action +
                                    gradient_flow_action_tolerance) {
      printf("Warning: gradient-flow action increased at step %zu, tau %.12g: "
             "%.12g -> %.12g\n",
             step, tau, previous_action, action_density);
    }

    GradientFlowGroupErrors errors{0.0, 0.0};
    if (flowParams.check_group_properties) {
      errors = measure_group_errors<rank, Nc>(V);
      const real_t tol = gradient_flow_group_tolerance<Nc>();
      if (errors.group_error_1 > tol || errors.group_error_2 > tol) {
        printf("Warning: gradient-flow group error at step %zu, tau %.12g: "
               "%.12g %.12g\n",
               step, tau, errors.group_error_1, errors.group_error_2);
      }
    }

    if (flowParams.measure_plaquette || flowParams.measure_action) {
      append_gradient_flow_obs_row(
          flowParams, step, gradient_flow_group_name<Nc>(), updateParams.beta,
          tau, target_t, plaquette, action_density,
          action_density - action_tau0,
          have_previous ? action_density - previous_action : 0.0, errors);
    }

    measure_flowed_wilson_loops<rank, Nc>(V, updateParams, gaugeParams,
                                          flowParams, step, tau, rng);

    previous_action = action_density;
    have_previous = true;
  }
}

} // namespace klft
