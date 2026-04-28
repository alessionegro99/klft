#include "core/compiled_theory.hpp"
#include "updates/gradient_flow.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <cstdio>

using namespace klft;
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace {

bool check_condition(const bool condition, const char *message) {
  if (!condition) {
    printf("FAIL: %s\n", message);
    return false;
  }
  printf("PASS: %s\n", message);
  return true;
}

template <size_t Nc> real_t cold_clover_q_error(const SUN<Nc> &q) {
  if constexpr (Nc == 1) {
    return complex_abs(q.comp - complex_t(4.0, 0.0));
  } else if constexpr (Nc == 2) {
    return Kokkos::sqrt((q.comp[0] - 4.0) * (q.comp[0] - 4.0) +
                        q.comp[1] * q.comp[1] + q.comp[2] * q.comp[2] +
                        q.comp[3] * q.comp[3]);
  } else {
    real_t sum = 0.0;
    for (index_t row = 0; row < 3; ++row) {
      for (index_t col = 0; col < 3; ++col) {
        const complex_t target(row == col ? 4.0 : 0.0, 0.0);
        const complex_t diff = matrix_ref(q, row, col) - target;
        sum += diff.real() * diff.real() + diff.imag() * diff.imag();
      }
    }
    return Kokkos::sqrt(sum);
  }
}

template <size_t rank> IndexArray<rank> check_dimensions() {
  IndexArray<rank> dims;
  for (index_t d = 0; d < static_cast<index_t>(rank); ++d) {
    dims[d] = 4;
  }
  return dims;
}

template <size_t rank, size_t Nc>
bool check_cold_configuration() {
  bool ok = true;
  const auto dims = check_dimensions<rank>();
  auto cold = make_gauge_field_with<rank, Nc>(dims, identitySUN<Nc>());
  auto force = make_gauge_field_with<rank, Nc>(dims, zeroSUN<Nc>());

  compute_flow_force<rank, Nc>(cold, force, 1.0);
  const real_t force_norm = max_algebra_norm<rank, Nc>(force);
  ok &= check_condition(force_norm < 1.0e-12,
                        "cold configuration has negligible flow force");

  GradientFlowWorkspace<rank, Nc> workspace(dims);
  real_t current_t = 0.0;
  flow_to_target_time<rank, Nc>(cold, workspace, current_t, 0.25, 0.01);

  const IndexArray<rank> origin{};
  const real_t q_error =
      cold_clover_q_error<Nc>(clover_q_mu_nu<rank, Nc>(cold, origin, 0, 1));
  const real_t clover_energy = measure_clover_energy_density<rank, Nc>(cold);
  const auto errors = measure_group_errors<rank, Nc>(cold);
  ok &= check_condition(q_error < 1.0e-12,
                        "cold clover Q_mu_nu equals four times identity");
  ok &= check_condition(Kokkos::abs(clover_energy) < 1.0e-12,
                        "cold clover energy density is zero");
  ok &= check_condition(errors.group_error_1 < gradient_flow_group_tolerance<Nc>(),
                        "cold flow preserves unitarity/modulus");
  ok &= check_condition(errors.group_error_2 < gradient_flow_group_tolerance<Nc>(),
                        "cold flow preserves determinant");
  return ok;
}

template <size_t rank, size_t Nc, class RNG>
bool check_zero_flow_consistency(RNG &rng) {
  bool ok = true;
  const auto dims = check_dimensions<rank>();
  auto original = make_random_gauge_field_with<rank, Nc>(dims, rng, 0.35);
  auto flowed = copy_gauge_field<rank, Nc>(original);

  const real_t e_original = measure_clover_energy_density<rank, Nc>(original);
  const real_t e_flowed = measure_clover_energy_density<rank, Nc>(flowed);
  ok &= check_condition(Kokkos::abs(e_original - e_flowed) < 1.0e-13,
                        "t=0 copy reproduces original clover energy");
  return ok;
}

template <size_t rank, size_t Nc, class RNG>
bool check_clover_energy_and_group(RNG &rng) {
  bool ok = true;
  const auto dims = check_dimensions<rank>();
  auto V = make_random_gauge_field_with<rank, Nc>(dims, rng, 0.55);
  GradientFlowWorkspace<rank, Nc> workspace(dims);

  real_t current_t = 0.0;
  const real_t targets_t[] = {0.03125, 0.0625, 0.125, 0.25};

  for (const real_t target_t : targets_t) {
    flow_to_target_time<rank, Nc>(V, workspace, current_t, target_t, 0.01);
    const real_t clover_energy = measure_clover_energy_density<rank, Nc>(V);
    const auto errors = measure_group_errors<rank, Nc>(V);

    ok &= check_condition(clover_energy > -1.0e-12,
                          "clover energy density is non-negative");
    ok &= check_condition(errors.group_error_1 <
                              10.0 * gradient_flow_group_tolerance<Nc>(),
                          "flow preserves unitarity/modulus on random field");
    ok &= check_condition(errors.group_error_2 <
                              10.0 * gradient_flow_group_tolerance<Nc>(),
                          "flow preserves determinant on random field");

  }

  return ok;
}

template <size_t Nc> constexpr real_t step_size_tolerance() {
  if constexpr (Nc == 3) {
    return 5.0e-5;
  } else {
    return 1.0e-6;
  }
}

template <size_t rank, size_t Nc, class RNG>
bool check_step_size_dependence(RNG &rng) {
  bool ok = true;
  const auto dims = check_dimensions<rank>();
  auto original = make_random_gauge_field_with<rank, Nc>(dims, rng, 0.45);
  auto coarse = copy_gauge_field<rank, Nc>(original);
  auto fine = copy_gauge_field<rank, Nc>(original);
  GradientFlowWorkspace<rank, Nc> coarse_workspace(dims);
  GradientFlowWorkspace<rank, Nc> fine_workspace(dims);

  real_t t_coarse = 0.0;
  real_t t_fine = 0.0;
  flow_to_target_time<rank, Nc>(coarse, coarse_workspace, t_coarse, 0.125,
                                0.01);
  flow_to_target_time<rank, Nc>(fine, fine_workspace, t_fine, 0.125, 0.005);

  const real_t e_coarse = measure_clover_energy_density<rank, Nc>(coarse);
  const real_t e_fine = measure_clover_energy_density<rank, Nc>(fine);
  ok &= check_condition(Kokkos::abs(e_coarse - e_fine) <
                            step_size_tolerance<Nc>(),
                        "RK3 clover-energy step-size comparison at t=0.125");
  return ok;
}

bool check_t0_interpolation() {
  real_t t0_over_a2 = 0.0;
  const bool found =
      gradient_flow_interpolate_t0(0.0, 0.25, 0.1, 0.6, 0.3, t0_over_a2);
  bool ok = check_condition(found, "t0 interpolation detects a crossing");
  ok &= check_condition(Kokkos::abs(t0_over_a2 - 0.1) < 1.0e-14,
                        "t0 interpolation is linear in t/a^2");
  return ok;
}

bool check_flowed_wilson_loop_multihit_policy() {
  GaugeObservableParams params;
  params.wilson_loop_multihit = 7;

  bool ok = check_condition(
      gradient_flow_wilson_loop_multihit(params, 0.0) == 7,
      "t=0 flowed Wilson loops use configured multihit");
  ok &= check_condition(
      gradient_flow_wilson_loop_multihit(params, 0.125) == 1,
      "t>0 flowed Wilson loops use one standard hit");
  return ok;
}

template <size_t rank, size_t Nc> bool run_checks() {
  RNGType rng(12345);
  bool ok = true;
  printf("Gradient-flow check for %zuD %s\n", rank,
         gradient_flow_group_name<Nc>());
  ok &= check_cold_configuration<rank, Nc>();
  ok &= check_zero_flow_consistency<rank, Nc>(rng);
  ok &= check_clover_energy_and_group<rank, Nc>(rng);
  ok &= check_step_size_dependence<rank, Nc>(rng);
  ok &= check_t0_interpolation();
  ok &= check_flowed_wilson_loop_multihit_policy();
  return ok;
}

} // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  const bool ok = run_checks<compiled_rank, compiled_nc>();
  Kokkos::finalize();
  return ok ? 0 : 1;
}
