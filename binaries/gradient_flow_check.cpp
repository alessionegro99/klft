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
  flow_to_target_time<rank, Nc>(cold, workspace, current_t, 0.25, 0.01, false);

  const real_t plaquette = GaugePlaquette<rank, Nc>(cold);
  const auto errors = measure_group_errors<rank, Nc>(cold);
  ok &= check_condition(Kokkos::abs(plaquette - 1.0) < 1.0e-12,
                        "cold plaquette remains one");
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

  const real_t p_original = GaugePlaquette<rank, Nc>(original);
  const real_t p_flowed = GaugePlaquette<rank, Nc>(flowed);
  ok &= check_condition(Kokkos::abs(p_original - p_flowed) < 1.0e-13,
                        "tau=0 copy reproduces original plaquette");
  return ok;
}

template <size_t rank, size_t Nc, class RNG>
bool check_monotonicity_and_group(RNG &rng) {
  bool ok = true;
  const auto dims = check_dimensions<rank>();
  auto V = make_random_gauge_field_with<rank, Nc>(dims, rng, 0.55);
  GradientFlowWorkspace<rank, Nc> workspace(dims);

  real_t current_t = 0.0;
  real_t previous_action =
      gradient_flow_action_density(GaugePlaquette<rank, Nc>(V));
  real_t previous_plaquette = GaugePlaquette<rank, Nc>(V);
  const real_t targets_tau[] = {0.25, 0.5, 1.0, 2.0};

  for (const real_t tau : targets_tau) {
    flow_to_target_time<rank, Nc>(V, workspace, current_t, tau / 8.0, 0.01,
                                  false);
    const real_t plaquette = GaugePlaquette<rank, Nc>(V);
    const real_t action = gradient_flow_action_density(plaquette);
    const auto errors = measure_group_errors<rank, Nc>(V);

    ok &= check_condition(action <= previous_action + 1.0e-10,
                          "Wilson action is monotone under flow");
    ok &= check_condition(plaquette + 1.0e-10 >= previous_plaquette,
                          "plaquette is monotone under flow");
    ok &= check_condition(errors.group_error_1 <
                              10.0 * gradient_flow_group_tolerance<Nc>(),
                          "flow preserves unitarity/modulus on random field");
    ok &= check_condition(errors.group_error_2 <
                              10.0 * gradient_flow_group_tolerance<Nc>(),
                          "flow preserves determinant on random field");

    previous_action = action;
    previous_plaquette = plaquette;
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
                                0.01, false);
  flow_to_target_time<rank, Nc>(fine, fine_workspace, t_fine, 0.125, 0.005,
                                false);

  const real_t p_coarse = GaugePlaquette<rank, Nc>(coarse);
  const real_t p_fine = GaugePlaquette<rank, Nc>(fine);
  ok &= check_condition(Kokkos::abs(p_coarse - p_fine) <
                            step_size_tolerance<Nc>(),
                        "RK3 step-size comparison at tau=1");
  return ok;
}

template <size_t rank, size_t Nc> bool run_checks() {
  RNGType rng(12345);
  bool ok = true;
  printf("Gradient-flow check for %zuD %s\n", rank,
         gradient_flow_group_name<Nc>());
  ok &= check_cold_configuration<rank, Nc>();
  ok &= check_zero_flow_consistency<rank, Nc>(rng);
  ok &= check_monotonicity_and_group<rank, Nc>(rng);
  ok &= check_step_size_dependence<rank, Nc>(rng);
  return ok;
}

} // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  const bool ok = run_checks<compiled_rank, compiled_nc>();
  Kokkos::finalize();
  return ok ? 0 : 1;
}
