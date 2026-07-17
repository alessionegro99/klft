#include "core/compiled_theory.hpp"
#include "core/temporal_dirichlet.hpp"
#include "observables/dirichlet_action.hpp"
#include "updates/heatbath.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace klft;
using RNG = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
using Gauge = typename DeviceGaugeFieldType<3, 2>::type;

struct Estimate {
  real_t mean = 0.0;
  real_t variance = 0.0;
  real_t tau_int = 0.5;
  real_t error = 0.0;
  real_t effective_samples = 0.0;
};

int failures = 0;

void check(const bool condition, const char *message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    ++failures;
  }
}

// Geyer's initial-positive-sequence estimator: adjacent autocovariance pairs
// are accumulated until their sum first becomes non-positive.
// C. J. Geyer, Statistical Science 7 (1992) 473, DOI 10.1214/ss/1177011137.
Estimate estimate_correlated_mean(const std::vector<real_t> &values) {
  Estimate result;
  if (values.empty()) {
    return result;
  }
  const size_t n = values.size();
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<real_t>(n);
  if (n == 1) {
    result.effective_samples = 1.0;
    return result;
  }
  std::vector<real_t> centered(n);
  for (size_t i = 0; i < n; ++i) {
    centered[i] = values[i] - result.mean;
    result.variance += centered[i] * centered[i];
  }
  result.variance /= static_cast<real_t>(n - 1);
  if (result.variance == 0.0 || n < 4) {
    result.error = 0.0;
    result.effective_samples = static_cast<real_t>(n);
    return result;
  }

  const real_t gamma0 =
      std::inner_product(centered.begin(), centered.end(), centered.begin(),
                         0.0) /
      static_cast<real_t>(n);
  real_t correlation_sum = 0.0;
  for (size_t even_lag = 1; even_lag + 1 < n / 2; even_lag += 2) {
    real_t gamma_even = 0.0;
    real_t gamma_odd = 0.0;
    for (size_t i = 0; i + even_lag < n; ++i) {
      gamma_even += centered[i] * centered[i + even_lag];
    }
    for (size_t i = 0; i + even_lag + 1 < n; ++i) {
      gamma_odd += centered[i] * centered[i + even_lag + 1];
    }
    gamma_even /= static_cast<real_t>(n - even_lag);
    gamma_odd /= static_cast<real_t>(n - even_lag - 1);
    if (gamma_even + gamma_odd <= 0.0) {
      break;
    }
    correlation_sum += (gamma_even + gamma_odd) / gamma0;
  }
  result.tau_int = std::max(real_t(0.5), real_t(0.5) + correlation_sum);
  result.error =
      std::sqrt(result.variance * 2.0 * result.tau_int /
                static_cast<real_t>(n));
  result.effective_samples =
      static_cast<real_t>(n) / (2.0 * result.tau_int);
  return result;
}

real_t z_distance(const Estimate &left, const Estimate &right) {
  const real_t error =
      std::sqrt(left.error * left.error + right.error * right.error);
  return error > 0.0 ? std::abs(left.mean - right.mean) / error : 0.0;
}

Gauge make_cold(const index_t nx, const index_t ny,
                const index_t slab_thickness) {
  auto gauge = make_identity_gauge_field<3, 2>(nx, ny, slab_thickness + 1, 1);
  apply_temporal_dirichlet_boundaries<3, 2>(gauge);
  return gauge;
}

Gauge make_hot(const index_t nx, const index_t ny,
               const index_t slab_thickness, const uint64_t seed) {
  RNG rng(seed);
  auto gauge =
      make_hot_gauge_field<3, 2>(nx, ny, slab_thickness + 1, 1, rng);
  apply_temporal_dirichlet_boundaries<3, 2>(gauge);
  return gauge;
}

real_t slab_plaquette_density(const Gauge &gauge, const real_t beta) {
  const index_t nt = slab_thickness_in_links<3>(gauge.dimensions);
  const real_t volume = static_cast<real_t>(gauge.dimensions[0]) *
                        static_cast<real_t>(gauge.dimensions[1]);
  const real_t dynamical_plaquettes = volume * (3.0 * nt - 1.0);
  const real_t action = DirichletSlabWilsonAction<2>(gauge, beta) -
                        FixedBoundarySpatialActionConstant<2>(gauge, beta);
  return -action / (beta * dynamical_plaquettes);
}

real_t direct_spin_correlator(const SlabHolonomyField<2> &spins,
                              const index_t separation) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        spins);
  const index_t nx = static_cast<index_t>(spins.extent(0));
  const index_t ny = static_cast<index_t>(spins.extent(1));
  real_t sum = 0.0;
  for (index_t x = 0; x < nx; ++x) {
    for (index_t y = 0; y < ny; ++y) {
      sum += trace(host(x, y) * conj(host((x + separation) % nx, y))).real() /
             2.0;
      sum += trace(host(x, y) * conj(host(x, (y + separation) % ny))).real() /
             2.0;
    }
  }
  return sum / (2.0 * static_cast<real_t>(nx * ny));
}

struct Chain {
  Gauge gauge;
  RNG rng;
  std::vector<real_t> thermalization_history;

  Chain(Gauge &&gauge, const uint64_t seed)
      : gauge(std::move(gauge)), rng(seed) {}
};

HeatbathParams heatbath_params(const index_t nx, const index_t ny,
                               const index_t slab_thickness,
                               const real_t beta) {
  HeatbathParams params;
  params.L0 = nx;
  params.L1 = ny;
  params.L2 = slab_thickness + 1;
  params.beta = beta;
  params.nOverrelax = 2;
  params.temporal_dirichlet = true;
  return params;
}

void advance(Chain &chain, const HeatbathParams &params, const index_t sweeps,
             const bool record_history) {
  for (index_t sweep = 0; sweep < sweeps; ++sweep) {
    full_heatbath_sweep<3, 2>(chain.gauge, params, chain.rng);
    if (record_history) {
      chain.thermalization_history.push_back(
          params.temporal_dirichlet
              ? slab_plaquette_density(chain.gauge, params.beta)
              : GaugePlaquette<3, 2>(chain.gauge));
    }
  }
}

bool segment_stationary(const std::vector<real_t> &history) {
  const size_t n = history.size();
  if (n < 400) {
    return false;
  }
  const size_t quarter = n / 4;
  const std::vector<real_t> previous(history.end() - 2 * quarter,
                                     history.end() - quarter);
  const std::vector<real_t> latest(history.end() - quarter, history.end());
  const Estimate previous_estimate = estimate_correlated_mean(previous);
  const Estimate latest_estimate = estimate_correlated_mean(latest);
  return latest_estimate.effective_samples >= 20.0 &&
         z_distance(previous_estimate, latest_estimate) < 4.0;
}

bool thermalize_hot_and_cold(Chain &cold, Chain &hot,
                             const HeatbathParams &params) {
  constexpr index_t batch = 100;
  constexpr index_t maximum = 4000;
  for (index_t sweeps = batch; sweeps <= maximum; sweeps += batch) {
    advance(cold, params, batch, true);
    advance(hot, params, batch, true);
    if (!segment_stationary(cold.thermalization_history) ||
        !segment_stationary(hot.thermalization_history)) {
      continue;
    }
    const size_t window = cold.thermalization_history.size() / 4;
    const std::vector<real_t> cold_tail(cold.thermalization_history.end() -
                                           static_cast<long>(window),
                                       cold.thermalization_history.end());
    const std::vector<real_t> hot_tail(hot.thermalization_history.end() -
                                          static_cast<long>(window),
                                      hot.thermalization_history.end());
    if (z_distance(estimate_correlated_mean(cold_tail),
                   estimate_correlated_mean(hot_tail)) < 4.0) {
      std::printf("thermalized after %d sweeps\n", sweeps);
      return true;
    }
  }
  return false;
}

std::vector<real_t> collect_density(Chain &chain,
                                    const HeatbathParams &params,
                                    const size_t minimum_effective_samples) {
  std::vector<real_t> values;
  constexpr index_t batch = 200;
  constexpr index_t maximum = 5000;
  for (index_t sweeps = 0; sweeps < maximum; sweeps += batch) {
    for (index_t i = 0; i < batch; ++i) {
      full_heatbath_sweep<3, 2>(chain.gauge, params, chain.rng);
      values.push_back(slab_plaquette_density(chain.gauge, params.beta));
    }
    if (values.size() >= 400 &&
        estimate_correlated_mean(values).effective_samples >=
            static_cast<real_t>(minimum_effective_samples)) {
      break;
    }
  }
  return values;
}

void check_hot_cold_and_symmetry() {
  constexpr index_t nx = 4;
  constexpr index_t ny = 4;
  constexpr index_t nt = 4;
  constexpr real_t beta = 1.8;
  const auto params = heatbath_params(nx, ny, nt, beta);
  Chain cold(make_cold(nx, ny, nt), 200001);
  Chain hot(make_hot(nx, ny, nt, 200002), 200003);
  check(thermalize_hot_and_cold(cold, hot, params),
        "hot and cold histories reach a common stationary regime");

  const auto cold_values = collect_density(cold, params, 80);
  const auto hot_values = collect_density(hot, params, 80);
  const Estimate cold_estimate = estimate_correlated_mean(cold_values);
  const Estimate hot_estimate = estimate_correlated_mean(hot_values);
  std::printf("hot/cold plaquette: %.8f(%.2g), %.8f(%.2g), tau %.2f %.2f\n",
              hot_estimate.mean, hot_estimate.error, cold_estimate.mean,
              cold_estimate.error, hot_estimate.tau_int,
              cold_estimate.tau_int);
  check(cold_estimate.effective_samples >= 80.0 &&
            hot_estimate.effective_samples >= 80.0,
        "hot/cold comparison has autocorrelation-adjusted sample size");
  check(z_distance(cold_estimate, hot_estimate) < 5.0,
        "hot and cold plaquettes agree within correlated errors");

  std::vector<real_t> spatial_reflection;
  std::vector<real_t> temporal_reflection;
  std::vector<real_t> orientation_relation;
  std::vector<real_t> correlator_symmetry;
  for (index_t measurement = 0; measurement < 1200; ++measurement) {
    full_heatbath_sweep<3, 2>(hot.gauge, params, hot.rng);
    const auto profile = MeasureDirichletPlaquetteProfiles<2>(hot.gauge);
    spatial_reflection.push_back(profile.spatial[1][1] -
                                 profile.spatial[nt - 1][1]);
    temporal_reflection.push_back(profile.temporal[0][1] -
                                  profile.temporal[nt - 1][1]);
    const auto holonomy = MeasureDirichletHolonomy<2>(hot.gauge, 1);
    orientation_relation.push_back(
        holonomy.correlator[1].legacy_double_trace -
        holonomy.correlator[1].chiral_correlator);
    const auto spins = SlabHolonomy<2>(hot.gauge);
    correlator_symmetry.push_back(direct_spin_correlator(spins, 1) -
                                  direct_spin_correlator(spins, nx - 1));
  }
  for (const auto &[series, description] :
       std::vector<std::pair<const std::vector<real_t> *, const char *>>{
           {&spatial_reflection, "spatial plaquette reflection profile"},
           {&temporal_reflection, "temporal plaquette reflection profile"},
           {&orientation_relation, "global-orientation group average"}}) {
    const Estimate estimate = estimate_correlated_mean(*series);
    const real_t z = estimate.error > 0.0
                         ? std::abs(estimate.mean) / estimate.error
                         : 0.0;
    std::printf("%s: %.4g +- %.3g, tau %.2f, z %.2f\n", description,
                estimate.mean, estimate.error, estimate.tau_int, z);
    check(estimate.effective_samples >= 25.0,
          "symmetry test has autocorrelation-adjusted samples");
    check(z < 5.0, description);
  }
  const Estimate correlator_symmetry_estimate =
      estimate_correlated_mean(correlator_symmetry);
  check(std::abs(correlator_symmetry_estimate.mean) < 2.0e-13,
        "C_G(R)=C_G(Ns-R) before half-range reduction");
}

real_t periodic_spatial_slice(const Gauge &gauge, const index_t t) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        gauge.field);
  real_t sum = 0.0;
  for (index_t x = 0; x < gauge.dimensions[0]; ++x) {
    for (index_t y = 0; y < gauge.dimensions[1]; ++y) {
      sum += dirichlet_observable_detail::plaquette_at<2>(
          host, IndexArray<3>{x, y, t}, 0, 1, gauge.dimensions);
    }
  }
  return sum /
         static_cast<real_t>(gauge.dimensions[0] * gauge.dimensions[1]);
}

void check_long_slab_bulk_recovery() {
  constexpr index_t nx = 4;
  constexpr index_t ny = 4;
  constexpr index_t nt = 12;
  constexpr real_t beta = 1.8;
  auto dirichlet_params = heatbath_params(nx, ny, nt, beta);
  Chain slab(make_hot(nx, ny, nt, 300001), 300002);
  Chain slab_reference(make_cold(nx, ny, nt), 300003);
  check(thermalize_hot_and_cold(slab_reference, slab, dirichlet_params),
        "long Dirichlet slab thermalizes from hot/cold starts");

  HeatbathParams periodic_params = heatbath_params(nx, ny, nt, beta);
  periodic_params.L2 = nt;
  periodic_params.temporal_dirichlet = false;
  RNG periodic_init_rng(300004);
  auto periodic_gauge = make_hot_gauge_field<3, 2>(nx, ny, nt, 1,
                                                    periodic_init_rng);
  Chain periodic(std::move(periodic_gauge), 300005);
  // Stationarity is tested by successive windows rather than assuming a fixed
  // burn-in length for the periodic comparison.
  for (index_t sweeps = 0; sweeps < 4000 &&
                           !segment_stationary(periodic.thermalization_history);
       sweeps += 100) {
    advance(periodic, periodic_params, 100, true);
  }
  check(segment_stationary(periodic.thermalization_history),
        "periodic bulk reference reaches stationarity");

  std::vector<real_t> slab_center;
  std::vector<real_t> periodic_bulk;
  for (index_t measurement = 0; measurement < 1600; ++measurement) {
    full_heatbath_sweep<3, 2>(slab.gauge, dirichlet_params, slab.rng);
    full_heatbath_sweep<3, 2>(periodic.gauge, periodic_params, periodic.rng);
    slab_center.push_back(
        MeasureDirichletPlaquetteProfiles<2>(slab.gauge).spatial[nt / 2][1]);
    periodic_bulk.push_back(periodic_spatial_slice(periodic.gauge, 0));
  }
  const Estimate slab_estimate = estimate_correlated_mean(slab_center);
  const Estimate periodic_estimate = estimate_correlated_mean(periodic_bulk);
  const real_t z = z_distance(slab_estimate, periodic_estimate);
  std::printf("long-slab/bulk spatial plaquette: %.8f(%.2g), %.8f(%.2g), "
              "z %.2f\n",
              slab_estimate.mean, slab_estimate.error, periodic_estimate.mean,
              periodic_estimate.error, z);
  check(slab_estimate.effective_samples >= 30.0 &&
            periodic_estimate.effective_samples >= 30.0,
        "bulk-recovery comparison has autocorrelation-adjusted samples");
  check(z < 5.0, "central long-slab plaquette agrees with periodic bulk");
}

} // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    check_hot_cold_and_symmetry();
    check_long_slab_bulk_recovery();
  }
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("All temporal Dirichlet statistical checks passed.\n");
  }
  return failures == 0 ? 0 : 1;
}
