#include "core/compiled_theory.hpp"
#include "updates/heatbath.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdio>
#include <cstdlib>

// Diagnostic timings only: this executable does not produce MC measurements.
// Warm up allocations/kernels before separately timing updates and observables.
int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: dirichlet_benchmark Ns Nt repeats\n");
    return 2;
  }
  const int ns = std::atoi(argv[1]), nt = std::atoi(argv[2]);
  const int repeats = std::atoi(argv[3]);
  if (ns < 2 || nt < 1 || repeats < 1) return 2;
  Kokkos::initialize(argc, argv);
  {
    using namespace klft;
    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> rng(20261014);
    auto gauge = make_hot_gauge_field<3, 2>(ns, ns, nt + 1, 1, rng, true);
    HeatbathParams params;
    params.L0 = ns; params.L1 = ns; params.L2 = nt + 1;
    params.beta = 3.; params.temporal_dirichlet = true; params.nOverrelax = 2;
    full_heatbath_sweep<3, 2>(gauge, params, rng);
    MeasureDirichletHolonomy<2>(gauge);
    MeasureDirichletPlaquetteProfiles<2>(gauge);
    Kokkos::Timer timer;
    for (int i = 0; i < repeats; ++i)
      full_heatbath_sweep<3, 2>(gauge, params, rng);
    Kokkos::fence();
    const double update = timer.seconds() / repeats;
    timer.reset();
    for (int i = 0; i < repeats; ++i) MeasureDirichletHolonomy<2>(gauge);
    Kokkos::fence();
    const double holonomy = timer.seconds() / repeats;
    timer.reset();
    for (int i = 0; i < repeats; ++i) MeasureDirichletPlaquetteProfiles<2>(gauge);
    Kokkos::fence();
    std::printf("Ns,Nt,repeats,update_seconds,holonomy_seconds,profile_seconds\n"
                "%d,%d,%d,%.9g,%.9g,%.9g\n", ns, nt, repeats, update,
                holonomy, timer.seconds() / repeats);
  }
  Kokkos::finalize();
}
