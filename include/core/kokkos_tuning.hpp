#pragma once

#include "core/klft_config.hpp"
#include "params/kokkos_tuning_params.hpp"

#include <Kokkos_Core.hpp>

#include <utility>

#if KLFT_ENABLE_KTUNE
#include <KTune/KTune.hpp>
#endif

namespace klft::kokkos_tuning {

inline bool &runtime_enabled_storage() {
  static bool enabled = false;
  return enabled;
}

inline bool enabled() { return runtime_enabled_storage(); }

inline void initialize(const KokkosTuningParams &params) {
  runtime_enabled_storage() = false;
  if (!params.enabled) {
    return;
  }

#if KLFT_ENABLE_KTUNE
  KTune::set_cache_file(params.cache_file);
  KTune::initialize();
  runtime_enabled_storage() = true;
#else
  (void)params;
#endif
}

#if KLFT_ENABLE_KTUNE
template <class... Views>
using BackupFunctor = KTune::Utils::BackupFunctor<Views...>;
#else
template <class... Views> struct BackupFunctor {
  explicit BackupFunctor(const Views &...) {}
};
#endif

template <class Name, class Policy, class Functor>
void parallel_for(const Name &name, const Policy &policy,
                  const Functor &functor) {
#if KLFT_ENABLE_KTUNE
  if (enabled()) {
    KTune::parallel_for(name, policy, functor);
    return;
  }
#endif
  Kokkos::parallel_for(name, policy, functor);
}

template <class Name, class Policy, class Functor, class ReducerOrResult>
void parallel_reduce(const Name &name, const Policy &policy,
                     const Functor &functor, ReducerOrResult &&result) {
#if KLFT_ENABLE_KTUNE
  if (enabled()) {
    KTune::parallel_reduce(name, policy, functor,
                           std::forward<ReducerOrResult>(result));
    return;
  }
#endif
  Kokkos::parallel_reduce(name, policy, functor,
                          std::forward<ReducerOrResult>(result));
}

} // namespace klft::kokkos_tuning
