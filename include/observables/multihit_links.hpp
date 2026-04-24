#pragma once

#include "groups/group_ops.hpp"
#include "updates/heatbath_link_updates.hpp"

namespace klft {

template <size_t Nc, class Generator>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
multihit_link_metropolis(const SUN<Nc> &link0, const SUN<Nc> &staple,
                         const index_t multihit, const real_t beta,
                         const real_t delta, const real_t epsilon1,
                         const real_t epsilon2, Generator &generator) {
  if (multihit <= 1) {
    return link0;
  }

  SUN<Nc> link = link0;
  SUN<Nc> avg = link0;

  for (index_t hit = 1; hit < multihit; ++hit) {
    const SUN<Nc> updated =
        apply_metropolis_proposal<Nc>(link, delta, generator);
    real_t dS = -(beta / static_cast<real_t>(Nc)) *
                (trace(updated * staple).real() - trace(link * staple).real());
    if (epsilon1 != 0.0) {
      dS += -0.5 * epsilon1 * (trace(updated).real() - trace(link).real());
    }
    if (epsilon2 != 0.0) {
      const real_t retr_updated = trace(updated).real();
      const real_t retr_link = trace(link).real();
      dS += -epsilon2 * (retr_updated * retr_updated - retr_link * retr_link);
    }
    bool accept = dS < 0.0;
    if (!accept) {
      accept = generator.drand(0.0, 1.0) < Kokkos::exp(-dS);
    }
    if (accept) {
      link = updated;
    }
    avg += link;
  }

  return avg * (1.0 / static_cast<real_t>(multihit));
}

template <size_t Nc, class Generator>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
multihit_link_heatbath(const SUN<Nc> &link0, const SUN<Nc> &staple,
                       const index_t multihit, const index_t nOverrelax,
                       const real_t beta, const real_t epsilon1,
                       Generator &generator) {
  if (multihit <= 1) {
    return link0;
  }

  const SUN<Nc> matrix = effective_local_matrix<Nc>(staple, beta, epsilon1);
  SUN<Nc> link = link0;
  SUN<Nc> avg = link0;

  for (index_t hit = 1; hit < multihit; ++hit) {
    heatbath_link(link, matrix, generator);
    restoreSUN(link);
    for (index_t i = 0; i < nOverrelax; ++i) {
      overrelax_link(link, matrix, generator);
      restoreSUN(link);
    }
    avg += link;
  }

  return avg * (1.0 / static_cast<real_t>(multihit));
}

} // namespace klft
