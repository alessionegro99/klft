#pragma once
#include "core/common.hpp"
#include "groups/group_ops.hpp"

#include <stdexcept>

namespace klft {

template <size_t Nd, size_t Nc> struct deviceGaugeField {

  deviceGaugeField() = delete;

  // initialize all sites to a given value
  deviceGaugeField(const index_t L0, const index_t L1, const index_t L2,
                   const index_t L3, const complex_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  // initialize all links to a given SUN matrix
  deviceGaugeField(const index_t L0, const index_t L1, const index_t L2,
                   const index_t L3, const SUN<Nc> &init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  // initialize all links to a random SUN matrix
  template <class RNG>
  deviceGaugeField(const index_t L0, const index_t L1, const index_t L2,
                   const index_t L3, RNG &rng, const real_t delta)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, rng, delta);
  }

  // initialize all sites to a random value
  template <class RNG>
  deviceGaugeField(const index_t L0, const index_t L1, const index_t L2,
                   const index_t L3, RNG &rng)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, rng);
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, GaugeField<Nd, Nc> &V, complex_t init) {
    const SUN<Nc> fill = identitySUN<Nc>() * init;
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = fill;
          }
        });
    Kokkos::fence();
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, GaugeField<Nd, Nc> &V, const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = init;
          }
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, GaugeField<Nd, Nc> &V, RNG &rng,
               const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) =
                make_metropolis_matrix<Nc>(delta, generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, GaugeField<Nd, Nc> &V, RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            rand_matrix(V(i0, i1, i2, i3, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  GaugeField<Nd, Nc> field;
  const IndexArray<4> dimensions;

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3, const index_t mu) const {
    return field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3, const index_t mu) {
    return field(i0, i1, i2, i3, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 4> site, const index_t mu) const {
    return field(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 4> site, const index_t mu) {
    return field(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const Kokkos::Array<indexType, 4> site, const index_t mu) const {
    // this only works if Nd == 4
    assert(Nd == 4);
    // get the indices
    const index_t i0 = site[0];
    const index_t i1 = site[1];
    const index_t i2 = site[2];
    const index_t i3 = site[3];
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
    const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
    const index_t i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      // do nothing for mu = nu
      if (nu == mu)
        continue;
      // get the x + nu indices
      const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
      const index_t i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
      // get the staple
      temp += field(i0pmu, i1pmu, i2pmu, i3pmu, nu) *
              conj(field(i0pnu, i1pnu, i2pnu, i3pnu, mu)) *
              conj(field(i0, i1, i2, i3, nu));
    } // loop over nu
// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      // do nothing for mu = nu
      if (nu == mu)
        continue;
      // get the x + mu - nu indices
      const index_t i0pmu_mnu =
          nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
      const index_t i1pmu_mnu =
          nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
      const index_t i2pmu_mnu =
          nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;
      const index_t i3pmu_mnu =
          nu == 3 ? (i3pmu - 1 + dimensions[3]) % dimensions[3] : i3pmu;
      // get the x - nu indices
      const index_t i0mnu =
          nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
      const index_t i1mnu =
          nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
      const index_t i2mnu =
          nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;
      const index_t i3mnu =
          nu == 3 ? (i3 - 1 + dimensions[3]) % dimensions[3] : i3;
      // get the staple
      temp += conj(field(i0pmu_mnu, i1pmu_mnu, i2pmu_mnu, i3pmu_mnu, nu)) *
              conj(field(i0mnu, i1mnu, i2mnu, i3mnu, mu)) *
              field(i0mnu, i1mnu, i2mnu, i3mnu, nu);
    } // loop over nu
    return temp;
  }
};

template <size_t Nd, size_t Nc> struct deviceGaugeField3D {

  deviceGaugeField3D() = delete;

  // initialize all sites to a given value
  deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                     const complex_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  // initialize all links to a given SUN matrix
  deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                     const SUN<Nc> &init, const bool open_time = false)
      : dimensions({L0, L1, L2}), open_time(open_time) {
    if (open_time) {
      allocate_slab();
      Kokkos::deep_copy(spatial, init);
      Kokkos::deep_copy(temporal, init);
    } else {
      do_init(L0, L1, L2, field, init);
    }
  }

  // initialize all links to a random SUN matrix
  template <class RNG>
  deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                     RNG &rng, const real_t delta)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, rng, delta);
  }

  // initialize all sites to a random value
  template <class RNG>
  deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                     RNG &rng, const bool open_time = false)
      : dimensions({L0, L1, L2}), open_time(open_time) {
    if (open_time) {
      allocate_slab();
      const auto links = *this;
      Kokkos::parallel_for("hot_open_slab",
          Policy<3>(IndexArray<3>{}, dimensions),
          KOKKOS_LAMBDA(const index_t x, const index_t y, const index_t t) {
            auto generator = rng.get_state();
            for (index_t mu = 0; mu < 2; ++mu) {
              if (t == 0 || t == L2 - 1) {
                links(x, y, t, mu) = identitySUN<Nc>();
              } else {
                rand_matrix(links(x, y, t, mu), generator);
              }
            }
            if (t < L2 - 1) rand_matrix(links(x, y, t, 2), generator);
            rng.free_state(generator);
          });
      Kokkos::fence();
    } else {
      do_init(L0, L1, L2, field, rng);
    }
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               GaugeField3D<Nd, Nc> &V, complex_t init) {
    const SUN<Nc> fill = identitySUN<Nc>() * init;
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = fill;
          }
        });
    Kokkos::fence();
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               GaugeField3D<Nd, Nc> &V, const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = init;
          }
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               GaugeField3D<Nd, Nc> &V, RNG &rng, const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = make_metropolis_matrix<Nc>(delta, generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               GaugeField3D<Nd, Nc> &V, RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            rand_matrix(V(i0, i1, i2, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  GaugeField3D<Nd, Nc> field;
  const IndexArray<3> dimensions;
  // Only one representation is allocated. An open slab has Nt+1 spatial
  // slices and Nt temporal-link layers; a closing temporal link does not exist.
  const bool open_time = false;
  GaugeField3D<2, Nc> spatial;
  Kokkos::View<SUN<Nc> ***> temporal;

  void allocate_slab() {
    static_assert(Nd == 3, "Open slabs require three dimensions.");
    if (dimensions[0] < 2 || dimensions[1] < 2 || dimensions[2] < 2)
      throw std::invalid_argument("An open slab needs L0,L1,L2 >= 2.");
    spatial = GaugeField3D<2, Nc>("slab_spatial", dimensions[0],
                                dimensions[1], dimensions[2]);
    temporal = Kokkos::View<SUN<Nc> ***>("slab_temporal", dimensions[0],
                                       dimensions[1], dimensions[2] - 1);
  }

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const index_t mu) const {
    if (open_time) {
      KOKKOS_ASSERT(mu < 2 || i2 < dimensions[2] - 1);
      return mu == 2 ? temporal(i0, i1, i2) : spatial(i0, i1, i2, mu);
    }
    return field(i0, i1, i2, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const index_t mu) {
    return static_cast<const deviceGaugeField3D &>(*this)(i0, i1, i2, mu);
  }

  // define accessors with 3D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 3> site, const index_t mu) const {
    return (*this)(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 3> site, const index_t mu) {
    return (*this)(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const Kokkos::Array<indexType, 3> site, const index_t mu) const {
    // Wilson staples of a dynamical slab link stay inside [0,Nt]. Spatial
    // directions are periodic; time is open and is never reduced modulo L2.
    static_assert(Nd == 3, "Three-dimensional staple required.");
    const IndexArray<3> origin{static_cast<index_t>(site[0]),
                               static_cast<index_t>(site[1]),
                               static_cast<index_t>(site[2])};
    const auto plus_mu = shifted(origin, mu, 1);
    SUN<Nc> temp = zeroSUN<Nc>();
    for (index_t nu = 0; nu < 3; ++nu) {
      if (nu == mu) continue;
      const auto plus_nu = shifted(origin, nu, 1);
      const auto minus_nu = shifted(origin, nu, -1);
      const auto plus_mu_minus_nu = shifted(plus_mu, nu, -1);
      temp += (*this)(plus_mu, nu) * conj((*this)(plus_nu, mu)) *
              conj((*this)(origin, nu));
      temp += conj((*this)(plus_mu_minus_nu, nu)) *
              conj((*this)(minus_nu, mu)) * (*this)(minus_nu, nu);
    }
    return temp;
  }

  KOKKOS_FORCEINLINE_FUNCTION IndexArray<3>
  shifted(IndexArray<3> site, const index_t mu, const index_t offset) const {
    site[mu] += offset;
    if (!open_time || mu != 2)
      site[mu] = (site[mu] + dimensions[mu]) % dimensions[mu];
    return site;
  }
};

template <size_t Nd, size_t Nc> struct deviceGaugeField2D {

  deviceGaugeField2D() = delete;

  // initialize all sites to a given value
  deviceGaugeField2D(const index_t L0, const index_t L1, const complex_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  // initialize all links to a given SUN matrix
  deviceGaugeField2D(const index_t L0, const index_t L1, const SUN<Nc> &init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  // initialize all links to a random SUN matrix
  template <class RNG>
  deviceGaugeField2D(const index_t L0, const index_t L1, RNG &rng,
                     const real_t delta)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, rng, delta);
  }

  // initialize all sites to a random value
  template <class RNG>
  deviceGaugeField2D(const index_t L0, const index_t L1, RNG &rng)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, rng);
  }

  void do_init(const index_t L0, const index_t L1, GaugeField2D<Nd, Nc> &V,
               complex_t init) {
    const SUN<Nc> fill = identitySUN<Nc>() * init;
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = fill;
          }
        });
    Kokkos::fence();
  }

  void do_init(const index_t L0, const index_t L1, GaugeField2D<Nd, Nc> &V,
               const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = init;
          }
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, GaugeField2D<Nd, Nc> &V,
               RNG &rng, const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = make_metropolis_matrix<Nc>(delta, generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0, const index_t L1, GaugeField2D<Nd, Nc> &V,
               RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            rand_matrix(V(i0, i1, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  GaugeField2D<Nd, Nc> field;
  const IndexArray<2> dimensions;

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const index_t mu) const {
    return field(i0, i1, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i0, const indexType i1, const index_t mu) {
    return field(i0, i1, mu);
  }

  // define accessors with 2D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 2> site, const index_t mu) const {
    return field(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 2> site, const index_t mu) {
    return field(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const Kokkos::Array<indexType, 2> site, const index_t mu) const {
    // this only works if Nd == 2
    assert(Nd == 2);
    // get the indices
    const index_t i0 = site[0];
    const index_t i1 = site[1];
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;

// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;

      temp += field(i0pmu, i1pmu, nu) * conj(field(i0pnu, i1pnu, mu)) *
              conj(field(i0, i1, nu));
    }

// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const index_t i0pmu_mnu =
          nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
      const index_t i1pmu_mnu =
          nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;

      const index_t i0mnu =
          nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
      const index_t i1mnu =
          nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;

      temp += conj(field(i0pmu_mnu, i1pmu_mnu, nu)) *
              conj(field(i0mnu, i1mnu, mu)) * field(i0mnu, i1mnu, nu);
    }

    return temp;
  }
};

} // namespace klft
