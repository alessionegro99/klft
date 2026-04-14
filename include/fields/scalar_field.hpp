#pragma once
#include "core/common.hpp"

namespace klft {

// Rank-4 real field wrapper with Kokkos allocation and reductions.
struct deviceScalarField {

  deviceScalarField() = delete;

  // initialize all sites to a given value
  deviceScalarField(const index_t L0, const index_t L1, const index_t L2,
                    const index_t L3, const real_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceScalarField(const IndexArray<4> &dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, ScalarField &V, const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) { V(i0, i1, i2, i3) = init; });
    Kokkos::fence();
  }

  ScalarField field;
  const IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceScalarField", Policy<4>({0, 0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2, const index_t i3,
                            real_t &lsum) { lsum += field(i0, i1, i2, i3); },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

struct deviceScalarField3D {

  deviceScalarField3D() = delete;

  // initialize all sites to a given value
  deviceScalarField3D(const index_t L0, const index_t L1, const index_t L2,
                      const real_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  deviceScalarField3D(const IndexArray<3> &dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               ScalarField3D &V, const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          V(i0, i1, i2) = init;
        });
    Kokkos::fence();
  }

  ScalarField3D field;
  const IndexArray<3> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const indexType i0, const indexType i1, const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const indexType i0, const indexType i1, const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 3D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 3> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 3> site) {
    return field(site[0], site[1], site[2]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceScalarField3D", Policy<3>({0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2,
                            real_t &lsum) { lsum += field(i0, i1, i2); },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

struct deviceScalarField2D {

  deviceScalarField2D() = delete;

  // initialize all sites to a given value
  deviceScalarField2D(const index_t L0, const index_t L1, const real_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  deviceScalarField2D(const IndexArray<2> &dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, init);
  }

  void do_init(const index_t L0, const index_t L1, ScalarField2D &V,
               const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          V(i0, i1) = init;
        });
    Kokkos::fence();
  }

  ScalarField2D field;
  const IndexArray<2> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &operator()(const indexType i0,
                                                 const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &operator()(const indexType i0,
                                                 const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 2D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 2> site) const {
    return field(site[0], site[1]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t &
  operator()(const Kokkos::Array<indexType, 2> site) {
    return field(site[0], site[1]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceScalarField2D", Policy<2>({0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1, real_t &lsum) {
          lsum += field(i0, i1);
        },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

} // namespace klft
