#pragma once
#include "core/common.hpp"

namespace klft {

// Rank-4 complex field wrapper with Kokkos allocation and reductions.
struct deviceField {

  deviceField() = delete;

  // initialize all sites to a given value
  deviceField(const index_t L0, const index_t L1, const index_t L2,
              const index_t L3, const complex_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceField(const IndexArray<4> &dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, Field &V, const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    Kokkos::parallel_for(
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) { V(i0, i1, i2, i3) = init; });
    Kokkos::fence();
  }

  Field field;
  const IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }

  complex_t sum() const {
    complex_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField", Policy<4>({0, 0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2, const index_t i3,
                            complex_t &lsum) { lsum += field(i0, i1, i2, i3); },
        Kokkos::Sum<complex_t>(sum));
    return sum;
  }
};

struct deviceField3D {

  deviceField3D() = delete;

  // initialize all sites to a given value
  deviceField3D(const index_t L0, const index_t L1, const index_t L2,
                const complex_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  deviceField3D(const IndexArray<3> &dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
  }

  void do_init(const index_t L0, const index_t L1, const index_t L2, Field3D &V,
               const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    Kokkos::parallel_for(
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          V(i0, i1, i2) = init;
        });
    Kokkos::fence();
  }

  Field3D field;
  const IndexArray<3> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const indexType i0, const indexType i1, const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const indexType i0, const indexType i1, const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 3D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 3> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 3> site) {
    return field(site[0], site[1], site[2]);
  }

  complex_t sum() const {
    complex_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField3D", Policy<3>({0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2,
                            complex_t &lsum) { lsum += field(i0, i1, i2); },
        Kokkos::Sum<complex_t>(sum));
    return sum;
  }
};

struct deviceField2D {

  deviceField2D() = delete;

  // initialize all sites to a given value
  deviceField2D(const index_t L0, const index_t L1, const complex_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  deviceField2D(const IndexArray<2> &dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, init);
  }

  void do_init(const index_t L0, const index_t L1, Field2D &V,
               const complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    Kokkos::parallel_for(
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          V(i0, i1) = init;
        });
    Kokkos::fence();
  }

  Field2D field;
  const IndexArray<2> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &operator()(const indexType i0,
                                                    const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &operator()(const indexType i0,
                                                    const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 2D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 2> site) const {
    return field(site[0], site[1]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION complex_t &
  operator()(const Kokkos::Array<indexType, 2> site) {
    return field(site[0], site[1]);
  }

  complex_t sum() const {
    complex_t sum = 0.0;
    Kokkos::parallel_reduce(
        "sum_deviceField2D", Policy<2>({0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            complex_t &lsum) { lsum += field(i0, i1); },
        Kokkos::Sum<complex_t>(sum));
    return sum;
  }
};

} // namespace klft
