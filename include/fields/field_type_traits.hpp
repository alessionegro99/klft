#pragma once
#include "fields/complex_field.hpp"
#include "fields/gauge_field.hpp"
#include "fields/scalar_field.hpp"

namespace klft {

// Map a lattice rank to the corresponding wrapper type.
template <size_t rank, size_t Nc> struct DeviceGaugeFieldType;

template <size_t Nc> struct DeviceGaugeFieldType<2, Nc> {
  using type = deviceGaugeField2D<2, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<3, Nc> {
  using type = deviceGaugeField3D<3, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<4, Nc> {
  using type = deviceGaugeField<4, Nc>;
};

template <size_t rank> struct DeviceFieldType;

template <> struct DeviceFieldType<2> {
  using type = deviceField2D;
};

template <> struct DeviceFieldType<3> {
  using type = deviceField3D;
};

template <> struct DeviceFieldType<4> {
  using type = deviceField;
};

template <size_t rank> struct DeviceScalarFieldType;

template <> struct DeviceScalarFieldType<2> {
  using type = deviceScalarField2D;
};

template <> struct DeviceScalarFieldType<3> {
  using type = deviceScalarField3D;
};

template <> struct DeviceScalarFieldType<4> {
  using type = deviceScalarField;
};

template <class Gauge>
KOKKOS_INLINE_FUNCTION bool gauge_uses_open_time(const Gauge &) { return false; }

template <size_t Nd, size_t Nc>
KOKKOS_INLINE_FUNCTION bool
gauge_uses_open_time(const deviceGaugeField3D<Nd, Nc> &gauge) {
  return gauge.open_time;
}

template <class Gauge, class Site>
KOKKOS_INLINE_FUNCTION bool
gauge_link_exists(const Gauge &, const Site &, const index_t) { return true; }

template <size_t Nd, size_t Nc, class Site>
KOKKOS_INLINE_FUNCTION bool gauge_link_exists(
    const deviceGaugeField3D<Nd, Nc> &gauge, const Site &site, const index_t mu) {
  return !gauge.open_time || mu != 2 || site[2] < gauge.dimensions[2] - 1;
}

template <size_t Nd, size_t Nc> struct HostGaugeLinks3D {
  typename GaugeField3D<Nd, Nc>::HostMirror field;
  typename GaugeField3D<2, Nc>::HostMirror spatial;
  typename Kokkos::View<SUN<Nc> ***>::HostMirror temporal;
  bool open_time;

  KOKKOS_INLINE_FUNCTION SUN<Nc> &operator()(const index_t x, const index_t y, const index_t t,
                      const index_t mu) const {
    if (open_time) {
      assert(mu != 2 || t < static_cast<index_t>(temporal.extent(2)));
      return mu == 2 ? temporal(x, y, t) : spatial(x, y, t, mu);
    }
    return field(x, y, t, mu);
  }
};

// create_mirror (without _view) always allocates independent staging storage,
// including on a CPU backend. A failed checkpoint read cannot mutate a field.
template <class Gauge> auto copy_gauge_to_host(const Gauge &gauge) {
  auto host = Kokkos::create_mirror(gauge.field);
  Kokkos::deep_copy(host, gauge.field);
  return host;
}

template <size_t Nd, size_t Nc>
auto copy_gauge_to_host(const deviceGaugeField3D<Nd, Nc> &gauge) {
  HostGaugeLinks3D<Nd, Nc> host{Kokkos::create_mirror(gauge.field),
      Kokkos::create_mirror(gauge.spatial), Kokkos::create_mirror(gauge.temporal),
      gauge.open_time};
  Kokkos::deep_copy(host.field, gauge.field);
  Kokkos::deep_copy(host.spatial, gauge.spatial);
  Kokkos::deep_copy(host.temporal, gauge.temporal);
  return host;
}

template <class Gauge, class Host>
void copy_gauge_from_host(Gauge &gauge, const Host &host) {
  Kokkos::deep_copy(gauge.field, host);
}

template <size_t Nd, size_t Nc>
void copy_gauge_from_host(deviceGaugeField3D<Nd, Nc> &gauge,
                          const HostGaugeLinks3D<Nd, Nc> &host) {
  Kokkos::deep_copy(gauge.field, host.field);
  Kokkos::deep_copy(gauge.spatial, host.spatial);
  Kokkos::deep_copy(gauge.temporal, host.temporal);
}

template <class Gauge> void copy_gauge_storage(Gauge &dst, const Gauge &src) {
  Kokkos::deep_copy(dst.field, src.field);
}

template <size_t Nd, size_t Nc>
void copy_gauge_storage(deviceGaugeField3D<Nd, Nc> &dst,
                         const deviceGaugeField3D<Nd, Nc> &src) {
  Kokkos::deep_copy(dst.field, src.field);
  Kokkos::deep_copy(dst.spatial, src.spatial);
  Kokkos::deep_copy(dst.temporal, src.temporal);
}
} // namespace klft
