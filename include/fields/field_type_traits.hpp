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
} // namespace klft
