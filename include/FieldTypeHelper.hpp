// this is a helper file to define the field types based on dimension
// for the gauge fields here we assume that Nd = rank (dimensionality)
// this would work for most use cases. But if you have a different
// dimensionality from Nd, you can not use the definitions here

#pragma once
#include "Field.hpp"
#include "GaugeField.hpp"
#include "SUNField.hpp"
#include "ScalarField.hpp"

namespace klft {

// define a function to get the gauge field type based on the rank
template <size_t rank, size_t Nc> struct DeviceGaugeFieldType;

// now define the specializations
template <size_t Nc> struct DeviceGaugeFieldType<2, Nc> {
  using type = deviceGaugeField2D<2, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<3, Nc> {
  using type = deviceGaugeField3D<3, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<4, Nc> {
  using type = deviceGaugeField<4, Nc>;
};

// define the same thing for SUN fields
template <size_t rank, size_t Nc> struct DeviceSUNFieldType;

template <size_t Nc> struct DeviceSUNFieldType<2, Nc> {
  using type = deviceSUNField2D<Nc>;
};

template <size_t Nc> struct DeviceSUNFieldType<3, Nc> {
  using type = deviceSUNField3D<Nc>;
};

template <size_t Nc> struct DeviceSUNFieldType<4, Nc> {
  using type = deviceSUNField<Nc>;
};

// repeat for field
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

// define the same for the scalar fields
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

// add the same for scalar fields here when needed
} // namespace klft
