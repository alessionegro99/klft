#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace klft {

using real_t = double;
using complex_t = Kokkos::complex<real_t>;

using index_t = int;

template <size_t rank> using IndexArray = Kokkos::Array<index_t, rank>;

template <size_t Nc>
using SUN = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

template <size_t Nd, size_t Nc>
using GaugeField =
    Kokkos::View<SUN<Nc> ****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField3D =
    Kokkos::View<SUN<Nc> ***[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField2D =
    Kokkos::View<SUN<Nc> **[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField =
    Kokkos::View<SUN<Nc> ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField3D =
    Kokkos::View<SUN<Nc> ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField2D =
    Kokkos::View<SUN<Nc> **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field =
    Kokkos::View<complex_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field3D =
    Kokkos::View<complex_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field2D =
    Kokkos::View<complex_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field1D =
    Kokkos::View<complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField =
    Kokkos::View<real_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField3D =
    Kokkos::View<real_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField2D =
    Kokkos::View<real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField1D =
    Kokkos::View<real_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField =
    Kokkos::View<real_t ****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField3D =
    Kokkos::View<real_t ***[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField2D =
    Kokkos::View<real_t **[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

#if defined(KOKKOS_ENABLE_CUDA)

template <size_t Nd, size_t Nc>
using constGaugeField =
    Kokkos::View<const SUN<Nc> ****[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constGaugeField3D =
    Kokkos::View<const SUN<Nc> ***[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constGaugeField2D =
    Kokkos::View<const SUN<Nc> **[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField = Kokkos::View<const SUN<Nc> ****,
                                   Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField3D =
    Kokkos::View<const SUN<Nc> ***, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField2D =
    Kokkos::View<const SUN<Nc> **, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField = Kokkos::View<const complex_t ****,
                                Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField3D = Kokkos::View<const complex_t ***,
                                  Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField2D = Kokkos::View<const complex_t **,
                                  Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField1D =
    Kokkos::View<const complex_t *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField =
    Kokkos::View<const real_t ****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField3D =
    Kokkos::View<const real_t ***, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField2D =
    Kokkos::View<const real_t **, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField1D =
    Kokkos::View<const real_t *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField =
    Kokkos::View<const real_t ****[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField3D =
    Kokkos::View<const real_t ***[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField2D =
    Kokkos::View<const real_t **[Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

#else

template <size_t Nd, size_t Nc>
using constGaugeField = Kokkos::View<const SUN<Nc> ****[Nd],
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constGaugeField3D =
    Kokkos::View<const SUN<Nc> ***[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constGaugeField2D =
    Kokkos::View<const SUN<Nc> **[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField =
    Kokkos::View<const SUN<Nc> ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField3D =
    Kokkos::View<const SUN<Nc> ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField2D =
    Kokkos::View<const SUN<Nc> **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField =
    Kokkos::View<const complex_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField3D =
    Kokkos::View<const complex_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField2D =
    Kokkos::View<const complex_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField1D =
    Kokkos::View<const complex_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField =
    Kokkos::View<const real_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField3D =
    Kokkos::View<const real_t ***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField2D =
    Kokkos::View<const real_t **, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField1D =
    Kokkos::View<const real_t *, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField =
    Kokkos::View<const real_t ****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField3D =
    Kokkos::View<const real_t ***[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField2D =
    Kokkos::View<const real_t **[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

#endif

template <size_t rank> using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

using Policy1D = Kokkos::RangePolicy<>;

// Build the zero matrix for the selected gauge group.
template <size_t Nc> constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> zeroSUN() {
  SUN<Nc> zero;
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
    for (index_t c2 = 0; c2 < Nc; ++c2) {
      zero[c1][c2] = complex_t(0.0, 0.0);
    }
  }
  return zero;
}

// Build the identity matrix for the selected gauge group.
template <size_t Nc>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> identitySUN() {
  SUN<Nc> id = zeroSUN<Nc>();
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
    id[c1][c1] = complex_t(1.0, 0.0);
  }
  return id;
}

inline int KLFT_VERBOSITY = 0;

inline void setVerbosity(int v) { KLFT_VERBOSITY = v; }

} // namespace klft
