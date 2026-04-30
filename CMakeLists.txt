cmake_minimum_required(VERSION 3.21.0)
project(klft VERSION 0.0.1 LANGUAGES CXX)
enable_testing()

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(KLFT_NDIM 4 CACHE STRING "Compile-time spacetime dimension (2, 3, or 4)")
set(KLFT_NC 3 CACHE STRING "Compile-time gauge group selector: 1=U(1), 2=SU(2), 3=SU(3)")

if(NOT KLFT_NDIM MATCHES "^[234]$")
  message(FATAL_ERROR "KLFT_NDIM must be one of: 2, 3, 4")
endif()

if(NOT KLFT_NC MATCHES "^[123]$")
  message(FATAL_ERROR "KLFT_NC must be one of: 1, 2, 3")
endif()

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/generated/core)

configure_file(
  ${PROJECT_SOURCE_DIR}/include/core/klft_config.hpp.in
  ${PROJECT_BINARY_DIR}/generated/core/klft_config.hpp
  @ONLY
)

#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${PROJECT_BINARY_DIR}/kokkos)

#
# yaml-cpp
#
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/yaml-cpp ${PROJECT_BINARY_DIR}/yaml-cpp)

# build the library
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ${PROJECT_BINARY_DIR}/lib)

# build the binaries
add_subdirectory(${PROJECT_SOURCE_DIR}/binaries ${PROJECT_BINARY_DIR}/binaries)

# stage Python analysis helpers into the build tree
add_subdirectory(${PROJECT_SOURCE_DIR}/analysis ${PROJECT_BINARY_DIR}/analysis)
