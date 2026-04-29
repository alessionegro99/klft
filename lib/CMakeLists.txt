add_library(klft STATIC
  metropolis.cpp
  heatbath.cpp)
target_include_directories(klft PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/generated)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)
