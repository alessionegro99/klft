add_library(klft STATIC
  metropolis.cpp
  heatbath.cpp)
target_include_directories(klft PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)
