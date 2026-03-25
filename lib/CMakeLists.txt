add_library(klft STATIC 
  klft.cpp)
target_link_libraries(klft PUBLIC Kokkos::kokkos yaml-cpp)
