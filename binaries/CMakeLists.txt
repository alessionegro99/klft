add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft Kokkos::kokkos yaml-cpp)
