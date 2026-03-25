add_executable(nemc nemc.cpp)
target_link_libraries(nemc PUBLIC klft Kokkos::kokkos yaml-cpp)
