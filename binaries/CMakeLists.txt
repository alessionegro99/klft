add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft)

add_executable(heatbath heatbath.cpp)
target_link_libraries(heatbath PUBLIC klft)
