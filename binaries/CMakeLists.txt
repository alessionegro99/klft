add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft)

add_executable(heatbath heatbath.cpp)
target_link_libraries(heatbath PUBLIC klft)

add_executable(heatbath_multilevel heatbath_multilevel.cpp)
target_link_libraries(heatbath_multilevel PUBLIC klft)
