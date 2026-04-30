add_executable(metropolis metropolis.cpp)
target_link_libraries(metropolis PUBLIC klft)

add_executable(heatbath heatbath.cpp)
target_link_libraries(heatbath PUBLIC klft)

add_executable(gradient_flow_check gradient_flow_check.cpp)
target_link_libraries(gradient_flow_check PUBLIC klft)
add_test(NAME gradient_flow_check COMMAND gradient_flow_check)
