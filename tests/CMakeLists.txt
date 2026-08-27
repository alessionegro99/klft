if(KLFT_NC STREQUAL "2")
  add_executable(partitioning_test partitioning_test.cpp)
  target_link_libraries(partitioning_test PRIVATE klft)
  add_test(NAME partitioning_linear
    COMMAND partitioning_test ${PROJECT_SOURCE_DIR}/partitionings/linear_m3.yaml)
  add_test(NAME partitioning_fibonacci
    COMMAND partitioning_test ${PROJECT_SOURCE_DIR}/partitionings/fibonacci_N88.yaml)
endif()

if(KLFT_NDIM STREQUAL "3" AND KLFT_NC STREQUAL "2")
  add_executable(dirichlet_test dirichlet_test.cpp)
  target_link_libraries(dirichlet_test PRIVATE klft)
  add_test(NAME temporal_dirichlet_deterministic
    COMMAND dirichlet_test ${PROJECT_SOURCE_DIR}/partitionings/linear_m3.yaml)
  set_tests_properties(temporal_dirichlet_deterministic PROPERTIES
    LABELS "dirichlet;fast")

  add_executable(dirichlet_statistical_test dirichlet_statistical_test.cpp)
  target_link_libraries(dirichlet_statistical_test PRIVATE klft)
  if(KLFT_ENABLE_STATISTICAL_TESTS)
    add_test(NAME temporal_dirichlet_statistical
      COMMAND dirichlet_statistical_test)
    set_tests_properties(temporal_dirichlet_statistical PROPERTIES
      LABELS "dirichlet;statistical;slow" TIMEOUT 180)
  endif()

endif()

if(KLFT_NDIM STREQUAL "4" AND KLFT_NC STREQUAL "3")
  add_executable(orbifold_test orbifold_test.cpp)
  target_link_libraries(orbifold_test PRIVATE klft)
  add_test(NAME orbifold_deterministic COMMAND orbifold_test)
  set_tests_properties(orbifold_deterministic PROPERTIES
    LABELS "orbifold;fast")
endif()
