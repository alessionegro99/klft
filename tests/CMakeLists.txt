if(KLFT_NC STREQUAL "2")
  add_executable(partitioning_test partitioning_test.cpp)
  target_link_libraries(partitioning_test PRIVATE klft)
  add_test(NAME partitioning_linear
    COMMAND partitioning_test ${PROJECT_SOURCE_DIR}/partitionings/linear_m3.yaml)
  add_test(NAME partitioning_fibonacci
    COMMAND partitioning_test ${PROJECT_SOURCE_DIR}/partitionings/fibonacci_N88.yaml)
endif()
