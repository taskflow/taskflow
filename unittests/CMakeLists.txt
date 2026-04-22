enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TF_UNITTESTS
  test_utility
  test_iterators
  test_wsq
  test_spsc
  test_basics
  test_notifiers
  test_work_stealing
  #test_serializer
  test_workers
  test_asyncs
  test_dependent_asyncs
  test_subflows
  test_control_flow
  test_semaphores
  test_movable
  test_cancellation
  test_for_each
  test_reduce
  test_transform
  test_sort
  test_scan
  test_find
  test_modules
  test_merge
  test_traversals
  test_runtimes
  test_task_group
  test_pipelines
  test_scalable_pipelines
  test_data_pipelines
  test_atomic_intrusive_stack
  test_fill
  test_generate
  test_graph_priorities
  test_prioritized_run
)

# we only do exception tests if sanitizer is not enabled
string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)
#message("sanitize='${sanitize}'")

if(${sanitize} EQUAL -1)
  message(STATUS "Add test_exception to unit tests")
  list(APPEND TF_UNITTESTS test_exceptions)
endif()


foreach(unittest IN LISTS TF_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
  target_include_directories(${unittest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

# Same prioritized_run tests with cross-worker exploit sweep enabled
add_executable(test_prioritized_run_enforced test_prioritized_run.cpp)
target_link_libraries(test_prioritized_run_enforced ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
target_include_directories(test_prioritized_run_enforced PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
target_compile_definitions(test_prioritized_run_enforced PRIVATE TF_ENFORCE_PRIORITY_EXPLOIT)
doctest_discover_tests(test_prioritized_run_enforced)

# Same graph_priorities tests with cross-worker exploit sweep enabled
add_executable(test_graph_priorities_enforced test_graph_priorities.cpp)
target_link_libraries(test_graph_priorities_enforced ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
target_include_directories(test_graph_priorities_enforced PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
target_compile_definitions(test_graph_priorities_enforced PRIVATE TF_ENFORCE_PRIORITY_EXPLOIT)
doctest_discover_tests(test_graph_priorities_enforced)

# Enforce-only test: verifies cross-worker sweep prioritizes HIGH over LOW.
# This test FAILS without TF_ENFORCE_PRIORITY_EXPLOIT (workers only drain local queues).
add_executable(test_prioritized_run_enforced_exclusively test_prioritized_run_enforced_exclusively.cpp)
target_link_libraries(test_prioritized_run_enforced_exclusively ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
target_include_directories(test_prioritized_run_enforced_exclusively PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
target_compile_definitions(test_prioritized_run_enforced_exclusively PRIVATE TF_ENFORCE_PRIORITY_EXPLOIT)
doctest_discover_tests(test_prioritized_run_enforced_exclusively)

# include C++ module tests
if(TF_BUILD_MODULES)
  add_subdirectory(${TF_UTEST_DIR}/modules)
endif()

# include CUDA tests
if(TF_BUILD_CUDA)
  add_subdirectory(${TF_UTEST_DIR}/cuda)
endif()

# include SYCL tests
#if(TF_BUILD_SYCL)
#  add_subdirectory(${TF_UTEST_DIR}/sycl)
#endif()




