enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TF_UNITTESTS 
  test_utility 
  test_work_stealing 
  #test_serializer 
  test_priorities
  test_basics 
  test_asyncs
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
  test_compositions
  test_traversals
  test_pipelines
  test_scalable_pipelines
  test_deferred_pipelines
  test_deferred_scalable_pipelines
  test_runtimes
  test_data_pipelines
  test_workers
)

foreach(unittest IN LISTS TF_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} tf::default_settings)
  target_include_directories(${unittest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

# include CUDA tests
if(TF_BUILD_CUDA)
  add_subdirectory(${TF_UTEST_DIR}/cuda)
endif()

# include SYCL tests
if(TF_BUILD_SYCL)
  add_subdirectory(${TF_UTEST_DIR}/sycl)
endif()




