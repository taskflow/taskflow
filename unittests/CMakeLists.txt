enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TF_UNITTESTS 
  utility 
  tsq 
  serializer 
  basics 
  control_flow
  movable
  cancellation
  semaphore
  algorithm 
  traverse 
  sorting
  pipelines
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



