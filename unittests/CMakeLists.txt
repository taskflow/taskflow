enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

set(TF_UNITTESTS utility tsq basics algorithm traverse sort)

foreach(unittest IN LISTS TF_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} tf::default_settings)
  target_include_directories(${unittest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  set(TF_CUDA_UNITTESTS cuda_basics cuda_matrix cuda_kmeans cuda_algorithm)
  foreach(cudatest IN LISTS TF_CUDA_UNITTESTS)
    add_executable(${cudatest} cuda/${cudatest}.cu)
    target_link_libraries(${cudatest} ${PROJECT_NAME} tf::default_settings)
    target_include_directories(${cudatest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
    doctest_discover_tests(${cudatest})
  endforeach()
endif()
