
list(APPEND TF_CUDA_UNITTESTS 
  test_cuda_objects
  test_cuda_memory
  test_cuda_basics 
  test_cuda_matrix 
  test_cuda_kmeans 
  test_cuda_for_each
  test_cuda_for_each_index
  test_cuda_transform
  test_cuda_reduce
  test_cuda_scan
  test_cuda_find
  test_cuda_min_max_element
  test_cuda_merge
  test_cuda_basic_updates
  test_cuda_capturer_optimizer
  test_cuda_capture
  
  #cuda_algorithms
  #cuda_algorithm_updates
)

foreach(cudatest IN LISTS TF_CUDA_UNITTESTS)
  add_executable(${cudatest} ${cudatest}.cu)
  target_link_libraries(${cudatest} ${PROJECT_NAME} tf::default_settings)
  target_include_directories(${cudatest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  
  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cudatest} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
  
  doctest_discover_tests(${cudatest})
endforeach()



