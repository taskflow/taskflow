
list(APPEND TF_CUDA_UNITTESTS 
  cuda_device
  cuda_memory
  cuda_basics 
  cuda_matrix 
  cuda_kmeans 
  cuda_algorithms
  cuda_updates
  cuda_rebinds
  cuda_capturer_optimizer
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



