
list(APPEND TF_CUDA_UNITTESTS 
  cuda_statgrab
  cuda_basics 
  cuda_matrix 
  cuda_kmeans 
  cuda_algorithm
  cuda_update
)

foreach(cudatest IN LISTS TF_CUDA_UNITTESTS)
  add_executable(${cudatest} ${cudatest}.cu)
  target_link_libraries(${cudatest} ${PROJECT_NAME} tf::default_settings)
  target_include_directories(${cudatest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${cudatest})
endforeach()
