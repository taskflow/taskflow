list(APPEND TF_SYCL_EXAMPLES 
  sycl_device
  sycl_ndrange
  sycl_saxpy
  sycl_vector_add
  sycl_atomic
  sycl_matmul
  sycl_reduce
)

foreach(sycl_example IN LISTS TF_SYCL_EXAMPLES)
  add_executable(${sycl_example} ${sycl_example}.cpp)

  #add_sycl_to_target(TARGET ${sycl_example} SOURCES ${sycl_example}.cpp)

  target_compile_options(${sycl_example} PRIVATE ${TF_SYCL_OPTIONS})
  target_link_options(${sycl_example} PRIVATE ${TF_SYCL_OPTIONS})
  target_link_libraries(${sycl_example}
    ${PROJECT_NAME} Threads::Threads tf::default_settings
  )
endforeach()
