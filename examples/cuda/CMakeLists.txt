set(TF_CUDA_EXAMPLES device_property saxpy matmul kmeans)

foreach(c in LISTS TF_CUDA_EXAMPLES)
  add_executable(${c} ${c}.cu)
  target_link_libraries(${c}
    ${PROJECT_NAME} Threads::Threads tf::default_settings)
endforeach()
