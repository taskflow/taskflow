# -----------------------------------------------------------------------------
# CUDA BLAS examples
# -----------------------------------------------------------------------------
list(APPEND TF_CUDA_BLAS_EXAMPLES
  nrm2
  trsv
  gemm
)

foreach(cublas_example IN LISTS TF_CUDA_BLAS_EXAMPLES)
  add_executable(${cublas_example} ${cublas_example}.cu)
  target_link_libraries(${cublas_example}
    ${PROJECT_NAME} Threads::Threads tf::default_settings ${CUBLAS_LIBRARIES}
  )
  target_include_directories(${cublas_example} PRIVATE ${CUBLAS_INCLUDE_DIRS})
  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cublas_example} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
endforeach()




