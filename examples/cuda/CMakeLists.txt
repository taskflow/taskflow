# -----------------------------------------------------------------------------
# cuda examples
# -----------------------------------------------------------------------------

list(APPEND TF_CUDA_EXAMPLES 
  device_property saxpy matmul kmeans
)

foreach(cuda_example IN LISTS TF_CUDA_EXAMPLES)
  add_executable(${cuda_example} ${cuda_example}.cu)
  target_link_libraries(${cuda_example}
    ${PROJECT_NAME} Threads::Threads tf::default_settings
  )

  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cuda_example} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
endforeach()

# -----------------------------------------------------------------------------
# CUDA BLAS examples
# -----------------------------------------------------------------------------
#list(APPEND TF_CUDA_BLAS_EXAMPLES
#  blas
#)
#
#foreach(cuda_blas_example IN LISTS TF_CUDA_BLAS_EXAMPLES)
#  add_executable(${cuda_blas_example} ${cuda_blas_example}.cu)
#  target_link_libraries(${cuda_blas_example}
#    ${PROJECT_NAME} Threads::Threads tf::default_settings ${CUBLAS_LIBRARIES}
#  )
#  # avoid cmake 3.18+ warning
#  # we let nvcc to decide the flag if the architecture is not given
#  if(NOT CUDA_ARCHITECTURES)
#    set_property(TARGET ${cuda_blas_example} PROPERTY CUDA_ARCHITECTURES OFF)
#  endif()
#endforeach()




