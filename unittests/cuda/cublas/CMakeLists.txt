# -----------------------------------------------------------------------------
# cuBLAS unittests
# -----------------------------------------------------------------------------
list(APPEND TF_CUBLAS_UNITTESTS
  helper
  level1
  level2
  level3
)

foreach(test IN LISTS TF_CUBLAS_UNITTESTS)
  add_executable(${test} ${test}.cu)
  target_link_libraries(${test}
    ${PROJECT_NAME} Threads::Threads tf::default_settings ${CUBLAS_LIBRARIES}
  )
  target_include_directories(${test} PRIVATE ${CUBLAS_INCLUDE_DIRS})
  target_include_directories(${test} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)

  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${test} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
  
  doctest_discover_tests(${test})
endforeach()




