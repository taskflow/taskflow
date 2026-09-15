
list(APPEND TF_CUDA_UNITTESTS 
  test_cuda_objects
  test_cuda_basics 
  test_cuda_updates
  test_cuda_matrix 
  test_cuda_kmeans 
  test_cuda_for_each
  test_cuda_transform
  test_cuda_type_safe_kernel
  #test_cuda_reduce
  #test_cuda_scan
  #test_cuda_find
  #test_cuda_min_max_element
  #test_cuda_merge

  #test_cuda_capturer_optimizer
  #test_cuda_capture
  
  #cuda_algorithms
  #cuda_algorithm_updates
)

foreach(cudatest IN LISTS TF_CUDA_UNITTESTS)
  add_executable(${cudatest} ${cudatest}.cu)
  target_link_libraries(${cudatest} ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
  target_include_directories(${cudatest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  
  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cudatest} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
  
  doctest_discover_tests(${cudatest})
endforeach()

# ---------------------------------------------------------------------------
# Negative compile tests — these files MUST fail to compile.
# try_compile returns TRUE if compilation succeeds, FALSE if it fails.
# We expect failure, so we REQUIRE the result to be FALSE.
# ---------------------------------------------------------------------------

try_compile(
    TF_COMPILE_FAIL_ARG_COUNT_RESULT
    ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_compile_fail_arg_count.cu
    CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${CMAKE_SOURCE_DIR}"
        "-DCMAKE_CUDA_STANDARD=20"
    OUTPUT_VARIABLE TF_COMPILE_FAIL_ARG_COUNT_OUTPUT
)
if(TF_COMPILE_FAIL_ARG_COUNT_RESULT)
    message(FATAL_ERROR
        "test_compile_fail_arg_count.cu should have failed to compile "
        "(static_assert on wrong arg count), but it succeeded.\n"
        "Output:\n${TF_COMPILE_FAIL_ARG_COUNT_OUTPUT}")
endif()

try_compile(
    TF_COMPILE_FAIL_SCALAR_MISMATCH_RESULT
    ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_compile_fail_scalar_mismatch.cu
    CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${CMAKE_SOURCE_DIR}"
        "-DCMAKE_CUDA_STANDARD=20"
    OUTPUT_VARIABLE TF_COMPILE_FAIL_SCALAR_MISMATCH_OUTPUT
)
if(TF_COMPILE_FAIL_SCALAR_MISMATCH_RESULT)
    message(FATAL_ERROR
        "test_compile_fail_scalar_mismatch.cu should have failed to compile "
        "(incompatible struct to scalar cast), but it succeeded.\n"
        "Output:\n${TF_COMPILE_FAIL_SCALAR_MISMATCH_OUTPUT}")
endif()
