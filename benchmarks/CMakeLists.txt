# -----------------------------------------------------------------------------
# eigen package 
# -----------------------------------------------------------------------------
if(NOT DEFINED EIGEN_ROOT)
  set(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/3rd-party/eigen-3.3.7)
endif()
include_directories(${EIGEN_ROOT})

# -----------------------------------------------------------------------------
# find OpenMP package
# -----------------------------------------------------------------------------
include(FindOpenMP)

if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP not found")
endif()
  
message(STATUS "OpenMP_VERSION: ${OpenMP_VERSION}")
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")

# -----------------------------------------------------------------------------
# tbb package
# -----------------------------------------------------------------------------
cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)
if(NOT DEFINED TBB_ROOT_DIR)
  set(TBB_ROOT_DIR ${TF_3RD_PARTY_DIR}/tbb)
endif()

include(${TBB_ROOT_DIR}/cmake/TBBBuild.cmake)
#tbb_build(TBB_ROOT ${TBB_ROOT_DIR} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
tbb_build(TBB_ROOT ${TBB_ROOT_DIR} CONFIG_DIR TBB_DIR)
unset(TBB_ROOT)  # unset is needed to pop CMP0074 without warning
cmake_policy(POP)
#find_package(TBB REQUIRED tbb_preview)
find_package(TBB REQUIRED)
#cmake_policy(POP)

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

## benchmark 1: wavefront computing
add_executable(
  wavefront 
  ${TF_BENCHMARK_DIR}/wavefront/main.cpp
  ${TF_BENCHMARK_DIR}/wavefront/omp.cpp
  ${TF_BENCHMARK_DIR}/wavefront/tbb.cpp
  ${TF_BENCHMARK_DIR}/wavefront/seq.cpp
  ${TF_BENCHMARK_DIR}/wavefront/taskflow.cpp
)
target_include_directories(wavefront PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  wavefront 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(wavefront PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 2: graph traversal
add_executable(
  graph_traversal 
  ${TF_BENCHMARK_DIR}/graph_traversal/main.cpp
  ${TF_BENCHMARK_DIR}/graph_traversal/omp.cpp
  ${TF_BENCHMARK_DIR}/graph_traversal/tbb.cpp
  ${TF_BENCHMARK_DIR}/graph_traversal/seq.cpp
  ${TF_BENCHMARK_DIR}/graph_traversal/taskflow.cpp
)
target_include_directories(graph_traversal PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  graph_traversal 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(graph_traversal PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 3: binary_tree
add_executable(
  binary_tree 
  ${TF_BENCHMARK_DIR}/binary_tree/main.cpp
  ${TF_BENCHMARK_DIR}/binary_tree/tbb.cpp
  ${TF_BENCHMARK_DIR}/binary_tree/omp.cpp
  ${TF_BENCHMARK_DIR}/binary_tree/taskflow.cpp
)
target_include_directories(binary_tree PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  binary_tree 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(binary_tree PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 4: linear_chain
add_executable(
  linear_chain
  ${TF_BENCHMARK_DIR}/linear_chain/main.cpp
  ${TF_BENCHMARK_DIR}/linear_chain/tbb.cpp
  ${TF_BENCHMARK_DIR}/linear_chain/omp.cpp
  ${TF_BENCHMARK_DIR}/linear_chain/taskflow.cpp
)
target_include_directories(linear_chain PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  linear_chain 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(linear_chain PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 5: MNIST
add_executable(
  mnist 
  ${TF_BENCHMARK_DIR}/mnist/main.cpp
  ${TF_BENCHMARK_DIR}/mnist/omp.cpp
  ${TF_BENCHMARK_DIR}/mnist/tbb.cpp
  ${TF_BENCHMARK_DIR}/mnist/seq.cpp
  ${TF_BENCHMARK_DIR}/mnist/taskflow.cpp
)
target_include_directories(mnist PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  mnist 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS}
  ${OpenMP_CXX_LIBRARIES} 
  stdc++fs 
  tf::default_settings
)
set_target_properties(mnist PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 6: matrix multiplication
add_executable(
  matrix_multiplication 
  ${TF_BENCHMARK_DIR}/matrix_multiplication/main.cpp
  ${TF_BENCHMARK_DIR}/matrix_multiplication/omp.cpp
  ${TF_BENCHMARK_DIR}/matrix_multiplication/tbb.cpp
  ${TF_BENCHMARK_DIR}/matrix_multiplication/taskflow.cpp
)
target_include_directories(matrix_multiplication PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  matrix_multiplication 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(matrix_multiplication PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 7: mandelbrot
add_executable(
  mandelbrot 
  ${TF_BENCHMARK_DIR}/mandelbrot/main.cpp
  ${TF_BENCHMARK_DIR}/mandelbrot/omp.cpp
  ${TF_BENCHMARK_DIR}/mandelbrot/tbb.cpp
  ${TF_BENCHMARK_DIR}/mandelbrot/taskflow.cpp
)
target_include_directories(mandelbrot PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  mandelbrot 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(mandelbrot PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 8: black_scholes
add_executable(
  black_scholes
  ${TF_BENCHMARK_DIR}/black_scholes/main.cpp
  ${TF_BENCHMARK_DIR}/black_scholes/omp.cpp
  ${TF_BENCHMARK_DIR}/black_scholes/tbb.cpp
  ${TF_BENCHMARK_DIR}/black_scholes/taskflow.cpp
)
target_include_directories(black_scholes PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  black_scholes 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(black_scholes PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 9: reduce_sum
add_executable(
  reduce_sum
  ${TF_BENCHMARK_DIR}/reduce_sum/main.cpp
  ${TF_BENCHMARK_DIR}/reduce_sum/omp.cpp
  ${TF_BENCHMARK_DIR}/reduce_sum/tbb.cpp
  ${TF_BENCHMARK_DIR}/reduce_sum/taskflow.cpp
)
target_include_directories(reduce_sum PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  reduce_sum 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(reduce_sum PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 10: sort
add_executable(
  sort
  ${TF_BENCHMARK_DIR}/sort/main.cpp
  ${TF_BENCHMARK_DIR}/sort/omp.cpp
  ${TF_BENCHMARK_DIR}/sort/tbb.cpp
  ${TF_BENCHMARK_DIR}/sort/taskflow.cpp
)
target_include_directories(sort PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  sort 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(sort PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})


## benchmark 11: linear_pipeline
add_executable(
  linear_pipeline
  ${TF_BENCHMARK_DIR}/linear_pipeline/main.cpp
  ${TF_BENCHMARK_DIR}/linear_pipeline/omp.cpp
  ${TF_BENCHMARK_DIR}/linear_pipeline/tbb.cpp
  ${TF_BENCHMARK_DIR}/linear_pipeline/taskflow.cpp
)
target_include_directories(linear_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  linear_pipeline 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(linear_pipeline PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})


## benchmark 12: graph_pipeline
add_executable(
  graph_pipeline
  ${TF_BENCHMARK_DIR}/graph_pipeline/main.cpp
  ${TF_BENCHMARK_DIR}/graph_pipeline/omp.cpp
  ${TF_BENCHMARK_DIR}/graph_pipeline/tbb.cpp
  ${TF_BENCHMARK_DIR}/graph_pipeline/taskflow.cpp
  ${TF_BENCHMARK_DIR}/graph_pipeline/gold.cpp
  #${TF_BENCHMARK_DIR}/graph_pipeline/fastflow.cpp
)
target_include_directories(graph_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_include_directories(graph_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/)
target_link_libraries(
  graph_pipeline 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(graph_pipeline PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 13: comparison with simple thread pool
add_executable(
  thread_pool
  ${TF_BENCHMARK_DIR}/thread_pool/thread_pool.cpp
)
target_include_directories(thread_pool PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  thread_pool
  ${PROJECT_NAME}
  ${TBB_IMPORTED_TARGETS}
  ${OpenMP_CXX_LIBRARIES}
  tf::default_settings
)
set_target_properties(thread_pool PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 14: data_pipeline
add_executable(
  data_pipeline
  ${TF_BENCHMARK_DIR}/data_pipeline/main.cpp
  ${TF_BENCHMARK_DIR}/data_pipeline/tbb.cpp
  ${TF_BENCHMARK_DIR}/data_pipeline/taskflow.cpp
)
target_include_directories(data_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  data_pipeline 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(data_pipeline PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 15: deferred_pipeline                                                                               
add_executable(
  deferred_pipeline
  ${TF_BENCHMARK_DIR}/deferred_pipeline/main.cpp
  ${TF_BENCHMARK_DIR}/deferred_pipeline/pthread.cpp
  ${TF_BENCHMARK_DIR}/deferred_pipeline/taskflow.cpp
)
target_include_directories(deferred_pipeline PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  deferred_pipeline
  ${PROJECT_NAME}
  ${OpenMP_CXX_LIBRARIES}
  tf::default_settings
)
set_target_properties(deferred_pipeline PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 16: for_each
add_executable(
  for_each
  ${TF_BENCHMARK_DIR}/for_each/main.cpp
  ${TF_BENCHMARK_DIR}/for_each/omp.cpp
  ${TF_BENCHMARK_DIR}/for_each/tbb.cpp
  ${TF_BENCHMARK_DIR}/for_each/taskflow.cpp
)
target_include_directories(for_each PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  for_each 
  ${PROJECT_NAME} 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(for_each PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})


###############################################################################
# CUDA benchmarks
###############################################################################

if(TF_BUILD_CUDA)

  ## cuda benchmark 1: heterogeneous traversal
  add_executable(
    hetero_traversal 
    ${TF_BENCHMARK_DIR}/hetero_traversal/main.cu
    ${TF_BENCHMARK_DIR}/hetero_traversal/taskflow.cu
    ${TF_BENCHMARK_DIR}/hetero_traversal/tbb.cu
    ${TF_BENCHMARK_DIR}/hetero_traversal/omp.cu
  )
  
  target_include_directories(
    hetero_traversal PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11
  )
  
  target_link_libraries(
    hetero_traversal 
    ${PROJECT_NAME} 
    Threads::Threads 
    ${TBB_IMPORTED_TARGETS} 
    ${OpenMP_CXX_LIBRARIES} 
    tf::default_settings
  )
  
  set_target_properties(
    hetero_traversal PROPERTIES COMPILE_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS}"
  )
  
  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET hetero_traversal PROPERTY CUDA_ARCHITECTURES OFF)
  endif()


endif(TF_BUILD_CUDA)

