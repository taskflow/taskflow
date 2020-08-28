# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Project name
project(Taskflow VERSION 2.6.0 LANGUAGES CXX)

# build options
option(TF_BUILD_BENCHMARKS "Enables build of benchmarks" OFF)
option(TF_BUILD_CUDA "Enables build of CUDA code" ON)
option(TF_BUILD_TESTS "Enables build of tests" ON)
option(TF_BUILD_EXAMPLES "Enables build of examples" ON)

# Include additional language check
include(CheckLanguage)

# Compiler vendors
## g++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires g++ at least v5.0")
  endif()
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.0")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires clang++ at least v4.0")
  endif() 
## AppleClang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0")
    message(FATAL_ERROR "\nTaskflow requires AppleClang at least v8.0")
  endif()
## microsoft visual c++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_definitions(/bigobj)
  if(NOT MSVC_VERSION GREATER_EQUAL 1914)
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires MSVC++ at least v14.14") 
  endif()
else()
  message(FATAL_ERROR "\n\
Taskflow currently supports the following compilers:\n\
  - g++ v5.0 or above\n\
  - clang++ v4.0 or above\n\
  - MSVC++ v19.14 or above\n\
  - AppleClang v8 or above\n\
")
endif()

# NVCC Compiler
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires CUDA at least v10")
  endif()
endif(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

# defult release build
set(TF_DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${TF_DEFAULT_BUILD_TYPE}'")
  set(
    CMAKE_BUILD_TYPE "${TF_DEFAULT_BUILD_TYPE}" 
    CACHE
    STRING "Choose the type of build." 
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE 
    CMAKE_BUILD_TYPE 
    PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()


# error setting
add_library(error_settings INTERFACE)
add_library(tf::error_settings ALIAS error_settings)

target_compile_options(
  error_settings
  INTERFACE
  $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wfatal-errors>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /permissive->
  #$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wfatal-errors>
  #$<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wfatal-errors>
  #$<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wfatal-errors>
  #$<$<CXX_COMPILER_ID:MSVC>:/W3 /permissive->
  #$<$<CXX_COMPILER_ID:NVIDIA>:-Xcompiler=-Wall,-Wextra,-Wfatal-errors>
)

if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  target_compile_options(
    error_settings
    BEFORE
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wfatal-errors>
  )
endif(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

# additional features
add_library(features_settings INTERFACE)
add_library(tf::features_settings ALIAS features_settings)
target_compile_definitions(
  features_settings 
  INTERFACE 
  $<$<CXX_COMPILER_ID:MSVC>:_ENABLE_EXTENDED_ALIGNED_STORAGE>
)

# optimization
#
##! Msvc flags info
# /Zi - Produces a program database (PDB) that contains type information 
#       and symbolic debugging information for use with the debugger.
# /FS - Allows multiple cl.exe processes to write to the same .pdb file
# /DEBUG - Enable debug during linking
# /Od - Disables optimization
# /Ox - Full optimization
# /Oy- do not suppress frame pointers (recommended for debugging)
#
#add_library(optimize_settings INTERFACE)
#add_library(tf::optimize_settings ALIAS optimize_settings)
#
#target_compile_options(
#  optimize_settings INTERFACE
#  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-O2 -march=native>
#  #$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang>>:-O2 -march=native>
#  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-O2 -march=native>
#  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2 -DNDEBUG /MP>
#  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0 -g>
#  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0 -g>
#  #$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:AppleClang>>:-O0 -g>
#  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi /FS /DEBUG /Od /MP /MDd /Oy->
#)
#
add_library(default_settings INTERFACE)
add_library(tf::default_settings ALIAS default_settings)
target_link_libraries(
  default_settings 
  INTERFACE 
  tf::error_settings 
  #tf::optimize_settings 
  tf::features_settings
)

# CXX target properties
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

# installation path
set(TF_INC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(TF_LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(TF_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
set(TF_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmarks)
set(TF_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
message(STATUS "TF_BUILD_BENCHMARKS: " ${TF_BUILD_BENCHMARKS})
message(STATUS "TF_BUILD_CUDA: " ${TF_BUILD_CUDA})
message(STATUS "TF_BUILD_TESTS: " ${TF_BUILD_TESTS})
message(STATUS "TF_BUILD_EXAMPLES: " ${TF_BUILD_EXAMPLES})
message(STATUS "TF_INC_INSTALL_DIR: " ${TF_INC_INSTALL_DIR})
message(STATUS "TF_LIB_INSTALL_DIR: " ${TF_LIB_INSTALL_DIR})
message(STATUS "TF_UTEST_DIR: " ${TF_UTEST_DIR})
message(STATUS "TF_EXAMPLE_DIR: " ${TF_EXAMPLE_DIR})
message(STATUS "TF_BENCHMARK_DIR: " ${TF_BENCHMARK_DIR})
message(STATUS "TF_3RD_PARTY_DIR: " ${TF_3RD_PARTY_DIR})

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

# Enable test
include(CTest)

# Find pthread package
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Taskflow library interface
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)
target_link_libraries(${PROJECT_NAME} INTERFACE Threads::Threads)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
if(TF_BUILD_EXAMPLES)

message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_EXAMPLE_DIR})

add_executable(simple ${TF_EXAMPLE_DIR}/simple.cpp)
target_link_libraries(
  simple ${PROJECT_NAME} tf::default_settings
)

add_executable(observer ${TF_EXAMPLE_DIR}/observer.cpp)
target_link_libraries(
  observer ${PROJECT_NAME} tf::default_settings
)

add_executable(subflow ${TF_EXAMPLE_DIR}/subflow.cpp)
target_link_libraries(
  subflow ${PROJECT_NAME} tf::default_settings
)

add_executable(fibonacci ${TF_EXAMPLE_DIR}/fibonacci.cpp)
target_link_libraries(
  fibonacci ${PROJECT_NAME} tf::default_settings
)

add_executable(condition ${TF_EXAMPLE_DIR}/condition.cpp)
target_link_libraries(
  condition ${PROJECT_NAME} tf::default_settings
)

add_executable(simple_dsl ${TF_EXAMPLE_DIR}/simple_dsl.cpp)
target_link_libraries(
  simple_dsl ${PROJECT_NAME} tf::default_settings
)

add_executable(visualization_dsl ${TF_EXAMPLE_DIR}/visualization_dsl.cpp)
target_link_libraries(
  visualization_dsl ${PROJECT_NAME} tf::default_settings
)

add_executable(visualization ${TF_EXAMPLE_DIR}/visualization.cpp)
target_link_libraries(
  visualization ${PROJECT_NAME} tf::default_settings
)

add_executable(reduce ${TF_EXAMPLE_DIR}/reduce.cpp)
target_link_libraries(
  reduce ${PROJECT_NAME} tf::default_settings
)

add_executable(parallel_for ${TF_EXAMPLE_DIR}/parallel_for.cpp)
target_link_libraries(
  parallel_for ${PROJECT_NAME} tf::default_settings
)

add_executable(run ${TF_EXAMPLE_DIR}/run.cpp)
target_link_libraries(
  run ${PROJECT_NAME} tf::default_settings
)

add_executable(composition ${TF_EXAMPLE_DIR}/composition.cpp)
target_link_libraries(
 composition ${PROJECT_NAME} tf::default_settings
)

#### CUDA examples
if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_EXAMPLE_DIR}/cuda)

add_executable(device_property ${TF_EXAMPLE_DIR}/cuda/device_property.cu)
target_link_libraries(
  device_property ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(saxpy ${TF_EXAMPLE_DIR}/cuda/saxpy.cu)
target_link_libraries(
  saxpy ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(matmul ${TF_EXAMPLE_DIR}/cuda/matmul.cu)
target_link_libraries(
  matmul ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(kmeans ${TF_EXAMPLE_DIR}/cuda/kmeans.cu)
target_link_libraries(
  kmeans ${PROJECT_NAME} Threads::Threads tf::default_settings
)
endif(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

endif(TF_BUILD_EXAMPLES)

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
if(TF_BUILD_TESTS)

enable_testing()
message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_UTEST_DIR})

# unittest for utility
add_executable(utility ${TF_UTEST_DIR}/utility.cpp)
target_link_libraries(utility ${PROJECT_NAME} tf::default_settings)
target_include_directories(utility PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_UTILITY ${TF_UTEST_DIR}/utility)
add_test(passive_vector  ${TF_UTEST_UTILITY} -tc=PassiveVector)
add_test(function_traits ${TF_UTEST_UTILITY} -tc=FunctionTraits)
add_test(uuid  ${TF_UTEST_UTILITY} -tc=uuid)
add_test(uuid.10threads ${TF_UTEST_UTILITY} -tc=uuid)
add_test(uuid.100threads ${TF_UTEST_UTILITY} -tc=uuid.100threads)
add_test(distance.integral ${TF_UTEST_UTILITY} -tc=distance.integral)
add_test(object_pool.sequential ${TF_UTEST_UTILITY} -tc=ObjectPool.Sequential)
add_test(object_pool.1thread ${TF_UTEST_UTILITY} -tc=ObjectPool.1thread)
add_test(object_pool.2threads ${TF_UTEST_UTILITY} -tc=ObjectPool.2threads)
add_test(object_pool.3threads ${TF_UTEST_UTILITY} -tc=ObjectPool.3threads)
add_test(object_pool.4threads ${TF_UTEST_UTILITY} -tc=ObjectPool.4threads)
add_test(object_pool.5threads ${TF_UTEST_UTILITY} -tc=ObjectPool.5threads)
add_test(object_pool.6threads ${TF_UTEST_UTILITY} -tc=ObjectPool.6threads)
add_test(object_pool.7threads ${TF_UTEST_UTILITY} -tc=ObjectPool.7threads)
add_test(object_pool.8threads ${TF_UTEST_UTILITY} -tc=ObjectPool.8threads)
add_test(object_pool.9threads ${TF_UTEST_UTILITY} -tc=ObjectPool.9threads)
add_test(object_pool.10threads ${TF_UTEST_UTILITY} -tc=ObjectPool.10threads)
add_test(object_pool.11threads ${TF_UTEST_UTILITY} -tc=ObjectPool.11threads)
add_test(object_pool.12threads ${TF_UTEST_UTILITY} -tc=ObjectPool.12threads)
add_test(object_pool.13threads ${TF_UTEST_UTILITY} -tc=ObjectPool.13threads)
add_test(object_pool.14threads ${TF_UTEST_UTILITY} -tc=ObjectPool.14threads)
add_test(object_pool.15threads ${TF_UTEST_UTILITY} -tc=ObjectPool.15threads)
add_test(object_pool.16threads ${TF_UTEST_UTILITY} -tc=ObjectPool.16threads)

# unittest for WorkStealingQueue 
add_executable(tsq ${TF_UTEST_DIR}/tsq.cpp)
target_link_libraries(tsq ${PROJECT_NAME})
target_include_directories(tsq PRIVATE ${TF_3RD_PARTY_DIR}/doctest tf::default_settings)
set(TF_UTEST_TSQ ${TF_UTEST_DIR}/tsq)
add_test(tsq.Owner  ${TF_UTEST_TSQ} -tc=TSQ.Owner)
add_test(tsq.1Thief ${TF_UTEST_TSQ} -tc=TSQ.1Thief)
add_test(tsq.2Thieves ${TF_UTEST_TSQ} -tc=TSQ.2Thieves)
add_test(tsq.3Thieves ${TF_UTEST_TSQ} -tc=TSQ.3Thieves)
add_test(tsq.4Thieves ${TF_UTEST_TSQ} -tc=TSQ.4Thieves)
add_test(tsq.5Thieves ${TF_UTEST_TSQ} -tc=TSQ.5Thieves)
add_test(tsq.6Thieves ${TF_UTEST_TSQ} -tc=TSQ.6Thieves)
add_test(tsq.7Thieves ${TF_UTEST_TSQ} -tc=TSQ.7Thieves)
add_test(tsq.8Thieves ${TF_UTEST_TSQ} -tc=TSQ.8Thieves)

# unittest for basics
add_executable(basics ${TF_UTEST_DIR}/basics.cpp)
target_link_libraries(basics ${PROJECT_NAME} tf::default_settings)
target_include_directories(basics PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_BASICS ${TF_UTEST_DIR}/basics)
add_test(builder ${TF_UTEST_BASICS} -tc=Builder)
add_test(creation ${TF_UTEST_BASICS} -tc=Creation)
add_test(iterators ${TF_UTEST_BASICS} -tc=Iterators)       
add_test(std_function ${TF_UTEST_BASICS} -tc=STDFunction)       
add_test(hash ${TF_UTEST_BASICS} -tc=Hash)
add_test(worker_id.1thread ${TF_UTEST_BASICS} -tc=WorkerID.1thread)
add_test(worker_id.2threads ${TF_UTEST_BASICS} -tc=WorkerID.2threads)
add_test(worker_id.3threads ${TF_UTEST_BASICS} -tc=WorkerID.3threads)
add_test(worker_id.4threads ${TF_UTEST_BASICS} -tc=WorkerID.4threads)
add_test(worker_id.5threads ${TF_UTEST_BASICS} -tc=WorkerID.5threads)
add_test(worker_id.6threads ${TF_UTEST_BASICS} -tc=WorkerID.6threads)
add_test(worker_id.7threads ${TF_UTEST_BASICS} -tc=WorkerID.7threads)
add_test(worker_id.8threads ${TF_UTEST_BASICS} -tc=WorkerID.8threads)
add_test(serial_runs.1thread ${TF_UTEST_BASICS} -tc=SerialRuns.1thread)
add_test(serial_runs.2threads ${TF_UTEST_BASICS} -tc=SerialRuns.2threads)
add_test(serial_runs.3threads ${TF_UTEST_BASICS} -tc=SerialRuns.3threads)
add_test(serial_runs.4threads ${TF_UTEST_BASICS} -tc=SerialRuns.4threads)
add_test(serial_runs.5threads ${TF_UTEST_BASICS} -tc=SerialRuns.5threads)
add_test(serial_runs.6threads ${TF_UTEST_BASICS} -tc=SerialRuns.6threads)
add_test(serial_runs.7threads ${TF_UTEST_BASICS} -tc=SerialRuns.7threads)
add_test(serial_runs.8threads ${TF_UTEST_BASICS} -tc=SerialRuns.8threads)
add_test(parallel_runs.1thread  ${TF_UTEST_BASICS} -tc=ParallelRuns.1thread)
add_test(parallel_runs.2threads ${TF_UTEST_BASICS} -tc=ParallelRuns.2threads)
add_test(parallel_runs.3threads ${TF_UTEST_BASICS} -tc=ParallelRuns.3threads)
add_test(parallel_runs.4threads ${TF_UTEST_BASICS} -tc=ParallelRuns.4threads)
add_test(parallel_runs.5threads ${TF_UTEST_BASICS} -tc=ParallelRuns.5threads)
add_test(parallel_runs.6threads ${TF_UTEST_BASICS} -tc=ParallelRuns.6threads)
add_test(parallel_runs.7threads ${TF_UTEST_BASICS} -tc=ParallelRuns.7threads)
add_test(parallel_runs.8threads ${TF_UTEST_BASICS} -tc=ParallelRuns.8threads)
add_test(nested_runs.1thread  ${TF_UTEST_BASICS} -tc=NestedRuns.1thread)
add_test(nested_runs.2threads ${TF_UTEST_BASICS} -tc=NestedRuns.2threads)
add_test(nested_runs.3threads ${TF_UTEST_BASICS} -tc=NestedRuns.3threads)
add_test(nested_runs.4threads ${TF_UTEST_BASICS} -tc=NestedRuns.4threads)
add_test(nested_runs.8threads ${TF_UTEST_BASICS} -tc=NestedRuns.8threads)
add_test(nested_runs.16threads ${TF_UTEST_BASICS} -tc=NestedRuns.16threads)
add_test(parallel_for.1thread ${TF_UTEST_BASICS} -tc=ParallelFor.1thread)
add_test(parallel_for.2threads ${TF_UTEST_BASICS} -tc=ParallelFor.2threads)
add_test(parallel_for.3threads ${TF_UTEST_BASICS} -tc=ParallelFor.3threads)
add_test(parallel_for.4threads ${TF_UTEST_BASICS} -tc=ParallelFor.4threads)
add_test(parallel_for.5threads ${TF_UTEST_BASICS} -tc=ParallelFor.5threads)
add_test(parallel_for.6threads ${TF_UTEST_BASICS} -tc=ParallelFor.6threads)
add_test(parallel_for.7threads ${TF_UTEST_BASICS} -tc=ParallelFor.7threads)
add_test(parallel_for.8threads ${TF_UTEST_BASICS} -tc=ParallelFor.8threads)
add_test(parallel_for_idx.1thread ${TF_UTEST_BASICS} -tc=ParallelForIndex.1thread)
add_test(parallel_for_idx.2threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.2threads)
add_test(parallel_for_idx.3threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.3threads)
add_test(parallel_for_idx.4threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.4threads)
add_test(parallel_for_idx.5threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.5threads)
add_test(parallel_for_idx.6threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.6threads)
add_test(parallel_for_idx.7threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.7threads)
add_test(parallel_for_idx.8threads ${TF_UTEST_BASICS} -tc=ParallelForIndex.8threads)
add_test(reduce     ${TF_UTEST_BASICS} -tc=Reduce)
add_test(reduce_min ${TF_UTEST_BASICS} -tc=ReduceMin)
add_test(reduce_max ${TF_UTEST_BASICS} -tc=ReduceMax)
add_test(joined_subflow.1thread ${TF_UTEST_BASICS} -tc=JoinedSubflow.1thread)
add_test(joined_subflow.2threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.2threads)
add_test(joined_subflow.3threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.3threads)
add_test(joined_subflow.4threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.4threads)
add_test(joined_subflow.5threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.5threads)
add_test(joined_subflow.6threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.6threads)
add_test(joined_subflow.7threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.7threads)
add_test(joined_subflow.8threads ${TF_UTEST_BASICS} -tc=JoinedSubflow.8threads)
add_test(detached_subflow.1thread ${TF_UTEST_BASICS} -tc=DetachedSubflow.1thread)
add_test(detached_subflow.2threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.2threads)
add_test(detached_subflow.3threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.3threads)
add_test(detached_subflow.4threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.4threads)
add_test(detached_subflow.5threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.5threads)
add_test(detached_subflow.6threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.6threads)
add_test(detached_subflow.7threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.7threads)
add_test(detached_subflow.8threads ${TF_UTEST_BASICS} -tc=DetachedSubflow.8threads)
add_test(tree_subflow ${TF_UTEST_BASICS} -tc=TreeSubflow)
add_test(fib_subflow.1thread ${TF_UTEST_BASICS} -tc=FibSubflow.1thread)
add_test(fib_subflow.2threads ${TF_UTEST_BASICS} -tc=FibSubflow.2threads)
add_test(fib_subflow.4threads ${TF_UTEST_BASICS} -tc=FibSubflow.4threads)
add_test(fib_subflow.5threads ${TF_UTEST_BASICS} -tc=FibSubflow.5threads)
add_test(fib_subflow.6threads ${TF_UTEST_BASICS} -tc=FibSubflow.6threads)
add_test(fib_subflow.7threads ${TF_UTEST_BASICS} -tc=FibSubflow.7threads)
add_test(fib_subflow.8threads ${TF_UTEST_BASICS} -tc=FibSubflow.8threads)
add_test(composition-1    ${TF_UTEST_BASICS} -tc=Composition-1)
add_test(composition-2    ${TF_UTEST_BASICS} -tc=Composition-2)
add_test(composition-3    ${TF_UTEST_BASICS} -tc=Composition-3)
add_test(observer.1thread  ${TF_UTEST_BASICS} -tc=Observer.1thread)
add_test(observer.2threads ${TF_UTEST_BASICS} -tc=Observer.2threads)
add_test(observer.3threads ${TF_UTEST_BASICS} -tc=Observer.3threads)
add_test(observer.4threads ${TF_UTEST_BASICS} -tc=Observer.4threads)
add_test(loop_cond.1thread  ${TF_UTEST_BASICS} -tc=LoopCond.1thread)
add_test(loop_cond.2threads ${TF_UTEST_BASICS} -tc=LoopCond.2threads)
add_test(loop_cond.3threads ${TF_UTEST_BASICS} -tc=LoopCond.3threads)
add_test(loop_cond.4threads ${TF_UTEST_BASICS} -tc=LoopCond.4threads)
add_test(flip_coin_cond.1thread  ${TF_UTEST_BASICS} -tc=FlipCoinCond.1thread)
add_test(flip_coin_cond.2threads ${TF_UTEST_BASICS} -tc=FlipCoinCond.2threads)
add_test(flip_coin_cond.3threads ${TF_UTEST_BASICS} -tc=FlipCoinCond.3threads)
add_test(flip_coin_cond.4threads ${TF_UTEST_BASICS} -tc=FlipCoinCond.4threads)
add_test(cyclic_cond.1thread  ${TF_UTEST_BASICS} -tc=CyclicCond.1thread)
add_test(cyclic_cond.2threads ${TF_UTEST_BASICS} -tc=CyclicCond.2threads)
add_test(cyclic_cond.3threads ${TF_UTEST_BASICS} -tc=CyclicCond.3threads)
add_test(cyclic_cond.4threads ${TF_UTEST_BASICS} -tc=CyclicCond.4threads)
add_test(cyclic_cond.5threads ${TF_UTEST_BASICS} -tc=CyclicCond.5threads)
add_test(cyclic_cond.6threads ${TF_UTEST_BASICS} -tc=CyclicCond.6threads)
add_test(cyclic_cond.7threads ${TF_UTEST_BASICS} -tc=CyclicCond.7threads)
add_test(cyclic_cond.8threads ${TF_UTEST_BASICS} -tc=CyclicCond.8threads)
add_test(btree_cond ${TF_UTEST_BASICS} -tc=BTreeCondition)
add_test(dynamic_btree_cond   ${TF_UTEST_BASICS} -tc=DynamicBTreeCondition)
add_test(nested_cond.1thread  ${TF_UTEST_BASICS} -tc=NestedCond.1thread)
add_test(nested_cond.2threads ${TF_UTEST_BASICS} -tc=NestedCond.2threads)
add_test(nested_cond.3threads ${TF_UTEST_BASICS} -tc=NestedCond.3threads)
add_test(nested_cond.4threads ${TF_UTEST_BASICS} -tc=NestedCond.4threads)
add_test(nested_cond.5threads ${TF_UTEST_BASICS} -tc=NestedCond.5threads)
add_test(nested_cond.6threads ${TF_UTEST_BASICS} -tc=NestedCond.6threads)
add_test(nested_cond.7threads ${TF_UTEST_BASICS} -tc=NestedCond.7threads)
add_test(nested_cond.8threads ${TF_UTEST_BASICS} -tc=NestedCond.8threads)
add_test(cond2cond.1thread  ${TF_UTEST_BASICS} -tc=Cond2Cond.1thread)
add_test(cond2cond.2threads ${TF_UTEST_BASICS} -tc=Cond2Cond.2threads)
add_test(cond2cond.3threads ${TF_UTEST_BASICS} -tc=Cond2Cond.3threads)
add_test(cond2cond.4threads ${TF_UTEST_BASICS} -tc=Cond2Cond.4threads)
add_test(cond2cond.5threads ${TF_UTEST_BASICS} -tc=Cond2Cond.5threads)
add_test(cond2cond.6threads ${TF_UTEST_BASICS} -tc=Cond2Cond.6threads)
add_test(cond2cond.7threads ${TF_UTEST_BASICS} -tc=Cond2Cond.7threads)
add_test(cond2cond.8threads ${TF_UTEST_BASICS} -tc=Cond2Cond.8threads)
add_test(hier_cond.1thread  ${TF_UTEST_BASICS} -tc=HierCondition.1thread)
add_test(hier_cond.2threads ${TF_UTEST_BASICS} -tc=HierCondition.2threads)
add_test(hier_cond.3threads ${TF_UTEST_BASICS} -tc=HierCondition.3threads)
add_test(hier_cond.4threads ${TF_UTEST_BASICS} -tc=HierCondition.4threads)
add_test(hier_cond.5threads ${TF_UTEST_BASICS} -tc=HierCondition.5threads)
add_test(hier_cond.6threads ${TF_UTEST_BASICS} -tc=HierCondition.6threads)
add_test(hier_cond.7threads ${TF_UTEST_BASICS} -tc=HierCondition.7threads)
add_test(hier_cond.8threads ${TF_UTEST_BASICS} -tc=HierCondition.8threads)
add_test(cond_subflow.1thread ${TF_UTEST_BASICS} -tc=CondSubflow.1thread)
add_test(cond_subflow.2threads ${TF_UTEST_BASICS} -tc=CondSubflow.2threads)
add_test(cond_subflow.3threads ${TF_UTEST_BASICS} -tc=CondSubflow.3threads)
add_test(cond_subflow.4threads ${TF_UTEST_BASICS} -tc=CondSubflow.4threads)
add_test(cond_subflow.5threads ${TF_UTEST_BASICS} -tc=CondSubflow.5threads)
add_test(cond_subflow.6threads ${TF_UTEST_BASICS} -tc=CondSubflow.6threads)
add_test(cond_subflow.7threads ${TF_UTEST_BASICS} -tc=CondSubflow.7threads)
add_test(cond_subflow.8threads ${TF_UTEST_BASICS} -tc=CondSubflow.8threads)

# unittest for algorithm
add_executable(algorithm ${TF_UTEST_DIR}/algorithm.cpp)
target_link_libraries(algorithm ${PROJECT_NAME} tf::default_settings)
target_include_directories(algorithm PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_ALGORITHM ${TF_UTEST_DIR}/algorithm)
add_test(pfg.1thread ${TF_UTEST_ALGORITHM} -tc=pfg.1thread)
add_test(pfg.2threads ${TF_UTEST_ALGORITHM} -tc=pfg.2threads)
add_test(pfg.3threads ${TF_UTEST_ALGORITHM} -tc=pfg.3threads)
add_test(pfg.4threads ${TF_UTEST_ALGORITHM} -tc=pfg.4threads)
add_test(pfg.5threads ${TF_UTEST_ALGORITHM} -tc=pfg.5threads)
add_test(pfg.6threads ${TF_UTEST_ALGORITHM} -tc=pfg.6threads)
add_test(pfg.7threads ${TF_UTEST_ALGORITHM} -tc=pfg.7threads)
add_test(pfg.8threads ${TF_UTEST_ALGORITHM} -tc=pfg.8threads)
add_test(pfg.9threads ${TF_UTEST_ALGORITHM} -tc=pfg.9threads)
add_test(pfg.10threads ${TF_UTEST_ALGORITHM} -tc=pfg.10threads)
add_test(pfg.11threads ${TF_UTEST_ALGORITHM} -tc=pfg.11threads)
add_test(pfg.12threads ${TF_UTEST_ALGORITHM} -tc=pfg.12threads)
add_test(pfd.1thread ${TF_UTEST_ALGORITHM} -tc=pfd.1thread)
add_test(pfd.2threads ${TF_UTEST_ALGORITHM} -tc=pfd.2threads)
add_test(pfd.3threads ${TF_UTEST_ALGORITHM} -tc=pfd.3threads)
add_test(pfd.4threads ${TF_UTEST_ALGORITHM} -tc=pfd.4threads)
add_test(pfd.5threads ${TF_UTEST_ALGORITHM} -tc=pfd.5threads)
add_test(pfd.6threads ${TF_UTEST_ALGORITHM} -tc=pfd.6threads)
add_test(pfd.7threads ${TF_UTEST_ALGORITHM} -tc=pfd.7threads)
add_test(pfd.8threads ${TF_UTEST_ALGORITHM} -tc=pfd.8threads)
add_test(pfd.9threads ${TF_UTEST_ALGORITHM} -tc=pfd.9threads)
add_test(pfd.10threads ${TF_UTEST_ALGORITHM} -tc=pfd.10threads)
add_test(pfd.11threads ${TF_UTEST_ALGORITHM} -tc=pfd.11threads)
add_test(pfd.12threads ${TF_UTEST_ALGORITHM} -tc=pfd.12threads)
add_test(pfs.1thread ${TF_UTEST_ALGORITHM} -tc=pfs.1thread)
add_test(pfs.2threads ${TF_UTEST_ALGORITHM} -tc=pfs.2threads)
add_test(pfs.3threads ${TF_UTEST_ALGORITHM} -tc=pfs.3threads)
add_test(pfs.4threads ${TF_UTEST_ALGORITHM} -tc=pfs.4threads)
add_test(pfs.5threads ${TF_UTEST_ALGORITHM} -tc=pfs.5threads)
add_test(pfs.6threads ${TF_UTEST_ALGORITHM} -tc=pfs.6threads)
add_test(pfs.7threads ${TF_UTEST_ALGORITHM} -tc=pfs.7threads)
add_test(pfs.8threads ${TF_UTEST_ALGORITHM} -tc=pfs.8threads)
add_test(pfs.9threads ${TF_UTEST_ALGORITHM} -tc=pfs.9threads)
add_test(pfs.10threads ${TF_UTEST_ALGORITHM} -tc=pfs.10threads)
add_test(pfs.11threads ${TF_UTEST_ALGORITHM} -tc=pfs.11threads)
add_test(pfs.12threads ${TF_UTEST_ALGORITHM} -tc=pfs.12threads)
add_test(statefulpfg.1thread ${TF_UTEST_ALGORITHM} -tc=statefulpfg.1thread)
add_test(statefulpfg.2threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.2threads)
add_test(statefulpfg.3threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.3threads)
add_test(statefulpfg.4threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.4threads)
add_test(statefulpfg.5threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.5threads)
add_test(statefulpfg.6threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.6threads)
add_test(statefulpfg.7threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.7threads)
add_test(statefulpfg.8threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.8threads)
add_test(statefulpfg.9threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.9threads)
add_test(statefulpfg.10threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.10threads)
add_test(statefulpfg.11threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.11threads)
add_test(statefulpfg.12threads ${TF_UTEST_ALGORITHM} -tc=statefulpfg.12threads)
add_test(statefulpfd.1thread ${TF_UTEST_ALGORITHM} -tc=statefulpfd.1thread)
add_test(statefulpfd.2threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.2threads)
add_test(statefulpfd.3threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.3threads)
add_test(statefulpfd.4threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.4threads)
add_test(statefulpfd.5threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.5threads)
add_test(statefulpfd.6threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.6threads)
add_test(statefulpfd.7threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.7threads)
add_test(statefulpfd.8threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.8threads)
add_test(statefulpfd.9threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.9threads)
add_test(statefulpfd.10threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.10threads)
add_test(statefulpfd.11threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.11threads)
add_test(statefulpfd.12threads ${TF_UTEST_ALGORITHM} -tc=statefulpfd.12threads)
add_test(statefulpfs.1thread ${TF_UTEST_ALGORITHM} -tc=statefulpfs.1thread)
add_test(statefulpfs.2threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.2threads)
add_test(statefulpfs.3threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.3threads)
add_test(statefulpfs.4threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.4threads)
add_test(statefulpfs.5threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.5threads)
add_test(statefulpfs.6threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.6threads)
add_test(statefulpfs.7threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.7threads)
add_test(statefulpfs.8threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.8threads)
add_test(statefulpfs.9threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.9threads)
add_test(statefulpfs.10threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.10threads)
add_test(statefulpfs.11threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.11threads)
add_test(statefulpfs.12threads ${TF_UTEST_ALGORITHM} -tc=statefulpfs.12threads)
add_test(prg.1thread ${TF_UTEST_ALGORITHM} -tc=prg.1thread)
add_test(prg.2threads ${TF_UTEST_ALGORITHM} -tc=prg.2threads)
add_test(prg.3threads ${TF_UTEST_ALGORITHM} -tc=prg.3threads)
add_test(prg.4threads ${TF_UTEST_ALGORITHM} -tc=prg.4threads)
add_test(prg.5threads ${TF_UTEST_ALGORITHM} -tc=prg.5threads)
add_test(prg.6threads ${TF_UTEST_ALGORITHM} -tc=prg.6threads)
add_test(prg.7threads ${TF_UTEST_ALGORITHM} -tc=prg.7threads)
add_test(prg.8threads ${TF_UTEST_ALGORITHM} -tc=prg.8threads)
add_test(prg.9threads ${TF_UTEST_ALGORITHM} -tc=prg.9threads)
add_test(prg.10threads ${TF_UTEST_ALGORITHM} -tc=prg.10threads)
add_test(prg.11threads ${TF_UTEST_ALGORITHM} -tc=prg.11threads)
add_test(prg.12threads ${TF_UTEST_ALGORITHM} -tc=prg.12threads)
add_test(prd.1thread ${TF_UTEST_ALGORITHM} -tc=prd.1thread)
add_test(prd.2threads ${TF_UTEST_ALGORITHM} -tc=prd.2threads)
add_test(prd.3threads ${TF_UTEST_ALGORITHM} -tc=prd.3threads)
add_test(prd.4threads ${TF_UTEST_ALGORITHM} -tc=prd.4threads)
add_test(prd.5threads ${TF_UTEST_ALGORITHM} -tc=prd.5threads)
add_test(prd.6threads ${TF_UTEST_ALGORITHM} -tc=prd.6threads)
add_test(prd.7threads ${TF_UTEST_ALGORITHM} -tc=prd.7threads)
add_test(prd.8threads ${TF_UTEST_ALGORITHM} -tc=prd.8threads)
add_test(prd.9threads ${TF_UTEST_ALGORITHM} -tc=prd.9threads)
add_test(prd.10threads ${TF_UTEST_ALGORITHM} -tc=prd.10threads)
add_test(prd.11threads ${TF_UTEST_ALGORITHM} -tc=prd.11threads)
add_test(prd.12threads ${TF_UTEST_ALGORITHM} -tc=prd.12threads)
add_test(prs.1thread ${TF_UTEST_ALGORITHM} -tc=prs.1thread)
add_test(prs.2threads ${TF_UTEST_ALGORITHM} -tc=prs.2threads)
add_test(prs.3threads ${TF_UTEST_ALGORITHM} -tc=prs.3threads)
add_test(prs.4threads ${TF_UTEST_ALGORITHM} -tc=prs.4threads)
add_test(prs.5threads ${TF_UTEST_ALGORITHM} -tc=prs.5threads)
add_test(prs.6threads ${TF_UTEST_ALGORITHM} -tc=prs.6threads)
add_test(prs.7threads ${TF_UTEST_ALGORITHM} -tc=prs.7threads)
add_test(prs.8threads ${TF_UTEST_ALGORITHM} -tc=prs.8threads)
add_test(prs.9threads ${TF_UTEST_ALGORITHM} -tc=prs.9threads)
add_test(prs.10threads ${TF_UTEST_ALGORITHM} -tc=prs.10threads)
add_test(prs.11threads ${TF_UTEST_ALGORITHM} -tc=prs.11threads)
add_test(prs.12threads ${TF_UTEST_ALGORITHM} -tc=prd.12threads)
add_test(ptrg.1thread ${TF_UTEST_ALGORITHM} -tc=ptrg.1thread)
add_test(ptrg.2threads ${TF_UTEST_ALGORITHM} -tc=ptrg.2threads)
add_test(ptrg.3threads ${TF_UTEST_ALGORITHM} -tc=ptrg.3threads)
add_test(ptrg.4threads ${TF_UTEST_ALGORITHM} -tc=ptrg.4threads)
add_test(ptrg.5threads ${TF_UTEST_ALGORITHM} -tc=ptrg.5threads)
add_test(ptrg.6threads ${TF_UTEST_ALGORITHM} -tc=ptrg.6threads)
add_test(ptrg.7threads ${TF_UTEST_ALGORITHM} -tc=ptrg.7threads)
add_test(ptrg.8threads ${TF_UTEST_ALGORITHM} -tc=ptrg.8threads)
add_test(ptrg.9threads ${TF_UTEST_ALGORITHM} -tc=ptrg.9threads)
add_test(ptrg.10threads ${TF_UTEST_ALGORITHM} -tc=ptrg.10threads)
add_test(ptrg.11threads ${TF_UTEST_ALGORITHM} -tc=ptrg.11threads)
add_test(ptrg.12threads ${TF_UTEST_ALGORITHM} -tc=ptrg.12threads)
add_test(ptrd.1thread ${TF_UTEST_ALGORITHM} -tc=ptrd.1thread)
add_test(ptrd.2threads ${TF_UTEST_ALGORITHM} -tc=ptrd.2threads)
add_test(ptrd.3threads ${TF_UTEST_ALGORITHM} -tc=ptrd.3threads)
add_test(ptrd.4threads ${TF_UTEST_ALGORITHM} -tc=ptrd.4threads)
add_test(ptrd.5threads ${TF_UTEST_ALGORITHM} -tc=ptrd.5threads)
add_test(ptrd.6threads ${TF_UTEST_ALGORITHM} -tc=ptrd.6threads)
add_test(ptrd.7threads ${TF_UTEST_ALGORITHM} -tc=ptrd.7threads)
add_test(ptrd.8threads ${TF_UTEST_ALGORITHM} -tc=ptrd.8threads)
add_test(ptrd.9threads ${TF_UTEST_ALGORITHM} -tc=ptrd.9threads)
add_test(ptrd.10threads ${TF_UTEST_ALGORITHM} -tc=ptrd.10threads)
add_test(ptrd.11threads ${TF_UTEST_ALGORITHM} -tc=ptrd.11threads)
add_test(ptrd.12threads ${TF_UTEST_ALGORITHM} -tc=ptrd.12threads)
add_test(ptrs.1thread ${TF_UTEST_ALGORITHM} -tc=ptrs.1thread)
add_test(ptrs.2threads ${TF_UTEST_ALGORITHM} -tc=ptrs.2threads)
add_test(ptrs.3threads ${TF_UTEST_ALGORITHM} -tc=ptrs.3threads)
add_test(ptrs.4threads ${TF_UTEST_ALGORITHM} -tc=ptrs.4threads)
add_test(ptrs.5threads ${TF_UTEST_ALGORITHM} -tc=ptrs.5threads)
add_test(ptrs.6threads ${TF_UTEST_ALGORITHM} -tc=ptrs.6threads)
add_test(ptrs.7threads ${TF_UTEST_ALGORITHM} -tc=ptrs.7threads)
add_test(ptrs.8threads ${TF_UTEST_ALGORITHM} -tc=ptrs.8threads)
add_test(ptrs.9threads ${TF_UTEST_ALGORITHM} -tc=ptrs.9threads)
add_test(ptrs.10threads ${TF_UTEST_ALGORITHM} -tc=ptrs.10threads)
add_test(ptrs.11threads ${TF_UTEST_ALGORITHM} -tc=ptrs.11threads)
add_test(ptrs.12threads ${TF_UTEST_ALGORITHM} -tc=ptrd.12threads)

# unittest for traverse
add_executable(traverse ${TF_UTEST_DIR}/traverse.cpp)
target_link_libraries(traverse ${PROJECT_NAME} tf::default_settings)
target_include_directories(traverse PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_TRAVERSE ${TF_UTEST_DIR}/traverse)
add_test(static_traverse  ${TF_UTEST_TRAVERSE} -tc=StaticTraverse)
add_test(dynamic_traverse ${TF_UTEST_TRAVERSE} -tc=DynamicTraverse)
add_test(parallel_traverse.1thread ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.1)
add_test(parallel_traverse.2threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.2)
add_test(parallel_traverse.3threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.3)
add_test(parallel_traverse.4threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.4)
add_test(parallel_traverse.5threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.5)
add_test(parallel_traverse.6threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.6)
add_test(parallel_traverse.7threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.7)
add_test(parallel_traverse.8threads ${TF_UTEST_TRAVERSE} -tc=ParallelTraverse.8)

# unittest for sorting
add_executable(sort ${TF_UTEST_DIR}/sort.cpp)
target_link_libraries(sort ${PROJECT_NAME} tf::default_settings)
target_include_directories(sort PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_SORT ${TF_UTEST_DIR}/sort)
add_test(bubble_sort ${TF_UTEST_SORT} -tc=BubbleSort)
add_test(selection_sort ${TF_UTEST_SORT} -tc=SelectionSort)
add_test(merge_sort ${TF_UTEST_SORT} -tc=MergeSort)
add_test(quick_sort ${TF_UTEST_SORT} -tc=QuickSort)


#### CUDA unittests
if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_UTEST_DIR}/cuda)

add_executable(cuda_basics ${TF_UTEST_DIR}/cuda/cuda_basics.cu)
target_link_libraries(
  cuda_basics ${PROJECT_NAME} Threads::Threads tf::default_settings
)
target_include_directories(cuda_basics PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_CUDA_BASICS ${TF_UTEST_DIR}/cuda/cuda_basics)
add_test(cuda_basics.builder ${TF_UTEST_CUDA_BASICS} -tc=Builder)
add_test(cuda_basics.worker_id.1C1G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.1C1G)
add_test(cuda_basics.worker_id.1C2G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.1C2G)
add_test(cuda_basics.worker_id.1C3G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.1C3G)
add_test(cuda_basics.worker_id.1C4G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.1C4G)
add_test(cuda_basics.worker_id.2C1G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.2C1G)
add_test(cuda_basics.worker_id.2C2G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.2C2G)
add_test(cuda_basics.worker_id.2C3G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.2C3G)
add_test(cuda_basics.worker_id.2C4G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.2C4G)
add_test(cuda_basics.worker_id.3C1G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.3C1G)
add_test(cuda_basics.worker_id.3C2G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.3C2G)
add_test(cuda_basics.worker_id.3C3G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.3C3G)
add_test(cuda_basics.worker_id.3C4G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.3C4G)
add_test(cuda_basics.worker_id.4C1G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.4C1G)
add_test(cuda_basics.worker_id.4C2G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.4C2G)
add_test(cuda_basics.worker_id.4C3G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.4C3G)
add_test(cuda_basics.worker_id.4C4G ${TF_UTEST_CUDA_BASICS} -tc=WorkerID.4C4G)
add_test(cuda_basics.set.i8  ${TF_UTEST_CUDA_BASICS} -tc=Set.i8)
add_test(cuda_basics.set.i16 ${TF_UTEST_CUDA_BASICS} -tc=Set.i16)
add_test(cuda_basics.set.i32 ${TF_UTEST_CUDA_BASICS} -tc=Set.i32)
add_test(cuda_basics.bset.i8  ${TF_UTEST_CUDA_BASICS} -tc=BSet.i8)
add_test(cuda_basics.bset.i16 ${TF_UTEST_CUDA_BASICS} -tc=BSet.i16)
add_test(cuda_basics.bset.i32 ${TF_UTEST_CUDA_BASICS} -tc=BSet.i32)
add_test(cuda_basics.add.i8  ${TF_UTEST_CUDA_BASICS} -tc=Add.i8)
add_test(cuda_basics.add.i16 ${TF_UTEST_CUDA_BASICS} -tc=Add.i16)
add_test(cuda_basics.add.i32 ${TF_UTEST_CUDA_BASICS} -tc=Add.i32)
add_test(cuda_basics.barrier.i8 ${TF_UTEST_CUDA_BASICS} -tc=Barrier.i8)
add_test(cuda_basics.barrier.i16 ${TF_UTEST_CUDA_BASICS} -tc=Barrier.i16)
add_test(cuda_basics.barrier.i32 ${TF_UTEST_CUDA_BASICS} -tc=Barrier.i32)
add_test(cuda_basics.memset ${TF_UTEST_CUDA_BASICS} -tc=Memset)
add_test(cuda_basics.memset0.i8 ${TF_UTEST_CUDA_BASICS} -tc=Memset0.i8)
add_test(cuda_basics.memset0.i16 ${TF_UTEST_CUDA_BASICS} -tc=Memset0.i16)
add_test(cuda_basics.memset0.i32 ${TF_UTEST_CUDA_BASICS} -tc=Memset0.i32)
add_test(cuda_basics.memset0.f32 ${TF_UTEST_CUDA_BASICS} -tc=Memset0.f32)
add_test(cuda_basics.memset0.f64 ${TF_UTEST_CUDA_BASICS} -tc=Memset0.f64)
add_test(cuda_basics.zero.i8 ${TF_UTEST_CUDA_BASICS} -tc=Zero.i8)
add_test(cuda_basics.zero.i16 ${TF_UTEST_CUDA_BASICS} -tc=Zero.i16)
add_test(cuda_basics.zero.i32 ${TF_UTEST_CUDA_BASICS} -tc=Zero.i32)
add_test(cuda_basics.zero.f32 ${TF_UTEST_CUDA_BASICS} -tc=Zero.f32)
add_test(cuda_basics.fill.i8 ${TF_UTEST_CUDA_BASICS} -tc=Fill.i8)
add_test(cuda_basics.fill.i16 ${TF_UTEST_CUDA_BASICS} -tc=Fill.i16)
add_test(cuda_basics.fill.i32 ${TF_UTEST_CUDA_BASICS} -tc=Fill.i32)
add_test(cuda_basics.fill.f32 ${TF_UTEST_CUDA_BASICS} -tc=Fill.f32)
add_test(cuda_basics.memcpy.i8 ${TF_UTEST_CUDA_BASICS} -tc=Memcpy.i8)
add_test(cuda_basics.memcpy.i16 ${TF_UTEST_CUDA_BASICS} -tc=Memcpy.i16)
add_test(cuda_basics.memcpy.i32 ${TF_UTEST_CUDA_BASICS} -tc=Memcpy.i32)
add_test(cuda_basics.memcpy.f32 ${TF_UTEST_CUDA_BASICS} -tc=Memcpy.f32)
add_test(cuda_basics.memcpy.f64 ${TF_UTEST_CUDA_BASICS} -tc=Memcpy.f64)
add_test(cuda_basics.empty ${TF_UTEST_CUDA_BASICS} -tc=Empty)
add_test(cuda_basics.nested_runs ${TF_UTEST_CUDA_BASICS} -tc=NestedRuns)
add_test(cuda_basics.multiruns.1C1G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.1C1G)
add_test(cuda_basics.multiruns.1C2G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.1C2G)
add_test(cuda_basics.multiruns.1C3G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.1C3G)
add_test(cuda_basics.multiruns.1C4G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.1C4G)
add_test(cuda_basics.multiruns.2C1G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.2C1G)
add_test(cuda_basics.multiruns.2C2G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.2C2G)
add_test(cuda_basics.multiruns.2C3G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.2C3G)
add_test(cuda_basics.multiruns.2C4G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.2C4G)
add_test(cuda_basics.multiruns.3C1G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.3C1G)
add_test(cuda_basics.multiruns.3C2G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.3C2G)
add_test(cuda_basics.multiruns.3C3G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.3C3G)
add_test(cuda_basics.multiruns.3C4G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.3C4G)
add_test(cuda_basics.multiruns.4C1G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.4C1G)
add_test(cuda_basics.multiruns.4C2G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.4C2G)
add_test(cuda_basics.multiruns.4C3G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.4C3G)
add_test(cuda_basics.multiruns.4C4G ${TF_UTEST_CUDA_BASICS} -tc=Multiruns.4C4G)
add_test(cuda_basics.subflow ${TF_UTEST_CUDA_BASICS} -tc=Subflow)
add_test(cuda_basics.nested_subflow ${TF_UTEST_CUDA_BASICS} -tc=NestedSubflow)
add_test(cuda_basics.detached_subflow ${TF_UTEST_CUDA_BASICS} -tc=DetachedSubflow)
add_test(cuda_basics.loop ${TF_UTEST_CUDA_BASICS} -tc=Loop)
add_test(cuda_basics.predicate ${TF_UTEST_CUDA_BASICS} -tc=Predicate)
add_test(cuda_basics.repeat ${TF_UTEST_CUDA_BASICS} -tc=Repeat)

# matrix operation tests
add_executable(cuda_matrix ${TF_UTEST_DIR}/cuda/cuda_matrix.cu)
target_link_libraries(
  cuda_matrix ${PROJECT_NAME} Threads::Threads tf::default_settings
)
target_include_directories(cuda_matrix PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_CUDA_MATRIX ${TF_UTEST_DIR}/cuda/cuda_matrix)
add_test(cuda_matrix.multiply ${TF_UTEST_CUDA_MATRIX} -tc=multiply)
add_test(cuda_matrix.transpose ${TF_UTEST_CUDA_MATRIX} -tc=transpose)
add_test(cuda_matrix.product ${TF_UTEST_CUDA_MATRIX} -tc=product)

# k-means test
add_executable(cuda_kmeans ${TF_UTEST_DIR}/cuda/cuda_kmeans.cu)
target_link_libraries(
  cuda_kmeans ${PROJECT_NAME} Threads::Threads tf::default_settings
)
target_include_directories(cuda_kmeans PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
set(TF_UTEST_CUDA_KMEANS  ${TF_UTEST_DIR}/cuda/cuda_kmeans)
add_test(cuda_kmeans.10.1C1G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.1C1G)
add_test(cuda_kmeans.10.1C2G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.1C2G)
add_test(cuda_kmeans.10.1C3G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.1C3G)
add_test(cuda_kmeans.10.1C4G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.1C4G)
add_test(cuda_kmeans.10.2C1G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.2C1G)
add_test(cuda_kmeans.10.2C2G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.2C2G)
add_test(cuda_kmeans.10.2C3G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.2C3G)
add_test(cuda_kmeans.10.2C4G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.10.2C4G)
add_test(cuda_kmeans.100.1C1G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.100.1C1G)
add_test(cuda_kmeans.100.2C2G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.100.2C2G)
add_test(cuda_kmeans.100.3C3G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.100.3C3G)
add_test(cuda_kmeans.100.4C4G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.100.4C4G)
add_test(cuda_kmeans.1000.1C1G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.1000.1C1G)
add_test(cuda_kmeans.1000.2C2G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.1000.2C2G)
add_test(cuda_kmeans.1000.4C4G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.1000.4C4G)
add_test(cuda_kmeans.1000.8C8G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.1000.8C8G)
add_test(cuda_kmeans.1000.16C16G ${TF_UTEST_CUDA_KMEANS} -tc=kmeans.1000.16C16G)

endif(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

endif(TF_BUILD_TESTS)

# -----------------------------------------------------------------------------
# Benchmarking (enabled by TF_BUILD_BENCHMARKS)
# -----------------------------------------------------------------------------

if(${TF_BUILD_BENCHMARKS})

# eigen package 
if(NOT DEFINED EIGEN_ROOT)
  set(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/3rd-party/eigen-3.3.7)
endif()
include_directories(${EIGEN_ROOT})

# find OpenMP package
include(FindOpenMP)

if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP not found")
endif()
  
message(STATUS "OpenMP_VERSION: ${OpenMP_VERSION}")
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")

# tbb package
if(NOT DEFINED TBB_ROOT)
  set(TBB_ROOT ${PROJECT_SOURCE_DIR}/3rd-party/tbb)
endif()

message(STATUS "TBB_ROOT: " ${TBB_ROOT})

include(${TBB_ROOT}/cmake/TBBBuild.cmake)
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
find_package(TBB REQUIRED tbb_preview)


## benchmark 1: wavefront computing
message(STATUS "benchmark 1: wavefront")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/wavefront)
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
message(STATUS "benchmark 2: graph traversal")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/graph_traversal)
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
message(STATUS "benchmark 3: binary tree")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/binary_tree)
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
message(STATUS "benchmark 4: linear_chain")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/linear_chain)
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
message(STATUS "benchmark 5: mnist")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/mnist)
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
message(STATUS "benchmark 6: matrix multiplication")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/matrix_multiplication)
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
message(STATUS "benchmark 7: mandelbrot")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/mandelbrot)
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
message(STATUS "benchmark 8: black scholes")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/black_scholes)
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
message(STATUS "benchmark 9: reduce_sum")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/reduce_sum)
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


if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
## cuda benchmark 1: heterogeneous traversal
message(STATUS "cuda benchmark 1: heterogeneous traversal")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/hetero_traversal)
add_executable(
  hetero_traversal 
  ${TF_BENCHMARK_DIR}/hetero_traversal/main.cu
  ${TF_BENCHMARK_DIR}/hetero_traversal/taskflow.cu
  ${TF_BENCHMARK_DIR}/hetero_traversal/tbb.cu
  ${TF_BENCHMARK_DIR}/hetero_traversal/omp.cu
)
target_include_directories(hetero_traversal PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  hetero_traversal 
  ${PROJECT_NAME} 
  Threads::Threads 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(hetero_traversal PROPERTIES COMPILE_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS}")
endif(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)

endif()

# -----------------------------------------------------------------------------
# create find_package(Taskflow CONFIG)
# -----------------------------------------------------------------------------

# install header
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/taskflow DESTINATION ${TF_INC_INSTALL_DIR})

# export target
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME ${PROJECT_NAME})

export(
  TARGETS ${PROJECT_NAME} 
  NAMESPACE ${PROJECT_NAME}:: 
  FILE ${PROJECT_NAME}Targets.cmake
)
export(PACKAGE ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
install(
  EXPORT ${PROJECT_NAME}Targets 
  NAMESPACE ${PROJECT_NAME}:: 
  DESTINATION ${TF_LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
)

# set up config
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${TF_LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
  PATH_VARS TF_INC_INSTALL_DIR
)

write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake 
  COMPATIBILITY SameMajorVersion
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${TF_LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
)

