# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-Taskflow VERSION 2.1.0 LANGUAGES CXX)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Compiler vendors
## g++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.3")
    message(FATAL_ERROR "\nCpp-Taskflow requires g++ at least v7.3")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "\nCpp-Taskflow requires clang++ at least v6.0")
  endif() 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
## microsoft visual c++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(NOT MSVC_VERSION GREATER_EQUAL 1914)
    message(FATAL_ERROR "\nCpp-Taskflow requires MSVC++ at least v14.14") 
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
else()
  message(FATAL_ERROR "\n\
Cpp-Taskflow currently supports the following compilers:\n\
- g++ v7.3 or above\n\
- clang++ v6.0 or above\n\
- MSVC++ v19.14 or above\n\
")
endif()

# CXX target properties
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

# build options
option(TF_BUILD_EXAMPLES "Enables build of examples" ON)
option(TF_BUILD_TESTS "Enables build of tests" ON)
option(TF_BUILD_BENCHMARKS "Enables build of benchmarks" OFF)

# installation path
set(TF_INC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(TF_LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittest)
set(TF_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/example)
set(TF_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmark)

message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
message(STATUS "TF_BUILD_EXAMPLES: " ${TF_BUILD_EXAMPLES})
message(STATUS "TF_BUILD_TESTS: " ${TF_BUILD_TESTS})
message(STATUS "TF_BUILD_BENCHMARKS: " ${TF_BUILD_BENCHMARKS})
message(STATUS "TF_INC_INSTALL_DIR: " ${TF_INC_INSTALL_DIR})
message(STATUS "TF_LIB_INSTALL_DIR: " ${TF_LIB_INSTALL_DIR})
message(STATUS "TF_UTEST_DIR: " ${TF_UTEST_DIR})
message(STATUS "TF_EXAMPLE_DIR: " ${TF_EXAMPLE_DIR})
message(STATUS "TF_BENCHMARK_DIR: " ${TF_BENCHMARK_DIR})

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
# Cpp-Taskflow library interface
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------

if(${TF_BUILD_EXAMPLES})

message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_EXAMPLE_DIR})

add_executable(simple ${TF_EXAMPLE_DIR}/simple.cpp)
target_link_libraries(simple ${PROJECT_NAME} Threads::Threads)

add_executable(subflow ${TF_EXAMPLE_DIR}/subflow.cpp)
target_link_libraries(subflow ${PROJECT_NAME} Threads::Threads)

add_executable(nested_subflow ${TF_EXAMPLE_DIR}/nested_subflow.cpp)
target_link_libraries(nested_subflow ${PROJECT_NAME} Threads::Threads)

add_executable(debug ${TF_EXAMPLE_DIR}/debug.cpp)
target_link_libraries(debug ${PROJECT_NAME} Threads::Threads)

add_executable(emplace ${TF_EXAMPLE_DIR}/emplace.cpp)
target_link_libraries(emplace ${PROJECT_NAME} Threads::Threads)

add_executable(reduce ${TF_EXAMPLE_DIR}/reduce.cpp)
target_link_libraries(reduce ${PROJECT_NAME} Threads::Threads)

add_executable(threadpool ${TF_EXAMPLE_DIR}/threadpool.cpp)
target_link_libraries(threadpool ${PROJECT_NAME} Threads::Threads)

add_executable(taskflow ${TF_EXAMPLE_DIR}/taskflow.cpp)
target_link_libraries(taskflow ${PROJECT_NAME} Threads::Threads)

add_executable(matrix ${TF_EXAMPLE_DIR}/matrix.cpp)
target_link_libraries(matrix ${PROJECT_NAME} Threads::Threads)

add_executable(parallel_for ${TF_EXAMPLE_DIR}/parallel_for.cpp)
target_link_libraries(parallel_for ${PROJECT_NAME} Threads::Threads)

add_executable(threadpool_cxx14 ${TF_EXAMPLE_DIR}/threadpool_cxx14.cpp)
set_property(TARGET threadpool_cxx14 PROPERTY CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14 Threads::Threads)

add_executable(multiple_dispatch ${TF_EXAMPLE_DIR}/multiple_dispatch.cpp)
target_link_libraries(multiple_dispatch ${PROJECT_NAME} Threads::Threads)

add_executable(dispatch ${TF_EXAMPLE_DIR}/dispatch.cpp)
target_link_libraries(dispatch ${PROJECT_NAME} Threads::Threads)

add_executable(executor ${TF_EXAMPLE_DIR}/executor.cpp)
target_link_libraries(executor ${PROJECT_NAME} Threads::Threads) 

add_executable(dynamic_traversal ${TF_EXAMPLE_DIR}/dynamic_traversal.cpp)
target_link_libraries(dynamic_traversal ${PROJECT_NAME} Threads::Threads) 

add_executable(framework ${TF_EXAMPLE_DIR}/framework.cpp)
target_link_libraries(framework ${PROJECT_NAME} Threads::Threads) 

add_executable(framework_dynamic_tasking ${TF_EXAMPLE_DIR}/framework_dynamic_tasking.cpp)
target_link_libraries(framework_dynamic_tasking ${PROJECT_NAME} Threads::Threads)

endif()

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------


if(${TF_BUILD_TESTS})
 
enable_testing()
message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_UTEST_DIR})

# unittest for utility
add_executable(utility unittest/utility.cpp)
target_link_libraries(utility ${PROJECT_NAME} Threads::Threads)
target_include_directories(utility PRIVATE ${PROJECT_SOURCE_DIR}/doctest)
add_test(passive_vector   ${TF_UTEST_DIR}/utility -tc=passive_vector)

# unittest for taskflow
add_executable(taskflow_test_tmp unittest/taskflow.cpp)
target_link_libraries(taskflow_test_tmp ${PROJECT_NAME} Threads::Threads)
target_include_directories(taskflow_test_tmp PRIVATE ${PROJECT_SOURCE_DIR}/doctest)
set_target_properties(taskflow_test_tmp PROPERTIES OUTPUT_NAME "taskflow")
add_test(builder          ${TF_UTEST_DIR}/taskflow -tc=Builder)
add_test(dispatch         ${TF_UTEST_DIR}/taskflow -tc=Dispatch)
add_test(executor         ${TF_UTEST_DIR}/taskflow -tc=Executor)
add_test(parallel_for     ${TF_UTEST_DIR}/taskflow -tc=ParallelFor)
add_test(parallel_for_idx ${TF_UTEST_DIR}/taskflow -tc=ParallelForOnIndex)
add_test(reduce           ${TF_UTEST_DIR}/taskflow -tc=Reduce)
add_test(reduce_min       ${TF_UTEST_DIR}/taskflow -tc=ReduceMin)
add_test(reduce_max       ${TF_UTEST_DIR}/taskflow -tc=ReduceMax)
add_test(joined_subflow   ${TF_UTEST_DIR}/taskflow -tc=JoinedSubflow)
add_test(detached_subflow ${TF_UTEST_DIR}/taskflow -tc=DetachedSubflow)
add_test(framework        ${TF_UTEST_DIR}/taskflow -tc=Framework)

# unittest for threadpool 
add_executable(threadpool_test_tmp unittest/threadpool.cpp)
target_link_libraries(threadpool_test_tmp ${PROJECT_NAME} Threads::Threads)
target_include_directories(threadpool_test_tmp PRIVATE ${PROJECT_SOURCE_DIR}/doctest)
set_target_properties(threadpool_test_tmp PROPERTIES OUTPUT_NAME "threadpool")
add_test(WorkStealingQueue.Owner  ${TF_UTEST_DIR}/threadpool -tc=WSQ.Owner)
add_test(WorkStealingQueue.1Thief ${TF_UTEST_DIR}/threadpool -tc=WSQ.1Thief)
add_test(WorkStealingQueue.2Thieves ${TF_UTEST_DIR}/threadpool -tc=WSQ.2Thieves)
add_test(WorkStealingQueue.3Thieves ${TF_UTEST_DIR}/threadpool -tc=WSQ.3Thieves)
add_test(WorkStealingQueue.4Thieves ${TF_UTEST_DIR}/threadpool -tc=WSQ.4Thieves)
add_test(simple_threadpool      ${TF_UTEST_DIR}/threadpool -tc=SimpleThreadpool)
add_test(proactive_threadpool   ${TF_UTEST_DIR}/threadpool -tc=ProactiveThreadpool)
add_test(speculative_threadpool ${TF_UTEST_DIR}/threadpool -tc=SpeculativeThreadpool)
add_test(work_stealing_threadpool  ${TF_UTEST_DIR}/threadpool -tc=WorkStealingThreadpool)

# threadpool_cxx14 unittest (contributed by Glen Fraser)
add_executable(threadpool_cxx14_tmp unittest/threadpool_cxx14.cpp)
set_target_properties(threadpool_cxx14_tmp PROPERTIES CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14_tmp Threads::Threads)
target_include_directories(threadpool_cxx14_tmp PRIVATE ${PROJECT_SOURCE_DIR}/doctest)
set_target_properties(threadpool_cxx14_tmp PROPERTIES OUTPUT_NAME "threadpool_cxx14")
add_test(threadpool_cxx14_basic        ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.Basic)
add_test(threadpool_cxx14_wait_for_all ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.WaitForAll)

# run all examples
#add_test(inttest.simple )

endif()

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
target_link_libraries(
  wavefront 
  ${PROJECT_NAME} Threads::Threads ${TBB_IMPORTED_TARGETS} ${OpenMP_CXX_LIBRARIES}
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
target_link_libraries(
  graph_traversal 
  ${PROJECT_NAME} Threads::Threads ${TBB_IMPORTED_TARGETS} ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(graph_traversal PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})


## benchmark 3: MNIST
message(STATUS "benchmark 3: mnist")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/mnist)
add_executable(
  mnist 
  ${TF_BENCHMARK_DIR}/mnist/main.cpp
  ${TF_BENCHMARK_DIR}/mnist/omp.cpp
  ${TF_BENCHMARK_DIR}/mnist/tbb.cpp
  ${TF_BENCHMARK_DIR}/mnist/seq.cpp
  ${TF_BENCHMARK_DIR}/mnist/taskflow.cpp
)
target_link_libraries(
  mnist 
  ${PROJECT_NAME} Threads::Threads ${TBB_IMPORTED_TARGETS}
  ${OpenMP_CXX_LIBRARIES} stdc++fs
)
set_target_properties(mnist PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

# Add benchmark command into Makefile 
add_custom_target(benchmark 
  COMMAND wavefront 
  COMMAND graph_traversal 
  COMMAND mnist)
add_dependencies(benchmark wavefront graph_traversal mnist)

## benchmark 4: Framework
message(STATUS "benchmark 4: Framework")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/framework)
add_executable(
  framework_benchmarking
  ${TF_BENCHMARK_DIR}/framework/main.cpp
  ${TF_BENCHMARK_DIR}/framework/omp.cpp
  ${TF_BENCHMARK_DIR}/framework/tbb.cpp
  ${TF_BENCHMARK_DIR}/framework/seq.cpp
  ${TF_BENCHMARK_DIR}/framework/taskflow.cpp
)
target_link_libraries(
  framework_benchmarking
  ${PROJECT_NAME} Threads::Threads ${TBB_IMPORTED_TARGETS} ${OpenMP_CXX_LIBRARIES}
)
set_target_properties(framework_benchmarking PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})



endif()

# -----------------------------------------------------------------------------
# create find_package(Cpp-Taskflow CONFIG)
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
  DESTINATION ${TF_LIB_INSTALL_DIR}/cmake
)

# set up config
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${TF_LIB_INSTALL_DIR}/cmake
  PATH_VARS TF_INC_INSTALL_DIR
)

write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake 
  COMPATIBILITY SameMajorVersion
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${TF_LIB_INSTALL_DIR}/cmake 
)

