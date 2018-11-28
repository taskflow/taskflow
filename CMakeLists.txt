# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-Taskflow VERSION 2.0.0)

# Enable memory test
include(CTest)

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
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
    message(FATAL_ERROR "\nCpp-Taskflow requires clang++ at least v5.0")
  endif() 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
## microsoft visual c++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(NOT MSVC_VERSION GREATER_EQUAL 1914)
    message(FATAL_ERROR "\nCpp-Taskflow requires MSVC++ at least v14.14") 
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
else()
  message(FATAL_ERROR "\n\
Cpp-Taskflow currently supports the following compilers:\n\
- g++ v7.3 or above\n\
- clang++ v5.0 or above\n\
- MSVC++ v19.14 or above\n\
")
endif()

# CXX target properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})
include_directories(doctest)

# Find pthread package
find_package(Threads REQUIRED)

## Find OpenMP package
#include(FindOpenMP)
#
#if(OpenMP_CXX_FOUND)
#  message(STATUS "OpenMP_CXX_SPEC_DATE: ${OpenMP_CXX_SPEC_DATE}")
#  message(STATUS "OpenMP_CXX_VERSION: ${OpenMP_CXX_VERSION}")
#  message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
#  message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
#else(OpenMP_CXX_FOUND)
#  message(WARNING "OpenMP CXX not found")
#endif(OpenMP_CXX_FOUND)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
OPTION(TF_BUILD_EXAMPLES "Enables build of examples" ON)

if(${TF_BUILD_EXAMPLES})

message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(simple example/simple.cpp)
target_link_libraries(simple Threads::Threads)

add_executable(subflow example/subflow.cpp)
target_link_libraries(subflow Threads::Threads)

add_executable(debug example/debug.cpp)
target_link_libraries(debug Threads::Threads)

add_executable(emplace example/emplace.cpp)
target_link_libraries(emplace Threads::Threads)

add_executable(reduce example/reduce.cpp)
target_link_libraries(reduce Threads::Threads)

add_executable(threadpool example/threadpool.cpp)
target_link_libraries(threadpool Threads::Threads)

add_executable(taskflow example/taskflow.cpp)
target_link_libraries(taskflow Threads::Threads)

add_executable(matrix example/matrix.cpp)
target_link_libraries(matrix Threads::Threads)

add_executable(parallel_for example/parallel_for.cpp)
target_link_libraries(parallel_for Threads::Threads)

add_executable(threadpool_cxx14 example/threadpool_cxx14.cpp)
set_property(TARGET threadpool_cxx14 PROPERTY CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14 Threads::Threads)

add_executable(multiple_dispatch example/multiple_dispatch.cpp)
target_link_libraries(multiple_dispatch Threads::Threads)

add_executable(dispatch example/dispatch.cpp)
target_link_libraries(dispatch Threads::Threads)

add_executable(executor example/executor.cpp)
target_link_libraries(executor Threads::Threads)

endif()

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
OPTION(TF_BUILD_TESTS "Enables build of tests" ON)

if(${TF_BUILD_TESTS})
 
enable_testing()
message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unittest)
set(TF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittest)

# unittest for taskflow
add_executable(taskflow_test_tmp unittest/taskflow.cpp)
target_link_libraries(taskflow_test_tmp Threads::Threads)
set_target_properties(taskflow_test_tmp PROPERTIES OUTPUT_NAME "taskflow")
add_test(builder          ${TF_UTEST_DIR}/taskflow -tc=Builder)
add_test(dispatch         ${TF_UTEST_DIR}/taskflow -tc=Dispatch)
add_test(executor         ${TF_UTEST_DIR}/taskflow -tc=Executor)
add_test(parallel_for     ${TF_UTEST_DIR}/taskflow -tc=ParallelFor)
add_test(reduce           ${TF_UTEST_DIR}/taskflow -tc=Reduce)
add_test(reduce_min       ${TF_UTEST_DIR}/taskflow -tc=ReduceMin)
add_test(reduce_max       ${TF_UTEST_DIR}/taskflow -tc=ReduceMax)
add_test(joined_subflow   ${TF_UTEST_DIR}/taskflow -tc=JoinedSubflow)
add_test(detached_subflow ${TF_UTEST_DIR}/taskflow -tc=DetachedSubflow)

# unittest for threadpool 
add_executable(threadpool_test_tmp unittest/threadpool.cpp)
target_link_libraries(threadpool_test_tmp Threads::Threads)
set_target_properties(threadpool_test_tmp PROPERTIES OUTPUT_NAME "threadpool")
add_test(WorkerQueue.OneThread  ${TF_UTEST_DIR}/threadpool -tc=WorkerQueue.OneThread)
add_test(WorkerQueue.TwoThread  ${TF_UTEST_DIR}/threadpool -tc=WorkerQueue.TwoThread)
add_test(WorkerQueue.TriThread  ${TF_UTEST_DIR}/threadpool -tc=WorkerQueue.TriThread)
add_test(simple_threadpool      ${TF_UTEST_DIR}/threadpool -tc=SimpleThreadpool)
add_test(proactive_threadpool   ${TF_UTEST_DIR}/threadpool -tc=ProactiveThreadpool)
add_test(speculative_threadpool ${TF_UTEST_DIR}/threadpool -tc=SpeculativeThreadpool)
add_test(privatized_threadpool  ${TF_UTEST_DIR}/threadpool -tc=PrivatizedThreadpool)

## threadpool_cxx14 unittest (contributed by Glen Fraser)
add_executable(threadpool_cxx14_tmp unittest/threadpool_cxx14.cpp)
set_target_properties(threadpool_cxx14_tmp PROPERTIES CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14_tmp Threads::Threads)
set_target_properties(threadpool_cxx14_tmp PROPERTIES OUTPUT_NAME "threadpool_cxx14")
add_test(threadpool_cxx14_basic        ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.Basic)
add_test(threadpool_cxx14_wait_for_all ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.WaitForAll)

endif()

# -----------------------------------------------------------------------------
# Benchmarking (enabled by TF_BUILD_BENCHMARKS)
# -----------------------------------------------------------------------------

#OPTION(TF_BUILD_BENCHMARKS "Enables build of benchmarks" OFF)
#if(${TF_BUILD_BENCHMARKS})
#
## Find OpenMP package
#include(FindOpenMP)
#
#if(NOT OpenMP_CXX_FOUND)
#  message(FATAL_ERROR "")
#  message(STATUS "OpenMP_CXX_SPEC_DATE: ${OpenMP_CXX_SPEC_DATE}")
#  message(STATUS "OpenMP_CXX_VERSION: ${OpenMP_CXX_VERSION}")
#  message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
#  message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
#else(OpenMP_CXX_FOUND)
#  message(WARNING "OpenMP CXX not found")
#endif(OpenMP_CXX_FOUND)
#
#endif()
#endif()

# -----------------------------------------------------------------------------
# create find_package(Cpp-Taskflow CONFIG)
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)

set(TF_INC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(TF_LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")

message(STATUS "TF_INC_INSTALL_DIR: " ${TF_INC_INSTALL_DIR})
message(STATUS "TF_LIB_INSTALL_DIR: " ${TF_LIB_INSTALL_DIR})

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
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${TF_LIB_INSTALL_DIR}/cmake)

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
