# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-TaskFlow VERSION 0.1.0)

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

message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})
include_directories(doctest)
include_directories(taskflow)

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
OPTION(TASKFLOW_BUILD_EXAMPLES "Enables build of examples" OFF)
if(${TASKFLOW_BUILD_EXAMPLES})
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
OPTION(TASKFLOW_TEST "Enables Tests" OFF)
if(${TASKFLOW_TEST})
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
# Benchmarking (enabled by TF_BENCHMARKING)
# -----------------------------------------------------------------------------

#if(TF_BENCHMARK EQUAL "1")
#message(STATUS "TF_BENCHMARK: ${TF_BENCHMARK}")
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

# -----------------------------------------------------------------------------
# Create importable library
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/taskflow>
    $<INSTALL_INTERFACE:taskflow/>  # <prefix>/include/mylib
)

SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "The directory the headers are installed in")
SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Directory where lib will install")

#Export as Package
set_target_properties (${PROJECT_NAME} PROPERTIES EXPORT_NAME ${PROJECT_NAME})
export(TARGETS ${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}:: FILE ${PROJECT_NAME}Targets.cmake)
export(PACKAGE ${PROJECT_NAME})

include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake
  PATH_VARS INCLUDE_INSTALL_DIR)
write_basic_package_version_file(${PROJECT_NAME}ConfigVersion.cmake COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/ DESTINATION ${INCLUDE_INSTALL_DIR}/taskflow)

install (TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
install (EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${LIB_INSTALL_DIR}/${PROJECT_NAME}/cmake)
