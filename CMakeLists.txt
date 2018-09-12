# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-TaskFlow)

# Enable memory test
include(CTest)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Compiler vendors
## g++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.2")
    message(FATAL_ERROR "\nCpp-Taskflow requires g++ at least v7.2")
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
- g++ v7.2 or above\n\
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
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})
include_directories(doctest)
include_directories(taskflow)

# Find pthread package
find_package(Threads REQUIRED)

# Find OpenMP package
include(FindOpenMP)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP_CXX_SPEC_DATE: ${OpenMP_CXX_SPEC_DATE}")
  message(STATUS "OpenMP_CXX_VERSION: ${OpenMP_CXX_VERSION}")
  message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP_CXX_LIB_NAMES: ${OpenMP_CXX_LIB_NAMES}")
else(OpenMP_CXX_FOUND)
  message(WARNING "OpenMP CXX not found")
endif(OpenMP_CXX_FOUND)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

set(EXAMPLE_CXX_FLAGS ${OpenMP_CXX_FLAGS})

message(STATUS "EXAMPLE_CXX_FLAGS: " ${EXAMPLE_CXX_FLAGS})
message(STATUS "EXAMPLE_EXE_LINKER_FLAGS: " ${EXAMPLE_EXE_LINKER_FLAGS})

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

add_executable(matrix example/matrix.cpp)
target_compile_options(matrix PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(matrix Threads::Threads OpenMP::OpenMP_CXX)

add_executable(parallel_for example/parallel_for.cpp)
target_compile_options(parallel_for PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(parallel_for Threads::Threads OpenMP::OpenMP_CXX)

add_executable(threadpool_cxx14 example/threadpool_cxx14.cpp)
set_property(TARGET threadpool_cxx14 PROPERTY CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14 Threads::Threads)

add_executable(multiple_dispatch example/multiple_dispatch.cpp)
target_link_libraries(multiple_dispatch Threads::Threads)

add_executable(subsum example/subsum.cpp)
target_link_libraries(subsum Threads::Threads)

add_executable(taskflow_subsum example/taskflow_subsum.cpp)
target_link_libraries(taskflow_subsum Threads::Threads)


# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
enable_testing()

message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unittest)
set(TF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittest)
add_executable(taskflow unittest/taskflow.cpp)
target_link_libraries(taskflow Threads::Threads)

# unittest for taskflow
add_test(builder          ${TF_UTEST_DIR}/taskflow -tc=Taskflow.Builder)
add_test(dispatch         ${TF_UTEST_DIR}/taskflow -tc=Taskflow.Dispatch)
add_test(parallel_for     ${TF_UTEST_DIR}/taskflow -tc=Taskflow.ParallelFor)
add_test(reduce           ${TF_UTEST_DIR}/taskflow -tc=Taskflow.Reduce)
add_test(reduce_min       ${TF_UTEST_DIR}/taskflow -tc=Taskflow.ReduceMin)
add_test(reduce_max       ${TF_UTEST_DIR}/taskflow -tc=Taskflow.ReduceMax)
add_test(joined_subflow   ${TF_UTEST_DIR}/taskflow -tc=Taskflow.JoinedSubflow)
add_test(detached_subflow ${TF_UTEST_DIR}/taskflow -tc=Taskflow.DetachedSubflow)

# unittest for threadpool 
add_executable(threadpool_test_tmp unittest/threadpool.cpp)
target_link_libraries(threadpool_test_tmp Threads::Threads)
set_target_properties(threadpool_test_tmp PROPERTIES OUTPUT_NAME "threadpool")
add_test(simple_threadpool    ${TF_UTEST_DIR}/threadpool -tc=SimpleThreadpool)
add_test(proactive_threadpool ${TF_UTEST_DIR}/threadpool -tc=ProactiveThreadpool)
add_test(speculative_threadpool ${TF_UTEST_DIR}/threadpool -tc=SpeculativeThreadpool)

## threadpool_cxx14 unittest (contributed by Glen Fraser)
add_executable(threadpool_cxx14_tmp unittest/threadpool_cxx14.cpp)
set_target_properties(threadpool_cxx14_tmp PROPERTIES CXX_STANDARD 14)
target_link_libraries(threadpool_cxx14_tmp Threads::Threads)
set_target_properties(threadpool_cxx14_tmp PROPERTIES OUTPUT_NAME "threadpool_cxx14")
add_test(threadpool_cxx14_basic        ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.Basic)
add_test(threadpool_cxx14_wait_for_all ${TF_UTEST_DIR}/threadpool_cxx14 -tc=Threadpool.WaitForAll)

