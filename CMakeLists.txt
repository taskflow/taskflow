cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-TaskFlow)

# Enable memory test
include(CTest)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# CXX target properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#set(USE_WERROR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})

# The version number
set (TASKFLOW_MAJOR_VERSION "2")
set (TASKFLOW_MINOR_VERSION "0")

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})
include_directories(doctest)

# Find pthread package
find_package(Threads REQUIRED)

# Find OpenMP package
include(FindOpenMP)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP_CXX_SPEC_DATE: ${OpenMP_CXX_SPEC_DATE}")
  message(STATUS "OpenMP_CXX_VERSION: ${OpenMP_CXX_VERSION}")
else(OpenMP_CXX_FOUND)
  message(WARNING "OpenMP CXX not found")
endif(OpenMP_CXX_FOUND)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

set(EXAMPLE_CXX_FLAGS ${OpenMP_CXX_FLAGS})

list(APPEND EXAMPLE_EXE_LINKER_FLAGS "")
list(APPEND EXAMPLE_EXE_LINKER_FLAGS Threads::Threads)
list(APPEND EXAMPLE_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)

message(STATUS "EXAMPLE_CXX_FLAGS: " ${EXAMPLE_CXX_FLAGS})
message(STATUS "EXAMPLE_EXE_LINKER_FLAGS: " ${EXAMPLE_EXE_LINKER_FLAGS})



add_executable(simple example/simple.cpp)
target_link_libraries(simple ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(subflow example/subflow.cpp)
target_link_libraries(subflow ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(debug example/debug.cpp)
target_link_libraries(debug ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(emplace example/emplace.cpp)
target_link_libraries(emplace ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(reduce example/reduce.cpp)
target_link_libraries(reduce ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(matrix example/matrix.cpp)
target_compile_options(matrix PRIVATE ${EXAMPLE_CXX_FLAGS})
target_link_libraries(matrix ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(parallel_for example/parallel_for.cpp)
target_compile_options(parallel_for PRIVATE ${EXAMPLE_CXX_FLAGS})
target_link_libraries(parallel_for ${EXAMPLE_EXE_LINKER_FLAGS})


# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
enable_testing()

message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/unittest)
add_executable(taskflow unittest/taskflow.cpp)
target_link_libraries(taskflow Threads::Threads)

add_test(builder ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.Builder)
add_test(dispatch ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.Dispatch)
add_test(parallel_for ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.ParallelFor)
add_test(reduce ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.Reduce)
add_test(reduce_min ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.ReduceMin)
add_test(reduce_max ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.ReduceMax)
add_test(subflow ${PROJECT_SOURCE_DIR}/unittest/taskflow -tc=Taskflow.Subflow)




