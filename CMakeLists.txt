cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-TaskFlow)

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
set (TASKFLOW_MAJOR_VERSION "1")
set (TASKFLOW_MINOR_VERSION "0")

# add the binayr tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})
include_directories(doctest)

# Find pthread package
find_package(Threads REQUIRED)

# Find OpenMP package
include(FindOpenMP)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
message(STATUS "Building examples ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

list(APPEND EXAMPLE_EXE_LINKER_FLAGS "")
list(APPEND EXAMPLE_EXE_LINKER_FLAGS Threads::Threads)

if(OpenMP_CXX_FOUND)
  set(EXAMPLE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  list(APPEND EXAMPLE_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
else(OpenMP_CXX_FOUND)
  message(WARNING "OpenMP CXX not found")
endif(OpenMP_CXX_FOUND)

message(STATUS "EXAMPLE_CXX_FLAGS: " ${EXAMPLE_CXX_FLAGS})
message(STATUS "EXAMPLE_EXE_LINKER_FLAGS: " ${EXAMPLE_EXE_LINKER_FLAGS})

add_executable(simple example/simple.cpp)
target_link_libraries(simple ${EXAMPLE_EXE_LINKER_FLAGS})

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




