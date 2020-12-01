cmake_minimum_required(VERSION 3.4.3)

if(CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message(FATAL_ERROR "host compiler - gcc version must be > 4.8")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
    message(FATAL_ERROR "host compiler - clang version must be > 3.6")
  endif()
endif()

if(MSVC)
  set(ComputeCpp_STL_CHECK_SRC __STL_check)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
    "#include <CL/sycl.hpp>  \n"
    "int main() { return 0; }\n")
  set(_stl_test_command ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
                        -sycl
                        ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
                        -isystem ${ComputeCpp_INCLUDE_DIRS}
                        -isystem ${OpenCL_INCLUDE_DIRS}
                        -o ${ComputeCpp_STL_CHECK_SRC}.sycl
                        -c ${ComputeCpp_STL_CHECK_SRC}.cpp)
  execute_process(
    COMMAND ${_stl_test_command}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
    ERROR_QUIET
    OUTPUT_QUIET)
  if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
    # Try disabling compiler version checks
    execute_process(
      COMMAND ${_stl_test_command}
              -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
      ERROR_QUIET
      OUTPUT_QUIET)
    if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
      # Try again with __CUDACC__ and _HAS_CONDITIONAL_EXPLICIT=0. This relaxes the restritions in the MSVC headers
      execute_process(
        COMMAND ${_stl_test_command}
                -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                -D_HAS_CONDITIONAL_EXPLICIT=0
                -D__CUDACC__
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
        ERROR_QUIET
        OUTPUT_QUIET)
        if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
          message(FATAL_ERROR "compute++ cannot consume hosted STL headers. This means that compute++ can't \
                               compile a simple program in this platform and will fail when used in this system.")
        else()
          list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                                                       -D_HAS_CONDITIONAL_EXPLICIT=0
                                                       -D__CUDACC__)
        endif()
    else()
      list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
    endif()
  endif()
  file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
              ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp.sycl)
endif(MSVC)
