# -------------------------------------------------------------------
# Taskflow CMake 
#
# To enable thread sanitizer: 
# -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
#
# To enable address and leak sanitizers
# -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g"
#
# To enable different c++ standards
# -DCMAKE_CXX_STANDARD=17
# -DCMAKE_CXX_STANDARD=20
# -DCMAKE_CXX_STANDARD=23
# -------------------------------------------------------------------


# CMake version
cmake_minimum_required (VERSION 3.18)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Taskflow VERSION 3.9.0 LANGUAGES CXX)

# build options
option(TF_BUILD_BENCHMARKS "Enables builds of benchmarks" OFF)
option(TF_BUILD_PROFILER "Enables builds of profiler" OFF)
option(TF_BUILD_CUDA "Enables builds of cuda code" OFF)
option(TF_BUILD_SYCL "Enables builds of sycl code" OFF)
option(TF_BUILD_TESTS "Enables builds of tests" ON)
option(TF_BUILD_EXAMPLES "Enables builds of examples" ON)

# project-specific variables
set(TF_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(TF_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
set(TF_BENCHMARK_DIR ${PROJECT_SOURCE_DIR}/benchmarks)
set(TF_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)
set(TF_CMAKE_DIR ${PROJECT_SOURCE_DIR}/cmake)
set(TF_DEFAULT_BUILD_TYPE "Release")

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Include additional language check
include(CheckLanguage)

# Adhere to GNU conventions
include(GNUInstallDirs)

# Check for correct settings for atomic library
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Atomic/CheckAtomic.cmake)

# Compiler vendors
## g++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.4")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires g++ at least v8.4")
  endif()
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires clang++ at least v6.0")
  endif() 
## AppleClang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    message(FATAL_ERROR "\nTaskflow requires AppleClang at least v12.0")
  endif()
## microsoft visual c++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_definitions(/bigobj)
  if(NOT MSVC_VERSION GREATER_EQUAL 1914)
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires MSVC++ at least v14.14") 
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0.1")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires icpc at least v19.0.1")
  endif()
else()
  message(FATAL_ERROR "\n\
Taskflow currently supports the following compilers:\n\
  - g++ v7.0 or above\n\
  - clang++ v6.0 or above\n\
  - MSVC++ v19.14 or above\n\
  - AppleClang v8 or above\n\
  - Intel v19.0.1 or above\n\
")
endif()

# -----------------------------------------------------------------------------
# find nvcc
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# -----------------------------------------------------------------------------
if(TF_BUILD_CUDA)
  message(STATUS "Configuring CUDA ...")
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "\nNo CUDA compiler found")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
  endif()
  
  ## configure cuBLAS
  #message(STATUS "Configuring CUDA BLAS ...")
  #list(APPEND CMAKE_MODULE_PATH ${TF_CMAKE_DIR}/cuda/modules/)
  #find_package(cuBLAS)
  #message(STATUS "CUBLAS_FOUND: ${CUBLAS_FOUND}")
  #message(STATUS "CUBLAS_INCLUDE_DIRS: ${CUBLAS_INCLUDE_DIRS}")
  #message(STATUS "CUBLAS_LIBRARIES: ${CUBLAS_LIBRARIES}")
endif()

# -----------------------------------------------------------------------------
# find ComputeCpp
# https://developer.codeplay.com/products/computecpp/ce/guides/integration-guide
# -----------------------------------------------------------------------------
# Users can specify:
# 1. -DComputeCpp_DIR=/path/to/ComputeCpp/
# 2. -DCOMPUTECPP_BITCODE=ptx64 (default spir64)
# For example:
# cmake ../ -DTF_BUILD_SYCL=ON 
#           -DCOMPUTECPP_BITCODE=ptx64 
#           -DComputeCpp_DIR=$COMPUTECPP_HOME
#if(TF_BUILD_SYCL)
#  message(STATUS "Configuring ComputeCpp for SYCL ...")
#  message(STATUS "You may specify '-DComputeCpp_DIR=/path/to/ComputeCpp/'")
#  message(STATUS "You may specify '-DCOMPUTECPP_BITCODE=ptx64' for Nvidia devices")
#  list(APPEND CMAKE_MODULE_PATH ${TF_CMAKE_DIR}/ComputeCpp/Modules/)
#  if(NOT ComputeCpp_DIR)
#    message(FATAL_ERROR "Please specify -DComputeCpp_DIR=/path/to/ComputeCpp/")
#  endif()
#  find_package(ComputeCpp REQUIRED)
#  message(STATUS "ComputeCpp_DIR: " ${ComputeCpp_DIR})
#  message(STATUS "ComputeCpp_INCLUDE_DIRS: " ${ComputeCpp_INCLUDE_DIRS})
#  include_directories(${ComputeCpp_INCLUDE_DIRS})
#endif()

# -----------------------------------------------------------------------------
# default to release build, unless explicitly specified
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# tf::default_setting
# -----------------------------------------------------------------------------

# error setting
add_library(error_settings INTERFACE)
add_library(tf::error_settings ALIAS error_settings)

target_compile_options(
  error_settings
  INTERFACE
  $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wfatal-errors -Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wfatal-errors -Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -Wfatal-errors -Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3 /permissive->
)

# cuda compiler options
if(TF_BUILD_CUDA)
  message(STATUS "Configuring CUDA compiler options ...")
  target_compile_options(
    error_settings
    BEFORE
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors,-Wshadow>
  )
endif(TF_BUILD_CUDA)

# sycl compiler options (currently support only DPC++)
if(TF_BUILD_SYCL)
  message(STATUS "Configuring SYCL compiler options ...")
  message(STATUS "You may specify -DTF_BUILD_SYCL_BITCODE=ptx64 for CUDA devices")
  list(APPEND TF_SYCL_OPTIONS -fsycl -fsycl-unnamed-lambda)
  if(TF_BUILD_SYCL_BITCODE STREQUAL "ptx64")
    message(STATUS "Configuring SYCL compiler options to CUDA ptx64 ...")
    list(APPEND TF_SYCL_OPTIONS -fsycl-targets=nvptx64-nvidia-cuda)
  endif()
endif()

# additional features
#add_library(features_settings INTERFACE)
#add_library(tf::features_settings ALIAS features_settings)
#target_compile_definitions(
#  features_settings 
#  INTERFACE 
#  $<$<CXX_COMPILER_ID:MSVC>:_ENABLE_EXTENDED_ALIGNED_STORAGE>
#)

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
  #tf::features_settings
)

# -----------------------------------------------------------------------------
# start build
# -----------------------------------------------------------------------------

message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_SYSTEM: " ${CMAKE_SYSTEM})
message(STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_STANDARD: " ${CMAKE_CXX_STANDARD})
message(STATUS "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(STATUS "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(STATUS "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(STATUS "CMAKE_CUDA_STANDARD: " ${CMAKE_CUDA_STANDARD})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_INSTALL_FULL_INCLUDEDIR: " ${CMAKE_INSTALL_FULL_INCLUDEDIR})
message(STATUS "CMAKE_INSTALL_FULL_LIBDIR: " ${CMAKE_INSTALL_FULL_LIBDIR})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "TF_BUILD_BENCHMARKS: " ${TF_BUILD_BENCHMARKS})
message(STATUS "TF_BUILD_PROFILER: " ${TF_BUILD_PROFILER})
message(STATUS "TF_BUILD_CUDA: " ${TF_BUILD_CUDA})
message(STATUS "TF_BUILD_SYCL: " ${TF_BUILD_SYCL})
message(STATUS "TF_BUILD_SYCL_BITCODE: " ${TF_BUILD_SYCL_BITCODE})
message(STATUS "TF_BUILD_TESTS: " ${TF_BUILD_TESTS})
message(STATUS "TF_BUILD_EXAMPLES: " ${TF_BUILD_EXAMPLES})
message(STATUS "TF_UTEST_DIR: " ${TF_UTEST_DIR})
message(STATUS "TF_EXAMPLE_DIR: " ${TF_EXAMPLE_DIR})
message(STATUS "TF_BENCHMARK_DIR: " ${TF_BENCHMARK_DIR})
message(STATUS "TF_3RD_PARTY_DIR: " ${TF_3RD_PARTY_DIR})

# add the binary tree to the search path for include files
include_directories(${PROJECT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# must-have package include
# -----------------------------------------------------------------------------

# Find pthread package
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------------
# Taskflow library interface
# -----------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_link_libraries(${PROJECT_NAME} INTERFACE ${ATOMIC_LIBRARY} Threads::Threads)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)

# -----------------------------------------------------------------------------
# Example program 
# -----------------------------------------------------------------------------
if(TF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(TF_BUILD_EXAMPLES)

# -----------------------------------------------------------------------------
# Unittest
# -----------------------------------------------------------------------------
if(TF_BUILD_TESTS)
  include(CTest)
  add_subdirectory(unittests)
endif(TF_BUILD_TESTS)

# -----------------------------------------------------------------------------
# Profiler
# -----------------------------------------------------------------------------

if(TF_BUILD_PROFILER)
  add_subdirectory(tfprof/server)
endif(TF_BUILD_PROFILER)

# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

if(TF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(TF_BUILD_BENCHMARKS)

# -----------------------------------------------------------------------------
# create find_package(Taskflow CONFIG)
# -----------------------------------------------------------------------------

# install header
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/taskflow DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# set up config
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  PATH_VARS CMAKE_INSTALL_FULL_INCLUDEDIR
)

write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake 
  COMPATIBILITY SameMinorVersion
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

