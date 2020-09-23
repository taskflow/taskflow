# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Project name
project(Taskflow VERSION 2.7.0 LANGUAGES CXX)

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
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors>
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
if(NOT CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  message(WARNING "TF_BUILD_CUDA set but no CUDA compiler found!\n"
    "NOT building CUDA examples/benchmarks/...")
endif()

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

#add_library(TaskflowDSL INTERFACE)
#target_compile_features(TaskflowDSL INTERFACE cxx_std_17)
#target_link_libraries(TaskflowDSL INTERFACE Threads::Threads)
#target_include_directories(TaskflowDSL INTERFACE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include/> 
#)
#
#add_library(TensorFrame INTERFACE)
#target_compile_features(TensorFrame INTERFACE cxx_std_17)
#target_link_libraries(TensorFrame INTERFACE Threads::Threads)
#target_include_directories(TensorFrame INTERFACE
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include/> 
#)

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
  add_subdirectory(unittests)
endif(TF_BUILD_TESTS)

# -----------------------------------------------------------------------------
# Benchmarking (enabled by TF_BUILD_BENCHMARKS)
# -----------------------------------------------------------------------------

if(TF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(TF_BUILD_BENCHMARKS)

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

