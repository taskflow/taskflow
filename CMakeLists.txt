# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(Cpp-Taskflow VERSION 2.3.0 LANGUAGES CXX)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# Compiler vendors
## g++
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.3")
    message(FATAL_ERROR "\nCpp-Taskflow requires g++ at least v7.3")
  endif()
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
    message(FATAL_ERROR "\nCpp-Taskflow requires clang++ at least v6.0")
  endif() 
## AppleClang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "\nCpp-Taskflow requires AppleClang at least v11.0")
  endif()
## microsoft visual c++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(NOT MSVC_VERSION GREATER_EQUAL 1914)
    message(FATAL_ERROR "\nCpp-Taskflow requires MSVC++ at least v14.14") 
  endif()
else()
  message(FATAL_ERROR "\n\
Cpp-Taskflow currently supports the following compilers:\n\
- g++ v7.3 or above\n\
- clang++ v6.0 or above\n\
- MSVC++ v19.14 or above\n\
")
endif()

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
  $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wfatal-errors>
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wfatal-errors>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /permissive->
)

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
add_library(optimize_settings INTERFACE)
add_library(tf::optimize_settings ALIAS optimize_settings)

target_compile_options(
  optimize_settings INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-O2 -march=native>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:AppleClang>>:-O2 -march=native>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-O2 -march=native>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2 -DNDEBUG /MP>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0 -g>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0 -g>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:AppleClang>>:-O0 -g>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi /FS /DEBUG /Od /MP /MDd /Oy->
)

add_library(default_settings INTERFACE)
add_library(tf::default_settings ALIAS default_settings)
target_link_libraries(
  default_settings 
  INTERFACE 
  tf::error_settings 
  tf::optimize_settings 
  tf::features_settings
)

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
set(TF_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

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
target_link_libraries(
  simple ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(observer ${TF_EXAMPLE_DIR}/observer.cpp)
target_link_libraries(
  observer ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(subflow ${TF_EXAMPLE_DIR}/subflow.cpp)
target_link_libraries(
  subflow ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(condition ${TF_EXAMPLE_DIR}/condition.cpp)
target_link_libraries(
  condition ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(visualization ${TF_EXAMPLE_DIR}/visualization.cpp)
target_link_libraries(
  visualization ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(reduce ${TF_EXAMPLE_DIR}/reduce.cpp)
target_link_libraries(
  reduce ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(parallel_for ${TF_EXAMPLE_DIR}/parallel_for.cpp)
target_link_libraries(
  parallel_for ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(run ${TF_EXAMPLE_DIR}/run.cpp)
target_link_libraries(
  run ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(composition ${TF_EXAMPLE_DIR}/composition.cpp)
target_link_libraries(
 composition ${PROJECT_NAME} Threads::Threads tf::default_settings
)

#add_executable(dice_pools ${TF_EXAMPLE_DIR}/dice_pools.cpp)
#target_link_libraries(dice_pools ${PROJECT_NAME} Threads::Threads)  
#
#add_executable(get_best_dice ${TF_EXAMPLE_DIR}/get_best_dice.cpp)
#target_link_libraries(get_best_dice ${PROJECT_NAME} Threads::Threads)  

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
target_link_libraries(utility ${PROJECT_NAME} Threads::Threads tf::default_settings)
target_include_directories(utility PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
add_test(passive_vector  ${TF_UTEST_DIR}/utility -tc=PassiveVector)
add_test(object_pool     ${TF_UTEST_DIR}/utility -tc=ObjectPool)
add_test(function_traits ${TF_UTEST_DIR}/utility -tc=FunctionTraits)
add_test(singular_alloc  ${TF_UTEST_DIR}/utility -tc=SingularAllocator)

# unittest for WorkStealingQueue 
add_executable(wsq unittest/wsq.cpp)
target_link_libraries(wsq ${PROJECT_NAME} Threads::Threads)
target_include_directories(wsq PRIVATE ${TF_3RD_PARTY_DIR}/doctest tf::default_settings)
add_test(work_stealing_queue.Owner  ${TF_UTEST_DIR}/wsq -tc=WSQ.Owner)
add_test(work_stealing_queue.1Thief ${TF_UTEST_DIR}/wsq -tc=WSQ.1Thief)
add_test(work_stealing_queue.2Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.2Thieves)
add_test(work_stealing_queue.3Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.3Thieves)
add_test(work_stealing_queue.4Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.4Thieves)
add_test(work_stealing_queue.5Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.5Thieves)
add_test(work_stealing_queue.6Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.6Thieves)
add_test(work_stealing_queue.7Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.7Thieves)
add_test(work_stealing_queue.8Thieves ${TF_UTEST_DIR}/wsq -tc=WSQ.8Thieves)

# unittest for basics
add_executable(basics unittest/basics.cpp)
target_link_libraries(basics ${PROJECT_NAME} Threads::Threads tf::default_settings)
target_include_directories(basics PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
add_test(builder          ${TF_UTEST_DIR}/basics -tc=Builder)
add_test(creation         ${TF_UTEST_DIR}/basics -tc=Creation)
add_test(iterators        ${TF_UTEST_DIR}/basics -tc=Iterators)       
add_test(sequential_runs  ${TF_UTEST_DIR}/basics -tc=SequentialRuns)
add_test(parallel_runs    ${TF_UTEST_DIR}/basics -tc=ParallelRuns)
add_test(parallel_for     ${TF_UTEST_DIR}/basics -tc=ParallelFor)
add_test(parallel_for_idx ${TF_UTEST_DIR}/basics -tc=ParallelForOnIndex)
add_test(reduce           ${TF_UTEST_DIR}/basics -tc=Reduce)
add_test(reduce_min       ${TF_UTEST_DIR}/basics -tc=ReduceMin)
add_test(reduce_max       ${TF_UTEST_DIR}/basics -tc=ReduceMax)
add_test(joined_subflow   ${TF_UTEST_DIR}/basics -tc=JoinedSubflow)
add_test(detached_subflow ${TF_UTEST_DIR}/basics -tc=DetachedSubflow)
add_test(tree_subflow     ${TF_UTEST_DIR}/basics -tc=TreeSubflow)
add_test(composition-1    ${TF_UTEST_DIR}/basics -tc=Composition-1)
add_test(composition-2    ${TF_UTEST_DIR}/basics -tc=Composition-2)
add_test(composition-3    ${TF_UTEST_DIR}/basics -tc=Composition-3)
add_test(observer         ${TF_UTEST_DIR}/basics -tc=Observer)
add_test(loop_cond                ${TF_UTEST_DIR}/basics -tc=LoopCondition)
add_test(flip_coin_cond           ${TF_UTEST_DIR}/basics -tc=FlipCoinCondition)
add_test(cyclic_cond              ${TF_UTEST_DIR}/basics -tc=CyclicCondition)
add_test(btree_cond               ${TF_UTEST_DIR}/basics -tc=BTreeCondition)
add_test(dynamic_btree_cond       ${TF_UTEST_DIR}/basics -tc=DynamicBTreeCondition)
add_test(nested_for_dynamic_cond  ${TF_UTEST_DIR}/basics -tc=NestedForDynamicCondition)
add_test(cond2cond                ${TF_UTEST_DIR}/basics -tc=Condition2Condition)
add_test(hierarchical_cond        ${TF_UTEST_DIR}/basics -tc=HierarchicalCondition)

# unittest for graphs
#add_executable(graphs unittest/graphs.cpp)
#target_link_libraries(graphs ${PROJECT_NAME} Threads::Threads tf::default_settings)
#target_include_directories(graphs PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
#add_test(DynamicLevelize  ${TF_UTEST_DIR}/graphs -tc=DynamicLevelize)

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
target_include_directories(wavefront PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
target_link_libraries(
  wavefront 
  ${PROJECT_NAME} 
  Threads::Threads 
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
  Threads::Threads 
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
  Threads::Threads 
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
  Threads::Threads 
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
  Threads::Threads 
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
  Threads::Threads 
  ${TBB_IMPORTED_TARGETS} 
  ${OpenMP_CXX_LIBRARIES} 
  tf::default_settings
)
set_target_properties(matrix_multiplication PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

## benchmark 6: Parallel DNN
#message(STATUS "benchmark 6: Parallel DNN")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/parallel_dnn)
#add_executable(
#  parallel_dnn 
#  ${TF_BENCHMARK_DIR}/parallel_dnn/main.cpp
#  ${TF_BENCHMARK_DIR}/parallel_dnn/omp.cpp
#  ${TF_BENCHMARK_DIR}/parallel_dnn/tbb.cpp
#  ${TF_BENCHMARK_DIR}/parallel_dnn/seq.cpp
#  ${TF_BENCHMARK_DIR}/parallel_dnn/taskflow.cpp
#)
#target_include_directories(parallel_dnn PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
#target_link_libraries(
#  parallel_dnn 
#  ${PROJECT_NAME} 
#  Threads::Threads 
#  ${TBB_IMPORTED_TARGETS}
#  ${OpenMP_CXX_LIBRARIES} 
#  stdc++fs 
#  tf::default_settings
#)
#set_target_properties(parallel_dnn PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
#


### benchmark 8: Mandelbrot set
#message(STATUS "benchmark 8: Mandelbrot set")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/mandelbrot_set)
#add_executable(
#  mandelbrot_set
#  ${TF_BENCHMARK_DIR}/mandelbrot_set/main.cpp
#  ${TF_BENCHMARK_DIR}/mandelbrot_set/omp.cpp
#  ${TF_BENCHMARK_DIR}/mandelbrot_set/tbb.cpp
#  ${TF_BENCHMARK_DIR}/mandelbrot_set/taskflow.cpp
#)
#target_include_directories(mandelbrot_set PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
#target_link_libraries(
#  mandelbrot_set
#  ${PROJECT_NAME} 
#  Threads::Threads 
#  ${TBB_IMPORTED_TARGETS} 
#  ${OpenMP_CXX_LIBRARIES} 
#  tf::default_settings
#)
#set_target_properties(mandelbrot_set PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
#
### benchmark 9: Black–Scholes
#message(STATUS "benchmark 9: Black-Scholes Partial Differential Equation")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/black_scholes)
#add_executable(
#  black_scholes
#  ${TF_BENCHMARK_DIR}/black_scholes/main.cpp
#  ${TF_BENCHMARK_DIR}/black_scholes/omp.cpp
#  ${TF_BENCHMARK_DIR}/black_scholes/tbb.cpp
#  ${TF_BENCHMARK_DIR}/black_scholes/taskflow.cpp
#)
#target_include_directories(black_scholes PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
#target_link_libraries(
#  black_scholes
#  ${PROJECT_NAME} 
#  Threads::Threads 
#  ${TBB_IMPORTED_TARGETS} 
#  ${OpenMP_CXX_LIBRARIES} 
#  tf::default_settings
#)
#set_target_properties(black_scholes PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
#
### benchmark 10: Strassen algorithm
#message(STATUS "benchmark 10: Strassen matrix multiplication algorithm")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_BENCHMARK_DIR}/strassen)
#add_executable(
#  strassen
#  ${TF_BENCHMARK_DIR}/strassen/main.cpp
#  ${TF_BENCHMARK_DIR}/strassen/omp.cpp
#  ${TF_BENCHMARK_DIR}/strassen/tbb.cpp
#  ${TF_BENCHMARK_DIR}/strassen/taskflow.cpp
#)
#target_include_directories(strassen PRIVATE ${PROJECT_SOURCE_DIR}/3rd-party/CLI11)
#target_link_libraries(
#  strassen
#  ${PROJECT_NAME} 
#  Threads::Threads 
#  ${TBB_IMPORTED_TARGETS} 
#  ${OpenMP_CXX_LIBRARIES} 
#  tf::default_settings
#)
#set_target_properties(strassen PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

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

