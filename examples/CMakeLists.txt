message(STATUS "Building examples ...")

set(TF_EXAMPLES
  simple observer subflow fibonacci condition
  visualization reduce parallel_for run composition
  )

foreach(example IN LISTS TF_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} ${PROJECT_NAME} tf::default_settings
    )
endforeach()

if (CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  add_subdirectory(cuda)
endif()


#### TensorFrame Project
##set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_EXAMPLE_DIR}/tensorframe)
#add_executable(add ${TF_EXAMPLE_DIR}/tensorframe/add.cpp)
#target_link_libraries(
#  add TensorFrame Threads::Threads tf::default_settings
#)


#### TaskflowDSL project
#### TODO: enable only when C++17 is available
##set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TF_EXAMPLE_DIR}/dsl)
#add_executable(condition_dsl ${TF_EXAMPLE_DIR}/dsl/condition_dsl.cpp)
#target_link_libraries(
#  condition_dsl TaskflowDSL tf::default_settings
#)
#
#add_executable(simple_dsl ${TF_EXAMPLE_DIR}/dsl/simple_dsl.cpp)
#target_link_libraries(
#  simple_dsl TaskflowDSL tf::default_settings
#)
#
#add_executable(visualization_dsl ${TF_EXAMPLE_DIR}/dsl/visualization_dsl.cpp)
#target_link_libraries(
#  visualization_dsl TaskflowDSL tf::default_settings
#)
