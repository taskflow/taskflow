list(APPEND TF_EXAMPLES
  simple 
  async
  subflow_async
  observer 
  subflow 
  fibonacci 
  condition
  multi_condition
  switch_case
  do_while
  while
  if_else
  nested_if_else
  visualization 
  reduce 
  parallel_for 
  parallel_sort
  run 
  composition
  limited_concurrency
  cancel
  cancel_async
  pipeline
  #pipeline_2
)

foreach(example IN LISTS TF_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} ${PROJECT_NAME} tf::default_settings
    )
endforeach()

# -----------------------------------------------------------------------------
# cuda examples
# -----------------------------------------------------------------------------
if(TF_BUILD_CUDA)
  add_subdirectory(cuda)
endif()

# -----------------------------------------------------------------------------
# sycl examples
# -----------------------------------------------------------------------------
if(TF_BUILD_SYCL)
  add_subdirectory(sycl)
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
