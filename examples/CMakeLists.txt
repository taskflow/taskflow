list(APPEND TF_EXAMPLES
  simple 
  attach_data
  async
  subflow_async
  dependent_async
  dependent_async_algorithm
  observer 
  subflow 
  fibonacci 
  condition
  multi_condition
  switch_case
  do_while_loop
  while_loop
  if_else
  nested_if_else
  priority
  visualization 
  parallel_for 
  parallel_sort
  reduce 
  inclusive_scan
  exclusive_scan
  pipeline
  scalable_pipeline
  scalable_pipeline_with_deferred_tokens
  text_pipeline
  taskflow_pipeline
  parallel_graph_pipeline
  parallel_data_pipeline
  pipeline_with_deferred_tokens
  run 
  corun
  runtime
  composition
  limited_concurrency
  cancel
  exception
)

foreach(example IN LISTS TF_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(
    ${example} ${PROJECT_NAME} tf::default_settings
    )
  # set emcc options
  if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
    target_link_options(${example} PUBLIC -sASSERTIONS=1 -sPROXY_TO_PTHREAD -sTOTAL_MEMORY=1536MB -sEXIT_RUNTIME=1 -sUSE_PTHREADS=1)
    target_compile_options(${example} PUBLIC -matomics)
  endif()
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
