enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TF_UNITTESTS 
  test_utility 
  test_iterators
  test_wsq
  test_spsc
  test_basics 
  test_notifiers
  test_work_stealing 
  #test_serializer 
  test_workers
  test_asyncs
  test_dependent_asyncs
  test_subflows
  test_control_flow
  test_semaphores
  test_movable
  test_cancellation
  test_for_each
  test_reduce
  test_transform
  test_sort
  test_scan
  test_find
  test_modules
  test_merge
  test_traversals
  test_runtimes
  test_task_group
  test_pipelines
  test_scalable_pipelines
  test_data_pipelines
  test_atomic_intrusive_stack
  test_fill
  test_generate
)

# we only do exception tests if sanitizer is not enabled
string(FIND '${CMAKE_CXX_FLAGS}' "-fsanitize" sanitize)
#message("sanitize='${sanitize}'")

if(${sanitize} EQUAL -1)
  message(STATUS "Add test_exception to unit tests")
  list(APPEND TF_UNITTESTS test_exceptions)
endif()


foreach(unittest IN LISTS TF_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings)
  target_include_directories(${unittest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

# test_for_each uses high-dimensional IndexRange templates that exceed MSVC's
# default object file section limit — /bigobj raises it from 2^16 to 2^32.
if(MSVC)
  target_compile_options(test_for_each PRIVATE /bigobj)
  target_compile_options(test_iterators PRIVATE /bigobj)
endif()

# include C++ module tests
if(TF_BUILD_MODULES)
  add_subdirectory(${TF_UTEST_DIR}/modules)
endif()

# include CUDA tests
if(TF_BUILD_CUDA)
  add_subdirectory(${TF_UTEST_DIR}/cuda)
endif()

