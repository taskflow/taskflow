enable_testing()

include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

list(APPEND TF_UNITTESTS 
  utility tsq basics algorithm traverse sort
)

foreach(unittest IN LISTS TF_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} ${PROJECT_NAME} tf::default_settings)
  target_include_directories(${unittest} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  doctest_discover_tests(${unittest})
endforeach()

if(CMAKE_CUDA_COMPILER AND TF_BUILD_CUDA)
  add_subdirectory(${TF_UTEST_DIR}/cuda)
endif()
