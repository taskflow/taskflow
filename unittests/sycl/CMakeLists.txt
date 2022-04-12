list(APPEND TF_SYCL_TESTS 
  sycl_basics
  #sycl_algorithms
  #sycl_rebinds
)

foreach(sycl_test IN LISTS TF_SYCL_TESTS)
  add_executable(${sycl_test} ${sycl_test}.cpp)

  #add_sycl_to_target(TARGET ${sycl_test} SOURCES ${sycl_test}.cpp)

  target_compile_options(${sycl_test} PRIVATE ${TF_SYCL_OPTIONS})
  target_link_options(${sycl_test} PRIVATE ${TF_SYCL_OPTIONS})
  target_link_libraries(${sycl_test}
    ${PROJECT_NAME} Threads::Threads tf::default_settings
  )
  target_include_directories(${sycl_test} PRIVATE ${TF_3RD_PARTY_DIR}/doctest)
  
  doctest_discover_tests(${sycl_test})
endforeach()
