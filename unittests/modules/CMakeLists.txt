include(${TF_3RD_PARTY_DIR}/doctest/doctest.cmake)

add_executable(test_cxx_modules test_cxx_modules.cpp)

set_target_properties(test_cxx_modules PROPERTIES CXX_SCAN_FOR_MODULES ON)

target_link_libraries(test_cxx_modules
  tf_module
  ${ATOMIC_LIBRARY}
  tf::default_settings
)

target_include_directories(test_cxx_modules PRIVATE ${TF_3RD_PARTY_DIR}/doctest)

doctest_discover_tests(test_cxx_modules)
