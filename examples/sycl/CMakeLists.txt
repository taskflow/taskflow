list(APPEND TF_SYCL_EXAMPLES 
  vector_add
)

foreach(sycl_example IN LISTS TF_SYCL_EXAMPLES)
  add_executable(${sycl_example} ${sycl_example}.cpp)
  add_sycl_to_target(TARGET ${sycl_example} SOURCES ${sycl_example}.cpp)
  target_link_libraries(${sycl_example}
    ${PROJECT_NAME} Threads::Threads tf::default_settings
  )
endforeach()
