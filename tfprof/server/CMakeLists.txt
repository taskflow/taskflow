
add_executable(tfprof tfprof.cpp)
target_link_libraries(
  tfprof ${PROJECT_NAME} tf::default_settings
)
