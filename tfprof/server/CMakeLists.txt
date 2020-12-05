
add_executable(tfprof tfprof.cpp)

target_link_libraries(
  tfprof ${PROJECT_NAME} tf::default_settings
)

target_include_directories(tfprof PRIVATE ${TF_3RD_PARTY_DIR})
