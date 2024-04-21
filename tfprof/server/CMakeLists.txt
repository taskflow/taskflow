
add_executable(tfprof tfprof.cpp)

target_link_libraries(
  tfprof ${PROJECT_NAME} ${ATOMIC_LIBRARY} tf::default_settings
)

target_include_directories(tfprof PRIVATE ${TF_3RD_PARTY_DIR})
