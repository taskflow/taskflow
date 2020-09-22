add_executable(device_property device_property.cu)
target_link_libraries(
  device_property ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(saxpy saxpy.cu)
target_link_libraries(
  saxpy ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(matmul matmul.cu)
target_link_libraries(
  matmul ${PROJECT_NAME} Threads::Threads tf::default_settings
)

add_executable(kmeans kmeans.cu)
target_link_libraries(
  kmeans ${PROJECT_NAME} Threads::Threads tf::default_settings
)
