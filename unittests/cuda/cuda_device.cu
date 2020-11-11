#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>

TEST_CASE("cuda.version") {
  REQUIRE(tf::cuda_get_driver_version() > 0);
  REQUIRE(tf::cuda_get_runtime_version() > 0);
}

TEST_CASE("cuda.device") {

  REQUIRE(tf::cuda_get_num_devices() > 0);
  REQUIRE(tf::cuda_get_device() >= 0);

  size_t num_devices = tf::cuda_get_num_devices();

  for(size_t d=0; d<num_devices; d++) {
    tf::cuda_set_device(d);
    REQUIRE(tf::cuda_get_device() == d);
    
    for(size_t d=0; d<num_devices; d++) {
      REQUIRE(tf::cuda_get_device_max_threads_per_block(d) > 0);
      REQUIRE(tf::cuda_get_device_max_x_dim_per_block(d) > 0);
      REQUIRE(tf::cuda_get_device_max_y_dim_per_block(d) > 0);
      REQUIRE(tf::cuda_get_device_max_z_dim_per_block(d) > 0);
      REQUIRE(tf::cuda_get_device_max_x_dim_per_grid(d) > 0);
      REQUIRE(tf::cuda_get_device_max_y_dim_per_grid(d) > 0);
      REQUIRE(tf::cuda_get_device_max_z_dim_per_grid(d) > 0);
      REQUIRE(tf::cuda_get_device_warp_size(d) > 0);
      REQUIRE(tf::cuda_get_device_max_shm_per_block(d) > 0);
      REQUIRE(tf::cuda_get_device_compute_capability_major(d) > 0);
      REQUIRE(tf::cuda_get_device_compute_capability_minor(d) >= 0);
      REQUIRE_NOTHROW(tf::cuda_get_device_unified_addressing(d));
    }
  }
  
  // going back to device 0
  tf::cuda_set_device(0);
}





