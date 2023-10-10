#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

TEST_CASE("cuda.version" * doctest::timeout(300) ) {
  REQUIRE(tf::cuda_get_driver_version() > 0);
  REQUIRE(tf::cuda_get_runtime_version() > 0);
}

TEST_CASE("cuda.device" * doctest::timeout(300) ) {

  REQUIRE(tf::cuda_get_num_devices() > 0);
  REQUIRE(tf::cuda_get_device() >= 0);

  size_t num_devices = tf::cuda_get_num_devices();

  for(size_t c=0; c<num_devices; c++) {
    tf::cuda_set_device(c);
    REQUIRE(tf::cuda_get_device() == c);
    
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

// ----------------------------------------------------------------------------
// stream
// ----------------------------------------------------------------------------

TEST_CASE("cudaStream" * doctest::timeout(300)) {
  
  // create a new stream s1 inside
  tf::cudaStream s1;
  
  // create another stream s2 from the outside
  cudaStream_t s2_source;
  cudaStreamCreate(&s2_source);
  tf::cudaStream s2(s2_source);
  
  REQUIRE(s2 == s2_source);

  cudaStream_t s1_source = s1;
  REQUIRE(s1 == s1_source);

  // query status
  REQUIRE(cudaStreamQuery(s1) == cudaSuccess);
  REQUIRE(cudaStreamQuery(s2) == cudaSuccess);

  s1 = std::move(s2);

  REQUIRE(s2 == nullptr);
  REQUIRE(s1 == s2_source);
  REQUIRE(cudaStreamQuery(s1) == cudaSuccess);

}

// ----------------------------------------------------------------------------
// event
// ----------------------------------------------------------------------------

TEST_CASE("cudaEvent" * doctest::timeout(300)) {
  
  // create a new event e1 inside
  tf::cudaEvent e1;
  
  // create another event e2 from the outside
  cudaEvent_t e2_source;
  cudaEventCreate(&e2_source);
  tf::cudaEvent e2(e2_source);
  
  REQUIRE(e2 == e2_source);

  cudaEvent_t e1_source = e1;
  REQUIRE(e1 == e1_source);

  // query status
  REQUIRE(cudaEventQuery(e1) == cudaSuccess);
  REQUIRE(cudaEventQuery(e2) == cudaSuccess);

  e1 = std::move(e2);

  REQUIRE(e2 == nullptr);
  REQUIRE(e1 == e2_source);
  REQUIRE(cudaEventQuery(e1) == cudaSuccess);
  REQUIRE(cudaEventQuery(e2) != cudaSuccess);
}

// ----------------------------------------------------------------------------
// CUDA Graph
// ----------------------------------------------------------------------------

TEST_CASE("cudaGraph" * doctest::timeout(300)) {
  
  // create a new graph g1 inside
  tf::cudaGraph g1;
  
  cudaGraph_t g1_source = g1;
  REQUIRE(g1 == g1_source);
  
  // create another graph g2 from the outside
  cudaGraph_t g2_source;
  cudaGraphCreate(&g2_source, 0);
  tf::cudaGraph g2(g2_source);
  
  REQUIRE(g2 == g2_source);

  g1 = std::move(g2);

  REQUIRE(g2 == nullptr);
  REQUIRE(g1 == g2_source);

  // reassign g1 (now holding g2_source) to g2
  g2.reset(g1.release());
  REQUIRE(g1 == nullptr);
  REQUIRE(g2 == g2_source);

  // clear
  g2.clear();
  g1.clear();

  REQUIRE(g1 == nullptr);
  REQUIRE(g2 == nullptr);
}


