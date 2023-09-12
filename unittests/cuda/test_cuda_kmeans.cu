#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>
#include <taskflow/cuda/algorithm/for_each.hpp>

#define L2(x1, y1, x2, y2) ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

template <typename T>
void run_and_wait(T& cf) {
  tf::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

// Each point (thread) computes its distance to each centroid 
// and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__global__ void assign_clusters(
  const float* px,
  const float* py,
  int N,
  const float* mx,
  const float* my,
  float* sx,
  float* sy,
  int k,
  int* c
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= N) {
    return;
  }

  // Make global loads once.
  const float x = px[index];
  const float y = py[index];

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = L2(x, y, mx[cluster], my[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  atomicAdd(&sx[best_cluster], x);
  atomicAdd(&sy[best_cluster], y);
  atomicAdd(&c [best_cluster], 1);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(
  float* mx, float* my, const float* sx, const float* sy, const int* c
) {
  const int cluster = threadIdx.x;
  const int count = max(1, c[cluster]);  // turn 0/0 to 0/1
  mx[cluster] = sx[cluster] / count;
  my[cluster] = sy[cluster] / count;
}

// k-means clustering
void kmeans(int N, int K, int M, size_t num_cpus, size_t num_gpus) {
  

  std::vector<float> h_px, h_py, h_mx, h_my, mx, my;
  
  std::vector<int> c(K), best_ks(N);
  std::vector<float> sx(K), sy(K);

  float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;
  
  // Randomly generate N points
  for(int i=0; i<N; ++i) {
    h_px.push_back(rand()%1000 - 500);
    h_py.push_back(rand()%1000 - 500);
    if(i < K) {
      mx.push_back(h_px.back());
      my.push_back(h_py.back());
      h_mx.push_back(h_px.back());
      h_my.push_back(h_py.back());
    }
  }
  
  tf::Taskflow taskflow;
  tf::Executor executor(num_cpus + num_gpus);
  
  // cpu version 
  auto init = taskflow.emplace([&](){
    for(int i=0; i<K; ++i) {
      mx[i] = h_px[i];
      my[i] = h_py[i];
    }
  }).name("init");
  
  // clear the storage
  auto clean_up = taskflow.emplace([&](){
    for(int k=0; k<K; ++k) {
      sx[k] = 0.0f;
      sy[k] = 0.0f;
      c [k] = 0;
    }
  }).name("clean_up");

  tf::Task pf;
  
  // update cluster
  pf = taskflow.for_each_index(0, N, 1, [&](int i){
    float x = h_px[i];
    float y = h_py[i];
    float best_d = std::numeric_limits<float>::max();
    int best_k = 0;
    for (int k = 0; k < K; ++k) {
      const float d = L2(x, y, mx[k], my[k]);
      if (d < best_d) {
        best_d = d;
        best_k = k;
      }
    }
    best_ks[i] = best_k;
  });

  auto update_cluster = taskflow.emplace([&](){
    for(int i=0; i<N; i++) {
      sx[best_ks[i]] += h_px[i];
      sy[best_ks[i]] += h_py[i];
      c [best_ks[i]] += 1;
    }
    
    for(int k=0; k<K; ++k) {
      auto count = max(1, c[k]);  // turn 0/0 to 0/1
      mx[k] = sx[k] / count;
      my[k] = sy[k] / count;
    }
  }).name("update_cluster");

  auto condition = taskflow.emplace([m=0, M]() mutable {
    return (m++ < M) ? 0 : 1;
  }).name("converged?");
  
  init.precede(clean_up);

  clean_up.precede(pf);
  pf.precede(update_cluster);

  condition.precede(clean_up)
           .succeed(update_cluster);
  
  // gpu version
  auto allocate_px = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_px, N*sizeof(float)) == cudaSuccess); 
  }).name("allocate_px");

  auto allocate_py = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_py, N*sizeof(float)) == cudaSuccess);
  }).name("allocate_py");
  
  auto allocate_mx = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_mx, K*sizeof(float)) == cudaSuccess);
  }).name("allocate_mx");

  auto allocate_my = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_my, K*sizeof(float)) == cudaSuccess);
  }).name("allocate_my");

  auto allocate_sx = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_sx, K*sizeof(float)) == cudaSuccess); 
  }).name("allocate_sx");

  auto allocate_sy = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_sy, K*sizeof(float)) == cudaSuccess);
  }).name("allocate_sy");

  auto allocate_c = taskflow.emplace([&](){
    REQUIRE(cudaMalloc(&d_c, K*sizeof(float)) == cudaSuccess);
  }).name("allocate_c");

  auto h2d = taskflow.emplace([&](){
    tf::cudaFlow cf;
    cf.copy(d_px, h_px.data(), N).name("h2d_px");
    cf.copy(d_py, h_py.data(), N).name("h2d_py");
    cf.copy(d_mx, h_mx.data(), K).name("h2d_mx");
    cf.copy(d_my, h_my.data(), K).name("h2d_my");
    run_and_wait(cf);
  }).name("h2d");

  auto kmeans = taskflow.emplace([&](){

    tf::cudaFlow cf;

    auto zero_c = cf.zero(d_c, K).name("zero_c");
    auto zero_sx = cf.zero(d_sx, K).name("zero_sx");
    auto zero_sy = cf.zero(d_sy, K).name("zero_sy");
    
    auto cluster = cf.kernel(
      (N+1024-1) / 1024, 1024, 0, 
      assign_clusters, d_px, d_py, N, d_mx, d_my, d_sx, d_sy, K, d_c
    ).name("cluster"); 
    
    auto new_centroid = cf.kernel(
      1, K, 0, 
      compute_new_means, d_mx, d_my, d_sx, d_sy, d_c
    ).name("new_centroid");

    cluster.precede(new_centroid)
           .succeed(zero_c, zero_sx, zero_sy);

    run_and_wait(cf);
  }).name("update_means");

  auto gpu_condition = taskflow.emplace([i=0, M] () mutable {
    return i++ < M ? 0 : 1;
  }).name("converged?");

  auto stop = taskflow.emplace([&](){
    tf::cudaFlow cf;
    cf.copy(h_mx.data(), d_mx, K).name("d2h_mx");
    cf.copy(h_my.data(), d_my, K).name("d2h_my");
    run_and_wait(cf);
  }).name("stop");

  auto free = taskflow.emplace([&](){
    REQUIRE(cudaFree(d_px)==cudaSuccess);
    REQUIRE(cudaFree(d_py)==cudaSuccess);
    REQUIRE(cudaFree(d_mx)==cudaSuccess);
    REQUIRE(cudaFree(d_my)==cudaSuccess);
    REQUIRE(cudaFree(d_sx)==cudaSuccess);
    REQUIRE(cudaFree(d_sy)==cudaSuccess);
    REQUIRE(cudaFree(d_c )==cudaSuccess);
  }).name("free");
  
  // build up the dependency
  h2d.succeed(allocate_px, allocate_py, allocate_mx, allocate_my);

  kmeans.succeed(allocate_sx, allocate_sy, allocate_c, h2d)
        .precede(gpu_condition);

  gpu_condition.precede(kmeans, stop);

  stop.precede(free);

  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);

  for(int k=0; k<K; k++) {
    REQUIRE(std::fabs(h_mx[k] - mx[k]) < 1.0f);
    REQUIRE(std::fabs(h_my[k] - my[k]) < 1.0f);
  }
}

TEST_CASE("kmeans.10.1C1G") {
  kmeans(10, 2, 10, 1, 1);
}

TEST_CASE("kmeans.10.1C2G") {
  kmeans(10, 2, 10, 1, 2);
}

TEST_CASE("kmeans.10.1C3G") {
  kmeans(10, 2, 10, 1, 3);
}

TEST_CASE("kmeans.10.1C4G") {
  kmeans(10, 2, 10, 1, 4);
}

TEST_CASE("kmeans.10.2C1G") {
  kmeans(10, 2, 10, 2, 1);
}

TEST_CASE("kmeans.10.2C2G") {
  kmeans(10, 2, 10, 2, 2);
}

TEST_CASE("kmeans.10.2C3G") {
  kmeans(10, 2, 10, 2, 3);
}

TEST_CASE("kmeans.10.2C4G") {
  kmeans(10, 2, 10, 2, 4);
}

TEST_CASE("kmeans.10.4C1G") {
  kmeans(10, 2, 10, 4, 1);
}

TEST_CASE("kmeans.10.4C2G") {
  kmeans(10, 2, 10, 4, 2);
}

TEST_CASE("kmeans.10.4C3G") {
  kmeans(10, 2, 10, 4, 3);
}

TEST_CASE("kmeans.10.4C4G") {
  kmeans(10, 2, 10, 4, 4);
}

TEST_CASE("kmeans.100.1C1G") {
  kmeans(100, 4, 100, 1, 1);
}

TEST_CASE("kmeans.100.2C2G") {
  kmeans(100, 4, 100, 2, 2);
}

TEST_CASE("kmeans.100.3C3G") {
  kmeans(100, 4, 100, 3, 3);
}

TEST_CASE("kmeans.100.4C4G") {
  kmeans(100, 4, 100, 4, 4);
}

TEST_CASE("kmeans.1000.1C1G") {
  kmeans(1000, 8, 1000, 1, 1);
}

TEST_CASE("kmeans.1000.2C2G") {
  kmeans(1000, 8, 1000, 2, 2);
}

TEST_CASE("kmeans.1000.4C4G") {
  kmeans(1000, 8, 1000, 4, 4);
}

TEST_CASE("kmeans.1000.8C8G") {
  kmeans(1000, 8, 1000, 8, 8);
}

TEST_CASE("kmeans.1000.16C16G") {
  kmeans(1000, 8, 1000, 16, 16);
}

