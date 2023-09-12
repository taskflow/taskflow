// This program implements the k-means clustering algorithm in three forms:
//  - sequential cpu
//  - parallel cpu
//  - gpu with conditional tasking
//  - gpu without conditional tasking

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/cuda/cudaflow.hpp>

#include <iomanip>
#include <cfloat>
#include <climits>

#define L2(x1, y1, x2, y2) ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

// ----------------------------------------------------------------------------
// CPU (sequential) implementation
// ----------------------------------------------------------------------------

// run k-means on cpu
std::pair<std::vector<float>, std::vector<float>> cpu_seq(
 const int N, 
 const int K, 
 const int M,
 const std::vector<float>& px,
 const std::vector<float>& py
) {

  std::vector<int> c(K);
  std::vector<float> sx(K), sy(K), mx(K), my(K);
  
  // initial centroids
  for(int i=0; i<K; ++i) {
    mx[i] = px[i];
    my[i] = py[i];
  }
  
  for(int m=0; m<M; m++) {
  
    // clear the storage
    for(int k=0; k<K; ++k) {
      sx[k] = 0.0f;
      sy[k] = 0.0f;
      c [k] = 0;
    }

    // find the best k (cluster id) for each point
    for(int i=0; i<N; ++i) {
      float x = px[i];
      float y = py[i];
      float best_d = std::numeric_limits<float>::max();
      int best_k = 0;
      for (int k = 0; k < K; ++k) {
        const float d = L2(x, y, mx[k], my[k]);
        if (d < best_d) {
          best_d = d;
          best_k = k;
        }
      }
      sx[best_k] += x;
      sy[best_k] += y;
      c [best_k] += 1;
    }
    
    // update the centroid
    for(int k=0; k<K; k++) {
      const int count = max(1, c[k]);  // turn 0/0 to 0/1
      mx[k] = sx[k] / count;
      my[k] = sy[k] / count;
    }
  }
  
  return {mx, my};
}

// ----------------------------------------------------------------------------
// CPU (parallel) implementation
// ----------------------------------------------------------------------------

// run k-means on cpu (parallel)
std::pair<std::vector<float>, std::vector<float>> cpu_par(
 const int N, 
 const int K, 
 const int M,
 const std::vector<float>& px,
 const std::vector<float>& py
) {

  const auto num_threads = std::thread::hardware_concurrency();

  tf::Executor executor;
  tf::Taskflow taskflow("K-Means");
  
  std::vector<int> c(K), best_ks(N);
  std::vector<float> sx(K), sy(K), mx(K), my(K);
  
  // initial centroids
  auto init = taskflow.emplace([&](){
    for(int i=0; i<K; ++i) {
      mx[i] = px[i];
      my[i] = py[i];
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
    float x = px[i];
    float y = py[i];
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

  pf.name("parallel-for");

  auto update_cluster = taskflow.emplace([&](){
    for(int i=0; i<N; i++) {
      sx[best_ks[i]] += px[i];
      sy[best_ks[i]] += py[i];
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

  executor.run(taskflow).wait();
  
  return {mx, my};
}

// ----------------------------------------------------------------------------
// GPU implementation
// ----------------------------------------------------------------------------

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

// Runs k-means on gpu
std::pair<std::vector<float>, std::vector<float>> gpu_predicate(
 const int N, 
 const int K, 
 const int M,
 const std::vector<float>& h_px,
 const std::vector<float>& h_py
) {
  
  std::vector<float> h_mx, h_my;
  float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;
  
  for(int i=0; i<K; ++i) {
    h_mx.push_back(h_px[i]);
    h_my.push_back(h_py[i]);
  }
  
  // create a taskflow graph
  tf::Executor executor;
  tf::Taskflow taskflow("K-Means");
  
  auto allocate_px = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_px, N*sizeof(float)), "failed to allocate d_px"); 
  }).name("allocate_px");

  auto allocate_py = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_py, N*sizeof(float)), "failed to allocate d_py"); 
  }).name("allocate_py");
  
  auto allocate_mx = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_mx, K*sizeof(float)), "failed to allocate d_mx"); 
  }).name("allocate_mx");

  auto allocate_my = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_my, K*sizeof(float)), "failed to allocate d_my"); 
  }).name("allocate_my");

  auto allocate_sx = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_sx, K*sizeof(float)), "failed to allocate d_sx"); 
  }).name("allocate_sx");

  auto allocate_sy = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_sy, K*sizeof(float)), "failed to allocate d_sy"); 
  }).name("allocate_sy");

  auto allocate_c = taskflow.emplace([&](){
    TF_CHECK_CUDA(cudaMalloc(&d_c, K*sizeof(float)), "failed to allocate dc");
  }).name("allocate_c");

  auto h2d = taskflow.emplace([&](){
    cudaMemcpy(d_px, h_px.data(), N*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_py, h_py.data(), N*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_mx, h_mx.data(), K*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_my, h_my.data(), K*sizeof(float), cudaMemcpyDefault);
  }).name("h2d");

  auto kmeans = taskflow.emplace([&](){

    tf::cudaFlow cf;

    auto zero_c = cf.zero(d_c, K).name("zero_c");
    auto zero_sx = cf.zero(d_sx, K).name("zero_sx");
    auto zero_sy = cf.zero(d_sy, K).name("zero_sy");
    
    auto cluster = cf.kernel(
      (N+512-1) / 512, 512, 0, 
      assign_clusters, d_px, d_py, N, d_mx, d_my, d_sx, d_sy, K, d_c
    ).name("cluster"); 
    
    auto new_centroid = cf.kernel(
      1, K, 0, 
      compute_new_means, d_mx, d_my, d_sx, d_sy, d_c
    ).name("new_centroid");

    cluster.precede(new_centroid)
           .succeed(zero_c, zero_sx, zero_sy);
    
    // Repeat the execution for M times
    tf::cudaStream stream;
    for(int i=0; i<M; i++) {
      cf.run(stream);
    }
    stream.synchronize();
  }).name("update_means");

  auto stop = taskflow.emplace([&](){
    cudaMemcpy(h_mx.data(), d_mx, K*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(h_my.data(), d_my, K*sizeof(float), cudaMemcpyDefault);
  }).name("d2h");

  auto free = taskflow.emplace([&](){
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_c);
  }).name("free");
  
  // build up the dependency
  h2d.succeed(allocate_px, allocate_py, allocate_mx, allocate_my);

  kmeans.succeed(allocate_sx, allocate_sy, allocate_c, h2d)
        .precede(stop);

  stop.precede(free);
  
  // run the taskflow
  executor.run(taskflow).wait();

  //std::cout << "dumping kmeans graph ...\n";
  taskflow.dump(std::cout);
  return {h_mx, h_my};
}

// Function: main
int main(int argc, const char* argv[]) {

  if(argc != 4) {
    std::cerr << "usage: ./kmeans num_points k num_iterations\n";
    std::exit(EXIT_FAILURE);
  }
  
  const int N = std::atoi(argv[1]);
  const int K = std::atoi(argv[2]);
  const int M = std::atoi(argv[3]);

  if(N < 1) {
    throw std::runtime_error("num_points must be at least one");
  }

  if(K >= N) {
    throw std::runtime_error("k must be smaller than the number of points");
  }

  if(M < 1) {
    throw std::runtime_error("num_iterations must be larger than 0");
  }

  std::vector<float> h_px, h_py, mx, my;
  
  // Randomly generate N points
  std::cout << "generating " << N << " random points ...\n";
  for(int i=0; i<N; ++i) {
    h_px.push_back(rand()%1000 - 500);
    h_py.push_back(rand()%1000 - 500);
  }

  // k-means on cpu_seq
  std::cout << "running k-means on cpu (sequential) ... ";
  auto sbeg = std::chrono::steady_clock::now();
  std::tie(mx, my) = cpu_seq(N, K, M, h_px, h_py);
  auto send = std::chrono::steady_clock::now();
  std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(send-sbeg).count()
            << " ms\n";
  
  std::cout << "k centroids found by cpu (sequential)\n";
  for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                          << std::setw(10) << my[k] << '\n';  
  }
  
  // k-means on cpu_par
  std::cout << "running k-means on cpu (parallel) ... ";
  auto pbeg = std::chrono::steady_clock::now();
  std::tie(mx, my) = cpu_par(N, K, M, h_px, h_py);
  auto pend = std::chrono::steady_clock::now();
  std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pend-pbeg).count()
            << " ms\n";
  
  std::cout << "k centroids found by cpu (parallel)\n";
  for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                          << std::setw(10) << my[k] << '\n';  
  }
  
  // k-means on gpu 
  std::cout << "running k-means on gpu ...";
  auto rbeg = std::chrono::steady_clock::now();
  std::tie(mx, my) = gpu_predicate(N, K, M, h_px, h_py);
  auto rend = std::chrono::steady_clock::now();
  std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(rend-rbeg).count()
            << " ms\n";
  
  std::cout << "k centroids found by gpu\n";
  for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                          << std::setw(10) << my[k] << '\n';  
  }

  return 0;
}



