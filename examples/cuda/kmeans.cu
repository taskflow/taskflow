#include <taskflow/taskflow.hpp>

#include <cfloat>
#include <climits>

// L2 distance
__device__ float l2(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
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
  if (index >= N) return;

  // Make global loads once.
  const float x = px[index];
  const float y = py[index];

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = l2(x, y, mx[cluster], my[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  // Slow but simple.
  atomicAdd(&sx[best_cluster], x);
  atomicAdd(&sy[best_cluster], y);
  atomicAdd(&c[best_cluster], 1);
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

// Function: main
int main(int argc, const char* argv[]) {

  if(argc != 4) {
    std::cerr << "usage: ./kmeans num_points k num_iterations\n";
    std::exit(EXIT_FAILURE);
  }
  
  const int N = std::atoi(argv[1]);
  const int K = std::atoi(argv[2]);
  const int M = std::atoi(argv[3]);

  if(K >= N) {
    throw std::runtime_error("k must be smaller than the number of points");
  }

  std::vector<float> h_px, h_py, h_mx, h_my;
  float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;
  
  // Randomly generate N points
  std::cout << "generating " << N << " random points ...\n";
  for(int i=0; i<N; ++i) {
    h_px.push_back(rand()%1000);
    h_py.push_back(rand()%1000);
    if(i<K) {
      h_mx.push_back(h_px.back());
      h_my.push_back(h_py.back());
    }
  }
  std::cout << "initial K centroids\n";
  for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << h_mx[k] << ' ' << h_my[k] << '\n';  
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

  auto h2d = taskflow.emplace([&](tf::cudaFlow& cf){
    cf.copy(d_px, h_px.data(), N).name("h2d_px");
    cf.copy(d_py, h_py.data(), N).name("h2d_py");
    cf.copy(d_mx, h_mx.data(), K).name("h2d_mx");
    cf.copy(d_my, h_my.data(), K).name("h2d_my");
  }).name("h2d");

  auto kmeans = taskflow.emplace([&](tf::cudaFlow& cf){

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
  }).name("update_means");

  auto condition = taskflow.emplace([i=0, M] () mutable {
    return i++ < M ? 0 : 1;
  }).name("converged?");

  auto stop = taskflow.emplace([&](tf::cudaFlow& cf){
    cf.copy(h_mx.data(), d_mx, K).name("d2h_mx");
    cf.copy(h_my.data(), d_my, K).name("d2h_my");
  }).name("stop");
  
  // build up the dependency
  h2d.succeed(allocate_px, allocate_py, allocate_mx, allocate_my);

  kmeans.succeed(allocate_sx, allocate_sy, allocate_c, h2d)
        .precede(condition);

  condition.precede(kmeans, stop);
  
  // run the taskflow
  executor.run(taskflow).wait();

  std::cout << "dumping kmeans graph ...\n";
  taskflow.dump(std::cout);
  
  std::cout << "final K centroids\n";
  for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << h_mx[k] << ' ' << h_my[k] << '\n';  
  }

  return 0;
}
