// This is considered as "handcrafted version" using levelization
// to avoid all tasking overhead. 
// The performance is typically the best.
#include "graph.hpp"
#include <queue>

struct cudaStream {

  std::vector<std::vector<cudaStream_t>> streams;

  cudaStream(unsigned N) : streams(tf::cuda_get_num_devices()) {
    for(size_t i=0; i<streams.size(); ++i) {
      streams[i].resize(N);
      tf::cudaScopedDevice ctx(i);
      for(unsigned j=0; j<N; ++j) {
        TF_CHECK_CUDA(cudaStreamCreate(&streams[i][j]), "failed to create a stream on ", i);
      }
    }
  }

  ~cudaStream() {
    for(size_t i=0; i<streams.size(); ++i) {
      tf::cudaScopedDevice ctx(i);
      for(unsigned j=0; j<streams[i].size(); ++j) {
        cudaStreamDestroy(streams[i][j]);
      }
    }
  }
  
  cudaStream_t per_thread_stream(int device) {
    auto id = std::hash<std::thread::id>()(std::this_thread::get_id()) % streams[device].size();
    return streams[device][id];
  }

};


void omp(const Graph& g, unsigned num_cpus, unsigned num_gpus) {

  std::vector<size_t> indegree(g.num_nodes, 0);
  std::vector<std::vector<int>> levellist(g.num_nodes);
  std::vector<std::vector<int>> adjlist(g.num_nodes);
  
  cudaStream streams(num_cpus + num_gpus);
  
  for(const auto& e : g.edges) {
    adjlist[e.u].push_back(e.v);
    indegree[e.v]++;
  }

  std::queue<std::pair<int, int>> queue;

  for(size_t i=0; i<indegree.size(); ++i) {
    if(indegree[i] == 0) {
      queue.push({i, 0});
    }
  }

  while(!queue.empty()) {
    auto u = queue.front();
    queue.pop();

    levellist[u.second].push_back(u.first);
    
    for(auto v : adjlist[u.first]) {
      indegree[v]--;
      if(indegree[v] == 0) {
        queue.push({v, u.second + 1});
      }
    }
  }

  //std::cout << "Levellist:\n";
  //for(size_t l=0; l<levellist.size(); ++l) {
  //  std::cout << l << ':';
  //  for(const auto u : levellist[l]) {
  //    std::cout << ' ' << u;
  //  }
  //  std::cout << '\n';
  //}
  
  std::atomic<int> counter {0};
  
  int* cx = new int[N];
  int* cy = new int[N];
  int* cz = new int[N];
  int* gx = nullptr;
  int* gy = nullptr;
  int* gz = nullptr;
  TF_CHECK_CUDA(cudaMallocManaged(&gx, N*sizeof(int)), "failed at cudaMallocManaged");
  TF_CHECK_CUDA(cudaMallocManaged(&gy, N*sizeof(int)), "failed at cudaMallocManaged");
  TF_CHECK_CUDA(cudaMallocManaged(&gz, N*sizeof(int)), "failed at cudaMallocManaged");
  
  for(size_t l=0; l<levellist.size(); ++l) {
    #pragma omp parallel for num_threads(num_cpus + num_gpus)
    for(size_t i=0; i<levellist[l].size(); ++i) {
      int u = levellist[l][i];
      if(g.nodes[u].g== -1) {
        ++counter;
        for(int i=0; i<N; ++i) {
          cz[i] = cx[i] + cy[i];
        }
      }
      else {
        tf::cudaScopedDevice device(g.nodes[u].g);

        tf::cudaStream stream;
        tf::cudaFlow cf;

        ++counter;
        auto sgx = cf.zero(gx, N);
        auto sgy = cf.zero(gy, N);
        auto sgz = cf.zero(gz, N);
        auto h2d_gx = cf.copy(gx, cx, N);
        auto h2d_gy = cf.copy(gy, cy, N);
        auto h2d_gz = cf.copy(gz, cz, N);
        auto kernel = cf.kernel((N+255)/256, 256, 0, add<int>, gx, gy, gz, N);
        auto d2h_gx = cf.copy(cx, gx, N);
        auto d2h_gy = cf.copy(cy, gy, N);
        auto d2h_gz = cf.copy(cz, gz, N);
        sgx.precede(h2d_gx);
        sgy.precede(h2d_gy);
        sgz.precede(h2d_gz);
        kernel.succeed(h2d_gx, h2d_gy, h2d_gz)
              .precede(d2h_gx, d2h_gy, d2h_gz);

        cf.run(stream);
        stream.synchronize();
      }
    }

  }
  
  delete [] cx;
  delete [] cy;
  delete [] cz;
  TF_CHECK_CUDA(cudaFree(gx), "failed at cudaFree");
  TF_CHECK_CUDA(cudaFree(gy), "failed at cudaFree");
  TF_CHECK_CUDA(cudaFree(gz), "failed at cudaFree");

  if(counter != g.num_nodes) {
    throw std::runtime_error("wrong result");
  }
}


std::chrono::microseconds measure_time_omp(
  const Graph& g, unsigned num_cpus, unsigned num_gpus
) {
  auto beg = std::chrono::high_resolution_clock::now();
  omp(g, num_cpus, num_gpus);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}
