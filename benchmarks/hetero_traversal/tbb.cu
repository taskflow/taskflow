#include "graph.hpp"

#include <tbb/global_control.h>
#include <tbb/flow_graph.h>

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

  
void TBB(const Graph& g, unsigned num_cpus, unsigned num_gpus) {

  using namespace tbb;
  using namespace tbb::flow;
    
  tbb::global_control c(
    tbb::global_control::max_allowed_parallelism, num_cpus + num_gpus
  );
  tbb::flow::graph G;

  cudaStream streams(num_cpus + num_gpus);
  
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

  std::vector<std::unique_ptr<continue_node<continue_msg>>> tasks(g.num_nodes);
  std::vector<size_t> indegree(g.num_nodes, 0);
  auto source = std::make_unique<continue_node<continue_msg>>(
    G, [](const continue_msg&){}
  );
  
  // create a task for each node
  for(const auto& v : g.nodes) {
    // cpu task
    if(v.g == -1) {
      tasks[v.v] = std::make_unique<continue_node<continue_msg>>(G, 
        [&](const continue_msg&){ 
          for(int i=0; i<N; ++i) {
            cz[i] = cx[i] + cy[i];
          }
          ++counter; 
        }
      );
    }
    else {
      tasks[v.v] = std::make_unique<continue_node<continue_msg>>(G,
        [&](const continue_msg&){
          tf::cudaScopedDevice device(v.g);

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
      );
    }
  }
  for(const auto& e : g.edges) {
    make_edge(*tasks[e.u], *tasks[e.v]);
    indegree[e.v]++;
  }
  for(size_t i=0; i<indegree.size(); ++i) {
    if(indegree[i] == 0) {
      make_edge(*source, *tasks[i]);
    }
  }
  source->try_put(continue_msg());
  G.wait_for_all();

  
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

std::chrono::microseconds measure_time_tbb(
  const Graph& g, unsigned num_cpus, unsigned num_gpus
) {
  auto beg = std::chrono::high_resolution_clock::now();
  TBB(g, num_cpus, num_gpus);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

