#include "graph.hpp"

void taskflow(const Graph& g, unsigned num_cpus, unsigned num_gpus) {

  std::atomic<int> counter{0};

  int* cx = new int[N];
  int* cy = new int[N];
  int* cz = new int[N];
  int* gx = nullptr;
  int* gy = nullptr;
  int* gz = nullptr;
  TF_CHECK_CUDA(cudaMallocManaged(&gx, N*sizeof(int)), "failed at cudaMalloc");
  TF_CHECK_CUDA(cudaMallocManaged(&gy, N*sizeof(int)), "failed at cudaMalloc");
  TF_CHECK_CUDA(cudaMallocManaged(&gz, N*sizeof(int)), "failed at cudaMalloc");

  tf::Taskflow taskflow;
  tf::Executor executor(num_cpus + num_gpus);

  std::vector<tf::Task> tasks(g.num_nodes);
  
  // create a task for each node
  for(const auto& v : g.nodes) {
    // cpu task
    if(v.g == -1) {
      tasks[v.v] = taskflow.emplace([&](){ 
        ++counter; 
        for(int i=0; i<N; ++i) {
          cz[i] = cx[i] + cy[i];
        }
      });
    }
    else {
      tasks[v.v] = taskflow.emplace([&](){

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

      });
    }
  }
  for(const auto& e : g.edges) {
    tasks[e.u].precede(tasks[e.v]);
  }
  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);

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

std::chrono::microseconds measure_time_taskflow(
  const Graph& g, unsigned num_cpus, unsigned num_gpus
) {
  auto beg = std::chrono::high_resolution_clock::now();
  taskflow(g, num_cpus, num_gpus);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

