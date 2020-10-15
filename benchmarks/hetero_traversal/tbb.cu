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
  
  const int N = 1000;

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
        [&, tgt_device=v.g](const continue_msg&){
          ++counter;
          int src_device = -1;
          TF_CHECK_CUDA(cudaGetDevice(&src_device), "get device failed");
          TF_CHECK_CUDA(cudaSetDevice(tgt_device), "set device failed");

          cudaGraph_t cuda_graph;
          TF_CHECK_CUDA(cudaGraphCreate(&cuda_graph, 0), "cudaGraphCreate failed");

          // memset parameter
          cudaMemsetParams msetp;
          msetp.value = 0;
          msetp.pitch = 0;
          msetp.elementSize = sizeof(int);  // either 1, 2, or 4
          msetp.width = N;
          msetp.height = 1;

          // sgx
          cudaGraphNode_t sgx;
          msetp.dst = gx;
          TF_CHECK_CUDA(cudaGraphAddMemsetNode(&sgx, cuda_graph, 0, 0, &msetp), "sgx failed");
          
          // sgy
          cudaGraphNode_t sgy;
          msetp.dst = gy;
          TF_CHECK_CUDA(cudaGraphAddMemsetNode(&sgy, cuda_graph, 0, 0, &msetp), "sgy failed");
          
          // sgz
          cudaGraphNode_t sgz;
          msetp.dst = gz;
          TF_CHECK_CUDA(cudaGraphAddMemsetNode(&sgz, cuda_graph, 0, 0, &msetp), "sgz failed");
      
          // copy parameter
          cudaMemcpy3DParms h2dp;
          h2dp.srcArray = nullptr;
          h2dp.srcPos = ::make_cudaPos(0, 0, 0);
          h2dp.dstArray = nullptr;
          h2dp.dstPos = ::make_cudaPos(0, 0, 0);
          h2dp.extent = ::make_cudaExtent(N*sizeof(int), 1, 1);
          h2dp.kind = cudaMemcpyDefault;

          // h2d_gx
          cudaGraphNode_t h2d_gx;
          h2dp.srcPtr = ::make_cudaPitchedPtr(cx, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_cudaPitchedPtr(gx, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&h2d_gx, cuda_graph, 0, 0, &h2dp), "h2d_gx failed");

          // h2d_gy
          cudaGraphNode_t h2d_gy;
          h2dp.srcPtr = ::make_cudaPitchedPtr(cy, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_cudaPitchedPtr(gy, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&h2d_gy, cuda_graph, 0, 0, &h2dp), "h2d_gy failed");

          // h2d_gz
          cudaGraphNode_t h2d_gz;
          h2dp.srcPtr = ::make_cudaPitchedPtr(cz, N*sizeof(int), N, 1);
          h2dp.dstPtr = ::make_cudaPitchedPtr(gz, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&h2d_gz, cuda_graph, 0, 0, &h2dp), "h2d_gz failed");
      
          // kernel
          cudaKernelNodeParams kp;
          void* arguments[4] = { (void*)(&gx), (void*)(&gy), (void*)(&gz), (void*)(&N) };
          kp.func = (void*)add<int>;
          kp.gridDim = (N+255)/256;
          kp.blockDim = 256;
          kp.sharedMemBytes = 0;
          kp.kernelParams = arguments;
          kp.extra = nullptr;
          
          cudaGraphNode_t kernel;
          TF_CHECK_CUDA(cudaGraphAddKernelNode(&kernel, cuda_graph, 0, 0, &kp), "kernel failed");
          
          // d2hp
          cudaMemcpy3DParms d2hp;
          d2hp.srcArray = nullptr;
          d2hp.srcPos = ::make_cudaPos(0, 0, 0);
          d2hp.dstArray = nullptr;
          d2hp.dstPos = ::make_cudaPos(0, 0, 0);
          d2hp.extent = ::make_cudaExtent(N*sizeof(int), 1, 1);
          d2hp.kind = cudaMemcpyDefault;
          
          // d2h_gx
          cudaGraphNode_t d2h_gx;
          d2hp.srcPtr = ::make_cudaPitchedPtr(gx, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_cudaPitchedPtr(cx, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&d2h_gx, cuda_graph, 0, 0, &d2hp), "d2h_gx failed");
          
          // d2h_gy
          cudaGraphNode_t d2h_gy;
          d2hp.srcPtr = ::make_cudaPitchedPtr(gy, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_cudaPitchedPtr(cy, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&d2h_gy, cuda_graph, 0, 0, &d2hp), "d2h_gy failed");
          
          // d2h_gz
          cudaGraphNode_t d2h_gz;
          d2hp.srcPtr = ::make_cudaPitchedPtr(gz, N*sizeof(int), N, 1);
          d2hp.dstPtr = ::make_cudaPitchedPtr(cz, N*sizeof(int), N, 1);
          TF_CHECK_CUDA(cudaGraphAddMemcpyNode(&d2h_gz, cuda_graph, 0, 0, &d2hp), "d2h_gz failed");
      
          // add dependency
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &sgx, &h2d_gx, 1), "sgx->h2d_gx");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &sgy, &h2d_gy, 1), "sgy->h2d_gy");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &sgz, &h2d_gz, 1), "sgz->h2d_gz");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &h2d_gx, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &h2d_gy, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &h2d_gz, &kernel, 1), "h2d_gz->kernel");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &kernel, &d2h_gx, 1), "kernel->d2h_gx");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &kernel, &d2h_gy, 1), "kernel->d2h_gy");
          TF_CHECK_CUDA(cudaGraphAddDependencies(cuda_graph, &kernel, &d2h_gz, 1), "kernel->d2h_gz");
          
          // launch the graph
          cudaGraphExec_t exe;

          auto pts = streams.per_thread_stream(tgt_device);

          TF_CHECK_CUDA(cudaGraphInstantiate(&exe, cuda_graph, 0, 0, 0), "inst failed");
          TF_CHECK_CUDA(cudaGraphLaunch(exe, pts), "failed to launch cudaGraph");
          TF_CHECK_CUDA(cudaStreamSynchronize(pts), "failed to sync cudaStream");
          TF_CHECK_CUDA(cudaGraphExecDestroy(exe), "destroy exe failed");
          TF_CHECK_CUDA(cudaGraphDestroy(cuda_graph), "cudaGraphDestroy failed");
          TF_CHECK_CUDA(cudaSetDevice(src_device), "set device failed");
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

