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
        ++counter;
        int tgt_device = g.nodes[u].g;
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
        cudaStream_t pts = streams.per_thread_stream(tgt_device);

        cudaGraphExec_t exe;
        TF_CHECK_CUDA(cudaGraphInstantiate(&exe, cuda_graph, 0, 0, 0), "inst failed");
        TF_CHECK_CUDA(cudaGraphLaunch(exe, pts), "failed to launch cudaGraph");
        TF_CHECK_CUDA(cudaStreamSynchronize(pts), "failed to sync cudaStream");
        TF_CHECK_CUDA(cudaGraphExecDestroy(exe), "destroy exe failed");
        TF_CHECK_CUDA(cudaGraphDestroy(cuda_graph), "cudaGraphDestroy failed");
        TF_CHECK_CUDA(cudaSetDevice(src_device), "set device failed");
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
