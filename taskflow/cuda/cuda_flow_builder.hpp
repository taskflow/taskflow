#pragma once

#include "cuda_task.hpp"

namespace tf {

/**
@class cudaFlow

@brief Building methods of a cuda task dependency graph.
*/
class cudaFlow {

  friend class Executor;

  public:
    
    /**
    @brief constructs a cudaFlow builder object

    @param graph a cudaGraph to manipulate
    */
    cudaFlow(cudaGraph& graph);

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;
    
    /**
    @brief creates a placeholder task
    */
    cudaTask placeholder();

    /**
    @brief creates a no-operation task

    An empty node performs no operation during execution, 
    but can be used for transitive ordering. 
    For example, a phased execution graph with 2 groups of n nodes 
    with a barrier between them can be represented using an empty node 
    and 2*n dependency edges, 
    rather than no empty node and n^2 dependency edges.
    */
    cudaTask noop();
    
    /**
    @brief creates a kernel task
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
    
    /**
    @brief creates a kernel task on a device
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type
    
    @param d device identifier to luanch the kernel
    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
    
    /**
    @brief creates an 1D copy task
    
    @tparam T element type (non-void)

    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param num number of elements to copy

    @return cudaTask handle

    A copy task transfers num*sizeof(T) bytes of data from a source location
    to a target location. Direction can be either cpu-to-gpu, gpu-to-cpu,
    gpu-to-gpu, or gpu-to-cpu.
    */
    template <
      typename T, 
      std::enable_if_t<!std::is_same<T, void>::value, void>* = nullptr
    >
    cudaTask copy(T* tgt, const T* src, size_t num);

    /**
    @brief assigns a device to launch the cudaFlow

    @param device target device identifier
    */
    void device(int device);

    /**
    @brief queries the device associated with the cudaFlow
    */
    int device() const;

    /**
    @brief assigns a stream to launch the cudaFlow

    @param stream target stream identifier
    */
    void stream(cudaStream_t stream);

    /**
    @brief queries the stream associated with the cudaFlow
    */
    cudaStream_t stream() const;

  private:

    cudaGraph& _graph;
    
    int _device {0};

    cudaStream_t _stream {nullptr};
};

// Constructor
inline cudaFlow::cudaFlow(cudaGraph& g) : _graph {g} {
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Procedure: device
inline void cudaFlow::device(int d) {
  _device = d;
}

// Function: device
inline int cudaFlow::device() const {
  return _device;
}

// Procedure: stream
inline void cudaFlow::stream(cudaStream_t s) {
  _stream = s;
}

// Function: stream
inline cudaStream_t cudaFlow::stream() const {
  return _stream;
}

// Function: noop
inline cudaTask cudaFlow::noop() {
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Noop>{}, 
    [](cudaGraph_t& graph, cudaGraphNode_t& node){
      TF_CHECK_CUDA(
        ::cudaGraphAddEmptyNode(&node, graph, nullptr, 0),
        "failed to create a no-operation (empty) node"
      );
    }
  );
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
  
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Kernel>{}, 
    [=] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaKernelNodeParams p;
      void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
      p.func = (void*)func;
      p.gridDim = grid;
      p.blockDim = block;
      p.sharedMemBytes = shm;
      p.kernelParams = arguments;
      p.extra = nullptr;

      TF_CHECK_CUDA(
        ::cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaKernel node"
      );
    }
  );
  
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  int dev, dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
  
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Kernel>{}, 
    [=] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaKernelNodeParams p;
      void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
      p.func = (void*)func;
      p.gridDim = grid;
      p.blockDim = block;
      p.sharedMemBytes = shm;
      p.kernelParams = arguments;
      p.extra = nullptr;

      cudaScopedDevice ctx(dev);
      TF_CHECK_CUDA(
        ::cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaKernel node"
      );
    }
  );
  
  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same<T, void>::value, void>*
>
cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Copy>{}, 
    [=] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaMemcpy3DParms p;
      p.srcArray = nullptr;
      p.srcPos = ::make_cudaPos(0, 0, 0);
      p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
      p.dstArray = nullptr;
      p.dstPos = ::make_cudaPos(0, 0, 0);
      p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
      p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
      p.kind = cudaMemcpyDefault;

      TF_CHECK_CUDA(
        cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaCopy node"
      );
    }
  );

  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------
