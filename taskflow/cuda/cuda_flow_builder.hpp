#pragma once

#include "cuda_task.hpp"

namespace tf {

/**
@class cudaFlow

@brief Building methods of a cuda task dependency graph.
*/
class cudaFlow {

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
    cudaTask copy(T* tgt, T* src, size_t num);

  private:

    cudaGraph& _graph;
};

// Constructor
inline cudaFlow::cudaFlow(cudaGraph& g) : _graph {g} {
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Function: noop
inline cudaTask cudaFlow::noop() {
  auto node = _graph.emplace_back();
  node->_handle.emplace<cudaNode::Noop>();
  TF_CHECK_CUDA(
    ::cudaGraphAddEmptyNode(&node->_node, _graph._handle, nullptr, 0),
    "failed to create a no-operation (empty) node"
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

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };

  auto node = _graph.emplace_back();

  auto& p = node->_handle.emplace<cudaNode::Kernel>().param;

  p.func = (void*)func;
  p.gridDim = grid;
  p.blockDim = block;
  p.sharedMemBytes = shm;
  p.kernelParams = arguments;
  p.extra = nullptr;
  
  TF_CHECK_CUDA(
    ::cudaGraphAddKernelNode(&node->_node, _graph._handle, nullptr, 0, &p),
    "failed to create a cudaKernel node"
  );
  
  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same<T, void>::value, void>*
>
cudaTask cudaFlow::copy(T* tgt, T* src, size_t num) {

  using U = std::decay_t<T>;

  auto node = _graph.emplace_back();
  auto& p = node->_handle.emplace<cudaNode::Copy>().param;

  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(src, num*sizeof(U), num, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
  p.kind = cudaMemcpyDefault;

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node->_node, _graph._handle, nullptr, 0, &p),
    "failed to create a cudaCopy node"
  );

  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------
