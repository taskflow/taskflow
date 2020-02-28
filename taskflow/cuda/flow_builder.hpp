#pragma once

#include "task.hpp"

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
    @brief constructs a kernel task
    
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
    @brief constructs an 1D copy task
    
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

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args) {

  void* arguments[sizeof...(ArgsT)] = { &args... };

  auto node = _graph.emplace_back();

  auto& param = node->_handle.emplace<cudaNode::Kernel>().param;

  param.func = (void*)func;
  param.gridDim = grid;
  param.blockDim = block;
  param.sharedMemBytes = shm;
  param.kernelParams = arguments;
  param.extra = nullptr;
  
  TF_CHECK_CUDA(
    ::cudaGraphAddKernelNode(&node->_node, _graph._handle, nullptr, 0, &param),
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

  auto node = _graph.emplace_back();
  auto& param = node->_handle.emplace<cudaNode::Copy>().param;

  param.srcArray = nullptr;
  param.srcPos = ::make_cudaPos(0, 0, 0);
  param.srcPtr = ::make_cudaPitchedPtr(src, num*sizeof(T), num, 1);
  param.dstArray = nullptr;
  param.dstPos = ::make_cudaPos(0, 0, 0);
  param.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(T), num, 1);
  param.extent = ::make_cudaExtent(num*sizeof(T), 1, 1);
  param.kind = cudaMemcpyDefault;

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node->_node, _graph._handle, nullptr, 0, &param),
    "failed to create a cudaCopy node"
  );

  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------
