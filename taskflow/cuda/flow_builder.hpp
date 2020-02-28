#pragma once

#include "task.hpp"

namespace tf {

class cudaFlow {

  public:
    
    cudaFlow(cudaGraph&);

    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args);

    cudaTask copy(void* tgt, void* src, size_t num, size_t size);

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
inline cudaTask cudaFlow::copy(void* tgt, void* src, size_t num, size_t size) {

  auto node = _graph.emplace_back();
  auto& param = node->_handle.emplace<cudaNode::Copy>().param;

  param.srcArray = nullptr;
  param.srcPos = ::make_cudaPos(0, 0, 0);
  param.srcPtr = ::make_cudaPitchedPtr(src, num*size, num, 1);
  param.dstArray = nullptr;
  param.dstPos = ::make_cudaPos(0, 0, 0);
  param.dstPtr = ::make_cudaPitchedPtr(tgt, num*size, num, 1);
  param.extent = ::make_cudaExtent(num*size, 1, 1);
  param.kind = cudaMemcpyDefault;

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node->_node, _graph._handle, nullptr, 0, &param),
    "failed to create a cudaCopy node"
  );

  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------
