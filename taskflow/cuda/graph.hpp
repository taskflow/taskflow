#pragma once

#include "device.hpp"

#include "../declarations.hpp"
#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"
#include "../nstd/variant.hpp"

namespace tf {

class cudaNode {
  
  friend class cudaGraph;

  public:
  
  //struct Host {
  //  cudaHostNodeParams param = {0};
  //  // need an any storage?
  //};

  struct Copy {
    
    template <typename... ArgsT>
    Copy(ArgsT&&... args) {
    }

    cudaMemcpy3DParms param = {0};
  };

  struct Kernel {
    
    template <typename... ArgsT>
    Kernel(ArgsT&&...) {}

    cudaKernelNodeParams param = {0};
    // need an any storage?
  };

  cudaNode() = default;

  private:

    std::string _name;
    
    nstd::variant<nstd::monostate, Copy, Kernel> _handle;

    cudaGraphNode_t _node {nullptr};

    PassiveVector<cudaNode*> _successors;

};

// class: cudaGraph
class cudaGraph {

  public:

    cudaGraph();
    ~cudaGraph();

    void placeholder() {
      auto node = std::make_unique<cudaNode>();
    }
    
    template <typename F, typename... ArgsT>
    cudaNode* kernel(dim3 grid, dim3 block, size_t shm, F&& func, ArgsT&&... args) {

      void* arguments[sizeof...(ArgsT)] = { &args... };

      auto node = std::make_unique<cudaNode>();
      auto& p = node->_handle.emplace<cudaNode::Kernel>();

      p.param.func = (void*)func;
      p.param.gridDim = grid;
      p.param.blockDim = block;
      p.param.sharedMemBytes = shm;
      p.param.kernelParams = arguments;
      p.param.extra = nullptr;
      
      TF_CHECK_CUDA(
        ::cudaGraphAddKernelNode(&node->_node, _graph, nullptr, 0, &p.param),
        "failed to create a kernel node"
      );

      return node.get();
    }

    cudaNode* copy(void* tgt, void* src, size_t num, size_t size) {
      auto node = std::make_unique<cudaNode>();
      auto& p = node->_handle.emplace<cudaNode::Copy>();
      p.param.srcArray = nullptr;
      p.param.srcPos = ::make_cudaPos(0, 0, 0);
      p.param.srcPtr = ::make_cudaPitchedPtr(src, num*size, num, 1);
      p.param.dstArray = nullptr;
      p.param.dstPos = ::make_cudaPos(0, 0, 0);
      p.param.dstPtr = ::make_cudaPitchedPtr(tgt, num*size, num, 1);
      p.param.extent = ::make_cudaExtent(num*size, 1, 1);
      p.param.kind = cudaMemcpyDefault;

      TF_CHECK_CUDA(
        ::cudaGraphAddMemcpyNode(&node->_node, _graph, nullptr, 0, &p.param),
        "failed to create a memcpy node"
      );

      return node.get();
    }

    void precede(cudaNode* u, cudaNode* v) {
      TF_CHECK_CUDA(
        ::cudaGraphAddDependencies(_graph, {&(u->_node)}, {&(v->_node)}, 1),
        "failed to add a preceding link"
      );
    }

    void run() {
      cudaGraphExec_t graphExec;
      cudaGraphInstantiate(&graphExec, _graph, nullptr, nullptr, 0);
      cudaGraphLaunch(graphExec, 0);
      cudaStreamSynchronize(0);
    }

  private:
    
    cudaGraph_t _graph {nullptr};

    std::vector<std::unique_ptr<cudaNode>> _nodes;
};

// Constructor
inline cudaGraph::cudaGraph() {
  TF_CHECK_CUDA(cudaGraphCreate(&_graph, 0), "failed to create a cudaGraph");
}

// Destructor
inline cudaGraph::~cudaGraph() {
  cudaGraphDestroy(_graph);
}



}  // end of namespace tf -----------------------------------------------------

