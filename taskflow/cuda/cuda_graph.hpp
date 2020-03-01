#pragma once

#include "cuda_device.hpp"

#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"
#include "../nstd/variant.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaNode class
// ----------------------------------------------------------------------------

// class: cudaNode
class cudaNode {
  
  friend class cudaFlow;
  friend class cudaGraph;
  friend class cudaTask;

  friend class Taskflow;
  friend class Executor;

  // Host handle
  //struct Host {
  //  cudaHostNodeParams param;
  //};

  struct Noop {

  };

  // Copy handle
  struct Copy {
    
    template <typename... ArgsT>
    Copy(ArgsT&&...);

    cudaMemcpy3DParms param;
  };
  
  // Kernel handle
  struct Kernel {
    
    template <typename... ArgsT>
    Kernel(ArgsT&&...);

    cudaKernelNodeParams param;
  };

  using handle_t = nstd::variant<nstd::monostate, Noop, Copy, Kernel>;
  
  // variant index
  constexpr static auto NOOP = get_index_v<Noop, handle_t>;
  constexpr static auto COPY = get_index_v<Copy, handle_t>; 
  constexpr static auto KERNEL = get_index_v<Kernel, handle_t>;

  public:
  
    cudaNode(cudaGraph&);

  private:

    cudaGraph& _graph;

    std::string _name;
    
    handle_t _handle;

    cudaGraphNode_t _node {nullptr};

    PassiveVector<cudaNode*> _successors;

    void _precede(cudaNode*);
};

// ----------------------------------------------------------------------------
// cudaGraph class
// ----------------------------------------------------------------------------

// class: cudaGraph
class cudaGraph {

  friend class cudaFlow;
  friend class cudaNode;
  friend class cudaTask;
  
  friend class Taskflow;
  friend class Executor;

  public:

    cudaGraph();
    ~cudaGraph();

    template <typename... ArgsT>
    cudaNode* emplace_back(ArgsT&&...);

    cudaGraph_t native_handle();

    void clear();

    bool empty() const;

  private:
    
    cudaGraph_t _handle {nullptr};

    std::vector<std::unique_ptr<cudaNode>> _nodes;
};

// ----------------------------------------------------------------------------
// cudaNode definitions
// ----------------------------------------------------------------------------

// Copy handle constructor
template <typename... ArgsT>
cudaNode::Copy::Copy(ArgsT&&... args) {
}

// Kernel handle constructor
template <typename... ArgsT>
cudaNode::Kernel::Kernel(ArgsT&&... args) {
}

// Constructor
inline cudaNode::cudaNode(cudaGraph& g) : _graph {g} {
}

// Procedure: _precede
inline void cudaNode::_precede(cudaNode* v) {
  _successors.push_back(v);
  TF_CHECK_CUDA(
    ::cudaGraphAddDependencies(_graph._handle, &_node, &(v->_node), 1),
    "failed to add a preceding link"
  );
}

// ----------------------------------------------------------------------------
// cudaGraph definitions
// ----------------------------------------------------------------------------

// Constructor
inline cudaGraph::cudaGraph() {
  TF_CHECK_CUDA(cudaGraphCreate(&_handle, 0), "failed to create a cudaGraph");
}

// Destructor
inline cudaGraph::~cudaGraph() {
  cudaGraphDestroy(_handle);
}

// Function: empty
inline bool cudaGraph::empty() const {
  return _nodes.empty();
}

// Procedure: clear
inline void cudaGraph::clear() {

  _nodes.clear();

  cudaGraphDestroy(_handle);
  TF_CHECK_CUDA(
    cudaGraphCreate(&_handle, 0), 
    "failed to create a cudaGraph after clear"
  );
}
    
// Function: emplace_back
template <typename... ArgsT>
cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
  auto node = std::make_unique<cudaNode>(*this, std::forward<ArgsT>(args)...);
  _nodes.emplace_back(std::move(node));
  return _nodes.back().get();
}

// Function: native_handle
inline cudaGraph_t cudaGraph::native_handle() {
  return _handle;
}


//inline void cudaGraph::run() {
//  cudaGraphExec_t graphExec;
//  TF_CHECK_CUDA(
//    cudaGraphInstantiate(&graphExec, _handle, nullptr, nullptr, 0),
//    "failed to create an executable cudaGraph"
//  );
//  TF_CHECK_CUDA(cudaGraphLaunch(graphExec, 0), "failed to launch cudaGraph")
//  TF_CHECK_CUDA(cudaStreamSynchronize(0), "failed to sync cudaStream");
//  TF_CHECK_CUDA(
//    cudaGraphExecDestroy(graphExec), "failed to destroy an executable cudaGraph"
//  );
//}





}  // end of namespace tf -----------------------------------------------------

