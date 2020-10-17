#pragma once

#include "cuda_memory.hpp"

#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"

namespace tf {

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
    
    cudaGraph() = default;
    ~cudaGraph();

    cudaGraph(const cudaGraph&) = delete;
    cudaGraph(cudaGraph&&);
    
    cudaGraph& operator = (const cudaGraph&) = delete;
    cudaGraph& operator = (cudaGraph&&);

    template <typename... ArgsT>
    cudaNode* emplace_back(ArgsT&&...);

    void clear();

    bool empty() const;

  private:

    cudaGraph_t _native_handle {nullptr};
    
    // TODO: nvcc complains deleter of unique_ptr
    //std::vector<std::unique_ptr<cudaNode>> _nodes;
    std::vector<cudaNode*> _nodes;

    void _create_native_graph();
    void _destroy_native_graph();
};

// ----------------------------------------------------------------------------
// cudaNode class
// ----------------------------------------------------------------------------

// class: cudaNode
// each create_native_node is wrapped in a function to call at runtime 
// in order to work with gpu context
class cudaNode {
  
  friend class cudaFlow;
  friend class cudaGraph;
  friend class cudaTask;

  friend class Taskflow;
  friend class Executor;
  
  // Noop handle
  struct Noop {
  };

  //// Host handle
  //struct Host {

  //  template <typename C>
  //  Host(C&&);

  //  std::function<void(cudaGraph_t&, cudaGraphNode_t&)> create_native_node;
  //};

  // Memset handle
  struct Memset {
  };

  // Memcpy handle
  struct Memcpy {
  };
  
  // Kernel handle
  struct Kernel {

    Kernel(void*);
    
    void* func {nullptr};
  };

  // Subflow handle
  struct Subflow {
    cudaGraph graph;
  };

  // Capture
  struct Capture {
  };

  using handle_t = std::variant<
    Noop, 
    Memset, 
    Memcpy, 
    Kernel,
    Subflow,
    Capture
  >;

  public:
  
  // variant index
  constexpr static auto CUDA_NOOP_TASK    = get_index_v<Noop, handle_t>;
  constexpr static auto CUDA_MEMSET_TASK  = get_index_v<Memset, handle_t>;
  constexpr static auto CUDA_MEMCPY_TASK  = get_index_v<Memcpy, handle_t>; 
  constexpr static auto CUDA_KERNEL_TASK  = get_index_v<Kernel, handle_t>;
  constexpr static auto CUDA_SUBFLOW_TASK = get_index_v<Subflow, handle_t>;
  constexpr static auto CUDA_CAPTURE_TASK = get_index_v<Capture, handle_t>;
    
    template <typename... ArgsT>
    cudaNode(cudaGraph&, ArgsT&&...);

  private:

    cudaGraph& _graph;

    std::string _name;
    
    handle_t _handle;

    cudaGraphNode_t _native_handle {nullptr};

    std::vector<cudaNode*> _successors;

    void _precede(cudaNode*);
};

// ----------------------------------------------------------------------------
// cudaNode definitions
// ----------------------------------------------------------------------------

//// Host handle constructor
//template <typename C>
//cudaNode::Host::Host(C&& c) : create_native_node {std::forward<C>(c)} {
//}

// Kernel handle constructor
inline cudaNode::Kernel::Kernel(void* ptr) : 
  func {ptr} {
}

// Constructor
template <typename... ArgsT>
cudaNode::cudaNode(cudaGraph& graph, ArgsT&&... args) : 
  _graph {graph},
  _handle {std::forward<ArgsT>(args)...} {
}

// Procedure: _precede
inline void cudaNode::_precede(cudaNode* v) {
  _successors.push_back(v);
  TF_CHECK_CUDA(
    ::cudaGraphAddDependencies(
      _graph._native_handle, &_native_handle, &v->_native_handle, 1
    ),
    "failed to add a preceding link ", this, "->", v
  );
}

// ----------------------------------------------------------------------------
// cudaGraph definitions
// ----------------------------------------------------------------------------

// Destructor
inline cudaGraph::~cudaGraph() {
  clear();
  assert(_native_handle == nullptr);
}

// Move constructor
inline cudaGraph::cudaGraph(cudaGraph&& g) :
  _native_handle {g._native_handle},
  _nodes         {std::move(g._nodes)} {
  
  g._native_handle = nullptr;

  assert(g._nodes.empty());
}

// Move assignment
inline cudaGraph& cudaGraph::operator = (cudaGraph&& rhs) {

  clear();
  
  // lhs
  _native_handle = rhs._native_handle;
  _nodes = std::move(rhs._nodes);

  assert(rhs._nodes.empty());

  // rhs
  rhs._native_handle = nullptr;

  return *this; 
}

// Function: empty
inline bool cudaGraph::empty() const {
  return _nodes.empty();
}

// Procedure: clear
inline void cudaGraph::clear() {
  for(auto n : _nodes) {
    delete n;
  }
  _nodes.clear();
}

// Procedure: clear the cudaGraph
inline void cudaGraph::_destroy_native_graph() {
  assert(_native_handle != nullptr);
  TF_CHECK_CUDA(
    cudaGraphDestroy(_native_handle), 
    "failed to destroy the native graph"
  );
  _native_handle = nullptr;
}
    
// Function: emplace_back
template <typename... ArgsT>
cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
  //auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
  //_nodes.emplace_back(std::move(node));
  //return _nodes.back().get();
  // TODO: object pool

  auto node = new cudaNode(*this, std::forward<ArgsT>(args)...);
  _nodes.push_back(node);
  return node;
}

// Procedure: _create_native_graph
inline void cudaGraph::_create_native_graph() {
  assert(_native_handle == nullptr);
  TF_CHECK_CUDA(
    cudaGraphCreate(&_native_handle, 0), 
    "failed to create a native graph"
  );
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

