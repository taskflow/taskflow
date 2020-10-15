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

  struct NativeHandle {
    cudaGraph_t graph {nullptr};
    cudaGraphExec_t image {nullptr};
  };

  public:
    
    cudaGraph() = default;
    ~cudaGraph();

    cudaGraph(const cudaGraph&) = delete;
    cudaGraph(cudaGraph&&);
    
    cudaGraph& operator = (const cudaGraph&);
    cudaGraph& operator = (cudaGraph&&);

    template <typename... ArgsT>
    cudaNode* emplace_back(ArgsT&&...);

    void clear();

    bool empty() const;

  private:
    
    NativeHandle _native_handle;
    
    // TODO: nvcc complains deleter of unique_ptr
    //std::vector<std::unique_ptr<cudaNode>> _nodes;
    std::vector<cudaNode*> _nodes;

    void _create_native_graph();
    void _destroy_native_graph();
    //void _update_native_graph();
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

    //template <typename C>
    //Noop(C&&);
  };

  // Host handle
  struct Host {

    //template <typename C>
    //Host(C&);
  };

  // Memset handle
  struct Memset {
    
    //template <typename C>
    //Memset(C&&);
  };

  // Copy handle
  struct Copy {
    
    //template <typename C>
    //Copy(C&&);
  };
  
  // Kernel handle
  struct Kernel {
    template <typename C>
    Kernel(C&& c) : func{ std::forward<C>(c) } {}
    
    void* func {nullptr};
  };

  // BLAS handle
  //struct BLAS {
  //  cudaGraph_t graph {nullptr};
  //};

  using handle_t = std::variant<
    std::monostate, 
    Noop, 
    Host, 
    Memset, 
    Copy, 
    Kernel
  >;

  public:
  
  // variant index
  constexpr static auto CUDA_NOOP_TASK   = get_index_v<Noop, handle_t>;
  constexpr static auto CUDA_HOST_TASK = get_index_v<Host, handle_t>;
  constexpr static auto CUDA_MEMSET_TASK = get_index_v<Memset, handle_t>;
  constexpr static auto CUDA_MEMCPY_TASK   = get_index_v<Copy, handle_t>; 
  constexpr static auto CUDA_KERNEL_TASK = get_index_v<Kernel, handle_t>;
    
    template <typename C, typename... ArgsT>
    cudaNode(C&&, ArgsT&&...);

  private:

    std::string _name;
    
    std::function<void(cudaGraph_t&, cudaGraphNode_t&)> _create_native_node;
    
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

/*// Noop handle constructor
template <typename C>
cudaNode::Noop::Noop(C&&) : create_native_node {std::forward<C>(c)} {
}

// Memset handle constructor
template <typename C>
cudaNode::Memset::Memset(C&& c) : create_native_node {std::forward<C>(c)} {
}

// Copy handle constructor
template <typename C>
cudaNode::Copy::Copy(C&& c) : create_native_node {std::forward<C>(c)} {
}

// Kernel handle constructor
template <typename C>
cudaNode::Kernel::Kernel(C&&) : create_native_node {std::forward<C>(c)} {
}*/

// Constructor
template <typename C, typename... ArgsT>
cudaNode::cudaNode(C&& c, ArgsT&&... args) : 
  _create_native_node {std::forward<C>(c)},
  _handle             {std::forward<ArgsT>(args)...} {
}

// Procedure: _precede
inline void cudaNode::_precede(cudaNode* v) {
  _successors.push_back(v);
}

// ----------------------------------------------------------------------------
// cudaGraph definitions
// ----------------------------------------------------------------------------

// Destructor
inline cudaGraph::~cudaGraph() {
  clear();
}

// Move constructor
inline cudaGraph::cudaGraph(cudaGraph&& g) :
  _native_handle {g._native_handle},
  _nodes         {std::move(g._nodes)} {
  
  g._native_handle.graph = nullptr;
  g._native_handle.image = nullptr;    

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
  rhs._native_handle.graph = nullptr;
  rhs._native_handle.image = nullptr;

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
  _destroy_native_graph();
}

// Procedure: clear the cudaGraph
inline void cudaGraph::_destroy_native_graph() {
  if(_native_handle.graph) {
    TF_CHECK_CUDA(
      cudaGraphExecDestroy(_native_handle.image), 
      "failed to destroy the native image"
    );
    _native_handle.image = nullptr;

    TF_CHECK_CUDA(
      cudaGraphDestroy(_native_handle.graph), 
      "failed to destroy the native graph"
    );
    _native_handle.graph = nullptr;
  }
  assert(_native_handle.image == nullptr);
}
    
// Function: emplace_back
template <typename... ArgsT>
cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
  //auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
  //_nodes.emplace_back(std::move(node));
  //return _nodes.back().get();
  
  assert(_native_handle.graph == nullptr);

  // TODO: object pool

  auto node = new cudaNode(std::forward<ArgsT>(args)...);
  _nodes.push_back(node);
  return node;
}

// Procedure: _create_native_graph
inline void cudaGraph::_create_native_graph() {

  assert(_native_handle.graph == nullptr);

  TF_CHECK_CUDA(
    cudaGraphCreate(&_native_handle.graph, 0), 
    "failed to create a native graph"
  );

  // create nodes
  for(auto& node : _nodes) {
    assert(node->_native_handle == nullptr);
    node->_create_native_node(_native_handle.graph, node->_native_handle);
  }

  // create edges
  for(auto& node : _nodes) {
    for(auto succ : node->_successors){
      TF_CHECK_CUDA(
        ::cudaGraphAddDependencies(
          _native_handle.graph, &(node->_native_handle), &(succ->_native_handle), 1
        ),
        "failed to add a preceding link"
      );
    }
  }
  
  // create the executable handle
  TF_CHECK_CUDA(
    cudaGraphInstantiate(
      &_native_handle.image, _native_handle.graph, nullptr, nullptr, 0
    ),
    "failed to create an executable cudaGraph"
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

