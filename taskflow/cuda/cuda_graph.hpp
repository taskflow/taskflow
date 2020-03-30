#pragma once

#include "cuda_device.hpp"

#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"
#include "../nstd/variant.hpp"
#include "../nstd/optional.hpp"

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

  
  // Noop handle
  struct Noop {

    template <typename C>
    Noop(C&&);

    std::function<void(cudaGraph_t&, cudaGraphNode_t&)> work;
  };

  //// Host handle
  //struct Host {

  //  template <typename C>
  //  Host(C&&);
  //  
  //  std::function<void(cudaGraph_t&, cudaGraphNode_t&)> work;
  //};

  // Memset handle
  struct Memset {
    
    template <typename C>
    Memset(C&&);

    std::function<void(cudaGraph_t&, cudaGraphNode_t&)> work;
  };

  // Copy handle
  struct Copy {
    
    template <typename C>
    Copy(C&&);

    std::function<void(cudaGraph_t&, cudaGraphNode_t&)> work;
  };
  
  // Kernel handle
  struct Kernel {
    
    template <typename C>
    Kernel(C&&);

    std::function<void(cudaGraph_t&, cudaGraphNode_t&)> work;
  };

  using handle_t = nstd::variant<
    nstd::monostate, 
    Noop, 
    //Host, 
    Memset, 
    Copy, 
    Kernel
  >;
  
  // variant index
  constexpr static auto NOOP   = get_index_v<Noop, handle_t>;
  //constexpr static auto HOST   = get_index_v<Host, handle_t>;
  constexpr static auto MEMSET = get_index_v<Memset, handle_t>;
  constexpr static auto COPY   = get_index_v<Copy, handle_t>; 
  constexpr static auto KERNEL = get_index_v<Kernel, handle_t>;

  public:
    
    template <typename... ArgsT>
    cudaNode(ArgsT&&...);

  private:

    std::string _name;
    
    handle_t _handle;

    cudaGraphNode_t _native_handle {nullptr};

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

    ~cudaGraph();

    template <typename... ArgsT>
    cudaNode* emplace_back(ArgsT&&...);

    cudaGraph_t native_handle();

    void clear();

    bool empty() const;

  private:
    
    cudaGraph_t _native_handle {nullptr};

    std::vector<std::unique_ptr<cudaNode>> _nodes;

    void _make_native_graph();
};

// ----------------------------------------------------------------------------
// cudaNode definitions
// ----------------------------------------------------------------------------

//// Host handle constructor
//template <typename C>
//cudaNode::Host::Host(C&& c) : work {std::forward<C>(c)} {
//}

// Noop handle constructor
template <typename C>
cudaNode::Noop::Noop(C&& c) : work {std::forward<C>(c)} {
}

// Memset handle constructor
template <typename C>
cudaNode::Memset::Memset(C&& c) : work {std::forward<C>(c)} {
}

// Copy handle constructor
template <typename C>
cudaNode::Copy::Copy(C&& c) : work {std::forward<C>(c)} {
}

// Kernel handle constructor
template <typename C>
cudaNode::Kernel::Kernel(C&& c) : work {std::forward<C>(c)} {
}

// Constructor
template <typename... ArgsT>
cudaNode::cudaNode(ArgsT&&... args) : _handle {std::forward<ArgsT>(args)...} {
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
  if(_native_handle) {
    cudaGraphDestroy(_native_handle);
  }
}

// Function: empty
inline bool cudaGraph::empty() const {
  return _nodes.empty();
}

// Procedure: clear
inline void cudaGraph::clear() {

  _nodes.clear();

  if(_native_handle) {
    TF_CHECK_CUDA(
      cudaGraphDestroy(_native_handle), "failed to destroy a cudaGraph on clear"
    );
    _native_handle = nullptr;
  }
}
    
// Function: emplace_back
template <typename... ArgsT>
cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
  auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
  _nodes.emplace_back(std::move(node));
  return _nodes.back().get();
}

// Function: native_handle
inline cudaGraph_t cudaGraph::native_handle() {
  return _native_handle;
}

// Procedure: _make_native_graph
inline void cudaGraph::_make_native_graph() {

  //// TODO: must be nullptr
  //if(_native_handle) {
  //  TF_CHECK_CUDA(
  //    cudaGraphDestroy(_native_handle), "failed to destroy the previous cudaGraph"
  //  );
  //  _native_handle = nullptr;
  //}
  //
  //cudaScopedDevice ctx {d};
  assert(_native_handle == nullptr);

  TF_CHECK_CUDA(
    cudaGraphCreate(&_native_handle, 0), "failed to create a cudaGraph"
  );

  // create nodes
  for(auto& node : _nodes) {
    switch(node->_handle.index()) {
      case cudaNode::NOOP:
        nstd::get<cudaNode::Noop>(node->_handle).work(
          _native_handle, node->_native_handle
        );
      break;
      
      //case cudaNode::HOST:
      //  nstd::get<cudaNode::Host>(node->_handle).work(
      //    _native_handle, node->_native_handle
      //  );
      //break;

      case cudaNode::MEMSET:
        nstd::get<cudaNode::Memset>(node->_handle).work(
          _native_handle, node->_native_handle
        );
      break;

      case cudaNode::COPY:
        nstd::get<cudaNode::Copy>(node->_handle).work(
          _native_handle, node->_native_handle
        );
      break;

      case cudaNode::KERNEL:
        nstd::get<cudaNode::Kernel>(node->_handle).work(
          _native_handle, node->_native_handle
        );
      break;
    }
  }

  // create edges
  for(auto& node : _nodes) {
    for(auto succ : node->_successors){
      TF_CHECK_CUDA(
        ::cudaGraphAddDependencies(
          _native_handle, &(node->_native_handle), &(succ->_native_handle), 1
        ),
        "failed to add a preceding link"
      );
    }
  }

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

