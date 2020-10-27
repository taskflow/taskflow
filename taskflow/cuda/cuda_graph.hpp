#pragma once

#include "cuda_memory.hpp"
#include "cuda_stream.hpp"

#include "../utility/object_pool.hpp"
#include "../utility/traits.hpp"
#include "../utility/passive_vector.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaGraph_t routines
// ----------------------------------------------------------------------------

/**
@brief queries the number of root nodes in a native CUDA graph
*/
inline size_t cuda_get_graph_num_root_nodes(cudaGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_CUDA(
    cudaGraphGetRootNodes(graph, nullptr, &num_nodes), 
    "failed to get native graph root nodes"
  );
  return num_nodes; 
}

/**
@brief queries the number of nodes in a native CUDA graph
*/
inline size_t cuda_get_graph_num_nodes(cudaGraph_t graph) {
  size_t num_nodes;
  TF_CHECK_CUDA(
    cudaGraphGetNodes(graph, nullptr, &num_nodes), 
    "failed to get native graph nodes"
  );
  return num_nodes; 
}

/**
@brief queries the number of edges in a native CUDA graph
*/
inline size_t cuda_get_graph_num_edges(cudaGraph_t graph) {
  size_t num_edges;
  TF_CHECK_CUDA(
    cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges), 
    "failed to get native graph edges"
  );
  return num_edges;
}

/**
@brief acquires the nodes in a native CUDA graph
*/
inline std::vector<cudaGraphNode_t> cuda_get_graph_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_get_graph_num_nodes(graph);
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  TF_CHECK_CUDA(
    cudaGraphGetNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the root nodes in a native CUDA graph
*/
inline std::vector<cudaGraphNode_t> cuda_get_graph_root_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_get_graph_num_root_nodes(graph);
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  TF_CHECK_CUDA(
    cudaGraphGetRootNodes(graph, nodes.data(), &num_nodes),
    "failed to get native graph nodes"
  );
  return nodes;
}

/**
@brief acquires the edges in a native CUDA graph
*/
inline std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>>
cuda_get_graph_edges(cudaGraph_t graph) {
  size_t num_edges = cuda_get_graph_num_edges(graph);
  std::vector<cudaGraphNode_t> froms(num_edges), tos(num_edges);
  TF_CHECK_CUDA(
    cudaGraphGetEdges(graph, froms.data(), tos.data(), &num_edges),
    "failed to get native graph edges"
  );
  std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges(num_edges);
  for(size_t i=0; i<num_edges; i++) {
    edges[i] = std::make_pair(froms[i], tos[i]);
  }
  return edges;
}

/**
@brief queries the type of a native CUDA graph node

valid type values are:
  + cudaGraphNodeTypeKernel      = 0x00
  + cudaGraphNodeTypeMemcpy      = 0x01
  + cudaGraphNodeTypeMemset      = 0x02
  + cudaGraphNodeTypeHost        = 0x03
  + cudaGraphNodeTypeGraph       = 0x04
  + cudaGraphNodeTypeEmpty       = 0x05
  + cudaGraphNodeTypeWaitEvent   = 0x06
  + cudaGraphNodeTypeEventRecord = 0x07
*/
inline cudaGraphNodeType cuda_get_graph_node_type(cudaGraphNode_t node) {
  cudaGraphNodeType type;
  TF_CHECK_CUDA(
    cudaGraphNodeGetType(node, &type), "failed to get native graph node type"
  );
  return type;
}

/**
@brief convert the type of a native CUDA graph node to a readable string
*/
inline const char* cuda_graph_node_type_to_string(cudaGraphNodeType type) {
  switch(type) {
    case cudaGraphNodeTypeKernel      : return "kernel";
    case cudaGraphNodeTypeMemcpy      : return "memcpy";
    case cudaGraphNodeTypeMemset      : return "memset";
    case cudaGraphNodeTypeHost        : return "host";
    case cudaGraphNodeTypeGraph       : return "subflow";
    case cudaGraphNodeTypeEmpty       : return "empty";
    case cudaGraphNodeTypeWaitEvent   : return "event_wait";
    case cudaGraphNodeTypeEventRecord : return "event_record";
    default                           : return "undefined";
  }
}

/**
@brief dumps a native CUDA graph and all associated child graphs to a DOT format

@tparam T output stream target
@param os target output stream
@param graph native CUDA graph
*/
template <typename T>
void cuda_dump_graph(T& os, cudaGraph_t graph) {
  
  os << "digraph cudaGraph {\n";

  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
  stack.push(std::make_tuple(graph, nullptr, 1));

  int pl = 0;

  while(stack.empty() == false) {

    auto [graph, parent, l] = stack.top();
    stack.pop();

    for(int i=0; i<pl-l+1; i++) {
      os << "}\n";
    }

    os << "subgraph cluster_p" << graph << " {\n"
       << "label=\"cudaGraph-L" << l << "\";\n"
       << "color=\"purple\";\n";

    auto nodes = cuda_get_graph_nodes(graph);
    auto edges = cuda_get_graph_edges(graph);

    for(auto& [from, to] : edges) {
      os << 'p' << from << " -> " << 'p' << to << ";\n";
    }
  
    for(auto& node : nodes) {
      auto type = cuda_get_graph_node_type(node);
      if(type == cudaGraphNodeTypeGraph) {

        cudaGraph_t graph;
        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &graph), "");
        stack.push(std::make_tuple(graph, node, l+1));

        os << 'p' << node << "["
           << "shape=folder, style=filled, fontcolor=white, fillcolor=purple, "
           << "label=\"cudaGraph-L" << l+1
           << "\"];\n";
      }
      else {
        os << 'p' << node << "[label=\"" 
           << cuda_graph_node_type_to_string(type) 
           << "\"];\n";
      }
    }

    // precede to parent
    if(parent != nullptr) {
      std::unordered_set<cudaGraphNode_t> successors;
      for(const auto& p : edges) {
        successors.insert(p.first);
      }
      for(auto node : nodes) {
        if(successors.find(node) == successors.end()) {
          os << 'p' << node << " -> " << 'p' << parent << ";\n";
        }
      }
    }
    
    // set the previous level
    pl = l;
  }

  for(int i=0; i<=pl; i++) {
    os << "}\n";
  }
}

// ----------------------------------------------------------------------------
// cudaGraph class
// ----------------------------------------------------------------------------

// class: cudaGraph
class cudaGraph {

  friend class cudaNode;
  friend class cudaTask;
  friend class cudaFlow;
  friend class cublasFlow;
  
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

    template <typename T>
    void dump(T&, const Node* = nullptr) const;

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
  
  friend class cudaGraph;
  friend class cudaTask;
  friend class cudaFlow;
  friend class cublasFlow;

  friend class Taskflow;
  friend class Executor;
  
  // Empty handle
  struct Empty {
  };

  // Host handle
  struct Host {
    template <typename C>
    Host(C&&);

    std::function<void()> func;
    
    static void callback(void*);
  };

  // Memset handle
  struct Memset {
  };

  // Memcpy handle
  struct Memcpy {
  };
  
  // Kernel handle
  struct Kernel {
    
    template <typename F>
    Kernel(F&& f);
    
    void* func {nullptr};
  };

  // Childflow handle
  struct Childflow {
    cudaGraph graph;
  };

  // Capture
  struct Capture {
    
    template <typename C>
    Capture(C&&);
    
    // TODO: probably better to use void(cudaStream_t)
    std::function<void(cudaStream_t)> work;
  };

  using handle_t = std::variant<
    Empty, 
    Host,
    Memset, 
    Memcpy, 
    Kernel,
    Childflow,
    Capture
  >;

  public:
  
  // variant index
  constexpr static auto CUDA_EMPTY_TASK     = get_index_v<Empty, handle_t>;
  constexpr static auto CUDA_HOST_TASK      = get_index_v<Host, handle_t>;
  constexpr static auto CUDA_MEMSET_TASK    = get_index_v<Memset, handle_t>;
  constexpr static auto CUDA_MEMCPY_TASK    = get_index_v<Memcpy, handle_t>; 
  constexpr static auto CUDA_KERNEL_TASK    = get_index_v<Kernel, handle_t>;
  constexpr static auto CUDA_CHILDFLOW_TASK = get_index_v<Childflow, handle_t>;
  constexpr static auto CUDA_CAPTURE_TASK   = get_index_v<Capture, handle_t>;
    
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

// Host handle constructor
template <typename C>
cudaNode::Host::Host(C&& c) : func {std::forward<C>(c)} {
}

// Host callback    
inline void cudaNode::Host::callback(void* data) { 
  static_cast<Host*>(data)->func(); 
};

// Kernel handle constructor
template <typename F>
cudaNode::Kernel::Kernel(F&& f) : 
  func {std::forward<F>(f)} {
}

// Capture handle constructor
template <typename C>
cudaNode::Capture::Capture(C&& work) : 
  work {std::forward<C>(work)} {
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

  // TODO: capture node doesn't have this
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

  auto node = new cudaNode(std::forward<ArgsT>(args)...);
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




