#pragma once

#include "cuda_memory.hpp"
#include "cuda_stream.hpp"
#include "cuda_meta.hpp"

#include "../utility/traits.hpp"

namespace tf {

// ----------------------------------------------------------------------------
// cudaGraph_t routines
// ----------------------------------------------------------------------------

/**
@brief gets the memcpy node parameter of a copy task
*/
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
>
cudaMemcpy3DParms cuda_get_copy_parms(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  cudaMemcpy3DParms p;

  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
  p.kind = cudaMemcpyDefault;

  return p;
}

/**
@brief gets the memcpy node parameter of a memcpy task (untyped)
*/
inline cudaMemcpy3DParms cuda_get_memcpy_parms(
  void* tgt, const void* src, size_t bytes
)  {

  // Parameters in cudaPitchedPtr
  // d   - Pointer to allocated memory
  // p   - Pitch of allocated memory in bytes
  // xsz - Logical width of allocation in elements
  // ysz - Logical height of allocation in elements
  cudaMemcpy3DParms p;
  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_cudaExtent(bytes, 1, 1);
  p.kind = cudaMemcpyDefault;

  return p;
}

/**
@brief gets the memset node parameter of a memcpy task (untyped)
*/
inline cudaMemsetParams cuda_get_memset_parms(void* dst, int ch, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;
  p.value = ch;
  p.pitch = 0;
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
  p.elementSize = 1;  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief gets the memset node parameter of a fill task (typed)
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
cudaMemsetParams cuda_get_fill_parms(T* dst, T value, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;

  // perform bit-wise copy
  p.value = 0;  // crucial
  static_assert(sizeof(T) <= sizeof(p.value), "internal error");
  std::memcpy(&p.value, &value, sizeof(T));

  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

/**
@brief gets the memset node parameter of a zero task (typed)
*/
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
>
cudaMemsetParams cuda_get_zero_parms(T* dst, size_t count) {

  cudaMemsetParams p;
  p.dst = dst;
  p.value = 0;
  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  return p;
}

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
    case cudaGraphNodeTypeGraph       : return "graph";
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
// cudaGraphNative
// ----------------------------------------------------------------------------

/**
@class cudaGraphNative

@brief class to create an RAII-styled wrapper over a CUDA executable graph

A cudaGraphNative object is an RAII-styled wrapper over 
a native CUDA executable graph (@c cudaGraphNative_t).
A cudaGraphNative object is move-only.
*/
class cudaGraphNative {

  struct cudaGraphNativeCreator {
    cudaGraph_t operator () () const { 
      cudaGraph_t g;
      TF_CHECK_CUDA(cudaGraphCreate(&g, 0), "failed to create a CUDA native graph");
      return g; 
    }
  };
  
  struct cudaGraphNativeDeleter {
    void operator () (cudaGraph_t g) const {
      if(g) {
        cudaGraphDestroy(g);
      }
    }
  };

  public:

    /**
    @brief constructs an RAII-styled object from the given CUDA exec

    Constructs a cudaGraphNative object which owns @c exec.
    */
    explicit cudaGraphNative(cudaGraph_t native) : _native(native) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new CUDA exec

    Equivalently calling @c cudaGraphNativeCreate to create a exec.
    */
    cudaGraphNative() : _native{ cudaGraphNativeCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    cudaGraphNative(const cudaGraphNative&) = delete;
    
    /**
    @brief move constructor
    */
    cudaGraphNative(cudaGraphNative&& rhs) : _native{rhs._native} {
      rhs._native = nullptr;
    }

    /**
    @brief destructs the CUDA exec
    */
    ~cudaGraphNative() {
      cudaGraphNativeDeleter {} (_native);
    }
    
    /**
    @brief disabled copy assignment
    */
    cudaGraphNative& operator = (const cudaGraphNative&) = delete;

    /**
    @brief move assignment
    */
    cudaGraphNative& operator = (cudaGraphNative&& rhs) {
      cudaGraphNativeDeleter {} (_native);
      _native = rhs._native;
      rhs._native = nullptr;
      return *this;
    }
    
    /**
    @brief implicit conversion to the native CUDA exec (cudaGraphNative_t)

    Returns the underlying exec of type @c cudaGraphNative_t.
    */
    operator cudaGraph_t () const {
      return _native;
    }
    
  private:

    cudaGraph_t _native {nullptr};
};

// ----------------------------------------------------------------------------
// cudaGraphExec
// ----------------------------------------------------------------------------

/**
@class cudaGraphExec

@brief class to create an RAII-styled wrapper over a CUDA executable graph

A cudaGraphExec object is an RAII-styled wrapper over 
a native CUDA executable graph (@c cudaGraphExec_t).
A cudaGraphExec object is move-only.
*/
class cudaGraphExec {

  struct cudaGraphExecCreator {
    cudaGraphExec_t operator () () const { return nullptr; }
  };
  
  struct cudaGraphExecDeleter {
    void operator () (cudaGraphExec_t executable) const {
      if(executable) {
        cudaGraphExecDestroy(executable);
      }
    }
  };

  public:

    /**
    @brief constructs an RAII-styled object from the given CUDA exec

    Constructs a cudaGraphExec object which owns @c exec.
    */
    explicit cudaGraphExec(cudaGraphExec_t exec) : _exec(exec) {
    }
    
    /**
    @brief constructs an RAII-styled object for a new CUDA exec

    Equivalently calling @c cudaGraphExecCreate to create a exec.
    */
    cudaGraphExec() : _exec{ cudaGraphExecCreator{}() } {
    }
    
    /**
    @brief disabled copy constructor
    */
    cudaGraphExec(const cudaGraphExec&) = delete;
    
    /**
    @brief move constructor
    */
    cudaGraphExec(cudaGraphExec&& rhs) : _exec{rhs._exec} {
      rhs._exec = nullptr;
    }

    /**
    @brief destructs the CUDA exec
    */
    ~cudaGraphExec() {
      cudaGraphExecDeleter {} (_exec);
    }
    
    /**
    @brief disabled copy assignment
    */
    cudaGraphExec& operator = (const cudaGraphExec&) = delete;

    /**
    @brief move assignment
    */
    cudaGraphExec& operator = (cudaGraphExec&& rhs) {
      cudaGraphExecDeleter {} (_exec);
      _exec = rhs._exec;
      rhs._exec = nullptr;
      return *this;
    }
    
    /**
    @brief replaces the managed executable graph with the given one

    Destructs the managed exec and resets it to the given exec.
    */
    void clear() {
      cudaGraphExecDeleter {} (_exec);
      _exec = nullptr;
    }
    
    /**
    @brief instantiates the exexutable from the given CUDA graph
    */
    void instantiate(cudaGraph_t graph) {
      cudaGraphExecDeleter {} (_exec);
      TF_CHECK_CUDA(
        cudaGraphInstantiate(&_exec, graph, nullptr, nullptr, 0),
        "failed to create an executable graph"
      );
    }
    
    /**
    @brief updates the exexutable from the given CUDA graph
    */
    cudaGraphExecUpdateResult update(cudaGraph_t graph) {
      cudaGraphNode_t error_node;
      cudaGraphExecUpdateResult error_result;
      cudaGraphExecUpdate(_exec, graph, &error_node, &error_result);
      return error_result;
    }
    
    /**
    @brief launchs the executable graph via the given stream
    */
    void launch(cudaStream_t stream) {
      TF_CHECK_CUDA(
        cudaGraphLaunch(_exec, stream), "failed to launch a CUDA executable graph"
      );
    }
  
    /**
    @brief implicit conversion to the native CUDA exec (cudaGraphExec_t)

    Returns the underlying exec of type @c cudaGraphExec_t.
    */
    operator cudaGraphExec_t () const {
      return _exec;
    }
    
  private:

    cudaGraphExec_t _exec {nullptr};
};

// ----------------------------------------------------------------------------
// cudaGraph class
// ----------------------------------------------------------------------------

// class: cudaGraph
class cudaGraph : public CustomGraphBase {

  friend class cudaNode;
  friend class cudaTask;
  friend class cudaFlowCapturerBase;
  friend class cudaFlowCapturer;
  friend class cudaFlow;
  friend class cudaCapturingBase;
  friend class cudaSequentialCapturing;
  friend class cudaLinearCapturing;
  friend class cudaRoundRobinCapturing;
  friend class Taskflow;
  friend class Executor;

  constexpr static int OFFLOADED = 0x01;
  constexpr static int CHANGED   = 0x02;
  constexpr static int UPDATED   = 0x04;

  public:

    cudaGraph() = default;
    ~cudaGraph();

    cudaGraph(const cudaGraph&) = delete;
    cudaGraph(cudaGraph&&);

    cudaGraph& operator = (const cudaGraph&) = delete;
    cudaGraph& operator = (cudaGraph&&);

    template <typename... ArgsT>
    cudaNode* emplace_back(ArgsT&&...);

    bool empty() const;

    void clear();
    void dump(std::ostream&, const void*, const std::string&) const override final;

  private:

    int _state{CHANGED};

    cudaGraph_t _native_handle {nullptr};

    std::vector<std::unique_ptr<cudaNode>> _nodes;
    //std::vector<cudaNode*> _nodes;
};

// ----------------------------------------------------------------------------
// cudaNode class
// ----------------------------------------------------------------------------

/**
@private
@class: cudaNode
*/
class cudaNode {

  friend class cudaGraph;
  friend class cudaTask;
  friend class cudaFlow;
  friend class cudaFlowCapturer;
  friend class cudaFlowCapturerBase;
  friend class cudaCapturingBase;
  friend class cudaSequentialCapturing;
  friend class cudaLinearCapturing;
  friend class cudaRoundRobinCapturing;
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

  // Subflow handle
  struct Subflow {
    cudaGraph graph;
  };

  // Capture
  struct Capture {

    template <typename C>
    Capture(C&&);

    std::function<void(cudaStream_t)> work;

    cudaEvent_t event;
    size_t level;
    size_t lid;
    size_t idx;
  };

  using handle_t = std::variant<
    Empty,
    Host,
    Memset,
    Memcpy,
    Kernel,
    Subflow,
    Capture
  >;

  public:

  // variant index
  constexpr static auto EMPTY   = get_index_v<Empty, handle_t>;
  constexpr static auto HOST    = get_index_v<Host, handle_t>;
  constexpr static auto MEMSET  = get_index_v<Memset, handle_t>;
  constexpr static auto MEMCPY  = get_index_v<Memcpy, handle_t>;
  constexpr static auto KERNEL  = get_index_v<Kernel, handle_t>;
  constexpr static auto SUBFLOW = get_index_v<Subflow, handle_t>;
  constexpr static auto CAPTURE = get_index_v<Capture, handle_t>;

    cudaNode() = delete;

    template <typename... ArgsT>
    cudaNode(cudaGraph&, ArgsT&&...);

  private:

    cudaGraph& _graph;

    std::string _name;

    handle_t _handle;

    cudaGraphNode_t _native_handle {nullptr};

    SmallVector<cudaNode*> _successors;
    SmallVector<cudaNode*> _dependents;

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

  _graph._state |= cudaGraph::CHANGED;

  _successors.push_back(v);
  v->_dependents.push_back(this);

  // capture node doesn't have the native graph yet
  if(_handle.index() != cudaNode::CAPTURE) {
    TF_CHECK_CUDA(
      cudaGraphAddDependencies(
        _graph._native_handle, &_native_handle, &v->_native_handle, 1
      ),
      "failed to add a preceding link ", this, "->", v
    );
  }
}

//// Procedure: _set_state
//inline void cudaNode::_set_state(int flag) {
//  _state |= flag;
//}
//
//// Procedure: _unset_state
//inline void cudaNode::_unset_state(int flag) {
//  _state &= ~flag;
//}
//
//// Procedure: _clear_state
//inline void cudaNode::_clear_state() {
//  _state = 0;
//}
//
//// Function: _has_state
//inline bool cudaNode::_has_state(int flag) const {
//  return _state & flag;
//}

// ----------------------------------------------------------------------------
// cudaGraph definitions
// ----------------------------------------------------------------------------

// Destructor
inline cudaGraph::~cudaGraph() {
  //clear();
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

  //clear();

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
  //for(auto n : _nodes) {
  //  delete n;
  //}
  _state = cudaGraph::CHANGED;
  _nodes.clear();
}

// Function: emplace_back
template <typename... ArgsT>
cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {

  _state |= cudaGraph::CHANGED;

  auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
  _nodes.emplace_back(std::move(node));
  return _nodes.back().get();

  // TODO: use object pool to save memory
  //auto node = new cudaNode(std::forward<ArgsT>(args)...);
  //_nodes.push_back(node);
  //return node;
}

// Procedure: dump the graph to a DOT format
inline void cudaGraph::dump(
  std::ostream& os, const void* root, const std::string& root_name
) const {

  // recursive dump with stack
  std::stack<std::tuple<const cudaGraph*, const cudaNode*, int>> stack;
  stack.push(std::make_tuple(this, nullptr, 1));

  int pl = 0;

  while(!stack.empty()) {

    auto [graph, parent, l] = stack.top();
    stack.pop();

    for(int i=0; i<pl-l+1; i++) {
      os << "}\n";
    }

    if(parent == nullptr) {
      if(root) {
        os << "subgraph cluster_p" << root << " {\nlabel=\"cudaFlow: ";
        if(root_name.empty()) os << 'p' << root;
        else os << root_name;
        os << "\";\n" << "color=\"purple\"\n";
      }
      else {
        os << "digraph cudaFlow {\n";
      }
    }
    else {
      os << "subgraph cluster_p" << parent << " {\nlabel=\"cudaSubflow: ";
      if(parent->_name.empty()) os << 'p' << parent;
      else os << parent->_name;
      os << "\";\n" << "color=\"purple\"\n";
    }

    for(auto& node : graph->_nodes) {

      auto v = node.get();

      os << 'p' << v << "[label=\"";
      if(v->_name.empty()) {
        os << 'p' << v << "\"";
      }
      else {
        os << v->_name << "\"";
      }

      switch(v->_handle.index()) {
        case cudaNode::KERNEL:
          os << " style=\"filled\""
             << " color=\"white\" fillcolor=\"black\""
             << " fontcolor=\"white\""
             << " shape=\"box3d\"";
        break;

        case cudaNode::SUBFLOW:
          stack.push(std::make_tuple(
            &(std::get_if<cudaNode::Subflow>(&v->_handle)->graph), v, l+1)
          );
          os << " style=\"filled\""
             << " color=\"black\" fillcolor=\"purple\""
             << " fontcolor=\"white\""
             << " shape=\"folder\"";
        break;

        default:
        break;
      }

      os << "];\n";

      for(const auto s : v->_successors) {
        os << 'p' << v << " -> " << 'p' << s << ";\n";
      }

      if(v->_successors.size() == 0) {
        if(parent == nullptr) {
          if(root) {
            os << 'p' << v << " -> p" << root << ";\n";
          }
        }
        else {
          os << 'p' << v << " -> p" << parent << ";\n";
        }
      }
    }

    // set the previous level
    pl = l;
  }

  for(int i=0; i<pl; i++) {
    os << "}\n";
  }

}


}  // end of namespace tf -----------------------------------------------------




