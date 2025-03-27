#pragma once

#include <filesystem>

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
inline size_t cuda_graph_get_num_root_nodes(cudaGraph_t graph) {
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
inline size_t cuda_graph_get_num_nodes(cudaGraph_t graph) {
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
inline size_t cuda_graph_get_num_edges(cudaGraph_t graph) {
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
inline std::vector<cudaGraphNode_t> cuda_graph_get_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_graph_get_num_nodes(graph);
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
inline std::vector<cudaGraphNode_t> cuda_graph_get_root_nodes(cudaGraph_t graph) {
  size_t num_nodes = cuda_graph_get_num_root_nodes(graph);
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
cuda_graph_get_edges(cudaGraph_t graph) {
  size_t num_edges = cuda_graph_get_num_edges(graph);
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

// ----------------------------------------------------------------------------
// cudaTask Types
// ----------------------------------------------------------------------------

/**
@brief convert a cuda_task type to a human-readable string
*/
constexpr const char* to_string(cudaGraphNodeType type) {
  switch (type) {
    case cudaGraphNodeTypeKernel:             return "Kernel";
    case cudaGraphNodeTypeMemcpy:             return "Memcpy";
    case cudaGraphNodeTypeMemset:             return "Memset";
    case cudaGraphNodeTypeHost:               return "Host";
    case cudaGraphNodeTypeGraph:              return "Graph";
    case cudaGraphNodeTypeEmpty:              return "Empty";
    case cudaGraphNodeTypeWaitEvent:          return "WaitEvent";
    case cudaGraphNodeTypeEventRecord:        return "EventRecord";
    case cudaGraphNodeTypeExtSemaphoreSignal: return "ExtSemaphoreSignal";
    case cudaGraphNodeTypeExtSemaphoreWait:   return "ExtSemaphoreWait";
    case cudaGraphNodeTypeMemAlloc:           return "MemAlloc";
    case cudaGraphNodeTypeMemFree:            return "MemFree";
    case cudaGraphNodeTypeConditional:        return "Conditional";
    default:                                  return "undefined";
  }
}

// ----------------------------------------------------------------------------
// cudaGraph
// ----------------------------------------------------------------------------

/**
 * @struct cudaGraphCreator
 * @brief  a functor for creating a CUDA graph
 *
 * This structure provides an overloaded function call operator to create a
 * new CUDA graph using `cudaGraphCreate`. 
 *
 */
struct cudaGraphCreator {

  /**
   * @brief creates a new CUDA graph
   *
   * Calls `cudaGraphCreate` to generate a CUDA native graph and returns it.
   * If the graph creation fails, an error is reported.
   *
   * @return A newly created `cudaGraph_t` instance.
   * @throws If CUDA graph creation fails, an error is logged.
   */
  cudaGraph_t operator () () const {
    cudaGraph_t g;
    TF_CHECK_CUDA(cudaGraphCreate(&g, 0), "failed to create a CUDA native graph");
    return g;
  }
  
  /**
  @brief return the given CUDA graph
  */
  cudaGraph_t operator () (cudaGraph_t graph) const {
    return graph;
  }
};

/**
 * @struct cudaGraphDeleter
 * @brief a functor for deleting a CUDA graph
 *
 * This structure provides an overloaded function call operator to safely
 * destroy a CUDA graph using `cudaGraphDestroy`.
 *
 */
struct cudaGraphDeleter {

  /**
   * @brief deletes a CUDA graph
   *
   * Calls `cudaGraphDestroy` to release the CUDA graph resource if it is valid.
   *
   * @param g the CUDA graph to be destroyed
   */
  void operator () (cudaGraph_t g) const {
    cudaGraphDestroy(g);
  }
};
  

/**
@class cudaGraph

@brief class to create a CUDA graph managed by C++ smart pointer

@tparam Creator functor to create the stream (used in constructor)
@tparam Deleter functor to delete the stream (used in destructor)

This class wraps a `cudaGraph_t` handle with std::unique_ptr to ensure proper 
resource management and automatic cleanup.
*/
template <typename Creator, typename Deleter>
class cudaGraphBase : public std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, cudaGraphDeleter> {
  
  static_assert(std::is_pointer_v<cudaGraph_t>, "cudaGraph_t is not a pointer type");

  public:
  
  /**
  @brief base std::unique_ptr type
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, Deleter>;

  /**
  @brief constructs a `cudaGraph` object by passing the given arguments to the executable CUDA graph creator

  Constructs a `cudaGraph` object by passing the given arguments to the executable CUDA graph creator

  @param args arguments to pass to the executable CUDA graph creator
  */
  template <typename... ArgsT>
  explicit cudaGraphBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {
  }  

  /**
  @brief queries the number of nodes in a native CUDA graph
  */
  size_t num_nodes() const;
  
  /**
  @brief queries the number of edges in a native CUDA graph
  */
  size_t num_edges() const;

  /**
  @brief queries if the graph is empty
  */
  bool empty() const;

  /**
  @brief dumps the CUDA graph to a DOT format through the given output stream
  
  @param os target output stream
  */
  void dump(std::ostream& os);
    
};

// query the number of nodes
template <typename Creator, typename Deleter>
size_t cudaGraphBase<Creator, Deleter>::num_nodes() const {
  size_t n;
  TF_CHECK_CUDA(
    cudaGraphGetNodes(this->get(), nullptr, &n),
    "failed to get native graph nodes"
  );
  return n;
}

// query the emptiness
template <typename Creator, typename Deleter>
bool cudaGraphBase<Creator, Deleter>::empty() const {
  return num_nodes() == 0;
}

// query the number of edges
template <typename Creator, typename Deleter>
size_t cudaGraphBase<Creator, Deleter>::num_edges() const {
  size_t num_edges;
  TF_CHECK_CUDA(
    cudaGraphGetEdges(this->get(), nullptr, nullptr, &num_edges),
    "failed to get native graph edges"
  );
  return num_edges;
}

//// dump the graph
//inline void cudaGraph::dump(std::ostream& os) {
//  
//  // acquire the native handle
//  auto g = this->get();
//
//  os << "digraph cudaGraph {\n";
//
//  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
//  stack.push(std::make_tuple(g, nullptr, 1));
//
//  int pl = 0;
//
//  while(stack.empty() == false) {
//
//    auto [graph, parent, l] = stack.top();
//    stack.pop();
//
//    for(int i=0; i<pl-l+1; i++) {
//      os << "}\n";
//    }
//
//    os << "subgraph cluster_p" << graph << " {\n"
//       << "label=\"cudaGraph-L" << l << "\";\n"
//       << "color=\"purple\";\n";
//
//    auto nodes = cuda_graph_get_nodes(graph);
//    auto edges = cuda_graph_get_edges(graph);
//
//    for(auto& [from, to] : edges) {
//      os << 'p' << from << " -> " << 'p' << to << ";\n";
//    }
//
//    for(auto& node : nodes) {
//      auto type = cuda_get_graph_node_type(node);
//      if(type == cudaGraphNodeTypeGraph) {
//
//        cudaGraph_t child_graph;
//        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &child_graph), "");
//        stack.push(std::make_tuple(child_graph, node, l+1));
//
//        os << 'p' << node << "["
//           << "shape=folder, style=filled, fontcolor=white, fillcolor=purple, "
//           << "label=\"cudaGraph-L" << l+1
//           << "\"];\n";
//      }
//      else {
//        os << 'p' << node << "[label=\""
//           << to_string(type)
//           << "\"];\n";
//      }
//    }
//
//    // precede to parent
//    if(parent != nullptr) {
//      std::unordered_set<cudaGraphNode_t> successors;
//      for(const auto& p : edges) {
//        successors.insert(p.first);
//      }
//      for(auto node : nodes) {
//        if(successors.find(node) == successors.end()) {
//          os << 'p' << node << " -> " << 'p' << parent << ";\n";
//        }
//      }
//    }
//
//    // set the previous level
//    pl = l;
//  }
//
//  for(int i=0; i<=pl; i++) {
//    os << "}\n";
//  }
//}

// dump the graph
template <typename Creator, typename Deleter>
void cudaGraphBase<Creator, Deleter>::dump(std::ostream& os) {

  // Generate a unique temporary filename in the system's temp directory using filesystem
  auto temp_path = std::filesystem::temp_directory_path() / "graph_";
  std::random_device rd;
  std::uniform_int_distribution<int> dist(100000, 999999); // Generates a random number
  temp_path += std::to_string(dist(rd)) + ".dot";

  // Call the original function with the temporary file
  TF_CHECK_CUDA(cudaGraphDebugDotPrint(this->get(), temp_path.string().c_str(), 0), "");

  // Read the file and write to the output stream
  std::ifstream file(temp_path);
  if (file) {
    os << file.rdbuf();  // Copy file contents to the stream
    file.close();
    std::filesystem::remove(temp_path);  // Clean up the temporary file
  } else {
    TF_THROW("failed to open ", temp_path, " for dumping the CUDA graph");
  }
}

/**
@brief default smart pointer type to manage a `cudaGraph_t` object with unique ownership
*/
using cudaGraph = cudaGraphBase<cudaGraphCreator, cudaGraphDeleter>;

// ----------------------------------------------------------------------------
// cudaGraphExec
// ----------------------------------------------------------------------------

/**
@struct cudaGraphExecCreator
@brief a functor for creating an executable CUDA graph

This structure provides an overloaded function call operator to create a
new executable CUDA graph using `cudaGraphCreate`. 
*/
struct cudaGraphExecCreator {
  /**
  @brief returns a null executable CUDA graph
  */
  cudaGraphExec_t operator () () const { 
    return nullptr;
  }
  
  /**
  @brief returns the given executable graph
  */
  cudaGraphExec_t operator () (cudaGraphExec_t exec) const {
    return exec;
  }
};
  
/**
@struct cudaGraphDeleter
@brief a functor for deleting an executable CUDA graph

This structure provides an overloaded function call operator to safely
destroy a CUDA graph using `cudaGraphDestroy`.
*/
struct cudaGraphExecDeleter {
  /**
   * @brief deletes an executable CUDA graph
   *
   * Calls `cudaGraphDestroy` to release the CUDA graph resource if it is valid.
   *
   * @param executable the executable CUDA graph to be destroyed
   */
  void operator () (cudaGraphExec_t executable) const {
    cudaGraphExecDestroy(executable);
  }
};

/**
@class cudaGraphExecBase

@brief class to create an executable CUDA graph managed by C++ smart pointer

@tparam Creator functor to create the stream (used in constructor)
@tparam Deleter functor to delete the stream (used in destructor)

This class wraps a `cudaGraphExec_t` handle with `std::unique_ptr` to ensure proper 
resource management and automatic cleanup.
*/
template <typename Creator, typename Deleter>
class cudaGraphExecBase : public std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, Deleter> {
  
  static_assert(std::is_pointer_v<cudaGraphExec_t>, "cudaGraphExec_t is not a pointer type");

  public:
  
  /**
  @brief base std::unique_ptr type
  */
  using base_type = std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, Deleter>;

  /**
  @brief constructs a `cudaGraphExec` object by passing the given arguments to the executable CUDA graph creator

  Constructs a `cudaGraphExec` object by passing the given arguments to the executable CUDA graph creator

  @param args arguments to pass to the executable CUDA graph creator
  */
  template <typename... ArgsT>
  explicit cudaGraphExecBase(ArgsT&& ... args) : base_type(
    Creator{}(std::forward<ArgsT>(args)...), Deleter()
  ) {
  }  

  /** 
  @brief runs the executable graph via the given CUDA stream
  */
  template <typename Stream>
  void run(Stream& stream) {
    // native cudaStream_t object
    if constexpr (std::is_same_v<Stream, cudaStream_t>) {
      TF_CHECK_CUDA(
        cudaGraphLaunch(this->get(), stream), "failed to launch a CUDA executable graph"
      );  
    }
    // cudaStreamBase object
    else {
      TF_CHECK_CUDA(
        cudaGraphLaunch(this->get(), stream.get()), "failed to launch a CUDA executable graph"
      );  
    }
  }
};

/**
@brief default smart pointer type to manage a `cudaGraphExec_t` object with unique ownership
*/
using cudaGraphExec = cudaGraphExecBase<cudaGraphExecCreator, cudaGraphExecDeleter>;

}  // end of namespace tf -----------------------------------------------------




