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
// cudaTask
// ----------------------------------------------------------------------------

/**
@class cudaTask

@brief class to create a task handle of a CUDA %Graph node
*/
class cudaTask {

  template <typename Creator, typename Deleter>
  friend class cudaGraphBase;
  
  template <typename Creator, typename Deleter>
  friend class cudaGraphExecBase;

  friend class cudaFlow;
  friend class cudaFlowCapturer;
  friend class cudaFlowCapturerBase;

  friend std::ostream& operator << (std::ostream&, const cudaTask&);

  public:

    /**
    @brief constructs an empty cudaTask
    */
    cudaTask() = default;

    /**
    @brief copy-constructs a cudaTask
    */
    cudaTask(const cudaTask&) = default;

    /**
    @brief copy-assigns a cudaTask
    */
    cudaTask& operator = (const cudaTask&) = default;

    /**
    @brief adds precedence links from this to other tasks

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& precede(Ts&&... tasks);

    /**
    @brief adds precedence links from other tasks to this

    @tparam Ts parameter pack

    @param tasks one or multiple tasks

    @return @c *this
    */
    template <typename... Ts>
    cudaTask& succeed(Ts&&... tasks);

    /**
    @brief queries the number of successors
    */
    size_t num_successors() const;

    /**
    @brief queries the number of dependents
    */
    size_t num_predecessors() const;

    /**
    @brief queries the type of this task
    */
    auto type() const;

    /**
    @brief dumps the task through an output stream

    @param os an output stream target
    */
    void dump(std::ostream& os) const;

  private:

    cudaTask(cudaGraph_t, cudaGraphNode_t);
    
    cudaGraph_t _native_graph {nullptr};
    cudaGraphNode_t _native_node {nullptr};
};

// Constructor
inline cudaTask::cudaTask(cudaGraph_t native_graph, cudaGraphNode_t native_node) : 
  _native_graph {native_graph}, _native_node  {native_node} {
}
  
// Function: precede
template <typename... Ts>
cudaTask& cudaTask::precede(Ts&&... tasks) {
  (
    cudaGraphAddDependencies(
      _native_graph, &_native_node, &(tasks._native_node), 1
    ), ...
  );
  return *this;
}

// Function: succeed
template <typename... Ts>
cudaTask& cudaTask::succeed(Ts&&... tasks) {
  (tasks.precede(*this), ...);
  return *this;
}

// Function: num_predecessors
inline size_t cudaTask::num_predecessors() const {
  size_t num_predecessors {0};
  cudaGraphNodeGetDependencies(_native_node, nullptr, &num_predecessors);
  return num_predecessors;
}

// Function: num_successors
inline size_t cudaTask::num_successors() const {
  size_t num_successors {0};
  cudaGraphNodeGetDependentNodes(_native_node, nullptr, &num_successors);
  return num_successors;
}

// Function: type
inline auto cudaTask::type() const {
  cudaGraphNodeType type;
  cudaGraphNodeGetType(_native_node, &type);
  return type;
}

// Function: dump
inline void cudaTask::dump(std::ostream& os) const {
  os << "cudaTask [type=" << to_string(type()) << ']';
}

/**
@brief overload of ostream inserter operator for cudaTask
*/
inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
  ct.dump(os);
  return os;
}

// ----------------------------------------------------------------------------
// cudaGraph
// ----------------------------------------------------------------------------

/**
 @class cudaGraphCreator

 @brief class to create functors that construct CUDA graphs
 
 This class define functors to new CUDA graphs using `cudaGraphCreate`. 
 
*/
class cudaGraphCreator {

  public:

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
 @class cudaGraphDeleter

 @brief class to create a functor that deletes a CUDA graph
 
 This structure provides an overloaded function call operator to safely
 destroy a CUDA graph using `cudaGraphDestroy`.
 
*/
class cudaGraphDeleter {

  public:
 
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
@class cudaGraphBase

@brief class to create a CUDA graph with uunique ownership

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
  @brief constructs a `cudaGraph` from the given rhs using move semantics
  */
  cudaGraphBase(cudaGraphBase&&) = default;

  /**
  @brief assign the rhs to `*this` using move semantics
  */
  cudaGraphBase& operator = (cudaGraphBase&&) = default;

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

  // ------------------------------------------------------------------------
  // Graph building routines
  // ------------------------------------------------------------------------

  /**
  @brief creates a no-operation task

  @return a tf::cudaTask handle

  An empty node performs no operation during execution,
  but can be used for transitive ordering.
  For example, a phased execution graph with 2 groups of @c n nodes
  with a barrier between them can be represented using an empty node
  and @c 2*n dependency edges,
  rather than no empty node and @c n^2 dependency edges.
  */
  cudaTask noop();

  /**
  @brief creates a host task that runs a callable on the host

  @tparam C callable type

  @param callable a callable object with neither arguments nor return
  (i.e., constructible from @c std::function<void()>)
  @param user_data a pointer to the user data

  @return a tf::cudaTask handle

  A host task can only execute CPU-specific functions and cannot do any CUDA calls
  (e.g., @c cudaMalloc).
  */
  template <typename C>
  cudaTask host(C&& callable, void* user_data);

  /**
  @brief creates a kernel task

  @tparam F kernel function type
  @tparam ArgsT kernel function parameters type

  @param g configured grid
  @param b configured block
  @param s configured shared memory size in bytes
  @param f kernel function
  @param args arguments to forward to the kernel function by copy

  @return a tf::cudaTask handle
  */
  template <typename F, typename... ArgsT>
  cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT... args);

  /**
  @brief creates a memset task that fills untyped data with a byte value

  @param dst pointer to the destination device memory area
  @param v value to set for each byte of specified memory
  @param count size in bytes to set

  @return a tf::cudaTask handle

  A memset task fills the first @c count bytes of device memory area
  pointed by @c dst with the byte value @c v.
  */
  cudaTask memset(void* dst, int v, size_t count);

  /**
  @brief creates a memcpy task that copies untyped data in bytes

  @param tgt pointer to the target memory block
  @param src pointer to the source memory block
  @param bytes bytes to copy

  @return a tf::cudaTask handle

  A memcpy task transfers @c bytes of data from a source location
  to a target location. Direction can be arbitrary among CPUs and GPUs.
  */
  cudaTask memcpy(void* tgt, const void* src, size_t bytes);

  /**
  @brief creates a memset task that sets a typed memory block to zero

  @tparam T element type (size of @c T must be either 1, 2, or 4)
  @param dst pointer to the destination device memory area
  @param count number of elements

  @return a tf::cudaTask handle

  A zero task zeroes the first @c count elements of type @c T
  in a device memory area pointed by @c dst.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  cudaTask zero(T* dst, size_t count);

  /**
  @brief creates a memset task that fills a typed memory block with a value

  @tparam T element type (size of @c T must be either 1, 2, or 4)

  @param dst pointer to the destination device memory area
  @param value value to fill for each element of type @c T
  @param count number of elements

  @return a tf::cudaTask handle

  A fill task fills the first @c count elements of type @c T with @c value
  in a device memory area pointed by @c dst.
  The value to fill is interpreted in type @c T rather than byte.
  */
  template <typename T, std::enable_if_t<
    is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
  >
  cudaTask fill(T* dst, T value, size_t count);

  /**
  @brief creates a memcopy task that copies typed data

  @tparam T element type (non-void)

  @param tgt pointer to the target memory block
  @param src pointer to the source memory block
  @param num number of elements to copy

  @return a tf::cudaTask handle

  A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
  to a target location. Direction can be arbitrary among CPUs and GPUs.
  */
  template <typename T,
    std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
  >
  cudaTask copy(T* tgt, const T* src, size_t num);
  
  // ------------------------------------------------------------------------
  // generic algorithms
  // ------------------------------------------------------------------------

  /**
  @brief runs a callable with only a single kernel thread

  @tparam C callable type

  @param c callable to run by a single kernel thread

  @return a tf::cudaTask handle
  */
  template <typename C>
  cudaTask single_task(C c);
  
  /**
  @brief applies a callable to each dereferenced element of the data array

  @tparam I iterator type
  @tparam C callable type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  @param callable a callable object to apply to the dereferenced iterator

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  for(auto itr = first; itr != last; itr++) {
    callable(*itr);
  }
  @endcode
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask for_each(I first, I last, C callable);
  
  /**
  @brief applies a callable to each index in the range with the step size

  @tparam I index type
  @tparam C callable type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first beginning index
  @param last last index
  @param step step size
  @param callable the callable to apply to each element in the data array

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  // step is positive [first, last)
  for(auto i=first; i<last; i+=step) {
    callable(i);
  }

  // step is negative [first, last)
  for(auto i=first; i>last; i+=step) {
    callable(i);
  }
  @endcode
  */
  template <typename I, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask for_each_index(I first, I last, I step, C callable);
  
  /**
  @brief applies a callable to a source range and stores the result in a target range

  @tparam I input iterator type
  @tparam O output iterator type
  @tparam C unary operator type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first iterator to the beginning of the input range
  @param last iterator to the end of the input range
  @param output iterator to the beginning of the output range
  @param op the operator to apply to transform each element in the range

  @return a tf::cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first != last) {
    *output++ = callable(*first++);
  }
  @endcode
  */
  template <typename I, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask transform(I first, I last, O output, C op);
  
  /**
  @brief creates a task to perform parallel transforms over two ranges of items

  @tparam I1 first input iterator type
  @tparam I2 second input iterator type
  @tparam O output iterator type
  @tparam C unary operator type
  @tparam E execution poligy (default tf::cudaDefaultExecutionPolicy)

  @param first1 iterator to the beginning of the input range
  @param last1 iterator to the end of the input range
  @param first2 iterato
  @param output iterator to the beginning of the output range
  @param op binary operator to apply to transform each pair of items in the
            two input ranges

  @return cudaTask handle

  This method is equivalent to the parallel execution of the following loop on a GPU:

  @code{.cpp}
  while (first1 != last1) {
    *output++ = op(*first1++, *first2++);
  }
  @endcode
  */
  template <typename I1, typename I2, typename O, typename C, typename E = cudaDefaultExecutionPolicy>
  cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);

  private:

  cudaGraphBase(const cudaGraphBase&) = delete;
  cudaGraphBase& operator = (const cudaGraphBase&) = delete;
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

// Function: noop
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::noop() {

  cudaGraphNode_t node;

  TF_CHECK_CUDA(
    cudaGraphAddEmptyNode(&node, this->get(), nullptr, 0),
    "failed to create a no-operation (empty) node"
  );

  return cudaTask(this->get(), node);
}

// Function: host
template <typename Creator, typename Deleter>
template <typename C>
cudaTask cudaGraphBase<Creator, Deleter>::host(C&& callable, void* user_data) {

  cudaGraphNode_t node;
  cudaHostNodeParams p {callable, user_data};

  TF_CHECK_CUDA(
    cudaGraphAddHostNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a host node"
  );

  return cudaTask(this->get(), node);
}

// Function: kernel
template <typename Creator, typename Deleter>
template <typename F, typename... ArgsT>
cudaTask cudaGraphBase<Creator, Deleter>::kernel(
  dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {

  cudaGraphNode_t node;
  cudaKernelNodeParams p;

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };

  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphAddKernelNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a kernel task"
  );

  return cudaTask(this->get(), node);
}

// Function: zero
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::zero(T* dst, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset (zero) task"
  );

  return cudaTask(this->get(), node);
}

// Function: fill
template <typename Creator, typename Deleter>
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::fill(T* dst, T value, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_fill_parms(dst, value, count);
  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset (fill) task"
  );

  return cudaTask(this->get(), node);
}

// Function: copy
template <typename Creator, typename Deleter>
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cudaGraphBase<Creator, Deleter>::copy(T* tgt, const T* src, size_t num) {

  cudaGraphNode_t node;
  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memcpy (copy) task"
  );

  return cudaTask(this->get(), node);
}

// Function: memset
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::memset(void* dst, int ch, size_t count) {

  cudaGraphNode_t node;
  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memset task"
  );

  return cudaTask(this->get(), node);
}

// Function: memcpy
template <typename Creator, typename Deleter>
cudaTask cudaGraphBase<Creator, Deleter>::memcpy(void* tgt, const void* src, size_t bytes) {

  cudaGraphNode_t node;
  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(&node, this->get(), nullptr, 0, &p),
    "failed to create a memcpy task"
  );

  return cudaTask(this->get(), node);
}





}  // end of namespace tf -----------------------------------------------------




