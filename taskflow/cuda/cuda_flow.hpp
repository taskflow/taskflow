#pragma once

#include "cuda_task.hpp"

namespace tf {

/**
@class cudaFlow

@brief methods for building a CUDA task dependency graph.

A cudaFlow is a high-level interface to manipulate GPU tasks using 
the task dependency graph model.
The class provides a set of methods for creating and launch different tasks
on one or multiple CUDA devices,
for instance, kernel tasks, data transfer tasks, and memory operation tasks.
*/
class cudaFlow {

  friend class Executor;

  public:
    
    /**
    @brief constructs a cudaFlow builder object

    @tparam P predicate type

    @param graph a cudaGraph to manipulate
    @param p predicate which return @c true if the launching should be contined
    */
    template <typename P>
    cudaFlow(cudaGraph& graph, P&& p);

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;
    
    /**
    @brief creates a no-operation task

    An empty node performs no operation during execution, 
    but can be used for transitive ordering. 
    For example, a phased execution graph with 2 groups of n nodes 
    with a barrier between them can be represented using an empty node 
    and 2*n dependency edges, 
    rather than no empty node and n^2 dependency edges.
    */
    cudaTask noop();
    
    // CUDA seems pretty restrictive about calling host in a cudaGraph.
    // We disable this function and wait for future stability.
    //
    //@brief creates a host execution task
    //
    //@tparam C callable type
    //
    //@param c a callable object constructible from std::function<void()>.

    //A host can only execute CPU-specific functions and cannot do any CUDA calls 
    //(e.g., cudaMalloc).
    //
    //template <typename C>
    //cudaTask host(C&& c);
    
    /**
    @brief creates a kernel task
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
    
    /**
    @brief creates a kernel task on a device
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type
    
    @param d device identifier to luanch the kernel
    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel_on(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

    /**
    @brief creates a memset task

    @param dst pointer to the destination device memory area
    @param v value to set for each byte of specified memory
    @param count size in bytes to set

    A memset task fills the first @c count bytes of device memory area 
    pointed by @c dst with the byte value @c v.
    */
    cudaTask memset(void* dst, int v, size_t count);
    
    /**
    @brief creates a memcpy task
    
    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param bytes bytes to copy

    @return cudaTask handle

    A memcpy task transfers @c bytes of data from a course location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */ 
    cudaTask memcpy(void* tgt, const void* src, size_t bytes);

    /**
    @brief creates a zero task that zeroes a typed memory block

    @tparam T element type (size of @c T must be either 1, 2, or 4)
    @param dst pointer to the destination device memory area
    @param count number of elements

    A zero task zeroes the first @c count elements of type @c T 
    in a device memory area pointed by @c dst.
    */
    template <typename T>
    std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
      cudaTask
    > 
    zero(T* dst, size_t count);

    /**
    @brief creates a fill task that fills a typed memory block with a value

    @tparam T element type (size of @c T must be either 1, 2, or 4)
    @param dst pointer to the destination device memory area
    @param value value to fill for each element of type @c T
    @param count number of elements

    A fill task fills the first @c count elements of type @c T with @c value
    in a device memory area pointed by @c dst.
    The value to fill is interpreted in type @c T rather than byte.
    */
    template <typename T>
    std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
      cudaTask
    >
    fill(T* dst, T value, size_t count);
    
    /**
    @brief creates a copy task
    
    @tparam T element type (non-void)

    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param num number of elements to copy

    @return cudaTask handle

    A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */
    template <
      typename T, 
      std::enable_if_t<!std::is_same<T, void>::value, void>* = nullptr
    >
    cudaTask copy(T* tgt, const T* src, size_t num);

    /**
    @brief assigns a device to launch the cudaFlow

    @param device target device identifier
    */
    void device(int device);

    /**
    @brief queries the device associated with the cudaFlow
    */
    int device() const;

    /**
    @brief assigns a stream to launch the cudaFlow

    @param stream target stream identifier
    */
    void stream(cudaStream_t stream);

    /**
    @brief assigns a predicate to loop the cudaFlow until the predicate is satisfied

    @tparam P predicate type
    @param p predicate which return @c true if the launching should be contined

    The execution of cudaFlow is equivalent to: <tt>while(!predicate()) { run cudaflow; }</tt>
    */
    template <typename P>
    void predicate(P&& p);
    
    /**
    @brief repeats the execution of the cudaFlow by @c n times
    */
    void repeat(size_t n);

  private:

    cudaGraph& _graph;
    
    int _device {0};

    nstd::optional<cudaStream_t> _stream;

    std::function<bool()> _predicate;
};

// Constructor
template <typename P>
cudaFlow::cudaFlow(cudaGraph& g, P&& p) : 
  _graph {g},
  _predicate {std::forward<P>(p)} {
}

// Procedure: predicate
template <typename P>
void cudaFlow::predicate(P&& pred) {
  _predicate = std::forward<P>(pred);
}

// Procedure: repeat
inline void cudaFlow::repeat(size_t n) {
  _predicate = [n] () mutable { return n-- == 0; };
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Procedure: device
inline void cudaFlow::device(int d) {
  _device = d;
}

// Function: device
inline int cudaFlow::device() const {
  return _device;
}

// Procedure: stream
inline void cudaFlow::stream(cudaStream_t s) {
  _stream = s;
}

// Function: noop
inline cudaTask cudaFlow::noop() {
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Noop>{}, 
    [](cudaGraph_t& graph, cudaGraphNode_t& node){
      TF_CHECK_CUDA(
        ::cudaGraphAddEmptyNode(&node, graph, nullptr, 0),
        "failed to create a no-operation (empty) node"
      );
    }
  );
  return cudaTask(node);
}

//// Function: host
//template <typename C>
//cudaTask cudaFlow::host(C&& c) {
//  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Host>{}, 
//    [c=std::forward<C>(c)](cudaGraph_t& graph, cudaGraphNode_t& node) mutable {
//      cudaHostNodeParams p;
//      p.fn = [] (void* data) { (*static_cast<C*>(data))(); };
//      p.userData = &c;
//      TF_CHECK_CUDA(
//        ::cudaGraphAddHostNode(&node, graph, nullptr, 0, &p),
//        "failed to create a host node"
//      );
//    }
//  );
//  return cudaTask(node);
//}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
  
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Kernel>{}, 
    [g, b, s, f=(void*)f, args...] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaKernelNodeParams p;
      void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
      p.func = f;
      p.gridDim = g;
      p.blockDim = b;
      p.sharedMemBytes = s;
      p.kernelParams = arguments;
      p.extra = nullptr;

      TF_CHECK_CUDA(
        ::cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in kernel task"
      );
    }
  );
  
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel_on(
  int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");
  
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Kernel>{}, 
    [d, g, b, s, f=(void*)f, args...] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaKernelNodeParams p;
      void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
      p.func = f;
      p.gridDim = g;
      p.blockDim = b;
      p.sharedMemBytes = s;
      p.kernelParams = arguments;
      p.extra = nullptr;

      cudaScopedDevice ctx(d);
      TF_CHECK_CUDA(
        ::cudaGraphAddKernelNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in kernel_on task"
      );
    }
  );
  
  return cudaTask(node);
}

// Function: zero
template <typename T>
std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
  cudaTask
> 
cudaFlow::zero(T* dst, size_t count) {
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Memset>{},
    [dst, count] (cudaGraph_t& graph, cudaGraphNode_t& node) {
      cudaMemsetParams p;
      p.dst = dst;
      p.value = 0;
      p.pitch = 0;
      p.elementSize = sizeof(T);  // either 1, 2, or 4
      p.width = count;
      p.height = 1;
      TF_CHECK_CUDA(
        cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in zero task"
      );
    }
  );
  return cudaTask(node);
}
    
// Function: fill
template <typename T>
std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), 
  cudaTask
>
cudaFlow::fill(T* dst, T value, size_t count) {
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Memset>{},
    [dst, value, count] (cudaGraph_t& graph, cudaGraphNode_t& node) {
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
      TF_CHECK_CUDA(
        cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in fill task"
      );
    }
  );
  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same<T, void>::value, void>*
>
cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Copy>{}, 
    [tgt, src, num] (cudaGraph_t& graph, cudaGraphNode_t& node) {

      cudaMemcpy3DParms p;
      p.srcArray = nullptr;
      p.srcPos = ::make_cudaPos(0, 0, 0);
      p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
      p.dstArray = nullptr;
      p.dstPos = ::make_cudaPos(0, 0, 0);
      p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
      p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
      p.kind = cudaMemcpyDefault;

      TF_CHECK_CUDA(
        cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in copy task"
      );
    }
  );

  return cudaTask(node);
}

// Function: memset
inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {

  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Memset>{},
    [dst, ch, count] (cudaGraph_t& graph, cudaGraphNode_t& node) {
      cudaMemsetParams p;
      p.dst = dst;
      p.value = ch;
      p.pitch = 0;
      //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
      //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
      p.elementSize = 1;  // either 1, 2, or 4
      p.width = count;
      p.height = 1;
      TF_CHECK_CUDA(
        cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in memset task"
      );
    }
  );
  
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
  auto node = _graph.emplace_back(nstd::in_place_type_t<cudaNode::Copy>{},
    [tgt, src, bytes] (cudaGraph_t& graph, cudaGraphNode_t& node) {
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
      TF_CHECK_CUDA(
        cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &p),
        "failed to create a cudaGraph node in memcpy task"
      );
    }
  );
  return cudaTask(node);
}

}  // end of namespace tf -----------------------------------------------------


