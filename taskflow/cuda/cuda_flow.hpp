#pragma once

#include "cuda_task.hpp"
#include "cuda_capturer.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"
#include "cuda_algorithm/cuda_transform.hpp"

/** 
@file cuda_flow.hpp
@brief cudaFlow include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: cudaFlow
// ----------------------------------------------------------------------------

/**
@class cudaFlow

@brief class for building a CUDA task dependency graph

A %cudaFlow is a high-level interface over CUDA Graph to perform GPU operations 
using the task dependency graph model.
The class provides a set of methods for creating and launch different tasks
on one or multiple CUDA devices,
for instance, kernel tasks, data transfer tasks, and memory operation tasks.
The following example creates a %cudaFlow of two kernel tasks, @c task_1 and 
@c task_2, where @c task_1 runs before @c task_2.

@code{.cpp}
tf::Taskflow taskflow;
tf::Executor executor;

taskflow.emplace([&](tf::cudaFlow& cf){
  // create two kernel tasks 
  tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
  tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);
  
  // kernel1 runs before kernel2
  task_1.precede(task2);
});

executor.run(taskflow).wait();
@endcode

A %cudaFlow is a task and will be run by one worker thread in the executor.
That is, the callable that defines how the given %cudaFlow runs 
will be executed sequentially.
*/
class cudaFlow {

  friend class Executor;

  struct External {
    cudaGraph graph;
  };

  struct Internal {
    Executor& executor;
    Internal(Executor& e) : executor {e} {}
  };

  using handle_t = std::variant<External, Internal>;

  public:

    /**
    @brief constructs a standalone %cudaFlow
    */
    cudaFlow();
    
    /**
    @brief destroys the %cudaFlow and its associated native CUDA graph
           and executable graph
     */
    ~cudaFlow();

    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;
    
    /**
    @brief dumps the %cudaFlow graph into a DOT format through an
           output stream that defines the stream insertion operator @c <<
    */
    template<typename T>
    void dump(T& os) const;
    
    /**
    @brief dumps the native CUDA graph into a DOT format through an
           output stream that defines the stream insertion operator @c <<

    The native CUDA graph may be different from the upper-level %cudaFlow 
    graph when flow capture is involved.
    */
    template<typename T>
    void dump_native_graph(T& os) const;

    // ------------------------------------------------------------------------
    // Graph building routines
    // ------------------------------------------------------------------------

    /**
    @brief creates a no-operation task
    
    @return a tf::cudaTask handle

    An empty node performs no operation during execution, 
    but can be used for transitive ordering. 
    For example, a phased execution graph with 2 groups of n nodes 
    with a barrier between them can be represented using an empty node 
    and 2*n dependency edges, 
    rather than no empty node and n^2 dependency edges.
    */
    cudaTask noop();
    
    /**
    @brief creates a host execution task
    
    @tparam C callable type
    
    @param callable a callable object with neither arguments nor return 
    (i.e., constructible from std::function<void()>)
    
    @return a tf::cudaTask handle

    A host task can only execute CPU-specific functions and cannot do any CUDA calls 
    (e.g., cudaMalloc).
    */
    template <typename C>
    cudaTask host(C&& callable);
    
    /**
    @brief creates a kernel task
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return a tf::cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
    
    /**
    @brief creates a kernel task on a device
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type
    
    @param d device identifier to launch the kernel
    @param g configured grid
    @param b configured block
    @param s configured shared memory
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return a tf::cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel_on(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

    /**
    @brief creates a memset task

    @param dst pointer to the destination device memory area
    @param v value to set for each byte of specified memory
    @param count size in bytes to set
    
    @return a tf::cudaTask handle

    A memset task fills the first @c count bytes of device memory area 
    pointed by @c dst with the byte value @c v.
    */
    cudaTask memset(void* dst, int v, size_t count);
    
    /**
    @brief creates a memcpy task
    
    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param bytes bytes to copy

    @return a tf::cudaTask handle

    A memcpy task transfers @c bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */ 
    cudaTask memcpy(void* tgt, const void* src, size_t bytes);

    /**
    @brief creates a zero task that zeroes a typed memory block

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
    @brief creates a fill task that fills a typed memory block with a value

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
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), cudaTask>* = nullptr
    >
    cudaTask fill(T* dst, T value, size_t count);
    
    /**
    @brief creates a copy task of typed data
    
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
    
    /**
    @brief offloads the %cudaFlow onto a GPU and repeatedly running it until 
    the predicate becomes true
    
    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the present %cudaFlow onto a GPU and
    repeatedly executes it until the predicate returns @c true.

    A offloaded %cudaFlow force the underlying graph to be instantiated.
    After the instantiation, you should not modify the graph topology
    but update node parameters.

    By default, if users do not offload the %cudaFlow, 
    the executor will offload it once.
    */
    template <typename P>
    void offload_until(P&& predicate);
    
    /**
    @brief offloads the %cudaFlow and executes it by the given times

    @param N number of executions
    */
    void offload_n(size_t N);

    /**
    @brief offloads the %cudaFlow and executes it once
    */
    void offload();

    // ------------------------------------------------------------------------
    // update methods
    // ------------------------------------------------------------------------
  
    // TODO update_kernel_on

    /**
    @brief updates parameters of a kernel task created from tf::cudaFlow::kernel

    The method updates the parameters of a kernel associated with the given 
    @c task. We do not allow you to change the kernel function.
    */
    template <typename... ArgsT>
    void update_kernel(cudaTask task, dim3 g, dim3 b, size_t shm, ArgsT&&... args);

    /**
    @brief updates parameters of a copy task created from tf::cudaFlow::copy

    The method updates the parameters of a copy task.
    The source/destination memory may have different address values but 
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    template <
      typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void update_copy(cudaTask task, T* tgt, const T* src, size_t num);

    /**
    @brief updates parameters of a memcpy task created from tf::cudaFlow::memcpy
    
    The method updates the parameters of a memcpy task.
    The source/destination memory may have different address values but 
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    void update_memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);

    /**
    @brief updates parameters of a memset task created from tf::cudaFlow::memset
    
    The method updates the parameters of a memset task.
    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    void update_memset(cudaTask task, void* dst, int ch, size_t count);

    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------
    
    /**
    @brief runs a callable with only a single kernel thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    
    @return a tf::cudaTask handle
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    /**
    @brief applies a callable to each dereferenced element of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable a callable object to apply to the dereferenced iterator 
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    for(auto itr = first; itr != last; i++) {
      callable(*itr);
    }
    @endcode
    */
    template <typename I, typename C>
    cudaTask for_each(I first, I last, C&& callable);

    /**
    @brief applies a callable to each index in the range with the step size
    
    @tparam I index type
    @tparam C callable type
    
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
    template <typename I, typename C>
    cudaTask for_each_index(I first, I last, I step, C&& callable);
  
    /**
    @brief applies a callable to a source range and stores the result in a target range
    
    @tparam I iterator type
    @tparam C callable type
    @tparam S source types

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable the callable to apply to each element in the range
    @param srcs iterators to the source ranges
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    while (first != last) {
      *first++ = callable(*src1++, *src2++, *src3++, ...);
    }
    @endcode
    */
    template <typename I, typename C, typename... S>
    cudaTask transform(I first, I last, C&& callable, S... srcs);
    
    // TODO: 
    //template <typename T, typename B>
    //cudaTask reduce(T* tgt, size_t N, T& init, B&& op);
    //
    
    // ------------------------------------------------------------------------
    // subflow
    // ------------------------------------------------------------------------
    
    /**
    @brief constructs a subflow graph through tf::cudaFlowCapturer

    @tparam C callable type constructible from 
              @c std::function<void(tf::cudaFlowCapturer&)>

    @param callable the callable to construct a capture flow

    @return a tf::cudaTask handle

    A captured subflow forms a sub-graph to the %cudaFlow and can be used to 
    capture custom (or third-party) kernels that cannot be directly constructed
    from the %cudaFlow.

    Example usage:

    @code{.cpp}
    taskflow.emplace([&](tf::cudaFlow& cf){
      
      tf::cudaTask my_kernel = cf.kernel(my_arguments);
      
      // create a flow capturer to capture custom kernels
      tf::cudaTask my_subflow = cf.capture([&](tf::cudaFlowCapturer& capturer){
        capturer.on([&](cudaStream_t stream){
          invoke_custom_kernel_with_stream(stream, custom_arguments);
        }); 
      });

      my_kernel.precede(my_subflow);
    });
    @endcode
     */
    template <typename C>
    cudaTask capture(C&& callable);

  private:

    handle_t _handle;
    
    cudaGraph& _graph;
    
    cudaGraphExec_t _executable {nullptr};
    
    cudaFlow(Executor&, cudaGraph&);
};

// Construct a standalone cudaFlow
inline cudaFlow::cudaFlow() :
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get<External>(_handle).graph} {
  
  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0), 
    "cudaFlow failed to create a native graph (external mode)"
  );
}

// Construct the cudaFlow from executor (internal graph)
inline cudaFlow::cudaFlow(Executor& e, cudaGraph& g) :
  _handle {std::in_place_type_t<Internal>{}, e},
  _graph  {g} {

  assert(_graph._native_handle == nullptr);

  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0), 
    "cudaFlow failed to create a native graph (internal mode)"
  );
}

// Destructor
inline cudaFlow::~cudaFlow() {
  if(_executable) {
    cudaGraphExecDestroy(_executable);
  }
  cudaGraphDestroy(_graph._native_handle);
  _graph._native_handle = nullptr;
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Procedure: dump
template <typename T>
void cudaFlow::dump(T& os) const {
  _graph.dump(os, nullptr, "");
}

// Procedure: dump
template <typename T>
void cudaFlow::dump_native_graph(T& os) const {
  cuda_dump_graph(os, _graph._native_handle);
}

// ----------------------------------------------------------------------------
// Graph building methods
// ----------------------------------------------------------------------------

// Function: noop
inline cudaTask cudaFlow::noop() {

  auto node = _graph.emplace_back( 
    _graph, std::in_place_type_t<cudaNode::Empty>{}
  );

  TF_CHECK_CUDA(
    cudaGraphAddEmptyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0
    ),
    "failed to create a no-operation (empty) node"
  );

  return cudaTask(node);
}

// Function: host
template <typename C>
cudaTask cudaFlow::host(C&& c) {
  
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Host>{}, std::forward<C>(c)
  );

  auto& h = std::get<cudaNode::Host>(node->_handle);

  cudaHostNodeParams p;
  p.fn = cudaNode::Host::callback;
  p.userData = &h;

  TF_CHECK_CUDA(
    ::cudaGraphAddHostNode(&node->_native_handle, _graph._native_handle, nullptr, 0, &p),
    "failed to create a host node"
  );
  
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
  using traits = function_traits<F>;

  static_assert(traits::arity == sizeof...(ArgsT), "arity mismatches");

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
  );
  
  cudaKernelNodeParams p;
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphAddKernelNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda kernel task"
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
  
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
  );
  
  cudaKernelNodeParams p;
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  cudaScopedDevice ctx(d);
  TF_CHECK_CUDA(
    ::cudaGraphAddKernelNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda kernel task on device ", d
  );

  return cudaTask(node);
}

// Function: zero
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaFlow::zero(T* dst, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  cudaMemsetParams p;
  p.dst = dst;
  p.value = 0;
  p.pitch = 0;
  p.elementSize = sizeof(T);  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset (zero) task"
  );

  return cudaTask(node);
}
    
// Function: fill
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), cudaTask>*
>
cudaTask cudaFlow::fill(T* dst, T value, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
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
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset (fill) task"
  );

  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {

  using U = std::decay_t<T>;

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );
  
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
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memcpy (copy) task"
  );

  return cudaTask(node);
}

// Function: memset
inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
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
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memset task"
  );
  
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );
  
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
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a cuda memcpy task"
  );

  return cudaTask(node);
}

// ------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------

// Function: update kernel parameters
template <typename... ArgsT>
void cudaFlow::update_kernel(
  cudaTask ct, dim3 g, dim3 b, size_t s, ArgsT&&... args
) {

  if(ct.type() != CUDA_KERNEL_TASK) {
    TF_THROW(ct, " is not a kernel task");
  }

  cudaKernelNodeParams p;
  
  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = std::get<cudaNode::Kernel>((ct._node)->_handle).func;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;
  
  //TF_CHECK_CUDA(
  //  cudaGraphKernelNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of kernel task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecKernelNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update kernel parameter on ", ct
  );
} 

// Function: update copy parameters
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void cudaFlow::update_copy(cudaTask ct, T* tgt, const T* src, size_t num) {
  
  if(ct.type() != CUDA_MEMCPY_TASK) {
    TF_THROW(ct, " is not a memcpy task");
  }

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

  //TF_CHECK_CUDA(
  //  cudaGraphMemcpyNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memcpy task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memcpy parameter on ", ct
  );
}

// Function: update memcpy parameters
inline void cudaFlow::update_memcpy(
  cudaTask ct, void* tgt, const void* src, size_t bytes
) {
  
  if(ct.type() != CUDA_MEMCPY_TASK) {
    TF_THROW(ct, " is not a memcpy task");
  }

  cudaMemcpy3DParms p;

  p.srcArray = nullptr;
  p.srcPos = ::make_cudaPos(0, 0, 0);
  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
  p.dstArray = nullptr;
  p.dstPos = ::make_cudaPos(0, 0, 0);
  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
  p.extent = ::make_cudaExtent(bytes, 1, 1);
  p.kind = cudaMemcpyDefault;

  //TF_CHECK_CUDA(
  //  cudaGraphMemcpyNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memcpy task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(_executable, ct._node->_native_handle, &p),
    "failed to update memcpy parameter on ", ct
  );
}

inline
void cudaFlow::update_memset(cudaTask ct, void* dst, int ch, size_t count) {

  if(ct.type() != CUDA_MEMSET_TASK) {
    TF_THROW(ct, " is not a memset task");
  }

  cudaMemsetParams p;
  p.dst = dst;
  p.value = ch;
  p.pitch = 0;
  //p.elementSize = (count & 1) == 0 ? ((count & 3) == 0 ? 4 : 2) : 1;
  //p.width = (count & 1) == 0 ? ((count & 3) == 0 ? count >> 2 : count >> 1) : count;
  p.elementSize = 1;  // either 1, 2, or 4
  p.width = count;
  p.height = 1;

  //TF_CHECK_CUDA(
  //  cudaGraphMemsetNodeSetParams(ct._node->_native_handle, &p),
  //  "failed to update a cudaGraph node of memset task"
  //);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memset parameter on ", ct
  );
}

// ----------------------------------------------------------------------------
// Generic Algorithm API
// ----------------------------------------------------------------------------
    
// Function: single_task
template <typename C>
cudaTask cudaFlow::single_task(C&& c) {
  return kernel(
    1, 1, 0, cuda_single_task<C>, std::forward<C>(c)
  );
}

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlow::for_each(I first, I last, C&& c) {
  
  size_t N = std::distance(first, last);
  size_t B = cuda_default_threads_per_block(N);
  
  // TODO: special case when N is 0?

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each<I, C>, first, N, std::forward<C>(c)
  );
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlow::for_each_index(I beg, I end, I inc, C&& c) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  // TODO: special case when N is 0?

  size_t N = distance(beg, end, inc);
  size_t B = cuda_default_threads_per_block(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_for_each_index<I, C>, beg, inc, N, std::forward<C>(c)
  );
}

// Function: transform
template <typename I, typename C, typename... S>
cudaTask cudaFlow::transform(I first, I last, C&& c, S... srcs) {
  
  // TODO: special case when N is 0?
  
  size_t N = std::distance(first, last);
  size_t B = cuda_default_threads_per_block(N);

  return kernel(
    (N+B-1) / B, B, 0, cuda_transform<I, C, S...>, 
    first, N, std::forward<C>(c), srcs...
  );
}

//template <typename T, typename B>>
//cudaTask cudaFlow::reduce(T* tgt, size_t N, T& init, B&& op) {
  //if(N == 0) {
    //return noop();
  //}
  //size_t B = cuda_default_threads_per_block(N);
//}

// ----------------------------------------------------------------------------
// captured flow 
// ----------------------------------------------------------------------------

// Function: capture
template <typename C>
cudaTask cudaFlow::capture(C&& c) {

  // insert a subflow node
  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Subflow>{}
  );
  
  // construct a captured flow from the callable
  auto& node_handle = std::get<cudaNode::Subflow>(node->_handle);
  cudaFlowCapturer capturer(node_handle.graph);

  c(capturer);
  
  // obtain the optimized captured graph
  auto captured = capturer._capture();
  //cuda_dump_graph(std::cout, captured);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, captured
    ), 
    "failed to add a cudaFlow capturer task"
  );
  
  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");

  return cudaTask(node);
}

// ----------------------------------------------------------------------------
// Offload methods
// ----------------------------------------------------------------------------

// Procedure: offload_until
template <typename P>
void cudaFlow::offload_until(P&& predicate) {

  //_executor->_invoke_cudaflow_task_internal(
  //  *this, std::forward<P>(predicate), false
  //);
  
  // transforms cudaFlow to a native cudaGraph under the specified device
  // and launches the graph through a given or an internal device stream
  if(_executable == nullptr) {
    TF_CHECK_CUDA(
      cudaGraphInstantiate(
        &_executable, _graph._native_handle, nullptr, nullptr, 0
      ),
      "failed to create an executable graph"
    );
    //cuda_dump_graph(std::cout, cf._graph._native_handle);
  }

  cudaScopedPerThreadStream s;

  while(!predicate()) {
    TF_CHECK_CUDA(
      cudaGraphLaunch(_executable, s), "failed to execute cudaFlow"
    );

    TF_CHECK_CUDA(
      cudaStreamSynchronize(s), "failed to synchronize cudaFlow execution"
    );
  }
}

// Procedure: offload_n
inline void cudaFlow::offload_n(size_t n) {
  offload_until([repeat=n] () mutable { return repeat-- == 0; });
}

// Procedure: offload
inline void cudaFlow::offload() {
  offload_until([repeat=1] () mutable { return repeat-- == 0; });
}

// ############################################################################
// Forward declaration: FlowBuilder
// ############################################################################
    
// FlowBuilder::emplace_on
template <typename C, typename D,
  std::enable_if_t<is_cudaflow_task_v<C>, void>*
>
Task FlowBuilder::emplace_on(C&& callable, D&& device) {
  auto n = _graph.emplace_back(
    std::in_place_type_t<Node::cudaFlowTask>{},
    [c=std::forward<C>(callable), d=std::forward<D>(device)]
    (Executor& executor, Node* node) mutable {
      cudaScopedDevice ctx(d);
      executor._invoke_cudaflow_task_entry(c, node);
    },
    std::make_unique<cudaGraph>()
  );
  return Task(n);
}

// FlowBuilder::emplace
template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>*>
Task FlowBuilder::emplace(C&& c) {
  return emplace_on(std::forward<C>(c), tf::cuda_get_device());
}

// ----------------------------------------------------------------------------
// Forward declaration: Executor
// ----------------------------------------------------------------------------

// Procedure: _invoke_cudaflow_task_entry
template <typename C,
  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlow&>, void>*
>
void Executor::_invoke_cudaflow_task_entry(C&& c, Node* node) {

  auto& h = std::get<Node::cudaFlowTask>(node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
  
  g->clear();

  cudaFlow cf(*this, *g);

  c(cf); 

  // join the cudaflow if never offloaded
  if(cf._executable == nullptr) {
    cf.offload();
  }
}

// Procedure: _invoke_cudaflow_task_entry
template <typename C, 
  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>*
>
void Executor::_invoke_cudaflow_task_entry(C&& c, Node* node) {

  auto& h = std::get<Node::cudaFlowTask>(node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
  
  g->clear();
  
  cudaFlowCapturer fc(*g);

  c(fc);
  
  auto captured = fc._capture();
  
  TF_CHECK_CUDA(
    cudaGraphInstantiate(
      &fc._executable, captured, nullptr, nullptr, 0
    ),
    "failed to create an executable graph"
  );
  
  cudaScopedPerThreadStream s;

  TF_CHECK_CUDA(cudaGraphLaunch(fc._executable, s), "failed to exec");
  TF_CHECK_CUDA(cudaStreamSynchronize(s), "failed to synchronize stream");
  TF_CHECK_CUDA(cudaGraphExecDestroy(fc._executable), "failed to destroy exec");

  fc._executable = nullptr;
  
  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");

  // TODO: how do we support the update?
}


}  // end of namespace tf -----------------------------------------------------


