#pragma once

#include "cuda_task.hpp"
#include "cuda_capturer.hpp"

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
The following example creates a %cudaFlow of two kernel tasks, @c task1 and 
@c task2, where @c task1 runs before @c task2.

@code{.cpp}
tf::Taskflow taskflow;
tf::Executor executor;

taskflow.emplace([&](tf::cudaFlow& cf){
  // create two kernel tasks 
  tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
  tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);
  
  // kernel1 runs before kernel2
  task1.precede(task2);
});

executor.run(taskflow).wait();
@endcode

A %cudaFlow is a task (tf::Task) created from tf::Taskflow 
and will be run by @em one worker thread in the executor.
That is, the callable that describes a %cudaFlow 
will be executed sequentially.
Inside a %cudaFlow task, different GPU tasks (tf::cudaTask) may run
in parallel scheduled by the CUDA runtime.

Please refer to @ref GPUTaskingcudaFlow for details.
*/
class cudaFlow {

  friend class Executor;

  struct External {
    cudaGraph graph;
  };

  struct Internal {
  };

  using handle_t = std::variant<External, Internal>;

  public:

    /**
    @brief constructs a standalone %cudaFlow

    A standalone %cudaFlow does not go through any taskflow and
    can be run by the caller thread using explicit offload methods 
    (e.g., tf::cudaFlow::offload).
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
    @brief queries the number of tasks
    */
    size_t num_tasks() const;

    /**
    @brief clears the %cudaFlow object
    */
    void clear();
    
    /**
    @brief dumps the %cudaFlow graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;
    
    /**
    @brief dumps the native CUDA graph into a DOT format through an
           output stream 

    The native CUDA graph may be different from the upper-level %cudaFlow 
    graph when flow capture is involved.
    */
    void dump_native_graph(std::ostream& os) const;

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
    
    @return a tf::cudaTask handle

    A host task can only execute CPU-specific functions and cannot do any CUDA calls 
    (e.g., @c cudaMalloc).
    */
    template <typename C>
    cudaTask host(C&& callable);
    
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
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
    
    /**
    @brief creates a kernel task on a specific GPU
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type
    
    @param d device identifier to launch the kernel
    @param g configured grid
    @param b configured block
    @param s configured shared memory size in bytes
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return a tf::cudaTask handle
    */
    template <typename F, typename... ArgsT>
    cudaTask kernel_on(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

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
    // offload methods
    // ------------------------------------------------------------------------
    
    /**
    @brief offloads the %cudaFlow onto a GPU and repeatedly runs it until 
    the predicate becomes true
    
    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the present %cudaFlow onto a GPU and
    repeatedly runs it until the predicate returns @c true.

    An offloaded %cudaFlow forces the underlying graph to be instantiated.
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
    for(auto itr = first; itr != last; itr++) {
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
    
    /**
    @brief performs parallel reduction over a range of items
    
    @tparam I input iterator type
    @tparam T value type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param result pointer to the result with an initialized value
    @param op binary reduction operator
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    cudaTask reduce(I first, I last, T* result, C&& op);
    
    /**
    @brief similar to tf::cudaFlow::reduce but does not assume any initial
           value to reduce
    
    This method is equivalent to the parallel execution of the following loop 
    on a GPU:
    
    @code{.cpp}
    *result = *first++;  // no initial values partitipcate in the loop
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    cudaTask uninitialized_reduce(I first, I last, T* result, C&& op);
    
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
    
    // ------------------------------------------------------------------------
    // update methods
    // ------------------------------------------------------------------------
  
    /**
    @brief updates parameters of a host task

    The method is similar to tf::cudaFlow::host but operates on a task
    of type tf::cudaTaskType::HOST.
    */
    template <typename C>
    void update_host(cudaTask task, C&& callable);

    /**
    @brief updates parameters of a kernel task

    The method is similar to tf::cudaFlow::kernel but operates on a task
    of type tf::cudaTaskType::KERNEL.
    The kernel function name must NOT change. 
    */
    template <typename... ArgsT>
    void update_kernel(cudaTask task, dim3 g, dim3 b, size_t shm, ArgsT&&... args);

    /**
    @brief updates parameters of a memcpy task to a copy task

    The method is similar to tf::cudaFlow::copy but operates on a task
    of type tf::cudaTaskType::MEMCPY.
    The source/destination memory may have different address values but 
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    template <typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void update_copy(cudaTask task, T* tgt, const T* src, size_t num);

    /**
    @brief updates parameters of a memcpy task
    
    The method is similar to tf::cudaFlow::memcpy but operates on a task
    of type tf::cudaTaskType::MEMCPY.     
    The source/destination memory may have different address values but 
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    void update_memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);

    /**
    @brief updates parameters of a memset task
    
    The method is similar to tf::cudaFlow::memset but operates on a task
    of type tf::cudaTaskType::MEMSET.
    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    void update_memset(cudaTask task, void* dst, int ch, size_t count);
    
    /**
    @brief updates parameters of a memset task to a fill task
    
    The method is similar to tf::cudaFlow::fill but operates on a task
    of type tf::cudaTaskType::MEMSET.

    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    template <typename T, std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
    >
    void update_fill(cudaTask task, T* dst, T value, size_t count);
    
    /**
    @brief updates parameters of a memset task to a zero task

    The method is similar to tf::cudaFlow::zero but operates on 
    a task of type tf::cudaTaskType::MEMSET.
    
    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original 
    source/destination memory.
    */
    template <typename T, std::enable_if_t<
      is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>* = nullptr
    >
    void update_zero(cudaTask task, T* dst, size_t count);
    
    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::for_each

    The type of the iterators and the callable must be the same as 
    the task created from tf::cudaFlow::for_each.
    */
    template <typename I, typename C>
    void update_for_each(cudaTask task, I first, I last, C&& callable);

    /**
    @brief updates parameters of a kernel task created from 
           tf::cudaFlow::for_each_index
    
    The type of the iterators and the callable must be the same as 
    the task created from tf::cudaFlow::for_each_index.
    */
    template <typename I, typename C>
    void update_for_each_index(
      cudaTask task, I first, I last, I step, C&& callable
    );
  
    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::transform of the same argument count
    
    The type of the iterators, callable, and source memory must 
    be the same as the task created from tf::cudaFlow::transform.
    */
    template <typename I, typename C, typename... S>
    void update_transform(
      cudaTask task, I first, I last, C&& callable, S... srcs
    );
    
    /**
    @brief updates parameters of a kernel task created from 
           tf::cudaFlow::reduce
    
    The type of the iterators, result, and callable must be the same as 
    the task created from tf::cudaFlow::reduce.
    */
    template <typename I, typename T, typename C>
    void update_reduce(cudaTask task, I first, I last, T* result, C&& op);
    
    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::uninitialized_reduce
    
    The type of the iterators, result, and callable must be the same as 
    the task created from tf::cudaFlow::uninitialized_reduce.
    */
    template <typename I, typename T, typename C>
    void update_uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C&& op
    );

  private:

    const size_t _MAX_BLOCK_SIZE;

    handle_t _handle;
    
    cudaGraph& _graph;
    
    cudaGraphExec_t _executable {nullptr};
    
    cudaFlow(cudaGraph&);

    size_t _default_block_size(size_t N) const;
};

// Construct a standalone cudaFlow
inline cudaFlow::cudaFlow() :
  _MAX_BLOCK_SIZE {
    cuda_get_device_max_threads_per_block(cuda_get_device())
  },
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get<External>(_handle).graph} {
  
  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0), 
    "cudaFlow failed to create a native graph (external mode)"
  );
}

// Construct the cudaFlow from executor (internal graph)
inline cudaFlow::cudaFlow(cudaGraph& g) :
  _MAX_BLOCK_SIZE {
    cuda_get_device_max_threads_per_block(cuda_get_device())
  },
  _handle {std::in_place_type_t<Internal>{}},
  _graph  {g} {

  assert(_graph._native_handle == nullptr);

  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0), 
    "failed to create a native graph (internal mode)"
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

// Function: _default_block_size
inline size_t cudaFlow::_default_block_size(size_t N) const {
  return N <= 32u ? 32u : std::min(_MAX_BLOCK_SIZE, next_pow2(N));
}

// Procedure: clear
inline void cudaFlow::clear() {

  if(_executable) {
    TF_CHECK_CUDA(
      cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
    );
    _executable = nullptr;
  }

  TF_CHECK_CUDA(
    cudaGraphDestroy(_graph._native_handle), "failed to destroy native graph"
  );
  
  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0), "failed to create native graph"
  );

  _graph._nodes.clear();
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _graph._nodes.empty();
}

// Function: num_tasks
inline size_t cudaFlow::num_tasks() const {
  return _graph._nodes.size();
}

// Procedure: dump
inline void cudaFlow::dump(std::ostream& os) const {
  _graph.dump(os, nullptr, "");
}

// Procedure: dump
inline void cudaFlow::dump_native_graph(std::ostream& os) const {
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
    cudaGraphAddHostNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a host node"
  );
  
  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
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
    "failed to create a kernel task"
  );

  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel_on(
  int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  
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
    cudaGraphAddKernelNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a kernel task on device ", d
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

  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a memset (zero) task"
  );

  return cudaTask(node);
}
    
// Function: fill
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaFlow::fill(T* dst, T value, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  auto p = cuda_get_fill_parms(dst, value, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a memset (fill) task"
  );

  return cudaTask(node);
}

// Function: copy
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );

  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a memcpy (copy) task"
  );

  return cudaTask(node);
}

// Function: memset
inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memset>{}
  );
  
  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a memset task"
  );
  
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {

  auto node = _graph.emplace_back(
    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
  );
  
  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _graph._native_handle, nullptr, 0, &p
    ),
    "failed to create a memcpy task"
  );

  return cudaTask(node);
}

// ------------------------------------------------------------------------
// update methods
// ------------------------------------------------------------------------

// Function: host
template <typename C>
void cudaFlow::update_host(cudaTask task, C&& c) {
  
  if(task.type() != cudaTaskType::HOST) {
    TF_THROW(task, " is not a host task");
  }

  auto& h = std::get<cudaNode::Host>(task._node->_handle);

  h.func = std::forward<C>(c);
}

// Function: update kernel parameters
template <typename... ArgsT>
void cudaFlow::update_kernel(
  cudaTask ct, dim3 g, dim3 b, size_t s, ArgsT&&... args
) {

  if(ct.type() != cudaTaskType::KERNEL) {
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
  
  TF_CHECK_CUDA(
    cudaGraphExecKernelNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update kernel parameters on ", ct
  );
} 

// Function: update copy parameters
template <
  typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void cudaFlow::update_copy(cudaTask ct, T* tgt, const T* src, size_t num) {
  
  if(ct.type() != cudaTaskType::MEMCPY) {
    TF_THROW(ct, " is not a memcpy task");
  }

  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memcpy parameters on ", ct
  );
}

// Function: update memcpy parameters
inline void cudaFlow::update_memcpy(
  cudaTask ct, void* tgt, const void* src, size_t bytes
) {
  
  if(ct.type() != cudaTaskType::MEMCPY) {
    TF_THROW(ct, " is not a memcpy task");
  }

  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(_executable, ct._node->_native_handle, &p),
    "failed to update memcpy parameters on ", ct
  );
}

// Procedure: update_memset
inline
void cudaFlow::update_memset(cudaTask ct, void* dst, int ch, size_t count) {

  if(ct.type() != cudaTaskType::MEMSET) {
    TF_THROW(ct, " is not a memset task");
  }

  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(
      _executable, ct._node->_native_handle, &p
    ),
    "failed to update memset parameters on ", ct
  );
}
    
// Procedure: update_fill
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaFlow::update_fill(cudaTask task, T* dst, T value, size_t count) {

  if(task.type() != cudaTaskType::MEMSET) {
    TF_THROW(task, " is not a memset task");
  }

  auto p = cuda_get_fill_parms(dst, value, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(
      _executable, task._node->_native_handle, &p
    ),
    "failed to update memset parameters on ", task
  );
}
    
// Procedure: update_zero
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaFlow::update_zero(cudaTask task, T* dst, size_t count) {

  if(task.type() != cudaTaskType::MEMSET) {
    TF_THROW(task, " is not a memset task");
  }
  
  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(
      _executable, task._node->_native_handle, &p
    ),
    "failed to update memset parameters on ", task
  );
}

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
Task FlowBuilder::emplace_on(C&& c, D&& d) {
  auto n = _graph.emplace_back(
    std::in_place_type_t<Node::cudaFlow>{},
    [c=std::forward<C>(c), d=std::forward<D>(d)] (Executor& e, Node* p) mutable {
      cudaScopedDevice ctx(d);
      e._invoke_cudaflow_task_entry(p, c);
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

// ############################################################################
// Forward declaration: Executor
// ############################################################################

// Procedure: _invoke_cudaflow_task_entry 
template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>*>
void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {
  
  using T = std::conditional_t<
    std::is_invocable_r_v<void, C, cudaFlow&>, cudaFlow, cudaFlowCapturer
  >;
  
  auto& h = std::get<Node::cudaFlow>(node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
  
  g->clear();

  T cf(*g);

  c(cf); 

  if(cf._executable == nullptr) {
    cf.offload();
  }
}

/*// Procedure: _invoke_cudaflow_task_entry (cudaFlow)
template <typename C,
  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlow&>, void>*
>
void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {

  auto& h = std::get<Node::cudaFlow>(node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
  
  g->clear();

  cudaFlow cf(*g);

  c(cf); 

  if(cf._executable == nullptr) {
    cf.offload();
  }
}

// Procedure: _invoke_cudaflow_task_entry (cudaFlowCapturer)
template <typename C, 
  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>*
>
void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {

  auto& h = std::get<Node::cudaFlow>(node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
  
  g->clear();
  
  cudaFlowCapturer fc(*g);

  c(fc);

  if(fc._executable == nullptr) {
    fc.offload();
  }
}*/


}  // end of namespace tf -----------------------------------------------------


