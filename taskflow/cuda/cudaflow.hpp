#pragma once

#include "../taskflow.hpp"
#include "cuda_task.hpp"
#include "cuda_capturer.hpp"

/**
@file taskflow/cuda/cudaflow.hpp
@brief cudaFlow include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: cudaFlow
// ----------------------------------------------------------------------------

/**
@class cudaFlow

@brief class to create a %cudaFlow task dependency graph

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
  
  public:

    /**
    @brief constructs a %cudaFlow
    */
    cudaFlow();

    /**
    @brief destroys the %cudaFlow and its associated native CUDA graph
           and executable graph
     */
    ~cudaFlow() = default;

    /**
    @brief default move constructor
    */
    cudaFlow(cudaFlow&&) = default;
    
    /**
    @brief default move assignment operator
    */
    cudaFlow& operator = (cudaFlow&&) = default;

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
    @brief updates parameters of a host task

    The method is similar to tf::cudaFlow::host but operates on a task
    of type tf::cudaTaskType::HOST.
    */
    template <typename C>
    void host(cudaTask task, C&& callable);

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
    @brief updates parameters of a kernel task

    The method is similar to tf::cudaFlow::kernel but operates on a task
    of type tf::cudaTaskType::KERNEL.
    The kernel function name must NOT change.
    */
    template <typename F, typename... ArgsT>
    void kernel(
      cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT... args
    );

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
    @brief updates parameters of a memset task

    The method is similar to tf::cudaFlow::memset but operates on a task
    of type tf::cudaTaskType::MEMSET.
    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original
    source/destination memory.
    */
    void memset(cudaTask task, void* dst, int ch, size_t count);

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
    @brief updates parameters of a memcpy task

    The method is similar to tf::cudaFlow::memcpy but operates on a task
    of type tf::cudaTaskType::MEMCPY.
    The source/destination memory may have different address values but
    must be allocated from the same contexts as the original
    source/destination memory.
    */
    void memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);

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
    void zero(cudaTask task, T* dst, size_t count);

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
    void fill(cudaTask task, T* dst, T value, size_t count);

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
    void copy(cudaTask task, T* tgt, const T* src, size_t num);

    // ------------------------------------------------------------------------
    // run method
    // ------------------------------------------------------------------------
    /**
    @brief offloads the %cudaFlow onto a GPU asynchronously via a stream

    @param stream stream for performing this operation

    Offloads the present %cudaFlow onto a GPU asynchronously via
    the given stream.

    An offloaded %cudaFlow forces the underlying graph to be instantiated.
    After the instantiation, you should not modify the graph topology
    but update node parameters.
    */
    void run(cudaStream_t stream);

    /**
    @brief acquires a reference to the underlying CUDA graph
    */
    cudaGraph_t native_graph();

    /**
    @brief acquires a reference to the underlying CUDA graph executable
    */
    cudaGraphExec_t native_executable();

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
    @brief updates a single-threaded kernel task

    This method is similar to cudaFlow::single_task but operates
    on an existing task.
    */
    template <typename C>
    void single_task(cudaTask task, C c);

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
    cudaTask for_each(I first, I last, C callable);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::for_each

    The type of the iterators and the callable must be the same as
    the task created from tf::cudaFlow::for_each.
    */
    template <typename I, typename C>
    void for_each(cudaTask task, I first, I last, C callable);

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
    cudaTask for_each_index(I first, I last, I step, C callable);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::for_each_index

    The type of the iterators and the callable must be the same as
    the task created from tf::cudaFlow::for_each_index.
    */
    template <typename I, typename C>
    void for_each_index(
      cudaTask task, I first, I last, I step, C callable
    );

    /**
    @brief applies a callable to a source range and stores the result in a target range

    @tparam I input iterator type
    @tparam O output iterator type
    @tparam C unary operator type

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
    template <typename I, typename O, typename C>
    cudaTask transform(I first, I last, O output, C op);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::transform

    The type of the iterators and the callable must be the same as
    the task created from tf::cudaFlow::for_each.
    */
    template <typename I, typename O, typename C>
    void transform(cudaTask task, I first, I last, O output, C c);

    /**
    @brief creates a task to perform parallel transforms over two ranges of items

    @tparam I1 first input iterator type
    @tparam I2 second input iterator type
    @tparam O output iterator type
    @tparam C unary operator type

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
    template <typename I1, typename I2, typename O, typename C>
    cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::transform

    The type of the iterators and the callable must be the same as
    the task created from tf::cudaFlow::for_each.
    */
    template <typename I1, typename I2, typename O, typename C>
    void transform(
      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
    );

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

    /**
    @brief updates the captured child graph

    The method is similar to tf::cudaFlow::capture but operates on a task
    of type tf::cudaTaskType::SUBFLOW.
    The new captured graph must be topologically identical to the original
    captured graph.
    */
    template <typename C>
    void capture(cudaTask task, C callable);

  private:

    cudaFlowGraph _cfg;
    cudaGraphExec _exe {nullptr};
};

// Construct a standalone cudaFlow
inline cudaFlow::cudaFlow() {
  _cfg._native_handle.create();
}

// Procedure: clear
inline void cudaFlow::clear() {
  _exe.clear();
  _cfg.clear();
  _cfg._native_handle.create();
}

// Function: empty
inline bool cudaFlow::empty() const {
  return _cfg._nodes.empty();
}

// Function: num_tasks
inline size_t cudaFlow::num_tasks() const {
  return _cfg._nodes.size();
}

// Procedure: dump
inline void cudaFlow::dump(std::ostream& os) const {
  _cfg.dump(os, nullptr, "");
}

// Procedure: dump
inline void cudaFlow::dump_native_graph(std::ostream& os) const {
  cuda_dump_graph(os, _cfg._native_handle);
}

// ----------------------------------------------------------------------------
// Graph building methods
// ----------------------------------------------------------------------------

// Function: noop
inline cudaTask cudaFlow::noop() {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Empty>{}
  );

  TF_CHECK_CUDA(
    cudaGraphAddEmptyNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0
    ),
    "failed to create a no-operation (empty) node"
  );

  return cudaTask(node);
}

// Function: host
template <typename C>
cudaTask cudaFlow::host(C&& c) {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Host>{}, std::forward<C>(c)
  );

  auto h = std::get_if<cudaFlowNode::Host>(&node->_handle);

  cudaHostNodeParams p;
  p.fn = cudaFlowNode::Host::callback;
  p.userData = h;

  TF_CHECK_CUDA(
    cudaGraphAddHostNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
    ),
    "failed to create a host node"
  );

  return cudaTask(node);
}

// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlow::kernel(
  dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Kernel>{}, (void*)f
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
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
    ),
    "failed to create a kernel task"
  );

  return cudaTask(node);
}

// Function: zero
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
cudaTask cudaFlow::zero(T* dst, size_t count) {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
  );

  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
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

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
  );

  auto p = cuda_get_fill_parms(dst, value, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
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

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Memcpy>{}
  );

  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
    ),
    "failed to create a memcpy (copy) task"
  );

  return cudaTask(node);
}

// Function: memset
inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
  );

  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphAddMemsetNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
    ),
    "failed to create a memset task"
  );

  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {

  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Memcpy>{}
  );

  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphAddMemcpyNode(
      &node->_native_handle, _cfg._native_handle, nullptr, 0, &p
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
void cudaFlow::host(cudaTask task, C&& c) {

  if(task.type() != cudaTaskType::HOST) {
    TF_THROW(task, " is not a host task");
  }

  auto h = std::get_if<cudaFlowNode::Host>(&task._node->_handle);

  h->func = std::forward<C>(c);
}

// Function: update kernel parameters
template <typename F, typename... ArgsT>
void cudaFlow::kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT... args
) {

  if(task.type() != cudaTaskType::KERNEL) {
    TF_THROW(task, " is not a kernel task");
  }

  cudaKernelNodeParams p;

  void* arguments[sizeof...(ArgsT)] = { (void*)(&args)... };
  p.func = (void*)f;
  p.gridDim = g;
  p.blockDim = b;
  p.sharedMemBytes = s;
  p.kernelParams = arguments;
  p.extra = nullptr;

  TF_CHECK_CUDA(
    cudaGraphExecKernelNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update kernel parameters on ", task
  );
}

// Function: update copy parameters
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
void cudaFlow::copy(cudaTask task, T* tgt, const T* src, size_t num) {

  if(task.type() != cudaTaskType::MEMCPY) {
    TF_THROW(task, " is not a memcpy task");
  }

  auto p = cuda_get_copy_parms(tgt, src, num);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Function: update memcpy parameters
inline void cudaFlow::memcpy(
  cudaTask task, void* tgt, const void* src, size_t bytes
) {

  if(task.type() != cudaTaskType::MEMCPY) {
    TF_THROW(task, " is not a memcpy task");
  }

  auto p = cuda_get_memcpy_parms(tgt, src, bytes);

  TF_CHECK_CUDA(
    cudaGraphExecMemcpyNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update memcpy parameters on ", task
  );
}

// Procedure: memset
inline void cudaFlow::memset(cudaTask task, void* dst, int ch, size_t count) {

  if(task.type() != cudaTaskType::MEMSET) {
    TF_THROW(task, " is not a memset task");
  }

  auto p = cuda_get_memset_parms(dst, ch, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: fill
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaFlow::fill(cudaTask task, T* dst, T value, size_t count) {

  if(task.type() != cudaTaskType::MEMSET) {
    TF_THROW(task, " is not a memset task");
  }

  auto p = cuda_get_fill_parms(dst, value, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update memset parameters on ", task
  );
}

// Procedure: zero
template <typename T, std::enable_if_t<
  is_pod_v<T> && (sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4), void>*
>
void cudaFlow::zero(cudaTask task, T* dst, size_t count) {

  if(task.type() != cudaTaskType::MEMSET) {
    TF_THROW(task, " is not a memset task");
  }

  auto p = cuda_get_zero_parms(dst, count);

  TF_CHECK_CUDA(
    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
    "failed to update memset parameters on ", task
  );
}

// Function: capture
template <typename C>
void cudaFlow::capture(cudaTask task, C c) {

  if(task.type() != cudaTaskType::SUBFLOW) {
    TF_THROW(task, " is not a subflow task");
  }

  // insert a subflow node
  // construct a captured flow from the callable
  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&task._node->_handle);
  //node_handle->graph.clear();

  cudaFlowCapturer capturer;
  c(capturer);

  // obtain the optimized captured graph
  capturer._cfg._native_handle.reset(capturer.capture());
  node_handle->cfg = std::move(capturer._cfg);

  TF_CHECK_CUDA(
    cudaGraphExecChildGraphNodeSetParams(
      _exe, 
      task._node->_native_handle, 
      node_handle->cfg._native_handle
    ),
    "failed to update a captured child graph"
  );
}

// ----------------------------------------------------------------------------
// captured flow
// ----------------------------------------------------------------------------

// Function: capture
template <typename C>
cudaTask cudaFlow::capture(C&& c) {

  // insert a subflow node
  auto node = _cfg.emplace_back(
    _cfg, std::in_place_type_t<cudaFlowNode::Subflow>{}
  );

  // construct a captured flow from the callable
  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&node->_handle);

  // perform capturing
  cudaFlowCapturer capturer;
  c(capturer);

  // obtain the optimized captured graph
  capturer._cfg._native_handle.reset(capturer.capture());

  // move capturer's cudaFlow graph into node
  node_handle->cfg = std::move(capturer._cfg);

  TF_CHECK_CUDA(
    cudaGraphAddChildGraphNode(
      &node->_native_handle, 
      _cfg._native_handle, 
      nullptr, 
      0, 
      node_handle->cfg._native_handle
    ), 
    "failed to add a cudaFlow capturer task"
  );

  return cudaTask(node);
}

// ----------------------------------------------------------------------------
// run method
// ----------------------------------------------------------------------------

// Procedure: run
inline void cudaFlow::run(cudaStream_t stream) {
  if(!_exe) {
    _exe.instantiate(_cfg._native_handle);
  }
  _exe.launch(stream);
  _cfg._state = cudaFlowGraph::OFFLOADED;
}

// Function: native_cfg
inline cudaGraph_t cudaFlow::native_graph() {
  return _cfg._native_handle;
}

// Function: native_executable
inline cudaGraphExec_t cudaFlow::native_executable() {
  return _exe;
}

}  // end of namespace tf -----------------------------------------------------


