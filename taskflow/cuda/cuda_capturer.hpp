#pragma once

#include "cuda_task.hpp"
#include "cuda_optimizer.hpp"

/** 
@file cuda_capturer.hpp
@brief %cudaFlow capturer include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// class definition: cudaFlowCapturer
// ----------------------------------------------------------------------------

/**
@class cudaFlowCapturer

@brief class for building a CUDA task dependency graph through stream capture

The usage of tf::cudaFlowCapturer is similar to tf::cudaFlow, except users can
call the method tf::cudaFlowCapturer::on to capture a sequence of asynchronous 
CUDA operations through the given stream.
The following example creates a CUDA graph that captures two kernel tasks,
@c task_1 and @c task_2, where @c task_1 runs before @c task_2. 

@code{.cpp}
taskflow.emplace([](tf::cudaFlowCapturer& capturer){

  // capture my_kernel_1 through the given stream managed by the capturer
  auto task_1 = capturer.on([&](cudaStream_t stream){ 
    my_kernel_1<<<grid_1, block_1, shm_size_1, stream>>>(my_parameters_1);
  });

  // capture my_kernel_2 through the given stream managed by the capturer
  auto task_2 = capturer.on([&](cudaStream_t stream){ 
    my_kernel_2<<<grid_2, block_2, shm_size_2, stream>>>(my_parameters_2);
  });

  task_1.precede(task_2);
});
@endcode

Similar to tf::cudaFlow, a %cudaFlowCapturer is a task (tf::Task) 
created from tf::Taskflow 
and will be run by @em one worker thread in the executor.
That is, the callable that describes a %cudaFlowCapturer 
will be executed sequentially.
Inside a %cudaFlow capturer task, different GPU tasks (tf::cudaTask) may run
in parallel depending on the selected optimization algorithm.
By default, we use tf::cudaSequentialCapturing to generate a sequential
CUDA graph.

Please refer to @ref GPUTaskingcudaFlowCapturer for details.
*/
class cudaFlowCapturer {

  friend class cudaFlow;
  friend class Executor;

  struct External {
    cudaGraph graph;
  };

  struct Internal {
  };

  using handle_t = std::variant<External, Internal>;

  using Optimizer = std::variant<
    cudaRoundRobinCapturing,
    cudaSequentialCapturing,
    cudaLinearCapturing
  >;

  public:
    
    /**
    @brief constrcts a standalone cudaFlowCapturer
    
    A standalone %cudaFlow capturer does not go through any taskflow and
    can be run by the caller thread using explicit offload methods 
    (e.g., tf::cudaFlow::offload).
    */
    cudaFlowCapturer();
    
    /**
    @brief destructs the cudaFlowCapturer
    */
    virtual ~cudaFlowCapturer();
    
    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;

    /**
    @brief queries the number of tasks
    */
    size_t num_tasks() const;

    /**
    @brief clear this %cudaFlow capturer
    */
    void clear();
   
    /**
    @brief dumps the capture graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;
    
    
    /**
    @brief selects a different optimization algorithm

    @tparam OPT optimizer type
    @tparam ArgsT arguments types

    @param args arguments to forward to construct the optimizer

    @return a reference to the optimizer

    We currently supports the following optimization algorithms to capture
    a user-described %cudaFlow:
      + tf::cudaSequentialCapturing
      + tf::cudaRoundRobinCapturing
    */
    template <typename OPT, typename... ArgsT>
    OPT& make_optimizer(ArgsT&&... args);

    // ------------------------------------------------------------------------
    // basic methods
    // ------------------------------------------------------------------------
    
    /**
    @brief captures a sequential CUDA operations from the given callable
    
    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
    @param callable a callable to capture CUDA operations with the stream

    This methods applies a stream created by the flow to capture 
    a sequence of CUDA operations defined in the callable.
    */
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
    >
    cudaTask on(C&& callable);
    
    /**
    @brief copies data between host and device asynchronously through a stream
    
    @param dst destination memory address
    @param src source memory address
    @param count size in bytes to copy
    
    The method captures a @c cudaMemcpyAsync operation through an 
    internal stream.
    */ 
    cudaTask memcpy(void* dst, const void* src, size_t count);

    /**
    @brief captures a copy task of typed data
    
    @tparam T element type (non-void)

    @param tgt pointer to the target memory block
    @param src pointer to the source memory block
    @param num number of elements to copy

    @return cudaTask handle

    A copy task transfers <tt>num*sizeof(T)</tt> bytes of data from a source location
    to a target location. Direction can be arbitrary among CPUs and GPUs.
    */
    template <typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    cudaTask copy(T* tgt, const T* src, size_t num);

    /**
    @brief initializes or sets GPU memory to the given value byte by byte
    
    @param ptr pointer to GPU mempry
    @param v value to set for each byte of the specified memory
    @param n size in bytes to set
    
    The method captures a @c cudaMemsetAsync operation through an
    internal stream to fill the first @c count bytes of the memory area 
    pointed to by @c devPtr with the constant byte value @c value.
    */ 
    cudaTask memset(void* ptr, int v, size_t n);

    /**
    @brief captures a kernel
    
    @tparam F kernel function type
    @tparam ArgsT kernel function parameters type

    @param g configured grid
    @param b configured block
    @param s configured shared memory size in bytes
    @param f kernel function
    @param args arguments to forward to the kernel function by copy

    @return cudaTask handle
    */ 
    template <typename F, typename... ArgsT>
    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);

    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------
    
    /**
    @brief capturers a kernel to runs the given callable with only one thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C callable);
    
    /**
    @brief captures a kernel that applies a callable to each dereferenced element 
           of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning 
    @param last iterator to the end 
    @param callable a callable object to apply to the dereferenced iterator 
    
    @return cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    for(auto itr = first; itr != last; i++) {
      callable(*itr);
    }
    @endcode
    */
    template <typename I, typename C>
    cudaTask for_each(I first, I last, C callable);

    /**
    @brief captures a kernel that applies a callable to each index in the range 
           with the step size
    
    @tparam I index type
    @tparam C callable type
    
    @param first beginning index
    @param last last index
    @param step step size
    @param callable the callable to apply to each element in the data array
    
    @return cudaTask handle
    
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
    @brief captures a kernel that transforms an input range to an output range 
    
    @tparam I input iterator type
    @tparam O output iterator type
    @tparam C unary operator type

    @param first iterator to the beginning of the input range
    @param last iterator to the end of the input range
    @param output iterator to the beginning of the output range
    @param op unary operator to apply to transform each item in the range
    
    @return cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    while (first != last) {
      *output++ = op(*first++);
    }
    @endcode
    */
    template <typename I, typename O, typename C>
    cudaTask transform(I first, I last, O output, C op);
    
    /**
    @brief captures a kernel that transforms two input ranges to an output range 
    
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
    @brief captures kernels that perform parallel reduction over a range of items

    @tparam I input iterator type
    @tparam T value type
    @tparam C binary operator type

    @param first iterator to the beginning 
    @param last iterator to the end 
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
    cudaTask reduce(I first, I last, T* result, C op);
    
    /**
    @brief similar to tf::cudaFlowCapturer::reduce but does not assume
           any initial value to reduce
    
    This method is equivalent to the parallel execution of the following loop 
    on a GPU:
    
    @code{.cpp}
    *result = *first++;  // initial value does not involve in the loop
    while (first != last) {
      *result = op(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename C>
    cudaTask uninitialized_reduce(I first, I last, T* result, C op);
    
    /**
    @brief captures kernels that perform parallel reduction over a range of 
           transformed items

    @tparam I input iterator type
    @tparam T value type
    @tparam C binary operator type
    @tparam U unary operator type

    @param first iterator to the beginning 
    @param last iterator to the end 
    @param result pointer to the result with an initialized value
    @param bop binary reduce operator
    @param uop unary transform operator
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename C, typename U>
    cudaTask transform_reduce(I first, I last, T* result, C bop, U uop);
    
    /**
    @brief similar to tf::cudaFlowCapturer::transform_reduce but does not assume
           any initial value to reduce
    
    This method is equivalent to the parallel execution of the following loop 
    on a GPU:
    
    @code{.cpp}
    *result = uop(*first++);  // initial value does not involve in the loop
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename C, typename U>
    cudaTask transform_uninitialized_reduce(I first, I last, T* result, C bop, U uop);
    
    /**
    @brief captures kernels that perform parallel inclusive scan 
           over a range of items

    @tparam I input iterator type
    @tparam O output iterator type
    @tparam C binary operator type

    @param first iterator to the beginning 
    @param last iterator to the end 
    @param output iterator to the beginning of the output
    @param op binary operator
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop on a GPU:
    
    @code{.cpp}
    for(size_t i=0; i<std::distance(first, last); i++) {
      *(output + i) = i ? op(*(first+i), *(output+i-1)) : *(first+i);
    }
    @endcode
    */
    template <typename I, typename O, typename C>
    cudaTask inclusive_scan(I first, I last, O output, C op);
    
    /**
    @brief similar to cudaFlowCapturer::inclusive_scan 
           but excludes the first value
    */
    template <typename I, typename O, typename C>
    cudaTask exclusive_scan(I first, I last, O output, C op);
    
    /**
    @brief captures kernels that perform parallel inclusive scan 
           over a range of transformed items
    
    @tparam I input iterator type
    @tparam O output iterator type
    @tparam B binary operator type
    @tparam U unary operator type

    @param first iterator to the beginning 
    @param last iterator to the end 
    @param output iterator to the beginning of the output
    @param bop binary operator
    @param uop unary operator
    
    @return a tf::cudaTask handle
    
    This method is equivalent to the parallel execution of the following loop
    on a GPU:
    
    @code{.cpp}
    for(size_t i=0; i<std::distance(first, last); i++) {
      *(output + i) = i ? op(uop(*(first+i)), *(output+i-1)) : uop(*(first+i));
    }
    @endcode
     */
    template <typename I, typename O, typename B, typename U>
    cudaTask transform_inclusive_scan(I first, I last, O output, B bop, U uop);
    
    /**
    @brief similar to cudaFlowCapturer::transform_inclusive_scan but 
           excludes the first value
    */
    template <typename I, typename O, typename B, typename U>
    cudaTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);
    
    /**
    @brief captures kernels that perform parallel merge on two sorted arrays
    
    @tparam A iterator type of the first input array
    @tparam B iterator type of the second input array
    @tparam C iterator type of the output array
    @tparam Comp comparator type

    @param a_first iterator to the beginning of the first input array
    @param a_last iterator to the end of the first input array
    @param b_first iterator to the beginning of the second input array
    @param b_last iterator to the end of the second input array
    @param c_first iterator to the beginning of the output array
    @param comp binary comparator
    
    @return a tf::cudaTask handle

    Merges two sorted ranges <tt>[a_first, a_last)</tt> and 
    <tt>[b_first, b_last)</tt> into one sorted range beginning at @c c_first.

    A sequence is said to be sorted with respect to a comparator @c comp 
    if for any iterator it pointing to the sequence and 
    any non-negative integer @c n such that <tt>it + n</tt> is a valid iterator 
    pointing to an element of the sequence, <tt>comp(*(it + n), *it)</tt> 
    evaluates to false.
     */
    template <typename A, typename B, typename C, typename Comp>
    cudaTask merge(A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp);
    
    /**
    @brief captures kernels that sort the given array
    
    @tparam I iterator type of the first input array
    @tparam C comparator type

    @param first iterator to the beginning of the input array
    @param last iterator to the end of the input array
    @param comp binary comparator
    
    @return a tf::cudaTask handle

    Sorts elements in the range <tt>[first, last)</tt> 
    with the given comparator.
     */
    template <typename I, typename C>
    cudaTask sort(I first, I last, C comp);

    // ------------------------------------------------------------------------
    // rebind methods to update captured tasks
    // ------------------------------------------------------------------------

    /**
    @brief rebinds a capture task to another sequential CUDA operations

    The method is similar to cudaFlowCapturer::on but operates 
    on an existing task.
    */
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
    >
    void rebind_on(cudaTask task, C&& callable);
    
    /**
    @brief rebinds a capture task to a memcpy operation

    The method is similar to cudaFlowCapturer::memcpy but operates on an
    existing task.
    */
    void rebind_memcpy(cudaTask task, void* dst, const void* src, size_t count);

    /**
    @brief rebinds a capture task to a copy operation

    The method is similar to cudaFlowCapturer::copy but operates on 
    an existing task.
    */
    template <typename T, 
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void rebind_copy(cudaTask task, T* tgt, const T* src, size_t num);

    /**
    @brief rebinds a capture task to a memset operation

    The method is similar to cudaFlowCapturer::memset but operates on
    an existing task.
    */
    void rebind_memset(cudaTask task, void* ptr, int value, size_t n);

    /**
    @brief rebinds a capture task to a kernel operation

    The method is similar to cudaFlowCapturer::kernel but operates on
    an existing task.
    */
    template <typename F, typename... ArgsT>
    void rebind_kernel(
      cudaTask task, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
    );
    
    /**
    @brief rebinds a capture task to a single-threaded kernel
    
    This method is similar to cudaFlowCapturer::single_task but operates
    on an existing task.
    */
    template <typename C>
    void rebind_single_task(cudaTask task, C callable);
    
    /**
    @brief rebinds a capture task to a for-each kernel task
    
    This method is similar to cudaFlowCapturer::for_each but operates
    on an existing task.
    */
    template <typename I, typename C>
    void rebind_for_each(cudaTask task, I first, I last, C callable);

    /**
    @brief rebinds a capture task to a for-each-index kernel task

    This method is similar to cudaFlowCapturer::for_each_index but operates
    on an existing task.
    */
    template <typename I, typename C>
    void rebind_for_each_index(
      cudaTask task, I first, I last, I step, C callable
    );
  
    /**
    @brief rebinds a capture task to a transform kernel task

    This method is similar to cudaFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I, typename O, typename C>
    void rebind_transform(cudaTask task, I first, I last, O output, C op);
    
    /**
    @brief rebinds a capture task to a transform kernel task

    This method is similar to cudaFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I1, typename I2, typename O, typename C>
    void rebind_transform(
      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
    );
      
    /**
    @brief rebinds a capture task to a reduction task

    This method is similar to cudaFlowCapturer::reduce but operates
    on an existing task.
    */
    template <typename I, typename T, typename C>
    void rebind_reduce(cudaTask task, I first, I last, T* result, C op);
    
    /**
    @brief rebinds a capture task to an uninitialized-reduction task

    This method is similar to cudaFlowCapturer::uninitialized_reduce
    but operates on an existing task.
    */
    template <typename I, typename T, typename C>
    void rebind_uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C op
    );
    
    /**
    @brief rebinds a capture task to a transform-reduce task

    This method is similar to cudaFlowCapturer::transform_reduce but 
    operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void rebind_transform_reduce(
      cudaTask task, I first, I last, T* result, C bop, U uop
    );
    
    /**
    @brief rebinds a capture task to a transform-reduce task of no initialized value

    This method is similar to cudaFlowCapturer::transform_uninitialized_reduce 
    but operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void rebind_transform_uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C bop, U uop
    );
    
    /**
    @brief rebinds a capture task to a merge task

    This method is similar to cudaFlowCapturer::merge but operates
    on an existing task.
     */ 
    template <typename A, typename B, typename C, typename Comp>
    void rebind_merge(
      cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
    );
    
    /**
    @brief rebinds a capture task to a sort task

    This method is similar to cudaFlowCapturer::sort but operates on 
    an existing task.
     */
    template <typename I, typename C>
    void rebind_sort(cudaTask task, I first, I last, C comp);
    
    // ------------------------------------------------------------------------
    // offload methods
    // ------------------------------------------------------------------------

    /**
    @brief offloads the captured %cudaFlow onto a GPU and repeatedly runs it until 
    the predicate becomes true
    
    @tparam P predicate type (a binary callable)

    @param predicate a binary predicate (returns @c true for stop)

    Immediately offloads the %cudaFlow captured so far onto a GPU and
    repeatedly runs it until the predicate returns @c true.

    By default, if users do not offload the %cudaFlow capturer, 
    the executor will offload it once.
    */
    template <typename P>
    void offload_until(P&& predicate);

    /**
    @brief offloads the captured %cudaFlow and executes it by the given times

    @param n number of executions
    */
    void offload_n(size_t n);

    /**
    @brief offloads the captured %cudaFlow and executes it once
    */
    void offload();

  private:

    const size_t _MAX_BLOCK_SIZE;
    
    handle_t _handle;

    cudaGraph& _graph;
    
    Optimizer _optimizer;

    cudaGraphExec_t _executable {nullptr};
    
    cudaFlowCapturer(cudaGraph&);

    cudaGraph_t _capture();

    void _destroy_executable();

    size_t _default_block_size(size_t) const;

};

// constructs a cudaFlow capturer from a taskflow
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
  _MAX_BLOCK_SIZE {
    cuda_get_device_max_threads_per_block(cuda_get_device())
  },
  _handle {std::in_place_type_t<Internal>{}},
  _graph  {g} {
}

// constructs a standalone cudaFlow capturer
inline cudaFlowCapturer::cudaFlowCapturer() : 
  _MAX_BLOCK_SIZE {
    cuda_get_device_max_threads_per_block(cuda_get_device())
  },
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get<External>(_handle).graph} {
}

inline cudaFlowCapturer::~cudaFlowCapturer() {

  if(_executable != nullptr) {
    cudaGraphExecDestroy(_executable);
  }
}

// Function: _default_block_size
inline size_t cudaFlowCapturer::_default_block_size(size_t N) const {
  return N <= 32u ? 32u : std::min(_MAX_BLOCK_SIZE, next_pow2(N));
}

// Function: empty
inline bool cudaFlowCapturer::empty() const {
  return _graph.empty();
}

// Function: num_tasks
inline size_t cudaFlowCapturer::num_tasks() const {
  return _graph._nodes.size();
}

// Procedure: clear
inline void cudaFlowCapturer::clear() {
  _destroy_executable();
  _graph._nodes.clear();
}

// Procedure: dump
inline void cudaFlowCapturer::dump(std::ostream& os) const {
  _graph.dump(os, nullptr, "");
}

// Procedure: _destroy_executable
inline void cudaFlowCapturer::_destroy_executable() {
  if(_executable != nullptr) {
    TF_CHECK_CUDA(
      cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
    );
    _executable = nullptr;
  }
}

// Function: capture
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
>
cudaTask cudaFlowCapturer::on(C&& callable) {
  auto node = _graph.emplace_back(_graph,
    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlowCapturer::memcpy(
  void* dst, const void* src, size_t count
) {
  return on([dst, src, count] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}
    
template <typename T, std::enable_if_t<!std::is_same_v<T, void>, void>*>
cudaTask cudaFlowCapturer::copy(T* tgt, const T* src, size_t num) {
  return on([tgt, src, num] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline cudaTask cudaFlowCapturer::memset(void* ptr, int v, size_t n) {
  return on([ptr, v, n] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}
    
// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlowCapturer::kernel(
  dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  return on([g, b, s, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: _capture
inline cudaGraph_t cudaFlowCapturer::_capture() {
  return std::visit(
    [this](auto&& opt){ return opt._optimize(_graph); }, _optimizer
  );
}

// Procedure: offload_until
template <typename P>
void cudaFlowCapturer::offload_until(P&& predicate) {

  // If the executable graph does not exist, instantiate it 
  if(_executable == nullptr) {

    auto captured = _capture();
    
    TF_CHECK_CUDA(
      cudaGraphInstantiate(&_executable, captured, nullptr, nullptr, 0),
      "failed to create an executable graph"
    );

    // TODO: store the native graph?
    TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
  }
  
  cudaScopedPerThreadStream s;

  while(!predicate()) {
    TF_CHECK_CUDA(
      cudaGraphLaunch(_executable, s), "failed to launch the exec graph"
    );

    TF_CHECK_CUDA(cudaStreamSynchronize(s), "failed to synchronize stream");
  }
}

// Procedure: offload_n
inline void cudaFlowCapturer::offload_n(size_t n) {
  offload_until([repeat=n] () mutable { return repeat-- == 0; });
}

// Procedure: offload
inline void cudaFlowCapturer::offload() {
  offload_until([repeat=1] () mutable { return repeat-- == 0; });
}

// Function: rebind_on
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>* 
>
void cudaFlowCapturer::rebind_on(cudaTask task, C&& callable) {
  
  if(task.type() != cudaTaskType::CAPTURE) {
    TF_THROW("invalid cudaTask type (must be CAPTURE)");
  }
  
  _destroy_executable();

  std::get<cudaNode::Capture>((task._node)->_handle).work = std::forward<C>(callable);
}

// Function: rebind_memcpy
inline void cudaFlowCapturer::rebind_memcpy(
  cudaTask task, void* dst, const void* src, size_t count
) {
  rebind_on(task, [dst, src, count](cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}

// Function: rebind_copy
template <typename T, 
  std::enable_if_t<!std::is_same_v<T, void>, void>* 
>
void cudaFlowCapturer::rebind_copy(
  cudaTask task, T* tgt, const T* src, size_t num
) {
  rebind_on(task, [tgt, src, num] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: rebind_memset
inline void cudaFlowCapturer::rebind_memset(
  cudaTask task, void* ptr, int v, size_t n
) {
  rebind_on(task, [ptr, v, n] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}

// Function: rebind_kernel
template <typename F, typename... ArgsT>
void cudaFlowCapturer::rebind_kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
) {
  rebind_on(task, [g, b, s, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: make_optimizer
template <typename OPT, typename ...ArgsT>
OPT& cudaFlowCapturer::make_optimizer(ArgsT&&... args) {
  return _optimizer.emplace<OPT>(std::forward<ArgsT>(args)...);
}

}  // end of namespace tf -----------------------------------------------------

