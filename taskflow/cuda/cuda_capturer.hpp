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

@brief class to create a %cudaFlow graph using stream capture

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
By default, we use tf::cudaFlowRoundRobinOptimizer to transform a user-level
graph into a native CUDA graph.

Please refer to @ref GPUTaskingcudaFlowCapturer for details.
*/
class cudaFlowCapturer {

  friend class cudaFlow;
  friend class Executor;

  // created by user
  struct External {
    cudaFlowGraph graph;
  };
  
  // created from cudaFlow
  struct Internal {
  };

  using handle_t = std::variant<External, Internal>;

  using Optimizer = std::variant<
    cudaFlowRoundRobinOptimizer,
    cudaFlowSequentialOptimizer,
    cudaFlowLinearOptimizer
  >;

  public:

    /**
    @brief constructs a standalone cudaFlowCapturer

    A standalone %cudaFlow capturer does not go through any taskflow and
    can be run by the caller thread using tf::cudaFlowCapturer::run.
    */
    cudaFlowCapturer() = default;

    /**
    @brief destructs the cudaFlowCapturer
    */
    ~cudaFlowCapturer() = default;
    
    /**
    @brief default move constructor
    */
    cudaFlowCapturer(cudaFlowCapturer&&) = default;
    
    /**
    @brief default move assignment operator
    */
    cudaFlowCapturer& operator = (cudaFlowCapturer&&) = default;

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
    @brief dumps the %cudaFlow graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;

    /**
    @brief dumps the native captured graph into a DOT format through 
           an output stream
    */
    void dump_native_graph(std::ostream& os) const;

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
    @brief updates a capture task to another sequential CUDA operations

    The method is similar to cudaFlowCapturer::on but operates
    on an existing task.
    */
    template <typename C, std::enable_if_t<
      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
    >
    void on(cudaTask task, C&& callable);

    /**
    @brief captures a no-operation task

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
    @brief updates a task to a no-operation task

    The method is similar to tf::cudaFlowCapturer::noop but
    operates on an existing task.
    */
    void noop(cudaTask task);

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
    @brief updates a capture task to a memcpy operation

    The method is similar to cudaFlowCapturer::memcpy but operates on an
    existing task.
    */
    void memcpy(cudaTask task, void* dst, const void* src, size_t count);

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
    @brief updates a capture task to a copy operation

    The method is similar to cudaFlowCapturer::copy but operates on
    an existing task.
    */
    template <typename T,
      std::enable_if_t<!std::is_same_v<T, void>, void>* = nullptr
    >
    void copy(cudaTask task, T* tgt, const T* src, size_t num);

    /**
    @brief initializes or sets GPU memory to the given value byte by byte

    @param ptr pointer to GPU memory
    @param v value to set for each byte of the specified memory
    @param n size in bytes to set

    The method captures a @c cudaMemsetAsync operation through an
    internal stream to fill the first @c count bytes of the memory area
    pointed to by @c devPtr with the constant byte value @c value.
    */
    cudaTask memset(void* ptr, int v, size_t n);

    /**
    @brief updates a capture task to a memset operation

    The method is similar to cudaFlowCapturer::memset but operates on
    an existing task.
    */
    void memset(cudaTask task, void* ptr, int value, size_t n);

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
    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);

    /**
    @brief updates a capture task to a kernel operation

    The method is similar to cudaFlowCapturer::kernel but operates on
    an existing task.
    */
    template <typename F, typename... ArgsT>
    void kernel(
      cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
    );

    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------

    /**
    @brief capturers a kernel to runs the given callable with only one thread

    @tparam C callable type

    @param c callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C c);

    /**
    @brief updates a capture task to a single-threaded kernel

    This method is similar to cudaFlowCapturer::single_task but operates
    on an existing task.
    */
    template <typename C>
    void single_task(cudaTask task, C c);

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
    @brief updates a capture task to a for-each kernel task

    This method is similar to cudaFlowCapturer::for_each but operates
    on an existing task.
    */
    template <typename I, typename C>
    void for_each(cudaTask task, I first, I last, C callable);

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
    @brief updates a capture task to a for-each-index kernel task

    This method is similar to cudaFlowCapturer::for_each_index but operates
    on an existing task.
    */
    template <typename I, typename C>
    void for_each_index(
      cudaTask task, I first, I last, I step, C callable
    );

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
    @brief updates a capture task to a transform kernel task

    This method is similar to cudaFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I, typename O, typename C>
    void transform(cudaTask task, I first, I last, O output, C op);

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
    @brief updates a capture task to a transform kernel task

    This method is similar to cudaFlowCapturer::transform but operates
    on an existing task.
    */
    template <typename I1, typename I2, typename O, typename C>
    void transform(
      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
    );

    // ------------------------------------------------------------------------
    // Capturing methods
    // ------------------------------------------------------------------------
    
    /**
    @brief selects a different optimization algorithm

    @tparam OPT optimizer type
    @tparam ArgsT arguments types

    @param args arguments to forward to construct the optimizer

    @return a reference to the optimizer

    We currently supports the following optimization algorithms to capture
    a user-described %cudaFlow:
      + tf::cudaFlowSequentialOptimizer
      + tf::cudaFlowRoundRobinOptimizer
      + tf::cudaFlowLinearOptimizer

    By default, tf::cudaFlowCapturer uses the round-robin optimization
    algorithm with four streams to transform a user-level graph into
    a native CUDA graph.
    */
    template <typename OPT, typename... ArgsT>
    OPT& make_optimizer(ArgsT&&... args);
    
    /**
    @brief captures the cudaFlow and turns it into a CUDA Graph
    */
    cudaGraph_t capture();

    // ------------------------------------------------------------------------
    // offload methods
    // ------------------------------------------------------------------------

    /**
    @brief offloads the %cudaFlowCapturer onto a GPU asynchronously via a stream

    @param stream stream for performing this operation

    Offloads the present %cudaFlowCapturer onto a GPU asynchronously via
    the given stream.

    An offloaded %cudaFlowCapturer forces the underlying graph to be instantiated.
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

  private:

    cudaFlowGraph _cfg;

    Optimizer _optimizer;

    cudaGraphExec _exe {nullptr};
};

// Function: empty
inline bool cudaFlowCapturer::empty() const {
  return _cfg.empty();
}

// Function: num_tasks
inline size_t cudaFlowCapturer::num_tasks() const {
  return _cfg._nodes.size();
}

// Procedure: clear
inline void cudaFlowCapturer::clear() {
  _exe.clear();
  _cfg.clear();
}

// Procedure: dump
inline void cudaFlowCapturer::dump(std::ostream& os) const {
  _cfg.dump(os, nullptr, "");
}

// Procedure: dump_native_graph
inline void cudaFlowCapturer::dump_native_graph(std::ostream& os) const {
  cuda_dump_graph(os, _cfg._native_handle);
}

// Function: capture
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
>
cudaTask cudaFlowCapturer::on(C&& callable) {
  auto node = _cfg.emplace_back(_cfg,
    std::in_place_type_t<cudaFlowNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Function: noop
inline cudaTask cudaFlowCapturer::noop() {
  return on([](cudaStream_t){});
}

// Function: noop
inline void cudaFlowCapturer::noop(cudaTask task) {
  on(task, [](cudaStream_t){});
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

// Function: copy
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
  dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
) {
  return on([g, b, s, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: capture
inline cudaGraph_t cudaFlowCapturer::capture() {
  return std::visit(
    [this](auto&& opt){ return opt._optimize(_cfg); }, _optimizer
  );
}

// Procedure: run
inline void cudaFlowCapturer::run(cudaStream_t stream) {

  // If the topology got changed, we need to destroy the executable
  // and create a new one
  if(_cfg._state & cudaFlowGraph::CHANGED) {
    _cfg._native_handle.reset(capture());
    _exe.instantiate(_cfg._native_handle);
  }
  // if the graph is just updated (i.e., topology does not change),
  // we can skip part of the optimization and just update the executable
  // with the new captured graph
  else if(_cfg._state & cudaFlowGraph::UPDATED) {
    // TODO: skip part of the optimization (e.g., levelization)
    _cfg._native_handle.reset(capture());
    if(_exe.update(_cfg._native_handle) != cudaGraphExecUpdateSuccess) {
      _exe.instantiate(_cfg._native_handle);
    }
  }

  // run the executable (should exist)
  _exe.launch(stream);

  _cfg._state = cudaFlowGraph::OFFLOADED;
}

// Function: native_graph
inline cudaGraph_t cudaFlowCapturer::native_graph() {
  return _cfg._native_handle;
}

// Function: native_executable
inline cudaGraphExec_t cudaFlowCapturer::native_executable() {
  return _exe;
}

// Function: on
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
>
void cudaFlowCapturer::on(cudaTask task, C&& callable) {

  if(task.type() != cudaTaskType::CAPTURE) {
    TF_THROW("invalid cudaTask type (must be CAPTURE)");
  }

  _cfg._state |= cudaFlowGraph::UPDATED;

  std::get_if<cudaFlowNode::Capture>(&task._node->_handle)->work =
    std::forward<C>(callable);
}

// Function: memcpy
inline void cudaFlowCapturer::memcpy(
  cudaTask task, void* dst, const void* src, size_t count
) {
  on(task, [dst, src, count](cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
      "failed to capture memcpy"
    );
  });
}

// Function: copy
template <typename T,
  std::enable_if_t<!std::is_same_v<T, void>, void>*
>
void cudaFlowCapturer::copy(
  cudaTask task, T* tgt, const T* src, size_t num
) {
  on(task, [tgt, src, num] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline void cudaFlowCapturer::memset(
  cudaTask task, void* ptr, int v, size_t n
) {
  on(task, [ptr, v, n] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}

// Function: kernel
template <typename F, typename... ArgsT>
void cudaFlowCapturer::kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
) {
  on(task, [g, b, s, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, s, stream>>>(args...);
  });
}

// Function: make_optimizer
template <typename OPT, typename ...ArgsT>
OPT& cudaFlowCapturer::make_optimizer(ArgsT&&... args) {
  return _optimizer.emplace<OPT>(std::forward<ArgsT>(args)...);
}

}  // end of namespace tf -----------------------------------------------------

