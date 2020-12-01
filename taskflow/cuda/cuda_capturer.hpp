#pragma once

#include "cuda_task.hpp"
#include "cuda_algorithm/cuda_for_each.hpp"
#include "cuda_algorithm/cuda_transform.hpp"
#include "cuda_algorithm/cuda_reduce.hpp"

/** 
@file cuda_capturer.hpp
@brief cudaFlowCapturer include file
*/

namespace tf {

/**
@brief queries the maximum threads allowed per block
*/
constexpr size_t cuda_default_max_threads_per_block() {
  return 512;
}

/**
@brief queries the default number of threads per block in an 1D vector of N elements
*/
constexpr size_t cuda_default_threads_per_block(size_t N) {
  // TODO: special case when N == 0?
  if(N <= 32) return 32;
  else {
    return std::min(cuda_default_max_threads_per_block(), next_pow2(N));
  }
}

// ----------------------------------------------------------------------------
// class definition: cudaFlowCapturerBase
// ----------------------------------------------------------------------------

/**
@class cudaFlowCapturerBase

@brief base class to construct a CUDA task graph through stream capture
*/
class cudaFlowCapturerBase {

  friend class cudaFlowCapturer;

  public:

    /**
    @brief default constructor
     */
    cudaFlowCapturerBase() = default;

    /**
    @brief default virtual destructor
     */
    virtual ~cudaFlowCapturerBase() = default;
    
    /**
    @brief dumps the capture graph into a DOT format through an
           output stream
    */
    void dump(std::ostream& os) const;
    
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
    
    @param devPtr pointer to GPU mempry
    @param value value to set for each byte of the specified memory
    @param count size in bytes to set
    
    The method captures a @c cudaMemsetAsync operation through an
    internal stream to fill the first @c count bytes of the memory area 
    pointed to by @c devPtr with the constant byte value @c value.
    */ 
    cudaTask memset(void* devPtr, int value, size_t count);

    /**
    @brief captures a kernel
    
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
    
    // ------------------------------------------------------------------------
    // generic algorithms
    // ------------------------------------------------------------------------
    
    /**
    @brief capturers a kernel to runs the given callable with only one thread

    @tparam C callable type

    @param callable callable to run by a single kernel thread
    */
    template <typename C>
    cudaTask single_task(C&& callable);
    
    /**
    @brief captures a kernel that applies a callable to each dereferenced element 
           of the data array

    @tparam I iterator type
    @tparam C callable type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
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
    cudaTask for_each(I first, I last, C&& callable);

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
    cudaTask for_each_index(I first, I last, I step, C&& callable);
  
    /**
    @brief captures a kernel that applies a callable to a source range and 
           stores the result in a target range
    
    @tparam I iterator type
    @tparam C callable type
    @tparam S source types

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param callable the callable to apply to each element in the range
    @param srcs iterators to the source ranges
    
    @return cudaTask handle
    
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
    @brief captures a kernel that performs parallel reduction over a range of items

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
    @brief similar to tf::cudaFlowCapturerBase::reduce but does not assum 
           any initial value to reduce
    
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

  private:

    cudaGraph* _graph {nullptr};

    cudaFlowCapturerBase(cudaGraph&);
};

// Constructor
inline cudaFlowCapturerBase::cudaFlowCapturerBase(cudaGraph& g) :
  _graph {&g} {
}

// Procedure: dump
inline void cudaFlowCapturerBase::dump(std::ostream& os) const {
  _graph->dump(os, nullptr, "");
}

// Function: capture
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
>
cudaTask cudaFlowCapturerBase::on(C&& callable) {
  auto node = _graph->emplace_back(*_graph,
    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
  );
  return cudaTask(node);
}

// Function: memcpy
inline cudaTask cudaFlowCapturerBase::memcpy(
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
cudaTask cudaFlowCapturerBase::copy(T* tgt, const T* src, size_t num) {
  return on([tgt, src, num] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
      "failed to capture copy"
    );
  });
}

// Function: memset
inline cudaTask cudaFlowCapturerBase::memset(void* ptr, int v, size_t n) {
  return on([ptr, v, n] (cudaStream_t stream) mutable {
    TF_CHECK_CUDA(
      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
    );
  });
}
    
// Function: kernel
template <typename F, typename... ArgsT>
cudaTask cudaFlowCapturerBase::kernel(
  dim3 g, dim3 b, size_t shm, F&& f, ArgsT&&... args
) {
  return on([g, b, shm, f, args...] (cudaStream_t stream) mutable {
    f<<<g, b, shm, stream>>>(args...);
  });
}

// Function: single_task
template <typename C>
cudaTask cudaFlowCapturerBase::single_task(C&& callable) {
  return on([c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
  });
}

// Function: for_each
template <typename I, typename C>
cudaTask cudaFlowCapturerBase::for_each(I first, I last, C&& c) {
  return on([first, last, c=std::forward<C>(c)](cudaStream_t stream) mutable {
    // TODO: special case for N == 0?
    size_t N = std::distance(first, last);
    size_t B = cuda_default_threads_per_block(N);
    cuda_for_each<I, C><<<(N+B-1)/B, B, 0, stream>>>(first, N, c);
  });
}

// Function: for_each_index
template <typename I, typename C>
cudaTask cudaFlowCapturerBase::for_each_index(I beg, I end, I inc, C&& c) {
      
  if(is_range_invalid(beg, end, inc)) {
    TF_THROW("invalid range [", beg, ", ", end, ") with inc size ", inc);
  }
  
  return on([beg, end, inc, c=std::forward<C>(c)] (cudaStream_t stream) mutable {
    // TODO: special case when N is 0?
    size_t N = distance(beg, end, inc);
    size_t B = cuda_default_threads_per_block(N);
    cuda_for_each_index<I, C><<<(N+B-1)/B, B, 0, stream>>>(beg, inc, N, c);
  });
}

// Function: transform
template <typename I, typename C, typename... S>
cudaTask cudaFlowCapturerBase::transform(I first, I last, C&& c, S... srcs) {
  return on([first, last, c=std::forward<C>(c), srcs...] 
  (cudaStream_t stream) mutable {
    // TODO: special case when N is 0?
    size_t N = std::distance(first, last);
    size_t B = cuda_default_threads_per_block(N);
    cuda_transform<I, C, S...><<<(N+B-1)/B, B, 0, stream>>>(first, N, c, srcs...);
  });
}

// Function: reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturerBase::reduce(I first, I last, T* result, C&& c) {
  
  return on([first, last, result, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    //using value_t = std::decay_t<decltype(*std::declval<I>())>;
    
    // TODO: special case N == 0?
    size_t N = std::distance(first, last);
    size_t B = cuda_default_threads_per_block(N);

    cuda_reduce<I, T, C, false><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
cudaTask cudaFlowCapturerBase::uninitialized_reduce(
  I first, I last, T* result, C&& c
) {
  
  return on([first, last, result, c=std::forward<C>(c)] 
  (cudaStream_t stream) mutable {
    //using value_t = std::decay_t<decltype(*std::declval<I>())>;
    
    // TODO: special case N == 0?
    size_t N = std::distance(first, last);
    size_t B = cuda_default_threads_per_block(N);

    cuda_reduce<I, T, C, true><<<1, B, B*sizeof(T), stream>>>(
      first, N, result, c
    );
  });
}

// ----------------------------------------------------------------------------
// class definition: cudaFlowCapturer
// ----------------------------------------------------------------------------

/**
@class cudaFlowCapturer

@brief class for building a CUDA task dependency graph through stream capture

A %cudaFlowCapturer inherits all the base methods from tf::cudaFlowCapturerBase 
to construct a CUDA task graph through <i>stream capturer</i>. 
This class also defines a factory interface tf::cudaFlowCapturer::make_capturer 
for users to create custom capturers with their lifetimes managed by the factory.

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
*/
class cudaFlowCapturer : public cudaFlowCapturerBase {

  friend class cudaFlow;
  friend class Executor;

  public:
    
    /**
    @brief queries the emptiness of the graph
    */
    bool empty() const;
    
    /**
    @brief creates a custom capturer derived from tf::cudaFlowCapturerBase

    @tparam T custom capturer type
    @tparam ArgsT arguments types

    @param args arguments to forward to construct the custom capturer

    This %cudaFlowCapturer object keeps a list of custom capturers
    and manages their lifetimes. The lifetime of each custom capturer is
    the same as the capturer.
     */
    template <typename T, typename... ArgsT>
    T* make_capturer(ArgsT&&... args);

  private:
    
    std::vector<std::unique_ptr<cudaFlowCapturerBase>> _capturers;

    cudaFlowCapturer(cudaGraph&);
    
    cudaGraph_t _capture();
    
    cudaGraphExec_t _executable {nullptr};
};

// constructor
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
  cudaFlowCapturerBase{g} {
}

// Function: empty
inline bool cudaFlowCapturer::empty() const {
  return _graph->empty();
}

//// Procedure: _create_executable
//inline void cudaFlowCapturer::_create_executable() {
//  assert(_executable == nullptr);
//  TF_CHECK_CUDA(
//    cudaGraphInstantiate(
//      &_executable, _graph->_native_handle, nullptr, nullptr, 0
//    ),
//    "failed to create an executable for captured graph"
//  );
//}
//
//// Procedure: _destroy_executable
//inline void cudaFlowCapturer::_destroy_executable() {
//  assert(_executable != nullptr);
//  TF_CHECK_CUDA(
//    cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
//  );
//  _executable = nullptr;
//}

// Function: make_capturer
template <typename T, typename... ArgsT>
T* cudaFlowCapturer::make_capturer(ArgsT&&... args) {

  static_assert(std::is_base_of_v<cudaFlowCapturerBase, T>);

  auto ptr = std::make_unique<T>(std::forward<ArgsT>(args)...);
  ptr->_graph = this->_graph;
  auto raw = ptr.get();
  _capturers.push_back(std::move(ptr));
  return raw;
}

// Function: _capture
inline cudaGraph_t cudaFlowCapturer::_capture() {

  // acquire per-thread stream and turn it into capture mode
  // we must use ThreadLocal mode to avoid clashing with CUDA global states
  cudaScopedPerThreadStream stream;
  
  TF_CHECK_CUDA(
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), 
    "failed to turn stream into per-thread capture mode"
  );

  // TODO: need an efficient algorithm
  auto ordered = _graph->_toposort();
  for(auto& node : ordered) {
    std::get<cudaNode::Capture>(node->_handle).work(stream);  
  }

  cudaGraph_t g;

  TF_CHECK_CUDA(
    cudaStreamEndCapture(stream, &g), "failed to end capture"
  );

  //cuda_dump_graph(std::cout, g);
  
  return g;
}

}  // end of namespace tf -----------------------------------------------------

