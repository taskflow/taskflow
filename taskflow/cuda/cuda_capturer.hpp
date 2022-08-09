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
By default, we use tf::cudaRoundRobinCapturing to transform a user-level
graph into a native CUDA graph.

Please refer to @ref GPUTaskingcudaFlowCapturer for details.
*/
class cudaFlowCapturer {

  friend class cudaFlow;
  friend class Executor;

  // created by user
  struct External {
    cudaGraph graph;
  };
  
  // created from executor
  struct Internal {
    Internal(Executor& e) : executor{e} {}
    Executor& executor;
  };
  
  // created from cudaFlow
  struct Proxy {
  };

  using handle_t = std::variant<External, Internal, Proxy>;

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
      + tf::cudaLinearCapturing

    By default, tf::cudaFlowCapturer uses the round-robin optimization
    algorithm with four streams to transform a user-level graph into
    a native CUDA graph.
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

    @param ptr pointer to GPU mempry
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
    @brief updates a capture task to a reduction task

    This method is similar to cudaFlowCapturer::reduce but operates
    on an existing task.
    */
    template <typename I, typename T, typename C>
    void reduce(cudaTask task, I first, I last, T* result, C op);

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
    @brief updates a capture task to an uninitialized-reduction task

    This method is similar to cudaFlowCapturer::uninitialized_reduce
    but operates on an existing task.
    */
    template <typename I, typename T, typename C>
    void uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C op
    );

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
    @brief updates a capture task to a transform-reduce task

    This method is similar to cudaFlowCapturer::transform_reduce but
    operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void transform_reduce(
      cudaTask task, I first, I last, T* result, C bop, U uop
    );

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
    @brief updates a capture task to a transform-reduce task of no initialized value

    This method is similar to cudaFlowCapturer::transform_uninitialized_reduce
    but operates on an existing task.
    */
    template <typename I, typename T, typename C, typename U>
    void transform_uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C bop, U uop
    );

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
    @brief updates a capture task to an inclusive scan task

    This method is similar to cudaFlowCapturer::inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void inclusive_scan(cudaTask task, I first, I last, O output, C op);

    /**
    @brief similar to cudaFlowCapturer::inclusive_scan
           but excludes the first value
    */
    template <typename I, typename O, typename C>
    cudaTask exclusive_scan(I first, I last, O output, C op);

    /**
    @brief updates a capture task to an exclusive scan task

    This method is similar to cudaFlowCapturer::exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void exclusive_scan(cudaTask task, I first, I last, O output, C op);

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
    @brief updates a capture task to a transform-inclusive scan task

    This method is similar to cudaFlowCapturer::transform_inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_inclusive_scan(
      cudaTask task, I first, I last, O output, B bop, U uop
    );

    /**
    @brief similar to cudaFlowCapturer::transform_inclusive_scan but
           excludes the first value
    */
    template <typename I, typename O, typename B, typename U>
    cudaTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);

    /**
    @brief updates a capture task to a transform-exclusive scan task

    This method is similar to cudaFlowCapturer::transform_exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_exclusive_scan(
      cudaTask task, I first, I last, O output, B bop, U uop
    );

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
    evaluates to @c false.
     */
    template <typename A, typename B, typename C, typename Comp>
    cudaTask merge(A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp);

    /**
    @brief updates a capture task to a merge task

    This method is similar to cudaFlowCapturer::merge but operates
    on an existing task.
     */
    template <typename A, typename B, typename C, typename Comp>
    void merge(
      cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
    );

    /**
    @brief captures kernels that perform parallel key-value merge

    @tparam a_keys_it first key iterator type
    @tparam a_vals_it first value iterator type
    @tparam b_keys_it second key iterator type
    @tparam b_vals_it second value iterator type
    @tparam c_keys_it output key iterator type
    @tparam c_vals_it output value iterator type
    @tparam C comparator type

    @param a_keys_first iterator to the beginning of the first key range
    @param a_keys_last iterator to the end of the first key range
    @param a_vals_first iterator to the beginning of the first value range
    @param b_keys_first iterator to the beginning of the second key range
    @param b_keys_last iterator to the end of the second key range
    @param b_vals_first iterator to the beginning of the second value range
    @param c_keys_first iterator to the beginning of the output key range
    @param c_vals_first iterator to the beginning of the output value range
    @param comp comparator

    Performs a key-value merge that copies elements from
    <tt>[a_keys_first, a_keys_last)</tt> and <tt>[b_keys_first, b_keys_last)</tt>
    into a single range, <tt>[c_keys_first, c_keys_last + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
    such that the resulting range is in ascending key order.

    At the same time, the merge copies elements from the two associated ranges
    <tt>[a_vals_first + (a_keys_last - a_keys_first))</tt> and
    <tt>[b_vals_first + (b_keys_last - b_keys_first))</tt> into a single range,
    <tt>[c_vals_first, c_vals_first + (a_keys_last - a_keys_first) + (b_keys_last - b_keys_first))</tt>
    such that the resulting range is in ascending order
    implied by each input element's associated key.

    For example, assume:
      + @c a_keys = <tt>{8, 1}</tt>
      + @c a_vals = <tt>{1, 2}</tt>
      + @c b_keys = <tt>{3, 7}</tt>
      + @c b_vals = <tt>{3, 4}</tt>

    After the merge, we have:
      + @c c_keys = <tt>{1, 3, 7, 8}</tt>
      + @c c_vals = <tt>{2, 3, 4, 1}</tt>
    */
    template<
      typename a_keys_it, typename a_vals_it,
      typename b_keys_it, typename b_vals_it,
      typename c_keys_it, typename c_vals_it,
      typename C
    >
    cudaTask merge_by_key(
      a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
      b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
      c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
    );

    /**
    @brief updates a capture task to a key-value merge task

    This method is similar to tf::cudaFlowCapturer::merge_by_key but operates
    on an existing task.
    */
    template<
      typename a_keys_it, typename a_vals_it,
      typename b_keys_it, typename b_vals_it,
      typename c_keys_it, typename c_vals_it,
      typename C
    >
    void merge_by_key(
      cudaTask task,
      a_keys_it a_keys_first, a_keys_it a_keys_last, a_vals_it a_vals_first,
      b_keys_it b_keys_first, b_keys_it b_keys_last, b_vals_it b_vals_first,
      c_keys_it c_keys_first, c_vals_it c_vals_first, C comp
    );

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

    /**
    @brief updates a capture task to a sort task

    This method is similar to cudaFlowCapturer::sort but operates on
    an existing task.
    */
    template <typename I, typename C>
    void sort(cudaTask task, I first, I last, C comp);

    /**
    @brief captures kernels that sort the given array

    @tparam K_it iterator type of the key
    @tparam V_it iterator type of the value
    @tparam C comparator type

    @param k_first iterator to the beginning of the key array
    @param k_last iterator to the end of the key array
    @param v_first iterator to the beginning of the value array
    @param comp binary comparator

    @return a tf::cudaTask handle

    Sorts key-value elements in <tt>[k_first, k_last)</tt> and
    <tt>[v_first, v_first + (k_last - k_first))</tt> into ascending key order
    using the given comparator @c comp.
    If @c i and @c j are any two valid iterators in <tt>[k_first, k_last)</tt>
    such that @c i precedes @c j, and @c p and @c q are iterators in
    <tt>[v_first, v_first + (k_last - k_first))</tt> corresponding to
    @c i and @c j respectively, then <tt>comp(*j, *i)</tt> evaluates to @c false.

    For example, assume:
      + @c keys are <tt>{1, 4, 2, 8, 5, 7}</tt>
      + @c values are <tt>{'a', 'b', 'c', 'd', 'e', 'f'}</tt>

    After sort:
      + @c keys are <tt>{1, 2, 4, 5, 7, 8}</tt>
      + @c values are <tt>{'a', 'c', 'b', 'e', 'f', 'd'}</tt>
    */
    template <typename K_it, typename V_it, typename C>
    cudaTask sort_by_key(K_it k_first, K_it k_last, V_it v_first, C comp);

    /**
    @brief updates a capture task to a key-value sort task

    This method is similar to tf::cudaFlowCapturer::sort_by_key
    but operates on an existing task.
    */
    template <typename K_it, typename V_it, typename C>
    void sort_by_key(
      cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
    );

    /**
    @brief creates a task to find the index of the first element in a range

    @tparam I input iterator type
    @tparam U unary operator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx pointer to the index of the found element
    @param op unary operator which returns @c true for the required element

    Finds the index @c idx of the first element in the range
    <tt>[first, last)</tt> such that <tt>op(*(first+idx))</tt> is true.
    This is equivalent to the parallel execution of the following loop:

    @code{.cpp}
    unsigned idx = 0;
    for(; first != last; ++first, ++idx) {
      if (p(*first)) {
        return idx;
      }
    }
    return idx;
    @endcode
    */
    template <typename I, typename U>
    cudaTask find_if(I first, I last, unsigned* idx, U op);

    /**
    @brief updates the parameters of a find-if task

    This method is similar to tf::cudaFlowCapturer::find_if but operates
    on an existing task.
    */
    template <typename I, typename U>
    void find_if(cudaTask task, I first, I last, unsigned* idx, U op);

    /**
    @brief finds the index of the minimum element in a range

    @tparam I input iterator type
    @tparam O comparator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx solution index of the minimum element
    @param op comparison function object

    The function launches kernels asynchronously to find
    the smallest element in the range <tt>[first, last)</tt>
    using the given comparator @c op.
    The function is equivalent to a parallel execution of the following loop:

    @code{.cpp}
    if(first == last) {
      return 0;
    }
    auto smallest = first;
    for (++first; first != last; ++first) {
      if (op(*first, *smallest)) {
        smallest = first;
      }
    }
    return std::distance(first, smallest);
    @endcode
    */
    template <typename I, typename O>
    cudaTask min_element(I first, I last, unsigned* idx, O op);

    /**
    @brief updates the parameters of a min-element task

    This method is similar to cudaFlowCapturer::min_element but operates
    on an existing task.
    */
    template <typename I, typename O>
    void min_element(cudaTask task, I first, I last, unsigned* idx, O op);

    /**
    @brief finds the index of the maximum element in a range

    @tparam I input iterator type
    @tparam O comparator type

    @param first iterator to the beginning of the range
    @param last iterator to the end of the range
    @param idx solution index of the maximum element
    @param op comparison function object

    The function launches kernels asynchronously to find
    the largest element in the range <tt>[first, last)</tt>
    using the given comparator @c op.
    The function is equivalent to a parallel execution of the following loop:

    @code{.cpp}
    if(first == last) {
      return 0;
    }
    auto largest = first;
    for (++first; first != last; ++first) {
      if (op(*largest, *first)) {
        largest = first;
      }
    }
    return std::distance(first, largest);
    @endcode
    */
    template <typename I, typename O>
    cudaTask max_element(I first, I last, unsigned* idx, O op);

    /**
    @brief updates the parameters of a max-element task

    This method is similar to cudaFlowCapturer::max_element but operates
    on an existing task.
     */
    template <typename I, typename O>
    void max_element(cudaTask task, I first, I last, unsigned* idx, O op);

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

    handle_t _handle;

    cudaGraph& _graph;

    Optimizer _optimizer;

    cudaGraphExec _exec {nullptr};

    cudaFlowCapturer(cudaGraph&, Executor& executor);
    cudaFlowCapturer(cudaGraph&);

    cudaGraph_t _capture();
};

// constructs a cudaFlow capturer from a taskflow
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
  _handle {std::in_place_type_t<Proxy>{}},
  _graph  {g} {
}

// constructs a cudaFlow capturer from a taskflow
inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g, Executor& e) :
  _handle {std::in_place_type_t<Internal>{}, e},
  _graph  {g} {
}

// constructs a standalone cudaFlow capturer
inline cudaFlowCapturer::cudaFlowCapturer() :
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get_if<External>(&_handle)->graph} {
}

inline cudaFlowCapturer::~cudaFlowCapturer() {
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
  _exec.clear();
  _graph._nodes.clear();
}

// Procedure: dump
inline void cudaFlowCapturer::dump(std::ostream& os) const {
  _graph.dump(os, nullptr, "");
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

// Function: _capture
inline cudaGraph_t cudaFlowCapturer::_capture() {
  return std::visit(
    [this](auto&& opt){ return opt._optimize(_graph); }, _optimizer
  );
}

// Procedure: offload_until
template <typename P>
void cudaFlowCapturer::offload_until(P&& predicate) {

  // If the topology got changed, we need to destroy the executable
  // and create a new one
  if(_graph._state & cudaGraph::CHANGED) {
    // TODO: store the native graph?
    cudaGraphNative g(_capture());
    _exec.instantiate(g);
  }
  // if the graph is just updated (i.e., topology does not change),
  // we can skip part of the optimization and just update the executable
  // with the new captured graph
  else if(_graph._state & cudaGraph::UPDATED) {
    // TODO: skip part of the optimization (e.g., levelization)
    cudaGraphNative g(_capture());
    if(_exec.update(g) != cudaGraphExecUpdateSuccess) {
      _exec.instantiate(g);
    }
    // TODO: store the native graph?
  }

  // offload the executable
  if(_exec) {
    cudaStream s;
    while(!predicate()) {
      _exec.launch(s);
      s.synchronize();
    }
  }

  _graph._state = cudaGraph::OFFLOADED;
}

// Procedure: offload_n
inline void cudaFlowCapturer::offload_n(size_t n) {
  offload_until([repeat=n] () mutable { return repeat-- == 0; });
}

// Procedure: offload
inline void cudaFlowCapturer::offload() {
  offload_until([repeat=1] () mutable { return repeat-- == 0; });
}

// Function: on
template <typename C, std::enable_if_t<
  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
>
void cudaFlowCapturer::on(cudaTask task, C&& callable) {

  if(task.type() != cudaTaskType::CAPTURE) {
    TF_THROW("invalid cudaTask type (must be CAPTURE)");
  }

  _graph._state |= cudaGraph::UPDATED;

  std::get_if<cudaNode::Capture>(&task._node->_handle)->work =
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

