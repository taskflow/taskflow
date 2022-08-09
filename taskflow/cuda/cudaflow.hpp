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

  friend class Executor;
  
  // created by user
  struct External {
    cudaGraph graph;
  };
  
  // created by executor
  struct Internal {
    Internal(Executor& e) : executor{e} {}
    Executor& executor;
  };

  using handle_t = std::variant<External, Internal>;
  
  // variant index
  constexpr static auto EXTERNAL = get_index_v<External, handle_t>;
  constexpr static auto INTERNAL = get_index_v<Internal, handle_t>;

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
    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);

    /**
    @brief updates parameters of a kernel task

    The method is similar to tf::cudaFlow::kernel but operates on a task
    of type tf::cudaTaskType::KERNEL.
    The kernel function name must NOT change.
    */
    template <typename F, typename... ArgsT>
    void kernel(
      cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT&&... args
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

    /**
    @brief performs parallel reduction over a range of items

    @tparam I input iterator type
    @tparam T value type
    @tparam B binary operator type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param result pointer to the result with an initialized value
    @param bop binary operator to apply to reduce items

    @return a tf::cudaTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first != last) {
      *result = bop(*result, *first++);
    }
    @endcode
    */
    template <typename I, typename T, typename B>
    cudaTask reduce(I first, I last, T* result, B bop);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::reduce

    The type of the iterators, result, and callable must be the same as
    the task created from tf::cudaFlow::reduce.
    */
    template <typename I, typename T, typename C>
    void reduce(cudaTask task, I first, I last, T* result, C op);

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
    template <typename I, typename T, typename B>
    cudaTask uninitialized_reduce(I first, I last, T* result, B bop);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::uninitialized_reduce

    The type of the iterators, result, and callable must be the same as
    the task created from tf::cudaFlow::uninitialized_reduce.
    */
    template <typename I, typename T, typename C>
    void uninitialized_reduce(
      cudaTask task, I first, I last, T* result, C op
    );

    /**
    @brief performs parallel reduction over a range of transformed items

    @tparam I input iterator type
    @tparam T value type
    @tparam B binary operator type
    @tparam U unary operator type

    @param first iterator to the beginning (inclusive)
    @param last iterator to the end (exclusive)
    @param result pointer to the result with an initialized value
    @param bop binary operator to apply to reduce items
    @param uop unary operator to transform each item before reduction

    @return a tf::cudaTask handle

    This method is equivalent to the parallel execution of the following loop on a GPU:

    @code{.cpp}
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename B, typename U>
    cudaTask transform_reduce(I first, I last, T* result, B bop, U uop);

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::transform_reduce
    */
    template <typename I, typename T, typename B, typename U>
    void transform_reduce(cudaTask, I first, I last, T* result, B bop, U uop);

    /**
    @brief similar to tf::cudaFlow::transform_reduce but does not assume any initial
           value to reduce

    This method is equivalent to the parallel execution of the following loop
    on a GPU:

    @code{.cpp}
    *result = uop(*first++);  // no initial values partitipcate in the loop
    while (first != last) {
      *result = bop(*result, uop(*first++));
    }
    @endcode
    */
    template <typename I, typename T, typename B, typename U>
    cudaTask transform_uninitialized_reduce(
      I first, I last, T* result, B bop, U uop
    );

    /**
    @brief updates parameters of a kernel task created from
           tf::cudaFlow::transform_uninitialized_reduce
    */
    template <typename I, typename T, typename B, typename U>
    void transform_uninitialized_reduce(
      cudaTask task, I first, I last, T* result, B bop, U uop
    );

    /**
    @brief creates a task to perform parallel inclusive scan
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
    @brief updates the parameters of a task created
           from tf::cudaFlow::inclusive_scan

    This method is similar to tf::cudaFlow::inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void inclusive_scan(cudaTask task, I first, I last, O output, C op);

    /**
    @brief similar to cudaFlow::inclusive_scan but excludes the first value
    */
    template <typename I, typename O, typename C>
    cudaTask exclusive_scan(I first, I last, O output, C op);

    /**
    @brief updates the parameters of a task created from
           tf::cudaFlow::exclusive_scan

    This method is similar to tf::cudaFlow::exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename C>
    void exclusive_scan(cudaTask task, I first, I last, O output, C op);

    /**
    @brief creates a task to perform parallel inclusive scan
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
    @brief updates the parameters of a task created from
           tf::cudaFlow::transform_inclusive_scan

    This method is similar to tf::cudaFlow::transform_inclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_inclusive_scan(
      cudaTask task, I first, I last, O output, B bop, U uop
    );

    /**
    @brief similar to cudaFlow::transform_inclusive_scan but
           excludes the first value
    */
    template <typename I, typename O, typename B, typename U>
    cudaTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);

    /**
    @brief updates the parameters of a task created from
           tf::cudaFlow::transform_exclusive_scan

    This method is similar to tf::cudaFlow::transform_exclusive_scan
    but operates on an existing task.
    */
    template <typename I, typename O, typename B, typename U>
    void transform_exclusive_scan(
      cudaTask task, I first, I last, O output, B bop, U uop
    );

    /**
    @brief creates a task to perform parallel merge on two sorted arrays

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
    @brief updates the parameters of a task created from
           tf::cudaFlow::merge

    This method is similar to tf::cudaFlow::merge but operates on
    an existing task.
    */
    template <typename A, typename B, typename C, typename Comp>
    void merge(
      cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
    );

    /**
    @brief creates a task to perform parallel sort an array

    @tparam I iterator type of the first input array
    @tparam C comparator type

    @param first iterator to the beginning of the input array
    @param last iterator to the end of the input array
    @param comp binary comparator

    @return a tf::cudaTask handle

    Sorts elements in the range <tt>[first, last)</tt>
    with the given comparator @c comp.
     */
    template <typename I, typename C>
    cudaTask sort(I first, I last, C comp);

    /**
    @brief updates the parameters of the task created from
           tf::cudaFlow::sort

    This method is similar to tf::cudaFlow::sort but operates on
    an existing task.
    */
    template <typename I, typename C>
    void sort(cudaTask task, I first, I last, C comp);

    /**
    @brief creates kernels that sort the given array

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
    @brief updates the parameters of a task created from
           tf::cudaFlow::sort_by_key

    This method is similar to tf::cudaFlow::sort_by_key but operates on
    an existing task.
    */
    template <typename K_it, typename V_it, typename C>
    void sort_by_key(
      cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
    );

    /**
    @brief creates a task to perform parallel key-value merge

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
    @brief updates the parameters of a task created from
           tf::cudaFlow::merge_by_key

    This method is similar to tf::cudaFlow::merge_by_key but operates
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
    @brief updates the parameters of the task created from
           tf::cudaFlow::find_if
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
    @brief updates the parameters of the task created from
           tf::cudaFlow::min_element
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
    @brief updates the parameters of the task created from
           tf::cudaFlow::max_element
     */
    template <typename I, typename O>
    void max_element(cudaTask task, I first, I last, unsigned* idx, O op);

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

    handle_t _handle;
    cudaGraph& _graph;
    cudaGraphExec _exec {nullptr};

    cudaFlow(cudaGraph&, Executor&);
    
    template <typename P>
    void _offload_until_external(P&&);
    
    template <typename P>
    void _offload_until_internal(P&&);
};

// Construct a standalone cudaFlow
inline cudaFlow::cudaFlow() :
  _handle {std::in_place_type_t<External>{}},
  _graph  {std::get_if<External>(&_handle)->graph} {

  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0),
    "cudaFlow failed to create a native graph (external mode)"
  );
}

// Construct the cudaFlow from executor (internal graph)
inline cudaFlow::cudaFlow(cudaGraph& g, Executor& executor) :
  _handle {std::in_place_type_t<Internal>{}, executor},
  _graph  {g} {

  assert(_graph._native_handle == nullptr);

  TF_CHECK_CUDA(
    cudaGraphCreate(&_graph._native_handle, 0),
    "failed to create a native graph (internal mode)"
  );
}

// Destructor
inline cudaFlow::~cudaFlow() {
  cudaGraphDestroy(_graph._native_handle);
  _graph._native_handle = nullptr;
}

// Procedure: clear
inline void cudaFlow::clear() {

  _exec.clear();

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

  auto h = std::get_if<cudaNode::Host>(&node->_handle);

  cudaHostNodeParams p;
  p.fn = cudaNode::Host::callback;
  p.userData = h;

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
  dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
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
void cudaFlow::host(cudaTask task, C&& c) {

  if(task.type() != cudaTaskType::HOST) {
    TF_THROW(task, " is not a host task");
  }

  auto h = std::get_if<cudaNode::Host>(&task._node->_handle);

  h->func = std::forward<C>(c);
}

// Function: update kernel parameters
template <typename F, typename... ArgsT>
void cudaFlow::kernel(
  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
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
    cudaGraphExecKernelNodeSetParams(_exec, task._node->_native_handle, &p),
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
    cudaGraphExecMemcpyNodeSetParams(_exec, task._node->_native_handle, &p),
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
    cudaGraphExecMemcpyNodeSetParams(_exec, task._node->_native_handle, &p),
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
    cudaGraphExecMemsetNodeSetParams(_exec, task._node->_native_handle, &p),
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
    cudaGraphExecMemsetNodeSetParams(_exec, task._node->_native_handle, &p),
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
    cudaGraphExecMemsetNodeSetParams(_exec, task._node->_native_handle, &p),
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
  auto node_handle = std::get_if<cudaNode::Subflow>(&task._node->_handle);
  node_handle->graph.clear();

  cudaFlowCapturer capturer(node_handle->graph);

  c(capturer);

  // obtain the optimized captured graph
  auto captured = capturer._capture();
  //cuda_dump_graph(std::cout, captured);

  TF_CHECK_CUDA(
    cudaGraphExecChildGraphNodeSetParams(_exec, task._node->_native_handle, captured),
    "failed to update a captured child graph"
  );

  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
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
  auto node_handle = std::get_if<cudaNode::Subflow>(&node->_handle);
  node_handle->graph.clear();
  cudaFlowCapturer capturer(node_handle->graph);

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

  _offload_until_external(std::forward<P>(predicate));
  
  /*
  // turns out the optimized version runs slower...
  switch(_handle.index()) {
    case EXTERNAL: {
      _offload_until_external(std::forward<P>(predicate));
    }
    break;
    case INTERNAL: {
      _offload_until_internal(std::forward<P>(predicate));
    }
    break;
    default:
    break;
  }*/
}

template <typename P>
void cudaFlow::_offload_until_external(P&& predicate) {
  if(!_exec) {
    _exec.instantiate(_graph._native_handle);
  }
  cudaStream stream;
  while(!predicate()) {
    _exec.launch(stream);
    stream.synchronize();
  }
  _graph._state = cudaGraph::OFFLOADED;
}

template <typename P>
void cudaFlow::_offload_until_internal(P&& predicate) {
  
  auto& executor = std::get<Internal>(_handle).executor;

  if(!_exec) {
    _exec.instantiate(_graph._native_handle);
  }
  
  cudaStream stream;
  cudaEvent event(cudaEventDisableTiming);

  while(!predicate()) {
    _exec.launch(stream);
    stream.record(event);
    executor.loop_until([&event] () -> bool { 
      return cudaEventQuery(event) == cudaSuccess;
    });
  }

  _graph._state = cudaGraph::OFFLOADED;
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
  auto n = _graph._emplace_back(
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

  auto h = std::get_if<Node::cudaFlow>(&node->_handle);

  cudaGraph* g = dynamic_cast<cudaGraph*>(h->graph.get());

  g->clear();

  T cf(*g, *this);

  c(cf);

  if(!(g->_state & cudaGraph::OFFLOADED)) {
    cf.offload();
  }
}


}  // end of namespace tf -----------------------------------------------------


