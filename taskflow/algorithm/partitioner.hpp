// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp

#pragma once

/**
@file partitioner.hpp
@brief partitioner include file
*/

namespace tf {

/**
@enum PartitionerType

@brief enumeration of all partitioner types
*/  
enum class PartitionerType : int {
  /** @brief static partitioner type */
  STATIC,
  /** @brief dynamic partitioner type */
  DYNAMIC
};


//template <typename C>
//class PartitionInvoker : public PartitionerBase {
//
//  protected
//
//  C _closure;
//
//  template <typename... ArgsT>
//  auto operator()(ArgsT&&... args) {
//    return std::invoke(closure, std::forward<ArgsT>(args)...);
//  }
//
//  template <typename... ArgsT>
//  auto operator()(ArgsT&&... args) const {
//    return std::invoke(closure, std::forward<ArgsT>(args)...);
//  }
//
//};

/**
@struct DefaultClosureWrapper

@brief default closure wrapper that simply runs the given closure as is
*/
struct DefaultClosureWrapper {
};

/**
@private
*/
struct IsPartitioner {
};

// ----------------------------------------------------------------------------
// Partitioner Base
// ----------------------------------------------------------------------------

/**
@class PartitionerBase

@brief class to derive a partitioner for scheduling parallel algorithms

@tparam C closure wrapper type

The class provides base methods to derive a partitioner that can be used
to schedule parallel iterations (e.g., tf::Taskflow::for_each).

An partitioner defines the scheduling method for running parallel algorithms,
such tf::Taskflow::for_each, tf::Taskflow::reduce, and so on.
By default, we provide the following partitioners: 

+ tf::GuidedPartitioner  to enable guided scheduling algorithm of adaptive chunk size
+ tf::DynamicPartitioner to enable dynamic scheduling algorithm of equal chunk size
+ tf::StaticPartitioner  to enable static scheduling algorithm of static chunk size
+ tf::RandomPartitioner  to enable random scheduling algorithm of random chunk size

Depending on applications, partitioning algorithms can impact the performance
a lot. 
For example, if a parallel-iteration workload contains a regular work unit per
iteration, tf::StaticPartitioner can deliver the best performance.
On the other hand, if the work unit per iteration is irregular and unbalanced,
tf::GuidedPartitioner or tf::DynamicPartitioner can outperform tf::StaticPartitioner.
In most situations, tf::GuidedPartitioner can deliver decent performance and
is thus used as our default partitioner.

@note
Giving the partition size of 0 lets the %Taskflow runtime automatically determines
the partition size for the given partitioner.


In addition to partition size, the application can specify a closure wrapper
for a partitioner.
A closure wrapper allows the application to wrapper a partitioned task 
(i.e., closure) with a custom function object that performs additional tasks.
For example:

@code{.cpp}
std::atomic<int> count = 0;
tf::Taskflow taskflow;
taskflow.for_each_index(0, 100, 1, 
  [](){                 
    printf("%d\n", i); 
  },
  tf::StaticPartitioner(0, [](auto&& closure){
    // do something before invoking the partitioned task
    // ...
    
    // invoke the partitioned task
    closure();

    // do something else after invoking the partitioned task
    // ...
  }
);
executor.run(taskflow).wait();
@endcode

@note
The default closure wrapper (tf::DefaultClosureWrapper) does nothing but invoke
the partitioned task (closure).

*/
template <typename C = DefaultClosureWrapper>
class PartitionerBase : public IsPartitioner {

  public:
  
  /** 
  @brief the closure type
  */
  using closure_wrapper_type = C;

  /**
  @brief default constructor
  */
  PartitionerBase() = default;

  /**
  @brief construct a partitioner with the given chunk size
  */
  explicit PartitionerBase(size_t chunk_size) : _chunk_size {chunk_size} {}
  
  /**
  @brief construct a partitioner with the given chunk size and closure wrapper
  */
  PartitionerBase(size_t chunk_size, C&& closure_wrapper) :
    _chunk_size {chunk_size},
    _closure_wrapper {std::forward<C>(closure_wrapper)} {
  }

  /**
  @brief query the chunk size of this partitioner
  */
  size_t chunk_size() const { return _chunk_size; }
  
  /**
  @brief update the chunk size of this partitioner
  */
  void chunk_size(size_t cz) { _chunk_size = cz; }

  /**
  @brief acquire an immutable access to the closure wrapper object
  */
  const C& closure_wrapper() const { return _closure_wrapper; }

  /**
  @brief modify the closure wrapper object
  */
  template <typename F>
  void closure_wrapper(F&& fn) { _closure_wrapper = std::forward<F>(fn); }

  protected:
  
  /**
  @brief chunk size 
  */
  size_t _chunk_size{0};

  /**
  @brief closure wrapper
  */
  C _closure_wrapper;
};

// ----------------------------------------------------------------------------
// Guided Partitioner
// ----------------------------------------------------------------------------

/**
@class GuidedPartitioner

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

@brief class to construct a guided partitioner for scheduling parallel algorithms

The size of a partition is proportional to the number of unassigned iterations 
divided by the number of workers, 
and the size will gradually decrease to the given chunk size.
The last partition may be smaller than the chunk size.

In addition to partition size, the application can specify a closure wrapper
for a guided partitioner.
A closure wrapper allows the application to wrapper a partitioned task 
(i.e., closure) with a custom function object that performs additional tasks.
For example:

@code{.cpp}
std::atomic<int> count = 0;
tf::Taskflow taskflow;
taskflow.for_each_index(0, 100, 1, 
  [](){                 
    printf("%d\n", i); 
  },
  tf::GuidedPartitioner(0, [](auto&& closure){
    // do something before invoking the partitioned task
    // ...
    
    // invoke the partitioned task
    closure();

    // do something else after invoking the partitioned task
    // ...
  }
);
executor.run(taskflow).wait();
@endcode
*/
template <typename C = DefaultClosureWrapper>
class GuidedPartitioner : public PartitionerBase<C> {

  public:
  
  /**
  @brief queries the partition type (dynamic)
  */
  static constexpr PartitionerType type() { return PartitionerType::DYNAMIC; }
  
  /**
  @brief default constructor
  */
  GuidedPartitioner() = default;

  /**
  @brief construct a guided partitioner with the given chunk size

  */
  explicit GuidedPartitioner(size_t sz) : PartitionerBase<C> (sz) {}
 
  /**
  @brief construct a guided partitioner with the given chunk size and the closure
  */ 
  explicit GuidedPartitioner(size_t sz, C&& closure) :
    PartitionerBase<C>(sz, std::forward<C>(closure)) {
  }
  
  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<void, F, size_t, size_t>, void>* = nullptr
  >
  void loop(
    size_t N, size_t W, std::atomic<size_t>& next, F&& func
  ) const {

    size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;

    size_t p1 = 2 * W * (chunk_size + 1);
    float  p2 = 0.5f / static_cast<float>(W);
    size_t curr_b = next.load(std::memory_order_relaxed);

    while(curr_b < N) {

      size_t r = N - curr_b;

      // fine-grained
      if(r < p1) {
        while(1) {
          curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
          if(curr_b >= N) {
            return;
          }
          func(curr_b, std::min(curr_b + chunk_size, N));
        }
        break;
      }
      // coarse-grained
      else {
        size_t q = static_cast<size_t>(p2 * r);
        if(q < chunk_size) {
          q = chunk_size;
        }
        //size_t curr_e = (q <= r) ? curr_b + q : N;
        size_t curr_e = std::min(curr_b + q, N);
        if(next.compare_exchange_strong(curr_b, curr_e, std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
          func(curr_b, curr_e);
          curr_b = next.load(std::memory_order_relaxed);
        }
      }
    }
  }
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<bool, F, size_t, size_t>, void>* = nullptr
  >
  void loop_until(
    size_t N, size_t W, std::atomic<size_t>& next, F&& func
  ) const {

    size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;

    size_t p1 = 2 * W * (chunk_size + 1);
    float  p2 = 0.5f / static_cast<float>(W);
    size_t curr_b = next.load(std::memory_order_relaxed);

    while(curr_b < N) {

      size_t r = N - curr_b;

      // fine-grained
      if(r < p1) {
        while(1) {
          curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
          if(curr_b >= N) {
            return;
          }
          if(func(curr_b, std::min(curr_b + chunk_size, N))) {
            return;
          }
        }
        break;
      }
      // coarse-grained
      else {
        size_t q = static_cast<size_t>(p2 * r);
        if(q < chunk_size) {
          q = chunk_size;
        }
        //size_t curr_e = (q <= r) ? curr_b + q : N;
        size_t curr_e = std::min(curr_b + q, N);
        if(next.compare_exchange_strong(curr_b, curr_e, std::memory_order_relaxed,
                                                        std::memory_order_relaxed)) {
          if(func(curr_b, curr_e)) {
            return;
          }
          curr_b = next.load(std::memory_order_relaxed);
        }
      }
    }
  }

};

// ----------------------------------------------------------------------------
// Dynamic Partitioner
// ----------------------------------------------------------------------------

/**
@class DynamicPartitioner

@brief class to construct a dynamic partitioner for scheduling parallel algorithms

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

The partitioner splits iterations into many partitions each of size equal to 
the given chunk size.
Different partitions are distributed dynamically to workers 
without any specific order.

In addition to partition size, the application can specify a closure wrapper
for a dynamic partitioner.
A closure wrapper allows the application to wrapper a partitioned task 
(i.e., closure) with a custom function object that performs additional tasks.
For example:

@code{.cpp}
std::atomic<int> count = 0;
tf::Taskflow taskflow;
taskflow.for_each_index(0, 100, 1, 
  [](){                 
    printf("%d\n", i); 
  },
  tf::DynamicPartitioner(0, [](auto&& closure){
    // do something before invoking the partitioned task
    // ...
    
    // invoke the partitioned task
    closure();

    // do something else after invoking the partitioned task
    // ...
  }
);
executor.run(taskflow).wait();
@endcode
*/
template <typename C = DefaultClosureWrapper>
class DynamicPartitioner : public PartitionerBase<C> {

  public:
  
  /**
  @brief queries the partition type (dynamic)
  */
  static constexpr PartitionerType type() { return PartitionerType::DYNAMIC; }

  /**
  @brief default constructor
  */
  DynamicPartitioner() = default;
  
  /**
  @brief construct a dynamic partitioner with the given chunk size
  */
  explicit DynamicPartitioner(size_t sz) : PartitionerBase<C>(sz) {}
  
  /**
  @brief construct a dynamic partitioner with the given chunk size and the closure
  */ 
  explicit DynamicPartitioner(size_t sz, C&& closure) :
    PartitionerBase<C>(sz, std::forward<C>(closure)) {
  }
  
  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------

  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<void, F, size_t, size_t>, void>* = nullptr
  >
  void loop(
    size_t N, size_t, std::atomic<size_t>& next, F&& func
  ) const {

    size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;
    size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

    while(curr_b < N) {
      func(curr_b, std::min(curr_b + chunk_size, N));
      curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    }
  }
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<bool, F, size_t, size_t>, void>* = nullptr
  >
  void loop_until(
    size_t N, size_t, std::atomic<size_t>& next, F&& func
  ) const {

    size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;
    size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

    while(curr_b < N) {
      if(func(curr_b, std::min(curr_b + chunk_size, N))) {
        return;
      }
      curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    }
  }

};

// ----------------------------------------------------------------------------
// Static Partitioner
// ----------------------------------------------------------------------------

/**
@class StaticPartitioner

@brief class to construct a static partitioner for scheduling parallel algorithms

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

The partitioner divides iterations into chunks and distributes chunks 
to workers in order.
If the chunk size is not specified (default @c 0), the partitioner resorts to a chunk size
that equally distributes iterations into workers.

@code{.cpp}
std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
taskflow.for_each(
  data.begin(), data.end(), [](int i){}, StaticPartitioner(0)
);
executor.run(taskflow).run();
@endcode

In addition to partition size, the application can specify a closure wrapper
for a static partitioner.
A closure wrapper allows the application to wrapper a partitioned task 
(i.e., closure) with a custom function object that performs additional tasks.
For example:

@code{.cpp}
std::atomic<int> count = 0;
tf::Taskflow taskflow;
taskflow.for_each_index(0, 100, 1, 
  [](){                 
    printf("%d\n", i); 
  },
  tf::StaticPartitioner(0, [](auto&& closure){
    // do something before invoking the partitioned task
    // ...
    
    // invoke the partitioned task
    closure();

    // do something else after invoking the partitioned task
    // ...
  }
);
executor.run(taskflow).wait();
@endcode
*/
template <typename C = DefaultClosureWrapper>
class StaticPartitioner : public PartitionerBase<C> {

  public:
  
  /**
  @brief queries the partition type (static)
  */
  static constexpr PartitionerType type() { return PartitionerType::STATIC; }

  /**
  @brief default constructor
  */
  StaticPartitioner() = default;
  
  /**
  @brief construct a static partitioner with the given chunk size
  */
  explicit StaticPartitioner(size_t sz) : PartitionerBase<C>(sz) {}
  
  /**
  @brief construct a static partitioner with the given chunk size and the closure
  */ 
  explicit StaticPartitioner(size_t sz, C&& closure) :
    PartitionerBase<C>(sz, std::forward<C>(closure)) {
  }
  
  /**
  @brief queries the adjusted chunk size
  
  Returns the given chunk size if it is not zero, or returns
  <tt>N/W + (w < N%W)</tt>, where @c N is the number of iterations,
  @c W is the number of workers, and @c w is the worker ID.
  */
  size_t adjusted_chunk_size(size_t N, size_t W, size_t w) const {
    return this->_chunk_size ? this->_chunk_size : N/W + (w < N%W);
  }
  
  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<void, F, size_t, size_t>, void>* = nullptr
  >
  void loop(
    size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func
  ) {
    size_t stride = W * chunk_size;
    while(curr_b < N) {
      size_t curr_e = std::min(curr_b + chunk_size, N);
      func(curr_b, curr_e);
      curr_b += stride;
    }
  }
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<bool, F, size_t, size_t>, void>* = nullptr
  >
  void loop_until(
    size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func
  ) {
    size_t stride = W * chunk_size;
    while(curr_b < N) {
      size_t curr_e = std::min(curr_b + chunk_size, N);
      if(func(curr_b, curr_e)) {
        return;
      }
      curr_b += stride;
    }
  }
};

// ----------------------------------------------------------------------------
// RandomPartitioner
// ----------------------------------------------------------------------------

/**
@class RandomPartitioner

@brief class to construct a random partitioner for scheduling parallel algorithms

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

Similar to tf::DynamicPartitioner, 
the partitioner splits iterations into many partitions but each with a random
chunk size in the range, <tt>c = [alpha * N * W, beta * N * W]</tt>.
By default, @c alpha is <tt>0.01</tt> and @c beta is <tt>0.5</tt>, respectively.

In addition to partition size, the application can specify a closure wrapper
for a random partitioner.
A closure wrapper allows the application to wrapper a partitioned task 
(i.e., closure) with a custom function object that performs additional tasks.
For example:

@code{.cpp}
std::atomic<int> count = 0;
tf::Taskflow taskflow;
taskflow.for_each_index(0, 100, 1, 
  [](){                 
    printf("%d\n", i); 
  },
  tf::RandomPartitioner(0, [](auto&& closure){
    // do something before invoking the partitioned task
    // ...
    
    // invoke the partitioned task
    closure();

    // do something else after invoking the partitioned task
    // ...
  }
);
executor.run(taskflow).wait();
@endcode
*/
template <typename C = DefaultClosureWrapper>
class RandomPartitioner : public PartitionerBase<C> {

  public:
  
  /**
  @brief queries the partition type (dynamic)
  */
  static constexpr PartitionerType type() { return PartitionerType::DYNAMIC; }

  /**
  @brief default constructor
  */
  RandomPartitioner() = default;
  
  /**
  @brief construct a dynamic partitioner with the given chunk size
  */
  explicit RandomPartitioner(size_t sz) : PartitionerBase<C>(sz) {}
  
  /**
  @brief construct a random partitioner with the given chunk size and the closure
  */ 
  explicit RandomPartitioner(size_t sz, C&& closure) :
    PartitionerBase<C>(sz, std::forward<C>(closure)) {
  }
  
  /**
  @brief constructs a random partitioner with the given parameters
  */
  RandomPartitioner(float alpha, float beta) : _alpha{alpha}, _beta{beta} {}

  /**
  @brief constructs a random partitioner with the given parameters and the closure
  */
  RandomPartitioner(float alpha, float beta, C&& closure) : 
    _alpha {alpha}, _beta {beta}, 
    PartitionerBase<C>(0, std::forward<C>(closure)) {
  }

  /**
  @brief queries the @c alpha value
  */
  float alpha() const { return _alpha; }
  
  /**
  @brief queries the @c beta value
  */
  float beta() const { return _beta; }
  
  /**
  @brief queries the range of chunk size
  
  @param N number of iterations
  @param W number of workers
  */
  std::pair<size_t, size_t> chunk_size_range(size_t N, size_t W) const {
    
    size_t b1 = static_cast<size_t>(_alpha * N * W);
    size_t b2 = static_cast<size_t>(_beta  * N * W);

    if(b1 > b2) {
      std::swap(b1, b2);
    }

    b1 = std::max(b1, size_t{1});
    b2 = std::max(b2, b1 + 1);

    return {b1, b2};
  }

  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------
  
  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<void, F, size_t, size_t>, void>* = nullptr
  >
  void loop(
    size_t N, size_t W, std::atomic<size_t>& next, F&& func
  ) const {

    auto [b1, b2] = chunk_size_range(N, W); 
    
    std::default_random_engine engine {std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(b1, b2);
    
    size_t chunk_size = dist(engine);
    size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

    while(curr_b < N) {
      func(curr_b, std::min(curr_b + chunk_size, N));
      chunk_size = dist(engine);
      curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    }
  }

  /**
  @private
  */
  template <typename F, 
    std::enable_if_t<std::is_invocable_r_v<bool, F, size_t, size_t>, void>* = nullptr
  >
  void loop_until(
    size_t N, size_t W, std::atomic<size_t>& next, F&& func
  ) const {

    auto [b1, b2] = chunk_size_range(N, W); 
    
    std::default_random_engine engine {std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(b1, b2);
    
    size_t chunk_size = dist(engine);
    size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

    while(curr_b < N) {
      if(func(curr_b, std::min(curr_b + chunk_size, N))){
        return;
      }
      chunk_size = dist(engine);
      curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    }
  }

  private:

  float _alpha {0.01f};
  float _beta  {0.5f};
};

/**
@brief default partitioner set to tf::GuidedPartitioner

Guided partitioning algorithm can achieve stable and decent performance
for most parallel algorithms.
*/
using DefaultPartitioner = GuidedPartitioner<>;

/**
@brief determines if a type is a partitioner 

A partitioner is a derived type from tf::PartitionerBase.
*/
template <typename P>
inline constexpr bool is_partitioner_v = std::is_base_of<IsPartitioner, P>::value;

}  // end of namespace tf -----------------------------------------------------



