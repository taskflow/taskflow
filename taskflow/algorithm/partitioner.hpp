// reference:
// - gomp: https://github.com/gcc-mirror/gcc/blob/master/libgomp/iter.c
// - komp: https://github.com/llvm-mirror/openmp/blob/master/runtime/src/kmp_dispatch.cpp

#pragma once

/**
@file partitioner.hpp
@brief partitioner include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Partitioner Base
// ----------------------------------------------------------------------------

/**
@class PartitionerBase

@brief class to derive a partitioner for scheduling parallel algorithms

The class provides base methods to derive a partitioner that can be used
to schedule parallel iterations (e.g., tf::Taskflow::for_each).

An partitioner defines the scheduling method for running parallel algorithms,
such tf::Taskflow::for_each, tf::Taskflow::reduce, and so on.
By default, we provide the following partitioners:

+ tf::GuidedPartitioner to enable guided scheduling algorithm of adaptive chunk size
+ tf::DynamicPartitioner to enable dynamic scheduling algorithm of equal chunk size
+ tf::StaticPartitioner to enable static scheduling algorithm of static chunk size
+ tf::RandomPartitioner to enable random scheduling algorithm of random chunk size

Depending on applications, partitioning algorithms can impact the performance
a lot. 
For example, if a parallel-iteration workload contains a regular work unit per
iteration, tf::StaticPartitioner can deliver the best performance.
On the other hand, if the work unit per iteration is irregular and unbalanced,
tf::GuidedPartitioner or tf::DynamicPartitioner can outperform tf::StaticPartitioner.
In most situations, tf::GuidedPartitioner can deliver decent performance and
is thus used as our default partitioner.
*/
class PartitionerBase {

  public:

  /**
  @brief default constructor
  */
  PartitionerBase() = default;

  /**
  @brief construct a partitioner with the given chunk size
  */
  explicit PartitionerBase(size_t chunk_size) : _chunk_size {chunk_size} {}

  /**
  @brief query the chunk size of this partitioner
  */
  size_t chunk_size() const { return _chunk_size; }
  
  /**
  @brief update the chunk size of this partitioner
  */
  void chunk_size(size_t cz) { _chunk_size = cz; }

  protected:
  
  /**
  @brief chunk size 
  */
  size_t _chunk_size{0};
};

// ----------------------------------------------------------------------------
// Guided Partitioner
// ----------------------------------------------------------------------------
  
/**
@class GuidedPartitioner

@brief class to construct a guided partitioner for scheduling parallel algorithms

The size of a partition is proportional to the number of unassigned iterations 
divided by the number of workers, 
and the size will gradually decrease to the given chunk size.
The last partition may be smaller than the chunk size.
*/
class GuidedPartitioner : public PartitionerBase {

  public:
  
  /**
  @brief default constructor
  */
  GuidedPartitioner() : PartitionerBase{1} {}

  /**
  @brief construct a guided partitioner with the given chunk size
  */
  explicit GuidedPartitioner(size_t sz) : PartitionerBase (sz) {}
  
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
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {

    size_t chunk_size = (_chunk_size == 0) ? size_t{1} : _chunk_size;

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
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {

    size_t chunk_size = (_chunk_size == 0) ? size_t{1} : _chunk_size;

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

The partitioner splits iterations into many partitions each of size equal to 
the given chunk size.
Different partitions are distributed dynamically to workers 
without any specific order.
*/
class DynamicPartitioner : public PartitionerBase {

  public:

  /**
  @brief default constructor
  */
  DynamicPartitioner() : PartitionerBase{1} {};
  
  /**
  @brief construct a dynamic partitioner with the given chunk size
  */
  explicit DynamicPartitioner(size_t sz) : PartitionerBase (sz) {}
  
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
    size_t N, 
    size_t, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {

    size_t chunk_size = (_chunk_size == 0) ? size_t{1} : _chunk_size;
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
    size_t N, 
    size_t, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {

    size_t chunk_size = (_chunk_size == 0) ? size_t{1} : _chunk_size;
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

@brief class to construct a dynamic partitioner for scheduling parallel algorithms

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
*/
class StaticPartitioner : public PartitionerBase {

  public:

  /**
  @brief default constructor
  */
  StaticPartitioner() : PartitionerBase{0} {};
  
  /**
  @brief construct a dynamic partitioner with the given chunk size
  */
  explicit StaticPartitioner(size_t sz) : PartitionerBase(sz) {}
  
  /**
  @brief queries the adjusted chunk size
  
  Returns the given chunk size if it is not zero, or returns
  <tt>N/W + (w < N%W)</tt>, where @c N is the number of iterations,
  @c W is the number of workers, and @c w is the worker ID.
  */
  size_t adjusted_chunk_size(size_t N, size_t W, size_t w) const {
    return _chunk_size ? _chunk_size : N/W + (w < N%W);
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
    size_t N, 
    size_t W, 
    size_t curr_b, 
    size_t chunk_size,
    F&& func
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
    size_t N, 
    size_t W, 
    size_t curr_b, 
    size_t chunk_size,
    F&& func
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

Similar to tf::DynamicPartitioner, 
the partitioner splits iterations into many partitions but each with a random
chunk size in the range, <tt>c = [alpha * N * W, beta * N * W]</tt>.
By default, @c alpha is <tt>0.01</tt> and @c beta is <tt>0.5</tt>, respectively.

*/
class RandomPartitioner : public PartitionerBase {

  public:

  /**
  @brief default constructor
  */
  RandomPartitioner() = default;
  
  /**
  @brief constructs a random partitioner 
  */
  RandomPartitioner(size_t cz) : PartitionerBase(cz) {}
  
  /**
  @brief constructs a random partitioner with the given parameters
  */
  RandomPartitioner(float alpha, float beta) : _alpha {alpha}, _beta {beta} {}

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
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
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
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
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

Guided partitioner can achieve decent performance for most parallel algorithms,
especially for those with irregular and unbalanced workload per iteration.
*/
using DefaultPartitioner = GuidedPartitioner;

/**
@brief determines if a type is a partitioner 

A partitioner is a derived type from tf::PartitionerBase.
*/
template <typename C>
inline constexpr bool is_partitioner_v = std::is_base_of<PartitionerBase, C>::value;

}  // end of namespace tf -----------------------------------------------------



