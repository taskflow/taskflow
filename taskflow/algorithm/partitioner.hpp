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
  explicit GuidedPartitioner(size_t sz) : PartitionerBase {sz} {}
  
  /**
  @private
  */
  template <typename F>
  void operator () (
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {

    size_t chunk_size = (_chunk_size == 0) ? size_t{1} : _chunk_size;

    size_t p1 = 2 * W * (chunk_size + 1);
    double p2 = 0.5 / static_cast<double>(W);
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
        size_t curr_e = (q <= r) ? curr_b + q : N;
        if(next.compare_exchange_strong(curr_b, curr_e, std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
          func(curr_b, curr_e);
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
  explicit DynamicPartitioner(size_t sz) : PartitionerBase {sz} {}
  
  /**
  @private
  */
  template <typename F>
  void operator () (
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
};

// ----------------------------------------------------------------------------
// Static Partitioner
// ----------------------------------------------------------------------------

/**
@class StaticPartitioner

@brief class to construct a dynamic partitioner for scheduling parallel algorithms

The partitioner divides iterations into chunks and distributes chunks 
to workers in order.
If the chunk size is not specified, i.e., 0, the partitioner resorts to a chunk size
that equally distributes iterations into workers.
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
  explicit StaticPartitioner(size_t sz) : PartitionerBase{sz} {}
  
  /**
  @private
  */
  template <typename F>
  void operator () (
    size_t N, 
    size_t W, 
    std::atomic<size_t>& next, 
    F&& func
  ) const {
    
    // TODO: strided version is not easy to work with reduce 
    //       
    //size_t chunk_size = (_chunk_size == 0) ? (N + W - 1) / W : _chunk_size;
    //size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    //size_t stride = W * chunk_size;
    //while(curr_b < N) {
    //  size_t curr_e = curr_b + chunk_size;
    //  if(curr_e > N) {
    //    curr_e = N;
    //  }
    //  func(curr_b, curr_e);
    //  curr_b += stride;
    //}
    
    // For now, implement something similar to dynamic partitioner but with
    // a different initial chunk_size value
    size_t chunk_size = (_chunk_size == 0) ? (N + W - 1) / W : _chunk_size;
    size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

    while(curr_b < N) {
      func(curr_b, std::min(curr_b + chunk_size, N));
      curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
    }
  }

};

// ----------------------------------------------------------------------------
// Utility
// ----------------------------------------------------------------------------

/**
@brief determines if a type is a partitioner 

A partitioner is a derived type from tf::PartitionerBase.
*/
template <typename C>
inline constexpr bool is_partitioner_v = std::is_base_of<PartitionerBase, C>::value;

}  // end of namespace tf -----------------------------------------------------



