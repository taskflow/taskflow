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

/**
@class DefaultClosureWrapper

@brief class to create a default closure wrapper
*/
class DefaultClosureWrapper {};

// ------------------------------------------------------------------------------------------------
// IndexRangesPartitioner
// ------------------------------------------------------------------------------------------------

/**
@private
*/
template <typename T, size_t N>
class IndexRangesPartitioner {

  public:

  using R = tf::IndexRanges<T, N>;

  explicit IndexRangesPartitioner(const R& ranges);

  size_t size() const;

  size_t active_rank() const;

  // N == 1: no stride walk, no recursion -- direct unravel(), mirroring
  // IndexRanges<T,1>::unravel() exactly (same as what the partitioners'
  // rank==1 branches already do today). Caller (the scheduler) guarantees
  // 0 <= flat_beg < flat_end <= size() -- no bounds checking performed here.
  template <typename F>
  bool for_each_box(size_t flat_beg, size_t flat_end, F&& visit) requires (N == 1);

  // N > 1: general recursive box decomposition. Visits every hyperplane-
  // aligned box covering [flat_beg, flat_end) of the cached ranges' active
  // dims. Caller (the scheduler) guarantees 0 <= flat_beg < flat_end <=
  // size() -- no bounds checking performed here (the recursion's own
  // if (b >= e) { return false; } would already have absorbed an
  // out-of-contract flat_beg >= flat_end regardless, so this was never
  // load-bearing for N > 1 -- only N == 1 needed it removed explicitly).
  template <typename F>
  bool for_each_box(size_t flat_beg, size_t flat_end, F&& visit) requires (N > 1);

  private:

  void _set_point(size_t dim, size_t coord);
  void _set_span(size_t dim, size_t b, size_t e);
  void _emit_middle(size_t dim, size_t b, size_t e);

  const R& _ranges;
  size_t _active_rank;
  std::array<size_t, N> _strides{};
  R _box;
};

// Constructor
template <typename T, size_t N>
IndexRangesPartitioner<T, N>::IndexRangesPartitioner(const R& ranges) : _ranges(ranges) {

  _active_rank = N;
  std::array<size_t, N> extent{};
  for (size_t d = 0; d < N; ++d) {
    extent[d] = _ranges.size(d);
    if (extent[d] == 0) {
      _active_rank = d;
      break;
    }
  }
  if (_active_rank > 0) {
    _strides[_active_rank - 1] = 1;
    for (size_t d = _active_rank - 1; d-- > 0; ) {
      _strides[d] = _strides[d + 1] * extent[d + 1];
    }
  }
  for (size_t d = _active_rank; d < N; ++d) {
    _box.dim(d) = _ranges.dim(d);
  }
}

// Function: size
template <typename T, size_t N>
size_t IndexRangesPartitioner<T, N>::size() const {
  return _ranges.size();
}

// Function: active_rank
template <typename T, size_t N>
size_t IndexRangesPartitioner<T, N>::active_rank() const {
  return _active_rank;
}

// Function: for_each_box
template <typename T, size_t N>
template <typename F>
bool IndexRangesPartitioner<T, N>::for_each_box(size_t flat_beg, size_t flat_end, F&& visit) 
requires (N == 1) {
  if (_active_rank == 0) {
    return false;
  }
  _box = _ranges.unravel(flat_beg, flat_end);
  if constexpr (std::is_same_v<std::invoke_result_t<F, R>, bool>) {
    return visit(_box);
  } else {
    visit(_box);
    return false;
  }
}

// Function: for_each_box
template <typename T, size_t N>
template <typename F>
bool IndexRangesPartitioner<T, N>::for_each_box(size_t flat_beg, size_t flat_end, F&& visit) 
requires (N > 1) {

  if (_active_rank == 0) {
    return false;
  }

  // Runtime recursion over active dims only -- runs once per claim, not
  // per element, so this is O(active_rank) stack frames with O(1) work
  // each (thanks to the precomputed _strides table and the incremental
  // _box updates in _set_point/_set_span/_emit_middle), never O(chunk_size).
  auto recurse = [&](auto& self, size_t dim, size_t b, size_t e) -> bool {
    if (b >= e) {
      return false;
    }

    if (dim == _active_rank - 1) {
      _set_span(dim, b, e);
      if constexpr (std::is_same_v<std::invoke_result_t<F, R>, bool>) {
        return visit(_box);
      } else {
        visit(_box);
        return false;
      }
    }

    size_t s = _strides[dim];
    size_t outer_b = b / s, inner_b = b % s;
    size_t outer_e = e / s, inner_e = e % s;

    if (outer_b == outer_e) {
      _set_point(dim, outer_b);
      return self(self, dim + 1, inner_b, inner_e);
    }
    if (inner_b != 0) {
      _set_point(dim, outer_b);
      if (self(self, dim + 1, inner_b, s)) {
        return true;
      }
      ++outer_b;
    }
    if (outer_b < outer_e) {
      _emit_middle(dim, outer_b, outer_e);
      if constexpr (std::is_same_v<std::invoke_result_t<F, R>, bool>) {
        if (visit(_box)) {
          return true;
        }
      } else {
        visit(_box);
      }
    }
    if (inner_e != 0) {
      _set_point(dim, outer_e);
      if (self(self, dim + 1, 0, inner_e)) {
        return true;
      }
    }
    return false;
  };

  return recurse(recurse, 0, flat_beg, flat_end);
}

// Function: _set_point
template <typename T, size_t N>
void IndexRangesPartitioner<T, N>::_set_point(size_t dim, size_t coord) {
  auto [bd, ed, sd] = _ranges.dim(dim);
  _box.dim(dim) = { static_cast<T>(bd + coord * sd), static_cast<T>(bd + (coord + 1) * sd), sd };
}

// Function: _set_span
template <typename T, size_t N>
void IndexRangesPartitioner<T, N>::_set_span(size_t dim, size_t b, size_t e) {
  auto [bd, ed, sd] = _ranges.dim(dim);
  _box.dim(dim) = { static_cast<T>(bd + b * sd), static_cast<T>(bd + e * sd), sd };
}

// Function: _emit_middle
template <typename T, size_t N>
void IndexRangesPartitioner<T, N>::_emit_middle(size_t dim, size_t b, size_t e) {
  _set_span(dim, b, e);
  for (size_t d = dim + 1; d < _active_rank; ++d) {
    _box.dim(d) = _ranges.dim(d);
  }
}

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

@attention
Giving the partition size of 0 lets the %Taskflow runtime automatically determines
the partition size for the given partitioner.


In addition to partition size, the application can specify a closure wrapper
for a partitioner.
A closure wrapper allows the application to wrap a partitioned task 
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

@attention
The default closure wrapper (tf::DefaultClosureWrapper) does nothing but invoke
the partitioned task (closure).

*/
template <typename C = DefaultClosureWrapper>
class PartitionerBase {

  public:

  /**
  @brief indicating if the given closure wrapper is a default wrapper (i.e., empty)
  */
  constexpr static bool is_default_wrapper_v = std::is_same_v<C, DefaultClosureWrapper>;

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
  explicit PartitionerBase(size_t chunk_size);

  /**
  @brief construct a partitioner with the given chunk size and closure wrapper
  */
  PartitionerBase(size_t chunk_size, C&& closure_wrapper);

  /**
  @brief query the chunk size of this partitioner
  */
  size_t chunk_size() const;

  /**
  @brief update the chunk size of this partitioner
  */
  void chunk_size(size_t cz);

  /**
  @brief acquire an immutable access to the closure wrapper object
  */
  const C& closure_wrapper() const;

  /**
  @brief acquire a mutable access to the closure wrapper object
  */
  C& closure_wrapper();

  /**
  @brief modify the closure wrapper object
  */
  template <typename F>
  void closure_wrapper(F&& fn);

  /**
  @brief wraps the given callable with the associated closure wrapper
  */
  template <typename F>
  decltype(auto) operator () (F&& callable);

  protected:

  /**
  @private
  */
  size_t _chunk_size{0};

  /**
  @private
  */
  C _closure_wrapper;
};

// Constructor
template <typename C>
PartitionerBase<C>::PartitionerBase(size_t chunk_size) : _chunk_size {chunk_size} {
}

// Constructor
template <typename C>
PartitionerBase<C>::PartitionerBase(size_t chunk_size, C&& closure_wrapper) :
  _chunk_size {chunk_size},
  _closure_wrapper {std::forward<C>(closure_wrapper)} {
}

// Function: chunk_size
template <typename C>
size_t PartitionerBase<C>::chunk_size() const {
  return _chunk_size;
}

// Function: chunk_size
template <typename C>
void PartitionerBase<C>::chunk_size(size_t cz) {
  _chunk_size = cz;
}

// Function: closure_wrapper
template <typename C>
const C& PartitionerBase<C>::closure_wrapper() const {
  return _closure_wrapper;
}

// Function: closure_wrapper
template <typename C>
C& PartitionerBase<C>::closure_wrapper() {
  return _closure_wrapper;
}

// Function: closure_wrapper
template <typename C>
template <typename F>
void PartitionerBase<C>::closure_wrapper(F&& fn) {
  _closure_wrapper = std::forward<F>(fn);
}

// Operator: ()
template <typename C>
template <typename F>
decltype(auto) PartitionerBase<C>::operator () (F&& callable) {
  if constexpr(is_default_wrapper_v) {
    return std::forward<F>(callable);
  }
  else {
    // closure wrapper is stateful - capture it by reference
    return [this, c=std::forward<F>(callable)]() mutable { _closure_wrapper(c); };
  }
}

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
A closure wrapper allows the application to wrap a partitioned task 
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
  explicit StaticPartitioner(size_t sz);

  /**
  @brief construct a static partitioner with the given chunk size and the closure
  */
  explicit StaticPartitioner(size_t sz, C&& closure);

  /**
  @brief queries the adjusted chunk size

  Returns the given chunk size if it is not zero, or returns
  <tt>N/W + (w < N%W)</tt>, where @c N is the number of iterations,
  @c W is the number of workers, and @c w is the worker ID.
  */
  size_t adjusted_chunk_size(size_t N, size_t W, size_t w) const;

  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------

  /**
  @private
  */
  template <typename F>
  void loop(size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func);

  /**
  @private

  Static partitioner loop for index ranges of any rank.

  Each worker is pre-assigned a flat quota (chunk_size) starting at curr_b,
  then strides by W*chunk_size to its next partition — the classic static
  strided pattern.  For a 1D range (rank == 1) the quota maps directly to a
  single subrange via unravel(), since there is no hyperplane alignment to
  respect.  For an N-D range (rank > 1) the worker repeatedly calls
  lower_slice to drain its quota in box-shaped pieces; lower_slice guarantees
  consumed <= remaining budget, so curr_b never overshoots curr_e and no
  elements are double-processed across partition boundaries.  Either way,
  curr_b ends up exactly stride past where it started once the quota is
  drained.
  */
  template <IndexRangesLike R, typename F>
  void loop(const R& range, size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func) const;

};

// Constructor
template <typename C>
StaticPartitioner<C>::StaticPartitioner(size_t sz) : PartitionerBase<C>(sz) {
}

// Constructor
template <typename C>
StaticPartitioner<C>::StaticPartitioner(size_t sz, C&& closure) :
  PartitionerBase<C>(sz, std::forward<C>(closure)) {
}

// Function: adjusted_chunk_size
template <typename C>
size_t StaticPartitioner<C>::adjusted_chunk_size(size_t N, size_t W, size_t w) const {
  return this->_chunk_size ? this->_chunk_size : N/W + (w < N%W);
}

// Function: loop
template <typename C>
template <typename F>
void StaticPartitioner<C>::loop(size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func) {
  size_t stride = W * chunk_size;
  while(curr_b < N) {
    size_t curr_e = (std::min)(curr_b + chunk_size, N);
    if constexpr (std::is_same_v<std::invoke_result_t<F, size_t, size_t>, bool>) {
      if(func(curr_b, curr_e)) {
        return;
      }
    } else {
      func(curr_b, curr_e);
    }
    curr_b += stride;
  }
}

// Function: loop
template <typename C>
template <IndexRangesLike R, typename F>
void StaticPartitioner<C>::loop(
  const R& range, size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func
) const {
  IndexRangesPartitioner irp(range);
  size_t stride = W * chunk_size;
  while(curr_b < N) {
    size_t curr_e = (std::min)(curr_b + chunk_size, N);
    if(irp.for_each_box(curr_b, curr_e, func)) {
      return;
    }
    curr_b += stride;
  }
}

// ----------------------------------------------------------------------------
// Guided Partitioner
// ----------------------------------------------------------------------------

/**
@class GuidedPartitioner

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

@brief class to create a guided partitioner for scheduling parallel algorithms

The size of a partition is proportional to the number of unassigned iterations 
divided by the number of workers, 
and the size will gradually decrease to the given chunk size.
The last partition may be smaller than the chunk size.

In addition to partition size, the application can specify a closure wrapper
for a guided partitioner.
A closure wrapper allows the application to wrap a partitioned task 
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
  explicit GuidedPartitioner(size_t sz);

  /**
  @brief construct a guided partitioner with the given chunk size and the closure
  */
  explicit GuidedPartitioner(size_t sz, C&& closure);

  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------

  /**
  @private
  */
  template <typename F>
  void loop(size_t N, size_t W, std::atomic<size_t>& next, F&& func) const;

  /**
  @private
  */
  template <IndexRangesLike R, typename F>
  void loop(const R& range, size_t N, size_t W, std::atomic<size_t>& next, F&& func) const;

};

// Constructor
template <typename C>
GuidedPartitioner<C>::GuidedPartitioner(size_t sz) : PartitionerBase<C> (sz) {
}

// Constructor
template <typename C>
GuidedPartitioner<C>::GuidedPartitioner(size_t sz, C&& closure) :
  PartitionerBase<C>(sz, std::forward<C>(closure)) {
}

// Function: loop
template <typename C>
template <typename F>
void GuidedPartitioner<C>::loop(
  size_t N, size_t W, std::atomic<size_t>& next, F&& func
) const {

  size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;
  size_t p1 = 2 * W * (chunk_size + 1);
  float  p2 = 0.5f / static_cast<float>(W);
  size_t curr_b = next.load(std::memory_order_relaxed);

  while(curr_b < N) {
    size_t r = N - curr_b;
    size_t csize = (r < p1) ? chunk_size : (std::max)(static_cast<size_t>(p2 * r), chunk_size);
    size_t curr_e = (std::min)(curr_b + csize, N);
    if(next.compare_exchange_weak(curr_b, curr_e,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
      if constexpr (std::is_same_v<std::invoke_result_t<F, size_t, size_t>, bool>) {
        if(func(curr_b, curr_e)) {
          return;
        }
      } else {
        func(curr_b, curr_e);
      }
      curr_b = curr_e;
    }
  }
}

// Function: loop
template <typename C>
template <IndexRangesLike R, typename F>
void GuidedPartitioner<C>::loop(
  const R& range, size_t N, size_t W, std::atomic<size_t>& next, F&& func
) const {

  IndexRangesPartitioner irp(range);

  size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;
  size_t p1 = 2 * W * (chunk_size + 1);
  float  p2 = 0.5f / static_cast<float>(W);
  size_t curr_b = next.load(std::memory_order_relaxed);

  while(curr_b < N) {
    size_t r = N - curr_b;
    size_t csize = (r < p1) ? chunk_size : (std::max)(static_cast<size_t>(p2 * r), chunk_size);
    size_t curr_e = (std::min)(curr_b + csize, N);
    if(next.compare_exchange_weak(curr_b, curr_e,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
      if(irp.for_each_box(curr_b, curr_e, func)) {
        return;
      }
      curr_b = curr_e;
    }
  }
}

// ----------------------------------------------------------------------------
// Dynamic Partitioner
// ----------------------------------------------------------------------------

/**
@class DynamicPartitioner

@brief class to create a dynamic partitioner for scheduling parallel algorithms

@tparam C closure wrapper type (default tf::DefaultClosureWrapper)

The partitioner splits iterations into many partitions each of size equal to 
the given chunk size.
Different partitions are distributed dynamically to workers 
without any specific order.

In addition to partition size, the application can specify a closure wrapper
for a dynamic partitioner.
A closure wrapper allows the application to wrap a partitioned task 
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
  explicit DynamicPartitioner(size_t sz);

  /**
  @brief construct a dynamic partitioner with the given chunk size and the closure
  */
  explicit DynamicPartitioner(size_t sz, C&& closure);

  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------

  /**
  @private
  */
  template <typename F>
  void loop(size_t N, size_t, std::atomic<size_t>& next, F&& func) const;

  /**
  @private
  */
  template <IndexRangesLike R, typename F>
  void loop(const R& range, size_t N, size_t, std::atomic<size_t>& next, F&& func) const;

};

// Constructor
template <typename C>
DynamicPartitioner<C>::DynamicPartitioner(size_t sz) : PartitionerBase<C>(sz) {
}

// Constructor
template <typename C>
DynamicPartitioner<C>::DynamicPartitioner(size_t sz, C&& closure) :
  PartitionerBase<C>(sz, std::forward<C>(closure)) {
}

// Function: loop
template <typename C>
template <typename F>
void DynamicPartitioner<C>::loop(size_t N, size_t, std::atomic<size_t>& next, F&& func) const {

  size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;
  size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

  while(curr_b < N) {
    if constexpr (std::is_same_v<std::invoke_result_t<F, size_t, size_t>, bool>) {
      if(func(curr_b, (std::min)(curr_b + chunk_size, N))) {
        return;
      }
    } else {
      func(curr_b, (std::min)(curr_b + chunk_size, N));
    }
    curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
  }
}

// Function: loop
template <typename C>
template <IndexRangesLike R, typename F>
void DynamicPartitioner<C>::loop(
  const R& range, size_t N, size_t, std::atomic<size_t>& next, F&& func
) const {

  IndexRangesPartitioner irp(range);

  size_t curr_b = next.load(std::memory_order_relaxed);
  size_t chunk_size = (this->_chunk_size == 0) ? size_t{1} : this->_chunk_size;

  while(curr_b < N) {

    size_t curr_e = (std::min)(curr_b + chunk_size, N);
    if(next.compare_exchange_weak(curr_b, curr_e,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
      if(irp.for_each_box(curr_b, curr_e, func)) {
        return;
      }
      curr_b = curr_e;
    }
  }
}

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
A closure wrapper allows the application to wrap a partitioned task 
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
  explicit RandomPartitioner(size_t sz);

  /**
  @brief construct a random partitioner with the given chunk size and the closure
  */
  explicit RandomPartitioner(size_t sz, C&& closure);

  /**
  @brief constructs a random partitioner with the given parameters
  */
  RandomPartitioner(float alpha, float beta);

  /**
  @brief constructs a random partitioner with the given parameters and the closure
  */
  RandomPartitioner(float alpha, float beta, C&& closure);

  /**
  @brief queries the @c alpha value
  */
  float alpha() const;

  /**
  @brief queries the @c beta value
  */
  float beta() const;

  /**
  @brief queries the range of chunk size

  @param N number of iterations
  @param W number of workers
  */
  std::pair<size_t, size_t> chunk_size_range(size_t N, size_t W) const;

  // --------------------------------------------------------------------------
  // scheduling methods
  // --------------------------------------------------------------------------

  /**
  @private
  */
  template <typename F>
  void loop(size_t N, size_t W, std::atomic<size_t>& next, F&& func) const;

  /**
  @private
  */
  template <IndexRangesLike R, typename F>
  void loop(const R& range, size_t N, size_t W, std::atomic<size_t>& next, F&& func) const;

  private:

  float _alpha {0.01f};
  float _beta  {0.50f};
};

// Constructor
template <typename C>
RandomPartitioner<C>::RandomPartitioner(size_t sz) : PartitionerBase<C>(sz) {
}

// Constructor
template <typename C>
RandomPartitioner<C>::RandomPartitioner(size_t sz, C&& closure) :
  PartitionerBase<C>(sz, std::forward<C>(closure)) {
}

// Constructor
template <typename C>
RandomPartitioner<C>::RandomPartitioner(float alpha, float beta) : _alpha{alpha}, _beta{beta} {
}

// Constructor
template <typename C>
RandomPartitioner<C>::RandomPartitioner(float alpha, float beta, C&& closure) :
  _alpha {alpha}, _beta {beta},
  PartitionerBase<C>(0, std::forward<C>(closure)) {
}

// Function: alpha
template <typename C>
float RandomPartitioner<C>::alpha() const {
  return _alpha;
}

// Function: beta
template <typename C>
float RandomPartitioner<C>::beta() const {
  return _beta;
}

// Function: chunk_size_range
template <typename C>
std::pair<size_t, size_t> RandomPartitioner<C>::chunk_size_range(size_t N, size_t W) const {

  size_t b1 = static_cast<size_t>(_alpha * N * W);
  size_t b2 = static_cast<size_t>(_beta  * N * W);

  if(b1 > b2) {
    std::swap(b1, b2);
  }

  b1 = (std::max)(b1, size_t{1});
  b2 = (std::max)(b2, b1 + 1);

  return {b1, b2};
}

// Function: loop
template <typename C>
template <typename F>
void RandomPartitioner<C>::loop(
  size_t N, size_t W, std::atomic<size_t>& next, F&& func
) const {

  auto [b1, b2] = chunk_size_range(N, W);

  std::default_random_engine engine {std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(b1, b2);

  size_t chunk_size = dist(engine);
  size_t curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);

  while(curr_b < N) {
    if constexpr (std::is_same_v<std::invoke_result_t<F, size_t, size_t>, bool>) {
      if(func(curr_b, (std::min)(curr_b + chunk_size, N))) {
        return;
      }
    } else {
      func(curr_b, (std::min)(curr_b + chunk_size, N));
    }
    chunk_size = dist(engine);
    curr_b = next.fetch_add(chunk_size, std::memory_order_relaxed);
  }
}

// Function: loop
template <typename C>
template <IndexRangesLike R, typename F>
void RandomPartitioner<C>::loop(
  const R& range, size_t N, size_t W, std::atomic<size_t>& next, F&& func
) const {

  IndexRangesPartitioner irp(range);

  auto [b1, b2] = chunk_size_range(N, W);

  std::default_random_engine engine{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(b1, b2);

  size_t curr_b = next.load(std::memory_order_relaxed);

  while(curr_b < N) {
    size_t curr_e = (std::min)(curr_b + dist(engine), N);
    if(next.compare_exchange_weak(curr_b, curr_e,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
      if(irp.for_each_box(curr_b, curr_e, func)) {
        return;
      }
      curr_b = curr_e;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Concept and Alias
// ------------------------------------------------------------------------------------------------

/**
@brief default partitioner set to tf::GuidedPartitioner

Guided partitioning algorithm can achieve stable and decent performance
for most parallel algorithms.
*/
using DefaultPartitioner = GuidedPartitioner<>;

/**
@brief concept to check if a type is a partitioner

A type satisfies tf::PartitionerLike if it is derived from tf::PartitionerBase.
*/
template <typename P>
concept PartitionerLike = std::derived_from<P, PartitionerBase<typename P::closure_wrapper_type>>;

/**
@brief concept to check if a type is a partitioner (variable template)

@tparam P type to check

Equivalent to tf::PartitionerLike<P>. Provided for backward compatibility.
*/
template <typename P>
inline constexpr bool is_partitioner_v = PartitionerLike<P>;

}  // end of namespace tf -------------------------------------------------------------------------
