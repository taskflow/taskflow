#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tf {
 
/**
@brief checks if the given index range is invalid
 
@tparam T integral type of the indices and step
 
@param beg starting index of the range
@param end ending index of the range
@param step step size to traverse the range
 
@return returns @c true if the range is invalid; @c false otherwise.
 
A range is considered invalid under the following conditions:
 + The step is zero and the begin and end values are not equal.
 + A positive range (begin < end) with a non-positive step.
 + A negative range (begin > end) with a non-negative step.
*/
template <std::integral T>
constexpr bool is_index_range_invalid(T beg, T end, T step) {
  return ((step == T{0} && beg != end) ||
          (beg < end && step <= T{0}) ||  // positive range
          (beg > end && step >= T{0}));   // negative range
}
 
/**
@brief calculates the number of iterations in the given index range
 
@tparam T integral type of the indices and step
 
@param beg starting index of the range
@param end ending index of the range
@param step step size to traverse the range
 
@return returns the number of required iterations to traverse the range
 
The distance of a range represents the number of required iterations to traverse the range
from the beginning index to the ending index (exclusive) with the given step size.
 
Example 1:
@code{.cpp}
// Range: 0 to 10 with step size 2
size_t dist = distance(0, 10, 2);  // Returns 5, the sequence is [0, 2, 4, 6, 8]
@endcode
 
Example 2:
@code{.cpp}
// Range: 10 to 0 with step size -2
size_t dist = distance(10, 0, -2);  // Returns 5, the sequence is [10, 8, 6, 4, 2]
@endcode
 
Example 3:
@code{.cpp}
// Range: 5 to 20 with step size 5
size_t dist = distance(5, 20, 5);  // Returns 3, the sequence is [5, 10, 15]
@endcode
 
@attention
It is user's responsibility to ensure the given index range is valid.
For instance, a range from 0 to 10 with a step size of -2 is invalid.
*/
template <std::integral T>
constexpr size_t distance(T beg, T end, T step) {
  if constexpr (std::is_unsigned_v<T>) {
    // step is always positive for unsigned types — standard ceiling division
    return static_cast<size_t>((end - beg + step - T{1}) / step);
  } else {
    // signed: step may be positive or negative
    return static_cast<size_t>((end - beg + step + (step > T{0} ? T{-1} : T{1})) / step);
  }
}
 

// ----------------------------------------------------------------------------
// IndexRange<T, N>  (primary template, N > 1)
// IndexRange<T, 1>  (partial specialization, mirrors original IndexRange<T>)
//
// Forward-declare the primary template so the specialization can reference it.
// ----------------------------------------------------------------------------

/**
@private
*/
template <std::integral T, size_t N = 1>
class IndexRange;


// ============================================================================
// Primary template: N-dimensional index range (N > 1)
//
// Represents the Cartesian product of N 1D IndexRange<T, 1> objects.
//
// Iteration order is row-major (last dimension is innermost / fastest),
// matching the natural loop nesting:
//
//   for i in dim[0]:        // outermost
//     for j in dim[1]:
//       ...
//         for k in dim[N-1]: // innermost
//
// Flat index 0 corresponds to (beg[0], beg[1], ..., beg[N-1]).
// ============================================================================

/**
@class IndexRange

@brief class to create an N-dimensional index range of integral indices

@tparam T the integral type of the indices
@tparam N the number of dimensions (must be > 1; use IndexRange<T, 1> for 1D)

This class represents the Cartesian product of N independent 1D index ranges,
each defined by a starting index, ending index, and step size.  Iteration order
is row-major: the last dimension varies fastest, matching the natural nesting of
C-style for-loops.

@code{.cpp}
// 3D range: i in [0,4), j in [0,6), k in [0,8), all step 1
tf::IndexRange<int, 3> r(
  tf::IndexRange<int>(0, 4, 1),
  tf::IndexRange<int>(0, 6, 1),
  tf::IndexRange<int>(0, 8, 1)
);
printf("%zu\n", r.size());  // 4*6*8 = 192
@endcode

@attention
It is the user's responsibility to ensure each per-dimension range is valid.
*/
template <std::integral T, size_t N>
class IndexRange {

public:

  /**
  @brief alias for the index type
  */
  using index_type = T;

  /**
  @brief the number of dimensions
  */
  static constexpr size_t rank = N;

  // --------------------------------------------------------------------------
  // Construction
  // --------------------------------------------------------------------------

  /**
  @brief constructs an N-dimensional index range without initialization
  */
  IndexRange() = default;

  /**
  @brief constructs an N-D index range from N 1D IndexRange<T, 1> objects

  @code{.cpp}
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 10, 2),   // dim 0: 0,2,4,6,8
    tf::IndexRange<int>(0,  6, 3)    // dim 1: 0,3
  );
  @endcode
  */
  template <typename... Ranges>
    requires (sizeof...(Ranges) == N) &&
             (std::same_as<std::remove_cvref_t<Ranges>, IndexRange<T, 1>> && ...)
  explicit IndexRange(Ranges&&... ranges)
    : _dims{ std::forward<Ranges>(ranges)... } {}

  /**
  @brief constructs an N-D index range from an array of 1D ranges
  */
  explicit IndexRange(const std::array<IndexRange<T, 1>, N>& dims);

  // --------------------------------------------------------------------------
  // Dimension access
  // --------------------------------------------------------------------------

  /**
  @brief returns the 1D range for dimension @c d (read-only)
  */
  const IndexRange<T, 1>& dim(size_t d) const { return _dims[d]; }

  /**
  @brief returns the 1D range for dimension @c d (mutable)
  */
  IndexRange<T, 1>& dim(size_t d) { return _dims[d]; }

  /**
  @brief returns the underlying array of per-dimension ranges
  */
  const std::array<IndexRange<T, 1>, N>& dims() const { return _dims; }

  // --------------------------------------------------------------------------
  // Size queries
  // --------------------------------------------------------------------------

  /**
  @brief returns the number of iterations along dimension @c d
  */
  size_t size(size_t d) const { return _dims[d].size(); }

  /**
  @brief returns the total number of iterations across all dimensions

  Equivalent to the product of each dimension's element count.

  @code{.cpp}
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 8, 1)
  );
  printf("%zu\n", r.size());  // 192
  @endcode
  */
  size_t size() const;

  // --------------------------------------------------------------------------
  // Flat <-> ND coordinate mapping
  // --------------------------------------------------------------------------

  /**
  @brief maps a zero-based flat index to its N-D index coordinates (row-major)

  @param flat zero-based flat index in `[0, size())`
  @return array of N index values, one per dimension

  @code{.cpp}
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),   // dim 0: 0, 1, 2
    tf::IndexRange<int>(0, 4, 1)    // dim 1: 0, 1, 2, 3
  );
  auto c = r.coords(5);  // flat 5 -> row 1, col 1 -> {1, 1}
  @endcode
  */
  std::array<T, N> coords(size_t flat) const;

  /**
  @brief consumes a linear chunk of the multidimensional space and returns the largest
         valid N-dimensional sub-box

  @param flat_beg the starting linear index (typically fetched from an atomic cursor)
  @param requested_size the maximum number of elements allowed to consume (chunk size hint)

  @return a pair containing the valid sub-box and the actual elements consumed

  This function maps a linear index from a flat iteration space back into the
  multidimensional geometric space. To ensure the returned sub-box remains a valid,
  perfectly orthogonal hyper-rectangle, the function enforces a boundary constraint
  (the "trailing zeros" rule): a dimension can only grow if all dimensions inner to it
  are currently at coordinate 0.

  Consequently, the actual number of elements consumed may be less than
  the @c requested_size if a geometric boundary is hit. This guarantees that workers
  never process discontiguous memory blocks or skip elements.

  @code{.cpp}
  // 3D range: 4 x 5 x 10
  tf::IndexRange<int, 3> range(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // Scenario 1: flat_beg=0, requested=30
  // Coords (0,0,0). inner_volume reaches 50 at dim-0 (first >= 30), so grow_dim=0,
  // steps_to_take=1. The box overshoots to the next orthogonal boundary.
  // box is [0,1) x [0,5) x [0,10), consumed=50
  auto [box1, consumed1] = range.consume_chunk(0, 30);

  // Scenario 2: flat_beg=30, requested=30
  // Coords (0,3,0). coords[1]=3 != 0, so trailing-zeros fires at d=0.
  // grow_dim=1, active=10. steps_left=2, steps_needed=3, steps_to_take=2.
  // box is [0,1) x [3,5) x [0,10), consumed=20 (geometry-constrained, < requested)
  auto [box2, consumed2] = range.consume_chunk(30, 30);

  // Scenario 3: flat_beg=55, requested=30
  // Coords (1,0,5). coords[2]=5 != 0, trailing-zeros fires at d=1.
  // grow_dim=2, active=1. steps_left=5, steps_needed=30, steps_to_take=5.
  // box is [1,2) x [0,1) x [5,10), consumed=5 (geometry-constrained, < requested)
  auto [box3, consumed3] = range.consume_chunk(55, 30);

  // Scenario 4: exact fit — requested equals a natural dimension boundary
  // flat_beg=0, requested=10. inner_volume reaches 10 at d=1 (10>=10), stops there.
  // grow_dim=1, active=10. steps_left=5, steps_needed=1, steps_to_take=1.
  // box is [0,1) x [0,1) x [0,10), consumed=10 (no overshoot needed)
  auto [box4, consumed4] = range.consume_chunk(0, 10);
  @endcode
  */
  std::pair<IndexRange<T, N>, size_t> consume_chunk(size_t flat_beg, size_t requested_size) const;

  private:

  std::array<IndexRange<T, 1>, N> _dims;
};

// ============================================================================
// Out-of-class definitions — IndexRange<T, N>  (primary template)
// ============================================================================

template <std::integral T, size_t N>
IndexRange<T, N>::IndexRange(const std::array<IndexRange<T, 1>, N>& dims) : _dims{dims} {}

template <std::integral T, size_t N>
size_t IndexRange<T, N>::size() const {
  size_t total = 1;
  for (size_t d = 0; d < N; ++d) total *= _dims[d].size();
  return total;
}

template <std::integral T, size_t N>
std::array<T, N> IndexRange<T, N>::coords(size_t flat) const {
  std::array<T, N> c;
  for (size_t d = N; d-- > 0; ) {
    size_t sz  = _dims[d].size();
    size_t pos = flat % sz;
    c[d] = _dims[d].begin() + static_cast<T>(pos) * _dims[d].step_size();
    flat /= sz;
  }
  return c;
}

template <std::integral T, size_t N>
std::pair<IndexRange<T, N>, size_t>
IndexRange<T, N>::consume_chunk(size_t flat_beg, size_t requested_size) const {

  if (requested_size == 0) {
    return { *this, 0 };
  }

  std::array<size_t, N> coords;
  size_t temp = flat_beg;

  // 1. Convert flat_beg to ND coordinates.
  // We use the standard unsigned reverse iteration idiom (d-- > 0) to avoid
  // -Wsign-compare warnings while correctly wrapping around after 0.
  for (size_t d = N; d-- > 0; ) {
    size_t dim_size = _dims[d].size();
    coords[d] = temp % dim_size;
    temp /= dim_size;
  }

  // 2. Find the outermost dimension whose cumulative inner volume first meets
  //    or exceeds requested_size (the "minimum overshooting box" rule).
  //
  //    We walk from the innermost dimension outward, recording grow_dim on
  //    every iteration BEFORE checking the budget.  This means grow_dim always
  //    refers to the current candidate and active_inner_vol is its cumulative
  //    volume.  We stop as soon as inner_volume >= requested_size (budget met)
  //    or the trailing-zeros rule fires (geometry prevents further growth).
  //
  //    Trailing-zeros rule: dimension d cannot be the grow dimension if any
  //    dimension inner to it is not at coordinate 0.  Growing d in that case
  //    would produce a discontiguous "stair-step" shape rather than an
  //    orthogonal hyper-rectangle.  When this fires we break immediately,
  //    returning the best box reachable regardless of requested_size.
  size_t grow_dim = N - 1;
  size_t inner_volume = 1;
  size_t active_inner_vol = 1;

  for (size_t d = N; d-- > 0; ) {
    // Trailing-zeros rule: an inner dimension is not at coordinate 0.
    // No outer dimension can grow; return the best box we can form here.
    if (d + 1 < N && coords[d + 1] != 0) {
      break;
    }

    // Record this dimension as the current candidate before the budget check.
    grow_dim = d;
    active_inner_vol = inner_volume;

    // Stop as soon as the cumulative volume meets the requested size.
    // This is the outermost dimension we need — going further would produce
    // a larger box than necessary.
    if (inner_volume >= requested_size) {
      break;
    }

    // Branchless accumulation: multiply unconditionally.  The potential
    // dead-store when d == 0 is harmless because the loop ends immediately.
    inner_volume *= _dims[d].size();
  }

  // 3. Determine how many steps to take along grow_dim.
  //
  //    active_inner_vol is the volume of one step along grow_dim.  When
  //    active_inner_vol >= requested_size (the budget was already met at this
  //    dimension or the trailing-zeros rule fired with active_inner_vol == 1),
  //    integer division yields 0 and std::max promotes it to 1, taking exactly
  //    one indivisible step — the minimum overshooting box.
  size_t steps_left   = _dims[grow_dim].size() - coords[grow_dim];
  size_t steps_needed = (std::max)(size_t{1}, requested_size / active_inner_vol);
  size_t steps_to_take = (std::min)(steps_left, steps_needed);

  // 4. Construct the orthogonal sub-box.
  std::array<IndexRange<T, 1>, N> box_dims;
  for (size_t d = 0; d < N; ++d) {
    if (d < grow_dim) {
      // Outer dimensions are locked at their current coordinate.
      // We use coords[d] + 1 for the end bound because C++ uses half-open
      // intervals [begin, end). To lock a dimension to exactly ONE element,
      // the end bound must be exactly 1 step past the begin bound.
      box_dims[d] = IndexRange<T, 1>(
        _dims[d].begin() + static_cast<T>(coords[d])     * _dims[d].step_size(),
        _dims[d].begin() + static_cast<T>(coords[d] + 1) * _dims[d].step_size(),
        _dims[d].step_size()
      );
    } else if (d == grow_dim) {
      // The chosen dimension expands by the calculated steps.
      box_dims[d] = IndexRange<T, 1>(
        _dims[d].begin() + static_cast<T>(coords[d])                 * _dims[d].step_size(),
        _dims[d].begin() + static_cast<T>(coords[d] + steps_to_take) * _dims[d].step_size(),
        _dims[d].step_size()
      );
    } else {
      // Inner dimensions take their full extent.
      box_dims[d] = _dims[d];
    }
  }

  return { IndexRange<T, N>(box_dims), steps_to_take * active_inner_vol };
}


// ============================================================================
// Partial specialization: IndexRange<T, 1>  (1D case)
//
// This is the original IndexRange<T> class, re-expressed as the N=1
// specialization so that IndexRange<int, 1> and IndexRange<int> (via the
// default argument N=1 on the primary template forward declaration above) both
// refer to the same type.
// ============================================================================

/**
@class IndexRange<T, 1>

@brief class to create a 1D index range of integral indices with a step size

@tparam T the integral type of the indices

This class provides functionality for managing a range of indices, where the range 
is defined by a starting index, an ending index, and a step size. 
Indices must be an integral type.
For example, the range `[0, 10)` with a step size 2 represents the five elements,
0, 2, 4, 6, and 8.

@code{.cpp}
tf::IndexRange<int> range(0, 10, 2);
for(auto i=range.begin(); i<range.end(); i+=range.step_size()) {
  printf("%d ", i);
}
@endcode

You can reset the range to a different value using tf::IndexRange::reset. 
This is particularly useful when the range value is only known at runtime.

@code{.cpp}
tf::IndexRange<int> range;
range.reset(0, 10, 2);
for(auto i=range.begin(); i<range.end(); i+=range.step_size()) {
  printf("%d ", i);
}
@endcode

@attention
It is user's responsibility to ensure the given range is valid.
For instance, a range from 0 to 10 with a step size of -2 is invalid.
*/
template <std::integral T>
class IndexRange<T, 1> {

  public:

  /**
  @brief alias for the index type used in the range
  */
  using index_type = T;

  /**
  @brief the number of dimensions (always 1 for this specialization)
  */
  static constexpr size_t rank = 1;

  /**
  @brief constructs an index range object without any initialization
  */
  IndexRange() = default;

  /**
  @brief constructs an IndexRange object
  @param beg starting index of the range
  @param end ending index of the range (exclusive)
  @param step_size step size between consecutive indices in the range
  */
  explicit IndexRange(T beg, T end, T step_size)
    : _beg{beg}, _end{end}, _step_size{step_size} {}

  /**
  @brief queries the starting index of the range
  */
  T begin() const { return _beg; }

  /**
  @brief queries the ending index of the range
  */
  T end() const { return _end; }

  /**
  @brief queries the step size of the range
  */
  T step_size() const { return _step_size; }

  /**
  @brief updates the range with the new starting index, ending index, and step size
  */
  IndexRange<T, 1>& reset(T begin, T end, T step_size) {
    _beg = begin;
    _end = end;
    _step_size = step_size;
    return *this;
  }

  /**
  @brief updates the starting index of the range
  */
  IndexRange<T, 1>& begin(T new_begin) { _beg = new_begin; return *this; }

  /**
  @brief updates the ending index of the range
  */
  IndexRange<T, 1>& end(T new_end) { _end = new_end; return *this; }

  /**
  @brief updates the step size of the range
  */
  IndexRange<T, 1>& step_size(T new_step_size) { _step_size = new_step_size; return *this; }

  /**
  @brief queries the number of elements in the range
  
  The number of elements is equivalent to the number of iterations in the range.
  For instance, the range [0, 10) with step size of 2 will iterate five elements,
  0, 2, 4, 6, and 8.

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);
  printf("%zu\n", range.size());        // 5 (0, 2, 4, 6, 8)
  @endcode
  */
  size_t size() const;

  /**
  @brief maps a contiguous index partition back to the corresponding subrange

  @param part_beg beginning index of the partition (inclusive)
  @param part_end ending index of the partition (exclusive)
  @return a new IndexRange<T, 1> covering the elements at positions
          [@c part_beg, @c part_end) in the original range

  Each element of the range can be addressed by a zero-based position index
  from @c 0 to @c size()-1. This function unravels a contiguous slice of those
  position indices back into the original iteration space, returning the
  sub-range whose elements correspond exactly to positions
  [@c part_beg, @c part_end).

  For example, the range [0, 10) with step size 2 contains five elements at
  positions 0->0, 1->2, 2->4, 3->6, 4->8. Unraveling the partition [1, 4) yields
  the subrange [2, 8) with the same step size 2, whose elements are 2, 4, and 6.

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);   // elements: 0, 2, 4, 6, 8
  auto sub = range.unravel(1, 4);        // elements at positions [1,4): 2, 4, 6
  // sub.begin() == 2, sub.end() == 8, sub.step_size() == 2
  @endcode

  This is particularly useful when partitioning work across parallel workers:
  each worker receives a position-space partition [@c part_beg, @c part_end) and
  calls @c unravel to recover the actual index subrange it should process.

  @attention
  Users must ensure [@c part_beg, @c part_end) is a valid partition of
  [0, @c size()), i.e., @c part_end <= size().
  */
  IndexRange<T, 1> unravel(size_t part_beg, size_t part_end) const;

  private:

  T _beg;
  T _end;
  T _step_size;

};



// ============================================================================
// Out-of-class definitions — IndexRange<T, 1>  (1D specialization)
// ============================================================================

template <std::integral T>
size_t IndexRange<T, 1>::size() const {
  return distance(_beg, _end, _step_size);
}

template <std::integral T>
IndexRange<T, 1> IndexRange<T, 1>::unravel(size_t part_beg, size_t part_end) const {
  return IndexRange<T, 1>(
    static_cast<T>(part_beg) * _step_size + _beg,
    static_cast<T>(part_end) * _step_size + _beg,
    _step_size
  );
}

// ----------------------------------------------------------------------------
// Deduction guide: IndexRange(beg, end, step) -> IndexRange<T, 1>
//
// Required because IndexRange is now a two-parameter template (T, N).
// Without this guide, CTAD cannot deduce N from a three-argument constructor
// call such as `tf::IndexRange range(0, 10, 2)`.
// ----------------------------------------------------------------------------

/**
@brief deduction guide for tf::IndexRange<T, 1>

@tparam T integral type, deduced from the three constructor arguments

Allows class template argument deduction (CTAD) from a three-argument
constructor call, mapping the common case @c IndexRange(beg, end, step)
to the 1D specialization tf::IndexRange<T, 1> without requiring an explicit
template argument.

@code{.cpp}
tf::IndexRange r(0, 10, 2);          // deduced as IndexRange<int, 1>
tf::IndexRange s(0ul, 100ul, 5ul);   // deduced as IndexRange<size_t, 1>
@endcode

Without this guide, CTAD cannot resolve @c N from a three-argument
call because @c IndexRange is a two-parameter template
(@c T and @c N). The guide explicitly pins @c N = 1 for the `(beg, end, step)` form.
*/
template <std::integral T>
IndexRange(T, T, T) -> IndexRange<T, 1>;

/**
@brief base type trait to detect if a type is an IndexRange
@tparam T The type to inspect.
*/
template <typename>
constexpr bool is_index_range_v = false;

/**
@brief specialization of the detector for tf::IndexRange<T, N>

Matches an IndexRange of ANY dimensionality (1D, 2D, 3D, etc.).

@tparam T the underlying coordinate type (e.g., size_t, int)
@tparam N the number of dimensions
*/
template <typename T, size_t N>
constexpr bool is_index_range_v<IndexRange<T, N>> = true;

// ==========================================
// C++20 Concept (Public API)
// ==========================================

/**
@brief concept to check if a type an tf::IndexRange, regardless of dimensionality
@tparam R the range type to evaluate

This concept strips cv-qualifiers and references (using std::unwrap_ref_decay_t)
before evaluating, allowing const and reference types to satisfy the constraint.
*/
template <typename R>
concept IndexRangeLike = is_index_range_v<std::decay_t<std::unwrap_ref_decay_t<R>>>;

/**
@brief base type trait to detect if a type is a 1D IndexRange
@tparam T the type to inspect
*/
template <typename T>
constexpr bool is_1d_index_range_v = false;

/**
@brief specialization of the detector for tf::IndexRange<T, 1>
@tparam T the underlying coordinate type (e.g., `size_t`, `int`)
*/
template <typename T>
constexpr bool is_1d_index_range_v<IndexRange<T, 1>> = true;

/**
@brief concept to check if a type is a tf::IndexRange<T, 1>.
@tparam R the range type to evaluate

@code{.cpp}
static_assert( tf::IndexRange1DLike<tf::IndexRange<int>>);      // true
static_assert( tf::IndexRange1DLike<tf::IndexRange<int, 1>>);   // true
static_assert(!tf::IndexRange1DLike<tf::IndexRange<int, 2>>);   // false
@endcode
*/
template <typename R>
concept IndexRange1DLike = is_1d_index_range_v<std::decay_t<std::unwrap_ref_decay_t<R>>>;

/**
@brief base type trait to detect if a type is a multi-dimensional IndexRange (rank > 1)
@tparam T the type to inspect
*/
template <typename T>
constexpr bool is_md_index_range_v = false;

/**
@brief specialization of the detector for tf::IndexRange<T, N> where N > 1
@tparam T the underlying coordinate type (e.g., `size_t`, `int`)
@tparam N the number of dimensions (must be > 1)
*/
template <typename T, size_t N>
requires (N > 1)
constexpr bool is_md_index_range_v<IndexRange<T, N>> = true;

/**
@brief concept to check if a type is a tf::IndexRange<T, N> with rank > 1
@tparam R the range type to evaluate

@code{.cpp}
static_assert(!tf::IndexRangeMDLike<tf::IndexRange<int>>);      // false, rank == 1
static_assert(!tf::IndexRangeMDLike<tf::IndexRange<int, 1>>);   // false, rank == 1
static_assert( tf::IndexRangeMDLike<tf::IndexRange<int, 2>>);   // true
static_assert( tf::IndexRangeMDLike<tf::IndexRange<int, 3>>);   // true
@endcode
*/
template <typename R>
concept IndexRangeMDLike = is_md_index_range_v<std::decay_t<std::unwrap_ref_decay_t<R>>>;

}  // end of namespace tf -----------------------------------------------------
