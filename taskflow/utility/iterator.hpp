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

@note
If any dimension has zero size (e.g. an empty range such as @c [0,0)), the total
@c size() is zero and the range produces no iterations.  This is consistent with
the behaviour of OpenMP @c collapse — collapsing a nest of loops where any loop
has zero iterations yields an empty iteration space regardless of the other loops:

@code{.cpp}
// j-loop has zero iterations -> total size = 100 * 0 * 100 = 0
tf::IndexRange<int, 3> r(
  tf::IndexRange<int>(0, 100, 1),   // i: 100 iters
  tf::IndexRange<int>(0,   0, 1),   // j:   0 iters
  tf::IndexRange<int>(0, 100, 1)    // k: 100 iters
);
printf("%zu\n", r.size());  // 0 — no work scheduled
@endcode
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

  Equivalent to the product of each dimension's element count.  If any
  dimension has zero size, the result is zero — consistent with OpenMP
  @c collapse semantics where a zero-size loop collapses the entire
  iteration space to empty.

  @code{.cpp}
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 8, 1)
  );
  printf("%zu\n", r.size());  // 192

  tf::IndexRange<int, 3> empty(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 0, 1),   // zero-size dimension
    tf::IndexRange<int>(0, 8, 1)
  );
  printf("%zu\n", empty.size());  // 0
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
  @brief returns the smallest perfectly-aligned hyperbox reachable from flat_beg,
         rounding up to the next hyperplane boundary when chunk_size is not aligned

  @param flat_beg  starting flat index (row-major) into the ND space
  @param chunk_size  hint for the desired number of elements

  @return a pair of (sub-box, consumed) where consumed == sub-box.size()

  Analogous to @c std::ceil: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact; otherwise it rounds up to the next clean
  orthogonal boundary, so @c consumed >= @c chunk_size.

  The returned sub-box is always a valid hyperrectangle.  A dimension can only
  expand if all dimensions inner to it start at coordinate 0 (the
  "trailing-zeros" rule); when this fires the function returns the best
  geometry-constrained box reachable from @c flat_beg.

  Used by dynamic partitioners: the atomic cursor advances by @c consumed and
  any overshoot is self-correcting.

  @code{.cpp}
  // 3D range: 4 x 5 x 10
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // chunk_size=10 aligns exactly to one inner row -> no overshoot
  auto [box1, c1] = r.slice_ceil(0, 10);  // consumed=10

  // chunk_size=30 does not align; rounds up to next boundary (one outer row = 50)
  auto [box2, c2] = r.slice_ceil(0, 30);  // consumed=50

  // geometry-constrained: mid-row start forces a smaller box
  auto [box3, c3] = r.slice_ceil(30, 30); // consumed=20 (<chunk_size)
  @endcode
  */
  std::pair<IndexRange<T, N>, size_t> slice_ceil(size_t flat_beg, size_t chunk_size) const;

  /**
  @brief returns the largest perfectly-aligned hyperbox reachable from flat_beg
         whose size does not exceed chunk_size, rounding down to the previous
         hyperplane boundary when chunk_size is not aligned

  @param flat_beg  starting flat index (row-major) into the ND space
  @param chunk_size  hint for the desired number of elements

  @return a pair of (sub-box, consumed) where consumed == sub-box.size()

  Analogous to @c std::floor: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact (identical to slice_ceil); otherwise
  it rounds down to the largest box that does not exceed @c chunk_size, so
  @c consumed <= @c chunk_size.

  The one irreducible exception: when @c chunk_size is smaller than the volume
  of a single indivisible inner step (which occurs when trailing-zeros forces
  @c grow_dim to a dimension whose per-step volume exceeds @c chunk_size), the
  function still returns that one step to guarantee forward progress.  With
  unit step sizes this exception never fires because the innermost dimension
  always has per-step volume of 1.

  The trailing-zeros rule applies identically to slice_ceil.

  Used by static partitioners: each worker's pre-assigned flat quota is never
  exceeded, so workers do not double-process elements at quota boundaries.

  @code{.cpp}
  // 3D range: 4 x 5 x 10
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // chunk_size=10 aligns exactly -> same as ceil
  auto [box1, c1] = r.slice_floor(0, 10);  // consumed=10

  // chunk_size=30 does not align; rounds down to largest fitting box (one inner row = 10)
  auto [box2, c2] = r.slice_floor(0, 30);  // consumed=10

  // chunk_size=50 aligns exactly to one outer row -> same as ceil
  auto [box3, c3] = r.slice_floor(0, 50);  // consumed=50
  @endcode
  */
  std::pair<IndexRange<T, N>, size_t> slice_floor(size_t flat_beg, size_t chunk_size) const;

  /**
  @brief returns the smallest hyperplane-aligned chunk size that is >= chunk_size,
         capped at size() when chunk_size exceeds the total range size

  @param chunk_size  hint for the desired number of flat elements

  @return the smallest natural per-step volume of this range that is
          >= @c chunk_size, or @c size() if @c chunk_size > @c size()

  Analogous to @c std::ceil but operating on the discrete set of hyperplane
  boundary sizes of this ND range.  The natural boundary sizes are the suffix
  products of the dimension sizes: 1, dim[N-1], dim[N-1]*dim[N-2], ...,
  size().  The returned value is always one of these suffix products.

  When @c chunk_size is already a natural boundary size the return equals
  @c chunk_size exactly — just like @c std::ceil of an integer.

  When @c chunk_size > @c size(), the function returns @c size() since no
  larger aligned boundary exists.  In this case the returned value is less
  than @c chunk_size.

  This is a lightweight query: no box is constructed and no flat index is
  needed.  Use this to snap a raw @c N/W estimate to the nearest hyperplane
  boundary before passing it to the static partitioner loop, so that
  @c slice_floor always returns one box per worker partition (inner while
  loop runs exactly once in the common case).

  @code{.cpp}
  // 3D range: 4 x 5 x 10  (boundary sizes: 1, 10, 50, 200)
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );
  r.ceil(1);   // 1   — already a boundary
  r.ceil(7);   // 10  — rounds up to next boundary (one inner row)
  r.ceil(10);  // 10  — already a boundary
  r.ceil(30);  // 50  — rounds up to next boundary (one outer row)
  r.ceil(50);  // 50  — already a boundary
  r.ceil(100); // 200 — rounds up to full range
  r.ceil(201); // 200 — capped at size() since no larger boundary exists
  @endcode
  */
  size_t ceil(size_t chunk_size) const;

  /**
  @brief returns the largest hyperplane-aligned chunk size that is <= chunk_size

  @param chunk_size  hint for the desired number of flat elements

  @return the largest natural per-step volume of this range that is
          <= @c chunk_size

  Analogous to @c std::floor but operating on the discrete set of hyperplane
  boundary sizes of this ND range.  The natural boundary sizes are the suffix
  products of the dimension sizes: 1, dim[N-1], dim[N-1]*dim[N-2], ...,
  size().  The returned value is always one of these suffix products.

  When @c chunk_size is already a natural boundary size the return equals
  @c chunk_size exactly — just like @c std::floor of an integer, and
  identical to @c ceil in that case.

  @code{.cpp}
  // 3D range: 4 x 5 x 10  (boundary sizes: 1, 10, 50, 200)
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );
  r.floor(1);   // 1   — already a boundary
  r.floor(7);   // 1   — rounds down to previous boundary
  r.floor(10);  // 10  — already a boundary
  r.floor(30);  // 10  — rounds down to previous boundary (one inner row)
  r.floor(50);  // 50  — already a boundary
  r.floor(100); // 50  — rounds down to previous boundary (one outer row)
  @endcode
  */
  size_t floor(size_t chunk_size) const;

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
size_t IndexRange<T, N>::ceil(size_t chunk_size) const {
  // The natural boundary sizes are the suffix products walked from innermost
  // outward: 1, dim[N-1], dim[N-1]*dim[N-2], ..., size().
  // Return the first suffix product that meets or exceeds chunk_size.
  // If any dimension is zero, size()==0 and we return 0 (no work).
  if (size() == 0) return 0;
  size_t inner_volume = 1;
  if (inner_volume >= chunk_size) return inner_volume;
  for (size_t d = N; d-- > 0; ) {
    inner_volume *= _dims[d].size();
    if (inner_volume >= chunk_size) {
      return inner_volume;
    }
  }
  return inner_volume;  // == size(), the largest possible boundary
}

template <std::integral T, size_t N>
size_t IndexRange<T, N>::floor(size_t chunk_size) const {
  // Walk suffix products outward; commit each one as long as it fits within
  // chunk_size.  The last committed value is the answer.
  // If any dimension is zero, size()==0 and we return 0 (no work).
  if (size() == 0) return 0;
  size_t inner_volume = 1;
  size_t last_fit     = 1;
  for (size_t d = N; d-- > 0; ) {
    size_t next = inner_volume * _dims[d].size();
    if (next > chunk_size) {
      break;
    }
    inner_volume = next;
    last_fit     = inner_volume;
  }
  return last_fit;
}

template <std::integral T, size_t N>
std::pair<IndexRange<T, N>, size_t>
IndexRange<T, N>::slice_ceil(size_t flat_beg, size_t chunk_size) const {

  if (chunk_size == 0) {
    return { *this, 0 };
  }

  // Fused pass: cache dim sizes and decode flat_beg to ND coords in one
  // backward traversal.  Also detects a zero-size dimension early (empty range).
  // A single backward pass suffices because the coord decode is naturally
  // innermost-first (temp %= dim_size[d] then /= dim_size[d]).
  size_t dim_sizes[N];
  size_t coords[N];
  size_t temp = flat_beg;
  for (size_t d = N; d-- > 0; ) {
    dim_sizes[d] = _dims[d].size();
    if (dim_sizes[d] == 0) return { *this, 0 };  // empty range
    coords[d] = temp % dim_sizes[d];
    temp     /= dim_sizes[d];
  }

  // Find grow_dim: walk innermost→outermost, record each dim as candidate
  // BEFORE the budget check (ceil: round up to the next boundary).
  // Stop when inner_volume meets chunk_size or trailing-zeros fires.
  size_t grow_dim        = N - 1;
  size_t inner_volume    = 1;
  size_t active_inner_vol = 1;

  for (size_t d = N; d-- > 0; ) {
    if (d + 1 < N && coords[d + 1] != 0) break;  // trailing-zeros rule

    grow_dim        = d;
    active_inner_vol = inner_volume;

    if (inner_volume >= chunk_size) break;

    inner_volume *= dim_sizes[d];
  }

  // Steps along grow_dim: ceil division, at least 1.
  size_t steps_left    = dim_sizes[grow_dim] - coords[grow_dim];
  size_t steps_needed  = (std::max)(size_t{1}, chunk_size / active_inner_vol);
  size_t steps_to_take = (std::min)(steps_left, steps_needed);

  // Construct the orthogonal sub-box.
  std::array<IndexRange<T, 1>, N> box_dims;
  for (size_t d = 0; d < N; ++d) {
    const auto& dim  = _dims[d];
    const T     step = dim.step_size();
    const T     beg  = dim.begin();
    if (d < grow_dim) {
      T b = beg + static_cast<T>(coords[d]) * step;
      box_dims[d] = IndexRange<T, 1>(b, b + step, step);
    } else if (d == grow_dim) {
      box_dims[d] = IndexRange<T, 1>(
        beg + static_cast<T>(coords[d])                 * step,
        beg + static_cast<T>(coords[d] + steps_to_take) * step,
        step
      );
    } else {
      box_dims[d] = dim;
    }
  }

  return { IndexRange<T, N>(box_dims), steps_to_take * active_inner_vol };
}

template <std::integral T, size_t N>
std::pair<IndexRange<T, N>, size_t>
IndexRange<T, N>::slice_floor(size_t flat_beg, size_t chunk_size) const {

  if (chunk_size == 0) {
    return { *this, 0 };
  }

  // Fused pass: cache dim sizes and decode flat_beg to ND coords in one
  // backward traversal.  Also detects a zero-size dimension early (empty range).
  size_t dim_sizes[N];
  size_t coords[N];
  size_t temp = flat_beg;
  for (size_t d = N; d-- > 0; ) {
    dim_sizes[d] = _dims[d].size();
    if (dim_sizes[d] == 0) return { *this, 0 };  // empty range
    coords[d] = temp % dim_sizes[d];
    temp     /= dim_sizes[d];
  }

  // Find grow_dim: walk innermost→outermost, commit each dim AFTER checking
  // the budget (floor: round down to the previous boundary).
  // Stop when next_vol would exceed chunk_size (and inner > 1 to ensure
  // the innermost dim is always committed as the minimum fallback).
  size_t grow_dim        = N - 1;
  size_t active_inner_vol = 1;
  size_t inner_volume    = 1;

  for (size_t d = N; d-- > 0; ) {
    if (d + 1 < N && coords[d + 1] != 0) break;  // trailing-zeros rule

    size_t next_vol = inner_volume * dim_sizes[d];
    if (next_vol > chunk_size && inner_volume > 1) break;

    grow_dim        = d;
    active_inner_vol = inner_volume;
    inner_volume    = next_vol;
  }

  // Steps along grow_dim: floor division, at least 1 for forward progress.
  size_t steps_left    = dim_sizes[grow_dim] - coords[grow_dim];
  size_t steps_needed  = (std::max)(size_t{1}, chunk_size / active_inner_vol);
  size_t steps_to_take = (std::min)(steps_left, steps_needed);

  // Construct the orthogonal sub-box.
  std::array<IndexRange<T, 1>, N> box_dims;
  for (size_t d = 0; d < N; ++d) {
    const auto& dim  = _dims[d];
    const T     step = dim.step_size();
    const T     beg  = dim.begin();
    if (d < grow_dim) {
      T b = beg + static_cast<T>(coords[d]) * step;
      box_dims[d] = IndexRange<T, 1>(b, b + step, step);
    } else if (d == grow_dim) {
      box_dims[d] = IndexRange<T, 1>(
        beg + static_cast<T>(coords[d])                 * step,
        beg + static_cast<T>(coords[d] + steps_to_take) * step,
        step
      );
    } else {
      box_dims[d] = dim;
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

// ==========================================
// traits
// ==========================================

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
