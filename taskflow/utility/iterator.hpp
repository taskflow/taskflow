#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <tuple>
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
An invalid index range will return 0.
*/
template <std::integral T>
constexpr size_t distance(T beg, T end, T step) {
  if constexpr (std::is_unsigned_v<T>) {
    return end > beg
      ? static_cast<size_t>((end - beg + step - T{1}) / step)
      : size_t{0};
  } else {
    return static_cast<size_t>(
      std::max(T{0}, (end - beg + step + (step > T{0} ? T{-1} : T{1})) / step)
    );
  }
}

// ============================================================================
// IndexRanges<T, N>
//
// A single class template representing an N-dimensional index range, the
// Cartesian product of N independent 1D ranges (begin, end, step).  Each
// dimension is stored as a std::tuple<T, T, T>, so dim(d) returns a tuple
// that can be read or mutated directly, including via structured bindings:
//
//   auto& [beg, end, step] = ranges.dim(0);
//
// Members that only make sense for a single dimension (begin(), end(),
// step_size(), reset(), unravel()) are gated with `requires (N == 1)`.
// Members that only make sense for more than one dimension (ceil(), floor(),
// upper_slice(), lower_slice()) are
// gated with `requires (N > 1)`.  This keeps
// everything in one class body — instead of a primary template plus a
// partial specialization — which Doxygen can parse and cross-reference
// cleanly.
//
// tf::IndexRange<T> is an alias for tf::IndexRanges<T, 1>, the common 1D
// case.
//
// Iteration order for N > 1 is row-major (the last dimension is innermost /
// fastest), matching the natural loop nesting:
//
//   for i in dim[0]:        // outermost
//     for j in dim[1]:
//       ...
//         for k in dim[N-1]: // innermost
//
// Flat index 0 corresponds to (beg[0], beg[1], ..., beg[N-1]).
// ============================================================================

/**
@class IndexRanges

@brief class to create an N-dimensional index range of integral indices

@tparam T the integral type of the indices
@tparam N the number of dimensions (defaults to 1)

This class represents the Cartesian product of @c N independent 1D index
ranges, each defined by a starting index, ending index, and step size.  Each
dimension is stored as a <tt>std::tuple<T, T, T></tt> of (begin, end, step),
accessible and mutable through @c dim(d).

For @c N == 1, the class behaves like a plain 1D range: @c tf::IndexRange<T>
(an alias for <tt>tf::IndexRanges<T, 1></tt>) exposes convenience accessors
@c begin(), @c end(), @c step_size(), @c reset(), and @c unravel() directly,
without going through @c dim(0).

@code{.cpp}
tf::IndexRange<int> range(0, 10, 2);
for(auto i=range.begin(); i<range.end(); i+=range.step_size()) {
  printf("%d ", i);
}
@endcode

You can reset the range to a different value using tf::IndexRanges::reset.
This is particularly useful when the range value is only known at runtime.

@code{.cpp}
tf::IndexRange<int> range;
range.reset(0, 10, 2);
for(auto i=range.begin(); i<range.end(); i+=range.step_size()) {
  printf("%d ", i);
}
@endcode

@attention
It is the user's responsibility to ensure the given range is valid.
For instance, a range from 0 to 10 with a step size of -2 is invalid.

For @c N > 1, iteration order is row-major: the last dimension varies
fastest, matching the natural nesting of C-style for-loops.

@code{.cpp}
// 3D range: i in [0,4), j in [0,6), k in [0,8), all step 1
tf::IndexRanges<int, 3> r(
  tf::IndexRange<int>(0, 4, 1),
  tf::IndexRange<int>(0, 6, 1),
  tf::IndexRange<int>(0, 8, 1)
);
printf("%zu\n", r.size());  // 4*6*8 = 192
@endcode

@note
If any dimension has zero size (e.g. an empty range such as @c [0,0)), the
active iteration space stops at that dimension.  @c size() returns the product
of all outer dimensions before the first zero, and @c upper_slice / @c lower_slice
copy the zero-size dimension and all inner
dimensions as full extent into each
returned sub-box.  This matches the behaviour of sequential nested loops:

@code{.cpp}
// Zero in the middle dimension: outer i-loop still runs, j/k body never executes.
tf::IndexRanges<int, 3> r(
  tf::IndexRange<int>(0, 100, 1),   // i: 100 iters (active)
  tf::IndexRange<int>(0,   0, 1),   // j:   0 iters (stops accumulation here)
  tf::IndexRange<int>(0, 100, 1)    // k: 100 iters (inactive — never reached)
);
r.size();  // 100  (only the i-dimension contributes)
@endcode
*/
template <std::integral T, size_t N = 1>
class IndexRanges {

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
  @brief constructs an index range without initialization

  The per-dimension ranges are left in an indeterminate state.
  Use this when the bounds will be set later via @c reset() (N==1) or
  @c dim(d) (any N).
  */
  IndexRanges() = default;

  /**
  @brief constructs a 1D index range (only available when @c N == 1)

  @param beg starting index of the range
  @param end ending index of the range (exclusive)
  @param step_size step size between consecutive indices in the range

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);   // elements: 0, 2, 4, 6, 8
  @endcode
  */
  explicit IndexRanges(T beg, T end, T step_size) requires (N == 1)
    : _dims{ std::tuple<T, T, T>{beg, end, step_size} } {}

  /**
  @brief constructs an N-D index range from N 1D ranges

  @param ranges  exactly N 1D ranges (each a <tt>tf::IndexRanges<T, 1></tt>,
                 i.e. a @c tf::IndexRange<T>), one per dimension in order
                 from outermost (dim 0) to innermost (dim N-1)

  Each 1D range defines the @c begin, @c end, and @c step_size for its
  dimension.  Dimensions are independent — any combination of positive,
  negative, or zero step sizes is supported, as long as each 1D range is
  individually valid.

  @code{.cpp}
  // 3D: mixed step sizes
  tf::IndexRanges<int, 3> r3(
    tf::IndexRange<int>(0,  4, 1),   // dim 0: 4 elements
    tf::IndexRange<int>(0, 10, 2),   // dim 1: 5 elements (0,2,4,6,8)
    tf::IndexRange<int>(0,  6, 1)    // dim 2: 6 elements
  );
  r3.size();   // 120
  @endcode
  */
  template <typename... Ranges>
  requires (sizeof...(Ranges) == N) && 
           (std::same_as<std::decay_t<Ranges>, IndexRanges<T, 1>> && ...)
  explicit IndexRanges(Ranges&&... ranges)
    : _dims{ ranges.dim(0)... } {}

  /**
  @brief constructs an index range from an array of (begin, end, step) tuples

  @param dims @c std::array of exactly N (begin, end, step) tuples

  Useful when the per-dimension bounds are constructed programmatically.

  @code{.cpp}
  std::array<std::tuple<int,int,int>, 2> dims = {
    std::tuple{0, 4, 1},
    std::tuple{0, 5, 1}
  };
  tf::IndexRanges<int, 2> r(dims);
  r.size();  // 20
  @endcode
  */
  explicit IndexRanges(const std::array<std::tuple<T, T, T>, N>& dims) : _dims{dims} {}

  // --------------------------------------------------------------------------
  // Dimension access
  // --------------------------------------------------------------------------

  /**
  @brief returns the (begin, end, step) tuple for dimension @c d (read-only)

  @param d zero-based dimension index in @c [0, N)

  @return a const reference to the <tt>std::tuple<T, T, T></tt> for
          dimension @c d, in (begin, end, step) order

  @code{.cpp}
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0,  4, 1),
    tf::IndexRange<int>(0, 10, 2),
    tf::IndexRange<int>(0,  6, 1)
  );

  auto [beg, end, step] = r.dim(1);  // beg=0, end=10, step=2
  @endcode
  */
  const std::tuple<T, T, T>& dim(size_t d) const { return _dims[d]; }

  /**
  @brief returns the (begin, end, step) tuple for dimension @c d (mutable)

  @param d zero-based dimension index in @c [0, N)

  @return a mutable reference to the <tt>std::tuple<T, T, T></tt> for
          dimension @c d

  Use this to update the bounds of an individual dimension at runtime, for
  example inside an upstream init task when using stateful ranges:

  @code{.cpp}
  tf::IndexRanges<int, 2> range;

  auto init = taskflow.emplace([&]() {
    range.dim(0) = {0, rows, 1};  // set row range at runtime
    range.dim(1) = {0, cols, 1};  // set col range at runtime
  });

  auto loop = taskflow.for_each_by_index(std::ref(range), callable);
  init.precede(loop);
  @endcode
  
  Native structured bindings can bind directly to the tuple's elements:

  @code{.cpp}
  // mutate a single field through a structured binding
  auto& [beg, end, step] = range.dim(0);
  beg = 0; end = rows; step = 1;
  @endcode
  */
  std::tuple<T, T, T>& dim(size_t d) { return _dims[d]; }

  // --------------------------------------------------------------------------
  // 1D convenience accessors (only available when N == 1)
  // --------------------------------------------------------------------------

  /**
  @brief queries the starting index of the range (only available when @c N == 1)

  @return the starting index of the range

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);
  auto b = range.begin();  // b == 0
  @endcode
  */
  T begin() const requires (N == 1) { return std::get<0>(_dims[0]); }

  /**
  @brief queries the ending index of the range (only available when @c N == 1)

  @return the ending index (exclusive) of the range

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);
  auto e = range.end();  // e == 10
  @endcode
  */
  T end() const requires (N == 1) { return std::get<1>(_dims[0]); }

  /**
  @brief queries the step size of the range (only available when @c N == 1)

  @return the step size between consecutive indices in the range

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);
  auto s = range.step_size();  // s == 2
  @endcode
  */
  T step_size() const requires (N == 1) { return std::get<2>(_dims[0]); }

  /**
  @brief updates the range with a new starting index, ending index, and step
         size (only available when @c N == 1)

  @param beg new starting index of the range
  @param end new ending index of the range (exclusive)
  @param step_size new step size between consecutive indices in the range

  @return a reference to @c *this

  Use this to rebind a stateful range to new bounds at runtime, for example
  inside an upstream init task:

  @code{.cpp}
  tf::IndexRange<int> range;

  auto init = taskflow.emplace([&]() {
    range.reset(0, 10, 2);  // elements: 0, 2, 4, 6, 8
  });

  auto loop = taskflow.for_each_by_index(std::ref(range), callable);
  init.precede(loop);
  @endcode
  */
  IndexRanges& reset(T beg, T end, T step_size) requires (N == 1);

  /**
  @brief updates the starting index of the range (only available when @c N == 1)

  @param new_begin the new starting index of the range

  @return a reference to @c *this

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);   // elements: 0, 2, 4, 6, 8
  range.begin(4);                        // elements become: 4, 6, 8
  @endcode
  */
  IndexRanges& begin(T new_begin) requires (N == 1);

  /**
  @brief updates the ending index of the range (only available when @c N == 1)

  @param new_end the new ending index (exclusive) of the range

  @return a reference to @c *this

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);   // elements: 0, 2, 4, 6, 8
  range.end(6);                          // elements become: 0, 2, 4
  @endcode
  */
  IndexRanges& end(T new_end) requires (N == 1);

  /**
  @brief updates the step size of the range (only available when @c N == 1)

  @param new_step_size the new step size between consecutive indices

  @return a reference to @c *this

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 1);   // elements: 0, 1, 2, ..., 9
  range.step_size(3);                    // elements become: 0, 3, 6, 9
  @endcode
  */
  IndexRanges& step_size(T new_step_size) requires (N == 1);

  /**
  @brief maps a contiguous index partition back to the corresponding subrange
         (only available when @c N == 1)

  @param part_beg beginning index of the partition (inclusive)
  @param part_end ending index of the partition (exclusive)
  @return a new range covering the elements at positions
          [@c part_beg, @c part_end) in the original range

  Each element of the range can be addressed by a zero-based position index
  from @c 0 to @c size()-1. This function unravels a contiguous slice of those
  position indices back into the original iteration space, returning the
  sub-range whose elements correspond exactly to positions
  [@c part_beg, @c part_end).

  @code{.cpp}
  tf::IndexRange<int> range(0, 10, 2);   // elements: 0, 2, 4, 6, 8
  auto sub = range.unravel(1, 4);        // elements at positions [1,4): 2, 4, 6
  // sub.begin() == 2, sub.end() == 8, sub.step_size() == 2
  @endcode

  @attention
  Users must ensure [@c part_beg, @c part_end) is a valid partition of
  [0, @c size()), i.e., @c part_end <= size().
  */
  IndexRanges unravel(size_t part_beg, size_t part_end) const requires (N == 1);

  // --------------------------------------------------------------------------
  // Size queries (available for any N)
  // --------------------------------------------------------------------------

  /**
  @brief returns the number of iterations along dimension @c d

  @param d zero-based dimension index in @c [0, N)

  @return the element count of dimension @c d

  This is a per-dimension count, independent of other dimensions and of the
  zero-size stopping rule that @c size() applies.

  @code{.cpp}
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0,  4, 1),
    tf::IndexRange<int>(0, 10, 2),
    tf::IndexRange<int>(0,  6, 1)
  );

  r.size(1);  // 5, since the range [0,10) with step 2 has 5 elements
  @endcode
  */
  size_t size(size_t d) const { return std::apply(distance<T>, _dims[d]); }

  /**
  @brief returns the number of active flat iterations

  For @c N == 1 this is simply the element count of the range.  For @c N > 1
  it returns the product of dimension sizes from the outermost dimension
  inward, stopping before the first zero-size dimension.  This matches the
  behaviour of sequential nested loops: a zero-size dimension prevents its
  own iterations and those of all deeper dimensions, but outer dimensions
  are unaffected.

  @code{.cpp}
  // All non-zero: 4 * 6 * 8 = 192
  tf::IndexRanges<int, 3> r1(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 8, 1)
  );
  r1.size();  // 192

  // Zero in middle (d=1): outer dim only -> 4
  tf::IndexRanges<int, 3> r2(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 0, 1),   // stops accumulation here
    tf::IndexRange<int>(0, 8, 1)
  );
  r2.size();  // 4
  @endcode
  */
  size_t size() const;

  // --------------------------------------------------------------------------
  // N-dimensional algorithms (only available when N > 1)
  // --------------------------------------------------------------------------

  /**
  @brief returns the smallest hyperplane-aligned chunk size that is >= chunk_size,
         capped at size() when chunk_size exceeds the total active range size
         (only available when @c N > 1)

  @param chunk_size  hint for the desired number of flat elements

  @return the smallest natural per-step volume of the active dimensions that is
          >= @c chunk_size, or @c size() if @c chunk_size > @c size(), or 0 if
          the outermost dimension is zero-size

  Analogous to @c std::ceil but operating on the discrete set of hyperplane
  boundary sizes of the active dimensions (those before the first zero-size
  dimension). Only active suffix products are considered — inactive inner
  dimensions do not contribute boundaries.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active boundaries: 1, 8, 32, 96  (inactive dims d=3,4,5 contribute nothing)
  tf::IndexRanges<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  r.ceil(1);   //  1 — already a boundary
  r.ceil(5);   //  8 — rounds up to next active boundary
  r.ceil(33);  // 96 — rounds up to full active size
  r.ceil(200); // 96 — capped at size()
  @endcode
  */
  size_t ceil(size_t chunk_size) const requires (N > 1);

  /**
  @brief returns the largest hyperplane-aligned chunk size that is <= chunk_size
         (only available when @c N > 1)

  @param chunk_size  hint for the desired number of flat elements

  @return the largest natural per-step volume of the active dimensions that is
          <= @c chunk_size, or 0 if the outermost dimension is zero-size

  Analogous to @c std::floor; see @c ceil for the boundary semantics.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active boundaries: 1, 8, 32, 96  (inactive dims d=3,4,5 contribute nothing)
  tf::IndexRanges<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  r.floor(5);   //  1 — rounds down to previous active boundary
  r.floor(33);  // 32 — rounds down
  r.floor(200); // 96 — capped at size()
  @endcode
  */
  size_t floor(size_t chunk_size) const requires (N > 1);

  /**
  @brief returns the smallest perfectly-aligned hyperbox reachable from flat_beg,
         rounding up to the next hyperplane boundary when chunk_size is not
         aligned (only available when @c N > 1)

  @param flat_beg  starting flat index (row-major) into the active ND space
  @param chunk_size  hint for the desired number of elements

  @return the sub-box; its @c size() gives the number of elements consumed

  Analogous to @c std::ceil: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact; otherwise it rounds up to the next clean
  orthogonal boundary, so @c size() >= @c chunk_size.

  Only the active dimensions — those before the first zero-size dimension — are
  partitioned. Dimensions from the first zero onward are copied into the
  returned box as full extent and do not affect the flat index space.

  The trailing-zeros rule applies within the active dimensions: a dimension can
  only expand if all active dimensions inner to it start at coordinate 0.
  When this fires the function returns the best geometry-constrained box
  reachable from @c flat_beg.

  Used by dynamic partitioners: the atomic cursor advances by the returned
  box's @c size() and any overshoot is self-correcting.
  */
  IndexRanges upper_slice(size_t flat_beg, size_t chunk_size) const requires (N > 1);

  /**
  @brief returns the largest perfectly-aligned hyperbox reachable from flat_beg
         whose size does not exceed chunk_size, rounding down to the previous
         hyperplane boundary when chunk_size is not aligned (only available
         when @c N > 1)

  @param flat_beg  starting flat index (row-major) into the active ND space
  @param chunk_size  hint for the desired number of elements

  @return the sub-box; its @c size() gives the number of elements consumed

  Analogous to @c std::floor: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact (identical to upper_slice); otherwise
  it rounds down to the largest box that does not exceed @c chunk_size, so
  @c size() <= @c chunk_size.

  Used by static partitioners: each worker's pre-assigned flat quota is never
  exceeded, so workers do not double-process elements at quota boundaries.
  */
  IndexRanges lower_slice(size_t flat_beg, size_t chunk_size) const requires (N > 1);

  /**
  @brief returns a box whose @c size() is 0 (only available when @c N > 1)

  @return a degenerate box with the same dimensions as @c *this, except
          dimension 0 is collapsed to a single point (begin == end)

  Collapsing dimension 0 alone is sufficient: @c size() stops accumulating
  at the first zero-size dimension, so the returned box reports 0 regardless
  of the other dimensions, which are copied through verbatim.

  Used by @c upper_slice and @c lower_slice for the @c chunk_size @c == @c 0
  case, since the returned box's @c size() doubles as the "consumed" count.
  */
  IndexRanges empty_box() const requires (N > 1);

private:

  std::array<std::tuple<T, T, T>, N> _dims;
};

// ============================================================================
// Out-of-class definitions — 1D convenience accessors and size queries
// ============================================================================

template <std::integral T, size_t N>
IndexRanges<T, N>&
IndexRanges<T, N>::reset(T beg, T end, T step_size) requires (N == 1) {
  _dims[0] = {beg, end, step_size};
  return *this;
}

template <std::integral T, size_t N>
IndexRanges<T, N>&
IndexRanges<T, N>::begin(T new_begin) requires (N == 1) {
  std::get<0>(_dims[0]) = new_begin;
  return *this;
}

template <std::integral T, size_t N>
IndexRanges<T, N>&
IndexRanges<T, N>::end(T new_end) requires (N == 1) {
  std::get<1>(_dims[0]) = new_end;
  return *this;
}

template <std::integral T, size_t N>
IndexRanges<T, N>&
IndexRanges<T, N>::step_size(T new_step_size) requires (N == 1) {
  std::get<2>(_dims[0]) = new_step_size;
  return *this;
}

template <std::integral T, size_t N>
IndexRanges<T, N>
IndexRanges<T, N>::unravel(size_t part_beg, size_t part_end) const requires (N == 1) {
  auto [beg, end, step] = _dims[0];
  return IndexRanges(
    static_cast<T>(part_beg) * step + beg,
    static_cast<T>(part_end) * step + beg,
    step
  );
}

template <std::integral T, size_t N>
size_t IndexRanges<T, N>::size() const {
  if constexpr (N == 1) {
    return size(0);
  } else {
    // Compile-time-unrolled recursion: D is a template parameter, so each
    // depth is a distinct instantiation and the recursion fully unrolls
    // (no runtime loop). `self` is passed explicitly because C++20 lambdas
    // cannot otherwise name themselves for recursion.
    auto compute = [this]<size_t D>(auto& self, size_t total) -> size_t {
      if constexpr (D == N) {
        return total;
      } else {
        size_t s = this->size(D);
        if (s == 0) return D == 0 ? 0 : total;  // outermost zero -> 0, inner zero -> outer product
        return self.template operator()<D + 1>(self, total * s);
      }
    };
    return compute.template operator()<0>(compute, 1);
  }
}

// ============================================================================
// Out-of-class definitions — N-dimensional algorithms (N > 1)
// ============================================================================

template <std::integral T, size_t N>
IndexRanges<T, N>
IndexRanges<T, N>::empty_box() const requires (N > 1) {
  std::array<std::tuple<T, T, T>, N> box_dims = _dims;
  box_dims[0] = std::tuple<T, T, T>{
    std::get<0>(_dims[0]), std::get<0>(_dims[0]), std::get<2>(_dims[0])
  };
  return IndexRanges(box_dims);
}

template <std::integral T, size_t N>
size_t IndexRanges<T, N>::ceil(size_t chunk_size) const requires (N > 1) {
  // Pass 1: cache all dim sizes (one call each) and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = size(d);
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }
  if (d_active == 0) return 0;

  // Pass 2: walk suffix products backward within [0, d_active) only.
  // Return the first suffix product >= chunk_size, capped at size().
  size_t inner_volume = 1;
  if (inner_volume >= chunk_size) return inner_volume;
  for (size_t d = d_active; d-- > 0; ) {
    inner_volume *= dim_sizes[d];
    if (inner_volume >= chunk_size) return inner_volume;
  }
  return inner_volume;  // == size() of active dims, capped
}

template <std::integral T, size_t N>
size_t IndexRanges<T, N>::floor(size_t chunk_size) const requires (N > 1) {
  // Pass 1: cache all dim sizes (one call each) and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = size(d);
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }
  if (d_active == 0) return 0;

  // Pass 2: walk suffix products backward within [0, d_active) only.
  // Return the largest suffix product <= chunk_size.
  size_t inner_volume = 1;
  size_t last_fit     = 1;
  for (size_t d = d_active; d-- > 0; ) {
    size_t next = inner_volume * dim_sizes[d];
    if (next > chunk_size) break;
    inner_volume = next;
    last_fit     = inner_volume;
  }
  return last_fit;
}

template <std::integral T, size_t N>
IndexRanges<T, N>
IndexRanges<T, N>::upper_slice(size_t flat_beg, size_t chunk_size) const requires (N > 1) {

  if (chunk_size == 0) {
    return empty_box();
  }

  // Pass 1 (forward): cache dim sizes and find d_active — the first zero-size
  // dimension.  Only [0, d_active) participates in the flat iteration space;
  // dims [d_active, N) are copied as full extent into the returned box.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = size(d);
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }

  if (d_active == 0) return *this;

  // Pass 2a (backward): decode flat_beg into per-dimension coords.
  size_t coords[N] = {};
  size_t temp      = flat_beg;
  for (size_t d = d_active; d-- > 0; ) {
    coords[d] = temp % dim_sizes[d];
    temp     /= dim_sizes[d];
  }

  // Pass 2b (backward, fused with grow_dim search): find grow_dim.
  // Ceil variant: commit grow_dim BEFORE the budget check (round up).
  size_t grow_dim         = d_active - 1;
  size_t inner_volume     = 1;
  size_t active_inner_vol = 1;

  for (size_t d = d_active; d-- > 0; ) {
    // trailing-zeros rule: inner coord non-zero -> can't expand further out
    if (d + 1 < d_active && coords[d + 1] != 0) break;

    // ceil: commit BEFORE budget check
    grow_dim         = d;
    active_inner_vol = inner_volume;

    if (inner_volume >= chunk_size) break;

    inner_volume *= dim_sizes[d];
  }

  // Steps along grow_dim: ceil division, at least 1 for forward progress.
  size_t steps_left    = dim_sizes[grow_dim] - coords[grow_dim];
  size_t steps_needed  = (std::max)(size_t{1}, chunk_size / active_inner_vol);
  size_t steps_to_take = (std::min)(steps_left, steps_needed);

  // Pass 3: construct the sub-box in three clean segments:
  //   [0, grow_dim)      — locked dims (one element each)
  //   [grow_dim]         — the grow dimension
  //   [grow_dim+1, N)    — full inner/inactive extent
  std::array<std::tuple<T, T, T>, N> box_dims;

  for (size_t d = 0; d < grow_dim; ++d) {
    const T beg  = std::get<0>(_dims[d]);
    const T step = std::get<2>(_dims[d]);
    box_dims[d] = std::tuple<T, T, T>{
      beg + static_cast<T>(coords[d]) * step,
      beg + static_cast<T>(coords[d] + 1) * step,
      step
    };
  }

  {
    const T beg  = std::get<0>(_dims[grow_dim]);
    const T step = std::get<2>(_dims[grow_dim]);
    box_dims[grow_dim] = std::tuple<T, T, T>{
      beg + static_cast<T>(coords[grow_dim])                 * step,
      beg + static_cast<T>(coords[grow_dim] + steps_to_take) * step,
      step
    };
  }

  for (size_t d = grow_dim + 1; d < N; ++d) {
    box_dims[d] = _dims[d];  // full inner or inactive extent
  }

  return IndexRanges<T, N>(box_dims);
}

template <std::integral T, size_t N>
IndexRanges<T, N>
IndexRanges<T, N>::lower_slice(size_t flat_beg, size_t chunk_size) const requires (N > 1) {

  if (chunk_size == 0) {
    return empty_box();
  }

  // Pass 1 (forward): cache dim sizes and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = size(d);
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }

  if (d_active == 0) return *this;

  // Pass 2a (backward): decode flat_beg into per-dimension coords.
  size_t coords[N] = {};
  size_t temp      = flat_beg;
  for (size_t d = d_active; d-- > 0; ) {
    coords[d] = temp % dim_sizes[d];
    temp     /= dim_sizes[d];
  }

  // Pass 2b (backward, fused with grow_dim search): find grow_dim.
  // Floor variant: commit grow_dim AFTER the budget check (round down).
  size_t grow_dim         = d_active - 1;
  size_t inner_volume     = 1;
  size_t active_inner_vol = 1;

  for (size_t d = d_active; d-- > 0; ) {
    // trailing-zeros rule
    if (d + 1 < d_active && coords[d + 1] != 0) break;

    // floor: budget check BEFORE committing
    size_t next_vol = inner_volume * dim_sizes[d];
    if (next_vol > chunk_size && inner_volume > 1) break;

    grow_dim         = d;
    active_inner_vol = inner_volume;
    inner_volume     = next_vol;
  }

  // Steps along grow_dim: floor division, at least 1 for forward progress.
  size_t steps_left    = dim_sizes[grow_dim] - coords[grow_dim];
  size_t steps_needed  = (std::max)(size_t{1}, chunk_size / active_inner_vol);
  size_t steps_to_take = (std::min)(steps_left, steps_needed);

  // Pass 3: construct the sub-box in three clean segments.
  std::array<std::tuple<T, T, T>, N> box_dims;

  for (size_t d = 0; d < grow_dim; ++d) {
    const T beg  = std::get<0>(_dims[d]);
    const T step = std::get<2>(_dims[d]);
    box_dims[d] = std::tuple<T, T, T>{
      beg + static_cast<T>(coords[d]) * step,
      beg + static_cast<T>(coords[d] + 1) * step,
      step
    };
  }

  {
    const T beg  = std::get<0>(_dims[grow_dim]);
    const T step = std::get<2>(_dims[grow_dim]);
    box_dims[grow_dim] = std::tuple<T, T, T>{
      beg + static_cast<T>(coords[grow_dim])                 * step,
      beg + static_cast<T>(coords[grow_dim] + steps_to_take) * step,
      step
    };
  }

  for (size_t d = grow_dim + 1; d < N; ++d) {
    box_dims[d] = _dims[d];  // full inner or inactive extent
  }

  return IndexRanges<T, N>(box_dims);
}

// ----------------------------------------------------------------------------
// IndexRange<T> — alias for the common 1D case
// ----------------------------------------------------------------------------

/**
@brief alias for the common 1D case of tf::IndexRanges

@tparam T the integral type of the indices

<tt>tf::IndexRange<T></tt> is equivalent to <tt>tf::IndexRanges<T, 1></tt>.
Class template argument deduction works through the alias, so the
three-argument constructor can be used without an explicit template argument:

@code{.cpp}
tf::IndexRange r(0, 10, 2);          // deduced as tf::IndexRanges<int, 1>
tf::IndexRange<int> s(0, 100, 5);    // same type, written explicitly
@endcode
*/
template <std::integral T>
using IndexRange = IndexRanges<T, 1>;

// ==========================================
// traits
// ==========================================

/**
@brief base type trait to detect if a type is a tf::IndexRanges
@tparam T The type to inspect.
*/
template <typename>
constexpr bool is_index_ranges_v = false;

/**
@brief specialization of the detector for tf::IndexRanges<T, N>

Matches an IndexRanges of ANY dimensionality (1D, 2D, 3D, etc.).

@tparam T the underlying coordinate type (e.g., size_t, int)
@tparam N the number of dimensions
*/
template <typename T, size_t N>
constexpr bool is_index_ranges_v<IndexRanges<T, N>> = true;

/**
@brief concept to check if a type is a tf::IndexRanges, regardless of dimensionality
@tparam R the range type to evaluate

This concept strips cv-qualifiers and references (using std::unwrap_ref_decay_t)
before evaluating, allowing const and reference (including std::reference_wrapper)
types to satisfy the constraint.
*/
template <typename R>
concept IndexRangesLike = is_index_ranges_v<std::decay_t<std::unwrap_ref_decay_t<R>>>;

/**
@brief concept to check if a type is a tf::IndexRanges<T, 1> (i.e., tf::IndexRange<T>)
@tparam R the range type to evaluate

@code{.cpp}
static_assert( tf::IndexRanges1DLike<tf::IndexRange<int>>);        // true
static_assert( tf::IndexRanges1DLike<tf::IndexRanges<int, 1>>);    // true
static_assert(!tf::IndexRanges1DLike<tf::IndexRanges<int, 2>>);    // false
@endcode
*/
template <typename R>
concept IndexRanges1DLike = IndexRangesLike<R> &&
  (std::decay_t<std::unwrap_ref_decay_t<R>>::rank == 1);

/**
@brief concept to check if a type is a tf::IndexRanges<T, N> with rank > 1
@tparam R the range type to evaluate

@code{.cpp}
static_assert(!tf::IndexRangesMDLike<tf::IndexRange<int>>);        // false, rank == 1
static_assert(!tf::IndexRangesMDLike<tf::IndexRanges<int, 1>>);    // false, rank == 1
static_assert( tf::IndexRangesMDLike<tf::IndexRanges<int, 2>>);    // true
static_assert( tf::IndexRangesMDLike<tf::IndexRanges<int, 3>>);    // true
@endcode
*/
template <typename R>
concept IndexRangesMDLike = IndexRangesLike<R> &&
  (std::decay_t<std::unwrap_ref_decay_t<R>>::rank > 1);

}  // end of namespace tf -----------------------------------------------------
