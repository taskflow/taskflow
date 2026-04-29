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
If any dimension has zero size (e.g. an empty range such as @c [0,0)), the
active iteration space stops at that dimension.  @c size() returns the product
of all outer dimensions before the first zero, and @c slice_ceil / @c slice_floor
copy the zero-size dimension and all inner dimensions as full extent into each
returned sub-box.  This matches the behaviour of sequential nested loops:

@code{.cpp}
// Zero in the middle dimension: outer i-loop still runs, j/k body never executes.
tf::IndexRange<int, 3> r(
  tf::IndexRange<int>(0, 100, 1),   // i: 100 iters (active)
  tf::IndexRange<int>(0,   0, 1),   // j:   0 iters (stops accumulation here)
  tf::IndexRange<int>(0, 100, 1)    // k: 100 iters (inactive — never reached)
);
r.size();  // 100  (only the i-dimension contributes)

// High-dimensional example: 15D with zero at d=7
// size() = product of dims [0..6] only
tf::IndexRange<int, 15> r2(
  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
  tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1), tf::IndexRange<int>(0,7,1),
  tf::IndexRange<int>(0,8,1), tf::IndexRange<int>(0,0,1),  // d=7: zero
  tf::IndexRange<int>(0,9,1), tf::IndexRange<int>(0,10,1), tf::IndexRange<int>(0,11,1),
  tf::IndexRange<int>(0,12,1),tf::IndexRange<int>(0,13,1), tf::IndexRange<int>(0,14,1),
  tf::IndexRange<int>(0,15,1)
);
r2.size();  // 2*3*4*5*6*7*8 = 40320
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

  The per-dimension ranges are left in an indeterminate state.
  Use this when the bounds will be set later via @c dim(d).reset().
  */
  IndexRange() = default;

  /**
  @brief constructs an N-D index range from N 1D IndexRange<T, 1> objects

  @param ranges  exactly N 1D ranges, one per dimension in order from
                 outermost (dim 0) to innermost (dim N-1)

  Each 1D range defines the @c begin, @c end, and @c step_size for its
  dimension.  Dimensions are independent — any combination of positive,
  negative, or zero step sizes is supported, as long as each 1D range is
  individually valid.

  @code{.cpp}
  // 2D: 4 rows, 5 columns (unit steps)
  tf::IndexRange<int, 2> r2(
    tf::IndexRange<int>(0,  4, 1),   // dim 0 (rows):    0,1,2,3
    tf::IndexRange<int>(0,  5, 1)    // dim 1 (columns): 0,1,2,3,4
  );
  r2.size();   // 20

  // 3D: mixed step sizes
  tf::IndexRange<int, 3> r3(
    tf::IndexRange<int>(0,  4, 1),   // dim 0: 4 elements
    tf::IndexRange<int>(0, 10, 2),   // dim 1: 5 elements (0,2,4,6,8)
    tf::IndexRange<int>(0,  6, 1)    // dim 2: 6 elements
  );
  r3.size();   // 120

  // 3D: innermost dim has zero size — outer two dims still active
  tf::IndexRange<int, 3> r4(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 0, 1)    // zero-size: only dims 0,1 contribute
  );
  r4.size();   // 20  (4 * 5, not 0)
  @endcode
  */
  template <typename... Ranges>
    requires (sizeof...(Ranges) == N) &&
             (std::same_as<std::remove_cvref_t<Ranges>, IndexRange<T, 1>> && ...)
  explicit IndexRange(Ranges&&... ranges)
    : _dims{ std::forward<Ranges>(ranges)... } {}

  /**
  @brief constructs an N-D index range from an array of 1D ranges

  @param dims  @c std::array of exactly N 1D ranges

  Equivalent to the variadic constructor but takes a pre-built array,
  which is useful when the ranges are constructed programmatically.

  @code{.cpp}
  std::array<tf::IndexRange<int, 1>, 3> dims = {
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  };
  tf::IndexRange<int, 3> r(dims);
  r.size();  // 120
  @endcode
  */
  explicit IndexRange(const std::array<IndexRange<T, 1>, N>& dims);

  // --------------------------------------------------------------------------
  // Dimension access
  // --------------------------------------------------------------------------

  /**
  @brief returns the 1D range for dimension @c d (read-only)

  @param d zero-based dimension index in @c [0, N)

  @return a const reference to the 1D range for dimension @c d,
          which exposes @c begin(), @c end(), and @c step_size()

  Use this inside a @c for_each_by_index callable to iterate the indices
  assigned to each dimension of the delivered sub-box.

  @code{.cpp}
  // 3D range: i in [0,4), j in [0,10,2) (step 2), k in [0,6)
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0,  4, 1),
    tf::IndexRange<int>(0, 10, 2),
    tf::IndexRange<int>(0,  6, 1)
  );

  r.dim(0).begin();      // 0
  r.dim(0).end();        // 4
  r.dim(0).step_size();  // 1
  r.dim(0).size();       // 4

  r.dim(1).begin();      // 0
  r.dim(1).end();        // 10
  r.dim(1).step_size();  // 2
  r.dim(1).size();       // 5  (values: 0, 2, 4, 6, 8)

  // Typical usage inside a for_each_by_index callable:
  taskflow.for_each_by_index(r, [](const tf::IndexRange<int, 3>& sub) {
    for(int i = sub.dim(0).begin(); i < sub.dim(0).end(); i += sub.dim(0).step_size()) {
      for(int j = sub.dim(1).begin(); j < sub.dim(1).end(); j += sub.dim(1).step_size()) {
        for(int k = sub.dim(2).begin(); k < sub.dim(2).end(); k += sub.dim(2).step_size()) {
          // process element at (i, j, k)
        }
      }
    }
  });
  @endcode
  */
  const IndexRange<T, 1>& dim(size_t d) const { return _dims[d]; }

  /**
  @brief returns the 1D range for dimension @c d (mutable)

  @param d zero-based dimension index in @c [0, N)

  @return a mutable reference to the 1D range for dimension @c d

  Use this to update the bounds of an individual dimension at runtime,
  for example inside an upstream init task when using stateful ranges.

  @code{.cpp}
  tf::IndexRange<int, 2> range(
    tf::IndexRange<int>(0, 0, 1),   // placeholder
    tf::IndexRange<int>(0, 0, 1)    // placeholder
  );

  auto init = taskflow.emplace([&]() {
    range.dim(0).reset(0, rows, 1);  // set row range at runtime
    range.dim(1).reset(0, cols, 1);  // set col range at runtime
  });

  auto loop = taskflow.for_each_by_index(std::ref(range), callable);
  init.precede(loop);
  @endcode
  */
  IndexRange<T, 1>& dim(size_t d) { return _dims[d]; }

  /**
  @brief returns the underlying array of all per-dimension 1D ranges

  @return a const reference to the @c std::array of N 1D ranges,
          one per dimension in order from outermost (0) to innermost (N-1)

  Useful when you need to inspect or iterate over all dimensions
  without knowing N at the call site, or when passing the full set of
  dimension ranges to another function.

  @code{.cpp}
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0,  4, 1),
    tf::IndexRange<int>(0, 10, 2),
    tf::IndexRange<int>(0,  6, 1)
  );

  // Print each dimension's bounds and step
  for(size_t d = 0; d < r.rank; ++d) {
    const auto& dim = r.dims()[d];
    printf("dim %zu: [%d, %d) step %d  size=%zu\n",
           d, dim.begin(), dim.end(), dim.step_size(), dim.size());
  }
  // dim 0: [0, 4)  step 1  size=4
  // dim 1: [0, 10) step 2  size=5
  // dim 2: [0, 6)  step 1  size=6
  @endcode
  */
  const std::array<IndexRange<T, 1>, N>& dims() const { return _dims; }

  // --------------------------------------------------------------------------
  // Size queries
  // --------------------------------------------------------------------------

  /**
  @brief returns the number of iterations along dimension @c d

  @param d zero-based dimension index in @c [0, N)

  @return the element count of dimension @c d, i.e.
          @c distance(dim(d).begin(), dim(d).end(), dim(d).step_size())

  This is a per-dimension count, independent of other dimensions and
  of the zero-size stopping rule that @c size() applies.

  @code{.cpp}
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0,  4, 1),   // 4 elements
    tf::IndexRange<int>(0, 10, 2),   // 5 elements (0,2,4,6,8)
    tf::IndexRange<int>(0,  6, 1)    // 6 elements
  );

  r.size(0);  // 4
  r.size(1);  // 5
  r.size(2);  // 6

  // Note: size(d) reports the raw dimension count regardless of
  // zero-size siblings — unlike size() which stops at the first zero.
  tf::IndexRange<int, 3> r2(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 0, 1),   // zero-size middle dim
    tf::IndexRange<int>(0, 6, 1)
  );
  r2.size(0);  // 4   (unaffected by the zero in dim 1)
  r2.size(1);  // 0
  r2.size(2);  // 6   (unaffected by the zero in dim 1)
  r2.size();   // 4   (stops at dim 1)
  @endcode
  */
  size_t size(size_t d) const { return _dims[d].size(); }

  /**
  @brief returns the number of active flat iterations

  Returns the product of dimension sizes from the outermost dimension inward,
  stopping before the first zero-size dimension.  This matches the behaviour
  of sequential nested loops: a zero-size dimension prevents its own iterations
  and those of all deeper dimensions, but outer dimensions are unaffected.

  @code{.cpp}
  // All non-zero: 4 * 6 * 8 = 192
  tf::IndexRange<int, 3> r1(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 8, 1)
  );
  r1.size();  // 192

  // Zero in middle (d=1): outer dim only -> 4
  tf::IndexRange<int, 3> r2(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 0, 1),   // stops accumulation here
    tf::IndexRange<int>(0, 8, 1)
  );
  r2.size();  // 4

  // Zero in innermost (d=2): outer two dims contribute -> 4*6 = 24
  tf::IndexRange<int, 3> r3(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 0, 1)    // stops accumulation here
  );
  r3.size();  // 24

  // Zero in outermost (d=0): no outer work -> 0
  tf::IndexRange<int, 3> r4(
    tf::IndexRange<int>(0, 0, 1),   // stops immediately
    tf::IndexRange<int>(0, 6, 1),
    tf::IndexRange<int>(0, 8, 1)
  );
  r4.size();  // 0

  // High-dimensional example: 19D with zero at d=9
  // -> product of dims [0..8] = 2^9 = 512
  tf::IndexRange<int, 19> r5(
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
    tf::IndexRange<int>(0,0,1),  // d=9: zero — dims 10..18 are inactive
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
    tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1)
  );
  r5.size();  // 512
  @endcode
  */
  size_t size() const;

  /**
  @brief returns the smallest perfectly-aligned hyperbox reachable from flat_beg,
         rounding up to the next hyperplane boundary when chunk_size is not aligned

  @param flat_beg  starting flat index (row-major) into the active ND space
  @param chunk_size  hint for the desired number of elements

  @return a pair of (sub-box, consumed) where consumed == sub-box.size()

  Analogous to @c std::ceil: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact; otherwise it rounds up to the next clean
  orthogonal boundary, so @c consumed >= @c chunk_size.

  Only the active dimensions — those before the first zero-size dimension — are
  partitioned.  Dimensions from the first zero onward are copied into the returned
  box as full extent and do not affect the flat index space.

  The trailing-zeros rule applies within the active dimensions: a dimension can
  only expand if all active dimensions inner to it start at coordinate 0.
  When this fires the function returns the best geometry-constrained box
  reachable from @c flat_beg.

  Used by dynamic partitioners: the atomic cursor advances by @c consumed and
  any overshoot is self-correcting.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active suffix products (boundaries): 1, 8, 32, 96
  // Dims d=3(0), d=4(2), d=5(3) are inactive — copied as full extent
  tf::IndexRange<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),   // zero — stops active space here
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );

  // chunk_size=8 aligns to one inner row — no overshoot
  // box: d0=[0,1), d1=[0,1), d2=[0,8), d3=[0,0), d4=[0,2), d5=[0,3)
  auto [box1, c1] = r.slice_ceil(0, 8);   // consumed=8

  // chunk_size=20 does not align; rounds up to next boundary (32)
  // box: d0=[0,1), d1=[0,4), d2=[0,8), d3=[0,0), d4=[0,2), d5=[0,3)
  auto [box2, c2] = r.slice_ceil(0, 20);  // consumed=32

  // geometry-constrained: flat=10 -> active coords (0,1,2), d2 not at 0
  // trailing-zeros fires; grow_dim=2, takes remaining 6 steps in that row
  auto [box3, c3] = r.slice_ceil(10, 20); // consumed=6 (<chunk_size)
  @endcode
  */
  std::pair<IndexRange<T, N>, size_t> slice_ceil(size_t flat_beg, size_t chunk_size) const;

  /**
  @brief returns the largest perfectly-aligned hyperbox reachable from flat_beg
         whose size does not exceed chunk_size, rounding down to the previous
         hyperplane boundary when chunk_size is not aligned

  @param flat_beg  starting flat index (row-major) into the active ND space
  @param chunk_size  hint for the desired number of elements

  @return a pair of (sub-box, consumed) where consumed == sub-box.size()

  Analogous to @c std::floor: if @c chunk_size already aligns to a hyperplane
  boundary the returned box is exact (identical to slice_ceil); otherwise
  it rounds down to the largest box that does not exceed @c chunk_size, so
  @c consumed <= @c chunk_size.

  Only the active dimensions — those before the first zero-size dimension — are
  partitioned.  Dimensions from the first zero onward are copied into the returned
  box as full extent.

  The trailing-zeros rule applies within the active dimensions identically to
  @c slice_ceil.

  Used by static partitioners: each worker's pre-assigned flat quota is never
  exceeded, so workers do not double-process elements at quota boundaries.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active suffix products (boundaries): 1, 8, 32, 96
  // Dims d=3(0), d=4(2), d=5(3) are inactive — copied as full extent
  tf::IndexRange<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),   // zero — stops active space here
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );

  // chunk_size=8 aligns exactly — same as slice_ceil
  auto [box1, c1] = r.slice_floor(0, 8);   // consumed=8

  // chunk_size=20 does not align; rounds down to largest boundary <= 20 (which is 8)
  auto [box2, c2] = r.slice_floor(0, 20);  // consumed=8

  // chunk_size=32 aligns exactly
  auto [box3, c3] = r.slice_floor(0, 32);  // consumed=32

  // chunk_size=50 does not align; rounds down to 32
  auto [box4, c4] = r.slice_floor(0, 50);  // consumed=32
  @endcode
  */
  std::pair<IndexRange<T, N>, size_t> slice_floor(size_t flat_beg, size_t chunk_size) const;

  /**
  @brief returns the smallest hyperplane-aligned chunk size that is >= chunk_size,
         capped at size() when chunk_size exceeds the total active range size

  @param chunk_size  hint for the desired number of flat elements

  @return the smallest natural per-step volume of the active dimensions that is
          >= @c chunk_size, or @c size() if @c chunk_size > @c size(), or 0 if
          the outermost dimension is zero-size

  Analogous to @c std::ceil but operating on the discrete set of hyperplane
  boundary sizes of the active dimensions (those before the first zero-size
  dimension).  Only active suffix products are considered — inactive inner
  dimensions do not contribute boundaries.

  When @c chunk_size is already a natural boundary size the return equals
  @c chunk_size exactly — just like @c std::ceil of an integer.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active boundaries: 1, 8, 32, 96  (inactive dims d=3,4,5 contribute nothing)
  tf::IndexRange<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  r.ceil(1);   //  1 — already a boundary
  r.ceil(5);   //  8 — rounds up to next active boundary
  r.ceil(8);   //  8 — already a boundary
  r.ceil(9);   // 32 — rounds up
  r.ceil(32);  // 32 — already a boundary
  r.ceil(33);  // 96 — rounds up to full active size
  r.ceil(96);  // 96 — already a boundary (== size())
  r.ceil(200); // 96 — capped at size()
  @endcode
  */
  size_t ceil(size_t chunk_size) const;

  /**
  @brief returns the largest hyperplane-aligned chunk size that is <= chunk_size

  @param chunk_size  hint for the desired number of flat elements

  @return the largest natural per-step volume of the active dimensions that is
          <= @c chunk_size, or 0 if the outermost dimension is zero-size

  Analogous to @c std::floor but operating on the discrete set of hyperplane
  boundary sizes of the active dimensions (those before the first zero-size
  dimension).  Only active suffix products are considered — inactive inner
  dimensions do not contribute boundaries.

  When @c chunk_size is already a natural boundary size the return equals
  @c chunk_size exactly — just like @c std::floor of an integer, and
  identical to @c ceil in that case.

  @code{.cpp}
  // 6D range: 3 x 4 x 8 x 0 x 2 x 3
  // Active dims: d=0(3), d=1(4), d=2(8) -> size()=96
  // Active boundaries: 1, 8, 32, 96  (inactive dims d=3,4,5 contribute nothing)
  tf::IndexRange<int, 6> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 8, 1),
    tf::IndexRange<int>(0, 0, 1),
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  r.floor(1);   //  1 — already a boundary
  r.floor(5);   //  1 — rounds down to previous active boundary
  r.floor(8);   //  8 — already a boundary
  r.floor(9);   //  8 — rounds down
  r.floor(32);  // 32 — already a boundary
  r.floor(33);  // 32 — rounds down
  r.floor(96);  // 96 — already a boundary (== size())
  r.floor(200); // 96 — capped at size()
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
  for (size_t d = 0; d < N; ++d) {
    size_t s = _dims[d].size();
    if (s == 0) return d == 0 ? 0 : total;  // outermost zero -> 0, inner zero -> outer product
    total *= s;
  }
  return total;
}

template <std::integral T, size_t N>
size_t IndexRange<T, N>::ceil(size_t chunk_size) const {
  // Pass 1: cache all dim sizes (one .size() call each) and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = _dims[d].size();
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
size_t IndexRange<T, N>::floor(size_t chunk_size) const {
  // Pass 1: cache all dim sizes (one .size() call each) and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = _dims[d].size();
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
std::pair<IndexRange<T, N>, size_t>
IndexRange<T, N>::slice_ceil(size_t flat_beg, size_t chunk_size) const {

  if (chunk_size == 0) {
    return { *this, 0 };
  }

  // Pass 1 (forward): cache dim sizes and find d_active — the first zero-size
  // dimension.  Only [0, d_active) participates in the flat iteration space;
  // dims [d_active, N) are copied as full extent into the returned box.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = _dims[d].size();
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }

  if (d_active == 0) return { *this, 0 };

  // Pass 2a (backward): decode flat_beg into per-dimension coords.
  // Must complete fully — all coords needed for box construction.
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
    // trailing-zeros rule: inner coord non-zero → can't expand further out
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

  // Pass 3: construct the sub-box in three clean segments with no branching
  // inside each loop:
  //   [0, grow_dim)      — locked dims (one element each)
  //   [grow_dim]         — the grow dimension
  //   [grow_dim+1, N)    — full inner/inactive extent
  std::array<IndexRange<T, 1>, N> box_dims;

  for (size_t d = 0; d < grow_dim; ++d) {
    const T beg  = _dims[d].begin();
    const T step = _dims[d].step_size();
    box_dims[d] = IndexRange<T, 1>(
      beg + static_cast<T>(coords[d]) * step,
      beg + static_cast<T>(coords[d] + 1) * step,
      step
    );
  }

  {
    const T beg  = _dims[grow_dim].begin();
    const T step = _dims[grow_dim].step_size();
    box_dims[grow_dim] = IndexRange<T, 1>(
      beg + static_cast<T>(coords[grow_dim])                 * step,
      beg + static_cast<T>(coords[grow_dim] + steps_to_take) * step,
      step
    );
  }

  for (size_t d = grow_dim + 1; d < N; ++d) {
    box_dims[d] = _dims[d];  // full inner or inactive extent
  }

  return { IndexRange<T, N>(box_dims), steps_to_take * active_inner_vol };
}

template <std::integral T, size_t N>
std::pair<IndexRange<T, N>, size_t>
IndexRange<T, N>::slice_floor(size_t flat_beg, size_t chunk_size) const {

  if (chunk_size == 0) {
    return { *this, 0 };
  }

  // Pass 1 (forward): cache dim sizes and find d_active.
  size_t dim_sizes[N];
  size_t d_active = N;
  for (size_t d = 0; d < N; ++d) {
    dim_sizes[d] = _dims[d].size();
    if (dim_sizes[d] == 0) { d_active = d; break; }
  }

  if (d_active == 0) return { *this, 0 };

  // Pass 2a (backward): decode flat_beg into per-dimension coords.
  // Must complete fully — all coords needed for box construction.
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
  std::array<IndexRange<T, 1>, N> box_dims;

  for (size_t d = 0; d < grow_dim; ++d) {
    const T beg  = _dims[d].begin();
    const T step = _dims[d].step_size();
    box_dims[d] = IndexRange<T, 1>(
      beg + static_cast<T>(coords[d]) * step,
      beg + static_cast<T>(coords[d] + 1) * step,
      step
    );
  }

  {
    const T beg  = _dims[grow_dim].begin();
    const T step = _dims[grow_dim].step_size();
    box_dims[grow_dim] = IndexRange<T, 1>(
      beg + static_cast<T>(coords[grow_dim])                 * step,
      beg + static_cast<T>(coords[grow_dim] + steps_to_take) * step,
      step
    );
  }

  for (size_t d = grow_dim + 1; d < N; ++d) {
    box_dims[d] = _dims[d];  // full inner or inactive extent
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
