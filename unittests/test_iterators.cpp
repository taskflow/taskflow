#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/utility/iterator.hpp>

// ============================================================================
// Helpers
// ============================================================================

// Enumerate every element produced by an IndexRange<T> via explicit loop.
template <typename T>
std::vector<T> enumerate_1d(const tf::IndexRange<T>& r) {
  std::vector<T> out;
  if (r.step_size() > T{0}) {
    for (T v = r.begin(); v < r.end(); v += r.step_size()) out.push_back(v);
  } else {
    for (T v = r.begin(); v > r.end(); v += r.step_size()) out.push_back(v);
  }
  return out;
}

// Enumerate every flat index produced by an IndexRanges<T, 2> in row-major order.
// Works for both positive and negative step sizes.
template <typename T>
std::vector<std::pair<T,T>> enumerate_2d(const tf::IndexRanges<T, 2>& r) {
  std::vector<std::pair<T,T>> out;
  auto [b0, e0, s0] = r.dim(0);
  auto [b1, e1, s1] = r.dim(1);
  auto in_range = [](T v, T end, T step) {
    return step > T{0} ? v < end : v > end;
  };
  for (T i = b0; in_range(i, e0, s0); i += s0)
    for (T j = b1; in_range(j, e1, s1); j += s1)
      out.push_back({i, j});
  return out;
}

// Enumerate every flat index produced by an IndexRanges<T, 3> in row-major order.
// Works for both positive and negative step sizes.
template <typename T>
std::vector<std::tuple<T,T,T>> enumerate_3d(const tf::IndexRanges<T, 3>& r) {
  std::vector<std::tuple<T,T,T>> out;
  auto [b0, e0, s0] = r.dim(0);
  auto [b1, e1, s1] = r.dim(1);
  auto [b2, e2, s2] = r.dim(2);
  auto in_range = [](T v, T end, T step) {
    return step > T{0} ? v < end : v > end;
  };
  for (T i = b0; in_range(i, e0, s0); i += s0)
    for (T j = b1; in_range(j, e1, s1); j += s1)
      for (T k = b2; in_range(k, e2, s2); k += s2)
        out.push_back({i, j, k});
  return out;
}

// ============================================================================
// Section 1: is_index_range_invalid
// ============================================================================

TEST_CASE("is_index_range_invalid.positive_range") {
  REQUIRE_FALSE(tf::is_index_range_invalid(0, 10, 1));
  REQUIRE_FALSE(tf::is_index_range_invalid(0, 10, 2));
  REQUIRE_FALSE(tf::is_index_range_invalid(-5, 5, 1));
  // zero step with beg != end
  REQUIRE(tf::is_index_range_invalid(0, 10, 0));
  // positive range with non-positive step
  REQUIRE(tf::is_index_range_invalid(0, 10, -1));
  REQUIRE(tf::is_index_range_invalid(0, 10, 0));
}

TEST_CASE("is_index_range_invalid.negative_range") {
  REQUIRE_FALSE(tf::is_index_range_invalid(10, 0, -1));
  REQUIRE_FALSE(tf::is_index_range_invalid(10, 0, -2));
  // negative range with non-negative step
  REQUIRE(tf::is_index_range_invalid(10, 0, 1));
  REQUIRE(tf::is_index_range_invalid(10, 0, 0));
}

TEST_CASE("is_index_range_invalid.empty_range") {
  // beg == end is always valid regardless of step
  REQUIRE_FALSE(tf::is_index_range_invalid(5, 5, 0));
  REQUIRE_FALSE(tf::is_index_range_invalid(5, 5, 1));
  REQUIRE_FALSE(tf::is_index_range_invalid(5, 5, -1));
}

TEST_CASE("is_index_range_invalid.unsigned") {
  REQUIRE_FALSE(tf::is_index_range_invalid(size_t{0}, size_t{10}, size_t{1}));
  REQUIRE_FALSE(tf::is_index_range_invalid(size_t{0}, size_t{10}, size_t{3}));
  REQUIRE(tf::is_index_range_invalid(size_t{0}, size_t{10}, size_t{0}));
}

// ============================================================================
// Section 2: distance
// ============================================================================

TEST_CASE("distance.positive_step") {
  REQUIRE(tf::distance(0, 10, 1) == 10);
  REQUIRE(tf::distance(0, 10, 2) == 5);
  REQUIRE(tf::distance(0, 10, 3) == 4);   // ceil(10/3)
  REQUIRE(tf::distance(0,  9, 3) == 3);
  REQUIRE(tf::distance(5, 20, 5) == 3);
  REQUIRE(tf::distance(0,  1, 1) == 1);
  REQUIRE(tf::distance(0,  0, 1) == 0);   // empty
}

TEST_CASE("distance.negative_step") {
  REQUIRE(tf::distance(10, 0, -1) == 10);
  REQUIRE(tf::distance(10, 0, -2) == 5);
  REQUIRE(tf::distance(10, 0, -3) == 4);
  REQUIRE(tf::distance(10, 1, -3) == 3);
}

TEST_CASE("distance.unsigned") {
  REQUIRE(tf::distance(size_t{0}, size_t{10}, size_t{1}) == 10);
  REQUIRE(tf::distance(size_t{0}, size_t{10}, size_t{2}) == 5);
  REQUIRE(tf::distance(size_t{0}, size_t{10}, size_t{3}) == 4);
}

TEST_CASE("distance.matches_size") {
  // distance() and IndexRange<T>::size() must agree for all valid ranges
  for (int beg = -5; beg <= 5; beg++) {
    for (int end = beg + 1; end <= 10; end++) {
      for (int s = 1; s <= 4; s++) {
        tf::IndexRange<int> r(beg, end, s);
        REQUIRE(tf::distance(beg, end, s) == r.size());
      }
    }
  }
}

// ============================================================================
// Section 3: IndexRange<T> — construction & accessors
// ============================================================================

TEST_CASE("IndexRange1D.basic_construction") {
  tf::IndexRange<int> r(0, 10, 2);
  REQUIRE(r.begin()     == 0);
  REQUIRE(r.end()       == 10);
  REQUIRE(r.step_size() == 2);
  REQUIRE(r.size()      == 5);
}

TEST_CASE("IndexRange1D.negative_step") {
  tf::IndexRange<int> r(10, 0, -2);
  REQUIRE(r.begin()     == 10);
  REQUIRE(r.end()       == 0);
  REQUIRE(r.step_size() == -2);
  REQUIRE(r.size()      == 5);
}

TEST_CASE("IndexRange1D.unit_step") {
  tf::IndexRange<int> r(3, 8, 1);
  REQUIRE(r.size() == 5);
  auto elems = enumerate_1d(r);
  REQUIRE(elems == std::vector<int>{3, 4, 5, 6, 7});
}

TEST_CASE("IndexRange1D.CTAD") {
  tf::IndexRange r(0, 10, 2);  // deduction guide -> IndexRange<int>
  static_assert(std::is_same_v<decltype(r), tf::IndexRange<int>>);
  REQUIRE(r.size() == 5);
}

TEST_CASE("IndexRange1D.reset") {
  tf::IndexRange<int> r(0, 10, 1);
  r.reset(5, 20, 3);
  REQUIRE(r.begin()     == 5);
  REQUIRE(r.end()       == 20);
  REQUIRE(r.step_size() == 3);
  REQUIRE(r.size()      == 5);
}

TEST_CASE("IndexRange1D.fluent_setters") {
  tf::IndexRange<int> r(0, 10, 1);
  r.begin(2).end(12).step_size(2);
  REQUIRE(r.begin()     == 2);
  REQUIRE(r.end()       == 12);
  REQUIRE(r.step_size() == 2);
  REQUIRE(r.size()      == 5);
}

TEST_CASE("IndexRange1D.rank") {
  static_assert(tf::IndexRange<int>::rank == 1);
  static_assert(tf::IndexRange<int>::rank    == 1);
}

// ============================================================================
// Section 4: IndexRange<T>::unravel
// ============================================================================

TEST_CASE("IndexRange1D.unravel.basic") {
  tf::IndexRange<int> r(0, 10, 2);   // elements: 0,2,4,6,8
  auto sub = r.unravel(1, 4);        // positions [1,4) -> elements 2,4,6
  REQUIRE(sub.begin()     == 2);
  REQUIRE(sub.end()       == 8);
  REQUIRE(sub.step_size() == 2);
  REQUIRE(sub.size()      == 3);
}

TEST_CASE("IndexRange1D.unravel.full_range") {
  tf::IndexRange<int> r(0, 10, 1);
  auto sub = r.unravel(0, 10);
  REQUIRE(sub.begin()     == r.begin());
  REQUIRE(sub.end()       == r.end());
  REQUIRE(sub.step_size() == r.step_size());
}

TEST_CASE("IndexRange1D.unravel.single_element") {
  tf::IndexRange<int> r(0, 10, 2);
  auto sub = r.unravel(3, 4);   // element at position 3 -> value 6
  REQUIRE(sub.begin()     == 6);
  REQUIRE(sub.size()      == 1);
  REQUIRE(sub.step_size() == 2);
}

TEST_CASE("IndexRange1D.unravel.negative_step") {
  tf::IndexRange<int> r(10, 0, -2);   // elements: 10,8,6,4,2
  auto sub = r.unravel(1, 4);         // positions [1,4) -> 8,6,4
  REQUIRE(sub.begin()     == 8);
  REQUIRE(sub.end()       == 2);
  REQUIRE(sub.step_size() == -2);
  REQUIRE(sub.size()      == 3);
}

TEST_CASE("IndexRange1D.unravel.covers_partition") {
  // All partitions from a sweep must together cover the full range exactly once
  tf::IndexRange<int> r(0, 30, 3);
  size_t N = r.size();  // 10
  std::vector<int> visited(N, 0);

  size_t chunk = 3;
  for (size_t b = 0; b < N; b += chunk) {
    size_t e = std::min(b + chunk, N);
    auto sub = r.unravel(b, e);
    auto elems = enumerate_1d(sub);
    for (int v : elems) {
      size_t pos = static_cast<size_t>((v - r.begin()) / r.step_size());
      visited[pos]++;
    }
  }
  for (int v : visited) REQUIRE(v == 1);
}

// ============================================================================
// Section 5: IndexRanges<T,N> — construction, size, rank
// ============================================================================

TEST_CASE("IndexRangeND.construction_2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  REQUIRE(r.rank     == 2);
  REQUIRE(r.size(0)  == 4);
  REQUIRE(r.size(1)  == 6);
  REQUIRE(r.size()   == 24);
}

TEST_CASE("IndexRangeND.construction_3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  REQUIRE(r.size() == 120);
}

TEST_CASE("IndexRangeND.construction_from_array") {
  std::array<std::tuple<int,int,int>, 2> dims = {
    std::tuple<int,int,int>{0, 3, 1},
    std::tuple<int,int,int>{0, 4, 1}
  };
  tf::IndexRanges<int, 2> r(dims);
  REQUIRE(r.size() == 12);
}

TEST_CASE("IndexRangeND.dim_accessor") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(2, 8, 2),
    tf::IndexRange<int>(1, 7, 3)
  );
  REQUIRE(std::get<0>(r.dim(0))     == 2);
  REQUIRE(std::get<2>(r.dim(0)) == 2);
  REQUIRE(r.size(0)      == 3);
  REQUIRE(std::get<0>(r.dim(1))     == 1);
  REQUIRE(std::get<2>(r.dim(1)) == 3);
  REQUIRE(r.size(1)      == 2);
  REQUIRE(r.size()             == 6);
}

TEST_CASE("IndexRangeND.non_unit_steps_size") {
  // 3D: steps 1, 2, 3
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(0, 10, 2),
    tf::IndexRange<int>(0,  9, 3)
  );
  REQUIRE(r.size(0) == 4);
  REQUIRE(r.size(1) == 5);
  REQUIRE(r.size(2) == 3);
  REQUIRE(r.size()  == 60);
}

TEST_CASE("IndexRangeND.rank") {
  static_assert(tf::IndexRanges<int, 2>::rank == 2);
  static_assert(tf::IndexRanges<int, 3>::rank == 3);
  static_assert(tf::IndexRanges<int, 4>::rank == 4);
}

// ============================================================================
// Section 17: zero-size dimension behaviour
//
// A zero-size dimension at position d stops the active flat space there.
// size() returns the product of dims [0, d), i.e. the outer dims only.
// This matches sequential nested loop behaviour — outer loops still execute.
//
// Tests cover 2D through 19D with zero at every dimension position.
// ============================================================================
TEST_CASE("IndexRangeND.zero_size.size.low_dimensions") {
  // 2D
  {
    // zero in outer
    tf::IndexRanges<int, 2> r(tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,7,1));
    REQUIRE(r.size() == 0);
  }
  {
    // zero in inner
    tf::IndexRanges<int, 2> r(tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,0,1));
    REQUIRE(r.size() == 5);
  }

  // 3D — zero at each position
  {
    tf::IndexRanges<int, 3> r(
      tf::IndexRange<int>(0, 0,1), tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,6,1));
    REQUIRE(r.size() == 0);
  }
  {
    tf::IndexRanges<int, 3> r(
      tf::IndexRange<int>(0, 3,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,6,1));
    REQUIRE(r.size() == 3);
  }
  {
    tf::IndexRanges<int, 3> r(
      tf::IndexRange<int>(0, 3,1), tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,0,1));
    REQUIRE(r.size() == 12);
  }

  // 5D — zero at each position
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,5,1));
    REQUIRE(r.size() == 0);
  }
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,3,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,5,1));
    REQUIRE(r.size() == 2);
  }
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,5,1));
    REQUIRE(r.size() == 6);
  }
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,5,1));
    REQUIRE(r.size() == 24);
  }
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,0,1));
    REQUIRE(r.size() == 120);
  }

  // 7D — zero in first, middle (d=3), and last position
  {
    tf::IndexRanges<int, 7> r(
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1));
    REQUIRE(r.size() == 0);
  }
  {
    tf::IndexRanges<int, 7> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1));
    REQUIRE(r.size() == 24);  // 2*3*4
  }
  {
    tf::IndexRanges<int, 7> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1), tf::IndexRange<int>(0,7,1),
      tf::IndexRange<int>(0,0,1));
    REQUIRE(r.size() == 2*3*4*5*6*7);
  }

  // 9D — zero at d=4
  {
    tf::IndexRanges<int, 9> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 16);  // 2^4
  }
}

TEST_CASE("IndexRangeND.zero_size.size.high_dimensions") {
  // 11D — zero at d=5
  {
    tf::IndexRanges<int, 11> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 32);  // 2^5
  }

  // 13D — zero at d=0 and d=12
  {
    tf::IndexRanges<int, 13> r(
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 0);  // outermost zero
  }
  {
    tf::IndexRanges<int, 13> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,0,1));
    REQUIRE(r.size() == 4096);  // 2^12, innermost is zero
  }

  // 15D — zero at d=7 (middle)
  {
    tf::IndexRanges<int, 15> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 128);  // 2^7
  }

  // 17D — zero at d=1
  {
    tf::IndexRanges<int, 17> r(
      tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 3);  // only d=0 contributes
  }

  // 19D — zero at d=9
  {
    tf::IndexRanges<int, 19> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 512);  // 2^9
  }
}

// ============================================================================
// Section 18: negative step sizes
// ============================================================================

// Helper: enumerate a 1D range with negative step into a vector of values
// already defined above via enumerate_1d.

TEST_CASE("NegativeStep.IndexRange1D.size_and_elements") {
  // beg=10, end=0, step=-2 -> 10,8,6,4,2 (5 elements)
  tf::IndexRange<int> r(10, 0, -2);
  REQUIRE(r.size() == 5);
  auto elems = enumerate_1d(r);
  REQUIRE(elems == std::vector<int>{10, 8, 6, 4, 2});
}

TEST_CASE("NegativeStep.IndexRange1D.unravel") {
  tf::IndexRange<int> r(10, 0, -2);   // elements: 10,8,6,4,2
  // positions [1,4) -> elements 8,6,4
  auto sub = r.unravel(1, 4);
  REQUIRE(sub.begin()     == 8);
  REQUIRE(sub.end()       == 2);
  REQUIRE(sub.step_size() == -2);
  REQUIRE(sub.size()      == 3);
  auto elems = enumerate_1d(sub);
  REQUIRE(elems == std::vector<int>{8, 6, 4});
}

TEST_CASE("NegativeStep.IndexRange1D.unravel.full_coverage") {
  // Same sweep as the positive-step unravel coverage test
  for (int beg = 10; beg >= 2; beg -= 2) {
    for (int s = 1; s <= 3; s++) {
      tf::IndexRange<int> r(beg, 0, -s);
      size_t N = r.size();
      std::vector<int> visited(N, 0);
      size_t chunk = 3;
      for (size_t b = 0; b < N; b += chunk) {
        size_t e = std::min(b + chunk, N);
        auto sub = r.unravel(b, e);
        for (auto v : enumerate_1d(sub)) {
          size_t pos = static_cast<size_t>((r.begin() - v) / s);
          visited[pos]++;
        }
      }
      for (int v : visited) REQUIRE(v == 1);
    }
  }
}

// 2D with one negative-step dimension
TEST_CASE("NegativeStep.2D.dim0_negative.size") {
  // dim0: 10 down to 0 step -2 -> 5 elements: 10,8,6,4,2
  // dim1: 0 up to 6 step 1    -> 6 elements
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(0,  6,  1)
  );
  REQUIRE(r.size(0) == 5);
  REQUIRE(r.size(1) == 6);
  REQUIRE(r.size()  == 30);
}

TEST_CASE("NegativeStep.2D.dim1_negative.size") {
  // dim0: 0 to 4 step 1
  // dim1: 9 to 0 step -3 -> 3 elements: 9,6,3
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(9, 0, -3)
  );
  REQUIRE(r.size(0) == 4);
  REQUIRE(r.size(1) == 3);
  REQUIRE(r.size()  == 12);
}

TEST_CASE("NegativeStep.2D.both_negative.size") {
  // dim0: 6 to 0 step -2 -> 3 elements: 6,4,2
  // dim1: 9 to 0 step -3 -> 3 elements: 9,6,3
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  REQUIRE(r.size() == 9);
}

// 3D with negative steps
TEST_CASE("NegativeStep.3D.all_negative.size") {
  // dim0: 4 to 0 step -1 -> 4 elements
  // dim1: 10 to 0 step -2 -> 5 elements
  // dim2: 9 to 0 step -3 -> 3 elements
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  REQUIRE(r.size(0) == 4);
  REQUIRE(r.size(1) == 5);
  REQUIRE(r.size(2) == 3);
  REQUIRE(r.size()  == 60);
}
// ============================================================================
// Section 19: concepts
// ============================================================================

TEST_CASE("Concepts.IndexRangesLike") {  // Section 19
  static_assert( tf::IndexRangesLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRangesLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRangesLike<tf::IndexRanges<int, 2>>);
  static_assert( tf::IndexRangesLike<tf::IndexRanges<int, 3>>);
  static_assert(!tf::IndexRangesLike<int>);
  static_assert(!tf::IndexRangesLike<std::vector<int>>);
  // cv-ref variants
  static_assert( tf::IndexRangesLike<const tf::IndexRange<int>&>);
  static_assert( tf::IndexRangesLike<tf::IndexRanges<int, 2>&&>);
  REQUIRE(true);  // static_asserts above are the real test
}

// ============================================================================
// Section 18a: IndexRange<T> — negative step basic properties
// ============================================================================

TEST_CASE("NegativeStep.IndexRange1D.size_and_elements") {
  // for(int i=10; i>0; i-=2)  -> 10,8,6,4,2  (5 elements)
  tf::IndexRange<int> r(10, 0, -2);
  REQUIRE(r.size() == 5);
  auto elems = enumerate_1d(r);
  REQUIRE(elems == std::vector<int>({10, 8, 6, 4, 2}));
}

TEST_CASE("NegativeStep.IndexRange1D.size_not_divisible") {
  // for(int i=10; i>0; i-=3) -> 10,7,4,1  (4 elements)
  tf::IndexRange<int> r(10, 0, -3);
  REQUIRE(r.size() == 4);
  auto elems = enumerate_1d(r);
  REQUIRE(elems == std::vector<int>({10, 7, 4, 1}));
}

TEST_CASE("NegativeStep.IndexRange1D.unravel") {
  // beg=10, end=0, step=-2 -> elements 10,8,6,4,2
  // unravel positions [1,4) -> elements at pos 1,2,3 -> 8,6,4
  tf::IndexRange<int> r(10, 0, -2);
  auto sub = r.unravel(1, 4);
  REQUIRE(sub.begin()     == 8);
  REQUIRE(sub.end()       == 2);
  REQUIRE(sub.step_size() == -2);
  REQUIRE(sub.size()      == 3);
  auto elems = enumerate_1d(sub);
  REQUIRE(elems == std::vector<int>({8, 6, 4}));
}

TEST_CASE("NegativeStep.IndexRange1D.unravel_coverage") {
  // Sweep partitions and confirm exact coverage
  tf::IndexRange<int> r(20, 0, -2);  // 20,18,...,2  (10 elements)
  size_t N = r.size();
  std::vector<int> visited(N, 0);
  size_t chunk = 3;
  for (size_t b = 0; b < N; b += chunk) {
    size_t e = std::min(b + chunk, N);
    auto sub = r.unravel(b, e);
    for (int v = sub.begin(); v > sub.end(); v += sub.step_size()) {
      size_t pos = static_cast<size_t>((r.begin() - v) / (-r.step_size()));
      visited[pos]++;
    }
  }
  for (int v : visited) REQUIRE(v == 1);
}

// ----------------------------------------------------------------------------
// InputIteratorLike concept test
// ----------------------------------------------------------------------------

TEST_CASE("Concept.InputIteratorLike" * doctest::timeout(300)) {

  using Iter  = std::vector<int>::iterator;
  using CIter = std::vector<int>::const_iterator;

  // Plain input iterators.
  static_assert(tf::InputIteratorLike<Iter>);
  static_assert(tf::InputIteratorLike<CIter>);
  static_assert(tf::InputIteratorLike<int*>);
  static_assert(tf::InputIteratorLike<const int*>);

  // std::reference_wrapper of input iterators.
  static_assert(tf::InputIteratorLike<std::reference_wrapper<Iter>>);
  static_assert(tf::InputIteratorLike<std::reference_wrapper<CIter>>);
  static_assert(tf::InputIteratorLike<std::reference_wrapper<int*>>);

  // Non-input-iterators.
  static_assert(!tf::InputIteratorLike<int>);
  static_assert(!tf::InputIteratorLike<double>);
  static_assert(!tf::InputIteratorLike<std::vector<int>>);
  static_assert(!tf::InputIteratorLike<std::reference_wrapper<int>>);
  static_assert(!tf::InputIteratorLike<std::nullptr_t>);
}






