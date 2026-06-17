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

// Drive upper_slice over the full flat space of an ND range and collect every
// element visited.  Returns the multiset of flat indices (encoded as size_t)
// so the caller can verify exactly-once coverage.
// Works for any N by having the caller supply a "visit" lambda that converts
// box -> flat indices and appends them.
template <typename R, typename VisitFn>
std::vector<size_t> drain_upper_slice(const R& range, size_t chunk_size, VisitFn visit) {
  size_t N = range.size();
  std::vector<size_t> visited;
  size_t cursor = 0;
  while (cursor < N) {
    auto box = range.upper_slice(cursor, chunk_size); size_t consumed = box.size();
    REQUIRE(consumed > 0);          // must make forward progress
    REQUIRE(consumed <= chunk_size + 1); // never wildly over-consume
    REQUIRE(cursor + consumed <= N);
    visit(box, visited);
    cursor += consumed;
  }
  return visited;
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
// Section 6: upper_slice — documented examples from the header
// ============================================================================

TEST_CASE("upper_slice.documented_examples") {
  // 3D range: 4 x 5 x 10 (from the header docstring)
  tf::IndexRanges<int, 3> range(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(0, 5,  1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // Scenario 1: flat_beg=0, requested=30
  // Coords (0,0,0). inner_volume reaches 50 at dim-0 (first >= 30), so grow_dim=0,
  // steps_to_take=1. The box overshoots to the next orthogonal boundary.
  // box is [0,1) x [0,5) x [0,10), consumed=50
  {
    auto box = range.upper_slice(0, 30); size_t consumed = box.size();
    REQUIRE(consumed == 50);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 5);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
    REQUIRE(box.size() == 50);
  }

  // Scenario 2: flat_beg=30, requested=30
  // Coords (0,3,0). coords[1]=3 != 0, so trailing-zeros fires at d=0.
  // grow_dim=1, active=10. steps_left=2, steps_needed=3, steps_to_take=2.
  // box is [0,1) x [3,5) x [0,10), consumed=20 (geometry-constrained, < requested)
  {
    auto box = range.upper_slice(30, 30); size_t consumed = box.size();
    REQUIRE(consumed == 20);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 3); REQUIRE(std::get<1>(box.dim(1)) == 5);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
    REQUIRE(box.size() == 20);
  }

  // Scenario 3: flat_beg=55, requested=30
  // Coords (1,0,5). coords[2]=5 != 0, trailing-zeros fires at d=1.
  // grow_dim=2, active=1. steps_left=5, steps_needed=30, steps_to_take=5.
  // box is [1,2) x [0,1) x [5,10), consumed=5 (geometry-constrained, < requested)
  {
    auto box = range.upper_slice(55, 30); size_t consumed = box.size();
    REQUIRE(consumed == 5);
    REQUIRE(std::get<0>(box.dim(0)) == 1); REQUIRE(std::get<1>(box.dim(0)) == 2);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 1);
    REQUIRE(std::get<0>(box.dim(2)) == 5); REQUIRE(std::get<1>(box.dim(2)) == 10);
    REQUIRE(box.size() == 5);
  }

  // Scenario 4: exact fit — requested equals a natural dimension boundary
  // flat_beg=0, requested=10. inner_volume reaches 10 at d=1 (10>=10), stops there.
  // grow_dim=1, active=10. steps_left=5, steps_needed=1, steps_to_take=1.
  // box is [0,1) x [0,1) x [0,10), consumed=10 (no overshoot needed)
  {
    auto box = range.upper_slice(0, 10); size_t consumed = box.size();
    REQUIRE(consumed == 10);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 1);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
    REQUIRE(box.size() == 10);
  }
}

// ============================================================================
// Section 8: upper_slice — zero chunk_size
// ============================================================================

TEST_CASE("upper_slice.zero_chunk_size") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  auto box = r.upper_slice(0, 0); size_t consumed = box.size();
  REQUIRE(consumed == 0);
}

// ============================================================================
// Section 9: upper_slice — forward progress guarantee
// For any chunk size >= 1 and any flat_beg < N, consumed must be >= 1.
// ============================================================================

TEST_CASE("upper_slice.forward_progress.2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 7, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 7, 13, 50}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
    }
  }
}

TEST_CASE("upper_slice.forward_progress.3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 5, 20, 60}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
    }
  }
}

// ============================================================================
// Section 10: upper_slice — box is an orthogonal sub-box of the original
// The sub-box must not exceed any dimension's bounds.
// ============================================================================

TEST_CASE("upper_slice.box_within_bounds.2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 6, 2),
    tf::IndexRange<int>(0, 9, 3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 6}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      for (size_t d = 0; d < 2; d++) {
        REQUIRE(std::get<0>(box.dim(d))     >= std::get<0>(r.dim(d)));
        REQUIRE(std::get<1>(box.dim(d))       <= std::get<1>(r.dim(d)));
        REQUIRE(std::get<2>(box.dim(d)) == std::get<2>(r.dim(d)));
      }
      REQUIRE(box.size() == consumed);
    }
  }
}

TEST_CASE("upper_slice.box_within_bounds.3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 5, 10, 30, 120}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      for (size_t d = 0; d < 3; d++) {
        REQUIRE(std::get<0>(box.dim(d))     >= std::get<0>(r.dim(d)));
        REQUIRE(std::get<1>(box.dim(d))       <= std::get<1>(r.dim(d)));
        REQUIRE(std::get<2>(box.dim(d)) == std::get<2>(r.dim(d)));
      }
      REQUIRE(box.size() == consumed);
    }
  }
}

// ============================================================================
// Section 11: upper_slice — full coverage (the critical invariant)
// Draining with upper_slice must visit every element exactly once.
// ============================================================================

// Helper: collect flat indices from a 2D box
auto collect_flat_2d(const tf::IndexRanges<int, 2>& box,
                     const tf::IndexRanges<int, 2>& full,
                     std::vector<size_t>& out) {
  size_t D1 = full.size(1);
  int S0 = std::get<2>(full.dim(0)), S1 = std::get<2>(full.dim(1));
  int B0 = std::get<0>(full.dim(0)),     B1 = std::get<0>(full.dim(1));
  auto pos = [](int v, int beg, int step) -> size_t {
    return static_cast<size_t>((v - beg) / step);
  };
  auto in_range = [](int v, int end, int step) {
    return step > 0 ? v < end : v > end;
  };
  for (int i = std::get<0>(box.dim(0)); in_range(i, std::get<1>(box.dim(0)), S0); i += S0)
    for (int j = std::get<0>(box.dim(1)); in_range(j, std::get<1>(box.dim(1)), S1); j += S1)
      out.push_back(pos(i, B0, S0) * D1 + pos(j, B1, S1));
}

auto collect_flat_3d(const tf::IndexRanges<int, 3>& box,
                     const tf::IndexRanges<int, 3>& full,
                     std::vector<size_t>& out) {
  size_t D1 = full.size(1), D2 = full.size(2);
  int S0 = std::get<2>(full.dim(0)), S1 = std::get<2>(full.dim(1)), S2 = std::get<2>(full.dim(2));
  int B0 = std::get<0>(full.dim(0)),     B1 = std::get<0>(full.dim(1)),     B2 = std::get<0>(full.dim(2));
  auto pos = [](int v, int beg, int step) -> size_t {
    return static_cast<size_t>((v - beg) / step);
  };
  auto in_range = [](int v, int end, int step) {
    return step > 0 ? v < end : v > end;
  };
  for (int i = std::get<0>(box.dim(0)); in_range(i, std::get<1>(box.dim(0)), S0); i += S0)
    for (int j = std::get<0>(box.dim(1)); in_range(j, std::get<1>(box.dim(1)), S1); j += S1)
      for (int k = std::get<0>(box.dim(2)); in_range(k, std::get<1>(box.dim(2)), S2); k += S2)
        out.push_back(pos(i, B0, S0) * (D1 * D2) + pos(j, B1, S1) * D2 + pos(k, B2, S2));
}

TEST_CASE("upper_slice.full_coverage.2d") {
  // Sweep multiple chunk sizes and range shapes
  for (int di : {1, 2, 3, 5, 7, 12}) {
    for (int dj : {1, 2, 3, 5, 7, 12}) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, di, 1),
        tf::IndexRange<int>(0, dj, 1)
      );
      size_t N = r.size();
      for (size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, size_t{13}, N/2 + 1, N}) {
        if (cs == 0) cs = 1;
        std::vector<size_t> visited;
        size_t cursor = 0;
        while (cursor < N) {
          auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
          REQUIRE(consumed >= 1);
          collect_flat_2d(box, r, visited);
          cursor += consumed;
        }
        REQUIRE(visited.size() == N);
        std::sort(visited.begin(), visited.end());
        for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
      }
    }
  }
}

TEST_CASE("upper_slice.full_coverage.3d") {
  for (int di : {1, 3, 5}) {
    for (int dj : {1, 4, 6}) {
      for (int dk : {1, 2, 7}) {
        tf::IndexRanges<int, 3> r(
          tf::IndexRange<int>(0, di, 1),
          tf::IndexRange<int>(0, dj, 1),
          tf::IndexRange<int>(0, dk, 1)
        );
        size_t N = r.size();
        for (size_t cs : {size_t{1}, size_t{3}, size_t{7}, N/2 + 1, N}) {
          if (cs == 0) cs = 1;
          std::vector<size_t> visited;
          size_t cursor = 0;
          while (cursor < N) {
            auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
            REQUIRE(consumed >= 1);
            collect_flat_3d(box, r, visited);
            cursor += consumed;
          }
          REQUIRE(visited.size() == N);
          std::sort(visited.begin(), visited.end());
          for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
        }
      }
    }
  }
}

TEST_CASE("upper_slice.full_coverage.non_unit_steps_2d") {
  // step sizes > 1 — the flat index is the logical position, not the raw value
  for (int si : {1, 2, 3}) {
    for (int sj : {1, 2, 3}) {
      for (int di : {1, 3, 5}) {
        for (int dj : {1, 3, 5}) {
          tf::IndexRanges<int, 2> r(
            tf::IndexRange<int>(0, di * si, si),
            tf::IndexRange<int>(0, dj * sj, sj)
          );
          size_t N = r.size();
          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
              REQUIRE(consumed >= 1);
              collect_flat_2d(box, r, visited);
              cursor += consumed;
            }
            REQUIRE(visited.size() == N);
            std::sort(visited.begin(), visited.end());
            for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
          }
        }
      }
    }
  }
}

TEST_CASE("upper_slice.full_coverage.non_unit_steps_3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4*1, 1),
    tf::IndexRange<int>(0, 5*2, 2),
    tf::IndexRange<int>(0, 6*3, 3)
  );
  size_t N = r.size();   // 4*5*6 = 120
  for (size_t cs : {1, 4, 7, 13, 30, 60, 120}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      collect_flat_3d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

// ============================================================================
// Section 12: upper_slice — chunk_size >= N must consume everything at once
// ============================================================================

TEST_CASE("upper_slice.oversized_chunk_2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  size_t N = r.size();  // 12
  auto box = r.upper_slice(0, N * 10); size_t consumed = box.size();
  REQUIRE(consumed == N);
  REQUIRE(box.size() == N);
  REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 3);
  REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 4);
}

TEST_CASE("upper_slice.oversized_chunk_3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  size_t N = r.size();  // 24
  auto box = r.upper_slice(0, 9999); size_t consumed = box.size();
  REQUIRE(consumed == N);
  REQUIRE(box.size() == N);
}

// ============================================================================
// Section 13: upper_slice — chunk_size=1 always produces unit boxes
// ============================================================================

TEST_CASE("upper_slice.unit_chunk_2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto box = r.upper_slice(flat, 1); size_t consumed = box.size();
    REQUIRE(consumed == 1);
    REQUIRE(box.size() == 1);
  }
}

TEST_CASE("upper_slice.unit_chunk_3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto box = r.upper_slice(flat, 1); size_t consumed = box.size();
    REQUIRE(consumed == 1);
    REQUIRE(box.size() == 1);
  }
}

// ============================================================================
// Section 14: upper_slice — trailing-zeros rule (orthogonality invariant)
// When an inner dimension is not at coordinate 0, grow_dim must be the
// innermost (N-1), which means outer dimensions are locked.
// ============================================================================

TEST_CASE("upper_slice.trailing_zeros.inner_not_at_zero") {
  // 2D: 4x6, unit steps
  // flat_beg=3 -> coords (0, 3): dim1 != 0, so grow_dim must be 1
  // Requesting 10 should be capped to the remaining 3 elements in the row.
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  {
    auto box = r.upper_slice(3, 10); size_t consumed = box.size();
    // coords: (0, 3); must stay on row 0, innermost grows
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 3);
    REQUIRE(consumed == 3);  // only 3 elements left in the row
  }

  // flat_beg=9 -> coords (1, 3): same constraint on row 1
  {
    auto box = r.upper_slice(9, 10); size_t consumed = box.size();
    REQUIRE(std::get<0>(box.dim(0)) == 1); REQUIRE(std::get<1>(box.dim(0)) == 2);
    REQUIRE(std::get<0>(box.dim(1)) == 3);
    REQUIRE(consumed == 3);
  }
}

TEST_CASE("upper_slice.trailing_zeros.3d_mid_row") {
  // 3D: 3x4x5; flat_beg at (0,0,3) -> dim2 != 0, outer dims locked
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t flat = 0*20 + 0*5 + 3;   // coords (0,0,3)
  auto box = r.upper_slice(flat, 100); size_t consumed = box.size();
  // inner row incomplete: only 2 elements left (indices 3,4)
  REQUIRE(consumed == 2);
  REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
  REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 1);
  REQUIRE(std::get<0>(box.dim(2)) == 3); REQUIRE(std::get<1>(box.dim(2)) == 5);
}

TEST_CASE("upper_slice.trailing_zeros.3d_clean_boundary") {
  // When coords are (row, 0, 0), can grow dim 0 if budget allows
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t flat = 1*20 + 0*5 + 0;   // coords (1,0,0)
  auto box = r.upper_slice(flat, 40); size_t consumed = box.size();
  // Budget 40 >= one full row slice (20 elements) -> can grow dim 0
  REQUIRE(consumed == 40);
  REQUIRE(std::get<0>(box.dim(0)) == 1); REQUIRE(std::get<1>(box.dim(0)) == 3);
  REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 4);
  REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 5);
}

// ============================================================================
// Section 16: upper_slice — 4D sanity check
// ============================================================================

TEST_CASE("upper_slice.full_coverage.4d") {
  tf::IndexRanges<int, 4> r(
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t N = r.size();   // 120

  for (size_t cs : {1, 5, 12, 60, 120}) {
    std::vector<int> visited(N, 0);
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      // Enumerate box elements and mark visited
      size_t D1 = r.size(1), D2 = r.size(2), D3 = r.size(3);
      for (int a = std::get<0>(box.dim(0)); a < std::get<1>(box.dim(0)); a++)
        for (int b = std::get<0>(box.dim(1)); b < std::get<1>(box.dim(1)); b++)
          for (int c = std::get<0>(box.dim(2)); c < std::get<1>(box.dim(2)); c++)
            for (int d = std::get<0>(box.dim(3)); d < std::get<1>(box.dim(3)); d++)
              visited[a*D1*D2*D3 + b*D2*D3 + c*D3 + d]++;
      cursor += consumed;
    }
    for (int v : visited) REQUIRE(v == 1);
  }
}

// ============================================================================
// Section 17: zero-size dimension behaviour
//
// A zero-size dimension at position d stops the active flat space there.
// size() returns the product of dims [0, d), i.e. the outer dims only.
// Dimensions [d, N) are copied as full extent into sub-boxes.
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

TEST_CASE("IndexRangeND.zero_size.ceil_floor.various_dimensions") {
  // For each N, verify that ceil/floor only see active suffix products.

  // 5D: 3x4x0x5x6 -> active dims: 3,4 -> size=12, boundaries: 1,4,12
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1));
    REQUIRE(r.size() == 12);
    REQUIRE(r.ceil(1)  == 1);
    REQUIRE(r.ceil(3)  == 4);   // rounds up to dim1 boundary
    REQUIRE(r.ceil(4)  == 4);
    REQUIRE(r.ceil(5)  == 12);  // rounds up to full active size
    REQUIRE(r.ceil(12) == 12);
    REQUIRE(r.ceil(99) == 12);  // capped

    REQUIRE(r.floor(1)  == 1);
    REQUIRE(r.floor(3)  == 1);
    REQUIRE(r.floor(4)  == 4);
    REQUIRE(r.floor(11) == 4);
    REQUIRE(r.floor(12) == 12);
    REQUIRE(r.floor(99) == 12);
  }

  // 7D: 2x3x4x0x5x6x7 -> active: 2,3,4 -> size=24, boundaries: 1,4,12,24
  {
    tf::IndexRanges<int, 7> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1));
    REQUIRE(r.size() == 24);
    REQUIRE(r.ceil(1)  == 1);
    REQUIRE(r.ceil(4)  == 4);
    REQUIRE(r.ceil(5)  == 12);
    REQUIRE(r.ceil(12) == 12);
    REQUIRE(r.ceil(13) == 24);
    REQUIRE(r.ceil(24) == 24);

    REQUIRE(r.floor(3)  == 1);
    REQUIRE(r.floor(4)  == 4);
    REQUIRE(r.floor(11) == 4);
    REQUIRE(r.floor(12) == 12);
    REQUIRE(r.floor(23) == 12);
    REQUIRE(r.floor(24) == 24);
  }

  // 11D: outermost zero -> size=0, ceil/floor return 0
  {
    tf::IndexRanges<int, 11> r(
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,3,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1), tf::IndexRange<int>(0,8,1), tf::IndexRange<int>(0,9,1),
      tf::IndexRange<int>(0,10,1),tf::IndexRange<int>(0,11,1));
    REQUIRE(r.size() == 0);
    REQUIRE(r.ceil(1)  == 0);
    REQUIRE(r.floor(1) == 0);
  }
}

TEST_CASE("IndexRangeND.zero_size.upper_slice.various_dimensions") {
  // 5D: 6x0x4x5x3 -> active: 6 -> size=6
  // upper_slice partitions [0,6); dims 1..4 are full extent in box.
  {
    tf::IndexRanges<int, 5> r(
      tf::IndexRange<int>(0,6,1), tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,3,1));
    REQUIRE(r.size() == 6);

    auto box = r.upper_slice(2, 3); size_t consumed = box.size();
    REQUIRE(consumed == 3);
    REQUIRE(std::get<0>(box.dim(0)) == 2); REQUIRE(std::get<1>(box.dim(0)) == 5);
    REQUIRE(box.size(1)  == 0);  // zero-size preserved
    REQUIRE(box.size(2)  == 4);  // full extent
    REQUIRE(box.size(3)  == 5);  // full extent
    REQUIRE(box.size(4)  == 3);  // full extent
  }

  // 9D: 4x3x0x2x2x2x2x2x2 -> active: 4,3 -> size=12
  {
    tf::IndexRanges<int, 9> r(
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 12);

    // flat=0, chunk=3: takes 3 steps along dim1 (active inner boundary=3)
    auto box = r.upper_slice(0, 3); size_t consumed = box.size();
    REQUIRE(consumed == 3);
    REQUIRE(box.size(0) == 1);  // locked at row 0
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 3);  // grow dim
    REQUIRE(box.size(2) == 0);  // zero-size
    // dims 3..8: full extent
    for (size_t d = 3; d < 9; ++d) REQUIRE(box.size(d) == 2);
  }

  // 13D: 5x0x... -> active: 5 -> size=5, full coverage
  {
    tf::IndexRanges<int, 13> r(
      tf::IndexRange<int>(0,5,1),  tf::IndexRange<int>(0,0,1),  tf::IndexRange<int>(0,3,1),
      tf::IndexRange<int>(0,4,1),  tf::IndexRange<int>(0,5,1),  tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1),  tf::IndexRange<int>(0,8,1),  tf::IndexRange<int>(0,9,1),
      tf::IndexRange<int>(0,10,1), tf::IndexRange<int>(0,11,1), tf::IndexRange<int>(0,12,1),
      tf::IndexRange<int>(0,13,1));
    size_t N = r.size();
    REQUIRE(N == 5);

    std::vector<int> visited(5, 0);
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, 2); size_t consumed = box.size();
      REQUIRE(consumed > 0);
      REQUIRE(box.size(1) == 0);  // zero-size dim always preserved
      for (int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); ++i) visited[i]++;
      cursor += consumed;
    }
    for (int v : visited) REQUIRE(v == 1);
  }
}

TEST_CASE("IndexRangeND.zero_size.lower_slice.various_dimensions") {
  // 7D: 3x5x0x4x4x4x4 -> active: 3,5 -> size=15
  {
    tf::IndexRanges<int, 7> r(
      tf::IndexRange<int>(0,3,1), tf::IndexRange<int>(0,5,1), tf::IndexRange<int>(0,0,1),
      tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,4,1), tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,4,1));
    REQUIRE(r.size() == 15);

    // chunk=4: active suffix products are 1, 5, 15.
    // grow_dim=1 (active_inner_vol=1), steps_needed=floor(4/1)=4, consumed=4 (<= 4)
    auto box = r.lower_slice(0, 4); size_t consumed = box.size();
    REQUIRE(consumed <= 4);
    REQUIRE(consumed == 4);  // 4 steps along dim1, each costing 1 element
    REQUIRE(box.size(2) == 0);  // zero-size preserved

    // chunk=5: exactly one inner row
    auto box2 = r.lower_slice(0, 5); size_t consumed2 = box2.size();
    REQUIRE(consumed2 == 5);
    REQUIRE(consumed2 <= 5);
    REQUIRE(std::get<0>(box2.dim(1)) == 0); REQUIRE(std::get<1>(box2.dim(1)) == 5);
  }

  // 11D: 2x3x4x5x0x... -> active: 2,3,4,5 -> size=120
  // full coverage with lower_slice
  {
    tf::IndexRanges<int, 11> r(
      tf::IndexRange<int>(0,2,1),  tf::IndexRange<int>(0,3,1),  tf::IndexRange<int>(0,4,1),
      tf::IndexRange<int>(0,5,1),  tf::IndexRange<int>(0,0,1),  tf::IndexRange<int>(0,6,1),
      tf::IndexRange<int>(0,7,1),  tf::IndexRange<int>(0,8,1),  tf::IndexRange<int>(0,9,1),
      tf::IndexRange<int>(0,10,1), tf::IndexRange<int>(0,11,1));
    size_t N = r.size();
    REQUIRE(N == 120);

    // drain with chunk=10 and verify no element double-visited
    std::vector<int> visited(N, 0);
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.lower_slice(cursor, 10); size_t consumed = box.size();
      REQUIRE(consumed > 0);
      REQUIRE(consumed <= 10);
      REQUIRE(box.size(4) == 0);  // zero-size dim preserved
      // count outer flat indices covered
      size_t box_outer = box.size(0) * box.size(1) *
                         box.size(2) * box.size(3);
      REQUIRE(box_outer == consumed);
      for (size_t k = cursor; k < cursor + consumed; ++k) visited[k]++;
      cursor += consumed;
    }
    for (int v : visited) REQUIRE(v == 1);
  }
}

TEST_CASE("IndexRangeND.zero_size.sequential_analogy.high_dimensions") {
  // 19D with zero at d=9: size() = 2^9 = 512 (outer 9 dims contribute)
  {
    tf::IndexRanges<int, 19> r(
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,0,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 512);

    // dims 9..18 are inactive — verify they appear as full extent in a sub-box
    auto box = r.upper_slice(0, 8); size_t consumed = box.size();
    REQUIRE(consumed > 0);
    REQUIRE(box.size(9)  == 0);  // zero-size preserved
    for (size_t d = 10; d < 19; ++d) REQUIRE(box.size(d) == 2);
  }

  // 15D with zero at d=0: size() = 0
  {
    tf::IndexRanges<int, 15> r(
      tf::IndexRange<int>(0,0,1),  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1),  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1),  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1),  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1),
      tf::IndexRange<int>(0,2,1),  tf::IndexRange<int>(0,2,1), tf::IndexRange<int>(0,2,1));
    REQUIRE(r.size() == 0);

    auto box = r.upper_slice(0, 1); size_t consumed = box.size();
    REQUIRE(consumed == 0);  // no active work
  }
}



// ============================================================================
// Section 18: IndexRanges<T,N>::ceil and floor
//
// ceil(chunk_size) returns the smallest suffix-product boundary >= chunk_size.
// floor(chunk_size) returns the largest suffix-product boundary <= chunk_size.
// Both are lightweight size queries — no box constructed, no flat_beg needed.
// When chunk_size is already a boundary, ceil == floor == chunk_size.
// ============================================================================

TEST_CASE("IndexRangeND.ceil.3d_documented_examples") {
  // 3D range: 4 x 5 x 10  — suffix-product boundaries: 1, 10, 50, 200
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // exact boundaries — ceil returns them unchanged
  REQUIRE(r.ceil(1)   == 1);
  REQUIRE(r.ceil(10)  == 10);
  REQUIRE(r.ceil(50)  == 50);
  REQUIRE(r.ceil(200) == 200);

  // non-boundaries — ceil rounds up
  REQUIRE(r.ceil(2)   == 10);   // up to first inner row
  REQUIRE(r.ceil(7)   == 10);
  REQUIRE(r.ceil(11)  == 50);   // up to first outer row
  REQUIRE(r.ceil(30)  == 50);
  REQUIRE(r.ceil(51)  == 200);  // up to full range
  REQUIRE(r.ceil(100) == 200);
  REQUIRE(r.ceil(199) == 200);

  // chunk_size > size() — capped at size()
  REQUIRE(r.ceil(201) == 200);
  REQUIRE(r.ceil(999) == 200);
}

TEST_CASE("IndexRangeND.floor.3d_documented_examples") {
  // 3D range: 4 x 5 x 10  — suffix-product boundaries: 1, 10, 50, 200
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // exact boundaries — floor returns them unchanged
  REQUIRE(r.floor(1)   == 1);
  REQUIRE(r.floor(10)  == 10);
  REQUIRE(r.floor(50)  == 50);
  REQUIRE(r.floor(200) == 200);

  // non-boundaries — floor rounds down
  REQUIRE(r.floor(2)   == 1);    // down to 1
  REQUIRE(r.floor(7)   == 1);
  REQUIRE(r.floor(9)   == 1);
  REQUIRE(r.floor(11)  == 10);   // down to one inner row
  REQUIRE(r.floor(30)  == 10);
  REQUIRE(r.floor(49)  == 10);
  REQUIRE(r.floor(51)  == 50);   // down to one outer row
  REQUIRE(r.floor(100) == 50);
  REQUIRE(r.floor(199) == 50);

  // chunk_size >= size() — capped at size()
  REQUIRE(r.floor(200) == 200);
  REQUIRE(r.floor(999) == 200);
}

TEST_CASE("IndexRangeND.ceil_floor.agree_on_boundaries.3d") {
  // On every natural boundary, ceil == floor == boundary value
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );
  for (size_t b : {size_t{1}, size_t{10}, size_t{50}, size_t{200}}) {
    REQUIRE(r.ceil(b)  == b);
    REQUIRE(r.floor(b) == b);
    REQUIRE(r.ceil(b)  == r.floor(b));
  }
}

TEST_CASE("IndexRangeND.ceil_floor.ordering") {
  // floor(x) <= x <= ceil(x) for all x, and floor(x) <= ceil(x)
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();
  for (size_t cs = 1; cs <= N + 10; ++cs) {
    size_t c = r.ceil(cs);
    size_t f = r.floor(cs);
    REQUIRE(f <= cs);
    REQUIRE(f <= c);
    // ceil is either >= cs, or capped at N when cs > N
    if (cs <= N) { REQUIRE(c >= cs); }
    else          { REQUIRE(c == N); }
  }
}

TEST_CASE("IndexRangeND.ceil_floor.2d") {
  // 2D range: 3 x 4 — suffix-product boundaries: 1, 4, 12
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  REQUIRE(r.ceil(1)  == 1);
  REQUIRE(r.ceil(2)  == 4);
  REQUIRE(r.ceil(4)  == 4);
  REQUIRE(r.ceil(5)  == 12);
  REQUIRE(r.ceil(12) == 12);
  REQUIRE(r.ceil(13) == 12);  // capped

  REQUIRE(r.floor(1)  == 1);
  REQUIRE(r.floor(3)  == 1);
  REQUIRE(r.floor(4)  == 4);
  REQUIRE(r.floor(11) == 4);
  REQUIRE(r.floor(12) == 12);
  REQUIRE(r.floor(99) == 12);
}

TEST_CASE("IndexRangeND.ceil_floor.exhaustive_2d") {
  // For all 2D shapes up to 6x6, verify:
  //   1. ceil and floor always return a valid suffix-product boundary
  //   2. floor(x) <= x, ceil(x) >= x (or capped at N)
  //   3. On boundaries ceil == floor
  for (int d0 = 1; d0 <= 6; ++d0) {
    for (int d1 = 1; d1 <= 6; ++d1) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, d0, 1),
        tf::IndexRange<int>(0, d1, 1)
      );
      size_t N     = r.size();
      size_t inner = static_cast<size_t>(d1);
      // valid suffix-product boundaries for this shape
      std::vector<size_t> boundaries = {1, inner, N};

      for (size_t cs = 1; cs <= N + 2; ++cs) {
        size_t c = r.ceil(cs);
        size_t f = r.floor(cs);

        // both must be valid boundaries
        REQUIRE(std::find(boundaries.begin(), boundaries.end(), c) != boundaries.end());
        REQUIRE(std::find(boundaries.begin(), boundaries.end(), f) != boundaries.end());

        // ordering
        REQUIRE(f <= cs);
        REQUIRE(f <= c);

        // on a boundary, ceil == floor
        if (std::find(boundaries.begin(), boundaries.end(), cs) != boundaries.end()) {
          REQUIRE(c == cs);
          REQUIRE(f == cs);
        }
      }
    }
  }
}

// ============================================================================
// Section 18: lower_slice — floor variant
//
// lower_slice guarantees consumed <= chunk_size, with one
// irreducible exception: when chunk_size is smaller than a single
// indivisible inner step, we still return that one step for forward progress.
//
// Key behavioral differences from upper_slice:
//  - grow_dim is the outermost dim whose per-step volume still fits within budget
//  - steps_needed uses floor division, not ceil
//  - no overshoot past chunk_size (except the irreducible minimum case)
// ============================================================================

// Helper: drain the full flat space using lower_slice and return
// the list of visited flat indices.  Analogous to drain_upper_slice above.
template <typename R, typename VisitFn>
std::vector<size_t> drain_lower_slice(
    const R& range, size_t chunk_size, VisitFn visit) {
  size_t N = range.size();
  std::vector<size_t> visited;
  size_t cursor = 0;
  while (cursor < N) {
    auto box = range.lower_slice(cursor, chunk_size); size_t consumed = box.size();
    REQUIRE(consumed > 0);            // must make forward progress
    REQUIRE(cursor + consumed <= N);  // must not go past the end
    visit(box, visited);
    cursor += consumed;
  }
  return visited;
}

// ---- Basic contract: consumed <= chunk_size (except irreducible min) ----

TEST_CASE("lower_slice.consumed_le_requested.2d") {
  // 3x7 range. For each starting flat index and various chunk sizes,
  // verify consumed <= chunk_size.
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 7, 1)
  );
  size_t N = r.size();  // 21
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 5, 7, 10, 21}) {
      auto box = r.lower_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed > 0);
      REQUIRE(consumed <= cs);
      REQUIRE(consumed == box.size());
    }
  }
}

TEST_CASE("lower_slice.consumed_le_requested.3d") {
  // 4x5x6 range
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();  // 120
  for (size_t flat = 0; flat < N; flat += 3) {  // stride to keep test fast
    for (size_t cs : {1, 6, 7, 12, 30, 60, 120}) {
      auto box = r.lower_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed > 0);
      REQUIRE(consumed <= cs);
      REQUIRE(consumed == box.size());
    }
  }
}

// ---- Specific scenarios mirroring the upper_slice doc examples -------

TEST_CASE("lower_slice.scenarios.3d_4x5x10") {
  // 3D range: 4x5x10, total=200
  tf::IndexRanges<int, 3> range(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // Scenario A: flat_beg=0, requested=30.
  // coords (0,0,0). grow into dim2 (inner_vol=1): 10 elements fit.
  // next try dim1 (next_vol=50 > 30 and inner=10 > 1) -> break.
  // grow_dim=2, active=1, steps_needed=floor(30/1)=30, steps_left=10 -> 10.
  // consumed=10 (atmost 30, no overshoot).
  {
    auto box = range.lower_slice(0, 30); size_t consumed = box.size();
    REQUIRE(consumed == 10);
    REQUIRE(consumed <= 30);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 1);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
  }

  // Scenario B: flat_beg=0, requested=50.
  // coords (0,0,0). dim2: next_vol=10 <=50 -> commit grow_dim=2, active=1, inner=10.
  // dim1: next_vol=50 <=50 -> commit grow_dim=1, active=10, inner=50.
  // dim0: next_vol=200 >50 and inner=50 > 1 -> break.
  // steps_needed=floor(50/10)=5, steps_left=5 -> 5. consumed=50.
  {
    auto box = range.lower_slice(0, 50); size_t consumed = box.size();
    REQUIRE(consumed == 50);
    REQUIRE(consumed <= 50);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);  // locked
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 5);  // full extent
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10); // full extent
  }

  // Scenario C: flat_beg=0, requested=100.
  // dim2: next_vol=10 <=100 -> grow_dim=2, active=1, inner=10.
  // dim1: next_vol=50 <=100 -> grow_dim=1, active=10, inner=50.
  // dim0: next_vol=200 >100, inner=50 > 1 -> break.
  // steps_needed=floor(100/10)=10 but steps_left(dim1)=5 -> steps_to_take=5. consumed=50.
  // (atmost 100, returns 50 — largest full-row slab that fits)
  {
    auto box = range.lower_slice(0, 100); size_t consumed = box.size();
    REQUIRE(consumed == 50);
    REQUIRE(consumed <= 100);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 5);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
  }

  // Scenario D: flat_beg=0, requested=200 (full range).
  // dim2: next_vol=10 -> grow_dim=2. dim1: next_vol=50 -> grow_dim=1, active=10.
  // dim0: next_vol=200 <=200 -> grow_dim=0, active=50, inner=200.
  // loop ends (d-- wraps past 0).
  // steps_needed=floor(200/50)=4, steps_left=4 -> 4. consumed=200.
  {
    auto box = range.lower_slice(0, 200); size_t consumed = box.size();
    REQUIRE(consumed == 200);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 4);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 5);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10);
  }

  // Scenario E: flat_beg=30, requested=30.
  // coords (0,3,0). d=2: coords[3] n/a. next_vol=10<=30 -> grow_dim=2, active=1, inner=10.
  // d=1: coords[2]=0 ok. next_vol=50>30, inner=10>1 -> break.
  // grow_dim=2, steps_needed=30, steps_left(dim2)=10-0=10 -> 10. consumed=10.
  {
    auto box = range.lower_slice(30, 30); size_t consumed = box.size();
    REQUIRE(consumed <= 30);
    REQUIRE(consumed > 0);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);  // locked row 0
    REQUIRE(std::get<0>(box.dim(1)) == 3); REQUIRE(std::get<1>(box.dim(1)) == 4);  // locked col 3
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 10); // full inner
  }
}

// ---- Alignment-clean cases: requested is a multiple of inner volume ----------

TEST_CASE("ceil_floor_agree_on_aligned_chunk.2d") {
  // For a 2D range, the natural boundary sizes are:
  //   inner_vol = dim1.size()       (one full inner row)
  //   outer_vol = dim0*dim1.size()  (entire range)
  //
  // When chunk_size is exactly one of these boundary values AND flat_beg
  // itself lands on a hyperplane boundary (coords that are multiples of that
  // boundary), ceil and floor must return identical boxes and consumed values —
  // analogous to std::ceil(x) == std::floor(x) when x is an integer.
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),   // dim0: 4 rows
    tf::IndexRange<int>(0, 6, 1)    // dim1: 6 cols
  );
  size_t d1 = r.size(1);  // 6  — one inner row
  size_t N  = r.size();   // 24 — full range

  // chunk_size == inner row (6): every flat index that is a multiple of 6
  // is aligned to an inner-row boundary.
  for (size_t flat = 0; flat < N; flat += d1) {
    auto box_c = r.upper_slice(flat, d1); size_t cons_c = box_c.size();
    auto box_f = r.lower_slice(flat, d1); size_t cons_f = box_f.size();
    REQUIRE(cons_c == d1);
    REQUIRE(cons_f == d1);
    REQUIRE(cons_c == cons_f);
    REQUIRE(std::get<0>(box_c.dim(0)) == std::get<0>(box_f.dim(0)));
    REQUIRE(std::get<1>(box_c.dim(0))   == std::get<1>(box_f.dim(0)));
    REQUIRE(std::get<0>(box_c.dim(1)) == std::get<0>(box_f.dim(1)));
    REQUIRE(std::get<1>(box_c.dim(1))   == std::get<1>(box_f.dim(1)));
  }

  // chunk_size == full range (24): only flat=0 is aligned.
  {
    auto box_c = r.upper_slice(0, N); size_t cons_c = box_c.size();
    auto box_f = r.lower_slice(0, N); size_t cons_f = box_f.size();
    REQUIRE(cons_c == N);
    REQUIRE(cons_f == N);
    REQUIRE(cons_c == cons_f);
    REQUIRE(std::get<0>(box_c.dim(0)) == std::get<0>(box_f.dim(0)));
    REQUIRE(std::get<1>(box_c.dim(0))   == std::get<1>(box_f.dim(0)));
    REQUIRE(std::get<0>(box_c.dim(1)) == std::get<0>(box_f.dim(1)));
    REQUIRE(std::get<1>(box_c.dim(1))   == std::get<1>(box_f.dim(1)));
  }
}

TEST_CASE("ceil_floor_agree_on_aligned_chunk.3d") {
  // 3D range: 3 x 4 x 5, total = 60.
  // Natural boundary sizes:
  //   innermost row  : dim2.size()       =  5
  //   middle slab    : dim1*dim2.size()  = 20
  //   full range     : dim0*dim1*dim2    = 60
  //
  // For each boundary size, sweep every flat index that is a multiple of that
  // boundary (i.e. aligned starts) and verify ceil == floor for both consumed
  // and the returned box.
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t d2      = r.size(2);           //  5
  size_t d1d2    = r.size(1) * d2;      // 20
  size_t N       = r.size();            // 60

  auto boxes_equal = [](const auto& a, const auto& b, size_t rank) {
    for (size_t d = 0; d < rank; ++d) {
      if (std::get<0>(a.dim(d)) != std::get<0>(b.dim(d))) return false;
      if (std::get<1>(a.dim(d))   != std::get<1>(b.dim(d)))   return false;
    }
    return true;
  };

  // Aligned on innermost-row boundary (stride = d2 = 5)
  for (size_t flat = 0; flat < N; flat += d2) {
    auto box_c = r.upper_slice(flat, d2); size_t cons_c = box_c.size();
    auto box_f = r.lower_slice(flat, d2); size_t cons_f = box_f.size();
    REQUIRE(cons_c == cons_f);
    REQUIRE(cons_c == d2);
    REQUIRE(boxes_equal(box_c, box_f, 3));
  }

  // Aligned on middle-slab boundary (stride = d1d2 = 20)
  for (size_t flat = 0; flat < N; flat += d1d2) {
    auto box_c = r.upper_slice(flat, d1d2); size_t cons_c = box_c.size();
    auto box_f = r.lower_slice(flat, d1d2); size_t cons_f = box_f.size();
    REQUIRE(cons_c == cons_f);
    REQUIRE(cons_c == d1d2);
    REQUIRE(boxes_equal(box_c, box_f, 3));
  }

  // Full range (only flat=0 is aligned)
  {
    auto box_c = r.upper_slice(0, N); size_t cons_c = box_c.size();
    auto box_f = r.lower_slice(0, N); size_t cons_f = box_f.size();
    REQUIRE(cons_c == cons_f);
    REQUIRE(cons_c == N);
    REQUIRE(boxes_equal(box_c, box_f, 3));
  }
}

TEST_CASE("ceil_floor_agree_on_aligned_chunk.exhaustive_2d") {
  // The ceil==floor contract holds when chunk_size is exactly a natural
  // per-step volume of the range — i.e. the volume of one step along some
  // dimension d, which equals the product of all inner dimension sizes.
  // For a 2D range these are: inner=d1 (one full row) and outer=d0*d1 (full range).
  // Both are tested at every valid aligned starting flat index (multiples of that volume).
  for (int d0 = 1; d0 <= 6; ++d0) {
    for (int d1 = 1; d1 <= 6; ++d1) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, d0, 1),
        tf::IndexRange<int>(0, d1, 1)
      );
      size_t N     = r.size();
      size_t inner = static_cast<size_t>(d1);  // per-step vol at grow_dim=1
      size_t outer = N;                         // per-step vol at grow_dim=0

      // chunk_size == inner: aligned starts are multiples of inner
      for (size_t flat = 0; flat < N; flat += inner) {
        auto box_c = r.upper_slice(flat, inner); size_t cons_c = box_c.size();
        auto box_f = r.lower_slice(flat, inner); size_t cons_f = box_f.size();
        REQUIRE(cons_c == cons_f);
        REQUIRE(std::get<0>(box_c.dim(0)) == std::get<0>(box_f.dim(0)));
        REQUIRE(std::get<1>(box_c.dim(0))   == std::get<1>(box_f.dim(0)));
        REQUIRE(std::get<0>(box_c.dim(1)) == std::get<0>(box_f.dim(1)));
        REQUIRE(std::get<1>(box_c.dim(1))   == std::get<1>(box_f.dim(1)));
      }

      // chunk_size == outer (full range): only flat=0 is aligned
      {
        auto box_c = r.upper_slice(0, outer); size_t cons_c = box_c.size();
        auto box_f = r.lower_slice(0, outer); size_t cons_f = box_f.size();
        REQUIRE(cons_c == cons_f);
        REQUIRE(std::get<0>(box_c.dim(0)) == std::get<0>(box_f.dim(0)));
        REQUIRE(std::get<1>(box_c.dim(0))   == std::get<1>(box_f.dim(0)));
        REQUIRE(std::get<0>(box_c.dim(1)) == std::get<0>(box_f.dim(1)));
        REQUIRE(std::get<1>(box_c.dim(1))   == std::get<1>(box_f.dim(1)));
      }
    }
  }
}

// ---- Full coverage: atmost must cover every element exactly once ------------

TEST_CASE("lower_slice.full_coverage.2d") {
  // Sweep multiple 2D shapes and chunk sizes; verify exactly-once coverage.
  for (int d0 : {1, 3, 5, 7}) {
    for (int d1 : {1, 4, 6, 10}) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, d0, 1),
        tf::IndexRange<int>(0, d1, 1)
      );
      size_t N = r.size();
      for (size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t(d1), N}) {
        std::vector<size_t> visited;
        size_t cursor = 0;
        while (cursor < N) {
          auto box = r.lower_slice(cursor, cs); size_t consumed = box.size();
          REQUIRE(consumed > 0);
          REQUIRE(consumed <= cs);
          REQUIRE(consumed == box.size());
          for (int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i++)
            for (int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j++)
              visited.push_back(static_cast<size_t>(i * d1 + j));
          cursor += consumed;
        }
        REQUIRE(visited.size() == N);
        std::sort(visited.begin(), visited.end());
        for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
      }
    }
  }
}

TEST_CASE("lower_slice.full_coverage.3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();  // 120
  size_t D1 = r.size(1), D2 = r.size(2);

  for (size_t cs : {1, 3, 6, 7, 10, 30, 60, 120}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.lower_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed > 0);
      REQUIRE(consumed <= cs);
      REQUIRE(consumed == box.size());
      for (int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i++)
        for (int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j++)
          for (int k = std::get<0>(box.dim(2)); k < std::get<1>(box.dim(2)); k++)
            visited.push_back(static_cast<size_t>(i*D1*D2 + j*D2 + k));
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

// ---- Trailing-zeros rule: mid-row starting position -------------------------

TEST_CASE("lower_slice.trailing_zeros.mid_row") {
  // 3x7, starting flat_beg=4 -> coords (0,4): dim1 at pos 4, 3 remain in row.
  // trailing-zeros fires: can't grow into dim0.
  // grow_dim stays at N-1=1. active_inner_vol=1, steps_needed=floor(cs/1).
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 7, 1)
  );
  // flat=4: coords (0,4). 3 elements remain in row 0 (pos 4,5,6).
  {
    auto box = r.lower_slice(4, 10); size_t consumed = box.size();
    REQUIRE(consumed == 3);   // min(steps_left=3, floor(10/1)=10) = 3
    REQUIRE(consumed <= 10);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1); // locked
    REQUIRE(std::get<0>(box.dim(1)) == 4); REQUIRE(std::get<1>(box.dim(1)) == 7);
  }
  // flat=4, cs=2: only 2 elements consumed even though 3 remain.
  {
    auto box = r.lower_slice(4, 2); size_t consumed = box.size();
    REQUIRE(consumed == 2);
    REQUIRE(consumed <= 2);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 4); REQUIRE(std::get<1>(box.dim(1)) == 6);
  }
}

// ---- Irreducible minimum: requested < innermost step volume -----------------

TEST_CASE("lower_slice.boundary_rounding.3d") {
  // Verify floor rounds down to the nearest hyperplane boundary.
  // 2x3x4 range: natural boundaries are 4 (inner row) and 12 (outer row).
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );

  // chunk_size=1: grows along dim2, steps_needed=1 -> consumed=1
  {
    auto box = r.lower_slice(0, 1); size_t consumed = box.size();
    REQUIRE(consumed == 1);
    REQUIRE(consumed <= 1);
    REQUIRE(box.size() == 1);
  }
  // chunk_size=3: grows along dim2, steps_needed=3 -> consumed=3
  {
    auto box = r.lower_slice(0, 3); size_t consumed = box.size();
    REQUIRE(consumed == 3);
    REQUIRE(consumed <= 3);
  }
  // chunk_size=4: exactly one inner row -> consumed=4
  {
    auto box = r.lower_slice(0, 4); size_t consumed = box.size();
    REQUIRE(consumed == 4);
    REQUIRE(consumed <= 4);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 1);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 4);
  }
  // chunk_size=5: not a boundary — floor rounds down to 4 (one inner row)
  {
    auto box = r.lower_slice(0, 5); size_t consumed = box.size();
    REQUIRE(consumed == 4);
    REQUIRE(consumed <= 5);
  }
  // chunk_size=11: not a boundary — floor rounds down to 4 (one inner row,
  // since promoting to grow_dim=1 would cost 12 per step which exceeds 11)
  {
    auto box = r.lower_slice(0, 11); size_t consumed = box.size();
    REQUIRE(consumed == 4);
    REQUIRE(consumed <= 11);
  }
  // chunk_size=12: exactly one outer row -> consumed=12
  {
    auto box = r.lower_slice(0, 12); size_t consumed = box.size();
    REQUIRE(consumed == 12);
    REQUIRE(consumed <= 12);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 0); REQUIRE(std::get<1>(box.dim(1)) == 3);
    REQUIRE(std::get<0>(box.dim(2)) == 0); REQUIRE(std::get<1>(box.dim(2)) == 4);
  }
}

// ---- ceil vs atmost: atmost never returns more than ceil --------------------

TEST_CASE("lower_slice.never_exceeds_ceil") {
  // For any (flat, cs), consumed_floor <= consumed_ceil.
  // With unit step sizes, consumed_floor <= cs always holds unconditionally.
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 3, 6, 7, 13, 30, 60, 120}) {
      auto box_c = r.upper_slice(flat, cs); size_t cons_c = box_c.size();
      auto box_f = r.lower_slice(flat, cs); size_t cons_f = box_f.size();
      REQUIRE(cons_f <= cons_c);      // atmost never returns more than ceil
      REQUIRE(cons_f == box_f.size());
      REQUIRE(cons_c == box_c.size());
    }
  }
}

// ---- Exhaustive sweep: atmost covers all elements exactly once --------------

TEST_CASE("lower_slice.exhaustive.2d_all_shapes") {
  for (int d0 = 1; d0 <= 8; d0++) {
    for (int d1 = 1; d1 <= 8; d1++) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, d0, 1),
        tf::IndexRange<int>(0, d1, 1)
      );
      size_t N = r.size();
      for (size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t(d1), N}) {
        std::vector<int> visited(N, 0);
        size_t cursor = 0;
        while (cursor < N) {
          auto box = r.lower_slice(cursor, cs); size_t consumed = box.size();
          REQUIRE(consumed > 0);
          REQUIRE(consumed <= cs);
          for (int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i++)
            for (int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j++)
              visited[i * d1 + j]++;
          cursor += consumed;
        }
        for (int v : visited) REQUIRE(v == 1);
      }
    }
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

// upper_slice full-coverage with negative steps
TEST_CASE("NegativeStep.2D.dim0_negative.full_coverage") {
  // dim0: 10 down to 0, step -2 (5 elements)
  // dim1: 0 up to 6,   step  1 (6 elements)
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(0,  6,  1)
  );
  size_t N = r.size();
  for (size_t cs : {1, 3, 6, 10, 30}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_2d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.2D.dim1_negative.full_coverage") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(9, 0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 2, 3, 6, 12}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_2d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.2D.both_negative.full_coverage") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 2, 3, 5, 9}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_2d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

// Exhaustive sweep: all combinations of sign and step magnitude for 2D
TEST_CASE("NegativeStep.2D.exhaustive_mixed_signs") {
  // All combinations of step sign for both dimensions,
  // with magnitudes 1 and 2, and sizes 2..5
  for (int si : {-2, -1, 1, 2}) {
    for (int sj : {-2, -1, 1, 2}) {
      for (int di : {2, 3, 5}) {
        for (int dj : {2, 3, 5}) {
          int beg_i = si > 0 ? 0        : di * abs(si);
          int end_i = si > 0 ? di * si  : 0;
          int beg_j = sj > 0 ? 0        : dj * abs(sj);
          int end_j = sj > 0 ? dj * sj  : 0;

          tf::IndexRanges<int, 2> r(
            tf::IndexRange<int>(beg_i, end_i, si),
            tf::IndexRange<int>(beg_j, end_j, sj)
          );
          REQUIRE(r.size() == static_cast<size_t>(di * dj));

          size_t N = r.size();
          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
              REQUIRE(consumed >= 1);
              REQUIRE(consumed == box.size());
              collect_flat_2d(box, r, visited);
              cursor += consumed;
            }
            REQUIRE(visited.size() == N);
            std::sort(visited.begin(), visited.end());
            for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
          }
        }
      }
    }
  }
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

TEST_CASE("NegativeStep.3D.all_negative.full_coverage") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 3, 5, 15, 60}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_3d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.3D.mixed_signs.full_coverage") {
  // dim0 positive, dim1 negative, dim2 positive
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3,   1),
    tf::IndexRange<int>(8, 0,  -2),
    tf::IndexRange<int>(0, 5,   1)
  );
  size_t N = r.size();   // 3*4*5 = 60
  for (size_t cs : {1, 4, 7, 30, 60}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_3d(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

// forward-progress guarantee with negative steps
TEST_CASE("NegativeStep.forward_progress.2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {size_t{1}, size_t{2}, size_t{5}, N}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
      REQUIRE(consumed == box.size());
    }
  }
}

TEST_CASE("NegativeStep.forward_progress.3d") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat += 5) {  // stride to keep test fast
    for (size_t cs : {1, 3, 15, 60}) {
      auto box = r.upper_slice(flat, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
      REQUIRE(consumed == box.size());
    }
  }
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

TEST_CASE("Concepts.IndexRanges1DLike") {
  static_assert( tf::IndexRanges1DLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRanges1DLike<tf::IndexRange<int>>);
  static_assert(!tf::IndexRanges1DLike<tf::IndexRanges<int, 2>>);
  static_assert(!tf::IndexRanges1DLike<tf::IndexRanges<int, 3>>);
  static_assert( tf::IndexRanges1DLike<const tf::IndexRange<int>&>);
  REQUIRE(true);
}

TEST_CASE("Concepts.IndexRangesMDLike") {
  static_assert(!tf::IndexRangesMDLike<tf::IndexRange<int>>);
  static_assert(!tf::IndexRangesMDLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRangesMDLike<tf::IndexRanges<int, 2>>);
  static_assert( tf::IndexRangesMDLike<tf::IndexRanges<int, 3>>);
  static_assert( tf::IndexRangesMDLike<tf::IndexRanges<int, 4>>);
  static_assert( tf::IndexRangesMDLike<const tf::IndexRanges<int, 2>&>);
  REQUIRE(true);
}

// ============================================================================
// Section 18: Negative step sizes
//
// Helpers
// ============================================================================

// Enumerate a 2D range with arbitrary (possibly negative) steps.
// Iteration condition flips based on sign, handled via enumerate_2d which
// already checks step sign. We need collect helpers that work for negative
// steps: the logical position of a value v along a dim with (beg, step) is
//   pos = (v - beg) / step
// which is always non-negative and integral for valid sub-box values.
auto collect_flat_2d_signed(
    const tf::IndexRanges<int, 2>& box,
    const tf::IndexRanges<int, 2>& full,
    std::vector<size_t>& out)
{
  size_t D1 = full.size(1);
  int s0 = std::get<2>(full.dim(0)), b0 = std::get<0>(full.dim(0));
  int s1 = std::get<2>(full.dim(1)), b1 = std::get<0>(full.dim(1));

  // step sign determines loop direction
  auto step0 = [&](int i){ return s0 > 0 ? i < std::get<1>(box.dim(0)) : i > std::get<1>(box.dim(0)); };
  auto step1 = [&](int j){ return s1 > 0 ? j < std::get<1>(box.dim(1)) : j > std::get<1>(box.dim(1)); };

  for (int i = std::get<0>(box.dim(0)); step0(i); i += s0) {
    size_t pi = static_cast<size_t>((i - b0) / s0);
    for (int j = std::get<0>(box.dim(1)); step1(j); j += s1) {
      size_t pj = static_cast<size_t>((j - b1) / s1);
      out.push_back(pi * D1 + pj);
    }
  }
}

auto collect_flat_3d_signed(
    const tf::IndexRanges<int, 3>& box,
    const tf::IndexRanges<int, 3>& full,
    std::vector<size_t>& out)
{
  size_t D1 = full.size(1), D2 = full.size(2);
  int s0 = std::get<2>(full.dim(0)), b0 = std::get<0>(full.dim(0));
  int s1 = std::get<2>(full.dim(1)), b1 = std::get<0>(full.dim(1));
  int s2 = std::get<2>(full.dim(2)), b2 = std::get<0>(full.dim(2));

  auto cmp0 = [&](int i){ return s0 > 0 ? i < std::get<1>(box.dim(0)) : i > std::get<1>(box.dim(0)); };
  auto cmp1 = [&](int j){ return s1 > 0 ? j < std::get<1>(box.dim(1)) : j > std::get<1>(box.dim(1)); };
  auto cmp2 = [&](int k){ return s2 > 0 ? k < std::get<1>(box.dim(2)) : k > std::get<1>(box.dim(2)); };

  for (int i = std::get<0>(box.dim(0)); cmp0(i); i += s0) {
    size_t pi = static_cast<size_t>((i - b0) / s0);
    for (int j = std::get<0>(box.dim(1)); cmp1(j); j += s1) {
      size_t pj = static_cast<size_t>((j - b1) / s1);
      for (int k = std::get<0>(box.dim(2)); cmp2(k); k += s2) {
        size_t pk = static_cast<size_t>((k - b2) / s2);
        out.push_back(pi * D1 * D2 + pj * D2 + pk);
      }
    }
  }
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

// ============================================================================
// Section 18c: upper_slice full coverage — negative step dims
// ============================================================================

TEST_CASE("NegativeStep.upper_slice.full_coverage.2d_both_negative") {
  // dim0: 10,8,6,4,2  dim1: 9,6,3  -> 5x3 = 15 elements
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>( 9, 0, -3)
  );
  size_t N = r.size();
  REQUIRE(N == 15);

  for (size_t cs : {1, 2, 3, 5, 8, 15}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_2d_signed(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.upper_slice.full_coverage.2d_mixed_steps") {
  // dim0: positive 0,1,2,3,4  dim1: negative 10,8,6,4,2,0... wait, end exclusive
  // dim1: beg=10, end=0, step=-2 -> 10,8,6,4,2  (5 elements)
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 5,   1),
    tf::IndexRange<int>(10, 0, -2)
  );
  size_t N = r.size();  // 5*5 = 25
  REQUIRE(N == 25);

  for (size_t cs : {1, 3, 5, 7, 13, 25}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_2d_signed(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.upper_slice.full_coverage.3d_all_negative") {
  // 3x4x5 in descending order on all axes
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(3,  0, -1),
    tf::IndexRange<int>(8,  0, -2),
    tf::IndexRange<int>(15, 0, -3)
  );
  size_t N = r.size();  // 3*4*5 = 60
  REQUIRE(N == 60);

  for (size_t cs : {1, 4, 5, 12, 30, 60}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_3d_signed(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.upper_slice.full_coverage.3d_mixed_steps") {
  // dim0 positive, dim1 negative, dim2 positive
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0,  4,  1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(0,  6,  1)
  );
  size_t N = r.size();  // 4*5*6 = 120
  REQUIRE(N == 120);

  for (size_t cs : {1, 6, 10, 30, 60, 120}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      collect_flat_3d_signed(box, r, visited);
      cursor += consumed;
    }
    REQUIRE(visited.size() == N);
    std::sort(visited.begin(), visited.end());
    for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
  }
}

TEST_CASE("NegativeStep.upper_slice.full_coverage.exhaustive_2d") {
  // Sweep all sign combinations and small dims exhaustively
  for (int s0 : {1, -1, 2, -2}) {
    for (int s1 : {1, -1, 2, -2}) {
      for (int d0 = 1; d0 <= 5; d0++) {
        for (int d1 = 1; d1 <= 5; d1++) {
          int beg0 = (s0 > 0) ? 0          : d0 * (-s0);
          int end0 = (s0 > 0) ? d0 * s0    : 0;
          int beg1 = (s1 > 0) ? 0          : d1 * (-s1);
          int end1 = (s1 > 0) ? d1 * s1    : 0;

          tf::IndexRanges<int, 2> r(
            tf::IndexRange<int>(beg0, end0, s0),
            tf::IndexRange<int>(beg1, end1, s1)
          );
          size_t N = r.size();

          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto box = r.upper_slice(cursor, cs); size_t consumed = box.size();
              REQUIRE(consumed >= 1);
              REQUIRE(consumed == box.size());
              collect_flat_2d_signed(box, r, visited);
              cursor += consumed;
            }
            REQUIRE(visited.size() == N);
            std::sort(visited.begin(), visited.end());
            for (size_t k = 0; k < N; k++) REQUIRE(visited[k] == k);
          }
        }
      }
    }
  }
}

// ============================================================================
// Section 18d: upper_slice — trailing-zeros rule with negative steps
// The orthogonality constraint operates in logical position space and must
// behave identically regardless of step sign.
// ============================================================================

TEST_CASE("NegativeStep.upper_slice.trailing_zeros.inner_mid_row") {
  // dim0: 6,4,2  (beg=6, step=-2, size=3)
  // dim1: 9,6,3  (beg=9, step=-3, size=3)
  // flat 1 -> coords (0, 1): dim1 at pos 1 != 0, so outer dim must be locked
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  // flat=1: coords (0,1) -> dim1 at pos 1, 2 elements remain in the row (pos 1 and 2)
  {
    auto box = r.upper_slice(1, 10); size_t consumed = box.size();
    REQUIRE(std::get<0>(box.dim(0)) == 6); REQUIRE(std::get<1>(box.dim(0)) == 4);  // locked to row 0
    REQUIRE(std::get<0>(box.dim(1)) == 6);  // col pos 1 -> value 9 + 1*(-3) = 6
    REQUIRE(std::get<1>(box.dim(1))   == 0);  // runs to end of dim1
    REQUIRE(consumed == 2);  // pos 1 and pos 2 remain in this row
  }

  // flat=2: coords (0,2) -> last element in the row, exactly 1 left
  {
    auto box = r.upper_slice(2, 10); size_t consumed = box.size();
    REQUIRE(consumed == 1);
    REQUIRE(std::get<0>(box.dim(0)) == 6); REQUIRE(std::get<1>(box.dim(0)) == 4);  // locked to row 0
    REQUIRE(std::get<0>(box.dim(1)) == 3); REQUIRE(std::get<1>(box.dim(1)) == 0);  // col pos 2 -> value 3
  }

  // flat=3: coords (1,0) -> col at 0, can grow into dim0
  // Budget 9 >= 2 full rows (2*3=6), consumes both remaining rows
  {
    auto box = r.upper_slice(3, 9); size_t consumed = box.size();
    REQUIRE(consumed == 6);
    REQUIRE(std::get<0>(box.dim(0)) == 4); REQUIRE(std::get<1>(box.dim(0)) == 0);  // rows 1 and 2: values 4,2
    REQUIRE(std::get<0>(box.dim(1)) == 9); REQUIRE(std::get<1>(box.dim(1)) == 0);  // full dim1 extent
  }
}

TEST_CASE("NegativeStep.upper_slice.trailing_zeros.3d_mixed") {
  // dim0 positive, dim1 and dim2 negative
  // dim0: 0,1,2  dim1: 4,2 (beg=4,step=-2,size=2)  dim2: 6,3 (beg=6,step=-3,size=2)
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0,  3,  1),
    tf::IndexRange<int>(4,  0, -2),
    tf::IndexRange<int>(6,  0, -3)
  );
  size_t N = r.size();  // 3*2*2 = 12
  REQUIRE(N == 12);

  // flat=1 -> coords (0,0,1): dim2 at pos 1, outer dims locked
  {
    auto box = r.upper_slice(1, 100); size_t consumed = box.size();
    REQUIRE(consumed == 1);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
    REQUIRE(std::get<0>(box.dim(1)) == 4); REQUIRE(std::get<1>(box.dim(1)) == 2);
    REQUIRE(std::get<0>(box.dim(2)) == 3); REQUIRE(std::get<1>(box.dim(2)) == 0);
  }

  // flat=2 -> coords (0,1,0): dim2 at 0 but dim1 at 1 != 0, grow_dim = 1
  {
    auto box = r.upper_slice(2, 100); size_t consumed = box.size();
    // dim1 has 1 step left (pos 1 -> value 2), inner vol = dim2.size() = 2
    // budget 100 >> 2 -> takes the remaining 1 step on dim1 = 2 elements
    REQUIRE(consumed == 2);
    REQUIRE(std::get<0>(box.dim(0)) == 0); REQUIRE(std::get<1>(box.dim(0)) == 1);
  }

  // flat=4 -> coords (1,0,0): all inner dims at 0, can grow into dim0
  {
    auto box = r.upper_slice(4, 100); size_t consumed = box.size();
    // 2 rows left (rows 1 and 2), each 2*2=4, budget 100 -> takes both = 8
    REQUIRE(consumed == 8);
    REQUIRE(std::get<0>(box.dim(0)) == 1); REQUIRE(std::get<1>(box.dim(0)) == 3);
    REQUIRE(std::get<0>(box.dim(1)) == 4); REQUIRE(std::get<1>(box.dim(1)) == 0);
    REQUIRE(std::get<0>(box.dim(2)) == 6); REQUIRE(std::get<1>(box.dim(2)) == 0);
  }
}

// ============================================================================
// Section 18e: box step_size must always match the original dim's step_size
// ============================================================================

TEST_CASE("NegativeStep.upper_slice.step_size_preserved") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0,  4,  1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto box = r.upper_slice(flat, 7);
    REQUIRE(std::get<2>(box.dim(0)) ==  1);
    REQUIRE(std::get<2>(box.dim(1)) == -2);
    REQUIRE(std::get<2>(box.dim(2)) == -3);
  }
}
