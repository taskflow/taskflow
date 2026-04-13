#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/utility/iterator.hpp>

// ============================================================================
// Helpers
// ============================================================================

// Enumerate every element produced by an IndexRange<T,1> via explicit loop.
template <typename T>
std::vector<T> enumerate_1d(const tf::IndexRange<T, 1>& r) {
  std::vector<T> out;
  if (r.step_size() > T{0}) {
    for (T v = r.begin(); v < r.end(); v += r.step_size()) out.push_back(v);
  } else {
    for (T v = r.begin(); v > r.end(); v += r.step_size()) out.push_back(v);
  }
  return out;
}

// Enumerate every flat index produced by an IndexRange<T,2> in row-major order.
// Works for both positive and negative step sizes.
template <typename T>
std::vector<std::pair<T,T>> enumerate_2d(const tf::IndexRange<T, 2>& r) {
  std::vector<std::pair<T,T>> out;
  auto& d0 = r.dim(0); auto& d1 = r.dim(1);
  auto in_range = [](T v, T end, T step) {
    return step > T{0} ? v < end : v > end;
  };
  for (T i = d0.begin(); in_range(i, d0.end(), d0.step_size()); i += d0.step_size())
    for (T j = d1.begin(); in_range(j, d1.end(), d1.step_size()); j += d1.step_size())
      out.push_back({i, j});
  return out;
}

// Enumerate every flat index produced by an IndexRange<T,3> in row-major order.
// Works for both positive and negative step sizes.
template <typename T>
std::vector<std::tuple<T,T,T>> enumerate_3d(const tf::IndexRange<T, 3>& r) {
  std::vector<std::tuple<T,T,T>> out;
  auto& d0 = r.dim(0); auto& d1 = r.dim(1); auto& d2 = r.dim(2);
  auto in_range = [](T v, T end, T step) {
    return step > T{0} ? v < end : v > end;
  };
  for (T i = d0.begin(); in_range(i, d0.end(), d0.step_size()); i += d0.step_size())
    for (T j = d1.begin(); in_range(j, d1.end(), d1.step_size()); j += d1.step_size())
      for (T k = d2.begin(); in_range(k, d2.end(), d2.step_size()); k += d2.step_size())
        out.push_back({i, j, k});
  return out;
}

// Drive consume_chunk over the full flat space of an ND range and collect every
// element visited.  Returns the multiset of flat indices (encoded as size_t)
// so the caller can verify exactly-once coverage.
// Works for any N by having the caller supply a "visit" lambda that converts
// box -> flat indices and appends them.
template <typename R, typename VisitFn>
std::vector<size_t> drain_consume_chunk(const R& range, size_t chunk_size, VisitFn visit) {
  size_t N = range.size();
  std::vector<size_t> visited;
  size_t cursor = 0;
  while (cursor < N) {
    auto [box, consumed] = range.consume_chunk(cursor, chunk_size);
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
  // distance() and IndexRange<T,1>::size() must agree for all valid ranges
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
// Section 3: IndexRange<T,1> — construction & accessors
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
  tf::IndexRange r(0, 10, 2);  // deduction guide -> IndexRange<int, 1>
  static_assert(std::is_same_v<decltype(r), tf::IndexRange<int, 1>>);
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
  static_assert(tf::IndexRange<int, 1>::rank == 1);
  static_assert(tf::IndexRange<int>::rank    == 1);
}

// ============================================================================
// Section 4: IndexRange<T,1>::unravel
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
// Section 5: IndexRange<T,N> — construction, size, rank
// ============================================================================

TEST_CASE("IndexRangeND.construction_2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  REQUIRE(r.rank     == 2);
  REQUIRE(r.size(0)  == 4);
  REQUIRE(r.size(1)  == 6);
  REQUIRE(r.size()   == 24);
}

TEST_CASE("IndexRangeND.construction_3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  REQUIRE(r.size() == 120);
}

TEST_CASE("IndexRangeND.construction_from_array") {
  std::array<tf::IndexRange<int,1>, 2> dims = {
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  };
  tf::IndexRange<int, 2> r(dims);
  REQUIRE(r.size() == 12);
}

TEST_CASE("IndexRangeND.dim_accessor") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(2, 8, 2),
    tf::IndexRange<int>(1, 7, 3)
  );
  REQUIRE(r.dim(0).begin()     == 2);
  REQUIRE(r.dim(0).step_size() == 2);
  REQUIRE(r.dim(0).size()      == 3);
  REQUIRE(r.dim(1).begin()     == 1);
  REQUIRE(r.dim(1).step_size() == 3);
  REQUIRE(r.dim(1).size()      == 2);
  REQUIRE(r.size()             == 6);
}

TEST_CASE("IndexRangeND.non_unit_steps_size") {
  // 3D: steps 1, 2, 3
  tf::IndexRange<int, 3> r(
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
  static_assert(tf::IndexRange<int, 2>::rank == 2);
  static_assert(tf::IndexRange<int, 3>::rank == 3);
  static_assert(tf::IndexRange<int, 4>::rank == 4);
}

// ============================================================================
// Section 6: IndexRange<T,N>::coords
// ============================================================================

TEST_CASE("IndexRangeND.coords.2d_unit_step") {
  // 3x4 range, unit steps
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  // row-major: flat 0 -> (0,0), flat 5 -> (1,1), flat 11 -> (2,3)
  auto c0  = r.coords(0);  REQUIRE(c0[0] == 0); REQUIRE(c0[1] == 0);
  auto c5  = r.coords(5);  REQUIRE(c5[0] == 1); REQUIRE(c5[1] == 1);
  auto c11 = r.coords(11); REQUIRE(c11[0] == 2); REQUIRE(c11[1] == 3);
}

TEST_CASE("IndexRangeND.coords.2d_non_unit_step") {
  // dim0: 0,2,4 (size 3); dim1: 1,4 (size 2)
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 6, 2),
    tf::IndexRange<int>(1, 7, 3)
  );
  // flat 0 -> (0,1), flat 1 -> (0,4), flat 2 -> (2,1), flat 5 -> (4,4)
  auto c0 = r.coords(0); REQUIRE(c0[0] == 0); REQUIRE(c0[1] == 1);
  auto c1 = r.coords(1); REQUIRE(c1[0] == 0); REQUIRE(c1[1] == 4);
  auto c2 = r.coords(2); REQUIRE(c2[0] == 2); REQUIRE(c2[1] == 1);
  auto c5 = r.coords(5); REQUIRE(c5[0] == 4); REQUIRE(c5[1] == 4);
}

TEST_CASE("IndexRangeND.coords.3d_roundtrip") {
  // For every flat index, coords() should recover the element that enumerate_3d
  // produces at that position.
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  auto all = enumerate_3d(r);
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == std::get<0>(all[flat]));
    REQUIRE(c[1] == std::get<1>(all[flat]));
    REQUIRE(c[2] == std::get<2>(all[flat]));
  }
}

TEST_CASE("IndexRangeND.coords.exhaustive_2d") {
  // All 2D sizes up to 5x5 with steps 1 and 2
  for (int si = 1; si <= 2; si++) {
    for (int sj = 1; sj <= 2; sj++) {
      for (int di = 1; di <= 5; di++) {
        for (int dj = 1; dj <= 5; dj++) {
          tf::IndexRange<int, 2> r(
            tf::IndexRange<int>(0, di * si, si),
            tf::IndexRange<int>(0, dj * sj, sj)
          );
          auto all = enumerate_2d(r);
          REQUIRE(all.size() == r.size());
          for (size_t flat = 0; flat < all.size(); flat++) {
            auto c = r.coords(flat);
            REQUIRE(c[0] == all[flat].first);
            REQUIRE(c[1] == all[flat].second);
          }
        }
      }
    }
  }
}

// ============================================================================
// Section 7: consume_chunk — documented examples from the header
// ============================================================================

TEST_CASE("consume_chunk.documented_examples") {
  // 3D range: 4 x 5 x 10 (from the header docstring)
  tf::IndexRange<int, 3> range(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(0, 5,  1),
    tf::IndexRange<int>(0, 10, 1)
  );

  // Scenario 1: flat_beg=0, requested=30
  // Coords (0,0,0). inner_volume reaches 50 at dim-0 (first >= 30), so grow_dim=0,
  // steps_to_take=1. The box overshoots to the next orthogonal boundary.
  // box is [0,1) x [0,5) x [0,10), consumed=50
  {
    auto [box, consumed] = range.consume_chunk(0, 30);
    REQUIRE(consumed == 50);
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
    REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 5);
    REQUIRE(box.dim(2).begin() == 0); REQUIRE(box.dim(2).end() == 10);
    REQUIRE(box.size() == 50);
  }

  // Scenario 2: flat_beg=30, requested=30
  // Coords (0,3,0). coords[1]=3 != 0, so trailing-zeros fires at d=0.
  // grow_dim=1, active=10. steps_left=2, steps_needed=3, steps_to_take=2.
  // box is [0,1) x [3,5) x [0,10), consumed=20 (geometry-constrained, < requested)
  {
    auto [box, consumed] = range.consume_chunk(30, 30);
    REQUIRE(consumed == 20);
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
    REQUIRE(box.dim(1).begin() == 3); REQUIRE(box.dim(1).end() == 5);
    REQUIRE(box.dim(2).begin() == 0); REQUIRE(box.dim(2).end() == 10);
    REQUIRE(box.size() == 20);
  }

  // Scenario 3: flat_beg=55, requested=30
  // Coords (1,0,5). coords[2]=5 != 0, trailing-zeros fires at d=1.
  // grow_dim=2, active=1. steps_left=5, steps_needed=30, steps_to_take=5.
  // box is [1,2) x [0,1) x [5,10), consumed=5 (geometry-constrained, < requested)
  {
    auto [box, consumed] = range.consume_chunk(55, 30);
    REQUIRE(consumed == 5);
    REQUIRE(box.dim(0).begin() == 1); REQUIRE(box.dim(0).end() == 2);
    REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 1);
    REQUIRE(box.dim(2).begin() == 5); REQUIRE(box.dim(2).end() == 10);
    REQUIRE(box.size() == 5);
  }

  // Scenario 4: exact fit — requested equals a natural dimension boundary
  // flat_beg=0, requested=10. inner_volume reaches 10 at d=1 (10>=10), stops there.
  // grow_dim=1, active=10. steps_left=5, steps_needed=1, steps_to_take=1.
  // box is [0,1) x [0,1) x [0,10), consumed=10 (no overshoot needed)
  {
    auto [box, consumed] = range.consume_chunk(0, 10);
    REQUIRE(consumed == 10);
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
    REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 1);
    REQUIRE(box.dim(2).begin() == 0); REQUIRE(box.dim(2).end() == 10);
    REQUIRE(box.size() == 10);
  }
}

// ============================================================================
// Section 8: consume_chunk — zero requested_size
// ============================================================================

TEST_CASE("consume_chunk.zero_requested_size") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  auto [box, consumed] = r.consume_chunk(0, 0);
  REQUIRE(consumed == 0);
}

// ============================================================================
// Section 9: consume_chunk — forward progress guarantee
// For any chunk size >= 1 and any flat_beg < N, consumed must be >= 1.
// ============================================================================

TEST_CASE("consume_chunk.forward_progress.2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 7, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 7, 13, 50}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
    }
  }
}

TEST_CASE("consume_chunk.forward_progress.3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 5, 20, 60}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
    }
  }
}

// ============================================================================
// Section 10: consume_chunk — box is an orthogonal sub-box of the original
// The sub-box must not exceed any dimension's bounds.
// ============================================================================

TEST_CASE("consume_chunk.box_within_bounds.2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 6, 2),
    tf::IndexRange<int>(0, 9, 3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 6}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      for (size_t d = 0; d < 2; d++) {
        REQUIRE(box.dim(d).begin()     >= r.dim(d).begin());
        REQUIRE(box.dim(d).end()       <= r.dim(d).end());
        REQUIRE(box.dim(d).step_size() == r.dim(d).step_size());
      }
      REQUIRE(box.size() == consumed);
    }
  }
}

TEST_CASE("consume_chunk.box_within_bounds.3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 5, 10, 30, 120}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      for (size_t d = 0; d < 3; d++) {
        REQUIRE(box.dim(d).begin()     >= r.dim(d).begin());
        REQUIRE(box.dim(d).end()       <= r.dim(d).end());
        REQUIRE(box.dim(d).step_size() == r.dim(d).step_size());
      }
      REQUIRE(box.size() == consumed);
    }
  }
}

// ============================================================================
// Section 11: consume_chunk — full coverage (the critical invariant)
// Draining with consume_chunk must visit every element exactly once.
// ============================================================================

// Helper: collect flat indices from a 2D box
auto collect_flat_2d(const tf::IndexRange<int, 2>& box,
                     const tf::IndexRange<int, 2>& full,
                     std::vector<size_t>& out) {
  size_t D1 = full.size(1);
  int S0 = full.dim(0).step_size(), S1 = full.dim(1).step_size();
  int B0 = full.dim(0).begin(),     B1 = full.dim(1).begin();
  auto pos = [](int v, int beg, int step) -> size_t {
    return static_cast<size_t>((v - beg) / step);
  };
  auto in_range = [](int v, int end, int step) {
    return step > 0 ? v < end : v > end;
  };
  for (int i = box.dim(0).begin(); in_range(i, box.dim(0).end(), S0); i += S0)
    for (int j = box.dim(1).begin(); in_range(j, box.dim(1).end(), S1); j += S1)
      out.push_back(pos(i, B0, S0) * D1 + pos(j, B1, S1));
}

auto collect_flat_3d(const tf::IndexRange<int, 3>& box,
                     const tf::IndexRange<int, 3>& full,
                     std::vector<size_t>& out) {
  size_t D1 = full.size(1), D2 = full.size(2);
  int S0 = full.dim(0).step_size(), S1 = full.dim(1).step_size(), S2 = full.dim(2).step_size();
  int B0 = full.dim(0).begin(),     B1 = full.dim(1).begin(),     B2 = full.dim(2).begin();
  auto pos = [](int v, int beg, int step) -> size_t {
    return static_cast<size_t>((v - beg) / step);
  };
  auto in_range = [](int v, int end, int step) {
    return step > 0 ? v < end : v > end;
  };
  for (int i = box.dim(0).begin(); in_range(i, box.dim(0).end(), S0); i += S0)
    for (int j = box.dim(1).begin(); in_range(j, box.dim(1).end(), S1); j += S1)
      for (int k = box.dim(2).begin(); in_range(k, box.dim(2).end(), S2); k += S2)
        out.push_back(pos(i, B0, S0) * (D1 * D2) + pos(j, B1, S1) * D2 + pos(k, B2, S2));
}

TEST_CASE("consume_chunk.full_coverage.2d") {
  // Sweep multiple chunk sizes and range shapes
  for (int di : {1, 2, 3, 5, 7, 12}) {
    for (int dj : {1, 2, 3, 5, 7, 12}) {
      tf::IndexRange<int, 2> r(
        tf::IndexRange<int>(0, di, 1),
        tf::IndexRange<int>(0, dj, 1)
      );
      size_t N = r.size();
      for (size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, size_t{13}, N/2 + 1, N}) {
        if (cs == 0) cs = 1;
        std::vector<size_t> visited;
        size_t cursor = 0;
        while (cursor < N) {
          auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("consume_chunk.full_coverage.3d") {
  for (int di : {1, 3, 5}) {
    for (int dj : {1, 4, 6}) {
      for (int dk : {1, 2, 7}) {
        tf::IndexRange<int, 3> r(
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
            auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("consume_chunk.full_coverage.non_unit_steps_2d") {
  // step sizes > 1 — the flat index is the logical position, not the raw value
  for (int si : {1, 2, 3}) {
    for (int sj : {1, 2, 3}) {
      for (int di : {1, 3, 5}) {
        for (int dj : {1, 3, 5}) {
          tf::IndexRange<int, 2> r(
            tf::IndexRange<int>(0, di * si, si),
            tf::IndexRange<int>(0, dj * sj, sj)
          );
          size_t N = r.size();
          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("consume_chunk.full_coverage.non_unit_steps_3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4*1, 1),
    tf::IndexRange<int>(0, 5*2, 2),
    tf::IndexRange<int>(0, 6*3, 3)
  );
  size_t N = r.size();   // 4*5*6 = 120
  for (size_t cs : {1, 4, 7, 13, 30, 60, 120}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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
// Section 12: consume_chunk — chunk_size >= N must consume everything at once
// ============================================================================

TEST_CASE("consume_chunk.oversized_chunk_2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  size_t N = r.size();  // 12
  auto [box, consumed] = r.consume_chunk(0, N * 10);
  REQUIRE(consumed == N);
  REQUIRE(box.size() == N);
  REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 3);
  REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 4);
}

TEST_CASE("consume_chunk.oversized_chunk_3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 2, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1)
  );
  size_t N = r.size();  // 24
  auto [box, consumed] = r.consume_chunk(0, 9999);
  REQUIRE(consumed == N);
  REQUIRE(box.size() == N);
}

// ============================================================================
// Section 13: consume_chunk — chunk_size=1 always produces unit boxes
// ============================================================================

TEST_CASE("consume_chunk.unit_chunk_2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto [box, consumed] = r.consume_chunk(flat, 1);
    REQUIRE(consumed == 1);
    REQUIRE(box.size() == 1);
  }
}

TEST_CASE("consume_chunk.unit_chunk_3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto [box, consumed] = r.consume_chunk(flat, 1);
    REQUIRE(consumed == 1);
    REQUIRE(box.size() == 1);
  }
}

// ============================================================================
// Section 14: consume_chunk — trailing-zeros rule (orthogonality invariant)
// When an inner dimension is not at coordinate 0, grow_dim must be the
// innermost (N-1), which means outer dimensions are locked.
// ============================================================================

TEST_CASE("consume_chunk.trailing_zeros.inner_not_at_zero") {
  // 2D: 4x6, unit steps
  // flat_beg=3 -> coords (0, 3): dim1 != 0, so grow_dim must be 1
  // Requesting 10 should be capped to the remaining 3 elements in the row.
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  {
    auto [box, consumed] = r.consume_chunk(3, 10);
    // coords: (0, 3); must stay on row 0, innermost grows
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
    REQUIRE(box.dim(1).begin() == 3);
    REQUIRE(consumed == 3);  // only 3 elements left in the row
  }

  // flat_beg=9 -> coords (1, 3): same constraint on row 1
  {
    auto [box, consumed] = r.consume_chunk(9, 10);
    REQUIRE(box.dim(0).begin() == 1); REQUIRE(box.dim(0).end() == 2);
    REQUIRE(box.dim(1).begin() == 3);
    REQUIRE(consumed == 3);
  }
}

TEST_CASE("consume_chunk.trailing_zeros.3d_mid_row") {
  // 3D: 3x4x5; flat_beg at (0,0,3) -> dim2 != 0, outer dims locked
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t flat = 0*20 + 0*5 + 3;   // coords (0,0,3)
  auto [box, consumed] = r.consume_chunk(flat, 100);
  // inner row incomplete: only 2 elements left (indices 3,4)
  REQUIRE(consumed == 2);
  REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
  REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 1);
  REQUIRE(box.dim(2).begin() == 3); REQUIRE(box.dim(2).end() == 5);
}

TEST_CASE("consume_chunk.trailing_zeros.3d_clean_boundary") {
  // When coords are (row, 0, 0), can grow dim 0 if budget allows
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  size_t flat = 1*20 + 0*5 + 0;   // coords (1,0,0)
  auto [box, consumed] = r.consume_chunk(flat, 40);
  // Budget 40 >= one full row slice (20 elements) -> can grow dim 0
  REQUIRE(consumed == 40);
  REQUIRE(box.dim(0).begin() == 1); REQUIRE(box.dim(0).end() == 3);
  REQUIRE(box.dim(1).begin() == 0); REQUIRE(box.dim(1).end() == 4);
  REQUIRE(box.dim(2).begin() == 0); REQUIRE(box.dim(2).end() == 5);
}

// ============================================================================
// Section 15: consume_chunk — consumed == box.size() always
// ============================================================================

TEST_CASE("consume_chunk.consumed_equals_box_size.2d") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 6, 2),
    tf::IndexRange<int>(0, 9, 3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {1, 2, 3, 6, 18}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed == box.size());
    }
  }
}

TEST_CASE("consume_chunk.consumed_equals_box_size.3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat += 7) {   // stride to keep test fast
    for (size_t cs : {1, 6, 12, 30, 120}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed == box.size());
    }
  }
}

// ============================================================================
// Section 16: consume_chunk — 4D sanity check
// ============================================================================

TEST_CASE("consume_chunk.full_coverage.4d") {
  tf::IndexRange<int, 4> r(
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
      auto [box, consumed] = r.consume_chunk(cursor, cs);
      REQUIRE(consumed >= 1);
      REQUIRE(consumed == box.size());
      // Enumerate box elements and mark visited
      size_t D1 = r.size(1), D2 = r.size(2), D3 = r.size(3);
      for (int a = box.dim(0).begin(); a < box.dim(0).end(); a++)
        for (int b = box.dim(1).begin(); b < box.dim(1).end(); b++)
          for (int c = box.dim(2).begin(); c < box.dim(2).end(); c++)
            for (int d = box.dim(3).begin(); d < box.dim(3).end(); d++)
              visited[a*D1*D2*D3 + b*D2*D3 + c*D3 + d]++;
      cursor += consumed;
    }
    for (int v : visited) REQUIRE(v == 1);
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
  tf::IndexRange<int, 2> r(
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
  tf::IndexRange<int, 2> r(
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
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  REQUIRE(r.size() == 9);
}

TEST_CASE("NegativeStep.2D.coords_roundtrip") {
  // dim0 negative, dim1 positive
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(0,  4,  1)
  );
  auto all = enumerate_2d(r);
  REQUIRE(all.size() == r.size());
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == all[flat].first);
    REQUIRE(c[1] == all[flat].second);
  }
}

TEST_CASE("NegativeStep.2D.both_negative.coords_roundtrip") {
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  auto all = enumerate_2d(r);
  REQUIRE(all.size() == r.size());
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == all[flat].first);
    REQUIRE(c[1] == all[flat].second);
  }
}

// consume_chunk full-coverage with negative steps
TEST_CASE("NegativeStep.2D.dim0_negative.full_coverage") {
  // dim0: 10 down to 0, step -2 (5 elements)
  // dim1: 0 up to 6,   step  1 (6 elements)
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(0,  6,  1)
  );
  size_t N = r.size();
  for (size_t cs : {1, 3, 6, 10, 30}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(9, 0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 2, 3, 6, 12}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 2, 3, 5, 9}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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

          tf::IndexRange<int, 2> r(
            tf::IndexRange<int>(beg_i, end_i, si),
            tf::IndexRange<int>(beg_j, end_j, sj)
          );
          REQUIRE(r.size() == static_cast<size_t>(di * dj));

          size_t N = r.size();
          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto [box, consumed] = r.consume_chunk(cursor, cs);
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
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  REQUIRE(r.size(0) == 4);
  REQUIRE(r.size(1) == 5);
  REQUIRE(r.size(2) == 3);
  REQUIRE(r.size()  == 60);
}

TEST_CASE("NegativeStep.3D.all_negative.coords_roundtrip") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  auto all = enumerate_3d(r);
  REQUIRE(all.size() == r.size());
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == std::get<0>(all[flat]));
    REQUIRE(c[1] == std::get<1>(all[flat]));
    REQUIRE(c[2] == std::get<2>(all[flat]));
  }
}

TEST_CASE("NegativeStep.3D.all_negative.full_coverage") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t cs : {1, 3, 5, 15, 60}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0, 3,   1),
    tf::IndexRange<int>(8, 0,  -2),
    tf::IndexRange<int>(0, 5,   1)
  );
  size_t N = r.size();   // 3*4*5 = 60
  for (size_t cs : {1, 4, 7, 30, 60}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    for (size_t cs : {size_t{1}, size_t{2}, size_t{5}, N}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
      REQUIRE(consumed == box.size());
    }
  }
}

TEST_CASE("NegativeStep.forward_progress.3d") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat += 5) {  // stride to keep test fast
    for (size_t cs : {1, 3, 15, 60}) {
      auto [box, consumed] = r.consume_chunk(flat, cs);
      REQUIRE(consumed >= 1);
      REQUIRE(flat + consumed <= N);
      REQUIRE(consumed == box.size());
    }
  }
}

// ============================================================================
// Section 19: concepts
// ============================================================================

TEST_CASE("Concepts.IndexRangeLike") {  // Section 19
  static_assert( tf::IndexRangeLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRangeLike<tf::IndexRange<int, 1>>);
  static_assert( tf::IndexRangeLike<tf::IndexRange<int, 2>>);
  static_assert( tf::IndexRangeLike<tf::IndexRange<int, 3>>);
  static_assert(!tf::IndexRangeLike<int>);
  static_assert(!tf::IndexRangeLike<std::vector<int>>);
  // cv-ref variants
  static_assert( tf::IndexRangeLike<const tf::IndexRange<int>&>);
  static_assert( tf::IndexRangeLike<tf::IndexRange<int, 2>&&>);
  REQUIRE(true);  // static_asserts above are the real test
}

TEST_CASE("Concepts.IndexRange1DLike") {
  static_assert( tf::IndexRange1DLike<tf::IndexRange<int>>);
  static_assert( tf::IndexRange1DLike<tf::IndexRange<int, 1>>);
  static_assert(!tf::IndexRange1DLike<tf::IndexRange<int, 2>>);
  static_assert(!tf::IndexRange1DLike<tf::IndexRange<int, 3>>);
  static_assert( tf::IndexRange1DLike<const tf::IndexRange<int>&>);
  REQUIRE(true);
}

TEST_CASE("Concepts.IndexRangeMDLike") {
  static_assert(!tf::IndexRangeMDLike<tf::IndexRange<int>>);
  static_assert(!tf::IndexRangeMDLike<tf::IndexRange<int, 1>>);
  static_assert( tf::IndexRangeMDLike<tf::IndexRange<int, 2>>);
  static_assert( tf::IndexRangeMDLike<tf::IndexRange<int, 3>>);
  static_assert( tf::IndexRangeMDLike<tf::IndexRange<int, 4>>);
  static_assert( tf::IndexRangeMDLike<const tf::IndexRange<int, 2>&>);
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
    const tf::IndexRange<int, 2>& box,
    const tf::IndexRange<int, 2>& full,
    std::vector<size_t>& out)
{
  size_t D1 = full.size(1);
  int s0 = full.dim(0).step_size(), b0 = full.dim(0).begin();
  int s1 = full.dim(1).step_size(), b1 = full.dim(1).begin();

  // step sign determines loop direction
  auto step0 = [&](int i){ return s0 > 0 ? i < box.dim(0).end() : i > box.dim(0).end(); };
  auto step1 = [&](int j){ return s1 > 0 ? j < box.dim(1).end() : j > box.dim(1).end(); };

  for (int i = box.dim(0).begin(); step0(i); i += s0) {
    size_t pi = static_cast<size_t>((i - b0) / s0);
    for (int j = box.dim(1).begin(); step1(j); j += s1) {
      size_t pj = static_cast<size_t>((j - b1) / s1);
      out.push_back(pi * D1 + pj);
    }
  }
}

auto collect_flat_3d_signed(
    const tf::IndexRange<int, 3>& box,
    const tf::IndexRange<int, 3>& full,
    std::vector<size_t>& out)
{
  size_t D1 = full.size(1), D2 = full.size(2);
  int s0 = full.dim(0).step_size(), b0 = full.dim(0).begin();
  int s1 = full.dim(1).step_size(), b1 = full.dim(1).begin();
  int s2 = full.dim(2).step_size(), b2 = full.dim(2).begin();

  auto cmp0 = [&](int i){ return s0 > 0 ? i < box.dim(0).end() : i > box.dim(0).end(); };
  auto cmp1 = [&](int j){ return s1 > 0 ? j < box.dim(1).end() : j > box.dim(1).end(); };
  auto cmp2 = [&](int k){ return s2 > 0 ? k < box.dim(2).end() : k > box.dim(2).end(); };

  for (int i = box.dim(0).begin(); cmp0(i); i += s0) {
    size_t pi = static_cast<size_t>((i - b0) / s0);
    for (int j = box.dim(1).begin(); cmp1(j); j += s1) {
      size_t pj = static_cast<size_t>((j - b1) / s1);
      for (int k = box.dim(2).begin(); cmp2(k); k += s2) {
        size_t pk = static_cast<size_t>((k - b2) / s2);
        out.push_back(pi * D1 * D2 + pj * D2 + pk);
      }
    }
  }
}

// ============================================================================
// Section 18a: IndexRange<T,1> — negative step basic properties
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
// Section 18b: coords() with negative step dimensions
// ============================================================================

TEST_CASE("NegativeStep.coords.2d_both_negative") {
  // dim0: 6,4,2 (beg=6, end=0, step=-2, size=3)
  // dim1: 9,6,3 (beg=9, end=0, step=-3, size=3)
  // Row-major flat order: (6,9),(6,6),(6,3),(4,9),(4,6),(4,3),(2,9),(2,6),(2,3)
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  REQUIRE(r.size() == 9);
  auto all = enumerate_2d(r);
  REQUIRE(all.size() == 9);
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == all[flat].first);
    REQUIRE(c[1] == all[flat].second);
  }
}

TEST_CASE("NegativeStep.coords.2d_mixed_steps") {
  // dim0: positive (0,2,4,6), dim1: negative (10,8,6,4,2)
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 8,  2),
    tf::IndexRange<int>(10, 0, -2)
  );
  REQUIRE(r.size() == 20);
  auto all = enumerate_2d(r);
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == all[flat].first);
    REQUIRE(c[1] == all[flat].second);
  }
}

TEST_CASE("NegativeStep.coords.3d_all_negative") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(4,  0, -1),
    tf::IndexRange<int>(6,  0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  // sizes: 4, 3, 3  -> 36 elements
  REQUIRE(r.size() == 36);
  auto all = enumerate_3d(r);
  REQUIRE(all.size() == 36);
  for (size_t flat = 0; flat < all.size(); flat++) {
    auto c = r.coords(flat);
    REQUIRE(c[0] == std::get<0>(all[flat]));
    REQUIRE(c[1] == std::get<1>(all[flat]));
    REQUIRE(c[2] == std::get<2>(all[flat]));
  }
}

TEST_CASE("NegativeStep.coords.exhaustive_2d_mixed") {
  // Sweep all sign combinations for step, small dims
  for (int s0 : {1, 2, -1, -2}) {
    for (int s1 : {1, 2, -1, -2}) {
      for (int d0 = 1; d0 <= 4; d0++) {
        for (int d1 = 1; d1 <= 4; d1++) {
          // construct begin/end so the range is always valid
          int beg0 = (s0 > 0) ? 0            : d0 * (-s0);
          int end0 = (s0 > 0) ? d0 * s0      : 0;
          int beg1 = (s1 > 0) ? 0            : d1 * (-s1);
          int end1 = (s1 > 0) ? d1 * s1      : 0;

          tf::IndexRange<int, 2> r(
            tf::IndexRange<int>(beg0, end0, s0),
            tf::IndexRange<int>(beg1, end1, s1)
          );
          REQUIRE(r.size() == static_cast<size_t>(d0 * d1));

          auto all = enumerate_2d(r);
          REQUIRE(all.size() == r.size());
          for (size_t flat = 0; flat < all.size(); flat++) {
            auto c = r.coords(flat);
            REQUIRE(c[0] == all[flat].first);
            REQUIRE(c[1] == all[flat].second);
          }
        }
      }
    }
  }
}

// ============================================================================
// Section 18c: consume_chunk full coverage — negative step dims
// ============================================================================

TEST_CASE("NegativeStep.consume_chunk.full_coverage.2d_both_negative") {
  // dim0: 10,8,6,4,2  dim1: 9,6,3  -> 5x3 = 15 elements
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>( 9, 0, -3)
  );
  size_t N = r.size();
  REQUIRE(N == 15);

  for (size_t cs : {1, 2, 3, 5, 8, 15}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("NegativeStep.consume_chunk.full_coverage.2d_mixed_steps") {
  // dim0: positive 0,1,2,3,4  dim1: negative 10,8,6,4,2,0... wait, end exclusive
  // dim1: beg=10, end=0, step=-2 -> 10,8,6,4,2  (5 elements)
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(0, 5,   1),
    tf::IndexRange<int>(10, 0, -2)
  );
  size_t N = r.size();  // 5*5 = 25
  REQUIRE(N == 25);

  for (size_t cs : {1, 3, 5, 7, 13, 25}) {
    std::vector<size_t> visited;
    size_t cursor = 0;
    while (cursor < N) {
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("NegativeStep.consume_chunk.full_coverage.3d_all_negative") {
  // 3x4x5 in descending order on all axes
  tf::IndexRange<int, 3> r(
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
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("NegativeStep.consume_chunk.full_coverage.3d_mixed_steps") {
  // dim0 positive, dim1 negative, dim2 positive
  tf::IndexRange<int, 3> r(
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
      auto [box, consumed] = r.consume_chunk(cursor, cs);
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

TEST_CASE("NegativeStep.consume_chunk.full_coverage.exhaustive_2d") {
  // Sweep all sign combinations and small dims exhaustively
  for (int s0 : {1, -1, 2, -2}) {
    for (int s1 : {1, -1, 2, -2}) {
      for (int d0 = 1; d0 <= 5; d0++) {
        for (int d1 = 1; d1 <= 5; d1++) {
          int beg0 = (s0 > 0) ? 0          : d0 * (-s0);
          int end0 = (s0 > 0) ? d0 * s0    : 0;
          int beg1 = (s1 > 0) ? 0          : d1 * (-s1);
          int end1 = (s1 > 0) ? d1 * s1    : 0;

          tf::IndexRange<int, 2> r(
            tf::IndexRange<int>(beg0, end0, s0),
            tf::IndexRange<int>(beg1, end1, s1)
          );
          size_t N = r.size();

          for (size_t cs : {size_t{1}, size_t{3}, N}) {
            std::vector<size_t> visited;
            size_t cursor = 0;
            while (cursor < N) {
              auto [box, consumed] = r.consume_chunk(cursor, cs);
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
// Section 18d: consume_chunk — trailing-zeros rule with negative steps
// The orthogonality constraint operates in logical position space and must
// behave identically regardless of step sign.
// ============================================================================

TEST_CASE("NegativeStep.consume_chunk.trailing_zeros.inner_mid_row") {
  // dim0: 6,4,2  (beg=6, step=-2, size=3)
  // dim1: 9,6,3  (beg=9, step=-3, size=3)
  // flat 1 -> coords (0, 1): dim1 at pos 1 != 0, so outer dim must be locked
  tf::IndexRange<int, 2> r(
    tf::IndexRange<int>(6, 0, -2),
    tf::IndexRange<int>(9, 0, -3)
  );
  // flat=1: coords (0,1) -> dim1 at pos 1, 2 elements remain in the row (pos 1 and 2)
  {
    auto [box, consumed] = r.consume_chunk(1, 10);
    REQUIRE(box.dim(0).begin() == 6); REQUIRE(box.dim(0).end() == 4);  // locked to row 0
    REQUIRE(box.dim(1).begin() == 6);  // col pos 1 -> value 9 + 1*(-3) = 6
    REQUIRE(box.dim(1).end()   == 0);  // runs to end of dim1
    REQUIRE(consumed == 2);  // pos 1 and pos 2 remain in this row
  }

  // flat=2: coords (0,2) -> last element in the row, exactly 1 left
  {
    auto [box, consumed] = r.consume_chunk(2, 10);
    REQUIRE(consumed == 1);
    REQUIRE(box.dim(0).begin() == 6); REQUIRE(box.dim(0).end() == 4);  // locked to row 0
    REQUIRE(box.dim(1).begin() == 3); REQUIRE(box.dim(1).end() == 0);  // col pos 2 -> value 3
  }

  // flat=3: coords (1,0) -> col at 0, can grow into dim0
  // Budget 9 >= 2 full rows (2*3=6), consumes both remaining rows
  {
    auto [box, consumed] = r.consume_chunk(3, 9);
    REQUIRE(consumed == 6);
    REQUIRE(box.dim(0).begin() == 4); REQUIRE(box.dim(0).end() == 0);  // rows 1 and 2: values 4,2
    REQUIRE(box.dim(1).begin() == 9); REQUIRE(box.dim(1).end() == 0);  // full dim1 extent
  }
}

TEST_CASE("NegativeStep.consume_chunk.trailing_zeros.3d_mixed") {
  // dim0 positive, dim1 and dim2 negative
  // dim0: 0,1,2  dim1: 4,2 (beg=4,step=-2,size=2)  dim2: 6,3 (beg=6,step=-3,size=2)
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0,  3,  1),
    tf::IndexRange<int>(4,  0, -2),
    tf::IndexRange<int>(6,  0, -3)
  );
  size_t N = r.size();  // 3*2*2 = 12
  REQUIRE(N == 12);

  // flat=1 -> coords (0,0,1): dim2 at pos 1, outer dims locked
  {
    auto [box, consumed] = r.consume_chunk(1, 100);
    REQUIRE(consumed == 1);
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
    REQUIRE(box.dim(1).begin() == 4); REQUIRE(box.dim(1).end() == 2);
    REQUIRE(box.dim(2).begin() == 3); REQUIRE(box.dim(2).end() == 0);
  }

  // flat=2 -> coords (0,1,0): dim2 at 0 but dim1 at 1 != 0, grow_dim = 1
  {
    auto [box, consumed] = r.consume_chunk(2, 100);
    // dim1 has 1 step left (pos 1 -> value 2), inner vol = dim2.size() = 2
    // budget 100 >> 2 -> takes the remaining 1 step on dim1 = 2 elements
    REQUIRE(consumed == 2);
    REQUIRE(box.dim(0).begin() == 0); REQUIRE(box.dim(0).end() == 1);
  }

  // flat=4 -> coords (1,0,0): all inner dims at 0, can grow into dim0
  {
    auto [box, consumed] = r.consume_chunk(4, 100);
    // 2 rows left (rows 1 and 2), each 2*2=4, budget 100 -> takes both = 8
    REQUIRE(consumed == 8);
    REQUIRE(box.dim(0).begin() == 1); REQUIRE(box.dim(0).end() == 3);
    REQUIRE(box.dim(1).begin() == 4); REQUIRE(box.dim(1).end() == 0);
    REQUIRE(box.dim(2).begin() == 6); REQUIRE(box.dim(2).end() == 0);
  }
}

// ============================================================================
// Section 18e: box step_size must always match the original dim's step_size
// ============================================================================

TEST_CASE("NegativeStep.consume_chunk.step_size_preserved") {
  tf::IndexRange<int, 3> r(
    tf::IndexRange<int>(0,  4,  1),
    tf::IndexRange<int>(10, 0, -2),
    tf::IndexRange<int>(9,  0, -3)
  );
  size_t N = r.size();
  for (size_t flat = 0; flat < N; flat++) {
    auto [box, consumed] = r.consume_chunk(flat, 7);
    REQUIRE(box.dim(0).step_size() ==  1);
    REQUIRE(box.dim(1).step_size() == -2);
    REQUIRE(box.dim(2).step_size() == -3);
  }
}
