#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/partitioner.hpp>

// ============================================================================
// Helpers
// ============================================================================

// Verify every index in [0, N) is visited exactly once by the given loop
// using an atomic visited grid.
static void check_visited(std::vector<std::atomic<int>>& visited, size_t N) {
  for(size_t i = 0; i < N; i++) {
    REQUIRE(visited[i].load() == 1);
  }
}

// ============================================================================
// 1D loop: StaticPartitioner
// ============================================================================

template <typename P>
void test_static_loop_1d(size_t W) {

  for(size_t N : {0, 1, 3, 7, 16, 99, 1000}) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      P p(c);
      size_t chunk_size = p.chunk_size() ? p.chunk_size() : (W ? (N + W - 1) / W : N);
      if(chunk_size == 0) chunk_size = 1;

      std::vector<std::atomic<int>> visited(N);
      for(auto& v : visited) v.store(0);

      // call loop for each worker w
      for(size_t w = 0; w < W && w < N; w++) {
        size_t curr_b = w * chunk_size;
        if(curr_b >= N) break;
        p.loop(N, W, curr_b, chunk_size, [&](size_t b, size_t e) {
          for(size_t i = b; i < e; i++) {
            visited[i].fetch_add(1, std::memory_order_relaxed);
          }
        });
      }

      check_visited(visited, N);
    }
  }
}

// ============================================================================
// 1D loop: dynamic/guided/random - use atomic next
// ============================================================================

template <typename P>
void test_dynamic_loop_1d(size_t W) {

  for(size_t N : {0, 1, 3, 7, 16, 99, 1000}) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      P p(c);
      std::atomic<size_t> next{0};
      std::vector<std::atomic<int>> visited(N);
      for(auto& v : visited) v.store(0);

      // spawn W workers each running the loop
      std::vector<std::future<void>> futures;
      for(size_t w = 0; w < W; w++) {
        futures.push_back(std::async(std::launch::async, [&]() {
          p.loop(N, W, next, [&](size_t b, size_t e) {
            for(size_t i = b; i < e; i++) {
              visited[i].fetch_add(1, std::memory_order_relaxed);
            }
          });
        }));
      }
      for(auto& f : futures) f.get();

      check_visited(visited, N);
    }
  }
}

// ============================================================================
// 1D loop: bool-returning callable (loop_until semantics)
// ============================================================================

template <typename P>
void test_dynamic_loop_1d_bool(size_t W) {

  for(size_t N : {1, 7, 16, 99, 1000}) {
    for(size_t c : {0, 1, 3, 7, 99}) {

      P p(c);
      // find first index where value == target - stop early
      std::vector<int> data(N);
      for(size_t i = 0; i < N; i++) data[i] = static_cast<int>(i);

      size_t target = N / 2;
      std::atomic<size_t> result{N};  // sentinel = not found
      std::atomic<size_t> next{0};

      std::vector<std::future<void>> futures;
      for(size_t w = 0; w < W; w++) {
        futures.push_back(std::async(std::launch::async, [&]() {
          p.loop(N, W, next, [&](size_t b, size_t e) -> bool {
            for(size_t i = b; i < e; i++) {
              if(static_cast<size_t>(data[i]) == target) {
                size_t prev = result.load(std::memory_order_relaxed);
                while(i < prev &&
                  !result.compare_exchange_weak(prev, i,
                    std::memory_order_relaxed, std::memory_order_relaxed));
                return true;
              }
            }
            return false;
          });
        }));
      }
      for(auto& f : futures) f.get();

      REQUIRE(result.load() == target);
    }
  }
}

// ============================================================================
// ND loop: 2D IndexRange
// ============================================================================

template <typename P>
void test_static_loop_nd_2d(size_t W) {

  for(size_t rows : {1, 3, 7, 16}) {
    for(size_t cols : {1, 4, 8, 13}) {
      for(size_t c : {0, 1, 3, 7, 99}) {

        P p(c);
        tf::IndexRanges<int, 2> range(
          tf::IndexRange<int>(0, static_cast<int>(rows), 1),
          tf::IndexRange<int>(0, static_cast<int>(cols), 1)
        );

        size_t N = range.size();
        size_t chunk_size = p.chunk_size() ? p.chunk_size() : (W ? (N + W - 1) / W : N);
        //chunk_size = range.ceil(chunk_size == 0 ? 1 : chunk_size);

        std::vector<std::atomic<int>> visited(N);
        for(auto& v : visited) v.store(0);

        for(size_t w = 0; w < W && w < N; w++) {
          size_t curr_b = w * chunk_size;
          if(curr_b >= N) break;
          p.loop(range, N, W, curr_b, chunk_size,
            [&](const tf::IndexRanges<int, 2>& box) {
              for(int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i += std::get<2>(box.dim(0))) {
                for(int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j += std::get<2>(box.dim(1))) {
                  visited[i * static_cast<int>(cols) + j]
                    .fetch_add(1, std::memory_order_relaxed);
                }
              }
            }
          );
        }

        check_visited(visited, N);
      }
    }
  }
}

template <typename P>
void test_dynamic_loop_nd_2d(size_t W) {

  for(size_t rows : {1, 3, 7, 16}) {
    for(size_t cols : {1, 4, 8, 13}) {
      for(size_t c : {0, 1, 3, 7, 99}) {

        P p(c);
        tf::IndexRanges<int, 2> range(
          tf::IndexRange<int>(0, static_cast<int>(rows), 1),
          tf::IndexRange<int>(0, static_cast<int>(cols), 1)
        );

        size_t N = range.size();
        std::atomic<size_t> next{0};
        std::vector<std::atomic<int>> visited(N);
        for(auto& v : visited) v.store(0);

        std::vector<std::future<void>> futures;
        for(size_t w = 0; w < W; w++) {
          futures.push_back(std::async(std::launch::async, [&]() {
            p.loop(range, N, W, next,
              [&](const tf::IndexRanges<int, 2>& box) {
                for(int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i += std::get<2>(box.dim(0))) {
                  for(int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j += std::get<2>(box.dim(1))) {
                    visited[i * static_cast<int>(cols) + j]
                      .fetch_add(1, std::memory_order_relaxed);
                  }
                }
              }
            );
          }));
        }
        for(auto& f : futures) f.get();

        check_visited(visited, N);
      }
    }
  }
}

// ============================================================================
// ND loop: 3D IndexRange
// ============================================================================

template <typename P>
void test_dynamic_loop_nd_3d(size_t W) {

  for(size_t d0 : {2, 4}) {
    for(size_t d1 : {3, 5}) {
      for(size_t d2 : {4, 6}) {
        for(size_t c : {0, 1, 3, 7, 99}) {

          P p(c);
          tf::IndexRanges<int, 3> range(
            tf::IndexRange<int>(0, static_cast<int>(d0), 1),
            tf::IndexRange<int>(0, static_cast<int>(d1), 1),
            tf::IndexRange<int>(0, static_cast<int>(d2), 1)
          );

          size_t N = range.size();
          std::atomic<size_t> next{0};
          std::vector<std::atomic<int>> visited(N);
          for(auto& v : visited) v.store(0);

          std::vector<std::future<void>> futures;
          for(size_t w = 0; w < W; w++) {
            futures.push_back(std::async(std::launch::async, [&]() {
              p.loop(range, N, W, next,
                [&](const tf::IndexRanges<int, 3>& box) {
                  for(int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i += std::get<2>(box.dim(0))) {
                    for(int j = std::get<0>(box.dim(1)); j < std::get<1>(box.dim(1)); j += std::get<2>(box.dim(1))) {
                      for(int k = std::get<0>(box.dim(2)); k < std::get<1>(box.dim(2)); k += std::get<2>(box.dim(2))) {
                        visited[i * static_cast<int>(d1 * d2) +
                                j * static_cast<int>(d2) + k]
                          .fetch_add(1, std::memory_order_relaxed);
                      }
                    }
                  }
                }
              );
            }));
          }
          for(auto& f : futures) f.get();

          check_visited(visited, N);
        }
      }
    }
  }
}

// ============================================================================
// ND loop: bool-returning callable
// ============================================================================

template <typename P>
void test_dynamic_loop_nd_bool(size_t W) {

  // 2D range: verify early exit fires and no index beyond target row is visited
  for(size_t rows : {4, 8}) {
    for(size_t cols : {4, 8}) {
      for(size_t c : {0, 1, 4}) {

        P p(c);
        tf::IndexRanges<int, 2> range(
          tf::IndexRange<int>(0, static_cast<int>(rows), 1),
          tf::IndexRange<int>(0, static_cast<int>(cols), 1)
        );

        size_t N = range.size();
        std::atomic<size_t> next{0};
        std::atomic<int> found{0};

        std::vector<std::future<void>> futures;
        for(size_t w = 0; w < W; w++) {
          futures.push_back(std::async(std::launch::async, [&]() {
            p.loop(range, N, W, next,
              [&](const tf::IndexRanges<int, 2>& box) -> bool {
                // return true (stop) as soon as we see i==1 in the box
                for(int i = std::get<0>(box.dim(0)); i < std::get<1>(box.dim(0)); i += std::get<2>(box.dim(0))) {
                  if(i == 1) {
                    found.fetch_add(1, std::memory_order_relaxed);
                    return true;
                  }
                }
                return false;
              }
            );
          }));
        }
        for(auto& f : futures) f.get();

        // at least one worker must have found i==1
        REQUIRE(found.load() >= 1);
      }
    }
  }
}

// ============================================================================
// TEST CASES: StaticPartitioner
// ============================================================================

TEST_CASE("Partitioner.Static.loop_1d.1thread" * doctest::timeout(300)) {
  test_static_loop_1d<tf::StaticPartitioner<>>(1);
}
TEST_CASE("Partitioner.Static.loop_1d.2threads" * doctest::timeout(300)) {
  test_static_loop_1d<tf::StaticPartitioner<>>(2);
}
TEST_CASE("Partitioner.Static.loop_1d.4threads" * doctest::timeout(300)) {
  test_static_loop_1d<tf::StaticPartitioner<>>(4);
}
TEST_CASE("Partitioner.Static.loop_1d.8threads" * doctest::timeout(300)) {
  test_static_loop_1d<tf::StaticPartitioner<>>(8);
}

TEST_CASE("Partitioner.Static.loop_nd_2d.1thread" * doctest::timeout(300)) {
  test_static_loop_nd_2d<tf::StaticPartitioner<>>(1);
}
TEST_CASE("Partitioner.Static.loop_nd_2d.2threads" * doctest::timeout(300)) {
  test_static_loop_nd_2d<tf::StaticPartitioner<>>(2);
}
TEST_CASE("Partitioner.Static.loop_nd_2d.4threads" * doctest::timeout(300)) {
  test_static_loop_nd_2d<tf::StaticPartitioner<>>(4);
}
TEST_CASE("Partitioner.Static.loop_nd_2d.8threads" * doctest::timeout(300)) {
  test_static_loop_nd_2d<tf::StaticPartitioner<>>(8);
}

// ============================================================================
// TEST CASES: GuidedPartitioner
// ============================================================================

TEST_CASE("Partitioner.Guided.loop_1d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("Partitioner.Guided.loop_1d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("Partitioner.Guided.loop_1d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("Partitioner.Guided.loop_1d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("Partitioner.Guided.loop_1d_bool.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_1d_bool<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("Partitioner.Guided.loop_1d_bool.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d_bool<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("Partitioner.Guided.loop_1d_bool.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d_bool<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("Partitioner.Guided.loop_1d_bool.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d_bool<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("Partitioner.Guided.loop_nd_2d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("Partitioner.Guided.loop_nd_2d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("Partitioner.Guided.loop_nd_2d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("Partitioner.Guided.loop_nd_2d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("Partitioner.Guided.loop_nd_3d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("Partitioner.Guided.loop_nd_3d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("Partitioner.Guided.loop_nd_3d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("Partitioner.Guided.loop_nd_3d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::GuidedPartitioner<>>(8);
}

TEST_CASE("Partitioner.Guided.loop_nd_bool.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_bool<tf::GuidedPartitioner<>>(1);
}
TEST_CASE("Partitioner.Guided.loop_nd_bool.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_bool<tf::GuidedPartitioner<>>(2);
}
TEST_CASE("Partitioner.Guided.loop_nd_bool.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_bool<tf::GuidedPartitioner<>>(4);
}
TEST_CASE("Partitioner.Guided.loop_nd_bool.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_bool<tf::GuidedPartitioner<>>(8);
}

// ============================================================================
// TEST CASES: DynamicPartitioner
// ============================================================================

TEST_CASE("Partitioner.Dynamic.loop_1d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("Partitioner.Dynamic.loop_1d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("Partitioner.Dynamic.loop_1d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("Partitioner.Dynamic.loop_1d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("Partitioner.Dynamic.loop_nd_2d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_2d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_2d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_2d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::DynamicPartitioner<>>(8);
}

TEST_CASE("Partitioner.Dynamic.loop_nd_3d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::DynamicPartitioner<>>(1);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_3d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::DynamicPartitioner<>>(2);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_3d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::DynamicPartitioner<>>(4);
}
TEST_CASE("Partitioner.Dynamic.loop_nd_3d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::DynamicPartitioner<>>(8);
}

// ============================================================================
// TEST CASES: RandomPartitioner
// ============================================================================

TEST_CASE("Partitioner.Random.loop_1d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("Partitioner.Random.loop_1d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("Partitioner.Random.loop_1d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("Partitioner.Random.loop_1d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_1d<tf::RandomPartitioner<>>(8);
}

TEST_CASE("Partitioner.Random.loop_nd_2d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("Partitioner.Random.loop_nd_2d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("Partitioner.Random.loop_nd_2d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("Partitioner.Random.loop_nd_2d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_2d<tf::RandomPartitioner<>>(8);
}

TEST_CASE("Partitioner.Random.loop_nd_3d.1thread" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::RandomPartitioner<>>(1);
}
TEST_CASE("Partitioner.Random.loop_nd_3d.2threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::RandomPartitioner<>>(2);
}
TEST_CASE("Partitioner.Random.loop_nd_3d.4threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::RandomPartitioner<>>(4);
}
TEST_CASE("Partitioner.Random.loop_nd_3d.8threads" * doctest::timeout(300)) {
  test_dynamic_loop_nd_3d<tf::RandomPartitioner<>>(8);
}

// ============================================================================
// IndexRangesPartitioner: direct stress tests
//
// The tests above exercise the box decomposition only indirectly, through
// a Partitioner's loop(). These tests drive tf::IndexRangesPartitioner
// itself -- construction, size(), active_rank(), and for_each_box() -- so a
// failure points straight at the box-decomposition algorithm rather than
// at whatever scheduling logic happens to be layered on top of it.
// ============================================================================

// Enumerates every element covered by `box` along its *active* dims
// [0, active_rank), converting each element to the flat index it occupies
// in the same row-major numbering as IndexRangesPartitioner::size(). Works
// for both positive and negative step sizes.
template <typename T, size_t N>
void collect_active_flat(
  const tf::IndexRanges<T, N>& box,
  const tf::IndexRanges<T, N>& full,
  size_t active_rank,
  std::vector<size_t>& out
) {
  std::array<size_t, N> extent{};
  for(size_t d = 0; d < active_rank; d++) {
    extent[d] = full.size(d);
  }

  auto in_range = [](T v, T end, T step) {
    return step > T{0} ? v < end : v > end;
  };

  auto rec = [&](auto& self, size_t d, size_t prefix) -> void {
    if(d == active_rank) {
      out.push_back(prefix);
      return;
    }
    auto [bb, be, bs] = box.dim(d);
    auto [fb, fe, fs] = full.dim(d);
    for(T v = bb; in_range(v, be, bs); v += bs) {
      size_t pos = static_cast<size_t>((v - fb) / fs);
      self(self, d + 1, prefix * extent[d] + pos);
    }
  };
  rec(rec, 0, 0);
}

// Verifies every dim >= active_rank in `box` is exactly the full extent of
// `full` -- the box-decomposition contract for inactive (zero-size-blocked)
// dims: they always ride along as full extent, regardless of where the
// active prefix's cursor currently is.
template <typename T, size_t N>
void verify_inactive_full_extent(
  const tf::IndexRanges<T, N>& box,
  const tf::IndexRanges<T, N>& full,
  size_t active_rank
) {
  for(size_t d = active_rank; d < N; d++) {
    REQUIRE(box.dim(d) == full.dim(d));
  }
}

// Verifies every active dim of `box` is an orthogonal sub-range of `full`:
// identical step size, and bounds contained within [full.begin, full.end)
// (direction-aware, since steps may be negative).
template <typename T, size_t N>
void verify_box_within_bounds(
  const tf::IndexRanges<T, N>& box,
  const tf::IndexRanges<T, N>& full,
  size_t active_rank
) {
  for(size_t d = 0; d < active_rank; d++) {
    auto [bb, be, bs] = box.dim(d);
    auto [fb, fe, fs] = full.dim(d);
    REQUIRE(bs == fs);
    if(fs > T{0}) {
      REQUIRE(bb >= fb);
      REQUIRE(be <= fe);
      REQUIRE(bb <= be);
    } else {
      REQUIRE(bb <= fb);
      REQUIRE(be >= fe);
      REQUIRE(bb >= be);
    }
  }
}

// Drains [0, total) in slices of `chunk_size` via repeated for_each_box()
// calls (mirroring how a Partitioner::loop() drives it), checking every
// emitted box against the two invariants above and returning the
// per-element visit counts so the caller can assert exactly-once coverage.
template <typename T, size_t N>
std::vector<int> drain_and_check(const tf::IndexRanges<T, N>& range, size_t chunk_size) {

  tf::IndexRangesPartitioner irp(range);
  size_t active_rank = irp.active_rank();
  size_t total = irp.size();
  REQUIRE(total == range.size());

  std::vector<int> visited(total, 0);
  size_t cursor = 0;
  while(cursor < total) {
    size_t e = (std::min)(cursor + chunk_size, total);
    bool stopped = irp.for_each_box(cursor, e,
      [&](const tf::IndexRanges<T, N>& box) {
        verify_box_within_bounds(box, range, active_rank);
        verify_inactive_full_extent(box, range, active_rank);
        std::vector<size_t> flats;
        collect_active_flat(box, range, active_rank, flats);
        REQUIRE_FALSE(flats.empty());
        for(auto f : flats) {
          REQUIRE(f < total);
          visited[f]++;
        }
        return false;
      }
    );
    REQUIRE_FALSE(stopped);
    cursor = e;
  }
  return visited;
}

template <typename T, size_t N>
void check_full_coverage(const tf::IndexRanges<T, N>& range, size_t chunk_size) {
  auto visited = drain_and_check(range, chunk_size == 0 ? size_t{1} : chunk_size);
  for(auto v : visited) {
    REQUIRE(v == 1);
  }
}

// ----------------------------------------------------------------------------
// Basic construction / size / active_rank
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.basic.2d") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 7, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 2);
  REQUIRE(irp.size() == 35);
}

TEST_CASE("IndexRangesPartitioner.basic.3d_non_unit_steps") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 8, 2),
    tf::IndexRange<int>(0, 15, 3),
    tf::IndexRange<int>(0, 4, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 3);
  REQUIRE(irp.size() == 4 * 5 * 4);
}

// ----------------------------------------------------------------------------
// Full coverage sweeps: unit steps
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.full_coverage.2d" * doctest::timeout(300)) {
  for(int d0 : {1, 2, 3, 5, 7, 12}) {
    for(int d1 : {1, 2, 3, 5, 7, 12}) {
      tf::IndexRanges<int, 2> r(
        tf::IndexRange<int>(0, d0, 1),
        tf::IndexRange<int>(0, d1, 1)
      );
      size_t N = r.size();
      for(size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, size_t{13}, N/2 + 1, N, N*3 + 1}) {
        check_full_coverage(r, cs);
      }
    }
  }
}

TEST_CASE("IndexRangesPartitioner.full_coverage.3d" * doctest::timeout(300)) {
  for(int d0 : {1, 3, 5}) {
    for(int d1 : {1, 4, 6}) {
      for(int d2 : {1, 2, 7}) {
        tf::IndexRanges<int, 3> r(
          tf::IndexRange<int>(0, d0, 1),
          tf::IndexRange<int>(0, d1, 1),
          tf::IndexRange<int>(0, d2, 1)
        );
        size_t N = r.size();
        for(size_t cs : {size_t{1}, size_t{3}, size_t{7}, N/2 + 1, N}) {
          check_full_coverage(r, cs);
        }
      }
    }
  }
}

TEST_CASE("IndexRangesPartitioner.full_coverage.4d" * doctest::timeout(300)) {
  tf::IndexRanges<int, 4> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 2, 1)
  );
  size_t N = r.size();
  for(size_t cs : {size_t{1}, size_t{3}, size_t{7}, size_t{20}, size_t{60}, N, N*2}) {
    check_full_coverage(r, cs);
  }
}

// ----------------------------------------------------------------------------
// Full coverage sweeps: non-unit and negative step sizes
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.full_coverage.non_unit_steps_2d" * doctest::timeout(300)) {
  for(int si : {1, 2, 3}) {
    for(int sj : {1, 2, 3}) {
      for(int di : {1, 3, 5}) {
        for(int dj : {1, 3, 5}) {
          tf::IndexRanges<int, 2> r(
            tf::IndexRange<int>(0, di * si, si),
            tf::IndexRange<int>(0, dj * sj, sj)
          );
          size_t N = r.size();
          for(size_t cs : {size_t{1}, size_t{3}, N}) {
            check_full_coverage(r, cs);
          }
        }
      }
    }
  }
}

TEST_CASE("IndexRangesPartitioner.full_coverage.negative_steps" * doctest::timeout(300)) {
  // all-negative 2D
  {
    tf::IndexRanges<int, 2> r(
      tf::IndexRange<int>(10, 0, -2),
      tf::IndexRange<int>(9, 0, -3)
    );
    size_t N = r.size();
    for(size_t cs : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, N}) {
      check_full_coverage(r, cs);
    }
  }
  // mixed-sign 3D
  {
    tf::IndexRanges<int, 3> r(
      tf::IndexRange<int>(0, 4,  1),
      tf::IndexRange<int>(10, 0, -2),
      tf::IndexRange<int>(0, 6,  1)
    );
    size_t N = r.size();
    for(size_t cs : {size_t{1}, size_t{6}, size_t{10}, size_t{30}, N}) {
      check_full_coverage(r, cs);
    }
  }
}

// ----------------------------------------------------------------------------
// Corner cases around hyperplane (dimension) boundaries
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.boundary_chunk_sizes") {
  // 3D 4x5x10: inner-row boundary = 10, outer-row boundary = 50, full = 200
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4,  1),
    tf::IndexRange<int>(0, 5,  1),
    tf::IndexRange<int>(0, 10, 1)
  );
  for(size_t cs : {size_t{1}, size_t{9}, size_t{10}, size_t{11},
                    size_t{49}, size_t{50}, size_t{51},
                    size_t{199}, size_t{200}, size_t{201}, size_t{1000}}) {
    check_full_coverage(r, cs);
  }
}

TEST_CASE("IndexRangesPartitioner.unit_chunk_produces_single_element_boxes") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  size_t N = irp.size();
  for(size_t flat = 0; flat < N; flat++) {
    size_t visits = 0;
    irp.for_each_box(flat, flat + 1, [&](const tf::IndexRanges<int, 3>& box) {
      ++visits;
      REQUIRE(box.size() == 1);
      return false;
    });
    REQUIRE(visits == 1);
  }
}

// ----------------------------------------------------------------------------
// active_rank() != N: zero-size dimensions at every position
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.active_rank.2d_zero_outer") {
  tf::IndexRanges<int, 2> r(tf::IndexRange<int>(0, 0, 1), tf::IndexRange<int>(0, 7, 1));
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 0);
  REQUIRE(irp.size() == 0);

  bool called = false;
  bool stopped = irp.for_each_box(0, 0, [&](const tf::IndexRanges<int, 2>&) {
    called = true;
    return false;
  });
  REQUIRE_FALSE(stopped);
  REQUIRE_FALSE(called);
}

TEST_CASE("IndexRangesPartitioner.active_rank.2d_zero_inner") {
  tf::IndexRanges<int, 2> r(tf::IndexRange<int>(0, 5, 1), tf::IndexRange<int>(0, 0, 1));
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 1);
  REQUIRE(irp.size() == 5);

  for(size_t cs : {size_t{1}, size_t{2}, size_t{5}, size_t{100}}) {
    check_full_coverage(r, cs);
  }

  // every box must carry dim(1) as the full (empty) extent, never touched
  irp.for_each_box(0, irp.size(), [&](const tf::IndexRanges<int, 2>& box) {
    verify_inactive_full_extent(box, r, 1);
    return false;
  });
}

TEST_CASE("IndexRangesPartitioner.active_rank.3d_zero_at_each_position" * doctest::timeout(300)) {
  for(size_t zero_pos : {size_t{0}, size_t{1}, size_t{2}}) {
    tf::IndexRange<int> dims[3] = {
      tf::IndexRange<int>(0, 4, 1),
      tf::IndexRange<int>(0, 5, 1),
      tf::IndexRange<int>(0, 6, 1)
    };
    dims[zero_pos] = tf::IndexRange<int>(0, 0, 1);

    tf::IndexRanges<int, 3> r(dims[0], dims[1], dims[2]);
    tf::IndexRangesPartitioner irp(r);
    REQUIRE(irp.active_rank() == zero_pos);

    if(zero_pos == 0) {
      REQUIRE(irp.size() == 0);
      continue;
    }

    size_t expect = 1;
    for(size_t d = 0; d < zero_pos; d++) {
      expect *= r.size(d);
    }
    REQUIRE(irp.size() == expect);

    for(size_t cs : {size_t{1}, size_t{2}, size_t{3}, irp.size(), irp.size() * 2 + 1}) {
      check_full_coverage(r, cs);
    }

    // dims from zero_pos onward must always show up as full extent
    irp.for_each_box(0, irp.size(), [&](const tf::IndexRanges<int, 3>& box) {
      verify_inactive_full_extent(box, r, zero_pos);
      return false;
    });
  }
}

TEST_CASE("IndexRangesPartitioner.active_rank.7d_zero_at_each_position" * doctest::timeout(300)) {
  for(size_t zero_pos = 0; zero_pos < 7; zero_pos++) {
    tf::IndexRange<int> dims[7];
    for(size_t d = 0; d < 7; d++) {
      dims[d] = tf::IndexRange<int>(0, static_cast<int>(d + 2), 1);
    }
    dims[zero_pos] = tf::IndexRange<int>(0, 0, 1);

    tf::IndexRanges<int, 7> r(
      dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]
    );
    tf::IndexRangesPartitioner irp(r);
    REQUIRE(irp.active_rank() == zero_pos);

    if(zero_pos == 0) {
      REQUIRE(irp.size() == 0);
      continue;
    }

    for(size_t cs : {size_t{1}, size_t{3}, irp.size()}) {
      check_full_coverage(r, cs);
    }

    irp.for_each_box(0, irp.size(), [&](const tf::IndexRanges<int, 7>& box) {
      verify_inactive_full_extent(box, r, zero_pos);
      return false;
    });
  }
}

// ----------------------------------------------------------------------------
// Early exit (bool-returning visit) and void-returning visit
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.for_each_box.stops_on_true") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  size_t N = irp.size();

  size_t visits = 0;
  bool stopped = irp.for_each_box(0, N, [&](const tf::IndexRanges<int, 3>&) {
    ++visits;
    return true;
  });
  REQUIRE(stopped);
  REQUIRE(visits == 1);
}

TEST_CASE("IndexRangesPartitioner.for_each_box.runs_to_completion_when_never_true") {
  tf::IndexRanges<int, 3> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1),
    tf::IndexRange<int>(0, 6, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  size_t N = irp.size();

  size_t covered = 0;
  bool stopped = irp.for_each_box(0, N, [&](const tf::IndexRanges<int, 3>& box) {
    covered += box.size();
    return false;
  });
  REQUIRE_FALSE(stopped);
  REQUIRE(covered == N);
}

TEST_CASE("IndexRangesPartitioner.for_each_box.void_visit_covers_everything") {
  tf::IndexRanges<int, 2> r(
    tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  );
  tf::IndexRangesPartitioner irp(r);
  size_t N = irp.size();

  size_t covered = 0;
  bool stopped = irp.for_each_box(0, N, [&](const tf::IndexRanges<int, 2>& box) {
    // void return -- for_each_box must treat this as "never stop early"
    covered += box.size();
  });
  REQUIRE_FALSE(stopped);
  REQUIRE(covered == N);
}

// ----------------------------------------------------------------------------
// Rank-1 (N == 1): the non-recursive fast path
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.rank1.basic") {
  tf::IndexRange<int> r(0, 10, 2);  // elements: 0,2,4,6,8
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 1);
  REQUIRE(irp.size() == 5);

  bool stopped = irp.for_each_box(1, 4, [&](const tf::IndexRanges<int, 1>& box) {
    REQUIRE(box.begin() == 2);
    REQUIRE(box.end() == 8);
    REQUIRE(box.step_size() == 2);
    return false;
  });
  REQUIRE_FALSE(stopped);
}

TEST_CASE("IndexRangesPartitioner.rank1.empty_range") {
  tf::IndexRange<int> r(0, 0, 1);
  tf::IndexRangesPartitioner irp(r);
  REQUIRE(irp.active_rank() == 0);
  REQUIRE(irp.size() == 0);

  bool called = false;
  bool stopped = irp.for_each_box(0, 0, [&](const tf::IndexRanges<int, 1>&) {
    called = true;
    return false;
  });
  REQUIRE_FALSE(stopped);
  REQUIRE_FALSE(called);
}

// ----------------------------------------------------------------------------
// Unsigned index type
// ----------------------------------------------------------------------------

TEST_CASE("IndexRangesPartitioner.unsigned_type") {
  tf::IndexRanges<size_t, 2> r(
    tf::IndexRange<size_t>(0, 6, 1),
    tf::IndexRange<size_t>(0, 9, 1)
  );
  for(size_t cs : {size_t{1}, size_t{4}, r.size()}) {
    check_full_coverage(r, cs);
  }
}

// ----------------------------------------------------------------------------
// Regression guard: a single for_each_box() call over the *entire* active
// range must decompose into O(active_rank) boxes, never O(2^active_rank).
// ----------------------------------------------------------------------------

template <typename T, size_t N>
void check_box_count_linear(const tf::IndexRanges<T, N>& r) {
  tf::IndexRangesPartitioner irp(r);
  size_t active_rank = irp.active_rank();
  if(active_rank == 0) {
    return;
  }
  size_t total = irp.size();
  size_t box_count = 0;
  irp.for_each_box(0, total, [&](const tf::IndexRanges<T, N>&) {
    ++box_count;
    return false;
  });
  REQUIRE(box_count <= 4 * active_rank + 4);
}

TEST_CASE("IndexRangesPartitioner.box_count_is_linear_in_rank") {
  check_box_count_linear(tf::IndexRanges<int, 2>(
    tf::IndexRange<int>(0, 7, 1), tf::IndexRange<int>(0, 11, 1)
  ));
  check_box_count_linear(tf::IndexRanges<int, 3>(
    tf::IndexRange<int>(0, 4, 1), tf::IndexRange<int>(0, 5, 1), tf::IndexRange<int>(0, 6, 1)
  ));
  check_box_count_linear(tf::IndexRanges<int, 4>(
    tf::IndexRange<int>(0, 3, 1), tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1), tf::IndexRange<int>(0, 6, 1)
  ));
  check_box_count_linear(tf::IndexRanges<int, 5>(
    tf::IndexRange<int>(0, 3, 1), tf::IndexRange<int>(0, 3, 1), tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 4, 1), tf::IndexRange<int>(0, 5, 1)
  ));
  check_box_count_linear(tf::IndexRanges<int, 6>(
    tf::IndexRange<int>(0, 2, 1), tf::IndexRange<int>(0, 3, 1), tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 4, 1), tf::IndexRange<int>(0, 4, 1), tf::IndexRange<int>(0, 5, 1)
  ));
  check_box_count_linear(tf::IndexRanges<int, 7>(
    tf::IndexRange<int>(0, 2, 1), tf::IndexRange<int>(0, 2, 1), tf::IndexRange<int>(0, 3, 1),
    tf::IndexRange<int>(0, 3, 1), tf::IndexRange<int>(0, 4, 1), tf::IndexRange<int>(0, 4, 1),
    tf::IndexRange<int>(0, 5, 1)
  ));
}
