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
        tf::IndexRange<int, 2> range(
          tf::IndexRange<int>(0, static_cast<int>(rows), 1),
          tf::IndexRange<int>(0, static_cast<int>(cols), 1)
        );

        size_t N = range.size();
        size_t chunk_size = p.chunk_size() ? p.chunk_size() : (W ? (N + W - 1) / W : N);
        chunk_size = range.ceil(chunk_size == 0 ? 1 : chunk_size);

        std::vector<std::atomic<int>> visited(N);
        for(auto& v : visited) v.store(0);

        for(size_t w = 0; w < W && w < N; w++) {
          size_t curr_b = w * chunk_size;
          if(curr_b >= N) break;
          p.loop(range, N, W, curr_b, chunk_size,
            [&](const tf::IndexRange<int, 2>& box) {
              for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
                for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
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
        tf::IndexRange<int, 2> range(
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
              [&](const tf::IndexRange<int, 2>& box) {
                for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
                  for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
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
          tf::IndexRange<int, 3> range(
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
                [&](const tf::IndexRange<int, 3>& box) {
                  for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
                    for(int j = box.dim(1).begin(); j < box.dim(1).end(); j += box.dim(1).step_size()) {
                      for(int k = box.dim(2).begin(); k < box.dim(2).end(); k += box.dim(2).step_size()) {
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
        tf::IndexRange<int, 2> range(
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
              [&](const tf::IndexRange<int, 2>& box) -> bool {
                // return true (stop) as soon as we see i==1 in the box
                for(int i = box.dim(0).begin(); i < box.dim(0).end(); i += box.dim(0).step_size()) {
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
