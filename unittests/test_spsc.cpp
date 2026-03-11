#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>

#include <atomic>
#include <thread>
#include <vector>

// ============================================================================
// Compile-time checks
// ============================================================================

static_assert(tf::SPSCRing<int, 7>::capacity() == 127);
static_assert(tf::SPSCRing<int, 1>::capacity() == 1);
// LogSize=30 would instantiate a 4 GB std::array — verify the capacity formula
// arithmetically rather than by instantiating the type.
static_assert(((std::size_t{1} << 30) - 1) == (std::size_t{1} << 30) - 1);

static_assert(std::is_same_v<tf::SPSCRing<int,   7>::value_type, std::optional<int>>);
static_assert(std::is_same_v<tf::SPSCRing<int*,  7>::value_type, int*>);
static_assert(std::is_same_v<tf::SPSCRing<void*, 7>::value_type, void*>);

// ============================================================================
// Non-pointer T (value_type = std::optional<T>)
// ============================================================================

TEST_CASE("SPSCRing<int,7>: capacity") {
  tf::SPSCRing<int, 7> ring;
  REQUIRE(ring.capacity() == 127);
  REQUIRE(ring.empty_approx() == true);
  REQUIRE(ring.full_approx()  == false);
  REQUIRE(ring.size_approx()  == 0);
}

TEST_CASE("SPSCRing<int,3>: push/pop basic") {
  tf::SPSCRing<int, 3> ring;  // capacity = 7

  for (int i = 0; i < 7; ++i) {
    REQUIRE(ring.push(i) == true);
  }
  REQUIRE(ring.full_approx() == true);
  REQUIRE(ring.push(99) == false);  // full

  for (int i = 0; i < 7; ++i) {
    auto v = ring.pop();
    REQUIRE(v.has_value() == true);
    REQUIRE(*v == i);
  }

  REQUIRE(ring.empty_approx() == true);
  REQUIRE(ring.pop() == std::nullopt);
}

TEST_CASE("SPSCRing<int,3>: wrap-around") {
  tf::SPSCRing<int, 3> ring;  // capacity = 7

  // fill, drain, fill again - exercises the modular wrap
  for (int round = 0; round < 4; ++round) {
    for (int i = 0; i < 7; ++i) {
      REQUIRE(ring.push(i * 10) == true);
    }
    REQUIRE(ring.size_approx() == 7);
    for (int i = 0; i < 7; ++i) {
      auto v = ring.pop();
      REQUIRE(v.has_value());
      REQUIRE(*v == i * 10);
    }
    REQUIRE(ring.empty_approx() == true);
  }
}

TEST_CASE("SPSCRing<int,3>: copy overload") {
  tf::SPSCRing<int, 3> ring;

  const int x = 42;
  REQUIRE(ring.push(x) == true);

  auto v = ring.pop();
  REQUIRE(v.has_value());
  REQUIRE(*v == 42);
  REQUIRE(x == 42);  // original unchanged
}

TEST_CASE("SPSCRing<int,3>: size_approx") {
  tf::SPSCRing<int, 3> ring;

  REQUIRE(ring.size_approx() == 0);
  (void)ring.push(1);
  REQUIRE(ring.size_approx() == 1);
  (void)ring.push(2);
  REQUIRE(ring.size_approx() == 2);
  (void)ring.pop();
  REQUIRE(ring.size_approx() == 1);
  (void)ring.pop();
  REQUIRE(ring.size_approx() == 0);
}

// ============================================================================
// Pointer T (value_type = T, empty = nullptr)
// ============================================================================

TEST_CASE("SPSCRing<int*,3>: nullptr sentinel") {
  tf::SPSCRing<int*, 3> ring;

  REQUIRE(ring.pop() == nullptr);

  int a = 10, b = 20;
  REQUIRE(ring.push(&a) == true);
  REQUIRE(ring.push(&b) == true);

  REQUIRE(ring.pop() == &a);
  REQUIRE(ring.pop() == &b);
  REQUIRE(ring.pop() == nullptr);
}

TEST_CASE("SPSCRing<int*,3>: full then drain") {
  tf::SPSCRing<int*, 3> ring;  // capacity = 7

  std::vector<int> data(7);
  for (int i = 0; i < 7; ++i) {
    data[i] = i;
    REQUIRE(ring.push(&data[i]) == true);
  }
  REQUIRE(ring.push(nullptr) == false);  // full

  for (int i = 0; i < 7; ++i) {
    int* p = ring.pop();
    REQUIRE(p != nullptr);
    REQUIRE(*p == i);
  }
  REQUIRE(ring.pop() == nullptr);
}

// ============================================================================
// Concurrent: one producer thread + one consumer thread
// ============================================================================

TEST_CASE("SPSCRing<int,10>: concurrent producer-consumer" * doctest::timeout(30)) {
  tf::SPSCRing<int, 10> ring;  // capacity = 1023

  constexpr int N = 100000;

  std::thread producer([&]() {
    for (int i = 0; i < N; ++i) {
      while (!ring.push(i)) { /* spin - ring full */ }
    }
  });

  std::thread consumer([&]() {
    int expected = 0;
    while (expected < N) {
      if (auto v = ring.pop()) {
        REQUIRE(*v == expected);
        ++expected;
      }
    }
  });

  producer.join();
  consumer.join();

  REQUIRE(ring.empty_approx() == true);
}

TEST_CASE("SPSCRing<int*,10>: concurrent producer-consumer pointer" * doctest::timeout(30)) {
  tf::SPSCRing<int*, 10> ring;

  constexpr int N = 100000;
  std::vector<int> data(N);
  for (int i = 0; i < N; ++i) data[i] = i;

  std::thread producer([&]() {
    for (int i = 0; i < N; ++i) {
      while (!ring.push(&data[i])) { /* spin */ }
    }
  });

  std::thread consumer([&]() {
    int expected = 0;
    while (expected < N) {
      int* p = ring.pop();
      if (p != nullptr) {
        REQUIRE(*p == expected);
        ++expected;
      }
    }
  });

  producer.join();
  consumer.join();

  REQUIRE(ring.pop() == nullptr);
}
