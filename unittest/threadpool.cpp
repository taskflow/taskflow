// conbributed by Guannan

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/threadpool/threadpool.hpp>
#include <taskflow/threadpool/proactive_threadpool.hpp>

template <typename ThreadpoolType>
void test_threadpool() {
  // ...
  // REQUIRE ...
}

// --------------------------------------------------------
// Testcase:
// --------------------------------------------------------
TEST_CASE("Threadpool.ProactiveThreadpool" * doctest::timeout(5)) {
  test_threadpool<tf::ProactiveThreadpool>();
}

