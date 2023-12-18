#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/transform.hpp>
#include <taskflow/algorithm/find.hpp>
#include <iostream>
#include <mutex>

namespace
{
  int& GetThreadSpecificContext()
  {
      thread_local int context = 0;
      return context;
  }

  const int UPPER = 1000;
}

// --------------------------------------------------------
// Testcase: chunk context
// --------------------------------------------------------

// ChunkContext
TEST_CASE("Chunk.Context.for_each_index.static" * doctest::timeout(300))
{
  for (int tc = 4; tc < 64; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(100, 0);
    taskflow.for_each_index(
        0, 100, 1, [&](int i)
		{ count++; range[i] = GetThreadSpecificContext(); },
        tf::StaticPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(count == 100);
    REQUIRE(range == std::vector<int>(100, tc));
    REQUIRE(wrapper_called_count == tc);
  }
}

// ChunkContext
TEST_CASE("Chunk.Context.for_each_index.dynamic" * doctest::timeout(300))
{
  for (int tc = 4; tc < 64; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    std::vector<int> range(UPPER, 0);
    tf::Taskflow taskflow;
    taskflow.for_each_index(
        0, UPPER, 1, [&](int i)
        { count++; range[i] = GetThreadSpecificContext();
        },
        tf::DynamicPartitioner(UPPER/tc/2), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(count == UPPER);
    REQUIRE(range == std::vector<int>(UPPER, tc));
    // Dynamic partitioner will spawn sometimes less tasks
    REQUIRE(wrapper_called_count <= tc);
  }
}


TEST_CASE("Chunk.Context.for_each.static" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER, 0);
    taskflow.for_each(
        begin(range), end(range), [&](int& i)
        { count++; i = GetThreadSpecificContext(); },
        tf::StaticPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(count == UPPER);
    REQUIRE(range == std::vector<int>(UPPER, tc));
    REQUIRE(wrapper_called_count == tc);
  }
}

TEST_CASE("Chunk.Context.for_each.dynamic" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;

    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER, 0);
    taskflow.for_each(
        begin(range), end(range), [](int& i)
        { i = GetThreadSpecificContext(); },
        tf::DynamicPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(range == std::vector<int>(UPPER, tc));
    REQUIRE(wrapper_called_count <= tc); // Dynamic scheduling is not obliged to load all threads with iterations, so <= is appropriate here
  }
}

TEST_CASE("Chunk.Context.transform.static" * doctest::timeout(300))
{
  // Write a test case for using the taskwrapper on tf::transform
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER, 0);
    std::vector<int> result(UPPER);
    taskflow.transform(
        begin(range), end(range), begin(result), [&](int)
        { return GetThreadSpecificContext(); },
        tf::StaticPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(result == std::vector<int>(UPPER, tc));
  }
}

// Implement for dynamic case for transform
TEST_CASE("Chunk.Context.transform.dynamic" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::iota(range.begin(), range.end(), 0);
    std::vector<int> result(UPPER);
    taskflow.transform(
        begin(range), end(range), begin(result), [&](int)
        { return GetThreadSpecificContext(); },
        tf::DynamicPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(result == std::vector<int>(UPPER, tc));
  }
}

// write a test for find using static partitioner
TEST_CASE("Chunk.Context.find.static" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if(
        begin(range), end(range), result, [&](int i)
        { return i == 500 && GetThreadSpecificContext() == tc; },
        tf::StaticPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == 500);
  }
}

// Same as above testcase but with dynamic partitioner
TEST_CASE("Chunk.Context.find.dynamic" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if(
        begin(range), end(range), result, [&](int i)
        { return i == 500 && GetThreadSpecificContext() == tc; },
        tf::DynamicPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count <= tc);
    REQUIRE(*result == 500);
  }
}

// dynamic and static partitioner for find_if_not
TEST_CASE("Chunk.Context.find_if_not.static" * doctest::timeout(300))
{
  for (int tc = 4; tc < 16; tc++)
  {
    tf::Executor executor(tc);
    std::atomic<int> wrapper_called_count = 0;
    auto task_wrapper = [&](auto &&task)
    {
      wrapper_called_count++;
      GetThreadSpecificContext() = tc;
      task();
      GetThreadSpecificContext() = 0;
    };
    std::atomic<int> count = 0;
    tf::Taskflow taskflow;
    std::vector<int> range(UPPER);
    std::vector<int>::iterator result;
    std::iota(range.begin(), range.end(), 0);
    taskflow.find_if_not(
        begin(range), end(range), result, [&](int i)
        { return i !=500 && GetThreadSpecificContext() == tc; },
        tf::StaticPartitioner(1), task_wrapper);
    executor.run(taskflow).wait();

    REQUIRE(wrapper_called_count == tc);
    REQUIRE(*result == 500);
  }
}
