#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/data_pipeline.hpp>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// --------------------------------------------------------
// Testcase: 1 pipe, L lines, w workers
// --------------------------------------------------------
void data_pipeline_1P(size_t L, unsigned w, tf::PipeType type) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  // iterate different data amount (1, 2, 3, 4, 5, ... 1000000)
  for (size_t N = 0; N <= maxN; N++) {

    // serial direction
    if (type == tf::PipeType::SERIAL) {
      tf::Taskflow taskflow;
      size_t j = 0;
      tf::DataPipeline pl (L, tf::make_data_pipe<tf::Pipeflow&, void>(type, [L, N, &j, &source](auto& pf) mutable {
        if (j == N) {
          pf.stop();
          return;
        }
        REQUIRE(j == source[j]);
        REQUIRE(pf.token() % L == pf.line());
        j++;
      }));

      auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");

      auto test = taskflow.emplace([&](){
        REQUIRE(j == N);
        REQUIRE(pl.num_tokens() == N);
      }).name("test");

      datapipeline.precede(test);

      executor.run_until(taskflow, [counter=3, j]() mutable{
        j = 0;
        return counter --== 0;
      }).get();
    }
  }
}

// serial pipe with one line
TEST_CASE("DataPipeline.1P(S).1L.1W" * doctest::timeout(300)) {
  data_pipeline_1P(1, 1, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).1L.2W" * doctest::timeout(300)) {
  data_pipeline_1P(1, 2, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).1L.3W" * doctest::timeout(300)) {
  data_pipeline_1P(1, 3, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).1L.4W" * doctest::timeout(300)) {
  data_pipeline_1P(1, 4, tf::PipeType::SERIAL);
}

// serial pipe with two lines
TEST_CASE("DataPipeline.1P(S).2L.1W" * doctest::timeout(300)) {
  data_pipeline_1P(2, 1, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).2L.2W" * doctest::timeout(300)) {
  data_pipeline_1P(2, 2, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).2L.3W" * doctest::timeout(300)) {
  data_pipeline_1P(2, 3, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).2L.4W" * doctest::timeout(300)) {
  data_pipeline_1P(2, 4, tf::PipeType::SERIAL);
}

// serial pipe with three lines
TEST_CASE("DataPipeline.1P(S).3L.1W" * doctest::timeout(300)) {
  data_pipeline_1P(3, 1, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).3L.2W" * doctest::timeout(300)) {
  data_pipeline_1P(3, 2, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).3L.3W" * doctest::timeout(300)) {
  data_pipeline_1P(3, 3, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).3L.4W" * doctest::timeout(300)) {
  data_pipeline_1P(3, 4, tf::PipeType::SERIAL);
}

// serial pipe with three lines
TEST_CASE("DataPipeline.1P(S).4L.1W" * doctest::timeout(300)) {
  data_pipeline_1P(4, 1, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).4L.2W" * doctest::timeout(300)) {
  data_pipeline_1P(4, 2, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).4L.3W" * doctest::timeout(300)) {
  data_pipeline_1P(4, 3, tf::PipeType::SERIAL);
}

TEST_CASE("DataPipeline.1P(S).4L.4W" * doctest::timeout(300)) {
  data_pipeline_1P(4, 4, tf::PipeType::SERIAL);
}

// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers
// ----------------------------------------------------------------------------

void data_pipeline_2P_SS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0, j2 = 0;
    size_t cnt = 1;

    tf::DataPipeline pl(
      L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j2, L](int& input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        // REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // j2++;
        REQUIRE(source[j2++] + 1 == input);
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);
      REQUIRE(pl.num_tokens() == cnt * N);
    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&]() mutable {
      j1 = 0;
      j2 = 0;
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }
      cnt++;
    }).get();
  }
}

// two pipes (SS)
TEST_CASE("DataPipeline.2P(SS).1L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(1, 1);
}

TEST_CASE("DataPipeline.2P(SS).1L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(1, 2);
}

TEST_CASE("DataPipeline.2P(SS).1L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(1, 3);
}

TEST_CASE("DataPipeline.2P(SS).1L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(1, 4);
}

TEST_CASE("DataPipeline.2P(SS).2L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(2, 1);
}

TEST_CASE("DataPipeline.2P(SS).2L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(2, 2);
}

TEST_CASE("DataPipeline.2P(SS).2L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(2, 3);
}

TEST_CASE("DataPipeline.2P(SS).2L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(2, 4);
}

TEST_CASE("DataPipeline.2P(SS).3L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(3, 1);
}

TEST_CASE("DataPipeline.2P(SS).3L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(3, 2);
}

TEST_CASE("DataPipeline.2P(SS).3L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(3, 3);
}

TEST_CASE("DataPipeline.2P(SS).3L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(3, 4);
}

TEST_CASE("DataPipeline.2P(SS).4L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(4, 1);
}

TEST_CASE("DataPipeline.2P(SS).4L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(4, 2);
}

TEST_CASE("DataPipeline.2P(SS).4L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(4, 3);
}

TEST_CASE("DataPipeline.2P(SS).4L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SS(4, 4);
}

// ----------------------------------------------------------------------------
// two pipes (SP), L lines, W workers
// ----------------------------------------------------------------------------
void data_pipeline_2P_SP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;
    size_t cnt = 1;

    tf::DataPipeline pl(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL,
      [N, &collection, &mutex, &j2, L](int& input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          // collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          collection.push_back(input);
        }
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);

      std::sort(collection.begin(), collection.end());
      for(size_t i = 0; i < N; i++) {
        REQUIRE(collection[i] == i + 1);
      }
      REQUIRE(pl.num_tokens() == cnt * N);
    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&]() mutable {
      j1 = j2 = 0;
      collection.clear();
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }
      cnt++;
    }).get();
  }
}

// two pipes (SP)
TEST_CASE("DataPipeline.2P(SP).1L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(1, 1);
}

TEST_CASE("DataPipeline.2P(SP).1L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(1, 2);
}

TEST_CASE("DataPipeline.2P(SP).1L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(1, 3);
}

TEST_CASE("DataPipeline.2P(SP).1L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(1, 4);
}

TEST_CASE("DataPipeline.2P(SP).2L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(2, 1);
}

TEST_CASE("DataPipeline.2P(SP).2L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(2, 2);
}

TEST_CASE("DataPipeline.2P(SP).2L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(2, 3);
}

TEST_CASE("DataPipeline.2P(SP).2L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(2, 4);
}

TEST_CASE("DataPipeline.2P(SP).3L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(3, 1);
}

TEST_CASE("DataPipeline.2P(SP).3L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(3, 2);
}

TEST_CASE("DataPipeline.2P(SP).3L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(3, 3);
}

TEST_CASE("DataPipeline.2P(SP).3L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(3, 4);
}

TEST_CASE("DataPipeline.2P(SP).4L.1W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(4, 1);
}

TEST_CASE("DataPipeline.2P(SP).4L.2W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(4, 2);
}

TEST_CASE("DataPipeline.2P(SP).4L.3W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(4, 3);
}

TEST_CASE("DataPipeline.2P(SP).4L.4W" * doctest::timeout(300)) {
  data_pipeline_2P_SP(4, 4);
}

// ----------------------------------------------------------------------------
// three pipes (SSS), L lines, W workers
// ----------------------------------------------------------------------------
void data_pipeline_3P_SSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0, j2 = 0, j3 = 0;
    size_t cnt = 1;

    tf::DataPipeline pl(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, std::string>(tf::PipeType::SERIAL, [N, &source, &j2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
        return std::to_string(input);
      }),

      tf::make_data_pipe<std::string, void>(tf::PipeType::SERIAL, [N, &source, &j3, L](std::string input, tf::Pipeflow& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(source[j3] + 1 == stoi(input));
        REQUIRE(pf.token() % L == pf.line());
        j3++;
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);
      REQUIRE(j3 == N);
      REQUIRE(pl.num_tokens() == cnt * N);
    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&]() mutable {
      j1 = j2 = j3 = 0;
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }
      cnt++;
    }).get();
  }
}

// three pipes (SSS)
TEST_CASE("DataPipeline.3P(SSS).1L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(1, 1);
}

TEST_CASE("DataPipeline.3P(SSS).1L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(1, 2);
}

TEST_CASE("DataPipeline.3P(SSS).1L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(1, 3);
}

TEST_CASE("DataPipeline.3P(SSS).1L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(1, 4);
}

TEST_CASE("DataPipeline.3P(SSS).2L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(2, 1);
}

TEST_CASE("DataPipeline.3P(SSS).2L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(2, 2);
}

TEST_CASE("DataPipeline.3P(SSS).2L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(2, 3);
}

TEST_CASE("DataPipeline.3P(SSS).2L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(2, 4);
}

TEST_CASE("DataPipeline.3P(SSS).3L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(3, 1);
}

TEST_CASE("DataPipeline.3P(SSS).3L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(3, 2);
}

TEST_CASE("DataPipeline.3P(SSS).3L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(3, 3);
}

TEST_CASE("DataPipeline.3P(SSS).3L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(3, 4);
}

TEST_CASE("DataPipeline.3P(SSS).4L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(4, 1);
}

TEST_CASE("DataPipeline.3P(SSS).4L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(4, 2);
}

TEST_CASE("DataPipeline.3P(SSS).4L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(4, 3);
}

TEST_CASE("DataPipeline.3P(SSS).4L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSS(4, 4);
}


// ----------------------------------------------------------------------------
// three pipes (SSP), L lines, W workers
// ----------------------------------------------------------------------------
void data_pipeline_3P_SSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0, j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex;
    std::vector<int> collection;
    size_t cnt = 1;

    tf::DataPipeline pl(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j2] + 1;
        // mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
        return input;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL, [N, &j3, &mutex, &collection, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          collection.push_back(input);
        }
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);
      REQUIRE(j3 == N);
      REQUIRE(collection.size() == N);

      std::sort(collection.begin(), collection.end());
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection[i] == i + 1);
      }
      REQUIRE(pl.num_tokens() == cnt * N);
    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&](){
      j1 = j2 = j3 = 0;
      collection.clear();
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }

      cnt++;
    }).get();
  }
}

// three pipes (SSP)
TEST_CASE("DataPipeline.3P(SSP).1L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(1, 1);
}

TEST_CASE("DataPipeline.3P(SSP).1L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(1, 2);
}

TEST_CASE("DataPipeline.3P(SSP).1L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(1, 3);
}

TEST_CASE("DataPipeline.3P(SSP).1L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(1, 4);
}

TEST_CASE("DataPipeline.3P(SSP).2L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(2, 1);
}

TEST_CASE("DataPipeline.3P(SSP).2L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(2, 2);
}

TEST_CASE("DataPipeline.3P(SSP).2L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(2, 3);
}

TEST_CASE("DataPipeline.3P(SSP).2L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(2, 4);
}

TEST_CASE("DataPipeline.3P(SSP).3L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(3, 1);
}

TEST_CASE("DataPipeline.3P(SSP).3L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(3, 2);
}

TEST_CASE("DataPipeline.3P(SSP).3L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(3, 3);
}

TEST_CASE("DataPipeline.3P(SSP).3L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(3, 4);
}

TEST_CASE("DataPipeline.3P(SSP).4L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(4, 1);
}

TEST_CASE("DataPipeline.3P(SSP).4L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(4, 2);
}

TEST_CASE("DataPipeline.3P(SSP).4L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(4, 3);
}

TEST_CASE("DataPipeline.3P(SSP).4L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SSP(4, 4);
}

// ----------------------------------------------------------------------------
// three pipes (SPS), L lines, W workers
// ----------------------------------------------------------------------------
void data_pipeline_3P_SPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0, j3 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;
    size_t cnt = 1;

    tf::DataPipeline pl(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](tf::Pipeflow& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [N, &j2, &mutex, &collection, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2++ < N);
        //*(pf.output()) = *(pf.input()) + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex);
          // mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          REQUIRE(pf.token() % L == pf.line());
          collection.push_back(input);
          return input + 1;
        }
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j3, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j3] + 2 == input);
        j3++;
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);
      REQUIRE(j3 == N);
      REQUIRE(collection.size() == N);

      std::sort(collection.begin(), collection.end());
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection[i] == i + 1);
      }
      REQUIRE(pl.num_tokens() == cnt * N);

    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&]() mutable {
      j1 = j2 = j3 = 0;
      collection.clear();
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }

      cnt++;
    }).get();
  }
}

// three pipes (SPS)
TEST_CASE("DataPipeline.3P(SPS).1L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(1, 1);
}

TEST_CASE("DataPipeline.3P(SPS).1L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(1, 2);
}

TEST_CASE("DataPipeline.3P(SPS).1L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(1, 3);
}

TEST_CASE("DataPipeline.3P(SPS).1L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(1, 4);
}

TEST_CASE("DataPipeline.3P(SPS).2L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(2, 1);
}

TEST_CASE("DataPipeline.3P(SPS).2L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(2, 2);
}

TEST_CASE("DataPipeline.3P(SPS).2L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(2, 3);
}

TEST_CASE("DataPipeline.3P(SPS).2L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(2, 4);
}

TEST_CASE("DataPipeline.3P(SPS).3L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(3, 1);
}

TEST_CASE("DataPipeline.3P(SPS).3L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(3, 2);
}

TEST_CASE("DataPipeline.3P(SPS).3L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(3, 3);
}

TEST_CASE("DataPipeline.3P(SPS).3L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(3, 4);
}

TEST_CASE("DataPipeline.3P(SPS).4L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(4, 1);
}

TEST_CASE("DataPipeline.3P(SPS).4L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(4, 2);
}

TEST_CASE("DataPipeline.3P(SPS).4L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(4, 3);
}

TEST_CASE("DataPipeline.3P(SPS).4L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPS(4, 4);
}

// ----------------------------------------------------------------------------
// three pipes (SPP), L lines, W workers
// ----------------------------------------------------------------------------


void data_pipeline_3P_SPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex2;
    std::mutex mutex3;
    std::vector<int> collection2;
    std::vector<int> collection3;
    size_t cnt = 1;

    tf::DataPipeline pl(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](tf::Pipeflow& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [N, &j2, &mutex2, &collection2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2++ < N);
        //*pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          REQUIRE(pf.token() % L == pf.line());
          // mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection2.push_back(input);
          return input + 1;
        }
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL, [N, &j3, &mutex3, &collection3, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          REQUIRE(pf.token() % L == pf.line());
          collection3.push_back(input);
        }
      })
    );

    auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(j1 == N);
      REQUIRE(j2 == N);
      REQUIRE(j3 == N);
      REQUIRE(collection2.size() == N);
      REQUIRE(collection3.size() == N);

      std::sort(collection2.begin(), collection2.end());
      std::sort(collection3.begin(), collection3.end());
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection2[i] == i + 1);
        REQUIRE(collection3[i] == i + 2);
      }
      REQUIRE(pl.num_tokens() == cnt * N);
    }).name("test");

    datapipeline.precede(test);

    executor.run_n(taskflow, 3, [&]() mutable {
      j1 = j2 = j3 = 0;
      collection2.clear();
      collection3.clear();
      // for(size_t i = 0; i < mybuffer.size(); ++i){
      //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
      //     mybuffer[i][j] = 0;
      //   }
      // }

      cnt++;
    }).get();
  }
}

// three pipes (SPP)
TEST_CASE("DataPipeline.3P(SPP).1L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(1, 1);
}

TEST_CASE("DataPipeline.3P(SPP).1L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(1, 2);
}

TEST_CASE("DataPipeline.3P(SPP).1L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(1, 3);
}

TEST_CASE("DataPipeline.3P(SPP).1L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(1, 4);
}

TEST_CASE("DataPipeline.3P(SPP).2L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(2, 1);
}

TEST_CASE("DataPipeline.3P(SPP).2L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(2, 2);
}

TEST_CASE("DataPipeline.3P(SPP).2L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(2, 3);
}

TEST_CASE("DataPipeline.3P(SPP).2L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(2, 4);
}

TEST_CASE("DataPipeline.3P(SPP).3L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(3, 1);
}

TEST_CASE("DataPipeline.3P(SPP).3L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(3, 2);
}

TEST_CASE("DataPipeline.3P(SPP).3L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(3, 3);
}

TEST_CASE("DataPipeline.3P(SPP).3L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(3, 4);
}

TEST_CASE("DataPipeline.3P(SPP).4L.1W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(4, 1);
}

TEST_CASE("DataPipeline.3P(SPP).4L.2W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(4, 2);
}

TEST_CASE("DataPipeline.3P(SPP).4L.3W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(4, 3);
}

TEST_CASE("DataPipeline.3P(SPP).4L.4W" * doctest::timeout(300)) {
  data_pipeline_3P_SPP(4, 4);
}

// ----------------------------------------------------------------------------
// three parallel pipelines. each pipeline with L lines.
// one with four pipes (SSSS), one with three pipes (SPP),
// One with two  Pipes (SP)
//
//      --> SSSS --> O --
//     |                 |
// O -> --> SSP  --> O -- --> O
//     |                 |
//      --> SP   --> O --
//
// ----------------------------------------------------------------------------

void three_parallel_data_pipelines(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 4>> mybuffer1(L);
  // std::vector<std::array<int, 3>> mybuffer2(L);
  // std::vector<std::array<int, 2>> mybuffer3(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1_1 = 0, j1_2 = 0, j1_3 = 0, j1_4 = 0;
    size_t cnt1 = 1;

    // pipeline 1 is SSSS
    tf::DataPipeline pl1(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1_1, L](tf::Pipeflow& pf) mutable {
        if(j1_1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1_1 == source[j1_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_1] + 1;
        // j1_1++;
        return source[j1_1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j1_2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j1_2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_2] + 1 == input);
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_2] + 1;
        j1_2++;
        return input;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j1_3, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j1_3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_3] + 1 == input);
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_3] + 1;
        j1_3++;
        return input;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j1_4, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j1_4 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_4] + 1 == input);
        j1_4++;
      })
    );

    auto datapipeline1 = taskflow.composed_of(pl1).name("module_of_datapipeline1");
    auto test1 = taskflow.emplace([&](){
      REQUIRE(j1_1 == N);
      REQUIRE(j1_2 == N);
      REQUIRE(j1_3 == N);
      REQUIRE(j1_4 == N);
      REQUIRE(pl1.num_tokens() == cnt1 * N);
    }).name("test1");

    datapipeline1.precede(test1);



    // the followings are definitions for pipeline 2
    size_t j2_1 = 0, j2_2 = 0;
    std::atomic<size_t> j2_3 = 0;
    std::mutex mutex2_3;
    std::vector<int> collection2_3;
    size_t cnt2 = 1;

    // pipeline 2 is SSP
    tf::DataPipeline pl2(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j2_1, L](tf::Pipeflow& pf) mutable {
        if(j2_1 == N) {
          pf.stop();
          return 0 ;
        }
        REQUIRE(j2_1 == source[j2_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer2[pf.line()][pf.pipe()] = source[j2_1] + 1;
        // j2_1++;
        return source[j2_1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j2_2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2_2 < N);
        REQUIRE(source[j2_2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer2[pf.line()][pf.pipe()] = source[j2_2] + 1;
        j2_2++;
        return input;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL, [N, &j2_3, &mutex2_3, &collection2_3, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2_3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex2_3);
          REQUIRE(pf.token() % L == pf.line());
          collection2_3.push_back(input);
        }
      })
    );

    auto datapipeline2 = taskflow.composed_of(pl2).name("module_of_datapipeline2");
    auto test2 = taskflow.emplace([&](){
      REQUIRE(j2_1 == N);
      REQUIRE(j2_2 == N);
      REQUIRE(j2_3 == N);
      REQUIRE(collection2_3.size() == N);

      std::sort(collection2_3.begin(), collection2_3.end());
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection2_3[i] == i + 1);
      }
      REQUIRE(pl2.num_tokens() == cnt2 * N);
    }).name("test2");

    datapipeline2.precede(test2);



    // the followings are definitions for pipeline 3
    size_t j3_1 = 0;
    std::atomic<size_t> j3_2 = 0;
    std::mutex mutex3_2;
    std::vector<int> collection3_2;
    size_t cnt3 = 1;

    // pipeline 3 is SP
    tf::DataPipeline pl3(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j3_1, L](tf::Pipeflow& pf) mutable {
        if(j3_1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j3_1 == source[j3_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer3[pf.line()][pf.pipe()] = source[j3_1] + 1;
        // j3_1++;
        return source[j3_1++] + 1;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL,
      [N, &collection3_2, &mutex3_2, &j3_2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j3_2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3_2);
          REQUIRE(pf.token() % L == pf.line());
          collection3_2.push_back(input);
        }
      })
    );

    auto datapipeline3 = taskflow.composed_of(pl3).name("module_of_datapipeline3");
    auto test3 = taskflow.emplace([&](){
      REQUIRE(j3_1 == N);
      REQUIRE(j3_2 == N);

      std::sort(collection3_2.begin(), collection3_2.end());
      for(size_t i = 0; i < N; i++) {
        REQUIRE(collection3_2[i] == i + 1);
      }
      REQUIRE(pl3.num_tokens() == cnt3 * N);
    }).name("test3");

    datapipeline3.precede(test3);


    auto initial  = taskflow.emplace([](){}).name("initial");
    auto terminal = taskflow.emplace([](){}).name("terminal");

    initial.precede(datapipeline1, datapipeline2, datapipeline3);
    terminal.succeed(test1, test2, test3);

    //taskflow.dump(std::cout);

    executor.run_n(taskflow, 3, [&]() mutable {
      // reset variables for pipeline 1
      j1_1 = j1_2 = j1_3 = j1_4 = 0;
      // for(size_t i = 0; i < mybuffer1.size(); ++i){
      //   for(size_t j = 0; j < mybuffer1[0].size(); ++j){
      //     mybuffer1[i][j] = 0;
      //   }
      // }
      cnt1++;

      // reset variables for pipeline 2
      j2_1 = j2_2 = j2_3 = 0;
      collection2_3.clear();
      // for(size_t i = 0; i < mybuffer2.size(); ++i){
      //   for(size_t j = 0; j < mybuffer2[0].size(); ++j){
      //     mybuffer2[i][j] = 0;
      //   }
      // }
      cnt2++;

      // reset variables for pipeline 3
      j3_1 = j3_2 = 0;
      collection3_2.clear();
      // for(size_t i = 0; i < mybuffer3.size(); ++i){
      //   for(size_t j = 0; j < mybuffer3[0].size(); ++j){
      //     mybuffer3[i][j] = 0;
      //   }
      // }
      cnt3++;
    }).get();


  }
}

// three parallel piplines
TEST_CASE("Three.Parallel.DataPipelines.1L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.1L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(1, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.2L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(2, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.3L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(3, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.4L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(4, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.5L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(5, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.6L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(6, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.7L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(7, 8);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.1W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 1);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.2W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 2);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.3W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 3);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.4W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 4);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.5W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 5);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.6W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 6);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.7W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 7);
}

TEST_CASE("Three.Parallel.DataPipelines.8L.8W" * doctest::timeout(300)) {
  three_parallel_data_pipelines(8, 8);
}

// ----------------------------------------------------------------------------
// three concatenated pipelines. each pipeline with L lines.
// one with four pipes (SSSS), one with three pipes (SSP),
// One with two  Pipes (SP)
//
// O -> SSSS -> O -> SSP -> O -> SP -> O
//
// ----------------------------------------------------------------------------

void three_concatenated_data_pipelines(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 4>> mybuffer1(L);
  // std::vector<std::array<int, 3>> mybuffer2(L);
  // std::vector<std::array<int, 2>> mybuffer3(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;

    size_t j1_1 = 0, j1_2 = 0, j1_3 = 0, j1_4 = 0;
    size_t cnt1 = 1;

    // pipeline 1 is SSSS
    tf::DataPipeline pl1(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1_1, L](auto& pf) mutable {
        if(j1_1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1_1 == source[j1_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_1] + 1;
        // j1_1++;
        return source[j1_1++] + 1;
      }),

      tf::make_data_pipe<int, std::string>(tf::PipeType::SERIAL, [N, &source, &j1_2, L](int input, auto& pf) mutable {
        REQUIRE(j1_2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_2] + 1 == input);
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_2] + 1;
        j1_2++;
        return std::to_string(input);
      }),

      tf::make_data_pipe<std::string, int>(tf::PipeType::SERIAL, [N, &source, &j1_3, L](std::string& input, auto& pf) mutable {
        REQUIRE(j1_3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_3] + 1 == stoi(input));
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_3] + 1;
        j1_3++;
        return stoi(input);
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j1_4, L](int input, auto& pf) mutable {
        REQUIRE(j1_4 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_4] + 1 == input);
        j1_4++;
      })
    );

    auto datapipeline1 = taskflow.composed_of(pl1).name("module_of_datapipeline1");
    auto test1 = taskflow.emplace([&](){
      REQUIRE(j1_1 == N);
      REQUIRE(j1_2 == N);
      REQUIRE(j1_3 == N);
      REQUIRE(j1_4 == N);
      REQUIRE(pl1.num_tokens() == cnt1 * N);
    }).name("test1");



    // the followings are definitions for pipeline 2
    size_t j2_1 = 0, j2_2 = 0;
    std::atomic<size_t> j2_3 = 0;
    std::mutex mutex2_3;
    std::vector<int> collection2_3;
    size_t cnt2 = 1;

    // pipeline 2 is SSP
    tf::DataPipeline pl2(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j2_1, L](auto& pf) mutable {
        if(j2_1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j2_1 == source[j2_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer2[pf.line()][pf.pipe()] = source[j2_1] + 1;
        // j2_1++;
        return source[j2_1++] + 1;
      }),

      tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j2_2, L](int input, auto& pf) mutable {
        REQUIRE(j2_2 < N);
        REQUIRE(source[j2_2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer2[pf.line()][pf.pipe()] = source[j2_2] + 1;
        j2_2++;
        return input;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL, [N, &j2_3, &mutex2_3, &collection2_3, L](int input, auto& pf) mutable {
        REQUIRE(j2_3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex2_3);
          REQUIRE(pf.token() % L == pf.line());
          collection2_3.push_back(input);
        }
      })
    );

    auto datapipeline2 = taskflow.composed_of(pl2).name("module_of_datapipeline2");
    auto test2 = taskflow.emplace([&](){
      REQUIRE(j2_1 == N);
      REQUIRE(j2_2 == N);
      REQUIRE(j2_3 == N);
      REQUIRE(collection2_3.size() == N);

      std::sort(collection2_3.begin(), collection2_3.end());
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection2_3[i] == i + 1);
      }
      REQUIRE(pl2.num_tokens() == cnt2 * N);
    }).name("test2");



    // the followings are definitions for pipeline 3
    size_t j3_1 = 0;
    std::atomic<size_t> j3_2 = 0;
    std::mutex mutex3_2;
    std::vector<int> collection3_2;
    size_t cnt3 = 1;

    // pipeline 3 is SP
    tf::DataPipeline pl3(L,
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j3_1, L](auto& pf) mutable {
        if(j3_1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j3_1 == source[j3_1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer3[pf.line()][pf.pipe()] = source[j3_1] + 1;
        // j3_1++;
        return source[j3_1++] + 1;
      }),

      tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL,
      [N, &collection3_2, &mutex3_2, &j3_2, L](int input, auto& pf) mutable {
        REQUIRE(j3_2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3_2);
          REQUIRE(pf.token() % L == pf.line());
          collection3_2.push_back(input);
        }
      })
    );

    auto datapipeline3 = taskflow.composed_of(pl3).name("module_of_datapipeline3");
    auto test3 = taskflow.emplace([&](){
      REQUIRE(j3_1 == N);
      REQUIRE(j3_2 == N);

      std::sort(collection3_2.begin(), collection3_2.end());
      for(size_t i = 0; i < N; i++) {
        REQUIRE(collection3_2[i] == i + 1);
      }
      REQUIRE(pl3.num_tokens() == cnt3 * N);
    }).name("test3");


    auto initial  = taskflow.emplace([](){}).name("initial");
    auto terminal = taskflow.emplace([](){}).name("terminal");

    initial.precede(datapipeline1);
    datapipeline1.precede(test1);
    test1.precede(datapipeline2);
    datapipeline2.precede(test2);
    test2.precede(datapipeline3);
    datapipeline3.precede(test3);
    test3.precede(terminal);

    //taskflow.dump(std::cout);

    executor.run_n(taskflow, 3, [&]() mutable {
      // reset variables for pipeline 1
      j1_1 = j1_2 = j1_3 = j1_4 = 0;
      // for(size_t i = 0; i < mybuffer1.size(); ++i){
      //   for(size_t j = 0; j < mybuffer1[0].size(); ++j){
      //     mybuffer1[i][j] = 0;
      //   }
      // }
      cnt1++;

      // reset variables for pipeline 2
      j2_1 = j2_2 = j2_3 = 0;
      collection2_3.clear();
      // for(size_t i = 0; i < mybuffer2.size(); ++i){
      //   for(size_t j = 0; j < mybuffer2[0].size(); ++j){
      //     mybuffer2[i][j] = 0;
      //   }
      // }
      cnt2++;

      // reset variables for pipeline 3
      j3_1 = j3_2 = 0;
      collection3_2.clear();
      // for(size_t i = 0; i < mybuffer3.size(); ++i){
      //   for(size_t j = 0; j < mybuffer3[0].size(); ++j){
      //     mybuffer3[i][j] = 0;
      //   }
      // }
      cnt3++;
    }).get();


  }
}

// three concatenated piplines
TEST_CASE("Three.Concatenated.DataPipelines.1L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.1L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(1, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.2L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(2, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.3L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(3, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.4L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(4, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.5L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(5, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.6L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(6, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.7L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(7, 8);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.1W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 1);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.2W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 2);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.3W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 3);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.4W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 4);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.5W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 5);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.6W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 6);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.7W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 7);
}

TEST_CASE("Three.Concatenated.DataPipelines.8L.8W" * doctest::timeout(300)) {
  three_concatenated_data_pipelines(8, 8);
}

// ----------------------------------------------------------------------------
// pipeline (SPSP) and conditional task.  pipeline has L lines, W workers
//
// O -> SPSP -> conditional_task
//        ^            |
//        |____________|
// ----------------------------------------------------------------------------

void looping_data_pipelines(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  // std::vector<std::array<int, 4>> mybuffer(L);

  tf::Taskflow taskflow;

  size_t j1 = 0, j3 = 0;
  std::atomic<size_t> j2 = 0;
  std::atomic<size_t> j4 = 0;
  std::mutex mutex2;
  std::mutex mutex4;
  std::vector<int> collection2;
  std::vector<int> collection4;
  size_t cnt = 0;

  size_t N = 0;

  tf::DataPipeline pl(L,
    tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [&N, &source, &j1, L](auto& pf) mutable {
      if(j1 == N) {
        pf.stop();
        return 0;
      }
      REQUIRE(j1 == source[j1]);
      REQUIRE(pf.token() % L == pf.line());
      // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
      // j1++;
      return source[j1++] + 1;
    }),

    tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&N, &j2, &mutex2, &collection2, L](int input, auto& pf) mutable {
      REQUIRE(j2++ < N);
      {
        std::scoped_lock<std::mutex> lock(mutex2);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
        collection2.push_back(input);
        return input + 1;
      }
    }),

    tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [&N, &source, &j3, L](int input, auto& pf) mutable {
      REQUIRE(j3 < N);
      REQUIRE(pf.token() % L == pf.line());
      REQUIRE(source[j3] + 2 == input);
      // mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
      j3++;
      return input + 1;
    }),

    tf::make_data_pipe<int, void>(tf::PipeType::PARALLEL, [&N, &j4, &mutex4, &collection4, L](int input, auto& pf) mutable {
      REQUIRE(j4++ < N);
      {
        std::scoped_lock<std::mutex> lock(mutex4);
        REQUIRE(pf.token() % L == pf.line());
        collection4.push_back(input);
      }
    })
  );

  auto datapipeline = taskflow.composed_of(pl).name("module_of_datapipeline");
  auto initial = taskflow.emplace([](){}).name("initial");

  auto conditional = taskflow.emplace([&](){
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection4[i] == i + 3);
    }
    REQUIRE(pl.num_tokens() == cnt);

    // reset variables
    j1 = j2 = j3 = j4 = 0;
    // for(size_t i = 0; i < mybuffer.size(); ++i){
    //   for(size_t j = 0; j < mybuffer[0].size(); ++j){
    //     mybuffer[i][j] = 0;
    //   }
    // }
    collection2.clear();
    collection4.clear();
    ++N;
    cnt+=N;

    return N < maxN ? 0 : 1;
  }).name("conditional");

  auto terminal = taskflow.emplace([](){}).name("terminal");

  initial.precede(datapipeline);
  datapipeline.precede(conditional);
  conditional.precede(datapipeline, terminal);

  executor.run(taskflow).wait();
}

// looping piplines
TEST_CASE("Looping.DataPipelines.1L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 1);
}

TEST_CASE("Looping.DataPipelines.1L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 2);
}

TEST_CASE("Looping.DataPipelines.1L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 3);
}

TEST_CASE("Looping.DataPipelines.1L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 4);
}

TEST_CASE("Looping.DataPipelines.1L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 5);
}

TEST_CASE("Looping.DataPipelines.1L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 6);
}

TEST_CASE("Looping.DataPipelines.1L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 7);
}

TEST_CASE("Looping.DataPipelines.1L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(1, 8);
}

TEST_CASE("Looping.DataPipelines.2L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 1);
}

TEST_CASE("Looping.DataPipelines.2L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 2);
}

TEST_CASE("Looping.DataPipelines.2L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 3);
}

TEST_CASE("Looping.DataPipelines.2L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 4);
}

TEST_CASE("Looping.DataPipelines.2L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 5);
}

TEST_CASE("Looping.DataPipelines.2L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 6);
}

TEST_CASE("Looping.DataPipelines.2L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 7);
}

TEST_CASE("Looping.DataPipelines.2L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(2, 8);
}

TEST_CASE("Looping.DataPipelines.3L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 1);
}

TEST_CASE("Looping.DataPipelines.3L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 2);
}

TEST_CASE("Looping.DataPipelines.3L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 3);
}

TEST_CASE("Looping.DataPipelines.3L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 4);
}

TEST_CASE("Looping.DataPipelines.3L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 5);
}

TEST_CASE("Looping.DataPipelines.3L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 6);
}

TEST_CASE("Looping.DataPipelines.3L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 7);
}

TEST_CASE("Looping.DataPipelines.3L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(3, 8);
}

TEST_CASE("Looping.DataPipelines.4L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 1);
}

TEST_CASE("Looping.DataPipelines.4L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 2);
}

TEST_CASE("Looping.DataPipelines.4L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 3);
}

TEST_CASE("Looping.DataPipelines.4L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 4);
}

TEST_CASE("Looping.DataPipelines.4L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 5);
}

TEST_CASE("Looping.DataPipelines.4L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 6);
}

TEST_CASE("Looping.DataPipelines.4L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 7);
}

TEST_CASE("Looping.DataPipelines.4L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(4, 8);
}

TEST_CASE("Looping.DataPipelines.5L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 1);
}

TEST_CASE("Looping.DataPipelines.5L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 2);
}

TEST_CASE("Looping.DataPipelines.5L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 3);
}

TEST_CASE("Looping.DataPipelines.5L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 4);
}

TEST_CASE("Looping.DataPipelines.5L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 5);
}

TEST_CASE("Looping.DataPipelines.5L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 6);
}

TEST_CASE("Looping.DataPipelines.5L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 7);
}

TEST_CASE("Looping.DataPipelines.5L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(5, 8);
}

TEST_CASE("Looping.DataPipelines.6L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 1);
}

TEST_CASE("Looping.DataPipelines.6L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 2);
}

TEST_CASE("Looping.DataPipelines.6L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 3);
}

TEST_CASE("Looping.DataPipelines.6L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 4);
}

TEST_CASE("Looping.DataPipelines.6L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 5);
}

TEST_CASE("Looping.DataPipelines.6L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 6);
}

TEST_CASE("Looping.DataPipelines.6L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 7);
}

TEST_CASE("Looping.DataPipelines.6L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(6, 8);
}

TEST_CASE("Looping.DataPipelines.7L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 1);
}

TEST_CASE("Looping.DataPipelines.7L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 2);
}

TEST_CASE("Looping.DataPipelines.7L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 3);
}

TEST_CASE("Looping.DataPipelines.7L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 4);
}

TEST_CASE("Looping.DataPipelines.7L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 5);
}

TEST_CASE("Looping.DataPipelines.7L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 6);
}

TEST_CASE("Looping.DataPipelines.7L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 7);
}

TEST_CASE("Looping.DataPipelines.7L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(7, 8);
}

TEST_CASE("Looping.DataPipelines.8L.1W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 1);
}

TEST_CASE("Looping.DataPipelines.8L.2W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 2);
}

TEST_CASE("Looping.DataPipelines.8L.3W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 3);
}

TEST_CASE("Looping.DataPipelines.8L.4W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 4);
}

TEST_CASE("Looping.DataPipelines.8L.5W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 5);
}

TEST_CASE("Looping.DataPipelines.8L.6W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 6);
}

TEST_CASE("Looping.DataPipelines.8L.7W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 7);
}

TEST_CASE("Looping.DataPipelines.8L.8W" * doctest::timeout(300)) {
  looping_data_pipelines(8, 8);
}

// ----------------------------------------------------------------------------
//
// ifelse pipeline has three pipes, L lines, w workers
//
// SPS
// ----------------------------------------------------------------------------

int ifelse_pipe_ans(int a) {
  // pipe 1
  if(a / 2 != 0) {
    a += 8;
  }
  // pipe 2
  if(a > 4897) {
    a -= 1834;
  }
  else {
    a += 3;
  }
  // pipe 3
  if((a + 9) / 4 < 50) {
    a += 1;
  }
  else {
    a += 17;
  }

  return a;
}

void ifelse_data_pipeline(size_t L, unsigned w) {
  srand(time(NULL));

  tf::Executor executor(w);
  size_t maxN = 200;

  std::vector<int> source(maxN);
  for(auto&& s: source) {
    s = rand() % 9962;
  }
  // std::vector<std::array<int, 4>> buffer(L);

  for(size_t N = 1; N < maxN; ++N) {
    tf::Taskflow taskflow;

    std::vector<int> collection;
    collection.reserve(N);

    tf::DataPipeline pl(L,
      // pipe 1
      tf::make_data_pipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [&, N](auto& pf){
        if(pf.token() == N) {
          pf.stop();
          return 0;
        }

        if(source[pf.token()] / 2 == 0) {
          // buffer[pf.line()][pf.pipe()] = source[pf.token()];
          return source[pf.token()];
        }
        else {
          // buffer[pf.line()][pf.pipe()] = source[pf.token()] + 8;
          return source[pf.token()] + 8;
        }

      }),

      // pipe 2
      tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&](int input){

        if(input > 4897) {
          // buffer[pf.line()][pf.pipe()] =  buffer[pf.line()][pf.pipe() - 1] - 1834;
          return input - 1834;
        }
        else {
          // buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 3;
          return input + 3;
        }

      }),

      // pipe 3
      tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [&](int input){
        int tmp = 0;
        if((input + 9) / 4 < 50) {
          // buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 1;
          tmp = input + 1;
        }
        else {
          // buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 17;
          tmp = input + 17;
        }

        collection.push_back(tmp);

      })
    );
    auto pl_t = taskflow.composed_of(pl).name("datapipeline");

    auto check_t = taskflow.emplace([&](){
      for(size_t n = 0; n < N; ++n) {
        REQUIRE(collection[n] == ifelse_pipe_ans(source[n]));
      }
    }).name("check");

    pl_t.precede(check_t);

    executor.run(taskflow).wait();

  }
}

TEST_CASE("Ifelse.DataPipelines.1L.1W" * doctest::timeout(300)) {
  ifelse_data_pipeline(1, 1);
}

TEST_CASE("Ifelse.DataPipelines.1L.2W" * doctest::timeout(300)) {
  ifelse_data_pipeline(1, 2);
}

TEST_CASE("Ifelse.DataPipelines.1L.3W" * doctest::timeout(300)) {
  ifelse_data_pipeline(1, 3);
}

TEST_CASE("Ifelse.DataPipelines.1L.4W" * doctest::timeout(300)) {
  ifelse_data_pipeline(1, 4);
}

TEST_CASE("Ifelse.DataPipelines.3L.1W" * doctest::timeout(300)) {
  ifelse_data_pipeline(3, 1);
}

TEST_CASE("Ifelse.DataPipelines.3L.2W" * doctest::timeout(300)) {
  ifelse_data_pipeline(3, 2);
}

TEST_CASE("Ifelse.DataPipelines.3L.3W" * doctest::timeout(300)) {
  ifelse_data_pipeline(3, 3);
}

TEST_CASE("Ifelse.DataPipelines.3L.4W" * doctest::timeout(300)) {
  ifelse_data_pipeline(3, 4);
}

TEST_CASE("Ifelse.DataPipelines.5L.1W" * doctest::timeout(300)) {
  ifelse_data_pipeline(5, 1);
}

TEST_CASE("Ifelse.DataPipelines.5L.2W" * doctest::timeout(300)) {
  ifelse_data_pipeline(5, 2);
}

TEST_CASE("Ifelse.DataPipelines.5L.3W" * doctest::timeout(300)) {
  ifelse_data_pipeline(5, 3);
}

TEST_CASE("Ifelse.DataPipelines.5L.4W" * doctest::timeout(300)) {
  ifelse_data_pipeline(5, 4);
}

TEST_CASE("Ifelse.DataPipelines.7L.1W" * doctest::timeout(300)) {
  ifelse_data_pipeline(7, 1);
}

TEST_CASE("Ifelse.DataPipelines.7L.2W" * doctest::timeout(300)) {
  ifelse_data_pipeline(7, 2);
}

TEST_CASE("Ifelse.DataPipelines.7L.3W" * doctest::timeout(300)) {
  ifelse_data_pipeline(7, 3);
}

TEST_CASE("Ifelse.DataPipelines.7L.4W" * doctest::timeout(300)) {
  ifelse_data_pipeline(7, 4);
}

// ----------------------------------------------------------------------------
// pipeline in pipeline
// pipeline has 4 pipes, L lines, W workers
// each subpipeline has 3 pipes, subL lines
//
// pipeline = SPPS
// each subpipeline = SPS
//
// ----------------------------------------------------------------------------

void pipeline_in_pipeline(size_t L, unsigned w, unsigned subL) {


  tf::Executor executor(w);

  const size_t maxN = 5;
  const size_t maxsubN = 4;

  std::vector<std::vector<int>> source(maxN);
  for(auto&& each: source) {
    each.resize(maxsubN);
    std::iota(each.begin(), each.end(), 0);
  }

  for (size_t N = 1; N < maxN; ++N) {
    for(size_t subN = 1; subN < maxsubN; ++subN) {

      size_t j1 = 0, j4 = 0;
      std::atomic<size_t> j2 = 0;
      std::atomic<size_t> j3 = 0;

      // begin of pipeline ---------------------------
      tf::DataPipeline pl(L,

        // begin of pipe 1 -----------------------------
        tf::make_data_pipe<void, int>(tf::PipeType::SERIAL, [&, N, subN, subL](auto& pf) mutable {
          if(j1 == N) {
            pf.stop();
            return 0;
          }

          size_t subj1 = 0, subj3 = 0;
          std::atomic<size_t> subj2 = 0;
          std::vector<int> subcollection;
          subcollection.reserve(subN);

          // subpipeline
          tf::DataPipeline subpl(subL,

            // subpipe 1
            tf::make_data_pipe<void, int>(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
              if(subj1 == subN) {
                subpf.stop();
                return 0;
              }
              REQUIRE(subpf.token() % subL == subpf.line());
              return source[pf.token()][subj1++] + 1;
            }),

            // subpipe 2
            tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&, subN](int input, auto& subpf) mutable {
              REQUIRE(subj2++ < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[pf.token()][subpf.token()] + 1 == input);
              return input;
            }),


            // subpipe 3
            tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [&, subN](int input, auto& subpf) mutable {
              REQUIRE(subj3 < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[pf.token()][subj3] + 1 == input);
              subcollection.push_back(input + 2);
              ++subj3;
            })
          );

          tf::Taskflow taskflow;

          // test task
          auto test_t = taskflow.emplace([&, subN](){
            REQUIRE(subj1 == subN);
            REQUIRE(subj2 == subN);
            REQUIRE(subj3 == subN);
            //REQUIRE(subpl.num_tokens() == subN);
            REQUIRE(subcollection.size() == subN);
          }).name("test");

          // subpipeline
          auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

          subpl_t.precede(test_t);
          executor.corun(taskflow);

          j1++;

          return std::accumulate(
            subcollection.begin(),
            subcollection.end(),
            0
          );
        }),
        // end of pipe 1 -----------------------------

         //begin of pipe 2 ---------------------------
        tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&, N, subN, subL](int input, auto& pf) mutable {
          REQUIRE(j2++ < N);
          int res = std::accumulate(
            source[pf.token()].begin(),
            source[pf.token()].begin() + subN,
            0
          );
          REQUIRE(input == res + 3 * subN);

          size_t subj1 = 0, subj3 = 0;
          std::atomic<size_t> subj2 = 0;
          std::vector<int> subcollection;
          subcollection.reserve(subN);

          // subpipeline
          tf::DataPipeline subpl(subL,

            // subpipe 1
            tf::make_data_pipe<void, int>(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
              if(subj1 == subN) {
                subpf.stop();
                return 0;
              }
              REQUIRE(subpf.token() % subL == subpf.line());
              return source[pf.token()][subj1++] + 1;
            }),

            // subpipe 2
            tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&, subN](int in, auto& subpf) mutable {
              REQUIRE(subj2++ < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[j2][subpf.token()] + 1 == in);
              return in;
            }),


            // subpipe 3
            tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [&, subN](int in, auto& subpf) mutable {
              REQUIRE(subj3 < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[pf.token()][subj3] + 1 == in);
              subcollection.push_back(in + 12);
              ++subj3;
            })
          );

          tf::Taskflow taskflow;

          // test task
          auto test_t = taskflow.emplace([&, subN](){
            REQUIRE(subj1 == subN);
            REQUIRE(subj2 == subN);
            REQUIRE(subj3 == subN);
            //REQUIRE(subpl.num_tokens() == subN);
            REQUIRE(subcollection.size() == subN);
          }).name("test");

          // subpipeline
          auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

          subpl_t.precede(test_t);
          executor.corun(taskflow);

          return std::accumulate(
            subcollection.begin(),
            subcollection.end(),
            0
          );

        }),
        // end of pipe 2 -----------------------------

        // begin of pipe 3 ---------------------------
        tf::make_data_pipe<int, int>(tf::PipeType::SERIAL, [&, N, subN, subL](int input, auto& pf) mutable {

          REQUIRE(j3++ < N);
          int res = std::accumulate(
            source[pf.token()].begin(),
            source[pf.token()].begin() + subN,
            0
          );

          REQUIRE(input == res + 13 * subN);

          size_t subj1 = 0, subj3 = 0;
          std::atomic<size_t> subj2 = 0;
          std::vector<int> subcollection;
          subcollection.reserve(subN);

          // subpipeline
          tf::DataPipeline subpl(subL,

            // subpipe 1
            tf::make_data_pipe<void, int>(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
              if(subj1 == subN) {
                subpf.stop();
                return 0;
              }
              REQUIRE(subpf.token() % subL == subpf.line());
              return source[pf.token()][subj1++] + 1;
            }),

            // subpipe 2
            tf::make_data_pipe<int, int>(tf::PipeType::PARALLEL, [&, subN](int in, auto& subpf) mutable {
              REQUIRE(subj2++ < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[pf.token()][subpf.token()] + 1 == in);
              return in;
            }),


            // subpipe 3
            tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [&, subN](int in, auto& subpf) mutable {
              REQUIRE(subj3 < subN);
              REQUIRE(subpf.token() % subL == subpf.line());
              REQUIRE(source[pf.token()][subj3] + 1 == in);
              subcollection.push_back(in + 6);
              ++subj3;
            })
          );

          tf::Taskflow taskflow;

          // test task
          auto test_t = taskflow.emplace([&, subN](){
            REQUIRE(subj1 == subN);
            REQUIRE(subj2 == subN);
            REQUIRE(subj3 == subN);
            //REQUIRE(subpl.num_tokens() == subN);
            REQUIRE(subcollection.size() == subN);
          }).name("test");

          // subpipeline
          auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

          subpl_t.precede(test_t);
          executor.corun(taskflow);

          return std::accumulate(
            subcollection.begin(),
            subcollection.end(),
            0
          );

        }),
        // end of pipe 3 -----------------------------

        // begin of pipe 4 ---------------------------
        tf::make_data_pipe<int, void>(tf::PipeType::SERIAL, [&, subN](int input) mutable {

          int res = std::accumulate(
            source[j4].begin(),
            source[j4].begin() + subN,
            0
          );
          REQUIRE(input == res + 7 * subN);
          j4++;
        })
        // end of pipe 4 -----------------------------
      );

      tf::Taskflow taskflow;
      taskflow.composed_of(pl).name("module_of_pipeline");
      executor.run(taskflow).wait();
    }
  }
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.1W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 1, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.1W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 1, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.1W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 1, 4);
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.2W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 2, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.2W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 2, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.1L.2W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(1, 2, 4);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.1W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 1, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.1W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 1, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.1W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 1, 4);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.2W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 2, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.2W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 2, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.3L.2W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(3, 2, 4);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.1W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 1, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.1W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 1, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.1W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 1, 4);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.2W.1subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 2, 1);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.2W.3subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 2, 3);
}

TEST_CASE("PipelineinPipeline.DataPipelines.5L.2W.4subL" * doctest::timeout(300)) {
  pipeline_in_pipeline(5, 2, 4);
}

