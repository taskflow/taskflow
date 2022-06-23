#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

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
      tf::DataPipeline pl (L, tf::make_datapipe<tf::Pipeflow&, void>(type, [L, N, &j, &source](auto& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
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

      tf::make_datapipe<int&, void>(tf::PipeType::SERIAL, [N, &source, &j2, L](int& input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
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

      tf::make_datapipe<int&, void>(tf::PipeType::PARALLEL,
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
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

      tf::make_datapipe<int, std::string>(tf::PipeType::SERIAL, [N, &source, &j2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
        return std::to_string(input);
      }),

      tf::make_datapipe<std::string, void>(tf::PipeType::SERIAL, [N, &source, &j3, L](std::string input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
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

      tf::make_datapipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j2] + 1;
        // mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
        return input;
      }),

      tf::make_datapipe<int, void>(tf::PipeType::PARALLEL, [N, &j3, &mutex, &collection, L](int input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, int>(tf::PipeType::PARALLEL, [N, &j2, &mutex, &collection, L](int input, tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j3, L](int input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1, L](tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, int>(tf::PipeType::PARALLEL, [N, &j2, &mutex2, &collection2, L](int input, tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, void>(tf::PipeType::PARALLEL, [N, &j3, &mutex3, &collection3, L](int input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j1_1, L](tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j1_2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j1_2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_2] + 1 == input);
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_2] + 1;
        j1_2++;
        return input;
      }),

      tf::make_datapipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j1_3, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j1_3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j1_3] + 1 == input);
        // mybuffer1[pf.line()][pf.pipe()] = source[j1_3] + 1;
        j1_3++;
        return input;
      }),

      tf::make_datapipe<int, void>(tf::PipeType::SERIAL, [N, &source, &j1_4, L](int input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j2_1, L](tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, int>(tf::PipeType::SERIAL, [N, &source, &j2_2, L](int input, tf::Pipeflow& pf) mutable {
        REQUIRE(j2_2 < N);
        REQUIRE(source[j2_2] + 1 == input);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer2[pf.line()][pf.pipe()] = source[j2_2] + 1;
        j2_2++;
        return input;
      }),

      tf::make_datapipe<int, void>(tf::PipeType::PARALLEL, [N, &j2_3, &mutex2_3, &collection2_3, L](int input, tf::Pipeflow& pf) mutable {
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
      tf::make_datapipe<tf::Pipeflow&, int>(tf::PipeType::SERIAL, [N, &source, &j3_1, L](tf::Pipeflow& pf) mutable {
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

      tf::make_datapipe<int, void>(tf::PipeType::PARALLEL,
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
