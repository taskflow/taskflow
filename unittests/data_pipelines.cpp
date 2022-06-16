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
      tf::DataPipeline pl (L, tf::DataPipe<tf::Pipeflow&, void>{type, [L, N, &j, &source](auto& pf) mutable {
        if (j == N) {
          pf.stop();
          return;
        }
        REQUIRE(j == source[j]);
        REQUIRE(pf.token() % L == pf.line());
        j++;
      }});

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
// TEST_CASE("DataPipeline.1P(S).1L.1W" * doctest::timeout(300)) {
//   data_pipeline_1P(1, 1, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).1L.2W" * doctest::timeout(300)) {
//   data_pipeline_1P(1, 2, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).1L.3W" * doctest::timeout(300)) {
//   data_pipeline_1P(1, 3, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).1L.4W" * doctest::timeout(300)) {
//   data_pipeline_1P(1, 4, tf::PipeType::SERIAL);
// }

// // serial pipe with two lines
// TEST_CASE("DataPipeline.1P(S).2L.1W" * doctest::timeout(300)) {
//   data_pipeline_1P(2, 1, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).2L.2W" * doctest::timeout(300)) {
//   data_pipeline_1P(2, 2, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).2L.3W" * doctest::timeout(300)) {
//   data_pipeline_1P(2, 3, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).2L.4W" * doctest::timeout(300)) {
//   data_pipeline_1P(2, 4, tf::PipeType::SERIAL);
// }

// // serial pipe with three lines
// TEST_CASE("DataPipeline.1P(S).3L.1W" * doctest::timeout(300)) {
//   data_pipeline_1P(3, 1, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).3L.2W" * doctest::timeout(300)) {
//   data_pipeline_1P(3, 2, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).3L.3W" * doctest::timeout(300)) {
//   data_pipeline_1P(3, 3, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).3L.4W" * doctest::timeout(300)) {
//   data_pipeline_1P(3, 4, tf::PipeType::SERIAL);
// }

// // serial pipe with three lines
// TEST_CASE("DataPipeline.1P(S).4L.1W" * doctest::timeout(300)) {
//   data_pipeline_1P(4, 1, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).4L.2W" * doctest::timeout(300)) {
//   data_pipeline_1P(4, 2, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).4L.3W" * doctest::timeout(300)) {
//   data_pipeline_1P(4, 3, tf::PipeType::SERIAL);
// }

// TEST_CASE("DataPipeline.1P(S).4L.4W" * doctest::timeout(300)) {
//   data_pipeline_1P(4, 4, tf::PipeType::SERIAL);
// }

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
      tf::DataPipe<tf::Pipeflow&, int>{tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }},

      tf::DataPipe<int, void>{tf::PipeType::SERIAL, [N, &source, &j2, L](int input) mutable {
        REQUIRE(j2 < N);
        // REQUIRE(pf.token() % L == pf.line());
        // REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // j2++;
        REQUIRE(source[j2++] + 1 == input);
      }}
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
// TEST_CASE("DataPipeline.2P(SS).1L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(1, 1);
// }

// TEST_CASE("DataPipeline.2P(SS).1L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(1, 2);
// }

// TEST_CASE("DataPipeline.2P(SS).1L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(1, 3);
// }

// TEST_CASE("DataPipeline.2P(SS).1L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(1, 4);
// }

// TEST_CASE("DataPipeline.2P(SS).2L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(2, 1);
// }

// TEST_CASE("DataPipeline.2P(SS).2L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(2, 2);
// }

// TEST_CASE("DataPipeline.2P(SS).2L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(2, 3);
// }

// TEST_CASE("DataPipeline.2P(SS).2L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(2, 4);
// }

// TEST_CASE("DataPipeline.2P(SS).3L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(3, 1);
// }

// TEST_CASE("DataPipeline.2P(SS).3L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(3, 2);
// }

// TEST_CASE("DataPipeline.2P(SS).3L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(3, 3);
// }

// TEST_CASE("DataPipeline.2P(SS).3L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(3, 4);
// }

// TEST_CASE("DataPipeline.2P(SS).4L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(4, 1);
// }

// TEST_CASE("DataPipeline.2P(SS).4L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(4, 2);
// }

// TEST_CASE("DataPipeline.2P(SS).4L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(4, 3);
// }

// TEST_CASE("DataPipeline.2P(SS).4L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SS(4, 4);
// }

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
      tf::DataPipe<tf::Pipeflow&, int>{tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }},

      tf::DataPipe<int, void>{tf::PipeType::PARALLEL,
      [N, &collection, &mutex, &j2, L](int input) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          // REQUIRE(pf.token() % L == pf.line());
          // collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          collection.push_back(input);
        }
      }}
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
// TEST_CASE("DataPipeline.2P(SP).1L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(1, 1);
// }

// TEST_CASE("DataPipeline.2P(SP).1L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(1, 2);
// }

// TEST_CASE("DataPipeline.2P(SP).1L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(1, 3);
// }

// TEST_CASE("DataPipeline.2P(SP).1L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(1, 4);
// }

// TEST_CASE("DataPipeline.2P(SP).2L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(2, 1);
// }

// TEST_CASE("DataPipeline.2P(SP).2L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(2, 2);
// }

// TEST_CASE("DataPipeline.2P(SP).2L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(2, 3);
// }

// TEST_CASE("DataPipeline.2P(SP).2L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(2, 4);
// }

// TEST_CASE("DataPipeline.2P(SP).3L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(3, 1);
// }

// TEST_CASE("DataPipeline.2P(SP).3L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(3, 2);
// }

// TEST_CASE("DataPipeline.2P(SP).3L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(3, 3);
// }

// TEST_CASE("DataPipeline.2P(SP).3L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(3, 4);
// }

// TEST_CASE("DataPipeline.2P(SP).4L.1W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(4, 1);
// }

// TEST_CASE("DataPipeline.2P(SP).4L.2W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(4, 2);
// }

// TEST_CASE("DataPipeline.2P(SP).4L.3W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(4, 3);
// }

// TEST_CASE("DataPipeline.2P(SP).4L.4W" * doctest::timeout(300)) {
//   data_pipeline_2P_SP(4, 4);
// }

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
      tf::DataPipe<tf::Pipeflow&, int>{tf::PipeType::SERIAL, [N, &source, &j1, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return 0;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // j1++;
        return source[j1++] + 1;
      }},

      tf::DataPipe<int, int>{tf::PipeType::SERIAL, [N, &source, &j2, L](int input) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == input);
        // REQUIRE(pf.token() % L == pf.line());
        // mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
        return input;
      }},

      tf::DataPipe<int, void>{tf::PipeType::SERIAL, [N, &source, &j3, L](int input) mutable {
        REQUIRE(j3 < N);
        REQUIRE(source[j3] + 1 == input);
        // REQUIRE(pf.token() % L == pf.line());
        j3++;
      }}
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
