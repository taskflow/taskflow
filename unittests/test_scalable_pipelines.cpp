#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

// ----------------------------------------------------------------------------
// Constructors and Assignments
// ----------------------------------------------------------------------------

TEST_CASE("ScalablePipeline.Basics" * doctest::timeout(300)) {

  size_t N = 10;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;

  for(size_t i=0; i<N; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow&) {});
  }

  using iterator_type = decltype(pipes)::iterator;

  tf::ScalablePipeline<iterator_type> rhs;

  REQUIRE(rhs.num_lines()  == 0);
  REQUIRE(rhs.num_pipes()  == 0);
  REQUIRE(rhs.num_tokens() == 0);

  rhs.reset(1, pipes.begin(), pipes.end());

  REQUIRE(rhs.num_lines()  == 1);
  REQUIRE(rhs.num_pipes()  == N);
  REQUIRE(rhs.num_tokens() == 0);

  tf::ScalablePipeline<iterator_type> lhs(std::move(rhs));

  REQUIRE(rhs.num_lines()  == 0);
  REQUIRE(rhs.num_pipes()  == 0);
  REQUIRE(rhs.num_tokens() == 0);
  REQUIRE(lhs.num_lines()  == 1);
  REQUIRE(lhs.num_pipes()  == N);
  REQUIRE(lhs.num_tokens() == 0);

  rhs = std::move(lhs);

  REQUIRE(lhs.num_lines()  == 0);
  REQUIRE(lhs.num_pipes()  == 0);
  REQUIRE(lhs.num_tokens() == 0);
  REQUIRE(rhs.num_lines()  == 1);
  REQUIRE(rhs.num_pipes()  == N);
  REQUIRE(rhs.num_tokens() == 0);
}

// ----------------------------------------------------------------------------
// Scalable Pipeline
// ----------------------------------------------------------------------------

void scalable_pipeline(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);

  for(size_t i=0; i<num_pipes; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable {

      switch(pf.pipe()) {
        case 0:
          if(pf.token() == 1111) {
            pf.stop();
            return;
          }
          data[pf.line()] = num_pipes * pf.token();
        break;

        default: {
          ++data[pf.line()];
        }
        break;
      }
      //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
      REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));
      if(pf.pipe() == num_pipes - 1) {
        N++;
      }
    });
  }

  tf::ScalablePipeline spl(num_lines, pipes.begin(), pipes.end());
  taskflow.composed_of(spl);
  executor.run(taskflow).wait();

  REQUIRE(N == 1111);
}

TEST_CASE("ScalablePipeline" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline(L, P);
    }
  }
}

// ----------------------------------------------------------------------------
// Scalable Pipeline using Reset
// ----------------------------------------------------------------------------

void scalable_pipeline_reset(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);

  tf::ScalablePipeline<typename decltype(pipes)::iterator> spl(num_lines);

  auto init = taskflow.emplace([&](){
    for(size_t i=0; i<num_pipes; i++) {
      pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable {

        switch(pf.pipe()) {
          case 0:
            if(pf.token() == 1111) {
              pf.stop();
              return;
            }
            data[pf.line()] = num_pipes * pf.token();
          break;

          default: {
            ++data[pf.line()];
          }
          break;
        }
        //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
        REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));

        if(pf.pipe() == num_pipes - 1) {
          N++;
        }
      });
    }
    spl.reset(pipes.begin(), pipes.end());
  });

  auto pipeline = taskflow.composed_of(spl);
  pipeline.succeed(init);
  executor.run(taskflow).wait();

  REQUIRE(N == 1111);
}

TEST_CASE("ScalablePipeline.Reset" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline_reset(L, P);
    }
  }
}

// ----------------------------------------------------------------------------
// Scalable Pipeline using Iterative Reset
// ----------------------------------------------------------------------------

void scalable_pipeline_iterative_reset(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;
  tf::Taskflow taskflow;

  size_t N = 0;

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;
  std::vector< int > data(num_lines, -1);

  tf::ScalablePipeline<typename decltype(pipes)::iterator> spl(num_lines);

  auto init = taskflow.emplace([&](){
    for(size_t i=0; i<num_pipes; i++) {
      pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable {

        switch(pf.pipe()) {
          case 0:
            if(pf.token() == 1111) {
              pf.stop();
              return;
            }
            data[pf.line()] = num_pipes * pf.token();
          break;

          default: {
            ++data[pf.line()];
          }
          break;
        }
        //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
        REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));

        if(pf.pipe() == num_pipes - 1) {
          N++;
        }
      });
    }
    spl.reset(pipes.begin(), pipes.end());
  });

  auto cond = taskflow.emplace([&, i=0]()mutable{
    REQUIRE(N == 1111*(i+1));
    spl.reset();
    return (i++ < 3) ? 0 : -1;
  });

  auto pipeline = taskflow.composed_of(spl);
  pipeline.succeed(init)
          .precede(cond);
  cond.precede(pipeline);
  executor.run(taskflow).wait();
}

TEST_CASE("ScalablePipeline.IterativeReset" * doctest::timeout(300)) {
  for(size_t L=1; L<=10; L++) {
    for(size_t P=1; P<=10; P++) {
      scalable_pipeline_iterative_reset(L, P);
    }
  }
}

// ----------------------------------------------------------------------------
// Scalable Pipeline Reset
//
// reset(num_lines, pipes.begin(), pipes.end())
// ----------------------------------------------------------------------------

void scalable_pipeline_lines_reset(size_t num_lines, size_t num_pipes) {

  tf::Executor executor;

  size_t N = 0;

  std::vector<tf::Pipe<>> pipes;
  tf::ScalablePipeline<typename decltype(pipes)::iterator> spl;

  for(size_t l = 1; l <= num_lines; ++l) {
    tf::Taskflow taskflow;
    std::vector<int> data(l, -1);

    auto init = taskflow.emplace([&](){
      for(size_t i=0; i<num_pipes; i++) {
        pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) mutable {

          switch(pf.pipe()) {
            case 0:
              if(pf.token() == 1111) {
                pf.stop();
                return;
              }
              data[pf.line()] = num_pipes * pf.token();
            break;

            default: {
              ++data[pf.line()];
            }
            break;
          }
          //printf("data[%zu]=%d\n", pf.line(), data[pf.line()]);
          REQUIRE(data[pf.line()] == (pf.token() * num_pipes + pf.pipe()));

          if(pf.pipe() == num_pipes - 1) {
            N++;
          }
        });
      }
      spl.reset(l, pipes.begin(), pipes.end());
    });

    auto check = taskflow.emplace([&]()mutable{
      REQUIRE(N == 1111 * l);
      pipes.clear();
    });

    auto pipeline = taskflow.composed_of(spl);
    pipeline.succeed(init)
            .precede(check);
    executor.run(taskflow).wait();
  }

}

TEST_CASE("ScalablePipeline.LinesReset" * doctest::timeout(300)) {
  for(size_t P=1; P<=10; P++) {
    scalable_pipeline_lines_reset(10, P);
  }
}

// ----------------------------------------------------------------------------
//
// ifelse ScalablePipeline has three pipes, L lines, w workers
//
// SPS
// ----------------------------------------------------------------------------

int ifelse_spipe_ans(int a) {
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

void ifelse_spipeline(size_t L, unsigned w) {
  srand(time(NULL));

  tf::Executor executor(w);
  size_t maxN = 200;

  std::vector<int> source(maxN);
  for(auto&& s: source) {
    s = rand() % 9962;
  }
  std::vector<std::array<int, 4>> buffer(L);

  std::vector<tf::Pipe<>> pipes;
  tf::ScalablePipeline<typename decltype(pipes)::iterator> pl;

  for(size_t N = 1; N < maxN; ++N) {
    tf::Taskflow taskflow;

    std::vector<int> collection;
    collection.reserve(N);

    // pipe 1
    pipes.emplace_back(tf::PipeType::SERIAL, [&, N](auto& pf){
        if(pf.token() == N) {
          pf.stop();
          return;
        }

        if(source[pf.token()] / 2 == 0) {
          buffer[pf.line()][pf.pipe()] = source[pf.token()];
        }
        else {
          buffer[pf.line()][pf.pipe()] = source[pf.token()] + 8;
        }

    });

      // pipe 2
    pipes.emplace_back(tf::PipeType::PARALLEL, [&](auto& pf){

        if(buffer[pf.line()][pf.pipe() - 1] > 4897) {
          buffer[pf.line()][pf.pipe()] =  buffer[pf.line()][pf.pipe() - 1] - 1834;
        }
        else {
          buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 3;
        }

    });

    // pipe 3
    pipes.emplace_back(tf::PipeType::SERIAL, [&](auto& pf){

        if((buffer[pf.line()][pf.pipe() - 1] + 9) / 4 < 50) {
          buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 1;
        }
        else {
          buffer[pf.line()][pf.pipe()] = buffer[pf.line()][pf.pipe() - 1] + 17;
        }

        collection.push_back(buffer[pf.line()][pf.pipe()]);

    });

    pl.reset(L, pipes.begin(), pipes.end());

    auto pl_t = taskflow.composed_of(pl).name("pipeline");

    auto check_t = taskflow.emplace([&](){
      for(size_t n = 0; n < N; ++n) {
        REQUIRE(collection[n] == ifelse_spipe_ans(source[n]));
      }
    }).name("check");

    pl_t.precede(check_t);

    executor.run(taskflow).wait();

    pipes.clear();
  }
}

TEST_CASE("ScalablePipeline.Ifelse.1L.1W" * doctest::timeout(300)) {
  ifelse_spipeline(1, 1);
}

TEST_CASE("ScalablePipeline.Ifelse.1L.2W" * doctest::timeout(300)) {
  ifelse_spipeline(1, 2);
}

TEST_CASE("ScalablePipeline.Ifelse.1L.3W" * doctest::timeout(300)) {
  ifelse_spipeline(1, 3);
}

TEST_CASE("ScalablePipeline.Ifelse.1L.4W" * doctest::timeout(300)) {
  ifelse_spipeline(1, 4);
}

TEST_CASE("ScalablePipeline.Ifelse.3L.1W" * doctest::timeout(300)) {
  ifelse_spipeline(3, 1);
}

TEST_CASE("ScalablePipeline.Ifelse.3L.2W" * doctest::timeout(300)) {
  ifelse_spipeline(3, 2);
}

TEST_CASE("ScalablePipeline.Ifelse.3L.3W" * doctest::timeout(300)) {
  ifelse_spipeline(3, 3);
}

TEST_CASE("ScalablePipeline.Ifelse.3L.4W" * doctest::timeout(300)) {
  ifelse_spipeline(3, 4);
}

TEST_CASE("ScalablePipeline.Ifelse.5L.1W" * doctest::timeout(300)) {
  ifelse_spipeline(5, 1);
}

TEST_CASE("ScalablePipeline.Ifelse.5L.2W" * doctest::timeout(300)) {
  ifelse_spipeline(5, 2);
}

TEST_CASE("ScalablePipeline.Ifelse.5L.3W" * doctest::timeout(300)) {
  ifelse_spipeline(5, 3);
}

TEST_CASE("ScalablePipeline.Ifelse.5L.4W" * doctest::timeout(300)) {
  ifelse_spipeline(5, 4);
}

TEST_CASE("ScalablePipeline.Ifelse.7L.1W" * doctest::timeout(300)) {
  ifelse_spipeline(7, 1);
}

TEST_CASE("ScalablePipeline.Ifelse.7L.2W" * doctest::timeout(300)) {
  ifelse_spipeline(7, 2);
}

TEST_CASE("ScalablePipeline.Ifelse.7L.3W" * doctest::timeout(300)) {
  ifelse_spipeline(7, 3);
}

TEST_CASE("ScalablePipeline.Ifelse.7L.4W" * doctest::timeout(300)) {
  ifelse_spipeline(7, 4);
}


// ----------------------------------------------------------------------------
// ScalablePipeline in ScalablePipeline
// pipeline has 4 pipes, L lines, W workers
// each subpipeline has 3 pipes, subL lines
//
// pipeline = SPPS
// each subpipeline = SPS
//
// ----------------------------------------------------------------------------

void spipeline_in_spipeline(size_t L, unsigned w, unsigned subL) {

  tf::Executor executor(w);

  const size_t maxN = 7;
  const size_t maxsubN = 7;

  std::vector<std::vector<int>> source(maxN);
  for(auto&& each: source) {
    each.resize(maxsubN);
    std::iota(each.begin(), each.end(), 0);
  }

  std::vector<std::array<int, 4>> buffer(L);

  std::vector<tf::Pipe<>> pipes;
  tf::ScalablePipeline<typename decltype(pipes)::iterator> pl;

  // each pipe contains one subpipeline
  // each subpipeline has three pipes, subL lines
  //
  // subbuffers[0][1][2][2] means
  // first line, second pipe, third subline, third subpipe
  std::vector<std::vector<std::vector<std::array<int, 3>>>> subbuffers(L);

  for(auto&& b: subbuffers) {
    b.resize(4);
    for(auto&& each: b) {
        each.resize(subL);
    }
  }

  for (size_t N = 1; N < maxN; ++N) {
    for(size_t subN = 1; subN < maxsubN; ++subN) {

      size_t j1 = 0, j4 = 0;
      std::atomic<size_t> j2 = 0;
      std::atomic<size_t> j3 = 0;

      // begin of pipeline ---------------------------

      // begin of pipe 1 -----------------------------
      pipes.emplace_back(tf::PipeType::SERIAL, [&, N, subN, subL](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }

        size_t subj1 = 0, subj3 = 0;
        std::atomic<size_t> subj2 = 0;
        std::vector<int> subcollection;
        subcollection.reserve(subN);
        std::vector<tf::Pipe<>> subpipes;
        tf::ScalablePipeline<typename decltype(subpipes)::iterator> subpl;

        // subpipe 1
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
            if(subj1 == subN) {
              subpf.stop();
              return;
            }

            REQUIRE(subpf.token() % subL == subpf.line());

            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
              = source[pf.token()][subj1] + 1;

            ++subj1;
        });

        // subpipe 2
        subpipes.emplace_back(tf::PipeType::PARALLEL, [&, subN](auto& subpf) mutable {
            REQUIRE(subj2++ < subN);
            REQUIRE(subpf.token() % subL == subpf.line());
            REQUIRE(
              source[pf.token()][subpf.token()] + 1 ==
              subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
            );
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
            = source[pf.token()][subpf.token()] + 1;
        });

        // subpipe 3
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
          REQUIRE(subj3 < subN);
          REQUIRE(subpf.token() % subL == subpf.line());
          REQUIRE(
            source[pf.token()][subj3] + 1 ==
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
          );
          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
            = source[pf.token()][subj3] + 3;
          subcollection.push_back(subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]);
          ++subj3;
        });

        tf::Taskflow taskflow;

        // test task
        auto test_t = taskflow.emplace([&, subN](){
          REQUIRE(subj1 == subN);
          REQUIRE(subj2 == subN);
          REQUIRE(subj3 == subN);
          REQUIRE(subpl.num_tokens() == subN);
          REQUIRE(subcollection.size() == subN);
        }).name("test");

        // subpipeline
        subpl.reset(subL, subpipes.begin(), subpipes.end());
        auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

        subpl_t.precede(test_t);
        executor.corun(taskflow);

        buffer[pf.line()][pf.pipe()] = std::accumulate(
          subcollection.begin(),
          subcollection.end(),
          0
        );

        j1++;
      });
      // end of pipe 1 -----------------------------

      //begin of pipe 2 ---------------------------
      pipes.emplace_back(tf::PipeType::PARALLEL, [&, subN, subL](auto& pf) mutable {

        REQUIRE(j2++ < N);
        int res = std::accumulate(
          source[pf.token()].begin(),
          source[pf.token()].begin() + subN,
          0
        );
        REQUIRE(buffer[pf.line()][pf.pipe() - 1] == res + 3 * subN);

        size_t subj1 = 0, subj3 = 0;
        std::atomic<size_t> subj2 = 0;
        std::vector<int> subcollection;
        subcollection.reserve(subN);
        std::vector<tf::Pipe<>> subpipes;
        tf::ScalablePipeline<typename decltype(subpipes)::iterator> subpl;

        // subpipe 1
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
          if(subj1 == subN) {
            subpf.stop();
            return;
          }

          REQUIRE(subpf.token() % subL == subpf.line());

          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()] =
          source[pf.token()][subj1] + 1;

          ++subj1;
        });

        // subpipe 2
        subpipes.emplace_back(tf::PipeType::PARALLEL, [&, subN](auto& subpf) mutable {
          REQUIRE(subj2++ < subN);
          REQUIRE(subpf.token() % subL == subpf.line());
          REQUIRE(
            source[pf.token()][subpf.token()] + 1 ==
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
          );
          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
          = source[pf.token()][subpf.token()] + 1;
        });

        // subpipe 3
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
          REQUIRE(subj3 < subN);
          REQUIRE(subpf.token() % subL == subpf.line());
          REQUIRE(
            source[pf.token()][subj3] + 1 ==
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
          );
          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
          = source[pf.token()][subj3] + 13;
          subcollection.push_back(subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]);
          ++subj3;
        });

        tf::Taskflow taskflow;

        // test task
        auto test_t = taskflow.emplace([&, subN](){
          REQUIRE(subj1 == subN);
          REQUIRE(subj2 == subN);
          REQUIRE(subj3 == subN);
          REQUIRE(subpl.num_tokens() == subN);
          REQUIRE(subcollection.size() == subN);
        }).name("test");

        // subpipeline
        subpl.reset(subL, subpipes.begin(), subpipes.end());
        auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

        subpl_t.precede(test_t);
        executor.corun(taskflow);

        buffer[pf.line()][pf.pipe()] = std::accumulate(
          subcollection.begin(),
          subcollection.end(),
          0
        );

      });
      // end of pipe 2 -----------------------------

      // begin of pipe 3 ---------------------------
      pipes.emplace_back(tf::PipeType::SERIAL, [&, N, subN, subL](auto& pf) mutable {

        REQUIRE(j3++ < N);
        int res = std::accumulate(
          source[pf.token()].begin(),
          source[pf.token()].begin() + subN,
          0
        );

        REQUIRE(buffer[pf.line()][pf.pipe() - 1] == res + 13 * subN);

        size_t subj1 = 0, subj3 = 0;
        std::atomic<size_t> subj2 = 0;
        std::vector<int> subcollection;
        subcollection.reserve(subN);
        std::vector<tf::Pipe<>> subpipes;
        tf::ScalablePipeline<typename decltype(subpipes)::iterator> subpl;

        // subpipe 1
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
          if(subj1 == subN) {
            subpf.stop();
            return;
          }

          REQUIRE(subpf.token() % subL == subpf.line());

          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]
            = source[pf.token()][subj1] + 1;

          ++subj1;
        });

        // subpipe 2
        subpipes.emplace_back(tf::PipeType::PARALLEL, [&, subN](auto& subpf) mutable {
          REQUIRE(subj2++ < subN);
          REQUIRE(subpf.token() % subL == subpf.line());
          REQUIRE(
            source[pf.token()][subpf.token()] + 1 ==
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
          );
          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()] =
          source[pf.token()][subpf.token()] + 1;
        });

        // subpipe 3
        subpipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& subpf) mutable {
          REQUIRE(subj3 < subN);
          REQUIRE(subpf.token() % subL == subpf.line());
          REQUIRE(
            source[pf.token()][subj3] + 1 ==
            subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe() - 1]
          );
          subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()] =
          source[pf.token()][subj3] + 7;
          subcollection.push_back(subbuffers[pf.line()][pf.pipe()][subpf.line()][subpf.pipe()]);
          ++subj3;
        });

        tf::Taskflow taskflow;

        // test task
        auto test_t = taskflow.emplace([&, subN](){
          REQUIRE(subj1 == subN);
          REQUIRE(subj2 == subN);
          REQUIRE(subj3 == subN);
          REQUIRE(subpl.num_tokens() == subN);
          REQUIRE(subcollection.size() == subN);
        }).name("test");

        // subpipeline
        subpl.reset(subL, subpipes.begin(), subpipes.end());
        auto subpl_t = taskflow.composed_of(subpl).name("module_of_subpipeline");

        subpl_t.precede(test_t);
        executor.corun(taskflow);

        buffer[pf.line()][pf.pipe()] = std::accumulate(
          subcollection.begin(),
          subcollection.end(),
          0
        );

      });
      // end of pipe 3 -----------------------------

      // begin of pipe 4 ---------------------------
      pipes.emplace_back(tf::PipeType::SERIAL, [&, subN](auto& pf) mutable {
        int res = std::accumulate(
          source[j4].begin(),
          source[j4].begin() + subN,
          0
        );
        REQUIRE(buffer[pf.line()][pf.pipe() - 1] == res + 7 * subN);
        j4++;
      });
      // end of pipe 4 -----------------------------

      pl.reset(L, pipes.begin(), pipes.end());

      tf::Taskflow taskflow;
      taskflow.composed_of(pl).name("module_of_pipeline");
      executor.run(taskflow).wait();

      pipes.clear();
    }
  }
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.1W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 1, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.1W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 1, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.1W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 1, 4);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.2W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 2, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.2W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 2, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.1L.2W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(1, 2, 4);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.1W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 1, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.1W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 1, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.1W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 1, 4);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.2W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 2, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.2W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 2, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.3L.2W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(3, 2, 4);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.1W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 1, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.1W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 1, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.1W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 1, 4);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.2W.1subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 2, 1);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.2W.3subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 2, 3);
}

TEST_CASE("ScalablePipeline.PipelineinPipeline.5L.2W.4subL" * doctest::timeout(300)) {
  spipeline_in_spipeline(5, 2, 4);
}

// ----------------------------------------------------------------------------
/* SNIG task graph
// o: normal task
// c: condition task
// p: pipeline
//
// four devices example:
//               o
//            / | | \
//          c  c  c  c -----
//          |  |  |  |     |
//   -----> p  p  p  p     |
//   |     | |   |  |      |
//   ----- c c   c  c      |
//         | |  |  |       |
//         o o  o  o       |
//         \ \  | /        |
//           \||/          |
//            o <-----------
//
// each pipeline has five pipes, L lines, W workers
// each pipeline = SPSPS
*/
// ----------------------------------------------------------------------------

void snig_spipeline(size_t L, unsigned w) {

  size_t NUM_SOURCE = 70000;
  size_t BATCH_SIZE = 100;

  std::array<size_t, 7> NUM_DEVICES = {1, 2, 4, 6, 9, 13, 17};
  std::atomic<size_t> finished{0};
  std::vector<int> source(NUM_SOURCE);
  std::iota(source.begin(), source.end(), 0);

  for(auto&& NUM_DEVICE: NUM_DEVICES) {
    std::vector<std::vector<std::array<int, 5>>> buffers(NUM_DEVICE);
    for(auto&& buffer: buffers) {
      buffer.resize(L);
    }

    tf::Taskflow taskflow;
    tf::Executor executor(w);

    auto start_t = taskflow.emplace([](){}).name("start");
    auto end_t = taskflow.emplace([](){}).name("end");

    std::vector<tf::Task> dev_ends(NUM_DEVICE);
    for(auto&& dev_end: dev_ends) {
      dev_end = taskflow.emplace([](){}).name("dev_end");
    }

    std::vector<tf::Task> first_fetches(NUM_DEVICE);
    std::vector<tf::Task> fetches(NUM_DEVICE);
    std::vector<std::vector<tf::Pipe<>>> pipes(NUM_DEVICE);

    // for type
    using pipeline_it = std::vector<tf::Pipe<>>::iterator;

    std::vector<tf::Task> module_of_pipelines(NUM_DEVICE);

    std::vector<tf::ScalablePipeline<pipeline_it>> pipelines(NUM_DEVICE);

    std::vector<size_t> dev_begins(NUM_DEVICE);

    std::vector<size_t> j1s(NUM_DEVICE, 0);
    std::vector<size_t> j3s(NUM_DEVICE, 0);
    std::vector<size_t> j5s(NUM_DEVICE, 0);
    std::vector<std::unique_ptr<std::atomic<size_t>>> j2s(NUM_DEVICE);
    std::vector<std::unique_ptr<std::atomic<size_t>>> j4s(NUM_DEVICE);

    for(size_t dev = 0; dev < NUM_DEVICE; ++dev) {
      j2s[dev] = std::make_unique<std::atomic<size_t>>(0);
      j4s[dev] = std::make_unique<std::atomic<size_t>>(0);
    }

    std::vector<std::vector<int>> collections(NUM_DEVICE);
    for(auto&& collection: collections) {
      collection.reserve(BATCH_SIZE);
    }

    for(size_t dev = 0; dev < NUM_DEVICE; ++dev) {
      first_fetches[dev] = taskflow.emplace([&, dev, BATCH_SIZE](){
        size_t num = finished.fetch_add(BATCH_SIZE);
        dev_begins[dev] = num;
        return num >= NUM_SOURCE;
      }).name("first_fetch");

      // pipe 1
      pipes[dev].emplace_back(
        tf::PipeType::SERIAL, 
        [&, dev, BATCH_SIZE](auto& pf) mutable {
          if(j1s[dev] == BATCH_SIZE) {
            pf.stop();
            return;
          }

          REQUIRE(pf.token() % L == pf.line());

          buffers[dev][pf.line()][pf.pipe()] = source[dev_begins[dev] + j1s[dev]] + 1;

          ++j1s[dev];
        }
      );

      // pipe 2
      pipes[dev].emplace_back(
        tf::PipeType::PARALLEL, [&, dev, BATCH_SIZE](auto& pf) mutable {
          REQUIRE((*j2s[dev])++ < BATCH_SIZE);
          REQUIRE(pf.token() % L == pf.line());
          REQUIRE(source[dev_begins[dev] + pf.token()] + 1 == buffers[dev][pf.line()][pf.pipe() - 1]);

          buffers[dev][pf.line()][pf.pipe()] = source[dev_begins[dev] + pf.token()] + 3;

        }
      );

      // pipe 3
      pipes[dev].emplace_back(
        tf::PipeType::SERIAL, [&, dev, BATCH_SIZE](auto& pf) mutable {
          REQUIRE(j3s[dev] < BATCH_SIZE);
          REQUIRE(pf.token() % L == pf.line());
          REQUIRE(source[dev_begins[dev] + j3s[dev]] + 3 == buffers[dev][pf.line()][pf.pipe() - 1]);

          buffers[dev][pf.line()][pf.pipe()] = source[dev_begins[dev] + j3s[dev]] + 8;

          ++j3s[dev];
        }
      );

      // pipe 4
      pipes[dev].emplace_back(
        tf::PipeType::PARALLEL, [&, dev, BATCH_SIZE](auto& pf) mutable {
          REQUIRE((*j4s[dev])++ < BATCH_SIZE);
          REQUIRE(pf.token() % L == pf.line());
          REQUIRE(source[dev_begins[dev] + pf.token()] + 8 == buffers[dev][pf.line()][pf.pipe() - 1]);

          buffers[dev][pf.line()][pf.pipe()] = source[dev_begins[dev] + pf.token()] + 9;
        }
      );

      // pipe 5
      pipes[dev].emplace_back(
        tf::PipeType::SERIAL, [&, dev, BATCH_SIZE](auto& pf) mutable {
          REQUIRE(j5s[dev] < BATCH_SIZE);
          REQUIRE(pf.token() % L == pf.line());
          REQUIRE(source[dev_begins[dev] + j5s[dev]] + 9 == buffers[dev][pf.line()][pf.pipe() - 1]);

          collections[dev].push_back(buffers[dev][pf.line()][pf.pipe() - 1] + 2);

          ++j5s[dev];
        }
      );


      fetches[dev] = taskflow.emplace([&, dev, NUM_SOURCE, BATCH_SIZE](){
        for(size_t b = 0; b < BATCH_SIZE; ++b) {
          REQUIRE(source[dev_begins[dev] + b] + 9 + 2 == collections[dev][b]);
        }
        collections[dev].clear();
        collections[dev].reserve(BATCH_SIZE);

        size_t num = finished.fetch_add(BATCH_SIZE);
        dev_begins[dev] = num;
        j1s[dev] = 0;
        *j2s[dev] = 0;
        j3s[dev] = 0;
        *j4s[dev] = 0;
        j5s[dev] = 0;
        pipelines[dev].reset();
        return num >= NUM_SOURCE;
      }).name("fetch");
    }

    for(size_t dev = 0; dev < NUM_DEVICE; ++dev) {
      pipelines[dev].reset(L, pipes[dev].begin(), pipes[dev].end());
      module_of_pipelines[dev] = taskflow.composed_of(pipelines[dev]).name("pipeline");
    }

    // dependencies
    for(size_t dev = 0; dev < NUM_DEVICE; ++dev) {
      start_t.precede(first_fetches[dev]);
      first_fetches[dev].precede(
        module_of_pipelines[dev],
        dev_ends[dev]
      );
      module_of_pipelines[dev].precede(fetches[dev]);
      fetches[dev].precede(module_of_pipelines[dev], dev_ends[dev]);
      dev_ends[dev].precede(end_t);
    }


    executor.run(taskflow).wait();
  }
}

TEST_CASE("ScalablePipeline.SNIG.1L.1W" * doctest::timeout(300)) {
  snig_spipeline(1, 1);
}

TEST_CASE("ScalablePipeline.SNIG.1L.2W" * doctest::timeout(300)) {
  snig_spipeline(1, 2);
}

TEST_CASE("ScalablePipeline.SNIG.1L.3W" * doctest::timeout(300)) {
  snig_spipeline(1, 3);
}

TEST_CASE("ScalablePipeline.SNIG.3L.1W" * doctest::timeout(300)) {
  snig_spipeline(3, 1);
}

TEST_CASE("ScalablePipeline.SNIG.3L.2W" * doctest::timeout(300)) {
  snig_spipeline(3, 2);
}

TEST_CASE("ScalablePipeline.SNIG.3L.3W" * doctest::timeout(300)) {
  snig_spipeline(3, 3);
}

TEST_CASE("ScalablePipeline.SNIG.5L.1W" * doctest::timeout(300)) {
  snig_spipeline(5, 1);
}

TEST_CASE("ScalablePipeline.SNIG.5L.2W" * doctest::timeout(300)) {
  snig_spipeline(5, 2);
}

TEST_CASE("ScalablePipeline.SNIG.5L.3W" * doctest::timeout(300)) {
  snig_spipeline(5, 3);
}

TEST_CASE("ScalablePipeline.SNIG.7L.1W" * doctest::timeout(300)) {
  snig_spipeline(7, 1);
}

TEST_CASE("ScalablePipeline.SNIG.7L.2W" * doctest::timeout(300)) {
  snig_spipeline(7, 2);
}

TEST_CASE("ScalablePipeline.SNIG.7L.3W" * doctest::timeout(300)) {
  snig_spipeline(7, 3);
}

// ----------------------------------------------------------------------
//  Subflow pipeline
// -----------------------------------------------------------------------

void spawn(
  tf::Subflow& sf,
  size_t L,
  size_t NUM_PIPES,
  size_t NUM_RECURS,
  size_t maxN,
  size_t r,
  std::vector<std::vector<int>>& buffer,
  std::vector<std::vector<int>>& source,
  std::vector<std::vector<tf::Pipe<>>>& pipes,
  std::vector<tf::ScalablePipeline<typename std::vector<tf::Pipe<>>::iterator>>& spls,
  size_t& counter
) {

  // construct pipes
  for(size_t p = 0; p < NUM_PIPES; ++p) {
    pipes[r].emplace_back(tf::PipeType::SERIAL, [&, maxN, r](tf::Pipeflow& pf) mutable {

      switch(pf.pipe()) {
        case 0:
          if(pf.token() == maxN) {
            pf.stop();
            ++counter;
            return;
          }
          buffer[r][pf.line()] = source[r][pf.token()];
        break;

        default:
          ++buffer[r][pf.line()];
      }

      REQUIRE(buffer[r][pf.line()] == source[r][pf.token()] + pf.pipe());
    });
  }

  spls[r].reset(L, pipes[r].begin(), pipes[r].end());
  auto spl_t = sf.composed_of(spls[r]).name("module_of_pipeline");

  if(r + 1 < NUM_RECURS) {
    auto spawn_t = sf.emplace([&, L, NUM_PIPES, NUM_RECURS, maxN, r](tf::Subflow& sf2) mutable {
      spawn(sf2, L, NUM_PIPES, NUM_RECURS, maxN, r + 1, buffer, source, pipes, spls, counter);
    });
    spawn_t.precede(spl_t);
  }

}

void subflow_spipeline(unsigned NUM_RECURS, unsigned w, size_t L) {

  tf::Executor executor(w);
  tf::Taskflow taskflow;
  std::vector<tf::ScalablePipeline<typename std::vector<tf::Pipe<>>::iterator>> spls(NUM_RECURS);
  std::vector<std::vector<tf::Pipe<>>> pipes(NUM_RECURS);

  size_t maxN = 1123;
  size_t NUM_PIPES = 5;
  size_t counter = 0;

  std::vector<std::vector<int>> source(NUM_RECURS);
  for(auto&& each: source) {
    each.resize(maxN);
    std::iota(each.begin(), each.end(), 0);
  }

  std::vector<std::vector<int>> buffer(NUM_RECURS);
  for(auto&& each:buffer) {
    each.resize(L);
  }

  auto subflows = taskflow.emplace([&, L, NUM_PIPES, NUM_RECURS, maxN](tf::Subflow& sf){
    spawn(sf, L, NUM_PIPES, NUM_RECURS, maxN, 0, buffer, source, pipes, spls, counter);
  });

  auto check = taskflow.emplace([&, NUM_RECURS](){
    REQUIRE(counter == NUM_RECURS);
  }).name("check");

  subflows.precede(check);

  executor.run(taskflow).wait();

}

TEST_CASE("ScalablePipeline.Subflow.1R.1W.1L" * doctest::timeout(300)) {
  subflow_spipeline(1, 1, 1);
}

TEST_CASE("ScalablePipeline.Subflow.1R.1W.3L" * doctest::timeout(300)) {
  subflow_spipeline(1, 1, 3);
}

TEST_CASE("ScalablePipeline.Subflow.1R.1W.4L" * doctest::timeout(300)) {
  subflow_spipeline(1, 1, 4);
}

TEST_CASE("ScalablePipeline.Subflow.1R.2W.1L" * doctest::timeout(300)) {
  subflow_spipeline(1, 2, 1);
}

TEST_CASE("ScalablePipeline.Subflow.1R.2W.3L" * doctest::timeout(300)) {
  subflow_spipeline(1, 2, 3);
}

TEST_CASE("ScalablePipeline.Subflow.1R.2W.4L" * doctest::timeout(300)) {
  subflow_spipeline(1, 2, 4);
}

TEST_CASE("ScalablePipeline.Subflow.3R.1W.1L" * doctest::timeout(300)) {
  subflow_spipeline(3, 1, 1);
}

TEST_CASE("ScalablePipeline.Subflow.3R.1W.3L" * doctest::timeout(300)) {
  subflow_spipeline(3, 1, 3);
}

TEST_CASE("ScalablePipeline.Subflow.3R.1W.4L" * doctest::timeout(300)) {
  subflow_spipeline(3, 1, 4);
}

TEST_CASE("ScalablePipeline.Subflow.3R.2W.1L" * doctest::timeout(300)) {
  subflow_spipeline(3, 2, 1);
}

TEST_CASE("ScalablePipeline.Subflow.3R.2W.3L" * doctest::timeout(300)) {
  subflow_spipeline(3, 2, 3);
}

TEST_CASE("ScalablePipeline.Subflow.3R.2W.4L" * doctest::timeout(300)) {
  subflow_spipeline(3, 2, 4);
}

TEST_CASE("ScalablePipeline.Subflow.5R.1W.1L" * doctest::timeout(300)) {
  subflow_spipeline(5, 1, 1);
}

TEST_CASE("ScalablePipeline.Subflow.5R.1W.3L" * doctest::timeout(300)) {
  subflow_spipeline(5, 1, 3);
}

TEST_CASE("ScalablePipeline.Subflow.5R.1W.4L" * doctest::timeout(300)) {
  subflow_spipeline(5, 1, 4);
}

TEST_CASE("ScalablePipeline.Subflow.5R.2W.1L" * doctest::timeout(300)) {
  subflow_spipeline(5, 2, 1);
}

TEST_CASE("ScalablePipeline.Subflow.5R.2W.3L" * doctest::timeout(300)) {
  subflow_spipeline(5, 2, 3);
}

TEST_CASE("ScalablePipeline.Subflow.5R.2W.4L" * doctest::timeout(300)) {
  subflow_spipeline(5, 2, 4);
}

TEST_CASE("ScalablePipeline.Subflow.7R.1W.1L" * doctest::timeout(300)) {
  subflow_spipeline(7, 1, 1);
}

TEST_CASE("ScalablePipeline.Subflow.7R.1W.3L" * doctest::timeout(300)) {
  subflow_spipeline(7, 1, 3);
}

TEST_CASE("ScalablePipeline.Subflow.7R.1W.4L" * doctest::timeout(300)) {
  subflow_spipeline(7, 1, 4);
}

TEST_CASE("ScalablePipeline.Subflow.7R.2W.1L" * doctest::timeout(300)) {
  subflow_spipeline(7, 2, 1);
}

TEST_CASE("ScalablePipeline.Subflow.7R.2W.3L" * doctest::timeout(300)) {
  subflow_spipeline(7, 2, 3);
}

TEST_CASE("ScalablePipeline.Subflow.7R.2W.4L" * doctest::timeout(300)) {
  subflow_spipeline(7, 2, 4);
}


// ------------------------------------------------------------------------
//  Scalable Pipeline with move constructor and move assignment constructor
// ------------------------------------------------------------------------

TEST_CASE("ScalablePipeline.move" * doctest::timeout(300)) {
  
  size_t N = 10;

  std::atomic<int> counter{0};

  std::vector< tf::Pipe<std::function<void(tf::Pipeflow&)>> > pipes;

  for(size_t i=0; i<N; i++) {
    pipes.emplace_back(tf::PipeType::SERIAL, [&](tf::Pipeflow& pf) {
      if (pf.token() == 5) {
        pf.stop();
      }
      else {
        ++counter;
      }
    });
  }

  using iterator_type = decltype(pipes)::iterator;

  tf::ScalablePipeline<iterator_type> rhs;

  REQUIRE(rhs.num_lines()  == 0);
  REQUIRE(rhs.num_pipes()  == 0);
  REQUIRE(rhs.num_tokens() == 0);

  rhs.reset(1, pipes.begin(), pipes.end());

  REQUIRE(rhs.num_lines()  == 1);
  REQUIRE(rhs.num_pipes()  == N);
  REQUIRE(rhs.num_tokens() == 0);
  
  {
    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.composed_of(rhs);
    executor.run(taskflow).wait();
    REQUIRE(counter == 50);
  }

  auto lhs = std::move(rhs);

  REQUIRE(rhs.num_lines()  == 0);
  REQUIRE(rhs.num_pipes()  == 0);
  REQUIRE(rhs.num_tokens() == 0);
  REQUIRE(lhs.num_lines()  == 1);
  REQUIRE(lhs.num_pipes()  == N);
  REQUIRE(lhs.num_tokens() == 5);
  
  {
    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.composed_of(lhs);
    executor.run(taskflow).wait();
    REQUIRE(counter == 50);
  }


  rhs = std::move(lhs);

  REQUIRE(lhs.num_lines()  == 0);
  REQUIRE(lhs.num_pipes()  == 0);
  REQUIRE(lhs.num_tokens() == 0);
  REQUIRE(rhs.num_lines()  == 1);
  REQUIRE(rhs.num_pipes()  == N);
  REQUIRE(rhs.num_tokens() == 5);
  
  {
    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.composed_of(rhs);
    executor.run(taskflow).wait();
    REQUIRE(counter == 50);
  }
}
