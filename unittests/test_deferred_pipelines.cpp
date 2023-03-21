#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/pipeline.hpp>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <mutex>
#include <algorithm>



// ----------------------------------------------------------------------------
// one pipe (S), L lines, W workers, defer to the previous token
// ----------------------------------------------------------------------------

void pipeline_1P_S_DeferPreviousToken(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 3;

  for(size_t N = 0; N <= maxN; N++) {
    
    std::vector<size_t> collection1;
    std::vector<size_t> deferrals;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, L, &deferrals](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() == 0) {
                //printf("Stage 1 : token %zu on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                deferrals.push_back(pf.num_deferrals());
              }
              else {
                pf.defer(pf.token()-1);
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu on line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              deferrals.push_back(pf.num_deferrals());
            break;
          }
          REQUIRE(pf.token() % L == pf.line());
        }
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(collection1.size() == N);

      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection1[i] == i);
      }

      REQUIRE(deferrals.size() == N);
      for (size_t i = 0; i < deferrals.size(); ++i) {
        if (i == 0) {
          REQUIRE(deferrals[i] == 0);
        }
        else {
          REQUIRE(deferrals[i] == 1);
        }
      }
    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      deferrals.clear();
    }).get();
  }
}

// one pipe (S)
TEST_CASE("Pipeline.1P(S).DeferPreviousToken.1L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(1, 1);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.1L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(1, 2);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.1L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(1, 3);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.1L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(1, 4);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.2L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(2, 1);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.2L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(2, 2);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.2L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(2, 3);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.2L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(2, 4);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.3L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(3, 1);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.3L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(3, 2);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.3L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(3, 3);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.3L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(3, 4);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.4L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(4, 1);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.4L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(4, 2);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.4L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(4, 3);
}

TEST_CASE("Pipeline.1P(S).DeferPreviousToken.4L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferPreviousToken(4, 4);
}


// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers, defer to the previous token
// ----------------------------------------------------------------------------

void pipeline_2P_SS_DeferPreviousToken(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<std::array<size_t, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {
    
    std::vector<size_t> collection1;
    std::vector<size_t> collection2;
    std::vector<size_t> deferrals1;
    std::vector<size_t> deferrals2;

    std::mutex mutex;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &mybuffer, L, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() == 0) {
                //printf("Stage 1 : token %zu on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                mybuffer[pf.line()][pf.pipe()] = pf.token(); 
                deferrals1.push_back(pf.num_deferrals());          
              }
              else {
                pf.defer(pf.token()-1);
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu on line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              mybuffer[pf.line()][pf.pipe()] = pf.token();           
              deferrals1.push_back(pf.num_deferrals());          
            break;
          }
          REQUIRE(pf.token() % L == pf.line());
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [&mybuffer, &mutex, &collection2, L, &deferrals2](auto& pf) mutable {
        REQUIRE(pf.token() % L == pf.line());
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals2.push_back(pf.num_deferrals());
        }

        if (pf.token() == 0) {
          REQUIRE(pf.num_deferrals() == 0);
        }
        else {
          REQUIRE(pf.num_deferrals() == 1);
        }
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(collection1.size() == N);
      REQUIRE(collection2.size() == N);
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection1[i] == i);
        REQUIRE(collection2[i] == i);
      }

      REQUIRE(deferrals1.size() == N);
      REQUIRE(deferrals2.size() == N);
      for (size_t i = 0; i < deferrals1.size(); ++i) {
        if (i == 0) {
          REQUIRE(deferrals1[i] == 0);
          REQUIRE(deferrals2[i] == 0);
        }
        else {
          REQUIRE(deferrals1[i] == 1);
          REQUIRE(deferrals2[i] == 1);
        }
      }
    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      collection2.clear();
      deferrals1.clear();
      deferrals2.clear();
    }).get();
  }
}

// two pipes (SS)
TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(1, 1);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(1, 2);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(1, 3);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(1, 4);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(2, 1);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(2, 2);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(2, 3);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(2, 4);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(3, 1);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(3, 2);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(3, 3);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(3, 4);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(4, 1);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(4, 2);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(4, 3);
}

TEST_CASE("Pipeline.2P(SS).DeferPreviousToken.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferPreviousToken(4, 4);
}


// ----------------------------------------------------------------------------
// two pipes (SP), L lines, W workers, defer to the previous token
// ----------------------------------------------------------------------------

void pipeline_2P_SP_DeferPreviousToken(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<std::array<size_t, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {
    
    std::vector<size_t> collection1;
    std::vector<size_t> collection2;
    std::vector<size_t> deferrals1;
    std::vector<size_t> deferrals2(N);
    std::mutex mutex;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &mybuffer, L, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() == 0) {
                //printf("Stage 1 : token %zu on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                deferrals1.push_back(pf.num_deferrals());
                mybuffer[pf.line()][pf.pipe()] = pf.token();           
              }
              else {
                pf.defer(pf.token()-1);
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu on line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              deferrals1.push_back(pf.num_deferrals());
              mybuffer[pf.line()][pf.pipe()] = pf.token();           
            break;
          }
          REQUIRE(pf.token() % L == pf.line());
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer, &mutex, &collection2, L, &deferrals2](auto& pf) mutable {
        REQUIRE(pf.token() % L == pf.line());
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals2[pf.token()] = pf.num_deferrals();
        }

        if (pf.token() == 0) {
          REQUIRE(pf.num_deferrals() == 0);
        }
        else {
          REQUIRE(pf.num_deferrals() == 1);
        }
    
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(collection1.size() == N);
      REQUIRE(collection2.size() == N);
      sort(collection2.begin(), collection2.end());
  
      for (size_t i = 0; i < N; ++i) {
        REQUIRE(collection1[i] == i);
        REQUIRE(collection2[i] == i);
      }

      REQUIRE(deferrals1.size() == N);
      REQUIRE(deferrals2.size() == N);
      for (size_t i = 0; i < N; ++i) {
        if (i == 0) {
          REQUIRE(deferrals1[0] == 0);
          REQUIRE(deferrals2[0] == 0);
        }
        else {
          REQUIRE(deferrals1[i] == 1);
          REQUIRE(deferrals2[i] == 1);
        }
      }
    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      collection2.clear();
      deferrals1.clear();
      deferrals2.clear();
    }).get();
  }
}

// two pipes (SP)
TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(1, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(1, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(1, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(1, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(2, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(2, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(2, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(2, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(3, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(3, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(3, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(3, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(4, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(4, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(4, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferPreviousToken.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferPreviousToken(4, 4);
}


// ----------------------------------------------------------------------------
// one pipe (S), L lines, W workers
//
// defer to the next token, pf.defer(pf.token()+1) except the max token
// ----------------------------------------------------------------------------

void pipeline_1P_S_DeferNextToken(size_t L, unsigned w, tf::PipeType) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  std::vector<size_t> collection1;
  std::vector<size_t> deferrals;

  for(size_t N = 1; N <= maxN; N++) {

    tf::Taskflow taskflow;
    deferrals.resize(N);

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &deferrals](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() < N-1) {
                pf.defer(pf.token()+1);
              }
              else {
                deferrals[pf.token()] = pf.num_deferrals();
                collection1.push_back(pf.token());
              }
            break;

            case 1:
              collection1.push_back(pf.token());
              deferrals[pf.token()] = pf.num_deferrals();
            break;
          }
        }
      }}
    );

    taskflow.composed_of(pl).name("module_of_pipeline");
    executor.run(taskflow).wait();
 
    REQUIRE(deferrals.size() == N); 
    for (size_t i = 0; i < deferrals.size()-1;++i) {
      REQUIRE(deferrals[i] == 1);
    }
    REQUIRE(deferrals[deferrals.size()-1] == 0);

    for (size_t i = 0; i < collection1.size(); ++i) {
      REQUIRE(i + collection1[i] == N-1);
    }
    
    collection1.clear();
    deferrals.clear();
  }
}

// one pipe 
TEST_CASE("Pipeline.1P(S).DeferNextToken.1L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(1, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.1L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(1, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.1L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(1, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.1L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(1, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.2L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(2, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.2L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(2, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.2L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(2, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.2L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(2, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.3L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(3, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.3L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(3, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.3L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(3, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.3L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(3, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.4L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(4, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.4L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(4, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.4L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(4, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1P(S).DeferNextToken.4L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_DeferNextToken(4, 4, tf::PipeType::SERIAL);
}


// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers
//
// defer to the next token, pf.defer(pf.token()+1) except the max token
// ----------------------------------------------------------------------------

void pipeline_2P_SS_DeferNextToken(size_t L, unsigned w, tf::PipeType second_type) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<size_t> mybuffer(L);

  std::vector<size_t> collection1;
  std::vector<size_t> collection2;
  std::vector<size_t> deferrals1;
  std::vector<size_t> deferrals2;

  for(size_t N = 1; N <= maxN; N++) {

    tf::Taskflow taskflow;
    deferrals1.resize(N);
    deferrals2.resize(N);

    //size_t value = (N-1)%L;
    //std::cout << "N = " << N << ", value = " << value << ", L = " << L << ", W = " << w << '\n';    
    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &mybuffer, &collection1, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() < N-1) {
                pf.defer(pf.token()+1);
              }
              else {
                collection1.push_back(pf.token());
                deferrals1[pf.token()] = pf.num_deferrals();
                mybuffer[pf.line()] = pf.token();              
              }
            break;

            case 1:
              collection1.push_back(pf.token());
              deferrals1[pf.token()] = pf.num_deferrals();
              mybuffer[pf.line()] = pf.token();              
            break;
          }
        }
      }},

      tf::Pipe{second_type, [&mybuffer, &collection2, &deferrals2](auto& pf) mutable {
        collection2.push_back(mybuffer[pf.line()]);
        deferrals2[pf.token()] = pf.num_deferrals();
      }}
    );

    taskflow.composed_of(pl).name("module_of_pipeline");
    executor.run(taskflow).wait();
   
    for (size_t i = 0; i < collection1.size(); ++i) {
      REQUIRE(i + collection1[i] == N-1);
      REQUIRE(i + collection2[i] == N-1);
    }
    
    REQUIRE(deferrals1.size() == N);
    REQUIRE(deferrals2.size() == N);
    for (size_t i = 0; i < deferrals1.size()-1; ++i) {
      REQUIRE(deferrals1[i] == 1);
      REQUIRE(deferrals2[i] == 1);
    }
    REQUIRE(deferrals1[deferrals1.size()-1] == 0);
    REQUIRE(deferrals2[deferrals2.size()-1] == 0);

    collection1.clear();
    collection2.clear();
    deferrals1.clear();
    deferrals2.clear();
  }
}

// two pipes 
TEST_CASE("Pipeline.2P(SS).DeferNextToken.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(1, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(1, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(1, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(1, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(2, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(2, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(2, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(2, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(3, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(3, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(3, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(3, 4, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(4, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(4, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(4, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.2P(SS).DeferNextToken.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_DeferNextToken(4, 4, tf::PipeType::SERIAL);
}

// ----------------------------------------------------------------------------
// two pipes (SP), L lines, W workers
//
// defer to the next token, pf.defer(pf.token()+1) except the max token
// ----------------------------------------------------------------------------

void pipeline_2P_SP_DeferNextToken(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<size_t> mybuffer(L);

  std::vector<size_t> collection1;
  std::vector<size_t> collection2;
  std::vector<size_t> deferrals1;
  std::vector<size_t> deferrals2;
  std::mutex mtx;

  for(size_t N = 1; N <= maxN; N++) {

    tf::Taskflow taskflow;
    deferrals1.resize(N);
    deferrals2.resize(N);

    //std::cout << "N = " << N << ", value = " << value << ", L = " << L << ", W = " << w << '\n';    
    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &mybuffer, &collection1, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (pf.token() < N-1) {
                pf.defer(pf.token()+1);
              }
              else {
                collection1.push_back(pf.token());
                deferrals1[pf.token()] = pf.num_deferrals();
                mybuffer[pf.line()] = pf.token();              
              }
            break;

            case 1:
              collection1.push_back(pf.token());
              deferrals1[pf.token()] = pf.num_deferrals();
              mybuffer[pf.line()] = pf.token();              
            break;
          }
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer, &collection2, &mtx, &deferrals2](auto& pf) mutable {
        {
          std::unique_lock lk(mtx);
          collection2.push_back(mybuffer[pf.line()]);
          deferrals2[pf.token()] = pf.num_deferrals();
        }
      }}
    );

    taskflow.composed_of(pl).name("module_of_pipeline");
    executor.run(taskflow).wait();
  
    sort(collection2.begin(), collection2.end()); 
    for (size_t i = 0; i < collection1.size(); ++i) {
      REQUIRE(i + collection1[i] == N-1);
      REQUIRE(collection2[i] == i);
    }
    
    REQUIRE(deferrals1.size() == N);
    REQUIRE(deferrals2.size() == N);
    for (size_t i = 0; i < deferrals1.size()-1; ++i) {
      REQUIRE(deferrals1[i] == 1);
      REQUIRE(deferrals2[i] == 1);
    }
    REQUIRE(deferrals1[deferrals1.size()-1] == 0);
    REQUIRE(deferrals2[deferrals2.size()-1] == 0);
    
    collection1.clear();
    collection2.clear();
    deferrals1.clear();
    deferrals2.clear();
  }
}

// two pipes 
TEST_CASE("Pipeline.2P(SP).DeferNextToken.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(1, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(1, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(1, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(1, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(2, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(2, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(2, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(2, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(3, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(3, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(3, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(3, 4);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(4, 1);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(4, 2);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(4, 3);
}

TEST_CASE("Pipeline.2P(SP).DeferNextToken.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_DeferNextToken(4, 4);
}



// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers, mimic 264 frame patterns
// ----------------------------------------------------------------------------

struct Frames {
  char type;
  size_t id;
  bool b_defer = false;
  std::vector<size_t> defers;
  Frames(char t, size_t i, std::vector<size_t>&& d) : type{t}, id{i}, defers{std::move(d)} {}
};

std::vector<char> types{'I','B','B','B','P','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','P','B','B','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','I','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','B','B','B','P','P'};


void construct_video(std::vector<Frames>& video, const size_t N) {
  for (size_t i = 0; i < N; ++i) {
    video.emplace_back(types[i], i, std::vector<size_t>{});

    if (types[i] == 'P') {
      size_t step = 1;
      size_t index;
      while (i >= step) {
        index = i - step;
        if (types[index] == 'P' || types[index] == 'I') {
          video[i].defers.push_back(index);
          break;
        }
        else {
          ++step;
        }
      }
    }
    else if (types[i] == 'B') {
      size_t step = 1;
      size_t index;
      while (i >= step) {
        index = i - step;
        if (types[index] == 'P' || types[index] == 'I') {
          video[i].defers.push_back(index);
          break;
        }
        else {
          ++step;
        }
      }
      step = 1;
      while (i+step < N) {
        index = i + step;
        if (types[index] == 'P' || types[index] == 'I') {
          video[i].defers.push_back(index);
          break;
        }
        else {
          ++step;
        }
      }
    }
  }
  //for (size_t i = 0; i < N; ++i) {
  //  std::cout << "video[" << i << "] has type = " << video[i].type
  //            << ", and id = " << video[i].id;
  //  
  //  if (video[i].defers.size()) {
  //     std::cout << ", and has depends = ";
  //     for (size_t j = 0; j < video[i].defers.size(); ++j) {
  //       std::cout << (video[i].defers[j])
  //                 << "(frame " << video[video[i].defers[j]].type << ") ";
  //     }
  //     std::cout << '\n';
  //  }
  //  else {
  //    std::cout << '\n';
  //  }
  //}
}


// ----------------------------------------------------------------------------
// one pipe (S), L lines, W workers, mimic 264 frame patterns
// ----------------------------------------------------------------------------
void pipeline_1P_S_264VideoFormat(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 512;


  for(size_t N = 0; N <= maxN; N++) {
    // declare a x264 format video
    std::vector<Frames> video;
    construct_video(video, N);
    
    std::vector<size_t> collection1;
    std::vector<size_t> deferrals1(N);

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &video, &deferrals1](auto& pf) mutable {
        //printf("toekn %zu, deferred = %zu, N=%zu\n", pf.token(), pf.num_deferrals(), N);
        if(pf.token() == N) {
          //printf("Token %zu stops on line %zu\n", pf.token() ,pf.line());
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (video[pf.token()].type == 'I') {
                //printf("Stage 1 : token %zu is a I frame on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                deferrals1[pf.token()] = 0;
              }
              else if (video[pf.token()].type == 'P') {
                //printf("Token %zu is a P frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
              else if (video[pf.token()].type == 'B') {
                //printf("Token %zu is a B frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    pf.defer(index);
                    break;
                  }
                  ++step;
                }
                step = 1;
                while (pf.token()+step < N) {
                  index = pf.token()+step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" and token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu is deferred 1 time at line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              deferrals1[pf.token()] = 1;
            break;
          }
        }
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){

      for (size_t i = 0; i < N; ++i) {
        std::vector<size_t>::iterator it;
        std::vector<size_t>::iterator it_dep;

        size_t index_it, index_it_dep;
        
        if (video[i].defers.size()) {
          it = std::find(collection1.begin(), collection1.end(), i);
          index_it = std::distance(collection1.begin(), it);
          REQUIRE(it != collection1.end());
          //if (it == collection1.end()) {
          //  printf("Token %zu is missing\n", i);
          //}
          for (size_t j = 0; j < video[i].defers.size(); ++j) {
            it_dep = std::find(collection1.begin(), collection1.end(), video[i].defers[j]);
            index_it_dep = std::distance(collection1.begin(), it_dep);
            
            REQUIRE(it != collection1.end());
            REQUIRE(it_dep != collection1.end());
            REQUIRE(index_it_dep < index_it);
          }
        }
      }

      REQUIRE(deferrals1.size() == N);
      for (size_t i = 0; i < N; ++i) {
        if (video[i].type == 'I') {
          REQUIRE(deferrals1[i] == 0);
        }
        else {
          REQUIRE(deferrals1[i] == 1);
        }
      }
    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      deferrals1.clear();
    }).get();
  }
}

TEST_CASE("Pipeline.1P(S).264VideoFormat.1L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(1,1);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.1L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(1,2);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.1L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(1,3);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.1L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(1,4);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.2L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(2,1);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.2L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(2,2);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.2L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(2,3);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.2L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(2,4);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.3L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(3,1);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.3L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(3,2);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.3L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(3,3);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.3L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(3,4);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.4L.1W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(4,1);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.4L.2W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(4,2);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.4L.3W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(4,3);
}
TEST_CASE("Pipeline.1P(S).264VideoFormat.4L.4W" * doctest::timeout(300)) {
  pipeline_1P_S_264VideoFormat(4,4);
}

// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers, mimic 264 frame patterns
// ----------------------------------------------------------------------------
void pipeline_2P_SS_264VideoFormat(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 512;

  std::vector<std::array<size_t, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {
    // declare a x264 format video
    std::vector<Frames> video;
    construct_video(video, N);
    
    std::vector<size_t> collection1;
    std::vector<size_t> collection2;
    std::vector<size_t> deferrals1(N);
    std::vector<size_t> deferrals2(N);
    
    std::mutex mutex;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &mybuffer, &video, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          //printf("Token %zu stops on line %zu\n", pf.token() ,pf.line());
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (video[pf.token()].type == 'I') {
                //printf("Stage 1 : token %zu is a I frame on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                mybuffer[pf.line()][pf.pipe()] = pf.token();
                deferrals1[pf.token()] = pf.num_deferrals();           
              }
              else if (video[pf.token()].type == 'P') {
                //printf("Token %zu is a P frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
              else if (video[pf.token()].type == 'B') {
                //printf("Token %zu is a B frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    pf.defer(index);
                    break;
                  }
                  ++step;
                }
                step = 1;
                while (pf.token()+step < N) {
                  index = pf.token()+step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" and token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu is deferred 1 time at line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              mybuffer[pf.line()][pf.pipe()] = pf.token();           
              deferrals1[pf.token()] = pf.num_deferrals();
            break;
          }
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [&mybuffer, &mutex, &collection2, &deferrals2](auto& pf) mutable {
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals2[pf.token()] = pf.num_deferrals();
        }
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      //printf("N = %zu and collection1.size() = %zu\n", N, collection1.size());
      for (size_t i = 0; i < collection1.size(); ++i) {
        //printf("collection1[%zu]=%zu, collection2[%zu]=%zu\n", i, collection1[i], i, collection2[i]);
        REQUIRE(collection1[i] == collection2[i]);
      }

      for (size_t i = 0; i < N; ++i) {
        std::vector<size_t>::iterator it;
        std::vector<size_t>::iterator it_dep;

        size_t index_it, index_it_dep;
        
        if (video[i].defers.size()) {
          it = std::find(collection1.begin(), collection1.end(), i);
          index_it = std::distance(collection1.begin(), it);
          //if (it == collection1.end()) {
          //  printf("Token %zu is missing\n", i);
          //}
          for (size_t j = 0; j < video[i].defers.size(); ++j) {
            it_dep = std::find(collection1.begin(), collection1.end(), video[i].defers[j]);
            index_it_dep = std::distance(collection1.begin(), it_dep);
            
            REQUIRE(it != collection1.end());
            REQUIRE(it_dep != collection1.end());
            REQUIRE(index_it_dep < index_it);
          }
        }
      }

      REQUIRE(deferrals1.size() == N);
      REQUIRE(deferrals2.size() == N);
      for (size_t i = 0; i < N; ++i) {
        if (video[i].type == 'I') {
          REQUIRE(deferrals1[i] == 0);
          REQUIRE(deferrals2[i] == 0);
        }
        else {
          REQUIRE(deferrals1[i] == 1);
          REQUIRE(deferrals2[i] == 1);
        }
      }
    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      collection2.clear();
      deferrals1.clear();
      deferrals2.clear();
    }).get();
  }
}

TEST_CASE("Pipeline.2P(SS).264VideoFormat.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(1,1);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(1,2);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(1,3);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(1,4);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(2,1);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(2,2);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(2,3);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(2,4);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(3,1);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(3,2);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(3,3);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(3,4);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(4,1);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(4,2);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(4,3);
}
TEST_CASE("Pipeline.2P(SS).264VideoFormat.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SS_264VideoFormat(4,4);
}



// ----------------------------------------------------------------------------
// two pipes (SP), L lines, W workers, mimic 264 frame patterns
// ----------------------------------------------------------------------------

void pipeline_2P_SP_264VideoFormat(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 512;

  std::vector<std::array<size_t, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {
    // declare a x264 format video
    std::vector<Frames> video;
    construct_video(video, N);
    
    std::vector<size_t> collection1;
    std::vector<size_t> collection2;
    std::vector<size_t> deferrals1(N);
    std::vector<size_t> deferrals2(N);
    
    std::mutex mutex;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &mybuffer, &video, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (video[pf.token()].type == 'I') {
                //printf("Stage 1 : token %zu is a I frame on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                mybuffer[pf.line()][pf.pipe()] = pf.token();           
                deferrals1[pf.token()] = pf.num_deferrals();
              }
              else if (video[pf.token()].type == 'P') {
                //printf("Token %zu is a P frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
              else if (video[pf.token()].type == 'B') {
                //printf("Token %zu is a B frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    pf.defer(index);
                    break;
                  }
                  ++step;
                }
                step = 1;
                while (pf.token()+step < N) {
                  index = pf.token()+step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" and token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
            break;

            case 1:
              //printf("Stage 1 : token %zu is deferred 1 time at line %zu\n", pf.token(), pf.line());
              collection1.push_back(pf.token());
              mybuffer[pf.line()][pf.pipe()] = pf.token();           
              deferrals1[pf.token()] = pf.num_deferrals();
            break;
          }
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer, &mutex, &collection2, &deferrals2](auto& pf) mutable {
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals2[pf.token()] = pf.num_deferrals();
        }
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(collection1.size() == N);
      REQUIRE(collection2.size() == N);

      for (size_t i = 0; i < N; ++i) {
        std::vector<size_t>::iterator it;
        std::vector<size_t>::iterator it_dep;

        if (video[i].defers.size()) {
          it = std::find(collection1.begin(), collection1.end(), i);

          for (size_t j = 0; j < video[i].defers.size(); ++j) {
            it_dep = std::find(collection1.begin(), collection1.end(), video[i].defers[j]);
            
            REQUIRE(it != collection1.end());
            REQUIRE(it_dep != collection1.end());
          }
        }
      }

      REQUIRE(deferrals1.size() == N);
      REQUIRE(deferrals2.size() == N);
      for (size_t i = 0; i < N; ++i) {
        if (video[i].type == 'I') {
          REQUIRE(deferrals1[i] == 0);
          REQUIRE(deferrals2[i] == 0);
        }
        else {
          REQUIRE(deferrals1[i] == 1);
          REQUIRE(deferrals2[i] == 1);
        }
      }

    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      collection2.clear();
      deferrals1.clear();
      deferrals2.clear();
    }).get();
  }
}

TEST_CASE("Pipeline.2P(SP).264VideoFormat.1L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(1,1);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.1L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(1,2);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.1L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(1,3);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.1L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(1,4);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.2L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(2,1);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.2L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(2,2);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.2L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(2,3);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.2L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(2,4);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.3L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(3,1);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.3L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(3,2);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.3L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(3,3);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.3L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(3,4);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.4L.1W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(4,1);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.4L.2W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(4,2);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.4L.3W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(4,3);
}
TEST_CASE("Pipeline.2P(SP).264VideoFormat.4L.4W" * doctest::timeout(300)) {
  pipeline_2P_SP_264VideoFormat(4,4);
}

// ----------------------------------------------------------------------------
// three pipes (SPP), L lines, W workers, mimic 264 frame patterns
// mainly test pf.num_deferrals()
// ----------------------------------------------------------------------------

void pipeline_3P_SPP_264VideoFormat(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 512;

  std::vector<std::array<size_t, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {
    //std::cout << "N = " << N << '\n';
    // declare a x264 format video
    std::vector<Frames> video;
    construct_video(video, N);
    
    std::vector<size_t> collection1;
    std::vector<size_t> collection2;
    std::vector<size_t> collection3;
    std::vector<size_t> deferrals1(N);
    std::vector<size_t> deferrals2(N);
    std::vector<size_t> deferrals3(N);
    
    std::mutex mutex;

    tf::Taskflow taskflow;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &collection1, &mybuffer, &video, &deferrals1](auto& pf) mutable {
        if(pf.token() == N) {
          pf.stop();
          return;
        }
        else {
          switch(pf.num_deferrals()) {
            case 0:
              if (video[pf.token()].type == 'I') {
                //printf("Stage 1 : token %zu is a I frame on line %zu\n", pf.token() ,pf.line());
                collection1.push_back(pf.token());
                mybuffer[pf.line()][pf.pipe()] = pf.token();           
                deferrals1[pf.token()] = pf.num_deferrals();
              }
              else if (video[pf.token()].type == 'P') {
                //printf("Token %zu is a P frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
              }
              else if (video[pf.token()].type == 'B') {
                //printf("Token %zu is a B frame", pf.token());
                size_t step = 1;
                size_t index = 0;
                
                while (pf.token() >= step) {
                  index = pf.token()-step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    //printf(" defers to token %zu which is a %c frame\n", index, video[index].type);
                    pf.defer(index);
                    break;
                  }
                  ++step;
                }
              }
            break;

            case 1:
              if (video[pf.token()].type == 'P') {
                //printf("Stage 1 : token %zu is deferred 1 time at line %zu\n", pf.token(), pf.line());
                collection1.push_back(pf.token());
                mybuffer[pf.line()][pf.pipe()] = pf.token();           
                deferrals1[pf.token()] = pf.num_deferrals();
              }
              else {
                size_t step = 1;
                size_t index = 0;
                while (pf.token()+step < N) {
                  index = pf.token()+step;
                  if (video[index].type == 'P' || video[index].type == 'I') {
                    pf.defer(index);
                    video[pf.token()].b_defer = true;
                    //printf(" and token %zu which is a %c frame\n", index, video[index].type);
                    break;
                  }
                  ++step;
                }
                if (video[pf.token()].b_defer == false) {
                  collection1.push_back(pf.token());
                  mybuffer[pf.line()][pf.pipe()] = pf.token();           
                  deferrals1[pf.token()] = pf.num_deferrals();
                }
              }
            break;

            case 2:
              collection1.push_back(pf.token());
              mybuffer[pf.line()][pf.pipe()] = pf.token();           
              deferrals1[pf.token()] = pf.num_deferrals();
            break;
          }
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer, &mutex, &collection2, &deferrals2](auto& pf) mutable {
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals2[pf.token()] = pf.num_deferrals();
        }
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [&mybuffer, &mutex, &collection3, &deferrals3](auto& pf) mutable {
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection3.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          deferrals3[pf.token()] = pf.num_deferrals();
        }
        //printf("Stage 2 : token %zu at line %zu\n", pf.token(), pf.line());
      }}
    );

    auto pipeline = taskflow.composed_of(pl).name("module_of_pipeline");
    auto test = taskflow.emplace([&](){
      REQUIRE(collection1.size() == N);
      REQUIRE(collection2.size() == N);
      REQUIRE(collection3.size() == N);

      for (size_t i = 0; i < N; ++i) {
        std::vector<size_t>::iterator it;
        std::vector<size_t>::iterator it_dep;

        if (video[i].defers.size()) {
          it = std::find(collection1.begin(), collection1.end(), i);

          for (size_t j = 0; j < video[i].defers.size(); ++j) {
            it_dep = std::find(collection1.begin(), collection1.end(), video[i].defers[j]);
            
            REQUIRE(it != collection1.end());
            REQUIRE(it_dep != collection1.end());
          }
        }
      }

      REQUIRE(deferrals1.size() == N);
      REQUIRE(deferrals2.size() == N);
      REQUIRE(deferrals3.size() == N);
      for (size_t i = 0; i < N; ++i) {
        if (video[i].type == 'I') {
          REQUIRE(deferrals1[i] == 0);
          REQUIRE(deferrals2[i] == 0);
          REQUIRE(deferrals3[i] == 0);
        }
        else if (video[i].type == 'P') {
          REQUIRE(deferrals1[i] == 1);
          REQUIRE(deferrals2[i] == 1);
          REQUIRE(deferrals3[i] == 1);
        }
        else {
          if (video[i].b_defer == true) {
            REQUIRE(deferrals1[i] == 2);
            REQUIRE(deferrals2[i] == 2);
            REQUIRE(deferrals3[i] == 2);
          }
          else {
            REQUIRE(deferrals1[i] == 1);
            REQUIRE(deferrals2[i] == 1);
            REQUIRE(deferrals3[i] == 1);
          }
        }
      }

    }).name("test");

    pipeline.precede(test);

    executor.run_n(taskflow, 1, [&]() mutable {
      collection1.clear();
      collection2.clear();
      collection3.clear();
      deferrals1.clear();
      deferrals2.clear();
      deferrals3.clear();
    }).get();
  }
}

TEST_CASE("Pipeline.3P(SPP).264VideoFormat.1L.1W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(1,1);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.1L.2W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(1,2);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.1L.3W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(1,3);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.1L.4W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(1,4);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.2L.1W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(2,1);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.2L.2W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(2,2);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.2L.3W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(2,3);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.2L.4W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(2,4);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.3L.1W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(3,1);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.3L.2W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(3,2);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.3L.3W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(3,3);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.3L.4W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(3,4);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.4L.1W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(4,1);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.4L.2W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(4,2);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.4L.3W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(4,3);
}
TEST_CASE("Pipeline.3P(SPP).264VideoFormat.4L.4W" * doctest::timeout(300)) {
  pipeline_3P_SPP_264VideoFormat(4,4);
}

