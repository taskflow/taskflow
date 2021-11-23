#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

// TODO (11/5):
// 1. change the testing logic => my pipe's data is data[line()][pipe()]
// 2. use std::vector<std::array<data, num_pipes>> data(L)

// --------------------------------------------------------
// Testcase: 1 pipe, L lines, w workers
// --------------------------------------------------------
void pipeline_1F(size_t L, unsigned w, tf::PipeType type) {

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
      tf::Pipeline pl (L, tf::Pipe{type, [L, N, &j, &source](auto& pf) mutable {
        if (j == N) {
          pf.stop();
          return;
        }
        REQUIRE(j == source[j]);
        REQUIRE(pf.token() % L == pf.line());
        j++;
      }});

      taskflow.pipeline(pl);
      executor.run(taskflow).wait();
      REQUIRE(j == N);
      REQUIRE(pl.num_tokens() == N);
      
      j = 0;
      executor.run(taskflow).wait();
      REQUIRE(j == N);
      REQUIRE(pl.num_tokens() == 2 * N);

      j = 0;
      executor.run(taskflow).wait();
      REQUIRE(j == N);
      REQUIRE(pl.num_tokens() == 3 * N);
    }
    // parallel pipe
    //else if(type == tf::PipeType::PARALLEL) {
    //  
    //  tf::Taskflow taskflow;

    //  std::atomic<size_t> j = 0;
    //  std::mutex mutex;
    //  std::vector<int> collection;

    //  tf::Pipeline pl(L, tf::Pipe{type, 
    //  [N, &j, &mutex, &collection](auto& pf) mutable {

    //    auto ticket = j.fetch_add(1);

    //    if(ticket >= N) {
    //      pf.stop();
    //      return;
    //    }
    //    std::scoped_lock<std::mutex> lock(mutex);
    //    collection.push_back(ticket);
    //  }});

    //  taskflow.pipeline(pl);
    //  executor.run(taskflow).wait();
    //  REQUIRE(collection.size() == N);
    //  std::sort(collection.begin(), collection.end());
    //  for(size_t k=0; k<N; k++) {
    //    REQUIRE(collection[k] == k);
    //  }

    //  j = 0;
    //  collection.clear();
    //  executor.run(taskflow).wait();
    //  REQUIRE(collection.size() == N);
    //  std::sort(collection.begin(), collection.end());
    //  for(size_t k=0; k<N; k++) {
    //    REQUIRE(collection[k] == k);
    //  }
    //}
  }
}

// ---- serial pipe ----

// serial pipe with one line
TEST_CASE("Pipeline.1F(S).1L.1W" * doctest::timeout(300)) {
  pipeline_1F(1, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.2W" * doctest::timeout(300)) {
  pipeline_1F(1, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.3W" * doctest::timeout(300)) {
  pipeline_1F(1, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.4W" * doctest::timeout(300)) {
  pipeline_1F(1, 4, tf::PipeType::SERIAL);
}

// serial pipe with two lines
TEST_CASE("Pipeline.1F(S).2L.1W" * doctest::timeout(300)) {
  pipeline_1F(2, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.2W" * doctest::timeout(300)) {
  pipeline_1F(2, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.3W" * doctest::timeout(300)) {
  pipeline_1F(2, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.4W" * doctest::timeout(300)) {
  pipeline_1F(2, 4, tf::PipeType::SERIAL);
}

// serial pipe with three lines
TEST_CASE("Pipeline.1F(S).3L.1W" * doctest::timeout(300)) {
  pipeline_1F(3, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.2W" * doctest::timeout(300)) {
  pipeline_1F(3, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.3W" * doctest::timeout(300)) {
  pipeline_1F(3, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.4W" * doctest::timeout(300)) {
  pipeline_1F(3, 4, tf::PipeType::SERIAL);
}

// serial pipe with three lines
TEST_CASE("Pipeline.1F(S).4L.1W" * doctest::timeout(300)) {
  pipeline_1F(4, 1, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.2W" * doctest::timeout(300)) {
  pipeline_1F(4, 2, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.3W" * doctest::timeout(300)) {
  pipeline_1F(4, 3, tf::PipeType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.4W" * doctest::timeout(300)) {
  pipeline_1F(4, 4, tf::PipeType::SERIAL);
}


//// ---- parallel pipe ----
//
//// parallel pipe with one line
//TEST_CASE("Pipeline.1F(P).1L.1W" * doctest::timeout(300)) {
//  pipeline_1F(1, 1, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).1L.2W" * doctest::timeout(300)) {
//  pipeline_1F(1, 2, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).1L.3W" * doctest::timeout(300)) {
//  pipeline_1F(1, 3, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).1L.4W" * doctest::timeout(300)) {
//  pipeline_1F(1, 4, tf::PipeType::PARALLEL);
//}
//
//// parallel pipe with two lines
//TEST_CASE("Pipeline.1F(P).2L.1W" * doctest::timeout(300)) {
//  pipeline_1F(2, 1, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).2L.2W" * doctest::timeout(300)) {
//  pipeline_1F(2, 2, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).2L.3W" * doctest::timeout(300)) {
//  pipeline_1F(2, 3, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).2L.4W" * doctest::timeout(300)) {
//  pipeline_1F(2, 4, tf::PipeType::PARALLEL);
//}
//
//// parallel pipe with three lines
//TEST_CASE("Pipeline.1F(P).3L.1W" * doctest::timeout(300)) {
//  pipeline_1F(3, 1, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).3L.2W" * doctest::timeout(300)) {
//  pipeline_1F(3, 2, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).3L.3W" * doctest::timeout(300)) {
//  pipeline_1F(3, 3, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).3L.4W" * doctest::timeout(300)) {
//  pipeline_1F(3, 4, tf::PipeType::PARALLEL);
//}
//
//// parallel pipe with four lines
//TEST_CASE("Pipeline.1F(P).4L.1W" * doctest::timeout(300)) {
//  pipeline_1F(4, 1, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).4L.2W" * doctest::timeout(300)) {
//  pipeline_1F(4, 2, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).4L.3W" * doctest::timeout(300)) {
//  pipeline_1F(4, 3, tf::PipeType::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F(P).4L.4W" * doctest::timeout(300)) {
//  pipeline_1F(4, 4, tf::PipeType::PARALLEL);
//}

// ----------------------------------------------------------------------------
// two pipes (SS), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_2FSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0;

    tf::Pipeline pl(
      L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        //REQUIRE(source[j2] + 1 == *(pf.input()));
        j2++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(pl.num_tokens() == N);
        
    j1 = j2 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(pl.num_tokens() == 2 * N);

    j1 = j2 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// two pipes (SS)
TEST_CASE("Pipeline.2F(SS).1L.1W" * doctest::timeout(300)) {
  pipeline_2FSS(1, 1);
}

TEST_CASE("Pipeline.2F(SS).1L.2W" * doctest::timeout(300)) {
  pipeline_2FSS(1, 2);
}

TEST_CASE("Pipeline.2F(SS).1L.3W" * doctest::timeout(300)) {
  pipeline_2FSS(1, 3);
}

TEST_CASE("Pipeline.2F(SS).1L.4W" * doctest::timeout(300)) {
  pipeline_2FSS(1, 4);
}

TEST_CASE("Pipeline.2F(SS).2L.1W" * doctest::timeout(300)) {
  pipeline_2FSS(2, 1);
}

TEST_CASE("Pipeline.2F(SS).2L.2W" * doctest::timeout(300)) {
  pipeline_2FSS(2, 2);
}

TEST_CASE("Pipeline.2F(SS).2L.3W" * doctest::timeout(300)) {
  pipeline_2FSS(2, 3);
}

TEST_CASE("Pipeline.2F(SS).2L.4W" * doctest::timeout(300)) {
  pipeline_2FSS(2, 4);
}

TEST_CASE("Pipeline.2F(SS).3L.1W" * doctest::timeout(300)) {
  pipeline_2FSS(3, 1);
}

TEST_CASE("Pipeline.2F(SS).3L.2W" * doctest::timeout(300)) {
  pipeline_2FSS(3, 2);
}

TEST_CASE("Pipeline.2F(SS).3L.3W" * doctest::timeout(300)) {
  pipeline_2FSS(3, 3);
}

TEST_CASE("Pipeline.2F(SS).3L.4W" * doctest::timeout(300)) {
  pipeline_2FSS(3, 4);
}

TEST_CASE("Pipeline.2F(SS).4L.1W" * doctest::timeout(300)) {
  pipeline_2FSS(4, 1);
}

TEST_CASE("Pipeline.2F(SS).4L.2W" * doctest::timeout(300)) {
  pipeline_2FSS(4, 2);
}

TEST_CASE("Pipeline.2F(SS).4L.3W" * doctest::timeout(300)) {
  pipeline_2FSS(4, 3);
}

TEST_CASE("Pipeline.2F(SS).4L.4W" * doctest::timeout(300)) {
  pipeline_2FSS(4, 4);
}

// ----------------------------------------------------------------------------
// two pipes (SP), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_2FSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 2>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L,
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &collection, &mutex, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    std::sort(collection.begin(), collection.end());
    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    std::sort(collection.begin(), collection.end());
    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    std::sort(collection.begin(), collection.end());
    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// two pipes (SP)
TEST_CASE("Pipeline.2F(SP).1L.1W" * doctest::timeout(300)) {
  pipeline_2FSP(1, 1);
}

TEST_CASE("Pipeline.2F(SP).1L.2W" * doctest::timeout(300)) {
  pipeline_2FSP(1, 2);
}

TEST_CASE("Pipeline.2F(SP).1L.3W" * doctest::timeout(300)) {
  pipeline_2FSP(1, 3);
}

TEST_CASE("Pipeline.2F(SP).1L.4W" * doctest::timeout(300)) {
  pipeline_2FSP(1, 4);
}

TEST_CASE("Pipeline.2F(SP).2L.1W" * doctest::timeout(300)) {
  pipeline_2FSP(2, 1);
}

TEST_CASE("Pipeline.2F(SP).2L.2W" * doctest::timeout(300)) {
  pipeline_2FSP(2, 2);
}

TEST_CASE("Pipeline.2F(SP).2L.3W" * doctest::timeout(300)) {
  pipeline_2FSP(2, 3);
}

TEST_CASE("Pipeline.2F(SP).2L.4W" * doctest::timeout(300)) {
  pipeline_2FSP(2, 4);
}

TEST_CASE("Pipeline.2F(SP).3L.1W" * doctest::timeout(300)) {
  pipeline_2FSP(3, 1);
}

TEST_CASE("Pipeline.2F(SP).3L.2W" * doctest::timeout(300)) {
  pipeline_2FSP(3, 2);
}

TEST_CASE("Pipeline.2F(SP).3L.3W" * doctest::timeout(300)) {
  pipeline_2FSP(3, 3);
}

TEST_CASE("Pipeline.2F(SP).3L.4W" * doctest::timeout(300)) {
  pipeline_2FSP(3, 4);
}

TEST_CASE("Pipeline.2F(SP).4L.1W" * doctest::timeout(300)) {
  pipeline_2FSP(4, 1);
}

TEST_CASE("Pipeline.2F(SP).4L.2W" * doctest::timeout(300)) {
  pipeline_2FSP(4, 2);
}

TEST_CASE("Pipeline.2F(SP).4L.3W" * doctest::timeout(300)) {
  pipeline_2FSP(4, 3);
}

TEST_CASE("Pipeline.2F(SP).4L.4W" * doctest::timeout(300)) {
  pipeline_2FSP(4, 4);
}

/*
// ----------------------------------------------------------------------------
// two pipes (PS), L lines, W workers
// ----------------------------------------------------------------------------

// TODO: need to discuss the interface
void pipeline_2FPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    size_t j2 = 0;
    std::mutex mutex;
    std::vector<int> collection1;
    std::vector<int> collection2;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &mutex, &collection1](auto& pf) mutable {

        auto ticket = j1.fetch_add(1);

        if(ticket >= N) {
          pf.stop();
          return;
        }

        *(pf.output()) = source[ticket] + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection1.push_back(source[ticket]);
        }
      }},
      tf::Pipe{tf::PipeType::SERIAL, 
      [N, &collection2, &source, &j2](auto& pf) mutable {
        REQUIRE(j2 < N);
        collection2.push_back(*(pf.input()));
        j2++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    
    REQUIRE(collection1.size() == N);
    REQUIRE(j2 == N);

    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());

    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i+1);
    }
    
    j1 = j2 = 0;
    collection1.clear();
    collection2.clear();
    executor.run(taskflow).wait();
    REQUIRE(collection1.size() == N);
    REQUIRE(j2 == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i+1);
    }
  }
}

// two pipes (PS)
//TEST_CASE("Pipeline.2F(PS).1L.1W" * doctest::timeout(300)) {
//  pipeline_2FPS(1, 1);
//}
//
//TEST_CASE("Pipeline.2F(PS).1L.2W" * doctest::timeout(300)) {
//  pipeline_2FPS(1, 2);
//}
//
//TEST_CASE("Pipeline.2F(PS).1L.3W" * doctest::timeout(300)) {
//  pipeline_2FPS(1, 3);
//}
//
//TEST_CASE("Pipeline.2F(PS).1L.4W" * doctest::timeout(300)) {
//  pipeline_2FPS(1, 4);
//}
//
//TEST_CASE("Pipeline.2F(PS).2L.1W" * doctest::timeout(300)) {
//  pipeline_2FPS(2, 1);
//}
//
//TEST_CASE("Pipeline.2F(PS).2L.2W" * doctest::timeout(300)) {
//  pipeline_2FPS(2, 2);
//}
//
//TEST_CASE("Pipeline.2F(PS).2L.3W" * doctest::timeout(300)) {
//  pipeline_2FPS(2, 3);
//}
//
//TEST_CASE("Pipeline.2F(PS).2L.4W" * doctest::timeout(300)) {
//  pipeline_2FPS(2, 4);
//}
//
//TEST_CASE("Pipeline.2F(PS).3L.1W" * doctest::timeout(300)) {
//  pipeline_2FPS(3, 1);
//}
//
//TEST_CASE("Pipeline.2F(PS).3L.2W" * doctest::timeout(300)) {
//  pipeline_2FPS(3, 2);
//}
//
//TEST_CASE("Pipeline.2F(PS).3L.3W" * doctest::timeout(300)) {
//  pipeline_2FPS(3, 3);
//}
//
//TEST_CASE("Pipeline.2F(PS).3L.4W" * doctest::timeout(300)) {
//  pipeline_2FPS(3, 4);
//}
//
//TEST_CASE("Pipeline.2F(PS).4L.1W" * doctest::timeout(300)) {
//  pipeline_2FPS(4, 1);
//}
//
//TEST_CASE("Pipeline.2F(PS).4L.2W" * doctest::timeout(300)) {
//  pipeline_2FPS(4, 2);
//}
//
//TEST_CASE("Pipeline.2F(PS).4L.3W" * doctest::timeout(300)) {
//  pipeline_2FPS(4, 3);
//}
//
//TEST_CASE("Pipeline.2F(PS).4L.4W" * doctest::timeout(300)) {
//  pipeline_2FPS(4, 4);
//}


// ----------------------------------------------------------------------------
// two pipes (PP), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_2FPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex1;
    std::mutex mutex2;
    std::vector<int> collection1;
    std::vector<int> collection2;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &mutex1, &collection1](auto& pf) mutable {
        auto ticket = j1.fetch_add(1);

        if(ticket >= N) {
          pf.stop();
          return;
        }

        *(pf.output()) = source[ticket] + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex1);
          collection1.push_back(source[ticket]);
        }
      }},
      tf::Pipe{tf::PipeType::SERIAL, 
      [N, &collection2, &source, &j2, &mutex2](auto& pf) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          collection2.push_back(*(pf.input()));
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);

    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());

    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
    }
    
    j1 = j2 = 0;
    collection1.clear();
    collection2.clear();
    executor.run(taskflow).wait();
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    for(size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
    }
  }
}

// two pipes (PP)
//TEST_CASE("Pipeline.2F(PP).1L.1W" * doctest::timeout(300)) {
//  pipeline_2FPP(1, 1);
//}
//
//TEST_CASE("Pipeline.2F(PP).1L.2W" * doctest::timeout(300)) {
//  pipeline_2FPP(1, 2);
//}
//
//TEST_CASE("Pipeline.2F(PP).1L.3W" * doctest::timeout(300)) {
//  pipeline_2FPP(1, 3);
//}
//
//TEST_CASE("Pipeline.2F(PP).1L.4W" * doctest::timeout(300)) {
//  pipeline_2FPP(1, 4);
//}
//
//TEST_CASE("Pipeline.2F(PP).2L.1W" * doctest::timeout(300)) {
//  pipeline_2FPP(2, 1);
//}
//
//TEST_CASE("Pipeline.2F(PP).2L.2W" * doctest::timeout(300)) {
//  pipeline_2FPP(2, 2);
//}
//
//TEST_CASE("Pipeline.2F(PP).2L.3W" * doctest::timeout(300)) {
//  pipeline_2FPP(2, 3);
//}
//
//TEST_CASE("Pipeline.2F(PP).2L.4W" * doctest::timeout(300)) {
//  pipeline_2FPP(2, 4);
//}
//
//TEST_CASE("Pipeline.2F(PP).3L.1W" * doctest::timeout(300)) {
//  pipeline_2FPP(3, 1);
//}
//
//TEST_CASE("Pipeline.2F(PP).3L.2W" * doctest::timeout(300)) {
//  pipeline_2FPP(3, 2);
//}
//
//TEST_CASE("Pipeline.2F(PP).3L.3W" * doctest::timeout(300)) {
//  pipeline_2FPP(3, 3);
//}
//
//TEST_CASE("Pipeline.2F(PP).3L.4W" * doctest::timeout(300)) {
//  pipeline_2FPP(3, 4);
//}
//
//TEST_CASE("Pipeline.2F(PP).4L.1W" * doctest::timeout(300)) {
//  pipeline_2FPP(4, 1);
//}
//
//TEST_CASE("Pipeline.2F(PP).4L.2W" * doctest::timeout(300)) {
//  pipeline_2FPP(4, 2);
//}
//
//TEST_CASE("Pipeline.2F(PP).4L.3W" * doctest::timeout(300)) {
//  pipeline_2FPP(4, 3);
//}
//
//TEST_CASE("Pipeline.2F(PP).4L.4W" * doctest::timeout(300)) {
//  pipeline_2FPP(4, 4);
//}
*/

// ----------------------------------------------------------------------------
// three pipes (SSS), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_3FSSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0, j3 = 0;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        REQUIRE(pf.token() % L == pf.line());

        //*(pf.output()) = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(source[j3] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        REQUIRE(pf.token() % L == pf.line());
        j3++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(pl.num_tokens() == 2 * N);
    

    j1 = j2 = j3 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// three pipes (SSS)
TEST_CASE("Pipeline.3F(SSS).1L.1W" * doctest::timeout(300)) {
  pipeline_3FSSS(1, 1);
}

TEST_CASE("Pipeline.3F(SSS).1L.2W" * doctest::timeout(300)) {
  pipeline_3FSSS(1, 2);
}

TEST_CASE("Pipeline.3F(SSS).1L.3W" * doctest::timeout(300)) {
  pipeline_3FSSS(1, 3);
}

TEST_CASE("Pipeline.3F(SSS).1L.4W" * doctest::timeout(300)) {
  pipeline_3FSSS(1, 4);
}

TEST_CASE("Pipeline.3F(SSS).2L.1W" * doctest::timeout(300)) {
  pipeline_3FSSS(2, 1);
}

TEST_CASE("Pipeline.3F(SSS).2L.2W" * doctest::timeout(300)) {
  pipeline_3FSSS(2, 2);
}

TEST_CASE("Pipeline.3F(SSS).2L.3W" * doctest::timeout(300)) {
  pipeline_3FSSS(2, 3);
}

TEST_CASE("Pipeline.3F(SSS).2L.4W" * doctest::timeout(300)) {
  pipeline_3FSSS(2, 4);
}

TEST_CASE("Pipeline.3F(SSS).3L.1W" * doctest::timeout(300)) {
  pipeline_3FSSS(3, 1);
}

TEST_CASE("Pipeline.3F(SSS).3L.2W" * doctest::timeout(300)) {
  pipeline_3FSSS(3, 2);
}

TEST_CASE("Pipeline.3F(SSS).3L.3W" * doctest::timeout(300)) {
  pipeline_3FSSS(3, 3);
}

TEST_CASE("Pipeline.3F(SSS).3L.4W" * doctest::timeout(300)) {
  pipeline_3FSSS(3, 4);
}

TEST_CASE("Pipeline.3F(SSS).4L.1W" * doctest::timeout(300)) {
  pipeline_3FSSS(4, 1);
}

TEST_CASE("Pipeline.3F(SSS).4L.2W" * doctest::timeout(300)) {
  pipeline_3FSSS(4, 2);
}

TEST_CASE("Pipeline.3F(SSS).4L.3W" * doctest::timeout(300)) {
  pipeline_3FSSS(4, 3);
}

TEST_CASE("Pipeline.3F(SSS).4L.4W" * doctest::timeout(300)) {
  pipeline_3FSSS(4, 4);
}



// ----------------------------------------------------------------------------
// three pipes (SSP), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_3FSSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &j3, &mutex, &collection, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());  
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// three pipes (SSP)
TEST_CASE("Pipeline.3F(SSP).1L.1W" * doctest::timeout(300)) {
  pipeline_3FSSP(1, 1);
}

TEST_CASE("Pipeline.3F(SSP).1L.2W" * doctest::timeout(300)) {
  pipeline_3FSSP(1, 2);
}

TEST_CASE("Pipeline.3F(SSP).1L.3W" * doctest::timeout(300)) {
  pipeline_3FSSP(1, 3);
}

TEST_CASE("Pipeline.3F(SSP).1L.4W" * doctest::timeout(300)) {
  pipeline_3FSSP(1, 4);
}

TEST_CASE("Pipeline.3F(SSP).2L.1W" * doctest::timeout(300)) {
  pipeline_3FSSP(2, 1);
}

TEST_CASE("Pipeline.3F(SSP).2L.2W" * doctest::timeout(300)) {
  pipeline_3FSSP(2, 2);
}

TEST_CASE("Pipeline.3F(SSP).2L.3W" * doctest::timeout(300)) {
  pipeline_3FSSP(2, 3);
}

TEST_CASE("Pipeline.3F(SSP).2L.4W" * doctest::timeout(300)) {
  pipeline_3FSSP(2, 4);
}

TEST_CASE("Pipeline.3F(SSP).3L.1W" * doctest::timeout(300)) {
  pipeline_3FSSP(3, 1);
}

TEST_CASE("Pipeline.3F(SSP).3L.2W" * doctest::timeout(300)) {
  pipeline_3FSSP(3, 2);
}

TEST_CASE("Pipeline.3F(SSP).3L.3W" * doctest::timeout(300)) {
  pipeline_3FSSP(3, 3);
}

TEST_CASE("Pipeline.3F(SSP).3L.4W" * doctest::timeout(300)) {
  pipeline_3FSSP(3, 4);
}

TEST_CASE("Pipeline.3F(SSP).4L.1W" * doctest::timeout(300)) {
  pipeline_3FSSP(4, 1);
}

TEST_CASE("Pipeline.3F(SSP).4L.2W" * doctest::timeout(300)) {
  pipeline_3FSSP(4, 2);
}

TEST_CASE("Pipeline.3F(SSP).4L.3W" * doctest::timeout(300)) {
  pipeline_3FSSP(4, 3);
}

TEST_CASE("Pipeline.3F(SSP).4L.4W" * doctest::timeout(300)) {
  pipeline_3FSSP(4, 4);
}



// ----------------------------------------------------------------------------
// three pipes (SPS), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_3FSPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j3 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &j2, &mutex, &collection, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        //*(pf.output()) = *(pf.input()) + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex);
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          REQUIRE(pf.token() % L == pf.line());
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j3] + 2 == mybuffer[pf.line()][pf.pipe() - 1]);
        j3++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// three pipes (SPS)
TEST_CASE("Pipeline.3F(SPS).1L.1W" * doctest::timeout(300)) {
  pipeline_3FSPS(1, 1);
}

TEST_CASE("Pipeline.3F(SPS).1L.2W" * doctest::timeout(300)) {
  pipeline_3FSPS(1, 2);
}

TEST_CASE("Pipeline.3F(SPS).1L.3W" * doctest::timeout(300)) {
  pipeline_3FSPS(1, 3);
}

TEST_CASE("Pipeline.3F(SPS).1L.4W" * doctest::timeout(300)) {
  pipeline_3FSPS(1, 4);
}

TEST_CASE("Pipeline.3F(SPS).2L.1W" * doctest::timeout(300)) {
  pipeline_3FSPS(2, 1);
}

TEST_CASE("Pipeline.3F(SPS).2L.2W" * doctest::timeout(300)) {
  pipeline_3FSPS(2, 2);
}

TEST_CASE("Pipeline.3F(SPS).2L.3W" * doctest::timeout(300)) {
  pipeline_3FSPS(2, 3);
}

TEST_CASE("Pipeline.3F(SPS).2L.4W" * doctest::timeout(300)) {
  pipeline_3FSPS(2, 4);
}

TEST_CASE("Pipeline.3F(SPS).3L.1W" * doctest::timeout(300)) {
  pipeline_3FSPS(3, 1);
}

TEST_CASE("Pipeline.3F(SPS).3L.2W" * doctest::timeout(300)) {
  pipeline_3FSPS(3, 2);
}

TEST_CASE("Pipeline.3F(SPS).3L.3W" * doctest::timeout(300)) {
  pipeline_3FSPS(3, 3);
}

TEST_CASE("Pipeline.3F(SPS).3L.4W" * doctest::timeout(300)) {
  pipeline_3FSPS(3, 4);
}

TEST_CASE("Pipeline.3F(SPS).4L.1W" * doctest::timeout(300)) {
  pipeline_3FSPS(4, 1);
}

TEST_CASE("Pipeline.3F(SPS).4L.2W" * doctest::timeout(300)) {
  pipeline_3FSPS(4, 2);
}

TEST_CASE("Pipeline.3F(SPS).4L.3W" * doctest::timeout(300)) {
  pipeline_3FSPS(4, 3);
}

TEST_CASE("Pipeline.3F(SPS).4L.4W" * doctest::timeout(300)) {
  pipeline_3FSPS(4, 4);
}


// ----------------------------------------------------------------------------
// three pipes (SPP), L lines, W workers
// ----------------------------------------------------------------------------


void pipeline_3FSPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 3>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex2;
    std::mutex mutex3;
    std::vector<int> collection2;
    std::vector<int> collection3;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        //*(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &j2, &mutex2, &collection2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        //*pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &j3, &mutex3, &collection3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          REQUIRE(pf.token() % L == pf.line());
          collection3.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = 0;
    collection2.clear();
    collection3.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = 0;
    collection2.clear();
    collection3.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// three pipes (SPP)
TEST_CASE("Pipeline.3F(SPP).1L.1W" * doctest::timeout(300)) {
  pipeline_3FSPP(1, 1);
}

TEST_CASE("Pipeline.3F(SPP).1L.2W" * doctest::timeout(300)) {
  pipeline_3FSPP(1, 2);
}

TEST_CASE("Pipeline.3F(SPP).1L.3W" * doctest::timeout(300)) {
  pipeline_3FSPP(1, 3);
}

TEST_CASE("Pipeline.3F(SPP).1L.4W" * doctest::timeout(300)) {
  pipeline_3FSPP(1, 4);
}

TEST_CASE("Pipeline.3F(SPP).2L.1W" * doctest::timeout(300)) {
  pipeline_3FSPP(2, 1);
}

TEST_CASE("Pipeline.3F(SPP).2L.2W" * doctest::timeout(300)) {
  pipeline_3FSPP(2, 2);
}

TEST_CASE("Pipeline.3F(SPP).2L.3W" * doctest::timeout(300)) {
  pipeline_3FSPP(2, 3);
}

TEST_CASE("Pipeline.3F(SPP).2L.4W" * doctest::timeout(300)) {
  pipeline_3FSPP(2, 4);
}

TEST_CASE("Pipeline.3F(SPP).3L.1W" * doctest::timeout(300)) {
  pipeline_3FSPP(3, 1);
}

TEST_CASE("Pipeline.3F(SPP).3L.2W" * doctest::timeout(300)) {
  pipeline_3FSPP(3, 2);
}

TEST_CASE("Pipeline.3F(SPP).3L.3W" * doctest::timeout(300)) {
  pipeline_3FSPP(3, 3);
}

TEST_CASE("Pipeline.3F(SPP).3L.4W" * doctest::timeout(300)) {
  pipeline_3FSPP(3, 4);
}

TEST_CASE("Pipeline.3F(SPP).4L.1W" * doctest::timeout(300)) {
  pipeline_3FSPP(4, 1);
}

TEST_CASE("Pipeline.3F(SPP).4L.2W" * doctest::timeout(300)) {
  pipeline_3FSPP(4, 2);
}

TEST_CASE("Pipeline.3F(SPP).4L.3W" * doctest::timeout(300)) {
  pipeline_3FSPP(4, 3);
}

TEST_CASE("Pipeline.3F(SPP).4L.4W" * doctest::timeout(300)) {
  pipeline_3FSPP(4, 4);
}

/*
// ----------------------------------------------------------------------------
// three pipes (PSS), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_3FPSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N=0; N<=maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    size_t j2 = 0, j3 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &collection, &mutex](auto& pf) mutable {
        auto ticket = j1.fetch_add(1);
        
        if(ticket >= N) {
          pf.stop();
          return;
        }
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection.push_back(ticket);
          *(pf.output()) = *(pf.input()) + 1;
        }
      }},
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == *(pf.input()));
        *(pf.output()) = source[j2] + 1;
        j2++;
      }},
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(source[j3] + 1 == *(pf.input()));
        j3++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i);
    }
   
    
    j1 = j2 = j3 = 0;
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i);
    }
    
    j1 = j2 = j3 = 0;
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection[i] == i);
    }
  }
}

// three pipes (PSS)
//TEST_CASE("Pipeline.3F(PSS).1L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSS(1, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSS).1L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSS(1, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSS).1L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSS(1, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSS).1L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSS(1, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSS).2L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSS(2, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSS).2L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSS(2, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSS).2L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSS(2, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSS).2L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSS(2, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSS).3L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSS(3, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSS).3L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSS(3, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSS).3L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSS(3, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSS).3L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSS(3, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSS).4L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSS(4, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSS).4L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSS(4, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSS).4L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSS(4, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSS).4L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSS(4, 4);
//}


// ----------------------------------------------------------------------------
// three pipes (PSP), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_3FPSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    std::atomic<size_t> j3 = 0;
    size_t j2 = 0;
    std::mutex mutex1;
    std::mutex mutex3;
    std::vector<int> collection1;
    std::vector<int> collection3;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &collection1, &mutex1](auto& pf) mutable {
        auto ticket = j1.fetch_add(1);
        
        if(ticket >= N) {
          pf.stop();
          return;
        }
        {
          std::scoped_lock<std::mutex> lock(mutex1);
          collection1.push_back(ticket);
          *(pf.output()) = source[ticket] + 1;
        }
      }},
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == *(pf.input()));
        *(pf.output()) = source[j2] + 1;
        j2++;
      }},
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j3, &mutex3, &collection3](auto& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          collection3.push_back(*(pf.input()));
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection3[i] == i + 2);
    }
   
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection3[i] == i + 2);
    }
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection3[i] == i + 2);
    }
  }
}

// three pipes (PSP)
//TEST_CASE("Pipeline.3F(PSP).1L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSP(1, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSP).1L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSP(1, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSP).1L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSP(1, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSP).1L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSP(1, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSP).2L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSP(2, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSP).2L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSP(2, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSP).2L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSP(2, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSP).2L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSP(2, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSP).3L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSP(3, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSP).3L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSP(3, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSP).3L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSP(3, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSP).3L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSP(3, 4);
//}
//
//TEST_CASE("Pipeline.3F(PSP).4L.1W" * doctest::timeout(300)) {
//  pipeline_3FPSP(4, 1);
//}
//
//TEST_CASE("Pipeline.3F(PSP).4L.2W" * doctest::timeout(300)) {
//  pipeline_3FPSP(4, 2);
//}
//
//TEST_CASE("Pipeline.3F(PSP).4L.3W" * doctest::timeout(300)) {
//  pipeline_3FPSP(4, 3);
//}
//
//TEST_CASE("Pipeline.3F(PSP).4L.4W" * doctest::timeout(300)) {
//  pipeline_3FPSP(4, 4);
//}


// ----------------------------------------------------------------------------
// three pipes (PPS), L lines, W workers
// ----------------------------------------------------------------------------


void pipeline_3FPPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    std::atomic<size_t> j2 = 0;
    size_t j3 = 0;
    std::mutex mutex1;
    std::mutex mutex2;
    std::vector<int> collection1;
    std::vector<int> collection2;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &collection1, &mutex1](auto& pf) mutable {
        auto ticket = j1.fetch_add(1);
        
        if(ticket >= N) {
          pf.stop();
          return;
        }
        {
          std::scoped_lock<std::mutex> lock(mutex1);
          collection1.push_back(ticket);
          *(pf.output()) = source[ticket] + 1;
        }
      }},
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j2, &mutex2, &collection2](auto& pf) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          collection2.push_back(*(pf.input()));
          *(pf.output()) = *(pf.input()) + 1;
        }
      }},
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(source[j3] + 1 == *(pf.input()));
        j3++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
    }
   
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection2.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
    }
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection2.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
    }
  }
}

// three pipes (PPS)
//TEST_CASE("Pipeline.3F(PPS).1L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPS(1, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPS).1L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPS(1, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPS).1L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPS(1, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPS).1L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPS(1, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPS).2L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPS(2, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPS).2L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPS(2, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPS).2L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPS(2, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPS).2L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPS(2, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPS).3L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPS(3, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPS).3L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPS(3, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPS).3L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPS(3, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPS).3L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPS(3, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPS).4L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPS(4, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPS).4L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPS(4, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPS).4L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPS(4, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPS).4L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPS(4, 4);
//}


// ----------------------------------------------------------------------------
// three pipes (PPP), L lines, W workers
// ----------------------------------------------------------------------------


void pipeline_3FPPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    std::atomic<size_t> j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex1;
    std::mutex mutex2;
    std::mutex mutex3;
    std::vector<int> collection1;
    std::vector<int> collection2;
    std::vector<int> collection3;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j1, &collection1, &mutex1](auto& pf) mutable {
        auto ticket = j1.fetch_add(1);
        
        if(ticket >= N) {
          pf.stop();
          return;
        }
        {
          std::scoped_lock<std::mutex> lock(mutex1);
          collection1.push_back(ticket);
          *(pf.output()) = source[ticket] + 1;
        }
      }},
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j2, &mutex2, &collection2](auto& pf) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          collection2.push_back(*(pf.input()));
          *(pf.output()) = *(pf.input()) + 1;
        }
      }},
      tf::Pipe{tf::PipeType::PARALLEL, 
      [N, &source, &j3, &mutex3, &collection3](auto& pf) mutable {
        REQUIRE(j3++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          collection3.push_back(*(pf.input()));
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
   
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection2.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
    
    j1 = j2 = j3 = 0;
    collection1.clear();
    collection2.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(collection1.size() == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection1.begin(), collection1.end());
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; i++) {
      REQUIRE(collection1[i] == i);
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
  }
}

// three pipes (PPP)
//TEST_CASE("Pipeline.3F(PPP).1L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPP(1, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPP).1L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPP(1, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPP).1L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPP(1, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPP).1L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPP(1, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPP).2L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPP(2, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPP).2L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPP(2, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPP).2L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPP(2, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPP).2L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPP(2, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPP).3L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPP(3, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPP).3L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPP(3, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPP).3L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPP(3, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPP).3L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPP(3, 4);
//}
//
//TEST_CASE("Pipeline.3F(PPP).4L.1W" * doctest::timeout(300)) {
//  pipeline_3FPPP(4, 1);
//}
//
//TEST_CASE("Pipeline.3F(PPP).4L.2W" * doctest::timeout(300)) {
//  pipeline_3FPPP(4, 2);
//}
//
//TEST_CASE("Pipeline.3F(PPP).4L.3W" * doctest::timeout(300)) {
//  pipeline_3FPPP(4, 3);
//}
//
//TEST_CASE("Pipeline.3F(PPP).4L.4W" * doctest::timeout(300)) {
//  pipeline_3FPPP(4, 4);
//}



// ----------------------------------------------------------------------------
// four pipes (SSSS), L lines, W workers
// ----------------------------------------------------------------------------


void pipeline_4FSSSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0, j3 = 0, j4 = 0;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // *(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // REQUIRE(source[j2] + 1 == *(pf.input()));
        // *(pf.output()) = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        // REQUIRE(source[j3] + 1 == *(pf.input()));
        REQUIRE(source[j3] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // *(pf.output()) = source[j3] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j3] + 1;
        j3++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4 < N);
        REQUIRE(pf.token() % L == pf.line());
        // REQUIRE(source[j4] + 1 == *(pf.input()));
        REQUIRE(source[j4] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        j4++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SSSS)
TEST_CASE("Pipeline.4F(SSSS).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 1);
}

TEST_CASE("Pipeline.4F(SSSS).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 2);
}

TEST_CASE("Pipeline.4F(SSSS).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 3);
}

TEST_CASE("Pipeline.4F(SSSS).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 4);
}

TEST_CASE("Pipeline.4F(SSSS).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 5);
}

TEST_CASE("Pipeline.4F(SSSS).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 6);
}

TEST_CASE("Pipeline.4F(SSSS).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 7);
}

TEST_CASE("Pipeline.4F(SSSS).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(1, 8);
}

TEST_CASE("Pipeline.4F(SSSS).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 1);
}

TEST_CASE("Pipeline.4F(SSSS).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 2);
}

TEST_CASE("Pipeline.4F(SSSS).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 3);
}

TEST_CASE("Pipeline.4F(SSSS).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 4);
}

TEST_CASE("Pipeline.4F(SSSS).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 5);
}

TEST_CASE("Pipeline.4F(SSSS).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 6);
}

TEST_CASE("Pipeline.4F(SSSS).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 7);
}

TEST_CASE("Pipeline.4F(SSSS).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(2, 8);
}

TEST_CASE("Pipeline.4F(SSSS).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 1);
}

TEST_CASE("Pipeline.4F(SSSS).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 2);
}

TEST_CASE("Pipeline.4F(SSSS).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 3);
}

TEST_CASE("Pipeline.4F(SSSS).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 4);
}

TEST_CASE("Pipeline.4F(SSSS).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 5);
}

TEST_CASE("Pipeline.4F(SSSS).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 6);
}

TEST_CASE("Pipeline.4F(SSSS).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 7);
}

TEST_CASE("Pipeline.4F(SSSS).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(3, 8);
}

TEST_CASE("Pipeline.4F(SSSS).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 1);
}

TEST_CASE("Pipeline.4F(SSSS).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 2);
}

TEST_CASE("Pipeline.4F(SSSS).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 3);
}

TEST_CASE("Pipeline.4F(SSSS).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 4);
}

TEST_CASE("Pipeline.4F(SSSS).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 5);
}

TEST_CASE("Pipeline.4F(SSSS).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 6);
}

TEST_CASE("Pipeline.4F(SSSS).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 7);
}

TEST_CASE("Pipeline.4F(SSSS).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(4, 8);
}

TEST_CASE("Pipeline.4F(SSSS).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 1);
}

TEST_CASE("Pipeline.4F(SSSS).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 2);
}

TEST_CASE("Pipeline.4F(SSSS).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 3);
}

TEST_CASE("Pipeline.4F(SSSS).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 4);
}

TEST_CASE("Pipeline.4F(SSSS).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 5);
}

TEST_CASE("Pipeline.4F(SSSS).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 6);
}

TEST_CASE("Pipeline.4F(SSSS).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 7);
}

TEST_CASE("Pipeline.4F(SSSS).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(5, 8);
}

TEST_CASE("Pipeline.4F(SSSS).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 1);
}

TEST_CASE("Pipeline.4F(SSSS).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 2);
}

TEST_CASE("Pipeline.4F(SSSS).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 3);
}

TEST_CASE("Pipeline.4F(SSSS).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 4);
}

TEST_CASE("Pipeline.4F(SSSS).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 5);
}

TEST_CASE("Pipeline.4F(SSSS).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 6);
}

TEST_CASE("Pipeline.4F(SSSS).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 7);
}

TEST_CASE("Pipeline.4F(SSSS).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(6, 8);
}

TEST_CASE("Pipeline.4F(SSSS).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 1);
}

TEST_CASE("Pipeline.4F(SSSS).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 2);
}

TEST_CASE("Pipeline.4F(SSSS).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 3);
}

TEST_CASE("Pipeline.4F(SSSS).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 4);
}

TEST_CASE("Pipeline.4F(SSSS).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 5);
}

TEST_CASE("Pipeline.4F(SSSS).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 6);
}

TEST_CASE("Pipeline.4F(SSSS).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 7);
}

TEST_CASE("Pipeline.4F(SSSS).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(7, 8);
}

TEST_CASE("Pipeline.4F(SSSS).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 1);
}

TEST_CASE("Pipeline.4F(SSSS).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 2);
}

TEST_CASE("Pipeline.4F(SSSS).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 3);
}

TEST_CASE("Pipeline.4F(SSSS).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 4);
}

TEST_CASE("Pipeline.4F(SSSS).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 5);
}

TEST_CASE("Pipeline.4F(SSSS).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 6);
}

TEST_CASE("Pipeline.4F(SSSS).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 7);
}

TEST_CASE("Pipeline.4F(SSSS).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSS(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SSSP), L lines, W workers
// ----------------------------------------------------------------------------


void pipeline_4FSSSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0, j3 = 0;
    std::atomic<size_t> j4 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // *(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        //REQUIRE(source[j2] + 1 == *(pf.input()));
        // *(pf.output()) = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j3] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        //REQUIRE(source[j3] + 1 == *(pf.input()));
        // *(pf.output()) = source[j3] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j3] + 1;
        j3++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j4, &mutex, &collection, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          //collection.push_back(*pf.input());
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    collection.clear();
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SSSP)
TEST_CASE("Pipeline.4F(SSSP).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 1);
}

TEST_CASE("Pipeline.4F(SSSP).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 2);
}

TEST_CASE("Pipeline.4F(SSSP).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 3);
}

TEST_CASE("Pipeline.4F(SSSP).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 4);
}

TEST_CASE("Pipeline.4F(SSSP).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 5);
}

TEST_CASE("Pipeline.4F(SSSP).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 6);
}

TEST_CASE("Pipeline.4F(SSSP).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 7);
}

TEST_CASE("Pipeline.4F(SSSP).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(1, 8);
}

TEST_CASE("Pipeline.4F(SSSP).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 1);
}

TEST_CASE("Pipeline.4F(SSSP).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 2);
}

TEST_CASE("Pipeline.4F(SSSP).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 3);
}

TEST_CASE("Pipeline.4F(SSSP).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 4);
}

TEST_CASE("Pipeline.4F(SSSP).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 5);
}

TEST_CASE("Pipeline.4F(SSSP).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 6);
}

TEST_CASE("Pipeline.4F(SSSP).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 7);
}

TEST_CASE("Pipeline.4F(SSSP).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(2, 8);
}

TEST_CASE("Pipeline.4F(SSSP).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 1);
}

TEST_CASE("Pipeline.4F(SSSP).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 2);
}

TEST_CASE("Pipeline.4F(SSSP).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 3);
}

TEST_CASE("Pipeline.4F(SSSP).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 4);
}

TEST_CASE("Pipeline.4F(SSSP).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 5);
}

TEST_CASE("Pipeline.4F(SSSP).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 6);
}

TEST_CASE("Pipeline.4F(SSSP).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 7);
}

TEST_CASE("Pipeline.4F(SSSP).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(3, 8);
}

TEST_CASE("Pipeline.4F(SSSP).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 1);
}

TEST_CASE("Pipeline.4F(SSSP).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 2);
}

TEST_CASE("Pipeline.4F(SSSP).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 3);
}

TEST_CASE("Pipeline.4F(SSSP).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 4);
}

TEST_CASE("Pipeline.4F(SSSP).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 5);
}

TEST_CASE("Pipeline.4F(SSSP).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 6);
}

TEST_CASE("Pipeline.4F(SSSP).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 7);
}

TEST_CASE("Pipeline.4F(SSSP).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(4, 8);
}

TEST_CASE("Pipeline.4F(SSSP).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 1);
}

TEST_CASE("Pipeline.4F(SSSP).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 2);
}

TEST_CASE("Pipeline.4F(SSSP).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 3);
}

TEST_CASE("Pipeline.4F(SSSP).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 4);
}

TEST_CASE("Pipeline.4F(SSSP).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 5);
}

TEST_CASE("Pipeline.4F(SSSP).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 6);
}

TEST_CASE("Pipeline.4F(SSSP).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 7);
}

TEST_CASE("Pipeline.4F(SSSP).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(5, 8);
}

TEST_CASE("Pipeline.4F(SSSP).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 1);
}

TEST_CASE("Pipeline.4F(SSSP).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 2);
}

TEST_CASE("Pipeline.4F(SSSP).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 3);
}

TEST_CASE("Pipeline.4F(SSSP).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 4);
}

TEST_CASE("Pipeline.4F(SSSP).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 5);
}

TEST_CASE("Pipeline.4F(SSSP).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 6);
}

TEST_CASE("Pipeline.4F(SSSP).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 7);
}

TEST_CASE("Pipeline.4F(SSSP).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(6, 8);
}

TEST_CASE("Pipeline.4F(SSSP).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 1);
}

TEST_CASE("Pipeline.4F(SSSP).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 2);
}

TEST_CASE("Pipeline.4F(SSSP).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 3);
}

TEST_CASE("Pipeline.4F(SSSP).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 4);
}

TEST_CASE("Pipeline.4F(SSSP).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 5);
}

TEST_CASE("Pipeline.4F(SSSP).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 6);
}

TEST_CASE("Pipeline.4F(SSSP).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 7);
}

TEST_CASE("Pipeline.4F(SSSP).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(7, 8);
}

TEST_CASE("Pipeline.4F(SSSP).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 1);
}

TEST_CASE("Pipeline.4F(SSSP).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 2);
}

TEST_CASE("Pipeline.4F(SSSP).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 3);
}

TEST_CASE("Pipeline.4F(SSSP).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 4);
}

TEST_CASE("Pipeline.4F(SSSP).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 5);
}

TEST_CASE("Pipeline.4F(SSSP).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 6);
}

TEST_CASE("Pipeline.4F(SSSP).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 7);
}

TEST_CASE("Pipeline.4F(SSSP).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSSSP(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SSPS), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_4FSSPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0, j4 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // *(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // REQUIRE(source[j2] + 1 == *(pf.input()));
        // *pf.output() = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j3, &mutex, &collection, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        // *(pf.output()) = *(pf.input()) + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          //collection.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4 < N);
        REQUIRE(pf.token() % L == pf.line());
        // REQUIRE(source[j4] + 2 == *(pf.input()));
        REQUIRE(source[j4] + 2 == mybuffer[pf.line()][pf.pipe() - 1]);
        j4++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SSPS)
TEST_CASE("Pipeline.4F(SSPS).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 1);
}

TEST_CASE("Pipeline.4F(SSPS).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 2);
}

TEST_CASE("Pipeline.4F(SSPS).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 3);
}

TEST_CASE("Pipeline.4F(SSPS).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 4);
}

TEST_CASE("Pipeline.4F(SSPS).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 5);
}

TEST_CASE("Pipeline.4F(SSPS).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 6);
}

TEST_CASE("Pipeline.4F(SSPS).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 7);
}

TEST_CASE("Pipeline.4F(SSPS).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(1, 8);
}

TEST_CASE("Pipeline.4F(SSPS).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 1);
}

TEST_CASE("Pipeline.4F(SSPS).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 2);
}

TEST_CASE("Pipeline.4F(SSPS).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 3);
}

TEST_CASE("Pipeline.4F(SSPS).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 4);
}

TEST_CASE("Pipeline.4F(SSPS).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 5);
}

TEST_CASE("Pipeline.4F(SSPS).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 6);
}

TEST_CASE("Pipeline.4F(SSPS).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 7);
}

TEST_CASE("Pipeline.4F(SSPS).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(2, 8);
}

TEST_CASE("Pipeline.4F(SSPS).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 1);
}

TEST_CASE("Pipeline.4F(SSPS).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 2);
}

TEST_CASE("Pipeline.4F(SSPS).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 3);
}

TEST_CASE("Pipeline.4F(SSPS).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 4);
}

TEST_CASE("Pipeline.4F(SSPS).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 5);
}

TEST_CASE("Pipeline.4F(SSPS).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 6);
}

TEST_CASE("Pipeline.4F(SSPS).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 7);
}

TEST_CASE("Pipeline.4F(SSPS).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(3, 8);
}

TEST_CASE("Pipeline.4F(SSPS).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 1);
}

TEST_CASE("Pipeline.4F(SSPS).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 2);
}

TEST_CASE("Pipeline.4F(SSPS).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 3);
}

TEST_CASE("Pipeline.4F(SSPS).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 4);
}

TEST_CASE("Pipeline.4F(SSPS).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 5);
}

TEST_CASE("Pipeline.4F(SSPS).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 6);
}

TEST_CASE("Pipeline.4F(SSPS).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 7);
}

TEST_CASE("Pipeline.4F(SSPS).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(4, 8);
}

TEST_CASE("Pipeline.4F(SSPS).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 1);
}

TEST_CASE("Pipeline.4F(SSPS).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 2);
}

TEST_CASE("Pipeline.4F(SSPS).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 3);
}

TEST_CASE("Pipeline.4F(SSPS).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 4);
}

TEST_CASE("Pipeline.4F(SSPS).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 5);
}

TEST_CASE("Pipeline.4F(SSPS).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 6);
}

TEST_CASE("Pipeline.4F(SSPS).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 7);
}

TEST_CASE("Pipeline.4F(SSPS).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(5, 8);
}

TEST_CASE("Pipeline.4F(SSPS).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 1);
}

TEST_CASE("Pipeline.4F(SSPS).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 2);
}

TEST_CASE("Pipeline.4F(SSPS).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 3);
}

TEST_CASE("Pipeline.4F(SSPS).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 4);
}

TEST_CASE("Pipeline.4F(SSPS).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 5);
}

TEST_CASE("Pipeline.4F(SSPS).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 6);
}

TEST_CASE("Pipeline.4F(SSPS).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 7);
}

TEST_CASE("Pipeline.4F(SSPS).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(6, 8);
}

TEST_CASE("Pipeline.4F(SSPS).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 1);
}

TEST_CASE("Pipeline.4F(SSPS).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 2);
}

TEST_CASE("Pipeline.4F(SSPS).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 3);
}

TEST_CASE("Pipeline.4F(SSPS).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 4);
}

TEST_CASE("Pipeline.4F(SSPS).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 5);
}

TEST_CASE("Pipeline.4F(SSPS).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 6);
}

TEST_CASE("Pipeline.4F(SSPS).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 7);
}

TEST_CASE("Pipeline.4F(SSPS).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(7, 8);
}

TEST_CASE("Pipeline.4F(SSPS).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 1);
}

TEST_CASE("Pipeline.4F(SSPS).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 2);
}

TEST_CASE("Pipeline.4F(SSPS).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 3);
}

TEST_CASE("Pipeline.4F(SSPS).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 4);
}

TEST_CASE("Pipeline.4F(SSPS).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 5);
}

TEST_CASE("Pipeline.4F(SSPS).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 6);
}

TEST_CASE("Pipeline.4F(SSPS).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 7);
}

TEST_CASE("Pipeline.4F(SSPS).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPS(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SSPP), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_4FSSPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::atomic<size_t> j4 = 0;
    std::mutex mutex3;
    std::mutex mutex4;
    std::vector<int> collection3;
    std::vector<int> collection4;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // *(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j2] + 1 == mybuffer[pf.line()][pf.pipe() - 1]);
        // REQUIRE(source[j2] + 1 == *(pf.input()));
        // *pf.output() = source[j2] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j2] + 1;
        j2++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j3, &mutex3, &collection3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        // *pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          //collection3.push_back(*pf.input());
          collection3.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j4, &mutex4, &collection4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex4);
          REQUIRE(pf.token() % L == pf.line());
          //collection4.push_back(*pf.input());
          collection4.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection3[i] == i + 1);
      REQUIRE(collection4[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection3.clear();
    collection4.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection3[i] == i + 1);
      REQUIRE(collection4[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection3.clear();
    collection4.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection3[i] == i + 1);
      REQUIRE(collection4[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SSPP)
TEST_CASE("Pipeline.4F(SSPP).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 1);
}

TEST_CASE("Pipeline.4F(SSPP).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 2);
}

TEST_CASE("Pipeline.4F(SSPP).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 3);
}

TEST_CASE("Pipeline.4F(SSPP).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 4);
}

TEST_CASE("Pipeline.4F(SSPP).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 5);
}

TEST_CASE("Pipeline.4F(SSPP).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 6);
}

TEST_CASE("Pipeline.4F(SSPP).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 7);
}

TEST_CASE("Pipeline.4F(SSPP).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(1, 8);
}

TEST_CASE("Pipeline.4F(SSPP).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 1);
}

TEST_CASE("Pipeline.4F(SSPP).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 2);
}

TEST_CASE("Pipeline.4F(SSPP).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 3);
}

TEST_CASE("Pipeline.4F(SSPP).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 4);
}

TEST_CASE("Pipeline.4F(SSPP).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 5);
}

TEST_CASE("Pipeline.4F(SSPP).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 6);
}

TEST_CASE("Pipeline.4F(SSPP).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 7);
}

TEST_CASE("Pipeline.4F(SSPP).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(2, 8);
}

TEST_CASE("Pipeline.4F(SSPP).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 1);
}

TEST_CASE("Pipeline.4F(SSPP).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 2);
}

TEST_CASE("Pipeline.4F(SSPP).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 3);
}

TEST_CASE("Pipeline.4F(SSPP).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 4);
}

TEST_CASE("Pipeline.4F(SSPP).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 5);
}

TEST_CASE("Pipeline.4F(SSPP).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 6);
}

TEST_CASE("Pipeline.4F(SSPP).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 7);
}

TEST_CASE("Pipeline.4F(SSPP).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(3, 8);
}

TEST_CASE("Pipeline.4F(SSPP).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 1);
}

TEST_CASE("Pipeline.4F(SSPP).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 2);
}

TEST_CASE("Pipeline.4F(SSPP).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 3);
}

TEST_CASE("Pipeline.4F(SSPP).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 4);
}

TEST_CASE("Pipeline.4F(SSPP).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 5);
}

TEST_CASE("Pipeline.4F(SSPP).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 6);
}

TEST_CASE("Pipeline.4F(SSPP).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 7);
}

TEST_CASE("Pipeline.4F(SSPP).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(4, 8);
}

TEST_CASE("Pipeline.4F(SSPP).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 1);
}

TEST_CASE("Pipeline.4F(SSPP).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 2);
}

TEST_CASE("Pipeline.4F(SSPP).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 3);
}

TEST_CASE("Pipeline.4F(SSPP).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 4);
}

TEST_CASE("Pipeline.4F(SSPP).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 5);
}

TEST_CASE("Pipeline.4F(SSPP).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 6);
}

TEST_CASE("Pipeline.4F(SSPP).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 7);
}

TEST_CASE("Pipeline.4F(SSPP).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(5, 8);
}

TEST_CASE("Pipeline.4F(SSPP).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 1);
}

TEST_CASE("Pipeline.4F(SSPP).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 2);
}

TEST_CASE("Pipeline.4F(SSPP).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 3);
}

TEST_CASE("Pipeline.4F(SSPP).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 4);
}

TEST_CASE("Pipeline.4F(SSPP).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 5);
}

TEST_CASE("Pipeline.4F(SSPP).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 6);
}

TEST_CASE("Pipeline.4F(SSPP).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 7);
}

TEST_CASE("Pipeline.4F(SSPP).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(6, 8);
}

TEST_CASE("Pipeline.4F(SSPP).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 1);
}

TEST_CASE("Pipeline.4F(SSPP).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 2);
}

TEST_CASE("Pipeline.4F(SSPP).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 3);
}

TEST_CASE("Pipeline.4F(SSPP).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 4);
}

TEST_CASE("Pipeline.4F(SSPP).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 5);
}

TEST_CASE("Pipeline.4F(SSPP).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 6);
}

TEST_CASE("Pipeline.4F(SSPP).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 7);
}

TEST_CASE("Pipeline.4F(SSPP).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(7, 8);
}

TEST_CASE("Pipeline.4F(SSPP).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 1);
}

TEST_CASE("Pipeline.4F(SSPP).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 2);
}

TEST_CASE("Pipeline.4F(SSPP).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 3);
}

TEST_CASE("Pipeline.4F(SSPP).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 4);
}

TEST_CASE("Pipeline.4F(SSPP).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 5);
}

TEST_CASE("Pipeline.4F(SSPP).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 6);
}

TEST_CASE("Pipeline.4F(SSPP).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 7);
}

TEST_CASE("Pipeline.4F(SSPP).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSSPP(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SPSS), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_4FSPSS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j3 = 0, j4 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        // *(pf.output()) = source[j1] + 1;
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j2, &mutex, &collection, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        // *(pf.output()) = *(pf.input()) + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          //collection.push_back(*pf.input());
          collection.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j3] + 2 == mybuffer[pf.line()][pf.pipe() - 1]);
        mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
        // REQUIRE(source[j3] + 2 == *(pf.input()));
        // *pf.output() = *pf.input() + 1;
        j3++;
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j4] + 3 == mybuffer[pf.line()][pf.pipe() - 1]);
        //REQUIRE(source[j4] + 3 == *(pf.input()));
        j4++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection.size() == N);
    std::sort(collection.begin(), collection.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection[i] == i + 1);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SPSS)
TEST_CASE("Pipeline.4F(SPSS).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 1);
}

TEST_CASE("Pipeline.4F(SPSS).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 2);
}

TEST_CASE("Pipeline.4F(SPSS).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 3);
}

TEST_CASE("Pipeline.4F(SPSS).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 4);
}

TEST_CASE("Pipeline.4F(SPSS).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 5);
}

TEST_CASE("Pipeline.4F(SPSS).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 6);
}

TEST_CASE("Pipeline.4F(SPSS).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 7);
}

TEST_CASE("Pipeline.4F(SPSS).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(1, 8);
}

TEST_CASE("Pipeline.4F(SPSS).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 1);
}

TEST_CASE("Pipeline.4F(SPSS).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 2);
}

TEST_CASE("Pipeline.4F(SPSS).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 3);
}

TEST_CASE("Pipeline.4F(SPSS).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 4);
}

TEST_CASE("Pipeline.4F(SPSS).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 5);
}

TEST_CASE("Pipeline.4F(SPSS).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 6);
}

TEST_CASE("Pipeline.4F(SPSS).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 7);
}

TEST_CASE("Pipeline.4F(SPSS).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(2, 8);
}

TEST_CASE("Pipeline.4F(SPSS).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 1);
}

TEST_CASE("Pipeline.4F(SPSS).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 2);
}

TEST_CASE("Pipeline.4F(SPSS).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 3);
}

TEST_CASE("Pipeline.4F(SPSS).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 4);
}

TEST_CASE("Pipeline.4F(SPSS).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 5);
}

TEST_CASE("Pipeline.4F(SPSS).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 6);
}

TEST_CASE("Pipeline.4F(SPSS).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 7);
}

TEST_CASE("Pipeline.4F(SPSS).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(3, 8);
}

TEST_CASE("Pipeline.4F(SPSS).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 1);
}

TEST_CASE("Pipeline.4F(SPSS).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 2);
}

TEST_CASE("Pipeline.4F(SPSS).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 3);
}

TEST_CASE("Pipeline.4F(SPSS).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 4);
}

TEST_CASE("Pipeline.4F(SPSS).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 5);
}

TEST_CASE("Pipeline.4F(SPSS).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 6);
}

TEST_CASE("Pipeline.4F(SPSS).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 7);
}

TEST_CASE("Pipeline.4F(SPSS).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(4, 8);
}

TEST_CASE("Pipeline.4F(SPSS).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 1);
}

TEST_CASE("Pipeline.4F(SPSS).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 2);
}

TEST_CASE("Pipeline.4F(SPSS).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 3);
}

TEST_CASE("Pipeline.4F(SPSS).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 4);
}

TEST_CASE("Pipeline.4F(SPSS).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 5);
}

TEST_CASE("Pipeline.4F(SPSS).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 6);
}

TEST_CASE("Pipeline.4F(SPSS).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 7);
}

TEST_CASE("Pipeline.4F(SPSS).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(5, 8);
}

TEST_CASE("Pipeline.4F(SPSS).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 1);
}

TEST_CASE("Pipeline.4F(SPSS).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 2);
}

TEST_CASE("Pipeline.4F(SPSS).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 3);
}

TEST_CASE("Pipeline.4F(SPSS).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 4);
}

TEST_CASE("Pipeline.4F(SPSS).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 5);
}

TEST_CASE("Pipeline.4F(SPSS).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 6);
}

TEST_CASE("Pipeline.4F(SPSS).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 7);
}

TEST_CASE("Pipeline.4F(SPSS).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(6, 8);
}

TEST_CASE("Pipeline.4F(SPSS).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 1);
}

TEST_CASE("Pipeline.4F(SPSS).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 2);
}

TEST_CASE("Pipeline.4F(SPSS).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 3);
}

TEST_CASE("Pipeline.4F(SPSS).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 4);
}

TEST_CASE("Pipeline.4F(SPSS).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 5);
}

TEST_CASE("Pipeline.4F(SPSS).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 6);
}

TEST_CASE("Pipeline.4F(SPSS).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 7);
}

TEST_CASE("Pipeline.4F(SPSS).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(7, 8);
}

TEST_CASE("Pipeline.4F(SPSS).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 1);
}

TEST_CASE("Pipeline.4F(SPSS).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 2);
}

TEST_CASE("Pipeline.4F(SPSS).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 3);
}

TEST_CASE("Pipeline.4F(SPSS).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 4);
}

TEST_CASE("Pipeline.4F(SPSS).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 5);
}

TEST_CASE("Pipeline.4F(SPSS).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 6);
}

TEST_CASE("Pipeline.4F(SPSS).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 7);
}

TEST_CASE("Pipeline.4F(SPSS).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSS(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SPSP), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_4FSPSP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j3 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j4 = 0;
    std::mutex mutex2;
    std::mutex mutex4;
    std::vector<int> collection2;
    std::vector<int> collection4;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // *(pf.output()) = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j2, &mutex2, &collection2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        // *(pf.output()) = *(pf.input()) + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          //collection2.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j3] + 2 == mybuffer[pf.line()][pf.pipe() - 1]);
        mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
        // REQUIRE(source[j3] + 2 == *(pf.input()));
        // *pf.output() = *pf.input() + 1;
        j3++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j4, &mutex4, &collection4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex4);
          REQUIRE(pf.token() % L == pf.line());
          collection4.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          //collection4.push_back(*pf.input());
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection4.clear();
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection4.clear();
    executor.run(taskflow).wait();
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
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SPSP)
TEST_CASE("Pipeline.4F(SPSP).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 1);
}

TEST_CASE("Pipeline.4F(SPSP).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 2);
}

TEST_CASE("Pipeline.4F(SPSP).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 3);
}

TEST_CASE("Pipeline.4F(SPSP).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 4);
}

TEST_CASE("Pipeline.4F(SPSP).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 5);
}

TEST_CASE("Pipeline.4F(SPSP).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 6);
}

TEST_CASE("Pipeline.4F(SPSP).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 7);
}

TEST_CASE("Pipeline.4F(SPSP).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(1, 8);
}

TEST_CASE("Pipeline.4F(SPSP).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 1);
}

TEST_CASE("Pipeline.4F(SPSP).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 2);
}

TEST_CASE("Pipeline.4F(SPSP).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 3);
}

TEST_CASE("Pipeline.4F(SPSP).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 4);
}

TEST_CASE("Pipeline.4F(SPSP).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 5);
}

TEST_CASE("Pipeline.4F(SPSP).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 6);
}

TEST_CASE("Pipeline.4F(SPSP).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 7);
}

TEST_CASE("Pipeline.4F(SPSP).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(2, 8);
}

TEST_CASE("Pipeline.4F(SPSP).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 1);
}

TEST_CASE("Pipeline.4F(SPSP).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 2);
}

TEST_CASE("Pipeline.4F(SPSP).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 3);
}

TEST_CASE("Pipeline.4F(SPSP).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 4);
}

TEST_CASE("Pipeline.4F(SPSP).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 5);
}

TEST_CASE("Pipeline.4F(SPSP).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 6);
}

TEST_CASE("Pipeline.4F(SPSP).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 7);
}

TEST_CASE("Pipeline.4F(SPSP).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(3, 8);
}

TEST_CASE("Pipeline.4F(SPSP).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 1);
}

TEST_CASE("Pipeline.4F(SPSP).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 2);
}

TEST_CASE("Pipeline.4F(SPSP).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 3);
}

TEST_CASE("Pipeline.4F(SPSP).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 4);
}

TEST_CASE("Pipeline.4F(SPSP).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 5);
}

TEST_CASE("Pipeline.4F(SPSP).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 6);
}

TEST_CASE("Pipeline.4F(SPSP).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 7);
}

TEST_CASE("Pipeline.4F(SPSP).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(4, 8);
}

TEST_CASE("Pipeline.4F(SPSP).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 1);
}

TEST_CASE("Pipeline.4F(SPSP).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 2);
}

TEST_CASE("Pipeline.4F(SPSP).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 3);
}

TEST_CASE("Pipeline.4F(SPSP).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 4);
}

TEST_CASE("Pipeline.4F(SPSP).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 5);
}

TEST_CASE("Pipeline.4F(SPSP).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 6);
}

TEST_CASE("Pipeline.4F(SPSP).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 7);
}

TEST_CASE("Pipeline.4F(SPSP).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(5, 8);
}

TEST_CASE("Pipeline.4F(SPSP).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 1);
}

TEST_CASE("Pipeline.4F(SPSP).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 2);
}

TEST_CASE("Pipeline.4F(SPSP).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 3);
}

TEST_CASE("Pipeline.4F(SPSP).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 4);
}

TEST_CASE("Pipeline.4F(SPSP).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 5);
}

TEST_CASE("Pipeline.4F(SPSP).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 6);
}

TEST_CASE("Pipeline.4F(SPSP).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 7);
}

TEST_CASE("Pipeline.4F(SPSP).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(6, 8);
}

TEST_CASE("Pipeline.4F(SPSP).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 1);
}

TEST_CASE("Pipeline.4F(SPSP).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 2);
}

TEST_CASE("Pipeline.4F(SPSP).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 3);
}

TEST_CASE("Pipeline.4F(SPSP).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 4);
}

TEST_CASE("Pipeline.4F(SPSP).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 5);
}

TEST_CASE("Pipeline.4F(SPSP).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 6);
}

TEST_CASE("Pipeline.4F(SPSP).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 7);
}

TEST_CASE("Pipeline.4F(SPSP).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(7, 8);
}

TEST_CASE("Pipeline.4F(SPSP).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 1);
}

TEST_CASE("Pipeline.4F(SPSP).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 2);
}

TEST_CASE("Pipeline.4F(SPSP).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 3);
}

TEST_CASE("Pipeline.4F(SPSP).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 4);
}

TEST_CASE("Pipeline.4F(SPSP).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 5);
}

TEST_CASE("Pipeline.4F(SPSP).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 6);
}

TEST_CASE("Pipeline.4F(SPSP).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 7);
}

TEST_CASE("Pipeline.4F(SPSP).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSPSP(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SPPS), L lines, W workers
// ----------------------------------------------------------------------------
void pipeline_4FSPPS(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j4 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::mutex mutex2;
    std::mutex mutex3;
    std::vector<int> collection2;
    std::vector<int> collection3;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // *(pf.output()) = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j2, &mutex2, &collection2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        // *pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          //collection2.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j3, &mutex3, &collection3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        // *pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection3.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          //collection3.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4 < N);
        REQUIRE(pf.token() % L == pf.line());
        REQUIRE(source[j4] + 3 == mybuffer[pf.line()][pf.pipe() - 1]);
        //REQUIRE(source[j4] + 3 == *(pf.input()));
        j4++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection3.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SPPS)
TEST_CASE("Pipeline.4F(SPPS).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 1);
}

TEST_CASE("Pipeline.4F(SPPS).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 2);
}

TEST_CASE("Pipeline.4F(SPPS).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 3);
}

TEST_CASE("Pipeline.4F(SPPS).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 4);
}

TEST_CASE("Pipeline.4F(SPPS).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 5);
}

TEST_CASE("Pipeline.4F(SPPS).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 6);
}

TEST_CASE("Pipeline.4F(SPPS).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 7);
}

TEST_CASE("Pipeline.4F(SPPS).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(1, 8);
}

TEST_CASE("Pipeline.4F(SPPS).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 1);
}

TEST_CASE("Pipeline.4F(SPPS).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 2);
}

TEST_CASE("Pipeline.4F(SPPS).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 3);
}

TEST_CASE("Pipeline.4F(SPPS).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 4);
}

TEST_CASE("Pipeline.4F(SPPS).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 5);
}

TEST_CASE("Pipeline.4F(SPPS).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 6);
}

TEST_CASE("Pipeline.4F(SPPS).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 7);
}

TEST_CASE("Pipeline.4F(SPPS).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(2, 8);
}

TEST_CASE("Pipeline.4F(SPPS).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 1);
}

TEST_CASE("Pipeline.4F(SPPS).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 2);
}

TEST_CASE("Pipeline.4F(SPPS).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 3);
}

TEST_CASE("Pipeline.4F(SPPS).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 4);
}

TEST_CASE("Pipeline.4F(SPPS).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 5);
}

TEST_CASE("Pipeline.4F(SPPS).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 6);
}

TEST_CASE("Pipeline.4F(SPPS).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 7);
}

TEST_CASE("Pipeline.4F(SPPS).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(3, 8);
}

TEST_CASE("Pipeline.4F(SPPS).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 1);
}

TEST_CASE("Pipeline.4F(SPPS).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 2);
}

TEST_CASE("Pipeline.4F(SPPS).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 3);
}

TEST_CASE("Pipeline.4F(SPPS).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 4);
}

TEST_CASE("Pipeline.4F(SPPS).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 5);
}

TEST_CASE("Pipeline.4F(SPPS).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 6);
}

TEST_CASE("Pipeline.4F(SPPS).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 7);
}

TEST_CASE("Pipeline.4F(SPPS).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(4, 8);
}

TEST_CASE("Pipeline.4F(SPPS).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 1);
}

TEST_CASE("Pipeline.4F(SPPS).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 2);
}

TEST_CASE("Pipeline.4F(SPPS).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 3);
}

TEST_CASE("Pipeline.4F(SPPS).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 4);
}

TEST_CASE("Pipeline.4F(SPPS).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 5);
}

TEST_CASE("Pipeline.4F(SPPS).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 6);
}

TEST_CASE("Pipeline.4F(SPPS).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 7);
}

TEST_CASE("Pipeline.4F(SPPS).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(5, 8);
}

TEST_CASE("Pipeline.4F(SPPS).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 1);
}

TEST_CASE("Pipeline.4F(SPPS).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 2);
}

TEST_CASE("Pipeline.4F(SPPS).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 3);
}

TEST_CASE("Pipeline.4F(SPPS).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 4);
}

TEST_CASE("Pipeline.4F(SPPS).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 5);
}

TEST_CASE("Pipeline.4F(SPPS).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 6);
}

TEST_CASE("Pipeline.4F(SPPS).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 7);
}

TEST_CASE("Pipeline.4F(SPPS).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(6, 8);
}

TEST_CASE("Pipeline.4F(SPPS).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 1);
}

TEST_CASE("Pipeline.4F(SPPS).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 2);
}

TEST_CASE("Pipeline.4F(SPPS).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 3);
}

TEST_CASE("Pipeline.4F(SPPS).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 4);
}

TEST_CASE("Pipeline.4F(SPPS).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 5);
}

TEST_CASE("Pipeline.4F(SPPS).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 6);
}

TEST_CASE("Pipeline.4F(SPPS).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 7);
}

TEST_CASE("Pipeline.4F(SPPS).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(7, 8);
}

TEST_CASE("Pipeline.4F(SPPS).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 1);
}

TEST_CASE("Pipeline.4F(SPPS).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 2);
}

TEST_CASE("Pipeline.4F(SPPS).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 3);
}

TEST_CASE("Pipeline.4F(SPPS).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 4);
}

TEST_CASE("Pipeline.4F(SPPS).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 5);
}

TEST_CASE("Pipeline.4F(SPPS).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 6);
}

TEST_CASE("Pipeline.4F(SPPS).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 7);
}

TEST_CASE("Pipeline.4F(SPPS).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPS(8, 8);
}

// ----------------------------------------------------------------------------
// four pipes (SPPP), L lines, W workers
// ----------------------------------------------------------------------------

void pipeline_4FSPPP(size_t L, unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);
  std::vector<std::array<int, 4>> mybuffer(L);

  for(size_t N = 0; N <= maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::atomic<size_t> j3 = 0;
    std::atomic<size_t> j4 = 0;
    std::mutex mutex2;
    std::mutex mutex3;
    std::mutex mutex4;
    std::vector<int> collection2;
    std::vector<int> collection3;
    std::vector<int> collection4;

    tf::Pipeline pl(L, 
      tf::Pipe{tf::PipeType::SERIAL, [N, &source, &j1, &mybuffer, L](auto& pf) mutable {
        if(j1 == N) {
          pf.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        REQUIRE(pf.token() % L == pf.line());
        mybuffer[pf.line()][pf.pipe()] = source[j1] + 1;
        // *(pf.output()) = source[j1] + 1;
        j1++;
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j2, &mutex2, &collection2, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j2++ < N);
        // *pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex2);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection2.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          //collection2.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j3, &mutex3, &collection3, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j3++ < N);
        // *pf.output() = *pf.input() + 1;
        {
          std::scoped_lock<std::mutex> lock(mutex3);
          REQUIRE(pf.token() % L == pf.line());
          mybuffer[pf.line()][pf.pipe()] = mybuffer[pf.line()][pf.pipe() - 1] + 1;
          collection3.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          // collection3.push_back(*pf.input());
        }
      }},

      tf::Pipe{tf::PipeType::PARALLEL, [N, &source, &j4, &mutex4, &collection4, &mybuffer, L](auto& pf) mutable {
        REQUIRE(j4++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex4);
          REQUIRE(pf.token() % L == pf.line());
          collection4.push_back(mybuffer[pf.line()][pf.pipe() - 1]);
          // collection4.push_back(*pf.input());
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
      REQUIRE(collection4[i] == i + 3);
    }
    REQUIRE(pl.num_tokens() == N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection3.clear();
    collection4.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
      REQUIRE(collection4[i] == i + 3);
    }
    REQUIRE(pl.num_tokens() == 2 * N);
    
    j1 = j2 = j3 = j4 = 0;
    for(size_t i = 0; i < mybuffer.size(); ++i){
      for(size_t j = 0; j < mybuffer[0].size(); ++j){
        mybuffer[i][j] = 0;
      }
    }
    collection2.clear();
    collection3.clear();
    collection4.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    REQUIRE(j3 == N);
    REQUIRE(j4 == N);
    REQUIRE(collection2.size() == N);
    REQUIRE(collection3.size() == N);
    REQUIRE(collection4.size() == N);
    std::sort(collection2.begin(), collection2.end());
    std::sort(collection3.begin(), collection3.end());
    std::sort(collection4.begin(), collection4.end());
    for (size_t i = 0; i < N; ++i) {
      REQUIRE(collection2[i] == i + 1);
      REQUIRE(collection3[i] == i + 2);
      REQUIRE(collection4[i] == i + 3);
    }
    REQUIRE(pl.num_tokens() == 3 * N);
  }
}

// four pipes (SPPP)
TEST_CASE("Pipeline.4F(SPPP).1L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 1);
}

TEST_CASE("Pipeline.4F(SPPP).1L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 2);
}

TEST_CASE("Pipeline.4F(SPPP).1L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 3);
}

TEST_CASE("Pipeline.4F(SPPP).1L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 4);
}

TEST_CASE("Pipeline.4F(SPPP).1L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 5);
}

TEST_CASE("Pipeline.4F(SPPP).1L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 6);
}

TEST_CASE("Pipeline.4F(SPPP).1L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 7);
}

TEST_CASE("Pipeline.4F(SPPP).1L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(1, 8);
}

TEST_CASE("Pipeline.4F(SPPP).2L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 1);
}

TEST_CASE("Pipeline.4F(SPPP).2L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 2);
}

TEST_CASE("Pipeline.4F(SPPP).2L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 3);
}

TEST_CASE("Pipeline.4F(SPPP).2L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 4);
}

TEST_CASE("Pipeline.4F(SPPP).2L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 5);
}

TEST_CASE("Pipeline.4F(SPPP).2L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 6);
}

TEST_CASE("Pipeline.4F(SPPP).2L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 7);
}

TEST_CASE("Pipeline.4F(SPPP).2L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(2, 8);
}

TEST_CASE("Pipeline.4F(SPPP).3L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 1);
}

TEST_CASE("Pipeline.4F(SPPP).3L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 2);
}

TEST_CASE("Pipeline.4F(SPPP).3L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 3);
}

TEST_CASE("Pipeline.4F(SPPP).3L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 4);
}

TEST_CASE("Pipeline.4F(SPPP).3L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 5);
}

TEST_CASE("Pipeline.4F(SPPP).3L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 6);
}

TEST_CASE("Pipeline.4F(SPPP).3L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 7);
}

TEST_CASE("Pipeline.4F(SPPP).3L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(3, 8);
}

TEST_CASE("Pipeline.4F(SPPP).4L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 1);
}

TEST_CASE("Pipeline.4F(SPPP).4L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 2);
}

TEST_CASE("Pipeline.4F(SPPP).4L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 3);
}

TEST_CASE("Pipeline.4F(SPPP).4L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 4);
}

TEST_CASE("Pipeline.4F(SPPP).4L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 5);
}

TEST_CASE("Pipeline.4F(SPPP).4L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 6);
}

TEST_CASE("Pipeline.4F(SPPP).4L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 7);
}

TEST_CASE("Pipeline.4F(SPPP).4L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(4, 8);
}

TEST_CASE("Pipeline.4F(SPPP).5L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 1);
}

TEST_CASE("Pipeline.4F(SPPP).5L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 2);
}

TEST_CASE("Pipeline.4F(SPPP).5L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 3);
}

TEST_CASE("Pipeline.4F(SPPP).5L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 4);
}

TEST_CASE("Pipeline.4F(SPPP).5L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 5);
}

TEST_CASE("Pipeline.4F(SPPP).5L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 6);
}

TEST_CASE("Pipeline.4F(SPPP).5L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 7);
}

TEST_CASE("Pipeline.4F(SPPP).5L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(5, 8);
}

TEST_CASE("Pipeline.4F(SPPP).6L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 1);
}

TEST_CASE("Pipeline.4F(SPPP).6L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 2);
}

TEST_CASE("Pipeline.4F(SPPP).6L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 3);
}

TEST_CASE("Pipeline.4F(SPPP).6L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 4);
}

TEST_CASE("Pipeline.4F(SPPP).6L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 5);
}

TEST_CASE("Pipeline.4F(SPPP).6L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 6);
}

TEST_CASE("Pipeline.4F(SPPP).6L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 7);
}

TEST_CASE("Pipeline.4F(SPPP).6L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(6, 8);
}

TEST_CASE("Pipeline.4F(SPPP).7L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 1);
}

TEST_CASE("Pipeline.4F(SPPP).7L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 2);
}

TEST_CASE("Pipeline.4F(SPPP).7L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 3);
}

TEST_CASE("Pipeline.4F(SPPP).7L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 4);
}

TEST_CASE("Pipeline.4F(SPPP).7L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 5);
}

TEST_CASE("Pipeline.4F(SPPP).7L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 6);
}

TEST_CASE("Pipeline.4F(SPPP).7L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 7);
}

TEST_CASE("Pipeline.4F(SPPP).7L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(7, 8);
}

TEST_CASE("Pipeline.4F(SPPP).8L.1W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 1);
}

TEST_CASE("Pipeline.4F(SPPP).8L.2W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 2);
}

TEST_CASE("Pipeline.4F(SPPP).8L.3W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 3);
}

TEST_CASE("Pipeline.4F(SPPP).8L.4W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 4);
}

TEST_CASE("Pipeline.4F(SPPP).8L.5W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 5);
}

TEST_CASE("Pipeline.4F(SPPP).8L.6W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 6);
}

TEST_CASE("Pipeline.4F(SPPP).8L.7W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 7);
}

TEST_CASE("Pipeline.4F(SPPP).8L.8W" * doctest::timeout(300)) {
  pipeline_4FSPPP(8, 8);
}  
*/
