#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <taskflow/pipeline.hpp>

// --------------------------------------------------------
// Testcase: 1 filter, L lines, w workers
// --------------------------------------------------------
template <size_t L>
void pipeline_1F(unsigned w, tf::FilterType type) {

  tf::Executor executor(w);
    
  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  // iterate different data amount (1, 2, 3, 4, 5, ... 1000000)
  for (size_t N=0; N<=maxN; N++) {
    
    // serial direction
    if(type == tf::FilterType::SERIAL) {
      tf::Taskflow taskflow;
      size_t j = 0;
      auto pl = tf::make_pipeline<int, L>(tf::Filter{type, [N, &j, &source](auto& df) mutable {
        if(j==N) {
          df.stop();
          return;
        }
        REQUIRE(j == source[j]);
        j++;
      }});
      taskflow.pipeline(pl);
      executor.run(taskflow).wait();
      REQUIRE(j == N);

      j = 0;
      executor.run(taskflow).wait();
      REQUIRE(j == N);

      j = 0;
      auto fu = executor.run(taskflow);
      fu.cancel();
      fu.get();

      j = 0;
      executor.run(taskflow).wait();
      REQUIRE(j == N);
    }
    // parallel filter
    else if(type == tf::FilterType::PARALLEL) {
      
      tf::Taskflow taskflow;

      std::atomic<size_t> j = 0;
      std::mutex mutex;
      std::vector<int> collection;

      auto pl = tf::make_pipeline<int, L>(tf::Filter{type, 
      [N, &j, &mutex, &collection](auto& df) mutable {

        auto ticket = j.fetch_add(1);

        if(ticket >= N) {
          df.stop();
          return;
        }
        std::scoped_lock<std::mutex> lock(mutex);
        collection.push_back(ticket);
      }});

      taskflow.pipeline(pl);
      executor.run(taskflow).wait();
      REQUIRE(collection.size() == N);
      std::sort(collection.begin(), collection.end());
      for(size_t k=0; k<N; k++) {
        REQUIRE(collection[k] == k);
      }

      j = 0;
      collection.clear();
      executor.run(taskflow).wait();
      REQUIRE(collection.size() == N);
      std::sort(collection.begin(), collection.end());
      for(size_t k=0; k<N; k++) {
        REQUIRE(collection[k] == k);
      }
    }
  }
}

// ---- serial filter ----

// serial filter with one line
TEST_CASE("Pipeline.1F(S).1L.1W" * doctest::timeout(300)) {
  pipeline_1F<1>(1, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.2W" * doctest::timeout(300)) {
  pipeline_1F<1>(2, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.3W" * doctest::timeout(300)) {
  pipeline_1F<1>(3, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).1L.4W" * doctest::timeout(300)) {
  pipeline_1F<1>(4, tf::FilterType::SERIAL);
}

// serial filter with two lines
TEST_CASE("Pipeline.1F(S).2L.1W" * doctest::timeout(300)) {
  pipeline_1F<2>(1, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.2W" * doctest::timeout(300)) {
  pipeline_1F<2>(2, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.3W" * doctest::timeout(300)) {
  pipeline_1F<2>(3, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).2L.4W" * doctest::timeout(300)) {
  pipeline_1F<2>(4, tf::FilterType::SERIAL);
}

// serial filter with three lines
TEST_CASE("Pipeline.1F(S).3L.1W" * doctest::timeout(300)) {
  pipeline_1F<3>(1, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.2W" * doctest::timeout(300)) {
  pipeline_1F<3>(2, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.3W" * doctest::timeout(300)) {
  pipeline_1F<3>(3, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).3L.4W" * doctest::timeout(300)) {
  pipeline_1F<3>(4, tf::FilterType::SERIAL);
}

// serial filter with three lines
TEST_CASE("Pipeline.1F(S).4L.1W" * doctest::timeout(300)) {
  pipeline_1F<4>(1, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.2W" * doctest::timeout(300)) {
  pipeline_1F<4>(2, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.3W" * doctest::timeout(300)) {
  pipeline_1F<4>(3, tf::FilterType::SERIAL);
}

TEST_CASE("Pipeline.1F(S).4L.4W" * doctest::timeout(300)) {
  pipeline_1F<4>(4, tf::FilterType::SERIAL);
}


// ---- parallel filter ----

// parallel filter with one line
TEST_CASE("Pipeline.1F(P).1L.1W" * doctest::timeout(300)) {
  pipeline_1F<1>(1, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).1L.2W" * doctest::timeout(300)) {
  pipeline_1F<1>(2, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).1L.3W" * doctest::timeout(300)) {
  pipeline_1F<1>(3, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).1L.4W" * doctest::timeout(300)) {
  pipeline_1F<1>(4, tf::FilterType::PARALLEL);
}

// parallel filter with two lines
TEST_CASE("Pipeline.1F(P).2L.1W" * doctest::timeout(300)) {
  pipeline_1F<2>(1, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).2L.2W" * doctest::timeout(300)) {
  pipeline_1F<2>(2, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).2L.3W" * doctest::timeout(300)) {
  pipeline_1F<2>(3, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).2L.4W" * doctest::timeout(300)) {
  pipeline_1F<2>(4, tf::FilterType::PARALLEL);
}

// parallel filter with three lines
TEST_CASE("Pipeline.1F(P).3L.1W" * doctest::timeout(300)) {
  pipeline_1F<3>(1, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).3L.2W" * doctest::timeout(300)) {
  pipeline_1F<3>(2, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).3L.3W" * doctest::timeout(300)) {
  pipeline_1F<3>(3, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).3L.4W" * doctest::timeout(300)) {
  pipeline_1F<3>(4, tf::FilterType::PARALLEL);
}

// parallel filter with four lines
TEST_CASE("Pipeline.1F(P).4L.1W" * doctest::timeout(300)) {
  pipeline_1F<4>(1, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).4L.2W" * doctest::timeout(300)) {
  pipeline_1F<4>(2, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).4L.3W" * doctest::timeout(300)) {
  pipeline_1F<4>(3, tf::FilterType::PARALLEL);
}

TEST_CASE("Pipeline.1F(P).4L.4W" * doctest::timeout(300)) {
  pipeline_1F<4>(4, tf::FilterType::PARALLEL);
}

// ----------------------------------------------------------------------------
// two filters (SS), L lines, W workers
// ----------------------------------------------------------------------------

template <size_t L>
void pipeline_2FSS(unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N=0; N<=maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0, j2 = 0;

    auto pl = tf::make_pipeline<int, L>(
      tf::Filter{tf::FilterType::SERIAL, [N, &source, &j1](auto& df) mutable {
        if(j1 == N) {
          df.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        *(df.output()) = source[j1] + 1;
        j1++;
      }},
      tf::Filter{tf::FilterType::SERIAL, [N, &source, &j2](auto& df) mutable {
        REQUIRE(j2 < N);
        REQUIRE(source[j2] + 1 == *(df.input()));
        j2++;
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    
    j1 = j2 = 0;
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    
    j1 = j2 = 0;
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
  }
}

// two filters (SS)
TEST_CASE("Pipeline.2F(SS).1L.1W" * doctest::timeout(300)) {
  pipeline_2FSS<1>(1);
}

TEST_CASE("Pipeline.2F(SS).1L.2W" * doctest::timeout(300)) {
  pipeline_2FSS<1>(2);
}

TEST_CASE("Pipeline.2F(SS).1L.3W" * doctest::timeout(300)) {
  pipeline_2FSS<1>(3);
}

TEST_CASE("Pipeline.2F(SS).1L.4W" * doctest::timeout(300)) {
  pipeline_2FSS<1>(4);
}

TEST_CASE("Pipeline.2F(SS).2L.1W" * doctest::timeout(300)) {
  pipeline_2FSS<2>(1);
}

TEST_CASE("Pipeline.2F(SS).2L.2W" * doctest::timeout(300)) {
  pipeline_2FSS<2>(2);
}

TEST_CASE("Pipeline.2F(SS).2L.3W" * doctest::timeout(300)) {
  pipeline_2FSS<2>(3);
}

TEST_CASE("Pipeline.2F(SS).2L.4W" * doctest::timeout(300)) {
  pipeline_2FSS<2>(4);
}

TEST_CASE("Pipeline.2F(SS).3L.1W" * doctest::timeout(300)) {
  pipeline_2FSS<3>(1);
}

TEST_CASE("Pipeline.2F(SS).3L.2W" * doctest::timeout(300)) {
  pipeline_2FSS<3>(2);
}

TEST_CASE("Pipeline.2F(SS).3L.3W" * doctest::timeout(300)) {
  pipeline_2FSS<3>(3);
}

TEST_CASE("Pipeline.2F(SS).3L.4W" * doctest::timeout(300)) {
  pipeline_2FSS<3>(4);
}

TEST_CASE("Pipeline.2F(SS).4L.1W" * doctest::timeout(300)) {
  pipeline_2FSS<4>(1);
}

TEST_CASE("Pipeline.2F(SS).4L.2W" * doctest::timeout(300)) {
  pipeline_2FSS<4>(2);
}

TEST_CASE("Pipeline.2F(SS).4L.3W" * doctest::timeout(300)) {
  pipeline_2FSS<4>(3);
}

TEST_CASE("Pipeline.2F(SS).4L.4W" * doctest::timeout(300)) {
  pipeline_2FSS<4>(4);
}

// ----------------------------------------------------------------------------
// two filters (SP), L lines, W workers
// ----------------------------------------------------------------------------

template <size_t L>
void pipeline_2FSP(unsigned w) {

  tf::Executor executor(w);

  const size_t maxN = 100;

  std::vector<int> source(maxN);
  std::iota(source.begin(), source.end(), 0);

  for(size_t N=0; N<=maxN; N++) {

    tf::Taskflow taskflow;
      
    size_t j1 = 0;
    std::atomic<size_t> j2 = 0;
    std::mutex mutex;
    std::vector<int> collection;

    auto pl = tf::make_pipeline<int, L>(
      tf::Filter{tf::FilterType::SERIAL, [N, &source, &j1](auto& df) mutable {
        if(j1 == N) {
          df.stop();
          return;
        }
        REQUIRE(j1 == source[j1]);
        *(df.output()) = source[j1] + 1;
        j1++;
      }},
      tf::Filter{tf::FilterType::PARALLEL, 
      [N, &collection, &source, &mutex, &j2](auto& df) mutable {
        REQUIRE(j2++ < N);
        {
          std::scoped_lock<std::mutex> lock(mutex);
          collection.push_back(*df.input());
        }
      }}
    );
    
    taskflow.pipeline(pl);
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    std::sort(collection.begin(), collection.end());
    for(size_t i=0; i<N; i++) {
      REQUIRE(collection[i] == i+1);
    }
    
    j1 = j2 = 0;
    collection.clear();
    executor.run(taskflow).wait();
    REQUIRE(j1 == N);
    REQUIRE(j2 == N);
    std::sort(collection.begin(), collection.end());
    for(size_t i=0; i<N; i++) {
      REQUIRE(collection[i] == i+1);
    }
  }
}

// two filters (SP)
TEST_CASE("Pipeline.2F(SP).1L.1W" * doctest::timeout(300)) {
  pipeline_2FSP<1>(1);
}

TEST_CASE("Pipeline.2F(SP).1L.2W" * doctest::timeout(300)) {
  pipeline_2FSP<1>(2);
}

TEST_CASE("Pipeline.2F(SP).1L.3W" * doctest::timeout(300)) {
  pipeline_2FSP<1>(3);
}

TEST_CASE("Pipeline.2F(SP).1L.4W" * doctest::timeout(300)) {
  pipeline_2FSP<1>(4);
}

TEST_CASE("Pipeline.2F(SP).2L.1W" * doctest::timeout(300)) {
  pipeline_2FSP<2>(1);
}

TEST_CASE("Pipeline.2F(SP).2L.2W" * doctest::timeout(300)) {
  pipeline_2FSP<2>(2);
}

TEST_CASE("Pipeline.2F(SP).2L.3W" * doctest::timeout(300)) {
  pipeline_2FSP<2>(3);
}

TEST_CASE("Pipeline.2F(SP).2L.4W" * doctest::timeout(300)) {
  pipeline_2FSP<2>(4);
}

TEST_CASE("Pipeline.2F(SP).3L.1W" * doctest::timeout(300)) {
  pipeline_2FSP<3>(1);
}

TEST_CASE("Pipeline.2F(SP).3L.2W" * doctest::timeout(300)) {
  pipeline_2FSP<3>(2);
}

TEST_CASE("Pipeline.2F(SP).3L.3W" * doctest::timeout(300)) {
  pipeline_2FSP<3>(3);
}

TEST_CASE("Pipeline.2F(SP).3L.4W" * doctest::timeout(300)) {
  pipeline_2FSP<3>(4);
}

TEST_CASE("Pipeline.2F(SP).4L.1W" * doctest::timeout(300)) {
  pipeline_2FSP<4>(1);
}

TEST_CASE("Pipeline.2F(SP).4L.2W" * doctest::timeout(300)) {
  pipeline_2FSP<4>(2);
}

TEST_CASE("Pipeline.2F(SP).4L.3W" * doctest::timeout(300)) {
  pipeline_2FSP<4>(3);
}

TEST_CASE("Pipeline.2F(SP).4L.4W" * doctest::timeout(300)) {
  pipeline_2FSP<4>(4);
}




















//
//TEST_CASE("Pipeline.1F.1L.1W.double" * doctest::timeout(300)) {
//  pipeline_1F_1L<double>(1, mode::SERIAL);
//  pipeline_1F_1L<double>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F.1L.1W.float" * doctest::timeout(300)) {
//  pipeline_1F_1L<float>(1, mode::SERIAL);
//  pipeline_1F_1L<float>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F.1L.1W.long" * doctest::timeout(300)) {
//  pipeline_1F_1L<long>(1, mode::SERIAL);
//  pipeline_1F_1L<long>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.1F.1L.1W.string" * doctest::timeout(300)) {
//  pipeline_1F_1L<std::string>(1, mode::SERIAL);
//  pipeline_1F_1L<std::string>(1, mode::PARALLEL);
//}



// --------------------------------------------------------
// Testcase: 2 filters, 1 line, w workers, different datatype  
// --------------------------------------------------------

//template <typename D>
//void pipeline_2F_1L(unsigned w, mode m) {
//  tf::Taskflow taskflow;
//  tf::Executor executor(w);
//  
//  // iterate different data amount (1, 2, 3, 4, 5, ... 1000000)
//  for (size_t N = 0; N < 10000; N++) {
//    std::vector<D> source(N);
//
//    if (static_cast<int>(m)) {
//      tf::make_pipeline<D, 1>(
//        tf::Filter{tf::FilterType::SERIAL,
//          [](auto& d){
//            REQUIRE(d.input() == nullptr);
//            REQUIRE(d.output() == nullptr);
//          }
//        }
//      );
//    }
//    else {
//      tf::make_pipeline<D, 1>(
//        tf::Filter{tf::FilterType::PARALLEL,
//          [](auto& d){
//            REQUIRE(d.input() == nullptr);
//            REQUIRE(d.output() == nullptr);
//          }
//        }
//      );
//    }
//  }
//  executor.run(taskflow).wait();
//}
//
//TEST_CASE("Pipeline.2F.1L.1W.int" * doctest::timeout(300)) {
//  pipeline_2F_1L<int>(1, mode::SERIAL);
//  pipeline_2F_1L<int>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.2F.1L.1W.double" * doctest::timeout(300)) {
//  pipeline_2F_1L<double>(1, mode::SERIAL);
//  pipeline_2F_1L<double>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.2F.1L.1W.float" * doctest::timeout(300)) {
//  pipeline_2F_1L<float>(1, mode::SERIAL);
//  pipeline_2F_1L<float>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.2F.1L.1W.long" * doctest::timeout(300)) {
//  pipeline_2F_1L<long>(1, mode::SERIAL);
//  pipeline_2F_1L<long>(1, mode::PARALLEL);
//}
//
//TEST_CASE("Pipeline.2F.1L.1W.string" * doctest::timeout(300)) {
//  pipeline_2F_1L<std::string>(1, mode::SERIAL);
//  pipeline_2F_1L<std::string>(1, mode::PARALLEL);
//}
//
//
//
//

// --------------------------------------------------------
// Testcase: 3 filters, 1 line, w workers, different datatype  
// --------------------------------------------------------

//template<typename output1, typename output2>
//void pipeline_3F_1L(unsigned w) {
//  tf::Taskflow taskflow;
//  tf::Executor executor(w);
// 
//  int cnt = 0; 
//  std::vector<int> source(N);
//  using data_type = std::variant<int, float, double, std::string>;
//
//  for (size_t i = 0; i < N; i++) {
//    source[i] = N-1-i;
//  }
//  
//  auto p1 = tf::make_pipeline<data_type, 1>(
//    // f1
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df) mutable {
//        if (source.empty()) {
//          df.stop();
//        }
//        else {
//          df.at_output() = static_cast<output1>(source.back());
//          std::cout << "f1 : output = " << std::get<output1>(df.at_output()) << std::endl;
//          source.pop_back();
//        }
//      }
//    },
//
//    // f2: output1 type now is f2's input type
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df){
//        std::cout << "f2 : input = " << std::get<output1>(df.at_input()) << std::endl;
//        REQUIRE(std::get<output1>(df.at_input()) == cnt);
//        df.at_output() = std::get<output1>(df.at_input());
//      }
//    },
//
//    // f3: output2 type now is f3's input type
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df){
//        std::cout << "f3 : input = " << std::get<output2>(df.at_input()) << std::endl;
//        REQUIRE(std::get<output2>(df.at_input()) == cnt++);
//        //REQUIRE(df.output() != nullptr);  
//      }
//    }
//  );
//  taskflow.pipeline(p1);
//  executor.run(taskflow).wait();
//}


//TEST_CASE("Pipeline.3F.1L.1W.sss.int.int" * doctest::timeout(300)) {
//  pipeline_3F_1L<int, int>(1);
//}


// --------------------------------------------------------
// Testcase: 3 filters, 2 lines, w workers, different datatype  
// --------------------------------------------------------

//template<typename output1, typename output2>
//void pipeline_3F_2L(unsigned w) {
//  tf::Taskflow taskflow;
//  tf::Executor executor(w);
//  
//  std::vector<int> source(N);
//  using data_type = std::variant<int, float, double, std::string>;
//
//  for (size_t i = 0; i < N; i++) {
//    source[i] = N-1-i;
//  }
//
//  int cnt = 0;
//
//  auto pl = tf::make_pipeline<data_type, 2>(
//    // f1
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df){
//        if (source.empty()) {
//          df.stop();
//        }
//        else {
//          df.at_output() = static_cast<output1>(source.back());
//          source.pop_back();
//        }
//      }
//    },
//
//    // f2: output1 type now is f2's input type
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df){
//        REQUIRE(std::get<output1>(df.at_input()) == cnt);
//        df.at_output() = std::get<output1>(df.at_input());
//      }
//    },
//
//    // f3: output2 type now is f3's input type
//    tf::Filter{tf::FilterType::SERIAL,
//      [&](auto& df){
//        REQUIRE(std::get<output2>(df.at_input()) == cnt++);
//        //REQUIRE(df.output() != nullptr);  
//      }
//    }
//  );
//
//  taskflow.pipeline(pl);
//  executor.run(taskflow).wait();
//}
//
//
//TEST_CASE("Pipeline.3F.2L.1W.sss.int.int" * doctest::timeout(300)) {
//  pipeline_3F_2L<int, int>(1);
//}
