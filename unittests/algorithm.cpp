#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/taskflow.hpp>
#include <vector>
#include <utility>
#include <chrono>
#include <limits.h>

// --------------------------------------------------------
// Testcase: parallel_for
// --------------------------------------------------------

enum TYPE {
  GUIDED,
  DYNAMIC,
  STATIC,
  FACTORING
};

void parallel_for(unsigned W, TYPE type) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;
  
  std::vector<int> vec(1024);
  for(int n = 0; n <= 150; n++) {

    std::fill_n(vec.begin(), vec.size(), -1);

    int beg = ::rand()%300 - 150;
    int end = beg + n;

    for(int s=1; s<=16; s*=2) {
      for(int c=0; c<=17; c=c*2+1) {
        taskflow.clear();
        std::atomic<int> counter {0};
        
        switch(type) {
          case GUIDED:
            taskflow.parallel_for_guided(beg, end, s, [&](int i){
              counter++;
              vec[i-beg] = i;
            }, c);
          break;

          case DYNAMIC:
            taskflow.parallel_for_dynamic(beg, end, s, [&](int i){
              counter++;
              vec[i-beg] = i;
            }, c);
          break;
          
          case STATIC:
            taskflow.parallel_for_static(beg, end, s, [&](int i){
              counter++;
              vec[i-beg] = i;
            }, c);
          break;
          
          case FACTORING:
            taskflow.parallel_for_factoring(beg, end, s, [&](int i){
              counter++;
              vec[i-beg] = i;
            });
          break;
        }

        executor.run(taskflow).wait();
        REQUIRE(counter == (n + s - 1) / s);

        for(int i=beg; i<end; i+=s) {
          REQUIRE(vec[i-beg] == i);
          vec[i-beg] = -1;
        }

        for(const auto i : vec) {
          REQUIRE(i == -1);
        }
      }
    }
  }

  for(size_t n = 0; n < 150; n++) {
    for(size_t c=0; c<=17; c=c*2+1) {
    
      std::fill_n(vec.begin(), vec.size(), -1);

      taskflow.clear();
      std::atomic<int> counter {0};
      
      switch(type) {
        case GUIDED:
          taskflow.parallel_for_guided(vec.begin(), vec.begin() + n, [&](int& i){
            counter++;
            i = 1;
          }, c);
        break;

        case DYNAMIC:
          taskflow.parallel_for_dynamic(vec.begin(), vec.begin() + n, [&](int& i){
            counter++;
            i = 1;
          }, c);
        break;
        
        case STATIC:
          taskflow.parallel_for_static(vec.begin(), vec.begin() + n, [&](int& i){
            counter++;
            i = 1;
          }, c);
        break;
        
        case FACTORING:
          taskflow.parallel_for_factoring(vec.begin(), vec.begin() + n, [&](int& i){
            counter++;
            i = 1;
          });
        break;
      }

      executor.run(taskflow).wait();
      REQUIRE(counter == n);

      for(size_t i=0; i<n; ++i) {
        REQUIRE(vec[i] == 1);
      }

      for(size_t i=n; i<vec.size(); ++i) {
        REQUIRE(vec[i] == -1);
      }
    }
  }
}

// guided
TEST_CASE("pfg.1thread" * doctest::timeout(300)) {
  parallel_for(1, GUIDED);
}

TEST_CASE("pfg.2threads" * doctest::timeout(300)) {
  parallel_for(2, GUIDED);
}

TEST_CASE("pfg.3threads" * doctest::timeout(300)) {
  parallel_for(3, GUIDED);
}

TEST_CASE("pfg.4threads" * doctest::timeout(300)) {
  parallel_for(4, GUIDED);
}

TEST_CASE("pfg.5threads" * doctest::timeout(300)) {
  parallel_for(5, GUIDED);
}

TEST_CASE("pfg.6threads" * doctest::timeout(300)) {
  parallel_for(6, GUIDED);
}

TEST_CASE("pfg.7threads" * doctest::timeout(300)) {
  parallel_for(7, GUIDED);
}

TEST_CASE("pfg.8threads" * doctest::timeout(300)) {
  parallel_for(8, GUIDED);
}

TEST_CASE("pfg.9threads" * doctest::timeout(300)) {
  parallel_for(9, GUIDED);
}

TEST_CASE("pfg.10threads" * doctest::timeout(300)) {
  parallel_for(10, GUIDED);
}

TEST_CASE("pfg.11threads" * doctest::timeout(300)) {
  parallel_for(11, GUIDED);
}

TEST_CASE("pfg.12threads" * doctest::timeout(300)) {
  parallel_for(12, GUIDED);
}

// dynamic
TEST_CASE("pfd.1thread" * doctest::timeout(300)) {
  parallel_for(1, DYNAMIC);
}

TEST_CASE("pfd.2threads" * doctest::timeout(300)) {
  parallel_for(2, DYNAMIC);
}

TEST_CASE("pfd.3threads" * doctest::timeout(300)) {
  parallel_for(3, DYNAMIC);
}

TEST_CASE("pfd.4threads" * doctest::timeout(300)) {
  parallel_for(4, DYNAMIC);
}

TEST_CASE("pfd.5threads" * doctest::timeout(300)) {
  parallel_for(5, DYNAMIC);
}

TEST_CASE("pfd.6threads" * doctest::timeout(300)) {
  parallel_for(6, DYNAMIC);
}

TEST_CASE("pfd.7threads" * doctest::timeout(300)) {
  parallel_for(7, DYNAMIC);
}

TEST_CASE("pfd.8threads" * doctest::timeout(300)) {
  parallel_for(8, DYNAMIC);
}

TEST_CASE("pfd.9threads" * doctest::timeout(300)) {
  parallel_for(9, DYNAMIC);
}

TEST_CASE("pfd.10threads" * doctest::timeout(300)) {
  parallel_for(10, DYNAMIC);
}

TEST_CASE("pfd.11threads" * doctest::timeout(300)) {
  parallel_for(11, DYNAMIC);
}

TEST_CASE("pfd.12threads" * doctest::timeout(300)) {
  parallel_for(12, DYNAMIC);
}

// static
TEST_CASE("pfs.1thread" * doctest::timeout(300)) {
  parallel_for(1, STATIC);
}

TEST_CASE("pfs.2threads" * doctest::timeout(300)) {
  parallel_for(2, STATIC);
}

TEST_CASE("pfs.3threads" * doctest::timeout(300)) {
  parallel_for(3, STATIC);
}

TEST_CASE("pfs.4threads" * doctest::timeout(300)) {
  parallel_for(4, STATIC);
}

TEST_CASE("pfs.5threads" * doctest::timeout(300)) {
  parallel_for(5, STATIC);
}

TEST_CASE("pfs.6threads" * doctest::timeout(300)) {
  parallel_for(6, STATIC);
}

TEST_CASE("pfs.7threads" * doctest::timeout(300)) {
  parallel_for(7, STATIC);
}

TEST_CASE("pfs.8threads" * doctest::timeout(300)) {
  parallel_for(8, STATIC);
}

TEST_CASE("pfs.9threads" * doctest::timeout(300)) {
  parallel_for(9, STATIC);
}

TEST_CASE("pfs.10threads" * doctest::timeout(300)) {
  parallel_for(10, STATIC);
}

TEST_CASE("pfs.11threads" * doctest::timeout(300)) {
  parallel_for(11, STATIC);
}

TEST_CASE("pfs.12threads" * doctest::timeout(300)) {
  parallel_for(12, STATIC);
}

// factoring
TEST_CASE("pff.1thread" * doctest::timeout(300)) {
  parallel_for(1, FACTORING);
}

TEST_CASE("pff.2threads" * doctest::timeout(300)) {
  parallel_for(2, FACTORING);
}

TEST_CASE("pff.3threads" * doctest::timeout(300)) {
  parallel_for(3, FACTORING);
}

TEST_CASE("pff.4threads" * doctest::timeout(300)) {
  parallel_for(4, FACTORING);
}

TEST_CASE("pff.5threads" * doctest::timeout(300)) {
  parallel_for(5, FACTORING);
}

TEST_CASE("pff.6threads" * doctest::timeout(300)) {
  parallel_for(6, FACTORING);
}

TEST_CASE("pff.7threads" * doctest::timeout(300)) {
  parallel_for(7, FACTORING);
}

TEST_CASE("pff.8threads" * doctest::timeout(300)) {
  parallel_for(8, FACTORING);
}

TEST_CASE("pff.9threads" * doctest::timeout(300)) {
  parallel_for(9, FACTORING);
}

TEST_CASE("pff.10threads" * doctest::timeout(300)) {
  parallel_for(10, FACTORING);
}

TEST_CASE("pff.11threads" * doctest::timeout(300)) {
  parallel_for(11, FACTORING);
}

TEST_CASE("pff.12threads" * doctest::timeout(300)) {
  parallel_for(12, FACTORING);
}

// ----------------------------------------------------------------------------
// stateful_parallel_for
// ----------------------------------------------------------------------------

void stateful_parallel_for(unsigned W, TYPE type) {
  
  tf::Executor executor(W);
  tf::Taskflow taskflow;
  std::vector<int> vec;
  std::atomic<int> counter {0};
  
  for(size_t n = 0; n <= 150; n++) {
    for(size_t c=0; c<=17; c++) {
  
    std::vector<int>::iterator beg, end;
    size_t ibeg = 0, iend = 0;
    size_t half = n/2;
    
    taskflow.clear();
    
    auto init = taskflow.emplace([&](){ 
      vec.resize(n);
      std::fill_n(vec.begin(), vec.size(), -1);

      beg = vec.begin();
      end = beg + half;

      ibeg = half;
      iend = n;

      counter = 0;
    });

    tf::Task pf1, pf2;
    
    switch (type) {

      case GUIDED:
        pf1 = taskflow.parallel_for_guided(
          std::ref(beg), std::ref(end), [&](int& i){
          counter++;
          i = 8;
        }, c);

        pf2 = taskflow.parallel_for_guided(
          std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
            counter++;
            vec[i] = -8;
        }, c);
      break;

      case DYNAMIC:
        pf1 = taskflow.parallel_for_dynamic(
          std::ref(beg), std::ref(end), [&](int& i){
          counter++;
          i = 8;
        }, c);

        pf2 = taskflow.parallel_for_dynamic(
          std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
            counter++;
            vec[i] = -8;
        }, c);
      break;
      
      case STATIC:
        pf1 = taskflow.parallel_for_static(
          std::ref(beg), std::ref(end), [&](int& i){
          counter++;
          i = 8;
        }, c);

        pf2 = taskflow.parallel_for_static(
          std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
            counter++;
            vec[i] = -8;
        }, c);
      break;
      
      case FACTORING:
        pf1 = taskflow.parallel_for_factoring(
          std::ref(beg), std::ref(end), [&](int& i){
          counter++;
          i = 8;
        });

        pf2 = taskflow.parallel_for_factoring(
          std::ref(ibeg), std::ref(iend), size_t{1}, [&] (size_t i) {
            counter++;
            vec[i] = -8;
        });
      break;
    }

    init.precede(pf1, pf2);

    executor.run(taskflow).wait();
    REQUIRE(counter == n);

    for(size_t i=0; i<half; ++i) {
      REQUIRE(vec[i] == 8);
      vec[i] = 0;
    }

    for(size_t i=half; i<n; ++i) {
      REQUIRE(vec[i] == -8);
      vec[i] = 0;
    }
    }
  }
}

// guided
TEST_CASE("statefulpfg.1thread" * doctest::timeout(300)) {
  stateful_parallel_for(1, GUIDED);
}

TEST_CASE("statefulpfg.2threads" * doctest::timeout(300)) {
  stateful_parallel_for(2, GUIDED);
}

TEST_CASE("statefulpfg.3threads" * doctest::timeout(300)) {
  stateful_parallel_for(3, GUIDED);
}

TEST_CASE("statefulpfg.4threads" * doctest::timeout(300)) {
  stateful_parallel_for(4, GUIDED);
}

TEST_CASE("statefulpfg.5threads" * doctest::timeout(300)) {
  stateful_parallel_for(5, GUIDED);
}

TEST_CASE("statefulpfg.6threads" * doctest::timeout(300)) {
  stateful_parallel_for(6, GUIDED);
}

TEST_CASE("statefulpfg.7threads" * doctest::timeout(300)) {
  stateful_parallel_for(7, GUIDED);
}

TEST_CASE("statefulpfg.8threads" * doctest::timeout(300)) {
  stateful_parallel_for(8, GUIDED);
}

TEST_CASE("statefulpfg.9threads" * doctest::timeout(300)) {
  stateful_parallel_for(9, GUIDED);
}

TEST_CASE("statefulpfg.10threads" * doctest::timeout(300)) {
  stateful_parallel_for(10, GUIDED);
}

TEST_CASE("statefulpfg.11threads" * doctest::timeout(300)) {
  stateful_parallel_for(11, GUIDED);
}

TEST_CASE("statefulpfg.12threads" * doctest::timeout(300)) {
  stateful_parallel_for(12, GUIDED);
}

// dynamic
TEST_CASE("statefulpfd.1thread" * doctest::timeout(300)) {
  stateful_parallel_for(1, DYNAMIC);
}

TEST_CASE("statefulpfd.2threads" * doctest::timeout(300)) {
  stateful_parallel_for(2, DYNAMIC);
}

TEST_CASE("statefulpfd.3threads" * doctest::timeout(300)) {
  stateful_parallel_for(3, DYNAMIC);
}

TEST_CASE("statefulpfd.4threads" * doctest::timeout(300)) {
  stateful_parallel_for(4, DYNAMIC);
}

TEST_CASE("statefulpfd.5threads" * doctest::timeout(300)) {
  stateful_parallel_for(5, DYNAMIC);
}

TEST_CASE("statefulpfd.6threads" * doctest::timeout(300)) {
  stateful_parallel_for(6, DYNAMIC);
}

TEST_CASE("statefulpfd.7threads" * doctest::timeout(300)) {
  stateful_parallel_for(7, DYNAMIC);
}

TEST_CASE("statefulpfd.8threads" * doctest::timeout(300)) {
  stateful_parallel_for(8, DYNAMIC);
}

TEST_CASE("statefulpfd.9threads" * doctest::timeout(300)) {
  stateful_parallel_for(9, DYNAMIC);
}

TEST_CASE("statefulpfd.10threads" * doctest::timeout(300)) {
  stateful_parallel_for(10, DYNAMIC);
}

TEST_CASE("statefulpfd.11threads" * doctest::timeout(300)) {
  stateful_parallel_for(11, DYNAMIC);
}

TEST_CASE("statefulpfd.12threads" * doctest::timeout(300)) {
  stateful_parallel_for(12, DYNAMIC);
}

// static
TEST_CASE("statefulpfs.1thread" * doctest::timeout(300)) {
  stateful_parallel_for(1, STATIC);
}

TEST_CASE("statefulpfs.2threads" * doctest::timeout(300)) {
  stateful_parallel_for(2, STATIC);
}

TEST_CASE("statefulpfs.3threads" * doctest::timeout(300)) {
  stateful_parallel_for(3, STATIC);
}

TEST_CASE("statefulpfs.4threads" * doctest::timeout(300)) {
  stateful_parallel_for(4, STATIC);
}

TEST_CASE("statefulpfs.5threads" * doctest::timeout(300)) {
  stateful_parallel_for(5, STATIC);
}

TEST_CASE("statefulpfs.6threads" * doctest::timeout(300)) {
  stateful_parallel_for(6, STATIC);
}

TEST_CASE("statefulpfs.7threads" * doctest::timeout(300)) {
  stateful_parallel_for(7, STATIC);
}

TEST_CASE("statefulpfs.8threads" * doctest::timeout(300)) {
  stateful_parallel_for(8, STATIC);
}

TEST_CASE("statefulpfs.9threads" * doctest::timeout(300)) {
  stateful_parallel_for(9, STATIC);
}

TEST_CASE("statefulpfs.10threads" * doctest::timeout(300)) {
  stateful_parallel_for(10, STATIC);
}

TEST_CASE("statefulpfs.11threads" * doctest::timeout(300)) {
  stateful_parallel_for(11, STATIC);
}

TEST_CASE("statefulpfs.12threads" * doctest::timeout(300)) {
  stateful_parallel_for(12, STATIC);
}

// factoring
TEST_CASE("statefulpff.1thread" * doctest::timeout(300)) {
  stateful_parallel_for(1, FACTORING);
}

TEST_CASE("statefulpff.2threads" * doctest::timeout(300)) {
  stateful_parallel_for(2, FACTORING);
}

TEST_CASE("statefulpff.3threads" * doctest::timeout(300)) {
  stateful_parallel_for(3, FACTORING);
}

TEST_CASE("statefulpff.4threads" * doctest::timeout(300)) {
  stateful_parallel_for(4, FACTORING);
}

TEST_CASE("statefulpff.5threads" * doctest::timeout(300)) {
  stateful_parallel_for(5, FACTORING);
}

TEST_CASE("statefulpff.6threads" * doctest::timeout(300)) {
  stateful_parallel_for(6, FACTORING);
}

TEST_CASE("statefulpff.7threads" * doctest::timeout(300)) {
  stateful_parallel_for(7, FACTORING);
}

TEST_CASE("statefulpff.8threads" * doctest::timeout(300)) {
  stateful_parallel_for(8, FACTORING);
}

TEST_CASE("statefulpff.9threads" * doctest::timeout(300)) {
  stateful_parallel_for(9, FACTORING);
}

TEST_CASE("statefulpff.10threads" * doctest::timeout(300)) {
  stateful_parallel_for(10, FACTORING);
}

TEST_CASE("statefulpff.11threads" * doctest::timeout(300)) {
  stateful_parallel_for(11, FACTORING);
}

TEST_CASE("statefulpff.12threads" * doctest::timeout(300)) {
  stateful_parallel_for(12, FACTORING);
}

// --------------------------------------------------------
// Testcase: parallel_reduce
// --------------------------------------------------------

void parallel_reduce(unsigned W, TYPE type) {

  tf::Executor executor(W);
  tf::Taskflow taskflow;

  std::vector<int> vec(1000);

  for(auto& i : vec) i = ::rand() % 100 - 50;

  for(size_t n=1; n<vec.size(); n++) {
    for(size_t c=0; c<=17; c=c*2+1) {

      int smin = std::numeric_limits<int>::max();
      int pmin = std::numeric_limits<int>::max();
      auto beg = vec.end();
      auto end = vec.end();

      taskflow.clear();
      auto stask = taskflow.emplace([&](){
        beg = vec.begin();
        end = vec.begin() + n;
        for(auto itr = beg; itr != end; itr++) {
          smin = std::min(*itr, smin);
        }
      });

      auto ptask = taskflow.parallel_reduce_guided(
        std::ref(beg), std::ref(end), pmin, [](int& l, int& r){
        return std::min(l, r);
      }, c);

      stask.precede(ptask);

      executor.run(taskflow).wait();
      
      REQUIRE(smin != std::numeric_limits<int>::max());
      REQUIRE(pmin != std::numeric_limits<int>::max());
      REQUIRE(smin == pmin);
    }
  }
}

TEST_CASE("prg.1thread" * doctest::timeout(300)) {
  parallel_reduce(1, GUIDED);
}

TEST_CASE("prg.2threads" * doctest::timeout(300)) {
  parallel_reduce(2, GUIDED);
}

TEST_CASE("prg.3threads" * doctest::timeout(300)) {
  parallel_reduce(3, GUIDED);
}

TEST_CASE("prg.4threads" * doctest::timeout(300)) {
  parallel_reduce(4, GUIDED);
}

TEST_CASE("prg.5threads" * doctest::timeout(300)) {
  parallel_reduce(5, GUIDED);
}

TEST_CASE("prg.6threads" * doctest::timeout(300)) {
  parallel_reduce(6, GUIDED);
}

TEST_CASE("prg.7threads" * doctest::timeout(300)) {
  parallel_reduce(7, GUIDED);
}

TEST_CASE("prg.8threads" * doctest::timeout(300)) {
  parallel_reduce(8, GUIDED);
}

TEST_CASE("prg.9threads" * doctest::timeout(300)) {
  parallel_reduce(9, GUIDED);
}

TEST_CASE("prg.10threads" * doctest::timeout(300)) {
  parallel_reduce(10, GUIDED);
}

TEST_CASE("prg.11threads" * doctest::timeout(300)) {
  parallel_reduce(11, GUIDED);
}

TEST_CASE("prg.12threads" * doctest::timeout(300)) {
  parallel_reduce(12, GUIDED);
}





