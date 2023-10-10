#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/sort.hpp>

// ----------------------------------------------------------------------------
// Data Type
// ----------------------------------------------------------------------------

struct MoveOnly1{

  int a {-1234};
  
  MoveOnly1() = default;

  MoveOnly1(const MoveOnly1&) = delete;
  MoveOnly1(MoveOnly1&&) = default;

  MoveOnly1& operator = (const MoveOnly1& rhs) = delete;
  MoveOnly1& operator = (MoveOnly1&& rhs) = default;

};

// ----------------------------------------------------------------------------
// parallel sort
// ----------------------------------------------------------------------------

template <typename T>
void ps_pod(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<T> data(N);

  for(auto& d : data) {
    d = ::rand() % 1000 - 500;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end());

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end()));
}

TEST_CASE("ParallelSort.int.1.100000") {
  ps_pod<int>(1, 100000);
}

TEST_CASE("ParallelSort.int.2.100000") {
  ps_pod<int>(2, 100000);
}

TEST_CASE("ParallelSort.int.3.100000") {
  ps_pod<int>(3, 100000);
}

TEST_CASE("ParallelSort.int.4.100000") {
  ps_pod<int>(4, 100000);
}

TEST_CASE("ParallelSort.ldouble.1.100000") {
  ps_pod<long double>(1, 100000);
}

TEST_CASE("ParallelSort.ldouble.2.100000") {
  ps_pod<long double>(2, 100000);
}

TEST_CASE("ParallelSort.ldouble.3.100000") {
  ps_pod<long double>(3, 100000);
}

TEST_CASE("ParallelSort.ldouble.4.100000") {
  ps_pod<long double>(4, 100000);
}

struct Object {

  std::array<int, 10> integers;

  int sum() const {
    int s = 0;
    for(const auto i : integers) {
      s += i;
    }
    return s;
  }
};

void ps_object(size_t W, size_t N) {

  std::srand(static_cast<unsigned int>(time(NULL)));

  std::vector<Object> data(N);

  for(auto& d : data) {
    for(auto& i : d.integers) {
      i = ::rand();
    }
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(data.begin(), data.end(), [](const auto& l, const auto& r){
    return l.sum() < r.sum();
  });

  executor.run(taskflow).wait();

  REQUIRE(std::is_sorted(data.begin(), data.end(),
    [](const auto& l, const auto& r){ return l.sum() < r.sum(); }
  ));
}

TEST_CASE("ParallelSort.object.1.100000") {
  ps_object(1, 100000);
}

TEST_CASE("ParallelSort.object.2.100000") {
  ps_object(2, 100000);
}

TEST_CASE("ParallelSort.object.3.100000") {
  ps_object(3, 100000);
}

TEST_CASE("ParallelSort.object.4.100000") {
  ps_object(4, 100000);
}

void move_only_ps(unsigned W) {
  
  std::vector<MoveOnly1> vec(1000000);
  for(auto& i : vec) {
    i.a = rand()%100;
  }

  tf::Taskflow taskflow;
  tf::Executor executor(W);

  taskflow.sort(vec.begin(), vec.end(),
    [](const MoveOnly1& m1, const MoveOnly1&m2) {
      return m1.a < m2.a;
    }
  );

  executor.run(taskflow).wait();

  for(size_t i=1; i<vec.size(); i++) {
    REQUIRE(vec[i-1].a <= vec[i].a);
  }

}

TEST_CASE("ParallelSort.MoveOnlyObject.1thread") {
  move_only_ps(1);
}

TEST_CASE("ParallelSort.MoveOnlyObject.2threads") {
  move_only_ps(2);
}

TEST_CASE("ParallelSort.MoveOnlyObject.3threads") {
  move_only_ps(3);
}

TEST_CASE("ParallelSort.MoveOnlyObject.4threads") {
  move_only_ps(4);
}

// --------------------------------------------------------
// Testcase: BubbleSort
// --------------------------------------------------------
TEST_CASE("BubbleSort" * doctest::timeout(300)) {

  for(unsigned w=1; w<=9; w+=2) {

    tf::Executor executor(w);

    for(int end=10; end <= 1000; end += 200) {

      tf::Taskflow taskflow("BubbleSort");

      std::vector<int> data(end);

      for(auto& d : data) d = ::rand()%100;

      auto gold = data;
      std::sort(gold.begin(), gold.end());

      std::atomic<bool>swapped;

      // init task
      auto init = taskflow.emplace([&swapped](){ swapped = false; });
      auto cond = taskflow.emplace([&swapped](){
        if(swapped) {
          swapped = false;
          return 0;
        }
        return 1;
      });
      auto stop = taskflow.emplace([](){});

      auto even_phase = taskflow.emplace([&](tf::Subflow& sf){
        for(size_t i=0; i<data.size(); i+=2) {
          sf.emplace([&, i](){
            if(i+1 < data.size() && data[i] > data[i+1]) {
              std::swap(data[i], data[i+1]);
              swapped = true;
            }
          });
        }
      });

      auto odd_phase = taskflow.emplace([&](tf::Subflow& sf) {
        for(size_t i=1; i<data.size(); i+=2) {
          sf.emplace([&, i](){
            if(i+1 < data.size() && data[i] > data[i+1]) {
              std::swap(data[i], data[i+1]);
              swapped = true;
            }
          });
        }
      });

      init.precede(even_phase).name("init");
      even_phase.precede(odd_phase).name("even-swap");
      odd_phase.precede(cond).name("odd-swap");
      cond.precede(even_phase, stop).name("cond");

      executor.run(taskflow).wait();

      REQUIRE(gold == data);
    }
  }
}

// --------------------------------------------------------
// Testcase: SelectionSort
// --------------------------------------------------------
TEST_CASE("SelectionSort" * doctest::timeout(300)) {

  std::function<
    void(tf::Subflow& sf, std::vector<int>&, int, int, int&)
  > spawn;

  spawn = [&] (
    tf::Subflow& sf,
    std::vector<int>& data,
    int beg,
    int end,
    int& min
  ) mutable {

    if(!(beg < end)) {
      min = -1;
      return;
    }

    if(end - beg == 1) {
      min = beg;
      return;
    }

    int m = (beg + end + 1) / 2;

    auto minl = new int(-1);
    auto minr = new int(-1);

    auto SL = sf.emplace(
      [&spawn, &data, beg, m, l=minl] (tf::Subflow& sf2) mutable {
      spawn(sf2, data, beg, m, *l);
    }).name(std::string("[")
          + std::to_string(beg)
          + ':'
          + std::to_string(m)
          + ')');

    auto SR = sf.emplace(
      [&spawn, &data, m, end, r=minr] (tf::Subflow& sf2) mutable {
      spawn(sf2, data, m, end, *r);
    }).name(std::string("[")
          + std::to_string(m)
          + ':'
          + std::to_string(end)
          + ')');

    auto SM = sf.emplace([&data, &min, minl, minr] () {
      if(*minl == -1) {
        min = *minr;
      }
      else if(*minr == -1) {
        min = *minl;
        return;
      }
      else {
        min = data[*minl] < data[*minr] ? *minl : *minr;
      }
      delete minl;
      delete minr;
    }).name(std::string("merge [")
          + std::to_string(beg)
          + ':'
          + std::to_string(end) + ')');

    SM.succeed(SL, SR);
  };

  for(unsigned w=1; w<=9; w+=2) {

    tf::Executor executor(w);

    for(int end=16; end <= 512; end <<= 1) {
      tf::Taskflow taskflow("SelectionSort");

      std::vector<int> data(end);

      for(auto& d : data) d = ::rand()%100;

      auto gold = data;
      std::sort(gold.begin(), gold.end());

      int beg = 0;
      int min = -1;

      auto start = taskflow.emplace([](){});

      auto argmin = taskflow.emplace(
        [&spawn, &data, &beg, end, &min](tf::Subflow& sf) mutable {
        spawn(sf, data, beg, end, min);
      }).name(std::string("[0")
            + ":"
            + std::to_string(end) + ")");

      auto putmin = taskflow.emplace([&](){
        std::swap(data[beg], data[min]);
        //std::cout << "select " << data[beg] << '\n';
        beg++;
        if(beg < end) {
          min = -1;
          return 0;
        }
        else return 1;
      });

      start.precede(argmin);
      argmin.precede(putmin);
      putmin.precede(argmin);

      executor.run(taskflow).wait();

      REQUIRE(gold == data);
      //std::exit(1);
    }
  }

}

// --------------------------------------------------------
// Testcase: MergeSort
// --------------------------------------------------------
TEST_CASE("MergeSort" * doctest::timeout(300)) {

  std::function<void(tf::Subflow& sf, std::vector<int>&, int, int)> spawn;

  spawn = [&] (tf::Subflow& sf, std::vector<int>& data, int beg, int end) mutable {

    if(!(beg < end) || end - beg == 1) {
      return;
    }

    if(end - beg <= 5) {
      std::sort(data.begin() + beg, data.begin() + end);
      return;
    }

    int m = (beg + end + 1) / 2;

    auto SL = sf.emplace([&spawn, &data, beg, m] (tf::Subflow& sf2) {
      spawn(sf2, data, beg, m);
    }).name(std::string("[")
          + std::to_string(beg)
          + ':'
          + std::to_string(m)
          + ')');

    auto SR = sf.emplace([&spawn, &data, m, end] (tf::Subflow& sf2) {
      spawn(sf2, data, m, end);
    }).name(std::string("[")
          + std::to_string(m)
          + ':'
          + std::to_string(end)
          + ')');

    auto SM = sf.emplace([&data, beg, end, m] () {
      std::vector<int> tmpl, tmpr;
      for(int i=beg; i<m; ++i) tmpl.push_back(data[i]);
      for(int i=m; i<end; ++i) tmpr.push_back(data[i]);

      // merge to data
      size_t i=0, j=0, k=beg;
      while(i<tmpl.size() && j<tmpr.size()) {
        data[k++] = (tmpl[i] < tmpr[j] ? tmpl[i++] : tmpr[j++]);
      }

      // remaining SL
      for(; i<tmpl.size(); ++i) data[k++] = tmpl[i];

      // remaining SR
      for(; j<tmpr.size(); ++j) data[k++] = tmpr[j];
    }).name(std::string("merge [")
          + std::to_string(beg)
          + ':'
          + std::to_string(end) + ')');

    SM.succeed(SL, SR);
  };

  for(unsigned w=1; w<=9; w+=2) {

    tf::Executor executor(w);

    for(int end=10; end <= 10000; end = end * 10) {
      tf::Taskflow taskflow("MergeSort");

      std::vector<int> data(end);

      for(auto& d : data) d = ::rand()%100;

      auto gold = data;

      taskflow.emplace([&spawn, &data, end](tf::Subflow& sf){
        spawn(sf, data, 0, end);
      }).name(std::string("[0")
            + ":"
            + std::to_string(end) + ")");

      executor.run(taskflow).wait();

      std::sort(gold.begin(), gold.end());

      REQUIRE(gold == data);
    }
  }
}

// --------------------------------------------------------
// Testcase: QuickSort
// --------------------------------------------------------
TEST_CASE("QuickSort" * doctest::timeout(300)) {

  using itr_t = std::vector<int>::iterator;

  std::function<void(tf::Subflow& sf, std::vector<int>&, itr_t, itr_t)> spawn;

  spawn = [&] (
    tf::Subflow& sf,
    std::vector<int>& data,
    itr_t beg,
    itr_t end
  ) mutable {

    if(!(beg < end) || std::distance(beg, end) == 1) {
      return;
    }

    if(std::distance(beg, end) <= 5) {
      std::sort(beg, end);
      return;
    }

    auto pvt = beg + std::distance(beg, end) / 2;

    std::iter_swap(pvt, end-1);

    pvt = std::partition(beg, end-1, [end] (int item) {
      return item < *(end - 1);
    });

    std::iter_swap(pvt, end-1);

    sf.emplace([&spawn, &data, beg, pvt] (tf::Subflow& sf2) {
      spawn(sf2, data, beg, pvt);
    }).name(std::string("[")
          + std::to_string(beg-data.begin())
          + ':'
          + std::to_string(pvt-data.begin())
          + ')');

    sf.emplace([&spawn, &data, pvt, end] (tf::Subflow& sf2) {
      spawn(sf2, data, pvt+1, end);
    }).name(std::string("[")
          + std::to_string(pvt-data.begin())
          + ':'
          + std::to_string(end-data.begin())
          + ')');
  };

  for(unsigned w=1; w<=9; w+=2) {

    tf::Executor executor(w);

    for(int end=16; end <= 16384; end <<= 1) {

      tf::Taskflow taskflow("QuickSort");

      std::vector<int> data(end);

      for(auto& d : data) d = ::rand()%100;

      auto gold = data;

      taskflow.emplace([&spawn, &data](tf::Subflow& sf){
        spawn(sf, data, data.begin(), data.end());
      }).name(std::string("[0")
            + ":"
            + std::to_string(end) + ")");

      executor.run(taskflow).wait();

      std::sort(gold.begin(), gold.end());

      REQUIRE(gold == data);
    }
  }
}

//// ----------------------------------------------------------------------------
//// Exception
//// ----------------------------------------------------------------------------
//
//void parallel_sort_exception(unsigned W) {
//
//  tf::Taskflow taskflow;
//  tf::Executor executor(W);
//
//  std::vector<int> data(1000000);
//
//  // for_each
//  taskflow.sort(data.begin(), data.end(), [](int a, int b){
//    throw std::runtime_error("x");
//    return a < b;
//  });
//  REQUIRE_THROWS_WITH_AS(executor.run(taskflow).get(), "x", std::runtime_error);
//}
//
//TEST_CASE("ParallelSort.Exception.1thread") {
//  parallel_sort_exception(1);
//}
//
//TEST_CASE("ParallelSort.Exception.2threads") {
//  parallel_sort_exception(2);
//}
//
//TEST_CASE("ParallelSort.Exception.3threads") {
//  parallel_sort_exception(3);
//}
//
//TEST_CASE("ParallelSort.Exception.4threads") {
//  parallel_sort_exception(4);
//}


