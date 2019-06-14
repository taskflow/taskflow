#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/utility/traits.hpp>
#include <taskflow/utility/object_pool.hpp>
#include <taskflow/utility/passive_vector.hpp>
#include <taskflow/utility/singular_allocator.hpp>

// --------------------------------------------------------
// Testcase: PassiveVector
// --------------------------------------------------------
TEST_CASE("PassiveVector" * doctest::timeout(300)) {

  SUBCASE("constructor") {
    tf::PassiveVector<int> vec1;
    REQUIRE(vec1.size() == 0);
    REQUIRE(vec1.empty() == true);

    tf::PassiveVector<int, 8> vec2;
    REQUIRE(vec2.size() == 0);
    REQUIRE(vec2.empty() == true);
    REQUIRE(vec2.capacity() == 8);
  }

  SUBCASE("constructor_n") {
    for(int N=0; N<=65536; ++N) {
      tf::PassiveVector<int> vec(N);
      REQUIRE(vec.size() == N);
      REQUIRE(vec.empty() == (N == 0));
      REQUIRE(vec.max_size() >= vec.size());
      REQUIRE(vec.capacity() >= vec.size());
    }
  }

  SUBCASE("copy_constructor") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }
      
      tf::PassiveVector<int> vec2(vec1);
      REQUIRE(vec1.size() == N);
      REQUIRE(vec2.size() == N);
      for(size_t i=0; i<vec1.size(); ++i) {
        REQUIRE(vec1[i] == vec2[i]);
        REQUIRE(vec1[i] == N);
      }
    }
  }
  
  SUBCASE("move_constructor") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }
      
      tf::PassiveVector<int> vec2(std::move(vec1));
      REQUIRE(vec1.size() == 0);
      REQUIRE(vec1.empty() == true);
      REQUIRE(vec2.size() == N);

      for(size_t i=0; i<vec2.size(); ++i) {
        REQUIRE(vec2[i] == N);
      }
    }
  }

  SUBCASE("push_back") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec;
      size_t pcap {0};
      size_t ncap {0};
      for(int n=0; n<N; ++n) {
        vec.push_back(n);
        REQUIRE(vec.size() == n+1);
        ncap = vec.capacity();
        REQUIRE(ncap >= pcap);
        pcap = ncap;
      }
      for(int n=0; n<N; ++n) {
        REQUIRE(vec[n] == n);
      }
      REQUIRE(vec.empty() == (N == 0));
    }
  }

  SUBCASE("pop_back") {
    size_t size {0};
    size_t pcap {0};
    size_t ncap {0};
    tf::PassiveVector<int> vec;
    for(int N=0; N<=65536; ++N) {
      vec.push_back(N);
      ++size;
      REQUIRE(vec.size() == size);
      if(N % 4 == 0) {
        vec.pop_back();
        --size;
        REQUIRE(vec.size() == size);
      }
      ncap = vec.capacity();
      REQUIRE(ncap >= pcap);
      pcap = ncap;
    }
    REQUIRE(vec.size() == size);
    for(size_t i=0; i<vec.size(); ++i) {
      REQUIRE(vec[i] % 4 != 0);
    }
  }

  SUBCASE("iterator") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec;
      for(int n=0; n<N; ++n) {
        vec.push_back(n);
        REQUIRE(vec.size() == n+1);
      }
      
      // non-constant iterator
      {
        int val {0};
        for(auto item : vec) {
          REQUIRE(item == val);
          ++val;
        }
      }
      
      // constant iterator
      {
        int val {0};
        for(const auto& item : vec) {
          REQUIRE(item == val);
          ++val;
        }
      }

      // change the value
      {
        for(auto& item : vec) {
          item = 1234;
        }
        for(auto& item : vec) {
          REQUIRE(item == 1234);
        }
      }
    }
  }

  SUBCASE("at") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec(N);
      REQUIRE_THROWS(vec.at(N));
      REQUIRE_THROWS(vec.at(N+1));
      for(int n=0; n<N; ++n) {
        REQUIRE_NOTHROW(vec.at(n));
      }
    }
  }

  SUBCASE("clear") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec(N);
      auto cap = vec.capacity();
      REQUIRE(vec.size() == N);
      vec.clear();
      REQUIRE(vec.size() == 0);
      REQUIRE(vec.capacity() == cap);
    }
  }

  SUBCASE("comparison") {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::PassiveVector<int> vec1;
      for(int i=0; i<N; ++i) {
        vec1.push_back(i);
      }
      tf::PassiveVector<int> vec2(vec1);
      REQUIRE(vec1 == vec2);
    }
  }
}

// --------------------------------------------------------
// Testcase: Pool
// --------------------------------------------------------
TEST_CASE("ObjectPool" * doctest::timeout(300)) {

  auto fork = [&] (unsigned N) {
    const int M = 2048 * N;
    std::atomic<int> counter = M;
    std::atomic<int> recycle = M;
    std::mutex mutex;
    std::vector<int*> objects;
    std::vector<std::thread> threads; 

    // allocate
    for(unsigned t=1; t<=N; ++t) {
      threads.emplace_back([&] () {
        while(1) {
          if(int c = --counter; c < 0) {
            break;
          }
          else {
            auto ptr = tf::per_thread_object_pool<int>().get(c);
            std::scoped_lock lock(mutex);
            objects.push_back(ptr);
          }
        }
      });
    }
    for(auto& thread : threads) {
      thread.join();
    }
    threads.clear();

    REQUIRE(objects.size() == M);

    auto sum = std::accumulate(objects.begin(), objects.end(), 0,
      [] (int s, int* v) { return s + *v; }
    );

    REQUIRE(sum == (M-1)*M / 2);

    // recycle
    for(unsigned t=1; t<=N; ++t) {
      threads.emplace_back([&] () {
        while(1) {
          if(int r = --recycle; r < 0) {
            break;
          }
          else {
            std::scoped_lock lock(mutex);
            REQUIRE(!objects.empty());
            tf::per_thread_object_pool<int>().recycle(objects.back());
            objects.pop_back();
          }
        }
      });
    }
    for(auto& thread : threads) {
      thread.join();
    }
    threads.clear();

    REQUIRE(objects.size() == 0);
  };

  SUBCASE("OneThread")    { fork(1); }
  SUBCASE("TwoThread")    { fork(2); }
  SUBCASE("ThreeThreads") { fork(3); }
  SUBCASE("FourThreads")  { fork(4); }
  SUBCASE("FiveThreads")  { fork(5); }
  SUBCASE("SixThreads")   { fork(6); }
  SUBCASE("SevenThreads") { fork(7); }
  SUBCASE("EightThreads") { fork(8); }
}

// --------------------------------------------------------
// Testcase: SingularAllocator
// --------------------------------------------------------
TEST_CASE("SingularAllocator" * doctest::timeout(300)) {

  tf::SingularAllocator<std::string> allocator;
  std::set<std::string*> set;
  for(int i=0; i<1024; ++i) {
   
    for(int j=0; j<i; ++j) {
      auto sptr = allocator.allocate(1);
      set.insert(sptr);
    }
    REQUIRE(set.size() == i);
    for(auto sptr : set) {
      allocator.deallocate(sptr);
    }

    for(size_t j=0; j<set.size(); ++j) {
      auto sptr = allocator.allocate(1);
      REQUIRE(set.find(sptr) != set.end());
    }
    for(auto sptr : set) {
      allocator.deallocate(sptr);
    }
  }
}






