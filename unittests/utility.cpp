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

  struct TestObject {

    TestObject(int v) : value {v} {
    }

    void animate(int v) {
      REQUIRE(value == 0);
      value = v;
    }
    
    void recycle() {
      value = 0;
    }

    int value;
  };
    
  thread_local tf::ObjectPool<TestObject> TestObjectPool;

  auto fork = [&] (unsigned N) {

    const int M = 2048 * N;
    std::atomic<int> counter = M;
    std::atomic<int> recycle = M;
    std::mutex mutex;
    std::vector<std::unique_ptr<TestObject>> objects;
    std::vector<std::thread> threads; 

    // allocate
    for(unsigned t=1; t<=N; ++t) {
      threads.emplace_back([&] () {
        while(1) {
          if(int c = --counter; c < 0) {
            break;
          }
          else {
            auto ptr = TestObjectPool.acquire(c);
            std::scoped_lock lock(mutex);
            objects.push_back(std::move(ptr));
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
      [] (int s, const auto& v) { return s + v->value; }
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
            TestObjectPool.release(std::move(objects.back()));
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
// Testcase: FunctionTraits
// --------------------------------------------------------
void func1() {
}

int func2(int, double, float, char) {
  return 0;
}

TEST_CASE("FunctionTraits" * doctest::timeout(300)) {
  
  SUBCASE("func1") {
    using func1_traits = tf::function_traits<decltype(func1)>;
    static_assert(std::is_same_v<func1_traits::return_type, void>);
    static_assert(func1_traits::arity == 0);
  }
  
  SUBCASE("func2") {
    using func2_traits = tf::function_traits<decltype(func2)>;
    static_assert(std::is_same_v<func2_traits::return_type, int>);
    static_assert(func2_traits::arity == 4);
    static_assert(std::is_same_v<func2_traits::argument_t<0>, int>);
    static_assert(std::is_same_v<func2_traits::argument_t<1>, double>);
    static_assert(std::is_same_v<func2_traits::argument_t<2>, float>);
    static_assert(std::is_same_v<func2_traits::argument_t<3>, char>);
  }

  SUBCASE("lambda1") {
    auto lambda1 = [] () mutable {
      return 1;
    };
    using lambda1_traits = tf::function_traits<decltype(lambda1)>;
    static_assert(std::is_same_v<lambda1_traits::return_type, int>);
    static_assert(lambda1_traits::arity == 0);
  }

  SUBCASE("lambda2") {
    auto lambda2 = [] (int, double, char&) {
    };
    using lambda2_traits = tf::function_traits<decltype(lambda2)>;
    static_assert(std::is_same_v<lambda2_traits::return_type, void>);
    static_assert(lambda2_traits::arity == 3);
    static_assert(std::is_same_v<lambda2_traits::argument_t<0>, int>);
    static_assert(std::is_same_v<lambda2_traits::argument_t<1>, double>);
    static_assert(std::is_same_v<lambda2_traits::argument_t<2>, char&>);
  }

  SUBCASE("class") {
    struct foo {
      int operator ()(int, float) const;
    };
    using foo_traits = tf::function_traits<foo>;
    static_assert(std::is_same_v<foo_traits::return_type, int>);
    static_assert(foo_traits::arity == 2);
    static_assert(std::is_same_v<foo_traits::argument_t<0>, int>);
    static_assert(std::is_same_v<foo_traits::argument_t<1>, float>);
  }

  SUBCASE("std-function") {
    using ft1 = tf::function_traits<std::function<void()>>;
    static_assert(std::is_same_v<ft1::return_type, void>);
    static_assert(ft1::arity == 0);

    using ft2 = tf::function_traits<std::function<int(int&, double&&)>&>;
    static_assert(std::is_same_v<ft2::return_type, int>);
    static_assert(ft2::arity == 2);
    static_assert(std::is_same_v<ft2::argument_t<0>, int&>);
    static_assert(std::is_same_v<ft2::argument_t<1>, double&&>);
    
    using ft3 = tf::function_traits<std::function<int(int&, double&&)>&&>;
    static_assert(std::is_same_v<ft3::return_type, int>);
    static_assert(ft3::arity == 2);
    static_assert(std::is_same_v<ft3::argument_t<0>, int&>);
    static_assert(std::is_same_v<ft3::argument_t<1>, double&&>);

    using ft4 = tf::function_traits<const std::function<void(int)>&>;
    static_assert(std::is_same_v<ft4::return_type, void>);
    static_assert(ft4::arity == 1);
    static_assert(std::is_same_v<ft4::argument_t<0>, int>);
  }
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






