#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/utility/traits.hpp>
#include <taskflow/utility/object_pool.hpp>
#include <taskflow/utility/passive_vector.hpp>

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
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
TEST_CASE("ObjectPool.Sequential" * doctest::timeout(300)) {

  struct Foo {
    std::string str;
    std::vector<int> vec;
    int a;
    char b;
  };

  for(unsigned w=1; w<=4; w++) {

    tf::ObjectPool<Foo> pool(w);

    REQUIRE(pool.num_heaps() > 0);
    REQUIRE(pool.num_local_heaps() > 0);
    REQUIRE(pool.num_global_heaps() > 0);
    REQUIRE(pool.num_bins_per_local_heap() > 0);
    REQUIRE(pool.num_objects_per_bin() > 0);
    REQUIRE(pool.num_objects_per_block() > 0);
    REQUIRE(pool.emptiness_threshold() > 0);
    
    // fill out all object objects
    int N = 1000*pool.num_objects_per_block();

    std::set<Foo*> set;

    for(int i=0; i<N; ++i) {
      auto item = pool.allocate();
      REQUIRE(set.find(item) == set.end());
      set.insert(item);
    }

    REQUIRE(set.size() == N);

    for(auto s : set) {
      pool.deallocate(s);
    }

    REQUIRE(N == pool.capacity());
    REQUIRE(N == pool.num_available_objects());
    REQUIRE(0 == pool.num_allocated_objects());
    
    for(int i=0; i<N; ++i) {
      auto item = pool.allocate();
      REQUIRE(set.find(item) != set.end());
    }

    REQUIRE(pool.num_available_objects() == 0);
    REQUIRE(pool.num_allocated_objects() == N);
  }
}

// --------------------------------------------------------
// Testcase: ObjectPool.Threaded
// --------------------------------------------------------

template <typename T>
void threaded_objectpool(unsigned W) {
  
  tf::ObjectPool<T> pool;

  std::vector<std::thread> threads;

  for(unsigned w=0; w<W; ++w) {
    threads.emplace_back([&pool](){
      std::vector<T*> items;
      for(int i=0; i<65536; ++i) {
        auto item = pool.allocate();
        new (item) T();
        items.push_back(item);
      }
      for(auto item : items) {
        item->~T();
        pool.deallocate(item);
      }
    });
  }

  for(auto& thread : threads) {
    thread.join();
  }

  REQUIRE(pool.num_allocated_objects() == 0);
  REQUIRE(pool.num_available_objects() == pool.capacity());
}

TEST_CASE("ObjectPool.1thread" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(1);
  threaded_objectpool<int16_t>(1);
  threaded_objectpool<int32_t>(1);
  threaded_objectpool<int64_t>(1);
  threaded_objectpool<std::string>(1);
}

TEST_CASE("ObjectPool.2threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(2);
  threaded_objectpool<int16_t>(2);
  threaded_objectpool<int32_t>(2);
  threaded_objectpool<int64_t>(2);
  threaded_objectpool<std::string>(2);
}

TEST_CASE("ObjectPool.3threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(3);
  threaded_objectpool<int16_t>(3);
  threaded_objectpool<int32_t>(3);
  threaded_objectpool<int64_t>(3);
  threaded_objectpool<std::string>(3);
}

TEST_CASE("ObjectPool.4threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(4);
  threaded_objectpool<int16_t>(4);
  threaded_objectpool<int32_t>(4);
  threaded_objectpool<int64_t>(4);
  threaded_objectpool<std::string>(4);
}

TEST_CASE("ObjectPool.5threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(5);
  threaded_objectpool<int16_t>(5);
  threaded_objectpool<int32_t>(5);
  threaded_objectpool<int64_t>(5);
  threaded_objectpool<std::string>(5);
}

TEST_CASE("ObjectPool.6threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(6);
  threaded_objectpool<int16_t>(6);
  threaded_objectpool<int32_t>(6);
  threaded_objectpool<int64_t>(6);
  threaded_objectpool<std::string>(6);
}

TEST_CASE("ObjectPool.7threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(7);
  threaded_objectpool<int16_t>(7);
  threaded_objectpool<int32_t>(7);
  threaded_objectpool<int64_t>(7);
  threaded_objectpool<std::string>(7);
}

TEST_CASE("ObjectPool.8threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(8);
  threaded_objectpool<int16_t>(8);
  threaded_objectpool<int32_t>(8);
  threaded_objectpool<int64_t>(8);
  threaded_objectpool<std::string>(8);
}

TEST_CASE("ObjectPool.9threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(9);
  threaded_objectpool<int16_t>(9);
  threaded_objectpool<int32_t>(9);
  threaded_objectpool<int64_t>(9);
  threaded_objectpool<std::string>(9);
}

TEST_CASE("ObjectPool.10threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(10);
  threaded_objectpool<int16_t>(10);
  threaded_objectpool<int32_t>(10);
  threaded_objectpool<int64_t>(10);
  threaded_objectpool<std::string>(10);
}

TEST_CASE("ObjectPool.11threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(11);
  threaded_objectpool<int16_t>(11);
  threaded_objectpool<int32_t>(11);
  threaded_objectpool<int64_t>(11);
  threaded_objectpool<std::string>(11);
}

TEST_CASE("ObjectPool.12threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(12);
  threaded_objectpool<int16_t>(12);
  threaded_objectpool<int32_t>(12);
  threaded_objectpool<int64_t>(12);
  threaded_objectpool<std::string>(12);
}

TEST_CASE("ObjectPool.13threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(13);
  threaded_objectpool<int16_t>(13);
  threaded_objectpool<int32_t>(13);
  threaded_objectpool<int64_t>(13);
  threaded_objectpool<std::string>(13);
}

TEST_CASE("ObjectPool.14threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(14);
  threaded_objectpool<int16_t>(14);
  threaded_objectpool<int32_t>(14);
  threaded_objectpool<int64_t>(14);
  threaded_objectpool<std::string>(14);
}

TEST_CASE("ObjectPool.15threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(15);
  threaded_objectpool<int16_t>(15);
  threaded_objectpool<int32_t>(15);
  threaded_objectpool<int64_t>(15);
  threaded_objectpool<std::string>(15);
}

TEST_CASE("ObjectPool.16threads" * doctest::timeout(300)) {
  threaded_objectpool<int8_t>(16);
  threaded_objectpool<int16_t>(16);
  threaded_objectpool<int32_t>(16);
  threaded_objectpool<int64_t>(16);
  threaded_objectpool<std::string>(16);
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
    static_assert(std::is_same<func1_traits::return_type, void>::value, "");
    static_assert(func1_traits::arity == 0, "");
  }
  
  SUBCASE("func2") {
    using func2_traits = tf::function_traits<decltype(func2)>;
    static_assert(std::is_same<func2_traits::return_type, int>::value, "");
    static_assert(func2_traits::arity == 4, "");
    static_assert(std::is_same<func2_traits::argument_t<0>, int>::value,   "");
    static_assert(std::is_same<func2_traits::argument_t<1>, double>::value,"");
    static_assert(std::is_same<func2_traits::argument_t<2>, float>::value, "");
    static_assert(std::is_same<func2_traits::argument_t<3>, char>::value,  "");
  }

  SUBCASE("lambda1") {
    auto lambda1 = [] () mutable {
      return 1;
    };
    using lambda1_traits = tf::function_traits<decltype(lambda1)>;
    static_assert(std::is_same<lambda1_traits::return_type, int>::value, "");
    static_assert(lambda1_traits::arity == 0, "");
  }

  SUBCASE("lambda2") {
    auto lambda2 = [] (int, double, char&) {
    };
    using lambda2_traits = tf::function_traits<decltype(lambda2)>;
    static_assert(std::is_same<lambda2_traits::return_type, void>::value, "");
    static_assert(lambda2_traits::arity == 3, "");
    static_assert(std::is_same<lambda2_traits::argument_t<0>, int>::value, "");
    static_assert(std::is_same<lambda2_traits::argument_t<1>, double>::value, "");
    static_assert(std::is_same<lambda2_traits::argument_t<2>, char&>::value, "");
  }

  SUBCASE("class") {
    struct foo {
      int operator ()(int, float) const;
    };
    using foo_traits = tf::function_traits<foo>;
    static_assert(std::is_same<foo_traits::return_type, int>::value, "");
    static_assert(foo_traits::arity == 2, "");
    static_assert(std::is_same<foo_traits::argument_t<0>, int>::value, "");
    static_assert(std::is_same<foo_traits::argument_t<1>, float>::value, "");
  }

  SUBCASE("std-function") {
    using ft1 = tf::function_traits<std::function<void()>>;
    static_assert(std::is_same<ft1::return_type, void>::value, "");
    static_assert(ft1::arity == 0, "");

    using ft2 = tf::function_traits<std::function<int(int&, double&&)>&>;
    static_assert(std::is_same<ft2::return_type, int>::value, "");
    static_assert(ft2::arity == 2, "");
    static_assert(std::is_same<ft2::argument_t<0>, int&>::value, "");
    static_assert(std::is_same<ft2::argument_t<1>, double&&>::value, "");
    
    using ft3 = tf::function_traits<std::function<int(int&, double&&)>&&>;
    static_assert(std::is_same<ft3::return_type, int>::value, "");
    static_assert(ft3::arity == 2, "");
    static_assert(std::is_same<ft3::argument_t<0>, int&>::value, "");
    static_assert(std::is_same<ft3::argument_t<1>, double&&>::value, "");

    using ft4 = tf::function_traits<const std::function<void(int)>&>;
    static_assert(std::is_same<ft4::return_type, void>::value, "");
    static_assert(ft4::arity == 1, "");
    static_assert(std::is_same<ft4::argument_t<0>, int>::value, "");
  }
}






