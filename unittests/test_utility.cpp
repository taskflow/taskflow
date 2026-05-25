#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <taskflow/utility/traits.hpp>
//#include <taskflow/utility/object_pool.hpp>
#include <taskflow/utility/small_vector.hpp>
#include <taskflow/utility/uuid.hpp>
#include <taskflow/utility/iterator.hpp>
#include <taskflow/utility/math.hpp>

// --------------------------------------------------------
// Testcase: SmallVector
// --------------------------------------------------------
TEST_CASE("SmallVector" * doctest::timeout(300)) {

  //SUBCASE("constructor")
  {
    tf::SmallVector<int> vec1;
    REQUIRE(vec1.size() == 0);
    REQUIRE(vec1.empty() == true);

    tf::SmallVector<int, 4> vec2;
    REQUIRE(vec2.data() != nullptr);
    REQUIRE(vec2.size() == 0);
    REQUIRE(vec2.empty() == true);
    REQUIRE(vec2.capacity() == 4);
  }

  //SUBCASE("constructor_n")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec(N);
      REQUIRE(vec.size() == N);
      REQUIRE(vec.empty() == (N == 0));
      REQUIRE(vec.max_size() >= vec.size());
      REQUIRE(vec.capacity() >= vec.size());
    }
  }

  //SUBCASE("copy_constructor")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }

      tf::SmallVector<int> vec2(vec1);
      REQUIRE(vec1.size() == N);
      REQUIRE(vec2.size() == N);
      for(size_t i=0; i<vec1.size(); ++i) {
        REQUIRE(vec1[i] == vec2[i]);
        REQUIRE(vec1[i] == N);
      }
    }
  }

  //SUBCASE("move_constructor")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec1(N);
      for(auto& item : vec1) {
        item = N;
      }

      tf::SmallVector<int> vec2(std::move(vec1));
      REQUIRE(vec1.size() == 0);
      REQUIRE(vec1.empty() == true);
      REQUIRE(vec2.size() == N);

      for(size_t i=0; i<vec2.size(); ++i) {
        REQUIRE(vec2[i] == N);
      }
    }
  }

  //SUBCASE("push_back")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec;
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

  //SUBCASE("pop_back")
  {
    size_t size {0};
    size_t pcap {0};
    size_t ncap {0};
    tf::SmallVector<int> vec;
    for(int N=0; N<=65536; N = (N ? N << 1 : N + 1)) {
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

  //SUBCASE("iterator")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec;
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

  //SUBCASE("clear")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec(N);
      auto cap = vec.capacity();
      REQUIRE(vec.size() == N);
      vec.clear();
      REQUIRE(vec.size() == 0);
      REQUIRE(vec.capacity() == cap);
    }
  }

  //SUBCASE("comparison")
  {
    for(int N=0; N<=65536; N = (N ? N << 1 : 1)) {
      tf::SmallVector<int> vec1;
      for(int i=0; i<N; ++i) {
        vec1.push_back(i);
      }
      tf::SmallVector<int> vec2(vec1);
      REQUIRE(vec1 == vec2);
    }
  }
}

// --------------------------------------------------------
// Testcase: distance
// --------------------------------------------------------
TEST_CASE("distance.integral" * doctest::timeout(300)) {

  auto count = [] (int beg, int end, int step) {
    size_t c = 0;
    for(int i=beg; step > 0 ? i < end : i > end; i += step) {
      ++c;
    }
    return c;
  };

  for(int beg=-50; beg<=50; ++beg) {
    for(int end=-50; end<=50; ++end) {
      if(beg < end) {   // positive step
        for(int s=1; s<=50; s++) {
          REQUIRE((tf::distance(beg, end, s) == count(beg, end, s)));
        }
      }
      else {            // negative step
        for(int s=-1; s>=-50; s--) {
          REQUIRE((tf::distance(beg, end, s) == count(beg, end, s)));
        }
      }
    }
  }

}

// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
/*
// Due to random # generation, this threaded program has a bug
void test_threaded_uuid(size_t N) {

  std::vector<tf::UUID> uuids(65536);

  // threaded
  std::mutex mutex;
  std::vector<std::thread> threads;

  for(size_t i=0; i<N; ++i) {
    threads.emplace_back([&](){
      for(int j=0; j<1000; ++j) {
        std::lock_guard<std::mutex> lock(mutex);
        uuids.push_back(tf::UUID());
      }
    });
  }

  for(auto& t : threads) {
    t.join();
  }

  auto size = uuids.size();
  std::sort(uuids.begin(), uuids.end());
  auto it = std::unique(uuids.begin(), uuids.end());
  REQUIRE(it - uuids.begin() == size);
}

TEST_CASE("uuid.10threads") {
  test_threaded_uuid(10);
}

TEST_CASE("uuid.100threads") {
  test_threaded_uuid(100);
}
*/

TEST_CASE("uuid") {

  tf::UUID u1, u2, u3, u4;

  // Comparator.
  REQUIRE(u1 == u1);

  // Copy
  u2 = u1;
  REQUIRE(u1 == u2);

  // Move
  u3 = std::move(u1);
  REQUIRE(u2 == u3);

  // Copy constructor
  tf::UUID u5(u4);
  REQUIRE(u5 == u4);

  // Move constructor.
  tf::UUID u6(std::move(u4));
  REQUIRE(u5 == u6);

  // Uniqueness
  std::vector<tf::UUID> uuids(65536);
  std::sort(uuids.begin(), uuids.end());
  auto it = std::unique(uuids.begin(), uuids.end());
  REQUIRE(it - uuids.begin() == 65536);

}



/*

// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
struct Poolable {
  std::string str;
  std::vector<int> vec;
  int a;
  char b;

  TF_ENABLE_POOLABLE_ON_THIS;
};

TEST_CASE("ObjectPool.Sequential" * doctest::timeout(300)) {

  for(unsigned w=1; w<=4; w++) {

    tf::ObjectPool<Poolable> pool(w);

    REQUIRE(pool.num_heaps() > 0);
    REQUIRE(pool.num_local_heaps() > 0);
    REQUIRE(pool.num_global_heaps() > 0);
    REQUIRE(pool.num_bins_per_local_heap() > 0);
    REQUIRE(pool.num_objects_per_bin() > 0);
    REQUIRE(pool.num_objects_per_block() > 0);
    REQUIRE(pool.emptiness_threshold() > 0);

    // fill out all objects
    size_t N = 100*pool.num_objects_per_block();

    std::set<Poolable*> set;

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
      REQUIRE(set.find(item) == set.end());
      set.insert(item);
    }

    REQUIRE(set.size() == N);

    for(auto s : set) {
      pool.recycle(s);
    }

    REQUIRE(N == pool.capacity());
    REQUIRE(N == pool.num_available_objects());
    REQUIRE(0 == pool.num_allocated_objects());

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
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
        auto item = pool.animate();
        items.push_back(item);
      }
      for(auto item : items) {
        pool.recycle(item);
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
  threaded_objectpool<Poolable>(1);
}

TEST_CASE("ObjectPool.2threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(2);
}

TEST_CASE("ObjectPool.3threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(3);
}

TEST_CASE("ObjectPool.4threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(4);
}

TEST_CASE("ObjectPool.5threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(5);
}

TEST_CASE("ObjectPool.6threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(6);
}

TEST_CASE("ObjectPool.7threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(7);
}

TEST_CASE("ObjectPool.8threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(8);
}

TEST_CASE("ObjectPool.9threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(9);
}

TEST_CASE("ObjectPool.10threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(10);
}

TEST_CASE("ObjectPool.11threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(11);
}

TEST_CASE("ObjectPool.12threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(12);
}

TEST_CASE("ObjectPool.13threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(13);
}

TEST_CASE("ObjectPool.14threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(14);
}

TEST_CASE("ObjectPool.15threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(15);
}

TEST_CASE("ObjectPool.16threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(16);
}
*/


// --------------------------------------------------------
// Testcase: Reference Wrapper
// --------------------------------------------------------

TEST_CASE("RefWrapper" * doctest::timeout(300)) {

  static_assert(std::is_same<
    std::unwrap_reference_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_reference_t<int&>, int&
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_reference_t<int&&>, int&&
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_reference_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_reference_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_ref_decay_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_ref_decay_t<int&>, int
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_ref_decay_t<int&&>, int
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_ref_decay_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    std::unwrap_ref_decay_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

}

// --------------------------------------------------------
// Math utilities
// --------------------------------------------------------
TEST_CASE("NextPow2") {

  static_assert(tf::next_pow2(0u) == 1);
  static_assert(tf::next_pow2(1u) == 1);
  static_assert(tf::next_pow2(100u) == 128u);
  static_assert(tf::next_pow2(245u) == 256u);
  static_assert(tf::next_pow2(512u) == 512u);
  static_assert(tf::next_pow2(513u) == 1024u);

  REQUIRE(tf::next_pow2(0u) == 1u);
  REQUIRE(tf::next_pow2(2u) == 2u);
  REQUIRE(tf::next_pow2(1u) == 1u);
  REQUIRE(tf::next_pow2(33u) == 64u);
  REQUIRE(tf::next_pow2(100u) == 128u);
  REQUIRE(tf::next_pow2(211u) == 256u);
  REQUIRE(tf::next_pow2(23u) == 32u);
  REQUIRE(tf::next_pow2(54u) == 64u);

  uint64_t z = 0;
  uint64_t a = 1;
  REQUIRE(tf::next_pow2(z) == 1);
  REQUIRE(tf::next_pow2(a) == a);
  REQUIRE(tf::next_pow2((a<<5)  + 0) == (a<<5));
  REQUIRE(tf::next_pow2((a<<5)  + 1) == (a<<6));
  REQUIRE(tf::next_pow2((a<<32) + 0) == (a<<32));
  REQUIRE(tf::next_pow2((a<<32) + 1) == (a<<33));
  REQUIRE(tf::next_pow2((a<<41) + 0) == (a<<41));
  REQUIRE(tf::next_pow2((a<<41) + 1) == (a<<42));

  REQUIRE(tf::is_pow2(0) == false);
  REQUIRE(tf::is_pow2(1) == true);
  REQUIRE(tf::is_pow2(2) == true);
  REQUIRE(tf::is_pow2(3) == false);
  REQUIRE(tf::is_pow2(0u) == false);
  REQUIRE(tf::is_pow2(1u) == true);
  REQUIRE(tf::is_pow2(54u) == false);
  REQUIRE(tf::is_pow2(64u) == true);
}


// ----------------------------------------------------------------------------
// test coprimes
// ----------------------------------------------------------------------------

TEST_CASE("Coprimes") {

  // Compile-time checks for known values
  static_assert(tf::coprime(1) == 1);
  static_assert(tf::coprime(2) == 1);
  static_assert(tf::coprime(3) == 2);
  static_assert(tf::coprime(4) == 3);
  static_assert(tf::coprime(5) == 4);
  static_assert(tf::coprime(6) == 5);
  static_assert(tf::coprime(7) == 6);
  static_assert(tf::coprime(8) == 7);
  static_assert(tf::coprime(9) == 8);
  static_assert(tf::coprime(10) == 9);
  static_assert(tf::coprime(11) == 10);
  static_assert(tf::coprime(12) == 11);
  static_assert(tf::coprime(13) == 12);
  static_assert(tf::coprime(14) == 13);
  static_assert(tf::coprime(15) == 14);
  static_assert(tf::coprime(16) == 15);
  static_assert(tf::coprime(17) == 16);
  static_assert(tf::coprime(18) == 17);
  static_assert(tf::coprime(19) == 18);
  static_assert(tf::coprime(20) == 19);

  constexpr auto coprime_table = tf::make_coprime_lut<51>();
  
  static_assert(coprime_table[1] == 1);
  static_assert(coprime_table[2] == 1);
  static_assert(coprime_table[3] == 2);
  static_assert(coprime_table[4] == 3);
  static_assert(coprime_table[5] == 4);
  static_assert(coprime_table[6] == 5);
  static_assert(coprime_table[7] == 6);
  static_assert(coprime_table[8] == 7);
  static_assert(coprime_table[9] == 8);
  static_assert(coprime_table[10] == 9);
  static_assert(coprime_table[11] == 10);
  static_assert(coprime_table[12] == 11);
  static_assert(coprime_table[13] == 12);
  static_assert(coprime_table[14] == 13);
  static_assert(coprime_table[15] == 14);
  static_assert(coprime_table[16] == 15);
  static_assert(coprime_table[17] == 16);
  static_assert(coprime_table[18] == 17);
  static_assert(coprime_table[19] == 18);
  static_assert(coprime_table[20] == 19);

  // Runtime assertions for all values up to 50
  for (size_t i = 1; i <= 50; ++i) {
    REQUIRE(std::gcd(i, coprime_table[i]) == 1);
    REQUIRE(tf::coprime(i) == coprime_table[i]);
    
    // randomly generate a coprime
    auto v = ::rand() % 1048 + 2;
    auto c = tf::coprime(v);
    REQUIRE(std::gcd(v, c) == 1);
    REQUIRE(c < v);
  }
  
}

// ----------------------------------------------------------------------------
// xorshift
// ----------------------------------------------------------------------------

TEST_CASE("Xorshift.Zero") {
  {
    tf::Xorshift<uint32_t> rng(0);
    for(int i = 0; i < 10; ++i) {
      assert(rng() == 0);
    }
  }

  { 
    tf::Xorshift<uint64_t> rng(0);
    for(int i = 0; i < 10; ++i) {
      assert(rng() == 0);
    }
  }
}

TEST_CASE("Xorshift.Determinism") {
  {
    tf::Xorshift<uint32_t> rng1(42);
    tf::Xorshift<uint32_t> rng2(42);
    for(int i = 0; i < 10; ++i) {
      REQUIRE(rng1() == rng2());
    }
  }
  {
    tf::Xorshift<uint64_t> rng1(42);
    tf::Xorshift<uint64_t> rng2(42);
    for(int i = 0; i < 10; ++i) {
      REQUIRE(rng1() == rng2());
    }
  }
}

template <typename T>
void xorshift_uniformity(size_t bits) {

  const size_t MASK = (1<<bits) - 1;
  const size_t N = 1000000;

  tf::Xorshift<T> rng(0xC0FFEE4U);
  uint64_t sum = 0;
  
  for(size_t i = 0; i < N; ++i) {
    sum += (rng() & MASK); 
  }
  double avg = static_cast<double>(sum) / N;
  double expected = ((1<<bits) - 1) / 2.0; 

  //std::cout << expected << " vs " << avg << " : delta = " 
  //          << std::abs(expected - avg)/expected*100 << "%\n";
  
  // Allow 1% tolerance
  REQUIRE((avg > expected * 0.99));
  REQUIRE((avg < expected * 1.01));
}

TEST_CASE("Xorshift.uint32.Uniformity.16bits") {
  xorshift_uniformity<uint32_t>(16);
}

TEST_CASE("Xorshift.uint32.Uniformity.15bits") {
  xorshift_uniformity<uint32_t>(15);
}

TEST_CASE("Xorshift.uint32.Uniformity.14bits") {
  xorshift_uniformity<uint32_t>(14);
}

TEST_CASE("Xorshift.uint32.Uniformity.13bits") {
  xorshift_uniformity<uint32_t>(13);
}

TEST_CASE("Xorshift.uint32.Uniformity.12bits") {
  xorshift_uniformity<uint32_t>(12);
}

TEST_CASE("Xorshift.uint32.Uniformity.11bits") {
  xorshift_uniformity<uint32_t>(11);
}

TEST_CASE("Xorshift.uint32.Uniformity.10bits") {
  xorshift_uniformity<uint32_t>(10);
}

TEST_CASE("Xorshift.uint32.Uniformity.9bits") {
  xorshift_uniformity<uint32_t>(9);
}

TEST_CASE("Xorshift.uint32.Uniformity.8bits") {
  xorshift_uniformity<uint32_t>(8);
}

TEST_CASE("Xorshift.uint32.Uniformity.7bits") {
  xorshift_uniformity<uint32_t>(7);
}

TEST_CASE("Xorshift.uint32.Uniformity.6bits") {
  xorshift_uniformity<uint32_t>(6);
}

TEST_CASE("Xorshift.uint32.Uniformity.5bits") {
  xorshift_uniformity<uint32_t>(5);
}

TEST_CASE("Xorshift.uint32.Uniformity.4bits") {
  xorshift_uniformity<uint32_t>(4);
}

TEST_CASE("Xorshift.uint32.Uniformity.3bits") {
  xorshift_uniformity<uint32_t>(3);
}

TEST_CASE("Xorshift.uint32.Uniformity.2bits") {
  xorshift_uniformity<uint32_t>(2);
}

TEST_CASE("Xorshift.uint32.Uniformity.1bits") {
  xorshift_uniformity<uint32_t>(1);
}

TEST_CASE("Xorshift.uint64.Uniformity.16bits") {
  xorshift_uniformity<uint64_t>(16);
}

TEST_CASE("Xorshift.uint64.Uniformity.15bits") {
  xorshift_uniformity<uint64_t>(15);
}

TEST_CASE("Xorshift.uint64.Uniformity.14bits") {
  xorshift_uniformity<uint64_t>(14);
}

TEST_CASE("Xorshift.uint64.Uniformity.13bits") {
  xorshift_uniformity<uint64_t>(13);
}

TEST_CASE("Xorshift.uint64.Uniformity.12bits") {
  xorshift_uniformity<uint64_t>(12);
}

TEST_CASE("Xorshift.uint64.Uniformity.11bits") {
  xorshift_uniformity<uint64_t>(11);
}

TEST_CASE("Xorshift.uint64.Uniformity.10bits") {
  xorshift_uniformity<uint64_t>(10);
}

TEST_CASE("Xorshift.uint64.Uniformity.9bits") {
  xorshift_uniformity<uint64_t>(9);
}

TEST_CASE("Xorshift.uint64.Uniformity.8bits") {
  xorshift_uniformity<uint64_t>(8);
}

TEST_CASE("Xorshift.uint64.Uniformity.7bits") {
  xorshift_uniformity<uint64_t>(7);
}

TEST_CASE("Xorshift.uint64.Uniformity.6bits") {
  xorshift_uniformity<uint64_t>(6);
}

TEST_CASE("Xorshift.uint64.Uniformity.5bits") {
  xorshift_uniformity<uint64_t>(5);
}

TEST_CASE("Xorshift.uint64.Uniformity.4bits") {
  xorshift_uniformity<uint64_t>(4);
}

TEST_CASE("Xorshift.uint64.Uniformity.3bits") {
  xorshift_uniformity<uint64_t>(3);
}

TEST_CASE("Xorshift.uint64.Uniformity.2bits") {
  xorshift_uniformity<uint64_t>(2);
}

TEST_CASE("Xorshift.uint64.Uniformity.1bits") {
  xorshift_uniformity<uint64_t>(1);
}



// --------------------------------------------------------
// Testcase: NaryOperatorLike.Basic
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.Basic" * doctest::timeout(300)) {

  // 0-ary
  {
    auto f0 = []() {};
    static_assert(tf::NaryOperatorLike<decltype(f0) , 0>);
    static_assert(!tf::NaryOperatorLike<decltype(f0), 1>);
    static_assert(!tf::NaryOperatorLike<decltype(f0), 2>);
    static_assert(!tf::NaryOperatorLike<decltype(f0), 3>);
  }

  // 1-ary
  {
    auto f1 = [](int) {};
    static_assert(!tf::NaryOperatorLike<decltype(f1), 0>);
    static_assert(tf::NaryOperatorLike<decltype(f1) , 1>);
    static_assert(!tf::NaryOperatorLike<decltype(f1), 2>);
    static_assert(!tf::NaryOperatorLike<decltype(f1), 3>);
  }

  // 2-ary
  {
    auto f2 = [](int, int) {};
    static_assert(!tf::NaryOperatorLike<decltype(f2), 0>);
    static_assert(!tf::NaryOperatorLike<decltype(f2), 1>);
    static_assert(tf::NaryOperatorLike<decltype(f2) , 2>);
    static_assert(!tf::NaryOperatorLike<decltype(f2), 3>);
  }

  // 3-ary
  {
    auto f3 = [](int, int, int) {};
    static_assert(!tf::NaryOperatorLike<decltype(f3), 0>);
    static_assert(!tf::NaryOperatorLike<decltype(f3), 1>);
    static_assert(!tf::NaryOperatorLike<decltype(f3), 2>);
    static_assert(tf::NaryOperatorLike<decltype(f3) , 3>);
  }
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.DefaultArguments
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.DefaultArguments" * doctest::timeout(300)) {

  // default arguments allow fewer args
  {
    auto f = [](int, int = 0) {};

    static_assert(!tf::NaryOperatorLike<decltype(f), 0>);
    static_assert(tf::NaryOperatorLike<decltype(f), 1>);
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
    static_assert(!tf::NaryOperatorLike<decltype(f), 3>);
  }

  {
    auto f = [](int, int = 0, int = 0) {};

    static_assert(tf::NaryOperatorLike<decltype(f), 1>);
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
    static_assert(tf::NaryOperatorLike<decltype(f), 3>);
    static_assert(!tf::NaryOperatorLike<decltype(f), 4>);
  }
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.Variadic
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.Variadic" * doctest::timeout(300)) {

  // variadic lambda accepts any N
  {
    auto f = [](auto...) {};

    static_assert(tf::NaryOperatorLike<decltype(f), 0>);
    static_assert(tf::NaryOperatorLike<decltype(f), 1>);
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
    static_assert(tf::NaryOperatorLike<decltype(f), 5>);
  }

  // variadic with fixed prefix
  {
    auto f = [](int, auto...) {};

    static_assert(!tf::NaryOperatorLike<decltype(f), 0>);
    static_assert(tf::NaryOperatorLike<decltype(f), 1>);
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
    static_assert(tf::NaryOperatorLike<decltype(f), 10>);
  }
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.FunctionPointer
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.FunctionPointer" * doctest::timeout(300)) {

  // function pointer
  {
    using F = void(*)(int, int);

    static_assert(tf::NaryOperatorLike<F, 2>);
    static_assert(!tf::NaryOperatorLike<F, 1>);
    static_assert(!tf::NaryOperatorLike<F, 3>);
  }

  // function reference
  {
    auto f = [](int, int, int) {};
    using F = decltype(f)&;

    static_assert(tf::NaryOperatorLike<F, 3>);
  }
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.Functor
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.Functor" * doctest::timeout(300)) {

  struct Unary {
    void operator()(int) const {}
  };

  struct Binary {
    void operator()(int, int) const {}
  };

  struct Overloaded {
    void operator()(int) const {}
    void operator()(int, int) const {}
  };

  static_assert(tf::NaryOperatorLike<Unary, 1>);
  static_assert(!tf::NaryOperatorLike<Unary, 2>);

  static_assert(tf::NaryOperatorLike<Binary, 2>);
  static_assert(!tf::NaryOperatorLike<Binary, 1>);

  // overloaded: satisfies both
  static_assert(tf::NaryOperatorLike<Overloaded, 1>);
  static_assert(tf::NaryOperatorLike<Overloaded, 2>);
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.StdFunction
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.StdFunction" * doctest::timeout(300)) {

  {
    std::function<void(int)> f;
    static_assert(tf::NaryOperatorLike<decltype(f), 1>);
    static_assert(!tf::NaryOperatorLike<decltype(f), 2>);
  }

  {
    std::function<void(int, int)> f;
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
    static_assert(!tf::NaryOperatorLike<decltype(f), 1>);
  }
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.RefWrapper
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.RefWrapper" * doctest::timeout(300)) {

  auto f = [](int, int) {};

  auto rf = std::ref(f);

  static_assert(tf::NaryOperatorLike<decltype(rf), 2>);
  static_assert(!tf::NaryOperatorLike<decltype(rf), 1>);
}

// --------------------------------------------------------
// Testcase: NaryOperatorLike.EdgeCases
// --------------------------------------------------------
TEST_CASE("NaryOperatorLike.EdgeCases" * doctest::timeout(300)) {

  // non-callable
  {
    static_assert(!tf::NaryOperatorLike<int, 0>);
    static_assert(!tf::NaryOperatorLike<int, 1>);
  }

  // callable returning something
  {
    auto f = [](int, int) { return 42; };
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
  }

  // weird signature (reference, const, etc.)
  {
    auto f = [](const int&, double&&) {};
    static_assert(tf::NaryOperatorLike<decltype(f), 2>);
  }
}


// ------------------------------------------------------------------------------------------------
// unroll
// ------------------------------------------------------------------------------------------------

TEST_CASE("Unroll" * doctest::timeout(300)) {

  // ── empty range ───────────────────────────────────────────────────
  {
    int count = 0;
    tf::unroll<0,  0, 1>([&](int){ count++; });
    tf::unroll<5,  5, 1>([&](int){ count++; });
    tf::unroll<5,  5, 3>([&](int){ count++; });
    tf::unroll<100,100,7>([&](int){ count++; });
    REQUIRE(count == 0);
  }

  // ── single iteration ──────────────────────────────────────────────
  {
    int seen = -1;
    tf::unroll<7, 8, 1>([&](int i){ seen = i; });
    REQUIRE(seen == 7);

    // step larger than the range → exactly one call at beg
    seen = -1;
    tf::unroll<3, 5, 99>([&](int i){ seen = i; });
    REQUIRE(seen == 3);
  }

  // ── exact indices, step = 1 ───────────────────────────────────────
  {
    std::vector<int> got;
    tf::unroll<0, 5, 1>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,1,2,3,4});

    got.clear();
    tf::unroll<10, 20, 1>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{10,11,12,13,14,15,16,17,18,19});
  }

  // ── step > 1, evenly divisible ────────────────────────────────────
  {
    std::vector<int> got;
    tf::unroll<0, 10, 2>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,2,4,6,8});

    got.clear();
    tf::unroll<0, 9, 3>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,3,6});
  }

  // ── step > 1, NOT divisible (ceil semantics) ──────────────────────
  // count = ceil((end - beg) / step); last visited index can be < end - 1
  {
    std::vector<int> got;
    tf::unroll<0, 10, 3>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,3,6,9});         // 9 < 10

    got.clear();
    tf::unroll<0, 11, 3>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,3,6,9});         // 12 would overshoot

    got.clear();
    tf::unroll<0, 13, 3>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0,3,6,9,12});

    got.clear();
    tf::unroll<10, 20, 3>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{10,13,16,19});

    got.clear();
    tf::unroll<10, 20, 2>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{10,12,14,16,18});
  }

  // ── step >= range ─────────────────────────────────────────────────
  {
    std::vector<int> got;
    tf::unroll<0, 5, 5>  ([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0});

    got.clear();
    tf::unroll<0, 5, 100>([&](int i){ got.push_back(i); });
    REQUIRE(got == std::vector<int>{0});
  }

  // ── left-to-right execution order ─────────────────────────────────
  {
    std::vector<int> order;
    tf::unroll<0, 8, 1>([&](int i){ order.push_back(i); });
    for (int k = 0; k < (int)order.size(); ++k) REQUIRE(order[k] == k);

    order.clear();
    tf::unroll<3, 9, 2>([&](int i){ order.push_back(i * 10); });
    REQUIRE(order == std::vector<int>{30, 50, 70});
  }

  // ── value-category of the index passed to f ───────────────────────
  // f receives a prvalue, so both by-value and by-const-ref must work.
  {
    int s1 = 0, s2 = 0;
    tf::unroll<0, 5, 1>([&](int i)         { s1 += i; });
    tf::unroll<0, 5, 1>([&](const int& i)  { s2 += i; });
    REQUIRE(s1 == 10);
    REQUIRE(s2 == 10);
  }

  // ── return value of f is discarded ────────────────────────────────
  {
    int hits = 0;
    tf::unroll<0, 4, 1>([&](int i) -> int { ++hits; return i + 1; });
    REQUIRE(hits == 4);
  }

  // ── different index types ─────────────────────────────────────────
  {
    std::vector<std::size_t> got;
    tf::unroll<std::size_t{0}, std::size_t{4}, std::size_t{1}>(
      [&](std::size_t i){ got.push_back(i); }
    );
    REQUIRE(got == std::vector<std::size_t>{0,1,2,3});
  }

  // ── nesting (the common HPC register-tile pattern) ────────────────
  {
    std::vector<std::pair<int,int>> got;
    tf::unroll<0, 3, 1>([&](int i){
      tf::unroll<0, 2, 1>([&](int j){
        got.emplace_back(i, j);
      });
    });
    REQUIRE(got == std::vector<std::pair<int,int>>{
      {0,0},{0,1},{1,0},{1,1},{2,0},{2,1}
    });
  }

  // ── constexpr / compile-time evaluation ───────────────────────────
  {
    constexpr auto sum_0_to_9 = []{
      int s = 0;
      tf::unroll<0, 10, 1>([&](int i){ s += i; });
      return s;
    }();
    static_assert(sum_0_to_9 == 45);
    REQUIRE(sum_0_to_9 == 45);

    constexpr auto count_step3 = []{
      int n = 0;
      tf::unroll<0, 13, 3>([&](int){ ++n; });
      return n;
    }();
    static_assert(count_step3 == 5);  // 0,3,6,9,12
  }

  // ── original cumulative baseline (preserved verbatim) ─────────────
  {
    int count = 0;
    tf::unroll<0,  0, 1>([&](int){ count++; }); REQUIRE(count == 0);
    tf::unroll<0,  1, 1>([&](int){ count++; }); REQUIRE(count == 1);
    tf::unroll<0,  3, 1>([&](int){ count++; }); REQUIRE(count == 4);
    tf::unroll<10,20, 1>([&](int){ count++; }); REQUIRE(count == 14);
    tf::unroll<10,20, 2>([&](int){ count++; }); REQUIRE(count == 19);
  }
}

TEST_CASE("UnrollUntil" * doctest::timeout(300)) {

  // ---- empty range: f never called, fold-over-empty-|| == false
  {
    int count = 0;
    bool r1 = tf::unroll_until<0, 0, 1>([&](int){ count++; return true;  });
    bool r2 = tf::unroll_until<5, 5, 1>([&](int){ count++; return true;  });
    bool r3 = tf::unroll_until<5, 5, 3>([&](int){ count++; return false; });
    bool r4 = tf::unroll_until<7, 7, 9>([&](int){ count++; return true;  });
    REQUIRE(count == 0);
    REQUIRE(r1 == false);
    REQUIRE(r2 == false);
    REQUIRE(r3 == false);
    REQUIRE(r4 == false);
  }

  // ---- single iteration, predicate false → visited once, returns false
  {
    int count = 0, seen = -1;
    bool r = tf::unroll_until<7, 8, 1>(
      [&](int i){ count++; seen = i; return false; }
    );
    REQUIRE(count == 1);
    REQUIRE(seen  == 7);
    REQUIRE(r == false);
  }

  // ---- single iteration, predicate true → visited once, returns true
  {
    int count = 0, seen = -1;
    bool r = tf::unroll_until<7, 8, 1>(
      [&](int i){ count++; seen = i; return true; }
    );
    REQUIRE(count == 1);
    REQUIRE(seen  == 7);
    REQUIRE(r == true);
  }

  // ---- predicate always false → all visited, returns false
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 5, 1>(
      [&](int i){ got.push_back(i); return false; }
    );
    REQUIRE(got == std::vector<int>{0,1,2,3,4});
    REQUIRE(r == false);
  }

  // ---- short-circuit on the first iteration → only beg visited
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 5, 1>(
      [&](int i){ got.push_back(i); return i == 0; }
    );
    REQUIRE(got == std::vector<int>{0});
    REQUIRE(r == true);
  }

  // ---- short-circuit in the middle → indices up to and including trigger
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 6, 1>(
      [&](int i){ got.push_back(i); return i == 3; }
    );
    REQUIRE(got == std::vector<int>{0,1,2,3});
    REQUIRE(r == true);
  }

  // ---- short-circuit on the last iteration → all visited, returns true
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 4, 1>(
      [&](int i){ got.push_back(i); return i == 3; }
    );
    REQUIRE(got == std::vector<int>{0,1,2,3});
    REQUIRE(r == true);
  }

  // ---- step > 1, short-circuit honors step
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 10, 2>(
      [&](int i){ got.push_back(i); return i == 4; }
    );
    REQUIRE(got == std::vector<int>{0, 2, 4});
    REQUIRE(r == true);
  }

  // ---- step > 1, predicate never true → ceil-many visits, returns false
  {
    std::vector<int> got;
    bool r = tf::unroll_until<0, 10, 3>(
      [&](int i){ got.push_back(i); return false; }
    );
    REQUIRE(got == std::vector<int>{0, 3, 6, 9});
    REQUIRE(r == false);
  }

  // ---- nonzero start + step, short-circuit
  {
    std::vector<int> got;
    bool r = tf::unroll_until<10, 20, 2>(
      [&](int i){ got.push_back(i); return i == 16; }
    );
    REQUIRE(got == std::vector<int>{10, 12, 14, 16});
    REQUIRE(r == true);
  }

  // ---- sanity: nothing past the trigger is ever touched
  {
    int after_trigger = 0;
    bool r = tf::unroll_until<0, 8, 1>([&](int i){
      if (i > 2) ++after_trigger;
      return i == 2;
    });
    REQUIRE(after_trigger == 0);
    REQUIRE(r == true);
  }

  // ---- f can take i by value or by const-ref
  {
    int last_v = -1, last_r = -1;
    bool rv = tf::unroll_until<0, 5, 1>([&](int i)        { last_v = i; return false; });
    bool rr = tf::unroll_until<0, 5, 1>([&](const int& i) { last_r = i; return false; });
    REQUIRE(last_v == 4);
    REQUIRE(last_r == 4);
    REQUIRE(rv == false);
    REQUIRE(rr == false);
  }

  // ---- non-bool but bool-convertible return (contextual conversion via ||)
  // int return: 0 → false-y, nonzero → true-y, so triggers on i == 1
  {
    int hits = 0;
    bool r = tf::unroll_until<0, 5, 1>([&](int i){ ++hits; return i; });
    REQUIRE(hits == 2);   // visited 0, then 1 (stopped)
    REQUIRE(r == true);
  }

  // ---- size_t bounds
  {
    std::vector<std::size_t> got;
    bool r = tf::unroll_until<std::size_t{0}, std::size_t{5}, std::size_t{1}>(
      [&](std::size_t i){ got.push_back(i); return i == 2; }
    );
    REQUIRE(got == std::vector<std::size_t>{0, 1, 2});
    REQUIRE(r == true);
  }

  // ---- nesting: outer stops as soon as inner finds a match
  {
    std::vector<std::pair<int,int>> got;
    bool r = tf::unroll_until<0, 4, 1>([&](int i){
      return tf::unroll_until<0, 3, 1>([&](int j){
        got.emplace_back(i, j);
        return (i == 1 && j == 2);
      });
    });
    REQUIRE(r == true);
    REQUIRE(got == std::vector<std::pair<int,int>>{
      {0,0},{0,1},{0,2},   // i=0: inner runs to completion, returns false
      {1,0},{1,1},{1,2}    // i=1: inner matches at j=2; outer stops here
    });
  }

  // ---- constexpr / compile-time evaluation
  {
    constexpr bool found_3 = tf::unroll_until<0, 5, 1>(
      [](int i){ return i == 3; }
    );
    static_assert(found_3 == true);

    constexpr bool found_99 = tf::unroll_until<0, 5, 1>(
      [](int i){ return i == 99; }
    );
    static_assert(found_99 == false);

    // confirm short-circuit fires at compile time
    constexpr int visits_until_2 = []{
      int n = 0;
      tf::unroll_until<0, 5, 1>([&](int i){ ++n; return i == 2; });
      return n;
    }();
    static_assert(visits_until_2 == 3);   // visits 0, 1, 2
    REQUIRE(visits_until_2 == 3);
  }
}








