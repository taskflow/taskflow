#pragma once

#include <atomic>

namespace tf {

// rounds the given 64-bit unsigned integer to the nearest power of 2
template <typename T, std::enable_if_t<
  (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8) , void
>* = nullptr>
constexpr T next_pow2(T x) {
  if(x == 0) return 1;
  x--;
  x |= x>>1;
	x |= x>>2;
	x |= x>>4;
	x |= x>>8;
	x |= x>>16;
	x |= x>>32;
  x++;
  return x;
}

// rounds the given 32-bit unsigned integer to the nearest power of 2
template <typename T, std::enable_if_t<
  (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 4), void
>* = nullptr>
constexpr T next_pow2(T x) {
  if(x == 0) return 1;
  x--;
  x |= x>>1;
	x |= x>>2;
	x |= x>>4;
	x |= x>>8;
	x |= x>>16;
  x++;
  return x;
}

// checks if the given number if a power of 2
template <typename T, std::enable_if_t<
  std::is_integral_v<std::decay_t<T>>, void>* = nullptr
>
constexpr bool is_pow2(const T& x) {
  return x && (!(x&(x-1)));
}

//// finds the ceil of x divided by b
//template <typename T, std::enable_if_t<
//  std::is_integral_v<std::decay_t<T>>, void>* = nullptr
//>
//constexpr T ceil(const T& x, const T& y) {
//  //return (x + y - 1) / y;
//  return (x-1) / y + 1;
//}

/**
@brief returns floor(log2(n)), assumes n > 0
*/
template<typename T>
constexpr int log2(T n) {
  int log = 0;
  while (n >>= 1) {
    ++log;
  }
  return log;
}

/**
@brief finds the median of three numbers of dereferenced iterators using
       the given comparator
*/
template <typename RandItr, typename C>
RandItr median_of_three(RandItr l, RandItr m, RandItr r, C cmp) {
  return cmp(*l, *m) ? (cmp(*m, *r) ? m : (cmp(*l, *r) ? r : l ))
                     : (cmp(*r, *m) ? m : (cmp(*r, *l) ? r : l ));
}

/**
@brief finds the pseudo median of a range of items using a spread of
       nine numbers
 */
template <typename RandItr, typename C>
RandItr pseudo_median_of_nine(RandItr beg, RandItr end, C cmp) {
  size_t N = std::distance(beg, end);
  size_t offset = N >> 3;
  return median_of_three(
    median_of_three(beg, beg+offset, beg+(offset*2), cmp),
    median_of_three(beg+(offset*3), beg+(offset*4), beg+(offset*5), cmp),
    median_of_three(beg+(offset*6), beg+(offset*7), end-1, cmp),
    cmp
  );
}

/**
@brief sorts two elements of dereferenced iterators using the given
       comparison function
*/
template<typename Iter, typename Compare>
void sort2(Iter a, Iter b, Compare comp) {
  if (comp(*b, *a)) std::iter_swap(a, b);
}

/**
@brief sorts three elements of dereferenced iterators using the given
       comparison function
*/
template<typename Iter, typename Compare>
void sort3(Iter a, Iter b, Iter c, Compare comp) {
  sort2(a, b, comp);
  sort2(b, c, comp);
  sort2(a, b, comp);
}

/**
@brief generates a program-wise unique id of the give type (thread-safe)
*/
template <typename T, std::enable_if_t<std::is_integral_v<T>, void>* = nullptr>
T unique_id() {
  static std::atomic<T> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

/**
@brief updates an atomic variable with a maximum value
*/
template <typename T>
inline void atomic_max(std::atomic<T>& v, const T& max_v) noexcept {
  T prev = v.load(std::memory_order_relaxed);
  while(prev < max_v && 
        !v.compare_exchange_weak(prev, max_v, std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
  }
}

/**
@brief updates an atomic variable with a minimum value
*/
template <typename T>
inline void atomic_min(std::atomic<T>& v, const T& min_v) noexcept {
  T prev = v.load(std::memory_order_relaxed);
  while(prev > min_v && 
        !v.compare_exchange_weak(prev, min_v, std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
  }
}

/**
@brief get a random seed
*/
template <typename T>
inline T seed() noexcept {
  return std::chrono::system_clock::now().time_since_epoch().count();
}

//class XorShift64 {
//
//  public:
//  
//  explicit XorShift64(uint64_t seed) : _state(seed) {}
//
//  uint64_t next() {
//    _state ^= _state >> 12;
//    _state ^= _state << 25;
//    _state ^= _state >> 27;
//    return _state * 0x2545F4914F6CDD1DULL; // Scramble for better randomness
//  }
//
//  size_t random_range(size_t min, size_t max) {
//    return min + (next() % (max - min + 1));
//  }
//
//  private:
//
//  uint64_t _state;
//};

//inline int generate_random_excluding(int worker_id, int W, XorShift64& rng) {
//    int random_number = rng.random_range(0, 2 * W - 2); // Range: [0, 2W-2]
//    return random_number + (random_number >= worker_id); // Skip worker_id
//}
//
//
//class Xoroshiro128Plus {
//
//  public:
//
//    explicit Xoroshiro128Plus(uint64_t seed1, uint64_t seed2) : _state{seed1, seed2} {}
//
//    uint64_t next() {
//      uint64_t s0 = _state[0];
//      uint64_t s1 = _state[1];
//      uint64_t result = s0 + s1;
//
//      s1 ^= s0;
//      _state[0] = _rotl(s0, 55) ^ s1 ^ (s1 << 14); // Scramble _state
//      _state[1] = _rotl(s1, 36);
//
//      return result;
//    }
//
//    int random_range(int min, int max) {
//      return min + (next() % (max - min + 1));
//    }
//
//  private:
//
//    std::array<uint64_t, 2> _state;
//
//    static uint64_t _rotl(uint64_t x, int k) {
//      return (x << k) | (x >> (64 - k));
//    }
//};


}  // end of namespace tf -----------------------------------------------------



