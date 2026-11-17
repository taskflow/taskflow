#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <numeric>

namespace tf {

/**
@brief rounds the given 64-bit unsigned integer to the nearest power of 2

@tparam T 64-bit unsigned integral type
@param x the number to round up
@return the smallest power of 2 that is greater than or equal to @c x

This overload participates in overload resolution only when @c T is an
8-byte unsigned integral type. It repeatedly fills in the lower bits of
<tt>x - 1</tt> until all bits below the highest set bit are 1, then adds 1
to obtain the next power of 2.

@code{.cpp}
tf::next_pow2(uint64_t{17});  // returns 32
tf::next_pow2(uint64_t{32});  // returns 32
@endcode
*/
template <typename T>
requires (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8)
constexpr T next_pow2(T x) {
  if(x == 0) return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

/**
@brief rounds the given 32-bit unsigned integer to the nearest power of 2

@tparam T 32-bit unsigned integral type
@param y the number to round up
@return the smallest power of 2 that is greater than or equal to @c y

This overload participates in overload resolution only when @c T is a
4-byte unsigned integral type. It uses the same bit-filling technique as
the 64-bit overload, but only propagates bits up to the 32-bit width.

@code{.cpp}
tf::next_pow2(uint32_t{17});  // returns 32
tf::next_pow2(uint32_t{32});  // returns 32
@endcode
*/
template <typename T>
requires (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 4)
constexpr T next_pow2(T y) {
  if(y == 0) return 1;
  y--;
  y |= y >> 1;
  y |= y >> 2;
  y |= y >> 4;
  y |= y >> 8;
  y |= y >> 16;
  y++;
  return y;
}

/**
@brief checks if the given number is a power of 2

@tparam T integral type of the input
@param x The integer to check.
@return `true` if `x` is a power of 2, otherwise `false`.

This function determines if the given integer is a power of 2 by testing
that exactly one bit is set, i.e., <tt>x & (x - 1) == 0</tt>, while also
excluding zero.

@code{.cpp}
tf::is_pow2(8);   // true
tf::is_pow2(10);  // false
@endcode

@note This function is constexpr and can be evaluated at compile time.
*/
template <std::integral T>
constexpr bool is_pow2(const T& x) {
  return x && (!(x&(x-1)));
}

/**
@brief returns the floor of `log2(N)` at compile time

@tparam N the input value
@return the largest integer `k` such that `2^k <= N`

This function recursively halves @c N until it is smaller than 2,
counting the number of halving steps performed, which equals the floor
of the base-2 logarithm of @c N.

@code{.cpp}
tf::static_floor_log2<16>();  // returns 4
tf::static_floor_log2<17>();  // returns 4
@endcode
*/
template <size_t N>
constexpr size_t static_floor_log2() {
  return (N < 2) ? 0 : 1 + static_floor_log2<N / 2>();
  //auto log = 0;
  //while (N >>= 1) {
  //  ++log;
  //}
  //return log;
}


/**
@brief finds the median of three numbers pointed to by iterators using the given comparator

@tparam RandItr The type of the random-access iterator.
@tparam C The type of the comparator.
@param l Iterator to the first element.
@param m Iterator to the second element.
@param r Iterator to the third element.
@param cmp The comparator used to compare the dereferenced iterator values.
@return The iterator pointing to the median value among the three elements.

This function determines the median value of the elements pointed to by
three random-access iterators using the provided comparator.

@code{.cpp}
std::vector<int> v = {5, 1, 3};
auto it = tf::median_of_three(v.begin(), v.begin()+1, v.begin()+2, std::less<int>{});
// *it == 3
@endcode
*/
template <typename RandItr, typename C>
RandItr median_of_three(RandItr l, RandItr m, RandItr r, C cmp) {
  return cmp(*l, *m) ? (cmp(*m, *r) ? m : (cmp(*l, *r) ? r : l ))
                     : (cmp(*r, *m) ? m : (cmp(*r, *l) ? r : l ));
}

/**
@brief finds the pseudo median of a range of items using a spread of nine numbers

@tparam RandItr The type of the random-access iterator.
@tparam C The type of the comparator.
@param beg Iterator to the beginning of the range.
@param end Iterator to the end of the range.
@param cmp The comparator used to compare the dereferenced iterator values.
@return The iterator pointing to the pseudo median of the range.

This function computes an approximate median of a range of items by sampling
nine values spread across the range and finding their median. It uses a
combination of the `median_of_three` function to determine the pseudo median.

@code{.cpp}
std::vector<int> v = {9, 4, 1, 7, 3, 8, 2, 6, 5};
auto it = tf::pseudo_median_of_nine(v.begin(), v.end(), std::less<int>{});
@endcode

@note The pseudo median is an approximation of the true median and may not
      be the exact middle value of the range.
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
@brief sorts two elements of dereferenced iterators using the given comparison function

@tparam Iter The type of the iterator.
@tparam Compare The type of the comparator.
@param a Iterator to the first element.
@param b Iterator to the second element.
@param comp The comparator used to compare the dereferenced iterator values.

This function compares two elements pointed to by iterators and swaps them
if they are out of order according to the provided comparator.

@code{.cpp}
std::vector<int> v = {3, 1};
tf::sort2(v.begin(), v.begin()+1, std::less<int>{});
// v == {1, 3}
@endcode
*/
template<typename Iter, typename Compare>
void sort2(Iter a, Iter b, Compare comp) {
  if (comp(*b, *a)) std::iter_swap(a, b);
}

/**
@brief Sorts three elements of dereferenced iterators using the given comparison function.

@tparam Iter The type of the iterator.
@tparam Compare The type of the comparator.
@param a Iterator to the first element.
@param b Iterator to the second element.
@param c Iterator to the third element.
@param comp The comparator used to compare the dereferenced iterator values.

This function sorts three elements pointed to by iterators in ascending order
according to the provided comparator. The sorting is performed using a sequence
of calls to the `sort2` function to ensure the correct order of elements.

@code{.cpp}
std::vector<int> v = {3, 1, 2};
tf::sort3(v.begin(), v.begin()+1, v.begin()+2, std::less<int>{});
// v == {1, 2, 3}
@endcode
*/
template<typename Iter, typename Compare>
void sort3(Iter a, Iter b, Iter c, Compare comp) {
  sort2(a, b, comp);
  sort2(b, c, comp);
  sort2(a, b, comp);
}

/**
@brief generates a program-wide unique ID of the given type in a thread-safe manner

@tparam T integral type of the ID to generate
@return A unique ID of type `T`.

This function provides a globally unique identifier of the specified integral type.
It uses a static `std::atomic` counter to ensure thread safety and increments the
counter in a relaxed memory ordering for efficiency.

@code{.cpp}
size_t id1 = tf::unique_id<size_t>();
size_t id2 = tf::unique_id<size_t>();
// id1 != id2
@endcode

@note The uniqueness of the ID is guaranteed only within the program's lifetime.
@note The function does not throw exceptions.
*/
template <std::integral T>
T unique_id() {
  static std::atomic<T> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

/**
@brief updates an atomic variable with the maximum value

@tparam T The type of the atomic variable. Must be trivially copyable and comparable.
@param v The atomic variable to update.
@param max_v The value to compare with the current value of `v`.

This function atomically updates the provided atomic variable `v` to hold
the maximum of its current value and `max_v`. The update is performed using
a relaxed memory ordering for efficiency in non-synchronizing contexts.

@code{.cpp}
std::atomic<int> v{5};
tf::atomic_max(v, 10);
// v.load() == 10
@endcode

@note If multiple threads call this function concurrently, the value of `v`
      will be the maximum value seen across all threads.
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
@brief updates an atomic variable with the minimum value

@tparam T The type of the atomic variable. Must be trivially copyable and comparable.
@param v The atomic variable to update.
@param min_v The value to compare with the current value of `v`.

This function atomically updates the provided atomic variable `v` to hold
the minimum of its current value and `min_v`. The update is performed using
a relaxed memory ordering for efficiency in non-synchronizing contexts.

@code{.cpp}
std::atomic<int> v{5};
tf::atomic_min(v, 2);
// v.load() == 2
@endcode

@note If multiple threads call this function concurrently, the value of `v`
      will be the minimum value seen across all threads.
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
@brief generates a random seed based on the current system clock

@tparam T The type of the returned seed. Must be an integral type.
@return A seed value based on the system clock.

This function returns a seed value derived from the number of clock ticks
since the epoch as measured by the system clock. The seed can be used
to initialize random number generators.

@code{.cpp}
auto s = tf::seed<size_t>();
tf::Xorshift<uint64_t> rng(s);
@endcode
*/
template <typename T>
inline T seed() noexcept {
  return std::chrono::system_clock::now().time_since_epoch().count();
}

// ------------------------------------------------------------------------------------------------
// coprime
// ------------------------------------------------------------------------------------------------

/**
@brief computes a coprime of a given number

@param N input number for which a coprime is to be found.
@return the largest number < @c N that is coprime to N

This function finds the largest number less than N that is coprime (i.e., has a greatest common divisor of 1) with @c N.
If @c N is less than 3, it returns 1 as a default coprime.

@code{.cpp}
tf::coprime(10);  // returns 9
@endcode
*/
constexpr size_t coprime(size_t N) {
  if(N < 3) {
    return 1;
  }
  for (size_t x = N; --x > 0;) {
    if (std::gcd(x, N) == 1) {
      return x;
    }
  }
  return 1;
}

/**
@brief generates a compile-time array of coprimes for numbers from 0 to N-1

@tparam N the size of the array to generate (should be greater than 0).
@return a constexpr array of size @c N where each index holds a coprime of its value.

This function constructs a constexpr array where each element at index `i` contains a coprime of `i`
(the largest number less than `i` that is coprime to it).

@code{.cpp}
constexpr auto lut = tf::make_coprime_lut<8>();
// lut[5] holds a coprime of 5
@endcode
*/
template <size_t N>
constexpr std::array<size_t, N> make_coprime_lut() {
  static_assert(N>0, "N must be greater than 0");
  std::array<size_t, N> coprimes{};
  for (size_t n = 0; n < N; ++n) {
    coprimes[n] = coprime(n);
  }
  return coprimes;
}

//template <typename T>
//constexpr T lemire_range(T x, T range) {
//  return (uint32_t)(((uint64_t)x * (uint64_t)range) >> 32);
//}


/**
@class Xorshift

@brief class to create a fast xorshift-based pseudo-random number generator

@tparam T unsigned integral type used as the internal state (supported uint32_t and uint64_t)

This class implements a lightweight xorshift pseudo-random number generator
suitable for performance-critical paths such as schedulers, work-stealing victim selection,
and randomized backoff.
The implementation is branchless on the hot path and has a very small state
footprint (one machine word). All operations are integer-only.

@note
The internal state must be seeded with a non-zero value.
This class is not thread-safe. Each thread should maintain its own instance.
*/
template <typename T>
class Xorshift {
  static_assert(std::is_unsigned<T>::value, "Xorshift requires an unsigned integral type.");

  public:

  /**
  @brief constructs an uninitialized xor-shift generator

  The internal state is not initialized. The user must call `seed()`
  with a non-zero value before generating numbers.

  @code{.cpp}
  tf::Xorshift<uint64_t> rng;
  rng.seed(12345);
  @endcode
  */
  Xorshift() = default;

  /**
  @brief constructs a xor-shift generator with the given seed

  @param value the new seed value to use

  The seed value must be non-zero.

  @code{.cpp}
  tf::Xorshift<uint64_t> rng(12345);
  auto r = rng();
  @endcode
  */
  Xorshift(T value) : _state(value) {}

  /**
  @brief seeds the generator with a new value

  @param value the new seed value

  The seed value must be non-zero. A zero seed results in a degenerated
  generator that always returns zero.

  @code{.cpp}
  tf::Xorshift<uint64_t> rng;
  rng.seed(12345);
  @endcode
  */
  void seed(T value) {
    _state = value;
  }

  /**
  @brief generates the next pseudo-random value

  @return a pseudo-random value of type `T`

  For 32-bit state, this function implements the Xorshift32 algorithm.
  For 64-bit state, this function implements the Xorshift64 algorithm with
  a multiplicative output transformation to improve distribution.

  @code{.cpp}
  tf::Xorshift<uint64_t> rng(12345);
  uint64_t r1 = rng();
  uint64_t r2 = rng();
  @endcode

  @warning
  Calling this function before seeding the generator with a non-zero value
  results in undefined behavior.
  */
  T operator()() {
    if constexpr (sizeof(T) == 8) {
      // Xorshift64 constants
      _state ^= _state << 13;
      _state ^= _state >> 7;
      _state ^= _state << 17;
      return _state * 0x2545F4914F6CDD1DULL;
    }
    else if constexpr (sizeof(T) == 4) {
      // Xorshift32 constants
      _state ^= _state << 13;
      _state ^= _state >> 17;
      _state ^= _state << 5;
      return _state;
    }
    else {
      static_assert(sizeof(T) == 0, "Unsupported bit-width for Xorshift. Use uint32_t or uint64_t.");
    }
  }

  private:

  T _state;  // must be initialized with non-zero
};

}  // end of namespace tf -----------------------------------------------------
