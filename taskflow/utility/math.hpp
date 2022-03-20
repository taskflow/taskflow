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
@brief finds the pseudo median of a range of items using spreaded
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

}  // end of namespace tf -----------------------------------------------------



