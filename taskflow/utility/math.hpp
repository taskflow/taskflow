#pragma once

namespace tf {

// rounds the given 64-bit unsigned integer to the nearest power of 2
template <typename T, std::enable_if_t<
  (std::is_unsigned_v<std::decay_t<T>> && sizeof(T) == 8) , void
>* = nullptr>
constexpr T next_pow2(T x) {
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

}  // end of namespace tf -----------------------------------------------------



