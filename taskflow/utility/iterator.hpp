#pragma once

#include <cstddef>
#include <type_traits>

namespace tf {

template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral<std::decay_t<B>>::value && 
                           std::is_integral<std::decay_t<E>>::value && 
                           std::is_integral<std::decay_t<S>>::value, bool>
is_range_invalid(B beg, E end, S step) {
  return ((step == 0 && beg != end) ||
          (beg < end && step <=  0) ||  // positive range
          (beg > end && step >=  0));   // negative range
}

template <typename B, typename E, typename S>
constexpr std::enable_if_t<std::is_integral<std::decay_t<B>>::value && 
                           std::is_integral<std::decay_t<E>>::value && 
                           std::is_integral<std::decay_t<S>>::value, size_t>
distance(B beg, E end, S step) {
  return (end - beg + step + (step > 0 ? -1 : 1)) / step;
}

}  // end of namespace tf -----------------------------------------------------
