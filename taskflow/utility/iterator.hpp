#pragma once

#include <type_traits>

namespace tf {

template <typename T>
constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, size_t>
distance(T beg, T end, T step) {
  return (end - beg + step + (step > 0 ? -1 : 1)) / step;
}

template <typename T>
constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, bool>
is_range_invalid(T beg, T end, T step) {
  return ((step == 0 && beg != end) || 
          (beg < end && step <=  0) || 
          (beg > end && step >=  0));
}

}  // end of namespace tf -----------------------------------------------------
