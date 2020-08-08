#pragma once

#include <type_traits>

namespace tf {

template <typename T>
constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, size_t>
distance(T beg, T end, T step) {
  return (end - beg + step + (step > 0 ? -1 : 1)) / step;
}



}  // end of namespace tf -----------------------------------------------------
