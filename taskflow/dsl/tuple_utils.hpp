// 2020/08/28 - Created by netcan: https://github.com/netcan
#pragma once
#include <cstddef>
#include <tuple>

namespace tf {
namespace dsl {
namespace detail {
// get tuple element index by f, if not exists then index >= tuple_size
template <typename TUP, template <typename> class F, typename = void>
struct TupleElementByF {
  constexpr static size_t Index = 0;
};

template <template <typename> class F, typename H, typename... Ts>
struct TupleElementByF<std::tuple<H, Ts...>, F, std::enable_if_t<F<H>::value>> {
  constexpr static size_t Index = 0;
};

template <template <typename> class F, typename H, typename... Ts>
struct TupleElementByF<std::tuple<H, Ts...>, F,
                       std::enable_if_t<!F<H>::value>> {
  constexpr static size_t Index =
      1 + TupleElementByF<std::tuple<Ts...>, F>::Index;
};

template <typename T, typename TUP, size_t... Is>
constexpr inline T AggregationByTupImpl(TUP &&tup, std::index_sequence<Is...>) {
  return T{std::get<Is>(tup)...};
}
} // namespace detail

template <typename TUP, template <typename> class F>
constexpr size_t TupleElementByF_v = detail::TupleElementByF<TUP, F>::Index;

template <typename T, typename TUP>
constexpr inline T AggregationByTup(TUP &&tup) {
  return detail::AggregationByTupImpl<T>(
      std::forward<TUP>(tup),
      std::make_index_sequence<std::tuple_size<std::decay_t<TUP>>::size>{});
}
} // namespace dsl
} // namespace tf
