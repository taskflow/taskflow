#pragma once

#include <version>
#include <concepts>
#include <latch>
#include <type_traits>
#include <iterator>
#include <iostream>
#include <fstream>
#include <mutex>
#include <stack>
#include <queue>
#include <vector>
#include <algorithm>
#include <memory>
#include <atomic>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <numeric>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <array>
#include <string>
#include <variant>
#include <optional>
#include "os.hpp"

namespace tf {

//-----------------------------------------------------------------------------
// Traits
//-----------------------------------------------------------------------------

//// Struct: dependent_false
//template <typename... T>
//struct dependent_false {
//  static constexpr bool value = false;
//};
//
//template <typename... T>
//constexpr auto dependent_false_v = dependent_false<T...>::value;

template<typename> inline constexpr bool dependent_false_v = false;

// ----------------------------------------------------------------------------
// is_pod
//-----------------------------------------------------------------------------
template <typename T>
struct is_pod {
  static const bool value = std::is_trivial_v<T> && 
                            std::is_standard_layout_v<T>;
};

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

//-----------------------------------------------------------------------------
// NoInit
//-----------------------------------------------------------------------------

template <typename T>
struct NoInit {

  //static_assert(is_pod_v<T>, "NoInit only supports POD type");

  // constructor without initialization
  NoInit () noexcept {}

  // implicit conversion T -> NoInit<T>
  constexpr  NoInit (T value) noexcept : v{value} {}

  // implicit conversion NoInit<T> -> T
  constexpr  operator T () const noexcept { return v; }

  T v;
};

//-----------------------------------------------------------------------------
// Move-On-Copy
//-----------------------------------------------------------------------------

// Struct: MoveOnCopyWrapper
template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }

  mutable T object;
};

template <typename T>
auto make_moc(T&& m) {
  return MoC<T>(std::forward<T>(m));
}

//-----------------------------------------------------------------------------
// Visitors.
//-----------------------------------------------------------------------------

//// Overloadded.
//template <typename... Ts>
//struct Visitors : Ts... {
//  using Ts::operator()... ;
//};
//
//template <typename... Ts>
//Visitors(Ts...) -> Visitors<Ts...>;

// ----------------------------------------------------------------------------
// std::variant
// ----------------------------------------------------------------------------
template <typename T, typename>
struct get_index;

template <size_t I, typename... Ts>
struct get_index_impl {};

template <size_t I, typename T, typename... Ts>
struct get_index_impl<I, T, T, Ts...> : std::integral_constant<size_t, I>{};

template <size_t I, typename T, typename U, typename... Ts>
struct get_index_impl<I, T, U, Ts...> : get_index_impl<I+1, T, Ts...>{};

template <typename T, typename... Ts>
struct get_index<T, std::variant<Ts...>> : get_index_impl<0, T, Ts...>{};

template <typename T, typename... Ts>
constexpr auto get_index_v = get_index<T, Ts...>::value;

// ----------------------------------------------------------------------------
// stateful iterators
// ----------------------------------------------------------------------------

// STL-styled iterator
template <typename B, typename E>
struct stateful_iterator {

  using TB = std::decay_t<std::unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<std::unwrap_ref_decay_t<E>>;

  static_assert(std::is_same_v<TB, TE>, "decayed iterator types must match");

  using type = TB;
};

template <typename B, typename E>
using stateful_iterator_t = typename stateful_iterator<B, E>::type;

// raw integral index
template <typename B, typename E, typename S>
struct stateful_index {

  using TB = std::decay_t<std::unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<std::unwrap_ref_decay_t<E>>;
  using TS = std::decay_t<std::unwrap_ref_decay_t<S>>;

  static_assert(
    std::is_integral_v<TB>, "decayed beg index must be an integral type"
  );

  static_assert(
    std::is_integral_v<TE>, "decayed end index must be an integral type"
  );

  static_assert(
    std::is_integral_v<TS>, "decayed step must be an integral type"
  );

  static_assert(
    std::is_same_v<TB, TE> && std::is_same_v<TE, TS>,
    "decayed index and step types must match"
  );

  using type = TB;
};

template <typename B, typename E, typename S>
using stateful_index_t = typename stateful_index<B, E, S>::type;

// ----------------------------------------------------------------------------
// visit a tuple with a functor at runtime
// ----------------------------------------------------------------------------

template <typename Func, typename Tuple>
void visit_tuple(Func&& func, Tuple& tup, size_t idx) {
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    ([&]{ if(Is == idx) { std::invoke(func, std::get<Is>(tup)); } }(), ...);
  }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// ----------------------------------------------------------------------------
// unroll loop
// ----------------------------------------------------------------------------

template<auto beg, auto end, auto step, typename F>
constexpr void unroll(F&& f) {
  [&]<auto... Is>(std::index_sequence<Is...>) {
    (f(beg + Is * step), ...);
  }(std::make_index_sequence<(end - beg + step - 1) / step>{});
}

// ----------------------------------------------------------------------------
// make types of variant unique
// ----------------------------------------------------------------------------

template <typename T, typename... Ts>
struct filter_duplicates { using type = T; };

template <template <typename...> class C, typename... Ts, typename U, typename... Us>
struct filter_duplicates<C<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
                       , filter_duplicates<C<Ts...>, Us...>
                       , filter_duplicates<C<Ts..., U>, Us...>> {};

template <typename T>
struct unique_variant;

template <typename... Ts>
struct unique_variant<std::variant<Ts...>> : filter_duplicates<std::variant<>, Ts...> {};

template <typename T>
using unique_variant_t = typename unique_variant<T>::type;


// ----------------------------------------------------------------------------
// check if it is default compare
// ----------------------------------------------------------------------------
template <typename T> struct is_std_compare : std::false_type { };
template <typename T> struct is_std_compare<std::less<T>> : std::true_type { };
template <typename T> struct is_std_compare<std::greater<T>> : std::true_type { };

template <typename T>
constexpr bool is_std_compare_v = is_std_compare<T>::value;

// ----------------------------------------------------------------------------
// check if all types are the same
// ----------------------------------------------------------------------------

template <typename T, typename... Ts>
concept all_same = (std::same_as<T, Ts> && ...);

// backward-compatible variable template
template <typename T, typename... Ts>
constexpr bool all_same_v = all_same<T, Ts...>;

// ----------------------------------------------------------------------------
// Iterator
// ----------------------------------------------------------------------------

// use std::iter_value_t instead of the custom deref_t
template <typename I>
using deref_t = std::iter_value_t<I>;

// use std::random_access_iterator concept instead of the custom variable
template <typename I>
constexpr bool is_random_access_iterator = std::random_access_iterator<I>;

}  // end of namespace tf. ----------------------------------------------------
