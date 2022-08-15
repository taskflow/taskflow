#pragma once

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
// unwrap_reference
// ----------------------------------------------------------------------------

template <class T>
struct unwrap_reference { using type = T; };

template <class U>
struct unwrap_reference<std::reference_wrapper<U>> { using type = U&; };

template<class T>
using unwrap_reference_t = typename unwrap_reference<T>::type;

template< class T >
struct unwrap_ref_decay : unwrap_reference<std::decay_t<T>> {};

template<class T>
using unwrap_ref_decay_t = typename unwrap_ref_decay<T>::type;

// ----------------------------------------------------------------------------
// stateful iterators
// ----------------------------------------------------------------------------

// STL-styled iterator
template <typename B, typename E>
struct stateful_iterator {

  using TB = std::decay_t<unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<unwrap_ref_decay_t<E>>;

  static_assert(std::is_same_v<TB, TE>, "decayed iterator types must match");

  using type = TB;
};

template <typename B, typename E>
using stateful_iterator_t = typename stateful_iterator<B, E>::type;

// raw integral index
template <typename B, typename E, typename S>
struct stateful_index {

  using TB = std::decay_t<unwrap_ref_decay_t<B>>;
  using TE = std::decay_t<unwrap_ref_decay_t<E>>;
  using TS = std::decay_t<unwrap_ref_decay_t<S>>;

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

template <typename Func, typename Tuple, size_t N = 0>
void visit_tuple(Func func, Tuple& tup, size_t idx) {
  if (N == idx) {
    std::invoke(func, std::get<N>(tup));
    return;
  }
  if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
    return visit_tuple<Func, Tuple, N + 1>(func, tup, idx);
  }
}

// ----------------------------------------------------------------------------
// unroll loop
// ----------------------------------------------------------------------------

// Template unrolled looping construct.
template<auto beg, auto end, auto step, bool valid = (beg < end)>
struct Unroll {
  template<typename F>
  static void eval(F f) {
    f(beg);
    Unroll<beg + step, end, step>::eval(f);
  }
};

template<auto beg, auto end, auto step>
struct Unroll<beg, end, step, false> {
  template<typename F>
  static void eval(F) { }
};

template<auto beg, auto end, auto step, typename F>
void unroll(F f) {
  Unroll<beg, end, step>::eval(f);
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





}  // end of namespace tf. ----------------------------------------------------



