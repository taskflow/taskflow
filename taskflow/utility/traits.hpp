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

// ----------------------------------------------------------------------------
// Callable
// ----------------------------------------------------------------------------

struct AnyArg {
  // Keep it simple: just one template conversion that can bind to anything.
  // We use a pointer trick to avoid needing a static dummy object.
  template <typename T>
  operator T&() const;
    
  template <typename T>
  operator T&&() const;
};

// Helper to provide N instances of AnyArg to a call check
template <typename F, typename Indices>
struct is_nary_invocable;

template <typename F, std::size_t... I>
struct is_nary_invocable<F, std::index_sequence<I...>> {
  static constexpr bool value = requires(F&& f) {
    std::forward<F>(f)( ((void)I, std::declval<AnyArg>())... );
  };
};

/**
@brief concept to check if a type is callable with `N` arguments of any types

@tparam F The function-like type to be tested.
@tparam N The required number of arguments (arity).

This concept validates whether a function-like object (lambda, functor, or function pointer)
can be invoked with a specific number of arguments. It uses an internal @ref AnyArg
helper to simulate arguments of any type, making it useful for generic API validation.

@code{.cppp}
auto f = [](int x, int y) { return x + y; };
static_assert(NaryOperatorLike<decltype(f), 2>); // Passes
static_assert(!NaryOperatorLike<decltype(f), 1>); // Fails: requires 2 args

auto g = [](auto... args) { return sizeof...(args); };
static_assert(NaryOperatorLike<decltype(g), 0>); // Passes
static_assert(NaryOperatorLike<decltype(g), 5>); // Passes
static_assert(NaryOperatorLike<decltype(g), 100>); // Passes

// When testing member functions via std::invoke, the first argument
// must be the object instance (or pointer).
struct Math { void add(int a) {} };

// Passes because std::invoke(ptr_to_mem, instance, arg) is 2 arguments total
static_assert(NaryOperatorLike<decltype(&Math::add), 2>);
@endcode
*/
template <typename F, size_t N>
concept NaryOperatorLike = is_nary_invocable<F, std::make_index_sequence<N>>::value;

/**
@brief concept to check if a type is callable with one argument of any type
*/
template <typename F>
concept UnaryOperatorLike = NaryOperatorLike<F, 1>;

/**
@brief concept to check if a type is callable with two arguments of any type
*/
template <typename F>
concept BinaryOperatorLike = NaryOperatorLike<F, 2>;

/**
@brief concept to check if a type is callable with three arguments of any type
*/
template <typename F>
concept TernaryOperatorLike = NaryOperatorLike<F, 3>;

}  // end of namespace tf. ----------------------------------------------------








