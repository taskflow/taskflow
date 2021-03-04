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
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <array>
#include <cstring>
#include <variant>
#include <optional>
#include <any>

namespace tf {

//-----------------------------------------------------------------------------
// Traits
//-----------------------------------------------------------------------------

// Struct: dependent_false
template <typename... T>
struct dependent_false { 
  static constexpr bool value = false; 
};

template <typename... T>
constexpr auto dependent_false_v = dependent_false<T...>::value;

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

// Overloadded.
template <typename... Ts>
struct Visitors : Ts... { 
  using Ts::operator()... ;
};

template <typename... Ts>
Visitors(Ts...) -> Visitors<Ts...>;

// ----------------------------------------------------------------------------
// Function Traits
// reference: https://github.com/ros2/rclcpp
// ----------------------------------------------------------------------------

template<typename T>
struct tuple_tail;

template<typename Head, typename ... Tail>
struct tuple_tail<std::tuple<Head, Tail ...>> {
  using type = std::tuple<Tail ...>;
};

// std::function
template<typename F>
struct function_traits
{
  using arguments = typename tuple_tail<
    typename function_traits<decltype(&F::operator())>::argument_tuple_type
  >::type;

  static constexpr size_t arity = std::tuple_size_v<arguments>;

  template <size_t N>
  struct argument {
    static_assert(N < arity, "error: invalid parameter index.");
    using type = std::tuple_element_t<N, arguments>;
  };
  
  template <size_t N>
  using argument_t = typename argument<N>::type;

  using return_type = typename function_traits<decltype(&F::operator())>::return_type;
};

// Free functions
template<typename R, typename... Args>
struct function_traits<R(Args...)> {

  using return_type = R;
  using argument_tuple_type = std::tuple<Args...>;
 
  static constexpr size_t arity = sizeof...(Args);
 
  template <size_t N>
  struct argument {
    static_assert(N < arity, "error: invalid parameter index.");
    using type = std::tuple_element_t<N, std::tuple<Args...>>;
  };

  template <size_t N>
  using argument_t = typename argument<N>::type;
};

// function pointer
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> {
};

// function reference
template<typename R, typename... Args>
struct function_traits<R(&)(Args...)> : function_traits<R(Args...)> {
};

// immutable lambda
template<typename C, typename R, typename ... Args>
struct function_traits<R(C::*)(Args ...) const>
  : function_traits<R(C &, Args ...)>
{};

// mutable lambda
template<typename C, typename R, typename ... Args>
struct function_traits<R(C::*)(Args ...)>
  : function_traits<R(C &, Args ...)>
{};

/*// std::bind for object methods
template<typename C, typename R, typename ... Args, typename ... FArgs>
#if defined _LIBCPP_VERSION  // libc++ (Clang)
struct function_traits<std::__bind<R (C::*)(Args ...), FArgs ...>>
#elif defined _GLIBCXX_RELEASE  // glibc++ (GNU C++ >= 7.1)
struct function_traits<std::_Bind<R(C::*(FArgs ...))(Args ...)>>
#elif defined __GLIBCXX__  // glibc++ (GNU C++)
struct function_traits<std::_Bind<std::_Mem_fn<R (C::*)(Args ...)>(FArgs ...)>>
#elif defined _MSC_VER  // MS Visual Studio
struct function_traits<
  std::_Binder<std::_Unforced, R (C::*)(Args ...), FArgs ...>>
#else
#error "Unsupported C++ compiler / standard library"
#endif
  : function_traits<R(Args ...)>
{};

// std::bind for object const methods
template<typename C, typename R, typename ... Args, typename ... FArgs>
#if defined _LIBCPP_VERSION  // libc++ (Clang)
struct function_traits<std::__bind<R (C::*)(Args ...) const, FArgs ...>>
#elif defined _GLIBCXX_RELEASE  // glibc++ (GNU C++ >= 7.1)
struct function_traits<std::_Bind<R(C::*(FArgs ...))(Args ...) const>>
#elif defined __GLIBCXX__  // glibc++ (GNU C++)
struct function_traits<std::_Bind<std::_Mem_fn<R (C::*)(Args ...) const>(FArgs ...)>>
#elif defined _MSC_VER  // MS Visual Studio
struct function_traits<
  std::_Binder<std::_Unforced, R (C::*)(Args ...) const, FArgs ...>>
#else
#error "Unsupported C++ compiler / standard library"
#endif
  : function_traits<R(Args ...)>
{};

// std::bind for free functions
template<typename R, typename ... Args, typename ... FArgs>
#if defined _LIBCPP_VERSION  // libc++ (Clang)
struct function_traits<std::__bind<R( &)(Args ...), FArgs ...>>
#elif defined __GLIBCXX__  // glibc++ (GNU C++)
struct function_traits<std::_Bind<R(*(FArgs ...))(Args ...)>>
#elif defined _MSC_VER  // MS Visual Studio
struct function_traits<std::_Binder<std::_Unforced, R( &)(Args ...), FArgs ...>>
#else
#error "Unsupported C++ compiler / standard library"
#endif
  : function_traits<R(Args ...)>
{}; */

// decay to the raw type
template <typename F>
struct function_traits<F&> : function_traits<F> {};

template <typename F>
struct function_traits<F&&> : function_traits<F> {};


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
// is_pod
//-----------------------------------------------------------------------------
template <typename T>
struct is_pod {
  static const bool value = std::is_trivial_v<T> && 
                            std::is_standard_layout_v<T>;
};

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

// ----------------------------------------------------------------------------
// bit_cast
//-----------------------------------------------------------------------------
template <class To, class From>
typename std::enable_if<
  (sizeof(To) == sizeof(From)) &&
  std::is_trivially_copyable_v<From> &&
  std::is_trivial_v<To>,
  // this implementation requires that To is trivially default constructible
  To
>::type
// constexpr support needs compiler magic
bit_cast(const From &src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

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


}  // end of namespace tf. ----------------------------------------------------



