#pragma once

#include <type_traits>
#include <iterator>
#include <iostream>
#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <list>
#include <forward_list>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <optional>
#include <variant>
#include <cmath>

namespace tf {

//-----------------------------------------------------------------------------
// Traits
//-----------------------------------------------------------------------------

// Macro to check whether a class has a member function
#define define_has_member(member_name)                                     \
template <typename T>                                                      \
class has_member_##member_name                                             \
{                                                                          \
  typedef char yes_type;                                                   \
  typedef long no_type;                                                    \
  template <typename U> static yes_type test(decltype(&U::member_name));   \
  template <typename U> static no_type  test(...);                         \
  public:                                                                  \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);  \
}

#define has_member(class_, member_name)  has_member_##member_name<class_>::value

// Struct: dependent_false
template <typename... T>
struct dependent_false { 
  static constexpr bool value = false; 
};

template <typename... T>
constexpr auto dependent_false_v = dependent_false<T...>::value;

// Struct: is_iterator
template <typename T, typename = void>
struct is_iterator {
  static constexpr bool value = false;
};

template <typename T>
struct is_iterator<
  T, 
  std::enable_if_t<!std::is_same_v<typename std::iterator_traits<T>::value_type, void>>
> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

// Struct: is_iterable
template <typename T, typename = void>
struct is_iterable : std::false_type {
};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin()),
                                  decltype(std::declval<T>().end())>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

// Struct: MoC
// Move-on-copy wrapper.
template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }
  
  mutable T object; 
};

//-----------------------------------------------------------------------------
// Functors.
//-----------------------------------------------------------------------------

// Overloadded.
template <typename... Ts>
struct Functors : Ts... { 
  using Ts::operator()... ;
};

template <typename... Ts>
Functors(Ts...) -> Functors<Ts...>;


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

  static constexpr size_t arity = std::tuple_size<arguments>::value;

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
struct function_traits<R(*)(Args...)> : public function_traits<R(Args...)> {
};

// function reference
template<typename R, typename... Args>
struct function_traits<R(&)(Args...)> : public function_traits<R(Args...)> {
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


}  // end of namespace tf. ---------------------------------------------------
