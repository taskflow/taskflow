// 2020/08/30 - Created by netcan: https://github.com/netcan
// ref https://github.com/Erlkoenig90/map-macro/
#pragma once
#define TF_EVAL0(...) __VA_ARGS__
#define TF_EVAL1(...) TF_EVAL0(TF_EVAL0(TF_EVAL0(__VA_ARGS__)))
#define TF_EVAL2(...) TF_EVAL1(TF_EVAL1(TF_EVAL1(__VA_ARGS__)))
#define TF_EVAL3(...) TF_EVAL2(TF_EVAL2(TF_EVAL2(__VA_ARGS__)))
#define TF_EVAL4(...) TF_EVAL3(TF_EVAL3(TF_EVAL3(__VA_ARGS__)))
#define TF_EVAL5(...) TF_EVAL4(TF_EVAL4(TF_EVAL4(__VA_ARGS__)))

#ifdef _MSC_VER
// MSVC needs more evaluations
#define TF_EVAL6(...) TF_EVAL5(TF_EVAL5(TF_EVAL5(__VA_ARGS__)))
#define TF_EVAL(...) TF_EVAL6(TF_EVAL6(__VA_ARGS__))
#else
#define TF_EVAL(...) TF_EVAL5(__VA_ARGS__)
#endif

#define TF_MAP_END(...)
#define TF_MAP_OUT

#define EMPTY()
#define DEFER(id) id EMPTY()

#define TF_MAP_GET_END2() 0, TF_MAP_END
#define TF_MAP_GET_END1(...) TF_MAP_GET_END2
#define TF_MAP_GET_END(...) TF_MAP_GET_END1
#define TF_MAP_NEXT0(test, next, ...) next TF_MAP_OUT
#define TF_MAP_NEXT1(test, next) DEFER(TF_MAP_NEXT0)(test, next, 0)
#define TF_MAP_NEXT(test, next) TF_MAP_NEXT1(TF_MAP_GET_END test, next)

#define TF_MAP0(f, x, peek, ...)                                               \
  f(x) DEFER(TF_MAP_NEXT(peek, TF_MAP1))(f, peek, __VA_ARGS__)
#define TF_MAP1(f, x, peek, ...)                                               \
  f(x) DEFER(TF_MAP_NEXT(peek, TF_MAP0))(f, peek, __VA_ARGS__)

#define TF_MAP(f, ...)                                                         \
  TF_EVAL(TF_MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))
