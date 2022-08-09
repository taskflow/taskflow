// 2020/08/30 - Created by netcan: https://github.com/netcan
// ref https://github.com/Erlkoenig90/map-macro/
#pragma once
#ifdef _MSC_VER
#define TF_EMPTY()
#define TF_GET_ARG_COUNT_(...)                                                 \
  TF_PASTE(TF_GET_ARG_COUNT_I(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, \
                              55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,  \
                              43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,  \
                              31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,  \
                              19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, \
                              6, 5, 4, 3, 2, 1, 0, ),                          \
           TF_EMPTY())

#else
#define TF_GET_ARG_COUNT_(...)                                                 \
  TF_GET_ARG_COUNT_I(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54,  \
                     53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40,   \
                     39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,   \
                     25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,   \
                     11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, )
#endif

#define TF_GET_ARG_COUNT(...) TF_GET_ARG_COUNT_(__dummy__, ##__VA_ARGS__)
#define TF_GET_ARG_COUNT_I(                                                    \
    e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, \
    e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, \
    e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, \
    e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, \
    e62, e63, e64, size, ...)                                                  \
  size

#define TF_GET_FIRST(a, ...) a
#define TF_GET_SECOND(a, b, ...) b
#define TF_CONCATE(x, y) x##y
#define TF_PASTE(x, y) TF_CONCATE(x, y)

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
