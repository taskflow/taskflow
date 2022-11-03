#pragma once

#if defined(_MSC_VER)
  #define TF_FORCE_INLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ > 3
  #define TF_FORCE_INLINE __attribute__((__always_inline__)) inline
#else
  #define TF_FORCE_INLINE inline
#endif

#if defined(_MSC_VER)
  #define TF_NO_INLINE __declspec(noinline)
#elif defined(__GNUC__) && __GNUC__ > 3
  #define TF_NO_INLINE __attribute__((__noinline__))
#else
  #define TF_NO_INLINE
#endif
