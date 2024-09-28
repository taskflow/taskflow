#pragma once

// ============================================================================
// inline and no-inline
// ============================================================================

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

// ============================================================================
// likely and unlikely
// ============================================================================

#if defined(__GNUC__)
  #define TF_LIKELY(x) (__builtin_expect((x), 1))
  #define TF_UNLIKELY(x) (__builtin_expect((x), 0))
#else
  #define TF_LIKELY(x) (x)
  #define TF_UNLIKELY(x) (x)
#endif

// ----------------------------------------------------------------------------

#ifdef TF_DISABLE_EXCEPTION_HANDLING
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block) \
    code_block;
#else
  #define TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, code_block)  \
    try {                                          \
      code_block;                                  \
    } catch(...) {                                 \
      _process_exception(worker, node);            \
    }
#endif

// ----------------------------------------------------------------------------    
