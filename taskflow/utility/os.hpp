#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>

#define TF_OS_LINUX 0
#define TF_OS_DRAGONFLY 0
#define TF_OS_FREEBSD 0
#define TF_OS_NETBSD 0
#define TF_OS_OPENBSD 0
#define TF_OS_DARWIN 0
#define TF_OS_WINDOWS 0
#define TF_OS_CNK 0
#define TF_OS_HURD 0
#define TF_OS_SOLARIS 0
#define TF_OS_UNIX 0

#ifdef _WIN32
#undef TF_OS_WINDOWS
#define TF_OS_WINDOWS 1
#endif

#ifdef __CYGWIN__
#undef TF_OS_WINDOWS
#define TF_OS_WINDOWS 1
#endif

#if (defined __APPLE__ && defined __MACH__)
#undef TF_OS_DARWIN
#define TF_OS_DARWIN 1
#endif

// in some ppc64 linux installations, only the second condition is met
#if (defined __linux)
#undef TF_OS_LINUX
#define TF_OS_LINUX 1
#elif (defined __linux__)
#undef TF_OS_LINUX
#define TF_OS_LINUX 1
#else
#endif

#if (defined __DragonFly__)
#undef TF_OS_DRAGONFLY
#define TF_OS_DRAGONFLY 1
#endif

#if (defined __FreeBSD__)
#undef TF_OS_FREEBSD
#define TF_OS_FREEBSD 1
#endif

#if (defined __NetBSD__)
#undef TF_OS_NETBSD
#define TF_OS_NETBSD 1
#endif

#if (defined __OpenBSD__)
#undef TF_OS_OPENBSD
#define TF_OS_OPENBSD 1
#endif

#if (defined __bgq__)
#undef TF_OS_CNK
#define TF_OS_CNK 1
#endif

#if (defined __GNU__)
#undef TF_OS_HURD
#define TF_OS_HURD 1
#endif

#if (defined __sun)
#undef TF_OS_SOLARIS
#define TF_OS_SOLARIS 1
#endif

#if (1 !=                                                                  \
     TF_OS_LINUX + TF_OS_DRAGONFLY + TF_OS_FREEBSD + TF_OS_NETBSD +        \
     TF_OS_OPENBSD + TF_OS_DARWIN + TF_OS_WINDOWS + TF_OS_HURD +           \
     TF_OS_SOLARIS)
#define TF_OS_UNKNOWN 1
#endif

#if TF_OS_LINUX || TF_OS_DRAGONFLY || TF_OS_FREEBSD || TF_OS_NETBSD ||     \
    TF_OS_OPENBSD || TF_OS_DARWIN || TF_OS_HURD || TF_OS_SOLARIS
#undef TF_OS_UNIX
#define TF_OS_UNIX 1
#endif


//-----------------------------------------------------------------------------
// Cache line alignment
//-----------------------------------------------------------------------------
#if defined(__i386__) || defined(__x86_64__)
  #define TF_CACHELINE_SIZE 64
#elif defined(__powerpc64__)
  // This is the L1 D-cache line size of our Power7 machines.
  // Need to check if this is appropriate for other PowerPC64 systems.
  #define TF_CACHELINE_SIZE 128
#elif defined(__arm__)
  // Cache line sizes for ARM: These values are not strictly correct since
  // cache line sizes depend on implementations, not architectures.
  // There are even implementations with cache line sizes configurable
  // at boot time.
  #if defined(__ARM_ARCH_5T__)
    #define TF_CACHELINE_SIZE 32
  #elif defined(__ARM_ARCH_7A__)
    #define TF_CACHELINE_SIZE 64
  #endif
#endif

#ifndef TF_CACHELINE_SIZE
// A reasonable default guess.  Note that overestimates tend to waste more
// space, while underestimates tend to waste more time.
  #define TF_CACHELINE_SIZE 64
#endif



//-----------------------------------------------------------------------------
// pause
//-----------------------------------------------------------------------------
//#if __has_include (<immintrin.h>)
//  #define TF_HAS_MM_PAUSE 1
//  #include <immintrin.h>
//#endif

namespace tf {

// Struct: CachelineAligned
// Due to prefetch, we typically do 2x cacheline for the alignment.
template <typename T>
struct CachelineAligned {
  alignas (2*TF_CACHELINE_SIZE) T data;
  T& get() { return data; }
};

// Function: get_env
inline std::string get_env(const std::string& str) {
#ifdef _MSC_VER
  char *ptr = nullptr;
  size_t len = 0;

  if(_dupenv_s(&ptr, &len, str.c_str()) == 0 && ptr != nullptr) {
    std::string res(ptr, len);
    std::free(ptr);
    return res;
  }
  return "";

#else
  auto ptr = std::getenv(str.c_str());
  return ptr ? ptr : "";
#endif
}

// Function: has_env
inline bool has_env(const std::string& str) {
#ifdef _MSC_VER
  char *ptr = nullptr;
  size_t len = 0;

  if(_dupenv_s(&ptr, &len, str.c_str()) == 0 && ptr != nullptr) {
    std::string res(ptr, len);
    std::free(ptr);
    return true;
  }
  return false;

#else
  auto ptr = std::getenv(str.c_str());
  return ptr ? true : false;
#endif
}

// Procedure: relax_cpu
//inline void relax_cpu() {
//#ifdef TF_HAS_MM_PAUSE
//  _mm_pause();
//#endif
//}



}  // end of namespace tf -----------------------------------------------------









