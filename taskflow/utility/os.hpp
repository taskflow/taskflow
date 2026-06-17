#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include <thread>
#include <new>

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

// ------------------------------------------------------------------------------------------------
// Number of bits used by the OS for user-space virtual addresses
// Used as the default PtrBits for TaggedHead64 to leave the remaining
// high bits free for the ABA version counter.
// ------------------------------------------------------------------------------------------------
//
// No standard C++ mechanism exposes the VA width; we derive it from
// architecture-specific predefined macros. Override by defining
// TF_POINTER_BITS before including this header if your environment
// differs (e.g. x86-64 with LA57 5-level paging uses 57 bits).

#if defined(TF_POINTER_BITS)
  // user-defined override — accepted as-is

#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
  // 4-level paging: 48 bits. If your kernel enables LA57 (5-level paging,
  // 57-bit VA), compile with -DTF_POINTER_BITS=57. Note that only 7 bits
  // remain for the ABA tag in that case; TaggedHead128 is a better choice.
  #define TF_POINTER_BITS 48

#elif defined(__aarch64__) || defined(_M_ARM64)
  #define TF_POINTER_BITS 48   // ARMv8 48-bit VA (TTBR0 range)

#elif defined(__riscv) && __riscv_xlen == 64
  #define TF_POINTER_BITS 48   // SV48 worst case; SV39 gives 25 free bits

#else
  #define TF_POINTER_BITS (sizeof(void*) * CHAR_BIT)  // 32-bit or unknown
#endif


// ------------------------------------------------------------------------------------------------
// Cache line size detection.
//
// Underestimating causes false sharing (hurts performance).
// Overestimating wastes memory.
// 64B is correct for the vast majority of modern server/desktop CPUs.
// ------------------------------------------------------------------------------------------------

#if defined(__i386__) || defined(__x86_64__) || \
    defined(_M_IX86)  || defined(_M_AMD64)
  // All modern x86/x86_64 CPUs (Intel, AMD) since Pentium 4.
  #define TF_CACHELINE_SIZE 64

#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  // 64-bit ARM: Apple Silicon (M1/M2/M3), AWS Graviton, Qualcomm Oryon, etc.
  #define TF_CACHELINE_SIZE 64

#elif defined(__arm__) || defined(_M_ARM)
  // 32-bit ARM — cache line size depends on the ARM architecture revision.
  #if defined(__ARM_ARCH_5T__)  || \
      defined(__ARM_ARCH_5TE__) || \
      defined(__ARM_ARCH_6__)
    #define TF_CACHELINE_SIZE 32
  #else
    // ARMv7-A (Cortex-A5/A7/A8/A9/A15) and later 32-bit ARM.
    #define TF_CACHELINE_SIZE 64
  #endif

#elif defined(__powerpc64__) || defined(__ppc64__)
  // IBM POWER7/8/9/10 and PowerPC64 (e.g., original Xbox 360-era G5).
  #define TF_CACHELINE_SIZE 128

#elif defined(__powerpc__) || defined(__ppc__)
  // Older 32-bit PowerPC (G3/G4 era embedded systems).
  #define TF_CACHELINE_SIZE 32

#elif defined(__s390x__) || defined(__zarch__)
  // IBM Z (z13 and later). Unusually large at 256B.
  #define TF_CACHELINE_SIZE 256

#elif defined(__riscv)
  // RISC-V: SiFive U74, Alibaba T-Head, etc.
  #define TF_CACHELINE_SIZE 64

#elif defined(__mips__) || defined(__mips64)
  // MIPS32/MIPS64 (embedded and networking SoCs).
  #define TF_CACHELINE_SIZE 64

#elif defined(__sparc__) || defined(__sparc64__)
  // Oracle/Fujitsu SPARC.
  #define TF_CACHELINE_SIZE 64

#elif defined(__loongarch64)
  // LoongArch (Loongson 3A5000 and later).
  #define TF_CACHELINE_SIZE 64

#elif defined(__alpha__)
  // DEC/Compaq Alpha.
  #define TF_CACHELINE_SIZE 64

#endif

#ifndef TF_CACHELINE_SIZE
  // Conservative fallback. If we land here, the architecture is unknown.
  // 64B is correct for virtually all modern CPUs. Overestimating wastes
  // a small amount of space; underestimating causes false sharing.
  #define TF_CACHELINE_SIZE 64
#endif

namespace tf {

/**
@class CachelineAligned

@brief class to ensure cacheline-aligned storage for an object.

@tparam T The type of the stored object.

This utility class aligns the stored object `data` to twice the size of a cacheline.
The alignment improves performance by optimizing data access in cache-sensitive scenarios.

@code{.cpp}
// create two integers on two separate cachelines to avoid false sharing
tf::CachelineAligned<int> counter1;
tf::CachelineAligned<int> counter2;

// two threads access the two counters without false sharing
std::thread t1([&]{ counter1.get() = 1; });
std::thread t2([&]{ counter2.get() = 2; });
t1.join();
t2.join();
@endcode
*/
template <typename T>
class CachelineAligned {
  public:
  /**
  @brief The stored object, aligned to twice the cacheline size.
  */
  alignas (TF_CACHELINE_SIZE) T data;

  /**
  @brief accesses the underlying object

  @return a reference to the underlying object.

  Returns a mutable reference to the stored object so it can be read
  or written directly without copying.

  @code{.cpp}
  tf::CachelineAligned<int> counter;
  counter.get() = 1;
  @endcode
  */
  T& get() { return data; }

  /**
  @brief accesses the underlying object as a constant reference

  @return a constant reference to the underlying object.

  Returns a read-only reference to the stored object, for use on
  const-qualified instances of `CachelineAligned`.

  @code{.cpp}
  const tf::CachelineAligned<int> counter;
  int v = counter.get();
  @endcode
  */
  const T& get() const { return data; }
};

/**
@brief retrieves the value of an environment variable

@param str The name of the environment variable to retrieve.
@return The value of the environment variable as a string, or an empty string if not found.

This function fetches the value of an environment variable by name.
If the variable is not found, it returns an empty string.

@code{.cpp}
std::string path = tf::get_env("PATH");
@endcode

@note The implementation differs between Windows and POSIX platforms:
 - On Windows, it uses `_dupenv_s` to fetch the value.
 - On POSIX, it uses `std::getenv`.
*/
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

/**
@brief checks whether an environment variable is defined

@param str The name of the environment variable to check.
@return `true` if the environment variable exists, `false` otherwise.

This function determines if a specific environment variable exists in the current environment.

@code{.cpp}
if(tf::has_env("TF_NUM_THREADS")) {
  // ...
}
@endcode

@note The implementation differs between Windows and POSIX platforms:
 - On Windows, it uses `_dupenv_s` to check for the variable's presence.
 - On POSIX, it uses `std::getenv` to check for the variable's presence.
*/
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


}  // end of namespace tf -----------------------------------------------------









