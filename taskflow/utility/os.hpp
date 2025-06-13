#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include <thread>

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
   * @brief The stored object, aligned to twice the cacheline size.
   */
  alignas (2*TF_CACHELINE_SIZE) T data;

  /**
   * @brief accesses the underlying object
   * 
   * @return a reference to the underlying object.
   */
  T& get() { return data; }
  
  /**
   * @brief accesses the underlying object as a constant reference
   * 
   * @return a constant reference to the underlying object.
   */
  const T& get() const { return data; }
};

/**
 * @brief retrieves the value of an environment variable
 *
 * This function fetches the value of an environment variable by name.
 * If the variable is not found, it returns an empty string.
 *
 * @param str The name of the environment variable to retrieve.
 * @return The value of the environment variable as a string, or an empty string if not found.
 *
 * @attention The implementation differs between Windows and POSIX platforms:
 *  - On Windows, it uses `_dupenv_s` to fetch the value.
 *  - On POSIX, it uses `std::getenv`.
 *
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
 * @brief checks whether an environment variable is defined
 *
 * This function determines if a specific environment variable exists in the current environment.
 *
 * @param str The name of the environment variable to check.
 * @return `true` if the environment variable exists, `false` otherwise.
 *
 * @attention The implementation differs between Windows and POSIX platforms:
 *  - On Windows, it uses `_dupenv_s` to check for the variable's presence.
 *  - On POSIX, it uses `std::getenv` to check for the variable's presence.
 *
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

/**
 * @fn pause
 * 
 * This function is used in spin-wait loops to hint the CPU that the current 
 * thread is in a busy-wait state. 
 * It helps reduce power consumption and improves performance on hyper-threaded processors 
 * by preventing the CPU from consuming unnecessary cycles while waiting. 
 * It is particularly useful in low-contention scenarios, where the thread 
 * is likely to quickly acquire the lock or condition it's waiting for, 
 * avoiding an expensive context switch. 
 * On modern x86 processors, this instruction can be invoked using @c __builtin_ia32_pause() 
 * in GCC/Clang or @c _mm_pause() in MSVC. 
 * In non-x86 architectures, alternative mechanisms such as yielding the CPU may be used instead.
 * 
 */
inline void pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86 and x86_64: Use the PAUSE instruction
  #if defined(_MSC_VER)
    // Microsoft Visual C++
    _mm_pause();
  #elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang
    __builtin_ia32_pause();
  #else
    asm volatile("pause" ::: "memory");
  #endif

#elif defined(__aarch64__) || defined(__arm__)
    // ARM and AArch64: Use the YIELD instruction
  #if defined(__GNUC__) || defined(__clang__)
    asm volatile("yield" ::: "memory");
  #endif

#else
  // Fallback: Portable yield for unknown architectures
  std::this_thread::yield();
#endif
}

/**
@brief pause CPU for a specified number of iterations
*/
inline void pause(size_t count) {
  while(count-- > 0) pause();
}

/**
 * @brief spins until the given predicate becomes true
 * 
 * @tparam P the type of the predicate function or callable.
 * @param predicate the callable that returns a boolean value, which is checked in the loop.
 * 
 * This function repeatedly checks the provided predicate in a spin-wait loop
 * and uses a backoff strategy to minimize CPU waste during the wait. Initially,
 * it uses the `pause()` instruction for the first 100 iterations to hint to the
 * CPU that the thread is waiting, thus reducing power consumption and avoiding
 * unnecessary cycles. After 100 iterations, it switches to yielding the CPU using
 * `std::this_thread::yield()` to allow other threads to run and improve system
 * responsiveness.
 * 
 * The function operates as follows:
 * 1. For the first 100 iterations, it invokes `pause()` to reduce power consumption
 *    during the spin-wait.
 * 2. After 100 iterations, it uses `std::this_thread::yield()` to relinquish the
 *    CPU, allowing other threads to execute.
 * 
 * @attention This function is useful when you need to wait for a condition to be true, but
 *       want to optimize CPU usage during the wait by using a busy-wait approach.
 * 
 */
template <typename P>
void spin_until(P&& predicate) {
  size_t num_pauses = 0;
  while(!predicate()) {
    (num_pauses++ < 100) ? pause() : std::this_thread::yield();
  }
}



}  // end of namespace tf -----------------------------------------------------









