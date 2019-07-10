// 2019/04/15 - created by Tsung-Wei Huang
//   - modified from boost/fiber/cpu_relax.hpp

#pragma once

#include <chrono>
#include <thread>

#include "../predef/compiler.hpp"
#include "../predef/architecture.hpp"

#if TF_COMP_MSVC || TF_COMP_MSVC_EMULATED
# define NOMINMAX
# include <windows.h>
#endif

namespace tf {

#if TF_ARCH_ARM
# if TF_COMP_MSVC
#  define TF_PAUSE() YieldProcessor();
# elif (defined(__ARM_ARCH_6K__) || \
        defined(__ARM_ARCH_6Z__) || \
        defined(__ARM_ARCH_6ZK__) || \
        defined(__ARM_ARCH_6T2__) || \
        defined(__ARM_ARCH_7__) || \
        defined(__ARM_ARCH_7A__) || \
        defined(__ARM_ARCH_7R__) || \
        defined(__ARM_ARCH_7M__) || \
        defined(__ARM_ARCH_7S__) || \
        defined(__ARM_ARCH_8A__) || \
        defined(__aarch64__))
// http://groups.google.com/a/chromium.org/forum/#!msg/chromium-dev/YGVrZbxYOlU/Vpgy__zeBQAJ
// mnemonic 'yield' is supported from ARMv6k onwards
#  define TF_PAUSE() asm volatile ("yield" ::: "memory");
# else
#  define TF_PAUSE() asm volatile ("nop" ::: "memory");
# endif
#elif TF_ARCH_MIPS
# define TF_PAUSE() asm volatile ("pause" ::: "memory");
#elif TF_ARCH_PPC
// http://code.metager.de/source/xref/gnu/glibc/sysdeps/powerpc/sys/platform/ppc.h
// http://stackoverflow.com/questions/5425506/equivalent-of-x86-pause-instruction-for-ppc
// mnemonic 'or' shared resource hints
// or 27, 27, 27 This form of 'or' provides a hint that performance
//               will probably be imrpoved if shared resources dedicated
//               to the executing processor are released for use by other
//               processors
// extended mnemonics (available with POWER7)
// yield   ==   or 27, 27, 27
# define TF_PAUSE() asm volatile ("or 27,27,27" ::: "memory");
#elif TF_ARCH_X86
# if TF_COMP_MSVC || TF_COMP_MSVC_EMULATED
#  define TF_PAUSE() YieldProcessor();
# else
#  define TF_PAUSE() asm volatile ("pause" ::: "memory");
# endif
#else
# define TF_PAUSE() { \
    static constexpr std::chrono::microseconds us0{ 0 }; \
    std::this_thread::sleep_for(us0); \
  }
#endif

// ------------------------------------------------------------------

// Procedure: relax_cpu
// pause cpu for a few rounds
inline void relax_cpu(int32_t cycles) {
  while(cycles > 0) {
    TF_PAUSE();
    cycles--;
  }
}

// Procedure: relax_cpu
inline void relax_cpu() {
  TF_PAUSE();
}

// ------------------------------------------------------------------

// Class that implements the exponential backoff
class ExponentialBackoff {

  static constexpr int LOOPS_BEFORE_YIELD = 16;
  
  public:
    
    ExponentialBackoff() = default;

    void backoff() {
      if(_count <= LOOPS_BEFORE_YIELD) {
        relax_cpu(_count);
        _count = _count << 1;
      }
      else {
        std::this_thread::yield();
      }
    }

    // pause for a few times and return false if saturated
    bool bounded_pause() {
      relax_cpu(_count);
      if(_count < LOOPS_BEFORE_YIELD) {
        _count = _count << 1;
        return true;
      } else {
        return false;
      }
    }

    void reset() {
      _count = 1;
    }

  private:

    int _count {1};

};

// Class that implements the exponential backoff
class LinearBackoff {
  
  public:

    LinearBackoff() = default;

    void backoff() {
      if(_count <= 16) {
        relax_cpu(_count);
        ++_count;
      }
      else {
        std::this_thread::yield();
      }
    }

    void reset() {
      _count = 1;
    }

  private:

    int32_t _count {1};

};

};  // end of namespace tf. ---------------------------------------------------

