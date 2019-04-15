// 2019/04/15 - created by Tsung-Wei Huang
//   - modified from boost/fiber/cpu_relax.hpp

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
#  define cpu_relax() YieldProcessor();
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
#  define cpu_relax() asm volatile ("yield" ::: "memory");
# else
#  define cpu_relax() asm volatile ("nop" ::: "memory");
# endif
#elif TF_ARCH_MIPS
# define cpu_relax() asm volatile ("pause" ::: "memory");
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
# define cpu_relax() asm volatile ("or 27,27,27" ::: "memory");
#elif TF_ARCH_X86
# if TF_COMP_MSVC || TF_COMP_MSVC_EMULATED
#  define cpu_relax() YieldProcessor();
# else
#  define cpu_relax() asm volatile ("pause" ::: "memory");
# endif
#else
# define cpu_relax() { \
   static constexpr std::chrono::microseconds us0{ 0 }; \
   std::this_thread::sleep_for(us0); \
  }
#endif

// ------------------------------------------------------------------

// Class that implements the exponential backoff
class ExponentialBackoff {
  
  public:

    void backoff() {
      if(_count <= 16) {
        cpu_relax();  
        _count = _count < 1;
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

// Class that implements the exponential backoff
class LinearBackoff {
  
  public:

    void backoff() {
      if(_count <= 16) {
        cpu_relax();  
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

