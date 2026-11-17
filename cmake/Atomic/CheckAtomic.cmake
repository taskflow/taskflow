# =============================================================================
# CheckAtomic.cmake
#
# Detects whether the platform needs -latomic for atomic operations and sets
# ATOMIC_LIBRARY accordingly. This is required on some platforms (e.g., ARM,
# 32-bit x86, older GCC) where std::atomic operations are not inlined by the
# compiler and require explicit linkage against libatomic.
#
# Sets:
#   ATOMIC_LIBRARY  — "atomic" if -latomic is required, empty otherwise
# =============================================================================

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# Test program: exercises both 64-bit and 128-bit std::atomic operations.
# The 128-bit test uses a struct with a pointer and a size_t tag — the same
# pattern used by AtomicIntrusiveStack's TaggedPointer. On platforms without
# native 128-bit CAS, std::atomic will use a mutex internally (still correct),
# but -latomic may still be required for the runtime support code.
set(_TF_ATOMIC_TEST_SRC "
#include <atomic>
#include <cstddef>
#include <cstdint>

// 64-bit atomic
static std::atomic<std::uint64_t> a64 {0};

// 128-bit struct atomic (TaggedPointer pattern used by AtomicIntrusiveStack)
struct TaggedPointer {
  void*       ptr {nullptr};
  std::size_t tag {0};
};
static std::atomic<TaggedPointer> a128 {};

int main() {
  // 64-bit operations
  a64.fetch_add(1, std::memory_order_relaxed);
  a64.fetch_sub(1, std::memory_order_relaxed);
  std::uint64_t e64 = 0;
  a64.compare_exchange_strong(e64, 1,
    std::memory_order_acq_rel, std::memory_order_acquire);

  // 128-bit operations
  TaggedPointer expected{};
  TaggedPointer desired{nullptr, 1};
  a128.compare_exchange_strong(expected, desired,
    std::memory_order_acq_rel, std::memory_order_acquire);

  return 0;
}
")

cmake_push_check_state()

# attempt 1: compile without -latomic
check_cxx_source_compiles("${_TF_ATOMIC_TEST_SRC}" TF_ATOMIC_BUILTIN)

if(TF_ATOMIC_BUILTIN)
  set(ATOMIC_LIBRARY "" CACHE STRING "Atomic library (empty = native)" FORCE)
  message(STATUS "Atomic operations: native (no -latomic needed)")
else()
  # attempt 2: compile with -latomic
  set(CMAKE_REQUIRED_LIBRARIES atomic)
  check_cxx_source_compiles("${_TF_ATOMIC_TEST_SRC}" TF_ATOMIC_WITH_LATOMIC)
  unset(CMAKE_REQUIRED_LIBRARIES)

  if(TF_ATOMIC_WITH_LATOMIC)
    set(ATOMIC_LIBRARY "atomic" CACHE STRING "Atomic library" FORCE)
    message(STATUS "Atomic operations: require -latomic")
  else()
    message(FATAL_ERROR
      "Taskflow requires std::atomic support. Neither native atomics nor "
      "-latomic compilation succeeded on this platform."
    )
  endif()
endif()

cmake_pop_check_state()

if(ATOMIC_LIBRARY)
  message(STATUS "ATOMIC_LIBRARY: -l${ATOMIC_LIBRARY}")
else()
  message(STATUS "ATOMIC_LIBRARY: (none)")
endif()
