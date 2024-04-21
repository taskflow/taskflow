# ==============================================================================
# LLVM Release License
# ==============================================================================
# University of Illinois/NCSA Open Source License
#
# Copyright (c) 2003-2018 University of Illinois at Urbana-Champaign. All rights
# reserved.
#
# Developed by:
#
# LLVM Team
#
# University of Illinois at Urbana-Champaign
#
# http://llvm.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# with the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimers.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimers in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the names of the LLVM Team, University of Illinois at
#   Urbana-Champaign, nor the names of its contributors may be used to endorse
#   or promote products derived from this Software without specific prior
#   written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
# THE SOFTWARE.

include(CheckCXXSourceCompiles)
include(CheckLibraryExists)

# Sometimes linking against libatomic is required for atomic ops, if the
# platform doesn't support lock-free atomics.

function(check_working_cxx_atomics varname)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++11")
  check_cxx_source_compiles(
    "
#include <atomic>
std::atomic<long long> x;
int main() {
  return std::atomic_is_lock_free(&x);
}
"
    ${varname})
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endfunction(check_working_cxx_atomics)

function(check_working_cxx_atomics64 varname)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "-std=c++11 ${CMAKE_REQUIRED_FLAGS}")
  check_cxx_source_compiles(
    "
#include <atomic>
#include <cstdint>
std::atomic<uint64_t> x (0);
int main() {
  uint64_t i = x.load(std::memory_order_relaxed);
  return std::atomic_is_lock_free(&x);
}
"
    ${varname})
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endfunction(check_working_cxx_atomics64)

function(check_working_cxx_atomics_2args varname)
  set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
  list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
  check_cxx_source_compiles(
    "
int main() {
  __atomic_load(nullptr, 0);
  return 0;
}
"
    ${varname})
  set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
endfunction(check_working_cxx_atomics_2args)

function(check_working_cxx_atomics64_2args varname)
  set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
  list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
  check_cxx_source_compiles(
    "
int main() {
  __atomic_load_8(nullptr, 0);
  return 0;
}
"
    ${varname})
  set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
endfunction(check_working_cxx_atomics64_2args)

# First check if atomics work without the library.
check_working_cxx_atomics(HAVE_CXX_ATOMICS_WITHOUT_LIB)

set(ATOMIC_LIBRARY "")

# If not, check if the library exists, and atomics work with it.
if(NOT HAVE_CXX_ATOMICS_WITHOUT_LIB)
  check_library_exists(atomic __atomic_fetch_add_4 "" HAVE_LIBATOMIC)
  if(NOT HAVE_LIBATOMIC)
    check_working_cxx_atomics_2args(HAVE_LIBATOMIC_2ARGS)
  endif()
  if(HAVE_LIBATOMIC OR HAVE_LIBATOMIC_2ARGS)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
    set(ATOMIC_LIBRARY "atomic")
    check_working_cxx_atomics(HAVE_CXX_ATOMICS_WITH_LIB)
    if(NOT HAVE_CXX_ATOMICS_WITH_LIB)
      message(FATAL_ERROR "Host compiler must support std::atomic!")
    endif()
  else()
    # Check for 64 bit atomic operations.
    if(MSVC)
      set(HAVE_CXX_ATOMICS64_WITHOUT_LIB True)
    else()
      check_working_cxx_atomics64(HAVE_CXX_ATOMICS64_WITHOUT_LIB)
    endif()

    # If not, check if the library exists, and atomics work with it.
    if(NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB)
      check_library_exists(atomic __atomic_load_8 "" HAVE_CXX_LIBATOMICS64)
      if(NOT HAVE_CXX_LIBATOMICS64)
        check_working_cxx_atomics64_2args(HAVE_CXX_LIBATOMICS64_2ARGS)
      endif()
      if(HAVE_CXX_LIBATOMICS64 OR HAVE_CXX_LIBATOMICS64_2ARGS)
        list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
        set(ATOMIC_LIBRARY "atomic")
        check_working_cxx_atomics64(HAVE_CXX_ATOMICS64_WITH_LIB)
        if(NOT HAVE_CXX_ATOMICS64_WITH_LIB)
          message(FATAL_ERROR "Host compiler must support std::atomic!")
        endif()
      else()
        message(
          FATAL_ERROR
            "Host compiler appears to require libatomic, but cannot find it.")
      endif()
    endif()

  endif()
endif()
