//
// doctest.h - the lightest feature-rich C++ single-header testing framework for unit tests and TDD
//
// Copyright (c) 2016-2018 Viktor Kirilov
//
// Distributed under the MIT Software License
// See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/MIT
//
// The documentation can be found at the library's page:
// https://github.com/onqtam/doctest/blob/master/doc/markdown/readme.md
//
// =================================================================================================
// =================================================================================================
// =================================================================================================
//
// The library is heavily influenced by Catch - https://github.com/philsquared/Catch
// which uses the Boost Software License - Version 1.0
// see here - https://github.com/philsquared/Catch/blob/master/LICENSE.txt
//
// The concept of subcases (sections in Catch) and expression decomposition are from there.
// Some parts of the code are taken directly:
// - stringification - the detection of "ostream& operator<<(ostream&, const T&)" and StringMaker<>
// - the Approx() helper class for floating point comparison
// - colors in the console
// - breaking into a debugger
// - signal / SEH handling
// - timer
//
// The expression decomposing templates are taken from lest - https://github.com/martinmoene/lest
// which uses the Boost Software License - Version 1.0
// see here - https://github.com/martinmoene/lest/blob/master/LICENSE.txt
//
// The type list and the foreach algorithm on it for C++98 are taken from Loki
// - http://loki-lib.sourceforge.net/
// - https://en.wikipedia.org/wiki/Loki_%28C%2B%2B%29
// - https://github.com/snaewe/loki-lib
// which uses the MIT Software License
//
// =================================================================================================
// =================================================================================================
// =================================================================================================

#ifndef DOCTEST_LIBRARY_INCLUDED
#define DOCTEST_LIBRARY_INCLUDED

// =================================================================================================
// == VERSION ======================================================================================
// =================================================================================================

#define DOCTEST_VERSION_MAJOR 1
#define DOCTEST_VERSION_MINOR 2
#define DOCTEST_VERSION_PATCH 8
#define DOCTEST_VERSION_STR "1.2.8"

#define DOCTEST_VERSION                                                                            \
    (DOCTEST_VERSION_MAJOR * 10000 + DOCTEST_VERSION_MINOR * 100 + DOCTEST_VERSION_PATCH)

// =================================================================================================
// == COMPILER VERSION =============================================================================
// =================================================================================================

// ideas for the version stuff are taken from here: https://github.com/cxxstuff/cxx_detect

#define DOCTEST_COMPILER(MAJOR, MINOR, PATCH) ((MAJOR)*10000000 + (MINOR)*100000 + (PATCH))

#if defined(_MSC_VER) && defined(_MSC_FULL_VER)
#if _MSC_VER == _MSC_FULL_VER / 10000
#define DOCTEST_MSVC DOCTEST_COMPILER(_MSC_VER / 100, _MSC_VER % 100, _MSC_FULL_VER % 10000)
#else
#define DOCTEST_MSVC                                                                               \
    DOCTEST_COMPILER(_MSC_VER / 100, (_MSC_FULL_VER / 100000) % 100, _MSC_FULL_VER % 100000)
#endif
#elif defined(__clang__) && defined(__clang_minor__)
#define DOCTEST_CLANG DOCTEST_COMPILER(__clang_major__, __clang_minor__, __clang_patchlevel__)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define DOCTEST_GCC DOCTEST_COMPILER(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#endif

#ifndef DOCTEST_MSVC
#define DOCTEST_MSVC 0
#endif // DOCTEST_MSVC
#ifndef DOCTEST_CLANG
#define DOCTEST_CLANG 0
#endif // DOCTEST_CLANG
#ifndef DOCTEST_GCC
#define DOCTEST_GCC 0
#endif // DOCTEST_GCC

// =================================================================================================
// == COMPILER WARNINGS HELPERS ====================================================================
// =================================================================================================

#if DOCTEST_CLANG
#ifdef __has_warning
#define DOCTEST_CLANG_HAS_WARNING(x) __has_warning(x)
#endif // __has_warning
#ifdef __has_feature
#define DOCTEST_CLANG_HAS_FEATURE(x) __has_feature(x)
#endif // __has_feature
#define DOCTEST_PRAGMA_TO_STR(x) _Pragma(#x)
#define DOCTEST_CLANG_SUPPRESS_WARNING_PUSH _Pragma("clang diagnostic push")
#define DOCTEST_MSVC_SUPPRESS_WARNING_PUSH
#define DOCTEST_GCC_SUPPRESS_WARNING_PUSH
#define DOCTEST_CLANG_SUPPRESS_WARNING(w) DOCTEST_PRAGMA_TO_STR(clang diagnostic ignored w)
#define DOCTEST_MSVC_SUPPRESS_WARNING(w)
#define DOCTEST_GCC_SUPPRESS_WARNING(w)
#define DOCTEST_CLANG_SUPPRESS_WARNING_POP _Pragma("clang diagnostic pop")
#define DOCTEST_MSVC_SUPPRESS_WARNING_POP
#define DOCTEST_GCC_SUPPRESS_WARNING_POP
#define DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)                                                \
    DOCTEST_CLANG_SUPPRESS_WARNING_PUSH DOCTEST_CLANG_SUPPRESS_WARNING(w)
#define DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(w)
#define DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH(w)
#elif DOCTEST_GCC
#define DOCTEST_PRAGMA_TO_STR(x) _Pragma(#x)
#define DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
#define DOCTEST_MSVC_SUPPRESS_WARNING_PUSH
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 7, 0)
#define DOCTEST_GCC_SUPPRESS_WARNING_PUSH _Pragma("GCC diagnostic push")
#else // GCC 4.7+
#define DOCTEST_GCC_SUPPRESS_WARNING_PUSH
#endif // GCC 4.7+
#define DOCTEST_CLANG_SUPPRESS_WARNING(w)
#define DOCTEST_MSVC_SUPPRESS_WARNING(w)
#define DOCTEST_GCC_SUPPRESS_WARNING(w) DOCTEST_PRAGMA_TO_STR(GCC diagnostic ignored w)
#define DOCTEST_CLANG_SUPPRESS_WARNING_POP
#define DOCTEST_MSVC_SUPPRESS_WARNING_POP
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 7, 0)
#define DOCTEST_GCC_SUPPRESS_WARNING_POP _Pragma("GCC diagnostic pop")
#else // GCC 4.7+
#define DOCTEST_GCC_SUPPRESS_WARNING_POP
#endif // GCC 4.7+
#define DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#define DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(w)
#define DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH(w)                                                  \
    DOCTEST_GCC_SUPPRESS_WARNING_PUSH DOCTEST_GCC_SUPPRESS_WARNING(w)
#elif DOCTEST_MSVC
#define DOCTEST_PRAGMA_TO_STR(x)
#define DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
#define DOCTEST_MSVC_SUPPRESS_WARNING_PUSH __pragma(warning(push))
#define DOCTEST_GCC_SUPPRESS_WARNING_PUSH
#define DOCTEST_CLANG_SUPPRESS_WARNING(w)
#define DOCTEST_MSVC_SUPPRESS_WARNING(w) __pragma(warning(disable : w))
#define DOCTEST_GCC_SUPPRESS_WARNING(w)
#define DOCTEST_CLANG_SUPPRESS_WARNING_POP
#define DOCTEST_MSVC_SUPPRESS_WARNING_POP __pragma(warning(pop))
#define DOCTEST_GCC_SUPPRESS_WARNING_POP
#define DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#define DOCTEST_MSVC_SUPPRESS_WARNING_WITH_PUSH(w)                                                 \
    DOCTEST_MSVC_SUPPRESS_WARNING_PUSH DOCTEST_MSVC_SUPPRESS_WARNING(w)
#define DOCTEST_GCC_SUPPRESS_WARNING_WITH_PUSH(w)
#endif // different compilers - warning suppression macros

#ifndef DOCTEST_CLANG_HAS_WARNING
#define DOCTEST_CLANG_HAS_WARNING(x) 1
#endif // DOCTEST_CLANG_HAS_WARNING

#ifndef DOCTEST_CLANG_HAS_FEATURE
#define DOCTEST_CLANG_HAS_FEATURE(x) 0
#endif // DOCTEST_CLANG_HAS_FEATURE

// =================================================================================================
// == COMPILER WARNINGS ============================================================================
// =================================================================================================

DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
DOCTEST_CLANG_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wnon-virtual-dtor")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wweak-vtables")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wpadded")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wdeprecated")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wmissing-prototypes")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wunused-local-typedef")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++11-long-long")
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_WARNING("-Wzero-as-null-pointer-constant")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wzero-as-null-pointer-constant")
#endif // clang - 0 as null

DOCTEST_GCC_SUPPRESS_WARNING_PUSH
DOCTEST_GCC_SUPPRESS_WARNING("-Wunknown-pragmas")
DOCTEST_GCC_SUPPRESS_WARNING("-Weffc++")
DOCTEST_GCC_SUPPRESS_WARNING("-Wstrict-overflow")
DOCTEST_GCC_SUPPRESS_WARNING("-Wstrict-aliasing")
DOCTEST_GCC_SUPPRESS_WARNING("-Wctor-dtor-privacy")
DOCTEST_GCC_SUPPRESS_WARNING("-Wmissing-declarations")
DOCTEST_GCC_SUPPRESS_WARNING("-Wnon-virtual-dtor")
DOCTEST_GCC_SUPPRESS_WARNING("-Winline")
DOCTEST_GCC_SUPPRESS_WARNING("-Wlong-long")
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 7, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wzero-as-null-pointer-constant")
#endif // GCC 4.7+
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 8, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wunused-local-typedefs")
#endif // GCC 4.8+
#if DOCTEST_GCC >= DOCTEST_COMPILER(5, 4, 0)
DOCTEST_GCC_SUPPRESS_WARNING("-Wuseless-cast")
#endif // GCC 5.4+

DOCTEST_MSVC_SUPPRESS_WARNING_PUSH
DOCTEST_MSVC_SUPPRESS_WARNING(4616) // invalid compiler warning
DOCTEST_MSVC_SUPPRESS_WARNING(4619) // invalid compiler warning
DOCTEST_MSVC_SUPPRESS_WARNING(4996) // The compiler encountered a deprecated declaration
DOCTEST_MSVC_SUPPRESS_WARNING(4706) // assignment within conditional expression
DOCTEST_MSVC_SUPPRESS_WARNING(4512) // 'class' : assignment operator could not be generated
DOCTEST_MSVC_SUPPRESS_WARNING(4127) // conditional expression is constant
DOCTEST_MSVC_SUPPRESS_WARNING(4820) // padding
DOCTEST_MSVC_SUPPRESS_WARNING(4625) // copy constructor was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(4626) // assignment operator was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(5027) // move assignment operator was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(5026) // move constructor was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(4623) // default constructor was implicitly defined as deleted
DOCTEST_MSVC_SUPPRESS_WARNING(4640) // construction of local static object is not thread-safe

// C4548 - expression before comma has no effect; expected expression with side - effect
// C4986 - exception specification does not match previous declaration
// C4350 - behavior change: 'member1' called instead of 'member2'
// C4668 - 'x' is not defined as a preprocessor macro, replacing with '0' for '#if/#elif'
// C4365 - conversion from 'int' to 'unsigned long', signed/unsigned mismatch
// C4774 - format string expected in argument 'x' is not a string literal
// C4820 - padding in structs

// only 4 should be disabled globally:
// - C4514 # unreferenced inline function has been removed
// - C4571 # SEH related
// - C4710 # function not inlined
// - C4711 # function 'x' selected for automatic inline expansion

#define DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN                                 \
    DOCTEST_MSVC_SUPPRESS_WARNING_PUSH                                                             \
    DOCTEST_MSVC_SUPPRESS_WARNING(4548)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4986)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4350)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4668)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4365)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4774)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4820)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4625)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4626)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(5027)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(5026)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(4623)                                                            \
    DOCTEST_MSVC_SUPPRESS_WARNING(5039)

#define DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END DOCTEST_MSVC_SUPPRESS_WARNING_POP

// =================================================================================================
// == FEATURE DETECTION ============================================================================
// =================================================================================================

#if __cplusplus >= 201103L
#ifndef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif // DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#ifndef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif // DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#ifndef DOCTEST_CONFIG_WITH_NULLPTR
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif // DOCTEST_CONFIG_WITH_NULLPTR
#ifndef DOCTEST_CONFIG_WITH_LONG_LONG
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif // DOCTEST_CONFIG_WITH_LONG_LONG
#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif // DOCTEST_CONFIG_WITH_STATIC_ASSERT
#ifndef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // __cplusplus >= 201103L

// MSVC C++11 feature support table: https://msdn.microsoft.com/en-us/library/hh567368.aspx
// GCC C++11 feature support table: https://gcc.gnu.org/projects/cxx-status.html
// MSVC version table:
// MSVC++ 15.0 _MSC_VER == 1910 (Visual Studio 2017)
// MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)
// MSVC++ 12.0 _MSC_VER == 1800 (Visual Studio 2013)
// MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)
// MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)
// MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)
// MSVC++ 8.0  _MSC_VER == 1400 (Visual Studio 2005)

// deleted functions

#ifndef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#if DOCTEST_MSVC >= DOCTEST_COMPILER(18, 0, 0)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif // MSVC
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_FEATURE(cxx_deleted_functions)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif // clang
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 5, 0) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif // GCC
#endif // DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS

#if defined(DOCTEST_CONFIG_NO_DELETED_FUNCTIONS) && defined(DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS)
#undef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif // DOCTEST_CONFIG_NO_DELETED_FUNCTIONS

// rvalue references

#ifndef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#if DOCTEST_MSVC >= DOCTEST_COMPILER(16, 0, 0)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif // MSVC
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_FEATURE(cxx_rvalue_references)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif // clang
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 3, 0) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif // GCC
#endif // DOCTEST_CONFIG_WITH_RVALUE_REFERENCES

#if defined(DOCTEST_CONFIG_NO_RVALUE_REFERENCES) && defined(DOCTEST_CONFIG_WITH_RVALUE_REFERENCES)
#undef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif // DOCTEST_CONFIG_NO_RVALUE_REFERENCES

// nullptr

#ifndef DOCTEST_CONFIG_WITH_NULLPTR
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_FEATURE(cxx_nullptr)
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif // clang
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 6, 0) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif // GCC
#if DOCTEST_MSVC >= DOCTEST_COMPILER(16, 0, 0)
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif // MSVC
#endif // DOCTEST_CONFIG_WITH_NULLPTR

#if defined(DOCTEST_CONFIG_NO_NULLPTR) && defined(DOCTEST_CONFIG_WITH_NULLPTR)
#undef DOCTEST_CONFIG_WITH_NULLPTR
#endif // DOCTEST_CONFIG_NO_NULLPTR

// variadic macros

#ifndef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#if DOCTEST_MSVC >= DOCTEST_COMPILER(14, 0, 0) && !defined(__EDGE__)
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // MSVC
#if(DOCTEST_CLANG || DOCTEST_GCC >= DOCTEST_COMPILER(4, 1, 0)) &&                                  \
        defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // GCC and clang
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#if defined(DOCTEST_CONFIG_NO_VARIADIC_MACROS) && defined(DOCTEST_CONFIG_WITH_VARIADIC_MACROS)
#undef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // DOCTEST_CONFIG_NO_VARIADIC_MACROS

// long long

#ifndef DOCTEST_CONFIG_WITH_LONG_LONG
#if DOCTEST_MSVC >= DOCTEST_COMPILER(14, 0, 0)
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif // MSVC
#if(DOCTEST_CLANG || DOCTEST_GCC >= DOCTEST_COMPILER(4, 5, 0)) &&                                  \
        defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif // GCC and clang
#endif // DOCTEST_CONFIG_WITH_LONG_LONG

#if defined(DOCTEST_CONFIG_NO_LONG_LONG) && defined(DOCTEST_CONFIG_WITH_LONG_LONG)
#undef DOCTEST_CONFIG_WITH_LONG_LONG
#endif // DOCTEST_CONFIG_NO_LONG_LONG

// static_assert

#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#if DOCTEST_CLANG && DOCTEST_CLANG_HAS_FEATURE(cxx_static_assert)
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif // clang
#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 3, 0) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif // GCC
#if DOCTEST_MSVC >= DOCTEST_COMPILER(16, 0, 0)
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif // MSVC
#endif // DOCTEST_CONFIG_WITH_STATIC_ASSERT

#if defined(DOCTEST_CONFIG_NO_STATIC_ASSERT) && defined(DOCTEST_CONFIG_WITH_STATIC_ASSERT)
#undef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif // DOCTEST_CONFIG_NO_STATIC_ASSERT

// other stuff...

#if defined(DOCTEST_CONFIG_WITH_RVALUE_REFERENCES) || defined(DOCTEST_CONFIG_WITH_LONG_LONG) ||    \
        defined(DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS) || defined(DOCTEST_CONFIG_WITH_NULLPTR) ||  \
        defined(DOCTEST_CONFIG_WITH_VARIADIC_MACROS) || defined(DOCTEST_CONFIG_WITH_STATIC_ASSERT)
#define DOCTEST_NO_CPP11_COMPAT
#endif // c++11 stuff

#if defined(DOCTEST_NO_CPP11_COMPAT)
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat")
DOCTEST_CLANG_SUPPRESS_WARNING("-Wc++98-compat-pedantic")
#endif // DOCTEST_NO_CPP11_COMPAT

#if DOCTEST_MSVC && !defined(DOCTEST_CONFIG_WINDOWS_SEH)
#define DOCTEST_CONFIG_WINDOWS_SEH
#endif // MSVC
#if defined(DOCTEST_CONFIG_NO_WINDOWS_SEH) && defined(DOCTEST_CONFIG_WINDOWS_SEH)
#undef DOCTEST_CONFIG_WINDOWS_SEH
#endif // DOCTEST_CONFIG_NO_WINDOWS_SEH

#if !defined(_WIN32) && !defined(__QNX__) && !defined(DOCTEST_CONFIG_POSIX_SIGNALS)
#define DOCTEST_CONFIG_POSIX_SIGNALS
#endif // _WIN32
#if defined(DOCTEST_CONFIG_NO_POSIX_SIGNALS) && defined(DOCTEST_CONFIG_POSIX_SIGNALS)
#undef DOCTEST_CONFIG_POSIX_SIGNALS
#endif // DOCTEST_CONFIG_NO_POSIX_SIGNALS

#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
#if(DOCTEST_GCC || DOCTEST_CLANG) && !defined(__EXCEPTIONS)
#define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif // clang and gcc
// in MSVC _HAS_EXCEPTIONS is defined in a header instead of as a project define
// so we can't do the automatic detection for MSVC without including some header
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
#define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS

#if defined(DOCTEST_CONFIG_NO_EXCEPTIONS) && !defined(DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS)
#define DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
#endif // DOCTEST_CONFIG_NO_EXCEPTIONS && !DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS

#if defined(DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN) && !defined(DOCTEST_CONFIG_IMPLEMENT)
#define DOCTEST_CONFIG_IMPLEMENT
#endif // DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#if defined _WIN32 || defined __CYGWIN__
#if DOCTEST_MSVC
#define DOCTEST_SYMBOL_EXPORT __declspec(dllexport)
#define DOCTEST_SYMBOL_IMPORT __declspec(dllimport)
#else // MSVC
#define DOCTEST_SYMBOL_EXPORT __attribute__((dllexport))
#define DOCTEST_SYMBOL_IMPORT __attribute__((dllimport))
#endif // MSVC
#else  // _WIN32
#define DOCTEST_SYMBOL_EXPORT __attribute__((visibility("default")))
#define DOCTEST_SYMBOL_IMPORT
#endif // _WIN32

#ifdef DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#ifdef DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_INTERFACE DOCTEST_SYMBOL_EXPORT
#else // DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_INTERFACE DOCTEST_SYMBOL_IMPORT
#endif // DOCTEST_CONFIG_IMPLEMENT
#else  // DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#define DOCTEST_INTERFACE
#endif // DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL

#if DOCTEST_MSVC
#define DOCTEST_NOINLINE __declspec(noinline)
#define DOCTEST_UNUSED
#define DOCTEST_ALIGNMENT(x)
#else // MSVC
#define DOCTEST_NOINLINE __attribute__((noinline))
#define DOCTEST_UNUSED __attribute__((unused))
#define DOCTEST_ALIGNMENT(x) __attribute__((aligned(x)))
#endif // MSVC

#ifndef DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK
#define DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK 5
#endif // DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK

// =================================================================================================
// == FEATURE DETECTION END ========================================================================
// =================================================================================================

// internal macros for string concatenation and anonymous variable name generation
#define DOCTEST_CAT_IMPL(s1, s2) s1##s2
#define DOCTEST_CAT(s1, s2) DOCTEST_CAT_IMPL(s1, s2)
#ifdef __COUNTER__ // not standard and may be missing for some compilers
#define DOCTEST_ANONYMOUS(x) DOCTEST_CAT(x, __COUNTER__)
#else // __COUNTER__
#define DOCTEST_ANONYMOUS(x) DOCTEST_CAT(x, __LINE__)
#endif // __COUNTER__

// macro for making a string out of an identifier
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TOSTR_IMPL(...) #__VA_ARGS__
#define DOCTEST_TOSTR(...) DOCTEST_TOSTR_IMPL(__VA_ARGS__)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TOSTR_IMPL(x) #x
#define DOCTEST_TOSTR(x) DOCTEST_TOSTR_IMPL(x)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

// counts the number of elements in a C string
#define DOCTEST_COUNTOF(x) (sizeof(x) / sizeof(x[0]))

#ifndef DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE
#define DOCTEST_REF_WRAP(x) x&
#else // DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE
#define DOCTEST_REF_WRAP(x) x
#endif // DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE

// not using __APPLE__ because... this is how Catch does it
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED)
#define DOCTEST_PLATFORM_MAC
#elif defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
#define DOCTEST_PLATFORM_IPHONE
#elif defined(_WIN32)
#define DOCTEST_PLATFORM_WINDOWS
#else
#define DOCTEST_PLATFORM_LINUX
#endif

#define DOCTEST_GLOBAL_NO_WARNINGS(var)                                                            \
    DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wglobal-constructors") static int var DOCTEST_UNUSED
#define DOCTEST_GLOBAL_NO_WARNINGS_END() DOCTEST_CLANG_SUPPRESS_WARNING_POP

// should probably take a look at https://github.com/scottt/debugbreak
#ifdef DOCTEST_PLATFORM_MAC
#define DOCTEST_BREAK_INTO_DEBUGGER() __asm__("int $3\n" : :)
#elif DOCTEST_MSVC
#define DOCTEST_BREAK_INTO_DEBUGGER() __debugbreak()
#elif defined(__MINGW32__)
extern "C" __declspec(dllimport) void __stdcall DebugBreak();
#define DOCTEST_BREAK_INTO_DEBUGGER() ::DebugBreak()
#else // linux
#define DOCTEST_BREAK_INTO_DEBUGGER() ((void)0)
#endif // linux

#if DOCTEST_CLANG
// to detect if libc++ is being used with clang (the _LIBCPP_VERSION identifier)
#include <ciso646>
#endif // clang

#ifdef _LIBCPP_VERSION
// not forward declaring ostream for libc++ because I had some problems (inline namespaces vs c++98)
// so the <iosfwd> header is used - also it is very light and doesn't drag a ton of stuff
#include <iosfwd>
#else // _LIBCPP_VERSION
#ifndef DOCTEST_CONFIG_USE_IOSFWD
namespace std
{
template <class charT>
struct char_traits;
template <>
struct char_traits<char>;
template <class charT, class traits>
class basic_ostream;
typedef basic_ostream<char, char_traits<char> > ostream;
} // namespace std
#else // DOCTEST_CONFIG_USE_IOSFWD
#include <iosfwd>
#endif // DOCTEST_CONFIG_USE_IOSFWD
#endif // _LIBCPP_VERSION

// static assert macro - because of the c++98 support requires that the message is an
// identifier (no spaces and not a C string) - example without quotes: I_am_a_message
// taken from here: http://stackoverflow.com/a/1980156/3162383
#ifdef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#define DOCTEST_STATIC_ASSERT(expression, message) static_assert(expression, #message)
#else // DOCTEST_CONFIG_WITH_STATIC_ASSERT
#define DOCTEST_STATIC_ASSERT(expression, message)                                                 \
    struct DOCTEST_CAT(__static_assertion_at_line_, __LINE__)                                      \
    {                                                                                              \
        doctest::detail::static_assert_impl::StaticAssertion<static_cast<bool>((expression))>      \
                DOCTEST_CAT(DOCTEST_CAT(DOCTEST_CAT(STATIC_ASSERTION_FAILED_AT_LINE_, __LINE__),   \
                                        _),                                                        \
                            message);                                                              \
    };                                                                                             \
    typedef doctest::detail::static_assert_impl::StaticAssertionTest<static_cast<int>(             \
            sizeof(DOCTEST_CAT(__static_assertion_at_line_, __LINE__)))>                           \
            DOCTEST_CAT(__static_assertion_test_at_line_, __LINE__)
#endif // DOCTEST_CONFIG_WITH_STATIC_ASSERT

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
#ifdef _LIBCPP_VERSION
#include <cstddef>
#else  // _LIBCPP_VERSION
namespace std
{
typedef decltype(nullptr) nullptr_t;
}
#endif // _LIBCPP_VERSION
#endif // DOCTEST_CONFIG_WITH_NULLPTR

#ifndef DOCTEST_CONFIG_DISABLE

#ifdef DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include <type_traits>
#endif // DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS

namespace doctest
{
namespace detail
{
    struct TestSuite
    {
        const char* m_test_suite;
        const char* m_description;
        bool        m_skip;
        bool        m_may_fail;
        bool        m_should_fail;
        int         m_expected_failures;
        double      m_timeout;

        TestSuite& operator*(const char* in) {
            m_test_suite = in;
            // clear state
            m_description       = 0;
            m_skip              = false;
            m_may_fail          = false;
            m_should_fail       = false;
            m_expected_failures = 0;
            m_timeout           = 0;
            return *this;
        }

        template <typename T>
        TestSuite& operator*(const T& in) {
            in.fill(*this);
            return *this;
        }
    };
} // namespace detail
} // namespace doctest

// in a separate namespace outside of doctest because the DOCTEST_TEST_SUITE macro
// introduces an anonymous namespace in which getCurrentTestSuite gets overridden
namespace doctest_detail_test_suite_ns
{
DOCTEST_INTERFACE doctest::detail::TestSuite& getCurrentTestSuite();
} // namespace doctest_detail_test_suite_ns

#endif // DOCTEST_CONFIG_DISABLE

namespace doctest
{
// A 24 byte string class (can be as small as 17 for x64 and 13 for x86) that can hold strings with length
// of up to 23 chars on the stack before going on the heap - the last byte of the buffer is used for:
// - "is small" bit - the highest bit - if "0" then it is small - otherwise its "1" (128)
// - if small - capacity left before going on the heap - using the lowest 5 bits
// - if small - 2 bits are left unused - the second and third highest ones
// - if small - acts as a null terminator if strlen() is 23 (24 including the null terminator)
//              and the "is small" bit remains "0" ("as well as the capacity left") so its OK
// Idea taken from this lecture about the string implementation of facebook/folly - fbstring
// https://www.youtube.com/watch?v=kPR8h4-qZdk
// TODO:
// - optimizations - like not deleting memory unnecessarily in operator= and etc.
// - resize/reserve/clear
// - substr
// - replace
// - back/front
// - iterator stuff
// - find & friends
// - push_back/pop_back
// - assign/insert/erase
// - relational operators as free functions - taking const char* as one of the params
class DOCTEST_INTERFACE String
{
    static const unsigned len  = 24;      //!OCLINT avoid private static members
    static const unsigned last = len - 1; //!OCLINT avoid private static members

    struct view // len should be more than sizeof(view) - because of the final byte for flags
    {
        char*    ptr;
        unsigned size;
        unsigned capacity;
    };

    union
    {
        char buf[len];
        view data;
    };

    void copy(const String& other);

    void setOnHeap() { *reinterpret_cast<unsigned char*>(&buf[last]) = 128; }
    void setLast(unsigned in = last) { buf[last] = char(in); }

public:
    String() {
        buf[0] = '\0';
        setLast();
    }

    String(const char* in);

    String(const String& other) { copy(other); }

    ~String() {
        if(!isOnStack())
            delete[] data.ptr;
    }

    // GCC 4.9/5/6 report Wstrict-overflow when optimizations are ON and it got inlined in the vector class somewhere...
    // see commit 574ef95f0cd379118be5011704664e4b5351f1e0 and build https://travis-ci.org/onqtam/doctest/builds/230671611
    DOCTEST_NOINLINE String& operator=(const String& other) {
        if(this != &other) {
            if(!isOnStack())
                delete[] data.ptr;

            copy(other);
        }

        return *this;
    }
    String& operator+=(const String& other);

    String operator+(const String& other) const { return String(*this) += other; }

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
    String(String&& other);
    String& operator=(String&& other);
#endif // DOCTEST_CONFIG_WITH_RVALUE_REFERENCES

    bool isOnStack() const { return (buf[last] & 128) == 0; }

    char operator[](unsigned i) const { return const_cast<String*>(this)->operator[](i); } // NOLINT
    char& operator[](unsigned i) {
        if(isOnStack())
            return reinterpret_cast<char*>(buf)[i];
        return data.ptr[i];
    }

    const char* c_str() const { return const_cast<String*>(this)->c_str(); } // NOLINT
    char*       c_str() {
        if(isOnStack())
            return reinterpret_cast<char*>(buf);
        return data.ptr;
    }

    unsigned size() const {
        if(isOnStack())
            return last - (unsigned(buf[last]) & 31); // using "last" would work only if "len" is 32
        return data.size;
    }

    unsigned capacity() const {
        if(isOnStack())
            return len;
        return data.capacity;
    }

    int compare(const char* other, bool no_case = false) const;
    int compare(const String& other, bool no_case = false) const;
};

// clang-format off
inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }
inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }
inline bool operator< (const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }
inline bool operator> (const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }
inline bool operator<=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) < 0 : true; }
inline bool operator>=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) > 0 : true; }
// clang-format on

DOCTEST_INTERFACE std::ostream& operator<<(std::ostream& stream, const String& in);

namespace detail
{
#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
    namespace static_assert_impl
    {
        template <bool>
        struct StaticAssertion;

        template <>
        struct StaticAssertion<true>
        {};

        template <int i>
        struct StaticAssertionTest
        {};
    }  // namespace static_assert_impl
#endif // DOCTEST_CONFIG_WITH_STATIC_ASSERT

    template <bool CONDITION, typename TYPE = void>
    struct enable_if
    {};

    template <typename TYPE>
    struct enable_if<true, TYPE>
    { typedef TYPE type; };

    template <typename T>
    struct deferred_false
    // cppcheck-suppress unusedStructMember
    { static const bool value = false; };

    // to silence the warning "-Wzero-as-null-pointer-constant" only for gcc 5 for the Approx template ctor - pragmas don't work for it...
    inline void* getNull() { return 0; }

    namespace has_insertion_operator_impl
    {
        typedef char no;
        typedef char yes[2];

        struct any_t
        {
            template <typename T>
            // cppcheck-suppress noExplicitConstructor
            any_t(const DOCTEST_REF_WRAP(T));
        };

        yes& testStreamable(std::ostream&);
        no   testStreamable(no);

        no operator<<(const std::ostream&, const any_t&);

        template <typename T>
        struct has_insertion_operator
        {
            static std::ostream& s;
            static const DOCTEST_REF_WRAP(T) t;
            static const bool value = sizeof(testStreamable(s << t)) == sizeof(yes);
        };
    } // namespace has_insertion_operator_impl

    template <typename T>
    struct has_insertion_operator : has_insertion_operator_impl::has_insertion_operator<T>
    {};

    DOCTEST_INTERFACE void     my_memcpy(void* dest, const void* src, unsigned num);
    DOCTEST_INTERFACE unsigned my_strlen(const char* in);

    DOCTEST_INTERFACE std::ostream* createStream();
    DOCTEST_INTERFACE String getStreamResult(std::ostream*);
    DOCTEST_INTERFACE void   freeStream(std::ostream*);

    template <bool C>
    struct StringMakerBase
    {
        template <typename T>
        static String convert(const DOCTEST_REF_WRAP(T)) {
            return "{?}";
        }
    };

    template <>
    struct StringMakerBase<true>
    {
        template <typename T>
        static String convert(const DOCTEST_REF_WRAP(T) in) {
            std::ostream* stream = createStream();
            *stream << in;
            String result = getStreamResult(stream);
            freeStream(stream);
            return result;
        }
    };

    DOCTEST_INTERFACE String rawMemoryToString(const void* object, unsigned size);

    template <typename T>
    String rawMemoryToString(const DOCTEST_REF_WRAP(T) object) {
        return rawMemoryToString(&object, sizeof(object));
    }

    class NullType
    {
    };

    template <class T, class U>
    struct Typelist
    {
        typedef T Head;
        typedef U Tail;
    };

    // type of recursive function
    template <class TList, class Callable>
    struct ForEachType;

    // Recursion rule
    template <class Head, class Tail, class Callable>
    struct ForEachType<Typelist<Head, Tail>, Callable> : public ForEachType<Tail, Callable>
    {
        enum
        {
            value = 1 + ForEachType<Tail, Callable>::value
        };

        explicit ForEachType(Callable& callable)
                : ForEachType<Tail, Callable>(callable) {
#if DOCTEST_MSVC && DOCTEST_MSVC < DOCTEST_COMPILER(19, 10, 0)
            callable.operator()<value, Head>();
#else  // MSVC
            callable.template operator()<value, Head>();
#endif // MSVC
        }
    };

    // Recursion end
    template <class Head, class Callable>
    struct ForEachType<Typelist<Head, NullType>, Callable>
    {
    public:
        enum
        {
            value = 0
        };

        explicit ForEachType(Callable& callable) {
#if DOCTEST_MSVC && DOCTEST_MSVC < DOCTEST_COMPILER(19, 10, 0)
            callable.operator()<value, Head>();
#else  // MSVC
            callable.template operator()<value, Head>();
#endif // MSVC
        }
    };

    template <typename T>
    const char* type_to_string() {
        return "<>";
    }
} // namespace detail

template <typename T1 = detail::NullType, typename T2 = detail::NullType,
          typename T3 = detail::NullType, typename T4 = detail::NullType,
          typename T5 = detail::NullType, typename T6 = detail::NullType,
          typename T7 = detail::NullType, typename T8 = detail::NullType,
          typename T9 = detail::NullType, typename T10 = detail::NullType,
          typename T11 = detail::NullType, typename T12 = detail::NullType,
          typename T13 = detail::NullType, typename T14 = detail::NullType,
          typename T15 = detail::NullType, typename T16 = detail::NullType,
          typename T17 = detail::NullType, typename T18 = detail::NullType,
          typename T19 = detail::NullType, typename T20 = detail::NullType,
          typename T21 = detail::NullType, typename T22 = detail::NullType,
          typename T23 = detail::NullType, typename T24 = detail::NullType,
          typename T25 = detail::NullType, typename T26 = detail::NullType,
          typename T27 = detail::NullType, typename T28 = detail::NullType,
          typename T29 = detail::NullType, typename T30 = detail::NullType,
          typename T31 = detail::NullType, typename T32 = detail::NullType,
          typename T33 = detail::NullType, typename T34 = detail::NullType,
          typename T35 = detail::NullType, typename T36 = detail::NullType,
          typename T37 = detail::NullType, typename T38 = detail::NullType,
          typename T39 = detail::NullType, typename T40 = detail::NullType,
          typename T41 = detail::NullType, typename T42 = detail::NullType,
          typename T43 = detail::NullType, typename T44 = detail::NullType,
          typename T45 = detail::NullType, typename T46 = detail::NullType,
          typename T47 = detail::NullType, typename T48 = detail::NullType,
          typename T49 = detail::NullType, typename T50 = detail::NullType,
          typename T51 = detail::NullType, typename T52 = detail::NullType,
          typename T53 = detail::NullType, typename T54 = detail::NullType,
          typename T55 = detail::NullType, typename T56 = detail::NullType,
          typename T57 = detail::NullType, typename T58 = detail::NullType,
          typename T59 = detail::NullType, typename T60 = detail::NullType>
struct Types
{
private:
    typedef typename Types<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17,
                           T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31,
                           T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
                           T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59,
                           T60>::Result TailResult;

public:
    typedef detail::Typelist<T1, TailResult> Result;
};

template <>
struct Types<>
{ typedef detail::NullType Result; };

template <typename T>
struct StringMaker : detail::StringMakerBase<detail::has_insertion_operator<T>::value>
{};

template <typename T>
struct StringMaker<T*>
{
    template <typename U>
    static String convert(U* p) {
        if(p)
            return detail::rawMemoryToString(p);
        return "NULL";
    }
};

template <typename R, typename C>
struct StringMaker<R C::*>
{
    static String convert(R C::*p) {
        if(p)
            return detail::rawMemoryToString(p);
        return "NULL";
    }
};

template <typename T>
String toString(const DOCTEST_REF_WRAP(T) value) {
    return StringMaker<T>::convert(value);
}

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
DOCTEST_INTERFACE String toString(char* in);
DOCTEST_INTERFACE String toString(const char* in);
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
DOCTEST_INTERFACE String toString(bool in);
DOCTEST_INTERFACE String toString(float in);
DOCTEST_INTERFACE String toString(double in);
DOCTEST_INTERFACE String toString(double long in);

DOCTEST_INTERFACE String toString(char in);
DOCTEST_INTERFACE String toString(char signed in);
DOCTEST_INTERFACE String toString(char unsigned in);
DOCTEST_INTERFACE String toString(int short in);
DOCTEST_INTERFACE String toString(int short unsigned in);
DOCTEST_INTERFACE String toString(int in);
DOCTEST_INTERFACE String toString(int unsigned in);
DOCTEST_INTERFACE String toString(int long in);
DOCTEST_INTERFACE String toString(int long unsigned in);

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
DOCTEST_INTERFACE String toString(int long long in);
DOCTEST_INTERFACE String toString(int long long unsigned in);
#endif // DOCTEST_CONFIG_WITH_LONG_LONG

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
DOCTEST_INTERFACE String toString(std::nullptr_t in);
#endif // DOCTEST_CONFIG_WITH_NULLPTR

class DOCTEST_INTERFACE Approx
{
public:
    explicit Approx(double value);

    Approx operator()(double value) const {
        Approx approx(value);
        approx.epsilon(m_epsilon);
        approx.scale(m_scale);
        return approx;
    }

#ifdef DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
    template <typename T>
    explicit Approx(const T& value,
                    typename detail::enable_if<std::is_constructible<double, T>::value>::type* =
                            static_cast<T*>(detail::getNull())) {
        *this = Approx(static_cast<double>(value));
    }
#endif // DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS

    // clang-format off
    // overloads for double - the first one is necessary as it is in the implementation part of doctest
    // as for the others - keeping them for potentially faster compile times
    DOCTEST_INTERFACE friend bool operator==(double lhs, Approx const& rhs);
    friend bool operator==(Approx const& lhs, double rhs) { return operator==(rhs, lhs); }
    friend bool operator!=(double lhs, Approx const& rhs) { return !operator==(lhs, rhs); }
    friend bool operator!=(Approx const& lhs, double rhs) { return !operator==(rhs, lhs); }
    friend bool operator<=(double lhs, Approx const& rhs) { return lhs < rhs.m_value || lhs == rhs; }
    friend bool operator<=(Approx const& lhs, double rhs) { return lhs.m_value < rhs || lhs == rhs; }
    friend bool operator>=(double lhs, Approx const& rhs) { return lhs > rhs.m_value || lhs == rhs; }
    friend bool operator>=(Approx const& lhs, double rhs) { return lhs.m_value > rhs || lhs == rhs; }
    friend bool operator< (double lhs, Approx const& rhs) { return lhs < rhs.m_value && lhs != rhs; }
    friend bool operator< (Approx const& lhs, double rhs) { return lhs.m_value < rhs && lhs != rhs; }
    friend bool operator> (double lhs, Approx const& rhs) { return lhs > rhs.m_value && lhs != rhs; }
    friend bool operator> (Approx const& lhs, double rhs) { return lhs.m_value > rhs && lhs != rhs; }

#ifdef DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#define DOCTEST_APPROX_PREFIX \
    template <typename T> friend typename detail::enable_if<std::is_constructible<double, T>::value, bool>::type

    DOCTEST_APPROX_PREFIX operator==(const T& lhs, const Approx& rhs) { return operator==(double(lhs), rhs); }
    DOCTEST_APPROX_PREFIX operator==(const Approx& lhs, const T& rhs) { return operator==(rhs, lhs); }
    DOCTEST_APPROX_PREFIX operator!=(const T& lhs, const Approx& rhs) { return !operator==(lhs, rhs); }
    DOCTEST_APPROX_PREFIX operator!=(const Approx& lhs, const T& rhs) { return !operator==(rhs, lhs); }
    DOCTEST_APPROX_PREFIX operator<=(const T& lhs, const Approx& rhs) { return double(lhs) < rhs.m_value || lhs == rhs; }
    DOCTEST_APPROX_PREFIX operator<=(const Approx& lhs, const T& rhs) { return lhs.m_value < double(rhs) || lhs == rhs; }
    DOCTEST_APPROX_PREFIX operator>=(const T& lhs, const Approx& rhs) { return double(lhs) > rhs.m_value || lhs == rhs; }
    DOCTEST_APPROX_PREFIX operator>=(const Approx& lhs, const T& rhs) { return lhs.m_value > double(rhs) || lhs == rhs; }
    DOCTEST_APPROX_PREFIX operator< (const T& lhs, const Approx& rhs) { return double(lhs) < rhs.m_value && lhs != rhs; }
    DOCTEST_APPROX_PREFIX operator< (const Approx& lhs, const T& rhs) { return lhs.m_value < double(rhs) && lhs != rhs; }
    DOCTEST_APPROX_PREFIX operator> (const T& lhs, const Approx& rhs) { return double(lhs) > rhs.m_value && lhs != rhs; }
    DOCTEST_APPROX_PREFIX operator> (const Approx& lhs, const T& rhs) { return lhs.m_value > double(rhs) && lhs != rhs; }
#undef DOCTEST_APPROX_PREFIX
#endif // DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS

    // clang-format on

    Approx& epsilon(double newEpsilon) {
        m_epsilon = newEpsilon;
        return *this;
    }

#ifdef DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
    template <typename T>
    typename detail::enable_if<std::is_constructible<double, T>::value, Approx&>::type epsilon(
            const T& newEpsilon) {
        m_epsilon = static_cast<double>(newEpsilon);
        return *this;
    }
#endif //  DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS

    Approx& scale(double newScale) {
        m_scale = newScale;
        return *this;
    }

#ifdef DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
    template <typename T>
    typename detail::enable_if<std::is_constructible<double, T>::value, Approx&>::type scale(
            const T& newScale) {
        m_scale = static_cast<double>(newScale);
        return *this;
    }
#endif // DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS

    String toString() const;

private:
    double m_epsilon;
    double m_scale;
    double m_value;
};

template <>
inline String toString<Approx>(const DOCTEST_REF_WRAP(Approx) value) {
    return value.toString();
}

#if !defined(DOCTEST_CONFIG_DISABLE)

namespace detail
{
    // the function type this library works with
    typedef void (*funcType)();

    namespace assertType
    {
        enum Enum
        {
            // macro traits

            is_warn    = 1,
            is_check   = 2,
            is_require = 4,

            is_throws    = 8,
            is_throws_as = 16,
            is_nothrow   = 32,

            is_fast  = 64, // not checked anywhere - used just to distinguish the types
            is_false = 128,
            is_unary = 256,

            is_eq = 512,
            is_ne = 1024,

            is_lt = 2048,
            is_gt = 4096,

            is_ge = 8192,
            is_le = 16384,

            // macro types

            DT_WARN    = is_warn,
            DT_CHECK   = is_check,
            DT_REQUIRE = is_require,

            DT_WARN_FALSE    = is_false | is_warn,
            DT_CHECK_FALSE   = is_false | is_check,
            DT_REQUIRE_FALSE = is_false | is_require,

            DT_WARN_THROWS    = is_throws | is_warn,
            DT_CHECK_THROWS   = is_throws | is_check,
            DT_REQUIRE_THROWS = is_throws | is_require,

            DT_WARN_THROWS_AS    = is_throws_as | is_warn,
            DT_CHECK_THROWS_AS   = is_throws_as | is_check,
            DT_REQUIRE_THROWS_AS = is_throws_as | is_require,

            DT_WARN_NOTHROW    = is_nothrow | is_warn,
            DT_CHECK_NOTHROW   = is_nothrow | is_check,
            DT_REQUIRE_NOTHROW = is_nothrow | is_require,

            DT_WARN_EQ    = is_eq | is_warn,
            DT_CHECK_EQ   = is_eq | is_check,
            DT_REQUIRE_EQ = is_eq | is_require,

            DT_WARN_NE    = is_ne | is_warn,
            DT_CHECK_NE   = is_ne | is_check,
            DT_REQUIRE_NE = is_ne | is_require,

            DT_WARN_GT    = is_gt | is_warn,
            DT_CHECK_GT   = is_gt | is_check,
            DT_REQUIRE_GT = is_gt | is_require,

            DT_WARN_LT    = is_lt | is_warn,
            DT_CHECK_LT   = is_lt | is_check,
            DT_REQUIRE_LT = is_lt | is_require,

            DT_WARN_GE    = is_ge | is_warn,
            DT_CHECK_GE   = is_ge | is_check,
            DT_REQUIRE_GE = is_ge | is_require,

            DT_WARN_LE    = is_le | is_warn,
            DT_CHECK_LE   = is_le | is_check,
            DT_REQUIRE_LE = is_le | is_require,

            DT_WARN_UNARY    = is_unary | is_warn,
            DT_CHECK_UNARY   = is_unary | is_check,
            DT_REQUIRE_UNARY = is_unary | is_require,

            DT_WARN_UNARY_FALSE    = is_false | is_unary | is_warn,
            DT_CHECK_UNARY_FALSE   = is_false | is_unary | is_check,
            DT_REQUIRE_UNARY_FALSE = is_false | is_unary | is_require,

            DT_FAST_WARN_EQ    = is_fast | is_eq | is_warn,
            DT_FAST_CHECK_EQ   = is_fast | is_eq | is_check,
            DT_FAST_REQUIRE_EQ = is_fast | is_eq | is_require,

            DT_FAST_WARN_NE    = is_fast | is_ne | is_warn,
            DT_FAST_CHECK_NE   = is_fast | is_ne | is_check,
            DT_FAST_REQUIRE_NE = is_fast | is_ne | is_require,

            DT_FAST_WARN_GT    = is_fast | is_gt | is_warn,
            DT_FAST_CHECK_GT   = is_fast | is_gt | is_check,
            DT_FAST_REQUIRE_GT = is_fast | is_gt | is_require,

            DT_FAST_WARN_LT    = is_fast | is_lt | is_warn,
            DT_FAST_CHECK_LT   = is_fast | is_lt | is_check,
            DT_FAST_REQUIRE_LT = is_fast | is_lt | is_require,

            DT_FAST_WARN_GE    = is_fast | is_ge | is_warn,
            DT_FAST_CHECK_GE   = is_fast | is_ge | is_check,
            DT_FAST_REQUIRE_GE = is_fast | is_ge | is_require,

            DT_FAST_WARN_LE    = is_fast | is_le | is_warn,
            DT_FAST_CHECK_LE   = is_fast | is_le | is_check,
            DT_FAST_REQUIRE_LE = is_fast | is_le | is_require,

            DT_FAST_WARN_UNARY    = is_fast | is_unary | is_warn,
            DT_FAST_CHECK_UNARY   = is_fast | is_unary | is_check,
            DT_FAST_REQUIRE_UNARY = is_fast | is_unary | is_require,

            DT_FAST_WARN_UNARY_FALSE    = is_fast | is_false | is_unary | is_warn,
            DT_FAST_CHECK_UNARY_FALSE   = is_fast | is_false | is_unary | is_check,
            DT_FAST_REQUIRE_UNARY_FALSE = is_fast | is_false | is_unary | is_require
        };
    } // namespace assertType

    DOCTEST_INTERFACE const char* getAssertString(assertType::Enum val);

    // clang-format off
    template<class T>               struct decay_array       { typedef T type; };
    template<class T, unsigned N>   struct decay_array<T[N]> { typedef T* type; };
    template<class T>               struct decay_array<T[]>  { typedef T* type; };

    template<class T>   struct not_char_pointer              { enum { value = 1 }; };
    template<>          struct not_char_pointer<char*>       { enum { value = 0 }; };
    template<>          struct not_char_pointer<const char*> { enum { value = 0 }; };

    template<class T> struct can_use_op : not_char_pointer<typename decay_array<T>::type> {};
    // clang-format on

    struct TestFailureException
    {
    };

    DOCTEST_INTERFACE bool checkIfShouldThrow(assertType::Enum assert_type);
    DOCTEST_INTERFACE void fastAssertThrowIfFlagSet(int flags);
    DOCTEST_INTERFACE void throwException();

    struct TestAccessibleContextState
    {
        bool no_throw; // to skip exceptions-related assertion macros
        bool success;  // include successful assertions in output
    };

    struct ContextState;

    DOCTEST_INTERFACE TestAccessibleContextState* getTestsContextState();

    struct DOCTEST_INTERFACE SubcaseSignature
    {
        const char* m_name;
        const char* m_file;
        int         m_line;

        SubcaseSignature(const char* name, const char* file, int line)
                : m_name(name)
                , m_file(file)
                , m_line(line) {}

        bool operator<(const SubcaseSignature& other) const;
    };

    // cppcheck-suppress copyCtorAndEqOperator
    struct DOCTEST_INTERFACE Subcase
    {
        SubcaseSignature m_signature;
        bool             m_entered;

        Subcase(const char* name, const char* file, int line);
        Subcase(const Subcase& other);
        ~Subcase();

        operator bool() const { return m_entered; }
    };

    template <typename L, typename R>
    String stringifyBinaryExpr(const DOCTEST_REF_WRAP(L) lhs, const char* op,
                               const DOCTEST_REF_WRAP(R) rhs) {
        return toString(lhs) + op + toString(rhs);
    }

    struct DOCTEST_INTERFACE Result
    {
        bool   m_passed;
        String m_decomposition;

        ~Result();

        DOCTEST_NOINLINE Result(bool passed = false, const String& decomposition = String())
                : m_passed(passed)
                , m_decomposition(decomposition) {}

        DOCTEST_NOINLINE Result(const Result& other)
                : m_passed(other.m_passed)
                , m_decomposition(other.m_decomposition) {}

        Result& operator=(const Result& other);

        operator bool() { return !m_passed; }

        // clang-format off
        // forbidding some expressions based on this table: http://en.cppreference.com/w/cpp/language/operator_precedence
        template <typename R> Result& operator&  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator^  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator|  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator&& (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator|| (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator== (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator!= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator<  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator>  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator<= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator>= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator=  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator+= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator-= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator*= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator/= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator%= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator<<=(const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator>>=(const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator&= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator^= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        template <typename R> Result& operator|= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
        // clang-format on
    };

#ifndef DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

    DOCTEST_CLANG_SUPPRESS_WARNING_PUSH
    DOCTEST_CLANG_SUPPRESS_WARNING("-Wsign-conversion")
    DOCTEST_CLANG_SUPPRESS_WARNING("-Wsign-compare")
    //DOCTEST_CLANG_SUPPRESS_WARNING("-Wdouble-promotion")
    //DOCTEST_CLANG_SUPPRESS_WARNING("-Wconversion")
    //DOCTEST_CLANG_SUPPRESS_WARNING("-Wfloat-equal")

    DOCTEST_GCC_SUPPRESS_WARNING_PUSH
    DOCTEST_GCC_SUPPRESS_WARNING("-Wsign-conversion")
    DOCTEST_GCC_SUPPRESS_WARNING("-Wsign-compare")
    //#if DOCTEST_GCC >= DOCTEST_COMPILER(4, 6, 0)
    //DOCTEST_GCC_SUPPRESS_WARNING("-Wdouble-promotion")
    //#endif // GCC
    //DOCTEST_GCC_SUPPRESS_WARNING("-Wconversion")
    //DOCTEST_GCC_SUPPRESS_WARNING("-Wfloat-equal")

    DOCTEST_MSVC_SUPPRESS_WARNING_PUSH
    // http://stackoverflow.com/questions/39479163 what's the difference between C4018 and C4389
    DOCTEST_MSVC_SUPPRESS_WARNING(4388) // signed/unsigned mismatch
    DOCTEST_MSVC_SUPPRESS_WARNING(4389) // 'operator' : signed/unsigned mismatch
    DOCTEST_MSVC_SUPPRESS_WARNING(4018) // 'expression' : signed/unsigned mismatch
    //DOCTEST_MSVC_SUPPRESS_WARNING(4805) // 'operation' : unsafe mix of type 'type' and type 'type' in operation

#endif // DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

// clang-format off
#ifndef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_COMPARISON_RETURN_TYPE bool
#else // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_COMPARISON_RETURN_TYPE typename enable_if<can_use_op<L>::value || can_use_op<R>::value, bool>::type
    inline bool eq(const char* lhs, const char* rhs) { return String(lhs) == String(rhs); }
    inline bool ne(const char* lhs, const char* rhs) { return String(lhs) != String(rhs); }
    inline bool lt(const char* lhs, const char* rhs) { return String(lhs) <  String(rhs); }
    inline bool gt(const char* lhs, const char* rhs) { return String(lhs) >  String(rhs); }
    inline bool le(const char* lhs, const char* rhs) { return String(lhs) <= String(rhs); }
    inline bool ge(const char* lhs, const char* rhs) { return String(lhs) >= String(rhs); }
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING

    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE eq(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs == rhs; }
    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE ne(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs != rhs; }
    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE lt(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs <  rhs; }
    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE gt(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs >  rhs; }
    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE le(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs <= rhs; }
    template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE ge(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs >= rhs; }
    // clang-format on

#ifndef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_CMP_EQ(l, r) l == r
#define DOCTEST_CMP_NE(l, r) l != r
#define DOCTEST_CMP_GT(l, r) l > r
#define DOCTEST_CMP_LT(l, r) l < r
#define DOCTEST_CMP_GE(l, r) l >= r
#define DOCTEST_CMP_LE(l, r) l <= r
#else // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_CMP_EQ(l, r) eq(l, r)
#define DOCTEST_CMP_NE(l, r) ne(l, r)
#define DOCTEST_CMP_GT(l, r) gt(l, r)
#define DOCTEST_CMP_LT(l, r) lt(l, r)
#define DOCTEST_CMP_GE(l, r) ge(l, r)
#define DOCTEST_CMP_LE(l, r) le(l, r)
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING

#define DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(op, op_str, op_macro)                              \
    template <typename R>                                                                          \
    DOCTEST_NOINLINE Result operator op(const DOCTEST_REF_WRAP(R) rhs) {                           \
        bool res = op_macro(lhs, rhs);                                                             \
        if(m_assert_type & assertType::is_false)                                                   \
            res = !res;                                                                            \
        if(!res || doctest::detail::getTestsContextState()->success)                               \
            return Result(res, stringifyBinaryExpr(lhs, op_str, rhs));                             \
        return Result(res);                                                                        \
    }

#define DOCTEST_FORBIT_EXPRESSION(op)                                                              \
    template <typename R>                                                                          \
    Expression_lhs& operator op(const R&) {                                                        \
        DOCTEST_STATIC_ASSERT(deferred_false<R>::value,                                            \
                              Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison);         \
        return *this;                                                                              \
    }

    template <typename L>
    // cppcheck-suppress copyCtorAndEqOperator
    struct Expression_lhs
    {
        L                lhs;
        assertType::Enum m_assert_type;

        explicit Expression_lhs(L in, assertType::Enum assert_type)
                : lhs(in)
                , m_assert_type(assert_type) {}

        DOCTEST_NOINLINE operator Result() {
            bool res = !!lhs;
            if(m_assert_type & assertType::is_false) //!OCLINT bitwise operator in conditional
                res = !res;

            if(!res || getTestsContextState()->success)
                return Result(res, toString(lhs));
            return Result(res);
        }

        // clang-format off
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(==, " == ", DOCTEST_CMP_EQ) //!OCLINT bitwise operator in conditional
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(!=, " != ", DOCTEST_CMP_NE) //!OCLINT bitwise operator in conditional
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(>, " >  ", DOCTEST_CMP_GT) //!OCLINT bitwise operator in conditional
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(<, " <  ", DOCTEST_CMP_LT) //!OCLINT bitwise operator in conditional
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(>=, " >= ", DOCTEST_CMP_GE) //!OCLINT bitwise operator in conditional
        DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(<=, " <= ", DOCTEST_CMP_LE) //!OCLINT bitwise operator in conditional
        // clang-format on

        // forbidding some expressions based on this table: http://en.cppreference.com/w/cpp/language/operator_precedence
        DOCTEST_FORBIT_EXPRESSION(&)
        DOCTEST_FORBIT_EXPRESSION (^)
        DOCTEST_FORBIT_EXPRESSION(|)
        DOCTEST_FORBIT_EXPRESSION(&&)
        DOCTEST_FORBIT_EXPRESSION(||)
        DOCTEST_FORBIT_EXPRESSION(=)
        DOCTEST_FORBIT_EXPRESSION(+=)
        DOCTEST_FORBIT_EXPRESSION(-=)
        DOCTEST_FORBIT_EXPRESSION(*=)
        DOCTEST_FORBIT_EXPRESSION(/=)
        DOCTEST_FORBIT_EXPRESSION(%=)
        DOCTEST_FORBIT_EXPRESSION(<<=)
        DOCTEST_FORBIT_EXPRESSION(>>=)
        DOCTEST_FORBIT_EXPRESSION(&=)
        DOCTEST_FORBIT_EXPRESSION(^=)
        DOCTEST_FORBIT_EXPRESSION(|=)
        // these 2 are unfortunate because they should be allowed - they have higher precedence over the comparisons, but the
        // ExpressionDecomposer class uses the left shift operator to capture the left operand of the binary expression...
        DOCTEST_FORBIT_EXPRESSION(<<)
        DOCTEST_FORBIT_EXPRESSION(>>)
    };

#ifndef DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

    DOCTEST_CLANG_SUPPRESS_WARNING_POP
    DOCTEST_MSVC_SUPPRESS_WARNING_POP
    DOCTEST_GCC_SUPPRESS_WARNING_POP

#endif // DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

    struct ExpressionDecomposer
    {
        assertType::Enum m_assert_type;

        ExpressionDecomposer(assertType::Enum assert_type)
                : m_assert_type(assert_type) {}

        // The right operator for capturing expressions is "<=" instead of "<<" (based on the operator precedence table)
        // but then there will be warnings from GCC about "-Wparentheses" and since "_Pragma()" is problematic this will stay for now...
        // https://github.com/philsquared/Catch/issues/870
        // https://github.com/philsquared/Catch/issues/565
        template <typename L>
        Expression_lhs<const DOCTEST_REF_WRAP(L)> operator<<(const DOCTEST_REF_WRAP(L) operand) {
            return Expression_lhs<const DOCTEST_REF_WRAP(L)>(operand, m_assert_type);
        }
    };

    struct DOCTEST_INTERFACE TestCase
    {
        // not used for determining uniqueness
        funcType m_test;    // a function pointer to the test case
        String m_full_name; // contains the name (only for templated test cases!) + the template type
        const char* m_name;       // name of the test case
        const char* m_type;       // for templated test cases - gets appended to the real name
        const char* m_test_suite; // the test suite in which the test was added
        const char* m_description;
        bool        m_skip;
        bool        m_may_fail;
        bool        m_should_fail;
        int         m_expected_failures;
        double      m_timeout;

        // fields by which uniqueness of test cases shall be determined
        const char* m_file; // the file in which the test was registered
        unsigned    m_line; // the line where the test was registered
        int m_template_id; // an ID used to distinguish between the different versions of a templated test case

        TestCase(funcType test, const char* file, unsigned line, const TestSuite& test_suite,
                 const char* type = "", int template_id = -1);

        // for gcc 4.7
        DOCTEST_NOINLINE ~TestCase() {}

        TestCase& operator*(const char* in);

        template <typename T>
        TestCase& operator*(const T& in) {
            in.fill(*this);
            return *this;
        }

        TestCase(const TestCase& other) { *this = other; }

        TestCase& operator=(const TestCase& other);

        bool operator<(const TestCase& other) const;
    };

    // forward declarations of functions used by the macros
    DOCTEST_INTERFACE int regTest(const TestCase& tc);
    DOCTEST_INTERFACE int setTestSuite(const TestSuite& ts);

    DOCTEST_INTERFACE void addFailedAssert(assertType::Enum assert_type);

    DOCTEST_INTERFACE void logTestStart(const TestCase& tc);
    DOCTEST_INTERFACE void logTestEnd();

    DOCTEST_INTERFACE void logTestException(const String& what, bool crash = false);

    DOCTEST_INTERFACE void logAssert(bool passed, const char* decomposition, bool threw,
                                     const String& exception, const char* expr,
                                     assertType::Enum assert_type, const char* file, int line);

    DOCTEST_INTERFACE void logAssertThrows(bool threw, const char* expr,
                                           assertType::Enum assert_type, const char* file,
                                           int line);

    DOCTEST_INTERFACE void logAssertThrowsAs(bool threw, bool threw_as, const char* as,
                                             const String& exception, const char* expr,
                                             assertType::Enum assert_type, const char* file,
                                             int line);

    DOCTEST_INTERFACE void logAssertNothrow(bool threw, const String& exception, const char* expr,
                                            assertType::Enum assert_type, const char* file,
                                            int line);

    DOCTEST_INTERFACE bool isDebuggerActive();
    DOCTEST_INTERFACE void writeToDebugConsole(const String&);

    namespace binaryAssertComparison
    {
        enum Enum
        {
            eq = 0,
            ne,
            gt,
            lt,
            ge,
            le
        };
    } // namespace binaryAssertComparison

    // clang-format off
    template <int, class L, class R> struct RelationalComparator     { bool operator()(const DOCTEST_REF_WRAP(L),     const DOCTEST_REF_WRAP(R)    ) const { return false;        } };
    template <class L, class R> struct RelationalComparator<0, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return eq(lhs, rhs); } };
    template <class L, class R> struct RelationalComparator<1, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return ne(lhs, rhs); } };
    template <class L, class R> struct RelationalComparator<2, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return gt(lhs, rhs); } };
    template <class L, class R> struct RelationalComparator<3, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return lt(lhs, rhs); } };
    template <class L, class R> struct RelationalComparator<4, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return ge(lhs, rhs); } };
    template <class L, class R> struct RelationalComparator<5, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return le(lhs, rhs); } };
    // clang-format on

    struct DOCTEST_INTERFACE ResultBuilder
    {
        assertType::Enum m_assert_type;
        const char*      m_file;
        int              m_line;
        const char*      m_expr;
        const char*      m_exception_type;

        Result m_result;
        bool   m_threw;
        bool   m_threw_as;
        bool   m_failed;
        String m_exception;

        ResultBuilder(assertType::Enum assert_type, const char* file, int line, const char* expr,
                      const char* exception_type = "");

        ~ResultBuilder();

        void setResult(const Result& res) { m_result = res; }

        template <int comparison, typename L, typename R>
        DOCTEST_NOINLINE void binary_assert(const DOCTEST_REF_WRAP(L) lhs,
                                            const DOCTEST_REF_WRAP(R) rhs) {
            m_result.m_passed = RelationalComparator<comparison, L, R>()(lhs, rhs);
            if(!m_result.m_passed || getTestsContextState()->success)
                m_result.m_decomposition = stringifyBinaryExpr(lhs, ", ", rhs);
        }

        template <typename L>
        DOCTEST_NOINLINE void unary_assert(const DOCTEST_REF_WRAP(L) val) {
            m_result.m_passed = !!val;

            if(m_assert_type & assertType::is_false) //!OCLINT bitwise operator in conditional
                m_result.m_passed = !m_result.m_passed;

            if(!m_result.m_passed || getTestsContextState()->success)
                m_result.m_decomposition = toString(val);
        }

        void unexpectedExceptionOccurred();

        bool log();
        void react() const;
    };

    namespace assertAction
    {
        enum Enum
        {
            nothing     = 0,
            dbgbreak    = 1,
            shouldthrow = 2
        };
    } // namespace assertAction

    template <int comparison, typename L, typename R>
    DOCTEST_NOINLINE int fast_binary_assert(assertType::Enum assert_type, const char* file,
                                            int line, const char* expr,
                                            const DOCTEST_REF_WRAP(L) lhs,
                                            const DOCTEST_REF_WRAP(R) rhs) {
        ResultBuilder rb(assert_type, file, line, expr);

        rb.m_result.m_passed = RelationalComparator<comparison, L, R>()(lhs, rhs);

        if(!rb.m_result.m_passed || getTestsContextState()->success)
            rb.m_result.m_decomposition = stringifyBinaryExpr(lhs, ", ", rhs);

        int res = 0;

        if(rb.log())
            res |= assertAction::dbgbreak;

        if(rb.m_failed && checkIfShouldThrow(assert_type))
            res |= assertAction::shouldthrow;

#ifdef DOCTEST_CONFIG_SUPER_FAST_ASSERTS
        // #########################################################################################
        // IF THE DEBUGGER BREAKS HERE - GO 1 LEVEL UP IN THE CALLSTACK TO SEE THE FAILING ASSERTION
        // THIS IS THE EFFECT OF HAVING 'DOCTEST_CONFIG_SUPER_FAST_ASSERTS' DEFINED
        // #########################################################################################
        if(res & assertAction::dbgbreak)
            DOCTEST_BREAK_INTO_DEBUGGER();
        fastAssertThrowIfFlagSet(res);
#endif // DOCTEST_CONFIG_SUPER_FAST_ASSERTS

        return res;
    }

    template <typename L>
    DOCTEST_NOINLINE int fast_unary_assert(assertType::Enum assert_type, const char* file, int line,
                                           const char* val_str, const DOCTEST_REF_WRAP(L) val) {
        ResultBuilder rb(assert_type, file, line, val_str);

        rb.m_result.m_passed = !!val;

        if(assert_type & assertType::is_false) //!OCLINT bitwise operator in conditional
            rb.m_result.m_passed = !rb.m_result.m_passed;

        if(!rb.m_result.m_passed || getTestsContextState()->success)
            rb.m_result.m_decomposition = toString(val);

        int res = 0;

        if(rb.log())
            res |= assertAction::dbgbreak;

        if(rb.m_failed && checkIfShouldThrow(assert_type))
            res |= assertAction::shouldthrow;

#ifdef DOCTEST_CONFIG_SUPER_FAST_ASSERTS
        // #########################################################################################
        // IF THE DEBUGGER BREAKS HERE - GO 1 LEVEL UP IN THE CALLSTACK TO SEE THE FAILING ASSERTION
        // THIS IS THE EFFECT OF HAVING 'DOCTEST_CONFIG_SUPER_FAST_ASSERTS' DEFINED
        // #########################################################################################
        if(res & assertAction::dbgbreak)
            DOCTEST_BREAK_INTO_DEBUGGER();
        fastAssertThrowIfFlagSet(res);
#endif // DOCTEST_CONFIG_SUPER_FAST_ASSERTS

        return res;
    }

    struct DOCTEST_INTERFACE IExceptionTranslator
    {
        virtual ~IExceptionTranslator() {}
        virtual bool translate(String&) const = 0;
    };

    template <typename T>
    class ExceptionTranslator : public IExceptionTranslator //!OCLINT destructor of virtual class
    {
    public:
        explicit ExceptionTranslator(String (*translateFunction)(T))
                : m_translateFunction(translateFunction) {}

        bool translate(String& res) const {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
            try {
                throw;
                // cppcheck-suppress catchExceptionByValue
            } catch(T ex) {                    // NOLINT
                res = m_translateFunction(ex); //!OCLINT parameter reassignment
                return true;
            } catch(...) {} //!OCLINT -  empty catch statement
#endif                      // DOCTEST_CONFIG_NO_EXCEPTIONS
            ((void)res);    // to silence -Wunused-parameter
            return false;
        }

    protected:
        String (*m_translateFunction)(T);
    };

    DOCTEST_INTERFACE void registerExceptionTranslatorImpl(
            const IExceptionTranslator* translateFunction);

    // FIX FOR VISUAL STUDIO VERSIONS PRIOR TO 2015 - they failed to compile the call to operator<< with
    // std::ostream passed as a reference noting that there is a use of an undefined type (which there isn't)
    DOCTEST_INTERFACE void writeStringToStream(std::ostream* stream, const String& str);

    template <bool C>
    struct StringStreamBase
    {
        template <typename T>
        static void convert(std::ostream* stream, const T& in) {
            writeStringToStream(stream, toString(in));
        }

        // always treat char* as a string in this context - no matter
        // if DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING is defined
        static void convert(std::ostream* stream, const char* in) {
            writeStringToStream(stream, String(in));
        }
    };

    template <>
    struct StringStreamBase<true>
    {
        template <typename T>
        static void convert(std::ostream* stream, const T& in) {
            *stream << in;
        }
    };

    template <typename T>
    struct StringStream : StringStreamBase<has_insertion_operator<T>::value>
    {};

    template <typename T>
    void toStream(std::ostream* stream, const T& value) {
        StringStream<T>::convert(stream, value);
    }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    DOCTEST_INTERFACE void toStream(std::ostream* stream, char* in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, const char* in);
#endif // DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
    DOCTEST_INTERFACE void toStream(std::ostream* stream, bool in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, float in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, double in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, double long in);

    DOCTEST_INTERFACE void toStream(std::ostream* stream, char in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, char signed in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, char unsigned in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int short in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int short unsigned in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int unsigned in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int long in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int long unsigned in);

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int long long in);
    DOCTEST_INTERFACE void toStream(std::ostream* stream, int long long unsigned in);
#endif // DOCTEST_CONFIG_WITH_LONG_LONG

    struct IContextScope
    {
        virtual ~IContextScope() {}
        virtual void build(std::ostream*) = 0;
    };

    DOCTEST_INTERFACE void addToContexts(IContextScope* ptr);
    DOCTEST_INTERFACE void popFromContexts();
    DOCTEST_INTERFACE void useContextIfExceptionOccurred(IContextScope* ptr);

    // cppcheck-suppress copyCtorAndEqOperator
    class ContextBuilder
    {
        friend class ContextScope;

        struct ICapture
        {
            virtual ~ICapture() {}
            virtual void toStream(std::ostream*) const = 0;
        };

        template <typename T>
        struct Capture : ICapture //!OCLINT destructor of virtual class
        {
            const T* capture;

            explicit Capture(const T* in)
                    : capture(in) {}
            virtual void toStream(std::ostream* stream) const { // override
                detail::toStream(stream, *capture);
            }
        };

        struct Chunk
        {
            char buf[sizeof(Capture<char>)] DOCTEST_ALIGNMENT(
                    2 * sizeof(void*)); // place to construct a Capture<T>
        };

        struct Node
        {
            Chunk chunk;
            Node* next;
        };

        Chunk stackChunks[DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK];
        int   numCaptures;
        Node* head;
        Node* tail;

        void build(std::ostream* stream) const {
            int curr = 0;
            // iterate over small buffer
            while(curr < numCaptures && curr < DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK)
                reinterpret_cast<const ICapture*>(stackChunks[curr++].buf)->toStream(stream);
            // iterate over list
            Node* curr_elem = head;
            while(curr < numCaptures) {
                reinterpret_cast<const ICapture*>(curr_elem->chunk.buf)->toStream(stream);
                curr_elem = curr_elem->next;
                ++curr;
            }
        }

        // steal the contents of the other - acting as a move constructor...
        DOCTEST_NOINLINE ContextBuilder(ContextBuilder& other)
                : numCaptures(other.numCaptures)
                , head(other.head)
                , tail(other.tail) {
            other.numCaptures = 0;
            other.head        = 0;
            other.tail        = 0;
            my_memcpy(stackChunks, other.stackChunks,
                      unsigned(int(sizeof(Chunk)) * DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK));
        }

        ContextBuilder& operator=(const ContextBuilder&); // NOLINT

    public:
        // cppcheck-suppress uninitMemberVar
        DOCTEST_NOINLINE ContextBuilder() // NOLINT
                : numCaptures(0)
                , head(0)
                , tail(0) {}

        template <typename T>
        DOCTEST_NOINLINE ContextBuilder& operator<<(T& in) {
            Capture<T> temp(&in);

            // construct either on stack or on heap
            // copy the bytes for the whole object - including the vtable because we cant construct
            // the object directly in the buffer using placement new - need the <new> header...
            if(numCaptures < DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK) {
                my_memcpy(stackChunks[numCaptures].buf, &temp, sizeof(Chunk));
            } else {
                Node* curr = new Node;
                curr->next = 0;
                if(tail) {
                    tail->next = curr;
                    tail       = curr;
                } else {
                    head = tail = curr;
                }

                my_memcpy(tail->chunk.buf, &temp, sizeof(Chunk));
            }
            ++numCaptures;
            return *this;
        }

        DOCTEST_NOINLINE ~ContextBuilder() {
            // free the linked list - the ones on the stack are left as-is
            // no destructors are called at all - there is no need
            while(head) {
                Node* next = head->next;
                delete head;
                head = next;
            }
        }

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
        template <typename T>
        ContextBuilder& operator<<(const T&&) {
            DOCTEST_STATIC_ASSERT(
                    deferred_false<T>::value,
                    Cannot_pass_temporaries_or_rvalues_to_the_streaming_operator_because_it_caches_pointers_to_the_passed_objects_for_lazy_evaluation);
            return *this;
        }
#endif // DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
    };

    class ContextScope : public IContextScope
    {
        ContextBuilder contextBuilder;
        bool           built;

    public:
        DOCTEST_NOINLINE explicit ContextScope(ContextBuilder& temp)
                : contextBuilder(temp)
                , built(false) {
            addToContexts(this);
        }

        DOCTEST_NOINLINE ~ContextScope() {
            if(!built)
                useContextIfExceptionOccurred(this);
            popFromContexts();
        }

        void build(std::ostream* stream) {
            built = true;
            contextBuilder.build(stream);
        }
    };

    class DOCTEST_INTERFACE MessageBuilder
    {
        std::ostream*    m_stream;
        const char*      m_file;
        int              m_line;
        assertType::Enum m_severity;

    public:
        MessageBuilder(const char* file, int line, assertType::Enum severity);
        ~MessageBuilder();

        template <typename T>
        MessageBuilder& operator<<(const T& in) {
            toStream(m_stream, in);
            return *this;
        }

        bool log();
        void react();
    };
} // namespace detail

struct test_suite
{
    const char* data;
    test_suite(const char* in)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_test_suite = data; }
    void fill(detail::TestSuite& state) const { state.m_test_suite = data; }
};

struct description
{
    const char* data;
    description(const char* in)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_description = data; }
    void fill(detail::TestSuite& state) const { state.m_description = data; }
};

struct skip
{
    bool data;
    skip(bool in = true)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_skip = data; }
    void fill(detail::TestSuite& state) const { state.m_skip = data; }
};

struct timeout
{
    double data;
    timeout(double in)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_timeout = data; }
    void fill(detail::TestSuite& state) const { state.m_timeout = data; }
};

struct may_fail
{
    bool data;
    may_fail(bool in = true)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_may_fail = data; }
    void fill(detail::TestSuite& state) const { state.m_may_fail = data; }
};

struct should_fail
{
    bool data;
    should_fail(bool in = true)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_should_fail = data; }
    void fill(detail::TestSuite& state) const { state.m_should_fail = data; }
};

struct expected_failures
{
    int data;
    expected_failures(int in)
            : data(in) {}
    void fill(detail::TestCase& state) const { state.m_expected_failures = data; }
    void fill(detail::TestSuite& state) const { state.m_expected_failures = data; }
};

#endif // DOCTEST_CONFIG_DISABLE

#ifndef DOCTEST_CONFIG_DISABLE
template <typename T>
int registerExceptionTranslator(String (*translateFunction)(T)) {
    DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wexit-time-destructors")
    static detail::ExceptionTranslator<T> exceptionTranslator(translateFunction);
    DOCTEST_CLANG_SUPPRESS_WARNING_POP
    detail::registerExceptionTranslatorImpl(&exceptionTranslator);
    return 0;
}

#else  // DOCTEST_CONFIG_DISABLE
template <typename T>
int registerExceptionTranslator(String (*)(T)) {
    return 0;
}
#endif // DOCTEST_CONFIG_DISABLE

DOCTEST_INTERFACE bool isRunningInTest();

// cppcheck-suppress noCopyConstructor
class DOCTEST_INTERFACE Context
{
#if !defined(DOCTEST_CONFIG_DISABLE)
    detail::ContextState* p;

    void parseArgs(int argc, const char* const* argv, bool withDefaults = false);

#endif // DOCTEST_CONFIG_DISABLE

public:
    explicit Context(int argc = 0, const char* const* argv = 0);

    ~Context();

    void applyCommandLine(int argc, const char* const* argv);

    void addFilter(const char* filter, const char* value);
    void clearFilters();
    void setOption(const char* option, int value);
    void setOption(const char* option, const char* value);

    bool shouldExit();

    int run();
};

} // namespace doctest

// if registering is not disabled
#if !defined(DOCTEST_CONFIG_DISABLE)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_EXPAND_VA_ARGS(...) __VA_ARGS__
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_EXPAND_VA_ARGS
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_STRIP_PARENS(x) x
#define DOCTEST_HANDLE_BRACED_VA_ARGS(expr) DOCTEST_STRIP_PARENS(DOCTEST_EXPAND_VA_ARGS expr)

// registers the test by initializing a dummy var with a function
#define DOCTEST_REGISTER_FUNCTION(f, decorators)                                                   \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) = doctest::detail::regTest(  \
            doctest::detail::TestCase(f, __FILE__, __LINE__,                                       \
                                      doctest_detail_test_suite_ns::getCurrentTestSuite()) *       \
            decorators);                                                                           \
    DOCTEST_GLOBAL_NO_WARNINGS_END()

#define DOCTEST_IMPLEMENT_FIXTURE(der, base, func, decorators)                                     \
    namespace                                                                                      \
    {                                                                                              \
        struct der : base                                                                          \
        {                                                                                          \
            void f();                                                                              \
        };                                                                                         \
        static void func() {                                                                       \
            der v;                                                                                 \
            v.f();                                                                                 \
        }                                                                                          \
        DOCTEST_REGISTER_FUNCTION(func, decorators)                                                \
    }                                                                                              \
    inline DOCTEST_NOINLINE void der::f()

#define DOCTEST_CREATE_AND_REGISTER_FUNCTION(f, decorators)                                        \
    static void f();                                                                               \
    DOCTEST_REGISTER_FUNCTION(f, decorators)                                                       \
    static void f()

// for registering tests
#define DOCTEST_TEST_CASE(decorators)                                                              \
    DOCTEST_CREATE_AND_REGISTER_FUNCTION(DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), decorators)

// for registering tests with a fixture
#define DOCTEST_TEST_CASE_FIXTURE(c, decorators)                                                   \
    DOCTEST_IMPLEMENT_FIXTURE(DOCTEST_ANONYMOUS(_DOCTEST_ANON_CLASS_), c,                          \
                              DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), decorators)

// for converting types to strings without the <typeinfo> header and demangling
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING_IMPL(...)                                                           \
    template <>                                                                                    \
    inline const char* type_to_string<__VA_ARGS__>() {                                             \
        return "<" #__VA_ARGS__ ">";                                                               \
    }
#define DOCTEST_TYPE_TO_STRING(...)                                                                \
    namespace doctest                                                                              \
    {                                                                                              \
        namespace detail                                                                           \
        {                                                                                          \
            DOCTEST_TYPE_TO_STRING_IMPL(__VA_ARGS__)                                               \
        }                                                                                          \
    }                                                                                              \
    typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING_IMPL(x)                                                             \
    template <>                                                                                    \
    inline const char* type_to_string<x>() {                                                       \
        return "<" #x ">";                                                                         \
    }
#define DOCTEST_TYPE_TO_STRING(x)                                                                  \
    namespace doctest                                                                              \
    {                                                                                              \
        namespace detail                                                                           \
        {                                                                                          \
            DOCTEST_TYPE_TO_STRING_IMPL(x)                                                         \
        }                                                                                          \
    }                                                                                              \
    typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

// for typed tests
#define DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, types, anon)                                \
    template <typename T>                                                                          \
    inline void anon();                                                                            \
    struct DOCTEST_CAT(anon, FUNCTOR)                                                              \
    {                                                                                              \
        template <int Index, typename Type>                                                        \
        void operator()() {                                                                        \
            doctest::detail::regTest(                                                              \
                    doctest::detail::TestCase(anon<Type>, __FILE__, __LINE__,                      \
                                              doctest_detail_test_suite_ns::getCurrentTestSuite(), \
                                              doctest::detail::type_to_string<Type>(), Index) *    \
                    decorators);                                                                   \
        }                                                                                          \
    };                                                                                             \
    inline int DOCTEST_CAT(anon, REG_FUNC)() {                                                     \
        DOCTEST_CAT(anon, FUNCTOR) registrar;                                                      \
        doctest::detail::ForEachType<DOCTEST_HANDLE_BRACED_VA_ARGS(types)::Result,                 \
                                     DOCTEST_CAT(anon, FUNCTOR)>                                   \
                doIt(registrar);                                                                   \
        return 0;                                                                                  \
    }                                                                                              \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_CAT(anon, DUMMY)) = DOCTEST_CAT(anon, REG_FUNC)();          \
    DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
    template <typename T>                                                                          \
    inline void anon()

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE(decorators, T, ...)                                             \
    DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, (__VA_ARGS__),                                  \
                                    DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE(decorators, T, types)                                           \
    DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, types, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE_IMPL(decorators, T, id, anon)                            \
    template <typename T>                                                                          \
    inline void anon();                                                                            \
    struct DOCTEST_CAT(id, _FUNCTOR)                                                               \
    {                                                                                              \
        int m_line;                                                                                \
        DOCTEST_CAT(id, _FUNCTOR)                                                                  \
        (int line)                                                                                 \
                : m_line(line) {}                                                                  \
        template <int Index, typename Type>                                                        \
        void operator()() {                                                                        \
            doctest::detail::regTest(                                                              \
                    doctest::detail::TestCase(anon<Type>, __FILE__, __LINE__,                      \
                                              doctest_detail_test_suite_ns::getCurrentTestSuite(), \
                                              doctest::detail::type_to_string<Type>(),             \
                                              m_line * 1000 + Index) *                             \
                    decorators);                                                                   \
        }                                                                                          \
    };                                                                                             \
    template <typename T>                                                                          \
    inline void anon()

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE(decorators, T, id)                                       \
    DOCTEST_TEST_CASE_TEMPLATE_DEFINE_IMPL(decorators, T, id, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))

#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, types, anon)                               \
    static int DOCTEST_CAT(anon, REG_FUNC)() {                                                     \
        DOCTEST_CAT(id, _FUNCTOR) registrar(__LINE__);                                             \
        doctest::detail::ForEachType<DOCTEST_HANDLE_BRACED_VA_ARGS(types)::Result,                 \
                                     DOCTEST_CAT(id, _FUNCTOR)>                                    \
                doIt(registrar);                                                                   \
        return 0;                                                                                  \
    }                                                                                              \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_CAT(anon, DUMMY)) = DOCTEST_CAT(anon, REG_FUNC)();          \
    DOCTEST_GLOBAL_NO_WARNINGS_END() typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, ...)                                            \
    DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, (__VA_ARGS__),                                 \
                                                DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, types)                                          \
    DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, types, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

// for subcases
#define DOCTEST_SUBCASE(name)                                                                      \
    if(const doctest::detail::Subcase & DOCTEST_ANONYMOUS(_DOCTEST_ANON_SUBCASE_) DOCTEST_UNUSED = \
               doctest::detail::Subcase(name, __FILE__, __LINE__))

// for grouping tests in test suites by using code blocks
#define DOCTEST_TEST_SUITE_IMPL(decorators, ns_name)                                               \
    namespace ns_name                                                                              \
    {                                                                                              \
        namespace doctest_detail_test_suite_ns                                                     \
        {                                                                                          \
            static DOCTEST_NOINLINE doctest::detail::TestSuite& getCurrentTestSuite() {            \
                static doctest::detail::TestSuite data;                                            \
                static bool                       inited = false;                                  \
                if(!inited) {                                                                      \
                    data* decorators;                                                              \
                    inited = true;                                                                 \
                }                                                                                  \
                return data;                                                                       \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    namespace ns_name

#define DOCTEST_TEST_SUITE(decorators)                                                             \
    DOCTEST_TEST_SUITE_IMPL(decorators, DOCTEST_ANONYMOUS(_DOCTEST_ANON_SUITE_))

// for starting a testsuite block
#define DOCTEST_TEST_SUITE_BEGIN(decorators)                                                       \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) =                            \
            doctest::detail::setTestSuite(doctest::detail::TestSuite() * decorators);              \
    DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
    typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

// for ending a testsuite block
#define DOCTEST_TEST_SUITE_END                                                                     \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) =                            \
            doctest::detail::setTestSuite(doctest::detail::TestSuite() * "");                      \
    DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
    typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

// for registering exception translators
#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR_IMPL(translatorName, signature)                      \
    inline doctest::String translatorName(signature);                                              \
    DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_)) =                     \
            doctest::registerExceptionTranslator(translatorName);                                  \
    DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
    doctest::String translatorName(signature)

#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR(signature)                                           \
    DOCTEST_REGISTER_EXCEPTION_TRANSLATOR_IMPL(DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_),       \
                                               signature)

// for logging
#define DOCTEST_INFO(x)                                                                            \
    doctest::detail::ContextScope DOCTEST_ANONYMOUS(_DOCTEST_CAPTURE_)(                            \
            doctest::detail::ContextBuilder() << x)
#define DOCTEST_CAPTURE(x) DOCTEST_INFO(#x " := " << x)

#define DOCTEST_ADD_AT_IMPL(type, file, line, mb, x)                                               \
    do {                                                                                           \
        doctest::detail::MessageBuilder mb(file, line, doctest::detail::assertType::type);         \
        mb << x;                                                                                   \
        if(mb.log())                                                                               \
            DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
        mb.react();                                                                                \
    } while((void)0, 0)

// clang-format off
#define DOCTEST_ADD_MESSAGE_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_warn, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)
#define DOCTEST_ADD_FAIL_CHECK_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_check, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)
#define DOCTEST_ADD_FAIL_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_require, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)
// clang-format on

#define DOCTEST_MESSAGE(x) DOCTEST_ADD_MESSAGE_AT(__FILE__, __LINE__, x)
#define DOCTEST_FAIL_CHECK(x) DOCTEST_ADD_FAIL_CHECK_AT(__FILE__, __LINE__, x)
#define DOCTEST_FAIL(x) DOCTEST_ADD_FAIL_AT(__FILE__, __LINE__, x)

#if __cplusplus >= 201402L || (DOCTEST_MSVC >= DOCTEST_COMPILER(19, 10, 0))
template <class T, T x>
constexpr T to_lvalue = x;
#define DOCTEST_TO_LVALUE(...) to_lvalue<decltype(__VA_ARGS__), __VA_ARGS__>
#else
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TO_LVALUE(...) TO_LVALUE_CAN_BE_USED_ONLY_IN_CPP14_MODE_OR_WITH_VS_2017_OR_NEWER
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TO_LVALUE(x) TO_LVALUE_CAN_BE_USED_ONLY_IN_CPP14_MODE_OR_WITH_VS_2017_OR_NEWER
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif // TO_LVALUE hack for logging macros like INFO()

// common code in asserts - for convenience
#define DOCTEST_ASSERT_LOG_AND_REACT(rb)                                                           \
    if(rb.log())                                                                                   \
        DOCTEST_BREAK_INTO_DEBUGGER();                                                             \
    rb.react()

#ifdef DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
#define DOCTEST_WRAP_IN_TRY(x) x;
#else // DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
#define DOCTEST_WRAP_IN_TRY(x)                                                                     \
    try {                                                                                          \
        x;                                                                                         \
    } catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }
#endif // DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS

#define DOCTEST_ASSERT_IMPLEMENT_2(expr, assert_type)                                              \
    DOCTEST_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Woverloaded-shift-op-parentheses")                  \
    doctest::detail::ResultBuilder _DOCTEST_RB(                                                    \
            doctest::detail::assertType::assert_type, __FILE__, __LINE__,                          \
            DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                                   \
    DOCTEST_WRAP_IN_TRY(_DOCTEST_RB.setResult(                                                     \
            doctest::detail::ExpressionDecomposer(doctest::detail::assertType::assert_type)        \
            << DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))                                               \
    DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB)                                                      \
    DOCTEST_CLANG_SUPPRESS_WARNING_POP

#define DOCTEST_ASSERT_IMPLEMENT_1(expr, assert_type)                                              \
    do {                                                                                           \
        DOCTEST_ASSERT_IMPLEMENT_2(expr, assert_type);                                             \
    } while((void)0, 0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_WARN)
#define DOCTEST_CHECK(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_CHECK)
#define DOCTEST_REQUIRE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_REQUIRE)
#define DOCTEST_WARN_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_WARN_FALSE)
#define DOCTEST_CHECK_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_CHECK_FALSE)
#define DOCTEST_REQUIRE_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_REQUIRE_FALSE)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_WARN)
#define DOCTEST_CHECK(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_CHECK)
#define DOCTEST_REQUIRE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_REQUIRE)
#define DOCTEST_WARN_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_WARN_FALSE)
#define DOCTEST_CHECK_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_CHECK_FALSE)
#define DOCTEST_REQUIRE_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_REQUIRE_FALSE)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

// clang-format off
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_WARN); } while((void)0, 0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_CHECK); } while((void)0, 0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_REQUIRE); } while((void)0, 0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_WARN_FALSE); } while((void)0, 0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_CHECK_FALSE); } while((void)0, 0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_REQUIRE_FALSE); } while((void)0, 0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_WARN); } while((void)0, 0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_CHECK); } while((void)0, 0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_REQUIRE); } while((void)0, 0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_WARN_FALSE); } while((void)0, 0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_CHECK_FALSE); } while((void)0, 0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_REQUIRE_FALSE); } while((void)0, 0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
// clang-format on

#define DOCTEST_ASSERT_THROWS(expr, assert_type)                                                   \
    do {                                                                                           \
        if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
            doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,   \
                                                       __FILE__, __LINE__, #expr);                 \
            try {                                                                                  \
                expr;                                                                              \
            } catch(...) { _DOCTEST_RB.m_threw = true; }                                           \
            DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
        }                                                                                          \
    } while((void)0, 0)

#define DOCTEST_ASSERT_THROWS_AS(expr, as, assert_type)                                            \
    do {                                                                                           \
        if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
            doctest::detail::ResultBuilder _DOCTEST_RB(                                            \
                    doctest::detail::assertType::assert_type, __FILE__, __LINE__, #expr,           \
                    DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(as)));                             \
            try {                                                                                  \
                expr;                                                                              \
            } catch(const DOCTEST_HANDLE_BRACED_VA_ARGS(as)&) {                                    \
                _DOCTEST_RB.m_threw    = true;                                                     \
                _DOCTEST_RB.m_threw_as = true;                                                     \
            } catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }                            \
            DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
        }                                                                                          \
    } while((void)0, 0)

#define DOCTEST_ASSERT_NOTHROW(expr, assert_type)                                                  \
    do {                                                                                           \
        if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
            doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,   \
                                                       __FILE__, __LINE__, #expr);                 \
            try {                                                                                  \
                expr;                                                                              \
            } catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }                            \
            DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
        }                                                                                          \
    } while((void)0, 0)

#define DOCTEST_WARN_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_WARN_THROWS)
#define DOCTEST_CHECK_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_CHECK_THROWS)
#define DOCTEST_REQUIRE_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_REQUIRE_THROWS)

// clang-format off
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_WARN_THROWS_AS)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_CHECK_THROWS_AS)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_REQUIRE_THROWS_AS)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_WARN_THROWS_AS)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_CHECK_THROWS_AS)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_REQUIRE_THROWS_AS)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
// clang-format on

#define DOCTEST_WARN_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_WARN_NOTHROW)
#define DOCTEST_CHECK_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_CHECK_NOTHROW)
#define DOCTEST_REQUIRE_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_REQUIRE_NOTHROW)

// clang-format off
#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_THROWS(expr); } while((void)0, 0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_THROWS(expr); } while((void)0, 0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_THROWS(expr); } while((void)0, 0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_NOTHROW(expr); } while((void)0, 0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_NOTHROW(expr); } while((void)0, 0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_NOTHROW(expr); } while((void)0, 0)
// clang-format on

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_BINARY_ASSERT(assert_type, expr, comp)                                             \
    do {                                                                                           \
        doctest::detail::ResultBuilder _DOCTEST_RB(                                                \
                doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
                DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                               \
        DOCTEST_WRAP_IN_TRY(                                                                       \
                _DOCTEST_RB.binary_assert<doctest::detail::binaryAssertComparison::comp>(          \
                        DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))                                      \
        DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
    } while((void)0, 0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_BINARY_ASSERT(assert_type, lhs, rhs, comp)                                         \
    do {                                                                                           \
        doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,       \
                                                   __FILE__, __LINE__, #lhs ", " #rhs);            \
        DOCTEST_WRAP_IN_TRY(                                                                       \
                _DOCTEST_RB.binary_assert<doctest::detail::binaryAssertComparison::comp>(lhs,      \
                                                                                         rhs))     \
        DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
    } while((void)0, 0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_UNARY_ASSERT(assert_type, expr)                                                    \
    do {                                                                                           \
        doctest::detail::ResultBuilder _DOCTEST_RB(                                                \
                doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
                DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                               \
        DOCTEST_WRAP_IN_TRY(_DOCTEST_RB.unary_assert(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))         \
        DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
    } while((void)0, 0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_EQ(...) DOCTEST_BINARY_ASSERT(DT_WARN_EQ, (__VA_ARGS__), eq)
#define DOCTEST_CHECK_EQ(...) DOCTEST_BINARY_ASSERT(DT_CHECK_EQ, (__VA_ARGS__), eq)
#define DOCTEST_REQUIRE_EQ(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_EQ, (__VA_ARGS__), eq)
#define DOCTEST_WARN_NE(...) DOCTEST_BINARY_ASSERT(DT_WARN_NE, (__VA_ARGS__), ne)
#define DOCTEST_CHECK_NE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_NE, (__VA_ARGS__), ne)
#define DOCTEST_REQUIRE_NE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_NE, (__VA_ARGS__), ne)
#define DOCTEST_WARN_GT(...) DOCTEST_BINARY_ASSERT(DT_WARN_GT, (__VA_ARGS__), gt)
#define DOCTEST_CHECK_GT(...) DOCTEST_BINARY_ASSERT(DT_CHECK_GT, (__VA_ARGS__), gt)
#define DOCTEST_REQUIRE_GT(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GT, (__VA_ARGS__), gt)
#define DOCTEST_WARN_LT(...) DOCTEST_BINARY_ASSERT(DT_WARN_LT, (__VA_ARGS__), lt)
#define DOCTEST_CHECK_LT(...) DOCTEST_BINARY_ASSERT(DT_CHECK_LT, (__VA_ARGS__), lt)
#define DOCTEST_REQUIRE_LT(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LT, (__VA_ARGS__), lt)
#define DOCTEST_WARN_GE(...) DOCTEST_BINARY_ASSERT(DT_WARN_GE, (__VA_ARGS__), ge)
#define DOCTEST_CHECK_GE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_GE, (__VA_ARGS__), ge)
#define DOCTEST_REQUIRE_GE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GE, (__VA_ARGS__), ge)
#define DOCTEST_WARN_LE(...) DOCTEST_BINARY_ASSERT(DT_WARN_LE, (__VA_ARGS__), le)
#define DOCTEST_CHECK_LE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_LE, (__VA_ARGS__), le)
#define DOCTEST_REQUIRE_LE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LE, (__VA_ARGS__), le)

#define DOCTEST_WARN_UNARY(...) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY, (__VA_ARGS__))
#define DOCTEST_CHECK_UNARY(...) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY, (__VA_ARGS__))
#define DOCTEST_REQUIRE_UNARY(...) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY, (__VA_ARGS__))
#define DOCTEST_WARN_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_CHECK_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_REQUIRE_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY_FALSE, (__VA_ARGS__))
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_EQ, lhs, rhs, eq)
#define DOCTEST_CHECK_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_EQ, lhs, rhs, eq)
#define DOCTEST_REQUIRE_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_EQ, lhs, rhs, eq)
#define DOCTEST_WARN_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_NE, lhs, rhs, ne)
#define DOCTEST_CHECK_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_NE, lhs, rhs, ne)
#define DOCTEST_REQUIRE_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_NE, lhs, rhs, ne)
#define DOCTEST_WARN_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_GT, lhs, rhs, gt)
#define DOCTEST_CHECK_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_GT, lhs, rhs, gt)
#define DOCTEST_REQUIRE_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GT, lhs, rhs, gt)
#define DOCTEST_WARN_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_LT, lhs, rhs, lt)
#define DOCTEST_CHECK_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_LT, lhs, rhs, lt)
#define DOCTEST_REQUIRE_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LT, lhs, rhs, lt)
#define DOCTEST_WARN_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_GE, lhs, rhs, ge)
#define DOCTEST_CHECK_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_GE, lhs, rhs, ge)
#define DOCTEST_REQUIRE_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GE, lhs, rhs, ge)
#define DOCTEST_WARN_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_LE, lhs, rhs, le)
#define DOCTEST_CHECK_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_LE, lhs, rhs, le)
#define DOCTEST_REQUIRE_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LE, lhs, rhs, le)

#define DOCTEST_WARN_UNARY(v) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY, v)
#define DOCTEST_CHECK_UNARY(v) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY, v)
#define DOCTEST_REQUIRE_UNARY(v) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY, v)
#define DOCTEST_WARN_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY_FALSE, v)
#define DOCTEST_CHECK_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY_FALSE, v)
#define DOCTEST_REQUIRE_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY_FALSE, v)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#ifndef DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, expr, comparison)                                  \
    do {                                                                                           \
        int _DOCTEST_FAST_RES = doctest::detail::fast_binary_assert<                               \
                doctest::detail::binaryAssertComparison::comparison>(                              \
                doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
                DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                \
                DOCTEST_HANDLE_BRACED_VA_ARGS(expr));                                              \
        if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
            DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
        doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
    } while((void)0, 0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, lhs, rhs, comparison)                              \
    do {                                                                                           \
        int _DOCTEST_FAST_RES = doctest::detail::fast_binary_assert<                               \
                doctest::detail::binaryAssertComparison::comparison>(                              \
                doctest::detail::assertType::assert_type, __FILE__, __LINE__, #lhs ", " #rhs, lhs, \
                rhs);                                                                              \
        if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
            DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
        doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
    } while((void)0, 0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_FAST_UNARY_ASSERT(assert_type, expr)                                               \
    do {                                                                                           \
        int _DOCTEST_FAST_RES = doctest::detail::fast_unary_assert(                                \
                doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
                DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                \
                DOCTEST_HANDLE_BRACED_VA_ARGS(expr));                                              \
        if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
            DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
        doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
    } while((void)0, 0)

#else // DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, expr, comparison)                                  \
    doctest::detail::fast_binary_assert<doctest::detail::binaryAssertComparison::comparison>(      \
            doctest::detail::assertType::assert_type, __FILE__, __LINE__,                          \
            DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                    \
            DOCTEST_HANDLE_BRACED_VA_ARGS(expr))
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, lhs, rhs, comparison)                              \
    doctest::detail::fast_binary_assert<doctest::detail::binaryAssertComparison::comparison>(      \
            doctest::detail::assertType::assert_type, __FILE__, __LINE__, #lhs ", " #rhs, lhs,     \
            rhs)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_FAST_UNARY_ASSERT(assert_type, expr)                                               \
    doctest::detail::fast_unary_assert(doctest::detail::assertType::assert_type, __FILE__,         \
                                       __LINE__,                                                   \
                                       DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),         \
                                       DOCTEST_HANDLE_BRACED_VA_ARGS(expr))

#endif // DOCTEST_CONFIG_SUPER_FAST_ASSERTS

// clang-format off
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_WARN_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_CHECK_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_REQUIRE_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_WARN_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_CHECK_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_REQUIRE_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_WARN_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_CHECK_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_REQUIRE_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_WARN_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_CHECK_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_REQUIRE_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_WARN_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_CHECK_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_REQUIRE_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_WARN_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LE, (__VA_ARGS__), le)
#define DOCTEST_FAST_CHECK_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LE, (__VA_ARGS__), le)
#define DOCTEST_FAST_REQUIRE_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LE, (__VA_ARGS__), le)

#define DOCTEST_FAST_WARN_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_CHECK_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_REQUIRE_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_WARN_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_FAST_CHECK_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY_FALSE, (__VA_ARGS__))
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_WARN_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_EQ, l, r, eq)
#define DOCTEST_FAST_CHECK_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_EQ, l, r, eq)
#define DOCTEST_FAST_REQUIRE_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_EQ, l, r, eq)
#define DOCTEST_FAST_WARN_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_NE, l, r, ne)
#define DOCTEST_FAST_CHECK_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_NE, l, r, ne)
#define DOCTEST_FAST_REQUIRE_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_NE, l, r, ne)
#define DOCTEST_FAST_WARN_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GT, l, r, gt)
#define DOCTEST_FAST_CHECK_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GT, l, r, gt)
#define DOCTEST_FAST_REQUIRE_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GT, l, r, gt)
#define DOCTEST_FAST_WARN_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LT, l, r, lt)
#define DOCTEST_FAST_CHECK_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LT, l, r, lt)
#define DOCTEST_FAST_REQUIRE_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LT, l, r, lt)
#define DOCTEST_FAST_WARN_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GE, l, r, ge)
#define DOCTEST_FAST_CHECK_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GE, l, r, ge)
#define DOCTEST_FAST_REQUIRE_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GE, l, r, ge)
#define DOCTEST_FAST_WARN_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LE, l, r, le)
#define DOCTEST_FAST_CHECK_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LE, l, r, le)
#define DOCTEST_FAST_REQUIRE_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LE, l, r, le)

#define DOCTEST_FAST_WARN_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY, v)
#define DOCTEST_FAST_CHECK_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY, v)
#define DOCTEST_FAST_REQUIRE_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY, v)
#define DOCTEST_FAST_WARN_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY_FALSE, v)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY_FALSE, v)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY_FALSE, v)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
// clang-format on

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS

#undef DOCTEST_WARN_THROWS
#undef DOCTEST_CHECK_THROWS
#undef DOCTEST_REQUIRE_THROWS
#undef DOCTEST_WARN_THROWS_AS
#undef DOCTEST_CHECK_THROWS_AS
#undef DOCTEST_REQUIRE_THROWS_AS
#undef DOCTEST_WARN_NOTHROW
#undef DOCTEST_CHECK_NOTHROW
#undef DOCTEST_REQUIRE_NOTHROW

#undef DOCTEST_WARN_THROWS_MESSAGE
#undef DOCTEST_CHECK_THROWS_MESSAGE
#undef DOCTEST_REQUIRE_THROWS_MESSAGE
#undef DOCTEST_WARN_THROWS_AS_MESSAGE
#undef DOCTEST_CHECK_THROWS_AS_MESSAGE
#undef DOCTEST_REQUIRE_THROWS_AS_MESSAGE
#undef DOCTEST_WARN_NOTHROW_MESSAGE
#undef DOCTEST_CHECK_NOTHROW_MESSAGE
#undef DOCTEST_REQUIRE_NOTHROW_MESSAGE

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS

#define DOCTEST_WARN_THROWS(expr) ((void)0)
#define DOCTEST_CHECK_THROWS(expr) ((void)0)
#define DOCTEST_REQUIRE_THROWS(expr) ((void)0)
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) ((void)0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) ((void)0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_NOTHROW(expr) ((void)0)
#define DOCTEST_CHECK_NOTHROW(expr) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW(expr) ((void)0)

#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) ((void)0)

#else // DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS

#undef DOCTEST_REQUIRE
#undef DOCTEST_REQUIRE_FALSE
#undef DOCTEST_REQUIRE_MESSAGE
#undef DOCTEST_REQUIRE_FALSE_MESSAGE
#undef DOCTEST_REQUIRE_EQ
#undef DOCTEST_REQUIRE_NE
#undef DOCTEST_REQUIRE_GT
#undef DOCTEST_REQUIRE_LT
#undef DOCTEST_REQUIRE_GE
#undef DOCTEST_REQUIRE_LE
#undef DOCTEST_REQUIRE_UNARY
#undef DOCTEST_REQUIRE_UNARY_FALSE
#undef DOCTEST_FAST_REQUIRE_EQ
#undef DOCTEST_FAST_REQUIRE_NE
#undef DOCTEST_FAST_REQUIRE_GT
#undef DOCTEST_FAST_REQUIRE_LT
#undef DOCTEST_FAST_REQUIRE_GE
#undef DOCTEST_FAST_REQUIRE_LE
#undef DOCTEST_FAST_REQUIRE_UNARY
#undef DOCTEST_FAST_REQUIRE_UNARY_FALSE

#endif // DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS

#endif // DOCTEST_CONFIG_NO_EXCEPTIONS

// =================================================================================================
// == WHAT FOLLOWS IS VERSIONS OF THE MACROS THAT DO NOT DO ANY REGISTERING!                      ==
// == THIS CAN BE ENABLED BY DEFINING DOCTEST_CONFIG_DISABLE GLOBALLY!                            ==
// =================================================================================================
#else // DOCTEST_CONFIG_DISABLE

#define DOCTEST_IMPLEMENT_FIXTURE(der, base, func, name)                                           \
    namespace                                                                                      \
    {                                                                                              \
        template <typename DOCTEST_UNUSED_TEMPLATE_TYPE>                                           \
        struct der : base                                                                          \
        { void f(); };                                                                             \
    }                                                                                              \
    template <typename DOCTEST_UNUSED_TEMPLATE_TYPE>                                               \
    inline void der<DOCTEST_UNUSED_TEMPLATE_TYPE>::f()

#define DOCTEST_CREATE_AND_REGISTER_FUNCTION(f, name)                                              \
    template <typename DOCTEST_UNUSED_TEMPLATE_TYPE>                                               \
    static inline void f()

// for registering tests
#define DOCTEST_TEST_CASE(name)                                                                    \
    DOCTEST_CREATE_AND_REGISTER_FUNCTION(DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), name)

// for registering tests with a fixture
#define DOCTEST_TEST_CASE_FIXTURE(x, name)                                                         \
    DOCTEST_IMPLEMENT_FIXTURE(DOCTEST_ANONYMOUS(_DOCTEST_ANON_CLASS_), x,                          \
                              DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), name)

// for converting types to strings without the <typeinfo> header and demangling
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING(...) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#define DOCTEST_TYPE_TO_STRING_IMPL(...)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING(x) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#define DOCTEST_TYPE_TO_STRING_IMPL(x)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

// for typed tests
#define DOCTEST_TEST_CASE_TEMPLATE(name, type, types)                                              \
    template <typename type>                                                                       \
    inline void DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_)()

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE(name, type, id)                                          \
    template <typename type>                                                                       \
    inline void DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_)()

#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, types)                                          \
    typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

// for subcases
#define DOCTEST_SUBCASE(name)

// for a testsuite block
#define DOCTEST_TEST_SUITE(name) namespace

// for starting a testsuite block
#define DOCTEST_TEST_SUITE_BEGIN(name) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

// for ending a testsuite block
#define DOCTEST_TEST_SUITE_END typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR(signature)                                           \
    template <typename DOCTEST_UNUSED_TEMPLATE_TYPE>                                               \
    static inline doctest::String DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_)(signature)

#define DOCTEST_INFO(x) ((void)0)
#define DOCTEST_CAPTURE(x) ((void)0)
#define DOCTEST_ADD_MESSAGE_AT(file, line, x) ((void)0)
#define DOCTEST_ADD_FAIL_CHECK_AT(file, line, x) ((void)0)
#define DOCTEST_ADD_FAIL_AT(file, line, x) ((void)0)
#define DOCTEST_MESSAGE(x) ((void)0)
#define DOCTEST_FAIL_CHECK(x) ((void)0)
#define DOCTEST_FAIL(x) ((void)0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(...) ((void)0)
#define DOCTEST_CHECK(...) ((void)0)
#define DOCTEST_REQUIRE(...) ((void)0)
#define DOCTEST_WARN_FALSE(...) ((void)0)
#define DOCTEST_CHECK_FALSE(...) ((void)0)
#define DOCTEST_REQUIRE_FALSE(...) ((void)0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(expr) ((void)0)
#define DOCTEST_CHECK(expr) ((void)0)
#define DOCTEST_REQUIRE(expr) ((void)0)
#define DOCTEST_WARN_FALSE(expr) ((void)0)
#define DOCTEST_CHECK_FALSE(expr) ((void)0)
#define DOCTEST_REQUIRE_FALSE(expr) ((void)0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_WARN_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) ((void)0)

#define DOCTEST_WARN_THROWS(expr) ((void)0)
#define DOCTEST_CHECK_THROWS(expr) ((void)0)
#define DOCTEST_REQUIRE_THROWS(expr) ((void)0)
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) ((void)0)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) ((void)0)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_NOTHROW(expr) ((void)0)
#define DOCTEST_CHECK_NOTHROW(expr) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW(expr) ((void)0)

#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) ((void)0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_WARN_EQ(...) ((void)0)
#define DOCTEST_CHECK_EQ(...) ((void)0)
#define DOCTEST_REQUIRE_EQ(...) ((void)0)
#define DOCTEST_WARN_NE(...) ((void)0)
#define DOCTEST_CHECK_NE(...) ((void)0)
#define DOCTEST_REQUIRE_NE(...) ((void)0)
#define DOCTEST_WARN_GT(...) ((void)0)
#define DOCTEST_CHECK_GT(...) ((void)0)
#define DOCTEST_REQUIRE_GT(...) ((void)0)
#define DOCTEST_WARN_LT(...) ((void)0)
#define DOCTEST_CHECK_LT(...) ((void)0)
#define DOCTEST_REQUIRE_LT(...) ((void)0)
#define DOCTEST_WARN_GE(...) ((void)0)
#define DOCTEST_CHECK_GE(...) ((void)0)
#define DOCTEST_REQUIRE_GE(...) ((void)0)
#define DOCTEST_WARN_LE(...) ((void)0)
#define DOCTEST_CHECK_LE(...) ((void)0)
#define DOCTEST_REQUIRE_LE(...) ((void)0)

#define DOCTEST_WARN_UNARY(...) ((void)0)
#define DOCTEST_CHECK_UNARY(...) ((void)0)
#define DOCTEST_REQUIRE_UNARY(...) ((void)0)
#define DOCTEST_WARN_UNARY_FALSE(...) ((void)0)
#define DOCTEST_CHECK_UNARY_FALSE(...) ((void)0)
#define DOCTEST_REQUIRE_UNARY_FALSE(...) ((void)0)

#define DOCTEST_FAST_WARN_EQ(...) ((void)0)
#define DOCTEST_FAST_CHECK_EQ(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_EQ(...) ((void)0)
#define DOCTEST_FAST_WARN_NE(...) ((void)0)
#define DOCTEST_FAST_CHECK_NE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_NE(...) ((void)0)
#define DOCTEST_FAST_WARN_GT(...) ((void)0)
#define DOCTEST_FAST_CHECK_GT(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_GT(...) ((void)0)
#define DOCTEST_FAST_WARN_LT(...) ((void)0)
#define DOCTEST_FAST_CHECK_LT(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_LT(...) ((void)0)
#define DOCTEST_FAST_WARN_GE(...) ((void)0)
#define DOCTEST_FAST_CHECK_GE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_GE(...) ((void)0)
#define DOCTEST_FAST_WARN_LE(...) ((void)0)
#define DOCTEST_FAST_CHECK_LE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_LE(...) ((void)0)

#define DOCTEST_FAST_WARN_UNARY(...) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY(...) ((void)0)
#define DOCTEST_FAST_WARN_UNARY_FALSE(...) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(...) ((void)0)

#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_WARN_EQ(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_EQ(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_EQ(lhs, rhs) ((void)0)
#define DOCTEST_WARN_NE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_NE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_NE(lhs, rhs) ((void)0)
#define DOCTEST_WARN_GT(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_GT(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_GT(lhs, rhs) ((void)0)
#define DOCTEST_WARN_LT(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_LT(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_LT(lhs, rhs) ((void)0)
#define DOCTEST_WARN_GE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_GE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_GE(lhs, rhs) ((void)0)
#define DOCTEST_WARN_LE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_LE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_LE(lhs, rhs) ((void)0)

#define DOCTEST_WARN_UNARY(val) ((void)0)
#define DOCTEST_CHECK_UNARY(val) ((void)0)
#define DOCTEST_REQUIRE_UNARY(val) ((void)0)
#define DOCTEST_WARN_UNARY_FALSE(val) ((void)0)
#define DOCTEST_CHECK_UNARY_FALSE(val) ((void)0)
#define DOCTEST_REQUIRE_UNARY_FALSE(val) ((void)0)

#define DOCTEST_FAST_WARN_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_LE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_LE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_LE(lhs, rhs) ((void)0)

#define DOCTEST_FAST_WARN_UNARY(val) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY(val) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY(val) ((void)0)
#define DOCTEST_FAST_WARN_UNARY_FALSE(val) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(val) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(val) ((void)0)

#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#endif // DOCTEST_CONFIG_DISABLE

// BDD style macros
// clang-format off
#define DOCTEST_SCENARIO(name)  TEST_CASE("  Scenario: " name)
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_SCENARIO_TEMPLATE(name, T, ...)  TEST_CASE_TEMPLATE("  Scenario: " name, T, __VA_ARGS__)
#else // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_SCENARIO_TEMPLATE(name, T, types) TEST_CASE_TEMPLATE("  Scenario: " name, T, types)
#endif // DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_SCENARIO_TEMPLATE_DEFINE(name, T, id) DOCTEST_TEST_CASE_TEMPLATE_DEFINE("  Scenario: " name, T, id)

#define DOCTEST_GIVEN(name)     SUBCASE("   Given: " name)
#define DOCTEST_WHEN(name)      SUBCASE("    When: " name)
#define DOCTEST_AND_WHEN(name)  SUBCASE("And when: " name)
#define DOCTEST_THEN(name)      SUBCASE("    Then: " name)
#define DOCTEST_AND_THEN(name)  SUBCASE("     And: " name)
// clang-format on

// == SHORT VERSIONS OF THE MACROS
#if !defined(DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES)

#define TEST_CASE DOCTEST_TEST_CASE
#define TEST_CASE_FIXTURE DOCTEST_TEST_CASE_FIXTURE
#define TYPE_TO_STRING DOCTEST_TYPE_TO_STRING
#define TEST_CASE_TEMPLATE DOCTEST_TEST_CASE_TEMPLATE
#define TEST_CASE_TEMPLATE_DEFINE DOCTEST_TEST_CASE_TEMPLATE_DEFINE
#define TEST_CASE_TEMPLATE_INSTANTIATE DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE
#define SUBCASE DOCTEST_SUBCASE
#define TEST_SUITE DOCTEST_TEST_SUITE
#define TEST_SUITE_BEGIN DOCTEST_TEST_SUITE_BEGIN
#define TEST_SUITE_END DOCTEST_TEST_SUITE_END
#define REGISTER_EXCEPTION_TRANSLATOR DOCTEST_REGISTER_EXCEPTION_TRANSLATOR
#define INFO DOCTEST_INFO
#define CAPTURE DOCTEST_CAPTURE
#define ADD_MESSAGE_AT DOCTEST_ADD_MESSAGE_AT
#define ADD_FAIL_CHECK_AT DOCTEST_ADD_FAIL_CHECK_AT
#define ADD_FAIL_AT DOCTEST_ADD_FAIL_AT
#define MESSAGE DOCTEST_MESSAGE
#define FAIL_CHECK DOCTEST_FAIL_CHECK
#define FAIL DOCTEST_FAIL
#define TO_LVALUE DOCTEST_TO_LVALUE

#define WARN DOCTEST_WARN
#define WARN_FALSE DOCTEST_WARN_FALSE
#define WARN_THROWS DOCTEST_WARN_THROWS
#define WARN_THROWS_AS DOCTEST_WARN_THROWS_AS
#define WARN_NOTHROW DOCTEST_WARN_NOTHROW
#define CHECK DOCTEST_CHECK
#define CHECK_FALSE DOCTEST_CHECK_FALSE
#define CHECK_THROWS DOCTEST_CHECK_THROWS
#define CHECK_THROWS_AS DOCTEST_CHECK_THROWS_AS
#define CHECK_NOTHROW DOCTEST_CHECK_NOTHROW
#define REQUIRE DOCTEST_REQUIRE
#define REQUIRE_FALSE DOCTEST_REQUIRE_FALSE
#define REQUIRE_THROWS DOCTEST_REQUIRE_THROWS
#define REQUIRE_THROWS_AS DOCTEST_REQUIRE_THROWS_AS
#define REQUIRE_NOTHROW DOCTEST_REQUIRE_NOTHROW

#define WARN_MESSAGE DOCTEST_WARN_MESSAGE
#define WARN_FALSE_MESSAGE DOCTEST_WARN_FALSE_MESSAGE
#define WARN_THROWS_MESSAGE DOCTEST_WARN_THROWS_MESSAGE
#define WARN_THROWS_AS_MESSAGE DOCTEST_WARN_THROWS_AS_MESSAGE
#define WARN_NOTHROW_MESSAGE DOCTEST_WARN_NOTHROW_MESSAGE
#define CHECK_MESSAGE DOCTEST_CHECK_MESSAGE
#define CHECK_FALSE_MESSAGE DOCTEST_CHECK_FALSE_MESSAGE
#define CHECK_THROWS_MESSAGE DOCTEST_CHECK_THROWS_MESSAGE
#define CHECK_THROWS_AS_MESSAGE DOCTEST_CHECK_THROWS_AS_MESSAGE
#define CHECK_NOTHROW_MESSAGE DOCTEST_CHECK_NOTHROW_MESSAGE
#define REQUIRE_MESSAGE DOCTEST_REQUIRE_MESSAGE
#define REQUIRE_FALSE_MESSAGE DOCTEST_REQUIRE_FALSE_MESSAGE
#define REQUIRE_THROWS_MESSAGE DOCTEST_REQUIRE_THROWS_MESSAGE
#define REQUIRE_THROWS_AS_MESSAGE DOCTEST_REQUIRE_THROWS_AS_MESSAGE
#define REQUIRE_NOTHROW_MESSAGE DOCTEST_REQUIRE_NOTHROW_MESSAGE

#define SCENARIO DOCTEST_SCENARIO
#define SCENARIO_TEMPLATE DOCTEST_SCENARIO_TEMPLATE
#define SCENARIO_TEMPLATE_DEFINE DOCTEST_SCENARIO_TEMPLATE_DEFINE
#define GIVEN DOCTEST_GIVEN
#define WHEN DOCTEST_WHEN
#define AND_WHEN DOCTEST_AND_WHEN
#define THEN DOCTEST_THEN
#define AND_THEN DOCTEST_AND_THEN

#define WARN_EQ DOCTEST_WARN_EQ
#define CHECK_EQ DOCTEST_CHECK_EQ
#define REQUIRE_EQ DOCTEST_REQUIRE_EQ
#define WARN_NE DOCTEST_WARN_NE
#define CHECK_NE DOCTEST_CHECK_NE
#define REQUIRE_NE DOCTEST_REQUIRE_NE
#define WARN_GT DOCTEST_WARN_GT
#define CHECK_GT DOCTEST_CHECK_GT
#define REQUIRE_GT DOCTEST_REQUIRE_GT
#define WARN_LT DOCTEST_WARN_LT
#define CHECK_LT DOCTEST_CHECK_LT
#define REQUIRE_LT DOCTEST_REQUIRE_LT
#define WARN_GE DOCTEST_WARN_GE
#define CHECK_GE DOCTEST_CHECK_GE
#define REQUIRE_GE DOCTEST_REQUIRE_GE
#define WARN_LE DOCTEST_WARN_LE
#define CHECK_LE DOCTEST_CHECK_LE
#define REQUIRE_LE DOCTEST_REQUIRE_LE
#define WARN_UNARY DOCTEST_WARN_UNARY
#define CHECK_UNARY DOCTEST_CHECK_UNARY
#define REQUIRE_UNARY DOCTEST_REQUIRE_UNARY
#define WARN_UNARY_FALSE DOCTEST_WARN_UNARY_FALSE
#define CHECK_UNARY_FALSE DOCTEST_CHECK_UNARY_FALSE
#define REQUIRE_UNARY_FALSE DOCTEST_REQUIRE_UNARY_FALSE

#define FAST_WARN_EQ DOCTEST_FAST_WARN_EQ
#define FAST_CHECK_EQ DOCTEST_FAST_CHECK_EQ
#define FAST_REQUIRE_EQ DOCTEST_FAST_REQUIRE_EQ
#define FAST_WARN_NE DOCTEST_FAST_WARN_NE
#define FAST_CHECK_NE DOCTEST_FAST_CHECK_NE
#define FAST_REQUIRE_NE DOCTEST_FAST_REQUIRE_NE
#define FAST_WARN_GT DOCTEST_FAST_WARN_GT
#define FAST_CHECK_GT DOCTEST_FAST_CHECK_GT
#define FAST_REQUIRE_GT DOCTEST_FAST_REQUIRE_GT
#define FAST_WARN_LT DOCTEST_FAST_WARN_LT
#define FAST_CHECK_LT DOCTEST_FAST_CHECK_LT
#define FAST_REQUIRE_LT DOCTEST_FAST_REQUIRE_LT
#define FAST_WARN_GE DOCTEST_FAST_WARN_GE
#define FAST_CHECK_GE DOCTEST_FAST_CHECK_GE
#define FAST_REQUIRE_GE DOCTEST_FAST_REQUIRE_GE
#define FAST_WARN_LE DOCTEST_FAST_WARN_LE
#define FAST_CHECK_LE DOCTEST_FAST_CHECK_LE
#define FAST_REQUIRE_LE DOCTEST_FAST_REQUIRE_LE
#define FAST_WARN_UNARY DOCTEST_FAST_WARN_UNARY
#define FAST_CHECK_UNARY DOCTEST_FAST_CHECK_UNARY
#define FAST_REQUIRE_UNARY DOCTEST_FAST_REQUIRE_UNARY
#define FAST_WARN_UNARY_FALSE DOCTEST_FAST_WARN_UNARY_FALSE
#define FAST_CHECK_UNARY_FALSE DOCTEST_FAST_CHECK_UNARY_FALSE
#define FAST_REQUIRE_UNARY_FALSE DOCTEST_FAST_REQUIRE_UNARY_FALSE

#endif // DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES

// this is here to clear the 'current test suite' for the current translation unit - at the top
DOCTEST_TEST_SUITE_END();

// add stringification for primitive/fundamental types
namespace doctest
{
namespace detail
{
    DOCTEST_TYPE_TO_STRING_IMPL(bool)
    DOCTEST_TYPE_TO_STRING_IMPL(float)
    DOCTEST_TYPE_TO_STRING_IMPL(double)
    DOCTEST_TYPE_TO_STRING_IMPL(long double)
    DOCTEST_TYPE_TO_STRING_IMPL(char)
    DOCTEST_TYPE_TO_STRING_IMPL(signed char)
    DOCTEST_TYPE_TO_STRING_IMPL(unsigned char)
    DOCTEST_TYPE_TO_STRING_IMPL(wchar_t)
    DOCTEST_TYPE_TO_STRING_IMPL(short int)
    DOCTEST_TYPE_TO_STRING_IMPL(unsigned short int)
    DOCTEST_TYPE_TO_STRING_IMPL(int)
    DOCTEST_TYPE_TO_STRING_IMPL(unsigned int)
    DOCTEST_TYPE_TO_STRING_IMPL(long int)
    DOCTEST_TYPE_TO_STRING_IMPL(unsigned long int)
#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
    DOCTEST_TYPE_TO_STRING_IMPL(long long int)
    DOCTEST_TYPE_TO_STRING_IMPL(unsigned long long int)
#endif // DOCTEST_CONFIG_WITH_LONG_LONG
} // namespace detail
} // namespace doctest

DOCTEST_CLANG_SUPPRESS_WARNING_POP
DOCTEST_MSVC_SUPPRESS_WARNING_POP
DOCTEST_GCC_SUPPRESS_WARNING_POP

#endif // DOCTEST_LIBRARY_INCLUDED
