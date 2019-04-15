// 2019-04-15 created by Tsung-Wei Huang
//   - modified from boost/predef/architecture/x86.hpp
//   - expanded BOOST_PREDEF_MAKE_10_VV00 in x86_32

#pragma once

#define TF_ARCH_X86_64 TF_VERSION_NUMBER_NOT_AVAILABLE

// x86-64 -------------------------------------------------
#if defined(__x86_64) || defined(__x86_64__) || \
    defined(__amd64__) || defined(__amd64) || \
    defined(_M_X64)
#   undef TF_ARCH_X86_64
#   define TF_ARCH_X86_64 TF_VERSION_NUMBER_AVAILABLE
#endif

#if TF_ARCH_X86_64
#   define TF_ARCH_X86_64_AVAILABLE
#endif

#define TF_ARCH_X86_64_NAME "Intel x86-64"

// x86-32 -------------------------------------------------
#define TF_ARCH_X86_32 TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(i386) || defined(__i386__) || \
    defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(__i386) || \
    defined(_M_IX86) || defined(_X86_) || \
    defined(__THW_INTEL__) || defined(__I86__) || \
    defined(__INTEL__)
#   undef TF_ARCH_X86_32
#   if !defined(TF_ARCH_X86_32) && defined(__I86__)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(__I86__,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32) && defined(_M_IX86)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(((_M_IX86)/100)%100,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32) && defined(__i686__)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(6,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32) && defined(__i586__)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(5,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32) && defined(__i486__)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(4,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32) && defined(__i386__)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER(3,0,0)
#   endif
#   if !defined(TF_ARCH_X86_32)
#       define TF_ARCH_X86_32 TF_VERSION_NUMBER_AVAILABLE
#   endif
#endif

#if TF_ARCH_X86_32
#   define TF_ARCH_X86_32_AVAILABLE
#endif

#define TF_ARCH_X86_32_NAME "Intel x86-32"

// x86 ----------------------------------------------------
#define TF_ARCH_X86 TF_VERSION_NUMBER_NOT_AVAILABLE

#if TF_ARCH_X86_32 || TF_ARCH_X86_64
#   undef TF_ARCH_X86
#   define TF_ARCH_X86 TF_VERSION_NUMBER_AVAILABLE
#endif


#if TF_ARCH_X86
#   define TF_ARCH_X86_AVAILABLE
#endif

#define TF_ARCH_X86_NAME "Intel x86"



