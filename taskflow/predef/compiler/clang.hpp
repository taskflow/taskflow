// 2019-04-11 created by Tsung-Wei Huang
//   - modifed from boost/predef/compiler/clang.h
//   - modified the include path of comp_detected.hpp

#pragma once

#include "../version_number.hpp"

#define TF_COMP_CLANG TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__clang__)
#  define TF_COMP_CLANG_DETECTION TF_VERSION_NUMBER(__clang_major__,__clang_minor__,__clang_patchlevel__)
#endif

#ifdef TF_COMP_CLANG_DETECTION
#  if defined(TF_PREDEF_DETAIL_COMP_DETECTED)
#    define TF_COMP_CLANG_EMULATED TF_COMP_CLANG_DETECTION
#  else
#    undef TF_COMP_CLANG
#    define TF_COMP_CLANG TF_COMP_CLANG_DETECTION
#  endif
#  define TF_COMP_CLANG_AVAILABLE
#  include "comp_detected.hpp"
#endif

#define TF_COMP_CLANG_NAME "Clang"

