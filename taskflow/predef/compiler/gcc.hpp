// 2019-04-11 created by Tsung-Wei Huang
//   - modifed from boost/predef/compiler/clang.h
//   - modified the include path of comp_detected.hpp

#pragma once

#include "../version_number.hpp"

#define TF_COMP_GNUC TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__GNUC__)
#   if !defined(TF_COMP_GNUC_DETECTION) && defined(__GNUC_PATCHLEVEL__)
#       define TF_COMP_GNUC_DETECTION \
            TF_VERSION_NUMBER(__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__)
#   endif
#   if !defined(TF_COMP_GNUC_DETECTION)
#       define TF_COMP_GNUC_DETECTION \
            TF_VERSION_NUMBER(__GNUC__,__GNUC_MINOR__,0)
#   endif
#endif

#ifdef TF_COMP_GNUC_DETECTION
#   if defined(TF_PREDEF_DETAIL_COMP_DETECTED)
#       define TF_COMP_GNUC_EMULATED TF_COMP_GNUC_DETECTION
#   else
#       undef TF_COMP_GNUC
#       define TF_COMP_GNUC TF_COMP_GNUC_DETECTION
#   endif
#   define TF_COMP_GNUC_AVAILABLE
#   include "comp_detected.hpp"
#endif

#define TF_COMP_GNUC_NAME "Gnu GCC C/C++"



