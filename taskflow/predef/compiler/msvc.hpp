// 2019-04-11 created by Tsung-Wei Huang
//   - modifed from boost/predef/compiler/visualc.h
//   - modified the include path of comp_detected.hpp

#pragma once

#include "../version_number.hpp"

#define TF_COMP_MSVC TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(_MSC_VER)
#   if !defined (_MSC_FULL_VER)
#       define TF_COMP_MSVC_BUILD 0
#   else
#       if _MSC_FULL_VER / 10000 == _MSC_VER
#           define TF_COMP_MSVC_BUILD (_MSC_FULL_VER % 10000)
#       elif _MSC_FULL_VER / 100000 == _MSC_VER
#           define TF_COMP_MSVC_BUILD (_MSC_FULL_VER % 100000)
#       else
#           error "Cannot determine build number from _MSC_FULL_VER"
#       endif
#   endif
    /*
    VS2014 was skipped in the release sequence for MS. Which
    means that the compiler and VS product versions are no longer
    in sync. Hence we need to use different formulas for
    mapping from MSC version to VS product version.
    */
#   if (_MSC_VER >= 1900)
#       define TF_COMP_MSVC_DETECTION TF_VERSION_NUMBER(\
            _MSC_VER/100-5,\
            _MSC_VER%100,\
            TF_COMP_MSVC_BUILD)
#   else
#       define TF_COMP_MSVC_DETECTION TF_VERSION_NUMBER(\
            _MSC_VER/100-6,\
            _MSC_VER%100,\
            TF_COMP_MSVC_BUILD)
#   endif
#endif

#ifdef TF_COMP_MSVC_DETECTION
#   if defined(TF_PREDEF_DETAIL_COMP_DETECTED)
#       define TF_COMP_MSVC_EMULATED TF_COMP_MSVC_DETECTION
#   else
#       undef TF_COMP_MSVC
#       define TF_COMP_MSVC TF_COMP_MSVC_DETECTION
#   endif
#   define TF_COMP_MSVC_AVAILABLE
#   include "comp_detected.hpp"
#endif

#define TF_COMP_MSVC_NAME "Microsoft Visual C/C++"
