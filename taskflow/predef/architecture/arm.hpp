// 2019-04-15 created by Tsung-Wei Huang
//   - modified from boost/predef/architecture/arm.hpp

#pragma once

#include "../version_number.hpp"

#define TF_ARCH_ARM TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__arm__) || defined(__arm64) || defined(__thumb__) || \
    defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || \
    defined(_M_ARM)
#   undef TF_ARCH_ARM
#   if !defined(TF_ARCH_ARM) && defined(__arm64)
#       define TF_ARCH_ARM TF_VERSION_NUMBER(8,0,0)
#   endif
#   if !defined(TF_ARCH_ARM) && defined(__TARGET_ARCH_ARM)
#       define TF_ARCH_ARM TF_VERSION_NUMBER(__TARGET_ARCH_ARM,0,0)
#   endif
#   if !defined(TF_ARCH_ARM) && defined(__TARGET_ARCH_THUMB)
#       define TF_ARCH_ARM TF_VERSION_NUMBER(__TARGET_ARCH_THUMB,0,0)
#   endif
#   if !defined(TF_ARCH_ARM) && defined(_M_ARM)
#       define TF_ARCH_ARM TF_VERSION_NUMBER(_M_ARM,0,0)
#   endif
#   if !defined(TF_ARCH_ARM)
#       define TF_ARCH_ARM TF_VERSION_NUMBER_AVAILABLE
#   endif
#endif

#if TF_ARCH_ARM
#   define TF_ARCH_ARM_AVAILABLE
#endif

#define TF_ARCH_ARM_NAME "ARM"
