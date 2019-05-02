// 2019-05-20 created by Chun-Xun Lin
//   - modifed from boost/predef/os/macos.h
//   - modified the include path of os_detected.hpp

#pragma once 

#include "ios.hpp"
#include "../version_number.hpp"

#define TF_OS_MACOS TF_VERSION_NUMBER_NOT_AVAILABLE

#if !defined(TF_PREDEF_DETAIL_OS_DETECTED) && ( \
    defined(macintosh) || defined(Macintosh) || \
    (defined(__APPLE__) && defined(__MACH__)) \
    )
#   undef TF_OS_MACOS
#   if !defined(TF_OS_MACOS) && defined(__APPLE__) && defined(__MACH__)
#       define TF_OS_MACOS TF_VERSION_NUMBER(10,0,0)
#   endif
#   if !defined(TF_OS_MACOS)
#       define TF_OS_MACOS TF_VERSION_NUMBER(9,0,0)
#   endif
#endif

#if TF_OS_MACOS
#   define TF_OS_MACOS_AVAILABLE 
#   include "os_detected.hpp"
#endif

#define TF_OS_MACOS_NAME "Mac OS"

