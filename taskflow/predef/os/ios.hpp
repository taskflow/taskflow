// 2019-05-20 created by Chun-Xun Lin
//   - modifed from boost/predef/os/ios.h
//   - modified the include path of os_detected.hpp

#pragma once

#include "../version_number.hpp"

#define TF_OS_IOS TF_VERSION_NUMBER_NOT_AVAILABLE

#if !defined(TF_PREDEF_DETAIL_OS_DETECTED) && ( \
    defined(__APPLE__) && defined(__MACH__) && \
    defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) \
    )
#   undef TF_OS_IOS
#   define TF_OS_IOS (__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__*1000)
#endif

#if TF_OS_IOS
#   define TF_OS_IOS_AVAILABLE
#   include "os_detected.hpp"
#endif

#define TF_OS_IOS_NAME "iOS"

