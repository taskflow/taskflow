//#if defined(_MSC_VER) && (_MSC_VER<=1500)
// For Visual Studio <= 2008 (ver 9)
#ifndef HAVE_STDINT_H
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
