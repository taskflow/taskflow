// 2019-04-15 created by Tsung-Wei Huang
//   - modified from boost/predef/architecture/ppc.hpp

#pragma once

#define TF_ARCH_PPC TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__powerpc) || defined(__powerpc__) || \
    defined(__POWERPC__) || defined(__ppc__) || \
    defined(_M_PPC) || defined(_ARCH_PPC) || \
    defined(__PPCGECKO__) || defined(__PPCBROADWAY__) || \
    defined(_XENON)
#   undef TF_ARCH_PPC
#   if !defined (TF_ARCH_PPC) && (defined(__ppc601__) || defined(_ARCH_601))
#       define TF_ARCH_PPC TF_VERSION_NUMBER(6,1,0)
#   endif
#   if !defined (TF_ARCH_PPC) && (defined(__ppc603__) || defined(_ARCH_603))
#       define TF_ARCH_PPC TF_VERSION_NUMBER(6,3,0)
#   endif
#   if !defined (TF_ARCH_PPC) && (defined(__ppc604__) || defined(__ppc604__))
#       define TF_ARCH_PPC TF_VERSION_NUMBER(6,4,0)
#   endif
#   if !defined (TF_ARCH_PPC)
#       define TF_ARCH_PPC TF_VERSION_NUMBER_AVAILABLE
#   endif
#endif

#if TF_ARCH_PPC
#   define TF_ARCH_PPC_AVAILABLE
#endif

#define TF_ARCH_PPC_NAME "PowerPC"

