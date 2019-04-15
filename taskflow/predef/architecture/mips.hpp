// 2019-04-15 created by Tsung-Wei Huang
//   - modified from boost/predef/architecture/mips.hpp

#pragma once

#define TF_ARCH_MIPS TF_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__mips__) || defined(__mips) || \
    defined(__MIPS__)
#   undef TF_ARCH_MIPS
#   if !defined(TF_ARCH_MIPS) && (defined(__mips))
#       define TF_ARCH_MIPS TF_VERSION_NUMBER(__mips,0,0)
#   endif
#   if !defined(TF_ARCH_MIPS) && (defined(_MIPS_ISA_MIPS1) || defined(_R3000))
#       define TF_ARCH_MIPS TF_VERSION_NUMBER(1,0,0)
#   endif
#   if !defined(TF_ARCH_MIPS) && (defined(_MIPS_ISA_MIPS2) || defined(__MIPS_ISA2__) || defined(_R4000))
#       define TF_ARCH_MIPS TF_VERSION_NUMBER(2,0,0)
#   endif
#   if !defined(TF_ARCH_MIPS) && (defined(_MIPS_ISA_MIPS3) || defined(__MIPS_ISA3__))
#       define TF_ARCH_MIPS TF_VERSION_NUMBER(3,0,0)
#   endif
#   if !defined(TF_ARCH_MIPS) && (defined(_MIPS_ISA_MIPS4) || defined(__MIPS_ISA4__))
#       define TF_ARCH_MIPS TF_VERSION_NUMBER(4,0,0)
#   endif
#   if !defined(TF_ARCH_MIPS)
#       define TF_ARCH_MIPS TF_VERSION_NUMBER_AVAILABLE
#   endif
#endif

#if TF_ARCH_MIPS
#   define TF_ARCH_MIPS_AVAILABLE
#endif

#define TF_ARCH_MIPS_NAME "MIPS"

