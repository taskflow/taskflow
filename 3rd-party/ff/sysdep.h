
/*
 * The following has been taken from Cilk (version 5.4.6) file cilk-sysdep.h. 
 * The Cilk Project web site is  http://supertech.csail.mit.edu/cilk/
 *
 */

/*
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 *
 *  This library is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation; either version 2.1 of the License, or (at
 *  your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307,
 *  USA.
 *
 */

/* Don't just include config.h, since that is not installed. */
/* Instead, we must actually #define the useful things here. */
/* #include "../config.h" */
/* Compiler-specific dependencies here, followed by the runtime system dependencies.
 * The compiler-specific dependencies were originally written by Eitan Ben Amos.
 * Modified by Bradley.
 */

#ifndef FF_SPIN_SYSDEP_H
#define FF_SPIN_SYSDEP_H

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif


/***********************************************************\
 * Various types of memory barriers and atomic operations
\***********************************************************/

/* RISCV
   Marco Aldinucci 
   10/04/2022 02:08
   RISC-V-Linux/linux/arch/riscv/include/asm/barrier.h
*/

#if defined(__riscv)
#pragma message "RISCV detected - experimental"

#define nop()		__asm__ __volatile__ ("nop")

#define RISCV_FENCE(p, s) \
	__asm__ __volatile__ ("fence " #p "," #s : : : "memory")

/* These barriers need to enforce ordering on both devices or memory. */
#define mb()		RISCV_FENCE(iorw,iorw)
#define rmb()		RISCV_FENCE(ir,ir)
#define wmb()		RISCV_FENCE(ow,ow)

/* These barriers do not need to enforce ordering on devices, just memory. */
#define __smp_mb()	RISCV_FENCE(rw,rw)
#define __smp_rmb()	RISCV_FENCE(r,r)
#define __smp_wmb()	RISCV_FENCE(w,w)

#define WMB() __smp_wmb()
#define PAUSE() 

#endif  
  
/*------------------------
       POWERPC 
 ------------------------*/
#if defined(__powerpc__) || defined(__ppc__)
/* This version contributed by Matteo Frigo Wed Jul 13 2005.   He wrote:
 *   lwsync is faster than eieio and has the desired store-barrier
 *   behavior.  The isync in the lock is necessary because the processor is
 *   allowed to speculate on loads following the branch, which makes the
 *   program without isync incorrect (in theory at least---I have never
 *   observed such a speculation).
 */

#define WMB()    __asm__ __volatile__ ("lwsync" : : : "memory")
#define PAUSE()

/* atomic swap operation */
static __inline__ int xchg(volatile int *ptr, int x)
{
    int result;
    __asm__ __volatile__ (
			  "0: lwarx %0,0,%1\n stwcx. %2,0,%1\n bne- 0b\n isync\n" :
			  "=&r"(result) : 
			  "r"(ptr), "r"(x) :
			  "cr0");
    
    return result;
}
#endif

/*------------------------
       IA64
 ------------------------*/
#ifdef __ia64__

#define WMB()    __asm__ __volatile__ ("mf" : : : "memory")
#define PAUSE()

/* atomic swap operation */
static inline int xchg(volatile int *ptr, int x)
{
    int result;
    __asm__ __volatile ("xchg4 %0=%1,%2" : "=r" (result)
			: "m" (*(int *) ptr), "r" (x) : "memory");
    return result;
}
#endif

/*------------------------
         I386 
 ------------------------*/
#ifdef __i386__ 

#define WMB()    __asm__ __volatile__ ("": : :"memory")
#define PAUSE()  __asm__ __volatile__ ("rep; nop" : : : "memory")

/* atomic swap operation 
   Note: no "lock" prefix even on SMP: xchg always implies lock anyway
*/
static inline int xchg(volatile int *ptr, int x)
{
    __asm__("xchgl %0,%1" :"=r" (x) :"m" (*(ptr)), "0" (x) :"memory");
    return x;
}
#endif /* __i386__ */

/*------------------------
   ARM (Mauro Mulatero)
 ------------------------*/
#if defined(__arm__) || defined(__aarch64__)

#define isb() __asm__ __volatile__ ("isb" : : : "memory")
#define dsb() __asm__ __volatile__ ("dsb" : : : "memory")
#define dmb() __asm__ __volatile__ ("dmb" : : : "memory")
#define smp_mb()  dmb()
#define smp_rmb() dmb()
#define smp_wmb() dmb()

#define WMB()   __asm__ __volatile__ ("dmb st": : : "memory")
#define PAUSE()

#define xchg(ptr,x) \
  ((__typeof__(*(ptr)))__xchg((unsigned long)(x),(ptr),sizeof(*(ptr))))

static inline unsigned long __xchg(unsigned long x, volatile void *ptr, int size)
{
  unsigned long ret;
  // MA: updated 12/08/22 unsigned int ==> unsignet long
  unsigned long tmp;

  smp_mb();
  // MA: updated 12/08/22 teq %1, #0 ==>  teq %w1, #0
  switch (size) {
  case 1:
    asm volatile("@ __xchg1\n"
    "1: ldrexb  %0, [%3]\n"
    " strexb  %1, %2, [%3]\n"
    " teq %w1, #0\n"            
    " bne 1b"
      : "=&r" (ret), "=&r" (tmp)
      : "r" (x), "r" (ptr)
      : "memory", "cc");
    break;
  case 4:
    asm volatile("@ __xchg4\n"
    "1: ldrex %0, [%3]\n"
    " strex %1, %2, [%3]\n"
    " teq %1, #0\n"
    " bne 1b"
      : "=&r" (ret), "=&r" (tmp)
      : "r" (x), "r" (ptr)
      : "memory", "cc");
    break;
  default:
    break;
  }

  smp_mb();

  return ret;
}

#endif /* __arm__ */

/*------------------------
         amd_64
 ------------------------*/
#ifdef __x86_64

#define WMB()    __asm__ __volatile__ ("": : :"memory")
#define PAUSE()  __asm__ __volatile__ ("rep; nop" : : : "memory")

/* atomic swap operation */
static inline int xchg(volatile int *ptr, int x)
{
    __asm__("xchgl %0,%1" :"=r" (x) :"m" (*(ptr)), "0" (x) :"memory");
    return x;
}
#endif /* __x86_64 */

/*------------------------
  (Marco Aldinucci)
 ------------------------*/

static inline void *getAlignedMemory(size_t align, size_t size) {
  void *ptr;
  
#if (defined(_WIN32)) // || defined(__INTEL_COMPILER)) && defined(_WIN32)
  if (posix_memalign(&ptr,align,size)!=0)  // defined in platform.h
    return NULL; 
  // Fallback solution in case of strange segfaults on memory allocator
  //ptr = ::malloc(size);
#else // linux or MacOS >= 10.6
  if (posix_memalign(&ptr,align,size)!=0)
    return NULL; 
#endif
  
  /* ptr = (void *)memalign(align, size);
     if (p == NULL) return NULL;
  */
  return ptr;
}

static inline void freeAlignedMemory(void* ptr) {
#if defined(_WIN32)
  if (ptr) posix_memalign_free(ptr); // defined in platform.h
  // Fallback solution in case of strange segfaults
  //::free(ptr);
#else	
  if (ptr) ::free(ptr);
#endif  
}

#endif /* FF_SPIN_SYSDEP_H */
