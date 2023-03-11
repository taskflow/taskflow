/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file spin-lock.hpp
 *  \ingroup building_blocks
 *
 *  \brief This file contains several alternative spin lock(s)
 *  implementations that can be used as FastFlow spin-lock
 *
 * CLH spin-lock, ticket lock, XCHG-based spin-lock,
 * AtomicFlagWrapper-based spin-lock, and counting ...
 */

/* ***************************************************************************
 *  
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
/* 
 *   Author: 
 *      Massimo Torquati <torquati@di.unipi.it> or <massimotor@gmail.com>    
 * 
 *    - April 2013 added CLHSpinLock 
 *    - February 2014 added AtomicFlagWrapper-based spin-lock
 *
 */
 
#ifndef FF_SPINLOCK_HPP
#define FF_SPINLOCK_HPP

// This code requires a c++11 compiler 
#include <ff/sysdep.h>
#include <ff/platforms/platform.h>
#include <ff/config.hpp>


#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
#include <atomic>
namespace ff {
#define _INLINE static inline

ALIGN_TO_PRE(CACHE_LINE_SIZE) struct AtomicFlagWrapper {
/* MA: MSVS 2013 does not allow initialisation of lock-free atomic_flag in the constructor. 
	Before removing the conditional compilation we should double-check that initialisation with .clear() really works in 
    all platforms.  
*/
#if defined(_MSC_VER)
	AtomicFlagWrapper() {
		F.clear();
		}
#endif
    // std::atomic_flag isn't copy-constructible, nor copy-assignable
    
    bool test_and_set(std::memory_order mo) {
        return F.test_and_set(mo);
    }
    void clear(std::memory_order mo) {
        F.clear(mo);
    }	
#if defined(_MSC_VER)
	std::atomic_flag F;
#else
    std::atomic_flag F=ATOMIC_FLAG_INIT;
#endif
}ALIGN_TO_POST(CACHE_LINE_SIZE);

typedef AtomicFlagWrapper lock_t[1];

_INLINE void init_unlocked(lock_t) { }
_INLINE void init_locked(lock_t)   { abort(); }
_INLINE void spin_lock(lock_t l) { 
    while(l->test_and_set(std::memory_order_acquire)) ;
}
_INLINE void spin_unlock(lock_t l) { l->clear(std::memory_order_release);}
}
#else
#pragma message ("FastFlow requires a c++11 compiler")
#endif


#if !defined(__GNUC__) && defined(_MSC_VER)
// An acquire-barrier exchange, despite the name

#define __sync_lock_test_and_set(_PTR,_VAL)  InterlockedExchangePointer( ( _PTR ), ( _VAL ))
#endif


#if defined(__GNUC__) || defined(_MSC_VER) || defined(__APPLE__)
/*
 * CLH spin-lock is a FIFO lock implementation which uses only the 
 * atomic swap operation. More info can be found in:  
 *
 * "Building FIFO and Priority-Queuing Spin Locks from Atomic Swap"
 * by Travis S. Craig in Techical Report 93-02-02 University of Washington
 * 
 * The following code is based on the implementation presented
 * in the synch1.0.1 (http://code.google.com/p/sim-universal-construction/) 
 * by Nikolaos D. Kallimanis (released under the New BSD licence).
 *
 */

ALIGN_TO_PRE(CACHE_LINE_SIZE) struct CLHSpinLock {
    typedef union CLHLockNode {
        bool locked;
        char align[CACHE_LINE_SIZE];
    } CLHLockNode;
    
    volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *Tail                    ALIGN_TO_POST(CACHE_LINE_SIZE);
    volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *MyNode[MAX_NUM_THREADS] ALIGN_TO_POST(CACHE_LINE_SIZE);
    volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *MyPred[MAX_NUM_THREADS] ALIGN_TO_POST(CACHE_LINE_SIZE);

    CLHSpinLock():Tail(NULL) {
        for (int j = 0; j < MAX_NUM_THREADS; j++) {
            MyNode[j] = NULL;
            MyPred[j] = NULL;
        }
    }
    ~CLHSpinLock() {
        if (Tail) freeAlignedMemory((void*)Tail);
        for (int j = 0; j < MAX_NUM_THREADS; j++) 
            if (MyNode[j]) freeAlignedMemory((void*)(MyNode[j]));
    }
    int init() {
        if (Tail != NULL) return -1;
        Tail = (CLHLockNode*)getAlignedMemory(CACHE_LINE_SIZE, sizeof(CLHLockNode));
        Tail->locked = false;
        for (int j = 0; j < MAX_NUM_THREADS; j++) 
            MyNode[j] = (CLHLockNode*)getAlignedMemory(CACHE_LINE_SIZE, sizeof(CLHLockNode));
        
        return 0;
    }

    // FIX
    inline void spin_lock(const int pid) {
        MyNode[pid]->locked = true;
#if defined(_MSC_VER) // To be tested
    MyPred[pid] = (CLHLockNode *) __sync_lock_test_and_set((void *volatile *)&Tail, (void *)MyNode[pid]);
#else //defined(__GNUC__)
    MyPred[pid] = (CLHLockNode *) __sync_lock_test_and_set((long *)&Tail, (long)MyNode[pid]);
#endif
    while (MyPred[pid]->locked == true) ;
    }
    
    inline void spin_unlock(const int pid) {
        MyNode[pid]->locked = false;
        MyNode[pid]= MyPred[pid];
    }
    
} ALIGN_TO_POST(CACHE_LINE_SIZE);
    
typedef CLHSpinLock clh_lock_t[1];

_INLINE void init_unlocked(clh_lock_t l) { l->init();}
_INLINE void init_locked(clh_lock_t) { abort(); }
_INLINE void spin_lock(clh_lock_t l, const int pid) { l->spin_lock(pid); }
_INLINE void spin_unlock(clh_lock_t l, const int pid) { l->spin_unlock(pid); }

#endif 
/*
#if !defined(__GNUC__) && defined(_MSC_VER)
// An acquire-barrier exchange, despite the name

#define __sync_lock_test_and_set(_PTR,_VAL)  InterlockedExchangePointer( ( _PTR ), ( _VAL ))
#endif
*/
#if 0

// -------------------------------------
// OLD IMPLEMENTATION -- TO BE DELETED
// -------------------------------------

// REW -- documentation
//#include <ff/sysdep.h>
//#include <ff/platforms/platform.h>
//#include <ff/config.hpp>
#include <ff/mpmc/asm/abstraction_dcas.h>

// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
#include <atomic>
#endif

#ifdef __cplusplus
namespace ff {
#define _INLINE static inline
#else
#define _INLINE __forceinline
#endif

// to be moved in platforms-specific, redefine some gcc intrinsics
#if !defined(__GNUC__) && defined(_MSC_VER)
// An acquire-barrier exchange, despite the name

#define __sync_lock_test_and_set(_PTR,_VAL)  InterlockedExchangePointer( ( _PTR ), ( _VAL ))
#endif

#if defined(USE_TICKETLOCK)

// NOTE: ticket lock implementation is experimental and for Linux only!

#if !defined(__linux__)
#error "Ticket-lock implementation only for Linux!"
#endif

#if !defined(LOCK_PREFIX)
#define LOCK_PREFIX "lock ; "
#endif

typedef  struct {unsigned int slock;}  lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0].slock=UNLOCKED;}
static inline void init_locked(lock_t l)   { abort(); }

/* Ticket-Lock from Linux kernel 2.6.37 */
static __always_inline void spin_lock(lock_t lock)
{
    int inc = 0x00010000;
    int tmp;

    asm volatile(LOCK_PREFIX "xaddl %0, %1\n"
         "movzwl %w0, %2\n\t"
         "shrl $16, %0\n\t"
         "1:\t"
         "cmpl %0, %2\n\t"
         "je 2f\n\t"
         "rep ; nop\n\t"
         "movzwl %1, %2\n\t"
         /* don't need lfence here, because loads are in-order */
         "jmp 1b\n"
         "2:"
         : "+r" (inc), "+m" (lock->slock), "=&r" (tmp)
         :
         : "memory", "cc");
}

static __always_inline void spin_unlock(lock_t lock)
{
    asm volatile(LOCK_PREFIX "incw %0"
         : "+m" (lock->slock)
         :
         : "memory", "cc");
}


#else /* TICKET_LOCK  */





// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))

ALIGN_TO_PRE(CACHE_LINE_SIZE) struct AtomicFlagWrapper {
/* MA: MSVS 2013 does not allow initialisation of lock-free atomic_flag in the constructor. 
	Before removing the conditional compilation we should double-check that initialisation with .clear() really works in all platforms. 
*/
#ifndef _MSC_VER
    AtomicFlagWrapper():F(ATOMIC_FLAG_INIT) {}
#else
	AtomicFlagWrapper() {
		F.clear();
		}
#endif
    // std::atomic_flag isn't copy-constructible, nor copy-assignable
    
    bool test_and_set(std::memory_order mo) {
        return F.test_and_set(mo);
    }
    void clear(std::memory_order mo) {
        F.clear(mo);
    }	

	std::atomic_flag F;
}ALIGN_TO_POST(CACHE_LINE_SIZE);

typedef AtomicFlagWrapper lock_t[1];

_INLINE void init_unlocked(lock_t l) { }
_INLINE void init_locked(lock_t l)   { abort(); }
_INLINE void spin_lock(lock_t l) { 
    while(l->test_and_set(std::memory_order_acquire)) ;
}
_INLINE void spin_unlock(lock_t l) { l->clear(std::memory_order_release);}

#else  // non C++11

/* -------- XCHG-based spin-lock --------- */
    
enum { UNLOCKED=0 };
typedef volatile int lock_t[1];

_INLINE void init_unlocked(lock_t l) { l[0]=UNLOCKED;}
_INLINE void init_locked(lock_t l)   { l[0]=!UNLOCKED;}

/*
 * NOTE: 
 *  The following 2 functions has been taken from Cilk (version 5.4.6). 
 *  The Cilk Project web site is  http://supertech.csail.mit.edu/cilk/
 *
 */

#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include <WinBase.h>
// windows platform
_INLINE void spin_lock(lock_t l) {
    while (InterlockedExchange((long *)l, 1) != UNLOCKED) {	
        while (l[0]) ;  /* spin using only reads - reduces bus traffic */
    }
}

_INLINE void spin_unlock(lock_t l) {
    /*
     *  Ensure that all previous writes are globally visible before any
     *  future writes become visible.
     */    
    WMB();
    l[0]=UNLOCKED;
}

#else // non windows platform

_INLINE void spin_lock(lock_t l) {
    while (xchg((int *)l, 1) != UNLOCKED) {
        while (l[0]) ;  /* spin using only reads - reduces bus traffic */
    }
}

_INLINE void spin_unlock(lock_t l) {
    /*
     *  Ensure that all previous writes are globally visible before any
     *  future writes become visible.
     */    
    WMB();
    l[0]=UNLOCKED;
}
#endif // windows platform spin_lock
#endif // C++11 check
#endif // TICKET_LOCK


#ifdef __cplusplus
} // namespace ff
#endif

#endif // 0

#endif /* FF_SPINLOCK_HPP */
