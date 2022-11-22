/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file MPMCqueues.hpp
 *  \ingroup aux_classes
 *
 *  \brief This file contains several MPMC queue implementations. Not
 *  currently used.
 * 
 * This file contains the following
 * Multi-Producer/Multi-Consumer queue implementations:
 * \li  MPMC_Ptr_Queue   bounded MPMC queue by Dmitry Vyukov 
 * \li  uMPMC_Ptr_Queue  unbounded MPMC queue by Massimo Torquati
 * \li  uMPMC_Ptr_Queue  unbounded MPMC queue by Massimo Torquati
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
 
#ifndef FF_MPMCQUEUE_HPP
#define FF_MPMCQUEUE_HPP

/* 
 * This file contains Multi-Producer/Multi-Consumer queue implementations.
 * 
 *   * MPMC_Ptr_Queue   bounded MPMC queue by Dmitry Vyukov 
 *   * uMPMC_Ptr_Queue  unbounded MPMC queue by Massimo Torquati 
 *   * MSqueue          unbounded MPMC queue by Michael & Scott
 *
 *  - Author: 
 *     Massimo Torquati <torquati@di.unipi.it> <massimotor@gmail.com>
 *  
 *  - History
 *    10 Jul 2012: M. Aldinucci: Minor fixes 
 *     4 Oct 2015: M. Aldinucci: cleaning related to better c++11 compliance
 */


#include <cstdlib>
#include <vector>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>
#include <ff/allocator.hpp>
#include <ff/platforms/platform.h>
#include <ff/mpmc/asm/abstraction_dcas.h>
#include <ff/spin-lock.hpp>

 
/*
 * NOTE: You should define NO_STD_C0X if you want to avoid c++0x and c++11
 *
 */
 
#if ( (!defined(NO_STD_C0X))  &&  !(__cplusplus >= 201103L))
#pragma message ("Define -DNO_STD_C0X to use a non c++0x/c++11 compiler")
#endif

//#define NO_STD_C0X


// // Check for g++ version >= 4.5
// #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)
//  #include <atomic>
// #else
//  // Check for g++ version >= 4.4
//  #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
//   #include <cstdatomic>
//  #else
//   #define USE_STD_0X
//  #endif
// #endif
//#endif // USE_STD_C0X


#define CAS abstraction_cas

namespace ff {

/* 
 *  In the following we implement two kinds of queues: 
 *   - the MPMC_Ptr_Queue is an implementation of the ** bounded ** 
 *     Multi-Producer/Multi-Consumer queue algorithm by Dmitry Vyukov 
 *     (www.1024cores.net). It stores pointers.
 *
 *   - the uMPMC_Ptr_Queue implements an ** unbounded ** 
 *     Multi-Producer/Multi-Consumer queue which does not require 
 *     any special memory allocator to avoid dangling pointers. 
 *     The implementation blends together the MPMC_Ptr_Queue and the 
 *     uSWSR_Ptr_Buffer.
 *  
 */

/*!
 * \class MPMC_Ptr_Queue
 *  \ingroup aux_classes
 *
 * \brief An implementation of the \a bounded Multi-Producer/Multi-Consumer queue. Not currently used.
 *
 * This class describes an implementation of the MPMC queue inspired by the solution
 * proposed by <a href="https://sites.google.com/site/1024cores/home/lock-free-algorithms/queues/bounded-mpmc-queue" target="_blank">Dmitry Vyukov</a>. \n
 *
 * \note There are two versions 1) with atomic operations 2) using new C++0X standard (compile with -DUSE_STD_C0X).
 *
 *
 */
#if !defined(NO_STD_C0X)
#include <atomic>
    
class MPMC_Ptr_Queue {
private:
    struct element_t {
        std::atomic<unsigned long> seq;
        void *                     data;
    };
    
public:
    /*
     * \brief Constructor
     */
    MPMC_Ptr_Queue() {}
    
    /*
     * \brief Destructor
     */
    ~MPMC_Ptr_Queue() {
        if (buf) {
            delete [] buf;
            buf=NULL;
        }
    }

    /*    |  data  | seq |        |  data  | seq |        |  data  | seq |
     *    |  NULL  |  0  | ------ |  NULL  |  1  | ------ |  NULL  | ... |
     *    ||||||||||||||||        ||||||||||||||||        ||||||||||||||||
     *                |
     *                |
     *                | 
     *          pwrite pread
     */
     
    /**
     * \brief init
     */
    inline bool init(size_t size) {
        if (size<2) size=2;
        // we need a size that is a power 2 in order to set the mask 
        if (!isPowerOf2(size)) size = nextPowerOf2(size);
        mask = size-1;

        buf = new element_t[size];
        if (!buf) return false;
        for(size_t i=0;i<size;++i) {
            buf[i].data = NULL;
            buf[i].seq.store(i,std::memory_order_relaxed);
            
            // store method
            // Atomically stores the value 'i'. 
            //
            // Memory is affected according to the value of memory_order:
            // memory_order must be one of 
            //      std::memory_order_relaxed 
            //      std::memory_order_release 
            //      std::memory_order_seq_cst. 
            // Otherwise the behavior is undefined.  
        }
        pwrite.store(0,std::memory_order_relaxed);        
        pread.store(0,std::memory_order_relaxed);
        return true;
    }
    
    /** 
     * \brief push: enqueue data
     *
     * This method is non-blocking and costs one CAS per operation. 
     */
    inline bool push(void *const data) {
        unsigned long pw, seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;
        do {
            pw    = pwrite.load(std::memory_order_relaxed);
            node  = &buf[pw & mask];
            seq   = node->seq.load(std::memory_order_acquire);
            
            // load method
            // Atomically loads and returns the current value of the atomic variable. 
            // Memory is affected according to the value of memory_order. 
            
            if (pw == seq) { // CAS 
                if (pwrite.compare_exchange_weak(pw, pw+1, std::memory_order_relaxed))
                    break;

                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            } else 
                if (pw > seq) return false; // queue full
        } while(1);
        node->data = data;
        node->seq.store(seq+1,std::memory_order_release);
        return true;
    }

    /**
     * pop method: dequeue data from the queue.
     *
     * This is a non-blocking method.
     *
     */
    inline bool pop(void** data) {
        unsigned long pr, seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;

        do {
            pr    = pread.load(std::memory_order_relaxed);
            node  = &buf[pr & mask];
            seq   = node->seq.load(std::memory_order_acquire);

            long diff = seq - (pr+1);
            if (diff == 0) { // CAS
                if (pread.compare_exchange_weak(pr, (pr+1), std::memory_order_relaxed))
                    break;

                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            } else { 
                if (diff < 0) return false; // queue empty
            }
        } while(1);
        *data = node->data;
        node->seq.store((pr+mask+1), std::memory_order_release);
        return true;
    }
    
private:
    union {
        std::atomic<unsigned long>  pwrite; /// Pointer to the location where to write to
        char padding1[CACHE_LINE_SIZE]; 
    };
    union {
        std::atomic<unsigned long>  pread;  /// Pointer to the location where to read from
        char padding2[CACHE_LINE_SIZE]; 
    };
    element_t *                 buf;
    unsigned long               mask;
};


#else  // using internal atomic operations
#include <ff/mpmc/asm/atomic.h>
    
class MPMC_Ptr_Queue {
protected:

    struct element_t {
        atomic_long_t seq;
        void *        data;
    };

public:
    /**
     *  \brief Constructor
     */
    MPMC_Ptr_Queue() {}

    /**
     *
     * \brief Destructor
     */
    ~MPMC_Ptr_Queue() { 
        if (buf) {
            freeAlignedMemory(buf);
            buf = NULL;
        }
    }
    
    /*    |  data  | seq |        |  data  | seq |        |  data  | seq |
     *    |  NULL  |  0  | ------ |  NULL  |  1  | ------ |  NULL  | ... |
     *    ||||||||||||||||        ||||||||||||||||        ||||||||||||||||
     *                |
     *                |
     *                |
     *          pwrite pread
     */
    
    /**
     * \brief init
     */
    inline bool init(size_t size) {
        if (size<2) size=2;
        // we need a size that is a power 2 in order to set the mask 
        if (!isPowerOf2(size)) size = nextPowerOf2(size);
        mask = (unsigned long) (size-1);

        buf=(element_t*)getAlignedMemory(longxCacheLine*sizeof(long),size*sizeof(element_t));
        if (!buf) return false;
        for(size_t i=0;i<size;++i) {
            buf[i].data = NULL;
            atomic_long_set(&buf[i].seq,long(i));
        }
        atomic_long_set(&pwrite,0);
        atomic_long_set(&pread,0);

        return true;
    }

    /**
     * Push method: enqueue data in the queue.
     *
     * This method is non-blocking and costs one CAS per operation.
     *
     */
    inline bool push(void *const data) {
        unsigned long pw, seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;

        do {
            pw    = atomic_long_read(&pwrite);
            node  = &buf[pw & mask];
            seq   = atomic_long_read(&node->seq);

            if (pw == seq) {
                if (abstraction_cas((volatile atom_t*)&pwrite, (atom_t)(pw+1), (atom_t)pw)==(atom_t)pw) 
                    break;

                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            } else 
                if (pw > seq) return false;

        } while(1);
        node->data = data;
        //atomic_long_inc(&node->seq);
        atomic_long_set(&node->seq, (seq+1));
        return true;
    }
        
    /**
     * Pop method: dequeue data from the queue.
     *
     * This is a non-blocking method.
     *
     */
    inline bool pop(void** data) {
        unsigned long pr , seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;

        do {
            pr    = atomic_long_read(&pread);
            node  = &buf[pr & mask];
            seq   = atomic_long_read(&node->seq);
            long diff = seq - (pr+1);
            if (diff == 0) {
                if (abstraction_cas((volatile atom_t*)&pread, (atom_t)(pr+1), (atom_t)pr)==(atom_t)pr) 
                    break;

                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            } else { 
                if (diff < 0) return false;
            }

        } while(1);
        *data = node->data;
        atomic_long_set(&node->seq,(pr+mask+1));
        return true;
    }
    
private:
    // WARNING: on 64bit Windows platform sizeof(unsigned long) = 32 !!
    union {
        atomic_long_t  pwrite;
        char           padding1[CACHE_LINE_SIZE];
    };
    union {
        atomic_long_t  pread;
        char           padding2[CACHE_LINE_SIZE];
    };
protected:
    element_t *    buf;
    unsigned long  mask;
};


 
/*! 
 * \class uMPMC_Ptr_Queue
 *  \ingroup building_blocks
 *
 * \brief An implementation of the \a unbounded Multi-Producer/Multi-Consumer queue
 *
 * This class implements an \a unbounded  MPMC queue which does not require 
 * any special memory allocator to avoid dangling pointers. The implementation blends 
 * together the MPMC_Ptr_Queue and the uSWSR_Ptr_Buffer. \n
 *
 * It uses internal atomic operations.
 *
 * This class is defined in \ref MPMCqueues.hpp
 *
 */
class uMPMC_Ptr_Queue {
protected:
    enum {DEFAULT_NUM_QUEUES=4, DEFAULT_uSPSC_SIZE=2048};

    typedef void *        data_element_t;
    typedef atomic_long_t sequenceP_t;
    typedef atomic_long_t sequenceC_t;

public:
    /**
     * \brief Constructor
     */
    uMPMC_Ptr_Queue() {}
    
    /**
     * \brief Destructor
     */
    ~uMPMC_Ptr_Queue() {
        if (buf) {
            for(size_t i=0;i<(mask+1);++i) {
                if (buf[i]) delete (uSWSR_Ptr_Buffer*)(buf[i]);
            }
            freeAlignedMemory(buf);
            buf = NULL;
        }
        if (seqP) freeAlignedMemory(seqP);        
        if (seqC) freeAlignedMemory(seqC);
    }

    /**
     * \brief init
     */
    inline bool init(unsigned long nqueues=DEFAULT_NUM_QUEUES, size_t size=DEFAULT_uSPSC_SIZE) {
        if (nqueues<2) nqueues=2;
        if (!isPowerOf2(nqueues)) nqueues = nextPowerOf2(nqueues);
        mask = nqueues-1;

        buf=(data_element_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(data_element_t));
        seqP=(sequenceP_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(sequenceP_t));
        seqC=(sequenceP_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(sequenceC_t));

        for(size_t i=0;i<nqueues;++i) {
            buf[i]= new uSWSR_Ptr_Buffer(size);
            ((uSWSR_Ptr_Buffer*)(buf[i]))->init();
            atomic_long_set(&(seqP[i]),long(i));
            atomic_long_set(&(seqC[i]),long(i));
        }
        atomic_long_set(&preadP,0);
        atomic_long_set(&preadC,0);
        return true;
    }

    /**
     * \brief nonblocking push
     *
     * \return It always returns true
     */
    inline bool push(void *const data) {
        unsigned long pw,seq,idx;
        unsigned long bk = BACKOFF_MIN;
        do {
            pw    = atomic_long_read(&preadP);
            idx   = pw & mask;
            seq   = atomic_long_read(&seqP[idx]);
            if (pw == seq) {
                if (abstraction_cas((volatile atom_t*)&preadP, (atom_t)(pw+1), (atom_t)pw)==(atom_t)pw) 
                    break;
                
                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            } 
        } while(1);
        ((uSWSR_Ptr_Buffer*)(buf[idx]))->push(data); // cannot fail
        atomic_long_set(&seqP[idx],(pw+mask+1));
        return true;               
    }
    
    /**
     * \brieg nonblocking pop
     *
     */
    inline bool pop(void ** data) {
        unsigned long pr,idx;
		long seq;
        unsigned long bk = BACKOFF_MIN;

        do {
            pr     = atomic_long_read(&preadC);
            idx    = pr & mask;
            seq    = atomic_long_read(&seqC[idx]);
            if (pr == (unsigned long)seq) { 
                if (atomic_long_read(&seqP[idx]) <= (unsigned long)seq) return false; // queue 
                if (abstraction_cas((volatile atom_t*)&preadC, (atom_t)(pr+1), (atom_t)pr)==(atom_t)pr) 
                    break;

                // exponential delay with max value
                for(volatile unsigned i=0;i<bk;++i) ;
                bk <<= 1;
                bk &= BACKOFF_MAX;
            }  
        } while(1);
        ((uSWSR_Ptr_Buffer*)(buf[idx]))->pop(data);
        atomic_long_set(&seqC[idx],(pr+mask+1));
        return true;
    }

private:
    union {
        atomic_long_t  preadP;
        char           padding1[CACHE_LINE_SIZE];
    };
    union {
        atomic_long_t  preadC;
        char           padding2[CACHE_LINE_SIZE];
    };
protected:
    data_element_t *  buf;
    sequenceP_t    *  seqP;
    sequenceC_t    *  seqC;
    unsigned long     mask;

};
    



/*! 
 * \class MSqueue
 * \ingroup aux_classes
 *
 * \brief Michael and Scott MPMC. Not currently used.
 *
 * See:  M. Michael and M. Scott, "Simple, Fast, and Practical
 * Non-Blocking and Blocking Concurrent Queue Algorithms", PODC 1996.
 *
 * The MSqueue implementation is inspired to the one in the \p liblfds 
 * libraly that is a portable, license-free, lock-free data structure 
 * library written in C. The liblfds implementation uses double-word CAS 
 * (aka DCAS) whereas this implementation uses only single-word CAS 
 * since it relies on a implementation of a memory allocator (used to 
 * allocate internal queue nodes) which implements a deferred reclamation 
 * algorithm able to solve both the ABA problem and the dangling pointer 
 * problem.
 *
 * More info about liblfds can be found at http://www.liblfds.org
 *
 */
class MSqueue {
private:
    enum {MSQUEUE_PTR=0 };

    // forward decl of Node type
    struct Node;
 
    struct Pointer {
        Pointer() { ptr[MSQUEUE_PTR]=0;}

        inline bool operator !() {
            return (ptr[MSQUEUE_PTR]==0);
        }
        inline Pointer& operator=(const Pointer & p) {
            ptr[MSQUEUE_PTR]=p.ptr[MSQUEUE_PTR];
            return *this;
        }

        inline Pointer& operator=(Node & node) {
            ptr[MSQUEUE_PTR]=&node;
            return *this;
        }

        inline Pointer & getNodeNext() {
            return ptr[MSQUEUE_PTR]->next;
        }
        inline Node * getNode() { return  ptr[MSQUEUE_PTR]; }

        inline bool operator==( const Pointer& r ) const {
            return ((ptr[MSQUEUE_PTR]==r.ptr[MSQUEUE_PTR]));
        }

        inline operator volatile atom_t * () const { 
            union { Node* const volatile* p1; volatile atom_t * p2;} pn;
            pn.p1 = ptr;
            return pn.p2; 
        }
        inline operator atom_t * () const { 
            union { Node* const volatile* p1; atom_t * p2;} pn;
            pn.p1 = ptr;
            return pn.p2; 
        }
        
        inline operator atom_t () const { 
            union { Node* volatile p1; atom_t p2;} pn;
            pn.p1 = ptr[MSQUEUE_PTR];
            return pn.p2; 
        }

        inline void set(Node & node) {
            ptr[MSQUEUE_PTR]=&node;
        }

        inline void * getData() const { return ptr[MSQUEUE_PTR]->getData(); }

        Node * volatile ptr[1];
    } ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
    
    struct Node {
        Node():data(0) { next.ptr[MSQUEUE_PTR]=0;}
        Node(void * data):data(data) {
            next.ptr[MSQUEUE_PTR]=0;
        }
        
        inline operator atom_t * () const { return (atom_t *)next; }

        inline void   setData(void * const d) { data=d;}
        inline void * getData() const { return data; }

        Pointer   next;
        void    * data;
    } ALIGN_TO_POST(ALIGN_DOUBLE_POINTER);

    Pointer  head;
    long     padding1[longxCacheLine-1];
    Pointer  tail;
    long     padding2[longxCacheLine-1];;
    FFAllocator *delayedAllocator;

private:
    inline void allocnode(Pointer & p, void * data) {
        union { Node * p1; void * p2;} pn;

        if (delayedAllocator->posix_memalign((void**)&pn.p2,ALIGN_DOUBLE_POINTER,sizeof(Node))!=0) {
            abort();
        }            
        new (pn.p2) Node(data);
        p.set(*pn.p1);
    }

    inline void deallocnode( Node * n) {
        n->~Node();
        delayedAllocator->free(n);
    }

public:
    MSqueue(): delayedAllocator(NULL) { }
    
    ~MSqueue() {
        if (delayedAllocator)  {
            delete delayedAllocator;
            delayedAllocator = NULL;
        }
    }

    MSqueue& operator=(const MSqueue& v) { 
        head=v.head;
        tail=v.tail;
        return *this;
    }

    /** initialize the MSqueue */
    int init() {
        if (delayedAllocator) return 0;
        delayedAllocator = new FFAllocator(2); 
        if (!delayedAllocator) {
            error("MSqueue::init, cannot allocate FFAllocator\n");
            return -1;
        }

        // create the first NULL node 
        // so the queue is never really empty
        Pointer dummy;
        allocnode(dummy,NULL);
        
        head = dummy;
        tail = dummy;
        return 1;
    }

    // insert method, it never fails
    inline bool push(void * const data) {
        bool done = false;

        Pointer tailptr ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
        Pointer next    ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
        Pointer node    ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
        allocnode(node,data);

        do {
            tailptr = tail;
            next    = tailptr.getNodeNext();

            if (tailptr == tail) {
                if (!next) { // tail was pointing to the last node
                    done = (CAS((volatile atom_t *)(tailptr.getNodeNext()), 
                                (atom_t)node, 
                                (atom_t)next) == (atom_t)next);
                } else {     // tail was not pointing to the last node
                    CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
                }
            }
        } while(!done);
        CAS((volatile atom_t *)tail, (atom_t)node, (atom_t) tailptr);
        return true;
    }
    
    // extract method, it returns false if the queue is empty
    inline bool  pop(void ** data) {        
        bool done = false;

        ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer headptr;
        ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer tailptr;
        ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer next;

        do {
            headptr = head;
            tailptr = tail;
            next    = headptr.getNodeNext();

            if (head == headptr) {
                if (headptr.getNode() == tailptr.getNode()) {
                    if (!next) return false; // empty
                    CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
                } else {
                    *data = next.getData();
                    done = (CAS((volatile atom_t *)head, (atom_t)next, (atom_t)headptr) == (atom_t)headptr);
                }
            }
        } while(!done);

        deallocnode(headptr.getNode());
        return true;
    } 

    // return true if the queue is empty 
    inline bool empty() { 
        if ((head.getNode() == tail.getNode()) && !(head.getNodeNext()))
            return true;
        return false;            
    }
};


/* ---------------------- experimental code -------------------------- */



class multiSWSR {
protected:
    enum {DEFAULT_NUM_QUEUES=4, DEFAULT_uSPSC_SIZE=2048};

public:
    multiSWSR() {}
    
    ~multiSWSR() {
        if (buf) {
            for(size_t i=0;i<(mask+1);++i) {
                if (buf[i]) delete buf[i];
            }
            freeAlignedMemory(buf);
            buf = NULL;
        }
        if (PLock) freeAlignedMemory(PLock);        
        if (CLock) freeAlignedMemory(CLock);
    }

    inline bool init(unsigned long nqueues=DEFAULT_NUM_QUEUES, size_t size=DEFAULT_uSPSC_SIZE) {
        if (nqueues<2) nqueues=2;
        if (!isPowerOf2(nqueues)) nqueues = nextPowerOf2(nqueues);
        mask = nqueues-1;

        buf=(uSWSR_Ptr_Buffer**)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(uSWSR_Ptr_Buffer*));
        PLock=(CLHSpinLock*)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(CLHSpinLock));
        CLock=(CLHSpinLock*)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(CLHSpinLock));

        for(size_t i=0;i<nqueues;++i) {
            buf[i]= new uSWSR_Ptr_Buffer(size);
            buf[i]->init();
            PLock[i].init();
            CLock[i].init();
        }
        atomic_long_set(&count, 0);
        atomic_long_set(&enqueue,0);
        atomic_long_set(&dequeue,0);
        return true;
    }

    // it always returns true
    inline bool push(void *const data, int tid) {
        long q = atomic_long_inc_return(&enqueue) & mask;
        PLock[q].spin_lock(tid);
        buf[q]->push(data);
        PLock[q].spin_unlock(tid);
        atomic_long_inc(&count);
        return true;
    }
    
    // non-blocking pop
    inline bool pop(void ** data, int tid) {
        if (!atomic_long_read(&count))  return false; // empty

        long q = atomic_long_inc_return(&dequeue) & mask;
        CLock[q].spin_lock(tid);
        bool r = buf[q]->pop(data);
        CLock[q].spin_unlock(tid);
        if (r) { atomic_long_dec(&count); return true;}
        return false;
    }

private:
    union {
        atomic_long_t  enqueue;
        char           padding1[CACHE_LINE_SIZE];
    };
    union {
        atomic_long_t  dequeue;
        char           padding2[CACHE_LINE_SIZE];
    };
    union {
        atomic_long_t  count;
        char           padding3[CACHE_LINE_SIZE];
    };
protected:
    uSWSR_Ptr_Buffer **buf;
    CLHSpinLock *PLock;    
    CLHSpinLock *CLock;    
    size_t   mask;
};


/*
 * Simple and scalable Multi-Producer/Multi-Consumer queue.
 * By defining at compile time MULTI_MPMC_RELAX_FIFO_ORDERING it is possible 
 * to improve performance relaxing FIFO ordering in the pop method.
 *
 * The underling MPMC queue (the Q template parameter) should export at least 
 * the following methods:
 *
 *   bool push(T)
 *   bool pop(T&)
 *   bool empty() 
 *
 *
 */
template <typename Q>
class scalableMPMCqueue {
public:
    enum {DEFAULT_POOL_SIZE=4};

    scalableMPMCqueue() {
        //enqueue.store(0);
        //count.store(0);
        atomic_long_set(&enqueue,0);
        atomic_long_set(&count,0);

#if !defined(MULTI_MPMC_RELAX_FIFO_ORDERING)
        // NOTE: dequeue must start from 1 because enqueue is incremented
        //       using atomic_long_inc_return which first increments and than
        //       return the value.
        //dequeue.store(1);
        atomic_long_set(&dequeue,1);
#else
        //dequeue.store(0);
        atomic_long_set(&dequeue,0);
#endif
    }
    
    int init(size_t poolsize = DEFAULT_POOL_SIZE) {
        if (poolsize > pool.size()) {
            pool.resize(poolsize);
        }
        
        // WARNING: depending on Q, pool elements may need to be initialized  

        return 1;
    }

    // insert method, it never fails if data is not NULL
    inline bool push(void * const data) {
        //long q = (1 + enqueue.fetch_add(1)) % pool.size();
        long q = atomic_long_inc_return(&enqueue) % pool.size();
        bool r = pool[q].push(data);
        if (r) atomic_long_inc(&count);
        //if (r) count.fetch_add(1);
        return r;
    }

    // extract method, it returns false if the queue is empty
    inline bool  pop(void ** data) {      
        if (!atomic_long_read(&count))  return false; // empty
        //if (!count.load()) return false;
#if !defined(MULTI_MPMC_RELAX_FIFO_ORDERING)
        unsigned long bk = BACKOFF_MIN;
        //
        // enforce FIFO ordering for the consumers
        //
        long q, q1;
        do {
            q  = atomic_long_read(&dequeue), q1 = atomic_long_read(&enqueue);
            //q = dequeue.load(); q1 = enqueue.load();
            if (q > q1) return false;
            if (CAS((volatile atom_t *)&dequeue, (atom_t)(q+1), (atom_t)q) == (atom_t)q) break;
            //if(dequeue.compare_exchange_strong(<#long &__e#>, <#long __d#>)
            // exponential delay with max value
            for(volatile unsigned i=0;i<bk;++i) ;
            bk <<= 1;
            bk &= BACKOFF_MAX;
        } while(1);
        
        q %= pool.size(); 
        if (pool[q].pop(data)) {
            atomic_long_dec(&count);
            //count.fetch_sub(1);
            return true;
        }
        return false;
        
#else  // MULTI_MPMC_RELAX_FIFO_ORDERING
        long q = atomic_long_inc_return(&dequeue) % pool.size();
        bool r = pool[q].pop(data);
        if (r) { atomic_long_dec(&count); return true;}
        return false;
#endif        
    }
    
    // check if the queue is empty
    inline bool empty() {
        for(size_t i=0;i<pool.size();++i)
            if (!pool[i].empty()) return false;
        return true;
    }
private:
    // std::atomic<long> enqueue;
    atomic_long_t enqueue;
    long padding1[longxCacheLine-sizeof(atomic_long_t)];
    //std::atomic<long> dequeue;
    atomic_long_t dequeue;
    long padding2[longxCacheLine-sizeof(atomic_long_t)];
    //std::atomic<long> count;
    atomic_long_t count;
    long padding3[longxCacheLine-sizeof(atomic_long_t)];
protected:
    std::vector<Q> pool;
};

/* 
 * multiMSqueue is a specialization of the scalableMPMCqueue which uses the MSqueue 
*/
    class multiMSqueue: public scalableMPMCqueue<MSqueue> {
    public:
        
        multiMSqueue(size_t poolsize = scalableMPMCqueue<MSqueue>::DEFAULT_POOL_SIZE) {
            if (! scalableMPMCqueue<MSqueue>::init(poolsize)) {
                error("multiMSqueue init ERROR\n");
                abort();
            }
            
            for(size_t i=0;i<poolsize;++i)
                if (pool[i].init()<0) {
                    error("multiMSqueue init ERROR\n");
                    abort();
                }
        }
    };




#endif // USE_STD_C0X



/* ---------------------- MaX experimental code -------------------------- */
#if 0
/*
 *
 *   bool push(T)
 *   bool pop(T&)
 *   bool empty() 
 *
 *
 */
    typedef struct{
        unsigned long data;
        unsigned long next;        
        long padding1[64-2*sizeof(unsigned long)];
    }utMPMC_list_node_t;

    typedef struct{
        /*HEAD*/
        utMPMC_list_node_t* head;
        long padding0[64-sizeof(unsigned long)];
        /*TAIL*/
        utMPMC_list_node_t* tail;        
        long padding1[64-sizeof(unsigned long)];
    }utMPMC_list_info_t;

    typedef struct{
        /*address*/
        utMPMC_list_info_t l; 
        /*status*/
        unsigned long s;       
        long padding0[64-sizeof(unsigned long)]; 
    }utMPMC_VB_note_t;

#if !defined(NEXT_SMALLEST_2_POW)
#define NEXT_SMALLEST_2_POW(A) (1 << (32 - __builtin_clz((A)-1)))
#endif

#if !defined(VOLATILE_READ)
#define VOLATILE_READ(X)  (*(volatile typeof(X)*)&X)

#if !defined(OPTIMIZED_MOD_ON_2_POW)
#define OPTIMIZED_MOD_ON_2_POW(X,Y) ((X) & (Y))
#endif

#define IS_WRITABLE(STATUS,MYEQC) (STATUS==MYEQC)
#define WRITABLE_STATUS(STATUS,MYEQC) (MYEQC)
#define UPDATE_AFTER_WRITE(STATUS) (STATUS+1)

#define IS_READABLE(STATUS,MYDQC) (STATUS==MYDQC+1)
#define READABLE_STATUS(STATUS,MYDQC) (MYDQC+1)
#define UPDATE_AFTER_READ(STATUS,LEN) (STATUS+LEN-1)
#endif

    template <typename Q>
    class utMPMC_VB {
    public:
        enum {DEFAULT_POOL_SIZE=4};

        utMPMC_VB() {
            dqc =0;
            eqc = 0;
            /*
             * Both push and pop start from index 0
             */
            dqc = 0;
            eqc = 0;
        }
    
        int init(size_t vector_len) {
   
            len_v = NEXT_SMALLEST_2_POW(vector_len);
            len_v_minus_one = len_v-1;
            /*
             * Allocation and Init of the Vector
             */
            int done = posix_memalign((void **) v, longxCacheLine,
                                      sizeof(utMPMC_VB_note_t) * len_v);
            if (done != 0) {
                return 0;
            }
            int i = 0;
            for (i = 0; i < len_v; i++) {
                v[i].s = i;
                utMPMC_list_node_t * new_node;
                do{new_node = (utMPMC_list_node_t *)
                    malloc (sizeof(utMPMC_list_node_t));}while(new_node);
                new_node->data=NULL;
                new_node->next=NULL;
                v[i].l.tail=new_node;
                v[i].l.head=new_node;
            }

            return 1;
        }

    // insert method, it never fails!!
    inline bool push(void * const p) {
        utMPMC_list_node_t * new_node;
        do{new_node = (utMPMC_list_node_t *) 
            malloc (sizeof(utMPMC_list_node_t));}while(new_node);
        new_node->data= (unsigned long) p;
        new_node->next=NULL;

		unsigned long myEQC = __sync_fetch_and_add (&eqc, 1UL);;
		unsigned long myI = OPTIMIZED_MOD_ON_2_POW(myEQC, len_v_minus_one);

		unsigned long target_status = WRITABLE_STATUS(target_status, myEQC);
		do{}while(VOLATILE_READ(v[myI].s) != target_status);

        /* List Stuff TODO*/
        v[myI].l.tail->next = new_node;
        v[myI].l.tail = new_node;
        target_status = UPDATE_AFTER_WRITE(target_status);
        /*barrier*/
        __sync_synchronize();
        v[myI].s = target_status;
        
		return true;
    }

    // extract method, it returns false if the queue is empty
    inline bool  pop(void ** ret_val) {      
        	for (;;) {
		unsigned long myDQC = VOLATILE_READ(dqc);
		unsigned long myI = OPTIMIZED_MOD_ON_2_POW(myDQC, len_v_minus_one);
		unsigned long target_status = v[myI].s;


		if (IS_READABLE(target_status,myDQC) && (v[myI].l.tail!=v[myI].l.head)) {
			int atomic_result = __sync_bool_compare_and_swap(&dqc, myDQC,
					myDQC + 1);
			if (atomic_result) {
				/*
				 * that is my lucky day!! I've fished something...
				 */
                utMPMC_list_node_t* to_be_remoed =  v[myI].l.head;
                /* First Advance */
                v[myI].l.head = v[myI].l.head->next;
                /* Secondly Extract elem */
                *ret_val = v[myI].l.head->data;
                /* update the rest */
				target_status = UPDATE_AFTER_READ(target_status,len_v);
                __sync_synchronize();                
				v[myI].s = target_status;
                free(to_be_remoed);
				return true;
			} else {
				continue;
			}
		} else {
			/*
			 * Check if someone changed the card while I was playing
			 */
			if (myDQC != VOLATILE_READ(dqc)) {
				continue;
			}
			if (VOLATILE_READ(eqc) != VOLATILE_READ(dqc)) {
				continue;
			}
			/*
			 * Sorry.. no space for you...
			 */
			return false;
		}
	}
	/*
	 * Impossible to reach this point!!!
	 */
	return true;
    }
    
//     inline bool empty() {
//         for(size_t i=0;i<pool.size();++i)
//             if (!pool[i].empty()) return false;
//          return true;
//     }
private:
        long padding0[64 - sizeof(unsigned long)];
        unsigned long eqc;
        long padding1[64 - sizeof(unsigned long)];
        unsigned long dqc;
        long padding2[64 - sizeof(unsigned long)];
        unsigned long len_v;
        unsigned long len_v_minus_one;
        utMPMC_VB_note_t * v;    
        long padding3[64 - 3*sizeof(unsigned long)];
    };

// /* 
//  * multiMSqueue is a specialization of the scalableMPMCqueue which uses the MSqueue 
// */
// class multiMSqueue: public scalableMPMCqueue<MSqueue> {
// public:

//     multiMSqueue(size_t poolsize = scalableMPMCqueue<MSqueue>::DEFAULT_POOL_SIZE) {
//         if (! scalableMPMCqueue<MSqueue>::init(poolsize)) {
//             std::cerr << "multiMSqueue init ERROR, abort....\n";
//             abort();
//         }
        
//         for(size_t i=0;i<poolsize;++i)
//             if (pool[i].init()<0) {
//                 std::cerr << "ERROR initializing MSqueue, abort....\n";
//                 abort();
//             }
//     }
// };
#endif

} // namespace

#endif /* FF_MPMCQUEUE_HPP */
