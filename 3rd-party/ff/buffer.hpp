/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file buffer.hpp
 *  \ingroup building_blocks
 *
 *  \brief This file contains the definition of the bounded \p SPSC channel
 *  buffer used in FastFlow
 *
 *  Single-Writer Single-Reader circular buffer.
 *  No lock is needed around pop and push methods.
 *  Wait-free and fence-free (in the TSO model). 
 * 
 *  A single NULL value is used to indicate buffer full and 
 *  buffer empty conditions.
 * 
 *  More details about the SWSR_Ptr_Buffer implementation 
 *  can be found in:
 *
 *  Massimo Torquati, "Single-Producer/Single-Consumer Queue on Shared Cache 
 *  Multi-Core Systems", TR-10-20, Computer Science Department, University
 *  of Pisa Italy,2010
 *  ( http://compass2.di.unipi.it/TR/Files/TR-10-20.pdf.gz )
 * 
 *  M. Aldinucci, M. Danelutto, P. Kilpatrick, M. Meneghin, and M. Torquati, 
 *  "An Efficient Unbounded Lock-Free Queue for Multi-core Systems," 
 *  in Proc. of 18th Intl. Euro-Par 2012 Parallel Processing, Rhodes Island, 
 *  Greece, 2012, pp. 662-673. doi:10.1007/978-3-642-32820-6_65
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
/* Author: Massimo Torquati
 *
 */

#ifndef FF_SWSR_PTR_BUFFER_HPP
#define FF_SWSR_PTR_BUFFER_HPP

#include <cstdlib>
#include <cstring>
//#include <atomic>
//#include <ff/atomic/abstraction_dcas.h>

#include <ff/sysdep.h>
#include <ff/config.hpp>

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

#include <ff/platforms/platform.h>

namespace ff {

// 64 bytes is the common size of a cache line
static const int longxCacheLine = (CACHE_LINE_SIZE/sizeof(long));
    
 
 /*! 
  * \class SWSR_Ptr_Buffer
  *  \ingroup building_blocks
  * 
  * \brief SPSC bound channel (Single-Writer/Single-Reader)
  *
  * This class describes the SWSR circular buffer, used in FastFlow to
  * implement a lock-free (wait-free) bounded FIFO queue. No lock is needed
  * around pop and push methods.
  *
  * A single NULL value is used to indicate buffer full and buffer empty
  * conditions.
  *
  * This class is defined in \ref buffer.hpp
  *
  */ 

class SWSR_Ptr_Buffer {
    /**
     * experimentally we found that a good value is between 
     * 2 and 6 cache lines (16 to 48 entries respectively)
     */
    enum {MULTIPUSH_BUFFER_SIZE=16};

private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
#if defined(NO_VOLATILE_POINTERS)
    unsigned long    pread;
    long padding1[longxCacheLine-1];
    unsigned long    pwrite;
    long padding2[longxCacheLine-1];
#else
    ALIGN_TO_PRE(CACHE_LINE_SIZE)
    volatile unsigned long pread;
    ALIGN_TO_POST(CACHE_LINE_SIZE)

    ALIGN_TO_PRE(CACHE_LINE_SIZE)
    volatile unsigned long pwrite;
    ALIGN_TO_POST(CACHE_LINE_SIZE)
#endif
    size_t     size;
    void    ** buf;
    
#if defined(SWSR_MULTIPUSH)
    /* massimot: experimental code (see multipush)
     *
     */
    long padding3[longxCacheLine-2];    
    // local multipush buffer used by the mpush method
    void  * multipush_buf[MULTIPUSH_BUFFER_SIZE];
    int     mcnt;
#endif

public:    
    /* pointer to member function for the push method */
    bool (SWSR_Ptr_Buffer::*pushPMF)(void * const);
    /* pointer to member function for the ppop method */
    bool (SWSR_Ptr_Buffer::*popPMF)(void **);
    
public:
    /** 
     *  Constructor.
     *
     *  \param n the size of the buffer
     */
    SWSR_Ptr_Buffer(unsigned long n, const bool=true):
        pread(0),pwrite(0),size(n),buf(0) {
        pushPMF=&SWSR_Ptr_Buffer::push;
        popPMF =&SWSR_Ptr_Buffer::pop;
        // Avoid unused private field warning on padding1, padding2
        //(void)padding1;
        //(void)padding2;
    }
    
    /** 
     * Default destructor 
     */
    ~SWSR_Ptr_Buffer() {
        // freeAlignedMemory is a function defined in 'sysdep.h'
        freeAlignedMemory(buf);
    }
    
    /** 
     *  It initialise the buffer. Allocate space (\p size) of possibly aligned
     *  memory and reset the pointers (read pointer and write pointer) by
     *  placing them at the beginning of the buffer.
     *
     *  \return TODO
     */
    bool init(const bool startatlineend=false) {
        if (buf || (size==0)) return false;

#if defined(SWSR_MULTIPUSH)
        if (size<MULTIPUSH_BUFFER_SIZE) return false;
#endif
        // getAlignedMemory is a function defined in 'sysdep.h'
        buf=(void**)getAlignedMemory(longxCacheLine*sizeof(long),size*sizeof(void*));
        if (!buf) return false;

        reset(startatlineend);

        return true;
    }

    /** 
     * It returns true if the buffer is empty.
     */
    inline bool empty() {
#if defined(NO_VOLATILE_POINTERS)
        return ((*(volatile unsigned long *)(&buf[pread]))==0);
#else
        return (buf[pread]==NULL);
#endif
    }
    
    /** 
     * It returns true if there is at least one room in the buffer.
     */
    inline bool available()   { 
#if defined(NO_VOLATILE_POINTERS)
        return ((*(volatile unsigned long *)(&buf[pwrite]))==0);
#else
        return (buf[pwrite]==NULL);
#endif
    }

    /** 
     * It returns the size of the buffer.
     *
     * \return The size of the buffer.
     */
    inline size_t buffersize() const { return size; };

    /**
     * It changes the size of the queue WITHOUT reallocating 
     * the internal buffer. It should be used mainly for 
     * reducing the size of the queue or to restore it after
     * is has been previously reduced. 
     * 
     * WARNING: this is very a dangerous operation if executed 
     * while the queue is being used; if wrongly used, it
     * may lead to data loss or memory corruption!
     *
     */
    size_t changesize(size_t newsz) {
        size_t tmp=size;
        size=newsz;
        return tmp;
    }

    
    /** 
     *  Push method: push the input value into the queue. A Write Memory
     *  Barrier (WMB) ensures that all previous memory writes are visible to
     *  the other processors before any later write is executed.  This is an
     *  "expensive" memory fence operation needed in all the architectures with
     *  a weak-ordering memory model, where stores can be executed out-of-order
     *  (e.g. PowerPc). This is a no-op on Intel x86/x86-64 CPUs.
     *
     *  \param data Element to be pushed in the buffer
     *
     *  \return TODO
     */
    inline bool push(void * const data) {     /* modify only pwrite pointer */
        assert(data != NULL);

        if (available()) {
            /**
             * Write Memory Barrier: ensure all previous memory write 
             * are visible to the other processors before any later
             * writes are executed.  This is an "expensive" memory fence
             * operation needed in all the architectures with a weak-ordering 
             * memory model where stores can be executed out-or-order 
             * (e.g. Powerpc). This is a no-op on Intel x86/x86-64 CPUs.
             */
            WMB(); 
            //std::atomic_thread_fence(std::memory_order_release);
            buf[pwrite] = data;
            pwrite = pwrite + ((pwrite+1 >=  size) ? (1-size): 1); // circular buffer
            return true;
        }
        return false;
    }

    /**
     * The multipush method, which pushes a batch of elements (array) in the
     * queue. NOTE: len should be a multiple of longxCacheLine/sizeof(void*)
     *
     */
    inline bool multipush(void * const data[], int len) {
        if ((unsigned)len>=size) return false;
        unsigned long last = pwrite + ((pwrite+ --len >= size) ? (len-size): len);
        unsigned long r    = len-(last+1), l=last;
        unsigned long i;

        if (buf[last]==NULL) {
            
            if (last < pwrite) {
                for(i=len;i>r;--i,--l) 
                    buf[l] = data[i];
                for(i=(size-1);i>=pwrite;--i,--r)
                    buf[i] = data[r];
                
            } else 
                for(int i=len;i>=0;--i) 
                    buf[pwrite+i] = data[i];
            
            WMB();
            pwrite = pwrite + ((last+1 >= size) ? 0 : (last+1));
#if defined(SWSR_MULTIPUSH)
            mcnt = 0; // reset mpush counter
#endif
            return true;
        }
        return false;
    }


#if defined(SWSR_MULTIPUSH)
    
    // massimot: experimental code
    /**
     * This method provides the same interface of the \p push method, but it
     * allows to provide a batch of items to
     * the consumer, thus ensuring better cache locality and 
     * lowering the cache trashing.
     *
     * \param data Element to be pushed in the buffer
     */
    inline bool mpush(void * const data) {
        assert(data);
        
        if (mcnt==MULTIPUSH_BUFFER_SIZE)
            return multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE);

        multipush_buf[mcnt++]=data;

        if (mcnt==MULTIPUSH_BUFFER_SIZE)
            return multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE);

        return true;
    }

    /* REW -- ? */
    inline bool flush() {
        return (mcnt ? multipush(multipush_buf,mcnt) : true);
    }
#endif /* SWSR_MULTIPUSH */
    

    /**
     * It is like pop but doesn't copy any data.
     *
     * \return \p true is alway returned.
     */
    inline bool  inc() {
        buf[pread]=NULL;
        pread = pread + ((pread+1 >= size) ? (1-size): 1); // circular buffer     
        return true;
    }           

    /** 
     *  Pop method: get the next value from the FIFO buffer.
     *
     *  \param data Pointer to the location where to store the 
     *  data popped from the buffer.
     */
    inline bool  pop(void ** data) {  /* modify only pread pointer */
        if (empty()) return false;        
        *data = buf[pread];
        //std::atomic_thread_fence(std::memory_order_acquire);
        return inc();
    } 
        
    /** 
     *  It returns the "head" of the buffer, i.e. the element pointed by the read
     *  pointer (it is a FIFO queue, so \p push on the tail and \p pop from the
     *  head). 
     *
     *  \return The head of the buffer.
     */
    inline void * top() const { return buf[pread]; }    

    /** 
     * Reset the buffer and move \p read and \p write pointers to the beginning
     * of the buffer (i.e. position 0). Also, the entire buffer is cleaned and
     * set to 0  
     */
    inline void reset(const bool startatlineend=false) { 
        if (startatlineend) {
            /**
             *  This is a good starting point if the multipush method will be
             *  used in order to reduce cache trashing.
             */
            pwrite = longxCacheLine-1;
            pread  = longxCacheLine-1;
        } else {
            pread=0;
            pwrite=0; 
        }
#if defined(SWSR_MULTIPUSH)        
        mcnt   = 0;
#endif  
        if (size<=512) for(unsigned long i=0;i<size;++i) buf[i]=0;
        else memset(buf,0,size*sizeof(void*));
    }

    /** 
     * It returns the length of the buffer 
     * (i.e. the actual number of elements it contains) 
     */
    inline unsigned long length() const {
        long tpread=pread, tpwrite=pwrite;
        long len = tpwrite-tpread;
        if (len>0) return (unsigned long)len;
        if (len<0) return (unsigned long)(size+len);
        if (buf[tpwrite]==NULL) return 0;
        return size;  
    }

    // Not yet implemented 
    inline bool mp_push(void *const) {
        abort();
        return false;
    }
    // Not yet implemented 
    inline bool mc_pop(void **) {
        abort();
        return false;
    }
    
    inline bool isFixedSize() const { return true; }
};

/*!
 * \class Lamport_Buffer.
 * \ingroup aux_classes
 *
 * \brief Implementation of the well-known Lamport's wait-free circular
 * buffer. Not currently used.
 *
 *
 */
class Lamport_Buffer {
private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
    volatile unsigned long    pread;
    long padding1[longxCacheLine-1];
    volatile unsigned long    pwrite;
    long padding2[longxCacheLine-1];

    const    size_t size;
    void         ** buf;
    
public:
    /**
     * Constructor
     */
    Lamport_Buffer(unsigned long n, const bool=true):
        pread(0),pwrite(0),size(n),buf(0) {
        // Avoid unused private field warning on padding1, padding2
        (void)padding1;
        (void)padding2;
    }
    
    /**
     * Destructor
     */
    ~Lamport_Buffer() {
        freeAlignedMemory(buf);
    }
    
    /**
     * It initialize the circular buffer. 
     *
     * \return If successful \p true is returned, otherwise \p false is
     * returned.
     */
    bool init() {
        assert(buf==0);
        buf=(void**)getAlignedMemory(longxCacheLine*sizeof(long),size*sizeof(void*));
        if (!buf) return false;
        reset();
        return true;
    }

    /**
     * It return true if the buffer is empty 
     */
    inline bool empty() { return (pwrite == pread);  }
    
    /**
     * It return true if there is at least one room in the buffer 
     */
    inline bool available()   { 
        const unsigned long next = pwrite + ((pwrite+1>=size)?(1-size):1);
        return (next != pread);
    }

    /**
     * TODO
     */
    inline size_t buffersize() const { return size; };
    
    /**
     * TODO
     */
    inline bool push(void * const data) {
        assert(data);

        const unsigned long next = pwrite + ((pwrite+1>=size)?(1-size):1);
        if (next != pread) {
            buf[pwrite] = data;
            /* We have to ensure that all writes have been committed 
             * in memory before we change the value of the pwrite
             * reference otherwise the reader can read stale data.
             */
            WMB(); 
            pwrite =next;
            return true;
        }
        return false;
    }

    /**
     * TODO
     */
    inline bool  pop(void ** data) {
        assert(data);

        if (empty()) return false;
        *data = buf[pread];
        pread = pread + ((pread+1 >= size) ? (1-size): 1);
        return true;
    }    
    
    /**
     * TODO
     */
    inline void reset() { 
        pread=0;
        pwrite=0; 
        if (size<=512) for(unsigned long i=0;i<size;++i) buf[i]=0;
        else memset(buf,0,size*sizeof(void*));
    }

    /**
     * TODO
     */
    inline unsigned long length() const {
        // long len = pwrite-pread;
        // if (len>=0) return len;
        //return size+len;
        long tpread=pread, tpwrite=pwrite;
        long len = tpwrite-tpread;
        if (len>0) return (unsigned long)len;
        if (len<0) return (unsigned long)(size+len);
        if (buf[tpwrite]==NULL) return 0;
        return size;  
    }
};

/*!
 *  @}
 *  \endlink
 */

} // namespace ff

#endif /* FF_SWSR_PTR_BUFFER_HPP */
