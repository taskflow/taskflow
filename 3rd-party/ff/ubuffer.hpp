/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file ubuffer.hpp
 *  \ingroup building_blocks
 *
 *  \brief This file contains the definition of the unbounded \p SWSR circular buffer used in 
 *  FastFlow
 *
 * Single-Writer/Single-Reader (SWSR) lock-free (wait-free) unbounded FIFO
 * queue.  No lock is needed around pop and push methods!!
 *
 * The key idea underneath the implementation is quite simple: the unbounded
 * queue is based on a pool of wait-free SWSR circular buffers (see
 * buffer.hpp).  The pool of buffers automatically grows and shrinks on demand.
 * The implementation of the pool of buffers carefully try to minimize the
 * impact of dynamic memory allocation/deallocation by using caching
 * strategies.
 *
 * More information about the uSWSR_Ptr_Buffer implementation and correctness
 * proof can be found in:
 *
 * M. Aldinucci, M. Danelutto, P. Kilpatrick, M. Meneghin, and M. Torquati,
 * "An Efficient Unbounded Lock-Free Queue for Multi-core Systems,"
 * in Proc. of 18th Intl. Euro-Par 2012 Parallel Processing, Rhodes Island,
 * Greece, 2012, pp. 662-673. doi:10.1007/978-3-642-32820-6_65
 *
 *
 * IMPORTANT:
 *
 * This implementation has been optimized for 1 producer and 1 consumer.
 * If you need to use more producers and/or more consumers you have
 * several possibilities (top-down order):
 *  1. to use an high level construct like the farm skeleton and compositions
 *     of multiple farms (in other words, use FastFlow ;-) ).
 *  2. to use one of the implementations in the MPMCqueues.hpp file
 *  3. to use the SWSR_Ptr_Buffer but with the mp_push and the mc_pop methods
 *     both protected by (spin-)locks in order to protect the internal data
 *     structures.
 *
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


 
#ifndef FF_uSWSR_PTR_BUFFER_HPP
#define FF_uSWSR_PTR_BUFFER_HPP

#include <assert.h>
#include <cassert>
#include <new>
#include <ff/dynqueue.hpp>
#include <ff/buffer.hpp>
#include <ff/spin-lock.hpp>
// #if defined(HAVE_ATOMIC_H)
// #include <asm/atomic.h>
// #else
// #include <ff/atomic/atomic.h>
// #endif
namespace ff {

/* Do not change the following define unless you know what you're doing */
#define INTERNAL_BUFFER_T SWSR_Ptr_Buffer  /* bounded SPSC buffer */

class BufferPool {
public:
    BufferPool(int cachesize, const bool fillcache=false, unsigned long size=-1)
        :inuse(cachesize),bufcache(cachesize) {
        bufcache.init(); // initialise the internal buffer and allocates memory

        if (fillcache) {
            assert(size>0);
            union { INTERNAL_BUFFER_T * buf; void * buf2;} p;
            for(int i=0;i<cachesize;++i) {
                p.buf = (INTERNAL_BUFFER_T*)malloc(sizeof(INTERNAL_BUFFER_T));
                new (p.buf) INTERNAL_BUFFER_T(size);
#if defined(uSWSR_MULTIPUSH)
                p.buf->init(true);
#else
                p.buf->init();
#endif
                bufcache.push(p.buf);
            }
        }

#if defined(UBUFFER_STATS)
        miss=0;hit=0;
        (void)padding1;        
#endif
    }
    
    ~BufferPool() {
        union { INTERNAL_BUFFER_T * b1; void * b2;} p;

        while(inuse.pop(&p.b2)) {
            p.b1->~INTERNAL_BUFFER_T();
            free(p.b2);
        }
        while(bufcache.pop(&p.b2)) {
            p.b1->~INTERNAL_BUFFER_T();	    
            free(p.b2);
        }
    }
    
    inline INTERNAL_BUFFER_T * next_w(unsigned long size)  { 
        union { INTERNAL_BUFFER_T * buf; void * buf2;} p;
        if (!bufcache.pop(&p.buf2)) {
#if defined(UBUFFER_STATS)
            ++miss;
#endif
            p.buf = (INTERNAL_BUFFER_T*)malloc(sizeof(INTERNAL_BUFFER_T));
            new (p.buf) INTERNAL_BUFFER_T(size);
#if defined(uSWSR_MULTIPUSH)        
            if (!p.buf->init(true)) return NULL;
#else
            if (!p.buf->init()) return NULL;
#endif
        }
#if defined(UBUFFER_STATS)
        else  ++hit;
#endif  
        inuse.push(p.buf);
        return p.buf;
    }
    

    inline INTERNAL_BUFFER_T * next_r()  { 
        union { INTERNAL_BUFFER_T * buf; void * buf2;} p;
        return (inuse.pop(&p.buf2)? p.buf : NULL);
    }
    

    inline void release(INTERNAL_BUFFER_T * const buf) {
        buf->reset();
        if (!bufcache.push(buf)) {
            buf->~INTERNAL_BUFFER_T();	    
            free(buf);
        }
    }

#if defined(UBUFFER_STATS)

    inline unsigned long readPoolMiss() {
        unsigned long m = miss;
        miss = 0;
        return m;
    }


    inline unsigned long readPoolHit() {
        unsigned long h = hit;
        hit = 0;
        return h;
    }
#endif

    // just empties the inuse bucket putting data in the cache
    void reset() {
        union { INTERNAL_BUFFER_T * b1; void * b2;} p;
        while(inuse.pop(&p.b2)) release(p.b1);
    }

    //
    // WARNING: using this call in the wrong place is very dangerous!!!!
    //
    void changesize(size_t newsz) {
        // the inuse queue should be empty
        union { INTERNAL_BUFFER_T * b1; void * b2;} p;
        while (inuse.pop(&p.b2))
            assert(1==0);
        dynqueue tmp;        
        while(bufcache.pop(&p.b2)) {
            p.b1->changesize(newsz);
            tmp.push(p.b2);
        }
        while(tmp.pop(&p.b2)) {
            bufcache.push(p.b2);
        }
    }

    
private:
#if defined(UBUFFER_STATS)
    unsigned long      miss,hit;
    long padding1[longxCacheLine-2];    
#endif

    dynqueue           inuse;    // of type dynqueue, that is a Dynamic (list-based) 
                                 // SWSR unbounded queue.
                                 // No lock is needed around pop and push methods.
    INTERNAL_BUFFER_T  bufcache; // This is a bounded buffer
};
    
// --------------------------------------------------------------------------------------
    
 /*! 
  * \class uSWSR_Ptr_Buffer
 *  \ingroup building_blocks
  *
  * \brief Unbounded Single-Writer/Single-Reader buffer (FastFlow unbound channel)
  *
  * The unbounded SWSR circular buffer is based on a pool of wait-free SWSR
  * circular buffers (see buffer.hpp). The pool of buffers automatically grows
  * and shrinks on demand. The implementation of the pool of buffers carefully
  * tries to minimize the impact of dynamic memory allocation/deallocation by
  * using caching strategies. The unbounded buffer is based on the
  * INTERNAL_BUFFER_T.
  *
  */ 
class uSWSR_Ptr_Buffer {
private:
    enum {CACHE_SIZE=32};

#if defined(uSWSR_MULTIPUSH)
    enum { MULTIPUSH_BUFFER_SIZE=16};

    // Multipush: push a bach of items.
    inline bool multipush() {
        if (buf_w->multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE)) {
            mcnt=0; 
            return true;
        }

        if (fixedsize) return false;
        // try to get a new buffer             
        INTERNAL_BUFFER_T * t = pool.next_w(size);
        assert(t); // if (!t) return false; // EWOULDBLOCK
        buf_w = t;
        in_use_buffers++;
        buf_w->multipush(multipush_buf,MULTIPUSH_BUFFER_SIZE);
        mcnt=0;
#if defined(UBUFFER_STATS)
        ++numBuffers
        //atomic_long_inc(&numBuffers);
#endif
        return true;
    }
#endif

public:
    /**
     *  \brief Constructor
     *
     */
    uSWSR_Ptr_Buffer(unsigned long n,
                     const bool fixedsize=false,
                     const bool fillcache=false):
        buf_r(0),buf_w(0),in_use_buffers(1),size(n),fixedsize(fixedsize),
        pool(CACHE_SIZE,fillcache,size) {
        init_unlocked(P_lock); init_unlocked(C_lock);
        pushPMF=&uSWSR_Ptr_Buffer::push;
        popPMF =&uSWSR_Ptr_Buffer::pop;
#if defined(UBUFFER_STATS)
        numBuffers = 0;
        //atomic_long_set(&numBuffers,0);
#endif
        // Avoid unused private field warning on padding fields
        //(void)padding1; (void)padding2; (void)padding3; (void)padding4;
        
    }
    
    /** \brief Destructor */
    ~uSWSR_Ptr_Buffer() {
        if (buf_r) {
            buf_r->~INTERNAL_BUFFER_T();
            free(buf_r);
        }
        // buf_w either is equal to buf_w or is freed by BufferPool destructor
    }
    
    /**
     *  \brief Initialise the unbounded buffer.
     */
    bool init() {
        if (buf_w || buf_r) return false;
#if defined(uSWSR_MULTIPUSH)
        if (size<=MULTIPUSH_BUFFER_SIZE) return false;
#endif
        buf_r = (INTERNAL_BUFFER_T*)::malloc(sizeof(INTERNAL_BUFFER_T));
        assert(buf_r);
        new ((void *)buf_r) INTERNAL_BUFFER_T(size);
#if defined(uSWSR_MULTIPUSH)        
        if (!buf_r->init(true)) return false;
#else
        if (!buf_r->init()) return false;
#endif
        buf_w = buf_r;

        return true;
    }
    
    /** 
     * \brief Returns true if the buffer is empty.
     *
     * \return \p true if empty, \p false otherwise
     */
    inline bool empty() {        
        if (buf_r != buf_w) return false;
        return buf_r->empty();
    }
    
    inline bool available()   { 
        return buf_w->available();
    }


    /**
     *  \brief Push
     *
     *  push the input value into the queue.\n
     *  If fixedsize has been set to \p true, this method may
     *  return false. This means EWOULDBLOCK 
     *  and the call should be retried.
     *
     *  \param data pointer to data to be pushed in the buffer
     *  \return \p false if \p fixedsize is set to \p true OR if \p data is NULL 
     *  OR if there is not a buffer to write to. 
     *  
     *  \return \p true if the push succedes. 
     */
    inline bool push(void * const data) {
        /* NULL values cannot be pushed in the queue */
        assert(data != NULL);

        // If fixedsize has been set to \p true, this method may
        // return false. This means EWOULDBLOCK 
        if (!available()) {

            if (fixedsize) return false;

            // try to get a new buffer             
            INTERNAL_BUFFER_T * t = pool.next_w(size);
            assert(t); //if (!t) return false; // EWOULDBLOCK
            buf_w = t;
            in_use_buffers++;
#if defined(UBUFFER_STATS)
            ++numBuffers;
            //atomic_long_inc(&numBuffers);
#endif
        }
        //DBG(assert(buf_w->push(data)); return true;);
        buf_w->push(data);
        return true;
    }

    inline bool mp_push(void *const data) {
        spin_lock(P_lock);
        bool r=push(data);  
        spin_unlock(P_lock);
        return r;
    }

#if defined(uSWSR_MULTIPUSH)
    /**
     *
     * massimot: experimental code
     *
     * This method provides the same interface of the push one but uses the
     * multipush method to provide a batch of items to the consumer thus
     * ensuring better cache locality and lowering the cache trashing.
     *
     * \return TODO
     */
    inline bool mpush(void * const data) {
        assert(data != NULL);
        
        if (mcnt==MULTIPUSH_BUFFER_SIZE) 
            return multipush();        

        multipush_buf[mcnt++]=data;

        if (mcnt==MULTIPUSH_BUFFER_SIZE) return multipush();

        return true;
    }

    inline bool flush() {
        return (mcnt ? multipush() : true);
    }
#endif /* uSWSR_MULTIPUSH */
    
    /**
     *  \brief Pop
     *  
     *  \param[out] data pointer-pointer to data
     *
     */
    inline bool  pop(void ** data) {
        assert(data != NULL);

        if (buf_r->empty()) { // current buffer is empty
            if (buf_r == buf_w) return false; 
            if (buf_r->empty()) { // we have to check again
                INTERNAL_BUFFER_T * tmp = pool.next_r();
                if (tmp) {
                    // there is another buffer, release the current one 
                    pool.release(buf_r); 
                    in_use_buffers--;
                    buf_r = tmp;                    

#if defined(UBUFFER_STATS)
                    --numBuffers;
                    //atomic_long_dec(&numBuffers);
#endif
                }
            }
        }
        //DBG(assert(buf_r->pop(data)); return true;);
        return buf_r->pop(data);
    }    


#if defined(UBUFFER_STATS)
    inline unsigned long queue_status() {
        return (unsigned long) numBuffers;
            //atomic_long_read(&numBuffers);
    }
    

    inline unsigned long readMiss() {
        return pool.readPoolMiss();
    }


    inline unsigned long readHit() {
        return pool.readPoolHit();
    }
#endif

    inline bool mc_pop(void ** data) {
        spin_lock(C_lock);
        bool r=pop(data);
        spin_unlock(C_lock);
        return r;
    }


    /** 
     * It returns the size of the buffer.
     *
     * \return The size of the buffer.
     */
    inline size_t buffersize() const { 
        if (!fixedsize) return (size_t)-1;        
        return buf_w->buffersize(); 
    };

    /**
     * It changes the size of the queue WITHOUT reallocating 
     * the internal buffers. It should be used mainly for 
     * reducing the size of the queue or to restore after
     * it has been previously reduced. 
     * NOTE: it forces the queue to behave as bounded queue!
     * 
     * WARNING: this is very a dangerous operation if executed 
     * while the queue is being used; if wrongly used, it
     * may lead to data loss or memory corruption!
     *
     */
    size_t changesize(size_t newsz) {
        assert(buf_r == buf_w); // just a sanity check that the queue is not being used
        size_t tmp=buf_w->changesize(newsz);
        assert(size == tmp);
        size = newsz;
        pool.changesize(newsz); 
        fixedsize=true;
        return tmp;
    }

    
    /**
     * \brief number of elements in the queue
     * \note This is just a rough estimation of the actual queue length. Not really possible
     * to be precise in a lock-free buffer.
     *
     */
    inline unsigned long length() const {
        unsigned long len = buf_r->length();
        if (buf_r == buf_w) return len;
        ssize_t in_use = in_use_buffers-2;
        return len+(in_use>0?in_use:0)*size+buf_w->length();
    }

    inline bool isFixedSize() const { return fixedsize; }

    inline void reset() {
        if (buf_r) buf_r->reset();
        if (buf_w) buf_w->reset();
        buf_w = buf_r;
        pool.reset();
    }

    /* pointer to member function for the push method */
    bool (uSWSR_Ptr_Buffer::*pushPMF)(void * const);
    /* pointer to member function for the ppop method */
    bool (uSWSR_Ptr_Buffer::*popPMF)(void **);


private:
    // Padding is required to avoid false-sharing between 
    // core's private cache
    ALIGN_TO_PRE(CACHE_LINE_SIZE) 
    INTERNAL_BUFFER_T * buf_r;
    ALIGN_TO_POST(CACHE_LINE_SIZE)

    ALIGN_TO_PRE(CACHE_LINE_SIZE)
    INTERNAL_BUFFER_T * buf_w;
    ALIGN_TO_POST(CACHE_LINE_SIZE)

    /* ----- two-lock used only in the mp_push and mc_pop methods ------- */
	ALIGN_TO_PRE(CACHE_LINE_SIZE) 
    lock_t P_lock;
    ALIGN_TO_POST(CACHE_LINE_SIZE)

	ALIGN_TO_PRE(CACHE_LINE_SIZE) 
    lock_t C_lock;
    ALIGN_TO_POST(CACHE_LINE_SIZE)
    /* -------------------------------------------------------------- */
#if defined(UBUFFER_STATS)
    std::atomic<unsigned long> numBuffers;
    //atomic_long_t numBuffers;
#endif

#if defined(uSWSR_MULTIPUSH)
    /* massimot: experimental code (see multipush)
     *
     */
    // local multipush buffer used by the mpush method
    void  * multipush_buf[MULTIPUSH_BUFFER_SIZE];
    int     mcnt;
#endif

    unsigned long       in_use_buffers; // used to estimate queue length
    unsigned long	    size;
    bool			    fixedsize;
    BufferPool			pool;
};



} // namespace ff

#endif /* FF_uSWSR_PTR_BUFFER_HPP */
