/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file allocator.hpp
 *  \ingroup building_blocks core_patterns
 *
 *  \brief Implementations of the FastFlow's lock-free allocator.
 *
 * Here we defined the \ref ff::ff_allocator (1) and the ff::FFAllocator (2).
 *
 * 1. The ff_allocator allocates only large chunks of memory, slicing them up
 *    into little chunks all with the same size. Only one thread can perform
 *    malloc operations while any number of threads may perform frees using
 *    the ff_allocator.
 *
 *    The ff_allocator is based on the idea of Slab Allocator, for more details
 *    about Slab Allocator please see:
 *     Bonwick, Jeff. "The Slab Allocator: An Object-Caching Kernel Memory
 *     Allocator." Boston USENIX Proceedings, 1994.
 *
 * 2. Based on the ff_allocator, the FFAllocator has been implemented.
 *    It might be used by any number of threads to dynamically
 *    allocate/deallocate memory. You have to include allocator.hpp and just use
 *    <tt> ff::ff_malloc, ff::ff_free(), ff::ff_realloc, ff::ff_posix_memalign </tt>
 *
 * \note In all the cases it is possible, it is better to use the ff_allocator
 *       as it allows more control and is more efficient.
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
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc., 59
 *  Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this file
 *  does not by itself cause the resulting executable to be covered by the GNU
 *  General Public License.  This exception does not however invalidate any
 *  other reasons why the executable file might be covered by the GNU General
 *  Public License.
 *
 * **************************************************************************/

/* Lock-free FastFlow allocator.
 *
 * Here we defined the ff_allocator (1) and the FFAllocator (2).
 *
 * 1. The ff_allocator allocates only large chunks of memory, slicing them up
 *    into little chunks all with the same size. Only one thread can perform
 *    malloc operations while any number of threads may perform frees using
 *    the ff_allocator.
 *
 *    The ff_allocator is based on the idea of Slab Allocator, for more details
 *    about Slab Allocator please see:
 *     Bonwick, Jeff. "The Slab Allocator: An Object-Caching Kernel Memory
 *     Allocator." Boston USENIX Proceedings, 1994.
 *
 * 2. Based on the ff_allocator, the FFAllocator has been implemented.
 *    It might be used by any number of threads to dynamically
 *    allocate/deallocate memory. You have to include allocator.hpp and just use
 *    ff_malloc, ff_free, ff_realloc, ff_posix_memalign.
 *
 * Note: In all the cases it is possible, it is better to use the ff_allocator
 *       as it allows more control and is more efficient.
 *
 */

/*
 *
 *   Author:
 *      Massimo Torquati <torquati@di.unipi.it> or <massimotor@gmail.com>
 *
 *  - June  2008 first version
 *  - March 2011 main rework
 *  - June  2012  - performance improvement in getfrom_fb
 *                - statistics cleanup
 *  - Sept  2015 Marco Aldinucci: porting to c++11
 *
 */


#ifndef FF_ALLOCATOR_HPP
#define FF_ALLOCATOR_HPP


#include <assert.h>
#include <algorithm>

#include <ff/platforms/platform.h>

// #if defined(HAVE_ATOMIC_H)
// #include <asm/atomic.h>
// #else 
// #include <ff/atomic/atomic.h>
// #endif
#include <atomic>

#if defined(ALLOCATOR_STATS)
#include <iostream>
#endif


//#include <pthread.h>
#include <ff/ubuffer.hpp>
#include <ff/spin-lock.hpp>
#include <ff/svector.hpp>
#include <ff/utils.hpp>


//#define DEBUG_ALLOCATOR 1
#if defined(DEBUG_ALLOCATOR)
#define DBG(X) X
#else
#define DBG(X)
#endif

namespace ff {

    /*
     *  \ingroup building_blocks
     *
     *
     */
    enum { N_SLABBUFFER=9, MAX_SLABBUFFER_SIZE=8192};
    static const int POW2_MIN  = 5;
    static const int POW2_MAX  = 13;

    // array containing different possbile sizes for a slab buffer
    static const int buffersize[N_SLABBUFFER] = 
        { 32, 64,128,256,512,1024,2048,4096,8192  };

    // array containing the allowed numbers (quantity) of buffers. These values will be 
    // used together with the array of possible sizes for buffers: they will be paired 
    // with a specific buffer size and will denote the number of buffers of that size, 
    // present inside the cache.
    static const int nslabs_default[N_SLABBUFFER] =
        { 512,512,512,512,128,  64,  32,  16,   8 };

#if defined(ALLOCATOR_STATS)
    
    struct all_stats {
        std::atomic_long      nmalloc;
        std::atomic_long      nfree;
        std::atomic_long      nrealloc;
        std::atomic_long      sysmalloc;
        std::atomic_long      sysfree;
        std::atomic_long      sysrealloc;
        std::atomic_long      hit;
        std::atomic_long      miss;
        std::atomic_long      leakremoved;
        std::atomic_long      memallocated;
        std::atomic_long      memfreed;
        std::atomic_long      segmalloc;
        std::atomic_long      segfree;
        std::atomic_long      newallocator;
        std::atomic_long      deleteallocator;

        all_stats() {
            nmalloc.store(0);
            nfree.store(0);
            nrealloc.store(0);
            sysmalloc.store(0);
            sysfree.store(0);
            sysrealloc.store(0);
            hit.store(0);
            miss.store(0);
            leakremoved.store(0);
            memallocated.store(0);
            memfreed.store(0);
            segmalloc.store(0);
            segfree.store(0);
            newallocator.store(0);
            deleteallocator.store(0);
        }
    
        static all_stats *instance() {
            static all_stats allstats;
            return &allstats;
        }

        void print(std::ostream & out = std::cout) {
            out << "\n--- Allocator Stats ---\n"
                << "malloc        = " << nmalloc << "\n"
                << "  cache hit   = " << hit << "\n"
                << "  cache miss  = " << miss << "\n"
                << "  leakremoved = " << leakremoved  << "\n\n"
                << "free          = " << nfree << "\n"
                << "realloc       = " << nrealloc << "\n\n"
                << "mem. allocated= " << memallocated << "\n"
                << "mem. freed    = " << memfreed << "\n\n"
                << "segmalloc     = " << segmalloc << "\n"
                << "segfree       = " << segfree << "\n\n"
                << "sysmalloc     = " << sysmalloc << "\n"
                << "sysfree       = " << sysfree << "\n"
                << "sysrealloc    = " << sysrealloc << "\n\n";
            if (newallocator>0) {
                out << "\n ---- How many allocators ----\n"
                    << "new allocator    = " << newallocator << "\n"
                    << "delete allocator = " << deleteallocator << "\n\n";
            }
        }
    };

#define ALLSTATS(X) X
#else
#define ALLSTATS(X)
#endif /* ALLOCATOR_STATS */


    static inline bool isPowerOfTwoMultiple(size_t alignment, size_t multiple) {
        return alignment && ((alignment & (alignment-multiple))==0);
    }

    /*
     * \class SegmentAllocator
     *
     * \brief Uses standard \p libc allocator to allocate chunks of (possibly
     * aligned) memory. It also keeps track of the amount of allocated memory.
     *
     */

    class SegmentAllocator {
    public:
        SegmentAllocator():memory_allocated(0) { }
        ~SegmentAllocator() {}
    
        /*
         * Allocates a new segment of (possibly aligned) memory.  \n This method
         * uses standard \p malloc to allocate memory (or \p posix_memalign when
         * possible). \p malloc should guarantee a sufficiently well aligned memory
         * for any purpose, while \p posix_memalign is used to effectively allocate
         * memory aligned in a defined way (i.e. the address of the allocated
         * memory will be a multiple of a specific value).
         *
         * \param segment_size the quantity of memory to be allocated.
         *
         * \returns a pointer to the chunk of memory allocated.
         */
        void * newsegment(size_t segment_size) {
            //void * ptr = ::malloc(segment_size);
            void * ptr=NULL;
            // MA: Check if this does not lead to a page swap
#if defined (_SC_PAGESIZE)
            ptr = getAlignedMemory(sysconf(_SC_PAGESIZE),segment_size);
#else
            ptr = getAlignedMemory(4096,segment_size); // To be fixed for MSC
#endif
            if (!ptr) return NULL;

            memory_allocated += segment_size; // updt quantity of allocated memory

            ALLSTATS(all_stats::instance()->segmalloc.fetch_add(1);
                     all_stats::instance()->memallocated.fetch_add(segment_size));

#if defined(MAKE_VALGRIND_HAPPY)
            memset(ptr,0,segment_size);
#endif
            return ptr;
        }

        /*
         * Free \p segment_size amount of memory from the memory chunk pointed by
         * \p ptr.\n First, it checks that the total allocated memory is at least
         * as big as \p segment_size. Then, it frees memory and updates the counter
         * of the total allocated memory.
         *
         * \param ptr pointer to the memory chunk to be freed
         * \param segment_size quantity of memory to free from the chunk
         * requested.
         */
        void   freesegment(void * ptr, size_t segment_size) {
            DBG(assert(memory_allocated >=segment_size));
            //::free(ptr);
            freeAlignedMemory(ptr);

            memory_allocated -= segment_size;
            ALLSTATS( all_stats::instance()->segfree.fetch_add(1);
                      all_stats::instance()->memfreed.fetch_add(segment_size) );
        }
    
        /*
         * \return Returns the amount of allocated memory
         */
        size_t getallocated() const { return memory_allocated; }
    
    private:
        size_t       memory_allocated;
    };

    // forward declarations
    class SlabCache;
    class ff_allocator;

    /*
     * \struct Buf_ctl
     *
     * \brief A buffer controller
     *
     * Each slab buffer has a \p Buf_ctl structure at the beginning of the data.
     * This structure holds a back-pointer to the controlling slab.
     *
     *
     */
    struct Buf_ctl {  SlabCache  * ptr; };

    /*
     * \struct Seg_ctl
     *
     * \brief The segment controller
     *
     * Each slab segment has a \p Seg_ctl structure at the beginning of the
     * segment, responsible for maintaining the linkage with other slabs in the
     * cache, the number of buffers in use and the number of available buffers.
     *
     */
    struct Seg_ctl {
        // number of buffers in use
        DBG(size_t refcount;)
        
        // reference to the SlabCache that owns the segment
        SlabCache * cacheentry;
        
        // reference to the allocator
        ff_allocator * allocator;
        
        // number of items in the buffer freelist (i.e. Buffers available)
        //atomic_long_t        availbuffers;
        size_t  availbuffers;
    };


    /*
     * \struct xThreadData
     *
     * \brief A buffer shared among threads (i.e. leak queue)
     *
     * A buffer that is used as a channel for exchanging shared data among threads.
     * It is built upon FastFlow's unbounded Single-Writer / Single-Reader queues
     * (\p uSWSR_Ptr_Buffer). These buffers guarantee deadlock avoidance and
     * scalability in stream-oriented applications.
     *
     *
     */
    struct xThreadData {
        enum { LEAK_CHUNK=4096 };
    
        xThreadData(const bool /*allocator*/, size_t /*nslabs*/, const pthread_t key)
            : leak(0), key(key) {
            //leak = (uSWSR_Ptr_Buffer*)::malloc(sizeof(uSWSR_Ptr_Buffer));
            leak = (uSWSR_Ptr_Buffer*)getAlignedMemory(128,sizeof(uSWSR_Ptr_Buffer));
            if (!leak) abort();
            new (leak) uSWSR_Ptr_Buffer(LEAK_CHUNK);
            if (!leak->init()) abort();
        }

        ~xThreadData() {
            if (leak) { leak->~uSWSR_Ptr_Buffer(); freeAlignedMemory(leak); } // free(leak)
        }
    
        uSWSR_Ptr_Buffer * leak;   //
        const pthread_t    key;    // used to identify a thread (threadID)
        long padding[longxCacheLine-((sizeof(const pthread_t)+sizeof(uSWSR_Ptr_Buffer*))/sizeof(long))]; //
    };

    /* 
     *      SlabCache
     *    ----------------- <---
     *   | size            |    |
     *   | cur ptr seg     |    |
     *   | availbuffers    |    |
     *   | vect of buf-ptr |    |
     *    -----------------     |
     *                          |      -------------<--------------------<-----------
     *                          |     |             |                    |
     *                          |     v             |                    |
     *      (Slab segment)      |   -------------------------------------------------
     *                          |   |              | p |            | p |
     *                          |---|- cacheentry  | t |    data    | t |    data
     *  <---------------------------|- allocator   | r |            | r |
     *   (ptr to ff_allocator)      |- availbuffers|   |            |   |
     *                              -------------------------------------------------
     *                              |    Seg_ctl   | ^ |               | ^ |
     *                                               |                   |
     *                                               |--- Buf_ctl        |--- Buf_ctl
     *
     *
     *  NOTE: the segment overhead is:  sizeof(Seg_ctl) + (nbuffer * BUFFER_OVERHEAD)
     *
     */

    /*
     * \class SlabCache
     *
     * \brief Cache of slab segments.
     *
     * A SlabCache is meant to keep commonly used object in an initialised state,
     * available for use by kernel. Each cache contains blocks of contiguous pages
     * in memory (i.e. \a slab segments).
     *
     * Each slab segment consists of contiguous memory blocks, carved up into
     * equal-size chunks, with a reference count to indicate the number of
     * allocated chunks.
     *
     * The SlabCache is used to implement the efficient lock-free allocator used in
     * FastFlow (\p ff_allocator), which is based on the
     *
     * <a
     * href="http://www.ibm.com/developerworks/linux/library/l-linux-slab-allocator/"
     * target="_blank">Slab allocator</a>
     *
     * At the beginning of each segment there is a Seg_ctl structure, that is
     * responsible for maintaining the linkage with other slabs in the cache, keeps
     * track of the number of buffers in use and the number of available buffers.
     *
     * Each buffer in the segment is managed by a Buf_ctl structure, that holds a
     * back-pointer to the controlling slab.
     *
     */
    class SlabCache {    
    private:    
        enum {TICKS_TO_WAIT=500,TICKS_CNT=3};
        enum {BUFFER_OVERHEAD=sizeof(Buf_ctl)};
        enum {MIN_FB_CAPACITY=32};
    
        inline Seg_ctl * getsegctl(Buf_ctl * buf) {
            return *((Seg_ctl **)buf);
        }

        /*
         * This method creates a new slab, that is, it allocates a new segment of
         * large and possibly aligned memory.
         *
         * \returns 0 if the creation of a new Slab succeedes; a negative value
         * otherwise.
         */
        inline int newslab() {
            /*
             * Allocate a large chunk of memory. The size of each chunk is equal to
             * the size of the slab buffer times the num of slubs in the segment
             * plus the segment overhead.
             *
             * 'alloc' is of type SegmentAllocator
             */
            void * ptr = alloc->newsegment( size*nslabs + sizeof(Seg_ctl) +
                                            nslabs * BUFFER_OVERHEAD );
        
            /*
             * If the allocation of the new segment succeedes, the pointer to the
             * newly allocated segment of memory becomes the new Segment controller
             * and all its embedded data are reset.
             */
            if (ptr) {
                Seg_ctl * seg      = (Seg_ctl *)ptr; // Segment controller
                DBG(seg->refcount  = 0);             // number of buffers in use
                seg->cacheentry    = this;           // owner of the segment
                seg->allocator     = mainalloc;      // ref to the ff_allocator
            
                //atomic_long_set(&seg->availbuffers,nslabs);
            
                seg->availbuffers = nslabs;
                ++seg;
                *(Seg_ctl **)seg = (Seg_ctl *)ptr;
                buffptr          = (Buf_ctl *)seg;
                availbuffers    += nslabs;
                seglist.push_back(ptr);             // add new slab into list of seg
                return 0;
            }

            return -1;
        }

        /*
         * Free a slab of memory previously allocated. This method frees the memory
         * segment pointed by 'ptr'. The size of the freed memory is exactly the
         * same size of the allocated segment: the size of the slab buffer times
         * the num of slubs in the segment plus the segment overhead.
         *
         * By default, the 'seglist_rem'' is set to true. It means that it has to
         * remove the given pointer from the list of pointers that point to active
         * memory chunks.
         *
         */
        inline void freeslab(void * ptr, bool seglist_rem=true) {
            alloc->freesegment( ptr, size*nslabs +  sizeof(Seg_ctl) + nslabs * BUFFER_OVERHEAD );
        
            if (seglist_rem) {      /* remove ptr from seglist */
                svector<void *>::iterator b(seglist.begin()), e(seglist.end());
                for(;b!=e;++b)
                    if ((*b)==ptr) {
                        seglist.erase(b);
                        return;
                    }
                DBG(assert(1==0));
            }
        }

        /*
         * This function is called when freeing a slab and the allocator has been
         * deregistered. It checks whether the memory occupied by the slab segment
         * given as argument can actually be deallocated. And this is possible when
         * none of the slab buffers are in use. In this case, the memory occupied
         * by the slab segment can be deallocated and the segment removed from the
         * list of segments.
         *
         * \returns true if the operation suceeds; false otherwise.
         */
        inline bool checkReclaim(Seg_ctl  * seg) {
            bool r=false;
            spin_lock(lock);    // per-cache locking
            DBG(assert(seg->availbuffers <= (unsigned)nslabs));
            if (++(seg->availbuffers) == (unsigned long)nslabs) {
                freeslab(seg); /* Free the given segment of memory (a slab) */
                r=true;
            }
            spin_unlock(lock);
            return r;
        }

        inline void checkReclaimD(Seg_ctl  * seg) {
            if (++(seg->availbuffers) == (unsigned long)nslabs) {
                freeslab(seg); /* Free the given segment of memory (a slab) */
            }
        }


        // REW
        // TODO: The following have to be checked if it can use the last queue
        //       index  as in the getfrom_fb() method !!!!!
        //

        inline void * getfrom_fb_delayed() {
            if (fb_size==1) return 0;
            for(unsigned i=1;i<fb_size;++i) {
                if (fb[i]->leak->length()<(unsigned)delayedReclaim)  {
                    return 0; // cache is empty
                }
            }

            union { Buf_ctl * buf; void * ptr; } b={NULL};

            for(unsigned i=2;i<fb_size;++i) {
                fb[i]->leak->pop(&b.ptr);
                DBG(assert(b.ptr));
                checkReclaimD(getsegctl(b.buf));
            }

            fb[1]->leak->pop(&b.ptr);
            ALLSTATS(all_stats::instance()->leakremoved.fetch_add(1));
            return b.ptr;
        }

        inline void * getfrom_fb() {
            void * buf = 0;
            for(unsigned i=0;i<fb_size;++i) {
                unsigned k=(lastqueue+i)%fb_size;
                if (fb[k]->leak->pop((void **)&buf)) {
                    ALLSTATS(all_stats::instance()->leakremoved.fetch_add(1));
                    lastqueue=k;
                    return buf;
                }
            }

            // the free buffers are empty
            return 0;
        }

        inline int searchfb(const pthread_t key) {
            for(unsigned i=0;i<fb_size;++i)
                if (fb[i]->key == key) return (int)i;
            return -1;
        }

    public: 
        /*
         * Default Constructor
         *
         *  \param mainalloc a pointer to the allocator used (must be of type
         *  \p ff_allocator).
         *  \param delayedReclaim a flag to... REW
         *  \param alloc a pointer to a \p SegmentAllocator object, used to obtain large
         *  (and possibly aligned) chunks of memory from the operating system.
         *  \param sz the size of each buffer in a segment of the slab cache.
         *  \param ns number of buffers (slabs) in each segment (note that, as the
         *  size of each buffer increases, the number of buffers in each segment
         *  decreases).
         *
         */
        SlabCache( ff_allocator * const mainalloc, const int delayedReclaim,
                   SegmentAllocator * const alloc, size_t sz, int ns )
            : size(sz), nslabs(ns), fb(0), fb_capacity(0), fb_size(0),
              buffptr(0), availbuffers(0), alloc(alloc),
              mainalloc(mainalloc), delayedReclaim(delayedReclaim),lastqueue(0) { }
    
        /**
         * Destructor
         */
        ~SlabCache() {
            if (!nslabs) return;

            svector<void *>::iterator b(seglist.begin()), e(seglist.end());
            for(;b!=e;++b) {        // just check the list is not empty
                freeslab(*b,false); // seglist_rem = false: nothing is removed
            }
            seglist.clear();        // clear the list
            if (fb) {
                for(unsigned i=0;i<fb_size;++i)
                    if (fb[i]) {
                        fb[i]->~xThreadData();
                        ::free(fb[i]);
                    }
                ::free(fb);
            }
        }

        /*
         * Initialise a new SlabCache.
         *
         * This method creates a new slab, that is,
         * it allocates space for an unbounded buffer built upon the
         * uSWSR_Ptr_Buffer. Then it creates a new segment of large and (possibly)
         * aligned memory. The initial size of the buffer - set to a minimum value
         * equal to 32 - can be later increased, if necessary.
         *
         * \param prealloc flag that act as a mask against the creation of a new
         * Slab. By default it is set to \p true.
         *
         * \returns 0 if the creation of a new Slab succeedes and the prealloc mask
         * is set to \p true OR if the creation of a new Slab fails and the
         * prealloc mask is set to \p false. Gives a negative value otherwise.
         */
        int init(bool prealloc=true) {
            if (!nslabs) return 0;

            nomoremalloc.store(0);   // atomic variable set to 0

            init_unlocked(lock);

            /* Allocate space for leak queue */
            fb = (xThreadData**)::malloc(MIN_FB_CAPACITY*sizeof(xThreadData*));
            if (!fb) return -1;

            fb_capacity = MIN_FB_CAPACITY;            
            if ( prealloc && (newslab()<0) ) return -1;
            return 0;
        }
    
        /*
         * Register the calling thread into the shared buffer (leak queue).\n
         * In case the thread (ID) is already registered in the buffer, it returns a
         * pointer to its position in the buffer. If it is not registered, then
         * allocates space for a new thread's buffer and initialise the new buffer
         * with the key of the new thread. The \p allocator flag is set to false.
         *
         * \param allocator \p true if the calling thread is the allocator thread;
         * \p false by default.
         * \returns a pointer to the allocated thread in the buffer.
         */
        inline xThreadData * register4free(const bool allocator=false) {
            DBG(assert(nslabs>0));
        
            pthread_t key= pthread_self();  // obtain ID of the calling thread
            int entry = searchfb(key);      // search for threadID in leak queue
            if (entry<0) {                  // if threadID not found
                // allocate a new buffer for thread 'key'
                xThreadData * xtd = (xThreadData*)::malloc(sizeof(xThreadData));
                if (!xtd) return NULL;
                new (xtd) xThreadData(allocator, nslabs, key);

                /*
                 * REW
                 * if the size of the existing leak queue has reached the max
                 * capacity, allocate more space (::realloc) and update the queue's
                 * capacity. These operations are thread safe (mutual exclusion).
                 */
                spin_lock(lock);
                if (fb_size==fb_capacity) {
                    xThreadData ** fb_new = (xThreadData**)::realloc( fb,
                                                                      (fb_capacity+MIN_FB_CAPACITY) * sizeof(xThreadData*) );
                    if (!fb_new) {
                        spin_unlock(lock);
                        return NULL;
                    }
                    fb = fb_new;
                    fb_capacity += MIN_FB_CAPACITY;
                }
                // add the new buffer to the leak queue and release lock
                fb[entry=(int) fb_size++] = xtd;
                spin_unlock(lock);
            }
            DBG(assert(fb[entry]->key == key));
            return fb[entry];                   // position of new entry in buffer
        }

        /*
         * Deregister the allocator thread and reclaim memory
         */
        inline void deregisterAllocator(const bool reclaim) {
            DBG(assert(nslabs>0));
            DBG(assert(delayedReclaim==0));

            // atomic variable set to 1: allocator has been deregistered
            // and prevented from allocating
            nomoremalloc.store(1);

            // try to reclaim some memory
            for(unsigned i=0;i<fb_size;++i) {
                DBG(assert(fb[i]));
                if (reclaim) {
                    union { Buf_ctl * buf2; void * ptr; } b={NULL};

                    while(fb[i]->leak->pop(&b.ptr)) {   // while pop  succeedes
                        checkReclaim(getsegctl(b.buf2));
                    }
                }
            }
        }

        /*
         * Get an item from a Slab segment.
         *
         * \returns a pointer to the item obtained
         */
        inline void * getitem() {
            DBG(assert(nslabs>0));
            void * item = 0;

            /* try to get one item from the available ones */
            if (availbuffers) {
            avail:
                Seg_ctl * seg = *((Seg_ctl **)buffptr);
                DBG(assert(seg));
                DBG(++seg->refcount);       // incr num buffers in use
                --(seg->availbuffers);      // decr available buffers

                DBG(if (seg->refcount==1) assert(availbuffers==nslabs));

                item = (char *)buffptr + BUFFER_OVERHEAD;   // set data pointer
                if (--availbuffers) {
                    Seg_ctl ** ctl   = (Seg_ctl **)((char *)item + size);
                    *ctl             = seg;
                    buffptr          = (Buf_ctl *)ctl;
                } else buffptr = 0;

                DBG( if ((getsegctl((Buf_ctl *)((char *)item -
                                                sizeof(Buf_ctl))))->allocator == NULL)
                         abort() );
                return item;
            }

            // else, try to get a free item from cache
            item = delayedReclaim ? getfrom_fb_delayed() : getfrom_fb();

            if (item) {
                ALLSTATS(all_stats::instance()->hit.fetch_add(1));
                DBG(if ((getsegctl((Buf_ctl *)item))->allocator == NULL) abort());
                return ((char *)item + BUFFER_OVERHEAD);
            }

            ALLSTATS(all_stats::instance()->miss.fetch_add(1));

            /* if there are not available items try to allocate a new slab */
            if (newslab()<0) return NULL;
            goto avail;
            return NULL; // not reached
        }

        /*
         * Push an item into the slab
         *
         * \return false if the operation succedes; true if some memory has been
         * reclaimed
         */
        inline bool putitem(Buf_ctl * buf) {
            DBG(assert(buf)); DBG(assert(nslabs>0));

            /*
             * If nomoremalloc is set to 1, it means the allocator has been
             * deregistered and the memory in segment 'seg' could have been
             * reclaimed.
             */
            if (nomoremalloc.load()) {
                // NOTE: if delayedReclaim is set, freeing threads cannot pass here
                DBG(assert(delayedReclaim==0));

                Seg_ctl  * seg = *(Seg_ctl **)buf;
                return checkReclaim(seg);   // true if some memory can be reclaimed
            }

            /*
             * If nomorealloc is 0,
             */
            int entry = searchfb(pthread_self());   // look for calling thread
            xThreadData * xtd = NULL;
            if (entry<0) xtd = register4free();     // if not present, register it
            else xtd = fb[entry];                   // else, point to its position
            DBG(if (!xtd) abort());
            xtd->leak->push((void *)buf);           // push the item in the buffer
            return false;
        }

        /// Get the size of the SlabCache
        inline size_t getsize() const { return size; }
        /// Get the number of slabs in the cache
        inline size_t getnslabs() const { return nslabs;}
        inline size_t allocatedsize() const {
            return (alloc?alloc->getallocated():0);
        }

    protected:
        size_t                     size;            /* size of slab buffer */
        size_t                     nslabs;          /* num of buffers in segment */

        xThreadData            **  fb;              /* leak queue */
        size_t                     fb_capacity;     /* Min is 32 */
        size_t                     fb_size;
        lock_t                     lock;
            
        Buf_ctl *             buffptr;
        size_t                availbuffers;

    private:
        std::atomic_long            nomoremalloc;

        SegmentAllocator * const alloc;
        ff_allocator     * const mainalloc;     // the main allocator
        const int                delayedReclaim;
        unsigned                 lastqueue;
        svector<void *>          seglist;       // list of pointers to mem segments
    };

    /*!
     * \class ff_allocator
     * \ingroup building_blocks
     * \brief The ff_allocator, based on the idea of the
     * <a href="http://www.ibm.com/developerworks/linux/library/l-linux-slab-allocator/" target="_blank"> Slab allocator</a>
     *
     * A ff_allocator is owner by a single ff_node, which is the ff_node
     * which <tt></tt> it. The owner can
     * <tt> malloc/realloc/posix_memalign</tt>. Other ff_node can \p free
     * provided they /p register4free. A ff_node can init more than one
     * ff_allocator. Every violation cause unexpected results (e.g. segfault).
     *
     * The ff_allocator works over the SlabCache and is tuned to outperform
     * standard allocators' performance in a multi-threaded envirnoment. When it is
     * initialised, it creates a (predefined) number of SlabCaches, each one
     * containing a (predefined) number of buffers of different sizes, from 32 to
     * 8192 bytes. The thread that calls first the allocator object and wants to
     * allocate memory has to register itself to the shared leak queue. Only one
     * thread can perform this operation. The allocator obtains memory from the
     * pre-allocated SlabCache: if there is a slab (i.e. a buffer) big enough to
     * contain an object of the requested size, the pointer to that buffer is
     * passed to the thread that requested the memory. This latter thread has to
     * register as well to the same shared leak queue, so that when it has finished
     * its operations, it can return memory to the allocator thread.
     *
     * \note Very efficient but delicate to use, expert only. Not
     * designed to end-user but to build FastFlow pattern runtime
     * support. Not expert should use \ref ff::FFAllocator.
     *
     * \example perf_test_alloc1.cpp
     */
    class ff_allocator {
        friend class FFAllocator;
    public:

        /*
         * Check if there is a slab big enough to contain an object as big as \p
         * size.
         *
         * Since there is a limited number of possible slab sizes, and these
         * predefined dimensions are all powers of 2, it is easy to find which slub
         * can contain the object of the given size.
         *
         * \param size the size of the object that must be contained in a slab.
         *
         * \return  0  if the object can be contained in the slab of the
         * smallest size
         * \return  the index of the suitable slab size in the array of possible sizes, if
         * one exists
         * \return -1 if the object is too big to be contained in one of the slabs.
         */
        inline int getslabs(size_t size) {
            if (!size) return -1;
            --size;
            size_t e=1, sz=size;
            while(sz >>= 1) ++e;
            if (e < (size_t)POW2_MIN) return 0;
            if (e > (size_t)POW2_MAX) return -1;
            /*
              {
              for(int i=POW2_MAX-POW2_MIN+1;i<N_SLABBUFFER;++i)
              if ((size_t)buffersize[i]>=size) return i;
              }
            */
            return (int) e - POW2_MIN;
        }

    private:

        inline Seg_ctl * getsegctl(Buf_ctl * buf) {
            return *((Seg_ctl **)buf);
        }

        inline SlabCache * getslabs(Buf_ctl * buf) {
            return (getsegctl(buf))->cacheentry;
        }

    protected:

        /*
         * Frees the buffer \p buf and returns it to its slab segment
         *
         * \return \p true if some memory has been reclaimed
         */
        virtual inline bool   free(Buf_ctl * buf) {
            SlabCache * const entry = getslabs(buf);
            DBG( if (!entry) abort() );
            ALLSTATS(all_stats::instance()->nfree.fetch_add(1));
            return entry->putitem(buf);
        }


        inline size_t allocatedsize() {
            size_t s=0;
            svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
            for(;b!=e;++b) {
                if ((*b) && (*b)->getnslabs())
                    s+=(*b)->allocatedsize();
            }
            return s;
        }

    public:
        // FIX: we have to implement max_size !!!!
        /// Default Constructor

        ff_allocator(size_t /*max_size*/=0, const int delayedReclaim=0) :
            alloc(0), /* max_size(max_size), */ delayedReclaim(delayedReclaim) { 
        }

        /*
         * Destructor
         */
        virtual ~ff_allocator() {
            for(size_t i=0;i<slabcache.size(); ++i) {
                if (slabcache[i]) {
                    slabcache[i]->~SlabCache();
                    ::free(slabcache[i]);
                    slabcache[i] = NULL;
                }
            }
            if (alloc) {
                alloc->~SegmentAllocator();
                ::free(alloc);
                alloc=NULL;
            }
        }

        // initialize the allocator - Check this -- REW
        /**
         * \brief init the allocator
         *
         * Initialise the allocator. This method is called by one ff_node for
         * each data-path. (e.g. the Emitter in a Farm skeleton).
         * It creates a number of SlabCaches objects, a number specified by the
         * \p N_SLABBUFFER constant. The size of each created segment
         * goes from 32 (the first one created) to 8192 (the last).
         * Typically the number of buffers in a slab segment
         * decreases as the size of the slab increases.
         *
         * Default:
         * \code{.ccp}
         * enum { N_SLABBUFFER=9, MAX_SLABBUFFER_SIZE=8192};
         * const int buffersize[N_SLABBUFFER] = { 32, 64,128,256,512,1024,2048,4096,8192  };
         * const int nslabs_default[N_SLABBUFFER] = { 512,512,512,512,128,  64,  32,  16,   8 };
         * \endcode
         *
         * The number of nslabs is dynamically increased if needed recaliming more
         * memory from OS. This is a reltevely slow and not lock-free path of the code.
         *
         * \param _nslabs an array specifying the allowed numbers of buffers in a
         * SlabCache (overwrite \p nslabs_default )
         * \param prealloc if \true use preallocated segments
         *
         * \return 0 if initialisation succedes; a negative value otherwise
         */
        int init(const int _nslabs[N_SLABBUFFER]=0, bool prealloc=true) {
            svector<int> nslabs;  // used to specify the num of buffers in a slab

            if (_nslabs)
                for (int i=0;i<N_SLABBUFFER; ++i)
                    nslabs.push_back(_nslabs[i]);
            else
                for (int i=0;i<N_SLABBUFFER; ++i)
                    nslabs.push_back(nslabs_default[i]);

            /*
             * Allocate space for a SegmenAllocator object and
             * initialise it.
             */
            alloc = (SegmentAllocator *)::malloc(sizeof(SegmentAllocator));
            if (!alloc) return -1;
            new (alloc) SegmentAllocator();

            SlabCache * s = 0;
            slabcache.reserve(N_SLABBUFFER);


            /*
             * Allocate space for 'N_SLABBUFFER' caches and create SlabCache
             * objects. Buffers size within a cache size grows from 32 to 8192. if
             * not otherwise specified, the number of slab buffers in a segment
             * decreases as the size of the slab increases.
             */
            for(int i=0; i<N_SLABBUFFER; ++i) {
                s = (SlabCache*)::malloc(sizeof(SlabCache));
                if (!s) return -1;
                new (s) SlabCache( this, delayedReclaim, alloc,
                                   buffersize[i], nslabs[i] );
                if (s->init(prealloc)<0) {
                    error("ff_allocator:init: slab init fails!\n");
                    return -1;
                }
                slabcache.push_back(s);     // create a list of SlabCaches
            }
            return 0;
        }

        /**
         * \brief register ff_allocator (get ownership)
         *
         * The ff_node that allocates memory have to call this method in order to
         * register itself to the shared buffer. With this method, a ff_node is
         * allowed to allocate memory. Only one ff_node can allocate memory.
         *
         * \returns 0 if operation succedes, -1 otherwise
         */
        inline int registerAllocator() {
            svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
            for(;b!=e;++b) {
                if ((*b)->getnslabs())
                    if ((*b)->register4free(true)==0) return -1;
            }
            return 0;
        }

        /**
         * \brief de-register the ff_allocator (release ownership)
         *
         * Deregister the ff_allocator (i.e. release ownership) and reclaim
         * allocated memory back to the allocator. Every ff_node can perform this
         * action. No <tt>malloc/realloc/posix_memalign</tt> requires than one
         * ff_node own the ff_allocator
         *
         * \param reclaimMemory \p true if reclaim; \p false if deregister only
         */
        inline void deregisterAllocator(bool reclaimMemory=true) {
            svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
            for(;b!=e;++b) {
                if ((*b)->getnslabs())
                    (*b)->deregisterAllocator(reclaimMemory);
            }
        }


        /**
         * \brief register for the free operation
         *
         * Threads different from the allocator (i.e. those threads that do not
         * allocate memory) have to register themselves to the shared buffer by
         * calling this method. In this way they are provided with a chunk of
         * memory. Since they are registered, once their taks terminates they free
         * the memory assigned and that memory returns back to the allocator
         * thread's buffer, so that it can be reused.
         *
         */
        inline int register4free() {
            svector<SlabCache *>::iterator b(slabcache.begin()), e(slabcache.end());
            for(;b!=e;++b) {
                if ((*b)->getnslabs())
                    if ((*b)->register4free()==0) return -1;
            }

            return 0;
        }

        /**
        * \brief malloc
        *
        * Request to allocate \p size bytes. If the size is too
        * large, use OS malloc to get a new chunk of memory.
        *
        * To be called on a \ref ff::ff_allocator initialised via
        * \p ff:ff_allocator.init() on the calling \ref ff::ff_node
        * (owner).
        * Each \ref ff::ff_allocator object is owned by the \ref ff::ff_node
        * which \p registerAllocator it.
        * Only the owner can malloc on it, whereas other ff_node can
        * free. More than one \ref ff::ff_allocator per ff_node can
        * be defined.
        *
        * Violating ownership cause segfault or unspected behaviour.
        *
        * \param size the size of memory requested
        * \return pointer to allocated memory chunk
        *
        * \note Very efficient but delicate to use, expert only. Not
        * designed to end-user but to build FastFlow pattern runtime
        * support. Not expert should use \ref ff::FFAllocator
        */
        inline void * malloc(size_t size) {
            /**
             * use standard allocator if the size is too big or
             * we don't want to use the ff_allocator for that size
             */
            if ( size>MAX_SLABBUFFER_SIZE ||
                 (slabcache[getslabs(size)]->getnslabs()==0) ) {
                ALLSTATS(all_stats::instance()->sysmalloc.fetch_add(1));
                void * ptr = ::malloc(size+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)ptr)->ptr = 0;
                return (char *)ptr + sizeof(Buf_ctl);
            }

            // otherwise
            int entry = getslabs(size);
            DBG(if (entry<0) abort());
            ALLSTATS(all_stats::instance()->nmalloc.fetch_add(1));
            void * buf = slabcache[entry]->getitem();
            return buf;
        }

        /**
         * \brief ff posix_memalign.
         *
         * \param[out] *memptr pointer to a chunk of memory where the aligned memory will
         * be returned
         * \param alignment allocation's base address is an exact multiple of \p alignment, which
         * must be a power of 2 at least as large as sizeof(void *)
         * \param size the size of memory requested.
         * \return 0 if successful; otherwise it returns an error value.
         *
         * \note Very efficient but delicate to use, expert only. Not
         * designed to end-user but to build FastFlow pattern runtime
         * support. Not expert should use \ref ff::FFAllocator
         *
         */
        inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
            if (!isPowerOfTwoMultiple(alignment, sizeof(void *))) return -1;

            size_t realsize = size+alignment;

            /*
             * if the size is too big or we don't want to use the ff_allocator
             * for that size, use the standard allocator and force alignment
             */
            if (realsize > MAX_SLABBUFFER_SIZE ||
                (slabcache[getslabs(realsize)]->getnslabs()==0)) {
                ALLSTATS(all_stats::instance()->sysmalloc.fetch_add(1));
                void * ptr = ::malloc(realsize+sizeof(Buf_ctl));
                if (!ptr) return -1;
                ((Buf_ctl *)ptr)->ptr = 0;
                void * ptraligned = (void *)(  ( (long)((char*)ptr +
                                                        sizeof(Buf_ctl)) + alignment ) & ~(alignment-1)  );
                ((Buf_ctl *)((char *)ptraligned - sizeof(Buf_ctl)))->ptr = 0;
                *memptr = ptraligned;
                return 0;
            }

            // else use
            int entry = getslabs(realsize);
            DBG(if (entry<0) abort());
            ALLSTATS(all_stats::instance()->nmalloc.fetch_add(1));
            void * buf = slabcache[entry]->getitem();

            Buf_ctl * backptr = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
            DBG(assert(backptr));
            void * ptraligned = (void *)(((long)((char*)buf) + alignment) & ~(alignment-1));

            for(Buf_ctl *p=(Buf_ctl*)buf;p!=(Buf_ctl*)ptraligned;p++) {
                p->ptr = backptr->ptr;
            }

            *memptr = ptraligned;
            return 0;
        }


        // BUG 2 FIX: free fails if memory has been previously allocated using the
        // posix_memalign method with a size grater than MAX_SLABBUFFER_SIZE!!!!
        /**
         * \brief free
         *
         * free the requested memory chunk on a given ff::ff_allocator object.
         *
         * The memory should have been allocated using \p malloc on the same
         * object (otherwise segfault), and the ff_node should have been
         * \p ff::register4free() on the calling \ref ff::ff_node.
         *
         * Wrong registering cause segfault or unspected behaviour.
         *
         * \note Very efficient but delicate to use, expert only. Not
         * designed to end-user but to build FastFlow pattern runtime
         * support. Not expert should use \ref ff::FFAllocator
         *
         * \param ptr a pointer to the buffer.
         *
         */
        inline void   free(void * ptr) {
            if (!ptr) return;
            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));

            if (!buf->ptr) {
                ALLSTATS(all_stats::instance()->sysfree.fetch_add(1));
                ::free(buf);
                return;
            }
            Buf_ctl  * buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
            while(buf->ptr == buf2->ptr) {
                buf = buf2;
                buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
            }
            free(buf);
        }

        /**
         * \brief realloc
         *
         * It changes the size of the memory block pointed to by
         * \p ptr to \p newsize bytes. If the size is too large,
         * use OS malloc to get a new chunk of memory.
         *
         * To be called on a \ref ff::ff_allocator initialised via
         * \p registerAllocator() on the calling \ref ff::ff_node
         * (owner).
         * Each \ref ff::ff_allocator object is owned by the \ref ff::ff_node
         * which \p registerAllocator it.
         * Only the owner can malloc/realloc on it, whereas other ff_node can
         * free. More than one \ref ff::ff_allocator per ff_node can
         * be defined.
         *
         * Violating ownership cause segfault or unspected behaviour.
         *
         * \param size the size of memory requested
         * \return pointer to allocated memory chunk
         *
         * \note Very efficient but delicate to use, expert only. Not
         * designed to end-user but to build FastFlow pattern runtime
         * support. Not expert should use \ref ff::FFAllocator
         */
        inline void * realloc(void * ptr, size_t newsize) {
            if (ptr) {
                size_t oldsize;
                Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));

                if (!buf->ptr) {
                    ALLSTATS(all_stats::instance()->sysrealloc.fetch_add(1));
                    void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                    if (!ptr) return 0;
                    ((Buf_ctl *)newptr)->ptr = 0;
                    return (char *)newptr + sizeof(Buf_ctl);
                }
                ALLSTATS(all_stats::instance()->nrealloc.fetch_add(1));
                SlabCache * const entry = getslabs(buf);

                if (!entry) return 0;
                if ((oldsize=entry->getsize()) >= newsize) {
                    return ptr;
                }
                void * newptr = this->malloc(newsize);
                memcpy(newptr,ptr,oldsize);
                entry->putitem(buf);
                return newptr;
            }
            return this->malloc(newsize);
        }


        inline void * growsup(void * ptr, size_t newsize) {
            if (ptr) {
                size_t oldsize;
                Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));

                if (!buf->ptr) { /* WARNING: this is probably an error !!! */
                    ALLSTATS(all_stats::instance()->sysrealloc.fetch_add(1));
                    void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                    if (!ptr) return 0;
                    ((Buf_ctl *)newptr)->ptr = 0;
                    return (char *)newptr + sizeof(Buf_ctl);
                }

                SlabCache * const entry = getslabs(buf);
                if (!entry) return 0;
                if ((oldsize=entry->getsize()) >= newsize) {
                    return ptr;
                }
                entry->putitem(buf);
            }
            return this->malloc(newsize);
        }

        ALLSTATS( void printstats(std::ostream & out) {
                all_stats::instance()->print(out); }
            )

        private:
        svector<SlabCache *>   slabcache;       // List of caches
        SegmentAllocator     * alloc;
        /* const size_t           max_size; */  // TODO
        const int              delayedReclaim;
    };

    // forward decl
    class ffa_wrapper;
    static void FFAkeyDestructorHandler(void * kv);
    static void killMyself(ffa_wrapper * ptr);

    /*
     * \class ffa_wrapper
     *
     * \brief A wrapper of the ff_allocator.
     *
     * This wrapper acts as an intermediate level between the ff_allocator (which
     * it extends) and the FFAllocator (which uses the wrapper through the
     * FFAxThreadData structure).
     *
     * This class is defined in \ref allocator.hpp
     */
    class ffa_wrapper: public ff_allocator {
    protected:

        virtual inline bool free(Buf_ctl * buf) {
            bool reclaimed = ff_allocator::free(buf);

            // Do I have to kill myself?
            if (reclaimed && nomorealloc.load() && (allocatedsize()==0)) {
                killMyself(this);
            }
            return reclaimed;
        }

    public:
        friend void FFAkeyDestructorHandler(void * kv);


        ffa_wrapper(size_t max_size=0,const int delayedReclaim=0) :
            ff_allocator(max_size,delayedReclaim) {
            nomorealloc.store(0);
        }


        virtual ~ffa_wrapper() {}


        inline void * malloc(size_t size) {
            return ff_allocator::malloc(size);
        }

        inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
            return ff_allocator::posix_memalign(memptr,alignment,size);
        }


        inline void   free(void *) {  abort();  }


        inline void * realloc(void * ptr, size_t newsize) {
            return ff_allocator::realloc(ptr,newsize);
        }


        inline void * growsup(void * ptr, size_t newsize) {
            return ff_allocator::growsup(ptr,newsize);
        }
    private:
        inline void nomoremalloc() {
            nomorealloc.store(1);
        }

    private:
        std::atomic_long    nomorealloc;
    };


    struct FFAxThreadData {
        FFAxThreadData(ffa_wrapper * f): f(f) { }
        ffa_wrapper * f;
        long padding[longxCacheLine-(sizeof(ffa_wrapper*)/sizeof(long))];
    };

    /**
     * \class FFAllocator
     * \ingroup core_patterns
     * \brief A user-space parallel allocator (process-wide)
     *
     * Based on the \p ff_allocator, the FFAllocator might be used by any number
     * of \ref ff:ff_node (i.e. threads) to dynamically allocate/deallocate memory. It consists on
     * a network of  \p ff_allocator per \ref ff_node. It uses \ref ff::uSWSR_Ptr_Buffer
     * to avoid deadlocks.
     *
     * FFAllocator will be created as a static object on the first call to \ref FFAllocator::instance()
     *
     * Usage example:
     * \code{.cpp}
     * #define INIT() // no init needed
     * #define MALLOC(size)   (FFAllocator::instance()->malloc(size))
     * #define FREE(ptr,unused) (FFAllocator::instance()->free(ptr))
     * \endcode
     *
     * \example perf_test_alloc2.cpp
     */
    class FFAllocator {
        enum {MIN_A_CAPACITY=32};

    protected:
        inline Seg_ctl * getsegctl(Buf_ctl * buf) {
            return *((Seg_ctl **)buf);
        }

        inline FFAxThreadData * newAllocator( bool prealloc,
                                              int _nslabs[N_SLABBUFFER],
                                              size_t max_size )
        {
            FFAxThreadData * ffaxtd =
                (FFAxThreadData *)::malloc(sizeof(FFAxThreadData) +
                                           sizeof(ffa_wrapper));

            if (!ffaxtd) return NULL;
            ffa_wrapper * f = (ffa_wrapper*)((char *) ffaxtd +
                                             sizeof(FFAxThreadData));

            new (f) ffa_wrapper(max_size,delayedReclaim);
            new (ffaxtd) FFAxThreadData(f);

            if (f->init(_nslabs, prealloc)<0) {
                error("FFAllocator:newAllocator: init fails!\n");
                return NULL;
            }
            if (f->registerAllocator()<0) {     // register allocator to leak queue
                error("FFAllocator:newAllocator: registerAllocator fails!\n");
                return NULL;
            }

            // REW -- like in register4free
            // if more space in the SlabCache is needed, allocate more space
            // (::realloc) and update counters. Thread-safe operation (mutex)
            spin_lock(lock);
            if (A_size == A_capacity) {
                FFAxThreadData ** A_new = (FFAxThreadData**) ::realloc( A,
                                                                        (A_capacity+MIN_A_CAPACITY)*sizeof(FFAxThreadData*) );

                if (!A_new) { spin_unlock(lock); return NULL;}
                A=A_new;
                A_capacity += MIN_A_CAPACITY;
            }
            A[A_size++] = ffaxtd;
            spin_unlock(lock);

            ALLSTATS(all_stats::instance()->newallocator.fetch_add(1));

            return ffaxtd;
        }


    public:

        /**
         * \brief Constructor
         *
         * \param delayedReclaim Deferred reclamation configuration
         */
        FFAllocator(int delayedReclaim=0) :
            A(0), A_capacity(0), A_size(0),
            delayedReclaim(delayedReclaim)
        {
            init_unlocked(lock);

            if (pthread_key_create( &A_key,
                                    (delayedReclaim ? NULL : FFAkeyDestructorHandler) )!=0)  {
                error("FFAllocator FATAL ERROR: pthread_key_create fails\n");
                abort();
            }
        }

        /**
         * \brief Destructor
         *
         * Delete the allocator and return pre-allocated memory to the OS
         */
        ~FFAllocator() {
            if (delayedReclaim) {
                if (A) {
                    for(unsigned i=0;i<A_size;++i)
                        if (A[i]) deleteAllocator(A[i]->f);
                    ::free(A);
                }
                pthread_key_delete(A_key);
            }

        }

        /**
         * \brief Returns an instance of the FFAllocator object
         *
         */
        static inline FFAllocator * instance() {
            static FFAllocator FFA;
            return &FFA;
        }

        /*
         * Allocator factory method.
         *
         * Each time this method is called it spawns a new memory allocator. The
         * calling thread is registered as an allocator thread. Other threads
         * willing to use this SlabCache have to register themselves to the shared
         * leak queue of the allocator.
         *
         * \param max_size PARAMETER NOT USED!
         * \param _nslabs  array specifying the allowed quantities of buffers. By
         * default it is initialised to 0s
         * \param prealloc flag that is used as a mask when creating a new slab.
         * Default is \p true.
         *
         * \returns a \p ffa_wrapper object, that is in turn an extension of
         * a \ff_allocator object.
         */
        inline ffa_wrapper * newAllocator( size_t max_size=0,
                                           int _nslabs[N_SLABBUFFER]=0,
                                           bool prealloc=true )
        {
            FFAxThreadData * ffaxtd = newAllocator(prealloc, _nslabs, max_size);
            if (!ffaxtd) return NULL;
            return ffaxtd->f;
        }

        /*
         * Delete the allocator \p f passed as argument and reclaim the used
         * memory.\n Also updates the counter of the available size in the slab
         *
         * \param f an object of type ffa_wrapper.
         * \param reclaimMemory flag to decide whether to reclaim memory or not.
         */
        inline void deleteAllocator(ffa_wrapper * f, bool reclaimMemory=true) {
            if (!f) return;

            spin_lock(lock);
            for (unsigned int i=0;i<A_size;++i) {
                if (A[i]->f == f) {
                    f->deregisterAllocator(reclaimMemory);
                    f->~ffa_wrapper();
                    A[i]->~FFAxThreadData();
                    ::free(A[i]);

                    for (unsigned int j=i+1;j<A_size;++i,++j)
                        A[i]=A[j];
                    --A_size;

                    break;
                }
            }
            spin_unlock(lock);
            ALLSTATS(all_stats::instance()->deleteallocator.fetch_add(1));
        }

        /*
         * Delete allocator owned by calling thread and reclaim memory
         *
         */
        inline void deleteAllocator() {
            FFAxThreadData * ffaxtd = (FFAxThreadData*)pthread_getspecific(A_key);
            if (!ffaxtd) return;
            deleteAllocator(ffaxtd->f,true);
        }

        /**
         * \brief malloc
         *
         * Request to allocate \p size bytes. If the size is too
         * large, use OS malloc to get a new chunk of memory. Otherwise use the
         * ff_allocator on the calling ff_node
         *
         * \param size the size of memory requested
         * \return pointer to allocated memory chunk
         */
        inline void * malloc(size_t size) {
            /* use standard allocator if the size is too big */
            if (size>MAX_SLABBUFFER_SIZE) {
                ALLSTATS(all_stats::instance()->sysmalloc.fetch_add(1));
                void * ptr = ::malloc(size+sizeof(Buf_ctl));
                if (!ptr) return 0;
                ((Buf_ctl *)ptr)->ptr = 0;
                return (char *)ptr + sizeof(Buf_ctl);
            }

            FFAxThreadData * ffaxtd = (FFAxThreadData*) pthread_getspecific(A_key);
            if (!ffaxtd) {
                // if no thread-data is associated to the key
                // initialise and register a new allocator
                // REW -- why prealloc is FALSE??
                //
                ffaxtd = newAllocator(false,0,0);

                if (!ffaxtd) {
                    error("FFAllocator:malloc: newAllocator fails!\n");
                    return NULL;
                }

                // REW -- ?
                spin_lock(lock);
                if (pthread_setspecific(A_key, ffaxtd)!=0) {
                    deleteAllocator(ffaxtd->f);
                    spin_unlock(lock);
                    return NULL;
                }
                spin_unlock(lock);
            }

            return ffaxtd->f->malloc(size);
        }

        /**
         * \brief ff posix_memalign
         *
         * \param[out] *memptr pointer to a chunk of memory where the aligned memory will
         * be returned
         * \param alignment allocation's base address is an exact multiple of \p alignment, which
         * must be a power of 2 at least as large as sizeof(void *)
         * \param size the size of memory requested.
         * \return 0 if successful; otherwise it returns an error value.
         *
         */
        inline int posix_memalign(void **memptr, size_t alignment, size_t size) {
            if (!isPowerOfTwoMultiple(alignment, sizeof(void *))) return -1;

            size_t realsize = size+alignment;

            if (realsize > MAX_SLABBUFFER_SIZE) {
                ALLSTATS(all_stats::instance()->sysmalloc.fetch_add(1));
                void * ptr = ::malloc(realsize+sizeof(Buf_ctl));
                if (!ptr) return -1;
                ((Buf_ctl *)ptr)->ptr = 0;
                void * ptraligned = (void *)(((long)((char*)ptr+sizeof(Buf_ctl)) + alignment) & ~(alignment-1));
                ((Buf_ctl *)((char *)ptraligned - sizeof(Buf_ctl)))->ptr = 0;
                *memptr = ptraligned;
                return 0;
            }
            FFAxThreadData * ffaxtd = (FFAxThreadData*)pthread_getspecific(A_key);
            if (!ffaxtd) {
                // initialise and register a new allocator
                // REW -- why prealloc FALSE??
                ffaxtd = newAllocator(false,0,0);

                if (!ffaxtd) {
                    error("FFAllocator:posix_memalign: newAllocator fails!\n");
                    return -1;
                }

                spin_lock(lock);
                if (pthread_setspecific(A_key, ffaxtd)!=0) {
                    deleteAllocator(ffaxtd->f);
                    spin_unlock(lock);
                    return -1;
                }
                spin_unlock(lock);
            }

            return ffaxtd->f->posix_memalign(memptr,alignment,size);
        }

        /**
         * \brief free
         *
         * free the requested memory chunk
         *
         * \param ptr a pointer to the buffer.
         *
         */
        inline void   free(void * ptr) {
            if (!ptr) return;

            Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
            if (!buf->ptr) {
                ALLSTATS(all_stats::instance()->sysfree.fetch_add(1));
                ::free(buf);
                return;
            }
            Buf_ctl  * buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
            while(buf->ptr == buf2->ptr) {
                buf = buf2;
                buf2 = (Buf_ctl *)((char *)buf - sizeof(Buf_ctl));
            }
            DBG(if (!getsegctl(buf)->allocator) abort());
            (getsegctl(buf)->allocator)->free(buf);
        }

        /**
         * \brief realloc.
         *
         * It changes the size of the memory block pointed to by \p ptr to \p newsize
         * bytes.
         *
         * \param ptr pointer to the buffer
         * \param newsize the new size.
         *
         */
        inline void * realloc(void * ptr, size_t newsize) {
            if (ptr) {
                Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
                if (!buf->ptr) { /* use standard allocator */
                    ALLSTATS(all_stats::instance()->sysrealloc.fetch_add(1));
                    void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                    if (!ptr) return 0;
                    ((Buf_ctl *)newptr)->ptr = 0;
                    return (char *)newptr + sizeof(Buf_ctl);
                }

                ffa_wrapper * f = (ffa_wrapper*) pthread_getspecific(A_key);
                if (!f) return NULL;
                return f->realloc(ptr,newsize);
            }
            return this->malloc(newsize);
        }

        inline void * growsup(void * ptr, size_t newsize) {
            if (ptr) {
                Buf_ctl  * buf = (Buf_ctl *)((char *)ptr - sizeof(Buf_ctl));
                if (!buf->ptr) { /* use standard allocator */
                    ALLSTATS(all_stats::instance()->sysrealloc.fetch_add(1));
                    void * newptr= ::realloc(buf, newsize+sizeof(Buf_ctl));
                    if (!ptr) return 0;
                    ((Buf_ctl *)newptr)->ptr = 0;
                    return (char *)newptr + sizeof(Buf_ctl);
                }

                ffa_wrapper * f = (ffa_wrapper*)pthread_getspecific(A_key);
                if (!f) return NULL;
                return f->growsup(ptr,newsize);
            }
            return this->malloc(newsize);
        }

        ALLSTATS(void printstats(std::ostream & out) {
                all_stats::instance()->print(out);
            })

        private:
        FFAxThreadData **A;             // ffa_wrapper : ff_allocator
        size_t          A_capacity;     //
        size_t          A_size;         //

        lock_t          lock;
        pthread_key_t   A_key;
        const int       delayedReclaim;
    };

    /* REW - Document these? */
    static void FFAkeyDestructorHandler(void * kv) {
        FFAxThreadData * ffaxtd = (FFAxThreadData*)kv;
        ffaxtd->f->nomoremalloc();
        ffaxtd->f->deregisterAllocator();
    }
    static inline void killMyself(ffa_wrapper * ptr) {
        FFAllocator::instance()->deleteAllocator(ptr);
    }


    // Static functions below are not documented, but their usage is quite obvious
    static inline void * ff_malloc(size_t size) {
        return FFAllocator::instance()->malloc(size);
    }
    static inline void   ff_free(void * ptr) {
        FFAllocator::instance()->free(ptr);
    }
    static inline void * ff_realloc(void * ptr, size_t newsize) {
        return FFAllocator::instance()->realloc(ptr,newsize);
    }
    static inline int    ff_posix_memalign(void **memptr, size_t alignment, size_t size) {
        return FFAllocator::instance()->posix_memalign(memptr,alignment,size);
    }


} // namespace ff

#endif /* FF_ALLOCATOR_HPP */
