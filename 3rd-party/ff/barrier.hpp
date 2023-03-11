/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file barrier.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow blocking and non-blocking barrier implementations
 *
 */

#ifndef FF_BARRIER_HPP
#define FF_BARRIER_HPP

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

#include <stdlib.h>
#include <ff/platforms/platform.h>
#include <ff/utils.hpp>
#include <ff/config.hpp>

// 
// Inside FastFlow barriers are used only for:
//   - to start all nodes synchronously (just for convinience could be avoided)
//   - inside the ParallelFor* to implement the barrier at the end of each 
//     loop iteration.
//

namespace ff {

    /**
     *  \class ffBarrier
     *  \ingroup building_blocks
     *
     *  \brief Just a barrier interface
     *  
     */
struct ffBarrier {
    virtual ~ffBarrier() {}
    virtual inline int  barrierSetup(size_t) { return 0;}
    virtual inline void doBarrier(size_t) { };
};

    /**
     *  \class Barrier
     *  \ingroup building_blocks
     *
     *  \brief Blocking barrier - Used only to start all nodes synchronously
     *  
     */

#if (defined(__APPLE__) || defined(_MSC_VER))
    /**
     *   No pthread_barrier available on these platforms.
     *   The implementation uses only mutex and cond variable.
     */ 
class Barrier: public ffBarrier {
public:   
    Barrier(const size_t=MAX_NUM_THREADS):_barrier(0) {
        if (pthread_mutex_init(&bLock,NULL)!=0) {
            error("FATAL ERROR: Barrier: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&bCond,NULL)!=0) {
            error("FATAL ERROR: Barrier: pthread_cond_init fails!\n");
            abort();
        }
    }

    /**
     *
     *  Note that the results are undefined if barrierSetup() is called
     *  while any threads is blocked on the barrier.
     *
     */      
    inline int barrierSetup(size_t init) {
        assert(init>0);
        if (_barrier == init) return 0;
        _barrier = init; 
        counter = 0;
        return 0;
    }

    inline void doBarrier(size_t id) {
        pthread_mutex_lock(&bLock);
        if (++counter == _barrier) {
            pthread_cond_broadcast(&bCond);
            counter = 0;
        }
        else pthread_cond_wait(&bCond, &bLock);
        pthread_mutex_unlock(&bLock);
    }
private:
    // it is the number of threads in the barrier. 
    size_t _barrier, counter;
    pthread_mutex_t bLock;  // Mutex variable
    pthread_cond_t  bCond;  // Condition variable
};

#else
    /**
     *   pthread_barrier based implementation
     *
     */ 
class Barrier: public ffBarrier {
public:
    Barrier(const size_t=MAX_NUM_THREADS):_barrier(0) { }
    ~Barrier() { if (_barrier>0) pthread_barrier_destroy(&bar); }

    inline int barrierSetup(size_t init) {
        assert(init>0);
        if (_barrier == init) return 0;
        if (_barrier==0) {
            if (pthread_barrier_init(&bar,NULL,init) != 0) {
                error("ERROR: pthread_barrier_init failed\n");
                return -1;
            }
            _barrier = init;
            return 0;
        }
        if (pthread_barrier_destroy(&bar) != 0) {
            error("ERROR: pthread_barrier_destroy failed\n");
            return -1;
        }
        if (pthread_barrier_init(&bar,NULL,init) == 0) {
            _barrier = init;
            return 0;
        }
        error("ERROR: pthread_barrier_init failed\n");
        return -1;
    }

    inline void doBarrier(size_t) {  
        pthread_barrier_wait(&bar); 
    }

private:
    // it is the number of threads in the barrier. 
    size_t _barrier;
    pthread_barrier_t bar;
};

#endif 

/**
 *  \class spinBarrier
 *  \ingroup building_blocks
 *
 *  \brief Non-blocking barrier 
 *
 */
class spinBarrier: public ffBarrier {
public:
   
    spinBarrier(const size_t _maxNThreads=MAX_NUM_THREADS):maxNThreads(_maxNThreads), _barrier(0) {
        barArray=new bool[maxNThreads];
        assert(barArray!=NULL);
    }

    ~spinBarrier() {
        if (barArray != NULL) delete [] barArray;
        barArray=NULL;
    }
    
    inline int barrierSetup(size_t init) {
        assert(init>0);
        if (init == _barrier) return -1;
        for(size_t i=0; i<maxNThreads; ++i) barArray[i]=false;
        B[0]=0; B[1]=0;
        _barrier = init; 
        return 0;
    }

    inline void doBarrier(size_t tid) {
        assert(tid<maxNThreads);
        const int whichBar = (barArray[tid] ^= true); // computes % 2
        long c = ++B[whichBar];
        if ((size_t)c == _barrier) {
            B[whichBar]=0;
            return;
        }
        // spin-wait
        while(c) {
            c = B[whichBar];
            PAUSE();  // TODO: define a spin policy !
        }
    }    
private:
    const size_t maxNThreads;
    size_t _barrier;
    bool* barArray;
    std::atomic<long> B[2];
};


template<bool spin>
struct barHelper {
    Barrier bar;
    inline int barrierSetup(size_t init) { return bar.barrierSetup(init); }
    inline void doBarrier(size_t tid)    { return bar.doBarrier(tid); }
};
template<>
struct barHelper<true> {
    spinBarrier bar;
    inline int barrierSetup(size_t init) { return bar.barrierSetup(init); }
    inline void doBarrier(size_t tid)    { return bar.doBarrier(tid); }
};

/**
 *  \class barrierSelector
 *  \ingroup building_blocks
 *
 *  \brief It allows to select (at compile time) between blocking (false) and non-blocking (true) barriers.
 *
 */
template<bool whichone>
struct barrierSelector: public barHelper<whichone> {};

} // namespace ff

#endif /* FF_BARRIER_HPP */
