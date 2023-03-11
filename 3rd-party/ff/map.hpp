/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file map.hpp
 *  \in group high_level_patterns
 *
 *  \brief map pattern
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
 *  This program is distributed in the hope that it will be useful, but WITHOUT_t
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

 
#ifndef FF_MAP_HPP
#define FF_MAP_HPP

//VS12
//#ifndef WIN32

// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_AUTO) && defined(HAS_CXX11_LAMBDA))
#include <ff/parallel_for.hpp>
#else
#error "C++ >= 201103L is required to use ff_Map"
#endif

namespace ff {


/*!
 * \class ff_Map
 *  \ingroup high_level_patterns
 *
 * \brief Map pattern
 *
 * Apply to all
 *
 * \todo Map to be documented and exemplified
 */
template<typename IN_t , typename OUT_t=IN_t , typename reduceT=int>
class ff_Map: public ff_node_t<IN_t, OUT_t> {
    using _node = ff_node_t<IN_t,OUT_t>;
protected:
    ParallelForReduce<reduceT> pfr;
protected:
    int prepare() {
        if (!_node::prepared) {
            // warmup phase
            pfr.resetskipwarmup();
            auto r=-1;
            if (pfr.run_then_freeze() != -1)         
                r = pfr.wait_freezing();            
            if (r<0) {
                error("ff_Map: preparing ParallelForReduce\n");
                return -1;
            }
            
            if (spinWait) { 
                if (pfr.enableSpinning() == -1) {
                    error("ParallelForReduce: enabling spinwait\n");
                    return -1;
                }
            }
            _node::prepared = true;
        }
        return 0;
    }
        
    int freeze_and_run(bool=false) {
        if (!_node::prepared) if (prepare()<0) return -1;
        return ff_node::freeze_and_run(true);
    }
    
public:
    
    typedef IN_t  in_type;
    typedef OUT_t out_type;

    ff_Map(size_t maxp=-1, bool spinWait=false, bool spinBarrier=false):
        pfr(maxp,false,true,spinBarrier),// skip loop warmup and disable spinwait
        spinWait(spinWait) {
        pfr.disableScheduler(true);
    }
    virtual ~ff_Map() {}

    /* --------------------------------------- */
    template <typename Function>
    inline void parallel_for(long first, long last, const Function& f, 
                             const long nw=FF_AUTO) {
        pfr.parallel_for(first,last,f,nw);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, const Function& f, 
                             const long nw=FF_AUTO) {
        pfr.parallel_for(first,last,step,f,nw);
    }
    template <typename Function>
    inline void parallel_for(long first, long last, long step, long grain, 
                             const Function& f, const long nw=FF_AUTO) {
        pfr.parallel_for(first,last,step,grain,f,nw);
    }    
    template <typename Function>
    inline void parallel_for_thid(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {
        pfr.parallel_for_thid(first,last,step,grain,f,nw);
    }    
    template <typename Function>
    inline void parallel_for_idx(long first, long last, long step, long grain, 
                                  const Function& f, const long nw=FF_AUTO) {
        pfr.parallel_for_idx(first,last,step,grain,f,nw);        
    }    
    template <typename Function>
    inline void parallel_for_static(long first, long last, long step, long grain, 
                                    const Function& f, const long nw=FF_AUTO) {
        pfr.parallel_for_static(first,last,step,grain,f,nw);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(reduceT& var, const reduceT& identity, 
                                long first, long last, 
                                const Function& partialreduce_body, const FReduction& finalreduce_body,
                                const long nw=FF_AUTO) {
        pfr.parallel_reduce(var,identity,first,last,partialreduce_body,finalreduce_body,nw);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(reduceT& var, const reduceT& identity, 
                                long first, long last, long step, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=FF_AUTO) {
        pfr.parallel_reduce(var,identity,first,last,step,body,finalreduce,nw);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce(reduceT& var, const reduceT& identity, 
                                long first, long last, long step, long grain, 
                                const Function& body, const FReduction& finalreduce,
                                const long nw=FF_AUTO) {
        pfr.parallel_reduce(var,identity,first,last,step,grain,body,finalreduce,nw);
    }

    template <typename Function, typename FReduction>
    inline void parallel_reduce_thid(reduceT& var, const reduceT& identity,
                                     long first, long last, long step, long grain,
                                     const Function& body, const FReduction& finalreduce,
                                     const long nw=FF_AUTO) {
        pfr.parallel_reduce_thid(var,identity,first,last,step,grain,body,finalreduce,nw);
    }
    template <typename Function, typename FReduction>
    inline void parallel_reduce_static(reduceT& var, const reduceT& identity,
                                       long first, long last, long step, long grain, 
                                       const Function& body, const FReduction& finalreduce,
                                       const long nw=FF_AUTO) {
        pfr.parallel_reduce_static(var,identity,first,last,step,grain,body,finalreduce,nw);
    }
    /* --------------------------------------- */

    
    virtual int run(bool=false) {
        if (!_node::prepared) if (prepare()<0) return -1;
        return ff_node::run(true);
    }

    virtual int run_then_freeze() {
        return freeze_and_run();
    }

    virtual int wait() { return ff_node::wait();}
    virtual int wait_freezing() { return ff_node::wait_freezing(); }


    int nodeInit() { if (!_node::prepared) return prepare(); return 0;  }
    void nodeEnd() {}

protected:
    bool spinWait;
};
    
} // namespace ff

//#endif //VS12
#endif /* FF_MAP_HPP */

