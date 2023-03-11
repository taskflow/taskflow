/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file parallel_for_internals.hpp
 *  \ingroup aux_classes
 *
 *  \brief Internal classes and functions for parallel_for/parallel_reduce skeletons.
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
 *  - Author: 
 *     Massimo Torquati <torquati@di.unipi.it>
 *
 *    History: 
 *      - started in May 2013
 *      - January 2014: code optimized
 *      - February 2014: 
 *          - avoided to start the Scheduler thread if it is not needed
 *            A new (non lock-free) decentralized scheduler has been implemented 
 *            for the case when adding an extra thread is not useful.
 *          - introduced the parallel_for functions
 *          - added the ParallelFor and ParallelForReduce classes
 *      - June 2014:
 *          - parallel_for_static
 *
 */

#ifndef FF_PARFOR_INTERNALS_HPP
#define FF_PARFOR_INTERNALS_HPP

// #ifndef __INTEL_COMPILER
// // see http://www.stroustrup.com/C++11FAQ.html#11
// #if __cplusplus <= 199711L
// #error "parallel_for requires C++11 features"
// #endif
// #endif

#include <atomic>
#include <algorithm>
#include <deque>
#include <vector>
#include <cmath>
#include <functional>
#include <ff/lb.hpp>
#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/spin-lock.hpp>

enum {FF_AUTO=-1};

#ifdef FF_PARFOR_PASSIVE_NOSTEALING
static int dummyTask;
static bool globalSchedRunning;
#endif

#if defined(__ICC)
#define PRAGMA_IVDEP _Pragma("ivdep")
#else
#define PRAGMA_IVDEP
#endif

namespace ff {

    /* -------------------- Parallel For/Reduce Macros -------------------- */
    /* Usage example:
     *                              // loop parallelization using 3 workers
     *                              // and a minimum task grain of 2
     *                              wthread = 3;
     *                              grain = 2;
     *  for(int i=0;i<N;++i)        FF_PARFOR_BEGIN(for,i,0,N,1,grain,wthread) {
     *    A[i]=f(i)          ---->    A[i]=f(i);
     *                              } FF_PARFOR_END(for);
     * 
     *   parallel for + reduction:
     *     
     *  s=4;                         
     *  for(int i=0;i<N;++i)        FF_PARFORREDUCE_BEGIN(for,s,0,i,0,N,1,grain,wthread) {
     *    s*=f(i)            ---->    s*=f(i);
     *                              } FF_PARFORREDUCE_END(for,s,*);
     *
     *                          
     *                              FF_PARFOR_INIT(pf,maxwthread);
     *                              ....
     *  while(k<nTime) {            while(k<nTime) {
     *    for(int i=0;i<N;++i)        FF_PARFORREDUCE_START(pf,s,0,i,0,N,1,grain,wthread) {
     *      s*=f(i,k);       ---->       s*=f(i,k);
     *  }                             } FF_PARFORREDUCE_STOP(pf,s,*);
     *                             }
     *                             ....
     *
     *                             FF_PARFOR_DONE(pf);
     *
     * 
     *  NOTE: inside the body of the PARFOR/PARFORREDUCE, it is possible to use the 
     *        '_ff_thread_id' const integer variable to identify the thread id 
     *        running the sequential portion of the loop.
     */

    /**
     *  name : of the parallel for
     *  idx  : iteration index
     *  begin: for starting point
     *  end  : for ending point
     *  step : for step
     *  chunk: chunk size
     *  nw   : n. of worker threads
     */
#define FF_PARFOR_BEGIN(name, idx, begin, end, step, chunk, nw)                   \
    ff_forall_farm<forallreduce_W<int> > name(nw,false,true);                     \
    name.setloop(begin,end,step,chunk,nw);                                        \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,     \
                         const int _ff_thread_id, const int) {                    \
        FF_IGNORE_UNUSED(_ff_thread_id);                                          \
        PRAGMA_IVDEP;                                                             \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 

    /* This is equivalent to the above one except that the user has to define
     * the for loop in the range (ff_start_idx,ff_stop_idx(
     * This can be useful if you have to perform some actions before starting
     * the local loop and/or some actions after the local loop finishes.
     * The onoff parameter allow to disable/enable the scheduler thread 
     * (by default the scheduler is active.
     */
#define FF_PARFOR_BEGIN_IDX(name, idx, begin, end, step, chunk, nw, onoff)        \
    ff_forall_farm<forallreduce_W<int> > name(nw,false,true);                     \
    name.setloop(begin,end,step,chunk, nw);                                       \
    name.disableScheduler(onoff);                                                 \
    auto F_##name = [&] (const long ff_start_idx, const long ff_stop_idx,         \
                         const int _ff_thread_id, const int) {                    \
    /* here you have to define the for loop using ff_start/stop_idx  */


#define FF_PARFOR_END(name)                                                       \
    };                                                                            \
    {                                                                             \
      if (name.getnw()>1) {                                                       \
        name.setF(F_##name);                                                      \
        if (name.run_and_wait_end()<0) {                                          \
			error("running parallel for\n");                                      \
        }                                                                         \
      } else F_##name(name.startIdx(),name.stopIdx(),0,0);                        \
    }

    
    /* ---------------------------------------------- */

    /**
     *  name    : of the parallel for
     *  var     : variable on which the reduce operator is applied
     *  identity: the value such that var == var op identity 
     *  idx     : iteration index
     *  begin   : for starting point
     *  end     : for ending point
     *  step    : for step
     *  chunk   : chunk size
     *  nw      : n. of worker threads
     * 
     *  op      : reduce operation (+ * ....) 
     */
#define FF_PARFORREDUCE_BEGIN(name, var,identity, idx,begin,end,step, chunk, nw)  \
    ff_forall_farm<forallreduce_W<decltype(var)> > name(nw,false,true);           \
    name.setloop(begin,end,step,chunk,nw);                                        \
    auto idtt_##name =identity;                                                   \
    auto F_##name =[&](const long start,const long stop,const int _ff_thread_id,  \
                       decltype(var) &var) {                                      \
        FF_IGNORE_UNUSED(_ff_thread_id);                                          \
        PRAGMA_IVDEP;                                                             \
        for(long idx=start;idx<stop;idx+=step)

#define FF_PARFORREDUCE_BEGIN_IDX(name, var,identity, idx,begin,end,step, chunk, nw, onoff)    \
    ff_forall_farm<forallreduce_W<decltype(var)> > name(nw,false,true);                        \
    name.setloop(begin,end,step,chunk,nw);                                                     \
    name.disableScheduler(onoff);                                                              \
    auto idtt_##name =identity;                                                                \
    auto F_##name =[&](const long ff_start_idx,const long ff_stop_idx,const int _ff_thread_id, \
                       decltype(var) &var) {                                                   \
        FF_IGNORE_UNUSED(_ff_thread_id);


    
#define FF_PARFORREDUCE_END(name, var, op)                                        \
        };                                                                        \
        if (name.getnw()>1) {                                                     \
          auto ovar_##name = var;                                                 \
          name.setF(F_##name,idtt_##name);                                        \
          if (name.run_and_wait_end()<0) {                                        \
			error("running forall_##name\n");                                     \
          }                                                                       \
          var = ovar_##name;                                                      \
          for(size_t i=0;i<name.getnw();++i)  {                                   \
              var op##= name.getres(i);                                           \
          }                                                                       \
        } else {                                                                  \
          var = ovar_##name;                                                      \
          F_##name(name.startIdx(),name.stopIdx(),0,var);                         \
        }


#define FF_PARFORREDUCE_F_END(name, var, F)                                       \
        };                                                                        \
        if (name.getnw()>1) {                                                     \
          auto ovar_##name = var;                                                 \
          name.setF(F_##name,idtt_##name);                                        \
          if (name.run_and_wait_end()<0)                                          \
            error("running ff_forall_farm (reduce F end)\n");	                  \
          var = ovar_##name;                                                      \
          for(size_t i=0;i<name.getnw();++i)  {                                   \
             F(var,name.getres(i));                                               \
          }                                                                       \
        } else {                                                                  \
            F_##name(name.startIdx(),name.stopIdx(),0,var);                       \
        }


    /* ---------------------------------------------- */

    /* FF_PARFOR_START and FF_PARFOR_STOP have the same meaning of 
     * FF_PARFOR_BEGIN and FF_PARFOR_END but they have to be used in 
     * conjunction with  FF_PARFOR_INIT FF_PARFOR_END.
     *
     * The same is for FF_PARFORREDUCE_START/STOP.
     */
#define FF_PARFOR_INIT(name, nw)                                                         \
    ff_forall_farm<forallreduce_W<int> > *name =                                         \
        new ff_forall_farm<forallreduce_W<int> >(nw)

#define FF_PARFOR_DECL(name)         ff_forall_farm<forallreduce_W<int> > * name
#define FF_PARFOR_ASSIGN(name,nw)    name=new ff_forall_farm<forallreduce_W<int> >(nw)
#define FF_PARFOR_DONE(name)         name->stop(); name->wait(); delete name;

#define FF_PARFORREDUCE_INIT(name, type, nw)                                             \
    ff_forall_farm<forallreduce_W<type> > *name =                                        \
        new ff_forall_farm<forallreduce_W<type> >(nw)

#define FF_PARFORREDUCE_DECL(name,type)      ff_forall_farm<forallreduce_W<type> > * name
#define FF_PARFORREDUCE_ASSIGN(name,type,nw) name=                                       \
        new ff_forall_farm<forallreduce_W<type> >(nw)
#define FF_PARFORREDUCE_DONE(name)           name->stop();name->wait();delete name

#define FF_PARFOR_START(name, idx, begin, end, step, chunk, nw)                          \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,            \
                         const int _ff_thread_id, const int) {                           \
        FF_IGNORE_UNUSED(_ff_thread_id);                                                 \
        PRAGMA_IVDEP;                                                                    \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 

#define FF_PARFOR2_START(name, idx, begin, end, step, chunk, nw)                         \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,            \
                         const int _ff_thread_id, const int) {                           \
        FF_IGNORE_UNUSED(_ff_thread_id);
    /* here you have to define the for loop using ff_start/stop_##idx  */

/* this is equivalent to FF_PARFOR2_START but the start/stop indexes have a fixed name */
#define FF_PARFOR_START_IDX(name, idx, begin, end, step, chunk, nw)                      \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_idx, const long ff_stop_idx,                \
                         const int _ff_thread_id, const int) {                           \
        FF_IGNORE_UNUSED(_ff_thread_id);
    /* here you have to define the for loop using ff_start/stop_idx  */


// just another variat that may be used together with FF_PARFORREDUCE_INIT
#define FF_PARFOR_T_START(name, type, idx, begin, end, step, chunk, nw)                  \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,            \
                         const int _ff_thread_id, const type&) {                         \
        FF_IGNORE_UNUSED(_ff_thread_id);                                                 \
        PRAGMA_IVDEP;                                                                    \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 


// just another variat that may be used together with FF_PARFORREDUCE_INIT
#define FF_PARFOR_T_START_STATIC(name, type, idx, begin, end, step, chunk, nw)           \
    assert(chunk<=0);                                                                    \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_##idx, const long ff_stop_##idx,            \
                         const int _ff_thread_id, const type&) {                         \
        const long _ff_jump0=(name->getnw())*(-chunk*step);                              \
        const long _ff_jump1=(-chunk*step);                                              \
        FF_IGNORE_UNUSED(_ff_thread_id);                                                 \
        PRAGMA_IVDEP;                                                                    \
        for(long _ff_##idx=ff_start_##idx;_ff_##idx<ff_stop_##idx;_ff_##idx+=_ff_jump0)  \
           for(long idx=_ff_##idx,_ff_end_##idx=std::min(ff_stop_##idx,_ff_##idx +_ff_jump1); \
               idx<_ff_end_##idx;idx+=step)


#define FF_PARFOR_T_START_IDX(name, type, idx, begin, end,step,chunk, nw)                \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto F_##name = [&] (const long ff_start_idx, const long ff_stop_idx,                \
                         const int _ff_thread_id, const type&) {                         \
        FF_IGNORE_UNUSED(_ff_thread_id);
    /* here you have to use the fixed indexes ff_idx_start, ff_idx_stop */

#define FF_PARFOR_STOP(name)                                                             \
    };                                                                                   \
    if (name->getnw()>1) {                                                               \
      name->setF(F_##name);                                                              \
      if (name->run_then_freeze(name->getnw())<0)                                        \
		 error("running ff_forall_farm (name)\n");                                       \
      name->wait_freezing();                                                             \
    } else F_##name(name->startIdx(),name->stopIdx(),0,0);

#define FF_PARFOR_T_STOP(name, type)                                                     \
    };                                                                                   \
    if (name->getnw()>1) {                                                               \
        name->setF(F_##name, type());                                                    \
        if (name->run_then_freeze(name->getnw())<0)                                      \
		  error("running ff_forall_farm (name)\n");                                      \
        name->wait_freezing();                                                           \
    } else {                                                                             \
        F_##name(name->startIdx(),name->stopIdx(),0,type());                             \
    }
    
#define FF_PARFORREDUCE_START(name, var,identity, idx,begin,end,step, chunk, nw)         \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto idtt_##name =identity;                                                          \
    auto F_##name =[&](const long ff_start_##idx, const long ff_stop_##idx,              \
                       const int _ff_thread_id, decltype(var) &var) {                    \
        FF_IGNORE_UNUSED(_ff_thread_id);                                                 \
        PRAGMA_IVDEP                                                                     \
        for(long idx=ff_start_##idx;idx<ff_stop_##idx;idx+=step) 

#define FF_PARFORREDUCE_START_IDX(name, var,identity, idx,begin,end,step, chunk, nw)     \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto idtt_##name =identity;                                                          \
    auto F_##name =[&](const long ff_start_idx, const long ff_stop_idx,                  \
                       const int _ff_thread_id, decltype(var) &var) {                    \
        FF_IGNORE_UNUSED(_ff_thread_id);



#define FF_PARFORREDUCE_START_STATIC(name, var,identity, idx,begin,end,step, chunk, nw)  \
    assert(chunk<=0);                                                                    \
    name->setloop(begin,end,step,chunk,nw);                                              \
    auto idtt_##name =identity;                                                          \
    auto F_##name =[&](const long ff_start_##idx, const long ff_stop_##idx,              \
                       const int _ff_thread_id, decltype(var) &var) {                    \
        const long _ff_jump0=(name->getnw())*(-chunk*step);                              \
        const long _ff_jump1=(-chunk*step);                                              \
        PRAGMA_IVDEP;                                                                    \
        for(long _ff_##idx=ff_start_##idx;_ff_##idx<ff_stop_##idx;_ff_##idx+=_ff_jump0)  \
           for(long idx=_ff_##idx,_ff_end_##idx=std::min(ff_stop_##idx,_ff_##idx +_ff_jump1); \
               idx<_ff_end_##idx;idx+=step)


#define FF_PARFORREDUCE_STOP(name, var, op)                                              \
        };                                                                               \
        if (name->getnw()>1) {                                                           \
          auto ovar_##name = var;                                                        \
          name->setF(F_##name,idtt_##name);                                              \
          if (name->run_then_freeze(name->getnw())<0)                                    \
			error("running ff_forall_farm (name)\n");                                    \
          name->wait_freezing();                                                         \
          var = ovar_##name;                                                             \
          for(size_t i=0;i<name->getnw();++i)  {                                         \
              var op##= name->getres(i);                                                 \
          }                                                                              \
        } else {                                                                         \
          F_##name(name->startIdx(),name->stopIdx(),0,var);                              \
        }


#define FF_PARFORREDUCE_F_STOP(name, var, F)                                             \
        };                                                                               \
        if (name->getnw()>1) {                                                           \
          auto ovar_##name = var;                                                        \
          name->setF(F_##name,idtt_##name);                                              \
          if (name->run_then_freeze(name->getnw())<0)                                    \
			 error("running ff_forall_farm (name)\n");                                   \
          name->wait_freezing();                                                         \
          var = ovar_##name;                                                             \
          for(size_t i=0;i<name->getnw();++i)  {                                         \
             F(var,name->getres(i));                                                     \
          }                                                                              \
        } else {                                                                         \
            F_##name(name->startIdx(),name->stopIdx(),0,var);                            \
        }



//
// see NOTE in setloop to understand the meaning of 'default static' 
// 'static with grain size' and 'dynamic with grain size'
//
#define PARFOR_STATIC(X)   (X>0?-X:X)
#define PARFOR_DYNAMIC(X)  (X<0?-X:X)

    /* ------------------------------------------------------------------- */



// parallel for task, it represents a range (start,end( of indexes
struct forall_task_t {
	forall_task_t() : end(0) {
		start.store(0); // MA: consistency of store to be checked
	}
    forall_task_t(const forall_task_t &t):end(t.end) {
		start.store(t.start.load(std::memory_order_relaxed)); // MA: consistency of store to be checked
	}
    forall_task_t & operator=(const forall_task_t &t) { 
        start=t.start.load(std::memory_order_relaxed), end=t.end; 
        return *this; 
    }
    void set(long s, long e)  { start=s,end=e; }

    std::atomic_long start;
    long             end;
};
struct dataPair {
    std::atomic_long ntask;
	ALIGN_TO_PRE(CACHE_LINE_SIZE)
	forall_task_t task;
	ALIGN_TO_POST(CACHE_LINE_SIZE)

    dataPair():task() {
		ntask.store(0); // MA: consistency of store to be checked
	};
    dataPair(const dataPair &d):task(d.task) {
		ntask.store(d.ntask.load(std::memory_order_relaxed)); // MA: consistency of store to be checked
	}
    dataPair& operator=(const dataPair &d) { ntask=d.ntask.load(std::memory_order_relaxed), task=d.task; return *this; }
};

// compare functiong
static inline bool data_cmp(const dataPair &a,const dataPair &b) {
    return a.ntask < b.ntask;
}
// delay function for worker threads
static inline void workerlosetime_in(const bool aggressive) {
    if (aggressive) PAUSE();
    else ff_relax(0);
}


// parallel for/reduce task scheduler
class forall_Scheduler: public ff_node {
protected:
    std::vector<bool>      eossent;
    std::vector<dataPair>  data;
    std::atomic_long       maxid;
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
    std::atomic_long       _nextIteration;
#endif
protected:
    // initialize the data vector
    virtual inline size_t init_data(ssize_t start, ssize_t stop) {
        static_scheduling = false;  // enable work stealing in the nextTaskConcurrent
        const long numtasks  = std::lrint(std::ceil((stop-start)/(double)_step));
        long totalnumtasks   = std::lrint(std::ceil(numtasks/(double)_chunk));
        long tt     = totalnumtasks;
        size_t ntxw = totalnumtasks / _nw;
        size_t r    = totalnumtasks % _nw;

        // try to keep the n. of tasks per worker as smaller as possible
        if (ntxw == 0 && r>=1) {  ntxw = 1, r = 0; }
        
        data.resize(_nw); eossent.resize(_nw);
        taskv.resize(8*_nw); // 8 is the maximum n. of jumps, see the heuristic below
        skip1=false,jump=0,maxid=-1;

        ssize_t end, t=0, e;
        for(size_t i=0;i<_nw && totalnumtasks>0;++i, totalnumtasks-=t) {
            t       = ntxw + ( (r>1 && (i<r)) ? 1 : 0 );
            e       = start + (t*_chunk - 1)*_step + 1;
            end     = (e<stop) ? e : stop;
            data[i].ntask=t;
            data[i].task.set(start,end);
            start   = (end-1)+_step;
        }

        if (totalnumtasks) {
            assert(totalnumtasks==1);
            // try to keep the n. of tasks per worker as smaller as possible
            if (ntxw > 1) data[_nw-1].ntask += totalnumtasks;
            else { --tt, _chunk*=2; }
            data[_nw-1].task.end = stop;
        } 
        // printf("init_data\n");
        // for(size_t i=0;i<_nw;++i) {
        //     printf("W=%ld %ld <%ld,%ld>\n", i, data[i].ntask.load(), data[i].task.start.load(), data[i].task.end);
        // }
        // printf("totaltasks=%ld\n", tt);
        return tt;
    }    
    // initialize the data vector
    virtual inline size_t init_data_static(long start, long stop) {
        assert(_chunk <= 0);
        static_scheduling = true;  // this forces static scheduling in the nextTaskConcurrent
        skip1=false,jump=0,maxid=-1;
        
        if (_chunk == 0) { 
            // default static scheduling, i.e. the iteration space is almost equally divided
            // in contiguous chunks among threads
            const long numtasks  = std::lrint(std::ceil((stop-start)/(double)_step));
            long totalnumtasks   = (long)_nw;
            size_t r             = numtasks % _nw;
            _chunk               = numtasks / long(_nw);
            
            data.resize(_nw); taskv.resize(_nw);eossent.resize(_nw);
            
            long end, e;
            for(size_t i=0; totalnumtasks>0; ++i,--totalnumtasks) {
                e       = start + (_chunk - 1)*_step + 1 + ((i<r) ? _step : 0 );
                end     = (e<stop) ? e : stop;
                data[i].ntask=1;
                data[i].task.set(start,end);
                start   = (end-1)+_step;
            }
            if (r) ++_chunk;
            return _nw;
        }
        // fill out the table with only the first task just to start the worker threads
        long chunk = -_chunk; 
        _chunk = stop; // needed because sendTask has to send the range (begin, stop(
        const long numtasks      = std::lrint(std::ceil((stop-start)/(double)_step));
        const long totalnumtasks = std::lrint(std::ceil(numtasks/(double)chunk));
        const size_t ntxw = (std::min)(_nw, (size_t)totalnumtasks);

        for(size_t i=0;i<ntxw;++i) {
            data[i].ntask = 1;
            data[i].task.set(start+long(i)*chunk,stop);                        
        }
        // printf("init_data_static\n");
        // for(size_t i=0;i<_nw;++i) {
        //     long start=data[i].task.start;
        //     long ntask=data[i].ntask;
        //     printf("W=%ld %ld <%ld,%ld>\n", i, ntask.load(), start.load(), data[i].task.end);
        // }
        // printf("total task=%ld\n", ntxw);

        return ntxw;
    }    
public:
    forall_Scheduler(ff_loadbalancer* lb, long start, long stop, long step, long chunk, size_t nw):
        lb(lb),_start(start),_stop(stop),_step(step),_chunk(chunk),totaltasks(0),_nw(nw),
        jump(0),skip1(false),workersspinwait(false),static_scheduling(false) {
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        _nextIteration = _start;
#endif
		maxid.store(-1); // MA: consistency of store to be checked
        if (_chunk<=0) totaltasks = init_data_static(start,stop);
        else           totaltasks = init_data(start,stop);
        assert(totaltasks>=1);
    }
    forall_Scheduler(ff_loadbalancer* lb, size_t nw):
        lb(lb),_start(0),_stop(0),_step(1),_chunk(1),totaltasks(0),_nw(nw),
        jump(0),skip1(false),workersspinwait(false),static_scheduling(false) {
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        _nextIteration = 0;
#endif
		maxid.store(-1); // MA: consistency of store to be checked
        totaltasks = init_data(0,0);
        assert(totaltasks==0);
    }

#ifdef FF_PARFOR_PASSIVE_NOSTEALING
    inline bool canUseNoStealing(){
        return !globalSchedRunning && !static_scheduling && _step == 1 && _chunk == 1;
    }
#endif
    inline bool sendTask(const bool skipmore=false) {
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        if(canUseNoStealing()){
            // Just start the workers and die.
            for(size_t wid=0;wid<_nw;++wid) {
                lb->ff_send_out_to((void*) &dummyTask, (int) wid);
           }
        return true;
        }
#endif
        size_t remaining    = totaltasks;
        const long endchunk = (_chunk-1)*_step + 1;

    more:
        for(size_t wid=0;wid<_nw;++wid) {
            if (data[wid].ntask >0) {
                long start = data[wid].task.start;
                long end   = (std::min)(start+endchunk, data[wid].task.end);
                taskv[wid+jump].set(start, end);
                lb->ff_send_out_to(&taskv[wid+jump], (int) wid);
                --remaining, --data[wid].ntask;
                (data[wid].task).start = (end-1)+_step;  
                eossent[wid]=false;
            } else  skip1=true; //skip2=skip3=true;
        }
        // January 2014 (massimo): this heuristic maight not be the best option in presence 
        // of very high load imbalance between iterations. 
        // Update: removed skip2 and skip3 so that it is less aggressive !
        
        jump+=long(_nw);
        assert((jump / _nw) <= 8);
        // heuristic: try to assign more task at the very beginning
        if (!skipmore && !skip1 && totaltasks>=4*_nw)   { skip1=true; goto more;}        
        //if (!skip2 && totaltasks>=64*_nw)  { skip1=false; skip2=true; goto moretask;}
        //if (!skip3 && totaltasks>=1024*_nw){ skip1=false; skip2=false; skip3=true; goto moretask;}
        return (remaining>0);
    }

    inline void sendWakeUp() {
        for(size_t id=0;id<_nw;++id) {
            taskv[id].set(0,0);
            lb->ff_send_out_to(&taskv[id], int(id));
        }
    }
    inline bool nextTaskConcurrentNoStealing(forall_task_t *task, const int wid) {  
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        long r = _nextIteration.fetch_add(_step);
        if(r >= _stop){return false;}
        task->set(r, r + _step);
        return true;
#else
        FF_IGNORE_UNUSED(task);
        FF_IGNORE_UNUSED(wid);
        error("To use nextTaskConcurrentNoStealing you need to define macro FF_PARFOR_PASSIVE_NOSTEALING\n");
        return false;
#endif
    }

    // this method is accessed concurrently by all worker threads
    inline bool nextTaskConcurrent(forall_task_t *task, const int wid) {
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        if(canUseNoStealing()){
            return nextTaskConcurrentNoStealing(task, wid);
        }
#endif
        const long endchunk = (_chunk-1)*_step + 1; // next end-point
        auto id  = wid;
    L1:
        if (data[id].ntask.load(std::memory_order_acquire)>0) {
            auto oldstart = data[id].task.start.load(std::memory_order_relaxed);
            auto end      = (std::min)(oldstart+endchunk, data[id].task.end);
            auto newstart = (end-1)+_step;
            
            if (!data[id].task.start.compare_exchange_weak(oldstart, newstart,
                                                           std::memory_order_release,
                                                           std::memory_order_relaxed)) {
                workerlosetime_in(_nw <= lb->getnworkers());
                goto L1; // restart the sequence from the beginning
            }
            
            // after fetch_sub ntask may be less than 0
            data[id].ntask.fetch_sub(1,std::memory_order_release);   
            if (oldstart<end) { // it might be possible that oldstart == end
                task->set(oldstart, end); 
                return true;
            }
        }

        // no available task for the current thread
        if (static_scheduling) return false;      // <------------------------------------

#if !defined(PARFOR_MULTIPLE_TASKS_STEALING)
        // the following scheduling policy for the tasks focuses mostly to load-balancing
        long _maxid = 0, ntask = 0;
        if (maxid.load(std::memory_order_acquire)<0)
            _maxid = (long) (std::max_element(data.begin(),data.end(),data_cmp) - data.begin());
        else _maxid = maxid;
        ntask  = data[_maxid].ntask.load(std::memory_order_relaxed);
        if (ntask>0) { 
            if (_maxid != maxid) maxid.store(_maxid, std::memory_order_release);
            id = _maxid; 
            goto L1; 
        }
        // no more tasks, exit

#else
        // the following scheduling policy for the tasks is a little bit more 
        // complex and costly. It tries to find a trade-off between 
        // task-to-thread localy and load-balancing by moving a bunch of tasks 
        // from one thread to another one
        long _maxid = 0, ntask = 0;
        if (maxid.load(std::memory_order_acquire)<0)
            _maxid = (std::max_element(data.begin(),data.end(),data_cmp) - data.begin());
        else _maxid = maxid;
    L2:
        ntask  = data[_maxid].ntask.load(std::memory_order_relaxed);
        if (ntask>0) { 
            if (_maxid != maxid) maxid.store(_maxid, std::memory_order_release);
            if (ntask<=3) { id = _maxid; goto L1; }
            
            // try to steal half of the tasks remaining to _maxid

            auto oldstart = data[_maxid].task.start.load(std::memory_order_relaxed);
            auto q = ((data[_maxid].task.end-oldstart)/_chunk) >> 1;
            if (q<=3) { id = _maxid; goto L1; }
            auto newstart = oldstart + (q*_chunk-1)*_step +1;
            if (!data[_maxid].task.start.compare_exchange_weak(oldstart, newstart,
                                                               std::memory_order_release,
                                                               std::memory_order_relaxed)) {
                workerlosetime_in(_nw <= lb->getnworkers());
                goto L2; // restart the sequence from the beginning
            }
            assert(newstart <= data[_maxid].task.end);
            
            data[_maxid].ntask.fetch_sub(q, std::memory_order_release);            
            data[wid].task.start.store(oldstart, std::memory_order_relaxed);
            data[wid].task.end = newstart;
            data[wid].ntask.store(q, std::memory_order_release);
            id = wid;
            goto L1;
        }
#endif
        return false; 
    }
    
    inline bool nextTask(forall_task_t *task, const int wid) {
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        if(canUseNoStealing()){
                return nextTaskConcurrentNoStealing(task, wid);
        }
#endif
        const long endchunk = (_chunk-1)*_step + 1;
        int id  = wid;
        if (data[id].ntask) {
        L1:
            long start = data[id].task.start;
            long end = (std::min)(start+endchunk, data[id].task.end);
            --data[id].ntask, (data[id].task).start = (end-1)+_step;
            task->set(start, end);
            return true;
        }
        // no available task for the current thread
#if !defined(PARFOR_MULTIPLE_TASKS_STEALING)
        // the following scheduling policy for the tasks focuses mostly to load-balancing
        if (maxid<0)  { //check if maxid has been set
        L2:
            maxid = (long) (std::max_element(data.begin(),data.end(),data_cmp) - data.begin());
            if (data[maxid].ntask > 0) {
                id=maxid;
                goto L1;
            }
            // no more tasks, exit
        } else {
            if (data[maxid].ntask > 0) { 
                id=maxid;
                goto L1;
            }
            goto L2;
        }        
#else
        auto flag=false;
        if (maxid<0)  {
        L2:
            maxid = (std::max_element(data.begin(),data.end(),data_cmp) - data.begin());
            flag=true;
        }
        id = maxid;
        if (data[id].ntask>0) {
            if (data[id].ntask<=3) goto L1;

            // steal half of the tasks
            auto q = data[id].ntask >> 1, r = data[id].ntask & 0x1; 
            data[id].ntask  = q;
            data[wid].ntask = q+r;
            data[wid].task.end   = data[id].task.end;
            data[id].task.end    = data[id].task.start + (q*_chunk-1)*_step +1;
            data[wid].task.start = data[id].task.end;
            id = wid;
            goto L1;
        } else if (!flag) goto L2;
#endif
        return false; 
    }

    inline void* svc(void* t) {
        if (t==NULL) {
            if (totaltasks==0) { lb->broadcast_task(GO_OUT); return GO_OUT;}
            sendTask();
            return GO_ON; 
        }
        auto wid =  lb->get_channel_id();
        assert(wid>=0);
        if (--totaltasks <=0) {
            if (!eossent[wid]) {
                lb->ff_send_out_to(workersspinwait?EOS_NOFREEZE:GO_OUT, int(wid));
                eossent[wid]=true;
            }
            return GO_OUT;
        }
        if (nextTask((forall_task_t*)t, (int) wid)) lb->ff_send_out_to(t, int(wid));            
        else  {
            if (!eossent[wid]) {
                lb->ff_send_out_to((workersspinwait?EOS_NOFREEZE:GO_OUT), int(wid));
                eossent[wid]=true;
            }
        }
        return GO_ON;
    }

    inline void setloop(long start, long stop, long step, long chunk, size_t nw) {
        _start=start, _stop=stop, _step=step, _chunk=chunk, _nw=nw;
        
#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        _nextIteration = _start;
#endif
        if (_chunk<=0) totaltasks = init_data_static(start,stop);
        else           totaltasks = init_data(start,stop);

        assert(totaltasks>=1);        
        // adjust the number of workers that have to be started
        if ( (totaltasks/(double)_nw) <= 1.0 || (totaltasks==1) )
           _nw = totaltasks;
    }

    inline long startIdx() const { return _start;}
    inline long stopIdx()  const { return _stop;}
    inline long stepIdx()  const { return _step;}
    inline size_t running() const { return _nw; }
    inline void workersSpinWait() { workersspinwait=true;}
    inline size_t getnumtasks() const { return totaltasks;}
protected:
    // the following fields are used only by the scheduler thread
    ff_loadbalancer *lb;
    long             _start,_stop,_step;  // for step
    long             _chunk;              // a chunk of indexes
    size_t           totaltasks;          // total n. of tasks
    size_t           _nw;                 // num. of workers
    long             jump;
    bool             skip1;
    bool             workersspinwait;
    bool             static_scheduling;
    std::vector<forall_task_t> taskv;
};

// parallel for/reduce  worker node
template<typename Tres>
class forallreduce_W: public ff_node {
public:
    typedef Tres Tres_t;
    typedef std::function<void(const long,const long, const int, Tres&)> F_t;
protected:
    virtual inline void losetime_in(unsigned long) {
        //FFTRACE(lostpopticks+=ff_node::TICKS2WAIT; ++popwait); // FIX
        workerlosetime_in(aggressive);
    }
public:
    forallreduce_W(forall_Scheduler *const sched, ffBarrier *const loopbar, F_t F):
        sched(sched),loopbar(loopbar), schedRunning(true), 
        spinwait(false), aggressive(true),F(F) {}
    
    inline void setSchedRunning(bool r) { schedRunning = r; }

    inline void* svc(void* t) {
        auto task = (forall_task_t*)t;
        auto myid = get_my_id();

#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        forall_task_t tmptask;
        if(t != (void*) &dummyTask || schedRunning){
           F(task->start,task->end,myid,res);
           if (schedRunning) return t;
        }else{
           task = &tmptask;
        }
#else
        F(task->start,task->end,myid,res);
        if (schedRunning) return t;
#endif

        // the code below is executed only if the scheduler thread is not running
        while(sched->nextTaskConcurrent(task,myid))
            F(task->start,task->end,myid,res);
        
        if (spinwait) {
            loopbar->doBarrier(myid);
            return GO_ON;
        }
        return GO_OUT;
    }

    inline void enableSpinWait() {  spinwait=true; }

    inline void setF(F_t _F, const Tres& idtt, bool a=true) { 
        F=_F, res=idtt, aggressive=a;
    }
    inline const Tres& getres() const { return res; }

protected:
    forall_Scheduler *const sched;
    ffBarrier *const loopbar;
    bool schedRunning;    
protected:
    bool spinwait,aggressive;
    F_t  F;
    Tres res;
};


class forallpipereduce_W: public forallreduce_W<ff_buffernode> {
public:
    typedef ff_buffernode Tres_t;
    typedef std::function<void(const long,const long, const int, ff_buffernode&)> F_t;
public:
    forallpipereduce_W(forall_Scheduler *const sched,ffBarrier *const loopbar, F_t F):
        forallreduce_W<ff_buffernode>(sched,loopbar,F) {
        res.set(8192,false,get_my_id());
        res.init_blocking_stuff();
    }

    inline void* svc(void* t) {
        auto task = (forall_task_t*)t;
        auto myid = get_my_id();

        F(task->start,task->end,myid,res);
        if (schedRunning) return t;

        // the code below is executed only if the scheduler thread is not running
        while(sched->nextTaskConcurrent(task,myid))
            F(task->start,task->end,myid,res);

        if (spinwait) {
            res.ff_send_out(EOS);
            loopbar->doBarrier(myid);
            return GO_ON;
        }
        return GO_OUT;
    }
    
    void svc_end() { res.ff_send_out(EOS); }

    inline void setF(F_t _F, const Tres_t&, bool a=true) { 
        F=_F, aggressive=a;
    }

    // The following methods are custom for this node which is not multi-output. FIX
    bool isMultiOutput() const { return true; }
    void get_out_nodes(svector<ff_node*> &w) { w.push_back(&res); }
    void get_out_nodes_feedback(svector<ff_node*> &w) { w.push_back(this); }
};



template <typename Worker_t>
class ff_forall_farm: public ff_farm {
public:
    typedef typename Worker_t::Tres_t Tres_t;
    typedef typename Worker_t::F_t    F_t;
protected:
    // removes possible EOS still in the input queues of the workers
    inline void resetqueues(const int _nw) {
        const svector<ff_node*> &nodes = getWorkers();
        for(int i=0;i<_nw;++i) nodes[i]->reset();
    }
    //  used just to redefine losetime_in
    class foralllb_t: public ff_loadbalancer {
    protected:
        virtual inline void losetime_in(unsigned long) { 
            if ((int)(getnworkers())>=ncores) {
                //FFTRACE(lostpopticks+=(100*TICKS2WAIT);++popwait); // FIX: adjust tracing
                ff_relax(0);
                return;
            }    
            //FFTRACE(lostpushticks+=TICKS2WAIT;++pushwait);
            PAUSE();            
        }
    public:
        foralllb_t(size_t n):ff_loadbalancer(n),ncores(ff_realNumCores()) {}
        inline int getNCores() const { return ncores;}
    private:
        const int ncores;
    };
    
private:
    Tres_t t; // not used
    size_t numCores;
    ffBarrier *loopbar;
public:

    ff_forall_farm(ssize_t maxnw, const bool spinwait=false, const bool skipwarmup=false, const bool spinbarrier=false):
        ff_farm(false,8*DEF_MAX_NUM_WORKERS,8*DEF_MAX_NUM_WORKERS,
                            true, DEF_MAX_NUM_WORKERS,true), // cleanup at exit !
        loopbar( (spinwait && spinbarrier) ? 
                 (ffBarrier*)(new spinBarrier(maxnw<=0?DEF_MAX_NUM_WORKERS+1:(size_t)(maxnw+1))) :
                 (ffBarrier*)(new Barrier(maxnw<=0?DEF_MAX_NUM_WORKERS+1:(size_t)(maxnw+1))) ),
        skipwarmup(skipwarmup),spinwait(spinwait) {

        foralllb_t* _lb = new foralllb_t(DEF_MAX_NUM_WORKERS);
        assert(_lb);
        ff_farm::setlb(_lb);
        
        numCores = ((foralllb_t*const)getlb())->getNCores();
        if (maxnw<=0) maxnw=numCores;
        std::vector<ff_node *> forall_w;
        auto donothing=[](const long,const long,const int,const Tres_t&) -> void { };
        forall_Scheduler *sched = new forall_Scheduler(getlb(),maxnw);
        ff_farm::add_emitter(sched);
        for(size_t i=0;i<(size_t)maxnw;++i)
            forall_w.push_back(new Worker_t(sched, loopbar, donothing));
        ff_farm::add_workers(forall_w);
        ff_farm::wrap_around();

        // needed to avoid the initial barrier (see (**) below)
        if (ff_farm::prepare() < 0) 
            error("running base forall farm(2)\n");
        
        // NOTE: the warmup phase has to be done, if not now later on. 
        // The run_then_freeze method will fail if skipwarmup is true.
        if (!skipwarmup) {
            auto r=-1;
            getlb()->freeze();
            if (getlb()->run() != -1) 
                r = getlb()->wait_freezing();            
            if (r<0) error("running base forall farm(1)\n");
        }

        if (spinwait) {
            sched->workersSpinWait();
            for(size_t i=0;i<(size_t)maxnw;++i) {
                //auto w = (forallreduce_W<Tres>*)forall_w[i];
                auto w = (Worker_t*)forall_w[i];
                w->enableSpinWait();
            }
            //resetqueues(maxnw);
        }
        ff_farm::cleanup_all(); // delete everything at exit
    }
    virtual ~ff_forall_farm() {
        if (loopbar) delete loopbar;
        if (ff_farm::getlb()) delete ff_farm::getlb();
    }


    // It returns true if the scheduler has to be started, false otherwise.
    //
    // Unless the removeSched flag is set, the scheduler thread will be started 
    // only if there are less threads than cores AND if the number of tasks per thread 
    // is greather than 1. In case of static scheduling (i.e. chunk<=0), the scheduler 
    // is never started because numtasks == nwtostart;
    //
    // By defining at compile time NO_PARFOR_SCHEDULER_THREAD the 
    // scheduler won't be started.
    //
    // To always start the scheduler thread, the PARFOR_SCHEDULER_THREAD 
    // may be defined at compile time.
    //
    inline bool startScheduler(const size_t nwtostart, const size_t numtasks) const { 
#if   defined(NO_PARFOR_SCHEDULER_THREAD)
        return false;
#elif defined(PARFOR_SCHEDULER_THREAD)
        return true;
#else
        if (removeSched) return false;
        return ((numtasks > nwtostart) && (nwtostart < numCores));
#endif
    }
    // set/reset removeSched flag
    // By calling this method with 'true' the scheduler will be disabled.
    //
    // NOTE:
    // Sometimes may be usefull (in terms of performance) to explicitly disable 
    // the scheduler thread when #numworkers > ff_realNumCores() on systems where
    // ff_numCores() > ff_realNumCores() (i.e. HT or SMT is enabled)
    inline void disableScheduler(bool onoff=true) { removeSched=onoff; }

    inline int run_then_freeze(ssize_t nw_=-1) {
        assert(skipwarmup == false);
        const ssize_t nwtostart = (nw_ == -1)?getNWorkers():nw_;
        auto r = -1;
        if (schedRunning) {
            getlb()->skipfirstpop(true);
            if (spinwait) {
                // NOTE: here we have to be sure to send one task to each worker!
                ((forall_Scheduler*)getEmitter())->sendTask(true);
            }
            r=ff_farm::run_then_freeze(nwtostart);
        } else {
            if (spinwait) {
                // all worker threads have already crossed the barrier so it is safe to restart it
                loopbar->barrierSetup(nwtostart+1);
                // NOTE: here is not possible to use sendTask because otherwise there could be 
                //       a race between the main thread and the workers in accessing the task table.
                ((forall_Scheduler*)getEmitter())->sendWakeUp(); 
            } else 
                ((forall_Scheduler*)getEmitter())->sendTask(true);

            r = getlb()->thawWorkers(true, nwtostart);
        }
        return r;
    }

    inline int run_and_wait_end() {
        assert(spinwait == false); 
        const size_t nwtostart = getnw();
        auto r= -1;
        if (schedRunning) {
            //resetqueues(nwtostart);
            getlb()->skipfirstpop(true); 
            // (**) this way we avoid the initial barrier
            if (getlb()->runlb()!= -1) {
                if (getlb()->runWorkers(nwtostart)!=-1)
                    r = getlb()->wait();
            }
        } else {
            ((forall_Scheduler*)getEmitter())->sendTask(true);
            if (getlb()->runWorkers(nwtostart) != -1)
                r = getlb()->waitWorkers();
        }
        return r;
    }
    
    // it puts all threads to sleep but does not disable the spinWait flag
    inline int stopSpinning() {
        if (!spinwait) return -1;
        // getnworkers() returns the number of threads that are running
        // it may be different from getnw() (i.e. the n. of threads currently 
        // executing the parallel iterations)
        size_t running = getlb()->getnworkers();
        if (running == (size_t)-1) return 0;
        getlb()->freezeWorkers();
        getlb()->broadcast_task(GO_OUT);
        return getlb()->wait_freezingWorkers();
    }

    inline int enableSpinning() {
        if (spinwait) return -1;
        const svector<ff_node*> &nodes = getWorkers();
        for(size_t i=0;i<nodes.size();++i) {
            auto w = (Worker_t*)nodes[i];
            w->enableSpinWait();
        }
        ((forall_Scheduler*)getEmitter())->workersSpinWait();
        spinwait = true;
        return 0;
    }

    inline int wait_freezing() {
        //if (startScheduler(getnw())) return getlb()->wait_lb_freezing();
        if (schedRunning) return getlb()->wait_lb_freezing();
        if (spinwait) { 
            loopbar->doBarrier(getnw()); 
            return 0;
        }
        return getlb()->wait_freezingWorkers();
    }
    
    inline int wait() {
        if (spinwait){
            const svector<ff_node*> &nodes = getWorkers();
            for(size_t i=0;i<nodes.size();++i) 
                getlb()->ff_send_out_to(EOS,i);
        }
        return ff_farm::wait();
    }

    inline void setF(F_t  _F, const Tres_t& idtt=Tres_t()) { //(Tres)0) { 
        const size_t nw                = getnw();
        const svector<ff_node*> &nodes = getWorkers();
        // aggressive mode enabled if the number of threads is less than
        // or equal to the number of cores
        const bool mode = (nw <= numCores);
    
        // NOTE: in case of static scheduling, the scheduler is never started !
        schedRunning = (!removeSched && startScheduler(nw, ((forall_Scheduler*)getEmitter())->getnumtasks()));

#ifdef FF_PARFOR_PASSIVE_NOSTEALING
        globalSchedRunning = schedRunning;
#endif

        if (schedRunning)  {
            for(size_t i=0;i<nw;++i) {
                //auto w = (forallreduce_W<Tres>*)nodes[i];
                auto w = (Worker_t*)nodes[i];
                w->setF(_F, idtt, mode);
                w->setSchedRunning(true);
            }
        } else {
            for(size_t i=0;i<nw;++i) {
                //auto w = (forallreduce_W<Tres>*)nodes[i];
                auto w = (Worker_t*)nodes[i];
                w->setF(_F, idtt, mode);
                w->setSchedRunning(false);
            }
        }
    }
    /* NOTE: - chunk>0   means dynamic scheduling with grain equal to chunk, that is,
     *                   no more than chunk iterations at a time is computed by 
     *                   one thread
     *       - chunk==0  means default static scheduling, that is, a bunch of ~(#iteration/nw) 
     *                   iterations per thread is computed by each thread
     *       - chunk<0   means static scheduling with grain equal to chunk, that is,
     *                   the iteration space is divided in chunks each one of no more 
     *                   than chunk iterations. Then chunks are assigned to the threads 
     *                   in a round-robin fashion.
     */
    inline void setloop(long begin,long end,long step,long chunk,long nw) {
        if (nw>(ssize_t)getNWorkers()) {
            error("The number of threads specified is greater than the number set in the ParallelFor* constructor, it will be downsized\n");
            nw = getNWorkers();
        }
        assert(nw<=(ssize_t)getNWorkers());
        forall_Scheduler *sched = (forall_Scheduler*)getEmitter();
        sched->setloop(begin,end,step,chunk,(nw<=0)?getNWorkers():(size_t)nw);
    }
    // return the number of workers running or supposed to run
    inline size_t getnw() { return ((const forall_Scheduler*)getEmitter())->running(); }
    
    inline const Tres_t& getres(int i) {
        //return  ((forallreduce_W<Tres>*)(getWorkers()[i]))->getres();
        return  ((Worker_t*)(getWorkers()[i]))->getres();
    }
    inline long startIdx(){ return ((const forall_Scheduler*)getEmitter())->startIdx(); }
    inline long stopIdx() { return ((const forall_Scheduler*)getEmitter())->stopIdx(); }
    inline long stepIdx() { return ((const forall_Scheduler*)getEmitter())->stepIdx(); }

    void resetskipwarmup() { assert(skipwarmup); skipwarmup=false;}
protected:
    bool   removeSched = false;
    bool   schedRunning= true;
    bool   skipwarmup  = false;
    bool   spinwait    = false;
};

    
} // namespace ff

#endif /* FF_PARFOR_INTERNALS_HPP */
    
