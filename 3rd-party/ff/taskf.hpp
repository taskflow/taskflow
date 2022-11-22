/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*! 
 *  \link
 *  \file taskf.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief This file implements a task parallel pattern whose tasks are functions.
 */
 
#ifndef FF_TASKF_HPP
#define FF_TASKF_HPP
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
 * Author: Massimo Torquati (September 2014)
 *
 */
#include <algorithm> 
#include <ff/farm.hpp>
#include <ff/task_internals.hpp>

namespace ff {

class ff_taskf: public ff_farm {
    enum {DEFAULT_OUTSTANDING_TASKS = 2048};
protected:
    /// task function
    template<typename F_t, typename... Param>
    struct ff_task_f_t: public base_f_t {
        ff_task_f_t(const F_t F, Param&... a):F(F) { args = std::make_tuple(a...);}	
        inline void call() { ffapply(F, args); }
        F_t F;
        std::tuple<Param...> args;	
    };


    inline task_f_t *alloc_task(std::vector<param_info> &P, base_f_t *wtask) {
        task_f_t *task = &(TASKS[ntasks++ % outstandingTasks]);
        task->P     = P;
        task->wtask = wtask;
        return task;
    }
    
    /* --------------  worker ------------------------------- */
    struct Worker: ff_node_t<task_f_t> {
        inline task_f_t *svc(task_f_t *task) {
            task->wtask->call();
            return task;
        }
    };
    
    /* --------------  Scheduler ---------------------------- */
    class Scheduler: public ff_node_t<task_f_t> {
    protected:
        inline bool fromInput() { return (lb->get_channel_id() == -1);	}
    public:
        Scheduler(ff_loadbalancer*const lb, const int):
            eosreceived(false),numtasks(0), lb(lb) {}
        
        ~Scheduler() { wait(); }

        int svc_init() { numtasks = 0; eosreceived = false;  return 0;}

        inline task_f_t *svc(task_f_t *task) { 
            if (fromInput()) { 
                ++numtasks; 
                return task;
            }
            delete task->wtask;
            if (--numtasks <= 0 && eosreceived) {
                lb->broadcast_task(GO_OUT);
                return GO_OUT;
            }
            return GO_ON; 
        }

        void thaw(bool freeze, ssize_t nw) { lb->thaw(freeze,nw);}
        int wait_freezing() {  return lb->wait_lb_freezing(); }
        int wait()          {  return lb->wait(); }
        void eosnotify(ssize_t id) { 
            if (id == -1) {
                eosreceived=true; 
                if (numtasks<=0) lb->broadcast_task(EOS);
            }
        }

    protected:	
        bool   eosreceived;
        size_t numtasks;
        
        ff_loadbalancer *const lb;
    };
public:
    // NOTE: by default the scheduling is round-robin (pseudo round-robin indeed).
    //       In order to select the ondemand scheduling policy, set the ondemand_buffer to a 
    //       value grather than 1.
    ff_taskf(int maxnw=ff_realNumCores(), 
             const size_t maxTasks=DEFAULT_OUTSTANDING_TASKS, 
             const int ondemand_buffer=0):
        ff_farm(true, 
                  (std::max)(maxTasks, (size_t)(MAX_NUM_THREADS*8)),
                  (std::max)(maxTasks, (size_t)(MAX_NUM_THREADS*8)),
                  true, maxnw, true),
        farmworkers(maxnw),ntasks(0),
        outstandingTasks((std::max)(maxTasks, (size_t)(MAX_NUM_THREADS*8))),taskscounter(0) {
        
        TASKS.resize(outstandingTasks); 
        std::vector<ff_node *> w;
        // NOTE: Worker objects are going to be destroyed by the farm destructor
        for(int i=0;i<maxnw;++i) w.push_back(new Worker);
        ff_farm::add_workers(w);
        ff_farm::add_emitter(sched = new Scheduler(ff_farm::getlb(), maxnw));
        ff_farm::wrap_around();
        ff_farm::set_scheduling_ondemand(ondemand_buffer);
        
        // needed to avoid the initial barrier
        if (ff_farm::prepare() < 0) 
            error("ff_taskf: running farm (1)\n");
        
        auto r=-1;
        getlb()->freeze();
        ff_farm::offload(GO_OUT);
        if (getlb()->run() != -1) {
            getlb()->broadcast_task(GO_OUT);
            r = getlb()->wait_freezing();   
        }
        if (r<0) error("ff_taskf: running farm (2)\n");
    }
    virtual ~ff_taskf() {
        if (sched) { delete sched; sched=nullptr;}
    }
    
    template<typename F_t, typename... Param>
    inline task_f_t* AddTask(const F_t F, Param... args) {	
        // FIX: use ff_allocator here !
        ff_task_f_t<F_t, Param...> *wtask = new ff_task_f_t<F_t, Param...>(F, args...);
        std::vector<param_info> useless;
        task_f_t *task = alloc_task(useless,wtask);	
        while(!ff_farm::offload(task, 1)) ff_relax(1);	
        ++taskscounter;
        return task;
    } 
    
    virtual inline int run_and_wait_end() {
        while(!ff_farm::offload(EOS, 1)) ff_relax(1);
        sched->thaw(true,farmworkers);
        sched->wait_freezing();
            return sched->wait();
    }
    virtual int run_then_freeze(ssize_t nw=-1) {
        while(!ff_farm::offload(EOS, 1)) ff_relax(1);
        sched->thaw(true,(nw>0) ? nw:taskscounter);
        int r=sched->wait_freezing();
        taskscounter=0;
        return r;
    }
    
    // it starts all workers
    virtual inline int run(bool=false) {
        sched->thaw(true,farmworkers);
        return 0;
    }
    virtual inline int wait() { 
        while(!ff_farm::offload(EOS, 1)) ff_relax(1);
        int r=sched->wait_freezing();
        taskscounter=0;
        return r;
    }

#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- taskf:\n";
        ff_farm::ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif
    
protected:
    int farmworkers;
    Scheduler *sched;
    size_t ntasks, outstandingTasks, taskscounter;
    std::vector<task_f_t> TASKS;    // FIX: svector should be used here
};

} // namespace

#endif /* FF_TASKF_HPP */
