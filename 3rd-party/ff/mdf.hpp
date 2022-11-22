/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*! 
 *  \link
 *  \file mdf.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief This file implements the macro dataflow pattern.
 */
 
#ifndef FF_MDF_HPP
#define FF_MDF_HPP
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
 * Author: Massimo Torquati (October 2013)
 *
 *
 * Acknowledgement:
 * This implementation is a refinement of the first implementation developed
 * at the Computer Science Department of University of Pisa in the early 2013 
 * together with:
 *  - Daniele Buono       (d.buono@di.unipi.it)
 *  - Tiziano De Matteis  (dematteis@di.unipi.it)
 *  - Gabriele Mencagli   (mencagli@di.unipi.it)
 *
 */

//VS12
//#ifndef WIN32

#include <functional>
#include <tuple>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/task_internals.hpp>

namespace ff {


    /** 
     * \class ff_mdf
     * \ingroup high_level_patterns
     * 
     * \brief Macro Data Flow executor
     */
class ff_mdf:public ff_pipeline {
public:
    enum {DEFAULT_OUTSTANDING_TASKS = 2048};    
protected:    
    
    /* --------------  graph descriptor ---------------------- */
    struct base_gd: public ff_node {
        virtual inline void setMaxTasks(size_t) {}
        virtual inline void activate(bool) {}
        virtual inline void alloc_and_send(std::vector<param_info> &, base_f_t *) {}
        virtual inline void thaw(bool /*freeze*/=false,ssize_t=-1) {};
        virtual inline int  wait_freezing() { return 0; };
    };
    template<typename T>
    class GD: public base_gd {
    public:
        GD(void(*F)(T*const), T*const args):
            active(false),F(F),args(args),ntasks(0),maxMsgs(DEFAULT_OUTSTANDING_TASKS),TASKS(maxMsgs) {}

        void setMaxTasks(size_t maxtasks) {
            maxMsgs = maxtasks;
            TASKS.resize(maxMsgs);
        }
        void activate(bool a) { active=a;}
        void thaw(bool freeze=false,ssize_t=-1) { ff_node::thaw(freeze); };
        int  wait_freezing() { return ff_node::wait_freezing(); };
        int  wait() { return ff_node::wait(); }
        inline void alloc_and_send(std::vector<param_info> &P, base_f_t *wtask) {
            task_f_t *task = &(TASKS[ntasks++ % maxMsgs]);
            task->P     = P;
            task->wtask = wtask;
            while(!ff_send_out(task, -1, 1)) ff_relax(1);
        }

        void *svc(void *) {
            if (!active) return EOS;
            F(args);
            std::vector<param_info> useless;
            alloc_and_send(useless, nullptr); // END task
            return EOS;
        }

    protected:
        bool active;
        void(*F)(T*const); // user's function
        T*const args;      // F's arguments
        unsigned long ntasks, maxMsgs;
        std::vector<task_f_t> TASKS;    // FIX: svector should be used here
    };
        
    /* --------------  scheduler ----------------------------- */
    template<typename compare_t=CompareTask_Par>
    class Scheduler: public TaskFScheduler<task_f_t, compare_t> {
        using baseSched = TaskFScheduler<task_f_t, compare_t>;
        using baseSched::lb;
        using baseSched::insertTask;
        using baseSched::schedule_task;       
        using baseSched::handleCompletedTask;
        enum { RELAX_MIN_BACKOFF=1, RELAX_MAX_BACKOFF=32};
    public:
        Scheduler(ff_loadbalancer* lb, const int maxnw, void (*schedRelaxF)(unsigned long)):
            TaskFScheduler<task_f_t, compare_t>(lb,maxnw),
            task_numb(0),task_completed(0),bk_count(0),schedRelaxF(schedRelaxF),
            gd_ended(false) {
        }
        virtual ~Scheduler() {}

        int svc_init() {
            if (baseSched::svc_init()<0) return -1;
            ff_node::input_active(true);

            task_numb = task_completed = 0, bk_count = 0;
            m=0; gd_ended = false;

            return 0;
        }

        task_f_t* svc(task_f_t* task) {
            if (!task) {
                if (!gd_ended && (task_numb-task_completed)<(unsigned long)baseSched::LOWER_TH)
                    ff_node::input_active(true); // start receiveing from input channel again
                else  if (schedRelaxF) schedRelaxF(++bk_count);
                return baseSched::GO_ON;
            }
            bk_count = 0;
            if (baseSched::fromInput()) {
                task_f_t *const msg = task;
                if (msg->wtask == nullptr) {
                    gd_ended = true;
                    ff_node::input_active(false); // we don't want to read FF_EOS
                    return ((task_numb!=task_completed) ?
                            baseSched::GO_ON:
                            baseSched::EOS);
                }
                ++task_numb;
                insertTask(msg);
                schedule_task(0);
                if ((task_numb-task_completed)>(unsigned long)baseSched::LOWER_TH) {
                    ff_node::input_active(false); // stop receiving from input channel
                } 
                return baseSched::GO_ON;     
            }            
            hash_task_t * t = (hash_task_t *)task;
            ++task_completed;
            handleCompletedTask(t,lb->get_channel_id());            
            schedule_task(1); // try once more
            
            if(task_numb==task_completed && gd_ended) return baseSched::EOS;
            return baseSched::GO_ON;
        }

        void eosnotify(ssize_t /*id*/=-1) { lb->broadcast_task(FF_EOS); }
        int wait_freezing()           { return lb->wait_lb_freezing(); }

    private:
        size_t                         task_numb, task_completed, bk_count,m;
        void                         (*schedRelaxF)(unsigned long);
        bool                           gd_ended;
    };

    inline void reset() {
        gd->reset(); farm->reset(); sched->reset();
    }
	
    /* --------------  worker ------------------------------- */
    struct TaskFWorker: ff_node_t<hash_task_t> {
        inline hash_task_t *svc(hash_task_t *task) {
            task->wtask->call();
            return task;
        }
    };

    /// task function
    template<typename F_t, typename... Param>
    struct ff_mdf_f_t: public base_f_t {
        ff_mdf_f_t(const F_t F, Param&... a):F(F) { args = std::make_tuple(a...);}	
        
        inline void call() { ffapply(F, args); }
        F_t F;
        std::tuple<Param...> args;	
    };


public:
    /**
     *  \brief Constructor
     *
     *  \param F = is the user's function
     *  \param args = is the argument of the function F
     *  \param maxnw = is the maximum number of farm's workers that can be used
     *  \param schedRelaxF = is a function for managing busy-waiting in the farm scheduler
     */
    template<typename T1, typename compare_t = CompareTask_Par>
    ff_mdf(void (*F)(T1*const), T1*const args, size_t outstandingTasks=DEFAULT_OUTSTANDING_TASKS,
           int maxnw=ff_realNumCores(), void (*schedRelaxF)(unsigned long)=NULL):
        ff_pipeline(false,outstandingTasks), farmworkers(maxnw) { //NOTE: pipe has fixed size queue by default 
        GD<T1> *_gd   = new GD<T1>(F,args);
        _gd->setMaxTasks(outstandingTasks+16); // NOTE: TASKS must be greater than pipe's queue!
        farm = new ff_farm(false,640*maxnw,1024*maxnw,true,maxnw,true);
	    
        std::vector<ff_node *> w;
        // NOTE: Worker objects are going to be destroyed by the farm destructor
        for(int i=0;i<maxnw;++i) w.push_back(new TaskFWorker);
        farm->add_workers(w);
        farm->add_emitter(sched = new Scheduler<compare_t>(farm->getlb(), maxnw, schedRelaxF));
        farm->wrap_around();
	    
        ff_pipeline::add_stage(_gd);
        ff_pipeline::add_stage(farm);
        if (ff_pipeline::run_then_freeze()<0) {
            error("ff_mdf: running pipeline\n");
        } else { 
            ff_pipeline::wait_freezing();
            _gd->activate(true);
            gd = _gd;
            reset();
        }
    }
    virtual ~ff_mdf() {
        if (gd)    delete gd;
        if (sched) delete sched;
        if (farm)  delete farm;
    }

    template<typename F_t, typename... Param>
    inline void AddTask(std::vector<param_info> &P, const F_t F, Param... args) {	
        ff_mdf_f_t<F_t, Param...> *wtask = new ff_mdf_f_t<F_t, Param...>(F, args...);
        gd->alloc_and_send(P,wtask);
    }
  
    void setNumWorkers(ssize_t nw) { 
        if (nw > ff_numCores())   // TODO: use the mapper to get the number of cores
            error("ff_mdf: setNumWorkers: too much workers, setting num worker to %d\n", 
                  ff_numCores());         
        farmworkers=(std::min)(ff_numCores(),nw); 
    }	
    void setThreshold(size_t /*th*/=0) {} // FIX: 
	

    // FIX: TODO
    void *svc(void*) { 
        // FIX: (IDEA) hashing tables received as input.
        return NULL;
    }
    
    virtual inline int run_and_wait_end() {
        ff_pipeline::thaw(true, farmworkers);
        if (ff_pipeline::wait_freezing() <0) return -1;
        return ff_pipeline::wait();
    }

    virtual inline int run_then_freeze(ssize_t nw=-1) {
        if (nw>0) setNumWorkers(nw);
        ff_pipeline::thaw(true, farmworkers);
        return ff_pipeline::wait_freezing();
    }

    double ffTime() { return ff_pipeline::ffTime(); }
    double ffwTime() { return ff_pipeline::ffwTime(); }
    
protected:
    int farmworkers;   // n. of workers in the farm
    base_gd   *gd;     // first stage
    ff_farm   *farm;   // second stage
    ff_node   *sched;  // farm's scheduler
};

} // namespace

//#endif //VS12
#endif /* FF_MDF_HPP */
