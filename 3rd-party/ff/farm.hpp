/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file farm.hpp
 *  \ingroup high_level_patterns building_blocks
 *  \brief Farm pattern
 *
 *  It works on a stream of tasks. Workers are non-blocking threads
 *  not tasks. It is composed by: Emitter (E), Workers (W), Collector (C).
 *  They all are C++ objects.
 *  Overall, it has one (optional) input stream and one (optional) output stream.
 *  Emitter gets stream items (tasks, i.e. C++ objects) and disptach them to 
 *  Workers (activating svc method). On svn return (or ff_send_out call), tasks
 *  are sent to Collector that gather them and output them in the output stream.

 *  Dispatching policy can be configured in the Emitter. Gathering policy in the
 *  Collector.
 * 
 *  In case of no output stream the Collector is usually not needed. Emitter 
 *  should always exist, even with no input stream.
 * 
 *  There exists several variants of the farm pattern, including
 * 
 *  \li Master-worker: no collector, tasks from Workers return to Emitter
 *  \li Ordering farm: default emitter and collector, tasks are gathered
 *  in the same order they are dispatched
 * 
 * \todo Includes classes at different levels. To be split sooner or later.
 * High level farm function to be wrapped in a separate class.
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
 *      Massimo Torquati
 */

#ifndef FF_FARM_HPP
#define FF_FARM_HPP

#include <iosfwd>
#include <vector>
#include <algorithm>
#include <memory>
#include <ff/platforms/platform.h>
#include <ff/lb.hpp>
#include <ff/gt.hpp>
#include <ff/node.hpp>
#include <ff/multinode.hpp>
#include <ff/ordering_policies.hpp>
#include <ff/all2all.hpp>

namespace ff {


/* This file provides the following classes:
 *   ff_farm    task-farm pattern 
 *   ff_Farm    typed version of the task-farm pattern (requires c++11)
 *
 */

// forward decls
class ff_farm;
static inline int optimize_static(ff_farm&, const OptLevel&);
    

/*!
 *  \class ff_farm
 * \ingroup  building_block
 *
 *  \brief The Farm skeleton, with Emitter (\p lb_t) and Collector (\p gt_t).
 *
 *  The Farm skeleton can be seen as a 3-stages pipeline. The first stage is
 *  the \a Emitter (\ref ff_loadbalancer "lb_t") that act as a load-balancer;
 *  the last (optional) stage would be the \a Collector (\ref ff_gatherer
 *  "gt_t") that gathers the results computed by the \a Workers, which are
 *  ff_nodes.
 *
 *  This class is defined in \ref farm.hpp
 */
class ff_farm: public ff_node {

    friend inline int optimize_static(ff_farm&, const OptLevel&);

protected:
    // -------- strict round-robin load balancer and gatherer -------
    // This the default policy used by the ff_farm when set_ordered
    // is called
    struct ofarm_lb: ff_loadbalancer {
        size_t victim = 0;        
        ofarm_lb(int max_num_workers):ff_loadbalancer(max_num_workers), victim(0) {}
        
        inline size_t selectworker() { return victim; }                
        inline bool schedule_task(void * task,unsigned long retry,unsigned long ticks) {
            auto s = ff_loadbalancer::schedule_task(task, retry, ticks);
            if (s) victim = (victim+1) % getnworkers();
            return s;
        }
        inline void broadcast_task(void * task) {
            const svector<ff_node*> &W = getWorkers();
            if (blocking_out) {
                size_t nw = getnworkers();
                for(size_t i=victim;i<nw;++i) {
                    while (!W[i]->put(task)) {
                        pthread_mutex_lock(prod_m);
                        struct timespec tv;
                        timedwait_timeout(tv);
                        pthread_cond_timedwait(prod_c, prod_m,&tv);
                        pthread_mutex_unlock(prod_m); 
                    }
                    put_done(i);
                }
                for(size_t i=0;i<victim;++i) {
                    while (!W[i]->put(task)) {
                        pthread_mutex_lock(prod_m);
                        struct timespec tv;
                        timedwait_timeout(tv);
                        pthread_cond_timedwait(prod_c, prod_m,&tv);
                        pthread_mutex_unlock(prod_m); 
                    }
                    put_done(i);
                }     
#if defined(FF_TASK_CALLBACK)
                callbackOut(this);
#endif
                return;
            }
            
            for(size_t i=victim;i<getnworkers();++i) {
                while (!W[i]->put(task)) losetime_out();
            }
            for(size_t i=0;i<victim;++i) {
                while (!W[i]->put(task)) losetime_out();
            }
            
#if defined(FF_TASK_CALLBACK)
            callbackOut(this);
#endif
        }
        inline void thaw(bool freeze=false, ssize_t nw=-1) {
            if (nw < (ssize_t)victim) victim = 0;
            ff_loadbalancer::thaw(freeze,nw);
        }
        inline int thawWorkers(bool freeze=false, ssize_t nw=-1) {
            if (nw < (ssize_t)victim) victim = 0;
            return ff_loadbalancer::thawWorkers(freeze,nw);
        }
        bool ff_send_out_to(void *task, int id,unsigned long retry,unsigned long ticks) {
            if (victim==(size_t)id) return ff_loadbalancer::ff_send_out_to(task, id, retry, ticks);
            return false; 
        }
    };
    struct ofarm_gt: ff_gatherer {
        ofarm_gt(int max_num_workers):
            ff_gatherer(max_num_workers),dead(max_num_workers) {
            dead.resize(max_num_workers);
        }
        inline ssize_t selectworker() { return victim; }
        void updatenextone() {
            size_t start = victim;
            do {
                victim= (victim+1) % getrunning();
            } while(dead[victim] && victim != start);
        }
        inline ssize_t gather_task(void ** task) {
            auto nextr = ff_gatherer::gather_task(task);
            assert((size_t)nextr == victim);
            if (*task == FF_EOS || *task==FF_EOSW || *task==FF_EOS_NOFREEZE)
                dead[victim] = true;
            updatenextone();
            return nextr;
        }
        int svc_init() {
            for(size_t i=0;i<dead.size();++i) dead[i]=false;
            return ff_gatherer::svc_init();
        }
        void thaw(bool freeze=false, ssize_t nw=-1) {
            if (nw < (ssize_t)victim) victim = 0;
            ff_gatherer::thaw(freeze,nw);
        }    
        size_t victim = 0;
        svector<bool> dead;
    };    
protected:
    inline int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(size_t i=0;i<workers.size();++i) 
            card += workers[i]->cardinality(barrier);
        
        lb->set_barrier(barrier);
        if (gt) gt->set_barrier(barrier);

        return (card + 1 + ((collector && !collector_removed)?1:0));
    }
    
    inline int prepare() {
        size_t nworkers = workers.size();
        if (nworkers==0 || nworkers > max_nworkers) {
            error("FARM: wrong number of workers\n");
            return -1;
        }
        for(size_t i=0;i<workers.size();++i) {
            workers[i]->set_id(int(i));
        }

        // NOTE: if the farm is in a master-worker configuration, all workers must be either
        //       sequential or parallel  building block
        if (lb->masterworker()) {
            bool is_parallel_bb = workers[0]->isAll2All() || workers[0]->isFarm() || workers[0]->isPipe();
            lb->parallel_workers = is_parallel_bb;
            for(size_t i=1;i<workers.size();++i) {
                bool tmp=workers[i]->isAll2All() || workers[i]->isFarm() || workers[i]->isPipe();
                if (tmp != is_parallel_bb) {
                    error("FARM, prepare, farm in master-worker configuration but with non homogeneous workers\n");
                    return -1;
                }
            }
        }
        
        // ordering
        if (ordered) {

            // TODO: this constraint must be relaxed!!!!!!
            if (workers[0]->isFarm() || workers[0]->isPipe() || workers[0]->isMultiInput()
                || workers[0]->isMultiOutput() || workers[0]->isAll2All() || workers[0]->isComp() ) {
                error("FARM: ordered farm is currently supported only for standard node!\n");
                return -1;
            }
            
            if (ondemand) {                                   
                ordered_lb* _lb= new ordered_lb(nworkers);
                ordered_gt* _gt= new ordered_gt(nworkers);
                assert(_lb); assert(_gt);
                ordering_Memory.resize(nworkers * (2*ff_farm::ondemand_buffer()+3)+ordering_memsize);
                _lb->init(ordering_Memory.begin(), ordering_Memory.size());
                _gt->init(ordering_memsize);
                setlb(_lb, true);
                setgt(_gt, true);
                
                for(size_t i=0;i<nworkers;++i) {
                    workers[i] = new OrderedWorkerWrapper(workers[i], worker_cleanup);
                    assert(workers[i]);
                }
                worker_cleanup = true;
            } else {
                ofarm_lb* _lb = new ofarm_lb(nworkers);
                ofarm_gt* _gt = new ofarm_gt(nworkers);
                assert(lb); assert(gt);
                setlb(_lb, true);
                setgt(_gt, true);
            }        
        }

        // accelerator
        if (has_input_channel) { 
            if (create_input_buffer(in_buffer_entries, fixedsizeIN)<0) {
                error("FARM, creating input buffer\n");
                return -1;
            }
            if (hasCollector()) {
                // NOTE: the queue is forced to be unbounded
                if (create_output_buffer(out_buffer_entries, false)<0) return -1;
            }
        }


        
        for(size_t i=0;i<nworkers;++i) {

            ff_a2a* a2a_first = nullptr;        // the Emitter sees this all-to-all
            ff_a2a* a2a_last  = nullptr;        // the Collector sees this all-to-all
            if (workers[i]->isAll2All()) {
                a2a_first = reinterpret_cast<ff_a2a*>(workers[i]);
                a2a_last  = a2a_first; 
            } else {  // TODO: farm with workers A2A or pipeline ending with A2A
                if (workers[i]->isPipe()) {
                    ff_pipeline* pipe=reinterpret_cast<ff_pipeline*>(workers[i]);
                    ff_node* node0 = pipe->get_node(0);
                    ff_node* nodeN = pipe->get_lastnode();
                    if (node0->isAll2All()) {
                        a2a_first = reinterpret_cast<ff_a2a*>(node0);
                    }
                    if (nodeN->isAll2All()) {
                        a2a_last  = reinterpret_cast<ff_a2a*>(nodeN);
                    }
                }
            }
            
            if (a2a_first) {  
                //NOTE: if the worker is an A2A or a pipe starting with an A2A, the L-Workers
                //      are transformed by the prepare, that's why the prepare must be done
                //      before adding workers to the emitter
                if (a2a_first->prepare()<0) {
                    error("FARM, preparing worker A2A %d\n", i);
                    return -1;
                }


                
                if (a2a_first->create_input_buffer((int) (ondemand ? ondemand: in_buffer_entries), 
                                             (ondemand ? true: fixedsizeIN))<0) return -1;
                
                const svector<ff_node*>& W1 = a2a_first->getFirstSet();
                for(size_t i=0;i<W1.size();++i) {
                    lb->register_worker(W1[i]);
                }
            } else {
                if (workers[i]->create_input_buffer((int) (ondemand ? ondemand: in_buffer_entries), 
                                                    (ondemand ? true: fixedsizeIN))<0) return -1;

                lb->register_worker(workers[i]);
            }

            if (a2a_last) {

                if (a2a_first != a2a_last) {
                    //NOTE: if the worker is an A2A or a pipe ending with an A2A, the R-Workers
                    //      are transformed by the prepare, that's why the prepare must be done
                    //      before adding workers to the collector
                    if (a2a_last->prepare()<0) {
                        error("FARM, preparing worker A2A %d (collector side)\n", i);
                        return -1;
                    }
                }
                                    
                const svector<ff_node*>& W2 = a2a_last->getSecondSet();                

                // TODO: If the internal A2A has feedbacks toward the Emitter (i.e. master-worker) and there is also the collector
                //       then this case is not properly handled because R-Workers are not connected to the collector
                if (collector && !collector_removed) {
                    if (!lb->masterworker()) {
                        
                        // NOTE: the following call might fail because the buffers were already created for example by
                        // the pipeline that contains this stage
                        a2a_last->create_output_buffer(out_buffer_entries,(lb->masterworker()?false:fixedsizeOUT));
                        
                        for(size_t i=0;i<W2.size();++i) {
                            svector<ff_node*> w(1);
                            W2[i]->get_out_nodes(w);
                            assert(w.size()>0);
                            for(size_t j=0;j<w.size();++j)
                                gt->register_worker(w[j]);
                        }
                    } else {
                        error("FARM feature not yet supported (1)\n");
                        abort(); // <---- FIX
                    }
                } else { // there is no collector
                    if (outputNodes.size()) { 
                        assert(W2.size() == outputNodes.size());
                        for(size_t i=0;i<W2.size();++i) {
                            if (W2[i]->set_output(outputNodes[i])<0) return -1;
                                                                                     }
                    } else {
                        // TODO: for the moment we support only feedback channels and not both feedback and forward channels
                        //       for the last stages of the last all-to-all
                        if (lb->masterworker()) {
                            if (a2a_last->create_output_buffer(out_buffer_entries,(lb->masterworker()?false:fixedsizeOUT))<0) {
                                error("FARM failed to create feedback channels\n");
                                return -1;
                            }

                            for(size_t i=0;i<W2.size();++i) {
                                svector<ff_node*> w(1);
                                W2[i]->get_out_nodes(w);
                                assert(w.size()>0);
                                for(size_t j=0;j<w.size();++j)
                                    lb->set_input_feedback(w[j]);
                            }
                        }
                    }
                }

                continue;
            }


            // helper function
            auto create_feedback_buffers = [&]() {
                static int idx=0;
                svector<ff_node*> w(1);
                workers[i]->get_out_nodes(w);
                for(size_t j=0;j<w.size();++j) {
                    ff_buffernode* t = new ff_buffernode(out_buffer_entries, false, idx++);
                    assert(t);
                    internalSupportNodes.push_back(t);
                    workers[i]->set_output_feedback(t);
                    
                    // REMEMBER: if the worker is not parallel (i.e., farm, a2a, pipeline) then
                    // the lb adds the workers to the 'availworkers' array directly,
                    // so set_input_feedback must be avoided to don't have duplicates.
                    if (lb->parallel_workers) {                                    
                        lb->set_input_feedback(t);
                    } else {
                        workers[i]->set_output_buffer(t->get_out_buffer());
                    }
                }                                        
            };
            
            
            if (collector && !collector_removed) {   // there is a collector         
                if (workers[i]->get_out_buffer()==NULL) {
                    if (workers[i]->isMultiOutput()) {   // the worker is multi-output
                        if (lb->masterworker()) {   // there is a feedback
                            create_feedback_buffers();
                        }
                        
                        
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        workers[i]->get_out_nodes(w);
                        if (w.size()>0) {                            
                            static int idx=0;
                            
                            for(size_t j=0;j<w.size();++j) {
                                ff_node* t = new ff_buffernode(out_buffer_entries,fixedsizeOUT, idx++);
                                assert(t);
                                internalSupportNodes.push_back(t);
                                if (w[j]->isMultiOutput()) {
                                    if (w[j]->set_output(t)<0) return -1;
                                } else {
                                    if (workers[i]->set_output(t)<0) return -1;
                                }
                                gt->register_worker(t);
                            }                            
                        } else  { // single node multi-output
                            ff_node* t = new ff_buffernode(out_buffer_entries,fixedsizeOUT, i); 
                            internalSupportNodes.push_back(t);
                            workers[i]->set_output(t);
                            if (!lb->masterworker()) workers[i]->set_output_buffer(t->get_out_buffer());
                            gt->register_worker(t);
                        }                        
                    } else { // standard worker or composition where the second stage is not multi-output
                        if (workers[i]->create_output_buffer(out_buffer_entries,(lb->masterworker()?false:fixedsizeOUT))<0)
                            return -1;
                        assert(!lb->masterworker());
                        gt->register_worker(workers[i]);
                    }
                    
                }

                // this is possible only if the collector filter is a multi-output node
                if (outputNodes.size()) { 
                    assert((collector != (ff_node*)gt) && collector->isMultiOutput());
                    collector->set_output(outputNodes);
                }
                
            } else { // there is not a collector
                if (workers[i]->get_out_buffer()==NULL) {
                    if (workers[i]->isMultiOutput()) {
                        if (lb->masterworker()) {    // feedback channels from workers
                            create_feedback_buffers();
                        }
                                                
                        if (outputNodes.size()) { 
                            workers[i]->set_output(outputNodes); 
                        }
                        
                    } else {  // the worker is not multi-output
                        if (outputNodes.size()) {
                            assert(outputNodesFeedback.size()==0);
                            assert(!lb->masterworker());    // no master-worker 
                            assert(outputNodes.size() == workers.size()); // same cardinality
                            workers[i]->set_output_buffer(outputNodes[i]->get_in_buffer());
                        }
                        else{
                            if (outputNodesFeedback.size()) {
                                assert(!lb->masterworker());    // no master-worker 
                                assert(outputNodesFeedback.size() == workers.size()); // same cardinality
                                workers[i]->set_output_buffer(outputNodesFeedback[i]->get_in_buffer());
                            } else {
                                if (lb->masterworker()) {
                                    create_feedback_buffers();
                                } 
                            }
                        }
                    }
                }
            }
        }
        
        // preparing emitter
        if (emitter) {
            if (emitter->isMultiOutput()) {
                // we can use a multi-output node as emitter or a composition where the 
                // last stage is a multi-output node                
                emitter->setlb(lb);
            }
            if (lb->set_filter(emitter)<0) {
                error("FARM, preparing emitter filter\n");
                return -1;
            }
            // if the emitter is a composition we have to call prepare
            // and to be sure to have a consistent gt
            if (emitter->isComp()) {
                svector<ff_node*> w(MAX_NUM_THREADS);
                lb->get_in_nodes(w);
                if (w.size())  emitter->set_input(w);
                if (emitter->prepare()<0) { 
                    error("FARM, preparing emitter filter\n");
                    return -1;
                }
            }
        }
        // preparing collector
        if (collector && !collector_removed && (collector != (ff_node*)gt) ) {
            if (collector->isMultiInput()){
                collector->setgt(gt);
            }
            if (collector->isMultiOutput())
                if (collector->prepare()<0) {
                    error("FARM, preparing multi-output collector filter\n");
                    return -1;
                }
            if (collector->isComp()) 
                if (collector->prepare()<0) { 
                    error("FARM, preparing collector filter\n");
                    return -1;
                }
            
            // NOTE: if the collector has a filter and the set_output_blocking has
            // been already executed (for example by the pipeline), then we
            // have to set the blocking stuff also for the filter
            if (gt->set_filter(collector)<0) {
                error("FARM, preparing collector filter\n");
                return -1;
            }
        }

        
        // blocking stuff --------------------------------
        for(size_t i=0;i<nworkers;++i) {
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (!workers[i]->init_input_blocking(m,c)) {
                error("FARM, init input blocking mode for worker %d\n", i);
                return -1;
            }
            if (!workers[i]->init_output_blocking(m,c)) {
                error("FARM, init output blocking mode for worker %d\n", i);
                return -1;
            }
        }

        // NOTE: if the emitter filter is a multi-output node, it shares
        //       the same lb of the farm.
        // NOTE: if the collector filter is a multi-input node, it shares
        //       the same gt of the farm.
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        if (!lb->init_output_blocking(m,c)) {
            error("FARM, init output blocking mode for LB\n");
            return -1;
        }
        if (collector && !collector_removed) {
            if (!gt->init_input_blocking(m,c)) {
                error("FARM, init output blocking mode for GT\n");
                return -1;
            }
            for(size_t j=0;j<nworkers;++j) {
                svector<ff_node*> w;
                workers[j]->get_out_nodes(w);
                assert(w.size()>0);
                for(size_t i=0;i<w.size(); ++i) 
                    w[i]->set_output_blocking(m,c);
            }
        }

        if (lb->masterworker()) {
            bool last_multioutput = [&]() {
                // WARNING: we consider homogeneous workers!
                if (lb->parallel_workers) {
                    svector<ff_node*> w;
                    workers[0]->get_out_nodes(w);
                    if (w[0]->isMultiOutput()) return true;
                    return false;
                }
                if (workers[0]->isMultiOutput()) return true;
                return false;
            } ();
            
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (!init_input_blocking(m,c)) {
                error("FARM, init input blocking mode for master-worker\n");
                return -1;
            }
            for(size_t j=0;j<nworkers;++j) {
                svector<ff_node*> w;
                if (last_multioutput) {
                    workers[j]->get_out_nodes_feedback(w);
                    if (w.size() == 0)
                        workers[j]->get_out_nodes(w);
                } else 
                    workers[j]->get_out_nodes(w);                    
                assert(w.size()>0);
                // NOTE: it is possible that we have to overwrite the 
                //       p_cons_* variables that could have been set in the
                //       wrap_around method. This can happen when the last
                //       stage is multi-output and it has a feedback channel
                //       and one (or more) forward channels to the next stage.
                for(size_t i=0;i<w.size(); ++i) 
                    w[i]->set_output_blocking(m,c, true);  
            }
        }

        if (has_input_channel) {
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (!init_input_blocking(m,c)) {
                error("FARM, init input blocking\n");
                return -1;
            }
            // this is to notify the Emitter when the queue
            // is not anymore empty 
            ff_node::set_output_blocking(m,c);
            // this is to setup the condition variable where
            // the thread (main?) that is using the accelerator
            // will sleep if the queue fill up
            if (!ff_node::init_output_blocking(m,c)) {
                error("FARM, init output blocking\n");
                return -1;
            } 
            
            if (hasCollector()) {
                m=NULL,c=NULL;
                if (!init_output_blocking(m,c)) {
                    error("FARM, init output blocking\n");
                    return -1;
                } 
                
                m=NULL,c=NULL;
                if (!ff_node::init_input_blocking(m,c)) {
                    error("FARM, init input blocking\n");
                    return -1;
                } 
                set_output_blocking(m,c);
            }
        }
        
        
        prepared=true;
        return 0;
    }

    int freeze_and_run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;
        freeze();
        return run(true);
    }

    inline void skipfirstpop(bool sk)   { 
        lb->skipfirstpop(sk);
        skip1pop=sk;
    }

 
#ifdef DFF_ENABLED
    void skipallpop(bool sk)   { 
        lb->skipallpop(sk);
        ff_node::skipallpop(sk);
    }
#endif   


    // consumer
    virtual inline bool init_input_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool /*feedback*/=true) {
        bool r = lb->init_input_blocking(m,c);
        if (!r) return false;
        // NOTE: for all registered input node (or buffernode) we have to set the 
        // blocking stuff (see also ff_minode::init_input_blocking)
        svector<ff_node*> w;
        lb->get_in_nodes(w);
        lb->get_in_nodes_feedback(w);
        for(size_t i=0;i<w.size();++i)
            w[i]->set_output_blocking(m,c);
        return true;        
    }
    // producer
    virtual inline bool init_output_blocking(pthread_mutex_t   *&m,
                                             pthread_cond_t    *&c,
                                             bool /*feedback*/=true) {
        if (collector && !collector_removed) {
            if (collector == (ff_node*)gt)
                return gt->init_output_blocking(m,c);
            return collector->init_output_blocking(m,c);
        }
        for(size_t i=0;i<workers.size();++i)
            if (!workers[i]->init_output_blocking(m,c)) return false;

        return true;
    }
    virtual inline void set_output_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool canoverwrite=false) {
        if (collector && !collector_removed) {
            if (collector == (ff_node*)gt)
                gt->set_output_blocking(m,c, canoverwrite);
            else
                collector->set_output_blocking(m,c, canoverwrite);
        }
        else {
            for(size_t i=0;i<workers.size();++i)
                workers[i]->set_output_blocking(m,c, canoverwrite);
        }
    }

    virtual inline pthread_cond_t    &get_cons_c()        { return *(lb->cons_c);}

public:

    using lb_t = ff_loadbalancer;
    using gt_t = ff_gatherer;    

    /*
     * \ingroup building_blocks
     * @brief farm building block 
     *
     * This is the farm constructor.
     * Note that, by using this constructor, the collector IS added automatically !
     *
     * @param W vector of workers
     * @param Emitter pointer to Emitter object (mandatory)
     * @param Collector pointer to Collector object (optional)
     * @param input_ch \p true for enabling the input stream
     */
    ff_farm(const std::vector<ff_node*>& W, ff_node *const Emitter=NULL, ff_node *const Collector=NULL, bool input_ch=false):
        has_input_channel(input_ch),collector_removed(false),ordered(false),fixedsizeIN(FF_FIXED_SIZE),fixedsizeOUT(FF_FIXED_SIZE),
        myownlb(true),myowngt(true),worker_cleanup(false),emitter_cleanup(false),
        collector_cleanup(false),ondemand(0),
        in_buffer_entries(DEFAULT_BUFFER_CAPACITY),
        out_buffer_entries(DEFAULT_BUFFER_CAPACITY),
        max_nworkers(DEF_MAX_NUM_WORKERS),ordering_memsize(0),
        emitter(NULL),collector(NULL),
        lb(new lb_t(max_nworkers)),gt(new gt_t(max_nworkers)),
        workers(W.size()) {

        assert(W.size()>0);
        add_workers(W);

        if (Emitter) add_emitter(Emitter); 

        // add default collector even if Collector is NULL, 
        // if you don't want the collector you have to call remove_collector
        add_collector(Collector); 
    }

    /*
     * \ingroup building_blocks
     * @brief farm building block 
     *
     * This is the farm constructor.
     */
    ff_farm(const std::vector<ff_node*>& W, ff_node& E, ff_node& C):ff_farm(W,&E,&C,false) {
    }
    
    /**
     * \ingroup building_blocks
     * \brief farm building block
     *
     *  This is the basic constructor for the farm building block. To be coupled with \p add_worker, \p add_emitter, and \p add_collector
     *  Note that, by using this constructor, the collector is NOT added automatically !
     *
     *  \param input_ch = true to set accelerator mode
     *  \param in_buffer_entries = input queue length
     *  \param out_buffer_entries = output queue length
     *  \param max_num_workers = highest number of farm's worker
     *  \param worker_cleanup = true deallocate worker object at exit
     *  \param fixedsize = true uses only fixed size queue (both between Emitter and Workers and between Workers and Collector)
     */
    explicit ff_farm(bool input_ch=false,
                     int in_buffer_entries=DEFAULT_BUFFER_CAPACITY,
                     int out_buffer_entries=DEFAULT_BUFFER_CAPACITY,
                     bool worker_cleanup=false, // NOTE: by default no cleanup at exit is done !
                     size_t max_num_workers=DEF_MAX_NUM_WORKERS,
                     bool fixedsize=FF_FIXED_SIZE): 
        has_input_channel(input_ch),collector_removed(false), ordered(false), fixedsizeIN(FF_FIXED_SIZE),fixedsizeOUT(FF_FIXED_SIZE),
        myownlb(true),myowngt(true),worker_cleanup(worker_cleanup),emitter_cleanup(false),
        collector_cleanup(false), ondemand(0),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries),        
        max_nworkers(max_num_workers),ordering_memsize(0),
        emitter(NULL),collector(NULL),
        lb(new lb_t(max_num_workers)),gt(new gt_t(max_num_workers)),
        workers(max_num_workers)  {

        for(size_t i=0;i<max_num_workers;++i) workers[i]=NULL;        
    }

    ff_farm(const ff_farm& f) : ff_node(f) { 
        if (f.prepared) {
            error("ff_farm, copy constructor, the input farm is already prepared\n");
            return;
        }

        has_input_channel = f.has_input_channel;
        collector_removed = f.collector_removed;
        ordered           = f.ordered;
        ordering_memsize  = f.ordering_memsize;
        ondemand = f.ondemand; in_buffer_entries = f.in_buffer_entries;
        out_buffer_entries = f.out_buffer_entries;
        worker_cleanup = f.worker_cleanup; 
        emitter_cleanup = f.emitter_cleanup;
        collector_cleanup = f.collector_cleanup;
        max_nworkers = f.max_nworkers;
        internalSupportNodes= f.internalSupportNodes;
        fixedsizeIN  = f.fixedsizeIN;
        fixedsizeOUT = f.fixedsizeOUT;
        myownlb = f.myownlb;
        myowngt = f.myowngt;
        workers = f.workers;
        emitter = nullptr;
        collector = nullptr;

        //lb = new lb_t(max_nworkers);
        //gt = new gt_t(max_nworkers);
        lb=nullptr;setlb(f.lb); myownlb = f.myownlb;
        gt=nullptr;setgt(f.gt); myowngt = f.myowngt;
        assert(lb); assert(gt);
        
        add_emitter(f.emitter);
        if (f.hasCollector()) add_collector(f.getCollector());
        
        // this is a dirty part, we modify a const object.....
        ff_farm *dirty         = const_cast<ff_farm*>(&f);
        ordering_Memory          = std::move(dirty->ordering_Memory);
        dirty->worker_cleanup    = false;
        dirty->emitter_cleanup   = false;
        dirty->collector_cleanup = false;        
        dirty->myownlb           = false;
        dirty->myowngt           = false;
        dirty->internalSupportNodes.resize(0);
    }
    
    /* move constructor */
    ff_farm(ff_farm &&f):ff_node(std::move(f)), workers(std::move(f.workers)), internalSupportNodes(std::move(f.internalSupportNodes)) {
        
        has_input_channel = f.has_input_channel;
        collector_removed = f.collector_removed;
        ordered           = f.ordered;
        ordering_memsize  = f.ordering_memsize;
        ordering_Memory   = std::move(f.ordering_Memory);
        ondemand = f.ondemand; in_buffer_entries = f.in_buffer_entries;
        out_buffer_entries = f.out_buffer_entries;
        worker_cleanup = f.worker_cleanup; 
        emitter_cleanup = f.emitter_cleanup;
        collector_cleanup = f.collector_cleanup;
        max_nworkers = f.max_nworkers;
        fixedsizeIN  = f.fixedsizeIN;
        fixedsizeOUT = f.fixedsizeOUT;

        emitter = f.emitter;  collector = f.collector;
        lb = f.lb;   gt = f.gt;     
        myownlb = f.myownlb; myowngt = f.myowngt;
        f.lb = nullptr;
        f.gt = nullptr;
        f.max_nworkers=0;
        f.worker_cleanup    = false;
        f.emitter_cleanup   = false;
        f.collector_cleanup = false;
        f.myownlb           = false;
        f.myowngt           = false;        
    }


    /** 
     * \brief Destructor
     *
     * Destruct the load balancer, the
     * gatherer, all the workers
     */
    virtual ~ff_farm() { 
        if (emitter_cleanup) {
            if (lb && myownlb && lb->get_filter()) delete lb->get_filter();
            else if (emitter) delete emitter;
        }
        if (collector_cleanup) {
            if (gt && myowngt && gt->get_filter()) delete gt->get_filter();
            else if (collector != (ff_node*)gt) delete collector;
        }
        if (lb && myownlb) { delete lb; lb=NULL;}
        if (gt && myowngt) { delete gt; gt=NULL;}
        if (worker_cleanup) {
            for(size_t i=0;i<workers.size(); ++i) 
                if (workers[i]) delete workers[i];
        }
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes[i];
        }
        
        if (barrier) {delete barrier; barrier=NULL;}
    }

    // used to redefine scheduling/gathering policy
    void setgt(ff_gatherer *external_gt, bool cleanup=false) {        
        assert(external_gt);
        if (gt) {
            *external_gt = std::move(*gt);
        }
        if (myowngt) {
            if (collector == (ff_node*)gt) collector = (ff_node*)external_gt;
            delete gt; gt = nullptr;
            myowngt=false;
        }
        gt = external_gt;
        myowngt = cleanup;
    }
    void setlb(ff_loadbalancer *external_lb, bool cleanup=false) {
        assert(external_lb);
        if (lb) { 
            *external_lb = std::move(*lb);
        }
        if (myownlb) {
            delete lb; lb=nullptr;
            myownlb=false;
        }
        lb = external_lb;
        myownlb = cleanup;
    }    
    
    /** 
     *
     *  \brief Adds the emitter
     *
     *  It adds an Emitter to the Farm. The Emitter is of type \p ff_node and
     *  there can be only one Emitter in a Farm skeleton. 
     *  
     *  \param e the \p ff_node acting as an Emitter 
     *
     *  \return Returns 0 if successful -1 otherwise
     *
     */
    template<typename T>
    int add_emitter(T * e) { 
        if (e==nullptr) return 0;
        if (e->isFarm() || e->isAll2All() || e->isPipe()) {
            error("FARM, add_emitter: wrong kind of node, the Emitter filter cannot be a parallel building block (i.e. farm, all2all, pipeline)\n");
            return -1;
        }
        if (e->isComp()) {
            // NOTE: if a comp is set as a filter in the emitter of a farm,
            // it must terminate with a multi-output node.
            // In the previous version, it was allowed to add whatever kind of sequantial BB
            // as filter, but in some cases we experienced problems with EOS propagation
            if (!e->isMultiOutput()) {
                error("FARM, add_emitter: wrong kind of node, if the filter is a combine building block, it must terminate with a multi-output node\n");

               
                abort();    // WARNING: here for debugging purposes, it must be removed!          
                return -1;
            }

            // the combine is forced to appear as multi-input even if it is not
            e->set_multiinput();                             
        }
        if (emitter) {
            error("FARM, add_emitter: emitter already present\n");
            return -1; 
        }
        emitter = e;
        
        // if the emitter is a real multi-input, then we have to register the callback for
        // the all_gather call        
        if (e->isMultiInput()) {
            e->registerAllGatherCallback(lb->ff_all_gather_emitter, lb);
        }      
        return 0;
    }
    template<typename T>
    int add_emitter(const T& e) {
        T* n = new T(e);
        assert(n);
        int r = add_emitter(n);
        if (r<0) return -1;
        emitter_cleanup=true;
        return 0;
    }

    template<typename T>
    int change_emitter(T *e, bool cleanup=false) {
        if (emitter==nullptr) return add_emitter(e);
        if (emitter_cleanup) {
            delete emitter;
            emitter_cleanup=false;
        }
        lb->reset_filter();
        emitter=nullptr;
        emitter_cleanup=cleanup;
        return this->add_emitter(e);
    }
    template<typename T>
    int change_emitter(const T& e, bool cleanup=false) {
        if (emitter==nullptr) return add_emitter(e);
        if (emitter_cleanup) {
            delete emitter;
            emitter_cleanup=false;
        }
        lb->reset_filter();
        emitter=nullptr;
        emitter_cleanup=cleanup;
        return this->add_emitter(e);
    }

    bool change_node(ff_node* old, ff_node* n, bool cleanup=false, bool remove_from_cleanuplist=false) {
        assert(old!=nullptr);
        assert(n!=nullptr);
        if (prepared) {
            error("FARM, change_node cannot be called because the FARM has already been prepared\n");
            return false;
        }

        if (emitter == old) return (change_emitter(n, cleanup)==0);

        if (collector && !collector_removed && collector == old) {
            if (collector_cleanup) {
                delete collector;
                collector_cleanup=false;
            }
            gt->reset_filter();
            collector=nullptr;
            collector_cleanup=cleanup;
            return (this->add_collector(n) == 0);
        }
        
        for(size_t i=0; i<workers.size();++i) {
            if (workers[i] == old) {
                if (remove_from_cleanuplist) {
                    int pos=-1;
                    for(size_t i=0;i<internalSupportNodes.size();++i)
                        if (internalSupportNodes[i] == old) { pos = i; break; }
                    if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);            
                }

                if (worker_cleanup)
                    internalSupportNodes.push_back(workers[i]);
                
                workers[i] = n;
                if (cleanup && !worker_cleanup) internalSupportNodes.push_back(n);                
                return true;
            }
        }
        
        return false;
    }
    
    /**
     *
     * \brief Set scheduling with on demand polity
     *
     * The default scheduling policy is round-robin, When there is a great
     * computational difference among tasks the round-robin scheduling policy
     * could lead to load imbalance in worker's workload (expecially with short
     * stream length). The on-demand scheduling policy can guarantee a near
     * optimal load balancing in lots of cases. Alternatively it is always
     * possible to define a complete application-level scheduling by redefining
     * the ff_loadbalancer class.
     *
     * \param inbufferentries sets the number of queue slot for one worker
     * threads. If the input parameter should be greater than 0. If it is 0
     * then the ondemand scheduling is NOT set.
     *
     */
    void set_scheduling_ondemand(const int inbufferentries=1) {
        if (prepared) {
            error("FARM, set_scheduling_ondemand, farm already prepared\n");
            return;
        }
        if (inbufferentries<=0) ondemand=1;
        else ondemand=inbufferentries;
    }
    /**
     * \brief Force ordering. 
     *  
     * The data elements will be produced in output respecting the
     * input ordering.
     *
     * The \param MemoryElements sets the maximum size of the buffer in the 
     * collector when the scheduling of elements is on-demand.
     */
    void set_ordered(const size_t MemoryElements=DEF_OFARM_ONDEMAND_MEMORY) {
        if (prepared) {
            error("FARM, set_ordered, farm already prepared\n");
            return;
        }
        ordered = true;
        ordering_memsize=MemoryElements;
    }

    void ordered_resize_memory(const size_t size) {
        ordering_Memory.resize(size);
    }
    ordering_pair_t* ordered_get_memory() { return ordering_Memory.begin(); }
    
    int ondemand_buffer() const { return ondemand; }
    ssize_t ordering_memory_size() const { return ordering_memsize; }
    
    /**
     *  \brief Adds workers to the form
     *
     *  Add workers to the Farm. There is a limit to the number of workers that
     *  can be added to a Farm. This limit is set by default to 64. This limit
     *  can be augmented by passing the desired limit as the fifth parameter of
     *  the \p ff_farm constructor.
     *
     *  \param w a vector of \p ff_nodes which are Workers to be attached to
     *  the Farm.
     *
     *  \return 0 if successsful, otherwise -1 is returned.
     */
    int add_workers(const std::vector<ff_node *> & w) { 
        if ((workers.size()+w.size())> max_nworkers) {
            error("FARM, try to add too many workers, please increase max_nworkers\n");
            return -1; 
        }
        if (w.size()==0) {
            error("FARM, try to add zero workers!\n");
            return -1; 
        }        
        for(size_t i=0;i<w.size();++i) {
            workers.push_back(w[i]);	   
        }

        return 0;
    }
    /**
     * replace the workers node. Note, that no cleanup of previous workers will be done.
     * For more fine-grained control you should use change_node
     */
    int change_workers(const std::vector<ff_node*>& w) {
        workers.clear();
        return add_workers(w);
    }

    
    /**
     *  \brief Adds the collector
     *
     *  It adds the Collector filter to the farm skeleton. If no object is
     *  passed as a colelctor, than a default collector will be added (i.e.
     *  \link ff_gatherer \endlink). Note that it is not possible to add more
     *  than one collector. 
     *
     *  \param c Collector object
     *
     *  \return The status of \p set_filter(x) if successful, otherwise -1 is
     */
    int add_collector(ff_node * c, bool cleanup=false) { 
        if (c && (c->isFarm() || c->isAll2All() || c->isPipe())) {
            error("FARM, add_collector: wrong kind of node, the Collector filter can be either a standard node or a multi-input node or a multi-output node\n");
            return -1;
        }
        
        if (collector && (collector != (ff_node*)gt) && !collector_removed) {
            error("add_collector: collector already defined!\n");
            return -1; 
        }
        if (!gt) return -1; //inconsist state

        // NOTE: if a comp is set as a filter in the collector of a farm,
        // it is a multiinput node even if the first stage of the composition
        // is not a multi-input.
        if (c && c->isComp()) {
            // NOTE: if a comp is set as a filter in the collector of a farm,
            // it should start with a multi-input node.
            // However, if it is not the case, the combine is forced to appear as multi-input
            // even if its first stage is not. This is because EOS should be propagated
            // only if all EOSs from previous stages were received (see eosnotify in the combine)
            c->set_multiinput();
        }
        if (c) {
            collector = c;
            if (cleanup) collector_cleanup=true;
        } else 
            collector=(ff_node*)gt;
        collector_removed = false;
        return 0;
    }
    
    /**
     *
     * \brief Sets the feedback channel from the collector to the emitter
     *
     * This method allows to estabilish a feedback channel from the Collector
     * to the Emitter. If the collector is present, than the collector output
     * queue will be connected to the emitter input queue (feedback channel)
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    int wrap_around() {
        if (!this->hasCollector()) { // all stuff are in the prepare method
            if (lb->set_masterworker()<0) return -1;           
            if (!has_input_channel) lb->skipfirstpop(true);
            return 0;
        }
        if (has_input_channel) {
            error("FARM: wrap_around: cannot create feedback if accelerator mode is set, and the collector is present!\n");
            return -1;
        }
        ff_buffernode *tmpbuffer = new ff_buffernode(out_buffer_entries, false);
        assert(tmpbuffer);
        internalSupportNodes.push_back(tmpbuffer);
        if (set_output_buffer(tmpbuffer->get_in_buffer())<0) {
            error("FARM, setting output buffer for multi-input configuration\n");
            return -1;
        }
        if (getCollector() && collector->isMultiOutput())
            collector->set_output_feedback(tmpbuffer);
        
        lb->set_input_feedback(tmpbuffer);
        
        // blocking stuff ------------------------
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        if (!init_input_blocking(m,c)) {
            error("FARM, wrap_around, init input blocking mode for emitter\n");
            return -1;
        }
        set_output_blocking(m,c);
        m=NULL,c=NULL;
        if (!init_output_blocking(m,c)) {
            error("FARM, wrap_around, init output blocking mode for collector\n");
            return -1;
        }
        // ---------------------------------------
        
        lb->skipfirstpop(true);
        return 0;
    }

    /**
     *
     * \brief Removes the collector
     *
     * It allows not to start the collector thread, whereas all worker's output
     * buffer will be created as if it were present.
     *
     * \return 0 is always returned.
     */
    int remove_collector() {
        if (ordered) {
            collector=(ff_node*)gt;
            return 0;
        }
        collector_removed = true;
        return 0;
    }

    inline bool isMultiInput() const { return true;}

    inline bool isMultiOutput() const {
        if (!collector || collector_removed) return true;
        if (collector == (ff_node*)gt) return false;
        return collector->isMultiOutput();
    }

    inline bool isFarm() const { return true; }
    inline bool isOFarm() const { return ordered; }
    inline bool isPrepared() const { return prepared;}
    
    inline bool hasCollector() const {
        return (ordered ? true: (collector && !collector_removed));
    }
            
    bool isset_cleanup_emitter() const { return emitter_cleanup; }
    bool isset_cleanup_workers() const { return worker_cleanup;}
    bool isset_cleanup_collector() const { return collector_cleanup; }
    
    /**
     * \internal
     * \brief Delete workers when the destructor is called.
     *
     */
    void cleanup_workers(bool onoff=true) {
        worker_cleanup = onoff;
    }
    void cleanup_emitter(bool onoff=true) {
        emitter_cleanup = onoff;
    }
    void cleanup_collector(bool onoff=true) {
        collector_cleanup = onoff;
    }
    
    void cleanup_all() {
        worker_cleanup   = true;
        emitter_cleanup  = true;
        collector_cleanup= true;
    }

    virtual void no_barrier() {
        initial_barrier = false;
    }
    virtual void no_mapping() {
        default_mapping = false;
        lb->no_mapping();
        if (gt) gt->no_mapping();
    }
    virtual void blocking_mode(bool blk=true) {
        // NOTE: blocking_mode for workers is managed by the load-balancer
        blocking_in = blocking_out = blk;
        lb->blocking_mode(blk);
        if (gt) gt->blocking_mode(blk);            
    }
    
    inline int cardinality() const { 
        int card=0;
        for(size_t i=0;i<workers.size();++i) 
            card += workers[i]->cardinality();
        
        return (card + 1 + ((collector && !collector_removed)?1:0));
    }

    
    /**
     * \brief Execute the Farm 
     *
     * It executes the farm.
     *
     * \param skip_init A booleon value showing if the initialization should be
     * skipped
     *
     * \return If successful 0, otherwise a negative is returned.
     *
     */
    int run(bool skip_init=false) {
        if (!skip_init) {
#if defined(FF_INITIAL_BARRIER)
            if (initial_barrier) {
                // set the initial value for the barrier 
                if (!barrier)  barrier = new BARRIER_T;
                const int nthreads = cardinality(barrier);
                if (nthreads > MAX_NUM_THREADS) {
                    error("FARM, too much threads, increase MAX_NUM_THREADS !\n");
                    return -1;
                }
                barrier->barrierSetup(nthreads);
            }
#endif
            lb->skipfirstpop(!has_input_channel);
        }
        
        if (!prepared) if (prepare()<0) return -1;

        // starting the emitter node
        if (lb->runlb()<0) {
            error("FARM, running load-balancer module\n");
            return -1;        
        }

        // starting the workers
        if (isfrozen()) {
            for(size_t i=0;i<workers.size();++i) {
                /* set the initial blocking mode
                 */
                assert(blocking_in==blocking_out);
                workers[i]->blocking_mode(blocking_in);
                if (!default_mapping) workers[i]->no_mapping();
                //workers[i]->skipfirstpop(false);
                 if (workers[i]->freeze_and_run(true)<0) {
                    error("FARM, spawning worker thread\n");
                    return -1;
                }      
            }
        } else {
            for(size_t i=0;i<workers.size();++i) {
                /* set the initial blocking mode
                 */
                assert(blocking_in==blocking_out);
                workers[i]->blocking_mode(blocking_in);
                if (!default_mapping) workers[i]->no_mapping();
                //workers[i]->skipfirstpop(false);
                 if (workers[i]->run(true)<0) {
                    error("FARM, spawning worker thread\n");
                    return -1;
                }                      
            }
        }
        // starting the collector node
        if (!collector_removed)
            if (collector && gt->run(true)<0) {
                error("FARM, running gather module\n");
                return -1;
            }
        return 0;
    }

    /** 
     * \brief Executs the farm and wait for workers to complete
     *
     * It executes the farm and waits for all workers to complete their
     * tasks.
     *
     * \return If successful 0, otherwise a negative value is returned.
     */
    virtual int run_and_wait_end() {
        if (isfrozen()) {
            stop();
            thaw();
            if (wait()<0) return -1;
            return 0;
        }
        stop();
        if (run(false)<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }

    /** 
     * \brief Executes the farm and then freeze.
     *
     * It executs the farm and then freezes the farm.
     * If workers are frozen, it is possible to wake up just a subset of them.
     *
     * \return If successful 0, otherwise a negative value
     */
    virtual int run_then_freeze(ssize_t nw=-1) {
        if (isfrozen()) {
            // true means that next time threads are frozen again
            thaw(true, nw); 
            return 0;
        }
        if (!prepared) if (prepare()<0) return -1;
        freeze();
        return run(false);
    }
    
    /** 
     * \brief Puts the thread in waiting state
     *
     * It puts the thread in waiting state.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int wait() {
        int ret=0;
        //if (lb->waitWorkers()<0) ret = -1;
        for(size_t i=0;i<workers.size();++i)
            if (workers[i]->wait()<0) {
                error("FARM, waiting worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        lb->running = -1;
        if (lb->waitlb()<0) ret=-1;
        if (!collector_removed && collector) if (gt->wait()<0) ret=-1;
        return ret;
    }
    int wait_collector() {
        int ret=-1;
        if (!collector_removed && collector) {
            if (gt->wait()<0) ret=-1;
            else ret = 0;
        }
        return ret;
    }
    
    /** 
     * \brief Waits for freezing
     *
     * It waits for thread to freeze.
     *
     * \return 0 if successful otherwise -1 is returned.
     */
    inline int wait_freezing(/* timeval */ ) {
        int ret=0;
        //if (lb->wait_freezingWorkers()<0) ret = -1;
        for(size_t i=0;i<workers.size();++i)
            if (workers[i]->wait_freezing()<0) {
                error("FARM, waiting freezing of worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        lb->running = -1;        
        if (lb->wait_lb_freezing()<0) ret=-1;
        if (!collector_removed && collector) if (gt->wait_freezing()<0) ret=-1;
        return ret; 
    } 

    /** 
     * \internal
     * \brief Forces a thread to stop at the next EOS message.
     *
     * It forces the thread to stop at the next EOS message.
     */
    inline void stop() {
        lb->stop();
        if (collector && !collector_removed) gt->stop();
    }

    /** 
     * \internal
     * \brief Forces the thread to freeze at next EOS.
     *
     * It forces to freeze the farm at next EOS.
     */
    inline void freeze() {
        lb->freeze();
        if (collector && !collector_removed) gt->freeze();
    }

    /**
     * \internal
     * \brief Checks if the Farm has completed the computation.
     *
     * It checks if the farm has completed the computation.
     * 
     *
     * \return true if the pattern is frozen or has terminated the execution.
     */
    inline bool done() const { 
        if (collector && !collector_removed) return (lb->done() && gt->done());
        return lb->done();
    }

    /**
     * \breif Offloads teh task to farm
     *
     * It offloads the given task to the farm.
     *
     * \param task is a void pointer
     * \param retry showing the number of tries to offload
     * \param ticks is the number of ticks to wait
     *
     * \return \p true if successful, otherwise \p false
     */
    inline bool offload(void * task,
                        unsigned long retry=((unsigned long)-1),
                        unsigned long ticks=ff_loadbalancer::TICKS2WAIT) { 
        FFBUFFER * inbuffer = get_in_buffer();

        if (inbuffer) {
            if (blocking_out) {
            _retry:
                const bool empty=inbuffer->empty();
                if (inbuffer->push(task)) {
                    if (empty) pthread_cond_signal(p_cons_c);
                    return true;
                }
                struct timespec tv;
                timedwait_timeout(tv);                
                pthread_mutex_lock(prod_m);
                pthread_cond_timedwait(prod_c, prod_m, &tv);
                pthread_mutex_unlock(prod_m);
                goto _retry;
            }
            for(unsigned long i=0;i<retry;++i) {
                if (inbuffer->push(task)) return true;
                losetime_out(ticks);
            } 
            return false;
        }        
        if (!has_input_channel) 
            error("FARM: accelerator is not set, offload not available");
        else
            error("FARM: input buffer creation failed");
        return false;
    }


    /**
     * \brief Loads results into gatherer
     *
     * It loads the results from the gatherer (if any).
     *
     * \param task is a void pointer
     * \param retry is the number of tries to load the results
     * \param ticks is the number of ticks to wait
     *
     * \return \p false if EOS arrived or too many retries, \p true if  there is a new value
     */
    inline bool load_result(void ** task,
                            unsigned long retry=((unsigned long)-1),
                            unsigned long ticks=ff_gatherer::TICKS2WAIT) {
        if (!collector) {
            error("FARM: load_result: no collector present!!");
            return false;
        }

        if (blocking_in) {
        _retry:
            if (gt->pop_nb(task)) {
                // NOTE: the queue between collector and the main thread is forced to be unbounded
                // therefore the collector cannot be blocked for the condition buffer full ! 
                
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            struct timespec tv;
            timedwait_timeout(tv);
            pthread_mutex_lock(cons_m);
            pthread_cond_timedwait(cons_c, cons_m,&tv);
            pthread_mutex_unlock(cons_m);
            goto _retry;
        }
        for(unsigned long i=0;i<retry;++i) {
            if (gt->pop_nb(task)) {
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            losetime_in(ticks);
        }
        return false;
    }
    /**
     * \brief Loads result with non-blocking
     *
     * It loads the result with non-blocking situation.
     *
     * \param task is a void pointer
     *
     * \return \false if no task is present, otherwise \true if there is a new
     * value. It should be checked if the task has a \p FF_EOS
     *
     */
    inline bool load_result_nb(void ** task) {
        if (!collector) {
            error("FARM: load_result_nb: no collector present!!");
            return false;
        }
        return gt->pop_nb(task);
    }
    
    /**
     * \internal
     * \brief Gets lb (Emitter) node
     *
     * It gets the lb node (Emitter)
     *
     * \return A pointer to the load balancer is returned.
     *
     */
    inline lb_t * getlb() const { return lb;}

    /**
     * \internal
     * \brief Gets gt (Collector) node
     *
     * It gets the gt node (collector)
     *
     * \return A pointer to the gatherer is returned.
     */
    inline gt_t * getgt() const { return gt;}

    /**
     * \internal
     * \brief Gets workers list
     *
     * It gets the list of the workers
     *
     * \return A list of workers is returned.
     */
    const svector<ff_node*>& getWorkers() const { return workers; }


    /**
     * \brief Gets Emitter
     *
     * It returns a pointer to the emitter.
     *
     * \return A pointer of the FastFlow node which is actually the emitter.
     */
    virtual ff_node* getEmitter() const   {
        return emitter;
    }

    /**
     * \brief Gets Collector
     * 
     * It returns a pointer to the collector filter (if present). 
     * It returns \p NULL even if the collector is present and it is the default one.
     * To check the presence of the collector it has to be used @hasCollector
     *
     * \return A pointer to collector node if exists, otherwise a \p NULL
     */
    virtual ff_node* getCollector() const { 
        if (collector == (ff_node*)gt || collector_removed) return nullptr;
        return collector;
    }



    /**
     * \internal
     * \brief Resets input/output queues.
     * 
     *  Warning: resetting queues while the node is running may 
     *           produce unexpected results.
     */
    void reset() {
        if (lb)  lb->reset();
        if (gt)  gt->reset();
        for(size_t i=0;i<workers.size();++i) workers[i]->reset();
    }

    /**
     * \internal
     * \brief Gets the number of workers
     *
     * The number of workers is returned.
     *
     * \return An integet value showing the number of workers.
     */
    size_t getNWorkers() const { return workers.size();}

    /**
     * \internal
     * \brief Returns the node that can produce output.
     * 
     */
    inline void get_out_nodes(svector<ff_node*>&w) {
        if (collector && !collector_removed) {
            if ((ff_node*)gt == collector) {
                assert(gt->get_out_buffer());
                w.push_back(this);                
            } else {
                collector->get_out_nodes(w);
                if (w.size()==0) w.push_back(collector);
            }
            return;
        }
        svector<ff_node*> wtmp;
        for(size_t i=0;i<workers.size();++i)
            workers[i]->get_out_nodes(wtmp);
        if (wtmp.size()==0) w += workers;
        else w += wtmp;
    }

    inline void get_out_nodes_feedback(svector<ff_node*>&w) {
        w += outputNodesFeedback;
    }

    
    inline void get_in_nodes(svector<ff_node*>&w) {
        w.push_back(this);
    }

    /*  WARNING: if these methods are called after prepare (i.e. after having called
     *  run_and_wait_end/run_then_freeze/run/....) they have no effect.     
     *
     */
    void setFixedSize(bool fs) { fixedsizeIN = fixedsizeOUT = fs; }
    void setInputQueueLength(int sz, bool fixedsize)  {
        in_buffer_entries = sz;
        fixedsizeIN       = fixedsize;
    }
    void setOutputQueueLength(int sz, bool fixedsize) {
        out_buffer_entries = sz;
        fixedsizeOUT       = fixedsize;
    }

    int numThreads() const { return cardinality(); }

    /**
     * \internal
     * \brief Gets the starting time
     *
     * It returns the starting time.
     *
     * \return A structure of \p timeval showing the starting time.
     *
     */
    const struct timeval getstarttime() const { return lb->getstarttime();}

    /**
     * \internal
     * \brief Gets the stoping time
     *
     * It returns the structure showing the finishing time. It
     * is the collector then return the finishing time of the farm. otherwise,
     * collects the finishing time in all workers and add them in a vector and
     * then return the vector, showing the collective finishing time of the
     * farm with no collector.
     *
     * \return A \timeval showing the finishing time of the farm.
     */
    const struct timeval  getstoptime()  const {
        if (collector && !collector_removed) return gt->getstoptime();
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers.size()+1,zero);
        for(size_t i=0;i<workers.size();++i)
            workertime[i]=workers[i]->getstoptime();
        workertime[workers.size()]=lb->getstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }

    /**
     * \internal
     * \brief Gets the starting time
     *
     * It returnes the starting time.
     *
     * \return A struct of type timeval showing the starting time.
     */
    const struct timeval  getwstartime() const { return lb->getwstartime(); }    

    /**
     * \internal
     * \brief Gets the finishing time
     *
     * It returns the finishing time if there exists a collector in the farm.
     * If there is no collector, then the finishing time of individual workers
     * is collected in the form of a vector and return that vector.
     *
     * \return The vector showing the finishing time.
     */
    const struct timeval  getwstoptime() const {
        if (collector && !collector_removed) return gt->getwstoptime();
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers.size()+1,zero);
        for(size_t i=0;i<workers.size();++i) {
            workertime[i]=workers[i]->getwstoptime();
        }
        workertime[workers.size()]=lb->getwstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    /**
     * \internal
     * \brief Gets the time spent in \p svc_init
     *
     * The returned time comprises the time spent in \p svc_init and in \p
     * svc_end methods.
     *
     * \return A double value showing the time taken in \p svc_init
     */
    double ffTime() {
        if (collector && !collector_removed)
            return diffmsec(gt->getstoptime(), lb->getstarttime());

        return diffmsec(getstoptime(),lb->getstarttime());
    }

    /**
     * \internal
     * \brief Gets the time spent in \p svc
     *
     * The returned time considers only the time spent in the svc methods.
     *
     * \return A double value showing the time taken in \p svc.
     */
    double ffwTime() {
        if (collector && !collector_removed)
            return diffmsec(gt->getwstoptime(), lb->getwstartime());

        return diffmsec(getwstoptime(),lb->getwstartime());
    }

#ifdef DFF_ENABLED
    virtual bool isSerializable(){ 
        svector<ff_node*> outputs; this->get_out_nodes(outputs);
        for(ff_node* output: outputs) if (!output->isSerializable()) return false;
        return true;
    }

    virtual bool isDeserializable(){ 
        svector<ff_node*> inputs; this->get_in_nodes(inputs);
        for(ff_node* input: inputs) if(!input->isDeserializable()) return false;
        return true; 
    }
#endif

    
#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- farm:\n";
        lb->ffStats(out);
        for(size_t i=0;i<workers.size();++i) workers[i]->ffStats(out);
        if (collector && !collector_removed) gt->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

protected:

    /**
     * \brief svc method
     */
    void* svc(void *) { return NULL; }

    /**
     * \brief The svc_init method
     */
    int svc_init()       { return -1; };

    /**
     * \brief The svc_end method
     */
    void svc_end()        {}

    ssize_t get_my_id() const { return -1; };


    void setAffinity(int) { 
        error("FARM, setAffinity: cannot set affinity for the farm\n");
    }

    int getCPUId() const { return -1;}

    /**
     * \internal
     * \brief Thaws the thread
     *
     * If the thread is frozen, then thaw it. 
     */
    inline void thaw(bool _freeze=false, ssize_t nw=-1) {
        lb->thaw(_freeze, nw);
        if (collector && !collector_removed) gt->thaw(_freeze, nw);
    }

    /**
     * \internal
     * \brief Checks if the Farm is frozen
     *
     * It checks if the farm is frozen.
     *
     * \return The status of \p isfrozen().
     */
    inline bool isfrozen() const { return lb->isfrozen(); }


    /** 
     *  \brief Creates the input buffer for the emitter node
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  It creates an input buffer for the Emitter node. 
     *
     *  \param nentries the size of the buffer
     *  \param fixedsize flag to decide whether the buffer is resizable. 
     *
     *  \return If successful 0, otherwsie a negative value.
     */
    int create_input_buffer(int nentries, bool fixedsize) {
        if (in) {
            error("FARM create_input_buffer, buffer already present\n");
            return -1;
        }
        if (emitter) {
            if (emitter->create_input_buffer(nentries,fixedsize)<0) return -1;
            if (emitter->isMultiInput()) {
                if (emitter->isComp()) 
                    in = emitter->get_in_buffer();
                else {
                    svector<ff_node*> w(1);
                    emitter->get_in_nodes(w);
                    assert(w.size()==1);
                    in = w[0]->get_in_buffer();
                }
            } else  in = emitter->get_in_buffer();
        } else {
            if (ff_node::create_input_buffer(nentries, fixedsize)<0) return -1;
        }
        lb->set_in_buffer(in);
        return 0;
    }
    
    /**
     * \internal
     * \brief Creates the output buffer for the collector
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  It create an output buffer for the Collector
     *
     *  \param nentries the size of the buffer
     *  \param fixedsize flag to decide whether the buffer is resizable. 
     *  Default is \p false
     *
     *  \return If successful 0, otherwise a negative value.
     */
    int create_output_buffer(int nentries, bool fixedsize=false) {
        if (out) {
            error("FARM create_output_buffer, buffer already present\n");
            return -1;
        }

        if (!this->hasCollector()) {
            size_t nworkers = workers.size();
            assert(nworkers>0);
            
            // check to see if workers' output buffer has been already created 
            if (workers[0]->get_out_buffer() == NULL) {

                // We can be here because we are in a pipeline and the next stage
                // is a multi-input stage. If the farm is a masterworker or if the node 
                // has multiple output then we are a multi-output node and thus all channels
                // have to be registered as output channels for the worker.

                for(size_t i=0;i<nworkers;++i) {
                    if (workers[i]->create_output_buffer(out_buffer_entries,fixedsize)<0)   return -1;
                }
            }
            return 0;
        }
        
        if (ff_node::create_output_buffer(nentries, fixedsize)<0) return -1;        
        if (gt->set_output_buffer(this->get_out_buffer())<0) return -1;
        if (collector && !collector_removed) {
            if (collector != (ff_node*)gt)
                collector->set_output_buffer(this->get_out_buffer());
        }

        return 0;
    }

    /**
     * \internal
     * \brief Sets multiple input nodes
     *
     * It sets multiple inputs to the node.
     *
     *
     * \return The status of \p set_input(x) otherwise -1 is returned.
     */
    inline int set_input(const svector<ff_node *> & w) { 
        return lb->set_input(w);
    }

    inline int set_input(ff_node *node) { 
        return lb->set_input(node);
    }

    inline int set_input_feedback(ff_node *node) { 
        return lb->set_input_feedback(node);
    }

    inline int set_output(const svector<ff_node *> & w) {
        if (collector && !collector_removed) {
            if (collector != (ff_node*)gt)
                if (collector->isMultiOutput()) {
                    collector->set_output(w);
                    return 0;
                }
            error("FARM, cannot add output nodes, the collector is not multi-output\n");
            return -1;
        }
        if (outputNodes.size()+w.size() > workers.size()) {
            return -1;
        }
        outputNodes +=w;
        return 0; 
    }
    inline int set_output(ff_node *node) {

        if (collector && !collector_removed) {
            if (collector != (ff_node*)gt)
                if (collector->isMultiOutput()) {
                    collector->set_output(node);
                    return 0;
                }
            error("FARM, cannot add output node\n");
            return -1;
        }
        svector<ff_node*> w(1);
        this->get_out_nodes(w);
        if (outputNodes.size()+1 > w.size()) {
            return -1;
        }
        outputNodes.push_back(node);
        return 0; 
    }

    inline int set_output_feedback(ff_node *node) { 
        outputNodesFeedback.push_back(node); 
        return 0;
    }

    
    /**
     *
     *  \brief Sets the output buffer of the collector 
     *
     *  This function redefines the ff_node's virtual method of the same name.
     *  Set the output buffer for the Collector.
     *
     *  \param o a buffer object, which can be of type \p SWSR_Ptr_Buffer or 
     *  \p uSWSR_Ptr_Buffer
     *
     *  \return 0 if successful, otherwise -1 is returned.
     */
    int set_output_buffer(FFBUFFER * const o) {
        if (!collector && !collector_removed) {
            error("FARM with no collector, cannot set output buffer\n");
            return -1;
        }
        if (gt->set_output_buffer(o)<0) return -1;
        if (this->getCollector()) collector->set_output_buffer(o);
        return 0;
    }

protected:
    bool has_input_channel; // for the accelerator mode
    bool collector_removed;
    bool ordered;          
    bool fixedsizeIN, fixedsizeOUT;
    bool myownlb,myowngt;
    bool worker_cleanup, emitter_cleanup,collector_cleanup;
    
    int ondemand;          // if >0, emulates on-demand scheduling
    int in_buffer_entries;
    int out_buffer_entries;
    size_t max_nworkers;
    size_t ordering_memsize;
    
    ff_node          *  emitter;
    ff_node          *  collector;

    lb_t             * lb;
    gt_t             * gt;
    svector<ff_node*>  workers;
    svector<ff_node*>  outputNodes;
    svector<ff_node*>  outputNodesFeedback;       
    svector<ff_node*>  internalSupportNodes;
    svector<ordering_pair_t>  ordering_Memory;     // used for ordering purposes
};






#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))

#include <ff/make_unique.hpp>

/*
 * \class ff_Farm
 * \ingroup  high_level_patterns
 *
 *  \brief The Farm pattern.
 */    
template<typename IN_t=char, typename OUT_t=IN_t>
class ff_Farm: public ff_farm {
protected:
    // unique_ptr based data
    std::vector<std::unique_ptr<ff_node> > Workers;
    std::unique_ptr<ff_node>               Emitter;
    std::unique_ptr<ff_node>               Collector;
public:    
    typedef IN_t  in_type;
    typedef OUT_t out_type;

    // NOTE: the ownership of the ff_node (unique) pointers is transferred to the farm !!!!
    //       All workers, the Emitter and the Collector will be deleted in the ff_Farm destructor !

    ff_Farm(std::vector<std::unique_ptr<ff_node> > &&W,
            std::unique_ptr<ff_node> E  =std::unique_ptr<ff_node>(nullptr), 
            std::unique_ptr<ff_node> C  =std::unique_ptr<ff_node>(nullptr), 
            bool input_ch=false): 
        ff_farm(input_ch,DEFAULT_BUFFER_CAPACITY,DEFAULT_BUFFER_CAPACITY,false), 
        Workers(std::move(W)), Emitter(std::move(E)), Collector(std::move(C)) { 

        const size_t nw = Workers.size();
        assert(nw>0);
        std::vector<ff_node*> w(nw);        
        for(size_t i=0;i<nw;++i) w[i]= Workers[i].get(); 
        ff_farm::add_workers(w);

        // add default collector even if Collector is NULL, 
        // if you don't want the collector you have to call remove_collector
        ff_farm::add_collector(Collector.get());
        ff_node *e = Emitter.get();
        if (e) ff_farm::add_emitter(e);         
    }

    ff_Farm(std::vector<std::unique_ptr<ff_node> > &&W,
            ff_node &E, ff_node &C, 
            bool input_ch=false):
        ff_farm(input_ch,DEFAULT_BUFFER_CAPACITY,DEFAULT_BUFFER_CAPACITY,false),
        Workers(std::move(W)) {

        const size_t nw = Workers.size();
        assert(nw>0);
        std::vector<ff_node*> w(nw);        
        for(size_t i=0;i<nw;++i) w[i]=Workers[i].get();
        ff_farm::add_workers(w);

        ff_farm::add_collector(&C);
        ff_farm::add_emitter(&E); 
    }
    ff_Farm(std::vector<std::unique_ptr<ff_node> > &&W,  
            ff_node &E, bool input_ch=false):
        ff_farm(input_ch,DEFAULT_BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY,false),
        Workers(std::move(W)) {

        const size_t nw = Workers.size();
        assert(nw>0);
        std::vector<ff_node*> w(nw);        
        for(size_t i=0;i<nw;++i) w[i]=Workers[i].get();
        ff_farm::add_workers(w);

        ff_farm::add_collector(nullptr);
        ff_farm::add_emitter(&E); 
    }

    ff_Farm(std::vector<std::unique_ptr<ff_node> > &&W, bool input_ch):
        ff_Farm(std::move(W), std::unique_ptr<ff_node>(nullptr), 
                std::unique_ptr<ff_node>(nullptr), input_ch) {
    }
   
    /* copy constructor */
    ff_Farm(const ff_Farm<IN_t, OUT_t> &f): ff_farm(f) {
    }

    
    /* move constructor */
    ff_Farm(ff_Farm<IN_t, OUT_t> &&f):ff_farm(std::move(f)) {

        Workers = std::move(f.Workers);
        Emitter = std::move(f.Emitter);
        Collector = std::move(f.Collector);

        f.worker_cleanup    = false;
        f.emitter_cleanup   = false;
        f.collector_cleanup = false;
    }


    /* --- */

    template <typename FUNC_t>
    explicit ff_Farm(FUNC_t F, ssize_t nw, bool input_ch=false): 
        ff_farm(input_ch,DEFAULT_BUFFER_CAPACITY,DEFAULT_BUFFER_CAPACITY,
                  true, nw) {

        std::vector<ff_node*> w(nw);        
        for(int i=0;i<nw;++i) w[i]=new ff_node_F<IN_t,OUT_t>(F);
        ff_farm::add_workers(w);
        ff_farm::add_collector(NULL);

        ff_farm::cleanup_workers();  
    }

    virtual ~ff_Farm() { }

    int add_emitter(ff_node &e) {
        int r =ff_farm::add_emitter(&e);
        if (r>=0) emitter_cleanup=false;
        return r;
    }
    int add_collector(ff_node &c) {
        ff_farm::remove_collector();
        int r=ff_farm::add_collector(&c);
        if (r>=0) collector_cleanup=false;
        ff_farm::collector_removed = false;
        return r;
    }

    bool load_result(OUT_t *&task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_gatherer::TICKS2WAIT) {
        return ff_farm::load_result((void**)&task, retry,ticks);
    }
    bool load_result_nb(OUT_t *&r) {
        return ff_farm::load_result_nb((void**)&r);
    }    

    // ------------------- deleted method --------------------------------- 
    int add_workers(std::vector<ff_node *> & w)                   = delete;
    int add_emitter(ff_node * e)                                  = delete;
    int add_collector(ff_node * c)                                = delete;    
    bool load_result(void ** task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_gatherer::TICKS2WAIT) = delete;
    void cleanup_workers()                                        = delete;
    void cleanup_all()                                            = delete;
    bool load_result_nb(void ** task)                             = delete;
};


    
/*
 * \class ff_Farm
 * \ingroup  high_level_patterns
 *
 * \brief The ordered Farm pattern.
 *
 * Ordered task-farm pattern based on ff_farm building-block
 *
 */
template<typename IN_t=char, typename OUT_t=IN_t>
class ff_OFarm: public ff_farm {
protected:
    // unique_ptr based data
    std::vector<std::unique_ptr<ff_node> > Workers;
public:    
    typedef IN_t  in_type;
    typedef OUT_t out_type;

    ff_OFarm(std::vector<std::unique_ptr<ff_node> > &&W,  bool input_ch=false): 
        ff_farm(input_ch, DEFAULT_BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY,false,W.size()), 
        Workers(std::move(W)) { 
        assert(Workers.size());
        const size_t nw = Workers.size();
        assert(nw>0);
        std::vector<ff_node*> w(nw);        
        for(size_t i=0;i<nw;++i) w[i]= Workers[i].get(); 
        ff_farm::add_workers(w);
        this->set_ordered();
        ff_farm::add_collector(nullptr);
    }

    template <typename FUNC_t>
    explicit ff_OFarm(FUNC_t F, size_t nw, bool input_ch=false): 
        ff_farm(input_ch,DEFAULT_BUFFER_CAPACITY,DEFAULT_BUFFER_CAPACITY,false,nw) {
        if (Workers.size()>0) {
            error("OFARM: workers already added\n");
            return;
        }
        assert(nw>0);
        for(size_t i=0;i<nw;++i)
            Workers.push_back(make_unique<ff_node_F<IN_t,OUT_t>>(F));
        std::vector<ff_node*> w(nw);        
        for(size_t i=0;i<nw;++i) w[i]= Workers[i].get(); 
        ff_farm::add_workers(w);        
        this->set_ordered();
        ff_farm::add_collector(nullptr);
    }

    virtual ~ff_OFarm() { }

    int add_emitter(ff_node &e) {
        int r =ff_farm::add_emitter(&e);
        if (r>=0) emitter_cleanup=false;
        return r;
    }
    int add_collector(ff_node &c) {
        ff_farm::remove_collector();
        int r=ff_farm::add_collector(&c);
        if (r>=0) collector_cleanup=false;
        ff_farm::collector_removed = false;
        return r;
    }


    
    int add_emitter(ff_node * e)  = delete;
    int add_collector(ff_node * c) = delete;

    int add_workers(std::vector<ff_node *> & w) = delete;
    int remove_collector()                      = delete;
    void cleanup_all()                          = delete;

    bool load_result(void ** task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_gatherer::TICKS2WAIT) = delete;
    bool load_result(OUT_t *&task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_gatherer::TICKS2WAIT) {
        return ff_farm::load_result((void**)&task, retry,ticks);
    }

    bool load_result_nb(void ** task) = delete;
    bool load_result_nb(OUT_t *&r) {
        return ff_farm::load_result_nb((void**)&r);
    }
};
#endif

} // namespace ff


#endif /* FF_FARM_HPP */
