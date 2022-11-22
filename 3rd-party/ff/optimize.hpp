/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file optimize.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow optimization heuristics 
 *
 * @detail FastFlow basic container for a shared-memory parallel activity 
 *
 */

#ifndef FF_OPTIMIZE_HPP
#define FF_OPTIMIZE_HPP

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
 *   Author: Massimo Torquati
 *      
 */
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/all2all.hpp>
#include <ff/combine.hpp>

namespace ff {

typedef enum { OPT_NORMAL = 1, OPT_INFO = 2 } reportkind_t;
static inline void opt_report(int verbose_level, reportkind_t kind, const char *str, ...) {
    if (verbose_level < kind) return;
    va_list argp;
    char * p=(char *)malloc(strlen(str)+512); // this is dangerous.....
    assert(p);
    strcpy(p, str);
    va_start(argp, str);
    vfprintf(stdout, p, argp);
    va_end(argp);
    free(p);
}

/**
 *  This function looks for internal farms with default collector in a farm building block.
 *  The internal default collectors are removed.
 */    
static inline int remove_internal_collectors(ff_farm& farm) {
    const svector<ff_node*>& W = farm.getWorkers();
    for(size_t i=0;i<W.size();++i) {
        if (W[i]->isFarm() && !W[i]->isOFarm()) {
            ff_farm* ifarm = reinterpret_cast<ff_farm*>(W[i]);            
            if (remove_internal_collectors(*ifarm)<0) return -1;
            if (ifarm->getCollector() == nullptr)
                ifarm->remove_collector();
        } else {
            if (W[i]->isPipe()) {
                ff_pipeline* ipipe = reinterpret_cast<ff_pipeline*>(W[i]);
                OptLevel iopt;
                iopt.remove_collector=true;
                if (optimize_static(*ipipe, iopt)<0) return -1;
            }
            if (W[i]->isAll2All()) {
                ff_a2a *a2a   = reinterpret_cast<ff_a2a*>(W[i]);
                const svector<ff_node*>& W1 = a2a->getFirstSet();
                const svector<ff_node*>& W2 = a2a->getSecondSet();
                for(size_t j=0;j<W1.size();++j) {
                    if (W1[j]->isPipe()) {
                        ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W1[j]);
                        OptLevel iopt;
                        iopt.remove_collector=true;
                        if (optimize_static(*ipipe, iopt)<0) return -1;
                    }
                }
                for(size_t j=0;j<W2.size();++j) {
                    if (W2[j]->isPipe()) {
                        ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W2[j]);
                        OptLevel iopt;
                        iopt.remove_collector=true;
                        if (optimize_static(*ipipe, iopt)<0) return -1;
                    }
                }                                           
            }
        }
    }
    return 0;
}

/**
 * It combines the node passed as second parameter with the farm's emitter. 
 * The node is added at the left-hand side of the emitter.
 * This transformation is logically equivalent to the following pipeline: ff_Pipe<> pipe(node, farm);
 */
static inline int combine_with_emitter(ff_farm& farm, ff_node*node, bool cleanup_node=false) {
    if (node->isFarm() || node->isPipe() || node->isAll2All()) {
        error("combine_with_emitter: the node to combine cannot be a parallel building block\n");
        return -1;
    }
    ff_node* emitter = farm.getEmitter();
    if (!emitter) {        
        farm.add_emitter(node);
        farm.cleanup_emitter(cleanup_node);
        return 0;
    }
    ff_comb* comb;

    if (!emitter->isMultiOutput()) {
        internal_mo_transformer *mo_emitter = new internal_mo_transformer(emitter,
                                                                          farm.isset_cleanup_emitter());
        comb = new ff_comb(node, mo_emitter, cleanup_node, true);
    } else {    
        comb = new ff_comb(node,emitter,
                           cleanup_node, farm.isset_cleanup_emitter());

    }
    if (farm.isset_cleanup_emitter())
        farm.cleanup_emitter(false);
    
    farm.change_emitter(comb, true);
    return 0;
}

/*
 * It combines the node passed as parameter with the farm's collector. 
 * The node is added at the right-hand side of the collector.
 * This transformation is logically equivalent to the following pipeline: ff_Pipe<> pipe(farm, node);
 */
static inline int combine_with_collector(ff_farm& farm, ff_node*node, bool cleanup_node=false) {
    if (!farm.getCollector()) {
        error("combine_with_collector: the farm passed as parameter does not have a collector\n");
        return -1;
    }
    if (node->isFarm() || node->isPipe() || node->isAll2All()) {
        error("combine_with_emitter: the node to combine cannot be a parallel building block\n");
        return -1;
    }
    ff_node* collector = farm.getCollector();
    ff_comb* comb    = new ff_comb(collector, node,
                                   farm.isset_cleanup_collector(), cleanup_node);
    if (farm.isset_cleanup_collector())
        farm.cleanup_collector(false);
    farm.remove_collector();
    farm.add_collector(comb, true);
    return 0;
}

/*
 * It combines the node passed as parameter with the first stage of the pipeline. 
 * The node is added at the left-hand side of the first pipeline node.
 * This transformation is logically equivalent to the following pipeline: ff_Pipe<> pipe2(node, pipe);
 */
template<typename T>
static inline int combine_with_firststage(ff_pipeline& pipe, T* node, bool cleanup_node) {
    pipe.flatten();
    ff_node* node0 = pipe.get_node(0); // it cannot be a pipeline
    if (!node0) {
        error("combine_with_firststage: empty pipeline\n");
        return -1;
    }
    if (node0->isAll2All()) {  
        return combine_left_with_a2a(*reinterpret_cast<ff_a2a*>(node0), node, cleanup_node);
    }
    if (node0->isFarm()) {
        ff_farm &farm=*(ff_farm*)node0;
        if (combine_with_emitter(farm, node, cleanup_node)<0) return -1;
        pipe.remove_stage(0);        
        pipe.insert_stage(0, node0);        
    } else {
        ff_comb* comb = new ff_comb(node, node0, cleanup_node);
        pipe.remove_stage(0);
        pipe.insert_stage(0, comb, true);
    }
    
    return 0;
}


template<typename T> 
static inline int combine_right_with_farm(ff_farm& farm, T* node, bool cleanup_node) {
    if (farm.hasCollector())
        return combine_with_collector(farm, node, cleanup_node);

    // farm with no collector
    const svector<ff_node*>& w= farm.getWorkers();
    assert(w.size()>0);

    if (w[0]->isPipe()) { // NOTE: we suppose that all workers are homogeneous
        if (combine_with_laststage(*reinterpret_cast<ff_pipeline*>(w[0]), node, cleanup_node)<0) return -1;
        int r=0;
        for(size_t i=1;i<w.size();++i) {
            ff_pipeline* pipe = reinterpret_cast<ff_pipeline*>(w[i]);
            r+=combine_with_laststage(*pipe, new T(*node), true);
        }        
        return (r>0?-1:0);
    }
    if (w[0]->isFarm()) {
        if (combine_right_with_farm(*reinterpret_cast<ff_farm*>(w[0]), node, cleanup_node)<0) return -1;
        int r=0;
        for(size_t i=1;i<w.size();++i) {
            ff_farm* farm = reinterpret_cast<ff_farm*>(w[i]);
            r+=combine_right_with_farm(*farm, new T(*node), true);
        }        
        return (r>0?-1:0);
    }
    if (w[0]->isAll2All()) {
        if (combine_right_with_a2a(*reinterpret_cast<ff_a2a*>(w[0]), node, cleanup_node)<0) return -1;
        int r=0;
        for(size_t i=1;i<w.size();++i) {
            ff_a2a* a2a = reinterpret_cast<ff_a2a*>(w[i]);
            r+=combine_right_with_a2a(*a2a, new T(*node), true);
        }        
        return (r>0?-1:0);
    }
    bool workers_cleanup = farm.isset_cleanup_workers();
    std::vector<ff_node*> new_workers;
    
    ff_comb* comb = new ff_comb(w[0], node, workers_cleanup, cleanup_node);
    assert(comb);
    new_workers.push_back(comb);
    for(size_t i=1;i<w.size();++i) {
        ff_comb* c = new ff_comb(w[i], new T(*node), workers_cleanup, true);
        assert(c);
        new_workers.push_back(c);
    }
    farm.change_workers(new_workers);    
    return 0;
}
template<typename T> 
static inline int combine_right_with_a2a(ff_a2a& a2a, T* node, bool cleanup_node) {
    const svector<ff_node*>& w= a2a.getSecondSet();
    if (w[0]->isPipe()) { // NOTE: we suppose that all workers are homogeneous
        if (combine_with_laststage(*reinterpret_cast<ff_pipeline*>(w[0]), node, cleanup_node)<0) return -1;
        int r=0;
        for(size_t i=1;i<w.size();++i) {
            assert(w[i]->isPipe());
            ff_pipeline* pipe = reinterpret_cast<ff_pipeline*>(w[i]);
            r+=combine_with_laststage(*pipe, new T(*node), true);
        }        
        return (r>0?-1:0); 
    }
    std::vector<ff_node*> new_secondset;
    
    ff_comb* comb = new ff_comb(w[0], node, false, cleanup_node);
    assert(comb);
    new_secondset.push_back(comb);
    for(size_t i=1;i<w.size();++i) {
        ff_comb* c = new ff_comb(w[i], new T(*node), false, true);
        assert(c);
        new_secondset.push_back(c);
    }
    a2a.change_secondset(new_secondset, true); 
    return 0;
}
template<typename T> 
static inline int combine_left_with_a2a(ff_a2a& a2a, T* node, bool cleanup_node) {
    const svector<ff_node*>& w= a2a.getFirstSet();
    if (w[0]->isPipe()) { // NOTE: we suppose that all workers are homogeneous

        if (combine_with_firststage(*reinterpret_cast<ff_pipeline*>(w[0]), node, cleanup_node)<0)
            return -1;
        int r=0;
        for(size_t i=1;i<w.size();++i) {
            assert(w[i]->isPipe());
            ff_pipeline* pipe = reinterpret_cast<ff_pipeline*>(w[i]);
            r+=combine_with_firststage(*pipe, new T(*node), true);
        }        
        return (r>0?-1:0); 
    }
    std::vector<ff_node*> new_firstset;    
    ff_comb* comb = new ff_comb(node, w[0], cleanup_node, false);
    assert(comb);
    new_firstset.push_back(comb);
    for(size_t i=1;i<w.size();++i) {
        ff_comb* c = new ff_comb(new T(*node), w[i], true, false);
        assert(c);
        new_firstset.push_back(c);
    }
    a2a.change_firstset(new_firstset, a2a.ondemand_buffer(), true); 
    return 0;
}
    
/*
 * It combines the node passed as second parameter with the last stage of the pipeline. 
 * The node is added at the right-hand side of the last pipeline stage.
 * This transformation is logically equivalent to the following pipeline: ff_Pipe<> pipe2(pipe, node);
 */
template<typename T>
static inline int combine_with_laststage(ff_pipeline& pipe, T* node, bool cleanup_node) {    
    pipe.flatten();
    ff_node* last = pipe.get_lastnode(); // it cannot be a pipeline
    if (!last) {
        error("combine_with_laststage: empty pipeline\n");
        return -1;
    }
    if (last->isAll2All()) {
        return combine_right_with_a2a(*reinterpret_cast<ff_a2a*>(last), node, cleanup_node);
    }
    if (last->isFarm()) {
        return combine_right_with_farm(*reinterpret_cast<ff_farm*>(last), node, cleanup_node);
    }
    bool node_cleanup = pipe.isset_cleanup_nodes();
    int nstages=static_cast<int>(pipe.nodes_list.size());    
    ff_comb* comb = new ff_comb(last, node, node_cleanup , cleanup_node);
    pipe.remove_stage(nstages-1);
    pipe.insert_stage((nstages-1)>0?(nstages-1):0, comb, true);        
    return 0;
}
    
    
/* This is farm specific. 
 *  - It basically sets the threshold for enabling blocking mode.
 *  - It can remove the collector of internal farms in a farm of farms composition.
 *  - TODO: Farm of farms ---> single farm with two emitters combined (external+internal)
 */    
static inline int optimize_static(ff_farm& farm, const OptLevel& opt=OptLevel1()) {
    if (farm.prepared) {
        error("optimize_static (farm) called after prepare\n");
        return -1;
    }
    // optimizing internal pipelines, if any
    OptLevel iopt(opt);
    iopt.blocking_mode      = false;
    iopt.no_initial_barrier = false;
    iopt.no_default_mapping = false;
    const svector<ff_node*> &Workers = farm.getWorkers();
    for(size_t i=0;i<Workers.size();++i) {
        if (Workers[i]->isPipe()) {
            opt_report(opt.verbose_level, OPT_NORMAL,
                       "OPT (farm): Looking for optimizations in the internal pipeline %ld\n",i);

            ff_pipeline *ipipe = reinterpret_cast<ff_pipeline*>(Workers[i]);
            if (optimize_static(*ipipe, iopt)) return -1;
        }
    }

    // here it looks for internal farms with null collectors
    if (opt.remove_collector) {
        auto optimize_internal_farm = [opt](ff_farm& ifarm) {
            OptLevel iopt;
            iopt.remove_collector=true;
            iopt.verbose_level = opt.verbose_level;
            if (optimize_static(ifarm, iopt)<0) return -1;
            if (ifarm.getCollector() == nullptr) {
                opt_report(opt.verbose_level, OPT_NORMAL, "OPT (farm): REMOVE_COLLECTOR: Removed farm collector\n");
                ifarm.remove_collector();
            }
            return 0;
        };
        
        const svector<ff_node*>& W = farm.getWorkers();
        for(size_t i=0;i<W.size();++i) {
            if (W[i]->isFarm() && !W[i]->isOFarm()) {
                ff_farm* ifarm = reinterpret_cast<ff_farm*>(W[i]);
                opt_report(opt.verbose_level, OPT_NORMAL,
                           "OPT (farm): Looking for optimizations in the internal farm %ld\n",i);
                
                if (optimize_internal_farm(*ifarm)<0) return -1;
            } else {
                if (W[i]->isPipe()) {
                    ff_pipeline* ipipe = reinterpret_cast<ff_pipeline*>(W[i]);
                    OptLevel iopt;
                    iopt.remove_collector=true;
                    iopt.verbose_level = opt.verbose_level;
                    if (optimize_static(*ipipe, iopt)<0) return -1;


#if 0                    
                    ff_node* last  = ipipe->get_lastnode();
                    if (last->isFarm() && !last->isOFarm()) {
                        ff_farm* ifarm = reinterpret_cast<ff_farm*>(last);

                        opt_report(opt.verbose_level, OPT_NORMAL,
                                   "OPT (farm): Looking for optimizations in the internal last farm of a pipeline %ld\n",i);

                        if (optimize_internal_farm(*ifarm)<0) return -1;
                    }
#endif                    
                }
                if (W[i]->isAll2All()) {
                    ff_a2a *a2a   = reinterpret_cast<ff_a2a*>(W[i]);
                    const svector<ff_node*>& W1 = a2a->getFirstSet();
                    const svector<ff_node*>& W2 = a2a->getSecondSet();
                    for(size_t j=0;j<W1.size();++j) {
                        if (W1[j]->isPipe()) {
                            ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W1[j]);
                            OptLevel iopt;
                            iopt.remove_collector=true;
                            iopt.verbose_level = opt.verbose_level;
                            if (optimize_static(*ipipe, iopt)<0) return -1;
                        }
                    }
                    for(size_t j=0;j<W2.size();++j) {
                        if (W2[j]->isPipe()) {
                            ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W2[j]);
                            OptLevel iopt;
                            iopt.remove_collector=true;
                            iopt.verbose_level = opt.verbose_level;
                            if (optimize_static(*ipipe, iopt)<0) return -1;
                        }
                    }                                           
                }
            }
        }
    }

    // swithing to blocking mode if the n. of threads is greater than the threshold
    if (opt.blocking_mode) {
        ssize_t card = farm.cardinality();
        if (opt.max_nb_threads < card) {
            opt_report(opt.verbose_level, OPT_NORMAL,
                       "OPT (farm): BLOCKING_MODE: Activating blocking mode, threshold=%ld, number of threads=%ld\n",opt.max_nb_threads, card);
            
            farm.blocking_mode(true);
        }
    }

    // turning off initial/default mapping if the n. of threads is greater than the threshold
    if (opt.no_default_mapping) {
        ssize_t card = farm.cardinality();
        if (opt.max_mapped_threads < card) {
            opt_report(opt.verbose_level, OPT_NORMAL,
                       "OPT (farm): MAPPING: Disabling mapping, threshold=%ld, number of threads=%ld\n",opt.max_mapped_threads, card);
            
            farm.no_mapping();
        }
    }
    // no initial barrier
    if (opt.no_initial_barrier) {
        opt_report(opt.verbose_level, OPT_NORMAL,
                   "OPT (farm): NO_INITIAL_BARRIER: Initial barrier disabled\n");
        farm.no_barrier();
   }
    return 0;
}
    
/* 
 *
 */    
static inline int optimize_static(ff_pipeline& pipe, const OptLevel& opt=OptLevel1()) {
    if (pipe.prepared) {
        error("optimize_static (pipeline) called after prepare\n");
        return -1;
    }

   // flattening the pipeline 
    pipe.flatten();
    int nstages=static_cast<int>(pipe.nodes_list.size());    

   // looking for farm and all-to-all because they might have pipeline inside
   // for each nested pipeline the optimize_pipeline function is recursively
   // called following a depth-first search

   OptLevel iopt(opt);
   iopt.blocking_mode      = false;
   iopt.no_initial_barrier = false;
   iopt.no_default_mapping = false;
   for(int i=0;i<nstages;++i) {
       if (pipe.nodes_list[i]->isFarm()) {
           ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[i]);
           const svector<ff_node*>& W = farm->getWorkers();
           for(size_t j=0;j<W.size();++j) {
               if (W[j]->isPipe()) {
                   ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W[j]);
                   if (optimize_static(*ipipe, iopt)) return -1;
               }
           }
       } else if (pipe.nodes_list[i]->isAll2All()) {
           ff_a2a *a2a   = reinterpret_cast<ff_a2a*>(pipe.nodes_list[i]);
           const svector<ff_node*>& W1 = a2a->getFirstSet();
           const svector<ff_node*>& W2 = a2a->getSecondSet();
           for(size_t j=0;j<W1.size();++j) {
               if (W1[j]->isPipe()) {
                   ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W1[j]);
                   if (optimize_static(*ipipe, iopt)) return -1;
               }
           }
           for(size_t j=0;j<W2.size();++j) {
               if (W2[j]->isPipe()) {
                   ff_pipeline* ipipe=reinterpret_cast<ff_pipeline*>(W2[j]);
                   if (optimize_static(*ipipe, iopt)) return -1;
               }
           }           
       }
   }
   
   // ------------------ helping function ----------------------
   auto find_farm_with_null_collector =
       [](const svector<ff_node*>& nodeslist, int start=0)->int {
       for(int i=start;i<(int)(nodeslist.size());++i) {
           if (nodeslist[i]->isFarm()) {
               ff_farm *farm = reinterpret_cast<ff_farm*>(nodeslist[i]);
               if (farm->getCollector() == nullptr)  return i;
           }
       }
       return -1;
   };
   auto find_farm_with_null_emitter =
       [](const svector<ff_node*>& nodeslist, int start=0)->int {
       for(int i=start;i<(int)(nodeslist.size());++i) {
           if (nodeslist[i]->isFarm() &&
               (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getEmitter()) 
               ) 
               return i;
       }
       return -1;
   };
   // looking for the longest sequence of farms (or ofarms) with the same number of workers
   auto farm_sequence =
       [&](const svector<ff_node*>& nodeslist, int& first_farm, int& last_farm) {
       bool ofarm = nodeslist[first_farm]->isOFarm();
       size_t nworkers = (reinterpret_cast<ff_farm*>(nodeslist[first_farm]))->getNWorkers();
       int starting_point=first_farm+1;
       int first = first_farm, last = last_farm;
       while (starting_point<static_cast<int>(nodeslist.size()))  {
           bool ok=true;
           int next = find_farm_with_null_emitter(nodeslist, starting_point);
           if (next == -1) break;
           else {
               for(int i=starting_point; i<=next;) {
                   if ((ofarm?nodeslist[i]->isOFarm():!nodeslist[i]->isOFarm()) &&
                       (nworkers == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getNWorkers())  &&
                       (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getCollector()) &&
                       (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getEmitter())
                       )
                       ++i;
                   else { ok = false; break; }
               }
               if (ok) 
                   if ((reinterpret_cast<ff_farm*>(nodeslist[next]))->getEmitter() != nullptr) ok=false;
           }
           if (ok) {
               last = next;
               starting_point = next+1;
           } else {
               if (last==-1) {
                   first = find_farm_with_null_collector(nodeslist, first+1);
                   if (first==-1) break;
                   starting_point = first_farm+1;
               } break;
           }                   
       }
       if (first != -1 && last != -1) {
           first_farm = first;
           last_farm  = last;
       }
   };
   // introduces the normal-form of a sequence of farms (or ofarms)
   auto combine_farm_sequence =
       [](const svector<ff_node*>& nodeslist, int first_farm, int last_farm) {

       svector<svector<ff_node*> > W(16);
       W.resize(last_farm-first_farm+1);
       for(int i=first_farm, j=0; i<=last_farm; ++i,++j) {
           W[j]=reinterpret_cast<ff_farm*>(nodeslist[i])->getWorkers();
       }
       size_t nfarms   = W.size();
       size_t nworkers = W[0].size();
       
       std::vector<ff_node*> Workers(nworkers);
       for(size_t j=0; j<nworkers; ++j) {
           if (nfarms==2)  {
               ff_comb *p = new ff_comb(W[0][j],W[1][j]);
               assert(p);
               Workers[j] = p;
           } else {
               const ff_comb *p = new ff_comb(W[0][j],W[1][j]);
               for(size_t i=2;i<nfarms;++i) {
                   const ff_comb* combtmp = new ff_comb(*p, W[i][j]);
                   assert(combtmp);
                   delete p;
                   p = combtmp;                           
               }
               Workers[j] = const_cast<ff_comb*>(p);
           }
       }
       ff_farm* firstfarm= reinterpret_cast<ff_farm*>(nodeslist[first_farm]);
       ff_farm* lastfarm = reinterpret_cast<ff_farm*>(nodeslist[last_farm]);
       ff_farm* newfarm = new ff_farm;
       if (firstfarm->isOFarm()) {
           assert(lastfarm->isOFarm());
           newfarm->set_ordered();
       }
       
       if (firstfarm->getEmitter()) newfarm->add_emitter(firstfarm->getEmitter());       
       if (lastfarm->hasCollector()) 
           newfarm->add_collector(lastfarm->getCollector());
       newfarm->add_workers(Workers);
       newfarm->cleanup_workers();
       newfarm->set_scheduling_ondemand(firstfarm->ondemand_buffer());
       
       return newfarm;
   };
   // ---------------- end helping function --------------------
       
   if (opt.merge_farms) {
       // find the first farm with default collector or with no collector
       int first_farm = find_farm_with_null_collector(pipe.nodes_list);
       if (first_farm!=-1) {
           do {
               int last_farm  = -1;
               farm_sequence(pipe.nodes_list,first_farm,last_farm);
               if (first_farm<last_farm) {  // normal form
                   ff_farm *newfarm = combine_farm_sequence(pipe.nodes_list,first_farm,last_farm);               
                   for(int i=first_farm; i<=last_farm; ++i) pipe.remove_stage(first_farm);
                   pipe.insert_stage(first_farm, newfarm, true);
                   opt_report(opt.verbose_level, OPT_NORMAL,
                              "OPT (pipe): MERGE_FARMS: Merged farms staged [%d-%d]\n", first_farm, last_farm);

               } 
               first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
           }while(first_farm!=-1 && first_farm < static_cast<int>(pipe.nodes_list.size()));
       }
   }
   if (opt.introduce_a2a) {
       int first_farm = find_farm_with_null_collector(pipe.nodes_list);
       while(first_farm != -1 && (first_farm < static_cast<int>(pipe.nodes_list.size()-1))) {
           if (!pipe.nodes_list[first_farm]->isOFarm()) {
               if (pipe.nodes_list[first_farm+1]->isFarm() && !pipe.nodes_list[first_farm+1]->isOFarm()) {
                   ff_farm *farm1 = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                   ff_farm *farm2 = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm+1]);
                   if (farm2->getEmitter() == nullptr) {
                       opt_report(opt.verbose_level, OPT_NORMAL,
                                  "OPT (pipe): INTRODUCE_A2A: Introducing all-to-all between %d and %d stages\n", first_farm, first_farm+1);
                       const ff_farm f = combine_farms_a2a(*farm1, *farm2);
                       ff_farm* newfarm = new ff_farm(f);
                       assert(newfarm);
                       pipe.remove_stage(first_farm);
                       pipe.remove_stage(first_farm);
                       pipe.insert_stage(first_farm, newfarm, true);
                       first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
                   }
               } else {
                   if (pipe.nodes_list[first_farm+1]->isOFarm())
                       opt_report(opt.verbose_level, OPT_INFO,
                                  "OPT (pipe): INTRODUCE_A2A: cannot introduce A2A because node %d is an ordered farm\n", first_farm+1);
                   first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+2);
               }
           } else {
               opt_report(opt.verbose_level, OPT_INFO,
                          "OPT (pipe): INTRODUCE_A2A: cannot introduce A2A because node %d is an ordered farm\n", first_farm);
               first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
           }
       }
   }

   if (opt.remove_collector) {
       // first, for all farms in the pipeline we try to optimize farms' workers
       OptLevel farmopt;
       farmopt.remove_collector = true;
       farmopt.verbose_level = opt.verbose_level;
       for(size_t i=0;i<pipe.nodes_list.size();++i) {
           if (pipe.nodes_list[i]->isFarm()) {
               ff_farm *farmnode = reinterpret_cast<ff_farm*>(pipe.nodes_list[i]);
               if (optimize_static(*farmnode, farmopt)<0) {
                   error("optimize_static, trying to optimize the farm at stage %ld\n", i);
                   return -1;
               }
           }
       }
       
       int first_farm, next=0; 
       while((first_farm=find_farm_with_null_collector(pipe.nodes_list, next)) != -1) {
           if (first_farm < static_cast<int>(pipe.nodes_list.size()-1)) {

               // TODO: if the next stage is A2A would be nice to have a rule
               //       that attaches the farm workers with the first set of nodes

               ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
               if (farm->hasCollector()) {
                   if (farm->isOFarm()) {
                       if ((!pipe.nodes_list[first_farm+1]->isAll2All()) &&
                           (!pipe.nodes_list[first_farm+1]->isFarm())) {
                           farm->add_collector(pipe.nodes_list[first_farm+1]);
                           pipe.remove_stage(first_farm+1);
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Merged next stage with ordered-farm collector\n");
                       }
                   } else {
                       if (!pipe.nodes_list[first_farm+1]->isAll2All()) {
                           farm->remove_collector();
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Removed farm collector\n");
                           
                           if (!pipe.nodes_list[first_farm+1]->isMultiInput()) {
                               // the next stage is a standard node
                               ff_node *next = pipe.nodes_list[first_farm+1];
                               pipe.remove_stage(first_farm+1);
                               ff_minode *mi = new internal_mi_transformer(next);
                               assert(mi);
                               pipe.insert_stage(first_farm+1, mi, true);
                               opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Transformed next stage to multi-input node\n");
                           } 
                       }
                   }
               }
           } else { // this is the last stage (or the only stage)
               ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
               if (!farm->isOFarm()) {
                   if (farm->hasCollector()) {
                       farm->remove_collector();
                       opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Removed farm collector\n");
                   }
               }
           }
           next=first_farm+1;
       }
   }
   if (opt.merge_with_emitter) {
       int first_farm = find_farm_with_null_emitter(pipe.nodes_list);
       if (first_farm!=-1) {
           if (first_farm>0) { // it is not the first one
               bool prev_single_standard  = (!pipe.nodes_list[first_farm-1]->isMultiOutput());
               if (prev_single_standard) {
                   // could be a farm with a collector
                   if (pipe.nodes_list[first_farm-1]->isFarm()) {
                       ff_farm *farm_prev = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm-1]);
                       if (farm_prev->hasCollector()) {
                           ff_node* collector=farm_prev->getCollector();
                           if (collector->isMultiInput() && !collector->isComp()) {
                               error("MERGING MULTI-INPUT COLLECTOR TO THE FARM EMITTER NOT YET SUPPORTED\n");
                               abort();
                               // TODO we have to create a multi-input comp to add to the emitter
                           }
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with farm emitter\n");
                           ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                           farm->add_emitter(collector);
                           farm_prev->remove_collector();
                       }
                   } else {
                       ff_node *node = pipe.nodes_list[first_farm-1];
                       if (node->isMultiInput() && !node->isComp()) {
                           error("MERGING MULTI-INPUT NODE TO THE FARM EMITTER NOT YET SUPPORTED\n");
                           //TODO: we have to create a multi-input comp to add to the emitter
                           abort();
                       }
                       ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                       if (pipe.nodes_list[first_farm]->isOFarm()) {
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with ordered-farm emitter\n");
                       } else {
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with farm emitter\n");                          
                       }
                       farm->add_emitter(node);
                       pipe.remove_stage(first_farm-1);
                   }
               }
           }
       }
   }
   
   // activate blocking mode if the n. of threads is greater than the threshold
   if (opt.blocking_mode) {
       ssize_t card = pipe.cardinality();
       if (opt.max_nb_threads < card) {
           opt_report(opt.verbose_level, OPT_NORMAL, 
                      "OPT (pipe): BLOCKING_MODE: Activating blocking mode, threshold=%ld, number of threads=%ld\n",opt.max_nb_threads, card);
           pipe.blocking_mode(true);
       } 
   }
    // turning off initial/default mapping if the n. of threads is greater than the threshold
   if (opt.no_default_mapping) {
       ssize_t card = pipe.cardinality();
       if (opt.max_mapped_threads < card) {
           opt_report(opt.verbose_level, OPT_NORMAL,
                      "OPT (pipe): MAPPING: Disabling mapping, threshold=%ld, number of threads=%ld\n",opt.max_mapped_threads, card);
           
           pipe.no_mapping();
       }
   }
   // no initial barrier
   if (opt.no_initial_barrier) {
       opt_report(opt.verbose_level, OPT_NORMAL,
                  "OPT (pipe): NO_INITIAL_BARRIER: Initial barrier disabled\n");
       pipe.no_barrier();
   }
   return 0;
}


} // namespace ff
#endif /* FF_OPTIMIZE_HPP */
