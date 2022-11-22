/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 * \file pipeline.hpp
 * \ingroup building_block high_level_patterns
 *
 * \brief This file implements the pipeline skeleton, both in the high-level pattern
 * syntax (\ref ff::ff_pipe) and low-level syntax (\ref ff::ff_pipeline)
 *
 */
/* ***************************************************************************
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */

#ifndef FF_PIPELINE_HPP
#define FF_PIPELINE_HPP

#include <cassert>
#include <memory>
#include <functional>
#include <ff/svector.hpp>
#include <ff/node.hpp>
#ifdef FF_OPENCL
#include <ff/ocl/clEnvironment.hpp>
#endif
#if defined(MAMMUT)
#include <mammut/mammut.hpp>
#endif


namespace ff {
static int nthreads = 8;
// forward declarations
class ff_pipeline;
static inline int optimize_static(ff_pipeline&, const OptLevel&);
template<typename T>    
static inline int combine_with_firststage(ff_pipeline&,T*,bool=false);
template<typename T>    
static inline int combine_with_laststage(ff_pipeline&,T*,bool=false);        

static bool isfarm_withcollector(ff_node*);
static bool isfarm_multimultioutput(ff_node*);
static const svector<ff_node*>& isa2a_getfirstset(ff_node*);
static const svector<ff_node*>& isa2a_getsecondset(ff_node*);
static const svector<ff_node*>& isfarm_getworkers(ff_node*);
    
/**
 * \class ff_pipeline
 * \ingroup core_patterns
 *
 *  \brief The Pipeline skeleton (low-level syntax)
 *
 */
class ff_pipeline: public ff_node {

    friend inline int optimize_static(ff_pipeline&, const OptLevel&);
    template<typename T> 
    friend inline int combine_with_firststage(ff_pipeline&,T*,bool);
    template<typename T>
    friend inline int combine_with_laststage(ff_pipeline&,T*,bool);        
    
protected:

    int prepare_wraparound() {
        const int last = static_cast<int>(nodes_list.size())-1;

        bool isa2a_first  = get_node(0)->isAll2All();
        bool isa2a_last   = get_lastnode()->isAll2All();
        bool isfarm_last  = get_lastnode()->isFarm();
        // possible cases:                                                                       [captured by]
        //
        // the last stage is a standard node   (the first stage can also be a multi-input node)  [last_single_standard]
        // the first stage is a standard node  (the last stage can also be a multi-output node)  [first_single_standard]
        //
        //
        // the last stage is a multi-output node:
        //          - it's a farm with a multi-output collector                                  [last_single_multioutput]
        //          - it's a farm without collector and workers are standard nodes               [last_multi_standard]
        //          - it's a farm without collector with multi-output workers                    [last_multi_multioutput]
        //          - it's a all2all with standard nodes                                         [last_multi_standard]
        //          - it's a all2all with multi-output nodes                                     [last_multi_multioutput]
        //          - it's a singolo nodo (comp or pipeline) with the last stage multi-output    [last_single_multioutput]
        //
        //
        // the first stage is multi-input
        //          - it's a farm (the emettitore is multi-input by default)                     [first_single_multiinput]
        //          - it's a single multi-input node                                             [first_single_multiinput]
        //          - it's all2all with standard nodes                                           [first_multi_standard]
        //          - its' all2all with multi-input nodes                                        [first_multi_multiinput]
        //

        bool last_isfarm_nocollector   = get_lastnode()->isFarm() && !isfarm_withcollector(get_lastnode()); 
        bool last_isfarm_withcollector = get_lastnode()->isFarm() && !last_isfarm_nocollector;

        bool first_single_standard     = (!nodes_list[0]->isMultiInput());
        // the farm is considered single_multiinput
        bool first_single_multiinput   = (nodes_list[0]->isMultiInput() && !isa2a_first);
        bool first_multi_standard      = [&]() {
            if (!isa2a_first) return false;
            const svector<ff_node*>& w1=isa2a_getfirstset(get_node(0));
            assert(w1.size()>0);
            if (w1[0]->isMultiInput()) return false; // NOTE: we suppose homogeneous first set
            return true;
        }();
        bool first_multi_multiinput    = [&]() {
            if (!isa2a_first) return false;
            const svector<ff_node*>& w1=isa2a_getfirstset(get_node(0));
            assert(w1.size()>0);
            if (w1[0]->isMultiInput()) return true; // NOTE: we suppose homogeneous first set
            return false;
        } ();

        bool last_single_standard      = (!nodes_list[last]->isMultiOutput());
        bool last_single_multioutput   = ((nodes_list[last]->isMultiOutput() && !isa2a_last && !isfarm_last) ||
                                          (last_isfarm_withcollector && nodes_list[last]->isMultiOutput()));        
        bool last_multi_standard       = [&]() {            
            if (last_isfarm_nocollector) {
                return !isfarm_multimultioutput(get_lastnode());
            } if (isa2a_last) {                
                const svector<ff_node*>& w1=isa2a_getsecondset(nodes_list[last]);
                assert(w1.size()>0);
                if (!w1[0]->isMultiOutput()) return true; // NOTE: we suppose homogeneous second set
            }
            return false;
        } ();
        bool last_multi_multioutput    = [&]() {
            if (last_isfarm_nocollector) {
                return isfarm_multimultioutput(get_lastnode());
            } if (isa2a_last) {
                const svector<ff_node*>& w1=isa2a_getsecondset(nodes_list[last]);
                assert(w1.size()>0);
                if (w1[0]->isMultiOutput()) return true; // NOTE: we suppose homogeneous second set
            }
            return false;
        } ();      

        // first stage: standard node
        if (first_single_standard) {
            if (create_input_buffer(out_buffer_entries, false)<0) return -1;
            if (last_single_standard) {
                if (set_output_buffer(get_in_buffer())<0)  return -1;

                // blocking stuff ------
                pthread_mutex_t   *m        = NULL;
                pthread_cond_t    *c        = NULL;
                if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                nodes_list[last]->set_output_blocking(m,c);
                // ---------------------
            } else {
                if (last_single_multioutput) {
                    ff_node *t = new ff_buffernode(last, get_in_buffer(), get_in_buffer());
                    internalSupportNodes.push_back(t);
                    assert(t);                   
                    nodes_list[last]->set_output_feedback(t);
                    
                    // blocking stuff ------
                    pthread_mutex_t   *m        = NULL;
                    pthread_cond_t    *c        = NULL;
                    if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                    if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                    // NOTE: if the node0 is multi-input, then the init_input_blocking
                    //       method already calls set_output_blocking
                    t->set_output_blocking(m,c);
                    // ---------------------
                    
                } else {
                    error("PIPE, cannot connect stage %d with node %d\n", last, 0);
                    return -1;                                        
                }
            }
        }
        // first stage: multi-input
        if (first_single_multiinput) {

            if (last_single_standard) {
                if (nodes_list[last]->create_output_buffer(out_buffer_entries,false)<0) return -1;
                nodes_list[0]->set_input_feedback(nodes_list[last]);

                // blocking stuff ......
                pthread_mutex_t   *m        = NULL;
                pthread_cond_t    *c        = NULL;
                if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                // multi-input nodes execute set_output_blocking
                if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                // ---------------------
            } else {
                if (last_single_multioutput) {
                    ff_node *t = new ff_buffernode(out_buffer_entries,false, last); 
                    assert(t);
                    internalSupportNodes.push_back(t);
                    nodes_list[last]->set_output_feedback(t);
                    nodes_list[0]->set_input_feedback(t);

                    // blocking stuff ......
                    pthread_mutex_t   *m        = NULL;
                    pthread_cond_t    *c        = NULL;
                    if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                    // multi-input nodes execute set_output_blocking
                    if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                    // ---------------------
                } else {
                    if (last_multi_standard) {
                        if (nodes_list[last]->create_output_buffer(out_buffer_entries, false) <0)
                            return -1;
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        nodes_list[last]->get_out_nodes(w);
                        assert(w.size());
                        for(size_t j=0;j<w.size();++j)
                            nodes_list[0]->set_input_feedback(w[j]);

                        // blocking stuff ......
                        pthread_mutex_t   *m        = NULL;
                        pthread_cond_t    *c        = NULL;
                        if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                        // multi-input nodes execute set_output_blocking
                        if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                        // ---------------------
                    } else {
                        if (last_multi_multioutput) {
                            svector<ff_node*> w;
                            ff_node *lastbb = get_lastnode();
                            if (lastbb->isAll2All()) 
                                w = isa2a_getsecondset(lastbb);
                            else {
                                assert(lastbb->isFarm());
                                w = isfarm_getworkers(lastbb);
                            }
                            assert(w.size());
                            
                            for(size_t j=0;j<w.size();++j) {
                                ff_node* t = new ff_buffernode(out_buffer_entries,false, j);
                                assert(t);
                                internalSupportNodes.push_back(t);
                                nodes_list[0]->set_input_feedback(t);
                                w[j]->set_output_feedback(t);
                            }                            

                            // blocking stuff ......
                            pthread_mutex_t   *m        = NULL;
                            pthread_cond_t    *c        = NULL;
                            if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
                            // multi-input nodes execute set_output_blocking
                            if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
                            // ---------------------                            
                        } else {
                            error("PIPE, wrap_around invalid stage\n");
                            return -1;
                        }
                    }
                }
            }
        }
        // first stage: multi standard
        if (first_multi_standard) {  // all-to-all
            assert(get_node(0)->isAll2All());
            ff_node* a2a = nodes_list[0];
            const svector<ff_node*>& firstSet=isa2a_getfirstset(a2a);
            assert(firstSet.size()>0);
             
            if (last_single_standard) {
                error("PIPE, wrap_around, cannot connect last with first stage\n");
                return -1;
            }
            if (last_single_multioutput) {
                for(size_t j=0;j<firstSet.size();++j) {
                    ff_node* t = new ff_buffernode(out_buffer_entries,false, j);
                    assert(t);
                    internalSupportNodes.push_back(t);
                    firstSet[j]->set_input(t);
                    nodes_list[last]->set_output_feedback(t);
                    
                    // blocking stuff ......
                    pthread_mutex_t   *m        = NULL;
                    pthread_cond_t    *c        = NULL;                    
                    if (!firstSet[j]->init_input_blocking(m,c)) return -1;
                    t->set_output_blocking(m,c);
                    // ---------------------                            
                }
                // blocking stuff ......
                pthread_mutex_t   *m        = NULL;
                pthread_cond_t    *c        = NULL;
                nodes_list[last]->init_output_blocking(m,c);
                // ---------------------                            
            }
            if (last_multi_standard) {
                svector<ff_node*> w(MAX_NUM_THREADS);
                nodes_list[last]->get_out_nodes(w);
                assert(w.size());
                assert(w.size() == firstSet.size());

                if (a2a->create_input_buffer(in_buffer_entries, false)<0) return -1;
                for(size_t j=0;j<w.size();++j)
                    w[j]->set_output_buffer(firstSet[j]->get_in_buffer());

                // blocking stuff ......
                pthread_mutex_t   *m        = NULL;
                pthread_cond_t    *c        = NULL;
                if (!a2a->init_output_blocking(m,c)) return -1;
                for(size_t j=0;j<w.size();++j) {
                    if (!firstSet[j]->init_input_blocking(m,c)) return -1;
                    w[j]->set_output_blocking(m,c);
                }
                // ---------------------                                            
            }
            if (last_multi_multioutput) {
                error("PIPE, wrap_around, cannot connect last stage with first stage\n");
                return -1;                
            }
        }
        // first stage: multi multi-input
        if (first_multi_multiinput) { 
            assert(get_node(0)->isAll2All());
            const svector<ff_node*>& firstSet=isa2a_getfirstset(nodes_list[0]);
            assert(firstSet.size()>0);
            
            if (last_single_standard) {
                error("PIPE, wrap_around, cannot connect last with first stage\n");
                return -1;
            }
            if (last_single_multioutput) {
                for(size_t j=0;j<firstSet.size();++j) {
                    ff_node* t = new ff_buffernode(out_buffer_entries,false, j);
                    assert(t);
                    internalSupportNodes.push_back(t);
                    firstSet[j]->set_input_feedback(t);
                    nodes_list[last]->set_output_feedback(t);
                }                                                         
            }
            if (last_multi_standard) {
                svector<ff_node*> w(MAX_NUM_THREADS);
                nodes_list[last]->get_out_nodes(w);
                assert(w.size());
                assert(w.size() == firstSet.size());

                for(size_t i=0;i<firstSet.size(); ++i) {
                    ff_node* t = new ff_buffernode(out_buffer_entries,false, i);
                    assert(t);
                    internalSupportNodes.push_back(t);
                    firstSet[i]->set_input_feedback(t);
                    w[i]->set_output(t);
                }
            }
            if (last_multi_multioutput) {
                svector<ff_node*> w;
                ff_node *lastbb = get_lastnode();
                if (lastbb->isAll2All()) 
                    w = isa2a_getsecondset(lastbb);
                else {
                    assert(lastbb->isFarm());
                    w = isfarm_getworkers(lastbb);
                }
                assert(w.size());

                // here we have to create all connections
                for(size_t i=0;i<firstSet.size(); ++i) {
                    for(size_t j=0;j<w.size();++j) {
                        ff_node* t = new ff_buffernode(out_buffer_entries,false, j);
                        assert(t);
                        internalSupportNodes.push_back(t);
                        firstSet[i]->set_input_feedback(t);
                        w[j]->set_output_feedback(t);
                    }
                }
            }
            // blocking stuff ......
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (!nodes_list[last]->init_output_blocking(m,c)) return -1;
            if (!nodes_list[0]->init_input_blocking(m,c)) return -1;
            // ---------------------              
        }        
        return 0;    
    }
    inline int prepare() {

        if (wraparound) {
            if (nodes_list.size()<2) {
                error("PIPE, too few pipeline nodes\n");
                return -1;
            }
            if (prepare_wraparound()<0) {
                error("PIPE, prepare_wraparound failed\n");
                return -1;
            }
        }        
        
        const int nstages=static_cast<int>(nodes_list.size());

        // possible cases:                                                                       [captured by]
        //
        // the current stage is a standard node (the previous stage can also be multi-output)    [curr_single_standard]
        //
        // the current stage is multi-input
        //          - it's a farm (the emettitore is multi-input by default)                     [curr_single_multiinput]
        //          - it's a single multi-input node                                             [curr_single_multiinput]
        //          - it's all2all with standard nodes                                           [curr_multi_standard]
        //          - it's all2all with multi-input nodes                                        [curr_multi_multiinput]
        //
        // the previous stage is a standard node                                                 [prev_single_standard]
        //
        // the previous stage is a multi-output node:
        //          - it's a farm with a multi-output collector                                  [prev_single_multioutput]
        //          - it's a farm without collector and workers are standard node                [prev_multi_standard]
        //          - it's a farm without collector with multi-output workers                    [prev_multi_multioutput]
        //          - it's a all2all with standard nodes                                         [prev_multi_standard]
        //          - it's a all2all with multi-output nodes                                     [prev_multi_multioutput]
        //          - it's a single node (comp or pipeline) with the last stage multi-output     [prev_single_multioutput]
        //
        //
        for(int i=1;i<nstages;++i) {            
            const bool isa2a_curr                = get_node(i)->isAll2All();
            const bool curr_single_standard      = (!nodes_list[i]->isMultiInput());
            // the farm is considered single_multiinput
            const bool curr_single_multiinput   = (!isa2a_curr && nodes_list[i]->isMultiInput());
            const bool curr_multi_multiinput    = [&]() {
                if (!isa2a_curr) return false;
                const svector<ff_node*>& w1=isa2a_getfirstset(get_node(i));
                assert(w1.size()>0);
                for(size_t k=0;k<w1.size();++k) {
                    svector<ff_node*> w2(1);
                    w1[k]->get_in_nodes(w2);
                    for(size_t j=0;j<w2.size();++j)
                        if (w2[j]->isMultiInput()) return true;
                }
                return false;
            } ();
            const bool curr_multi_standard      = [&]() {
                if (!isa2a_curr) return false;
                return !curr_multi_multiinput;
            }();
            const bool isa2a_prev   = get_node_last(i-1)->isAll2All();
            const bool isfarm_prev  = get_node_last(i-1)->isFarm();
            const bool prev_isfarm_nocollector   = (isfarm_prev && !get_node_last(i-1)->isOFarm() && !(isfarm_withcollector(get_node_last(i-1))));
            const bool prev_isfarm_withcollector = isfarm_withcollector(get_node_last(i-1));
            
            
            const bool prev_single_standard      = (!get_node_last(i-1)->isMultiOutput());
            const bool prev_single_multioutput   = ((get_node_last(i-1)->isMultiOutput() && !isa2a_prev && !isfarm_prev) ||
                                                    (prev_isfarm_withcollector && get_node_last(i-1)->isMultiOutput()));        
            const bool prev_multi_standard       = [&]() {
                if (prev_isfarm_nocollector) {
                    svector<ff_node*> w1;
                    nodes_list[i-1]->get_out_nodes(w1);
                    if (!w1[0]->isMultiOutput()) return true;  // NOTE: we suppose homogeneous workers
                } if (isa2a_prev) {
                    const svector<ff_node*>& w1=isa2a_getsecondset(get_node_last(i-1));
                    assert(w1.size()>0);
                    if (!w1[0]->isMultiOutput()) return true; // NOTE: we suppose homogeneous workers
                }
                return false;
            } ();
            const bool prev_multi_multioutput    = [&]() {
                if (prev_isfarm_nocollector) {
                    svector<ff_node*> w1;
                    nodes_list[i-1]->get_out_nodes(w1);
                    if (w1[0]->isMultiOutput()) return true; // NOTE: we suppose homogeneous workers
                } if (isa2a_prev) {
                    const svector<ff_node*>& w1=isa2a_getsecondset(get_node_last(i-1));
                    assert(w1.size()>0);
                    if (w1[0]->isMultiOutput()) return true;  // NOTE: we suppose homogeneous workers
                }
                return false;
            } ();
            
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (curr_single_standard) {
                bool skip_set_output_blocking = false;
                if (nodes_list[i]->create_input_buffer(in_buffer_entries, fixedsizeIN)<0)  return -1;
                if (prev_single_standard) {
                    if (nodes_list[i-1]->set_output_buffer(nodes_list[i]->get_in_buffer())<0) return -1;
                } else {
                    skip_set_output_blocking = true;
                    // WARNING: here we add as output node of the previous stage the
                    //          current node and not a buffer-node.                      
                    if (prev_multi_standard || prev_multi_multioutput) {                        
                        svector<ff_node*> w(1);
                        nodes_list[i-1]->get_out_nodes(w);
                        if (w.size()>1) {
                            error("PIPE, cannot connect stage %d with stage %d\n", i-1, i);
                            return -1;
                        }
                        nodes_list[i-1]->set_output(nodes_list[i]);
                    } else {
                        assert(prev_single_multioutput);
                        nodes_list[i-1]->set_output(nodes_list[i]);
                    }
                }
                // blocking stuff --------------------------------------------
                if (!nodes_list[i]->init_input_blocking(m,c)) {
                    error("PIPE, init input blocking mode for node %d\n", i);
                    return -1;
                }                
                if (!skip_set_output_blocking) // we do not want to overwrite previous setting
                    nodes_list[i-1]->set_output_blocking(m,c); 
                if (!nodes_list[i-1]->init_output_blocking(m,c,false)) {
                    error("PIPE, init output blocking mode for node %d\n", i-1);
                    return -1;
                }
                // ------------------------------------------------------------
            }
            if (curr_single_multiinput) {
                if (prev_single_standard) {
                    if (nodes_list[i-1]->create_output_buffer(in_buffer_entries, fixedsizeOUT)<0) return -1;
                    if (nodes_list[i]->set_input(nodes_list[i-1])<0) return -1;

                    // blocking stuff --------------------------------------------
                    if (!nodes_list[i]->init_input_blocking(m,c)) {
                        error("PIPE, init input blocking mode for node %d\n", i);
                        return -1;
                    }
                    // since the curr node is multi-input, the following op has been executed
                    // by the previous one.
                    //nodes_list[i-1]->set_output_blocking(m,c);
                    if (!nodes_list[i-1]->init_output_blocking(m,c)) {
                        error("PIPE, init output blocking mode for node %d\n", i-1);
                        return -1;
                    }
                    // ------------------------------------------------------------
                } else {
                    if (prev_single_multioutput) {
                        ff_node* t = new ff_buffernode(in_buffer_entries,fixedsizeIN|fixedsizeOUT, i);
                        internalSupportNodes.push_back(t);
                        nodes_list[i-1]->set_output(t);
                        nodes_list[i]->set_input(t);                        
                    }
                    if (prev_multi_standard) {
                        if (nodes_list[i-1]->create_output_buffer(out_buffer_entries, fixedsizeOUT)<0) return -1;
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        nodes_list[i-1]->get_out_nodes(w);
                        assert(w.size());
                        nodes_list[i]->set_input(w);
                        
                        //if (w.size() == 0) nodes_list[i]->set_input(nodes_list[i-1]);                         
                    }
                    if (prev_multi_multioutput) {
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        nodes_list[i-1]->get_out_nodes(w);
                        assert(w.size());
                        for(size_t j=0;j<w.size();++j) {
                            ff_node* t = new ff_buffernode(in_buffer_entries,fixedsizeIN|fixedsizeOUT, j);
                            internalSupportNodes.push_back(t);
                            w[j]->set_output(t);
                            nodes_list[i]->set_input(t);
                        }                    
                    }
                    // blocking stuff --------------------------------------------
                    if (!nodes_list[i]->init_input_blocking(m,c)) {
                        error("PIPE, init input blocking mode for node %d\n", i);
                        return -1;
                    }
                    svector<ff_node*> w(MAX_NUM_THREADS);
                    nodes_list[i-1]->get_out_nodes(w);
                    assert(w.size());
                    for(size_t j=0;j<w.size();++j) {
                        // we set p_cons_* for each buffernode
                        w[j]->set_output_blocking(m,c);
                    }
                    if (!nodes_list[i-1]->init_output_blocking(m,c)) {
                        error("PIPE, init output blocking mode for node %d\n", i);
                        return -1;
                    }
                    // ------------------------------------------------------------
                }
            }
            if (curr_multi_standard) {
                ff_node* a2a = nodes_list[i];
                const svector<ff_node*>& W1 = isa2a_getfirstset(a2a);
                assert(W1.size()>0);
                
                if (prev_single_standard) {
                    if (W1.size()>1) {
                        error("PIPE, cannot connect stage %d with stage %d because of different cardinality\n", i-1, i);
                        return -1;
                    }
                    if (a2a->create_input_buffer(in_buffer_entries, fixedsizeIN)<0) return -1;
                    svector<ff_node*> w(1);
                    nodes_list[i]->get_in_nodes(w);
                    assert(w.size() == 1);
                    if (nodes_list[i-1]->set_output_buffer(w[0]->get_in_buffer())<0) return -1;

                    // blocking stuff --------------------------------------------
                    if (!nodes_list[i]->init_input_blocking(m,c)) {
                        error("PIPE, init input blocking mode for node %d\n", i);
                        return -1;
                    }                
                    nodes_list[i-1]->set_output_blocking(m,c); 
                    if (!nodes_list[i-1]->init_output_blocking(m,c)) {
                        error("PIPE, init output blocking mode for node %d\n", i-1);
                        return -1;
                    }
                    // -----------------------------------------------------------
                } else {
                    if (prev_single_multioutput) {
                        if (a2a->create_input_buffer(in_buffer_entries, fixedsizeIN)<0) return -1;
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        nodes_list[i]->get_in_nodes(w);
                        if (nodes_list[i-1]->set_output(w)<0) return -1;
                        
                        // blocking stuff --------------------------------------------
                        if (!a2a->init_input_blocking(m,c)) return -1;
                        if (!nodes_list[i-1]->init_output_blocking(m,c)) {
                            error("PIPE, init output blocking mode for node %d\n", i-1);
                            return -1;
                        }
                        // -----------------------------------------------------------
                    } else {
                        assert(prev_multi_multioutput || prev_multi_standard);

                        svector<ff_node*> w1(1);
                        nodes_list[i-1]->get_out_nodes(w1);
                        assert(w1.size());
                        // here we pretend to have point-to-point connections
                        if (w1.size() != W1.size()) {
                            error("PIPE, cannot connect stage %d with stage %d, because of different input/output cardinality (**)\n", i-1, i);
                            return -1;
                        }
                        if (a2a->create_input_buffer(in_buffer_entries, fixedsizeIN)<0) return -1;
                        // the previous node can be a farm or a all2all                       
                        for(size_t i=0;i<w1.size();++i) {
                            if (prev_multi_multioutput) {
                                if (w1[i]->set_output(W1[i])<0) return -1;
                            } else {
                                if (w1[i]->set_output_buffer(W1[i]->get_in_buffer())<0) return -1;
                            }
                        }
                        //if (nodes_list[i-1]->set_output(w1)<0) return -1; 
                        
                        // blocking stuff --------------------------------------------
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        nodes_list[i-1]->get_out_nodes(w);
                        assert(w.size());
                        for(size_t i=0;i<w.size();++i) {
                            if (!W1[i]->init_input_blocking(m,c)) return -1;
                            w[i]->set_output_blocking(m,c);
                            
                            if (!w[i]->init_output_blocking(m,c)) return -1;
                        }       
                        // -----------------------------------------------------------
                    }
                }
            }
            // it could be an all-to-all or a pipeline containing as first stage an all-to-all
            if (curr_multi_multiinput) { 
                ff_node* a2a = get_node(i);
                assert(a2a->isAll2All());
                const svector<ff_node*>& W1 = isa2a_getfirstset(a2a);
                assert(W1.size()>0);

                if (prev_single_standard) {
                    if (W1.size()>1) {
                        error("PIPE, cannot connect stage %d with stage %d\n", i-1, i);
                        return -1;
                    }
                    if (a2a->create_input_buffer(in_buffer_entries, fixedsizeIN)<0) return -1;
                    svector<ff_node*> w(1);
                    nodes_list[i]->get_in_nodes(w);
                    assert(w.size() == 1);
                    if (nodes_list[i-1]->set_output_buffer(w[0]->get_in_buffer())<0) return -1;
                }
                if (prev_single_multioutput) {
                    for(size_t j=0;j<W1.size();++j) {
                        svector<ff_node*> w(MAX_NUM_THREADS);
                        W1[j]->get_in_nodes(w);
                        for(size_t k=0;k<w.size();++k) {
                            ff_node* t = new ff_buffernode(in_buffer_entries,fixedsizeIN|fixedsizeOUT, j);
                            internalSupportNodes.push_back(t);
                            w[k]->set_input(t);
                            nodes_list[i-1]->set_output(t);
                        }
                    }
                }                
                if (prev_multi_standard) {

                    svector<ff_node*> w(MAX_NUM_THREADS);
                    nodes_list[i-1]->get_out_nodes(w);
                    assert(w.size());
                    // here we pretend to have point-to-point connections
                    if (w.size() != W1.size()) {
                        error("PIPE, cannot connect stage %d with stage %d, because of different input/output cardinality (**)\n", i-1, i);
                        return -1;
                    }
                    w.clear();
                    if (nodes_list[i-1]->create_output_buffer(in_buffer_entries, fixedsizeOUT)<0) return -1;
                    nodes_list[i-1]->get_out_nodes(w);
                    assert(w.size()== W1.size());

                    for(size_t j=0;j<W1.size();++j) {
                        W1[j]->set_input(w[j]);
                    }                    
                }
                if (prev_multi_multioutput) {
                    svector<ff_node*> w(MAX_NUM_THREADS);
                    nodes_list[i-1]->get_out_nodes(w);
                    assert(w.size());

                    for(size_t k=0;k<W1.size(); ++k) {
                        for(size_t j=0;j<w.size();++j) {
                            ff_node* t = new ff_buffernode(in_buffer_entries,fixedsizeIN|fixedsizeOUT, j);
                            internalSupportNodes.push_back(t);
                            W1[k]->set_input(t);
                            w[j]->set_output(t);
                        }
                    }
                }
                
                // blocking stuff --------------------------------------------
                if (prev_multi_standard) {
                    svector<ff_node*> w(MAX_NUM_THREADS);
                    nodes_list[i-1]->get_out_nodes(w);
                    assert(w.size());
                    for(size_t i=0;i<w.size();++i) {
                        if (!W1[i]->init_input_blocking(m,c)) return -1;
                        w[i]->set_output_blocking(m,c);
                        
                        if (!w[i]->init_output_blocking(m,c)) return -1;
                    }
                } else {
                    if (!a2a->init_input_blocking(m,c)) {
                        error("PIPE, init input blocking mode for node %d\n", i);
                        return -1;
                    }
                    if (!nodes_list[i-1]->init_output_blocking(m,c)) {
                        error("PIPE, init output blocking mode for node %d\n", i-1);
                        return -1;
                    }
                }
                // -----------------------------------------------------------
                
            }
        }
        
        // Preparation of buffers for the accelerator
        int ret = 0;
        if (has_input_channel) {
            if (create_input_buffer(in_buffer_entries, fixedsizeIN)<0) {
                error("PIPE, creating input buffer for the accelerator\n");
                ret=-1;
            } else {             
                if (get_out_buffer()) {
                    error("PIPE, output buffer already present for the accelerator\n");
                    ret=-1;
                } else {
                    // NOTE: the last buffer is forced to be unbounded |
                    if (create_output_buffer(out_buffer_entries, false)<0) {
                        error("PIPE, creating output buffer for the accelerator\n");
                        ret = -1;
                    }
                }
            }

            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;

            // set blocking input for the first stage (cons_m,...)
            if (!init_input_blocking(m,c)) {
                error("PIPE, init input blocking mode for accelerator\n");
            }
            // set my pointers to the first stage input blocking stuff
            ff_node::set_output_blocking(m,c);
            
            m=NULL,c=NULL;

            // set my blocking output (prod_m, ....)
            if (!ff_node::init_output_blocking(m,c)) {
                error("PIPE, init output blocking mode for accelerator\n");
            }

            m=NULL,c=NULL;

            // pipeline's first stage blocking output stuff (prod_m, ....)
            if (!init_output_blocking(m,c)) {
                error("FARM, add_collector, init input blocking mode for accelerator\n");
            }

            m=NULL,c=NULL;
            
            // set my blocking input (cons_m, ....)
            if (!ff_node::init_input_blocking(m,c)) {
                error("PIPE, init input blocking mode for accelerator\n");
            }
            // give pointers to my blocking input to the last pipeline stage (p_cons_m,...)
            set_output_blocking(m,c);
        }

        prepared=true; 
        return ret;
    }


    int freeze_and_run(bool skip_init=false) {
        int nstages=static_cast<int>(nodes_list.size());
        if (!skip_init) {        
#if defined(FF_INITIAL_BARRIER)
            if (initial_barrier) {
                // set the initial value for the barrier 
                if (!barrier)  barrier = new BARRIER_T;
                //const int nthreads = cardinality(barrier);
                if (nthreads > MAX_NUM_THREADS) {
                    error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                    return -1;
                }
                barrier->barrierSetup(nthreads);
            }
#endif
            // only the first stage has to skip the first pop
            skipfirstpop(!has_input_channel);
        }
        if (!prepared) if (prepare()<0) return -1;
        ssize_t startid = (get_my_id()>0)?get_my_id():0;
        for(ssize_t i=0;i<nstages;++i) {
            nodes_list[i]->set_id(i+startid);
            assert(blocking_in == blocking_out);
            nodes_list[i]->blocking_mode(blocking_in);            
            if (nodes_list[i]->freeze_and_run(true)<0) {
                error("ERROR: PIPE, (frbbeezing and) running stage %d\n", i);
                return -1;
            }
        }
        return 0;
    } 

public:

    /**
     *  \brief Constructor
     *
     *  \param input_ch \p true set accelerator mode
     *  \param in_buffer_entries input queue length
     *  \param out_buffer_entries output queue length
     *  \param fixedsize \p true uses bound channels (SPSC queue)
     */
    explicit ff_pipeline(bool input_ch=false,
                         int in_buffer_entries=DEFAULT_BUFFER_CAPACITY,
                         int out_buffer_entries=DEFAULT_BUFFER_CAPACITY, 
                         bool fixedsize=FF_FIXED_SIZE):  
        has_input_channel(input_ch),
        node_cleanup(false),fixedsizeIN(fixedsize),fixedsizeOUT(fixedsize),
        in_buffer_entries(in_buffer_entries),
        out_buffer_entries(out_buffer_entries) {            
    }

    ff_pipeline(const ff_pipeline& p) : ff_node(p) {
        if (p.prepared) {
            error("ff_pipeline, copy constructor, the input pipeline is already prepared\n");
            return;
        }
        has_input_channel = p.has_input_channel;
        node_cleanup = p.node_cleanup;
        fixedsizeIN    = p.fixedsizeIN;
        fixedsizeOUT   = p.fixedsizeOUT;
        wraparound     = p.wraparound;
        in_buffer_entries  = p.in_buffer_entries;
        out_buffer_entries = p.out_buffer_entries;
        nodes_list = p.nodes_list;
        internalSupportNodes = p.internalSupportNodes;
        dontcleanup = p.dontcleanup;
        
        // this is a dirty part, we modify a const object.....
        ff_pipeline *dirty= const_cast<ff_pipeline*>(&p);
        dirty->node_cleanup = false;
        dirty->internalSupportNodes.resize(0);
        dirty->dontcleanup.resize(0);
    }
    
    /**
     * \brief Destructor
     */
    virtual ~ff_pipeline() {        
        if (barrier) delete barrier;
        if (node_cleanup) {
            while(nodes_list.size()>0) {
                ff_node *n = nodes_list.back();

                bool found = false;
                for(size_t i=0;i<internalSupportNodes.size();++i)
                    if (internalSupportNodes[i] == n) { found = true; break; }
                if (!found) {
                    for(size_t i=0;i<dontcleanup.size();++i)
                        if (dontcleanup[i] == n) { found = true; break; }
                }
                nodes_list.pop_back();
                if (!found) delete n;
            }
        }
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes[i];
        }
    }



    /*  WARNING: if these methods are called after prepare (i.e. after having called
     *  run_and_wait_end/run_then_freeze/run/....) they have no effect.     
     *
     */
    void setFixedSize(bool fs) { fixedsizeIN = fixedsizeOUT= fs; }
    void setXNodeInputQueueLength(int sz, bool fixedsize)  {
        in_buffer_entries = sz;
        fixedsizeIN       = fixedsize;
    }
    void setXNodeOutputQueueLength(int sz, bool fixedsize) {
        out_buffer_entries = sz;
        fixedsizeOUT       = fixedsize;
    }


    int numThreads() const { return cardinality(); }
    

    /**
     *  \brief It adds a stage to the pipeline
     *
     *  \param s a ff_node (or derived, e.g. farm) object that is the stage to be added 
     *  to the pipeline
     */
    template<typename T>
    int add_stage(T *s, bool cleanup=false) {  
        nodes_list.push_back(s);
        if (cleanup) internalSupportNodes.push_back(s);
        return 0;
    }
    template<typename T>
    int add_stage(const T& s) {
        ff_node *newstage = new T(s);
        if (!newstage) {
            error("add_stage not enough memory\n");
            return -1;
        }
        return add_stage(newstage, true);
    }

    /* remove one stage already inserted into the pipeline. 
     * NOTE: The node is not deleted if the pipeline cleanup flag is set.
     * NOTE: If remove_from_cleanuplist is true the node is removed (if it is present)
     *       from the list of internalSupportNodes. 
     *
     */
    void remove_stage(int pos, bool remove_from_cleanuplist=false) {
        if (prepared) {
            error("PIPE, remove_stage, stage %d cannot be removed because the PIPE has already been prepared\n");
            return;
        }
        if (pos<0 || pos>static_cast<int>(nodes_list.size())) {
            error("PIPE, remove_stage, stage number %d does not exist\n", pos);
            return;
        }
        svector<ff_node*>::iterator it=nodes_list.begin();
        assert(it+pos < nodes_list.end());
        if (remove_from_cleanuplist) {
            ff_node* node = nodes_list[pos];
            int pos2=-1;
            for(size_t i=0;i<internalSupportNodes.size();++i)
                if (internalSupportNodes[i] == node) { pos2 = i; break; }
            if (pos2>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos2);            
        }        
        nodes_list.erase(it+pos);
    }
    void insert_stage(int pos, ff_node* node, bool cleanup=false) {
        if (prepared) {
            error("PIPE, insert_stage, stage %d cannot be added because the PIPE has already been prepared\n");
            return;
        }

        if (pos<0 || pos>static_cast<int>(nodes_list.size())) {
            error("PIPE, insert_stage, invalid position\n");
            return;
        }
        svector<ff_node*>::iterator it=nodes_list.begin();
        assert(it+pos <= nodes_list.end());
        nodes_list.insert(it+pos, node);
        if (cleanup) internalSupportNodes.push_back(node);
    }

    ssize_t get_stageindex(const ff_node* stage){
        if (!stage) return -1;
        for(ssize_t i=0; i<(ssize_t)nodes_list.size(); ++i) 
            if(nodes_list[i] == stage) return i;
        return -1;
    }

    // returns true if the old node has been changed with the new one
    // false in case of error of if the old node has not been found
    bool change_node(ff_node* old, ff_node* n, bool cleanup=false, bool remove_from_cleanuplist=false) {
        if (prepared) {
            error("PIPE, change_node cannot be called because the PIPE has already been prepared\n");
            return false;
        }
        for(size_t i=0; i<nodes_list.size(); ++i) {
            if (nodes_list[i] == old) {
                insert_stage(i+1, n, cleanup);
                remove_stage(i, remove_from_cleanuplist);
                return true;
            }
        }
        return false;
    }

    /*
     * returns the list of nodes removing them from the pipeline(s)
     */
    const svector<std::pair<ff_node*,bool>> get_and_remove_nodes() {
        int nstages=static_cast<int>(this->nodes_list.size());
        svector<std::pair<ff_node*,bool>> newvector;
        for(int i=0;i<nstages;++i)  {
            if (nodes_list[i]->isPipe()) {
                ff_pipeline * p = reinterpret_cast<ff_pipeline*>(nodes_list[i]);
                const svector<std::pair<ff_node*,bool>>& W = p->get_and_remove_nodes();
                newvector+=W;
                if (node_cleanup) {
                    bool found=false;
                    for(size_t i=0;i<internalSupportNodes.size();++i)
                        if (internalSupportNodes[i]==p) {found =true; break;}
                    if (!found) internalSupportNodes.push_back(p);
                }
            } else 
                newvector.push_back(std::make_pair(nodes_list[i], node_cleanup));
        }
        for(int i=0;i<nstages;++i)  this->remove_stage(0);
        return newvector;
    }

    /**
     * \brief Feedback channel (pattern modifier)
     * 
     * The last stage output stream will be connected to the first stage 
     * input stream in a cycle (feedback channel)
     */
    int wrap_around() { wraparound=true; return 0; };
    bool isset_wraparound() { return wraparound; }
    
    bool isset_cleanup_nodes() const { return node_cleanup; }
    
    inline void cleanup_nodes(bool onoff=true) { node_cleanup = onoff; }

    /**
     * returns the stages added to the pipeline
     */
    const svector<ff_node*>& getStages() const { return nodes_list; }

    /**
     *  \brief returns all nodes of the pipeline, where stages are not pipeline.
     *  In the list returned, no single stage is a pipeline. 
     */
    const svector<ff_node*> get_pipeline_nodes() const {
        const int nstages=static_cast<int>(nodes_list.size());
        svector<ff_node*> newvector;
        for (int i=0;i<nstages;++i) {
            if (nodes_list[i]->isPipe()) {
                ff_pipeline * p = reinterpret_cast<ff_pipeline*>(nodes_list[i]);
                const svector<ff_node*>& tmp = p->get_pipeline_nodes();
                newvector+=tmp;
            } else
                newvector.push_back(nodes_list[i]);
        }
        return newvector;
    };
    /**
     *  \brief returns the stage i of the pipeline. If the stage is a pipeline
     *  the function is called recursively extracting its first stage. 
     */
    ff_node* get_node(int i) const {
        if (!nodes_list.size()) return nullptr;
        if (i<0 || i>=(int)nodes_list.size()) return nullptr;
        if (nodes_list[i]->isPipe()) {
            ff_pipeline * p = reinterpret_cast<ff_pipeline*>(nodes_list[i]);
            return p->get_node(0);
        }
        return nodes_list[i];
    }
    /**
     *  \brief returns the stage i of the pipeline. If the stage is a pipeline
     *  the function is called recursively extracting its last stage. 
     */
    ff_node* get_node_last(int i) const {
        if (!nodes_list.size()) return nullptr;
        if (i<0 || i>=(int)nodes_list.size()) return nullptr;
        if (nodes_list[i]->isPipe()) {
            ff_pipeline * p = reinterpret_cast<ff_pipeline*>(nodes_list[i]);
            return p->get_lastnode();
        }
        return nodes_list[i];
    }
    /**
     *  \brief returns the last stage of the pipeline recursively. 
     */
    ff_node* get_lastnode() const {
        if (!nodes_list.size()) return nullptr;
        const int last = static_cast<int>(nodes_list.size())-1;
        if (nodes_list[last]->isPipe()) {
            ff_pipeline * p = reinterpret_cast<ff_pipeline*>(nodes_list[last]);
            return p->get_lastnode();
        }
        return nodes_list[last];
    }

     /**
     *  \brief returns the last stage of the pipeline. 
     */
    ff_node* get_laststage() const {
        if (!nodes_list.size()) return nullptr;
        const int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last];
    }
    
     /**
     *  \brief returns the first stage of the pipeline. 
     */
    ff_node* get_firststage() const {
        if (!nodes_list.size()) return nullptr;
        return nodes_list[0];
    }

    ff_node* get_nextstage(const ff_node* s){
        ssize_t index = get_stageindex(s);
        if (index == -1 || (index+1) == (ssize_t)nodes_list.size())
            return nullptr;
        return nodes_list[index+1];
    }

    ff_node* get_prevstage(const ff_node* s){
        ssize_t index = get_stageindex(s);
        if (index <= 0) return nullptr;
        return nodes_list[index-1];
    }
    
    inline void get_out_nodes(svector<ff_node*>&w) {
        assert(nodes_list.size()>0);
        int last = static_cast<int>(nodes_list.size())-1;
        nodes_list[last]->get_out_nodes(w);
    }

    inline void get_out_nodes_feedback(svector<ff_node*>&w) {
        assert(nodes_list.size()>0);
        int last = static_cast<int>(nodes_list.size())-1;
        nodes_list[last]->get_out_nodes_feedback(w);
    }

    
    inline void get_in_nodes(svector<ff_node*>&w) {
        assert(nodes_list.size()>0);
        nodes_list[0]->get_in_nodes(w);
    }
    
    void skipfirstpop(bool sk)   { 
        get_node(0)->skipfirstpop(sk);
        for(size_t i=1;i<nodes_list.size();++i)
            nodes_list[i]->skipfirstpop(false);            
    }

#ifdef DFF_ENABLED
    void skipallpop(bool sk)   { 
        get_node(0)->skipallpop(sk);       
    }
#endif

    
    /* WARNING: these methods must be called before the run() method */
    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }
    void no_barrier() {
        initial_barrier = false;
    }

    void no_mapping() {
        default_mapping = false;
    }
    
    /**
     * \brief Run the pipeline skeleton asynchronously
     * 
     * Run the pipeline, the method call return immediately. To be coupled with 
     * \ref ff_pipeline::wait()
     */
    int run(bool skip_init=false) {
        int nstages=static_cast<int>(nodes_list.size());

        if (!skip_init) {            

#if defined(MAMMUT)
            mammut::energy::Energy *e = mammut.getInstanceEnergy();
            mammutcounter = e->getCounter();
            if (mammutcounter) {
                mammutcounter->reset();
                printf("Starting Joules = %g\n", mammutcounter->getJoules());
            }
#endif
#if defined(FF_INITIAL_BARRIER)    
            if (initial_barrier) {
                // set the initial value for the barrier 
                if (!barrier)  barrier = new BARRIER_T;
                //const int nthreads = cardinality(barrier);
                printf("nthreads = %d\n", nthreads);
                if (nthreads > MAX_NUM_THREADS) {
                    error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                    return -1;
                }
                barrier->barrierSetup(nthreads);
            }
#endif          
#ifdef FF_OPENCL

            // TODO: check if the pipeline has at least 1 oclNode.
            // setup openCL environment
            clEnvironment::instance();
            
#if 0            
            // REMOVE THIS ?
            // check if we have to setup the OpenCL environment !
            if (fftree_ptr->hasOpenCLNode()) {
                // setup openCL environment
                clEnvironment::instance();
            }
#endif            
#endif
            
            // only the first stage has to skip the first pop
            skipfirstpop(!has_input_channel);
        }
        if (!prepared) if (prepare()<0) return -1;

        ssize_t startid = (get_my_id()>0)?get_my_id():0;
        for(int i=0;i<nstages;++i) {
            if (i>0) startid += nodes_list[i-1]->cardinality();
            nodes_list[i]->set_id(startid);
            assert(blocking_in == blocking_out);
            nodes_list[i]->blocking_mode(blocking_in);
            if (!default_mapping) nodes_list[i]->no_mapping();
            if (nodes_list[i]->run(true)<0) {
                error("ERROR: PIPE, running stage %d\n", i);
                return -1;
            }
        }        
        return 0;
    }

    int dryrun() {  
        if (!prepared) 
            if (prepare()<0) return -1; 
        return 0;
    }

    /**
     * \relates ff_pipe
     * \brief run the pipeline, waits that all stages received the End-Of-Stream (EOS),
     * and destroy the pipeline run-time 
     * 
     * Blocking behaviour w.r.t. main thread to be clarified
     */
#ifdef DFF_ENABLED
    int run_and_wait_end();
#else
    int run_and_wait_end() {
        if (isfrozen()) {  // TODO 
            error("PIPE: Error: feature not yet supported\n");
            return -1;
        } 
        stop();
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }
#endif

    /**
     * \related ff_pipe
     * \brief run the pipeline, waits that all stages received the End-Of-Stream (EOS),
     * and suspend the pipeline run-time
     * 
     * Run-time threads are suspended by way of a distrubuted protocol. 
     * The same pipeline can be re-started by calling again run_then_freeze 
     */
    int run_then_freeze(ssize_t nw=-1) {
        if (isfrozen()) {
            // true means that next time threads are frozen again
            thaw(true, nw);
            return 0;
        }
        if (!prepared) if (prepare()<0) return -1;
        //freeze();
        //return run();
        /* freeze_and_run is required because in the pipeline 
         * there isn't no manager thread, which allows to freeze other 
         * threads before starting the computation
         */
        return freeze_and_run(false);
    }
    
    /**
     * \brief wait for pipeline termination (all stages received EOS)
     */
    int wait(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait()<0) {
                error("PIPE, waiting stage thread, id = %d\n",nodes_list[i]->get_my_id());
                ret = -1;
            } 
        
#if defined(MAMMUT)
        if (mammutcounter) joules = mammutcounter->getJoules();
#endif
        return ret;
    }
    
    int wait_last() {
        int ret=-1;
        const int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return ret;
        if (nodes_list[0]->wait()<0) return ret;
        return nodes_list[last]->wait();
    }
    
    /**
     * \brief wait for pipeline to complete and suspend (all stages received EOS)
     * 
     * 
     */
    inline int wait_freezing(/* timeval */ ) {
        int ret=0;
        for(unsigned int i=0;i<nodes_list.size();++i)
            if (nodes_list[i]->wait_freezing()<0) {
                error("PIPE, waiting freezing of stage thread, id = %d\n",
                      nodes_list[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    } 
       
    inline void stop() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->stop();
    }

  
    inline void freeze() {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->freeze();
    }

    /**
     * \brief checks if the pipeline is still running or not
     *
     */
    inline bool done() const { 
        int nstages=static_cast<int>(nodes_list.size());
        for(int i=0;i<nstages;++i) 
            if (!nodes_list[i]->done()) return false;
        return true;
    }
    
    inline bool isPipe() const { return true; }
    inline bool isPrepared() const { return prepared;}    
    inline bool isMultiInput() const { 
        if (nodes_list.size()==0) return false;
        return nodes_list[0]->isMultiInput();
    }
    inline bool isMultiOutput() const {
        if (nodes_list.size()==0) return false;
        int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last]->isMultiOutput();
    }
    
    // remove internal pipeline
    // WARNING: if there are feedback channels calling this method is dangerous!
    void flatten() {
        const svector<std::pair<ff_node*,bool>>& W = this->get_and_remove_nodes();
        assert(get_pipeline_nodes().size() == 0);
        for(size_t i=0;i<W.size();++i) this->add_stage(W[i].first);
        if (node_cleanup) {
            for(size_t i=0;i<W.size();++i)
                if (!W[i].second) dontcleanup.push_back(W[i].first);                    
        } else {
            for(size_t i=0;i<W.size();++i)
                if (W[i].second) internalSupportNodes.push_back(W[i].first);                                
        }
    }

    
    /** 
     * \brief offload a task to the pipeline from the offloading thread (accelerator mode)
     * 
     * Offload a task onto a pipeline accelerator, tipically the offloading 
     * entity is the main thread (even if it can be used from any 
     * \ref ff_node::svc method)  
     *
     * \note to be used in accelerator mode only
     */
    inline bool offload(void * task,
                        unsigned long retry=((unsigned long)-1),
                        unsigned long ticks=ff_node::TICKS2WAIT) { 
         FFBUFFER * inbuffer = get_in_buffer();
         assert(inbuffer != NULL);

         if (ff_node::blocking_out) {
         _retry:
             if (inbuffer->push(task)) {
                 pthread_cond_signal(p_cons_c);
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

    /** 
     * \brief gets a result from a task to the pipeline from the main thread 
     * (accelator mode)
     * 
     * Total call: return when a result is available. To be used in accelerator mode only
     *
     * \param[out] task
     * \param retry number of attempts to get a result before failing 
     * (related to nonblocking get from channel - expert use only)
     * \param ticks number of clock cycles between successive attempts 
     * (related to nonblocking get from channel - expert use only)
     * \return \p true is a task is returned, \p false if End-Of-Stream (EOS)
     */
    inline bool load_result(void ** task, 
                            unsigned long retry=((unsigned long)-1),
                            unsigned long ticks=ff_node::TICKS2WAIT) {
        FFBUFFER * outbuffer = get_out_buffer();

        if (!outbuffer) {
            if (!has_input_channel) 
                error("PIPE: accelerator is not set, offload not available");
            else
                error("PIPE: output buffer not created");
            return false;            
        }

        if (ff_node::blocking_in) {
        _retry:
            if (outbuffer->pop(task)) {
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            struct timespec tv;
            timedwait_timeout(tv);
            pthread_mutex_lock(cons_m);
            pthread_cond_timedwait(cons_c, cons_m, &tv);
            pthread_mutex_unlock(cons_m);
            goto _retry;
        }
        for(unsigned long i=0;i<retry;++i) {
            if (outbuffer->pop(task)) {
                if ((*task != (void *)FF_EOS)) return true;
                else return false;
            }
            losetime_in(ticks);
        }     
        return false;
    }
    /** 
     * \brief try to get a result from a task to the pipeline from the main thread 
     * (accelator mode)
     * 
     * Partial call: can return no result. To be used in accelerator mode only
     *
     * \param[out] task
     * \return \p true is a task is returned (including EOS), 
     * \p false if no task is returned     */
    inline bool load_result_nb(void ** task) {
        FFBUFFER * outbuffer = get_out_buffer();
        if (outbuffer) {
            if (outbuffer->pop(task)) return true;
            else return false;
        }
        
        if (!has_input_channel) 
            error("PIPE: accelerator is not set, offload not available");
        else
            error("PIPE: output buffer not created");
        return false;        
    }
    

    int cardinality() const { 
        int card=0;
        for(unsigned int i=0;i<nodes_list.size();++i) 
            card += nodes_list[i]->cardinality();        
        return card;
    }

    
    /* 
     * \brief Misure execution time (including init and finalise)
     *
     * \return pipeline execution time (including init and finalise)
     */
    double ffTime() {
        return diffmsec(nodes_list[nodes_list.size()-1]->getstoptime(),
                        nodes_list[0]->getstarttime());
    }
    
    /* 
     * \brief Misure execution time (excluding init and finalise)
     *
     * \return pipeline execution time (excluding runtime setup)
     */
    double ffwTime() {
        return diffmsec(nodes_list[nodes_list.size()-1]->getwstoptime(),
                        nodes_list[0]->getwstartime());
    }
    
#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- pipeline:\n";
        for(unsigned int i=0;i<nodes_list.size();++i)
            nodes_list[i]->ffStats(out);

#if defined(MAMMUT)
        out << "Joules: " << joules << "\n";
#endif
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

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
    
protected:

    int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(unsigned int i=0;i<nodes_list.size();++i) 
            card += nodes_list[i]->cardinality(barrier);
        
        return card;
    }
    
    /* The pipeline has not been flattened and its first stage is a multi-input node used as 
     * a standard node. 
     */
    inline bool  put(void * ptr) { 
        return nodes_list[0]->put(ptr);
    }
    inline FFBUFFER * get_in_buffer() const {
        return nodes_list[0]->get_in_buffer();
    }
    
    // returns the pipeline starting time
    const struct timeval startTime() { return nodes_list[0]->getstarttime(); }

    void* svc(void *) { return NULL; }    
    int   svc_init() { return -1; };    
    void  svc_end()  {}

    void  setAffinity(int) { 
        error("PIPE, setAffinity: cannot set affinity for the pipeline\n");
    }
    
    int   getCPUId() const { return -1;}

    inline void thaw(bool _freeze=false, ssize_t nw=-1) {
        for(unsigned int i=0;i<nodes_list.size();++i) nodes_list[i]->thaw(_freeze, nw);
    }
    
    inline bool isfrozen() const { 
        int nstages=static_cast<int>(nodes_list.size());
        for(int i=0;i<nstages;++i) 
            if (!nodes_list[i]->isfrozen()) return false;
        return true;
    }

    // consumer
    virtual inline bool init_input_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool /*feedback*/=true) {
        return nodes_list[0]->init_input_blocking(m,c);
    }
    // producer
    virtual inline bool init_output_blocking(pthread_mutex_t   *&m,
                                             pthread_cond_t    *&c,
                                             bool /*feedback*/=true) {
        const int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return false;
        return nodes_list[last]->init_output_blocking(m,c);
    }
    virtual inline void set_output_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool canoverwrite=false) {
        const int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return;
        nodes_list[last]->set_output_blocking(m,c, canoverwrite);
    }

    virtual inline pthread_cond_t    &get_cons_c()        { return nodes_list[0]->get_cons_c();}

    int create_input_buffer(int nentries, bool fixedsize) { 
        if (in) return -1;  
        
        if (nodes_list[0]->create_input_buffer(nentries, fixedsize)<0) {
            error("PIPE, creating input buffer for node 0\n");
            return -1;
        }
        if (!nodes_list[0]->isMultiInput() || get_node(0)->isFarm()) 
            ff_node::set_input_buffer(get_node(0)->get_in_buffer());
        return 0;
    }
    
    int create_output_buffer(int nentries, bool fixedsize=false) {
        int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return -1;

        if (nodes_list[last]->create_output_buffer(nentries, fixedsize)<0) {
            error("PIPE, creating output buffer for node %d\n",last);
            return -1;
        }
        ff_node::set_output_buffer(nodes_list[last]->get_out_buffer());
        return 0;
    }

    int set_output_buffer(FFBUFFER * const o) {
        int last = static_cast<int>(nodes_list.size())-1;
        if (last<0) return -1;

        if (nodes_list[last]->set_output_buffer(o)<0) {
            error("PIPE, setting output buffer for node %d\n",last);
            return -1;
        }
        return 0;
    }

    inline int set_input(ff_node *node) { 
        return nodes_list[0]->set_input(node);
    }
    inline int set_input(const svector<ff_node *> & w) { 
             return nodes_list[0]->set_input(w);
    }
    inline int set_input_feedback(ff_node *node) { 
        return nodes_list[0]->set_input_feedback(node);
    }   

    inline int set_output(ff_node *node) {
        int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last]->set_output(node);
    }
    inline int set_output(const svector<ff_node *> &w) {
        int last = static_cast<int>(nodes_list.size())-1;
        return nodes_list[last]->set_output(w);
    }

    inline int set_output_feedback(ff_node *node) {
        int last = static_cast<int>(nodes_list.size())-1;

        if (nodes_list[last]->isMultiOutput())
            return nodes_list[last]->set_output_feedback(node);

        assert(node->get_out_buffer());
        return nodes_list[last]->set_output_buffer(node->get_out_buffer());
    }

    inline int ondemand_buffer() const {
        int last = static_cast<int>(nodes_list.size())-1;
        
        svector<ff_node*> w;
        nodes_list[last]->get_out_nodes(w);
        return w[0]->ondemand_buffer();  // NOTE: we suppose that all others are the same !!!!!
    }
              
private:
    bool has_input_channel; // for accelerator
    bool node_cleanup;
    bool fixedsizeIN, fixedsizeOUT;
    bool wraparound=false;
    int in_buffer_entries;
    int out_buffer_entries;
    svector<ff_node *> nodes_list;
    svector<ff_node*>  internalSupportNodes;
    svector<ff_node*>  dontcleanup;  // used by the flatten method

#if defined(MAMMUT)
    mammut::Mammut           mammut;
    mammut::energy::Joules   joules        = -1;
    mammut::energy::Counter* mammutcounter = nullptr;
#endif
};


//#ifndef WIN32 //VS12
    // ------------------------ high-level (simpler) pipeline ------------------
#if ((__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__)) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))

#include <ff/make_unique.hpp>

    /*! 
     * \class ff_Pipe
     * \ingroup high_level_patterns
     * 
     * \brief Pipeline pattern (high-level pattern syntax)
     *
     * Set up a parallel for pipeline pattern run-time support object. 
     * Run with \p run_and_wait_end or \p run_the_freeze. See related functions.
     *
     * \note Don't use to model a workflow of tasks, stages are nonblocking threads 
     * and
     * require one core per stage. If you need to model a workflow use \ref ff::ff_mdf
     *
     * \example pipe_basic.cpp
     */
    
    template<typename IN_t=char,typename OUT_t=IN_t>
    class ff_Pipe: public ff_pipeline {
    private:

#if !defined(__CUDACC__) && !defined(WIN32) && !defined(__ICC)
        // 
        // Thanks to Suter Toni (HSR) for suggesting the following code for checking
        // correct input-output types ordering.
        //
		
        template<class A, class...>
        struct valid_stage_types : std::true_type {};
        
        template<class A, class B, class... Bs>
        struct valid_stage_types<A&&, B&&, Bs &&...> : std::integral_constant<bool, std::is_same<typename A::out_type, typename B::in_type>{} && valid_stage_types<B, Bs...>{}> {};        

        template<class A, class B, class... Bs>
        struct valid_stage_types<std::unique_ptr<A>&&, std::unique_ptr<B>&&, Bs &&...> : std::integral_constant<bool, std::is_same<typename A::out_type, typename B::in_type>{} && valid_stage_types<std::unique_ptr<B>, Bs...>{}> {}; 

        template<class A, class B, class... Bs>
        struct valid_stage_types<std::unique_ptr<A>&&, B&&, Bs &&...> : std::integral_constant<bool, std::is_same<typename A::out_type, typename B::in_type>{} && valid_stage_types<B, Bs...>{}> {}; 
        template<class A, class B, class... Bs>
        struct valid_stage_types<A&&, std::unique_ptr<B>&&, Bs &&...> : std::integral_constant<bool, std::is_same<typename A::out_type, typename B::in_type>{} && valid_stage_types<std::unique_ptr<B>, Bs...>{}> {};   

        //struct valid_stage_types<A, B, Bs ...> : std::integral_constant<bool, std::is_same<typename A::out_type, typename B::in_type>{} && valid_stage_types<B, Bs...>{}> {};        

#endif

        // 
        // Thanks to Peter Sommerlad for suggesting the following simpler code
        //
        void add2pipeall(){} // base case

        // need to see this before add2pipeall variadic template function
        inline void add2pipe(ff_node &node) { ff_pipeline::add_stage(&node); }
        // need to see this before add2pipeall variadic template function
        inline void add2pipe(ff_node *node) { ff_pipeline::add_stage(node);  }

        template<typename FIRST,typename ...ARGS>
        void add2pipeall(FIRST &stage,ARGS&...args){
        	add2pipe(stage);
        	add2pipeall(args...); // recurse
        }

        template<typename FIRST,typename ...ARGS>
        void add2pipeall(const FIRST &stage,ARGS&...args){
            FIRST *f = new FIRST(stage);
            add2pipe(f);
        	add2pipeall(args...); // recurse
        }

        
        template<typename FIRST,typename ...ARGS>
        void add2pipeall(std::unique_ptr<FIRST> & stage,ARGS&...args){
            ff_node* node = stage.release();
        	add2pipe(*node); //stage.release());
            cleanup_stages.push_back(node);
        	add2pipeall(args...); // recurse
        }

    protected:
        svector<ff_node*> cleanup_stages;
        
    public:

        // NOTE: The ff_Pipe accepts as stages either l-value references or std::unique_ptr l-value references.
        //       The ownership of the (unique) pointer stage is transferred to the pipeline !!!!

        typedef IN_t  in_type;
        typedef OUT_t out_type;

        /**
         * \brief Create a stand-alone pipeline (no input/output streams). Run with \p run_and_wait_end or \p run_the_freeze.
         *
         * Identifies an stream parallel construct in which stages are executed 
         * in parallel. 
         * It does require a stream of tasks, either external of created by the 
         * first stage.
         * \param stages pipeline stages
         * 
         * Example: \ref pipe_basic.cpp
         */
        template<typename... STAGES>
        ff_Pipe(STAGES &&...stages) {    // forwarding reference (aka universal reference)
#if !defined(__CUDACC__) && !defined(WIN32) && !defined(__ICC)
        	static_assert(valid_stage_types<STAGES...>{}, "Input & output types of the pipe's stages don't match");
#endif
        	this->add2pipeall(stages...); //this->add2pipeall(std::forward<STAGES>(stages)...);
        }
        /**
         * \brief Create a pipeline (with input stream). Run with \p run_and_wait_end or \p run_the_freeze.
         *
         * Identifies an stream parallel construct in which stages are executed 
         * in parallel. 
         * It does require a stream of tasks, either external of created by the 
         * first stage.
         * \param input_ch \p true to enable first stage input stream
         * \param stages pipeline stages
         *
         * Example: \ref pipe_basic.cpp
         */
        template<typename... STAGES>
        explicit ff_Pipe(bool input_ch, STAGES &&...stages):ff_pipeline(input_ch) {
#if !defined(__CUDACC__) && !defined(WIN32) && !defined(__ICC)
        	static_assert(valid_stage_types<STAGES...>{},
                          "Input & output types of the pipe's stages don't match");
#endif
        	this->add2pipeall(stages...);
        }
        
        ~ff_Pipe() {
            for (auto s: cleanup_stages) delete s;
        }

        operator ff_node* () { return this;}

        bool load_result(OUT_t *&task,
                         unsigned long retry=((unsigned long)-1),
                         unsigned long ticks=ff_node::TICKS2WAIT) {
            return ff_pipeline::load_result((void**)&task, retry,ticks);
        }
        
        // deleted members
        bool load_result(void ** task,
                         unsigned long retry=((unsigned long)-1),
                         unsigned long ticks=ff_node::TICKS2WAIT) = delete;

        /*
         *  using the following two add_stage method, no static check on the input/output 
         *  types is executed 
         */
        int add_stage(ff_node &s) { 
            add2pipe(s); 
            return 0; 
        }
        int add_stage(std::unique_ptr<ff_node> &&s) { 
            add2pipe(s.release()); 
            return 0; 
        }

        // deleted functions
        int add_stage(ff_node * s) = delete;
        void cleanup_nodes() = delete;       
    };
#endif /* HAS_CXX11_VARIADIC_TEMPLATES */
//#endif //VS12

} // namespace ff

// to avoid warning when optimize_static is not used
#include<ff/optimize.hpp>

#endif /* FF_PIPELINE_HPP */
