/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file all2all.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow all-2-all building block
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_A2A_HPP
#define FF_A2A_HPP

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

#include <ff/node.hpp>
#include <ff/multinode.hpp>

namespace ff {

// forward declarations
static ff_node* ispipe_getlast(ff_node*);
    
class ff_a2a: public ff_node {
    friend class ff_farm;
    friend class ff_pipeline;    
protected:
    inline int cardinality(BARRIER_T * const barrier)  { 
        int card=0;
        for(size_t i=0;i<workers1.size();++i) 
            card += workers1[i]->cardinality(barrier);
        for(size_t i=0;i<workers2.size();++i) 
            card += workers2[i]->cardinality(barrier);
        
        return card;
    }

    inline int prepare() {
        /* ----------------------- */
        if (wraparound) {   
            if (workers2[0]->isMultiOutput()) { // NOTE: we suppose that all others are the same
                if (workers1[0]->isMultiInput()) { // NOTE: we suppose that all others are the same
                    for(size_t i=0;i<workers2.size(); ++i) {
                        for(size_t j=0;j<workers1.size();++j) {
                            ff_node* t = new ff_buffernode(in_buffer_entries,false);
                            t->set_id(i);
                            internalSupportNodes.push_back(t);
                            workers2[i]->set_output_feedback(t);
                            workers1[j]->set_input_feedback(t);
                        }                    
                    }
                } else {
                    // the cardinatlity of the first and second set of workers must be the same
                    if (workers1.size() != workers2.size()) {
                        error("A2A, wrap_around, the workers of the second set are not multi-output nodes so the cardinatlity of the first and second set must be the same\n");
                        return -1;
                    }
                    
                    if (create_input_buffer(in_buffer_entries, false) <0) {
                        error("A2A, error creating input buffers\n");
                        return -1;
                    }
                    
                    for(size_t i=0;i<workers2.size(); ++i)
                        workers2[i]->set_output_feedback(workers1[i]);
                    
                }
            } else {
                // the cardinatlity of the first and second set of workers must be the same
                if (workers1.size() != workers2.size()) {
                    error("A2A, wrap_around, the workers of the second set are not multi-output nodes so the cardinatlity of the first and second set must be the same\n");
                    return -1;
                }
                if (!workers1[0]->isMultiInput()) {  // we suppose that all others are the same
                    if (create_input_buffer(in_buffer_entries, false) <0) {
                        error("A2A, error creating input buffers\n");
                        return -1;
                    }
                    
                    for(size_t i=0;i<workers2.size(); ++i)
                        workers2[i]->set_output_buffer(workers1[i]->get_in_buffer());
                    
                } else {
                    if (create_output_buffer(out_buffer_entries, false) <0) {
                        error("A2A, error creating output buffers\n");
                        return -1;
                    }
                    
                    for(size_t i=0;i<workers1.size(); ++i)
                        if (workers1[i]->set_input_feedback(workers2[i])<0) {
                            error("A2A, wrap_around, the nodes of the first set are not multi-input\n");
                            return -1;
                        }
                }
            }

            // blocking stuff --------------------------------------------
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            for(size_t i=0;i<workers2.size(); ++i) {
                if (!workers2[i]->init_output_blocking(m,c)) return -1;
            }
            if (!workers2[0]->isMultiOutput()) {
                assert(workers1.size() == workers2.size());
            }
            if (workers1[0]->isMultiInput()) {
                for(size_t i=0;i<workers1.size();++i) {
                    if (!workers1[i]->init_input_blocking(m,c)) return -1;
                }
            } else {
                assert(workers1.size() == workers2.size());
                for(size_t i=0;i<workers1.size();++i) {
                    if (!workers1[i]->init_input_blocking(m,c)) return -1;
                    workers2[i]->set_output_blocking(m,c);
                }
            }
            // -----------------------------------------------------------
        } // wraparound

        
        if (workers1[0]->isFarm() || workers1[0]->isAll2All()) {
            error("A2A, nodes of the first set cannot be farm or all-to-all\n");
            return -1;
        }
        if (workers2[0]->isFarm() || workers2[0]->isAll2All()) {
            error("A2A, nodes of the second set cannot be farm or all-to-all\n");
            return -1;
        }
        
        if (workers1[0]->isPipe()) {
            // all other Workers must be pipe
            for(size_t i=1;i<workers1.size();++i) {
                if (!workers1[i]->isPipe()) {
                    error("A2A, workers of the first set are not homogeneous, all of them must be of the same kind of building-block (e.g., all pipelines)\n");
                    return -1;
                }
            }
            
            if (!workers1[0]->isMultiOutput()) {
                error("A2A, workers of the first set can be pipelines but only if they are multi-output (automatic transformation NOT YET SUPPORTED)\n");
                return -1;
            }

            ff_node* last = ispipe_getlast(workers1[0]); // NOTE: we suppose homogeneous first set
            assert(last);
            if (last->isFarm() && !last->isOFarm()) { // standard farm ...
                if (!isfarm_withcollector(last)) { // ... with no collector
                    svector<ff_node*> w1;
                    last->get_out_nodes(w1);
                    if (!w1[0]->isMultiOutput()) { // NOTE: we suppose homogeneous workers
                        error("A2A, workers (farm/ofarm) of the first set are pipelines but their last stage is not multi-output (automatic transformation NOT YET SUPPORTED)\n");
                        return -1;
                    }
                }
            }
            // since by default the a2a is multi-output, we have to check if its workers
            // are multi-output
            if (last->isAll2All()) {
                svector<ff_node*> w1;
                last->get_out_nodes(w1);
                if (!w1[0]->isMultiOutput()) { // NOTE: we suppose homogeneous second set
                    error("A2A, workers (a2a) of the first set are pipelines but their last stage is not multi-output (automatic transformation NOT YET SUPPORTED)\n");
                    return -1;
                }
            }
        }
        if (workers2[0]->isPipe()) {
            // all other Workers must be pipe
            for(size_t i=1;i<workers2.size();++i) {
                if (!workers2[i]->isPipe()) {
                    error("A2A, workers of the second set are not homogeneous, all of them must be of the same kind (e.g., all pipelines)\n");
                    return -1;
                }
            }
            
            if (!workers2[0]->isMultiInput()) {
                error("A2A, workers of the second set can be pipelines but only if they are multi-input (automatic transformation NOT YET SUPPORTED)\n");
                return -1;
            }
            // since by default the a2a is multi-input, we have to check if its workers
            // are multi-input
            svector<ff_node*> w1;
            workers2[0]->get_in_nodes(w1);
            if (!w1[0]->isMultiInput()) { // NOTE: we suppose homogeneous second set
                error("A2A, workers of the second set are pipelines but their first stage is not multi-input (automatic transformation NOT YET SUPPORTED)\n");
                return -1;
            }
        }
        
        // checking L-Workers
        if (!workers1[0]->isMultiOutput()) {  // NOTE: we suppose all others to be the same
            // the nodes in the first set cannot be multi-input nodes without being
            // also multi-output
            if (workers1[0]->isMultiInput()) { // NOTE: we suppose all others to be the same
                error("A2A, the nodes of the first set cannot be multi-input nodes without being also multi-output (i.e., a composition of nodes). The node must be either standard node or multi-output node or compositions where the second stage is a multi-output node\n");
                return -1;
            }
            // it is a standard node or a pipeline with a standard node as last stage, so we transform it to a multi-output node
            for(size_t i=0;i<workers1.size();++i) {
                internal_mo_transformer *mo = new internal_mo_transformer(workers1[i], false);
                if (!mo) {
                    error("A2A, FATAL ERROR not enough memory\n");
                    return -1;
                }
                BARRIER_T *bar =workers1[i]->get_barrier();
                if (bar) mo->set_barrier(bar);
                workers1[i] = mo; // replacing old node
                internalSupportNodes.push_back(mo);
            }
        } 
        for(size_t i=0;i<workers1.size();++i) {
            if (ondemand_chunk && (workers1[i]->ondemand_buffer()==0)) {
                svector<ff_node*> w;
                workers1[i]->get_out_nodes(w);
                for(size_t k=0;k<w.size(); ++k)
                    w[k]->set_scheduling_ondemand(ondemand_chunk);                
                //workers1[i]->set_scheduling_ondemand(ondemand_chunk);
            }
            workers1[i]->set_id(int(i));
        }
        // checking R-Workers
        if (!workers2[0]->isMultiInput()) { // NOTE: we suppose that all others are the same        
            if (workers2[0]->isMultiOutput()) {
                error("A2A, the nodes of the second set cannot be multi-output nodes without being also multi-input (i.e., a composition of nodes). The node must be either standard node or multi-input node or compositions where the first stage is a multi-input node\n");
                return -1;
            }

            // here we have to transform the standard node into a multi-input node
            for(size_t i=0;i<workers2.size();++i) {
                internal_mi_transformer *mi = new internal_mi_transformer(workers2[i], false);
                if (!mi) {
                    error("A2A, FATAL ERROR not enough memory\n");
                    return -1;
                }
                BARRIER_T *bar =workers2[i]->get_barrier();
                if (bar) mi->set_barrier(bar);
                workers2[i] = mi; // replacing old node
                internalSupportNodes.push_back(mi);
            }
        }
        for(size_t i=0;i<workers2.size();++i) {
            workers2[i]->set_id(int(i));
        }
        
        size_t nworkers1 = workers1.size();
        size_t nworkers2 = workers2.size();

        {
            int ondemand = workers1[0]->ondemand_buffer();  // NOTE: here we suppose that all nodes in workers1 are homogeneous!

            svector<ff_node*> L;
            for(size_t i=0;i<nworkers1;++i)
                workers1[i]->get_out_nodes(L);
            if (L.size()==0) L=workers1;
            svector<ff_node*> R;
            for(size_t i=0;i<nworkers2;++i)
                workers2[i]->get_in_nodes(R);
            if (R.size()==0) R=workers2;

            for(size_t i=0;i<R.size(); ++i) {
                for(size_t j=0;j<L.size();++j) {
                    ff_node* t = new ff_buffernode(ondemand?ondemand:in_buffer_entries,ondemand?true:(fixedsizeIN|fixedsizeOUT), j);
                    assert(t);
                    internalSupportNodes.push_back(t);                    
                    L[j]->set_output(t);
                    R[i]->set_input(t);
                }
            }
        }
        if (outputNodes.size()) {

            svector<ff_node*> w;
            for(size_t i=0;i<workers2.size();++i) 
                workers2[i]->get_out_nodes(w);

            if (outputNodes.size() != w.size()) {
                error("A2A, prepare, invalid state detected\n");
                return -1;
            }

            for(size_t i=0;i<w.size(); ++i) {
                if (w[i]->isMultiOutput()) {
                    error("A2A, prepare, invalid state, unexpected multi-output node\n");
                    return -1;
                }

                assert(outputNodes[i]->get_in_buffer() != nullptr);
                if (w[i]->set_output_buffer(outputNodes[i]->get_in_buffer()) < 0)  {
                    error("A2A, prepare, invalid state, setting output buffer\n");
                    return -1;
                }
            }
        }


     
        // blocking stuff --------------------------------------------
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        for(size_t i=0;i<nworkers2;++i) {
            // initialize worker2 local cons_* stuff and sets all p_cons_*
            if (!workers2[i]->init_input_blocking(m,c)) return -1;
        }
        for(size_t i=0;i<nworkers1;++i) {
            // initialize worker1 local prod_* stuff and sets all p_prod_*
            if (!workers1[i]->init_output_blocking(m,c)) return -1;
        }
        // ------------------------------------------------------------            
        prepared = true;
        return 0;
    }

    void *svc(void*) { return FF_EOS; }

    
public:

    ff_a2a(bool notusedanymore=false,
           int in_buffer_entries=DEFAULT_BUFFER_CAPACITY,
           int out_buffer_entries=DEFAULT_BUFFER_CAPACITY,
           bool fixedsize=FF_FIXED_SIZE):prepared(false),fixedsizeIN(fixedsize),fixedsizeOUT(fixedsize),
                                 in_buffer_entries(in_buffer_entries),
                                 out_buffer_entries(out_buffer_entries)
    {}

    ff_a2a(const ff_a2a& p):prepared(false) {
        if (p.prepared) {
            error("ff_a2a, copy constructor, the input all-to-all has already been prepared\n");
            return;
        }

        workers1             = p.workers1;
        workers2             = p.workers2;
        workers1_cleanup     = p.workers1_cleanup;
        workers2_cleanup     = p.workers2_cleanup;
        fixedsizeIN          = p.fixedsizeIN;
        fixedsizeOUT         = p.fixedsizeOUT;
        in_buffer_entries    = p.in_buffer_entries;
        out_buffer_entries   = p.out_buffer_entries;
        wraparound           = p.wraparound;
        ondemand_chunk       = p.ondemand_chunk;
        outputNodes          = p.outputNodes;
        internalSupportNodes = p.internalSupportNodes;

        // this is a dirty part, we modify a const object.....
        ff_a2a* dirty= const_cast<ff_a2a*>(&p);
        dirty->internalSupportNodes.resize(0);
    }
    
    virtual ~ff_a2a() {
        if (barrier) delete barrier;
        for(size_t i=0;i<workers1.size();++i)
            workers1[i] = nullptr;        
        for(size_t i=0;i<workers2.size();++i) 
            workers2[i] = nullptr;
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes[i];
        }
    }

    /**
     * The nodes of the first set must be either standard ff_node or a node that is multi-output, 
     * e.g., a composition where the last stage is a multi-output node
     * 
     */
    template<typename T>
    int add_firstset(const std::vector<T*> & w, int ondemand=0, bool cleanup=false) {
        if (workers1.size()>0) {
            error("A2A, add_firstset cannot be called multiple times\n");
            return -1;
        }        
        if (w.size()==0) {
            error("A2A, try to add zero workers to the first set!\n");
            return -1; 
        }        
        for(size_t i=0;i<w.size();++i) {
            workers1.push_back(w[i]);
        }
        workers1_cleanup = cleanup;
        if (cleanup) {
            for(size_t i=0;i<w.size();++i) {
                internalSupportNodes.push_back(w[i]);
            }
        }
        ondemand_chunk   = ondemand;
        return 0;        
    }
    template<typename T>
    int change_firstset(const std::vector<T*>& w, int ondemand=0, bool cleanup=false, bool remove_from_cleanuplist=false) {
        if (remove_from_cleanuplist) {
            for(size_t j=0;j<workers1.size(); ++j) {
                int pos=-1;
                for(size_t i=0;i<internalSupportNodes.size();++i)
                    if (internalSupportNodes[i] == workers1[j]) { pos = i; break; }
                if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);
            }
        }
        workers1.clear();
        return add_firstset(w, ondemand, cleanup);
    }
    /**
     * The nodes of the second set must be either standard ff_node or a node that is multi-input.
     * 
     */
    template<typename T>
    int add_secondset(const std::vector<T*> & w, bool cleanup=false) {
        if (workers2.size()>0) {
            error("A2A, add_secondset cannot be called multiple times\n");
            return -1;
        }
        if (w.size()==0) {
            error("A2A, try to add zero workers to the second set!\n");
            return -1; 
        }        
        for(size_t i=0;i<w.size();++i) {
            workers2.push_back(w[i]);
        }
        workers2_cleanup = cleanup;
        if (cleanup) {
            for(size_t i=0;i<w.size();++i) {
                internalSupportNodes.push_back(w[i]);
            }
        }
        return 0;
    }
    template<typename T>
    int change_secondset(const std::vector<T*>& w, bool cleanup=false, bool remove_from_cleanuplist=false) {
        if (remove_from_cleanuplist) {
            for(size_t j=0;j<workers2.size(); ++j) {
                int pos=-1;
                for(size_t i=0;i<internalSupportNodes.size();++i)
                    if (internalSupportNodes[i] == workers2[j]) { pos = i; break; }
                if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);
            }                     
        }        
        workers2.clear();
        return add_secondset(w, cleanup);
    }

    bool change_node(ff_node* old, ff_node* n, bool cleanup=false, bool remove_from_cleanuplist=false) {
        if (prepared) {
            error("A2A, change_node cannot be called because the A2A has already been prepared\n");
            return false;
        }
        for(size_t i=0; i<workers1.size();++i) {
            if (workers1[i] == old) {
                if (remove_from_cleanuplist) {
                    int pos=-1;
                    for(size_t i=0;i<internalSupportNodes.size();++i)
                        if (internalSupportNodes[i] == old) { pos = i; break; }
                    if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);            
                }
                workers1[i] = n;
                if (cleanup) internalSupportNodes.push_back(n);
                return true;
            }
        }
        for(size_t i=0; i<workers2.size();++i) {
            if (workers2[i] == old) {
                if (remove_from_cleanuplist) {
                    int pos=-1;
                    for(size_t i=0;i<internalSupportNodes.size();++i)
                        if (internalSupportNodes[i] == old) { pos = i; break; }
                    if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);            
                }
                workers2[i] = n;
                if (cleanup) internalSupportNodes.push_back(n);                
                return true;
            }
        }
        
        return false;
    }
    
    
    void skipfirstpop(bool sk)   { 
        for(size_t i=0;i<workers1.size(); ++i)
            workers1[i]->skipfirstpop(sk);
        skip1pop=sk;
    }

#ifdef DFF_ENABLED
    void skipallpop(bool sk)   { 
        for(size_t i=0;i<workers1.size(); ++i)
            workers1[i]->skipallpop(sk);
        ff_node::skipallpop(sk);
    }
#endif

    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }

    void no_mapping() {
        default_mapping = false;
    }
    
    void no_barrier() {
        initial_barrier=false;
    }

    int cardinality() const { 
        int card=0;
        for(size_t i=0;i<workers1.size();++i) card += workers1[i]->cardinality();
        for(size_t i=0;i<workers2.size();++i) card += workers2[i]->cardinality();
        return card;
    }
    
    int run(bool skip_init=false) {
        if (!skip_init) {        
#if defined(FF_INITIAL_BARRIER)
            if (initial_barrier) {
                // set the initial value for the barrier 
                if (!barrier)  barrier = new BARRIER_T;
                const int nthreads = cardinality(barrier);
                if (nthreads > MAX_NUM_THREADS) {
                    error("PIPE, too much threads, increase MAX_NUM_THREADS !\n");
                    return -1;
                }
                barrier->barrierSetup(nthreads);
            }
#endif
            skipfirstpop(true);
        }
        if (!prepared) if (prepare()<0) return -1;
        
        const size_t nworkers1 = workers1.size();
        const size_t nworkers2 = workers2.size();
        
        for(size_t i=0;i<nworkers1; ++i) {
            workers1[i]->blocking_mode(blocking_in);
            if (!default_mapping) workers1[i]->no_mapping();
            if (workers1[i]->run(true)<0) {
                error("ERROR: A2A, running worker (first set) %d\n", i);
                return -1;
            }
        }
        for(size_t i=0;i<nworkers2; ++i) {
            workers2[i]->blocking_mode(blocking_in);
            if (!default_mapping) workers2[i]->no_mapping();
            if (workers2[i]->run(true)<0) {
                error("ERROR: A2A, running worker (second set) %d\n", i);
                return -1;
            }
        }
        return 0;
    }
    
    int wait() {
        int ret=0;
        const size_t nworkers1 = workers1.size();
        const size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers1; ++i)
            if (workers1[i]->wait()<0) {
                error("A2A, waiting Worker1 thread, id = %d\n",workers1[i]->get_my_id());
                ret = -1;
            } 
        for(size_t i=0;i<nworkers2; ++i)
            if (workers2[i]->wait()<0) {
                error("A2A, waiting Worker2 thread, id = %d\n",workers2[i]->get_my_id());
                ret = -1;
            } 
        
        return ret;
    }
    
#ifdef DFF_ENABLED
    int run_and_wait_end();
#else
    int run_and_wait_end() {
        if (isfrozen()) {  // TODO 
            error("A2A: Error: feature not yet supported\n");
            return -1;
        } 
        if (run()<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }
#endif

    /**
     * \brief checks if the node is running 
     *
     */
    bool done() const { 
        const size_t nworkers1 = workers1.size();
        const size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers1;++i) 
            if (!workers1[i]->done()) return false;
        for(size_t i=0;i<nworkers2;++i) 
            if (!workers2[i]->done()) return false;
        return true;
    }

    void remove_from_cleanuplist(const svector<ff_node*>& w) {
        for(size_t j=0;j<w.size(); ++j) {
            int pos=-1;
            for(size_t i=0;i<internalSupportNodes.size();++i)
                if (internalSupportNodes[i] == w[j]) { pos = i; break; }
            if (pos>=0) internalSupportNodes.erase(internalSupportNodes.begin()+pos);
        }
    }
    
    const svector<ff_node*>& getFirstSet() const  { return workers1; }
    const svector<ff_node*>& getSecondSet() const { return workers2; }

    int ondemand_buffer() const { return ondemand_chunk; }
    
    int numThreads() const { return cardinality(); }

    int set_output(const svector<ff_node *> & w) {
        outputNodes +=w;
        return 0; 
    }

    int set_output(ff_node *node) {
        outputNodes.push_back(node); 
        return 0;
    }

    
    void get_out_nodes(svector<ff_node*>&w) {
        for(size_t i=0;i<workers2.size();++i)
            workers2[i]->get_out_nodes(w);
        if (w.size() == 0)
            w += getSecondSet();
    }

    void get_out_nodes_feedback(svector<ff_node*>& w) {
        for(size_t i=0;i<workers2.size();++i)
            workers2[i]->get_out_nodes_feedback(w);
    }
   
    void get_in_nodes(svector<ff_node*>&w) {
        size_t len=w.size();
        for(size_t i=0;i<workers1.size();++i)
            workers1[i]->get_in_nodes(w);
        if (len == w.size())
            w += getFirstSet();
    }


    /**
     * \brief Feedback channel (pattern modifier)
     * 
     * The last stage output stream will be connected to the first stage 
     * input stream in a cycle (feedback channel)
     */
    int wrap_around() { wraparound=true; return 0;} 
    bool isset_wraparound() { return wraparound; }
    
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
    
    // time functions --------------------------------

    const struct timeval  getstarttime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers1.size(),zero);
        for(size_t i=0;i<workers1.size();++i)
            workertime[i]=workers1[i]->getstarttime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    const struct timeval  getwstartime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers1.size(),zero);
        for(size_t i=0;i<workers1.size();++i)
            workertime[i]=workers1[i]->getwstartime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    
    const struct timeval  getstoptime()  const {
        const struct timeval zero={0,0};        
        std::vector<struct timeval > workertime(workers2.size(),zero);
        for(size_t i=0;i<workers2.size();++i)
            workertime[i]=workers2[i]->getstoptime();
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    const struct timeval  getwstoptime() const {
        const struct timeval zero={0,0};
        std::vector<struct timeval > workertime(workers2.size(),zero);
        for(size_t i=0;i<workers2.size();++i) {
            workertime[i]=workers2[i]->getwstoptime();
        }
        std::vector<struct timeval >::iterator it=
            std::max_element(workertime.begin(),workertime.end(),time_compare);
        return (*it);
    }
    
    double ffTime() { return diffmsec(getstoptime(),getstarttime()); }

    double ffwTime() { return diffmsec(getwstoptime(),getwstartime()); }

#if defined(TRACE_FASTFLOW)
    void ffStats(std::ostream & out) { 
        out << "--- a2a:\n";
        out << "--- L-Workers:\n";
        for(size_t i=0;i<workers1.size();++i) workers1[i]->ffStats(out);
        out << "--- R-Workers:\n";
        for(size_t i=0;i<workers2.size();++i) workers2[i]->ffStats(out);
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
    bool isMultiInput()  const { return true;}
    bool isMultiOutput() const { return true;}
    bool isAll2All()     const { return true; }    

    int create_input_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i)
            if (workers1[i]->create_input_buffer(nentries,fixedsize)==-1) return -1;
        return 0;
    }

    int create_output_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        int id=0;
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            if (workers2[i]->isMultiOutput()) {
                svector<ff_node*> w(1);
                workers2[i]->get_out_nodes(w);
                assert(w.size());
                for(size_t j=0;j<w.size();++j) {
                    ff_node* t = new ff_buffernode(nentries,fixedsize); 
                    t->set_id(id++);
                    internalSupportNodes.push_back(t);
                    if (w[j]->isMultiOutput()) {
                        if (w[j]->set_output(t)<0) return -1;
                    } else {
                        if (workers2[i]->set_output(t)<0) return -1;
                    }
                }
            } else{ 
                if (workers2[i]->create_output_buffer(nentries,fixedsize)==-1) return -1;
                id++;
            }
        }
        return 0;        
    }
    
    bool init_input_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             bool /*feedback*/=true) {
        size_t nworkers1 = workers1.size();
        for(size_t i=0;i<nworkers1; ++i) {
            pthread_mutex_t   *m1        = NULL;
            pthread_cond_t    *c1        = NULL;
            if (!workers1[i]->init_input_blocking(m1,c1)) return false;
            if (nworkers1==1) { m=m1; c=c1; }
        }
        return true;
    }
    bool init_output_blocking(pthread_mutex_t   *&,
                              pthread_cond_t    *&,
                              bool /*feedback*/=true) {
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            pthread_mutex_t   *m        = NULL;
            pthread_cond_t    *c        = NULL;
            if (!workers2[i]->init_output_blocking(m,c)) return false;
        }
        return true;
    }
    void set_output_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             bool canoverwrite=false) {
        size_t nworkers2 = workers2.size();
        for(size_t i=0;i<nworkers2; ++i) {
            workers2[i]->set_output_blocking(m,c, canoverwrite);
        }
    }
   
protected:
    bool workers1_cleanup=false;
    bool workers2_cleanup=false;
    bool prepared, fixedsizeIN, fixedsizeOUT;
    bool wraparound=false;
    int in_buffer_entries, out_buffer_entries;
    int ondemand_chunk=0;
    svector<ff_node*>  workers1;  // first set, nodes must be multi-output
    svector<ff_node*>  workers2;  // second set, nodes must be multi-input
    svector<ff_node*>  outputNodes;
    svector<ff_node*>  internalSupportNodes;
};
    
} // namespace ff

#endif /* FF_A2A_HPP */
