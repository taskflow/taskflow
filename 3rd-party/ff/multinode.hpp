/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file multinode.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow ff_minode ff_monode and typed versions. 
 *
 * @detail FastFlow multi-input and multi-output nodes.
 *
 */

#ifndef FF_MULTINODE_HPP
#define FF_MULTINODE_HPP

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
#include <ff/lb.hpp>
#include <ff/gt.hpp>

namespace ff {

/* This file provides the following classes:
 *   ff_minode
 *   ff_monode
 *   ff_minode_t (typed version of the ff_minode -- requires c++11)
 *   ff_monode_t (typed version of the ff_monode -- requires c++11)
 *
 */

/* ************************* Multi-Input node ************************* */

/*!
 * \ingroup building_blocks
 *
 * \brief Multiple input ff_node (the SPMC mediator)
 *
 * The ff_node with many input channels.
 *
 * This class is defined in \ref farm.hpp
 */

class ff_minode: public ff_node {
    friend class ff_farm;
    friend class ff_comb;
protected:
    /**
     * \brief Gets the number of input channels
     */
    inline int cardinality(BARRIER_T * const barrier)  { 
        this->set_barrier(barrier);
        return 1;
    }

    /**
     * \brief Creates the input channels
     *
     * This function may be called because the multi-input node is  
     * used just as a standard node (for example as a farm's worker).
     *
     * \return >=0 if successful, otherwise -1 is returned.
     */
    int create_input_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        assert(inputNodes.size() == 0);

        ff_node* t = new ff_buffernode(nentries,fixedsize); 
        t->set_id(-1);
        internalSupportNodes.push_back(t);
        set_input(t);
        return ff_node::set_input_buffer(t->get_in_buffer());
    }

    /* The multi-input node is used as a standard node. 
     */
    inline bool  put(void * ptr) { 
        assert(inputNodes.size() == 1);
        return inputNodes[0]->put(ptr);
    }
    inline FFBUFFER *get_in_buffer() const {
        if (inputNodes.size() == 0) return nullptr;
        assert(inputNodes.size() == 1);
        return inputNodes[0]->get_in_buffer();
    }

    
    int dryrun() {
        if (prepared) return 0;
        for(size_t i=0;i<inputNodesFeedback.size();++i)
            gt->register_worker(inputNodesFeedback[i]);        
        if (inputNodesFeedback.size()>0)
            gt->set_feedbackid_threshold(inputNodesFeedback.size());        
        for(size_t i=0;i<inputNodes.size();++i)
            gt->register_worker(inputNodes[i]);
        if (gt->dryrun()<0) return -1;
        return 0;
    }
    
    int prepare() {
        if (prepared) return 0;
        if (ff_minode::dryrun()<0) return -1;
        prepared=true;
        return 0;
    }
        
    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
        gt->blocking_mode(blk);
    }
    template<typename T>
    int all_gather(T* in, T** V) { return gt->all_gather(in,(void**)V); }

    void registerAllGatherCallback(int (*cb)(void *,void **,void*), void * arg) {
        gt->registerAllGatherCallback(cb,arg);
    }
    
    // consumer
    virtual inline bool init_input_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool feedback=true) {
        bool r = gt->init_input_blocking(m,c, feedback);
        if (!r) return false;
        // NOTE: for all registered input node (or buffernode) we have to set the  
        // blocking stuff both for input nodes as well as "feedback nodes"
        for(size_t i=0;i<inputNodes.size(); ++i) 
            inputNodes[i]->set_output_blocking(m,c);
        if (feedback) {
            for(size_t i=0;i<inputNodesFeedback.size(); ++i) 
                inputNodesFeedback[i]->set_output_blocking(m,c);
        }
        return true;
    }
    // producer
    virtual inline bool init_output_blocking(pthread_mutex_t   *&m,
                                             pthread_cond_t    *&c,
                                             bool feedback=true) {
        // This is a multi-input node, so it has only one output channel
        // unless it is a combine node whose right part is a multi-output
        // node. If this is not the case, we have to initialize the
        // local-node output blocking
        ff_node* filter = gt->get_filter();
        if (filter &&
            ( (filter->get_out_buffer()!=nullptr) || filter->isMultiOutput() ) )  { // see gt.hpp
            return filter->init_output_blocking(m, c, feedback);
        }
        return ff_node::init_output_blocking(m,c, feedback);
        //return gt->init_output_blocking(m,c);
    }
    virtual inline void set_output_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool canoverwrite=false) {

        ff_node* filter = gt->get_filter();
        if (filter &&
            ( (filter->get_out_buffer()!=nullptr) || filter->isMultiOutput() ) )  { // see gt.hpp
            filter->set_output_blocking(m, c, canoverwrite);
        }
        //gt->set_output_blocking(m,c);
        ff_node::set_output_blocking(m,c, canoverwrite);
    }

    inline pthread_cond_t    &get_cons_c()  { return gt->get_cons_c(); }
    
    virtual inline void get_in_nodes(svector<ff_node*>&w) {
        size_t len=w.size();
        // it is possible that the multi-input node is register
        // as collector of farm
        if (inputNodes.size() == 0 && gt->getNWorkers()>0) {
            w += gt->getWorkers();
        }
        w += inputNodes;
        
        if (len == w.size())  w.push_back(this);
    }

    virtual void get_in_nodes_feedback(svector<ff_node*>&w) {
        w += inputNodesFeedback;
    }
    
    virtual inline void get_out_nodes(svector<ff_node*>&w) {
        w.push_back(this);
    }
    
public:
    /**
     * \brief Constructor
     */
    ff_minode(int max_num_workers=DEF_MAX_NUM_WORKERS):
        ff_node(), gt(new ff_gatherer(max_num_workers)),myowngt(true) { }

    ff_minode(ff_node *filter, int max_num_workers=DEF_MAX_NUM_WORKERS):
        ff_node(), gt(new ff_gatherer(max_num_workers)),myowngt(true) {
        if (filter == nullptr) {
            delete gt;
            gt = nullptr;
            return;
        }
        gt->set_filter(filter);
    }
    ff_minode(const ff_minode& n) : ff_node(n) {
        // here we re-initialize a new gatherer
        gt = new ff_gatherer(n.gt->max_nworkers);
        if (!gt) {
            error("ff_minode, not enough memory\n");
            return;
        }
        if (n.gt->get_filter())
            gt->set_filter(n.gt->get_filter());
        myowngt=true;

        inputNodes=n.inputNodes;
        inputNodesFeedback=n.inputNodesFeedback;
        internalSupportNodes = n.internalSupportNodes;
                
        // this is a dirty part, we modify a const object.....
        ff_minode *dirty= const_cast<ff_minode*>(&n);
        dirty->internalSupportNodes.resize(0);
    }
    
    /**
     * \brief Destructor 
     */
    virtual ~ff_minode() { 
        if (gt && myowngt) delete gt;
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes[i];
        }
    }

    int set_filter(ff_node *filter) {
        return gt->set_filter(filter);
    }

    // used when a multi-input node is a filter of a collector or of a comp
    // it can also be used to set a particular gathering policy like for
    // example the ones pre-defined in the file ordering_policies.hpp
    void setgt(ff_gatherer *external_gt, bool cleanup=false) {
        if (myowngt) {
            delete gt;
            myowngt=false;
        }
        gt = external_gt;
        myowngt = cleanup;
    }

    
    inline void set_barrier(BARRIER_T * const barrier) {
        gt->set_barrier(barrier);
    }
    inline BARRIER_T* get_barrier() const { return gt->get_barrier(); }
    
    /**
     * \brief Assembly input channels
     *
     * Assembly input channelnames to ff_node channels
     */
    virtual inline int set_input(const svector<ff_node *> & w) { 
        inputNodes += w;
        return 0; 
    }

    void set_input_channelid(ssize_t id, bool fromin=true) {
        gt->set_input_channelid(id, fromin);
    }
    
    /**
     * \brief Assembly a input channel
     *
     * Assembly a input channelname to a ff_node channel
     */
    virtual inline int set_input(ff_node *node) { 
        inputNodes.push_back(node); 
        return 0;
    }

    virtual inline int set_input_feedback(ff_node *node) { 
        inputNodesFeedback.push_back(node); 
        return 0;
    }
    
    virtual bool isMultiInput() const { return true;}


    void set_running(int r) {
        gt->running = r;
    }
    
    /**
     * \brief Skip first pop
     *
     * Set up spontaneous start
     */
    inline void skipfirstpop(bool sk)   {
        gt->skipfirstpop(sk);
        ff_node::skipfirstpop(sk);
    }

#ifdef DFF_ENABLED
    inline void skipallpop(bool sk) {
        gt->skipallpop(sk);
        ff_node::skipallpop(sk);
    }
#endif

    /**
     * \brief run
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    int run(bool=false) {
        if (!gt) return -1;
        if (gt->get_filter() == nullptr)
            gt->set_filter(this);

        if (!prepared) if (prepare()<0) return -1;

        if (ff_node::skipfirstpop()) gt->skipfirstpop();
        if (!default_mapping) gt->no_mapping();
        if (gt->run()<0) {
            error("ff_minode, running gather module\n");
            return -1;
        }
        return 0;
    }
    int freeze_and_run(bool=false) { 
        gt->freeze();
        return run();
    }
    int run_then_freeze(ssize_t nw=-1) {
        if (gt->isfrozen()) {
            // true means that next time threads are frozen again
            gt->thaw(true, nw); 
            return 0;
        }
        gt->freeze();
        return run();
    }

    int  wait(/* timeout */) { 
        if (gt->wait()<0) return -1;
        return 0;
    }
    
    int wait_freezing() { return gt->wait_freezing(); }


    /**
     * \brief checks if the node is running 
     *
     */
    bool done() const { 
        return gt->done();
    }
    
    bool fromInput() const { return gt->fromInput(); }
    
    /**
     * \brief Gets the channel id from which the data has just been received
     *
     */
    ssize_t get_channel_id() const { return gt->get_channel_id();}

    size_t get_num_inchannels()       const { return gt->get_num_inchannels();  } 
    size_t get_num_outchannels()      const {
        if (gt->get_filter() == (ff_node*)this)
            return (gt->get_out_buffer()?1:0);
            
        return gt->get_num_outchannels();
    }
    size_t get_num_feedbackchannels() const {
        return gt->get_num_feedbackchannels();
    }
    
    /**
     * For a multi-input node the number of EOS to receive before terminating is equal to 
     * the current number of input channels.
     */
    ssize_t get_neos() const { return get_num_inchannels(); }
    
    /**
     * \internal
     * \brief Gets the gt
     *
     * It gets the internal gatherer.
     *
     * \return A pointer to the FastFlow gatherer.
     */
    inline ff_gatherer *getgt() const { return gt;}

    const struct timeval getstarttime() const { return gt->getstarttime();}
    const struct timeval getstoptime()  const { return gt->getstoptime();}
    const struct timeval getwstartime() const { return gt->getwstartime();}
    const struct timeval getwstoptime() const { return gt->getwstoptime();}    

    
#if defined(TRACE_FASTFLOW) 
    /**
     * \brief Prints the FastFlow trace
     *
     * It prints the trace of FastFlow.
     */
    inline void ffStats(std::ostream & out) {
        out << "--- multi-input:\n";
        gt->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

    
private:
    ff_gatherer* gt;
    bool myowngt;
    svector<ff_node*> inputNodes;
    svector<ff_node*> inputNodesFeedback;
    svector<ff_node*> internalSupportNodes;
};


    /* ************************* Multi-Ouput node ************************* */

/*!
 *  \ingroup building_blocks
 *
 * \brief Multiple output ff_node (the MPSC mediator)
 *
 * The ff_node with many output channels.
 *
 * This class is defined in \ref farm.hpp
 */

class ff_monode: public ff_node {
    friend class ff_a2a;
protected:
    /**
     * \brief Cardinatlity
     *
     * Defines the cardinatlity of the FastFlow node.
     *
     * \param barrier defines the barrier
     *
     * \return 1 is always returned.
     */
    inline int   cardinality(BARRIER_T * const barrier)  { 
        this->set_barrier(barrier);
        return 1;
    }

    int create_output_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {

        // this is needed for example when a worker of a farm is a multi-output node
        // (even without a feedback channel)
        
        if (ff_node::create_output_buffer(nentries,fixedsize) <0) return -1;
        ff_node *t = new ff_buffernode(-1, get_out_buffer(), get_out_buffer());
        assert(t);
        internalSupportNodes.push_back(t);
        set_output(t);
        return 0;
    }

    int dryrun() {
        if (prepared) return 0;
        for(size_t i=0;i<outputNodesFeedback.size();++i)
            lb->register_worker(outputNodesFeedback[i]);
        if (outputNodesFeedback.size()>0)
            lb->set_feedbackid_threshold(outputNodesFeedback.size());        
        for(size_t i=0;i<outputNodes.size();++i)
            lb->register_worker(outputNodes[i]);
        if (lb->dryrun()<0) return -1;
        return 0;
    }
    
    int prepare() {
        if (prepared) return 0;
        if (ff_monode::dryrun()<0) return -1;
        prepared=true;
        return 0;
    }

    void propagateEOS(void*task=FF_EOS) {
        if (lb->getnworkers() == 0) ff_send_out(task);
        lb->propagateEOS(task);
    }
        
    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
        lb->blocking_mode(blk);
    }
    
    // consumer
    virtual inline bool init_input_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool feedback=true) {
        return lb->init_input_blocking(m,c, feedback);
    }
    // producer
    virtual inline bool init_output_blocking(pthread_mutex_t   *&m,
                                             pthread_cond_t    *&c,
                                             bool feedback=true) {
        return lb->init_output_blocking(m,c, feedback);
    }
    virtual inline void set_output_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool canoverwrite=false) {
        ff_node::set_output_blocking(m,c, canoverwrite);
    }

    virtual inline void  set_cons_c(pthread_cond_t *c) {
        lb->set_cons_c(c);
    }        
    virtual inline pthread_cond_t    &get_cons_c()        { return lb->get_cons_c();}

public:
    /**
     * \brief Constructor
     *
     * \param max_num_workers defines the maximum number of workers
     *
     */
    ff_monode(int max_num_workers=DEF_MAX_NUM_WORKERS):
        ff_node(), lb(new ff_loadbalancer(max_num_workers)), myownlb(true) {
    }

    ff_monode(ff_node *filter, int max_num_workers=DEF_MAX_NUM_WORKERS):
        ff_node(), lb(new ff_loadbalancer(max_num_workers)),myownlb(true) {
        if (filter == nullptr) {
            delete lb;
            lb = nullptr;
            return;
        }
        lb->set_filter(filter);
    }

    ff_monode(const ff_monode& n) : ff_node(n) {
        // here we re-initialize a new gatherer
        lb = new ff_loadbalancer(n.lb->max_nworkers);
        if (!lb) {
            error("ff_monode, not enough memory\n");
            return;
        }
        if (n.lb->get_filter()) 
            lb->set_filter(n.lb->get_filter());
        myownlb=true;

        outputNodes=n.outputNodes;
        outputNodesFeedback=n.outputNodesFeedback;
        internalSupportNodes = n.internalSupportNodes;
                
        // this is a dirty part, we modify a const object.....
        ff_monode *dirty= const_cast<ff_monode*>(&n);
        dirty->internalSupportNodes.resize(0);
    }

    
    /**
     * \brief Destructor 
     */
    virtual ~ff_monode() {
        if (lb && myownlb) delete lb;
        for(size_t i=0;i<internalSupportNodes.size();++i) {
            delete internalSupportNodes[i];
        }
    }

    void set_scheduling_ondemand(const int inbufferentries=1) { 
        if (inbufferentries<0) ondemand=1;
        else ondemand=inbufferentries;
    }
    int ondemand_buffer() const { return ondemand; } 

    
    int set_filter(ff_node *filter) {
        return lb->set_filter(filter);
    }

    inline void set_barrier(BARRIER_T * const barrier) {
        lb->set_barrier(barrier);
    }
    inline BARRIER_T* get_barrier() const { return lb->get_barrier(); }

    
    /**
     * \brief Assembly the output channels
     *
     * Attach output channelnames to ff_node channels
     */
    virtual inline int set_output(const svector<ff_node *> & w) {
        outputNodes += w;
        return 0; 
    }

    /**
     * \brief Assembly an output channels
     *
     * Attach a output channelname to ff_node channel
     */
    virtual inline int set_output(ff_node *node) { 
        outputNodes.push_back(node); 
        return 0;
    }

    /**
     * \brief Assembly an output channels
     *
     * Attach a output channelname to ff_node channel
     */
    virtual inline int set_output_feedback(ff_node *node) { 
        outputNodesFeedback.push_back(node); 
        return 0;
    }


    virtual bool isMultiOutput() const { return true;}

    virtual inline void get_out_nodes(svector<ff_node*>&w) {
        // it is possible that the multi-output node is register
        // as emitter of farm
        if (outputNodes.size() == 0) {
            if (lb->getNWorkers()>0) w += lb->getWorkers();
            else
                w.push_back(this);
            return;
        }
        w += outputNodes;
    }
    virtual inline void get_out_nodes_feedback(svector<ff_node*>&w) {
        w += outputNodesFeedback;
    }

    void set_running(int r) {
        lb->running = r;
    }
    
    /**
     * \brief Skips the first pop
     *
     * Set up spontaneous start
     */
    inline void skipfirstpop(bool sk)   {
        lb->skipfirstpop(sk);
        ff_node::skipfirstpop(sk);
    }

#ifdef DFF_ENABLED
    void skipallpop(bool sk) {
        lb->skipallpop(sk);
        ff_node::skipallpop(sk);
    }
    void set_virtual_outchannels(int n){
        noutchannels=n;
    }
    void set_virtual_feedbackchannels(int n) {
        nfeedbackchannels=n;
    }
    
#endif

    /**
     * \brief Provides the next channel id that will be selected for sending out the task
     *  
     */    
    int get_next_free_channel(bool forever=true) {
        return lb->get_next_free_channel(forever);
    }
    
    /**
     * \brief Sends one task to a specific node id.
     *
     * \return true if successful, false otherwise
     */
    virtual inline bool ff_send_out_to(void *task, int id, unsigned long retry=((unsigned long)-1),
                                       unsigned long ticks=(ff_node::TICKS2WAIT)) {
        // NOTE: this callback should be set only if the multi-output node is part of
        // a composition and the node is not the last stage
        if (callback) return  callback(task,id, retry,ticks, callback_arg);
        assert(id>=0);
        return lb->ff_send_out_to(task,id,retry,ticks);
    }
    
    inline bool ff_send_out(void * task, int id=-1,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=(ff_node::TICKS2WAIT)) {
        // NOTE: this callback should be set only if the multi-output node is part of
        // a composition and it is not the last stage 
        if (callback) return  callback(task,id, retry,ticks,callback_arg);
        return lb->schedule_task(task,retry,ticks);
    }

    // TODO: broadcast_task should have callback as in ff_send_out
    //
    inline void broadcast_task(void *task) {
        lb->broadcast_task(task);
    }

    
    /**
     * \brief run
     *
     * \param skip_init defines if the initilization should be skipped
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int run(bool /*skip_init*/=false) {
        if (!lb) return -1;
        if (lb->get_filter() == nullptr)
            lb->set_filter(this);

        if (!prepared) if (prepare()<0) return -1;
       
        if (ff_node::skipfirstpop()) lb->skipfirstpop(true);
        if (!default_mapping) lb->no_mapping();
        if (lb->runlb()<0) {
            error("ff_monode, running loadbalancer module\n");
            return -1;
        }
        return 0;
    }

    int freeze_and_run(bool=false) {
        lb->freeze();
        return run(true);
    }

    int  wait(/* timeout */) { 
        if (lb->waitlb()<0) return -1;
        return 0;
    }

    /**
     * \brief checks if the node is running 
     *
     */
    bool done() const { 
        return lb->done();
    }
    
    /**
     * \internal
     * \brief Gets the internal lb (Emitter)
     *
     * It gets the internal lb (Emitter)
     *
     * \return A pointer to the lb
     */
    inline ff_loadbalancer * getlb() const { return lb;}

    // used when a multi-output node is used as a filter in the emitter of a farm 
    // it can also be used to set a particular scheduling policy like for
    // example the ones pre-defined in the file ordering_policies.hpp
    void setlb(ff_loadbalancer *elb, bool cleanup=false) {
        if (lb && myownlb) {
            delete lb;
            myownlb = false;
        }
        lb = elb;
        myownlb=cleanup;
    }

    /**
     * \brief Gets the channel id from which the data has just been received
     *
     */
    ssize_t get_channel_id() const { return lb->get_channel_id();}
    
    size_t get_num_feedbackchannels() const {
#ifdef DFF_ENABLED        
        if (nfeedbackchannels!=-1) return nfeedbackchannels;
#endif        
        return lb->get_num_feedbackchannels();
    }
    size_t get_num_outchannels() const      {
#ifdef DFF_ENABLED        
        if (noutchannels!=-1) return noutchannels;
#endif        
        return lb->get_num_outchannels();
    }
    size_t get_num_inchannels()  const      { return lb->get_num_inchannels(); }
    
    const struct timeval getstarttime() const { return lb->getstarttime();}
    const struct timeval getstoptime()  const { return lb->getstoptime();}
    const struct timeval getwstartime() const { return lb->getwstartime();}
    const struct timeval getwstoptime() const { return lb->getwstoptime();}    

    
#if defined(TRACE_FASTFLOW) 
    /*
     * \brief Prints the FastFlow trace
     *
     * It prints the trace of FastFlow.
     */
    inline void ffStats(std::ostream & out) {
        out << "--- multi-output:\n";
        lb->ffStats(out);
    }
#else
    void ffStats(std::ostream & out) { 
        out << "FastFlow trace not enabled\n";
    }
#endif

    
protected:
    ff_loadbalancer* lb;
    bool myownlb;
    int  ondemand=0;
#ifdef DFF_ENABLED    
    int  noutchannels=-1;
    int  nfeedbackchannels=-1;
#endif     
    svector<ff_node*> outputNodes;
    svector<ff_node*> outputNodesFeedback;
    svector<ff_node*> internalSupportNodes;    
};


/* ************************* Multi-Input and Multi-Output node ************************* */
/*                   (typed version based on ff_minode and ff_monode )                   */

/*!
 *  \class ff_minode_t
 *  \ingroup building_blocks
 *
 *  \brief Typed multiple input ff_node (the SPMC mediator).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template<typename IN_t, typename OUT_t = IN_t>
struct ff_minode_t: ff_minode {
    typedef IN_t  in_type;
    typedef OUT_t out_type;
    ff_minode_t():
        GO_ON((OUT_t*)FF_GO_ON),
        EOS((OUT_t*)FF_EOS),EOSW((OUT_t*)FF_EOSW),
        GO_OUT((OUT_t*)FF_GO_OUT),
        EOS_NOFREEZE((OUT_t*) FF_EOS_NOFREEZE) {
#ifdef DFF_ENABLED

        /* WARNING: 
         *    the definition of functions alloctaskF, freetaskF, serializeF, deserializeF
         *    IS DUPLICATED for the ff_node_t (see file node.hpp).
         *
         */        
     if constexpr (traits::has_alloctask_v<IN_t>) {        
         this->alloctaskF = [](char* ptr, size_t sz) -> void* {
                                IN_t* p = nullptr;
                                alloctaskWrapper<IN_t>(ptr, sz, p);
                                assert(p);
                                return p;
                           };
     } else {
         this->alloctaskF = [](char*, size_t ) -> void* {
                               IN_t* o = new IN_t;
                               assert(o);
                               return o;
                           };
     }
        
     if constexpr (traits::has_freetask_v<OUT_t>) {
        this->freetaskF = [](void* o) {
                              freetaskWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o));
                          };

     } else {
         this->freetaskF = [](void* o) { delete reinterpret_cast<OUT_t*>(o); };
     }

        
     // check on Serialization capabilities on the OUTPUT type!
     if constexpr (traits::is_serializable_v<OUT_t>){
        this->serializeF = [](void* o, dataBuffer& b) -> bool {
                               bool datacopied=true;
                               std::pair<char*, size_t> p = serializeWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o), datacopied);
                               b.setBuffer(p.first, p.second);
                               return datacopied;
                           };
    } else if constexpr (cereal::traits::is_output_serializable<OUT_t, cereal::PortableBinaryOutputArchive>::value) {
        this->serializeF = [](void* o, dataBuffer& b) -> bool {
                               std::ostream oss(&b);
                               cereal::PortableBinaryOutputArchive ar(oss);
                               ar << *reinterpret_cast<OUT_t*>(o);
                               return true;
                           };
    }
    
    // check on Serialization capabilities on the INPUT type!
    if constexpr (traits::is_deserializable_v<IN_t>){
        this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                                 IN_t* ptr=(IN_t*)this->alloctaskF(b.getPtr(), b.getLen());
                                 datacopied = deserializeWrapper<IN_t>(b.getPtr(), b.getLen(), ptr);
                                 assert(ptr);
                                 return ptr;
                             };
    } else if constexpr(cereal::traits::is_input_serializable<IN_t, cereal::PortableBinaryInputArchive>::value) {
            this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                                     std::istream iss(&b);
                                     cereal::PortableBinaryInputArchive ar(iss);
                                     IN_t* o = (IN_t*)this->alloctaskF(nullptr,0);
                                     assert(o);
                                     ar >> *o;
                                     datacopied = true;
                                     return o;
                                 };
    }
#endif
	}
    OUT_t * const GO_ON,  *const EOS, *const EOSW, *const GO_OUT, *const EOS_NOFREEZE;
    virtual ~ff_minode_t()  {}
    virtual OUT_t* svc(IN_t*)=0;
    inline  void *svc(void *task) { return svc(reinterpret_cast<IN_t*>(task)); };
    inline  int all_gather(IN_t* in, std::vector<IN_t*>& V) {
        size_t nw = get_num_inchannels();
        svector<IN_t*> v(nw);
        v.resize(nw);
        for(size_t i=0;i<v.size();++i) v[i]=nullptr;
        IN_t **data = v.begin();
        int r = ff_minode::all_gather(in,data);
        V.resize(nw);
        for(size_t i=0;i<v.size();++i) V[i]=v[i];
        return r;
    }
};

/*!
 *  \class ff_monode_t
 *  \ingroup building_blocks
 *
 *  \brief Typed multiple output ff_node (the MPSC mediator).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template<typename IN_t, typename OUT_t = IN_t>
struct ff_monode_t: ff_monode {
    typedef IN_t  in_type;
    typedef OUT_t out_type;
    ff_monode_t():
        GO_ON((OUT_t*)FF_GO_ON),
        EOS((OUT_t*)FF_EOS),EOSW((OUT_t*)FF_EOSW),
        GO_OUT((OUT_t*)FF_GO_OUT),
        EOS_NOFREEZE((OUT_t*) FF_EOS_NOFREEZE) {
#ifdef DFF_ENABLED

     if constexpr (traits::has_alloctask_v<IN_t>) {        
        this->alloctaskF = [](char* ptr, size_t sz) -> void* {
                               IN_t* p = nullptr;
                               alloctaskWrapper<IN_t>(ptr, sz, p);
                               assert(p);
                               return p;
                           };
     } else {
         this->alloctaskF = [](char*, size_t) -> void* {
                               IN_t* o = new IN_t;
                               assert(o);
                               return o;
                           };
     }
        
     if constexpr (traits::has_freetask_v<OUT_t>) {
        this->freetaskF = [](void* o) {
                              freetaskWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o));
                          };

     } else {
         this->freetaskF = [](void* o) { delete reinterpret_cast<OUT_t*>(o); };
     }

        
    // check on Serialization capabilities on the OUTPUT type!
    if constexpr (traits::is_serializable_v<OUT_t>){
        this->serializeF = [](void* o, dataBuffer& b) -> bool {
                               bool datacopied=true;
                               std::pair<char*, size_t> p = serializeWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o),datacopied);
                               b.setBuffer(p.first, p.second);
                               return datacopied;
                           };
    } else if constexpr (cereal::traits::is_output_serializable<OUT_t, cereal::PortableBinaryOutputArchive>::value) {
            this->serializeF = [](void* o, dataBuffer& b) -> bool {
                                   std::ostream oss(&b);
                                   cereal::PortableBinaryOutputArchive ar(oss);
                                   ar << *reinterpret_cast<OUT_t*>(o);
                                   return true;
                               };
        }
    
    // check on Serialization capabilities on the INPUT type!
    if constexpr (traits::is_deserializable_v<IN_t>){
        this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                                 IN_t* ptr=(IN_t*)this->alloctaskF(b.getPtr(), b.getLen());
                                 datacopied = deserializeWrapper<IN_t>(b.getPtr(), b.getLen(), ptr);
                                 assert(ptr);
                                 return ptr;
                             };
    } else if constexpr(cereal::traits::is_input_serializable<IN_t, cereal::PortableBinaryInputArchive>::value){
            this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                                     std::istream iss(&b);cereal::PortableBinaryInputArchive ar(iss);
                                     IN_t* o = (IN_t*)this->alloctaskF(nullptr,0);
                                     assert(o);
                                     ar >> *o;
                                     datacopied = true;
                                     return o;
                                 };
    }
#endif
	}
    OUT_t * const GO_ON,  *const EOS, *const EOSW, *const GO_OUT, *const EOS_NOFREEZE;
    virtual ~ff_monode_t()  {}
    virtual OUT_t* svc(IN_t*)=0;
    inline  void *svc(void *task) { return svc(reinterpret_cast<IN_t*>(task)); };
};


/**
 *   Transforms a standard node into a multi-output node 
 */
struct internal_mo_transformer: ff_monode {
    internal_mo_transformer(ff_node* n, bool cleanup=false):
        ff_monode(n),cleanup(cleanup),n(n) {
    }

    template<typename T>
    internal_mo_transformer(const T& _n) {
        T *t = new T(_n);
        assert(t);
        n = t;
        cleanup=true;
        ff_monode::set_filter(n);
    }
    internal_mo_transformer(const internal_mo_transformer& t) : ff_monode(t) {
        cleanup=t.cleanup;
        n = t.n;
        ff_monode::set_filter(n);

        // this is a dirty part, we modify a const object.....
        internal_mo_transformer *dirty= const_cast<internal_mo_transformer*>(&t);
        dirty->cleanup=false;
    }
    ~internal_mo_transformer() {
        if (cleanup && n) {
            delete n;
            n=nullptr;
        }
    }
    
    inline int svc_init() { return n->svc_init(); }
    inline void* svc(void* task) { return n->svc(task);}
    inline void svc_end() { return n->svc_end(); }
    inline void eosnotify(ssize_t id) { n->eosnotify(id); }

    int create_input_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        int r= ff_monode::create_input_buffer(nentries,fixedsize);
        if (r<0) return -1;
        ff_monode::getlb()->get_filter()->set_input_buffer(ff_monode::get_in_buffer());
        return 0;
    }    
    
    void set_id(ssize_t id) {
        if (n) n->set_id(id);
        ff_monode::set_id(id);
    }

    void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
        n->registerCallback(cb,arg);
    }

    int dryrun() {
        if (prepared) return 0;
        if (n) {
            if (!n->callback)
                n->registerCallback(ff_send_out_motransformer, this);
            if (n->isMultiOutput()) {
                assert(!n->isComp());
                n->setlb(ff_monode::getlb(), false);
            }
        }
        return ff_monode::dryrun();
    }
    
    int run(bool skip_init=false) {
        assert(n);
        if (!prepared) {
            if (n && n->prepare()<0) return -1;
        }
        assert(blocking_in == blocking_out);
        n->blocking_mode(blocking_in);
        if (!default_mapping) {
            n->no_mapping();
            ff_monode::no_mapping();
        }
        ff_monode::getlb()->get_filter()->set_id(get_my_id());
        return ff_monode::run(skip_init);
    }

    static inline bool ff_send_out_motransformer(void * task, int id, 
                                                 unsigned long retry,
                                                 unsigned long ticks, void *obj) {
        bool r= ((internal_mo_transformer *)obj)->ff_send_out_to(task, (id<0?0:id), retry, ticks);
        return r;
    }

    bool cleanup;
    ff_node *n=nullptr;
};

/**
 *   Transforms a standard node into a multi-input node 
 *
 *   NOTE: it is importat to call the eosnotify method only if 
 *         all input EOSs have been received and not at each EOS.
 */
struct internal_mi_transformer: ff_minode {
    internal_mi_transformer(ff_node* n, bool cleanup=false):ff_minode(n),cleanup(cleanup),n(n) {}

    template<typename T>
    internal_mi_transformer(const T& _n) {
        T *t = new T(_n);
        assert(t);
        n = t;
        cleanup=true;
        ff_minode::set_filter(n);
    }
    
    internal_mi_transformer(const internal_mi_transformer& t) : ff_minode(t) {
        cleanup=t.cleanup;
        n = t.n;
        ff_minode::set_filter(n);

        // this is a dirty part, we modify a const object.....
        internal_mi_transformer *dirty= const_cast<internal_mi_transformer*>(&t);
        dirty->cleanup=false;
        dirty->n = nullptr;
    }

    ~internal_mi_transformer() {
        if (cleanup && n)
            delete n;
    }
    inline int svc_init() { return n->svc_init(); }
    inline void* svc(void*task) { return n->svc(task);  }
    inline void svc_end() { return n->svc_end(); }
    int set_input(const svector<ff_node *> & w) {
        n->neos += w.size();
        return ff_minode::set_input(w);
    }

    int set_input(ff_node *node) {
        n->neos+=1;
        return ff_minode::set_input(node);
    }
#if 0
    int set_output(ff_node *node) {
        return ff_minode::set_output(node);
    }
#endif
    
    int create_output_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        if (ff_minode::getgt()->get_out_buffer()) return -1;
        int r= ff_minode::create_output_buffer(nentries,fixedsize);
        if (r<0) return -1;
        ff_minode::getgt()->set_output_buffer(ff_minode::get_out_buffer());
        return 0;
    }    

    void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
        n->registerCallback(cb,arg);
    }

    int set_output_buffer(FFBUFFER * const o) {
        return n->set_output_buffer(o);
    }

    FFBUFFER * get_out_buffer() const { return n->get_out_buffer();}

    bool ff_send_out(void * task, int id=-1,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=(TICKS2WAIT)) { 
        return n->ff_send_out(task,id,retry,ticks);
    }
    
    bool init_output_blocking(pthread_mutex_t   *&m,
                              pthread_cond_t    *&c,
                              bool feedback=true) {
        return n->init_output_blocking(m,c,feedback);
    }
    
    void set_output_blocking(pthread_mutex_t   *&m,
                             pthread_cond_t    *&c,
                             bool canoverwrite=false) {
        n->set_output_blocking(m,c, canoverwrite);
    }
        
    inline void eosnotify(ssize_t id) { n->eosnotify(id); }

    void set_id(ssize_t id) {
        if (n) n->set_id(id);
        ff_minode::set_id(id);
    }

    int dryrun() {
        if (prepared) return 0;
        if (n) {
            if (n->isMultiInput()) {
                assert(!n->isComp());
                n->setgt(ff_minode::getgt(), false);
            }
        }
        return ff_minode::dryrun();
    }

    int run(bool skip_init=false) {
        assert(n);
        if (!prepared) {
            if (n && n->prepare()<0) return -1;
        }

        assert(blocking_in == blocking_out);
        n->blocking_mode(blocking_in);
        if (!default_mapping) {
            n->no_mapping();
            ff_minode::no_mapping();
        }
        
        ff_minode::getgt()->get_filter()->set_id(get_my_id());
        return ff_minode::run(skip_init);
    }

    bool cleanup;
    ff_node *n=nullptr;
};

    
} // namespace

#endif /* FF_MULTINODE_HPP */
