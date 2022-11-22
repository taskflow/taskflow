/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file lb.hpp
 *  \ingroup building_blocks
 *
 *  \brief Farm Emitter (not a ff_node)
 *
 *  Contains the \p ff_loadbalancer class and methods used to model the \a Emitter node,
 *  which is used to schedule tasks to workers.
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

 
#ifndef FF_LB_HPP
#define FF_LB_HPP

#include <iosfwd>
#include <deque>

#include <ff/utils.hpp>
#include <ff/node.hpp>

namespace ff {


/*!
 *  \class ff_loadbalancer
 *  \ingroup building_blocks
 *
 *  \brief A class representing the \a Emitter node in a typical \a Farm
 *  skeleton.
 *
 *  This class models the \p loadbalancer, which wraps all the methods and
 *  structures used by the \a Emitter node in a \p Farm skeleton. The \a
 *  emitter node is used to generate the stream of tasks for the pool of \a
 *  workers. The \a emitter can also be used as sequential preprocessor if the
 *  stream is coming from outside the farm, as is the case when the stream is
 *  coming from a previous node of a pipeline chain or from an external
 *  device.\n The \p Farm skeleton must have the \a emitter node defined: if
 *  the user does not add it to the farm, the run-time support adds a default
 *  \a emitter, which acts as a stream filter and schedules tasks in a
 *  round-robin fashion towards the \a workers.
 *
 *  This class is defined in \ref lb.hpp
 *
 */

class ff_loadbalancer: public ff_thread {
    friend class ff_farm;
    friend class ff_ofarm;
    friend class ff_monode;
public:    

    // NOTE:
    //  - TICKS2WAIT should be a valued profiled for the application
    //    Consider to redefine losetime_in and losetime_out for your app.
    //
    enum {TICKS2WAIT=1000};
protected:

    inline void put_done(int id) {
        // here we access the cond variable of the worker, that must be initialized
        pthread_cond_signal(&workers[id]->get_cons_c());
    }
    
    inline bool init_input_blocking(pthread_mutex_t   *&m,
                                    pthread_cond_t    *&c,
                                    bool /*feedback*/=true) {
        if (cons_m == nullptr) {
            assert(cons_c == nullptr);
            cons_m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
            cons_c = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
            assert(cons_m); assert(cons_c);
            if (pthread_mutex_init(cons_m, NULL) != 0) return false;
            if (pthread_cond_init(cons_c, NULL) != 0)  return false;
        } 
        m = cons_m,  c = cons_c;
        return true;
    }
    // producer
    inline bool init_output_blocking(pthread_mutex_t   *&m,
                                     pthread_cond_t    *&c,
                                     bool /*feedback*/=true) {
        if (prod_m == nullptr) {
            assert(prod_c == nullptr);
            prod_m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
            prod_c = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
            assert(prod_m); assert(prod_c);
            if (pthread_mutex_init(prod_m, NULL) != 0) return false;
            if (pthread_cond_init(prod_c, NULL) != 0)  return false;
        } 
        m = prod_m, c = prod_c;
        return true;
    }
    inline void set_output_blocking(pthread_mutex_t   *&,
                                    pthread_cond_t    *&,
                                    bool /*canoverwrite*/=false) {
        assert(1==0);
    }
    
    virtual inline void  set_cons_c(pthread_cond_t *c) {
        assert(cons_c == nullptr);
        assert(cons_m == nullptr);
        cons_c = c;
    }        
    virtual inline pthread_cond_t    &get_cons_c()       { return *cons_c;}
    
    /**
     * \brief Pushes EOS to the worker
     *
     * It pushes the EOS into the queue of all workers.
     *
     */
    inline void push_eos(void *task=NULL) {
        //register int cnt=0;

        if (!task) task = FF_EOS;
        broadcast_task(task);
        if (feedbackid > 0) {
            for(size_t i=feedbackid; i<workers.size();++i)
                this->ff_send_out_to(task, i);
        }
    }
    inline void push_goon() {
        void * goon = FF_GO_ON;
        broadcast_task(goon);
    }

    void propagateEOS(void *task=FF_EOS) { push_eos(task); }
    
    /** 
     * \brief Virtual function that can be redefined to implement a new scheduling
     * policy.
     *
     * It is a virtual function that can be redefined to implement a new
     * scheduling polity.
     *
     * \return The number of worker to be selected.
     */
    virtual inline size_t selectworker() { return (++nextw % running); }

#if defined(LB_CALLBACK)

    /**
     * \brief Defines callback
     *
     * It defines the call back
     *
     * \parm n TODO
     */
    virtual inline void callback(int /*n*/) { }
    
    /**
     * \brief Defines callback
     *
     * It defines the callback and returns a pointer.
     * 
     * \parm n TODO
     * \parm task is a void pointer
     *
     * \return \p NULL pointer is returned.
     */
    virtual inline void * callback(int /*n*/, void * /*task*/) { return NULL;}
#endif

    /**
     * \brief Gets the number of attempts before wasting some times
     *
     * The number of attempts before wasting some times and than retry.
     *
     * \return The number of workers.
     */
    virtual inline size_t nattempts() { return running;}

    /**
     * \brief Loses some time before sending the message to output buffer
     *
     * It loses some time before the message is sent to the output buffer.
     *
     */
    virtual inline void losetime_out(unsigned long ticks=TICKS2WAIT) {
        FFTRACE(lostpushticks+=ticks; ++pushwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }

    /**
     * \brief Loses time before retrying to get a message from the input buffer
     *
     * It loses time before retrying to get a message from the input buffer.
     */
    virtual inline void losetime_in(unsigned long ticks=TICKS2WAIT) {
        FFTRACE(lostpopticks+=ticks; ++popwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }

    /** 
     * \brief Scheduling of tasks
     *
     * It is the main scheduling function. This is a virtual function and can
     * be redefined to implement a custom scheduling policy. 
     *
     * \parm task is a void pointer
     * \parm retry is the number of tries to schedule a task
     * \parm ticks are the number of ticks to be lost
     *
     * \return \p true, if successful, or \p false if not successful.
     *
     */
    virtual inline bool schedule_task(void * task, 
                                      unsigned long retry=((unsigned long)-1), 
                                      unsigned long ticks=TICKS2WAIT) {
        unsigned long cnt;
        if (blocking_out) {
            unsigned long r = 0;
            do {
                cnt=0;
                do {
                    nextw = selectworker();
                    assert(nextw>=0);                    
#if defined(LB_CALLBACK)
                    task = callback(nextw, task);
#endif
                    bool empty=workers[nextw]->get_in_buffer()->empty();
                    if(workers[nextw]->put(task)) {
                        FFTRACE(++taskcnt);
                        if (empty) put_done(nextw);
                        return true;
                    } 
                    ++cnt;
                    if (cnt == nattempts()) break; 
                } while(1);

                if (++r >= retry) return false;
                
                struct timespec tv;
                timedwait_timeout(tv);                
                pthread_mutex_lock(prod_m);
                pthread_cond_timedwait(prod_c, prod_m, &tv);
                pthread_mutex_unlock(prod_m);
            } while(1);
            return true;
        } // blocking 
        do {
            cnt=0;
            do {
                nextw = selectworker();
                if (nextw<0) return false;
#if defined(LB_CALLBACK)
                task = callback(nextw, task);
#endif
                if(workers[nextw]->put(task)) {
                    FFTRACE(++taskcnt);
                    return true;
                }
                ++cnt;
                if (cnt>=retry) { nextw=-1; return false; }
                if (cnt == nattempts()) break; 
            } while(1);
            losetime_out(ticks);
        } while(1);
        return false;
    }

    /**
     * \brief Collects tasks
     *
     * It collects tasks from the worker and returns in the form of deque.
     *
     * \parm task is a void pointer
     * \parm availworkers is a queue of available workers
     * \parm start is a queue of TODO
     *
     * \return The deque of the tasks.
     */
    virtual std::deque<ff_node *>::iterator  collect_task(void ** task, 
                                                           std::deque<ff_node *> & availworkers,
                                                           std::deque<ff_node *>::iterator & start) {
        int cnt, nw= (int)(availworkers.end()-availworkers.begin());
        const std::deque<ff_node *>::iterator & ite(availworkers.end());
        do {
            cnt=0;
            do {
                if (++start == ite) start=availworkers.begin();
                //if (start == ite) start=availworkers.begin();  // read again next time from the same, this needs (**)

                const size_t idx = (start-availworkers.begin());
                if (filter && !filter->in_active && (idx >= multi_input_start)) continue;
                
                if((*start)->get(task)) {
                    
                    input_channelid = (idx >= multi_input_start) ? (*start)->get_my_id():-1;
                    channelid = idx;
                    if (idx >= multi_input_start) {
                        channelid = -1;
                        // NOTE: the filter can be a multi-input node so this call allows
                        //       to set the proper channelid of the gatherer (gt).
                        if (filter) filter->set_input_channelid((ssize_t)(idx-multi_input_start), true);
                    }
                    else {
                        if (idx == (size_t)managerpos) {
                            channelid = (*start)->get_my_id();
                            if (filter) filter->set_input_channelid(channelid, true);
                        } else 
                            if (filter) filter->set_input_channelid((ssize_t)idx, false);
                    }
                    

                    return start;
                }
                else { 
                    //++start;             // (**)
                    if (++cnt == nw) {
                        if (filter && !filter->in_active) { *task=NULL; channelid=-2; return ite;}
                        if (buffer && buffer->pop(task)) {
                            channelid = -1;
                            return ite;
                        }
                        break;
                    }
                }
            } while(1);
            if (blocking_in) {
                struct timespec tv;
                timedwait_timeout(tv);
                pthread_mutex_lock(cons_m);
                pthread_cond_timedwait(cons_c, cons_m, &tv);
                pthread_mutex_unlock(cons_m);
            } else losetime_in();
        } while(1);
        return ite;
    }

    /**
     * \brief Pop a task from buffer
     *
     * It pops the task from buffer.
     *
     * \parm task is a void pointer
     *
     * \return \p true if successful
     */
    bool pop(void ** task) {
        //register int cnt = 0;       
        if (blocking_in) {
            if (!filter) {
                while (! buffer->pop(task)) {
                    struct timespec tv;
                    timedwait_timeout(tv);
                    pthread_mutex_lock(cons_m);
                    pthread_cond_timedwait(cons_c, cons_m, &tv);
                    pthread_mutex_unlock(cons_m);
                } // while
            } else  {                
                if (cons_m) {                
                    while (! filter->pop(task)) {
                        struct timespec tv;
                        timedwait_timeout(tv);
                        pthread_mutex_lock(cons_m);
                        pthread_cond_timedwait(cons_c, cons_m, &tv);
                        pthread_mutex_unlock(cons_m);
                    } //while 
                } else {
                    // NOTE:
                    // it may happen that the filter has been transformed
                    // into a multi-output node (e.g., by using internal_mo_transformer,
                    // see the all-to-all) and so the blocking stuff could have been
                    // initialized for the filter and not for the multi-output node
                    // because the transformation has been postponed until the very last
                    // moment (e.g., in the prepare method of the all-to-all)
                    filter->Pop(task);  
                }
            }
            return true;
        }
        if (!filter) 
            while (! buffer->pop(task)) losetime_in();
        else 
            while (! filter->pop(task)) losetime_in();
        return true;
    }
    
    /**
     *
     * \brief Task scheduler
     *
     * It defines the static version of the task scheduler.
     *
     * \return The status of scheduled task, which can be either \p true or \p
     * false.
     */
    static inline bool ff_send_out_emitter(void * task,int id,
                                           unsigned long retry,
                                           unsigned long ticks, void *obj) {
        
        (void)id;
        bool r= ((ff_loadbalancer *)obj)->schedule_task(task, retry, ticks);
#if defined(FF_TASK_CALLBACK)
        // used to notify that a task has been sent out
        if (r) ((ff_loadbalancer *)obj)->callbackOut(obj);
#endif
        return r;
    }

    /**
     *
     * \brief It gathers all tasks from input channels.
     *
     *
     */
    static inline int ff_all_gather_emitter(void *task, void **V, void*obj) {
        return ((ff_loadbalancer*)obj)->all_gather(task,V);
    }

    // removes all dangling EOSs
    void absorb_eos(svector<ff_node*>& W, size_t size) {
        void *task;
        for(size_t i=0;i<size;++i) {
            do {} while(!W[i]->get(&task));
            assert((task == FF_EOS) || (task == FF_EOS_NOFREEZE));
        }
    }

    // FIX: this function is too costly, it should be re-implemented!
    //
    int get_next_free_channel(bool forever=true) {
        long x=1;
        const size_t attempts = (forever ? (size_t)-1 : running);
        do {
            int nextone = (nextw + x) % running;
            FFBUFFER* buf = workers[nextone]->get_in_buffer();
            
            if (buf->buffersize()>buf->length()) {
                nextw = (nextw + x - 1) % running;
                return nextone;
            }
            if ((x % running) == 0) losetime_out();
        } while((size_t)++x <= attempts);
        return -1;
    }

#if defined(FF_TASK_CALLBACK)
    virtual void callbackIn(void  *t=NULL) { if (filter) filter->callbackIn(t);  }
    virtual void callbackOut(void *t=NULL) { if (filter) filter->callbackOut(t); }
#endif

public:
    /** 
     *  \brief Default constructor 
     *
     *  It is the defauls constructor 
     *
     *  \param max_num_workers The max number of workers allowed
     */
    ff_loadbalancer(size_t max_num_workers): 
        running(-1),max_nworkers(max_num_workers),nextw(-1),feedbackid(0),
        channelid(-2),input_channelid(-1),
        filter(NULL),workers(max_num_workers),
        buffer(NULL),skip1pop(false),master_worker(false),parallel_workers(false),
        multi_input(MAX_NUM_THREADS), inputNodesFeedback(MAX_NUM_THREADS), multi_input_start((size_t)-1) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;

        blocking_in = blocking_out = FF_RUNTIME_MODE;

        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
    }

    ff_loadbalancer& operator=(ff_loadbalancer&& lbin) {
        set_barrier(lbin.get_barrier());

        multi_input = lbin.multi_input;
        inputNodesFeedback= lbin.inputNodesFeedback;
        cons_m       = lbin.cons_m;
        cons_c       = lbin.cons_c;
        prod_m       = lbin.prod_m;
        prod_c       = lbin.prod_c;
        buffer         = lbin.buffer;
        blocking_in    = lbin.blocking_in;
        blocking_out   = lbin.blocking_out;
        skip1pop       = lbin.skip1pop;
        filter         = lbin.filter;
        workers        = lbin.workers;
        manager        = lbin.manager;
        
        lbin.cons_m = nullptr;
        lbin.cons_c = nullptr;
        lbin.prod_m = nullptr;
        lbin.prod_c = nullptr;
        lbin.set_barrier(nullptr);
        return *this;
    }

    /** 
     *  \brief Destructor
     *
     *  It deallocates dynamic memory spaces previoulsy allocated for workers.
     */
    virtual ~ff_loadbalancer() {
        if (cons_m) {
            pthread_mutex_destroy(cons_m);
            free(cons_m);
            cons_m = nullptr;
        }
        if (cons_c && cons_m) {
            pthread_cond_destroy(cons_c);
            free(cons_c);
            cons_c = nullptr;
        }
        if (prod_m) {
            pthread_mutex_destroy(prod_m);
            free(prod_m);
            prod_m = nullptr;
        }
        if (prod_c) {
            pthread_cond_destroy(prod_c);
            free(prod_c);
            prod_c = nullptr;
        }
    }

    /**
     * \brief Sets filter node
     *
     * It sets the filter with the FastFlow node.
     *
     * \parm f is FastFlow node
     *
     * \return 0 if successful, otherwise -1 
     */
    int set_filter(ff_node * f) { 
        if (filter) {
            error("LB, setting emitter filter\n");
            return -1;
        }
        filter = f;
        return 0;
    }

    ff_node *get_filter() const { return filter;}

    void reset_filter() {
        if (filter == NULL) return;
        filter->registerCallback(NULL,NULL);
        filter->setThread(NULL);
        filter = NULL;
    }
    

    /**
     * \brief Sets input buffer
     *
     * It sets the input buffer with the instance of FFBUFFER
     *
     * \parm buff is a pointer of FFBUFFER
     */
    void set_in_buffer(FFBUFFER * const buff) { 
        buffer=buff; 
        skip1pop=false;
    }

    /**
     * \brief Gets input buffer
     *
     * It gets the input buffer
     *
     * \return The buffer
     */
    FFBUFFER * get_in_buffer() const { return buffer;}
    
    /**
     *
     * \brief Gets channel id
     *
     * It returns the identifier of the channel.
     *
     * \return the channel id
     */
    ssize_t get_channel_id() const { return channelid;}

    size_t get_num_inchannels() const {
        size_t nw=multi_input.size();
        if (manager) nw +=1;
        if (multi_input.size()==0 && (get_in_buffer()!=NULL)) nw+=1;
        return nw;
    }
    size_t get_num_outchannels() const { return workers.size(); }
    size_t get_num_feedbackchannels() const {
        return feedbackid;
    }
    
    /**
     * \brief Resets the channel id
     *
     * It reset the channel id to -2
     *
     */
    void reset_channel_id() { channelid=-2;}

    /**
     *  \brief Resets input buffer
     *  
     *   Warning resetting the buffer while the node is running may produce unexpected results.
     */
    void reset() { if (buffer) buffer->reset();}

    /**
     * \brief Get the number of workers
     *
     * It returns the number of workers running
     *
     * \return Number of worker
     */
    inline size_t getnworkers() const { return (size_t)running;} 

    /**
     * \brief Get the number of workers
     *
     * It returns the number of total workers registered
     *
     * \return Number of worker
     */
    inline size_t getNWorkers() const { return workers.size();}

    const svector<ff_node*>& getWorkers() const { return workers; }

    void set_feedbackid_threshold(size_t id) {
        feedbackid = id;
    }
    
    /**
     * \brief Skips first pop
     *
     * It sets \p skip1pop to \p true
     *
     */
    void skipfirstpop(bool sk) {
        skip1pop=sk;
        for(size_t i=0;i<workers.size();++i)
            workers[i]->skipfirstpop(false);
    }

#ifdef DFF_ENABLED
    void skipallpop(bool sk) { _skipallpop = sk; }
#endif


    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }

    void no_mapping() {
        default_mapping = false;
    }
    
    /**
     * \brief Decides master-worker schema
     *
     * It desides the master-worker schema.
     *
     * \return 0 if successful, or -1 if unsuccessful.
     *
     */
    int set_masterworker() {
        if (master_worker) {
            error("LB, master_worker flag already set\n");
            return -1;
        }
        master_worker=true;
        return 0;
    }

    inline int getTid(ff_node *node) const {
        return node->getTid();
    }
    
    /**
     * \brief Sets multiple input buffers
     *
     * It sets the multiple input buffers.
     *
     * \return 0 if successful, otherwise -1.
     */
    int set_input(const svector<ff_node*> &mi) {
        multi_input += mi;
        return 0;
    }

    int set_input(ff_node * node) {
        multi_input.push_back(node);
        return 0;
    }
    int set_input_feedback(ff_node * node) {
        inputNodesFeedback.push_back(node);
        return 0;
    }
    void get_in_nodes(svector<ff_node*>&w) {
        w+=multi_input;
    }
    void get_in_nodes_feedback(svector<ff_node*>&w) {
        w += inputNodesFeedback;
    }

    virtual inline bool ff_send_out_to(void *task, int id,  
                               unsigned long retry=((unsigned long)-1),
                               unsigned long ticks=(TICKS2WAIT)) {        
        if (blocking_out) {
            unsigned long r=0;
        _retry:
            bool empty=workers[id]->get_in_buffer()->empty();
            if (workers[id]->put(task)) {
                FFTRACE(++taskcnt);
                if (empty) put_done(id);
            } else {
                if (++r >= retry) return false;
                struct timespec tv;
                timedwait_timeout(tv);
                pthread_mutex_lock(prod_m);
                pthread_cond_timedwait(prod_c, prod_m, &tv);
                pthread_mutex_unlock(prod_m);     
                goto _retry;
            }
#if defined(FF_TASK_CALLBACK)
            callbackOut(this);
#endif
            return true;
        }
        for(unsigned long i=0;i<retry;++i) {
            if (workers[id]->put(task)) {
                FFTRACE(++taskcnt);
#if defined(FF_TASK_CALLBACK)
                callbackOut(this);
#endif
               return true;
            }
            losetime_out(ticks);
        }    
        return false;
    }

    /** 
     * \brief Send the same task to all workers 
     *
     * It sends the same task to all workers.   
     */
    virtual inline void broadcast_task(void * task) {
       std::vector<size_t> retry;
       if (blocking_out) {
           for(ssize_t i=0;i<running;++i) {
               bool empty=workers[i]->get_in_buffer()->empty();
               if(!workers[i]->put(task))
                   retry.push_back(i);
               else if (empty) put_done(i);
           }
           while(retry.size()) {
               bool empty=workers[retry.back()]->get_in_buffer()->empty();
               if(workers[retry.back()]->put(task)) {
                   if (empty) put_done(retry.back());
                   retry.pop_back();
               } else {
                   struct timespec tv;
                   timedwait_timeout(tv);
                   pthread_mutex_lock(prod_m);
                   pthread_cond_timedwait(prod_c, prod_m, &tv);
                   pthread_mutex_unlock(prod_m); 
               }
           }           
#if defined(FF_TASK_CALLBACK)
           callbackOut(this);
#endif
           return;
       }
       for(ssize_t i=0;i<running;++i) {
           if(!workers[i]->put(task))
               retry.push_back(i);
       }
       while(retry.size()) {
           if(workers[retry.back()]->put(task))
               retry.pop_back();
           else losetime_out();
       }       
#if defined(FF_TASK_CALLBACK)
       callbackOut(this);
#endif
    }

    int all_gather(void* task, void**V) {
        V[input_channelid]=task;
        if (multi_input.size()==0) return -1;
        size_t _nw=0;
        svector<ff_node*> _workers(MAX_NUM_THREADS);
        for(size_t i=0;i<multi_input.size(); ++i) {
            if (!offline[i]) {
                ++_nw;
                _workers.push_back(multi_input[i]);
            }
            else _workers.push_back(nullptr);
        }
        svector<size_t> retry(_nw);
        for(size_t i=0;i<_workers.size();++i) {
            if(i!=(size_t)input_channelid) {
                if (_workers[i]) {
                    if (!_workers[i]->get(&V[i])) retry.push_back(i);
                }
            }
        }
        while(retry.size()) {
            input_channelid = retry.back();
            if(_workers[input_channelid]->get(&V[input_channelid])) {
                retry.pop_back();
            }
            else {
                if (blocking_in) {
                    struct timespec tv;
                    timedwait_timeout(tv);
                    pthread_mutex_lock(cons_m);
                    pthread_cond_timedwait(cons_c, cons_m, &tv);
                    pthread_mutex_unlock(cons_m);
                } else losetime_in();
            }
        }
        bool eos=false;
        for(size_t i=0;i<_nw;++i) {
            if (V[i] == FF_EOS || V[i] == FF_EOS_NOFREEZE) {
                eos=true;
                offline[i] = true;
                availworkers.erase(availworkers.begin()+i);
                V[i]=nullptr;
                FFTRACE(taskcnt--);
            } 
        }
        FFTRACE(taskcnt+=_nw-1);
        return eos?-1:0;
    }
        
    
    /**
     * \brief Gets the masterworker flags
     */
    bool masterworker() const { return master_worker;}
    
    /**
     * \brief Registers the given node into the workers' list
     *
     * It registers the given node in the worker list.
     *
     * \param w is the worker
     *
     * \return 0 if successful, or -1 if not successful
     */
    int  register_worker(ff_node * w) {
        if (workers.size()>=max_nworkers) {
            error("LB, max number of workers reached (max=%ld)\n",max_nworkers);
            return -1;
        }
        workers.push_back(w);
        return 0;
    }


    /**
     *
     * \brief Schedule engine.
     *
     *  It is the function used to schedule the tasks.
     */
    virtual void * svc(void *) {
        void * task = NULL;
        void * ret  = FF_EOS;
        bool inpresent  = (get_in_buffer() != NULL);
        bool skipfirstpop = skip1pop;

        if (!inpresent && filter && (filter->get_in_buffer()!=NULL)) {
            inpresent = true;
            set_in_buffer(filter->get_in_buffer());
        }

        gettimeofday(&wtstart,NULL);
        if (!master_worker && (multi_input.size()==0) && (inputNodesFeedback.size()==0)) {

            // it is possible that the input queue has been configured as multi-producer
            // therefore multiple node write in that queue and so the EOS has to be
            // notified only when 'neos' EOSs have been received. By default neos = 1
            int neos = filter?filter->neos:1;
            
            do {
#ifdef DFF_ENABLED
                if (!_skipallpop && inpresent){
#else
                if (inpresent) {
#endif
                    if (!skipfirstpop) pop(&task);
                    else skipfirstpop=false;

                    // ignoring EOSW in input
                    if (task == FF_EOSW) continue;                     
                    if (task == FF_EOS) {
                        if (--neos>0) continue;
                        if (filter) filter->eosnotify();
                        push_eos(); 
                        break;
                    } else if (task == FF_GO_OUT) 
                        break;
                    else if (task == FF_EOS_NOFREEZE) {
                        if (--neos>0) continue;
                        if (filter) 
                            filter->eosnotify();
                        ret = task;
                        break;
                    } 
                }

                if (filter) {
                    FFTRACE(ticks t0 = getticks());

#if defined(FF_TASK_CALLBACK)
                    callbackIn(this);
#endif
                    task = filter->svc(task);

                    
#if defined(TRACE_FASTFLOW)
                    ticks diff=(getticks()-t0);
                    tickstot +=diff;
                    ticksmin=(std::min)(ticksmin,diff);
                    ticksmax=(std::max)(ticksmax,diff);
#endif  

                    if (task == FF_GO_ON) continue;
                    if ((task == FF_GO_OUT) || (task == FF_EOS_NOFREEZE)) {
                        ret = task;
                        break; // exiting from the loop without sending out the task
                    }
                    // if the filter returns NULL/EOS we exit immediatly
                    if (!task || (task==FF_EOS) || (task==FF_EOSW)) {  // EOSW is propagated to workers
                        push_eos(task);
                        ret = FF_EOS;
                        break;
                    }
                } else 
                    if (!inpresent) { 
                        push_goon(); 
                        push_eos();
                        ret = FF_EOS; 
                        break;
                    }
                
                const bool r = schedule_task(task);
                assert(r); (void)r;
#if defined(FF_TASK_CALLBACK)
                callbackOut(this);
#endif
            } while(true);
        } else {
            size_t nw=0;
            availworkers.resize(0);
            if (master_worker && !parallel_workers) {
                for(int i=0;i<running;++i) {
                    availworkers.push_back(workers[i]);
                }
                nw = running;
            }
            // the manager has a complete separate channel that we want to listen to
            // as for all other input channels. The run-time sees the manager as an extra worker.
            if (manager) { 
                managerpos = availworkers.size();
                availworkers.push_back(manager);
                nw += 1;
            }
            if (inputNodesFeedback.size()>0) {
                for(size_t i=0;i<inputNodesFeedback.size();++i)
                    availworkers.push_back(inputNodesFeedback[i]);
                nw += inputNodesFeedback.size();
            }
            if (multi_input.size()>0) {
                multi_input_start = availworkers.size();
                for(size_t i=0;i<multi_input.size();++i) {
                    offline[i]=false;
                    availworkers.push_back(multi_input[i]);
                }
                nw += multi_input.size();
            } 
            if (multi_input.size()==0 && inpresent) {
                nw += 1;
            }
            std::deque<ff_node *>::iterator start(availworkers.begin());
            std::deque<ff_node *>::iterator victim(availworkers.begin());
            do {
                if (!skipfirstpop) {  
                    victim=collect_task(&task, availworkers, start);
                } else skipfirstpop=false;
                
                if (task == FF_GO_OUT) { 
                    ret = task; 
                    break; 
                }
                // ignoring EOSW in input
                if (task == FF_EOSW) continue; 

                if ((task == FF_EOS) || 
                    (task == FF_EOS_NOFREEZE)) {
                    if (filter) {
                        filter->eosnotify(channelid);
                    }
                    if ((victim != availworkers.end())) {
                        if ((task != FF_EOS_NOFREEZE) && channelid>0 && 
                            (channelid == managerpos ||  ((size_t)channelid<workers.size() && !workers[channelid]->isfrozen()))) {  

                            availworkers.erase(victim);
                            start=availworkers.begin(); // restart iterator
                            if (channelid == managerpos) { 
                                // the manager has been added as a worker
                                // so when it terminates we have to decrease the 
                                // starting point of the multi-input channels
                                multi_input_start -= 1;  
                            }
                        }
                    }
                    if (!--nw) {
                        // this conditions means that if there is a loop
                        // we don't want to send an additional
                        // EOS since all EOSs have already been received
                        if (!master_worker && (task==FF_EOS) && (inputNodesFeedback.size()==0)) {
                            push_eos();
                        }
                        ret = task;
                        break; // received all EOS, exit
                    }
                    //}
                } else {
                    if (filter) {
                        FFTRACE(ticks t0 = getticks());

#if defined(FF_TASK_CALLBACK)
                        callbackIn(this);
#endif   
                        task = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                        ticks diff=(getticks()-t0);
                        tickstot +=diff;
                        ticksmin=(std::min)(ticksmin,diff);
                        ticksmax=(std::max)(ticksmax,diff);
#endif  

                        if (task == FF_GO_ON) continue;
                        if ((task == FF_GO_OUT) || (task == FF_EOS_NOFREEZE)){
                            ret = task;
                            break; // exiting from the loop without sending out the task
                        }
                        // if the filter returns NULL we exit immediatly
                        if (!task || (task == FF_EOS) || (task == FF_EOSW) ) {
                            push_eos(task);
                            // try to remove the additional EOS due to 
                            // the feedback channel
                            
                            if (inpresent || multi_input.size()>0 || isfrozen()) {
                                if (master_worker && !parallel_workers) absorb_eos(workers, running);
                                if (inputNodesFeedback.size()>0) absorb_eos(inputNodesFeedback, 
                                                                            inputNodesFeedback.size());
                            }
                            ret = FF_EOS;
                            break;
                        }
                    }
                    schedule_task(task);
#if defined(FF_TASK_CALLBACK)
                    callbackOut(this);
#endif   
                }
            } while(1);
        }
        gettimeofday(&wtstop,NULL);
        wttime+=diffmsec(wtstop,wtstart);

        return ret;
    }

    /**
     * \brief Initializes the load balancer task
     *
     * It is a virtual function which initialises the loadbalancer task.
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    virtual int svc_init() {
#if !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)
        if (this->get_mapping()) {
            int cpuId = (filter)?filter->getCPUId():-1;
            if (ff_mapThreadToCpu((cpuId<0) ? (cpuId=threadMapper::instance()->getCoreId(tid)) : cpuId)!=0)
                error("Cannot map thread %d to CPU %d, mask is %u,  size is %u,  going on...\n",tid, (cpuId<0) ? threadMapper::instance()->getCoreId(tid) : cpuId, threadMapper::instance()->getMask(), threadMapper::instance()->getCListSize());            
            if (filter) filter->setCPUId(cpuId);
        }
#endif        
        gettimeofday(&tstart,NULL);
        if (filter) {
            if (filter->svc_init() <0) return -1;
        }

        return 0;
    }

    /**
     * \brief Finalizes the loadbalancer task
     *
     * It is a virtual function which finalises the loadbalancer task.
     *
     */
    virtual void svc_end() {
        if (filter) filter->svc_end();        
        gettimeofday(&tstop,NULL);
    }

    int dryrun() {
        // if there are feedback channels, we want ff_send_out will do
        // a round-robin on those channels, whereas ff_send_out_to could be
        // used to target forward channels
        // by setting running=feedbackid, then selectworkers will skip forward channels
        if (feedbackid>0)
            running = feedbackid;
        else 
            running = workers.size();
        
        if (filter) {
            // WARNING: If the last node of a composition is a multi-output node, then the
            // callback must not be set.
            if (!filter->isMultiOutput()) {
                filter->registerCallback(ff_send_out_emitter, this);
            }
            // setting the thread for the filter
            filter->setThread(this);
            
            assert(blocking_in==blocking_out);
            filter->blocking_mode(blocking_in);
        }        
        return 0;
    }
    
    /**
     * \brief Runs the loadbalancer
     *
     * It runs the load balancer.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int runlb(bool=false, ssize_t nw=-1) {
        ff_loadbalancer::dryrun();        
        running = (nw<=0)?(feedbackid>0?feedbackid:workers.size()):nw;
        
        if (this->spawn(filter?filter->getCPUId():-1) == -2) {
            error("LB, spawning LB thread\n");
            running = -1;
            return -1;
        }
        return 0;
    }

    int runWorkers(ssize_t nw=-1) {
        running = (nw<=0)?workers.size():nw;
        if (isfrozen()) {
            for(size_t i=0;i<(size_t)running;++i) {
                /* set the initial blocking mode
                 */
                assert(blocking_in==blocking_out);
                workers[i]->blocking_mode(blocking_in);
                if (!default_mapping) workers[i]->no_mapping();
                //workers[i]->skipfirstpop(false);
                if (workers[i]->freeze_and_run(true)<0) {
                    error("LB, spawning worker thread\n");
                    return -1;
                }            
            }
        } else {
            for(size_t i=0;i<(size_t)running;++i) {
                /* set the initial blocking mode
                 */
                assert(blocking_in==blocking_out);
                workers[i]->blocking_mode(blocking_in);
                if (!default_mapping) workers[i]->no_mapping();
                //workers[i]->skipfirstpop(false);
                if (workers[i]->run(true)<0) {
                    error("LB, spawning worker thread\n");
                    return -1;
                }            
            }
        }
        return 0;
    }
    virtual int thawWorkers(bool _freeze=false, ssize_t nw=-1) {
        if (nw == -1 || (size_t)nw > workers.size()) running = workers.size();
        else running = nw;
        for(ssize_t i=0;i<running;++i)
            workers[i]->thaw(_freeze);
        return 0;
    }
    inline int wait_freezingWorkers() {
        int ret = 0;
        for(ssize_t i=0;i<running;++i)
            if (workers[i]->wait_freezing()<0) {
                error("LB, waiting freezing of worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        running = -1;
        return ret;
    }
    inline int waitWorkers() {
        int ret=0;
        for(size_t i=0;i<workers.size();++i)
            if (workers[i]->wait()<0) {
                error("LB, waiting worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        running = -1;
        return ret;
    }

    /**
     * \brief Spawns workers threads
     *
     * It spawns workers threads.
     *
     * \return 0 if successful, otherwise -1 is returned
     */
    virtual int run(bool=false) {
        if (runlb(false, -1) <0) {
            error("LB, spawning LB thread\n");
            return -1;
        }
        if (runWorkers() <0) {
            error("LB, spawning worker thread\n");
            return -1;
        }            
        return 0;
    }

    /**
     * \brief Waits for load balancer
     *
     * It waits for the load balancer.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int waitlb() {
        if (ff_thread::wait()<0) {
            error("LB, waiting LB thread\n");
            return -1;
        }
        return 0;
    }


    /**
     * \brief Waits for workers to finish their task
     *
     * It waits for all workers to finish their tasks.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    virtual int wait() {
        int ret=0;
        for(size_t i=0;i<workers.size();++i)
            if (workers[i]->wait()<0) {
                error("LB, waiting worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        running = -1;
        if (ff_thread::wait()<0) {
            error("LB, waiting LB thread\n");
            ret = -1;
        }
        return ret;
    }


    inline int wait_lb_freezing() {
        if (ff_thread::wait_freezing()<0) {
            error("LB, waiting LB thread freezing\n");
            return -1;
        }
        running = -1;
        return 0;
    }

    /**
     * \brief Waits for freezing
     *
     * It waits for the freezing of all threads.
     *
     * \return 0 if successful, otherwise -1 is returned.
     *
     */
    virtual inline int wait_freezing() {
        int ret = 0;
        for(ssize_t i=0;i<running;++i)
            if (workers[i]->wait_freezing()<0) {
                error("LB, waiting freezing of worker thread, id = %d\n",workers[i]->get_my_id());
                ret = -1;
            }
        if (ff_thread::wait_freezing()<0) {
            error("LB, waiting LB thread freezing\n");
            ret = -1;
        }
        running = -1;
        return ret;
    }

    /**
     * \brief Waits for freezing for one single worker thread
     *
     */
    inline int wait_freezing(const size_t n) {
        assert(n<(size_t)running);
        if (workers[n]->wait_freezing()<0) {
            error("LB, waiting freezing of worker thread, id = %d\n",workers[n]->get_my_id());
            return -1;
        }
        return 0;
    }


    /**
     * \brief Stops the thread
     *
     * It stops all workers and the emitter.
     */
    inline void stop() {
        for(size_t i=0;i<workers.size();++i) workers[i]->stop();
        ff_thread::stop();
    }

    /**
     * \brief Freezes all threads registered with the lb and the lb itself
     *
     * It freezes all workers and the emitter.
     */
    inline void freeze() {
        for(ssize_t i=0;i<running;++i) workers[i]->freeze();
        ff_thread::freeze();
    }

    /**
     * \brief Freezes all workers registered with the lb.
     *
     * It freezes all worker threads.
     */
    inline void freezeWorkers() {
        for(ssize_t i=0;i<running;++i) workers[i]->freeze();
    }

    /**
     * \brief Freezes one worker thread
     */
    inline void freeze(const size_t n) {
        assert(n<workers.size());
        workers[n]->freeze();

        // FIX: should I have to decrease running ? CHECK        
    }

    /**
     * \brief Thaws all threads register with the lb and the lb itself
     *
     * 
     */
    virtual inline void thaw(bool _freeze=false, ssize_t nw=-1) {
        if (nw == -1 || (size_t)nw > workers.size()) running = workers.size();
        else running = nw;
        ff_thread::thaw(_freeze); // NOTE:start scheduler first
        for(ssize_t i=0;i<running;++i) workers[i]->thaw(_freeze);
    }

    /**
     * \brief Thaws one single worker thread 
     *
     */
    inline void thaw(const size_t n, bool _freeze=false) {
        assert(n<workers.size());
        workers[n]->thaw(_freeze);
    }

    void addManagerChannel(ff_node *m) {
        manager = m;
    }


    /**
     * \brief FastFlow start timing
     *
     * It returns the starting of FastFlow timing.
     *
     * \return The difference in FastFlow timing.
     */
    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    /**
     * \brief FastFlow finish timing
     *
     * It returns the finishing of FastFlow timing.
     *
     * \return The difference in FastFlow timing.
     */
    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

    virtual const struct timeval & getstarttime() const { return tstart;}
    virtual const struct timeval & getstoptime()  const { return tstop;}
    virtual const struct timeval & getwstartime() const { return wtstart;}
    virtual const struct timeval & getwstoptime() const { return wtstop;}
    
#if defined(TRACE_FASTFLOW) 
    /**
     * \brief Prints the FastFlow trace
     *
     * It prints the trace of FastFlow.
     */
    virtual void ffStats(std::ostream & out) { 
        out << "Emitter: "
            << "  work-time (ms): " << wttime    << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << (filter?ticksmin:0) << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }

    virtual double getworktime() const { return wttime; }
    virtual size_t getnumtask()  const { return taskcnt; }
    virtual ticks  getsvcticks() const { return tickstot; }
    virtual size_t getpushlost() const { return pushwait;}
    virtual size_t getpoplost()  const { return popwait; }

#endif

private:
    ssize_t            running;             /// Number of workers running
    size_t             max_nworkers;        /// Max number of workers allowed
    ssize_t            nextw;               /// out index
    ssize_t            feedbackid;          /// threshold index
    ssize_t            channelid;
    ssize_t            input_channelid;     
    ff_node         *  filter;              /// user's filter
    ff_node         *  manager = nullptr;   /// manager node, typically not present    
    svector<ff_node*>  workers;             /// farm's workers
    std::deque<ff_node *> availworkers;     /// contains current worker, used in multi-input mode
    svector<bool>      offline;             /// input workers that are offline
    FFBUFFER        *  buffer;
    bool               skip1pop;
    bool               master_worker;
    bool               parallel_workers;    /// true if workers are A2As, pipes or farms
    svector<ff_node*>  multi_input;         /// nodes coming from other stages
    svector<ff_node*>  inputNodesFeedback;  /// nodes coming node feedback channels
    size_t             multi_input_start;   /// position in the availworkers array
    ssize_t            managerpos=-1;       /// position in the availworkers array of the manager

    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;
    double wttime;

 protected:

    // for the input queue
    pthread_mutex_t    *cons_m = nullptr;
    pthread_cond_t     *cons_c = nullptr;

    // for the output queue
    pthread_mutex_t    *prod_m = nullptr;
    pthread_cond_t     *prod_c = nullptr;

    bool               blocking_in;
    bool               blocking_out;

#ifdef DFF_ENABLED
    bool               _skipallpop = false;    
#endif

#if defined(TRACE_FASTFLOW)
    unsigned long taskcnt;
    ticks         lostpushticks;
    unsigned long pushwait;
    ticks         lostpopticks;
    unsigned long popwait;
    ticks         ticksmin;
    ticks         ticksmax;
    ticks         tickstot;
#endif
};


} // namespace ff

#endif  /* FF_LB_HPP */
