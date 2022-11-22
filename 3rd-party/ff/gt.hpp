/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file gt.hpp
 *  \ingroup building_blocks
 *
 *  \brief Farm Collector (it is not a ff_node)
 *
 * It Contains the \p ff_gatherer class and methods which are used to model the \a
 *  Collector node, which is optionally used to gather tasks coming from
 *  workers.
 *
 * \todo Documentation to be rewritten. To be substituted with ff_minode?
 */

/* ***************************************************************************
 *
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

#ifndef FF_GT_HPP
#define FF_GT_HPP

#include <iosfwd>
#include <ff/svector.hpp>
#include <ff/utils.hpp>
#include <ff/node.hpp>

namespace ff {


/*!
 *  \class ff_gatherer
 *  \ingroup building_blocks
 *
 *  \brief A class representing the \a Collector node in a \a Farm skeleton.
 *
 *  This class models the \p gatherer, which wraps all the methods and
 *  structures used by the \a Collector node in a \p Farm skeleton. The \p farm
 *  can be seen as a three-stages \p pipeline, the stages being a \p
 *  ff_loadbalancer called \a emitter, a pool of \p ff_node called \a workers
 *  and - optionally - a \p ff_gatherer called \a collector. The \a Collector
 *  node can be used to gather the results coming from the computations
 *  executed by the pool of \a workers. The \a collector can also be
 *  connected to the \a emitter node via a feedback channel, in order to create
 *  a \p farm-with-feedback skeleton.
 *
 *  This class is defined in \ref gt.hpp
 *
 */

class ff_gatherer: public ff_thread {

    friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_minode;
public:
    enum {TICKS2WAIT=5000};

protected:

    inline bool init_input_blocking(pthread_mutex_t   *&m,
                                    pthread_cond_t    *&c,
                                    bool /*feedback*/=true) {
        if (cons_m == nullptr) {
            assert(cons_c==nullptr);
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
                                     bool feedback=true) {
        if (filter &&
            ( (filter->get_out_buffer()!=nullptr) || filter->isMultiOutput() ) )  { // (*)
            return filter->init_output_blocking(m, c, feedback);
        }
            
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
    inline void set_output_blocking(pthread_mutex_t   *&m,
                                    pthread_cond_t    *&c,
                                    bool canoverwrite=false) {
        p_cons_m = m, p_cons_c = c;

        if (filter && (filter->get_out_buffer()!=nullptr)) { // CHECK: asimmetry with test here (*)
            if (filter->isMultiInput() && !filter->isComp()) return;  // to avoid recursive calls to set_output_blocking
            filter->set_output_blocking(m,c, canoverwrite);
        }
    }
    inline pthread_cond_t    &get_cons_c()  { return *cons_c; }

    
    /**
     * \brief Selects a worker.
     * 
     * It gets the next worker using the Round Robin policy. The selected
     * worker has to be alive (and kicking).
     *
     * \return The next worker to be selected.
     *
     */
    virtual inline ssize_t selectworker() { 
        do 
            nextr = (nextr+1) % running;
        while(offline[nextr]);
        return nextr;
    }

    /**
     * \brief Notifies the EOS
     *
     * It is a virtual function and is used to notify EOS
     */
    virtual inline void notifyeos(int /*id*/) {}

    /**
     * \brief Gets the number of attempts.
     *
     * The number of attempts before wasting some times and than retry 
     */
    virtual inline size_t nattempts() { return getnworkers();}

    /**
     * \brief Loses the time out.
     *
     * It is a virutal function which defines the number of ticks to be waited.
     *
     */
    virtual inline void losetime_out(unsigned long ticks=TICKS2WAIT) { 
        FFTRACE(lostpushticks+=ticks;++pushwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }
    
    /**
     * \brief Loses the time in 
     *
     * It is a virutal function which defines the number of ticks to be waited.
     *
     */
    virtual inline void losetime_in(unsigned long ticks=TICKS2WAIT) { 
        FFTRACE(lostpopticks+=ticks;++popwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }    

    /**
     * \brief It gathers the tasks.
     *
     * It keeps selecting the worker. If a worker has task, then the worker is
     * returned. Otherwise a tick is wasted and then keep looking for the
     * worker with the task.
     *
     * \return It returns the workers with a taks if successful. Otherwise -1
     * is returned.
     */
    virtual ssize_t gather_task(void ** task) {
        unsigned int cnt;
        do {
            cnt=0;
            do {
                nextr = selectworker();
                //assert(offline[nextr]==false);
                if (workers[nextr]->get(task)) {
                    return nextr;
                }
                else if (++cnt == nattempts()) break;
            } while(1);
            if (blocking_in) {
                struct timespec tv;
                timedwait_timeout(tv);
                pthread_mutex_lock(cons_m);
                pthread_cond_timedwait(cons_c, cons_m, &tv);
                pthread_mutex_unlock(cons_m);
            } else losetime_in();
        } while(1);
        return -1;
    }

    /**
     * \brief Pushes the task in the tasks queue.
     *
     * It pushes the tasks in a queue. 
     */
    inline bool push(void * task, unsigned long retry=((unsigned long)-1), unsigned long ticks=(TICKS2WAIT)) {
        if (blocking_out) {
            if (!filter) {
                bool empty=buffer->empty();
                while(!buffer->push(task)) {
                    empty = false;
                    struct timespec tv;
                    timedwait_timeout(tv);
                    pthread_mutex_lock(prod_m);
                    pthread_cond_timedwait(prod_c,prod_m, &tv);
                    pthread_mutex_unlock(prod_m);  
                }
                if (empty) pthread_cond_signal(p_cons_c);
            } else {
                bool empty=filter->get_out_buffer()->empty();
                while(!filter->push(task)) {
                    empty=false;
                    struct timespec tv;
                    timedwait_timeout(tv);
                    pthread_mutex_lock(prod_m);
                    pthread_cond_timedwait(prod_c,prod_m,&tv);
                    pthread_mutex_unlock(prod_m);      
                }
                if (empty) pthread_cond_signal(p_cons_c);
            }
            return true;
        }
        if (!filter) {
            for(unsigned long i=0;i<retry;++i) {
                if (buffer->push(task)) return true;
                losetime_out(ticks);
            }           
        } else 
            for(unsigned long i=0;i<retry;++i) {
                if (filter->push(task)) return true;
                losetime_out();
            }
        return false;        
    }

    /**
     * \brief Pop a task out of the queue.
     *
     * It pops the task out of the queue.
     *
     * \return \p false if not successful, otherwise \p true is returned.
     *
     */
    bool pop(void ** task) {
        if (!buffer) return false;
        while (! buffer->pop(task)) {
            losetime_in();
        } 
        return true;
    }

    /**
     * \brief Pop a task
     *
     * It pops the task.
     *
     * \return The task popped from the buffer.
     */
    bool pop_nb(void ** task) {
        if (!buffer) return false;
        return buffer->pop(task);
    }


    void set_input_channelid(ssize_t id, bool fromin=true) { channelid = id; frominput=fromin;}
    
    static bool ff_send_out_collector(void * task, int id,
                                      unsigned long retry, 
                                      unsigned long ticks, void *obj) {
        (void)id;
        bool r = ((ff_gatherer *)obj)->push(task, retry, ticks);
#if defined(FF_TASK_CALLBACK)
        if (r) ((ff_gatherer *)obj)->callbackOut(obj);
#endif   
        return r;
    }

    bool fromInput() const { return frominput; }
    
#if defined(FF_TASK_CALLBACK)
    void callbackIn(void  *t=NULL) { filter->callbackIn(t);  }
    void callbackOut(void *t=NULL) { filter->callbackOut(t); }
#endif

public:

    /**
     *  \brief Constructor
     *
     *  It creates \p max_num_workers and \p NULL pointers to worker objects.
     */
    ff_gatherer(int max_num_workers):
        max_nworkers(max_num_workers), running(-1), nextr(-1),
        neos(0),neosnofreeze(0),channelid(-1),feedbackid(0),
        filter(NULL), workers(max_nworkers), offline(max_nworkers), buffer(NULL),
        skip1pop(false),frominput(false) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        p_cons_m = NULL, p_cons_c = NULL;

        offline.resize(max_nworkers);

        blocking_in = blocking_out = FF_RUNTIME_MODE;

        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
    }

    ff_gatherer& operator=(ff_gatherer&& gtin) {
        set_barrier(gtin.get_barrier());

        cons_m       = gtin.cons_m;
        cons_c       = gtin.cons_c;
        prod_m       = gtin.prod_m;
        prod_c       = gtin.prod_c;
        p_cons_m       = gtin.p_cons_m;
        p_cons_c       = gtin.p_cons_c;
        buffer         = gtin.buffer;
        blocking_in    = gtin.blocking_in;
        blocking_out   = gtin.blocking_out;
        skip1pop       = gtin.skip1pop;
        frominput      = gtin.frominput;
        filter         = gtin.filter;
        workers        = gtin.workers;
        
        gtin.cons_m = nullptr;
        gtin.cons_c = nullptr;
        gtin.prod_m = nullptr;
        gtin.prod_c = nullptr;
        
        gtin.set_barrier(nullptr);
        return *this;
    }
    
    virtual ~ff_gatherer() {
        if (cons_m) {
            pthread_mutex_destroy(cons_m);
            free(cons_m);
            cons_m = nullptr;
        }
        if (cons_c) {
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
     * \brief Sets the filer
     *
     * It sents the \p ff_node to the filter.
     *
     * \return 0 if successful, otherwise a negative value is returned.
     */
    int set_filter(ff_node * f) { 
        if (filter) {
            error("GT, setting collector filter\n");
            return -1;
        }
        filter = f;

        // NOTE: if set_output_blocking has been already executed (for example by the pipeline),
        // then we have to set the blocking stuff also for the filter
        if ( ((filter->get_out_buffer() != nullptr) || filter->isMultiOutput())  &&
             p_cons_m!=nullptr) {
            filter->set_output_blocking(p_cons_m, p_cons_c);
        }
        
        return 0;
    }

    // Sets where feedback channels end (if any)
    // All channels were registered in input, first feedback ones then the input ones. 
    void set_feedbackid_threshold(size_t id) { feedbackid = id; }
    
    ff_node *get_filter() const { return (filter==(ff_node*)this)?NULL:filter; }
    
    void reset_filter() {
        if (filter == NULL) return;
        filter->registerCallback(NULL,NULL);
        filter->setThread(NULL);
        filter = NULL;
    }
    
    
    /**
     * \brief Sets output buffer
     *
     * It sets the output buffer.
     */
    int set_output_buffer(FFBUFFER * const buff) {
        if (filter) {

            if (filter->set_output_buffer(buff)<0) return -1;
            
            // NOTE: if set_output_blocking has been already executed (for example by the pipeline),
            // then we have to set the blocking stuff also for the filter
            if ( p_cons_m!=nullptr) {
                filter->set_output_blocking(p_cons_m, p_cons_c);
            }
        }
        if (buffer) return -1;
        buffer=buff;
        return 0;
    }

    /**
     * \brief Gets the channel id
     *
     * It gets the \p channelid.
     *
     * \return The \p channelid is returned.
     */
    ssize_t get_channel_id() const { return channelid;}

    size_t get_num_inchannels()  const { return (size_t)(running); }
    size_t get_num_outchannels() const {
        if (filter && (filter != (ff_node*)this))
            return filter->get_num_outchannels();
        return (buffer?1:0);
    }

    size_t get_num_feedbackchannels() const {
        return feedbackid;
    }
    
    /**
     * \brief Gets the number of worker threads currently running.
     *
     * It gets the number of threads currently running.
     *
     * \return Number of worker threads
     */
    inline size_t getnworkers() const { return (size_t)(running-neos-neosnofreeze); }

    
    inline size_t getrunning() const { return (size_t)running;}
    

    /**
     * \brief Get the number of workers
     *
     * It returns the number of total workers registered
     *
     * \return Number of worker
     */
    inline size_t getNWorkers() const { return workers.size();}

    const svector<ff_node*>& getWorkers() const { return workers; }
    

    /**
     * \brief Skips the first pop
     *
     * It determine whether the first pop should be skipped or not.
     *
     * \return Always \true is returned.
     */
    void skipfirstpop(bool sk=true) { skip1pop=sk; }

#ifdef DFF_ENABLED
    void skipallpop(bool sk = true) { _skipallpop = sk;}
#endif

    /**
     * \brief Gets the ouput buffer
     *
     * It gets the output buffer
     *
     * \return \p buffer is returned. 
     */
    FFBUFFER * get_out_buffer() const { return buffer;}

    /**
     * \brief Register the given worker to the list of workers.
     *
     * It registers the given worker to the list of workers.
     *
     * \return 0 if successful, or -1 if not successful.
     */
    int  register_worker(ff_node * w) {
        if (workers.size()>=max_nworkers) {
            error("GT, max number of workers reached (max=%ld)\n",max_nworkers);
            return -1;
        }
        workers.push_back(w);
        return 0;
    }


    /**
     * \brief Initializes the gatherer task.
     *
     * It is a virtual function to initialise the gatherer task.
     *
     * \return It returns the task if successful, otherwise 0 is returned.
     */
    virtual int svc_init() {
#if !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)
        if (this->get_mapping()) {
            int cpuId = filter?filter->getCPUId():-1;
            if (ff_mapThreadToCpu((cpuId<0) ? (cpuId=threadMapper::instance()->getCoreId(tid)) : cpuId)!=0)
                error("Cannot map thread %d to CPU %d, mask is %u,  size is %u,  going on...\n",tid, (cpuId<0) ? threadMapper::instance()->getCoreId(tid) : cpuId, threadMapper::instance()->getMask(), threadMapper::instance()->getCListSize());            
            if (filter) filter->setCPUId(cpuId);
        }
#endif        
        gettimeofday(&tstart,NULL);
        for(ssize_t i=0;i<running;++i)  offline[i]=false;
        if (filter) {
            if (filter->isComp() && !filter->isMultiInput())
                filter->set_neos(running);
            return filter->svc_init();
        }
        return 0;
    }

    /**
     * \brief The gatherer task
     *
     * It is a virtual function to be used as the gatherer task.
     *
     * \return It returns the task.
     */
    virtual void * svc(void *) {
        void * ret  = FF_EOS;
        void * task = NULL;
        bool filter_outpresent = false;
        bool outpresent  = (buffer != NULL);
        bool skipfirstpop = skip1pop;

        // the following case is possible when the there is a filter that is a composition
        if ( filter && 
             ( (filter->get_out_buffer()!=NULL)  || filter->isMultiOutput() ) ) {

            filter_outpresent=true;
            //set_out_buffer(filter->get_in_buffer());
        }


        // it is possible that a standard node has been automatically (and transparently)
        // transformed into a multi-input node (see mi_transformer).
        // In this case we want to call notifyeos only when we have received EOS from all
        // input channel.
        bool notify_each_eos = filter ? (filter->neos==1): false;

        // TODO: skipallpop missing!

        gettimeofday(&wtstart,NULL);
        do {
            task = NULL;
            if (!skipfirstpop) 
                nextr = gather_task(&task); 
            else skipfirstpop=false;

            if (task == FF_GO_ON) continue;
            channelid = (nextr-feedbackid);
            frominput=true;
            if (feedbackid>0) { // there are feedback channels
                if (nextr<feedbackid)  {
                    frominput=false;
                    channelid=nextr;
                }
            }

            if ((task == FF_EOS) || (task == FF_EOSW)) {
                if (filter && notify_each_eos) 
                    filter->eosnotify(channelid); //workers[nextr]->get_my_id());                
                offline[nextr]=true;
                ++neos;
                ret=task;
            } else if (task == FF_EOS_NOFREEZE) {
                if (filter && notify_each_eos)
                    filter->eosnotify(channelid); //workers[nextr]->get_my_id());
                offline[nextr]=true;
                ++neosnofreeze;
                ret = task;
            } else {
                FFTRACE(++taskcnt);
                if (filter)  {                    
                    FFTRACE(ticks t0 = getticks());

#if defined(FF_TASK_CALLBACK)
                    if (filter) callbackIn(this);
#endif
                    task = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                    ticks diff=(getticks()-t0);
                    tickstot +=diff;
                    ticksmin=(std::min)(ticksmin,diff);
                    ticksmax=(std::max)(ticksmax,diff);
#endif    
                }

                // if the filter returns EOS or GO_OUT we exit immediatly
                if (task == FF_GO_ON) continue;                
                if ((task == FF_GO_OUT) || (task == FF_EOS_NOFREEZE) || (task == FF_EOSW) ) {
                    ret = task;
                    break;   // exiting from the loop without sending the task
                } 
                if (!task || (task == FF_EOS)) {
                    ret = FF_EOS;
                    break;
                }

                if (filter_outpresent) filter->ff_send_out(task);
                else  if (outpresent) push(task);
#if defined(FF_TASK_CALLBACK)
                else 
                    if (filter) callbackOut(this);
#endif
            }
        } while((neos<(size_t)running) && (neosnofreeze<(size_t)running));

        // GO_OUT, EOS_NOFREEZE and EOSW are not propagated !
        if (ret == FF_EOS) {
            // we notify the filter only when we have received all EOSs
            if (!notify_each_eos && filter) filter->eosnotify();

            // push EOS
            task = ret;
            if (filter_outpresent) {
                if (filter->isMultiOutput()) filter->propagateEOS();
                else filter->ff_send_out(FF_EOS);
            } else
                if (outpresent) push(FF_EOS);
        }
        if (ret == FF_EOSW) ret = FF_EOS; // EOSW is like an EOS but it is not propagated
        
        gettimeofday(&wtstop,NULL);
        wttime+=diffmsec(wtstop,wtstart);
        if (neos>=(size_t)running) neos=0;
        if (neosnofreeze>=(size_t)running) neosnofreeze=0;

        return ret;
    }

    /**
     * \brief Finializes the gatherer.
     *
     * It is a virtual function used to finalise the gatherer task.
     *
     */
    virtual void svc_end() {
        if (filter) filter->svc_end();
        gettimeofday(&tstop,NULL);
    }

    int dryrun() {
        running=workers.size();
        if (filter) {
            if ((filter->get_out_buffer() == nullptr) && buffer) 
                filter->registerCallback(ff_send_out_collector, this);
            // setting the thread for the filter
            filter->setThread(this);

            assert(blocking_in==blocking_out);
            filter->blocking_mode(blocking_in);
        }
        return 0;
    }

    /**
     * \brief Execute the gatherer task.
     *
     * It executes the gatherer task.
     *
     * \return 0 if successful, otherwise -1 is returned.
     */
    int run(bool=false) {
        ff_gatherer::dryrun();
        
        if (this->spawn(filter?filter->getCPUId():-1)== -2) {
            error("GT, spawning GT thread\n");
            return -1; 
        }
        return 0;
    }

    void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }

    void no_mapping() {
        default_mapping = false;
    }

    
    inline int wait_freezing() {
        int r = ff_thread::wait_freezing();
        running = -1;
        return r;
    }

    /**
     *
     * \brief It gathers all tasks.
     *
     * It is a virtual function, and gathers results from the workers. 
     *
     * \return It returns 0 if the tasks from all the workers are collected.
     * Otherwise a negative value is returned meaning that at least an EOS has been received.
     *
     */
    virtual int all_gather(void *task, void **V) {
        if (ag_callback)  return ag_callback(task,V,ag_callback_arg);

        V[channelid]=task;
        size_t nw=getnworkers();
        svector<ff_node*> _workers(nw);
        for(ssize_t i=0;i<running;++i) {
            if (!offline[i]) _workers.push_back(workers[i]);
            else _workers.push_back(nullptr);
        }
        svector<size_t> retry(nw);

        for(ssize_t i=0;i<running;++i) {
            if(i != channelid) {
                if (_workers[i]) {
                    if (!_workers[i]->get(&V[i])) retry.push_back(i);
                }
            }
        }
        while(retry.size()) {
            channelid = retry.back();
            if(_workers[channelid]->get(&V[channelid])) {
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
        for(ssize_t i=0;i<running;++i) {
            if (V[i] == FF_EOS) {
                eos=true;
                ++neos;
                V[i]=nullptr;
                FFTRACE(taskcnt--);
            } else 
                if (V[i] == FF_EOS_NOFREEZE) {
                    eos=true;
                    ++neosnofreeze;
                    V[i]=nullptr;
                    FFTRACE(taskcnt--);
                }
        }
        FFTRACE(taskcnt+=nw-1);
        return eos?-1:0;
    }

    void registerAllGatherCallback(int (*cb)(void *,void **, void*), void * arg) {
        ag_callback = cb;
        ag_callback_arg = arg;
    }
    
    /**
     * \brief Thaws all threads register with the gt and the gt itself
     *
     * 
     */
    inline void thaw(bool _freeze=false, ssize_t nw=-1) {
        assert(running==-1);
        if (nw == -1 || (size_t)nw > workers.size()) running = workers.size();
        else running = nw;
        ff_thread::thaw(_freeze);
    }

    /**
     *  \brief Resets output buffer
     *  
     *   Warning resetting the buffer while the node is running may produce unexpected results.
     */
    void reset() { if (buffer) buffer->reset();}


    /**
     * \brief Start counting time
     *
     * It defines the counting of start time.
     *
     * \return Difference in milli seconds.
     */
    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    /**
     * \brief Complete counting time
     *
     * It defines the counting of finished time.
     *
     * \return Difference in milli seconds.
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
     * \brief The trace of FastFlow
     *
     * It prints the trace for FastFlow.
     *
     */
    virtual void ffStats(std::ostream & out) { 
        out << "Collector: "
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
    size_t            max_nworkers;
    ssize_t           running;       /// Number of workers running
    ssize_t           nextr;

    size_t            neos;
    size_t            neosnofreeze;
    ssize_t           channelid;
    // if we have feedback channels in input, feedbackid tells us how many
    // they are
    ssize_t           feedbackid; 

    ff_node         * filter;
    svector<ff_node*> workers;
    svector<bool>     offline;
    FFBUFFER        * buffer; 
    bool              skip1pop;
#ifdef DFF_ENABLED
    bool             _skipallpop;
#endif
    bool              frominput;
    int  (*ag_callback)(void *,void **, void*);
    void  * ag_callback_arg;

    
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

    // for synchronizing with the next multi-input stage
    pthread_mutex_t   *p_cons_m = nullptr;
    pthread_cond_t    *p_cons_c = nullptr;

    bool               blocking_in;
    bool               blocking_out;

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

#endif /* FF_GT_HPP */
