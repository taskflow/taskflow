/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file node.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow ff_node 
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_NODE_HPP
#define FF_NODE_HPP

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

#include <stdlib.h>
#include <iosfwd>
#include <functional>
#include <ff/platforms/platform.h>
#include <ff/cycle.h>
#include <ff/utils.hpp>
#include <ff/buffer.hpp>
#include <ff/ubuffer.hpp>
#include <ff/mapper.hpp>
#include <ff/config.hpp>
#include <ff/svector.hpp>
#include <ff/barrier.hpp>
#include <atomic>

#ifdef DFF_ENABLED

#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_typetraits.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

#endif


namespace ff {

// distributed rts related type, but always defined
struct GroupInterface; 


static void* FF_EOS           = (void*)(ULLONG_MAX);     /// automatically propagated
static void* FF_EOS_NOFREEZE  = (void*)(ULLONG_MAX-1);   /// not automatically propagated
static void* FF_EOSW          = (void*)(ULLONG_MAX-2);   /// propagated only by farm's stages
static void* FF_GO_ON         = (void*)(ULLONG_MAX-3);   /// not automatically propagated
static void* FF_GO_OUT        = (void*)(ULLONG_MAX-4);   /// not automatically propagated
static void* FF_TAG_MIN       = (void*)(ULLONG_MAX-10);  /// just a lower bound mark
// The FF_GO_OUT is quite similar to the FF_EOS_NOFREEZE. Both of them are not propagated automatically to
// the next stage, but while the first one is used to exit the main computation loop and, if this is the case, to be frozen,
// the second one is used to exit the computation loop and keep spinning on the input queue waiting for a new task
// without being frozen.
// EOSW is like EOS but it is not propagated outside a farm pattern. If an emitter receives EOSW in input,
// then it will be discarded.
//
    
/* optimization levels used in the optimize_static call (see optimize.hpp) */    
struct OptLevel {
    ssize_t  max_nb_threads{MAX_NUM_THREADS};
    ssize_t  max_mapped_threads{MAX_NUM_THREADS};
    int      verbose_level{0};
    bool     no_initial_barrier{false};
    bool     no_default_mapping{false};
    bool     blocking_mode{false};
    bool     merge_with_emitter{false};
    bool     remove_collector{false};
    bool     merge_farms{false};
    bool     introduce_a2a{false};
};
struct OptLevel1: OptLevel {
    OptLevel1() {
        max_nb_threads=ff_numCores();   // TODO: use the mapper
        blocking_mode=true;
        no_initial_barrier=true;
        remove_collector=true;
    }
};
struct OptLevel2: OptLevel {
    OptLevel2() {
        max_nb_threads=ff_numCores();   // TODO: use the mapper
        blocking_mode=true;
        no_initial_barrier=true;
        merge_with_emitter=true;
        remove_collector=true;
        merge_farms= true;
    }
};
/* ----------------------------------------------------------------------- */

// This is just a counter, and is used to set the ff_node::tid value.
// The _noBarrier counter is to use with threads that are not part of a topology,
// such for example stand-alone nodes or manager node or ...etc...    
static std::atomic_ulong   internal_threadCounter{0};
static std::atomic_ulong   internal_threadCounter_noBarrier{MAX_NUM_THREADS};
    
// TODO: Should be rewritten in terms of mapping_utils.hpp 
#if defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)

    /*
     *
     * \brief Initialize thread affinity 
     * It initializes thread affinity i.e. which cpu the thread should be
     * assigned.
     *
     * \note Linux-specific code
     *
     * \param attr is the pthread attribute
     * \param cpuID is the identifier the core
     * \return -2  if error, the cpu identifier if successful
     */
static inline int init_thread_affinity(pthread_attr_t*attr, int cpuId) {
    // This is linux-specific code
    cpu_set_t cpuset;    
    CPU_ZERO(&cpuset);

    int id;
    if (cpuId<0) {
        id = threadMapper::instance()->getCoreId();
        CPU_SET (id, &cpuset);
    } else  {
        id = cpuId;
        CPU_SET (cpuId, &cpuset);
    }

    if (pthread_attr_setaffinity_np (attr, sizeof(cpuset), &cpuset)<0) {
        perror("pthread_attr_setaffinity_np");
        return -2;
    }
    return id;    
}
#elif !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)

/*
 * \brief Initializes thread affinity
 *
 * It initializes thread affinity i.e. it defines to which core ths thread
 * should be assigned.
 *
 * \return always return -1 because no thread mapping is done
 */
static inline int init_thread_affinity(pthread_attr_t*,int) {
    // Ensure that the threadMapper constructor is called
    threadMapper::instance();
    return -1;
}
#else
/*
 * \brief Initializes thread affinity
 *
 * It initializes thread affinity i.e. it defines to which core ths thread
 * should be assigned.
 *
 * \return always return -1 because no thread mapping is done
 */
static inline int init_thread_affinity(pthread_attr_t*,int) {
    // Do nothing
    return -1;
}
#endif /* HAVE_PTHREAD_SETAFFINITY_NP */


// forward decl
/*
 * \brief Proxy thread routine
 *
 */
static void * proxy_thread_routine(void * arg);

/*!
 *  \class ff_thread
 *  \ingroup buiding_blocks
 *
 *  \brief thread container for (leaves) ff_node
 *
 * It defines FastFlow's threading abstraction to run ff_node in parallel
 * in the shared-memory runtime
 *
 * \note Should not be used directly, it is called by ff_node
 */
class ff_thread {

    friend void * proxy_thread_routine(void *arg);

protected:
    ff_thread(BARRIER_T * barrier=NULL, bool default_mapping=true):
        tid((size_t)-1),threadid(0), default_mapping(default_mapping),
        barrier(barrier), stp(true), // only one shot by default
        spawned(false), freezing(0), frozen(false),isdone(false),
        init_error(false), attr(NULL) {
        (void)FF_TAG_MIN; // to avoid warnings
        
        /* Attr is NULL, default mutex attributes are used. Upon successful
         * initialization, the state of the mutex becomes initialized and
         * unlocked. 
         * */
        if (pthread_mutex_init(&mutex,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_mutex_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
        if (pthread_cond_init(&cond_frozen,NULL)!=0) {
            error("FATAL ERROR: ff_thread: pthread_cond_init fails!\n");
            abort();
        }
    }

    virtual ~ff_thread() {}
    
    void thread_routine() {
        threadid = ff_getThreadID();
#if defined(FF_INITIAL_BARRIER)
        if (barrier) {
            barrier->doBarrier(tid);
        }
        /* else {
         *    printf("THREAD %ld skip barrier\n", threadid);
         * }
         */
#endif
        void * ret;
        do {
            init_error=false;
            if (svc_init()<0) {
                error("ff_thread, svc_init failed, thread exit!!!\n");
                init_error=true;
                break;
            } else  {
                ret = svc(NULL);
            }
            svc_end();
            
            if (disable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }

            // acquire lock. While freezing is true,
            // freeze and wait. 
            pthread_mutex_lock(&mutex);
            if (ret != FF_EOS_NOFREEZE && !stp) {
                if ((freezing == 0) && (ret == FF_EOS)) stp = true;
                while(freezing==1) { // NOTE: freezing can change to 2
                    frozen=true; 
                    pthread_cond_signal(&cond_frozen);
                    pthread_cond_wait(&cond,&mutex);
                }
            }
            
            //thawed=true;
            //pthread_cond_signal(&cond);
            //frozen=false; 
            if (freezing != 0) freezing = 1; // freeze again next time 
            pthread_mutex_unlock(&mutex);

            if (enable_cancelability()) {
                error("ff_thread, thread_routine, could not change thread cancelability");
                return;
            }
        } while(!stp);
        
        if (freezing) {
            pthread_mutex_lock(&mutex);
            frozen=true;
            pthread_cond_signal(&cond_frozen);
            pthread_mutex_unlock(&mutex);
        }
        isdone = true;
    }

    int disable_cancelability() {
        if (pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancelstate)) {
            perror("pthread_setcanceltype");
            return -1;
        }
        return 0;
    }

    int enable_cancelability() {
        if (pthread_setcancelstate(old_cancelstate, 0)) {
            perror("pthread_setcanceltype");
            return -1;
        }
        return 0;
    }


#if defined(FF_TASK_CALLBACK)
    virtual void callbackIn(void  * =NULL) { }
    virtual void callbackOut(void * =NULL) { }
#endif
    
public:
 
    virtual void* svc(void * task) = 0;
    virtual int   svc_init() { return 0; };
    virtual void  svc_end()  {}

    virtual void set_barrier(BARRIER_T * const b) { barrier=b;}
    virtual BARRIER_T* get_barrier() const { return barrier; }

    virtual void no_mapping() { default_mapping=false; }
    bool get_mapping() const { return default_mapping; }
    
    virtual int run(bool=false) { return spawn(); }
    
    virtual int spawn(int cpuId=-1) {
        if (spawned) return -1;

        if ((attr = (pthread_attr_t*)malloc(sizeof(pthread_attr_t))) == NULL) {
            error("spawn: pthread can not be created, malloc failed\n");
            return -1;
        }
        if (pthread_attr_init(attr)) {
                perror("pthread_attr_init: pthread can not be created.");
                return -1;
        }

        int CPUId = -1;
        if (default_mapping)
            init_thread_affinity(attr, cpuId);
        if (CPUId==-2) return -2;

        if (barrier)
            tid= internal_threadCounter.fetch_add(1);
        else
            tid= internal_threadCounter_noBarrier.fetch_add(1);
        int r=0;
        if ((r=pthread_create(&th_handle, attr,
                              proxy_thread_routine, this)) != 0) {
            errno=r;
            perror("pthread_create: pthread creation failed.");
            barrier?--internal_threadCounter:--internal_threadCounter_noBarrier;
            return -2;
        }
        spawned = true;
        return CPUId;
    }
     
    virtual int wait() {
        int r=0;
        stp=true;
        if (isfrozen()) {
            wait_freezing();
            thaw();
        }
        if (spawned) {
            pthread_join(th_handle, NULL);
            barrier ? --internal_threadCounter: --internal_threadCounter_noBarrier;
        }
        if (attr) {
            if (pthread_attr_destroy(attr)) {
                error("ERROR: ff_thread.wait: pthread_attr_destroy fails!");
                r=-1;
            }        
            free(attr);
            attr = NULL;
        }
        spawned=false;
        return r;
    }

    virtual int wait_freezing() {
        pthread_mutex_lock(&mutex);
        while(!frozen) pthread_cond_wait(&cond_frozen,&mutex);
        pthread_mutex_unlock(&mutex);
        return (init_error?-1:0);
    }

    virtual void stop() { stp = true; };

    virtual void freeze() {  
        stp=false;
        freezing = 1;
    }
    
    virtual void thaw(bool _freeze=false, ssize_t=-1) {
        pthread_mutex_lock(&mutex);
        // if this function is called even if the thread is not 
        // in frozen state, then freezing has to be set to 1 and not 2
        //if (_freeze) freezing= (frozen?2:1); // next time freeze again the thread
        // October 2014, changed the above policy.
        // If thaw is called and the thread is not in the frozen stage, 
        // then the thread won't fall to sleep at the next freezing point

        if (_freeze) freezing = 2; // next time freeze again the thread
        else freezing=0;
        //assert(thawed==false);
        frozen=false; 
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);

        //pthread_mutex_lock(&mutex);
        //while(!thawed) pthread_cond_wait(&cond, &mutex);
        //thawed=false;
        //pthread_mutex_unlock(&mutex);
    }
    virtual bool isfrozen() const { return freezing>0;} 
    virtual bool done()     const { return isdone || (frozen && !stp);}

    pthread_t get_handle() const { return th_handle;}

    inline size_t getTid() const { return tid; }
    inline size_t getOSThreadId() const { return threadid; }

protected:
    size_t          tid;                /// unique logical id of the thread
    size_t          threadid;           /// OS specific thread ID
    bool            default_mapping;
private:
    BARRIER_T    *  barrier;            /// A \p Barrier object
    bool            stp;
    bool            spawned;
    int             freezing;  
    bool            frozen,isdone;
    bool            init_error;
    pthread_t       th_handle;
    pthread_attr_t *attr;
    pthread_mutex_t mutex; 
    pthread_cond_t  cond;
    pthread_cond_t  cond_frozen;
    int             old_cancelstate;
};
    
static void * proxy_thread_routine(void * arg) {
    ff_thread & obj = *(ff_thread *)arg;
    obj.thread_routine();
    pthread_exit(NULL);
    return NULL;
}

// forward declaration    
class ff_loadbalancer;
class ff_gatherer;

/*!
 *  \class ff_node
 *  \ingroup building_blocks
 *
 *  \brief The FastFlow abstract contanier for a parallel activity (actor).
 *
 * Implements \p ff_node, i.e. the general container for a parallel
 * activity. From the orchestration viewpoint, the process model to
 * be employed is a CSP/Actor hybrid model where activities (\p
 * ff_nodes) are named and the data paths between processes are
 * clearly identified. \p ff_nodes synchronise each another via
 * abstract units of SPSC communications and synchronisation (namely
 * 1:1 channels), which models data dependency between two
 * \p ff_nodes.  It is used to encapsulate
 * sequential portions of code implementing functions. 
 *
 * \p In a multicore, a ff_node is implemented as non-blocking thread. 
 * It is not and should
 * not be confused with a task. Typically a \p ff_node uses the 100% of one CPU
 * context (i.e. one core, either physical or HT, if any). Overall, the number of
 * ff_nodes running should not exceed the number of logical cores of the platform.
 * 
 * \p A ff_node behaves as a loop that gets an input (i.e. the parameter of \p svc 
 * method) and produces one or more outputs (i.e. return parameter of \p svc method 
 * or parameter of the \p ff_send_out method that can be called in the \p svc method). 
 * The loop complete on the output of the special value "end-of_stream" (EOS). 
 * The EOS is propagated across channels to the next \p ff_node.  
 * 
 * Key methods are: \p svc_init, \p svc_end (optional), and \p svc (pure virtual, 
 * mandatory). The \p svc_init method is called once at node initialization,
 * while the \p svn_end method is called after a EOS task has been returned. 
 *
 *  This class is defined in \ref node.hpp
 */

class ff_node {
private:

    friend class ff_farm;
    friend class ff_pipeline;
    friend class ff_map;
    template <typename IN,typename OUT>
    friend class ff_nodeSelector;
    friend class ff_loadbalancer;
    friend class ff_gatherer;
    friend class ff_minode;
    friend class ff_monode;
    friend class ff_a2a;
    friend class ff_comb;
    friend struct internal_mo_transformer;
    friend struct internal_mi_transformer;

#ifdef DFF_ENABLED
    friend class dGroups;
    friend class dGroup;
#endif

private:
    FFBUFFER        * in;           ///< Input buffer, built upon SWSR lock-free (wait-free) 
                                    ///< (un)bounded FIFO queue                                 
    FFBUFFER        * out;          ///< Output buffer, built upon SWSR lock-free (wait-free) 
                                    ///< (un)bounded FIFO queue 
    ssize_t           myid;         ///< This is the node id, it is valid only for farm's workers
    ssize_t           CPUId;
    ssize_t           neos=1;       ///< n. of EOS the node expects to receive before terminating 
    bool              myoutbuffer;
    bool              myinbuffer;
    bool              skip1pop;
#ifdef DFF_ENABLED
    bool _skipallpop;
#endif

    bool              in_active;    // allows to disable/enable input tasks receiving   
    bool              my_own_thread;

    ff_thread       * thread;       /// A \p thWorker object, which extends the \p ff_thread class 
    bool (*callback)(void *, int, unsigned long,unsigned long, void *);
    void            * callback_arg;
    BARRIER_T       * barrier;      /// A \p Barrier object
    struct timeval tstart;
    struct timeval tstop;
    struct timeval wtstart;
    struct timeval wtstop;
    double wttime;

protected:
    
    virtual void set_id(ssize_t id) {
        myid = id;
    }
    // sets how many EOSs the node has to receive before terminating,
    // it also sets when eosnotify has to be called, by default at each input EOS    
    virtual void set_neos(ssize_t n) { neos = n; }
    
    virtual inline bool push(void * ptr) { return out->push(ptr); }
    virtual inline bool pop(void ** ptr) { 
        if (!in_active) return false; // it does not want to receive data
        return in->pop(ptr);
    }
    virtual inline bool Push(void *ptr, unsigned long retry=((unsigned long)-1), unsigned long ticks=(TICKS2WAIT)) {
        if (blocking_out) {
        retry:
            bool empty=out->empty();
            bool r = push(ptr);
            if (r) { // OK
                if (empty) pthread_cond_signal(p_cons_c);
            } else { // FULL
                struct timespec tv;
                timedwait_timeout(tv);
                pthread_mutex_lock(prod_m);
                pthread_cond_timedwait(prod_c,prod_m,&tv);
                pthread_mutex_unlock(prod_m);
                goto retry;
            }
            return true;
        }
        for(unsigned long i=0;i<retry;++i) {
            if (push(ptr)) return true;
            losetime_out(ticks);
        }     
        return false;
    }
    
    virtual inline bool Pop(void **ptr, unsigned long retry=((unsigned long)-1), unsigned long ticks=(TICKS2WAIT)) {
        if (blocking_in) {
            if (!in_active) { *ptr=NULL; return false; }
        retry:
            bool r = in->pop(ptr);
            if (!r) { // EMPTY                
                struct timespec tv;
                timedwait_timeout(tv);
                pthread_mutex_lock(cons_m);
                pthread_cond_timedwait(cons_c, cons_m,&tv);
                pthread_mutex_unlock(cons_m);
                goto retry;
            }
            return true;
        }
        for(unsigned long i=0;i<retry;++i) {
            if (!in_active) { *ptr=NULL; return false; }
            if (pop(ptr)) return true;
            losetime_in(ticks);
        } 
        return true;
    }


    // consumer
    virtual inline bool init_input_blocking(pthread_mutex_t   *&m,
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
    virtual inline bool init_output_blocking(pthread_mutex_t   *&m,
                                             pthread_cond_t    *&c,
                                             bool /*feedback*/=true) {
        if (prod_m == nullptr) {
            assert(prod_c==nullptr);
            prod_m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
            prod_c = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
            assert(prod_m); assert(prod_c);
            if (pthread_mutex_init(prod_m, NULL) != 0) return false;
            if (pthread_cond_init(prod_c, NULL) != 0)  return false;
        } 
        m = prod_m, c = prod_c;
        return true;
    }
    virtual inline void set_output_blocking(pthread_mutex_t   *&m,
                                            pthread_cond_t    *&c,
                                            bool canoverwrite=false) {
        assert(canoverwrite ||
               (p_cons_c == nullptr) ||
               (p_cons_c == c));
        FF_IGNORE_UNUSED(canoverwrite);
        FF_IGNORE_UNUSED(m);
        p_cons_c = c;
    }

    // this function is used mainly for combined node where the cond variable must
    // be shared with the first internal node 
    virtual inline void  set_cons_c(pthread_cond_t *c) {
        assert(cons_c == nullptr);
        assert(cons_m == nullptr);
        cons_c = c;
    }        
    virtual inline pthread_cond_t    &get_cons_c()       { return *cons_c;}

    /**
     * \brief Set the ff_node to start with no input task
     *
     * Setting it to true let the \p ff_node execute the \p svc method spontaneusly 
     * before receiving a task on the input channel. \p skipfirstpop makes it possible
     * to define a "producer" node that starts the network.
     *
     * \param sk \p true start spontaneously (*task will be NULL)
     *
     */
    virtual inline void skipfirstpop(bool sk)   { skip1pop=sk;}

#ifdef DFF_ENABLED
    virtual inline void skipallpop(bool sk) {_skipallpop = sk;}
#endif

    /** 
     * \brief Gets the status of spontaneous start
     * 
     * If \p true the \p ff_node execute the \p svc method spontaneusly
     * before receiving a task on the input channel. \p skipfirstpop makes it possible
     * to define a "producer" node that produce the stream.
     * 
     * \return \p true if skip-the-first-element mode is set, \p false otherwise
     * 
     * Example: \ref l1_ff_nodes_graph.cpp
     */
    bool skipfirstpop() const { return skip1pop; }

#ifdef DFF_ENABLED
    bool skipallpop() {return _skipallpop;}
#endif

    
    /** 
     * \brief Creates the input channel 
     *
     *  \param nentries: the number of elements of the buffer
     *  \param fixedsize flag to decide whether the buffer is bound or unbound.
     *  Default is \p true.
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int create_input_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        if (in) return -1;
        if (nentries<=0) return -1;
        in = new FFBUFFER(nentries,fixedsize);
        if (!in) return -1;
        myinbuffer=true;
        if (!in->init()) return -1;
        return 0;
    }

    virtual int create_input_buffer_mp(int nentries, bool fixedsize=FF_FIXED_SIZE, int neos=1) {
        if (create_input_buffer(nentries,fixedsize)<0) return -1;
        // setting multi-producer push
        in->pushPMF = &FFBUFFER::mp_push;
        set_neos(neos);
        return 0;
    }
    
    /** 
     *  \brief Creates the output channel
     *
     *  \param nentries: the number of elements of the buffer
     *  \param fixedsize flag to decide whether the buffer is bound or unbound.
     *  Default is \p true.
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int create_output_buffer(int nentries, bool fixedsize=FF_FIXED_SIZE) {
        if (out) return -1;
        if (nentries<=0) return -1;
        out = new FFBUFFER(nentries,fixedsize); 
        if (!out) return -1;
        myoutbuffer=true;
        if (!out->init()) return -1;
        return 0;
    }

    /** 
     *  \brief Assign the output channelname to a channel
     *
     * Attach the output of a \p ff_node to an existing channel, typically the input 
     * channel of another \p ff_node
     *
     *  \param o reference to a channel of type \p FFBUFFER
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int set_output_buffer(FFBUFFER * const o) {
        if (myoutbuffer) return -1;        
        out = o;
        return 0;
    }


    /** 
     *  \brief Assign the input channelname to a channel
     *
     * Attach the input of a \p ff_node to an existing channel, typically the output 
     * channel of another \p ff_node
     *
     *  \param i a buffer object of type \p FFBUFFER
     *
     *  \return 0 if successful, -1 otherwise
     */
    virtual int set_input_buffer(FFBUFFER * const i) {
        if (myinbuffer) return -1;
        in = i;
        return 0;
    }

    virtual inline int set_input(const svector<ff_node *> &) { return -1;}
    virtual inline int set_input(ff_node *n) {
        return set_input_buffer(n->get_in_buffer());
    }
    virtual inline int set_input_feedback(ff_node *) { return -1;}
    virtual inline int set_output(const svector<ff_node *> &) { return -1;}
    virtual inline int set_output(ff_node *n) {
        return set_output_buffer(n->get_in_buffer());
    }
    virtual inline int set_output_feedback(ff_node *) { return -1;}
    virtual inline void set_input_channelid(ssize_t, bool=true) {}
        
    virtual int prepare() { prepared=true; return 0; }
    virtual int dryrun() { if (!prepared) return prepare(); return 0; }

    virtual void set_scheduling_ondemand(const int /*inbufferentries*/=1) {} 
    virtual int ondemand_buffer() const { return 0;} 

    
    /**
     * \brief Run the ff_node
     *
     * \return 0 success, -1 otherwise
     */
    virtual int run(bool=false) { 
        if (thread) delete reinterpret_cast<thWorker*>(thread);
        thread = new thWorker(this,neos);
        if (!thread) return -1;
        return thread->run();
    }

    #ifdef DFF_ENABLED
    virtual int run(ff_node*, bool=false) {return 0;}
    #endif     
    
    /**
     * \brief Suspend (freeze) the ff_node and run it
     *
     * Only initialisation will be performed
     *
     * \return 0 success, -1 otherwise
     */
    virtual int freeze_and_run(bool=false) {
        if (thread) delete reinterpret_cast<thWorker*>(thread);
        thread = new thWorker(this,neos);
        if (!thread) return 0;
        freeze();
        return thread->run();
    }

    /**
     * \brief Wait ff_node termination
     *
     * \return 0 success, -1 otherwise
     */
    virtual int  wait() { 
        if (!thread) return 0;
        return thread->wait(); 
    }
    
    /**
     * \brief Wait the freezing state
     *
     * It will happen on EOS arrival on the input channel
     *
     * \return 0 success, -1 otherwise
     */
    virtual int  wait_freezing() { 
        if (!thread) return 0;
        return thread->wait_freezing(); 
    }
    
    virtual void stop() {
        if (!thread) return; 
        thread->stop(); 
    }
    
    /**
     * \brief Freeze (suspend) a ff_node
     */
    virtual void freeze() { 
        if (!thread) return; 
        thread->freeze(); 
    }
    
    /**
     * \brief Thaw (resume) a ff_node
     */
    virtual void thaw(bool _freeze=false, ssize_t=-1) { 
        if (!thread) return; 
        thread->thaw(_freeze);
    }
    
    /**
     * \brief Checks if a ff_node is frozen
     * \return \p true is it frozen
     */
    virtual bool isfrozen() const { 
        if (!thread) return false;
        return thread->isfrozen();
    }

    /**
     * \brief checks if the node is running 
     *
     */
    virtual bool done() const  {
        if (!thread) return true;
        return thread->done();
    }


    virtual bool isoutbuffermine() const { return myoutbuffer;}

    virtual int  cardinality(BARRIER_T * const b) { 
        barrier = b;
        return 1;
    }
    virtual int  cardinality() const { return 1; }

    virtual inline void setlb(ff_loadbalancer*,bool=false) {}
    virtual inline void setgt(ff_gatherer*,bool=false) {}

    
    /**
     * \brief Misure \ref ff::ff_node execution time
     *
     * \return time (ms)
     */
    virtual double ffTime() {
        return diffmsec(tstop,tstart);
    }

    /**
     * \brief Misure \ref ff_node::svc execution time
     *
     * \return time (ms)
     */
    virtual double wffTime() {
        return diffmsec(wtstop,wtstart);
    }

public:
    /*
     * \brief Default retry delay in nonblocking get/put on channels
     */
    enum {TICKS2WAIT=1000};

    void *const GO_ON        = FF_GO_ON;
    void *const GO_OUT       = FF_GO_OUT;
    void *const EOS_NOFREEZE = FF_EOS_NOFREEZE;
    void *const EOS          = FF_EOS;
    void *const EOSW         = FF_EOSW;

    
    ff_node(const ff_node&):ff_node() {}
 
    /** 
     *  \brief Destructor, polymorphic deletion through base pointer is allowed.
     *
     *  
     */
    virtual  ~ff_node() {
        if (in && myinbuffer) delete in;
        if (out && myoutbuffer) delete out;
        if (thread && my_own_thread) delete reinterpret_cast<thWorker*>(thread);
        if (cons_c && cons_m) {
            pthread_cond_destroy(cons_c);
            free(cons_c);
            cons_c = nullptr;
        }
        if (cons_m) {
            pthread_mutex_destroy(cons_m);
            free(cons_m);
            cons_m = nullptr;
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
    };

    /**
     * \brief The service callback (should be filled by user with parallel activity business code)
     *
     * \param task is a the input data stream item pointer (task)
     * \return output data stream item pointer
     */
    virtual void* svc(void * task) = 0;
        
    /**
     * \brief Service initialisation
     *
     * Called after run-time initialisation (e.g. thread spawning) but before 
     * to start to get items from input stream (can be useful for initialisation
     * of parallel activities, e.g. manual thread pinning that cannot be done in
     * the costructor because threads stil do not exist).
     *
     * \return 0
     */
    virtual int svc_init() { return 0; }
    
    /**
     *
     * \brief Service finalisation
     *
     * Called after EOS arrived (logical termination) but before shutdding down
     * runtime support (can be useful for housekeeping)
     */
    virtual void  svc_end() {}
    

    /**
     * \brief Node initialisation
     *
     * This is a different initialization method with respect to svc_init (the default method).
     * This can be used to explicitly initialize the object when the node is not running as a thread.
     *
     * \return 0
     */
    virtual int   nodeInit() { return 0; }

    /**
     * \brief Node finalisation.
     *
     * This is a different finalisation method with respect to svc_end (the default method).
     * This can be used to explicitly finalise the object when the node is not running as a thread.
     */
    virtual void  nodeEnd()  { }

    /**
     * \brief EOS callback
     *
     * This method is called when an EOS has just been received from one input channel. 
     * Inside this method it is possible to call ff_send_out to produce data elements in output 
     * (this is not possible in the svc_end method).
     * The parameter \param id is the ID of the channel that received the EOS. 
     */
    virtual void eosnotify(ssize_t /*id*/=-1) {}

    /**
     * \brief Returns the number of EOS the node has to receive before terminating.
     */    
    virtual ssize_t get_neos() const { return neos;}

    /**
     *  \brief Returns the identifier of the node (not unique)
     */    
    virtual ssize_t get_my_id() const { return myid; };

    /**
     * \brief Returns the OS specific thread id of the node.
     *
     * The returned id is valid (>0) only if the node is an active node (i.e. the thread has been created).
     *
     */
    inline size_t getOSThreadId() const { if (thread) return thread->getOSThreadId(); return 0; }

    virtual bool change_node(ff_node* old, ff_node* n, bool cleanup=false, bool remove_from_cleanuplist=false) { return false;}

    /**
     * Change the size of the outputchannel. 
     * WARNING: this method should not be used if the queue is being used!!!!
     *
     */
    virtual bool change_outputqueuesize(size_t newsz, size_t &oldsz) {
        if (!out) { oldsz=0; return false; }
        oldsz = out->changesize(newsz);
        return true;
    }
    /**
     * Change the size of the inputchannel. 
     * WARNING: this method should not be used if the queue is being used!!!!
     *
     */
    virtual bool change_inputqueuesize(size_t newsz, size_t &oldsz) {
        if (!in) { oldsz=0; return false; }
        oldsz = in->changesize(newsz);
        return true;
    }

    
#if defined(FF_TASK_CALLBACK)
    virtual void callbackIn(void * =NULL)  { }
    virtual void callbackOut(void * =NULL) { }
#endif

    virtual inline void get_out_nodes(svector<ff_node*>&w) { w.push_back(this); }
    virtual inline void get_out_nodes_feedback(svector<ff_node*>&) {}
    virtual inline void get_in_nodes(svector<ff_node*>&w) { w.push_back(this); }
    virtual inline void get_in_nodes_feedback(svector<ff_node*>&) {}

    
    /**
     * \brief Force ff_node-to-core pinning
     *
     * \param cpuID is the ID of the CPU to which the thread will be pinned.
     */
    virtual void  setAffinity(int cpuID) { 
        if (cpuID<0 || !threadMapper::instance()->checkCPUId(cpuID) ) {
            error("setAffinity, invalid cpuID\n");
        }
        CPUId=cpuID;
    }

    virtual void set_barrier(BARRIER_T * const b) {
        barrier = b;
    }
    virtual BARRIER_T* get_barrier() const { return barrier; }
    
    /** 
     * \internal
     * \brief Gets the CPU id (if set) of this node is pinned
     *
     * It gets the ID of the CPU where the ff_node is running.
     *
     * \return The identifier of the CPU.
     */
    virtual int getCPUId() const { return CPUId; }

    /**
     * \brief Nonblocking put into the input channel
     *
     * Wait-free and fence-free (under TSO)
     * This is called by a different node (e.g., lb) to push data
     * into the node's input queue.
     *
     * \param ptr is a pointer to the task
     *
     */
    virtual inline bool  put(void * ptr) { 
        //return in->push(ptr);
        return (in->*in->pushPMF)(ptr);
    }
    
    /**
     * \brief Noblocking pop from the output channel
     *
     * Wait-free and fence-free (under TSO)
     *
     * \param ptr is a pointer to the task
     *
     */
    virtual inline bool  get(void **ptr) { return out->pop(ptr);}
   
    virtual inline void losetime_out(unsigned long ticks=ff_node::TICKS2WAIT) {
        FFTRACE(lostpushticks+=ticks; ++pushwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }

    virtual inline void losetime_in(unsigned long ticks=ff_node::TICKS2WAIT) {
        FFTRACE(lostpopticks+=ticks; ++popwait);
#if defined(SPIN_USE_PAUSE)
        const long n = (long)ticks/2000;
        for(int i=0;i<=n;++i) PAUSE();
#else
        ticks_wait(ticks);
#endif /* SPIN_USE_PAUSE */
    }

    /**
     * \brief Gets input channel
     *
     * It returns a pointer to the input buffer.
     *
     * \return A pointer to the input buffer
     */
    virtual FFBUFFER * get_in_buffer() const { return in;}

    /**
     * \brief Gets pointer to the output channel
     *
     * It returns a pointer to the output buffer.
     *
     * \return A pointer to the output buffer.
     */
    virtual FFBUFFER * get_out_buffer() const { return out;}

    virtual const struct timeval getstarttime() const { return tstart;}

    virtual const struct timeval getstoptime()  const { return tstop;}

    virtual const struct timeval getwstartime() const { return wtstart;}

    virtual const struct timeval getwstoptime() const { return wtstop;}    

#if defined(TRACE_FASTFLOW)
    virtual void ffStats(std::ostream & out) {
        out << "ID: " << get_my_id()
            << "  work-time (ms): " << wttime    << "\n"
            << "  n. tasks      : " << taskcnt   << "\n"
            << "  svc ticks     : " << tickstot  << " (min= " << ticksmin << " max= " << ticksmax << ")\n"
            << "  n. push lost  : " << pushwait  << " (ticks=" << lostpushticks << ")" << "\n"
            << "  n. pop lost   : " << popwait   << " (ticks=" << lostpopticks  << ")" << "\n";
    }

    virtual double getworktime() const { return wttime; }
    virtual size_t getnumtask()  const { return taskcnt; }
    virtual ticks  getsvcticks() const { return tickstot; }
    virtual size_t getpushlost() const { return pushwait;}
    virtual size_t getpoplost()  const { return popwait; }
#endif

    /**
     * \brief Sends out the task
     *
     * It allows to emit tasks on output stream without returning from the \p svc method.
     * Make the ff_node to emit zero or more tasks per input task
     *
     * \param task a pointer to the task
     * \param retry number of tries to put (nonbloking partial) the task to output channel
     * \param ticks delay between successive retries
     * 
     */
    virtual bool ff_send_out(void * task, int id=-1,
                             unsigned long retry=((unsigned long)-1),
                             unsigned long ticks=(TICKS2WAIT)) { 
        if (callback) return  callback(task,id,retry,ticks,callback_arg);
        bool r =Push(task,retry,ticks);
#if defined(FF_TASK_CALLBACK)
        if (r) callbackOut();
#endif
        return r;
    }

    // Warning resetting queues while the node is running may produce unexpected results.
    virtual void reset() {
        if (in)  in->reset();
        if (out) out->reset();
    }

    /** 
     *  checking for multi-input/output, all-to-all, farm, pipe
     *
     */
    virtual inline bool isMultiInput() const {  return false; }
    virtual inline bool isMultiOutput() const { return false; }
    virtual inline bool isAll2All() const     { return false; }
    virtual inline bool isFarm() const        { return false; }
    virtual inline bool isOFarm() const       { return false; }
    virtual inline bool isComp() const        { return false; }
    virtual inline bool isPipe() const        { return false; }

    virtual inline void set_multiinput()  {}
    
#if defined(FF_REPARA)
    struct rpr_measure_t {
        size_t schedule_id;
        size_t time_before, time_after;
        size_t problemSize;  // computed if the rpr::task_size attribute is defined otherwise is 0
        size_t bytesIn, bytesOut;
        size_t vmSize, vmPeak;
        double energy;
    };
    
    using RPR_devices_measure = std::vector<std::pair<int, std::vector<rpr_measure_t> > >;
    using RPR_measures_vector = std::vector<std::vector<RPR_devices_measure> >;

    /** 
     *  Returns input data size
     */
    virtual size_t rpr_get_sizeIn()  const { return rpr_sizeIn; }

    /** 
     *  Returns output data size
     */
    virtual size_t rpr_get_sizeOut() const { return rpr_sizeOut; }

    /** 
     *  gets/sets energy flag
     */
    virtual bool rpr_get_measure_energy() const { return measureEnergy; }
    virtual void rpr_set_measure_energy(bool v) { measureEnergy = v; }

    /**
     *  Returns all measures collected by the node.
     *  The structure is:
     *    - the outermost vector is greater than 1 if the node is a pipeline or a farm
     *    - each stage of a pipeline or a worker of a farm can be a pipeline or a farm as well
     *      therefore the second level vector is grater than 1 only if the stage is a pipeline or a farm
     *    - each entry of a stage is a vector containing info for each device associated to the stage.
     *      The device is identified by the first entry of the std::pair, the second element of the pair 
     *      is a vector containing the measurments for the period considered.
     */
    virtual RPR_measures_vector rpr_get_measures() { return RPR_measures_vector(); }

    
protected: 
    bool   measureEnergy = false;
    size_t rpr_sizeIn      = {0};
    size_t rpr_sizeOut     = {0};
#endif  /* FF_REPARA */

    /** 
     *  used for composition (see ff_comb)
     */
    static inline bool ff_send_out_comp(void * task, int, unsigned long /*retry*/,unsigned long /*ticks*/, void *obj) {
        return ((ff_node *)obj)->push_comp_local(task);
    }


    virtual bool push_comp_local(void *task) {
        (void)task;
        abort();  // to be removed, just for debugging purposes
    }


    virtual inline ssize_t get_channel_id() const           { return -1; }
    /** returns the total number of output channels */
    virtual inline size_t  get_num_outchannels() const      { return 0; }
    /** returns the total number of input channels */
    virtual inline size_t  get_num_inchannels() const       { return 0; } //(in?1:0); }
    virtual inline size_t  get_num_feedbackchannels() const { return 0; } //(out?1:0);}
    
    virtual void propagateEOS(void* task=FF_EOS) { (void)task; }
    
#ifdef DFF_ENABLED
    std::function<bool(void*, dataBuffer&)> serializeF;
    std::function<void(void*)> freetaskF;
    std::function<void*(dataBuffer&, bool&)> deserializeF;
    std::function<void*(char*, size_t)> alloctaskF;

    
    virtual bool isSerializable(){ return (bool)serializeF; }
    virtual bool isDeserializable(){ return (bool)deserializeF; }
    virtual std::pair<decltype(serializeF), decltype(freetaskF)> getSerializationFunction(){return std::make_pair(serializeF,freetaskF);}
    virtual std::pair<decltype(deserializeF), decltype(alloctaskF)> getDeserializationFunction(){ return std::make_pair(deserializeF,alloctaskF);}

#endif
    // always defined, the body will implement a no-op if the distributed runtime is disabled
    GroupInterface createGroup(std::string);
    
protected:

    ff_node():in(0),out(0),myid(-1),CPUId(-1),
              myoutbuffer(false),myinbuffer(false),
              skip1pop(false), in_active(true), 
              my_own_thread(true),
              thread(NULL),callback(NULL),barrier(NULL) {
        time_setzero(tstart);time_setzero(tstop);
        time_setzero(wtstart);time_setzero(wtstop);
        wttime=0;
        FFTRACE(taskcnt=0;lostpushticks=0;pushwait=0;lostpopticks=0;popwait=0;ticksmin=(ticks)-1;ticksmax=0;tickstot=0);
        
        p_cons_c = NULL;

        blocking_in = blocking_out = FF_RUNTIME_MODE;
    };

    
    // move constructor
    ff_node(ff_node &&n) {
        tstart = n.tstart;
        tstop  = n.tstop;
        wtstart = n.wtstart;
        wtstop = n.wtstop;
        wttime = n.wttime;
        p_cons_c = n.p_cons_c;
        blocking_in = n.blocking_in;
        blocking_out = n.blocking_out;
        default_mapping = n.default_mapping;
        in_active = n.in_active;
        cons_m = n.cons_m;  cons_c = n.cons_c;
        prod_m = n.prod_m;  prod_c = n.prod_c;
        barrier = n.barrier;

        // TODO trace <------
        
        in = n.in;
        myinbuffer = n.myinbuffer;
        out = n.out;
        myoutbuffer = n.myoutbuffer;
        thread = n.thread;
        my_own_thread = n.my_own_thread;

        n.in = nullptr;
        n.myinbuffer = false;
        n.out = nullptr;
        n.myoutbuffer = false;
        n.thread = nullptr;
        n.my_own_thread = false;
        n.barrier = nullptr;
        n.cons_m = nullptr; n.cons_c = nullptr;
        n.prod_m = nullptr; n.prod_c = nullptr;
    }

    virtual inline void input_active(const bool onoff) {
        if (in_active != onoff)
            in_active= onoff;
    }

    virtual void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
        callback=cb;
        callback_arg=arg;
    }
    virtual void registerAllGatherCallback(int (* /*cb*/)(void *,void **, void*), void * /*arg*/) {}

    /* WARNING: these method must be called before the run() method */
    virtual void blocking_mode(bool blk=true) {
        blocking_in = blocking_out = blk;
    }
    virtual void no_barrier() {
        initial_barrier=false;
    }
    virtual void no_mapping() {
        default_mapping=false;
    }
    
private:  
    /* ------------------------------------------------------------------------------------- */
    class thWorker: public ff_thread {
    public:
        thWorker(ff_node * const filter, const ssize_t input_neos=1):
            ff_thread(filter->barrier, filter->default_mapping),filter(filter),input_neos(input_neos) {}
        
        inline bool push(void * task) {
            /* NOTE: filter->push and not buffer->push because of the filter can be a dnode
             *  
             * It is not correct to call filter->Push because the filter could be a composition
             * so the ff_send_out allows to call the callback
             */
            //return filter->Push(task);
            return filter->ff_send_out(task);
        }
        
        inline bool pop(void ** task) {
            /* 
             * NOTE: filter->pop and not buffer->pop because of the filter can be a dnode
             */
            return filter->Pop(task);
        }

        inline bool put(void * ptr) { return filter->put(ptr);}

        inline bool get(void **ptr) { return filter->get(ptr);}

        inline void* svc(void * ) {
            void * task = NULL;
            void * ret  = FF_EOS;
            bool inpresent  = (filter->get_in_buffer() != NULL);
            bool outpresent = (filter->get_out_buffer() != NULL);
            bool skipfirstpop = filter->skipfirstpop(); 
            bool exit=false;            
            bool filter_outpresent = false;
            size_t neos=input_neos;

            
            // if the node is a combine where the last stage is a multi-output
            if ( filter && ( !outpresent && filter->isMultiOutput() ) ) {
                filter_outpresent=true;
            }
            gettimeofday(&filter->wtstart,NULL);
            do {
#ifdef DFF_ENABLED
                if (!filter->skipallpop() && inpresent){
#else
                if (inpresent) {
#endif
                    if (!skipfirstpop) pop(&task); 
                    else skipfirstpop=false;
                    if ((task == FF_EOS) || (task == FF_EOSW) ||
                        (task == FF_EOS_NOFREEZE)) {
                        ret = task;
                        
                        if (--neos > 0) continue;  
                        filter->eosnotify();

                        // only EOS and EOSW are propagated
                        if ( (task == FF_EOS) || (task == FF_EOSW) )  {
                            if (outpresent)  push(task);
                            if (filter_outpresent) filter->propagateEOS();
                        }
                        break;
                    }
                    if (task == FF_GO_OUT) break;
                }
                FFTRACE(++filter->taskcnt);
                FFTRACE(ticks t0 = getticks());

#if defined(FF_TASK_CALLBACK)
                if (filter) callbackIn();
#endif                    

                ret = filter->svc(task);

#if defined(TRACE_FASTFLOW)
                ticks diff=(getticks()-t0);
                filter->tickstot +=diff;
                filter->ticksmin=(std::min)(filter->ticksmin,diff); // (std::min) for win portability)
                filter->ticksmax=(std::max)(filter->ticksmax,diff);
#endif           

                if (ret == FF_GO_OUT) break;     
                if (!ret || (ret >= FF_EOSW)) { // EOS or EOS_NOFREEZE or EOSW
                    // NOTE: The EOS is gonna be produced in the output queue
                    // and the thread exits even if there might be some tasks
                    // in the input queue !!!
                    if (!ret) ret = FF_EOS;
                    exit=true;
                }
                if ( outpresent && ((ret != FF_GO_ON) && (ret != FF_EOS_NOFREEZE)) ) { 
                    push(ret);
#if defined(FF_TASK_CALLBACK)
                    if (filter) callbackOut();
#endif
                }
            } while(!exit);
            
            gettimeofday(&filter->wtstop,NULL);
            filter->wttime+=diffmsec(filter->wtstop,filter->wtstart);
            
            return ret;
        }
        
        int svc_init() {
#if !defined(HAVE_PTHREAD_SETAFFINITY_NP) && !defined(NO_DEFAULT_MAPPING)
            if (filter->default_mapping) {
                int cpuId = filter->getCPUId();
                if (ff_mapThreadToCpu((cpuId<0) ? (cpuId=threadMapper::instance()->getCoreId(tid)) : cpuId)!=0)
                    error("Cannot map thread %d to CPU %d, mask is %u,  size is %u,  going on...\n",tid, (cpuId<0) ? threadMapper::instance()->getCoreId(tid) : cpuId, threadMapper::instance()->getMask(), threadMapper::instance()->getCListSize());            
                filter->setCPUId(cpuId);
            }
#endif
            gettimeofday(&filter->tstart,NULL);
            return filter->svc_init();
        }
        
        void svc_end() {
            filter->svc_end();
            gettimeofday(&filter->tstop,NULL);            
        }
        
        int run(bool=false) { 
            int CPUId = ff_thread::spawn(filter->getCPUId());             
            filter->setCPUId(CPUId);
            return (CPUId==-2)?-1:0;
        }

        inline int  wait() { return ff_thread::wait();}
        inline int  wait_freezing() { return ff_thread::wait_freezing();}
        inline void freeze() { ff_thread::freeze();}
        inline bool isfrozen() const { return ff_thread::isfrozen();}
        inline bool done()     const { return ff_thread::done();}
        inline int  get_my_id() const { return filter->get_my_id(); };
        
    protected:
#if defined(FF_TASK_CALLBACK)
        void callbackIn(void  *t=NULL) { filter->callbackIn(t);  }
        void callbackOut(void *t=NULL) { filter->callbackOut(t); }
#endif        
    protected:            
        ff_node * const filter;
        const ssize_t input_neos;
    };
    /* ------------------------------------------------------------------------------------- */

    inline void   setCPUId(int id) { CPUId = id;}
    inline void   setThread(ff_thread *const th) { my_own_thread = false; thread = th; }        
    inline size_t getTid() const {
        if (!thread) return (size_t)-1;
        return thread->getTid();
    } 

protected:

#if defined(TRACE_FASTFLOW)
    size_t        taskcnt;
    ticks         lostpushticks;
    size_t        pushwait;
    ticks         lostpopticks;
    size_t        popwait;
    ticks         ticksmin;
    ticks         ticksmax;
    ticks         tickstot;
#endif
    
    // for the input queue
    pthread_mutex_t    *cons_m = nullptr;
    pthread_cond_t     *cons_c = nullptr;


    // for the output queue
    pthread_mutex_t    *prod_m = nullptr;
    pthread_cond_t     *prod_c = nullptr;

    // for synchronizing with the next multi-input stage
    pthread_cond_t     *p_cons_c = nullptr;

    bool               FF_MEM_ALIGN(blocking_in,32); 
    bool               FF_MEM_ALIGN(blocking_out,32);

    bool                  prepared = false;
    bool                  initial_barrier = true;
    bool                  default_mapping = true;
};  // ff_node


/* *************************** Typed node ************************* */

//#ifndef WIN32 //VS12
/*!
 *  \class ff_node_base_t
 *  \ingroup building_blocks
 *
 *  \brief The FastFlow typed abstract contanier for a parallel activity (actor).
 *
 *  Key method is: \p svc (pure virtual).
 *
 *  This class is defined in \ref node.hpp
 */

template<typename IN_t, typename OUT_t = IN_t>
struct ff_node_t: ff_node {
    typedef IN_t  in_type;
    typedef OUT_t out_type;

    using ff_node::registerCallback;
    using ff_node::ff_send_out;
    
    ff_node_t():
        GO_ON((OUT_t*)FF_GO_ON),
        EOS((OUT_t*)FF_EOS),
        EOSW((OUT_t*)FF_EOSW),
        GO_OUT((OUT_t*)FF_GO_OUT),
        EOS_NOFREEZE((OUT_t*) FF_EOS_NOFREEZE) {
#ifdef DFF_ENABLED

        /* WARNING: 
         *    the definition of functions alloctaskF, freetaskF, serializeF, deserializeF
         *    IS DUPLICATED for the ff_minode_t and ff_monode_t (see file multinode.hpp).
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
         this->freetaskF = [](void* o) {
                               if constexpr (!std::is_void_v<OUT_t>) {
                                       OUT_t* obj = reinterpret_cast<OUT_t*>(o);
                                       delete obj;
                               }
                           };
     }
        
    // check on Serialization capabilities on the OUTPUT type!
    if constexpr (traits::is_serializable_v<OUT_t>){
        this->serializeF = [](void* o, dataBuffer& b) -> bool {
                               bool datacopied = true;
                               std::pair<char*, size_t> p = serializeWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o,datacopied));
                               b.setBuffer(p.first, p.second);
                               return datacopied;
                           };
    } else if constexpr (cereal::traits::is_output_serializable<OUT_t, cereal::PortableBinaryOutputArchive>::value){
        this->serializeF = [](void* o, dataBuffer& b) -> bool {
                               std::ostream oss(&b);
                               cereal::PortableBinaryOutputArchive ar(oss);
                               ar << *reinterpret_cast<OUT_t*>(o);
                               return true;
                           };
    }
    
    // check on Serialization capabilities on the INPUT type!
    if constexpr (traits::is_deserializable_v<IN_t>) {
        this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                                 IN_t* ptr=(IN_t*)this->alloctaskF(b.getPtr(), b.getLen());
                                 datacopied = deserializeWrapper<IN_t>(b.getPtr(), b.getLen(), ptr);
                                 assert(ptr);
                                 return ptr;
                             };
    } else if constexpr(cereal::traits::is_input_serializable<IN_t, cereal::PortableBinaryInputArchive>::value){
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
    virtual ~ff_node_t()  {}
    virtual OUT_t* svc(IN_t*)=0;
    inline  void *svc(void *task) { return svc(reinterpret_cast<IN_t*>(task)); };
private:
    // deleting some functions that do not have to be used in the svc
    using ff_node::push;
    using ff_node::pop;
    using ff_node::Push;
    using ff_node::Pop;

};

#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))

/*!
 *  \class ff_node_F
 *  \ingroup building_blocks
 *
 *  \brief The FastFlow typed abstract contanier for a parallel activity (actor).
 *
 *  Creates an ff_node_t from a lambdas, function pointer, etc
 *
 *  This class is defined in \ref node.hpp
 */
template<typename TIN, typename TOUT=TIN, 
         typename FUNC=std::function<TOUT*(TIN*,ff_node*const)> >
struct ff_node_F: public ff_node_t<TIN,TOUT> {
   ff_node_F(FUNC f):F(f) {}
   TOUT* svc(TIN* task) { return F(task, this); }
   FUNC F;
};

#endif
//#endif



/* ------------------------ internal node implementations, should not be used -------- */

/* just a node interface for the input and output buffers 
 * This is used in the internal implementation but can be used also
 * at the user level. In this second case
 */
struct ff_buffernode: ff_node {
    ff_buffernode() {}
    ff_buffernode(int nentries, bool fixedsize=FF_FIXED_SIZE, int id=-1, int multi_producer_eos=-1) {
        set(nentries,fixedsize,id, multi_producer_eos);
    }
    // NOTE: this constructor is supposed to be used only for implementing 
    // internal FastFlow features!
    ff_buffernode(int id, FFBUFFER *in, FFBUFFER *out) {
        set_id(id);
        set_input_buffer(in);
        set_output_buffer(out);
    }
    void set(int nentries, bool fixedsize=FF_FIXED_SIZE, int id=-1, int multi_producer_eos=-1) {
        set_id(id);
        if (multi_producer_eos<0) {
            if (create_input_buffer(nentries,fixedsize) < 0) {
                error("FATAL ERROR: ff_buffernode::set: create_input_buffer fails!\n");
                abort();
            }
            set_output_buffer(ff_node::get_in_buffer());
        } else {
            if (create_input_buffer_mp(nentries,fixedsize, multi_producer_eos) < 0) {
                error("FATAL ERROR: ff_buffernode::set: create_input_buffer_mp fails!\n");
                abort();
            }
            set_output_buffer(ff_node::get_in_buffer());
        }
    }

    int init_blocking_stuff() {
        // blocking stuff
        pthread_mutex_t   *m        = NULL;
        pthread_cond_t    *c        = NULL;
        if (!ff_node::init_output_blocking(m,c)) {
            error("buffernode, FATAL ERROR, init input blocking mode for accelerator\n");
            return -1;
        }
        if (!ff_node::init_input_blocking(m,c)) {
            error("buffernode, FATAL ERROR, init input blocking mode for accelerator\n");
            return -1;
        }
        return 0;
    }

    void reset_blocking_out() { blocking_out = false; }
    
    bool ff_send_out(void *ptr, int id=-1,
                     unsigned long retry=((unsigned long)-1), unsigned long ticks=(ff_node::TICKS2WAIT)) {
        return ff_node::ff_send_out(ptr,id,retry,ticks);
    }
    bool gather_task(void **task, unsigned long retry=((unsigned long)-1), unsigned long ticks=(ff_node::TICKS2WAIT)) {    
        bool r =ff_node::Pop(task,retry,ticks);
        return r;
    }

    template<typename T> 
    bool gather_task(T *&task, unsigned long retry=((unsigned long)-1), unsigned long ticks=(ff_node::TICKS2WAIT)) {    
        return gather_task((void **)&task, retry, ticks);
    }


protected:
    void* svc(void*){return NULL;}

    pthread_cond_t    &get_cons_c()       { return *p_cons_c;}


    // New Blocking protocol (both for bounded and unbounded buffer):
    // init_output_blocking initializes prod_*
    // set_output_blocking sets p_cons_*
    // init_input_blocking initializes cons_*

    // sender:
    //   empty=channel.empty();
    //   r=channel.send()
    //   if (!r) timewait(prod_c);
    //   if empty then signal(p_cons_c) // the channel was empty

    // receive:
    //  r=channel.receive()
    //  if (!r) timewait(cons_c);       // channel empty
};


    

    

} // namespace ff

#endif /* FF_NODE_HPP */
