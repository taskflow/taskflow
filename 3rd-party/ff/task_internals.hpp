/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file task_internals.hpp
 *  \ingroup aux_classes
 *
 *  \brief Internal classes and helping functions for tasks management.
 */

#ifndef FF_TASK_INTERNALS_HPP
#define FF_TASK_INTERNALS_HPP
 
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
 *
 */
#include <functional>
#include <tuple>
#include <vector>
#include <deque>
#include <queue>
#include <ff/allocator.hpp>
#include "icl_hash.h"

namespace ff {

/* -------------- expanding tuple to func. arguments ------------------------- */
template<size_t N>
struct ffApply {
    template<typename F, typename T, typename... A>
    static inline auto apply(F && f, T && t, A &&... a)
        -> decltype(ffApply<N-1>::apply(
            ::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...))
    {
        return ffApply<N-1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...
        );
    }
};

template<>
struct ffApply<0> {
    template<typename F, typename T, typename... A>
    static inline auto apply(F && f, T &&, A &&... a)
        -> decltype(::std::forward<F>(f)(::std::forward<A>(a)...))
    {
        return ::std::forward<F>(f)(::std::forward<A>(a)...);
    }
};

template<typename F, typename T>
inline auto ffapply(F && f, T && t)
    -> decltype(ffApply< ::std::tuple_size<typename ::std::decay<T>::type
					 >::value>::apply(::std::forward<F>(f), ::std::forward<T>(t)))
{
    return ffApply< ::std::tuple_size<typename ::std::decay<T>::type>::value>::apply(::std::forward<F>(f), ::std::forward<T>(t));
}
/* --------------------------------------------------------------------------- */ 

/// kind of dependency
typedef enum {INPUT=0,OUTPUT=1,VALUE=2} data_direction_t;
/// generic pameter information (tag and kind of dependency)
struct param_info {
    uintptr_t        tag;  // unique tag for the parameter
    data_direction_t dir;
};
/// base class for a generic function call
struct base_f_t {
    virtual inline void call() {};
    virtual inline void call(void*) {};
    virtual ~base_f_t() {};
};    
/// task function basic type
struct task_f_t { 
    std::vector<param_info> P;  // FIX: svector should be used here
    base_f_t *wtask;
};
  

/* ---------------------------------------------------------------------- 
 * Hashing funtions
 * Well known hash function: Fowler/Noll/Vo - 32 bit version
 */
static inline unsigned int fnv_hash_function( void *key, int len ) {
    unsigned char *p = (unsigned char*)key;
    unsigned int h = 2166136261u;
    int i;
    for ( i = 0; i < len; i++ )
        h = ( h * 16777619 ) ^ p[i];
    return h;
}
/**
 * Hash function to map addresses, cut into "long" size chunks, then
 * XOR. The result will be matched to hash table size using mod in the
 * hash table implementation
 */
static inline unsigned int address_hash_function(void *address) {
    int len = sizeof(void *);
    unsigned int hashval = fnv_hash_function( &address, len );
    return hashval;
}
/* Adress compare function for hash table */
static inline int address_key_compare(void *addr1, void *addr2) {
    return (addr1 == addr2);
}

static inline unsigned int ulong_hash_function( void *key ) {
    int len = sizeof(unsigned long);
    unsigned int hashval = fnv_hash_function( key, len );
    return hashval;
}
static inline int ulong_key_compare( void *key1, void *key2  ) {
    return ( *(unsigned long*)key1 == *(unsigned long*)key2 );
}

typedef enum {NOT_READY, READY, DONE, PENDING, PENDING_DONE} task_status_t;


struct hash_task_t {
    union{
        struct {
            base_f_t *wtask;
            unsigned long id;
            task_status_t status;
            bool     is_dummy;
            long     remaining_dep;  // dependencies counter
            long     unblock_numb;   // task list counter
            long     num_out;        // output edges
            //task list that have to be "unblocked"
            unsigned long *unblock_task_ids;
            long     unblock_act_numb; // current task list size
        };
        char padding[CACHE_LINE_SIZE];
    };
};
    
// Parallelism priority
struct CompareTask_Par {
    // Returns true if t1 is earlier than t2
    bool operator()(hash_task_t * &t1, hash_task_t* &t2) {
        if (t1->unblock_numb < t2->unblock_numb) return true;
        return false;
    }
};
// LIFO priority
struct CompareTask_LIFO {
    bool operator()(hash_task_t * &t1, hash_task_t* &t2) {
        if (t1->id < t2->id) return true;
        return false;
    }
};
// FIFO priority
struct CompareTask_FIFO {
    bool operator()(hash_task_t * &t1, hash_task_t* &t2) {
        if (t1->id > t2->id) return true;
        return false;
    }
};

/* --------------------------------------- */

/* --------------------------------------- */

#if !defined(DONT_USE_FFALLOC)
#define FFALLOC ffalloc->
#else
#define FFALLOC
#endif

#define TASK_MALLOC(size)          (FFALLOC malloc(size))
#define TASK_FREE(ptr)             (FFALLOC free(ptr))
#define TASK_REALLOC(ptr,newsize)  (FFALLOC realloc(ptr,newsize))


/* --------------------------------------- */

template<typename TaskT, typename compare_t = CompareTask_LIFO>
class TaskFScheduler: public ff_node_t<TaskT> {
private:
    typedef std::priority_queue<hash_task_t*, std::vector<hash_task_t*>, compare_t> priority_queue_t;    
protected:
    enum { UNBLOCK_SIZE=16, TASK_PER_WORKER=128};

    // FIX: needed to deallocate address hash !!

    inline void task_hash_delete(hash_task_t *t) {
        icl_hash_delete(task_set,&t->id,NULL,NULL);
        TASK_FREE(t->unblock_task_ids); TASK_FREE(t);
    }
           
    inline void setDep(hash_task_t *t, unsigned long dep) {
        if(t->unblock_numb == t->unblock_act_numb) {
            t->unblock_act_numb+=UNBLOCK_SIZE;
            t->unblock_task_ids=(unsigned long *)TASK_REALLOC(t->unblock_task_ids,t->unblock_act_numb*sizeof(unsigned long));
        }
        t->unblock_task_ids[t->unblock_numb]= dep;
        t->unblock_numb++;
    }

    inline hash_task_t* createTask(unsigned long id, task_status_t status, base_f_t *wtask) {
        hash_task_t *t=(hash_task_t*)TASK_MALLOC(sizeof(hash_task_t));
        
        t->id=id;  t->status=status;  t->remaining_dep=0;
        t->unblock_numb=0; t->wtask=wtask; t->is_dummy=false;
        t->unblock_task_ids=(unsigned long *)TASK_MALLOC(UNBLOCK_SIZE*sizeof(unsigned long));
        t->unblock_act_numb=UNBLOCK_SIZE;  t->num_out=0;
        return t;        
    }

    inline hash_task_t *insertTask(task_f_t *const msg,
                                   hash_task_t *waittask=nullptr) {
        unsigned long act_id=task_id++;
        hash_task_t *act_task=createTask(act_id,NOT_READY,msg->wtask);	    
        icl_hash_insert(task_set, &act_task->id, act_task); 
        
        for (auto p: msg->P) {
            auto d    = p.tag;
            auto dir  = p.dir;
            if(dir==INPUT) {
                //hash_task_t * t=(hash_task_t *)icl_hash_find(address_set,(void*)d);		
                unsigned long t_id=(unsigned long)icl_hash_find(address_set,(void *)d);
                hash_task_t * t=NULL;
                if(t_id)    //t_id==0 if the hash table does not contains info for d
                    t=(hash_task_t *)icl_hash_find(task_set,&t_id);


	    
                if(t==NULL) { // no writer for this tag
                    hash_task_t *dummy=createTask(task_id,DONE,NULL);
                    dummy->is_dummy=true;
                    // the dummy task uses current data
                    icl_hash_insert(address_set,(void*)d,(void*)(dummy->id));
                    // the dummy task unblocks the current data
                    dummy->unblock_task_ids[dummy->unblock_numb]=act_id;
                    dummy->unblock_numb++;
                    dummy->num_out++;
                    icl_hash_insert(task_set,&dummy->id,dummy);
                    task_id++;
                } else {
                    if(t->unblock_numb == t->unblock_act_numb) {
                        t->unblock_act_numb+=UNBLOCK_SIZE;
                        t->unblock_task_ids=(unsigned long *)TASK_REALLOC(t->unblock_task_ids,t->unblock_act_numb*sizeof(unsigned long));
                    }
                    t->unblock_task_ids[t->unblock_numb]=act_id;
                    t->unblock_numb++;
                    if(t->status!=DONE) act_task->remaining_dep++;
                }
            } else
                if (dir==OUTPUT) {
                    hash_task_t * t=NULL;
                    unsigned long t_id = (unsigned long)icl_hash_find(address_set,(void*)d);

                    if (t_id) t=(hash_task_t *)icl_hash_find(task_set,&t_id);

                    // the task has been already written 
                    if(t != NULL) {
                        if (t->unblock_numb>0) {
                            // for each unblocked task, checks if that task unblock also act_task (WAR dependency)
                            for(long ii=0;ii<t->unblock_numb;ii++) {							
                                hash_task_t* t2=(hash_task_t*)icl_hash_find(task_set,&t->unblock_task_ids[ii]);
                                if(t2!=NULL && t2!=act_task && t2->status!=DONE) {
                                    if(t2->unblock_numb == t2->unblock_act_numb) {
                                        t2->unblock_act_numb+=UNBLOCK_SIZE;
                                        t2->unblock_task_ids=(unsigned long *)TASK_REALLOC(t2->unblock_task_ids,t2->unblock_act_numb*sizeof(unsigned long));
                                    }
                                    t2->unblock_task_ids[t2->unblock_numb]=act_id;
                                    t2->unblock_numb++;
                                    act_task->remaining_dep++;
                                }
                            }
                        } else { 
                            if(t->status!=DONE) {
                                t->unblock_task_ids[t->unblock_numb]=act_id;
                                t->unblock_numb++;
                                act_task->remaining_dep++;
                            }
                        }
                        t->num_out--;
                        if (t->status==DONE && t->num_out==0)
                            task_hash_delete(t);
                    }
                    if(t_id) icl_hash_delete(address_set,(void *)d,NULL,NULL);
                    icl_hash_insert(address_set, (void*)d,(void *)(act_task->id));
                    act_task->num_out++;
                }
        }
        if ((act_task->remaining_dep==0) && !waittask) {
            act_task->status=READY;
            readytasks++;
            ready_queues[m].push(act_task);
            m = (m + 1) % runningworkers;
        }
        return act_task;
    }

    // try to send at least one task to workers
    inline bool schedule_task(const unsigned long th) {
        bool ret = false;
        for(size_t i=0;(readytasks>0)&&(i<runningworkers);i++){
            if(nscheduled[i]<=th){
                if(ready_queues[i].size()>0){
                    hash_task_t *top = ready_queues[i].top();
                    assert(top->status == READY);
                    //printf("schedule task %ld to %ld\n", top->id, i);
                    lb->ff_send_out_to(top,i);
                    //taskscheduled[i].push_back(top);// remember the task just sent
                    ready_queues[i].pop();
                    ++nscheduled[i], --readytasks;
                    ret = true;
                } else{
                    bool found = false;
                    for(size_t j=0; !found && (j<runningworkers);j++){
                        if(ready_queues[mmax].size()>0){
                            hash_task_t *top = ready_queues[mmax].top();
                            assert(top->status == READY);
                            //printf("schedule task %ld to %ld\n", top->id, i);
                            lb->ff_send_out_to(top,i);
                            //taskscheduled[i].push_back(top); // remember the task just sent
                            ready_queues[mmax].pop();
                            ++nscheduled[i], --readytasks;
                            ret = found = true;
                        }
                        mmax = (mmax + 1) % runningworkers;
                    }
                }
            }
        }
        return ret;
    }
    
    inline void handleTask(hash_task_t *t, int workerid) {
         for(long i=0;i<t->unblock_numb;i++) {
             hash_task_t *tmp=(hash_task_t*)icl_hash_find(task_set,&t->unblock_task_ids[i]);
             assert(tmp);
             tmp->remaining_dep--;
             if(tmp->remaining_dep==0) {
                 if (tmp->status == PENDING_DONE) {
                     handleTask(tmp,workerid);
                 } else {
                     if (tmp->status == NOT_READY) {
                         tmp->status=READY;
                         ++readytasks;
                         ready_queues[workerid].push(tmp);
                     }
                 }
             }
         }        
         schedule_task(0); 
         
         t->status=DONE;
         if(!t->num_out) {
             delete t->wtask;
             task_hash_delete(t);
         }
    }
    inline void handleCompletedTask(hash_task_t *t, int workerid) {
        --nscheduled[workerid];
        //assert(taskscheduled[workerid].front() != nullptr);
        //taskscheduled[workerid].pop_front(); 
        handleTask(t,workerid);
    }

    inline bool fromInput() { return (lb->get_channel_id() == -1);	}
    
public:       
    TaskFScheduler(ff_loadbalancer* lb, const int maxnw):
        lb(lb),ffalloc(NULL),runningworkers(0),address_set(NULL),task_set(NULL),
        ready_queues(maxnw),nscheduled(maxnw) /* ,taskscheduled(maxnw) */ {
#if !defined(DONT_USE_FFALLOC)
        ffalloc=new ff_allocator;
        assert(ffalloc);
        int nslabs[N_SLABBUFFER]={0,2048,512,64,0,0,0,0,0 };
        if (ffalloc->init(nslabs)<0) {
            error("FATAL ERROR: allocator init failed\n");
            abort();
        }
#endif            
        
        LOWER_TH = (std::max)(1024, TASK_PER_WORKER*maxnw); //FIX: check for deadlock problems !!!
        UPPER_TH = LOWER_TH+TASK_PER_WORKER;
    }
    virtual ~TaskFScheduler() {
#if !defined(DONT_USE_FFALLOC)
        if (ffalloc) delete ffalloc;
#endif
        if (task_set)    icl_hash_destroy(task_set,NULL,NULL);
        if (address_set) icl_hash_destroy(address_set,NULL,NULL);
    }
    
    virtual int svc_init() {
        runningworkers = lb->getnworkers();
        mmax = readytasks = m = 0;        
        task_id = 1;
        if (task_set) icl_hash_destroy(task_set,NULL,NULL);
        if (task_set) icl_hash_destroy(address_set,NULL,NULL);
        task_set    = icl_hash_create( UPPER_TH*8, ulong_hash_function, ulong_key_compare ); 
        address_set = icl_hash_create( 0x01<<12, address_hash_function, address_key_compare);
        const size_t maxnw = nscheduled.size();
        for(size_t i=0; i<maxnw;++i) { 
            nscheduled[i]    = 0;
            ready_queues[i]  = priority_queue_t();
        }
        return 0;
    }

protected:
    ff_loadbalancer               *lb;
    ff_allocator                  *ffalloc;
    size_t                         task_id, runningworkers;
    icl_hash_t                    *address_set, *task_set;
    int                            mmax, readytasks,m;
    int                            LOWER_TH, UPPER_TH;
    std::vector<priority_queue_t>  ready_queues;
    std::vector<size_t>            nscheduled;
    //std::vector<std::deque<hash_task_t*> > taskscheduled;
};

/* ------------------------------------------------------ */

class TaskFKeyOnce {
private:
    pthread_key_t key;
protected:
    TaskFKeyOnce() {
        if (pthread_key_create( &key, NULL)!=0)  {                                 
            error("TaskFKeyOnce FATAL ERROR: pthread_key_create fails\n");
            abort();
        }
    }
    ~TaskFKeyOnce() { pthread_key_delete(key); }
public:
    static inline pthread_key_t getTaskFKey() {
        static TaskFKeyOnce TFKey;
        return TFKey.key;
    }
};


  
} // namespace


#endif // FF_TASK_INTERNALS_HPP
