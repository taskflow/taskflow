/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

#ifndef FF_DAC_MDF_HPP
#define FF_DAC_MDF_HPP
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
 ****************************************************************************
 *
 *  Author:   
 *          Tiziano De Matteis <dematteis@di.unipi.it>
 *          Massimo Torquati <torquati@di.unipi.it>       
 *
 */


#include <functional>
#include <tuple>
#include <vector>
#include <ff/ff.hpp>

#define DONT_USE_FFALLOC 1
#include <ff/task_internals.hpp>

namespace ff{


    /**
     * \class ff_DC
     * \ingroup high_level_patterns
     * 
     * \brief Macro Data Flow executor
     */

template<typename IN_t, 
         typename OUT_t = IN_t, 
         typename OperandType = IN_t, 
         typename ResultType = OUT_t, typename compare_t=CompareTask_Par>
class ff_DC:public ff_node_t<IN_t, OUT_t> {
    using divide_f_t   = std::function<void (const OperandType&, std::vector<OperandType> &)>;
    using combine_f_t  = std::function<void(std::vector<ResultType>&,ResultType&)>;
    using seq_f_t      = std::function<void(const OperandType&, ResultType&)>;
    using cond_f_t     = std::function<bool(const OperandType&)>;
public:
    enum {DEFAULT_OUTSTANDING_TASKS = 2048};    

protected:    
    bool prepared = false;

    int prepare() {
        if (!prepared) {
            auto r=-1;
            farm->freeze();
            if (farm->run(true) != -1) {
                r = farm->wait_freezing();
            }
            if (r<0) {
                error("ff_DC: preparing DAC\n"); 
                return -1;
            }
            prepared = true;
        }
        return 0;
    }


    virtual OperandType *setTaskIn(IN_t *in) {
        sched->setTaskIn((OperandType*)in);
        return (OperandType*)in;
    }
    virtual ResultType *setTaskOut(IN_t *in) {
        sched->setTaskOut((ResultType*)in);
        return (ResultType*)in;
    }
    virtual OUT_t *releaseTask(ResultType *in) {
        return (OUT_t*)in;
    }


    OUT_t *svc(IN_t *in) {
        OperandType *op  = setTaskIn(in); 
        sched->setTaskIn(op);
        ResultType  *res = setTaskOut(in);
        sched->setTaskOut(res);
        farm->run_then_freeze();
        farm->wait_freezing();
        return releaseTask(res);
    }

    // ---- data types used between the Scheduler and the Workers 

    // generic_task_t: encapsulate both task_f_t and hash_task_t
    // plus a boolean to distinguish between them
    struct generic_task_t{
        union{
            struct hash_task_t *ht;
            struct task_f_t     tf;
        };
        bool is_new_task;   //true if you have to read tf field
    };
    template<typename F_t, typename... Param>
    struct ff_dac_f_t: public base_f_t {
        ff_dac_f_t(const F_t F, Param&... a):F(F) { args = std::make_tuple(a...);}	
        
        inline void call(void *w) {
            //Before invoking the function, pass to it the context as first parameter (i.e. a reference to the current worker)
            std::get<0>(args)=w;
            ffapply(F, args);
        }
        
        F_t F;
        std::tuple<Param...> args;	
    };
        
    /**
     * @brief CombineFunction it represents the task for the Combine part of a Divide and Conquer algorithm
     * @param w worker context (suppplied by the caller in dac_task_internals.hpp)
     * @param combine_fn the combine function
     * @param ress partial result that have to be combined
     * @param res combine result
     * @param subops suboperands for this recursion step. They are deleted after the Combine
     */    
    static void CombineFunction(void * /*w*/,const std::function<void(std::vector<ResultType>&,ResultType&)>& combine_fn, std::vector<ResultType>* ress, ResultType* res, std::vector<OperandType> *subops)
    {
        combine_fn(*ress,*res);
        

        //clean up memory, sub operand and partial results are no more used
        //for(size_t i=0;i<subops->size();i++)
        //  {
        //      delete (*ress)[i];
                //delete subops[i];
        //  }
        
        delete ress;
        delete subops;
    }
    
    /**
     * @brief DACFunction it represents the generic (recursive) task of a Divide and Conquer algorithm. Threfore the operand is 'divided', then it is called
     *      the DACFunction for each suboperands and finally the Combine produces the final output. It supports unknown branch factor (i.e. unknown number
     *      of sub operands). For each recursive call, if n is the branch factor, are created:
     *      - n-1 DACFunction tasks (each operating on a different suboperand)
     *      - a recursive call on the last suboperand
     *      - a CombineFunction task to produce the result
     * @param w worker contest
     * @param _divide_fn    divide function passed by the user
     * @param _combine_fn   combine function passed by the user
     * @param _seq_fn       sequential (base case) function passed by the user
     * @param _condition_fn condition (for base case)
     * @param op operand
     * @param ret pointer to memory area in which store the result
     */
    static void DACFunction(void *w,const std::function<void (const OperandType&, std::vector<OperandType> &)>& _divide_fn, const std::function<void(std::vector<ResultType>&,ResultType&)>& _combine_fn,
                            const std::function<void(const OperandType&,ResultType&)>& _seq_fn,const std::function<bool(const OperandType&)>& _condition_fn,
                            OperandType* op, ResultType *ret )
	{
		if(!_condition_fn(*op))  //this is not the base case
            {
                //divide
                //std::vector<OperandType*> ops=_divide_fn(*op);
     
				std::vector<OperandType> *ops=new std::vector<OperandType>();
                _divide_fn(*op, *ops);
                int branch_factor=ops->size();
                
                //create the space for the partial results
                std::vector<ResultType> *ress=new std::vector<ResultType>(branch_factor);
                //for(int i=0;i<branch_factor;i++) ress->push_back(new ResultType());
                
                //conquer: create branch_factor-1 tasks and recur on the last sub-operand
                std::vector<param_info> params;
                
                for(int i=0;i<branch_factor-1;i++)
                    {
                        params.clear();
                        const param_info r={(uintptr_t)&(*ress)[i],OUTPUT};
                        params.push_back(r);
                        ((DACWorker*)w)->AddTaskWorker(params,
                                                       ff_DC<IN_t, OUT_t, OperandType, ResultType, compare_t>::DACFunction, 
                                                       w,_divide_fn,_combine_fn,_seq_fn,_condition_fn, &(*ops)[i], &(*ress)[i]);
                    }
                
                
                DACFunction(w,_divide_fn,_combine_fn,_seq_fn,_condition_fn, &(*ops)[branch_factor-1], &(*ress)[branch_factor-1]);
                
                //combine task
                params.clear();
                for(int i=0;i<branch_factor;i++)
                    {
                        const param_info c={(uintptr_t) &(*ress)[i],INPUT};
                        params.push_back(c);
                    }
                
                const param_info cret={(uintptr_t) (ret),OUTPUT};
                params.push_back(cret);
                
                ((DACWorker*)w)->AddTaskWorker(params, 
                                               ff_DC<IN_t, OUT_t, OperandType, ResultType, compare_t>::CombineFunction,
                                               w, _combine_fn,ress, ret, ops);

            }
        else{
            //base case
            _seq_fn(*op,*ret);
        }
    }
    
    
    /* --------------  worker ------------------------------- */
    struct DACWorker: ff_node_t<hash_task_t,generic_task_t> {

		//DEBUG
//		DACWorker():_task_exe(0), _task_created(0),_times(0){}

        inline generic_task_t *svc(hash_task_t *task) {

//			//DEBUG
//			_task_exe++;
//			gettimeofday(&_time, NULL);
//			long start_t = (_time.tv_sec)*1000000L + _time.tv_usec;
//			//END DEBUG
			task->wtask->call(this);

//			//DEBUG
//			gettimeofday(&_time, NULL);
//			_times += ((_time.tv_sec)*1000000L + _time.tv_usec-start_t);
//			//END DEBUG
            //ritorna un generic task che incapsula l'hash_task_t

            generic_task_t * gt = (generic_task_t*)TASK_MALLOC(sizeof(generic_task_t));
            gt->is_new_task = false;
            gt->ht=task;
            return gt;
        }
        
        template<typename F_t, typename... Param>
        inline void AddTaskWorker(std::vector<param_info> &P, const F_t F, Param... args) {
                        
            ff_dac_f_t<F_t, Param...> *wtask = new ff_dac_f_t<F_t,Param...>(F, args...);
            generic_task_t * gt = (generic_task_t*)TASK_MALLOC(sizeof(generic_task_t));
            new (gt) task_f_t();    // calling the constructor of the internal data structure
            gt->is_new_task = true;
            gt->tf.P     = P;
            gt->tf.wtask = wtask;
            this->ff_send_out(gt);
			//DEBUG
//			_task_created++;
        }


//		//DEBUG
//		void svc_end()
//		{
//			printf("%d\tExecuted\t%d\tCreated\t%d\tTime(us)\t%Ld\n",this->get_my_id(),_task_exe,_task_created,_times);
//		}

//		//DEBUG
//		int _task_exe;
//		int _task_created;
//		long _times;
//		struct timeval _time;

    };

    /* --------------  scheduler ----------------------------- */
    class Scheduler: public TaskFScheduler<generic_task_t, compare_t> {
        using baseSched = TaskFScheduler<generic_task_t, compare_t>;
        using baseSched::lb;
        using baseSched::schedule_task;       
        using baseSched::handleCompletedTask;
        enum { RELAX_MIN_BACKOFF=1, RELAX_MAX_BACKOFF=32};

    protected:
        inline hash_task_t *insertTask(task_f_t *const msg,
                                       hash_task_t *waittask=nullptr) {
            unsigned long act_id=baseSched::task_id++;
            hash_task_t *act_task=baseSched::createTask(act_id,NOT_READY,msg->wtask);	    
            icl_hash_insert(baseSched::task_set, &act_task->id, act_task); 
            for (auto p: msg->P) {
                auto d    = p.tag;
                auto dir  = p.dir;
                if(dir==INPUT) {
                    // hash_task_t * t=(hash_task_t *)icl_hash_find(address_set,(void*)d);
                    
                    //address_set is an hash table data->task_id. The task_id is memorized in plain format (i.e. it is not a pointer to a memory area that contains the id)
                    unsigned long t_id=(unsigned long)icl_hash_find(baseSched::address_set,(void *)d);
                    hash_task_t * t=NULL;
                    if(t_id)    //t_id==0 if the hash table does not contains info for d
                        t=(hash_task_t *)icl_hash_find(baseSched::task_set,&t_id);
                    
                    //we have to check that the task exists
                    if(t!=NULL) {
                        if(t->unblock_numb == t->unblock_act_numb) {
                            t->unblock_act_numb+=baseSched::UNBLOCK_SIZE;
                            t->unblock_task_ids=(unsigned long *)TASK_REALLOC(t->unblock_task_ids,t->unblock_act_numb*sizeof(unsigned long));
                        }
                        t->unblock_task_ids[t->unblock_numb]=act_id;
                        t->unblock_numb++;
                        if(t->status!=DONE)
                            act_task->remaining_dep++;
                    }
                } else
                    if (dir==OUTPUT) {
                        hash_task_t * t=NULL;
                        unsigned long t_id=(unsigned long)icl_hash_find(baseSched::address_set,(void *)d);
                        if(t_id)
                            t=(hash_task_t *)icl_hash_find(baseSched::task_set,&t_id);
                        
                        if(t != NULL) { //someone write that data
                            
                            if (t->unblock_numb>0) {
                                
                                //HERE THERE IS THE MAIN DIFFERENCE WRT CLASSICAL FF_MDF
                                
                            //Before: for each unblocked task, checks if that task unblock also act_task (-> WAR dependency)
                            //Now: due to the order in which tasks are generated if an existing task uses the same parameter
                            //  in output, it has to be unblocked by act_task (for example it is a combine at level i that is
                            //  unblocked by a combine spawned at level i+1; the tasks for combine at level i+1 arrives later wrt to
                            //  the one at level i, that, in turns, depends on the DAC task of the same level)


                                std::vector<long> todelete;

                            for(long ii=0;ii<t->unblock_numb;ii++) {							
                                hash_task_t* t2=(hash_task_t*)icl_hash_find(baseSched::task_set,&t->unblock_task_ids[ii]);


                                //t2 must not be a READY task (false positive dependence)
                                if(t2!=NULL && t2!=act_task && t2->status!=DONE && t2->status!=READY) {

                                    //act_task will unblock t2
                                    if(act_task->unblock_numb == act_task->unblock_act_numb) {
                                        act_task->unblock_act_numb+=baseSched::UNBLOCK_SIZE;
                                        act_task->unblock_task_ids=(unsigned long *)TASK_REALLOC(act_task->unblock_task_ids,act_task->unblock_act_numb*sizeof(unsigned long));
                                    }
                                    act_task->unblock_task_ids[act_task->unblock_numb]=t2->id;
                                    act_task->unblock_numb++;


                                    //in every case t2 is still the last task to write on d

                                    //t will not unblock t2. It is act_task that will do it
                                    t->unblock_task_ids[ii]=0;
                                    //the number of remaining dependencies of t2 is the same
                                    todelete.push_back(ii);
                                }
                            }

                            for(size_t i=0;i<todelete.size(); ++i) {
                                t->unblock_task_ids[todelete[i]] = t->unblock_task_ids[t->unblock_numb];
                                t->unblock_numb--;
                            }


                        } else {

                            if(t->status!=DONE) {
                                t->unblock_task_ids[t->unblock_numb]=act_id;
                                t->unblock_numb++;
                                act_task->remaining_dep++;
                            }
                        }

                    }
                    if(t_id)
                        icl_hash_delete(baseSched::address_set,(void *)d,NULL,NULL);
                    //icl_hash_update_insert(address_set, (void*)d, act_task);
                    icl_hash_insert(baseSched::address_set, (void*)d,(void *)(act_task->id));

                }
        }

        if ((act_task->remaining_dep==0) && !waittask) {
            act_task->status=READY;
            baseSched::readytasks++;
            baseSched::ready_queues[m].push(act_task);
            m = (m + 1) % baseSched::runningworkers;
        }
        return act_task;
    }

    public:
        Scheduler(ff_loadbalancer* lb, const int maxnw, void (*schedRelaxF)(unsigned long), const std::function<void (const OperandType&, std::vector<OperandType> &)>& divide_fn,
                  const std::function<void(std::vector<ResultType>&,ResultType&)>& combine_fn,
                  const std::function<void(const OperandType&,ResultType&)>& seq_fn, const std::function<bool(const OperandType&)>& cond_fn,
                  const OperandType* op,ResultType* res):
            TaskFScheduler<generic_task_t,compare_t>(lb,maxnw),
            task_numb(0),task_completed(0),bk_count(0),schedRelaxF(schedRelaxF),
            _divide_fn(divide_fn), _combine_fn(combine_fn), _seq_fn(seq_fn), _condition_fn(cond_fn),_op(op),_res(res)
            {
            }
        virtual ~Scheduler() {}

        int svc_init() {
            if (baseSched::svc_init()<0) return -1;

            task_numb = task_completed = 0, bk_count = 0;
            m=0;

            return 0;
        }

        void setTaskIn(const OperandType *op)   { _op  = op; }
        void setTaskOut(ResultType *res)        { _res = res; }


        generic_task_t* svc(generic_task_t* t) {

            // TODO: EVITARE LA BARRIERA 
            if (!_op) return baseSched::EOS; 

            if(!t)  //first call: generate initial tasks
            {
                

                //this is very similar to DACFunction, apart from the fact that we do not recur
				if(!_condition_fn(*_op))
                {
                    //std::vector<OperandType *> ops=_divide_fn(*_op);
					std::vector<OperandType> *ops=new std::vector<OperandType>();
                    _divide_fn(*_op, *ops);


                    //create the tasks according to the branch fractor (i.e. the number of suboperands returned by the divide)
                    int branch_factor=ops->size();
                    std::vector<ResultType> *ress=new std::vector<ResultType>(branch_factor);
                    //for(int i=0;i<branch_factor;i++) ress->push_back(new ResultType());

                    std::vector<param_info> params;

                    for(int i=0;i<branch_factor;i++)
                    {
                        params.clear();
                        const param_info r={(uintptr_t)&(*ress)[i],OUTPUT};
                        params.push_back(r);
                        this->AddTaskScheduler(params,
                                               ff_DC<IN_t, OUT_t, OperandType, ResultType, compare_t>::DACFunction, 
                                               (void*)0,_divide_fn,_combine_fn,_seq_fn,_condition_fn, &(*ops)[i], &(*ress)[i]);
                    }


                    params.clear();
                    for(int i=0;i<branch_factor;i++)
                    {
                        const param_info c={(uintptr_t) &(*ress)[i],INPUT};
                        params.push_back(c);
                    }

                    const param_info cret={(uintptr_t) (_res),OUTPUT};
                    params.push_back(cret);

                    this->AddTaskScheduler(params,CombineFunction,(void*)0, _combine_fn,ress, _res,ops);

                    return baseSched::GO_ON;
                }
                else{
                    //base case
                    _seq_fn(*_op,*(_res));
                    return baseSched::EOS;
                }
            }
            else
                {
                bk_count = 0;
                if(t->is_new_task)
                {
                    //new task generated by a worker
                    ++task_numb;
                    insertTask(&(t->tf));
                    schedule_task(0);
                    (&(t->tf))->~task_f_t();
                    TASK_FREE(t);
                    return baseSched::GO_ON;
                }
                else
                {
                    //completed task
                    hash_task_t * task = (hash_task_t *)t->ht;
                    ++task_completed;
                    handleCompletedTask(task,lb->get_channel_id());

                    schedule_task(1); // try once more

                    TASK_FREE(t);
                    if(task_numb==task_completed)
                    {
                        //std::cout << "Created tasks: "<<task_numb<<std::endl;
                        return baseSched::EOS;
                    }
                    return baseSched::GO_ON;
                }
            }
        }

        void eosnotify(ssize_t /*id*/=-1) { lb->broadcast_task(FF_EOS); }
        int wait_freezing()           { return lb->wait_lb_freezing(); }

    private:
        size_t                         task_numb, task_completed, bk_count,m;
        void                         (*schedRelaxF)(unsigned long);
        //function pointers
        const std::function<void(const OperandType&, std::vector<OperandType>&)>& _divide_fn;
        const std::function<void(std::vector<ResultType>&,ResultType&)>&          _combine_fn;
        const std::function<void(const OperandType& ,  ResultType&)>&             _seq_fn;
        const std::function<bool(const OperandType&)>&                            _condition_fn;

        const OperandType* _op;                                            //primo operando
        ResultType* _res;

        template<typename F_t, typename... Param>
        inline void AddTaskScheduler(std::vector<param_info> &P, const F_t F, Param... args)
        {
            //create task
            ff_dac_f_t<F_t, Param...> *wtask = new ff_dac_f_t<F_t,Param...>(F, args...);
            task_f_t *task = new task_f_t();
            task->P=P;
            task->wtask=wtask;
            //insert and schedule
            //task_f_t *const task=wtask;
            ++task_numb;
            insertTask(task);
            schedule_task(0);
            delete task;

        }
        inline void schedule()
        {
            schedule_task(0);
        }

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
    ff_DC(const divide_f_t& divide_fn, const combine_f_t& combine_fn,
          const seq_f_t& seq_fn, const cond_f_t& cond_fn, 
          const OperandType& op, ResultType& res,
          int numw, size_t /*outstandingTasks*/=DEFAULT_OUTSTANDING_TASKS,
          int maxnw=ff_numCores(), void (*schedRelaxF)(unsigned long)=NULL):
        _divide_fn(divide_fn), _combine_fn(combine_fn), _seq_fn(seq_fn), _condition_fn(cond_fn) {
        
        farm = new ff_farm(false,640*maxnw,1024*maxnw,true,maxnw,true);
        std::vector<ff_node *> w;
        // NOTE: Worker objects are going to be destroyed by the farm destructor
        for(int i=0;i<numw;++i) w.push_back(new DACWorker);
        farm->add_workers(w);
        
        farm->add_emitter(sched = new Scheduler(farm->getlb(), numw, schedRelaxF,_divide_fn,_combine_fn,_seq_fn,_condition_fn,&op,&res));
        farm->wrap_around();        
    }
    ff_DC(const std::function<void (const OperandType&, std::vector<OperandType> &)>& divide_fn, const std::function<void(std::vector<ResultType>&,ResultType&)>& combine_fn,
        const std::function<void(const OperandType&, ResultType&)>& seq_fn, const std::function<bool(const OperandType&)>& cond_fn, int numw, 
               size_t outstandingTasks=DEFAULT_OUTSTANDING_TASKS,int maxnw=ff_numCores(), void (*schedRelaxF)(unsigned long)=NULL):
                               _divide_fn(divide_fn), _combine_fn(combine_fn), _seq_fn(seq_fn), _condition_fn(cond_fn)
    {

        farm = new ff_farm(false,640*maxnw,1024*maxnw,true,maxnw,true);
        std::vector<ff_node *> w;
        // NOTE: Worker objects are going to be destroyed by the farm destructor
        for(int i=0;i<numw;++i) w.push_back(new DACWorker);
        farm->add_workers(w);

        farm->add_emitter(sched = new Scheduler(farm->getlb(), numw, schedRelaxF,_divide_fn,_combine_fn,_seq_fn,_condition_fn, nullptr, nullptr));
        farm->wrap_around();
    }

    virtual ~ff_DC() {
        if (sched) delete sched;
        if (farm)  delete farm;
        //TODO: andrebbe cancellato l'allocatore
    }

    void setNumWorkers(ssize_t nw) { 
        if (nw > ff_numCores())
            error("ff_DC: setNumWorkers: too much workers, setting num worker to %d\n", 
                  ff_numCores());         
        farmworkers=(std::min)(ff_numCores(),nw); 
    }	
    
    inline int run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;
        return ff_node::run(true);
    }


    virtual inline int run_and_wait_end() {
        //ff_Farm::thaw(true,farmworkers);
        //if (farm->wait_freezing() <0) return -1;
        return farm->run_and_wait_end();
    }

    virtual inline int run_then_freeze(ssize_t nw=-1) {
        //if (nw>0) setNumWorkers(nw);
       return farm->run_then_freeze(nw);
    }

    inline int  wait_freezing() { return farm->wait_freezing();}

    inline int wait() { 
        int r=ff_node::wait();
        if (r!=-1) r = farm->wait();
        return r;
    }

    double ffTime() { return ff_node::ffTime(); }
  //  double ffwTime() { return ff_pipeline::ffwTime(); }
    
protected:
    //function pointers
    const divide_f_t&   _divide_fn;
    const combine_f_t&  _combine_fn;
    const seq_f_t&      _seq_fn;
    const cond_f_t&     _condition_fn;

    int farmworkers;   // n. of workers in the farm
    ff_farm   *farm;   
    Scheduler *sched;  // farm's scheduler
};




} // namespace

#endif /* FF_DAC_MDF_HPP */
