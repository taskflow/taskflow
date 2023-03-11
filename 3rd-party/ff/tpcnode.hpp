/* -*- Mode: C++; tab-width: 2; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file tpcnode.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow Thread Pool Composer (TPC) interface node
 *
 * This class bridges multicore with FPGAs using the TPC library.
 *
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


/*  Author: Massimo Torquati torquati@di.unipi.it
 *  - September 2015 first version
 *
 */

#ifndef FF_TPCNODE_HPP
#define FF_TPCNODE_HPP

#include <vector>
#include <algorithm>
#include <ff/node.hpp>
#include <ff/bitflags.hpp>

#include <ff/tpc/tpcEnvironment.hpp>
#include <ff/tpcallocator.hpp>
#include <ff/tpc/tpc_api.h>
using namespace rpr;
using namespace tpc;

namespace ff{


namespace internal {
    /**
     *  iovector-like data structure plus memory flags
     */
    struct Arg_t {
        Arg_t():
            ptr(NULL),size(0),
            byvalue(false),
            copy(false),reuse(false),release(false) {}
        Arg_t(void *ptr, size_t size):
            ptr(ptr),size(size),
            byvalue(true),
            copy(true),reuse(false),release(false) {}
        Arg_t(void *ptr, size_t size, bool copy, bool reuse, bool release):
            ptr(ptr),size(size),
            byvalue(false),
            copy(copy),reuse(reuse),release(release) {}

        void        *ptr;
        size_t       size;
        const bool   byvalue;
        const bool   copy,reuse,release;
    };
} // namespace internal



/**
 * Interface of the task to be executed by a TPC-based node.
 */
template<typename TaskT_in, typename TaskT_out=TaskT_in>
class baseTPCTask {
    template <typename IN_t, typename TPC_t, typename OUT_t> 
    friend class ff_tpcNode_t;
public:
    
    /** 
     *  Default constructor.
     */
    baseTPCTask():tpc_kernel_id(-1), retvar{nullptr,0} {}

    /** 
     *  Destructor.
     */
    virtual ~baseTPCTask() { }
    
    /**
     * Provides to the run-time the task to be computed. 
     * This method must be overridden.
     *
     * @param t input task
     * 
     */ 
    virtual void setTask(TaskT_in *t) = 0;
    
    /**
     * Releases the input task just computed.
     * This method is called at the of the task computation.
     * It may be used to  perform per-task host memory cleanup 
     * (i.e. releasing the host memory previously allocated in the setTask 
     * function) or to execute a  post-elaboration phase
     *
     * @param t input task
     * @return the output task
     */
    virtual TaskT_out *releaseTask(TaskT_in *t) { return t; }

    /**
     * Sets the host-pointer to the input parameter.
     * The order of calls determin the order of parametes.
     * 
     * @param inPtr the host-pointer
     * @param size the number of elements in the input array (bytesize=size*sizeof(ptrT))
     * @param copy if CopyFlags::COPY is set the data will be copied into the device
     * @param reuse if BitFlags::REUSE is set then the run-time looks for a previously 
     * allocated device handle associated with the inPtr host pointer
     * @param release if BitFlags::RELEASE is set the device memory will be released 
     * at the end of the kernel execution
     */
    template <typename ptrT>
    void setInPtr(const ptrT* inPtr, size_t size, 
                  const CopyFlags    copy   =CopyFlags::COPY, 
                  const ReuseFlags   reuse  =ReuseFlags::DONTREUSE, 
                  const ReleaseFlags release=ReleaseFlags::DONTRELEASE)  { 
        internal::Arg_t arg(const_cast<ptrT*>(inPtr),size*sizeof(ptrT),
                            copy==CopyFlags::COPY,
                            reuse==ReuseFlags::REUSE,
                            release==ReleaseFlags::RELEASE);
        tpcInput.push_back(arg);
    }

    /**
     * Sets the host-pointer to the input parameter.
     * The order of calls determin the order of parametes.
     */
    template <typename ptrT>
    void setInPtr(const ptrT* inPtr, size_t size, const MemoryFlags &flags) {
        internal::Arg_t arg(const_cast<ptrT*>(inPtr),size*sizeof(ptrT),
                            flags.copy==CopyFlags::COPY,
                            flags.reuse==ReuseFlags::REUSE,
                            flags.release==ReleaseFlags::RELEASE);
        tpcInput.push_back(arg);
    }


    /**
     * Sets the host-pointer to the input parameter that is passed by-value.
     *
     * @param inPtr the host-pointer
     */
    template <typename ptrT>
    void setInVal(const ptrT* inPtr) {
        internal::Arg_t arg(const_cast<ptrT*>(inPtr), sizeof(ptrT));
        tpcInput.push_back(arg);
    }

    /**
     * Sets the host-pointer to the output parameter
     *
     * @see setInPtr()
     */
    template <typename ptrT>
    void setOutPtr(const ptrT* _outPtr, size_t size, 
                   const CopyFlags copyback =CopyFlags::COPY, 
                   const ReuseFlags reuse    =ReuseFlags::DONTREUSE, 
                   const ReleaseFlags release  =ReleaseFlags::DONTRELEASE)  { 
        internal::Arg_t arg(const_cast<ptrT*>(_outPtr),size*sizeof(ptrT),
                            copyback==CopyFlags::COPY,
                            reuse==ReuseFlags::REUSE,
                            release==ReleaseFlags::RELEASE);
        tpcOutput.push_back(arg);
    }

    /**
     * Sets the host-pointer to the output parameter
     *
     * @see setInPtr()
     */
    template <typename ptrT>
    void setOutPtr(const ptrT* _outPtr, size_t size, const MemoryFlags &flags) {       
        internal::Arg_t arg(const_cast<ptrT*>(_outPtr),size*sizeof(ptrT),
                            flags.copy==CopyFlags::COPY,
                            flags.reuse==ReuseFlags::REUSE,
                            flags.release==ReleaseFlags::RELEASE);
        tpcOutput.push_back(arg);
    }


    /**
     * Sets the kernel id
     * 
     * @param id function id
     */
    void setFunctionId(const tpc_func_id_t id) { tpc_kernel_id = id; }

    /** 
     * Should be used only if the kernel executed on the
     * device is a function. It allows to get back the return value.
     *
     * @param r host variable where the return value is copied
     */
    template<typename TresT>
    void setReturnVar(TresT *const r) { 
        retvar.first = r; 
        retvar.second= sizeof(TresT); 
    }

protected:

    const std::vector<internal::Arg_t> &getInputArgs()  const { return tpcInput; }
    const std::vector<internal::Arg_t> &getOutputArgs() const { return tpcOutput;}
    void * getReturnVar(size_t &size)     const { 
        size=retvar.second;
        return retvar.first; 
    }
    const tpc_func_id_t  getKernelId()    const { return tpc_kernel_id;}

    void resetTask() {
        tpcInput.resize(0);
        tpcOutput.resize(0);
        retvar.first = nullptr;
        retvar.second= 0;
    }

protected:
    tpc_func_id_t                tpc_kernel_id;
    std::vector<internal::Arg_t> tpcInput;
    std::vector<internal::Arg_t> tpcOutput;
    std::pair<void*,size_t>      retvar;
};



/*!
 *  \class ff_tpcNode
 *  \ingroup buiding_blocks
 *
 *  \brief TPC specialisation of the ff_node class
 *
 *  Implements the node that is serving as TPC device.
 *
 *
 */
template<typename IN_t, typename TPC_t = IN_t, typename OUT_t = IN_t>
class ff_tpcNode_t : public ff_node {
    struct handle_t {
        void         *ptr   = nullptr;
        size_t        size  = 0;
        tpc_handle_t  handle= 0;
    };
    
public:
    typedef IN_t  in_type;
    typedef OUT_t out_type;


    /**
     * Constructor used for stream-based computation.
     * It builds the TPC node for the device.
     * 
     * @param alloc the allocator to use for allocating 
     * device memory. By default it is used an internal
     * private allocator.
     *
     */
    ff_tpcNode_t(ff_tpcallocator *alloc = nullptr):
        tpcId(-1),deviceId(-1),my_own_allocator(false),oneshot(false),oneshotTask(nullptr),allocator(alloc) {

        if (allocator == nullptr) {
            my_own_allocator = true;
            allocator = new ff_tpcallocator;
            assert(allocator);
        }

        // TODO: management of multiple TPC devices
    };  

    /**
     * Constructor used for the "oneshot" mode, i.e. non 
     * stream-based computations.
     *
     * It builds the TPC node for the device.
     *
     */
    ff_tpcNode_t(const IN_t &task, ff_tpcallocator *alloc = nullptr):
        tpcId(-1),deviceId(-1), my_own_allocator(false), oneshot(true),oneshotTask(&task),allocator(alloc) {
        ff_node::skipfirstpop(true);
        Task.setTask(const_cast<TPC_t*>(oneshotTask));

        if (allocator == nullptr) {
            my_own_allocator = true;
            allocator = new ff_tpcallocator;
            assert(allocator);
        }

        // TODO: management of multiple TPC devices
    };  
    /**
     * Destructor
     *
     */
    virtual ~ff_tpcNode_t()  {
        if (my_own_allocator && allocator != nullptr) {
            allocator->releaseAllBuffers(dev_ctx);
            delete allocator;
            allocator = nullptr;
        }
    }
    /**
     * Starts the execution of the device node. 
     * A thread executing the object instance is spawned.
     *
     * @return 0 success, -1 otherwise
     */ 
    virtual int run(bool = false) {
        return ff_node::run();
    }
    /**
     * Waits for termination of the thread associated 
     * to the object.
     *
     * @return 0 success, -1 otherwise
     */    
    virtual int wait() {
        return ff_node::wait();
    }

    virtual int run_and_wait_end() {
        if (nodeInit()<0) return -1;   
        svc(nullptr);                  
        return 0;
    }
    /**
     * Starts the execution of the device node. 
     * A thread executing the object instance is spawned if 
     * no previous thread was associated to the object.
     * The thread will be put to sleep at the end of 
     * the kernel execution.
     * 
     * @return 0 success, -1 otherwise
     */
    virtual int run_then_freeze() {
        if (ff_node::isfrozen()) {
            ff_node::thaw(true);
            return 0;
        }
        return ff_node::freeze_and_run();
    }

    /**
     * Waits for the termination of the kernel
     * execution. The associated thread is put
     * to sleep and not destroyed.
     *
     * @return 0 success, -1 otherwise
     */
    virtual int wait_freezing() {
        return ff_node::wait_freezing();
    }
    
    /**
     *  Allows to set a new task when in "oneshot" mode.
     *
     * @param task  task to be computed
     */
    void setTask(IN_t &task) {
        Task.resetTask();
        Task.setTask(&task);
    }

    // returns the internal task
    const TPC_t* getTask() const {
        return &Task;
    }
    
    // returns the kind of node
    virtual fftype getFFType() const   { return TPC_WORKER; }

    // TODO currently only one devices can be used
    void setDeviceId(tpc_dev_id_t id)  { deviceId = id; }
    tpc_dev_id_t  getDeviceId()  const  { return deviceId; }    
    int getTPCID() const { return tpcId; }

#if defined(FF_REPARA)
    /** 
     *  Returns input data size
     */
    size_t rpr_get_sizeIn()  const { return rpr_sizeIn; }

    /** 
     *  Returns output data size
     */
    size_t rpr_get_sizeOut() const { return rpr_sizeOut; }
#endif
    
protected:

    int nodeInit() {
        if (tpcId<0) {
            tpcId   = tpcEnvironment::instance()->getTPCID();

            if (deviceId == (tpc_dev_id_t)-1) { // the user didn't set any specific device
                dev_ctx = tpcEnvironment::instance()->getTPCDevice();
            } else 
                abort(); // TODO;

            if (dev_ctx == NULL) return -1;
        }
        return 0;
    }

    int svc_init() { return nodeInit(); }

    // NOTE: this implementation supposes that the number of arguments does not 
    //       change between two different tasks ! Only the size does change.
    void *svc(void *task) {
        if (task) {
            Task.resetTask();
            Task.setTask((IN_t*)task);
        }

#if defined(FF_REPARA)
        rpr_sizeIn = rpr_sizeOut = 0;
#endif

        const tpc_func_id_t f_id = Task.getKernelId();
        if (tpc_device_func_instance_count(dev_ctx, f_id) == 0) {
            error("No instances for function with id %d\n", (int)f_id);
            return GO_ON;
        }
        tpc_res_t res;
        tpc_job_id_t j_id = tpc_device_acquire_job_id(dev_ctx, f_id, TPC_DEVICE_ACQUIRE_JOB_ID_BLOCKING);
        
        const std::vector<internal::Arg_t> &inV  = Task.getInputArgs();
        const std::vector<internal::Arg_t> &outV = Task.getOutputArgs();
        size_t ret_val_size=0;
        void *ret_val = Task.getReturnVar(ret_val_size);

        if (inHandles.size() == 0)  inHandles.resize(inV.size());
        if (outHandles.size() == 0) outHandles.resize(outV.size());


        uint32_t arg_idx = 0;

        for(size_t i=0;i<inV.size();++i) {

            if ( ! inV[i].byvalue ) {
                tpc_handle_t handle = inHandles[i].handle;  // previous handle
                if (inHandles[i].size < inV[i].size) {
                    
                    if (inHandles[i].handle) { // not first time
                        assert(!inV[i].reuse);
                        allocator->releaseBuffer(inHandles[i].ptr, dev_ctx, inHandles[i].handle);
                    }
                    if (inV[i].reuse) {
                        handle = allocator->createBufferUnique(inV[i].ptr, dev_ctx, TPC_DEVICE_ALLOC_FLAGS_NONE, inV[i].size);
                    } else {
                        handle = allocator->createBuffer(inV[i].ptr, dev_ctx, TPC_DEVICE_ALLOC_FLAGS_NONE, inV[i].size);
                    }
                    
                    if (!handle) {
                        error("ff_tpcNode::svc unable to allocate memory on the TPC device (IN)");
                        inHandles[i].size = 0, inHandles[i].ptr = nullptr, inHandles[i].handle = 0;
                        return (oneshot?NULL:GO_ON);
                    }
                    inHandles[i].size = inV[i].size, inHandles[i].ptr = inV[i].ptr, inHandles[i].handle = handle;
                }
                
                if (inV[i].copy) {
                    res = tpc_device_copy_to(dev_ctx, inV[i].ptr, handle, inV[i].size, TPC_DEVICE_COPY_BLOCKING);
                    if (res != TPC_SUCCESS) {
                        error("ff_tpcNode::svc unable to copy data into the device\n");
                        return (oneshot?NULL:GO_ON);
                    }
#if defined(FF_REPARA)
                    rpr_sizeIn += inV[i].size;
#endif
                } 
                
                res = tpc_device_job_set_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);

            } else { // argument passed by-value
                res = tpc_device_job_set_arg(dev_ctx, j_id, arg_idx, inV[i].size, inV[i].ptr);
            }
            
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to set argument for the device (IN)\n");
                return (oneshot?NULL:GO_ON);
            }
            ++arg_idx;
        }

        for(size_t i=0;i<outV.size();++i) {
            void *ptr = outV[i].ptr;

            // check if ptr is an in/out parameter
            typename std::vector<handle_t>::iterator it;
            it = std::find_if(inHandles.begin(), inHandles.end(), [ptr](const handle_t &arg) { return arg.ptr == ptr; } );
            
            tpc_handle_t handle;
            if (it == inHandles.end()) { // not found                
                handle = outHandles[i].handle;  // previous handle
                if (outHandles[i].size < outV[i].size) {
                    
                    if (outHandles[i].handle) { // not first time
                        assert(!outV[i].reuse);
                        allocator->releaseBuffer(outHandles[i].ptr, dev_ctx, outHandles[i].handle);
                    }

                    if (outV[i].reuse) {
                        handle = allocator->createBufferUnique(ptr, dev_ctx, TPC_DEVICE_ALLOC_FLAGS_NONE, outV[i].size);
                    } else {
                        handle = allocator->createBuffer(ptr, dev_ctx, TPC_DEVICE_ALLOC_FLAGS_NONE, outV[i].size);
                    }
                    
                    if (!handle) {
                        error("ff_tpcNode::svc unable to allocate memory on the TPC device (OUT)");
                        outHandles[i].size = 0, outHandles[i].ptr = nullptr, outHandles[i].handle = 0;
                        return (oneshot?NULL:GO_ON);
                    }
                    outHandles[i].size = outV[i].size, outHandles[i].ptr = ptr, outHandles[i].handle = handle;
                }
            } else handle= (*it).handle; // in/out parameter
           
            res = tpc_device_job_set_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to set argument for the device (OUT)\n");
                return (oneshot?NULL:GO_ON);
            }
            ++arg_idx;
        }
        res = tpc_device_job_launch(dev_ctx, j_id, TPC_DEVICE_JOB_LAUNCH_BLOCKING);
        if (res != TPC_SUCCESS) {
            error("ff_tpcNode::svc error launching kernel on TPC device\n");
            return (oneshot?NULL:GO_ON);
        }

        // reset the arg_idx to the initial value for the out parameters
        arg_idx=inV.size();
        
        // TODO: async data transfer
        for(size_t i=0;i<outV.size();++i) {
            tpc_handle_t handle;
            res = tpc_device_job_get_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to get argument handle\n");
                return (oneshot?NULL:GO_ON);
            }
            if (outV[i].copy) {
                res = tpc_device_copy_from(dev_ctx, handle, outV[i].ptr, outV[i].size, TPC_DEVICE_COPY_BLOCKING);
                if (res != TPC_SUCCESS) {
                    error("ff_tpcNode::svc unable to copy data back from the device\n");
                    return (oneshot?NULL:GO_ON);
                }
#if defined(FF_REPARA)
                rpr_sizeOut += outV[i].size;
#endif
            } 

            // By default the buffers are not released !
            if (outV[i].release) {
                assert(outHandles[i].handle == handle);
                allocator->releaseBuffer(outHandles[i].ptr, dev_ctx, handle);
                outHandles[i].size = 0, outHandles[i].ptr = nullptr, outHandles[i].handle = 0;
            } 

            ++arg_idx;
        }

        // now check if we have to get back the return value (if the kernel is a function)
        if (ret_val && ret_val_size>0) {
            res = tpc_device_job_get_return(dev_ctx, j_id, ret_val_size, ret_val);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc error getting back the return value of the kernel function\n");
                return (oneshot?NULL:GO_ON);
            }
        }
            
        // By default the buffers are not released !
        for(size_t i=0;i<inV.size();++i) {
            if (inV[i].release) {
                allocator->releaseBuffer(inHandles[i].ptr, dev_ctx, inHandles[i].handle);
                inHandles[i].size = 0, inHandles[i].ptr = nullptr, inHandles[i].handle = 0;
            }
        }

        // release job id
        tpc_device_release_job_id(dev_ctx, j_id);

        // per task host memory cleanup phase
        OUT_t * task_out;
        if (task) task_out = Task.releaseTask((IN_t*)task);
        else      task_out = Task.releaseTask(const_cast<IN_t*>(oneshotTask));

        return (oneshot ? NULL : task_out);  
    }
    
protected:    
    int                 tpcId;
    tpc_dev_id_t        deviceId;    
    tpc_dev_ctx_t      *dev_ctx; 

    TPC_t               Task;

    bool                my_own_allocator;
    const bool          oneshot; 
    const IN_t *const   oneshotTask;
    ff_tpcallocator    *allocator;

    std::vector<handle_t> inHandles;
    std::vector<handle_t> outHandles;
};
    
}
#endif /* FF_TPCNODE_HPP */
