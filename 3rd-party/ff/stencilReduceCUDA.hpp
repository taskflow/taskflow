/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
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
 *
 *  Author:
 *    Maurizio Drocco
 *  Contributors:  
 *     Guilherme Peretti Pezzi 
 *     Marco Aldinucci
 *     Massimo Torquati
 *
 *  First version: February 2014
 */

#ifndef FF_STENCILCUDA_HPP
#define FF_STENCILCUDA_HPP

// FF_CUDA must be defined in order to use this code!!!!
#ifdef FF_CUDA
#include <cuda.h>
#include <ff/mapper.hpp>
#include <ff/node.hpp>
#include <cmath>
#include <iostream>

namespace ff {

// map base task for CUDA implementation
// char type is a void type - clang++ don't like using void, g++ accept it
template<typename Tin_, typename Tout_ = Tin_, typename Tenv1_=char, typename Tenv2_=char, typename Tenv3_=char, typename Tenv4_=char, typename Tenv5_=char, typename Tenv6_=char>
class baseCUDATask {
public:
    typedef Tin_ Tin;
    typedef Tout_ Tout;
    typedef Tenv1_ Tenv1;
    typedef Tenv2_ Tenv2;
    typedef Tenv3_ Tenv3;
    typedef Tenv4_ Tenv4;
    typedef Tenv5_ Tenv5;
    typedef Tenv6_ Tenv6;
    
    baseCUDATask() : inPtr(NULL), outPtr(NULL), env1Ptr(NULL), env2Ptr(NULL), 
                     env3Ptr(NULL), env4Ptr(NULL), env5Ptr(NULL), env6Ptr(NULL),
                     inDevicePtr(NULL), outDevicePtr(NULL), env1DevicePtr(NULL), 
                     env2DevicePtr(NULL), env3DevicePtr(NULL), env4DevicePtr(NULL), 
                     env5DevicePtr(NULL), env6DevicePtr(NULL) {
        size_in = size_out = size_env1 = size_env2 = size_env3 = size_env4 = size_env5 = size_env6 = 0;
    }
    virtual ~baseCUDATask() { }
    
    // user must override this method
    virtual void setTask(void* t) = 0;
    //user may override this code - BEGIN
    virtual void startMR()        {}
    virtual void beforeMR()       {}
    virtual void afterMR(void *)  {}
    virtual void endMR(void *)    {}    
    virtual bool iterCondition(Tout rVar, size_t iter) { return true; }
    virtual void swap() {}
    //user may override this code - END

    size_t getBytesizeIn()   const { return getSizeIn() * sizeof(Tin);     }
    size_t getBytesizeOut()  const { return getSizeOut() * sizeof(Tout);    }
    size_t getBytesizeEnv1() const { return getSizeEnv1() * sizeof(Tenv1); }
    size_t getBytesizeEnv2() const { return getSizeEnv2() * sizeof(Tenv2); }
    size_t getBytesizeEnv3() const { return getSizeEnv3() * sizeof(Tenv3); }
    size_t getBytesizeEnv4() const { return getSizeEnv4() * sizeof(Tenv4); }
    size_t getBytesizeEnv5() const { return getSizeEnv5() * sizeof(Tenv5); }
    size_t getBytesizeEnv6() const { return getSizeEnv6() * sizeof(Tenv6); }

    void setSizeIn(size_t   sizeIn)   {	size_in   = sizeIn;   }
    void setSizeOut(size_t  sizeOut)  {	size_out  = sizeOut;   }
    void setSizeEnv1(size_t sizeEnv1) { size_env1 = sizeEnv1; }
    void setSizeEnv2(size_t sizeEnv2) { size_env2 = sizeEnv2; }
    void setSizeEnv3(size_t sizeEnv3) {	size_env3 = sizeEnv3; }
    void setSizeEnv4(size_t sizeEnv4) { size_env4 = sizeEnv4; }
    void setSizeEnv5(size_t sizeEnv5) { size_env5 = sizeEnv5; }
    void setSizeEnv6(size_t sizeEnv6) { size_env6 = sizeEnv6; }

    size_t getSizeIn()   const { return size_in;   }
    size_t getSizeOut()  const { return (size_out==0)?size_in:size_out;  }
    size_t getSizeEnv1() const { return size_env1; }
    size_t getSizeEnv2() const { return size_env2; }    
    size_t getSizeEnv3() const { return size_env3; }
    size_t getSizeEnv4() const { return size_env4; }
    size_t getSizeEnv5() const { return size_env5; }
    size_t getSizeEnv6() const { return size_env6; }

    void setInPtr(Tin*     _inPtr)     { inPtr   = _inPtr;  }
    void setOutPtr(Tout*   _outPtr)    { outPtr  = _outPtr;  }
    void setEnv1Ptr(Tenv1* _env1Ptr)   { env1Ptr = _env1Ptr; }
    void setEnv2Ptr(Tenv2* _env2Ptr)   { env2Ptr = _env2Ptr; }
    void setEnv3Ptr(Tenv3* _env3Ptr)   { env3Ptr = _env3Ptr; }
    void setEnv4Ptr(Tenv4* _env4Ptr)   { env4Ptr = _env4Ptr; }
    void setEnv5Ptr(Tenv5* _env5Ptr)   { env5Ptr = _env5Ptr; }
    void setEnv6Ptr(Tenv6* _env6Ptr)   { env6Ptr = _env6Ptr; }

    Tin*   getInPtr()   const { return inPtr;   }
    Tout*  getOutPtr()  const { return outPtr;  }
    Tenv1* getEnv1Ptr() const { return env1Ptr; }
    Tenv2* getEnv2Ptr() const { return env2Ptr; }
    Tenv3* getEnv3Ptr() const { return env3Ptr; }
    Tenv4* getEnv4Ptr() const { return env4Ptr; }
    Tenv5* getEnv5Ptr() const { return env5Ptr; }
    Tenv6* getEnv6Ptr() const { return env6Ptr; }


    Tin*   getInDevicePtr() const   { return inDevicePtr;   }
    Tout*  getOutDevicePtr() const  { return outDevicePtr;  }
    Tenv1* getEnv1DevicePtr() const { return env1DevicePtr; }
    Tenv2* getEnv2DevicePtr() const { return env2DevicePtr; }
    Tenv3* getEnv3DevicePtr() const { return env3DevicePtr; }
    Tenv4* getEnv4DevicePtr() const { return env4DevicePtr; }
    Tenv5* getEnv5DevicePtr() const { return env5DevicePtr; }
    Tenv6* getEnv6DevicePtr() const { return env6DevicePtr; }

    void setInDevicePtr(Tin*     _inDevicePtr)   { inDevicePtr   = _inDevicePtr;  }
    void setOutDevicePtr(Tout*   _outDevicePtr)  { outDevicePtr  = _outDevicePtr; }
    void setEnv1DevicePtr(Tenv1* _env1DevicePtr) { env1DevicePtr = _env1DevicePtr; }
    void setEnv2DevicePtr(Tenv2* _env2DevicePtr) { env2DevicePtr = _env2DevicePtr; }
    void setEnv3DevicePtr(Tenv3* _env3DevicePtr) { env3DevicePtr = _env3DevicePtr; }
    void setEnv4DevicePtr(Tenv4* _env4DevicePtr) { env4DevicePtr = _env4DevicePtr; }
    void setEnv5DevicePtr(Tenv5* _env5DevicePtr) { env5DevicePtr = _env5DevicePtr; }
    void setEnv6DevicePtr(Tenv6* _env6DevicePtr) { env6DevicePtr = _env6DevicePtr; }

    void setReduceVar(Tout r) {	reduceVar = r;  }
    Tout getReduceVar() const { return reduceVar;  }
    
protected:
    Tin *inPtr, *inDevicePtr;
    Tout *outPtr, *outDevicePtr;
    Tenv1 *env1Ptr, *env1DevicePtr;
    Tenv2 *env2Ptr, *env2DevicePtr;
    Tenv3 *env3Ptr, *env3DevicePtr;
    Tenv4 *env4Ptr, *env4DevicePtr;
    Tenv5 *env5Ptr, *env5DevicePtr;
    Tenv6 *env6Ptr, *env6DevicePtr;
    size_t size_in, size_out, size_env1, size_env2, size_env3, size_env4, size_env5, size_env6;
    Tout reduceVar;
};

#define FFMAPFUNC(name, T, param, code )                \
    struct name {                                                       \
        __device__ T K(T param, void *param1, void *param2, void *param3, void *param4, void *param5, void *param6) { \
            code                                                        \
                }                                                       \
    }
    
#define FFMAPFUNC2(name, outT, inT, param, code )                   \
    struct name {                                                       \
        __device__ outT K(inT param, void *param1, void *param2, void *param3, void *param4, void *param5, void *param6) { \
            code                                                        \
                }                                                       \
    }

#define FFMAPFUNC3(name, outT, inT, param, env1T, param1, env2T, param2, env3T, param3, code) \
    struct name {                                                       \
        __device__ outT K(inT param, env1T *param1, env2T *param2, env3T *param3, void *dummy0, void *dummy1, void *dummy2) { \
            code                                                        \
                }                                                       \
    }

#define FFMAPFUNC4(name, outT, inT, param, env1T, param1, env2T, param2, env3T, param3, env4T, param4, code) \
    struct name {                                                       \
        __device__ outT K(inT param, env1T *param1, env2T *param2, env3T *param3, env4T *param4, void *dummy0, void *dummy1) { \
            code                                                        \
                }                                                       \
    }
    
#define FFMAPFUNC5(name, outT, inT, param, env1T, param1, env2T, param2, env3T, param3, env4T, param4, env5T, param5, code) \
    struct name {                                                       \
        __device__ outT K(inT param, env1T *param1, env2T *param2, env3T *param3, env4T *param4, env5T *param5, void *dummy1) { \
            code                                                        \
                }                                                       \
    }
    
#define FFMAPFUNC6(name, outT, inT, param, env1T, param1, env2T, param2, env3T, param3, env4T, param4, env5T, param5, env6T, param6, code) \
    struct name {                                                       \
        __device__ outT K(inT param, env1T *param1, env2T *param2, env3T *param3, env4T *param4, env5T *param5, env6T *param6) { \
            code                                                        \
                }                                                       \
    }
    
#define FFREDUCEFUNC(name, T, param1, param2, code)	\
    struct device##name {                                               \
        __device__ T K(T param1, T param2, void *, void *, void *, void *, void *, void *) { \
            code                                                        \
                }                                                       \
    };                                                                  \
    struct host##name {                                                 \
        T K(T param1, T param2, void *, void *, void *, void *, void *, void *) { \
            code                                                        \
                }                                                       \
    }
    
    
#define FFSTENCILREDUCECUDA(taskT, mapT, reduceT)			 \
    ff_stencilReduceCUDA<taskT, mapT, device##reduceT, host##reduceT>

#define FFMAPCUDA(taskT, mapT)			                         \
    ff_mapCUDA<taskT, mapT>

#define FFREDUCECUDA(taskT, reduceT)				         \
    ff_reduceCUDA<taskT, device##reduceT, host##reduceT>

//following 2 macros are for backward compatibility
#define NEWMAP(name, task_t, f, input, iter)               \
    ff_mapCUDA<task_t,f> *name =                      \
        new ff_mapCUDA<task_t, f>(input, iter)
#define NEWMAPONSTREAM(task_t, f)                        \
    new ff_mapCUDA<task_t, f>()

//kernel for buffer initialization
template<typename T>
__global__ void initCUDAKernel(T* data, T value, size_t size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    while (i < size) {
	data[i] = value;
	i += gridSize;
    }
}
    
/* The following code (mapCUDAKernerl, SharedMemory and reduceCUDAKernel)
 * has been taken from the SkePU CUDA code
 * http://www.ida.liu.se/~chrke/skepu/
 *
 */
template<typename kernelF, typename Tin, typename Tout, typename Tenv1, typename Tenv2, typename Tenv3, typename Tenv4, typename Tenv5, typename Tenv6>
__global__ void mapCUDAKernel(kernelF K, Tin* input, Tout* output, Tenv1 *env1, Tenv2 *env2, Tenv3 *env3, Tenv4 *env4, Tenv5 *env5, Tenv6 *env6, size_t size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    
    while (i < size) {
	output[i] = K.K(input[i], env1, env2, env3, env4, env5, env6);
	i += gridSize;
    }
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
    __device__ inline operator T*() {
	extern __shared__ int __smem[];
	return (T*)__smem;
    }
    
    __device__ inline operator const T*() const {
	extern __shared__ int __smem[];
	return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<> struct SharedMemory<double> {
    __device__ inline operator double*() {
	extern __shared__ double __smem_d[];
	return (double*)__smem_d;
    }
    __device__ inline operator const double*() const {
	extern __shared__ double __smem_d[];
	return (double*)__smem_d;
    }
};

template<typename kernelF, typename T, typename Tenv1, typename Tenv2, typename Tenv3, typename Tenv4, typename Tenv5, typename Tenv6>
__global__ void reduceCUDAKernel(kernelF K, T *output, T *input, Tenv1 *env1, Tenv2 *env2, Tenv3 *env3, Tenv4 *env4, Tenv5 *env5, Tenv6 *env6, 
				 size_t size, unsigned int blockSize, bool nIsPow2, T identityVal) {

    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T result = identityVal;
    
    if(i < size) {
        result = input[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        
        // There we pass it always false
        if (nIsPow2 || i + blockSize < size)
            result = K.K(result, input[i+blockSize], env1, env2, env3, env4, env5, env6);
        i += gridSize;
    }
    
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while(i < size) {
        result = K.K(result, input[i], env1, env2, env3, env4, env5, env6);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < size)
            result = K.K(result, input[i+blockSize], env1, env2, env3, env4, env5, env6);
        i += gridSize;
    }
    
    // each thread puts its local sum into shared memory
    sdata[tid] = result;
    
    __syncthreads();
    
    // do reduction in shared mem
    if (blockSize >= 1024) {if (tid < 512) {sdata[tid] = result = K.K(result, sdata[tid + 512], env1, env2, env3, env4, env5, env6);}__syncthreads();}
    if (blockSize >= 512) {if (tid < 256) {sdata[tid] = result = K.K(result, sdata[tid + 256], env1, env2, env3, env4, env5, env6);}__syncthreads();}
    if (blockSize >= 256) {if (tid < 128) {sdata[tid] = result = K.K(result, sdata[tid + 128], env1, env2, env3, env4, env5, env6);}__syncthreads();}
    if (blockSize >= 128) {if (tid < 64)  {sdata[tid] = result = K.K(result, sdata[tid + 64],  env1, env2, env3, env4, env5, env6);}__syncthreads();}
    
    if (tid < 32) {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >= 64) {smem[tid] = result = K.K(result, smem[tid + 32], env1, env2, env3, env4, env5, env6);}
        if (blockSize >= 32) {smem[tid] = result = K.K(result, smem[tid + 16], env1, env2, env3, env4, env5, env6);}
        if (blockSize >= 16) {smem[tid] = result = K.K(result, smem[tid + 8],  env1, env2, env3, env4, env5, env6);}
        if (blockSize >= 8)  {smem[tid] = result = K.K(result, smem[tid + 4],  env1, env2, env3, env4, env5, env6);}
        if (blockSize >= 4)  {smem[tid] = result = K.K(result, smem[tid + 2],  env1, env2, env3, env4, env5, env6);}
        if (blockSize >= 2)  {smem[tid] = result = K.K(result, smem[tid + 1],  env1, env2, env3, env4, env5, env6);}
    }
    
    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
    
template<typename taskT, typename TkernelMap, typename TkernelReduce, typename ThostReduce>
class ff_stencilReduceCUDA: public ff_node {
public:
    typedef typename taskT::Tin Tin;
    typedef typename taskT::Tout Tout;
    typedef typename taskT::Tenv1 Tenv1;
    typedef typename taskT::Tenv2 Tenv2;
    typedef typename taskT::Tenv3 Tenv3;
    typedef typename taskT::Tenv4 Tenv4;
    typedef typename taskT::Tenv5 Tenv5;
    typedef typename taskT::Tenv6 Tenv6;
    
    ff_stencilReduceCUDA(size_t maxIter_ = 1, Tout identityValue_ = Tout()) :
        oneShot(NULL), identityValue(identityValue_), iter(0), maxIter(maxIter_) {
        maxThreads = maxBlocks = 0;
        oldSize_in = oldSize_out = oldSize_env1 = oldSize_env2 = oldSize_env3 = oldSize_env4 = oldSize_env5 = oldSize_env6 = 0;
        deviceID=-1;
        stream = NULL;
        kernelMap = new TkernelMap();
        kernelReduce = new TkernelReduce();
        hostReduce = new ThostReduce();
        assert(kernelMap != NULL && kernelReduce != NULL && hostReduce != NULL);
        in_buffer   = NULL;   out_buffer  = NULL; 
        env1_buffer = NULL;   env2_buffer = NULL;
        env3_buffer = NULL;   env4_buffer = NULL;
        env5_buffer = NULL;   env6_buffer = NULL;

                
        if (cudaStreamCreate(&stream) != cudaSuccess)
            error("mapCUDA, error creating stream\n");
        
    }
    ff_stencilReduceCUDA(const taskT &task, size_t maxIter_ = 1, Tout identityValue_ = Tout()) :
        oneShot(&task), identityValue(identityValue_), iter(0), maxIter(maxIter_) {
        maxThreads = maxBlocks = 0;
        oldSize_in = oldSize_out = oldSize_env1 = oldSize_env2 = oldSize_env3 = oldSize_env4 = oldSize_env5 = oldSize_env6 = 0;
        deviceID=-1;
        stream = NULL;
        kernelMap = new TkernelMap();
        kernelReduce = new TkernelReduce();
        hostReduce = new ThostReduce();
        assert(kernelMap != NULL && kernelReduce != NULL && hostReduce != NULL);
        in_buffer   = NULL;   out_buffer  = NULL; 
        env1_buffer = NULL;   env2_buffer = NULL;
        env3_buffer = NULL;   env4_buffer = NULL;
        env5_buffer = NULL;   env6_buffer = NULL;
        Task.setTask((void *)&task);
        
        if (cudaStreamCreate(&stream) != cudaSuccess)
            error("mapCUDA, error creating stream\n");
    }
    virtual ~ff_stencilReduceCUDA() {
        if ((void *)Task.getInDevicePtr() != (void *)Task.getOutDevicePtr())
            if (Task.getOutDevicePtr()) cudaFree(Task.getOutDevicePtr());
        if (Task.getInDevicePtr()) cudaFree(Task.getInDevicePtr());
        if(Task.getEnv1DevicePtr()) cudaFree(Task.getEnv1DevicePtr());
        if(Task.getEnv2DevicePtr()) cudaFree(Task.getEnv2DevicePtr());
        if(Task.getEnv3DevicePtr()) cudaFree(Task.getEnv3DevicePtr());
        if(Task.getEnv4DevicePtr()) cudaFree(Task.getEnv4DevicePtr());
        if(Task.getEnv5DevicePtr()) cudaFree(Task.getEnv5DevicePtr());
        if(Task.getEnv6DevicePtr()) cudaFree(Task.getEnv6DevicePtr());
        if(cudaStreamDestroy(stream) != cudaSuccess)
            error("mapCUDA, error destroying stream\n");
    }

    virtual void setMaxThreads(const size_t mt) { maxThreads = mt; } 

protected:
    
    virtual bool isPureMap() {return false;}
    virtual bool isPureReduce() {return false;}
    
    inline int svc_init() {
        // set the CUDA device where the node will be executed
        if(get_my_id() < 0)
        	deviceID = 0;
         else
             deviceID = get_my_id() % threadMapper::instance()->getNumCUDADevices();
        cudaDeviceProp deviceProp;
        
        cudaSetDevice(deviceID);
        if (cudaGetDeviceProperties(&deviceProp, deviceID) != cudaSuccess)
            error("mapCUDA, error getting device properties\n");
        
        const size_t mtxb = deviceProp.maxThreadsPerBlock;
        if (maxThreads == 0) {
            if (deviceProp.major == 1 && deviceProp.minor < 2)
                maxThreads = 256;
            else {
            	if (mtxb > 1024)
            		maxThreads = 1024;
            	else
	                maxThreads = mtxb;
	        }
        } else 
            maxThreads = std::min(maxThreads, (size_t)1024);
        
        maxBlocks = deviceProp.maxGridSize[0];
        
        // allocate memory on device having the initial size
        //			if (cudaMalloc(&in_buffer, Task.getBytesizeIn()) != cudaSuccess)
        //			error("mapCUDA error while allocating memory on device\n");
        //			oldSize_in = Task.getBytesizeIn();
        
        return 0;
    }

    
    inline void *svc(void *task) {
        if (task) Task.setTask(task);
        size_t size    = Task.getSizeIn();
        Tin *inPtr     = Task.getInPtr();
        Tout *outPtr   = Task.getOutPtr();
        Tenv1 *env1Ptr = Task.getEnv1Ptr();
        Tenv2 *env2Ptr = Task.getEnv2Ptr();
        Tenv3 *env3Ptr = Task.getEnv3Ptr();
        Tenv4 *env4Ptr = Task.getEnv4Ptr();
        Tenv5 *env5Ptr = Task.getEnv5Ptr();
        Tenv6 *env6Ptr = Task.getEnv6Ptr();
        
        size_t thxblock = std::min(maxThreads, size);
        size_t blockcnt = std::min(size / thxblock + (size % thxblock == 0 ? 0 : 1), maxBlocks);
        
        size_t padded_size = (size_t)pow(2, ceil(std::log2((float)size)));
        size_t thxblock_r = std::min(maxThreads, padded_size);
        size_t blockcnt_r = std::min(padded_size / thxblock_r + (padded_size % thxblock_r == 0 ? 0 : 1), maxBlocks);
        
        Task.startMR();

        //in
        if (oldSize_in < Task.getBytesizeIn()) {
            cudaFree(in_buffer);
            if (cudaMalloc(&in_buffer, padded_size * sizeof(Tin)) != cudaSuccess)
                error("mapCUDA error while allocating memory on device (in buffer)\n");
            oldSize_in = Task.getBytesizeIn();
        }
        Task.setInDevicePtr(in_buffer);
        //initCUDAKernel<Tin><<<blockcnt_r, thxblock_r, 0, stream>>>(Task.getInDevicePtr(), (Tin)identityValue, padded_size);
        cudaMemcpyAsync(in_buffer, inPtr, Task.getBytesizeIn(),
                        cudaMemcpyHostToDevice, stream);
        
        //env1
        if(env1Ptr) {
            if (oldSize_env1 < Task.getBytesizeEnv1()) {
                cudaFree(env1_buffer);
                if (cudaMalloc(&env1_buffer, Task.getBytesizeEnv1()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env1 buffer)\n");
                oldSize_env1 = Task.getBytesizeEnv1();
            }
            Task.setEnv1DevicePtr(env1_buffer);
            cudaMemcpyAsync(env1_buffer, env1Ptr, Task.getBytesizeEnv1(), cudaMemcpyHostToDevice, stream);
        }
        //env2
        if(env2Ptr) {
            if (oldSize_env2 < Task.getBytesizeEnv2()) {
                cudaFree(env2_buffer);
                if (cudaMalloc(&env2_buffer, Task.getBytesizeEnv2()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env2 buffer)\n");
                oldSize_env2 = Task.getBytesizeEnv2();
            }
            Task.setEnv2DevicePtr(env2_buffer);
            cudaMemcpyAsync(env2_buffer, env2Ptr, Task.getBytesizeEnv2(), cudaMemcpyHostToDevice, stream);
        }
        //env3
        if(env3Ptr) {
            if (oldSize_env3 < Task.getBytesizeEnv3()) {
                cudaFree(env3_buffer);
                if (cudaMalloc(&env3_buffer, Task.getBytesizeEnv3()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env3 buffer)\n");
                oldSize_env3 = Task.getBytesizeEnv3();
            }
            Task.setEnv3DevicePtr(env3_buffer);
            cudaMemcpyAsync(env3_buffer, env3Ptr, Task.getBytesizeEnv3(), cudaMemcpyHostToDevice, stream);
        }
        //env4
        if(env4Ptr) {
            if (oldSize_env4 < Task.getBytesizeEnv4()) {
                cudaFree(env4_buffer);
                if (cudaMalloc(&env4_buffer, Task.getBytesizeEnv4()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env4 buffer)\n");
                oldSize_env4 = Task.getBytesizeEnv4();
            }
            Task.setEnv4DevicePtr(env4_buffer);
            cudaMemcpyAsync(env4_buffer, env4Ptr, Task.getBytesizeEnv4(), cudaMemcpyHostToDevice, stream);
        }
        //env5
        if(env5Ptr) {
            if (oldSize_env5 < Task.getBytesizeEnv5()) {
                cudaFree(env5_buffer);
                if (cudaMalloc(&env5_buffer, Task.getBytesizeEnv5()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env5 buffer)\n");
                oldSize_env5 = Task.getBytesizeEnv5();
            }
            Task.setEnv5DevicePtr(env5_buffer);
            cudaMemcpyAsync(env5_buffer, env5Ptr, Task.getBytesizeEnv5(), cudaMemcpyHostToDevice, stream);
        }
        //env6
        if(env6Ptr) {
            if (oldSize_env6 < Task.getBytesizeEnv6()) {
                cudaFree(env6_buffer);
                if (cudaMalloc(&env6_buffer, Task.getBytesizeEnv6()) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (env6 buffer)\n");
                oldSize_env6 = Task.getBytesizeEnv6();
            }
            Task.setEnv6DevicePtr(env6_buffer);
            cudaMemcpyAsync(env6_buffer, env6Ptr, Task.getBytesizeEnv6(), cudaMemcpyHostToDevice, stream);
        }
        
        //TODO: in-place        
        if ((void*)inPtr != (void*)outPtr) {
            if (oldSize_out < Task.getBytesizeOut()) {
                if (out_buffer) {
                    cudaFree(out_buffer);
                }
                if (cudaMalloc(&out_buffer, padded_size * sizeof(Tout)) != cudaSuccess)
                    error("mapCUDA error while allocating memory on device (out buffer)\n");
                oldSize_out = Task.getBytesizeOut();
                
                //init kernels
                initCUDAKernel<Tout><<<blockcnt_r, thxblock_r, 0, stream>>>(out_buffer, (Tout)identityValue, padded_size);
            }
            Task.setOutDevicePtr(out_buffer);
        } else Task.setOutDevicePtr((Tout*)in_buffer); 

               
        iter = 0;        
        if(isPureMap()) {
            Task.swap();			// because of the next swap op
            do {
                Task.swap();
                
                Task.beforeMR();
                
                //CUDA Map
                mapCUDAKernel<TkernelMap, Tin, Tout, Tenv1, Tenv2, Tenv3, Tenv4, Tenv5, Tenv6><<<blockcnt, thxblock, 0, stream>>>(*kernelMap, Task.getInDevicePtr(), Task.getOutDevicePtr(), Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr(), size);
                
                Task.afterMR(task?task:(void*)oneShot);
            } while (Task.iterCondition(reduceVar, ++iter) && iter < maxIter);
            
            cudaMemcpyAsync(Task.getOutPtr(), Task.getOutDevicePtr(), Task.getBytesizeOut(), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
        } else {

            //allocate memory for reduce
            Tout *reduceBlocksPtr, *reduce_buffer;
            reduceBlocksPtr = (Tout *)malloc(blockcnt_r * sizeof(Tout));
            if (cudaMalloc(&reduce_buffer, blockcnt_r * sizeof(Tout)) != cudaSuccess)
                error("mapCUDA error while allocating memory on device (reduce buffer)\n");


            if(isPureReduce()) {
                Task.beforeMR();
                //Reduce
                //CUDA: blockwise reduce
                reduceCUDAKernel<TkernelReduce, Tout, Tenv1, Tenv2, Tenv3, Tenv4, Tenv5, Tenv6><<<blockcnt_r, thxblock_r, thxblock_r * sizeof(Tout), stream>>>(*kernelReduce, reduce_buffer, (Tout *)Task.getInDevicePtr(), Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr(), padded_size, thxblock_r, false, (Tout)identityValue);
                //copy reduce-blocks back to host
                cudaMemcpyAsync(reduceBlocksPtr, reduce_buffer, blockcnt_r * sizeof(Tout), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                //host: reduce blocks into reduceVar
                reduceVar = identityValue;
                for(size_t i=0; i<blockcnt_r; ++i)
                    reduceVar = hostReduce->K(reduceVar, reduceBlocksPtr[i], Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr());
                Task.setReduceVar(reduceVar);            
                Task.afterMR(task?task:(void*)oneShot);
            }
            else {
                Task.swap();			// because of the next swap op
                do {
                    Task.swap();
                    
                    Task.beforeMR();
                    
                    //CUDA Map
                    mapCUDAKernel<TkernelMap, Tin, Tout, Tenv1, Tenv2, Tenv3, Tenv4, Tenv5, Tenv6><<<blockcnt, thxblock, 0, stream>>>(*kernelMap, Task.getInDevicePtr(), Task.getOutDevicePtr(), Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr(), size);
                    
                    cudaError err = cudaGetLastError();
                    if(err!=cudaSuccess)
                    	std::cerr << "Problem launching mapCUDAKernel, code = " << err <<" \n";

                    //Reduce
                    //CUDA: blockwise reduce
                    reduceCUDAKernel<TkernelReduce, Tout, Tenv1, Tenv2, Tenv3, Tenv4, Tenv5, Tenv6><<<blockcnt_r, thxblock_r, thxblock_r * sizeof(Tout), stream>>>(*kernelReduce, reduce_buffer, Task.getOutDevicePtr(), Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr(), padded_size, thxblock_r, false, (Tout)identityValue);
                    //copy reduce-blocks back to host
                    cudaMemcpyAsync(reduceBlocksPtr, reduce_buffer, blockcnt_r * sizeof(Tout), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    err = cudaGetLastError();
                    if(err!=cudaSuccess)
                    	std::cerr << "Problem launching reduceCUDAKernel, code = " << err <<" \n";

                    //host: reduce blocks into reduceVar
                    reduceVar = identityValue;
                    for(size_t i=0; i<blockcnt_r; ++i)
                        reduceVar = hostReduce->K(reduceVar, reduceBlocksPtr[i], Task.getEnv1DevicePtr(), Task.getEnv2DevicePtr(), Task.getEnv3DevicePtr(), Task.getEnv4DevicePtr(), Task.getEnv5DevicePtr(), Task.getEnv6DevicePtr());
                    Task.setReduceVar(reduceVar);
                    Task.afterMR(task?task:(void*)oneShot);                
                } while (Task.iterCondition(reduceVar, ++iter) && iter < maxIter);
                
                cudaMemcpyAsync(Task.getOutPtr(), Task.getOutDevicePtr(), Task.getBytesizeOut(), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                cudaError err = cudaGetLastError();
                if(err!=cudaSuccess)
                	std::cerr << "Problem launching cudaMemcpyAsync (after kernel calls), code = " << err <<" \n";
            }
            free(reduceBlocksPtr);
            cudaFree(reduce_buffer);
        }

        Task.endMR(task?task:(void*)oneShot);

        return (oneShot?NULL:task);
    }
    
    size_t  getIter()     const { return iter; }
    ssize_t getDeviceID() const { return deviceID;}
    
public:
    virtual inline int run_and_wait_end() {
            if (isfrozen()) {
                stop();
                thaw();
                if (wait() < 0) return -1;
                return 0;
            }
            stop();
            if (run() < 0)  return -1;
            if (wait() < 0) return -1;
            return 0;
        }
    
protected:
    const taskT *oneShot;
    Tout identityValue;
    size_t iter;
    size_t maxIter;
    taskT Task;
    
private:
    ssize_t deviceID;
    cudaStream_t stream;
    size_t maxThreads;
    size_t maxBlocks;
    size_t oldSize_in, oldSize_out, oldSize_env1, oldSize_env2, oldSize_env3, oldSize_env4, oldSize_env5, oldSize_env6;
    TkernelMap *kernelMap;
    TkernelReduce *kernelReduce;
    ThostReduce *hostReduce;
    Tout reduceVar;
    //device buffers
    Tin* in_buffer;
    Tout* out_buffer;
    Tenv1* env1_buffer;
    Tenv2* env2_buffer;
    Tenv3* env3_buffer;
    Tenv4* env4_buffer;
    Tenv5* env5_buffer;
    Tenv6* env6_buffer;
};
    
template<typename taskT>
struct dummyMapF {
    __device__ typename taskT::Tout K(typename taskT::Tin, typename taskT::Tenv1 *, typename taskT::Tenv2 *, typename taskT::Tenv3 *, typename taskT::Tenv4 *, typename taskT::Tenv5 *, typename taskT::Tenv6 *)
    {	return (typename taskT::Tout)0;}
};
    
template<typename taskT>
struct dummyReduceF {
    __device__ typename taskT::Tout K(typename taskT::Tout a, typename taskT::Tout,typename taskT::Tenv1 *, typename taskT::Tenv2 *, typename taskT::Tenv3 *, typename taskT::Tenv4 *, typename taskT::Tenv5 *, typename taskT::Tenv6 *) {return a;}
};
    
template<typename taskT>
struct dummyHostReduceF {
    typename taskT::Tout K(typename taskT::Tout a, typename taskT::Tout,typename taskT::Tenv1 *, typename taskT::Tenv2 *, typename taskT::Tenv3 *, typename taskT::Tenv4 *, typename taskT::Tenv5 *, typename taskT::Tenv6 *) {return a;}
};
    
template<typename taskT, typename TkernelMap>
class ff_mapCUDA: public ff_stencilReduceCUDA<taskT, TkernelMap, dummyReduceF<taskT>, dummyHostReduceF<taskT> > {
    bool isPureMap() {return true;}
public:
    ff_mapCUDA(size_t maxIter = 1) :
    	ff_stencilReduceCUDA<taskT, TkernelMap, dummyReduceF<taskT>, dummyHostReduceF<taskT> >(maxIter) {}
    ff_mapCUDA(const taskT &task, size_t maxIter = 1) :
        	ff_stencilReduceCUDA<taskT, TkernelMap, dummyReduceF<taskT>, dummyHostReduceF<taskT> >(task, maxIter) {}
};
    
template<typename taskT, typename TkernelReduce, typename ThostReduce>
class ff_reduceCUDA: public ff_stencilReduceCUDA<taskT, dummyMapF<taskT>, TkernelReduce, ThostReduce> {
    bool isPureReduce() {return true;}
public:
    ff_reduceCUDA(size_t maxIter = 1) :
    	ff_stencilReduceCUDA<taskT, dummyMapF<taskT>, TkernelReduce, ThostReduce>(maxIter) {}
    ff_reduceCUDA(const taskT &task, size_t maxIter = 1) :
    	ff_stencilReduceCUDA<taskT, dummyMapF<taskT>, TkernelReduce, ThostReduce>(task, maxIter) {}
};

    
} //namespace

#endif  // FF_CUDA 

#endif /* FF_STENCILCUDA_HPP */
