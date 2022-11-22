/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file mapCUDAManaged.hpp
 *  \ingroup high_level_patterns_shared_memory
 *
 *  \brief This file describes the map skeleton.
 *
 * Author: Massimo Torquati / Guilherme Peretti Pezzi
 *         torquati@di.unipi.it  massimotor@gmail.com  / peretti@di.unito.it
 */

#ifndef _FF_MAPCUDAMANAGED_HPP_
#define _FF_MAPCUDAMANAGED_HPP_
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

#include <ff/map.hpp>

namespace ff {

/**
 * User data class should extend ff_cuda_managed in order to use pattern mapCUDAManaged
 * (if data was not allocated with cudaMallocManaged())
 */
class ff_cuda_managed {
public:
	void *operator new(size_t len) {
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

	void operator delete(void *ptr) {
		cudaFree(ptr);
	}
};

/**
 * map base task for CUDA managed map implementation
 */
class baseCUDATaskManaged {
public:
	baseCUDATaskManaged(): envPtr(NULL), inPtr(NULL), outPtr(NULL) {};
	baseCUDATaskManaged(void * env, void * in, void * out) : envPtr(env), inPtr(in), outPtr(out) {}

	//user may override this code - BEGIN
	virtual void setTask(void * in) { if (in) inPtr=in;}
	virtual void*  getEnvPtr()     { return envPtr;}
	virtual void*  getInPtr()     { return inPtr;}

	// by default the map works in-place
	virtual void*  newOutPtr()    { return outPtr; }
	virtual void   deleteOutPtr() {}

	virtual void beforeMR() {}
	virtual void afterMR() {}

	//user may override this code - END

	virtual ~baseCUDATaskManaged() {
	}

protected:
	void *envPtr;
	void *inPtr;
	void *outPtr;
};

template<typename Tenv, typename Tinout, typename kernelF>
__global__ void mapCUDAKernelManaged(kernelF K, Tenv * env, Tinout * in, Tinout* out, size_t size) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int gridSize = blockDim.x*gridDim.x;
	if(i<size)
		out[i]= K.K(env, in[i]);
}

#define FFMAPFUNCMANAGED(name, basictype, Tenv, env, Tinout, in, code )  \
		struct name {                                                                 \
	__device__ basictype K(Tenv env, Tinout in) {                    \
			code ;                                                                   \
		}                                                                            \
}

/*!
 * \class ff_mapCUDAManaged
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The ff_mapCUDAManaged skeleton.
 *
 * The map skeleton using CUDA
 *
 */
template<typename T, typename kernelF>
class ff_mapCUDAManaged: public ff_node {
public:

	ff_mapCUDAManaged(kernelF *mapF, void * env, void * in, void * out, size_t s):
		oneshot(true),
		Task( (typename T::env_type *)env, (typename T::inout_type *)in, (typename T::inout_type *)out, s ),
		kernel(mapF) {
		assert(in);
		ff_node::skipfirstpop(true);
		maxThreads=maxBlocks=0;
		oldSize=0;
		inPtr = outPtr = NULL;
		envPtr = NULL;
	}

	int  run(bool=false) { return  ff_node::run(); }
	int  wait() { return ff_node::wait(); }

	int run_and_wait_end() {
		if (run()<0) return -1;
		if (wait()<0) return -1;
		return 0;
	}

	double ffTime()  { return ff_node::ffTime();  }
	double ffwTime() { return ff_node::wffTime(); }

	const T* getTask() const { return &Task; }

	void cleanup() { if (kernel) delete kernel; }

protected:

	int svc_init() {
		int deviceID = 0;         // FIX:  we have to manage multiple devices
		cudaDeviceProp deviceProp;

		cudaSetDevice(deviceID);
		if (cudaGetDeviceProperties(&deviceProp, deviceID) != cudaSuccess)
			error("mapCUDA, error getting device properties\n");

		if(deviceProp.major == 1 && deviceProp.minor < 2)
			maxThreads = 256;
		else
			maxThreads = deviceProp.maxThreadsPerBlock;
		maxBlocks = deviceProp.maxGridSize[0];

		if(cudaStreamCreate(&stream) != cudaSuccess)
			error("mapCUDA, error creating stream\n");

		return 0;
	}

	void * svc(void* task) {
		Task.setTask(task);
		size_t size = Task.size();
		inPtr  = (typename T::inout_type*) Task.getInPtr();
		outPtr = (typename T::inout_type*) Task.newOutPtr();
		envPtr = (typename T::env_type*) Task.getEnvPtr();

		size_t thxblock = std::min(maxThreads, size);
		size_t blockcnt = std::min(size/thxblock + (size%thxblock == 0 ?0:1), maxBlocks);

		mapCUDAKernelManaged<typename T::env_type, typename T::inout_type, kernelF>
		<<<blockcnt,thxblock,0,stream>>>(*kernel, envPtr, inPtr, outPtr, size);

		cudaStreamSynchronize(stream);

		//        return (NULL);
		return (oneshot?NULL:outPtr);
	}

	void svc_end() {

		if(cudaStreamDestroy(stream) != cudaSuccess)
			error("mapCUDA, error destroying stream\n");
	}
private:
	const bool   oneshot;
	T            Task;
	kernelF     *kernel;     // user function
	cudaStream_t stream;
	size_t       maxThreads;
	size_t       maxBlocks;
	size_t       oldSize;
	typename T::inout_type* inPtr;
	typename T::inout_type* outPtr;
	typename T::env_type* envPtr;
};

#define NEWMAPMANAGED(name, task_t, f, env, input, output, size)           \
		ff_mapCUDAManaged<task_t, mapf> *name =                            \
		new ff_mapCUDAManaged<task_t, f>( new f, env, input, output, size)

/*!
 *  @}
 *  \endlink
 */

} // namespace ff

#endif /* _FF_MAPCUDAMANAGED_HPP_ */
