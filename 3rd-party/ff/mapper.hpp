/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file mapper.hpp
 *  \ingroup shared_memory_fastflow
 *
 *  \brief This file contains the thread mapper definition used in FastFlow
 */

#ifndef __THREAD_MAPPER_HPP_
#define __THREAD_MAPPER_HPP_

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
#include <ff/config.hpp>
#include <ff/svector.hpp>
#include <ff/utils.hpp>
#include <ff/mapping_utils.hpp>
#include <vector>
#if defined(MAMMUT)
#include <mammut/mammut.hpp>
#endif


#if defined(FF_CUDA) 
#include <cuda.h>
#endif

#if 0
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#include <ff/ocl/clEnvironment.hpp>
#endif





#endif

namespace ff {

/*!
 *  \ingroup shared_memory_fastflow
 *
 *  @{
 */

/*! 
 * \class threadMapper
 * \ingroup shared_memory_fastflow
 *
 * \brief The thread mapper allows to map threads to specific core using a
 * predefined mapping policy.
 *
 * The threadMapper stores a list of CPU ids. By default the list is simply a
 * linear sequence of core ids of the system, for example in a quad-core
 * system the default list is 0 1 2 3. It is possible to change the default
 * list using the method setMappingList by passing a string of space-serated
 * (or comma-separated) CPU ids. The policy implemented in the threadManager
 * is to pick up a CPU id from the list using a round-robin policy.
 *
 * This class is defined in \ref mapper.hpp
 *
 */
class threadMapper {
public:
	/**
	 * Get a static instance of the threadMapper object
	 *
	 * \return TODO
	 */
	static inline threadMapper* instance() {
		static threadMapper thm;
		return &thm;
	}

	/**
	 * Default constructor.
	 */
	threadMapper() :
			rrcnt(-1), mask(0) {
        unsigned int size = -1;
#if defined(MAMMUT)
        mammut::Mammut m;
        std::vector<mammut::topology::Cpu*> cpus = m.getInstanceTopology()->getCpus();
		if (cpus.size()<=0 || cpus[0]->getPhysicalCores().size() <=0) {
            error("threadMapper: invalid number of cores\n");
            return ;
        }
        size_t virtualPerPhysical = cpus[0]->getPhysicalCores()[0]->getVirtualCores().size();
        for(size_t k = 0; k < virtualPerPhysical; k++){
            for(size_t i = 0; i < cpus.size(); i++){
                std::vector<mammut::topology::PhysicalCore*> phyCores = cpus.at(i)->getPhysicalCores();
                for(size_t j = 0; j < phyCores.size(); j++){
                    std::vector<mammut::topology::VirtualCore*> virtCores = phyCores.at(j)->getVirtualCores();
                    CList.push_back(virtCores[k]->getVirtualCoreId());
                }
            }
        }
        int nc;
        size = nc = num_cores = CList.size();
		// usually num_cores is a power of two....!
		if (!isPowerOf2(size)) {
			size = nextPowerOf2(size);
            for(size_t i =CList.size(), j = 0; i< size; ++i, j++)
                CList.push_back(CList[j]);            
        }        
#else
        const std::string ff_mapping_string = FF_MAPPING_STRING;
        if (ff_mapping_string.length()) {
            num_cores = setMappingList(ff_mapping_string.c_str());
            assert(isPowerOf2(CList.size()));
            size = CList.size();
        } else {
            int nc = ff_numCores();
            if (nc <= 0) {
                error("threadMapper: invalid num_cores\n");
                return;
            }
            size = num_cores = nc;
            CList.reserve(size);
            for (int i = 0; i < nc; ++i)
                CList.push_back(i);

            // usually num_cores is a power of two....!
            if (!isPowerOf2(size)) {
                size = nextPowerOf2(size);
                for(size_t i =CList.size(), j = 0; i< size; ++i, j++)
                    CList.push_back(CList[j]);       
            }
        }
#endif /* MAMMUT */

        mask = size - 1;
		rrcnt = 0;
        /*
          printf("CList:\n");
          for(size_t i =0 ; i < CList.size(); ++i) {
          printf("%ld ", CList[i]);
          }
          printf("\n");
        */


#if 0
		const int max_supported_platforms = 10;
		const int max_supported_devices = 10;
		cl_uint n_platforms;
		cl_platform_id platforms[max_supported_platforms];
		cl_device_id devices[max_supported_devices];
		cl_int status = clGetPlatformIDs(max_supported_platforms, platforms,
				&n_platforms); //TODO max 10 platforms
		checkResult(status, "clGetPlatformIDs");
		for (cl_uint i = 0; i < n_platforms; ++i) {
			cl_uint n_devices;
			//GPUs
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
					max_supported_devices, devices, &n_devices);
			//checkResult(status, "clGetDeviceIDs GPU");
			if(!status)
			for (cl_uint j = 0; j < n_devices; ++j)
				ocl_gpus.push_back(devices[j]);
			//CPUs
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU,
					max_supported_devices, devices, &n_devices);
			//checkResult(status, "clGetDeviceIDs CPU");
			if(!status)
			for (cl_uint j = 0; j < n_devices; ++j)
				ocl_cpus.push_back(devices[j]);
			//accelerators
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ACCELERATOR,
                                    max_supported_devices, devices, &n_devices);
			//checkResult(status, "clGetDeviceIDs Accelerators");
			if(!status)
                for (cl_uint j = 0; j < n_devices; ++j)
                    ocl_accelerators.push_back(devices[j]);
		}
		ocl_cpu_id = ocl_gpu_id = ocl_accelerator_id = 0;
#endif
	}

	/**
	 * It allows to set a new list of CPU ids.
	 *
	 * The str variable should contain a space-separated or a comma-separated
	 * list of CPU ids. For example if the string str is "0 1 1 2 3", then the
	 * first thread will be bind to CPU 0, the second to CPU 1, the third to
	 * CPU 1, the fourth to CPU 2, the fifth to CPU 3. Then it follows the same
	 * rule for the subsequent threads.
	 *
	 * \return -1 for errors, otherwise it returns the number of elements in str
     *
	 */    
	int setMappingList(const char* str) {
		rrcnt = 0;        // reset rrcnt

		if (str == NULL) return -1; // use the previous mapping list
		char* _str = const_cast<char*>(str), *_str_end;
		svector<int> List(64);
		do {
			while (*_str == ' ' || *_str == '\t' || *_str == ',')
				++_str;
			unsigned cpuid = strtoul(_str, &_str_end, 0);
			if (_str == _str_end) {
				error("setMapping, invalid mapping string\n");
				return -1;
			}
			if (cpuid > (num_cores - 1)) {
				error("setMapping, invalid cpu id in the mapping string\n");
				return -1;
			}
			_str = _str_end;
			List.push_back(cpuid);

			if (*_str == '\0')
				break;
		} while (1);

		unsigned int size = (unsigned int) List.size();
        int ret = size;
		if (!isPowerOf2(size)) {
			size = nextPowerOf2(size);
			List.reserve(size);
		}
		mask = size - 1;
		for (size_t i = List.size(), j = 0; i < size; ++i, j++)
			List.push_back(List[j]);
		CList = List;
        return ret;
	}

    void setMappingList(const std::vector<size_t> &mapping) {
		rrcnt = 0;        // reset rrcnt

		if ((mapping.size() > (mask+1)) || (mapping.size()==0)) {
            error("Invalid pinng vector: ignoring it\n");
			return; // use the previous mapping list
        }
		svector<int> List(mask + 1);
        for (size_t i=0; i<mapping.size(); ++i) {
			auto cpuid = mapping[i];
            if (cpuid > (num_cores - 1)) {
				error("setMapping, invalid cpu id in the mapping string\n");
				return;
            }
            List.push_back(cpuid);
        }  
        
		unsigned int size = (unsigned int) List.size();
		if (!isPowerOf2(size)) {
			size = nextPowerOf2(size);
			List.reserve(size);
		}
		mask = size - 1;
		for (size_t i = List.size(), j = 0; i < size; ++i, j++)
			List.push_back(List[j]);
		CList = List;
	}

    
	/**
	 *  Returns the next CPU id using a round-robin mapping access on the mapping list. 
	 *
	 *  \return The identifier of the core.
	 */
	int getCoreId() {
		assert(rrcnt >= 0);
		int id = CList[rrcnt++];
		rrcnt &= mask;
		return id;
	}

	/**
	 * It is used for debugging.
	 *
	 * \return TODO
	 */
	unsigned int getMask() {
		return mask;
	}

	/**
	 * It is used for debugging.
	 *
	 * \return TODO
	 */
	unsigned int getCListSize() {
		return (unsigned int) CList.size();
	}

	/**
	 * It is used to get the identifier of the core.
	 *
	 * \return The identifier of the core.
	 */
	ssize_t getCoreId(unsigned int tid) {
		ssize_t id = CList[tid & mask];
		//std::cerr << "Mask is " << mask << "\n";
		//int id = CList[tid % (mask+1)];
		return id;
	}

	/**
	 * It checks whether the taken core is within the range of the cores
	 * available on the machine.
	 *
	 * \return It will return either \p true of \p false.
	 */
	inline bool checkCPUId(const int cpuId) const {
		return ((unsigned) cpuId < num_cores);
	}

#if defined(FF_CUDA) 
	inline int getNumCUDADevices() const {
		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
		if (error_id != cudaSuccess) {
			error("getNumCUDADevices: cannot get the number of cuda devices\n");
			return -1;
		}
		return deviceCount;
	}
#endif

#if 0
	cl_device_id getOCLcpu() {
		cl_device_id res = ocl_cpus[(ocl_cpu_id++) % ocl_cpus.size()];
		char tmp[1024];
		clGetDeviceInfo(res, CL_DEVICE_NAME, 1024 * sizeof(char), tmp, NULL);
		std::cerr << "picked CPU device: " << tmp << std::endl;
		return res;
	}

	cl_device_id getOCLgpu() {
		cl_device_id res = ocl_gpus[(ocl_gpu_id++) % ocl_gpus.size()];
		char tmp[1024];
		clGetDeviceInfo(res, CL_DEVICE_NAME, 1024 * sizeof(char), tmp, NULL);
		std::cerr << "picked GPU device: " << tmp << std::endl;
		return res;
	}

	cl_device_id getOCLaccelerator() {
		cl_device_id res = ocl_accelerators[(ocl_accelerator_id++) % ocl_accelerators.size()];
		char tmp[1024];
		clGetDeviceInfo(res, CL_DEVICE_NAME, 1024 * sizeof(char), tmp, NULL);
		std::cerr << "picked Accelerator device: " << tmp << std::endl;
		return res;
	}
#endif

protected:
	long rrcnt;
	unsigned int mask;
	unsigned int num_cores;
	svector<int> CList;
#if 0
	svector<cl_device_id> ocl_cpus, ocl_gpus, ocl_accelerators;
	std::atomic<unsigned int> ocl_cpu_id, ocl_gpu_id, ocl_accelerator_id;
#endif
};

} // namespace ff

/*!
 *
 * @}
 * \link
 */

#endif /* __THREAD_MAPPER_HPP_ */
