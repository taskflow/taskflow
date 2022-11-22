/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file clEnvironment.hpp
 *  \ingroup aux_classes
 *
 *  \brief This file includes the bsic support for OpenCL platforms
 *
 *  Realises a singleton class that keep the status of the OpenCL platform
 *  creates contexts, command queues etc.
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

/*
 * Mehdi Goli:         m.goli@rgu.ac.uk  goli.mehdi@gmail.com
 * Massimo Torquati:   torquati@di.unipi.it
 * Marco Aldinucci:    aldinuc@di.unito.it
 *
 */

#ifndef FF_OCLENVIRONMENT_HPP
#define FF_OCLENVIRONMENT_HPP

#if defined(FF_OPENCL)

// to avoid deprecated warnings
#if !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS 1
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 1
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <pthread.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>   // FIX: check if it is possible to remove this include
#include <sstream>
#include <map>
#include <vector>
//#include <ff/atomic/atomic.h>
#include <atomic>

namespace ff {

static pthread_mutex_t instanceMutex = PTHREAD_MUTEX_INITIALIZER;

struct oclParameter {
    oclParameter(cl_device_id d_id):d_id(d_id){}
    cl_device_id d_id;
    cl_context context;
    cl_command_queue commandQueue;    
};


/*!
 *  \class clEnvironment
 *  \ingroup aux_classes
 *
 *  \brief OpenCL platform inspection and setup
 *
 * \note Multiple paltforms are not managed. Platforms[0] is always adopted. Support for multiple 
 * platforms will be implemented if needed.
 *
 */

class clEnvironment {
private:
    cl_platform_id *platforms;
    cl_uint numPlatforms;
    cl_uint numDevices;
    //cl_device_id* devlist_for_platform;
    cl_device_id* deviceIds;
    
protected:
    clEnvironment(): platforms(NULL), numPlatforms(0),lastAssigned(0) {
        oclId=0;
        
        // FIX: what is this ???
#if defined(FF_GPUCOMPONETS)
        numGPU=FF_GPUCOMPONETS;
#else
        numGPU=10000;
#endif       
        clGetPlatformIDs(0, NULL, &numPlatforms);
        assert(numPlatforms>0);
        platforms = new cl_platform_id[numPlatforms]; 
        assert(platforms);
        clGetPlatformIDs(numPlatforms, platforms, NULL);
        
#ifdef FF_OPENCL_LOG
        if (numPlatforms>1) {
            printf("Multiple OpenCL platforms detected. Experimental code\n");
        }
#endif

        for (unsigned int i = 0; i< numPlatforms; ++i) {
            clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_ALL,0,NULL,&(numDevices));
            deviceIds = new cl_device_id[numDevices];  
            assert(deviceIds); 
            // Fill in CLDevice with clGetDeviceIDs()            
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,numDevices,deviceIds,NULL);
            //std::cerr << "OpenCL platform detection - begin\n";
            for(size_t j=0; j<numDevices; j++)   {
                // estimating max number of thread per device 
                cl_bool b;
                cl_context context;
                cl_int status;
                cl_device_type dt;       
                
                clGetDeviceInfo(deviceIds[j], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &(b), NULL);
                context = clCreateContext(NULL,1,&deviceIds[j],NULL,NULL,&status);
                clGetDeviceInfo(deviceIds[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &(dt), NULL);
                
                // if((dt) & CL_DEVICE_TYPE_GPU)
                //     std::cerr << "#" << j << " CPU device\n";
                // else if((dt) & CL_DEVICE_TYPE_CPU)
                //     std::cerr << "#" << j << " GPU device\n";
                // else std::cerr << "#" << j << " Other device (not yet implemented)\n";
                
                if((b & CL_TRUE) && (status == CL_SUCCESS)) {
                    clDeviceInUse.push_back(false);
                    clDevices.push_back(deviceIds[j]);
                }
                clReleaseContext(context);
            }
        }
        delete [] deviceIds;
        delete [] platforms;
        //std::cerr << "OpenCL platform detection - end \n";
        
        // prepare per device parameters: context and command queue
        for(std::vector<cl_device_id>::iterator iter=clDevices.begin(); iter < clDevices.end(); ++iter) {
            cl_device_id dId = *iter;
            oclParameter* oclParams = new oclParameter(dId);
            assert(oclParams);
            cl_int status;
            oclParams->context = clCreateContext(NULL,1,&dId,NULL,NULL,&status);
            
            cl_command_queue_properties prop = 0;
            oclParams->commandQueue = clCreateCommandQueue(oclParams->context, dId, prop, &status);
            
            dynamicParameters[dId]=oclParams;
        }
    }
 
public:
    ~clEnvironment() {
    }
   
    static inline clEnvironment * instance() {
        while (!m_clEnvironment) {
            //std::cerr << "clEnvironment instance\n";
            pthread_mutex_lock(&instanceMutex);
            if (!m_clEnvironment) {
                m_clEnvironment = new clEnvironment();
                //std::cerr << "clEnvironment instance\n";
            }
            assert(m_clEnvironment);
            pthread_mutex_unlock(&instanceMutex);
         }
         return m_clEnvironment; 
    }

    unsigned long getOCLID() {  return ++oclId; }

    /**
     * allocate multiple GPU devices.
     * Return a list of allocated GPU devices,
     * picked from round-robin scan of the device list
     *
     * @param n is the number of GPU devices to be allocated
     * @param preferred_dev is the logical-indexed starting device of the round-robin scan (ignored if <0)
     * @param exclusive if true, do not consider devices already allocated
     * @param identical TODO
     * @return the vector of the logical-indexed allocated GPU devices.
     * If allocation request cannot be fulfilled,
     * an empty vector is returned
     */
    std::vector<ssize_t> coAllocateGPUDeviceRR(size_t n=1, ssize_t preferred_dev=-1, bool exclusive=false, bool identical=false) {
        cl_device_type dt;
        size_t count = n;
        std::vector<ssize_t> ret;
        pthread_mutex_lock(&instanceMutex);
        //start from either the user-defined preferred_dev or the last RR-allocated device
        size_t dev = (preferred_dev>=0)? (preferred_dev%clDevices.size()): lastAssigned;
        //perform multiple passes over the device list,
        //stop if no allocation happens in one pass
        size_t count_pre = count;
        while (true) {
			count_pre = count;
			for (size_t i = 0; i < clDevices.size(); i++) {
				clGetDeviceInfo(clDevices[dev], CL_DEVICE_TYPE,
						sizeof(cl_device_type), &(dt), NULL);
				if ((!clDeviceInUse[dev] | !exclusive) //dev is free or not exclusive mode
						&& ((dt) & CL_DEVICE_TYPE_GPU)) { //dev is a GPU
					ret.push_back(dev);
					if (--count == 0)
						break;
				}
				++dev;
				dev %= clDevices.size();
			} //end pass
			if(!count) { // commit
				// TODO check if identical
				for (size_t i=0; i<ret.size();++i)
					clDeviceInUse[ret[i]]=true;
				lastAssigned=dev;
				break;
			}
			if(count_pre == count) { // roll back
				//std::cerr << "Not enough GPUs: aborting\n";
				ret.clear();
				break;
			}
			//continue to next pass
		}
        pthread_mutex_unlock(&instanceMutex);
        return ret;
    }
   
    
    ssize_t getGPUDeviceRR(bool exclusive=false) {
        std::vector<ssize_t> r = coAllocateGPUDeviceRR(1, false);
        if (r.size()>0) return r[0];
        else return -1;
    }

    ssize_t getGPUDevice() { return getGPUDeviceRR(); }

    ssize_t getCPUDevice(bool exclusive=false) {
        cl_device_type dt;
        ssize_t ret=-1;
        pthread_mutex_lock(&instanceMutex);
        for(size_t i=0; i<clDevices.size(); i++) {
            clGetDeviceInfo(clDevices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &(dt), NULL);
            if ((!clDeviceInUse[i] | !exclusive) && ((dt) & CL_DEVICE_TYPE_CPU)) {
                clDeviceInUse[i]=true;
//                char buf[128];
//                clGetDeviceInfo(clDevices[i], CL_DEVICE_NAME, 128, buf, NULL);
//                std::cerr << "clEnvironment: assigned CPU "<< i << " " << buf << "\n";
                ret=i;
                break;
            }
        }
        pthread_mutex_unlock(&instanceMutex);
        if (ret==-1)  std::cerr << "CPU not available or in exclusive use: aborting\n";
        return ret;
    }

    void releaseDevice(ssize_t id) {
        std::cerr << "Not yet implemented\n";
    }
    
    
    std::vector<ssize_t> getAllGPUDevices() {
        cl_device_type dt;
        std::vector<ssize_t> ret;
        for(size_t i=0; i<clDevices.size(); i++) {
            clGetDeviceInfo(clDevices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &(dt), NULL);
            if((dt) & CL_DEVICE_TYPE_GPU)
                ret.push_back(i);
        }
        return ret;
    }


    int getNumGPU() const { return numGPU; }

    inline cl_device_id getDevice(size_t id) const { return clDevices[id]; }
    
    oclParameter *getParameter(cl_device_id id) { return dynamicParameters[id]; }
     
    std::vector<std::string> getDevicesInfo( ) {
        std::vector<std::string> res;
        //fprintf(stdout, "%d\n", numDevices);
        for(size_t j = 0; j < clDevices.size(); j++) {
            /*
            char buf[128];
            std::string s1, s2;
            clGetDeviceInfo(clDevices[j], CL_DEVICE_NAME, 128, buf, NULL);
            //fprintf(stdout, "Device %s supports ", buf);
            s1 = std::string(buf);
            clGetDeviceInfo(clDevices[j], CL_DEVICE_VERSION, 128, buf, NULL);
            //fprintf(stdout, "%s\n", buf);
            s2 = std::string(buf);
            size_t max_workgroup_size = 0;
            clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                &max_workgroup_size, NULL);
            std::stringstream s3;
            s3 << max_workgroup_size;
            res.push_back(s1+" "+s2 + "MAX Work Group size " + s3.str());
            */
            res.push_back(getDeviceInfo(clDevices[j]));
        }
        return res;
    }
    
    
    std::string getDeviceInfo(cl_device_id dev) {
        char buf[128];
        std::string s1, s2;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 128, buf, NULL);
        s1 = std::string(buf);
        clGetDeviceInfo(dev, CL_DEVICE_VERSION, 128, buf, NULL);
        s2 = std::string(buf);
        size_t max_workgroup_size = 0;
        clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                            &max_workgroup_size, NULL);
        std::stringstream s3;
        s3 << max_workgroup_size;
        std::string res;
        res = s1+" "+s2 + "Max-WorkGroup-size " + s3.str();
        return res;
    }
    
    
    
private:
    clEnvironment(clEnvironment const&){};
    clEnvironment& operator=(clEnvironment const&){ return *this;};
private:    
    static clEnvironment * m_clEnvironment;
    std::atomic_long oclId;

    std::map<cl_device_id, oclParameter*> dynamicParameters;
	std::vector<cl_device_id> clDevices;
    std::vector<bool> clDeviceInUse;
    size_t lastAssigned;
    //std::vector<bool> clDEviceBusy;
	int numGPU;
};

clEnvironment* clEnvironment::m_clEnvironment = NULL;

static inline void printOCLErrorString(cl_int error, std::ostream & out) {
	switch (error) {
	case CL_SUCCESS:
		out << "CL_SUCCESS" << std::endl;
		break;
	case CL_DEVICE_NOT_FOUND:
		out << "CL_DEVICE_NOT_FOUND" << std::endl;
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		out << "CL_DEVICE_NOT_AVAILABLE" << std::endl;
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		out << "CL_COMPILER_NOT_AVAILABLE" << std::endl;
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		out << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << std::endl;
		break;
	case CL_OUT_OF_RESOURCES:
		out << "CL_OUT_OF_RESOURCES" << std::endl;
		break;
	case CL_OUT_OF_HOST_MEMORY:
		out << "CL_OUT_OF_HOST_MEMORY" << std::endl;
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		out << "CL_PROFILING_INFO_NOT_AVAILABLE" << std::endl;
		break;
	case CL_MEM_COPY_OVERLAP:
		out << "CL_MEM_COPY_OVERLAP" << std::endl;
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		out << "CL_IMAGE_FORMAT_MISMATCH" << std::endl;
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		out << "CL_IMAGE_FORMAT_NOT_SUPPORTED" << std::endl;
		break;
	case CL_BUILD_PROGRAM_FAILURE:
		out << "CL_BUILD_PROGRAM_FAILURE" << std::endl;
		break;
	case CL_MAP_FAILURE:
		out << "CL_MAP_FAILURE" << std::endl;
		break;
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		out << "CL_MISALIGNED_SUB_BUFFER_OFFSET" << std::endl;
		break;
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		out << "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" << std::endl;
		break;
	case CL_INVALID_VALUE:
		out << "CL_INVALID_VALUE" << std::endl;
		break;
	case CL_INVALID_DEVICE_TYPE:
		out << "CL_INVALID_DEVICE_TYPE" << std::endl;
		break;
	case CL_INVALID_PLATFORM:
		out << "CL_INVALID_PLATFORM" << std::endl;
		break;
	case CL_INVALID_DEVICE:
		out << "CL_INVALID_DEVICE" << std::endl;
		break;
	case CL_INVALID_CONTEXT:
		out << "CL_INVALID_CONTEXT" << std::endl;
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		out << "CL_INVALID_QUEUE_PROPERTIES" << std::endl;
		break;
	case CL_INVALID_COMMAND_QUEUE:
		out << "CL_INVALID_COMMAND_QUEUE" << std::endl;
		break;
	case CL_INVALID_HOST_PTR:
		out << "CL_INVALID_HOST_PTR" << std::endl;
		break;
	case CL_INVALID_MEM_OBJECT:
		out << "CL_INVALID_MEM_OBJECT" << std::endl;
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		out << "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" << std::endl;
		break;
	case CL_INVALID_IMAGE_SIZE:
		out << "CL_INVALID_IMAGE_SIZE" << std::endl;
		break;
	case CL_INVALID_SAMPLER:
		out << "CL_INVALID_SAMPLER" << std::endl;
		break;
	case CL_INVALID_BINARY:
		out << "CL_INVALID_BINARY" << std::endl;
		break;
	case CL_INVALID_BUILD_OPTIONS:
		out << "CL_INVALID_BUILD_OPTIONS" << std::endl;
		break;
	case CL_INVALID_PROGRAM:
		out << "CL_INVALID_PROGRAM" << std::endl;
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		out << "CL_INVALID_PROGRAM_EXECUTABLE" << std::endl;
		break;
	case CL_INVALID_KERNEL_NAME:
		out << "CL_INVALID_KERNEL_NAME" << std::endl;
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		out << "CL_INVALID_KERNEL_DEFINITION" << std::endl;
		break;
	case CL_INVALID_KERNEL:
		out << "CL_INVALID_KERNEL" << std::endl;
		break;
	case CL_INVALID_ARG_INDEX:
		out << "CL_INVALID_ARG_INDEX" << std::endl;
		break;
	case CL_INVALID_ARG_VALUE:
		out << "CL_INVALID_ARG_VALUE" << std::endl;
		break;
	case CL_INVALID_ARG_SIZE:
		out << "CL_INVALID_ARG_SIZE" << std::endl;
		break;
	case CL_INVALID_KERNEL_ARGS:
		out << "CL_INVALID_KERNEL_ARGS" << std::endl;
		break;
	case CL_INVALID_WORK_DIMENSION:
		out << "CL_INVALID_WORK_DIMENSION" << std::endl;
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		out << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		out << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		out << "CL_INVALID_GLOBAL_OFFSET" << std::endl;
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		out << "CL_INVALID_EVENT_WAIT_LIST" << std::endl;
		break;
	case CL_INVALID_EVENT:
		out << "CL_INVALID_EVENT" << std::endl;
		break;
	case CL_INVALID_OPERATION:
		out << "CL_INVALID_OPERATION" << std::endl;
		break;
	case CL_INVALID_GL_OBJECT:
		out << "CL_INVALID_GL_OBJECT" << std::endl;
		break;
	case CL_INVALID_BUFFER_SIZE:
		out << "CL_INVALID_BUFFER_SIZE" << std::endl;
		break;
	case CL_INVALID_MIP_LEVEL:
		out << "CL_INVALID_MIP_LEVEL" << std::endl;
		break;
	case CL_INVALID_GLOBAL_WORK_SIZE:
		out << "CL_INVALID_GLOBAL_WORK_SIZE" << std::endl;
		break;
	case CL_INVALID_PROPERTY:
		out << "CL_INVALID_PROPERTY" << std::endl;
		break;
	default:
		out << "Unknown OpenCL error " << error << std::endl;
	}
}

static  inline bool checkResult(cl_int s, const char* msg) {
    if(s != CL_SUCCESS) {
        std::cerr << msg << ":";
        printOCLErrorString(s,std::cerr);
        return (false);
        // Not Ok
    }
    return (true);
    // Ok
}    

} // namespace

#else

namespace ff {
class clEnvironment{
private:
    clEnvironment() {}
public:
    static inline clEnvironment * instance() { return NULL; }
};
} // namespace
#endif /* FASTFLOW_OPENCL */

#endif /* FF_OCLENVIRONMENT_HPP */
