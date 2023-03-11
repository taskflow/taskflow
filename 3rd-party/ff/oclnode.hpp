/* -*- Mode: C++; tab-width: 2; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file oclnode.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow OpenCL interface node
 *
 * This class bridges multicore with GPGPUs using OpenCL
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

/*  Mehdi Goli:            m.goli@rgu.ac.uk    goli.mehdi@gmail.com
 *  Massimo Torquati:      torquati@di.unipi.it
 *  Marco Aldinucci
 *
 */

#ifndef FF_OCLNODE_HPP
#define FF_OCLNODE_HPP

#include <ff/ocl/clEnvironment.hpp>
#include <ff/node.hpp>
#include <vector>

namespace ff{

/*!
 *  \class ff_oclNode
 *  \ingroup buiding_blocks
 *
 *  \brief OpenCL specialisation of the ff_node class
 *
 *  Implements the node that is serving as OpenCL device. In general there is one  ff_oclNode
 *  per OpenCL device. Anyway, there is a command queue per device. Concurrency for accessing 
 *  the command queue from different  ff_oclNode is managed by FF.
 *
 */
    
class ff_oclNode : public ff_node {
public:
/* cl_device_type - bitfield 
   #define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
   #define CL_DEVICE_TYPE_CPU                          (1 << 1)
   #define CL_DEVICE_TYPE_GPU                          (1 << 2)
   #define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
   #define CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
   #define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
*/
                              
    // returns the kind of node
    virtual fftype getFFType() const   { return OCL_WORKER; }

    void setDeviceId(cl_device_id id)  { deviceId = id; }
    void setDeviceType(cl_device_type dt = CL_DEVICE_TYPE_ALL) { dtype = dt; }

    cl_device_id   getDeviceId()  const  { return deviceId; }
    cl_device_type getDeviceType() const {return dtype;}
    
    int getOCLID() const { return oclId; }
    
protected:
    /**
     * \brief Constructor
     *
     * It construct the OpenCL node for the device.
     *
     */
    ff_oclNode():oclId(-1),deviceId(NULL),  dtype(CL_DEVICE_TYPE_ALL) {
        clEnvironment::instance();
    };
  
    ~ff_oclNode() { }
   
    int svc_init() {
        
        if (oclId < 0) oclId = clEnvironment::instance()->getOCLID();
        
        // the user has set a specific device
        if (deviceId != NULL) return 0;
        
        switch (dtype) {
        case CL_DEVICE_TYPE_ALL: {
            // no user choice, a static greedy algorithm is used to allocate openCL components
            ssize_t GPUdevId =clEnvironment::instance()->getGPUDeviceRR();
            if( (GPUdevId !=-1) && ( oclId < clEnvironment::instance()->getNumGPU())) { 
                printf("%d: Allocated a GPU device, the id is %ld\n", oclId, GPUdevId);
                deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                return 0;
            }
            // fall back to CPU either GPU has reached its max or there is no GPU available
            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
            if (CPUdevId != -1) {
                printf("%d: Allocated a CPU device as either no GPU device is available or no GPU slot is available (cpuId=%ld)\n",oclId, CPUdevId);
                deviceId=clEnvironment::instance()->getDevice(CPUdevId);
                return 0;
            }
            printf("%d: cannot allocate neither a GPU nor a CPU device\n", oclId);            
            return -1;
        } break;
        case CL_DEVICE_TYPE_GPU: {
            ssize_t GPUdevId =clEnvironment::instance()->getGPUDeviceRR();
            if( (GPUdevId !=-1) && ( oclId < clEnvironment::instance()->getNumGPU())) { 
                printf("%d: Allocated a GPU device, the id is %ld\n", oclId, GPUdevId);
                deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                return 0;
            }
            printf("%d: cannot allocate a GPU device\n", oclId);            
            return -1;
        } break;
        case CL_DEVICE_TYPE_CPU: {
            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
            if (CPUdevId != -1) {
                printf("%d: Allocated a CPU device (cpuId=%ld)\n",oclId, CPUdevId);
                deviceId=clEnvironment::instance()->getDevice(CPUdevId);        
                return 0;
            }
            printf("%d: cannot allocate a CPU device\n", oclId);
            return -1;
        } break;
        default : std::cerr << "Unknown/not supported device type\n";
            return -1;
        }
        return 0;
    } 
    
    void svc_end() {}
    
protected:    
    int               oclId;      // the OpenCL node id
    cl_device_id      deviceId;   // is the id which is provided for user
    cl_device_type    dtype;
};


/*!
 *  \class ff_oclNode_t
 *  \ingroup buiding_blocks
 *
 *  \brief OpenCL specialisation of the ff_node class (typed)
 *
 *
 */    
template<typename IN, typename OUT=IN>
struct ff_oclNode_t: ff_oclNode {
    typedef IN  in_type;
    typedef OUT out_type;
    ff_oclNode_t():
        GO_ON((OUT*)FF_GO_ON),
        EOS((OUT*)FF_EOS),
        GO_OUT((OUT*)FF_GO_OUT),
        EOS_NOFREEZE((OUT*)FF_EOS_NOFREEZE) {}
    OUT *GO_ON, *EOS, *GO_OUT, *EOS_NOFREEZE;
    virtual ~ff_oclNode_t()  {}
    virtual OUT* svc(IN*)=0;
    inline  void *svc(void *task) { return svc(reinterpret_cast<IN*>(task));};
};

}
#endif /* FF_OCLNODE_HPP */
