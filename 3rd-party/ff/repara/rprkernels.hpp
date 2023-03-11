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
 * Authors: Massimo Torquati
 *          Rafael Sotomayor
 *
 * Date:    October   2015
 *          February  2016 (major improvements -- Massimo)
 *          May       2016 (minor improvements -- Massimo)
 *          July      2016 (fixing scope env variables -- Massimo)
 */

#ifndef REPARA_KERNELS_HPP
#define REPARA_KERNELS_HPP

// enables REPARA support in the FastFlow run-time
#if !defined(FF_REPARA)
#error "FF_REPARA not defined"
#endif
// defines FastFlow blocking mode
#if !defined(BLOCKING_MODE)
#error "BLOCKING_MODE not defined"
#endif


#include <vector>
#include <map>
#include <cassert>
#include <cstdlib>
#include <ff/node.hpp>
#include <ff/selector.hpp>
#include <ff/map.hpp>               // CPU map
#include <ff/stencilReduceOCL.hpp>  // GPU map
#include <ff/tpcnode.hpp>           // TPC devices
#include <ff/pipeline.hpp>          // pipeline
#include <ff/farm.hpp>              // task-ffarm 
#include <ff/repara/baseKernelTask.hpp>    // repara base task


#if defined(ENERGY_MEASUREMENT)

#include <Method.h>
#include <PicoScopeMethod.h>
#include <RaplMethod.h>
#include <iostream>

using namespace repara::measurement;
using namespace repara::measurement::scope;
using namespace repara::measurement::rapl;
#endif  // ENERGY_MEASUREMENT 

using namespace ff;

#if defined(DPE_PROTOCOL)
#define DPE(X) X
#include <ff/repara/dpe.hpp>
#else 
#define DPE(X)
#define NO_DPE_SCHEDULE 1
#endif

namespace repara {
namespace rprkernels {

// queue size between two pipeline stages
static const int DEFAULT_PIPELINE_QUEUE_SIZE = 8;

// FIFO path and its default value
static char *FIFO_FILE = nullptr;
static char DEFAULT_FIFO_FILE[]       = "/tmp/RMEASURE_FIFO";

// queue size between the scheduler and the workers
static const int DEFAULT_FARM_QUEUE_SIZE     = 2;

// logical target device number 
typedef  enum { 
    CPU=0, 
    GPU0=10, GPU1, GPU2, GPU3, GPU4,
    FPGA0=20, FPGA1, FPGA2, FPGA3, FPGA4,
    DSP0=30, DSP1, DSP2, DSP3, DSP4 
}  Kernel_Target_t;

static std::map<std::string, Kernel_Target_t> String2Target = { 
    {"GPU:0",GPU0},   {"GPU:1",GPU1},   {"GPU:2",GPU2},
    {"FPGA:0",FPGA0}, {"FPGA:1",FPGA1}, {"FPGA:2",FPGA2},
    {"DSP:0",DSP0},   {"DSP:1",DSP1},   {"DSP:2",DSP2}
};

/**
 *  This table contains the association between the logical device id
 *  (CPU:0, GPU:0, GPU:1, FPGA:0, DSP:0, DSP:1) and the real device id
 *  of the corresponding family of device.
 */
static std::map<std::string, size_t > device_table;


/**
 *  Global initialization.
 *  @param program_name the program name
 *  @return the maximum amount of measures that each kernel or pipeline
 *  may store at a given time.
 */    
static int Setup(const std::string &program_name) {        

#if defined(ENERGY_MEASUREMENT)
#ifdef SCOPE
    static const char DEFAULT_SCOPESERVICE[]    = "http://scopemachine:8080/RPC2";
    char *scopeservice = getenv("SCOPESERVICE");
    if (scopeservice == nullptr) {
        std::cerr << "SCOPESERVICE env variable not set, using default value (" << DEFAULT_SCOPESERVICE << ")\n";    
        setenv("SCOPESERVICE",DEFAULT_SCOPESERVICE, 1);
    } 
#endif
    static char DEFAULT_RMEASURESERVICE[] = "http://localhost:8081/RPC2";
    char *rmeasureservice = getenv("RMEASURESERVICE");
    if (rmeasureservice == nullptr) {
        std::cerr << "RMEASURESERVICE  env variable not set, using default value (" << DEFAULT_RMEASURESERVICE << ")\n";    
        setenv("RMEASURESERVICE",DEFAULT_RMEASURESERVICE, 1);
    }
    char* fifoFile = getenv ("RMEASURE_FIFO");
    if (fifoFile == nullptr) {
        std::cerr << "RMEASURE_FIFO  env variable not set, using default value (" << DEFAULT_FIFO_FILE << ")\n";    
        FIFO_FILE = DEFAULT_FIFO_FILE;
    } else FIFO_FILE = fifoFile;
#endif
    int r = 0;
    DPE(r = rpr::setup(program_name));
    return r;
}
/**
 *  Performs global cleanup before exiting.
 */
static void Cleanup() {
    DPE(rpr::cleanup());
}    
/**
 *  This function is used to get the schedule string from the DPE.
 *  @param kernel_id is the id of the REPARA kernel.
 *  @return the string containing the schedule for the kernels
 */
static inline const std::string schedule_kernel(int kernel_id, size_t problem_size = 0) {
#if defined(DPE_PROTOCOL)
    return rpr::schedule_kernel(kernel_id, problem_size);
#else   
    return std::string("");
#endif
}  
/**
 *  This function is used to get the schedule string from the DPE.
 *  @param pipeline_id is the id of the REPARA pipeline.
 *  @return the string containing the schedule for all the pipeline.
 */  
static inline const std::string schedule_pipeline(int pipeline_id, size_t problem_size = 0) {
#if defined(DPE_PROTOCOL)
    return rpr::schedule_pipeline(pipeline_id, problem_size);
#else
    return std::string("");
#endif
}

static inline void register_kernel(ff_node &node, int kernel_id) {
    DPE(rpr::register_kernel(node, kernel_id));
}
static inline void deregister_kernel(int kernel_id) {
    DPE(rpr::deregister_kernel(kernel_id));
}
static inline int register_pipeline(ff_node &node, int pipeline_id) {
#if defined(DPE_PROTOCOL)
    return rpr::register_pipeline(node, pipeline_id);
#else
    return DEFAULT_PIPELINE_QUEUE_SIZE; 
#endif
}
static inline void deregister_pipeline(int pipeline_id) {
    DPE(rpr::deregister_pipeline(pipeline_id));
}
static inline int register_farm(ff_node &node, int kernel_id) {
#if defined(DPE_PROTOCOL)
    //return rpr::register_farm(node, kernel_id);
    return 0;
#else
    return DEFAULT_FARM_QUEUE_SIZE;  
#endif
}
static inline void deregister_farm(int kernel_id) {
    //DPE(rpr::deregister_farm(kernel_id));
}
static inline int register_async(ff_node &node, int kernel_id) {
#if defined(DPE_PROTOCOL)
    //return rpr::register_async(node, kernel_id);
    return 0;
#else
    return DEFAULT_FARM_QUEUE_SIZE;  
#endif
}
static inline void deregister_async(int kernel_id) {
    //DPE(rpr::deregister_async(kernel_id));
}


/* ------------------  Utility functions ---------------------- */
static inline void print_Measures(const ff_node::RPR_measures_vector &V, const std::string &str) {
    std::cout << "Metrics " << str << "\n";
    const size_t entries = V.size();
    for(size_t i=0;i<entries;++i) {
        std::cout << i+1 << "/" << entries << ":\n";
        for(size_t m=0;m<V[i].size();++m) {
            for(size_t j=0;j<V[i][m].size(); ++j) {
                std::cout << "  " << "device=" << V[i][m][j].first << "\n";
                for(size_t k=0;k<V[i][m][j].second.size(); ++k) {
                    std::cout << "  " 
                              << V[i][m][j].second[k].time_before << "," 
                              << V[i][m][j].second[k].time_after << ", "  
                              << V[i][m][j].second[k].bytesIn << "," 
                              << V[i][m][j].second[k].bytesOut << ", "
                              << V[i][m][j].second[k].vmSize   << ","
                              << V[i][m][j].second[k].vmPeak   << ", "
                              << V[i][m][j].second[k].problemSize << ", "
                              << V[i][m][j].second[k].energy   << "\n";
                }
            }
        }
    }
    std::cout << "\n";
}

static inline void printMemoryFlags(const memoryflagsVector &mf, const std::string &str) {
    std::cout << str << "\n";
    for(size_t i=0;i<mf.size();++i) {
        std::cout << "param: " <<  i << "\n";
        std::cout << "    copy   :  " << (mf[i].copy==CopyFlags::COPY ? "COPY" : "DONTCOPY") << "\n";
        std::cout << "    reuse  :  " << (mf[i].reuse==ReuseFlags::REUSE ? "REUSE" : "DONTREUSE") << "\n";
        std::cout << "    release:  " << (mf[i].release==ReleaseFlags::RELEASE ? "RELEASE" : "DONTRELEASE") << "\n";
    }
    std::cout << "\n";
}

#if defined(ENERGY_MEASUREMENT)
static inline int sendFifoMsg(const char* msg, int isBegin) {
    if (access (FIFO_FILE, F_OK) == -1) {
        perror("access");
        return -1;
    }
    FILE *fp;
    fp = fopen(FIFO_FILE, "w");
    if (fp == NULL) {
        perror("fopen");
        return -1;
    }
    if (isBegin == 1) {
        char rKernelName[strlen(msg)+5];
        memset(rKernelName, 0, sizeof rKernelName);
        strcat(rKernelName,"B:\0");
        strcat(rKernelName, msg);
        strcat(rKernelName, ";\0");
        if (fputs(rKernelName, fp) == EOF) {
            perror("fputs");
            return -1;
        }
    } else {
            if (fputs(msg, fp) == EOF) {
                perror("fputs");
                return -1;
            }
    }
    fclose(fp);
    return 0;
}

static double getSourceResults(const Measurement& measurement) {
    double result = 0.0;
    
    // we have just one single kernel instance
    Measurement::KernelSourceMap::const_iterator kernelSourceIt = measurement.kernelSourceMap().begin();
    Measurement::SourceContainer::const_iterator resultsIt = kernelSourceIt->second.begin();
    const Measurement::SourceMap& sourceMap = *resultsIt;
    Measurement::SourceMap::const_iterator sourceIt = sourceMap.begin();
    for (; sourceIt != sourceMap.end(); ++sourceIt) {
        const Measurement::DataMap& dataMap = sourceIt->second;
        Measurement::DataMap::const_iterator dataIt = dataMap.begin();
        for (; dataIt != dataMap.end(); ++dataIt) {
            // we are interested in only for the Energy field
            if (dataIt->first == SourceCapability::Energy) {
                result +=dataIt->second;
                break;
            }
        }
    }       
    return result;
}


static inline void EnergyMeasure_Begin(const std::string &kernel) {
#if defined(SCOPE)
    PicoScopeMethod* scopeMethod = PicoScopeMethod::getInstance();
    //scopeMethod->setSampleRate(10, TIME_US);
    //scopeMethod->startMeasurement(scopeMethod->configuration());
    PicoScopeConfiguration scopeConfig;
    scopeConfig.setSampleRate(TIME_US, 10);
    scopeMethod->startMeasurement(scopeConfig);
#else
    RaplMethod *raplMethod = RaplMethod::getInstance();
    raplMethod->startMeasurement(raplMethod->configuration());
#endif
    sendFifoMsg(kernel.c_str(), 1);
}

static inline void EnergyMeasure_End(const std::string &kernel, double &energy) {
    sendFifoMsg("E;", 0);
    energy = 0.0;
#if defined(SCOPE)
    PicoScopeMeasurement* measurement = PicoScopeMethod::getInstance()->stopMeasurement();
#else
    RaplMeasurement* measurement = RaplMethod::getInstance()->stopMeasurement();
#endif
    if (measurement) 
        energy = getSourceResults(*measurement);
}

#endif // ENERGY_MEASUREMENT

/* ---------------------------------------------------------- */

template <typename T>
static inline void default_init_F(T &) {}

/**
 * REPARA stream generator kernel always running on the CPU host
 *
 */
template<typename T, 
         typename CONDF_t, 
         typename INCF_t, 
         typename INITF_t = void (*)(T&) >
class streamGen_Kernel: public ff_node_t<T> {
    using basenode = ff_node_t<T>;
public:

    /**
     * Constructor.
     * 
     * @param pipeline_id  id of the pipeline kernel
     * @condF function used to determine the end of stream, returns true if another task
     * has to be produced in output false if the EOS has to be generated.
     */
    streamGen_Kernel(int pipeline_id, 
                     CONDF_t condF, INCF_t incF, INITF_t initF = default_init_F<T>, 
                     const size_t init_value=0):
        pipeline_id(pipeline_id),init_value(init_value), 
        condF(condF),incF(incF),initF(initF) {
    }

protected:    
    T *svc(T*) {
        size_t idx=init_value;
        while(condF(idx)) {
            const std::string &cmd =schedule_pipeline(pipeline_id);
            T *out = new T(idx, cmd);
            initF(*out);            
            basenode::ff_send_out(out);
            incF(idx);
        }
        return basenode::EOS;
    }

    int      pipeline_id;
    size_t   init_value;
    CONDF_t  condF;   
    INCF_t   incF;
    INITF_t  initF;
};

template<typename T>
static size_t zeroProblemSize(const T&) { return 0;}

/**
 * REPARA sequential stage kernel. It may be used when the first stage of a pipeline 
 * is a task-farm or a map to bring "external data" into the pipeline.
 *
 */
template<typename T, typename F_t>
class fillTask_Kernel: public ff_node_t<T> {
    using basenode = ff_node_t<T>;
public:
    fillTask_Kernel(F_t F):F(F) {}
protected:    
    T *svc(T* in) {
        F(*in);
        return in;
    }
    F_t  F;   
};


/**
 * REPARA kernel interface
 *
 */
template<typename IN_t, typename OUT_t=IN_t>
class Kernel: public ff_nodeSelector<IN_t, OUT_t> {
    using selector = ff_nodeSelector<IN_t,OUT_t>;    
    using RPR_measures_vector = ff_node::RPR_measures_vector;

public:    
    /** 
     *  Constructor
     *  @param kernel_id it is the id of the kernel
     *
     *  \brief REPARA kernel constructor for implementing kernels in a pipeline.
     */ 
    Kernel(int kernel_id): kernel_id(kernel_id), Task(nullptr),
                           problemSize_F(zeroProblemSize<IN_t>) {
        pw.store(nullptr);
    }

    Kernel(int kernel_id, std::function<size_t(const IN_t&)> F): 
        kernel_id(kernel_id), Task(nullptr), problemSize_F(std::move(F)) {
        pw.store(nullptr);
    }

    /** 
     *  Constructor
     *  @param kernel_id it is the id of the kernel
     *  @param task      input task pointer, passed only if the kernel is not 
     *  a stage of a pipeline ("oneshot" mode)
     *
     *  \brief REPARA kernel constructor for implementing stand-alone kernels.
     */ 
    Kernel(int kernel_id, IN_t &task):
        kernel_id(kernel_id), Task(&task), problemSize_F(zeroProblemSize<IN_t>) {
        pw.store(nullptr);
    }

    Kernel(int kernel_id, IN_t &task, size_t (*F)(const IN_t&)): 
        kernel_id(kernel_id), Task(&task), problemSize_F(F) {
        pw.store(nullptr);
    }



    /** 
     *  Destructor.
     */
    virtual ~Kernel() {
        RPR_measures_vector *p = pw.load();
        if (p != nullptr) delete p;
    }
    /**
     *  Adds a new implementation for the kernel 
     *  @param node  the new implementation 
     *  @param type  the target device type
     */
    void addNode(ff_node &node, Kernel_Target_t type) {
        node.rpr_set_measure_energy(true);
        size_t id = selector::addNode(node);
        kernel2node[type] = id;
    }
    void addNode(std::unique_ptr<ff_node> &&node, Kernel_Target_t type) {
        node.get()->rpr_set_measure_energy(true);
        size_t id = selector::addNode(std::move(node));
        kernel2node[type] = id;
    }

    /**
     *  Starts the execution of the kernel.
     *  A thread executing the object instance is spawned.
     *
     *  @return 0 success, -1 otherwise
     */
    int run(bool = false) { return ff_node::run();  }
    /**
     * Waits for termination of the thread associated 
     * to the object.
     *
     * @return 0 success, -1 otherwise
     */   
    int wait() { return ff_node::wait(); }

    /**
     *  Starts the execution of the kernel and waits for its termination.
     *  No thread is spawned.
     *
     *  @return 0 success, -1 otherwise
     */
    int run_and_wait_end() {        
        register_kernel(*this, kernel_id);                       
        if (selector::run_and_wait_end()<0) {
            deregister_kernel(kernel_id);
            return -1;
        }
        deregister_kernel(kernel_id);
        return 0;
    }

    ssize_t get_my_id() const { return ff_node::get_my_id(); };
    
    /** 
     *  Atomically gets the internal measurements stored by the kernel.
     *  @return the array containing the data
     */
    RPR_measures_vector rpr_get_measures() {
        RPR_measures_vector *v = new RPR_measures_vector(1);
        (*v)[0].resize(1);
        (*v)[0][0].resize(selector::numNodes());
        RPR_measures_vector *p  = std::atomic_exchange(&pw, v);

        if (p == nullptr) { // maybe too early....
            return RPR_measures_vector();
        }
        RPR_measures_vector tmp = std::move(*p);
        delete p;
        return tmp;
    }

protected:
    /**
     *  Extracts the device-id and the logical target device from the 
     *  command string
     *  @param cmd command string
     *  @target logical target device
     *  @return device-id     
     *
     *  cmd format:
     *   schedule_id;$kernel_1;GPU:0; URF;SF;...;$kernel_2;CPU:0;.... ;$kernel_N; ...;$
     *  S: send to (COPY) 
     *  U: reUse (REUSE)
     *  R: receive from (COPY)
     *  F: free/remoove (RELEASE)
     */
    inline size_t getDeviceID(const std::string &cmd, Kernel_Target_t &target) {
        if (cmd == "") return 0;
        const std::string kid = "kernel_"+std::to_string(kernel_id);
        const char *semicolon = ";";
        size_t n = cmd.rfind(kid);
        assert(n != std::string::npos);
        n = cmd.find_first_of(semicolon, n);
        assert(n != std::string::npos);
        size_t m = cmd.find_first_of(semicolon, n+1);
        assert(m != std::string::npos);
        const std::string &device = cmd.substr(n+1, m-n-1);
        target = String2Target[device];
        return kernel2node[target];
    }
    inline size_t getScheduleID(const std::string &cmd) {
        if (cmd == "") return 0;
        size_t n = cmd.find_first_of(";");
        assert(n != std::string::npos);
        return std::stol(cmd.substr(0,n));
    }

    // called once at the very beginning
    int nodeInit() {
        RPR_measures_vector *p = pw.load();
        if (p != nullptr) delete p;
        RPR_measures_vector *c = new RPR_measures_vector(1);
        assert(c);
        (*c)[0].resize(1);
        (*c)[0][0].resize(selector::numNodes());
        pw.store(c);
        statusStr = "/proc/"+std::to_string(getpid())+"/status";
        return selector::nodeInit();
    }

    // TODO: leaks check
    void nodeEnd() {}

    OUT_t *svc(IN_t *in) {
        bool oneshot = false;
        size_t selectedDevice = 0;
        size_t problem_size = 0;
        Kernel_Target_t target;
        if (in == nullptr) {  // non-streaming kernel execution
            assert(Task != nullptr);
            oneshot = true;  
            problem_size = problemSize_F(*Task);
#if !defined(NO_DPE_SCHEDULE)
            const std::string &cmd =schedule_kernel(kernel_id, problem_size);
            Task->cmd = cmd;
#endif
            Task->kernel_id = kernel_id;
            selectedDevice = getDeviceID(Task->cmd, target);
            in = Task;
        } else {
            assert(Task == nullptr);
            problem_size = problemSize_F(*in);
            selectedDevice = getDeviceID(in->cmd, target);
            in->kernel_id = kernel_id;
        }
        selector::selectNode(selectedDevice);

        // this is the node that is going to be executed
        ff_node *node = selector::getNode(selectedDevice);

        ff_node::rpr_measure_t measure;
        size_t stop, start, pstop,pstart;
        measure.schedule_id = getScheduleID(in->cmd);
        measure.problemSize = problem_size;        
        memory_Stats(statusStr.c_str(), start, pstart);
#if defined(ENERGY_MEASUREMENT)
        bool measureEnergy = node->rpr_get_measure_energy();
        if (measureEnergy) EnergyMeasure_Begin("kernel");
#endif
        measure.time_before = getusec();
        OUT_t *out  = selector::svc(in);  // executes the kernel!
        measure.time_after  = getusec();
#if defined(ENERGY_MEASUREMENT)
        if (measureEnergy) EnergyMeasure_End("kernel",measure.energy);
        else measure.energy = 0.0;
#endif
        memory_Stats(statusStr.c_str(), stop, pstop);
        // gets input/output size and virtual memory 
        measure.bytesIn  = node->rpr_get_sizeIn();
        measure.bytesOut = node->rpr_get_sizeOut();
        measure.vmSize   = (stop>start)?(stop-start):0;
        measure.vmPeak   = (pstop>pstart)?(pstop-pstart):0;

        // stores the collected measures
        RPR_measures_vector *pw_vector = pw.load();
        (*pw_vector)[0][0][selectedDevice].first = target;
        (*pw_vector)[0][0][selectedDevice].second.push_back(measure);

        return oneshot?selector::EOS:out;
    }
    
private:
    const int kernel_id;
    IN_t     *Task;    
    std::string    statusStr;
    std::function<size_t(const IN_t&)>  problemSize_F;
    std::map<Kernel_Target_t, size_t>   kernel2node;   // multimap ???
    std::atomic<RPR_measures_vector *>  pw;
};


/**
 * REPARA pipeline interface
 *
 */
class Pipe_Kernel: public ff_Pipe<> {
    using pipe = ff_Pipe<>;

public:
    template<typename... STAGES>
    explicit Pipe_Kernel(const int kernel_id, STAGES &&...stages): 
        ff_Pipe<>(stages...), kernel_id(kernel_id) {
        //sets bounded queues 
        ff_Pipe<>::setFixedSize(true);

        // Energy is measured for the entire pipeline and not for the single kernel
        const svector<ff_node*> &nodes = getStages();
        for(size_t i=0;i<nodes.size();++i) 
            nodes[i]->rpr_set_measure_energy(false);
    }
    virtual ~Pipe_Kernel() {}

    int run_and_wait_end() {
        int queue_size = register_pipeline(*this, kernel_id);
        ff_Pipe<>::setXNodeInputQueueLength(queue_size);
        ff_Pipe<>::setXNodeOutputQueueLength(queue_size);

#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_Begin("kernel");
#endif
        if (pipe::run_and_wait_end()<0) {
            deregister_pipeline(kernel_id);
            return -1;
        }
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_End("kernel", energy);
#else
        energy = 0.0;
#endif
        deregister_pipeline(kernel_id);
        return 0;
    }
    
    /** 
     *  Atomically gets the internal measurements stored by the kernel.
     *
     */
    RPR_measures_vector rpr_get_measures() {
        const svector<ff_node*> &nodes = getStages();
        RPR_measures_vector measures;
        measures.reserve(nodes.size()+1);

        // energy is stored for the first stage (the Generator stage) which is 
        // always a sequential stage running on the CPU
        std::vector<RPR_devices_measure> tmp(1);
        tmp[0].resize(1);
        tmp[0][CPU].second.resize(1);
        (tmp[0][CPU].second)[0].energy = energy;

        (tmp[0][CPU].second)[0].time_before= getusec(ff_Pipe<>::startTime()); // stores pipeline starting time
        (tmp[0][CPU].second)[0].time_after = getusec(); // stores current time

        measures.push_back(tmp);

        for(size_t i=1;i<nodes.size();++i) {
            const RPR_measures_vector &V = nodes[i]->rpr_get_measures();
            measures.insert(measures.end(), V.begin(), V.end());
        }
        return measures;
    }
protected:
    const int kernel_id;
    double energy = 0.0;
};


/**
 * REPARA farm interface
 *
 */
class Farm_Kernel: public ff_Farm<> {
    using farm = ff_Farm<>;

public:
    explicit Farm_Kernel(const int farm_id, 
                         std::vector<std::unique_ptr<ff_node> > &&W,
                         std::unique_ptr<ff_node> E  =std::unique_ptr<ff_node>(nullptr), 
                         std::unique_ptr<ff_node> C  =std::unique_ptr<ff_node>(nullptr)):
        ff_Farm<>(std::move(W),std::move(E),std::move(C),false), farm_id(farm_id) {
        //sets bounded queues 
        ff_Farm<>::setFixedSize(true);        

        // Energy is measured for the entire farm and not for the single kernel
        const svector<ff_node*>& workers =  getWorkers();        
        for(size_t i=0;i<workers.size();++i) 
            workers[i]->rpr_set_measure_energy(false);
    }

    virtual ~Farm_Kernel() {}

    int run_and_wait_end() {
        int queue_size = register_farm(*this, farm_id);
        farm::set_scheduling_ondemand(queue_size);
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_Begin("kernel");
#endif
        if (farm::run_and_wait_end()<0) {
            deregister_farm(farm_id);
            return -1;
        }
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_End("kernel", energy);
#else
        energy = 0.0;
#endif
        deregister_farm(farm_id);
        return 0;
    }    
    /** 
     *  Atomically gets the internal measurements stored by the kernel.
     *
     */
    RPR_measures_vector rpr_get_measures() {
        const svector<ff_node*>& workers =  getWorkers();        

        RPR_measures_vector measures;
        measures.reserve(workers.size()+1);

        std::vector<RPR_devices_measure> tmp(1);
        tmp[0].resize(1);
        tmp[0][CPU].second.resize(1);
        (tmp[0][CPU].second)[0].energy = energy;
        measures.push_back(tmp);

        for(size_t i=0;i<workers.size();++i) {
            const RPR_measures_vector &V = workers[i]->rpr_get_measures();
            measures.insert(measures.end(), V.begin(), V.end());
        }

        return measures;
    }
protected:
    const int farm_id; 
    double energy = 0.0;
};


/**
 * REPARA ordered farm interface
 *
 */
class OFarm_Kernel: public ff_OFarm<> {
    using farm = ff_OFarm<>;
    std::unique_ptr<ff_node>               EmitterF;
    std::unique_ptr<ff_node>               CollectorF;
public:
    explicit OFarm_Kernel(const int farm_id, 
                          std::vector<std::unique_ptr<ff_node> > &&W,
                          std::unique_ptr<ff_node> E  =std::unique_ptr<ff_node>(nullptr), 
                          std::unique_ptr<ff_node> C  =std::unique_ptr<ff_node>(nullptr)):
        ff_OFarm<>(std::move(W),false), EmitterF(std::move(E)),CollectorF(std::move(C)), farm_id(farm_id) {
        //sets bounded queues 
        ff_OFarm<>::setFixedSize(true);        

        ff_node *e = EmitterF.get();
        if (e) farm::setEmitterF(*e);
        ff_node *c = CollectorF.get();
        if (c) farm::setEmitterF(*c);

        // Energy is measured for the entire farm and not for the single kernel
        const svector<ff_node*>& workers =  getWorkers();        
        for(size_t i=0;i<workers.size();++i) 
            workers[i]->rpr_set_measure_energy(false);
    }

    virtual ~OFarm_Kernel() {}

    int run_and_wait_end() {
        const size_t nw = farm::getNWorkers();
        int queue_size = register_farm(*this, farm_id);
        farm::setInputQueueLength(queue_size*nw);
        farm::setOutputQueueLength(queue_size*nw);
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_Begin("kernel");
#endif
        if (farm::run_and_wait_end()<0) {
            deregister_farm(farm_id);
            return -1;
        }
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_End("kernel", energy);
#else
        energy = 0.0;
#endif
        deregister_farm(farm_id);
        return 0;
    }    
    /** 
     *  Atomically gets the internal measurements stored by the kernel.
     *
     */
    RPR_measures_vector rpr_get_measures() {
        const svector<ff_node*>& workers =  getWorkers();        

        RPR_measures_vector measures;
        measures.reserve(workers.size()+1);

        std::vector<RPR_devices_measure> tmp(1);
        tmp[0].resize(1);
        tmp[0][CPU].second.resize(1);
        (tmp[0][CPU].second)[0].energy = energy;
        measures.push_back(tmp);

        for(size_t i=0;i<workers.size();++i) {
            const RPR_measures_vector &V = workers[i]->rpr_get_measures();
            measures.insert(measures.end(), V.begin(), V.end());
        }

        return measures;
    }
protected:
    const int farm_id; 
    double energy = 0.0;
};


#if 0
/**
 * REPARA async/sync kernel interface
 *
 */
class Async_Kernel: public ff_taskf {
    using async = ff_taskf;

public:
    explicit Async_Kernel(const int kernel_id): 
        ff_taskf(), kernel_id(kernel_id) {
    }

    virtual ~Async_Kernel() {}

    template<typename F_t, typename... Param>
    void AddTask(const F_t F, Param... args);

    /**
     *  Starts the execution of the kernels added.
     */
    int run();

    /**
     * sync
     */   
    int wait();


    /**
     *  run + wait
     */
    int run_and_wait_end() {
        int queue_size = register_async(*this, kernel_id);
        //sets bounded queues 
        async::setFixedSize(true);        
        async::setXNodeInputQueueLength(queue_size);
        async::setXNodeOutputQueueLength(queue_size);

#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_Begin("kernel");
#endif
        if (pipe::run_and_wait_end()<0) {
            deregister_async(kernel_id);
            return -1;
        }
#if defined(ENERGY_MEASUREMENT)
        EnergyMeasure_End("kernel", energy);
#else
        energy = 0.0;
#endif
        deregister_async(kernel_id);
        return 0;
    }
    
    /** 
     *  Atomically gets the internal measurements stored by the kernel.
     *
     */
    RPR_measures_vector rpr_get_measures() {
        RPR_measures_vector measures;
        return measures;
    }
protected:
    const int kernel_id;
    double energy = 0.0;
};
#endif  // if 0



}; // namespace rprkernels
}; // namespace repara
#endif /* REPARA_KERNELS_HPP */
