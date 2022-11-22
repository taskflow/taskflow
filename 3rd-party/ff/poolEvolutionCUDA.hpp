/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file poolEvolution.hpp
 *  \ingroup high_level_patterns_shared_memory
 *
 *  \brief This file describes the pool evolution pattern.
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

#ifndef FF_POOL_HPP
#define FF_POOL_HPP

#if !defined(FF_CUDA)
#define FF_CUDA
#endif

#include <vector>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/stencilReduceCUDA.hpp>

namespace ff {


template<typename T, typename kernelEvol>
class poolEvolutionCUDA : public ff_node {
public:
    typedef typename std::vector<T>::iterator       iter_t;
    typedef typename std::vector<T>::const_iterator const_iter_t;

    struct cudaEvolTask: public baseCUDATask<T> {
        void setTask(void *t) {
            cudaEvolTask *task=(cudaEvolTask *)t;
            this->setInPtr(task->buffer);
            this->setOutPtr(task->buffer);
            this->setSizeIn(task->size);
        }

        T      *buffer;
        size_t  size;
    };

protected:
    size_t pE,pF,pT,pS;
    std::vector<T>  *input;
    std::vector<T>                buffer;
    std::vector<std::vector<T> >  bufferPool;
    void (*selection)(const_iter_t start, const_iter_t stop, std::vector<T> &out);
    void (*filter)(const_iter_t start, const_iter_t stop, std::vector<T> &out);
    bool (*termination)(const std::vector<T> &pop); 
    ff_pipeline pipeevol;

    ff_mapCUDA<cudaEvolTask, kernelEvol> *mapEvol;

public :
    // constructor : to be used in non-streaming applications
    poolEvolutionCUDA (std::vector<T> & pop,                       // the initial population
                       void (*sel)(const_iter_t start, const_iter_t stop,
                                   std::vector<T> &out),           // the selection function
                       void (*fil)(const_iter_t start, const_iter_t stop,
                                   std::vector<T> &out),           // the filter function
                       bool (*term)(const std::vector<T> &pop))    // the termination function
        :pE(0),pF(1),pT(1),pS(1),input(&pop),selection(sel),filter(fil),termination(term),
         pipeevol(true),mapEvol(NULL) { 
        
        mapEvol = new ff_mapCUDA<cudaEvolTask,kernelEvol>;
        pipeevol.add_stage(mapEvol);
        if (pipeevol.run_then_freeze()<0) {
            error("poolEvolutionCUDA: running pipeevol\n");
            abort();
        }
        pipeevol.offload(GO_OUT);
        pipeevol.wait_freezing();
    
    }
    // constructor : to be used in streaming applications
    poolEvolutionCUDA (void (*sel)(const_iter_t start, const_iter_t stop,
                                   std::vector<T> &out),           // the selection function
                       void (*fil)(const_iter_t start, const_iter_t stop,
                                   std::vector<T> &out),           // the filter function
                       bool (*term)(const std::vector<T> &pop))    // the termination function
        :pE(0),pF(1),pT(1),pS(1),input(NULL),selection(sel),filter(fil),termination(term),
         pipeevol(true),mapEvol(NULL) { 
        mapEvol = new ff_mapCUDA<cudaEvolTask,kernelEvol>;
        pipeevol.add_stage(mapEvol);
        if (pipeevol.run_then_freeze()<0) {
            error("poolEvolutionCUDA: running pipeevol\n");
            abort();
        }
        pipeevol.offload(GO_OUT);
        pipeevol.wait_freezing();
    }
    
    ~poolEvolutionCUDA() {
        pipeevol.wait();
        if (mapEvol) delete mapEvol;
    }

    // the function returning the result in non streaming applications
    const std::vector<T>& get_result() { return *input; }
    
    void setParEvolution(size_t pardegree)   { pE = pardegree; }
    // currently the termination condition is computed sequentially
    void setParTermination(size_t )          { pT = 1; }
    void setParSelection(size_t pardegree)   { pS = pardegree; }
    void setParFilter (size_t pardegree)     { pF = pardegree; }
    
    
    int run_and_wait_end() {
        // TODO:
        // if (isfrozen()) {
        //     stop();
        //     thaw();
        //     if (wait()<0) return -1;
        //     return 0;
        // }
        // stop();
        if (ff_node::run()<0) return -1;
        if (ff_node::wait()<0) return -1;
        return 0;
    }

protected:

    void* svc(void * task) {
        if (task) input = ((std::vector<T>*)task);
        
        cudaEvolTask   evolTask;
        cudaEvolTask  *pevolTask = &evolTask;

        pipeevol.run_then_freeze();
        while(!termination(*input)) {
            // selection phase
            buffer.clear();
            selection(input->begin(),input->end(), buffer);

            // evolution phase
            evolTask.buffer = buffer.data();
            evolTask.size   = buffer.size();
            pipeevol.offload((void*)pevolTask);
            pipeevol.load_result((void**)&pevolTask);

            // filtering phase
            input->clear();
            filter(buffer.begin(),buffer.end(), *input);
        }
        if (task) ff_send_out(task);
        pipeevol.offload(GO_OUT);
        pipeevol.wait_freezing();
        return (task?GO_ON:NULL);
    }
};
    
} // namespace ff


#endif /* FF_POOL_HPP */
