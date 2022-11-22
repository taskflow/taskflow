/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file poolEvolution.hpp
 *  \ingroup high_level_patterns
 *
 *
 *  \brief  The PoolEvolution pattern models the evolution of a given population. 
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

/*
 * The PoolEvolution pattern models the evolution of a population. 
 * In the pattern, a “candidate selection” function (s) selects a subset of objects belonging 
 * to an unstructured object pool (P). 
 * The selected objects are processed by means of an “evolution” function (e). 
 * The evolution function may produce any number of new/modified  objects out of the input one. 
 * The set of objects computed by the evolution function on the selected object are filtered 
 * through a “filter” function (f) and eventually inserted into the object pool. 
 * At any insertion/extraction into/from the object pool a “termination” function (t) is 
 * evaluated on the object pool, to determine whether the evolution process has to be stopped or 
 * continued for further iterations.
 * A pool evolution pattern therefore computes P as result of the following algorithm:
 *
 *  while not( t(P) ) do
 *    N  = e ( s(P) )
 *    P += f (N, P)
 *  end while
 *
 */

// TODO:
// reuse the pool pattern multiple times in a non streaming fashion

#ifndef FF_POOL_HPP
#define FF_POOL_HPP

#include <iosfwd>
#include <vector>
#include <ff/node.hpp>
#include <ff/parallel_for.hpp>

namespace ff {

/*! 
  * \class poolEvolution
  * \ingroup high_level_patterns
  * 
  * \brief The pool evolution parallel pattern.
  *
  * The pool pattern computes the set P as result of the following algorithm:
  *
  *  while not( t(P) ) do
  *    N  = e ( s(P) )
  *    P += f (N, P)
  *  end while
  * 
  * where 's' is a “candidate selection” function, which selects a subset of objects belonging 
  * to an unstructured object pool (P), 'e' is the "evolution" function, 'f' a "filter" function
  * and 't' a "termination" function.
  *
  * \example funcmin.cpp
  */ 
template<typename T, typename env_t=char>
class poolEvolution : public ff_node {
public:

    typedef void     (*selection_t)  (ParallelForReduce<T> &, std::vector<T> &, std::vector<T> &, env_t &);
    typedef const T& (*evolution_t)  (T&, const env_t&, const int); 
    typedef void     (*filtering_t)  (ParallelForReduce<T> &, std::vector<T> &, std::vector<T> &, env_t &);
    typedef bool     (*termination_t)(const std::vector<T> &pop, env_t &);

    typedef env_t envT;

protected:
    size_t maxp,pE;
    env_t  env;
    std::vector<T>               *input;
    std::vector<T>                buffer;

    selection_t   selection;
    evolution_t   evolution;
    filtering_t   filter;
    termination_t termination;

    ParallelForReduce<T> loopevol;

public :

    /* selection_t is the selection function type, it takes the popolution and returns a sub-population 
     * evolution_t is the evolution function type, it works on the single element 
     * filtering_t is the filter function type, it takes the population produced at the previous step and 
     * produces a new population 
     */

    // constructor : to be used in non-streaming applications
    poolEvolution (size_t maxp,                                // maximum parallelism degree 
                   std::vector<T> & pop,                       // the initial population
                   selection_t sel                 ,           // the selection function
                   evolution_t evol,                           // the evolution function
                   filtering_t fil,                            // the filter function
                   termination_t term,                         // the termination function
                   const env_t &E= env_t(), bool spinWait=true) // NOTE: spinWait does not enable spinBarrier !
        :maxp(maxp), pE(maxp),env(E),input(&pop),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp,spinWait) { 
        loopevol.disableScheduler(true);
    }
    // constructor : to be used in streaming applications
    poolEvolution (size_t maxp,                                // maximum parallelism degree 
                   selection_t sel                 ,           // the selection function
                   evolution_t evol,                           // the evolution function
                   filtering_t fil,                            // the filter function
                   termination_t term,                         // the termination function
                   const env_t &E= env_t(), bool spinWait=true)
        :maxp(maxp), pE(maxp),env(E),input(NULL),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp, spinWait) { 
        loopevol.disableScheduler(true);
    }
    
    // the function returning the result in non streaming applications
    const std::vector<T>& get_result() const { return *input; }

    // changing the parallelism degree
    void setParEvolution(size_t pardegree)   { 
        if (pardegree>maxp)
            error("setParEvolution: pardegree too high, it should be less than or equal to %ld\n",maxp);
        else pE = pardegree;        
    }

    const env_t& getEnv() const { return env;}

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

        while(!termination(*input,env)) {
            // selection phase
            buffer.clear();            
            selection(loopevol, *input, buffer, env);
            
            // evolution phase
            auto E = [&](const long i, const int thid) {
                buffer[i]=evolution(buffer[i], env, thid); 
            };
            // TODO: to add dynamic scheduling option
            loopevol.parallel_for_thid(0,buffer.size(),1,
                                       PARFOR_STATIC(0),E, pE); 
            
            // filtering phase
            filter(loopevol, *input, buffer, env);

            input->swap(buffer);
        }
        loopevol.threadPause();
        return (task?input:NULL);
    }    
};
    
} // namespace ff


#endif /* FF_POOL_HPP */
