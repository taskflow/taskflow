/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file ordering_policy.hpp
 *  \ingroup building_blocks
 *  \brief Implements ordering policy
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
 *   Author:
 *      Massimo Torquati
 */

#ifndef FF_ORDERING_POLICY_HPP
#define FF_ORDERING_POLICY_HPP

#include <vector>
#include <queue>

#include <ff/lb.hpp>
#include <ff/gt.hpp>
#include <ff/node.hpp>

namespace ff {


// -------- ondemand load balancer and gatherer -----------------
// first is the unique id, second.first is the data element
// second.second is used to store the sender
using ordering_pair_t = std::pair<size_t, std::pair<void*,ssize_t> >;

struct ordered_lb:ff_loadbalancer {
    ordered_lb(int max_num_workers):ff_loadbalancer(max_num_workers) {}
    void init(ordering_pair_t* v, const size_t size) {
        _M=v; _M_size=size; cnt=0; idx=0;
    }
    inline bool schedule_task(void * task, unsigned long retry, unsigned long ticks) {
        _M[idx].first  = cnt;
        _M[idx].second.first = task;
        auto r = ff_loadbalancer::schedule_task(&_M[idx], retry, ticks);
        assert(r);
        ++cnt; ++idx %= _M_size;
        return r;
    }
    inline void broadcast_task(void * task) {
        if (task > FF_TAG_MIN) {
            ff_loadbalancer::broadcast_task(task);
            return;
        }
        _M[idx].first  = cnt;
        _M[idx].second.first = task;
        ff_loadbalancer::broadcast_task(&_M[idx]);
        ++cnt; ++idx %= _M_size;
    }
    inline bool ff_send_out_to(void *task, int id, unsigned long retry, unsigned long ticks) {
        assert(task<FF_TAG_MIN);
        _M[idx].first  = cnt;
        _M[idx].second.first = task;
        auto r = ff_loadbalancer::ff_send_out_to(&_M[idx], id, retry, ticks);
        if (r) {++cnt; ++idx %= _M_size;}
        return r;
    }        
    size_t idx,cnt,_M_size=0;
    ordering_pair_t* _M=nullptr;    
};

struct ordered_gt: ff_gatherer {
    struct PairCmp {
        bool operator()(const ordering_pair_t* lhs, const ordering_pair_t* rhs) const { 
            return lhs->first > rhs->first;
        }
    };
    ordered_gt(int max_num_workers): ff_gatherer(max_num_workers) {}
    void init(const size_t size) { MemSize=size; cnt =0; }
    inline ssize_t gather_task(void ** task) {
        if (!Q.empty()) {
	    auto next = Q.top();
	    if (cnt == next->first) {
            ++cnt;
            *task = next->second.first;
            Q.pop();
            return next->second.second;
	    }
        }
        ssize_t nextr=  ff_gatherer::gather_task(task);
        if (*task < FF_TAG_MIN) {
            ordering_pair_t *in =  reinterpret_cast<ordering_pair_t*>(*task);
            if (cnt == in->first) { // it's the next to send out
                cnt++;
                *task = in->second.first;
                return nextr;
            }
            in->second.second = nextr;
            Q.push(in);
            if (Q.size()>MemSize) {
                error("FATAL ERROR: OFARM, ondemand, not enough memory, increase MemoryElements\n");
            }
            *task = FF_GO_ON;
        }
        return nextr;                            
    }
    int all_gather(void *task, void **V) {
        ssize_t sender = ff_gatherer::get_channel_id(); // set current sender of element task
        int r= ff_gatherer::all_gather(task,V);
        size_t nw = getnworkers();
        for(size_t i=0;i<nw;++i) {
            if (V[i]) {
                if (i!=(size_t)sender)
                    V[i] = (reinterpret_cast<ordering_pair_t*>(V[i]))->second.first;
            }
        }
        return r;
    }
    
    size_t cnt;
    std::priority_queue<ordering_pair_t*,std::vector<ordering_pair_t*>, PairCmp> Q;
    size_t MemSize;       
};
// Worker wrapper to be used when ordering_pair_t is added to the data elements
class OrderedWorkerWrapper: public ff_node_t<ordering_pair_t> {
public:    
    OrderedWorkerWrapper(ff_node* worker, bool cleanup=false):
        worker(worker),cleanup(cleanup) {
        set_barrier(worker->get_barrier());
        worker->set_barrier(nullptr);
    }
    ~OrderedWorkerWrapper() {
        if (cleanup) delete worker;
    }
    ordering_pair_t *svc(ordering_pair_t *in) {
        auto out=worker->svc(in->second.first);
        in->second.first=out;
        return in;
    }
    int  svc_init() { return worker->svc_init();}
    void svc_end() { worker->svc_end(); }
    void eosnotify(ssize_t id) { worker->eosnotify(id);}
protected:    
    ff_node* worker;
    bool cleanup;
};
// A node that removes the ordering_pair_t around the data element
template<typename IN_t>    
class OrderedEmitterWrapper: public ff_node_t<IN_t, ordering_pair_t> {
public:
    OrderedEmitterWrapper(ordering_pair_t*const  m, const size_t size):
        idx(0),cnt(0),Memory(m), MemSize(size) {}

    int svc_init() {
        idx=0;
        return 0;
    } 
    inline ordering_pair_t* svc(IN_t* in) {
        Memory[idx].first=cnt;
        Memory[idx].second.first = in;
        this->ff_send_out(&Memory[idx]);
        ++cnt;
        ++idx %= MemSize;
        return this->GO_ON;
    }
    size_t idx,cnt;
    ordering_pair_t* Memory;
    size_t MemSize;
};
    
// A node that removes the ordering_pair_t around the data element
class OrderedCollectorWrapper: public ff_node_t<ordering_pair_t, void> {
public:
    struct PairCmp {
        bool operator()(const ordering_pair_t* lhs, const ordering_pair_t* rhs) const { 
            return lhs->first > rhs->first;
        }
    };
    OrderedCollectorWrapper(const size_t size):cnt(0),MemSize(size) {}
    int svc_init() {
        cnt=0;
        return 0;
    } 
    inline void* svc(ordering_pair_t* in) {
        if (cnt == in->first) { // it's the next to send out
            ff_send_out(in->second.first);
            cnt++;
            while(!Q.empty()) {
                auto next = Q.top();
                if (cnt == next->first) {
                    ff_send_out(next->second.first);
                    ++cnt;
                    Q.pop();
                } else break;
            }
            return GO_ON;
        }
        Q.push(in);
        if (Q.size()>MemSize) {
            error("FATAL ERROR: OFARM, ondemand, not enough memory, increase MemoryElements\n");
        }
        return GO_ON;
    }
    size_t cnt;
    std::priority_queue<ordering_pair_t*,std::vector<ordering_pair_t*>, PairCmp> Q;
    size_t MemSize;       
};

    
// --------------------------------------------------------------

} // namespace ff    
#endif /* FF_ORDERING_POLICY_HPP */
