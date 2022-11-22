/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file gsearch.hpp
 *  \ingroup high_level_pattern
 *
 *  \brief This file implements the graph search skeleton.
 */
 
#ifndef FF_GSEARCH_HPP
#define FF_GSEARCH_HPP
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

#include <deque>
#include <vector>
#include <bitset>
#include <algorithm>

#include <ff/ff.hpp>

namespace ff {

// this is a generic graph node
template <typename T>
class gnode_t {
public:
    typedef T node_type;
    
    gnode_t(const long nodeid, const T& elem):nodeid(nodeid),elem(elem) {}
    
    /// compare operator
    inline bool operator==(const gnode_t &node) const {
        return (elem == node.getElement());
    }
    
    /// returns the list of output nodes
    inline void out_nodes(std::deque<gnode_t*> &out) const {
        out = outNodes;
    }
    
    /// adds one node to the graph
    inline void add_node(gnode_t*const t) { outNodes.push_back(t); }  
    /// returns the base element of the node
    inline const T& getElement() const { return elem;}
    /// returns the node identifier
    inline unsigned long getId() const { return nodeid; }

protected:
    const unsigned long  nodeid;    /// node's identifier
    T                    elem;      /// base graph element
    std::deque<gnode_t*> outNodes;  /// list of output nodes
};



template<typename T, unsigned N=10485760>
class ff_graphsearch: public ff_node {
protected:
    // worker class
    class Worker: public ff_node {
    public:
	Worker(long *flag, bool all=false): 
	    tosearch(NULL),foundflag(flag),all(all) {}
	
	void setNodeToSearch(T*const n, bool a=false) { 
	    tosearch=n; 
	    found.clear();
	    all=a;
	}
	inline void *svc(void *t) {
	    if (!all && *foundflag) return NULL;       
	    T *node = static_cast<T*>(t);
	    if (*tosearch == *node) { 
            found.push_back(node); 
            *foundflag=true; 
            return (all?t:NULL);
	    }
	    return t;
	}

        const std::deque<T*>& Found() const { return found;}
    protected:
        T *tosearch;
        std::deque<T*> found;
        long *const foundflag;
        bool all;
    };
    
    // scheduler class
    class Emitter: public ff_node {
    public:	
        enum {CHECK_FOUND_N=256};
        
        Emitter(const std::vector<ff_node*> &W, const long &foundflag, const bool all=false):
            foundflag(foundflag),W(W),start(NULL),counter(0),all(all) {}
        
        void setStart(T*const n) { start = n;}    
        void setNodeToSearch(T*const n, bool a=false) {
            all = a;
            for(auto w: W) ((Worker*)w)->setNodeToSearch(n,all);
        }
        
        inline int svc_init() { 
            mask.reset(); 
            counter=0; 
            return 0;
        }
        inline void *svc(void *t) {
            if (t == NULL) {
                if (start==NULL) return NULL;
                mask.set(start->getId());
                ff_send_out((void*)start);
                ++counter;
                std::deque<T*> outNodes;
                start->out_nodes(outNodes);
                for(T *n: outNodes) {
                    ++counter;
                    mask.set(n->getId());
                    ff_send_out((void*)n);
                }
                return GO_ON;      
            }
            --counter;
            if (!all && foundflag>0) return NULL;
            const T &node = *(static_cast<T*>(t));
            
            std::deque<T*> outNodes;
            node.out_nodes(outNodes);
            auto k =0;
            for(T *n: outNodes) {
                if (!all && (++k == CHECK_FOUND_N)) {
                    if (foundflag>0) return NULL;
                    k=0;
                }
                if (!mask.test(n->getId())) {
                    ++counter;
                    mask.set(n->getId());
                    ff_send_out((void*)n);
                }
            }
            if (counter == 0) 	return NULL;
            return ((!all && foundflag>0) ? NULL : GO_ON);
        }
    protected:
        const long                    &foundflag;
        const std::vector<ff_node*>   &W;
        T                             *start;
        unsigned long                  counter;
        bool                           all;
        std::bitset<N>                 mask;
    };

    inline void resetqueues() {  for(auto n: W) n->reset();  }

private:
    long        found;
    ff_farm    *farm;
    Emitter    *scheduler;
    T          *start;
    std::vector<ff_node*> W;
    bool        all;
public:
    
    ff_graphsearch(const int nw=ff_numCores(), const bool all=false):found(0),start(NULL),all(all) {
        for(int i=0;i<nw;++i) W.push_back(new Worker(&found,all));
        farm = new ff_farm(false, 524288*nw, 524800*nw, false, nw, true);
        farm->add_workers(W);
        scheduler = new Emitter(W,found);
        farm->add_emitter(scheduler);
        farm->wrap_around();
        if (farm->run_then_freeze()<0) {
            error("running farm ff_graphsearch\n");
        } else farm->wait_freezing();
    }
    
    ~ff_graphsearch() {
        found = 1;
        if (farm) farm->run_and_wait_end();
        while(W.size()){
            Worker * w = static_cast<Worker*>(W.back());
            delete w;
            W.pop_back();
        }
        if (scheduler) delete scheduler;
        if (farm) delete farm;
    }
    
    /// sets the starting node
    void setStart(T *const st) { start = st;}
    
    // One shot search. It returns just one result (valid only if the return value is true)
    inline bool search(T *const st, T* const search, T *&result, const int nw=-1) {
        if (nw > (int)W.size()) {
            error("ff_graphsearch:search: nw too big, using nw=%d\n", W.size());
        }
        found = 0;
        scheduler->setStart(st);
        scheduler->setNodeToSearch(search); 
        resetqueues();
        farm->run_then_freeze(nw);
        farm->wait_freezing();
        if (found > 0) {
            for(auto w : W) {
                const std::deque<T*> &r = ((Worker*)w)->Found(); 
                if (r.size()) {
                    result = r.back();
                    return true;
                }
            }
        }
        return false;
    }
    
    /// One shot search. It returns all results.
    inline bool search(T *const st, T* const search, std::deque<T*> &result, const int nw=-1) {
        if (nw > (int)W.size()) {
            error("ff_graphsearch:search: nw too big, using nw=%d\n", W.size());
        }
        found = 0;
        scheduler->setStart(st);
        scheduler->setNodeToSearch(search,true); 
        resetqueues();
        farm->run_then_freeze(nw);
        farm->wait_freezing();
        if (found > 0) {
            for(auto w : W) {
                const std::deque<T*> &r = ((Worker*)w)->Found(); 
                if (r.size()) {
                    for(auto r1: r) result.push_back(r1);
                }
            }
            return true;
        }
        return false;
    }
    
    int svc_init() {
        if (!start) {
            error("ff_graphsearch:svc_init starting pointer not set\n");
            return -1;
        }
        return 0;
    }
    inline void *svc(void *task) {	
        T *tosearch  = static_cast<T*>(task);
        T *result    = tosearch;
        search(start, tosearch, result);
        return result;
    }

    //FIX: add run_and_wait_end()

};

} // namespace

#endif /* FF_GSEARCH_HPP */
