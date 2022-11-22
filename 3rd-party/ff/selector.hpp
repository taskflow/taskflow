/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file stecilReduceOCL.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief OpenCL map and non-iterative data-parallel patterns
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
 *  Authors:
 *    Massimo Torquati  (August 2015)
 *
 *  
 */

#ifndef FF_NODE_SELECTOR_HPP
#define FF_NODE_SELECTOR_HPP

#include <memory>
#include <vector>
#include <ff/node.hpp>

namespace ff {


template<typename IN_t, typename OUT_t=IN_t>
class ff_nodeSelector: public ff_node_t<IN_t,OUT_t> {
protected:
    static bool ff_send_out_selector(void * task,int id,
                                     unsigned long retry,
                                     unsigned long ticks, void *obj) {
        return reinterpret_cast<ff_node*>(obj)->ff_send_out(task, id, retry, ticks);
    }


    inline void add2selector(ff_node &node) { ff_nodeSelector<IN_t,OUT_t>::addNode(node); }
    inline void add2selector(ff_node *node) { 
        cleanup_devices.push_back(node);
        ff_nodeSelector<IN_t,OUT_t>::addNode(*node);
    }
    void add2selectorall(const IN_t &task){
        setTask(task);
    }
    void add2selectorall(IN_t &task){
        setTask(task);
    }

    void add2selectorall(){} 
    template<typename FIRST,typename ...ARGS>
    void add2selectorall(FIRST &stage,ARGS&...args){
        add2selector(stage);
        add2selectorall(args...);
    }
    template<typename FIRST,typename ...ARGS>
    void add2selectorall(std::unique_ptr<FIRST> & stage,ARGS&...args){            
        add2selector(stage.release());
        add2selectorall(args...);
    }
    
public:

    typedef IN_t  in_type;
    typedef OUT_t out_type;

    ff_nodeSelector():selected(0) {}
    //ff_nodeSelector(const IN_t &task):selected(0), inTask(const_cast<IN_t*>(&task)) {}
    template<typename... NODES>
    ff_nodeSelector(NODES &&...nodes):selected(0) {
        this->add2selectorall(nodes...);
    }
    
    // used to set tasks when running in a passive mode
    void setTask(const IN_t &task) { inTask = const_cast<IN_t*>(&task);  }

    void selectNode(size_t id) { selected = id; }

    int svc_init() { return nodeInit(); }

    OUT_t* svc(IN_t *in) {
        if (in == nullptr) {
            devices[selected]->svc(inTask);
            return ff_node_t<IN_t,OUT_t>::EOS;
        }
        OUT_t* out = (OUT_t*)(devices[selected]->svc(in));
        return out;
    }

    void svc_end() { nodeEnd(); }

    ff_node *getNode(size_t id) { 
        if (id >= devices.size()) return nullptr;
        return devices[id];
    }

    int nodeInit() {
        if (devices.size() == 0) return -1;
        for(size_t i=0;i<devices.size();++i) {
            if (devices[i]->nodeInit()<0) return -1;
            devices[i]->set_id(ff_node::get_my_id());
        }
        return 0;
    }

    void nodeEnd() {
        for(size_t i=0;i<devices.size();++i)
            devices[i]->nodeEnd();
    }

    size_t addNode(ff_node &node) { 
        devices.push_back(&node); 
        node.registerCallback(ff_send_out_selector, this);
        return devices.size()-1;
    }
    size_t addNode(std::unique_ptr<ff_node> node) {
        ff_node *n = node.get();
        n->registerCallback(ff_send_out_selector, this);
        devices.push_back(n); 
        node.release();
        cleanup_devices.push_back(n);
        return devices.size()-1;
    }

    size_t numNodes() const { return devices.size(); }

    int run(bool = false) { return ff_node::run();  }

    int wait() { return ff_node::wait(); }
    
    int run_and_wait_end() {
        if (nodeInit() < 0)	return -1;
        svc(nullptr);
        nodeEnd();
        return 0;
    }

#if defined(FF_REPARA)
    bool rpr_get_measure_energy() const { return false; }
    void rpr_set_measure_energy(bool v) { 
        for(size_t i=0;i<devices.size();++i)
            devices[i]->rpr_set_measure_energy(v);
    }
#endif           
protected:
    size_t      selected;
    IN_t       *inTask;
    std::vector<ff_node*> devices; 
    std::vector<ff_node*> cleanup_devices;
};


} // namespace

#endif /* FF_NODE_SELECTOR_HPP */
