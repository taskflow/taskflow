#ifndef FF_DINTERFACE_H
#define FF_DINTERFACE_H

#include <ff/ff.hpp>
#include <map>


#ifdef DFF_ENABLED
#include <ff/distributed/ff_dgroups.hpp>
#endif


namespace ff {

struct GroupInterface {
    std::string name;
    GroupInterface(std::string name) : name(name){}

    GroupInterface& operator<<(ff_node* node){
#ifdef DFF_ENABLED
        auto& annotated = dGroups::Instance()->annotated;
        auto handler = annotated.find(node);
        if (handler == annotated.end())
            annotated[node] = name;
        else if (handler->second != name){
            std::cerr << "Node has been annotated in group " << name << " and in group " << handler->second << "! Aborting\n";
            abort();
        }
#endif
        return *this;
    }
	
    GroupInterface& operator<<(ff_node& node){
		return *this << &node;
	}   
};


GroupInterface ff_node::createGroup(std::string name){
#ifdef DFF_ENABLED
    dGroups::Instance()->annotateGroup(name, this);
#endif
    return GroupInterface(name);
}

}

#endif
