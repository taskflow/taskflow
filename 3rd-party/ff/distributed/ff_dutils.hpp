
#include <set>
#include "ff/pipeline.hpp"

namespace ff {
enum NodeIOTypes { IN, OUT, INOUT};
enum SetEnum {L, R};

/**
 * Check the coverage of the first level building blocks of the main pipe, which is the building block needed to create a distributed FF program.
 * Specifically this method, given the main pipe and and the set of buildingblock from which the user created groups, check if all stages of the pipe belong to a group.
 **/
static inline bool checkCoverageFirstLevel(ff_pipeline* mainPipe, const std::set<ff_node*>& groupBuildingBlock) {
    ff::svector<ff_node*> stages = mainPipe->getStages();
    for(size_t i = 0; i< stages.size(); i++)
        if (!groupBuildingBlock.contains(stages[i])){
            std::cerr << "Stage #" << i << " was not annotated in any group! Aborting!\n";
            abort();
        }
    
    return true;
}

/**
 * Helper function to detect sequential node from a bare node pointer.
 **/
static inline bool isSeq(const ff_node* n){return (!n->isAll2All() && !n->isComp() && !n->isFarm() && !n->isOFarm() && !n->isPipe());}

/**
 * Return the children builiding block of the given building block. We implemented only a2a and pipeline since, groups can be created just only from this two building block.
 **/
static inline std::set<std::pair<ff_node*, SetEnum>> getChildBB(ff_node* parent){
    std::set<std::pair<ff_node*, SetEnum>> out;
    if (parent->isAll2All()){
        for (ff_node* bb : reinterpret_cast<ff_a2a*>(parent)->getFirstSet())
            out.emplace(bb, SetEnum::L);
        
        for(ff_node* bb : reinterpret_cast<ff_a2a*>(parent)->getSecondSet())
            out.emplace(bb, SetEnum::R);
    }

    if (parent->isPipe())
        for(ff_node* bb : reinterpret_cast<ff_pipeline*>(parent)->getStages())
            out.emplace(bb, SetEnum::L); // for pipelines the default List is L (left)
    
    return out;
}

static inline bool isSource(const ff_node* n, const ff_pipeline* p){
    return p->getStages().front() == n;
}

static inline bool isSink(const ff_node* n, const ff_pipeline* p){
    return p->getStages().back() == n;
}

static inline ff_node* getPreviousStage(ff_pipeline* p, ff_node* s){
    ff::svector<ff_node*> stages = p->getStages();
    for(size_t i = 1; i < stages.size(); i++)
        if (stages[i] == s) return stages[--i];
    
    return nullptr;
}

static inline ff_node* getNextStage(ff_pipeline* p, ff_node* s){
    ff::svector<ff_node*> stages = p->getStages();
    for(size_t i = 0; i < stages.size() - 1; i++)
        if(stages[i] == s) return stages[++i];
    
    return nullptr;
}








}