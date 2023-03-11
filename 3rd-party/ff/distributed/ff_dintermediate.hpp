#ifndef IR_HPP
#define IR_HPP
#include <ff/distributed/ff_network.hpp>
#include <ff/node.hpp>
#include <ff/all2all.hpp>
#include <list>
#include <vector>
#include <map>
#include <numeric>

namespace ff {

class ff_IR {
    friend class dGroups;
protected:
    // set to true if the group contains the whole parent building block
    bool wholeParent = false;
    void computeCoverage(){
        if (!parentBB->isAll2All()) return;
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(parentBB);
        coverageL = coverageR = true;
        for(ff_node* n : a2a->getFirstSet())
            if (!L.contains(n)) {coverageL = false; break;}
        for(ff_node* n : a2a->getSecondSet())
            if (!R.contains(n)) {coverageR = false; break;}
    }

    void buildIndexes(){

        if (!L.empty()){
            if (parentBB->isPipe() && !wholeParent){
                assert(L.size() == 1);
                ff::svector<ff_node*> inputs, outputs;
                for (ff_node* n : L){
                    n->get_in_nodes(inputs);
                    n->get_out_nodes(outputs);
                }
                for (size_t i = 0; i < inputs.size(); i++) inputL.push_back(i);
                for (size_t i = 0; i < outputs.size(); i++) outputL.push_back(i);
                return;
            }
            
            ff::svector<ff_node*> parentInputs;
            parentBB->get_in_nodes(parentInputs);

            ff::svector<ff_node*> LOutputs;
            if (!parentBB->isAll2All() || wholeParent) parentBB->get_out_nodes(LOutputs);
            else for(ff_node* n : reinterpret_cast<ff_a2a*>(parentBB)->getFirstSet()) n->get_out_nodes(LOutputs);
            
            for(ff_node* n : L){
                ff::svector<ff_node*> bbInputs; n->get_in_nodes(bbInputs);     
                for(ff_node* bbInput : bbInputs)
                    inputL.push_back(std::find(parentInputs.begin(), parentInputs.end(), bbInput) - parentInputs.begin());

                ff::svector<ff_node*> bbOutputs; n->get_out_nodes(bbOutputs);
                for(ff_node* bbOutput : bbOutputs)
                    outputL.push_back(std::find(LOutputs.begin(), LOutputs.end(), bbOutput) - LOutputs.begin());
            }
        }

        if (!R.empty() && parentBB->isAll2All() && !wholeParent){
            ff::svector<ff_node*> RInputs;
            for(ff_node* n : reinterpret_cast<ff_a2a*>(parentBB)->getSecondSet()) n->get_in_nodes(RInputs);

            ff::svector<ff_node*> parentOutputs;
            parentBB->get_out_nodes(parentOutputs);
            
            for(ff_node* n : R){
                ff::svector<ff_node*> bbInputs; n->get_in_nodes(bbInputs);
                for(ff_node* bbInput : bbInputs)
                    inputR.push_back(std::find(RInputs.begin(), RInputs.end(), bbInput) - RInputs.begin());

                ff::svector<ff_node*> bbOutputs; n->get_out_nodes(bbOutputs);
                for(ff_node* bbOutput : bbOutputs)
                    outputR.push_back(std::find(parentOutputs.begin(), parentOutputs.end(), bbOutput) - parentOutputs.begin());
            }
        }

    }

    
public:
    std::set<ff_node*> L, R;
    bool coverageL = false, coverageR = false;
    bool isSource = false, isSink = false;
    bool hasReceiver = false, hasSender = false;
    Proto protocol;

    ff_node* parentBB;

    ff_endpoint listenEndpoint;
    std::vector<std::pair<ChannelType, ff_endpoint>> destinationEndpoints;

    std::set<std::string> otherGroupsFromSameParentBB;
    size_t expectedEOS = 0;
    int outBatchSize = 1;
    int messageOTF, internalMessageOTF;
    // liste degli index dei nodi input/output nel builiding block in the shared memory context. The first list: inputL will become the rouitng table
    std::vector<int> inputL, outputL, inputR, outputR;

    // pre computed routing table for the sender module (shoud coincide with the one exchanged actually at runtime)
    std::map<std::string, std::pair<std::vector<int>, ChannelType>> routingTable;
    
    // TODO: implmentare l'assegnamento di questi campi
    int leftTotalOuputs;
    int rightTotalInputs;

    bool isVertical(){return (L.empty() + R.empty()) == 1;}

    bool hasLeftChildren() {return !L.empty();}
    bool hasRightChildren() {return !R.empty();}

    void insertInList(std::pair<ff_node*, SetEnum> bb, bool _wholeParent = false){
        wholeParent = _wholeParent;
        switch(bb.second){
            case SetEnum::L: L.insert(bb.first); return;
            case SetEnum::R: R.insert(bb.first); return;
        }
    }

    std::vector<int> getInputIndexes(bool internal){
        if ((isVertical() && hasRightChildren()) || (internal && !R.empty())) return inputR;
        return inputL;
    }

    void print(){
        ff::cout << "###### BEGIN GROUP ######\n";
        ff::cout << "Group Orientation: " << (isVertical() ? "vertical" : "horizontal") << std::endl;
        ff::cout << std::boolalpha << "Source group: " << isSource << std::endl;
        ff::cout << std::boolalpha << "Sink group: " << isSink << std::endl;
        ff::cout << std::boolalpha << "Coverage Left: " << coverageL << std::endl;
        ff::cout << std::boolalpha << "Coverage Right: " << coverageR << std::endl << std::endl;

        ff::cout << std::boolalpha << "Has Receiver: " << hasReceiver << std::endl;
        ff::cout << "Expected input connections: " << expectedEOS << std::endl;
        ff::cout << "Listen endpoint: " << listenEndpoint.address << ":" << listenEndpoint.port << std::endl << std::endl;

        ff::cout << std::boolalpha << "Has Sender: " << hasSender << std::endl;
        ff::cout << "Destination endpoints: " << std::endl;
        for(auto& [ct, e] : destinationEndpoints)
            ff::cout << "\t* " << e.groupName << "\t[[" << e.address << ":" << e.port << "]]  - " << (ct==ChannelType::FBK ? "Feedback" : (ct==ChannelType::INT ? "Internal" : "Forward")) << std::endl;

        ff::cout << "Precomputed routing table: \n";
        for(auto& [gName, p] : routingTable){
            ff::cout << "\t* " << gName << (p.second==ChannelType::FBK ? "Feedback" : (p.second==ChannelType::INT ? "Internal" : "Forward")) << ":";
            for(auto i : p.first) ff::cout << i << " ";
            ff::cout << std::endl;
        }

        ff::cout << "\nPrecomputed FWD destinations: " << std::accumulate(routingTable.begin(), routingTable.end(), 0, [](const auto& s, const auto& f){return s+(f.second.second == ChannelType::FWD ? f.second.first.size() : 0);}) << std::endl;
        ff::cout << "Precomputed INT destinations: " << std::accumulate(routingTable.begin(), routingTable.end(), 0, [](const auto& s, const auto& f){return s+(f.second.second == ChannelType::INT ? f.second.first.size() : 0);}) << std::endl;
        ff::cout << "Precomputed FBK destinations: " << std::accumulate(routingTable.begin(), routingTable.end(), 0, [](const auto& s, const auto& f){return s+(f.second.second == ChannelType::FBK ? f.second.first.size() : 0);}) << std::endl;

        ff::cout << "\n\nIndex Input Left: ";
        for(int i : inputL) ff::cout << i << " ";
        ff::cout << "\n";

        ff::cout << "Index Output Left: ";
        for(int i : outputL) ff::cout << i << " ";
        ff::cout << "\n";

        ff::cout << "Index Input Right: ";
        for(int i : inputR) ff::cout << i << " ";
        ff::cout << "\n";

        ff::cout << "Index Output Right: ";
        for(int i : outputR) ff::cout << i << " ";
        ff::cout << "\n";
        
        ff::cout << "######  END GROUP  ######\n";
    }


};


}

#endif
