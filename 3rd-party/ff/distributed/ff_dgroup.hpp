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

/* Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */


#ifndef FF_DGROUP_H
#define FF_DGROUP_H

#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_wrappers.hpp>
#include <ff/distributed/ff_dreceiver.hpp>
#include <ff/distributed/ff_dsender.hpp>
#include <ff/distributed/ff_dadapters.hpp>

#include <numeric>

#ifdef DFF_MPI
#include <ff/distributed/ff_dreceiverMPI.hpp>
#include <ff/distributed/ff_dsenderMPI.hpp>
#endif


template<typename T>
T getBackAndPop(std::vector<T>& v){
    T b = v.back();
    v.pop_back();
    return b;
}

namespace ff{
class dGroup : public ff::ff_farm {
	
    static inline std::unordered_map<int, int> vector2UMap(const std::vector<int> v){
        std::unordered_map<int,int> output;
        for(size_t i = 0; i < v.size(); i++) output[v[i]] = i;
        return output;
    }

    static inline std::map<int, int> vector2Map(const std::vector<int> v){
        std::map<int,int> output;
        for(size_t i = 0; i < v.size(); i++) output[v[i]] = i;
        return output;
    }

    struct ForwarderNode : ff_node { 
        ForwarderNode(std::function<bool(void*, dataBuffer&)> f,
					  std::function<void(void*)> d) {			
            this->serializeF = f;
			this->freetaskF  = d;
        }
        ForwarderNode(std::function<void*(dataBuffer&,bool&)> f,
					  std::function<void*(char*,size_t)> a) {
			this->alloctaskF   = a;
            this->deserializeF = f;
        }
        void* svc(void* input){ return input;}
    };

    static ff_node* buildWrapperIN(ff_node* n){
        if (n->isMultiOutput())
			return new ff_comb(new WrapperIN(new ForwarderNode(n->deserializeF, n->alloctaskF)), n, true, false);
        return new WrapperIN(n);
    }

    static ff_node* buildWrapperOUT(ff_node* n, int id, int outputChannels, int feedbackChannels = 0){
        if (n->isMultiInput()) return new ff_comb(n, new WrapperOUT(new ForwarderNode(n->serializeF, n->freetaskF), id, outputChannels, feedbackChannels, true), false, true);
        return new WrapperOUT(n, id, outputChannels, feedbackChannels);
    }

public:
    dGroup(ff_IR& ir){
            
        if (ir.isVertical()){
            int outputChannels = 0;
            int feedbacksChannels = 0;
            if (ir.hasSender){
                outputChannels = std::accumulate(ir.routingTable.begin(), ir.routingTable.end(), 0, [](const auto& s, const auto& f){return s+ (f.second.second != ChannelType::FBK ? f.second.first.size() : 0);});
                feedbacksChannels = std::accumulate(ir.routingTable.begin(), ir.routingTable.end(), 0, [](const auto& s, const auto& f){return s+(f.second.second == ChannelType::FBK ? f.second.first.size() : 0);});
            }

            std::vector<int> reverseOutputIndexes(ir.hasLeftChildren() ? ir.outputL.rbegin() : ir.outputR.rbegin(), ir.hasLeftChildren() ? ir.outputL.rend() : ir.outputR.rend());
            for(ff_node* child: (ir.hasLeftChildren() ? ir.L : ir.R)){
                ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                
                // handle the case we have a pipe (or more nested) with just one sequential stage (not a combine)
                if (inputs.size() == 1 && outputs.size() == 1 && inputs[0] == outputs[0])
                    child = inputs[0];

               if (isSeq(child)){
				   ff_node* wrapper = nullptr;
				   if (ir.hasReceiver && ir.hasSender) {
					   wrapper = new WrapperINOUT(child, getBackAndPop(reverseOutputIndexes), 1, feedbacksChannels);
					   workers.push_back(wrapper);
                       if (ir.isSource && ir.isVertical() && ir.hasLeftChildren()) wrapper->skipfirstpop(true);
				   } else if (ir.hasReceiver) {
					   wrapper = buildWrapperIN(child);
					   if (ir.isSource && ir.isVertical() && ir.hasLeftChildren()) wrapper->skipfirstpop(true);
					   workers.push_back(wrapper);
				   } else  {
					   wrapper = buildWrapperOUT(child, getBackAndPop(reverseOutputIndexes), outputChannels, feedbacksChannels);
					   workers.push_back(wrapper);
				   }
				   // TODO: in case there are feedback channels we cannot skip all pops!
				   /*if (ir.isSource)
					   wrapper->skip(true);*/
               } else {

				   // TODO: in case there are feedback channels we cannot skip all pops!
				   /* if (ir.isSource)
					    child->skipallpop(true);*/
				   
				    if (ir.hasReceiver){
                        for(ff_node* input : inputs){
                            ff_node* inputParent = getBB(child, input);
                            if (inputParent) {
								ff_node* wrapper = buildWrapperIN(input);							
								inputParent->change_node(input, wrapper, true); //cleanup?? removefromcleanuplist??
								if (ir.isSource && ir.isVertical() && ir.hasLeftChildren())
									inputParent->skipfirstpop(true);
							}
                        }
                    }

                    if (ir.hasSender){
                        for(ff_node* output : outputs){
                            ff_node* outputParent = getBB(child, output);
                            if (outputParent) outputParent->change_node(output, buildWrapperOUT(output, getBackAndPop(reverseOutputIndexes), outputChannels, feedbacksChannels), true); // cleanup?? removefromcleanuplist??
                        }
                    }
                    
                   workers.push_back(child);
               }
            }

            if (ir.hasReceiver){
                if (ir.protocol == Proto::TCP)
                    this->add_emitter(new ff_dreceiver(ir.listenEndpoint, ir.expectedEOS, vector2Map(ir.hasLeftChildren() ? ir.inputL : ir.inputR)));
                
#ifdef DFF_MPI
                else
                   this->add_emitter(new ff_dreceiverMPI(ir.expectedEOS, vector2Map(ir.hasLeftChildren() ? ir.inputL : ir.inputR)));
#endif 
            }

            if (ir.hasSender){
                if(ir.protocol == Proto::TCP)
                    this->add_collector(new ff_dsender(ir.destinationEndpoints, &ir.routingTable, ir.listenEndpoint.groupName, ir.outBatchSize, ir.messageOTF), true);
               
#ifdef DFF_MPI
                else
                   this->add_collector(new ff_dsenderMPI(ir.destinationEndpoints, &ir.routingTable, ir.listenEndpoint.groupName, ir.outBatchSize, ir.messageOTF), true);
#endif      
            }
			
			if (ir.hasRightChildren() && ir.parentBB->isAll2All()) {
				ff_a2a *a2a = reinterpret_cast<ff_a2a*>(ir.parentBB);
				if (a2a->ondemand_buffer() > 0) {
					this->set_scheduling_ondemand(a2a->ondemand_buffer());
				}
			}			
        }
        else { // the group is horizontal!
            ff_a2a* innerA2A = new ff_a2a();
            
            std::vector<int> reverseLeftOutputIndexes(ir.outputL.rbegin(), ir.outputL.rend());

            std::unordered_map<int, int> localRightWorkers = vector2UMap(ir.inputR);
            std::vector<ff_node*> firstSet;
            for(ff_node* child : ir.L){
                if (isSeq(child))
                    if (ir.isSource){
                        ff_node* wrapped = new EmitterAdapter(child, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers);
                        //if (ir.hasReceiver)
						wrapped->skipallpop(true);
                        firstSet.push_back(wrapped);
                    } else {
						auto d = child->getDeserializationFunction();
                        firstSet.push_back(new ff_comb(new WrapperIN(new ForwarderNode(d.first,d.second), 1, true), new EmitterAdapter(child, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers), true, true));
                    }
                else {
                    
                    if (ir.isSource){
                        //if (ir.hasReceiver)
						child->skipallpop(true);
                    } else {
                        ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                        for(ff_node* input : inputs){
                            ff_node* inputParent = getBB(child, input);
                            if (inputParent) inputParent->change_node(input, buildWrapperIN(input), true); // cleanup??? remove_fromcleanuplist??
                        }
                    }
                    
                    ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                    for(ff_node* output : outputs){
                        ff_node* outputParent = getBB(child, output);
                        if (outputParent) outputParent->change_node(output,  new EmitterAdapter(output, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers) , true); // cleanup??? remove_fromcleanuplist??
                    }
                    firstSet.push_back(child); //ondemand?? cleanup??
                }
            }
            // add the Square Box Left, just if we have a receiver!
            if (ir.hasReceiver)
                firstSet.push_back(new SquareBoxLeft(localRightWorkers)); 

            std::transform(firstSet.begin(), firstSet.end(), firstSet.begin(), [](ff_node* n) -> ff_node* {
                if (!n->isPipe())
					return new ff_Pipe(n);
				return n;
            });
            innerA2A->add_firstset(firstSet, reinterpret_cast<ff_a2a*>(ir.parentBB)->ondemand_buffer()); // note the ondemand!!
            
            int outputChannels = std::accumulate(ir.routingTable.begin(), ir.routingTable.end(), 0, [](const auto& s, const auto& f){return s+ (f.second.second == ChannelType::FWD ? f.second.first.size() : 0);});
            int feedbacksChannels = std::accumulate(ir.routingTable.begin(), ir.routingTable.end(), 0, [](const auto& s, const auto& f){return s+(f.second.second == ChannelType::FBK ? f.second.first.size() : 0);});
            
            std::vector<int> reverseRightOutputIndexes(ir.outputR.rbegin(), ir.outputR.rend());
            std::vector<ff_node*> secondSet;
            for(ff_node* child : ir.R){
                if (isSeq(child)) {
					auto s = child->getSerializationFunction();
                    secondSet.push_back(
                        (ir.isSink) ? (ff_node*)new CollectorAdapter(child, ir.outputL) 
						: (ff_node*)new ff_comb(new CollectorAdapter(child, ir.outputL), new WrapperOUT(new ForwarderNode(s.first, s.second), getBackAndPop(reverseRightOutputIndexes), outputChannels, feedbacksChannels, true), true, true)
                    );
				} else {
                    ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                    for(ff_node* input : inputs){
                        ff_node* inputParent = getBB(child, input);
                        if (inputParent) inputParent->change_node(input, new CollectorAdapter(input, ir.outputL), true); //cleanup?? remove_fromcleanuplist??
                    }

                    if (!ir.isSink){
                        ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                        for(ff_node* output : outputs){
                            ff_node* outputParent = getBB(child, output);
                            if (outputParent) outputParent->change_node(output, buildWrapperOUT(output, getBackAndPop(reverseRightOutputIndexes), outputChannels, feedbacksChannels), true); //cleanup?? removefromcleanuplist?
                        }
                    }

                    secondSet.push_back(child);
                }
            }
            
            // add the SQuareBox Right, iif there is a sender!
            if (ir.hasSender)
                secondSet.push_back(new SquareBoxRight);

            std::transform(secondSet.begin(), secondSet.end(), secondSet.begin(), [](ff_node* n) -> ff_node* {
                if (!n->isPipe())
					return new ff_Pipe(n);
				return n;
            });

            innerA2A->add_secondset<ff_node>(secondSet); // cleanup??
            workers.push_back(innerA2A);

            
            if (ir.hasReceiver){
                if (ir.protocol == Proto::TCP)
                    this->add_emitter(new ff_dreceiverH(ir.listenEndpoint, ir.expectedEOS, vector2Map(ir.inputL)));
#ifdef DFF_MPI
                else
                   this->add_emitter(new ff_dreceiverHMPI(ir.expectedEOS, vector2Map(ir.inputL)));
#endif 
            }
            
            if (ir.hasSender){
                if(ir.protocol == Proto::TCP)
                    this->add_collector(new ff_dsenderH(ir.destinationEndpoints, &ir.routingTable, ir.listenEndpoint.groupName, ir.outBatchSize, ir.messageOTF, ir.internalMessageOTF) , true);
#ifdef DFF_MPI
                else
                   this->add_collector(new ff_dsenderHMPI(ir.destinationEndpoints, &ir.routingTable, ir.listenEndpoint.groupName, ir.outBatchSize, ir.messageOTF, ir.internalMessageOTF), true);
#endif   
            }
        }  

        if (this->getNWorkers() == 0){
            std::cerr << "The farm implementing the distributed group is empty! There might be an error! :(\n";
            abort();
        }
    }
         

};
}
#endif

