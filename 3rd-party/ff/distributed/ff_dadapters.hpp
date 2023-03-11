#ifndef ADAPTERS_H
#define ADAPTERS_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

#define FARM_GATEWAY -10

using namespace ff;

class SquareBoxRight : public ff_minode {
    ssize_t neos = 0;
public:	

	int svc_init() {
		// change the size of the queue towards the Sender
		// forcing the queue to be bounded of capacity 1
		size_t oldsz;
		change_outputqueuesize(1, oldsz);
		assert(oldsz != 0);		
		return 0;
	}

	void* svc(void* in) {
		return in;
	}

	void eosnotify(ssize_t id) {
		if (id == (ssize_t)(this->get_num_inchannels() - 1)) return;   // EOS coming from the SquareLeft, we must ignore it
		if (++neos == (ssize_t)(this->get_num_inchannels() - 1))
			this->ff_send_out(this->EOS);
	}
};

class SquareBoxLeft : public ff_monode {
	std::unordered_map<int, int> destinations;
	long next_rr_destination = 0;
public:
	/*
	 *  - localWorkers: list of pairs <logical_destination, physical_destination> where logical_destination is the original destination of the shared-memory graph
	 */
	SquareBoxLeft(const std::unordered_map<int, int> localDestinations) : destinations(localDestinations) {}
    
	void* svc(void* in){
		if (reinterpret_cast<message_t*>(in)->chid == -1) {
			ff_send_out_to(in, next_rr_destination);
			next_rr_destination = (next_rr_destination + 1) % destinations.size();
		}
		else this->ff_send_out_to(in, destinations[reinterpret_cast<message_t*>(in)->chid]);
		return this->GO_ON;
    }
};

class EmitterAdapter: public internal_mo_transformer {
private:
    int totalWorkers, index;
	std::unordered_map<int, int> localWorkersMap;
    int nextDestination = -1;
public:
	/** Parameters:
	 * 	- n: rightmost sequential node of the builiding block representing the left-set worker
	 * 	- totalWorkers: number of the workers on the right set (i.e., all the possible destinations) of the original entire a2a
	 *  - index: index of nodde n in the output list of the left set of the orgiginal entire a2a
	 *  - localWorkers: list of pairs <logical_destination, physical_destination> where logical_destination is the original destination of the shared-memory graph
	 *  - cleanup 
	 **/
	EmitterAdapter(ff_node* n, int totalWorkers, int index, std::unordered_map<int, int> localWorkers = {0,0}, bool cleanup=false): internal_mo_transformer(this, false), totalWorkers(totalWorkers), index(index), localWorkersMap(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}

	int svc_init() {
		if (this->n->isMultiOutput()) {
			ff_monode* mo = reinterpret_cast<ff_monode*>(this->n);
			//mo->set_running(localWorkersMap.size() + 1); // the last worker is the forwarder to the remote workers
			mo->set_virtual_outchannels(totalWorkers);
		}

		// change the size of the queue to the SquareBoxRight (if present),
		// since the distributed-memory communications are all on-demand
		svector<ff_node*> w;
        this->get_out_nodes(w);
		assert(w.size()>0);
		if (w.size() > localWorkersMap.size()) {
			assert(w.size() == localWorkersMap.size()+1);
			size_t oldsz;		
			w[localWorkersMap.size()]->change_inputqueuesize(1, oldsz);
		}
		return n->svc_init();
	}
	
	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;					
		
        this->forward(out, -1);

		return GO_ON;
	}

    bool forward(void* task, int destination){

		if (destination == -1) {
			message_t* msg = nullptr;
			bool datacopied = true;
			do {
				for(size_t i = 0; i < localWorkersMap.size(); i++){
					long actualDestination = (nextDestination + 1 + i) % localWorkersMap.size();
					if (ff_send_out_to(task, actualDestination, 1)){ // non blcking ff_send_out_to, we try just once                                                         
                        nextDestination = actualDestination;
                        if (msg) {
							if (!datacopied) msg->data.doNotCleanup();
							delete msg;
						}
                        return true;
					}
				}
				if (!msg) {
					msg = new message_t(index, destination);
					datacopied = this->n->serializeF(task, msg->data);
					if (!datacopied) {
						msg->data.freetaskF = this->n->freetaskF;
					}
				}
				if (ff_send_out_to(msg, localWorkersMap.size(), 1)) {
					if (datacopied) this->n->freetaskF(task);
					return true;
				}
			} while(1);
		}

        auto pyshicalDestination = localWorkersMap.find(destination);
		if (pyshicalDestination != localWorkersMap.end()) {
			return ff_send_out_to(task, pyshicalDestination->second);
		} else {
			message_t* msg = new message_t(index, destination);
			bool datacopied = this->n->serializeF(task, msg->data);
			if (!datacopied) {
				msg->data.freetaskF = this->n->freetaskF;
			}
			ff_send_out_to(msg, localWorkersMap.size());
			if (datacopied) this->n->freetaskF(task);
			return true;
		}
    }

	void svc_end(){n->svc_end();}
	
	int run(bool skip_init=false) {
		return internal_mo_transformer::run(skip_init);
	}
	
	ff::ff_node* getOriginal(){return this->n;}
	
	static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {
        return ((EmitterAdapter*)obj)->forward(task, id);
	}
};


class CollectorAdapter: public internal_mi_transformer {

private:
	std::vector<int> localWorkers;
public:
	/* localWorkers: contains the ids of "real" (square boxes not included) left-workers 
	 *               connected with this node
	 *
	 */
	CollectorAdapter(ff_node* n, std::vector<int> localWorkers, bool cleanup=false): internal_mi_transformer(this, false), localWorkers(localWorkers) {
		this->n       = n;
		this->cleanup = cleanup;
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	int svc_init() {
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(localWorkers.size() + 1);
		}
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
        ssize_t channel;

		// if the results come from the "square box", it is a result from a remote workers so i have to read from which worker it come from 
		if ((size_t)get_channel_id() == localWorkers.size()){
			message_t * msg = reinterpret_cast<message_t*>(in);
            channel = msg->sender;
			bool datacopied = true;
			in = this->n->deserializeF(msg->data, datacopied);
			if (!datacopied) msg->data.doNotCleanup();
			delete msg;
		} else {  // the result come from a local worker, just pass it to collector and compute the right worker id
            channel = localWorkers.at(get_channel_id());
		}

		// update the input channel id field only if the wrapped node is a multi input
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channel, true);
		}	

		return n->svc(in);
	}

	ff_node* getOriginal(){ return this->n;	}
};

#endif
