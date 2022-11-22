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

#ifndef WRAPPER_H
#define WRAPPER_H

#include <iostream>
#include <type_traits>
#include <functional>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>

using namespace ff;

template<typename Tin, typename Tout = Tin>
struct DummyNode : public ff_node_t<Tin, Tout> {
    Tout* svc(Tin* in){ return nullptr;}
};

/*
	Wrapper IN class
*/
class WrapperIN: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
	int feedbackChannels = 0;
public:


	WrapperIN(ff_node* n, int inchannels=1, bool cleanup=false): internal_mi_transformer(this, false), inchannels(inchannels){
		this->n = n;
		this->cleanup= cleanup;
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

	int svc_init() {
		if (this->n->isMultiInput()) {
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_running(get_num_feedbackchannels()+1);
		}
		return n->svc_init();
	}
	
	void * svc(void* in) {
		// with feedback channels it might be null
		if (in == nullptr) return n->svc(nullptr);
		message_t* msg = (message_t*)in;

		
		if (this->n->isMultiInput()) {
			int channelid = msg->sender; 
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
			mi->set_input_channelid(channelid, fromInput());
			if (!this->fromInput()) return n->svc(in);
		}

		// received a logical EOS
		if (msg->data.getLen() == 0){
			this->n->eosnotify(msg->sender);
			delete msg;
			return GO_ON;
		}
		
		bool datacopied=true;
		void* inputData = this->n->deserializeF(msg->data, datacopied);
		if (!datacopied) msg->data.doNotCleanup();
		delete msg;	
		return n->svc(inputData);
	}

	void svc_end(){this->n->svc_end();}
	
	ff_node* getOriginal(){ return this->n;	}
};


/*
	Wrapper OUT class
*/

class WrapperOUT: public internal_mo_transformer {
private:
    int outchannels; // number of output channels the wrapped node is supposed to have
	int defaultDestination;
	int myID;
	int feedbackChannels, localFeedbacks, remoteFeedbacks;
public:
	
	WrapperOUT(ff_node* n, int id, int outchannels=-1, int remoteFeedbacks = 0, bool cleanup=false, int defaultDestination = -1): internal_mo_transformer(this, false), outchannels(outchannels), defaultDestination(defaultDestination), myID(id), remoteFeedbacks(remoteFeedbacks){
		this->n = n;
		this->cleanup= cleanup;
		registerCallback(ff_send_out_to_cbk, this);
	}
	
	bool serialize(void* in, int id) {
		if (localFeedbacks){
			if (id < localFeedbacks) {
				if (id == -1) return ff_send_out(in);
				return ff_send_out_to(in, id);
			}
			// from 0 to feedbackChannels-1 are feedback channels
			// from feedbackChannels to outchannels-1 are forward channels
			id -= localFeedbacks; 
		}

		message_t* msg = new message_t;
		msg->feedback = (id < remoteFeedbacks && remoteFeedbacks);
		if (!msg->feedback) id -= remoteFeedbacks;
		
		bool datacopied = this->n->serializeF(in, msg->data);
		msg->sender = myID;
		msg->chid   = id;
		if (!datacopied)  msg->data.freetaskF = this->n->freetaskF;
		if (localFeedbacks) {
			// all forward channels are multiplexed in feedbackChannels which coicide with the sender node!
			ff_send_out_to(msg, localFeedbacks);
		} else
			ff_send_out(msg);
		if (datacopied) this->n->freetaskF(in);
		return true;
	}

	int svc_init() {
		// save the channel id fo the sender, useful for when there are feedbacks in the application

		// these are local feedback channels
		localFeedbacks = internal_mo_transformer::get_num_feedbackchannels();
		feedbackChannels = localFeedbacks + remoteFeedbacks;

		if (this->n->isMultiOutput()) {
			ff_monode* mo = reinterpret_cast<ff_monode*>(this->n);
			mo->set_virtual_outchannels(outchannels);
			mo->set_virtual_feedbackchannels(feedbackChannels); 
		}

		return n->svc_init();
	}

	void * svc(void* in) {
		void* out = n->svc(in);
		if (out > FF_TAG_MIN) return out;
		serialize(out, defaultDestination);
		return GO_ON;
	}

	void svc_end(){n->svc_end();}
	
	int run(bool skip_init=false) {
		return internal_mo_transformer::run(skip_init);
	}


    /** returns the total number of output channels */
    size_t  get_num_outchannels() const      { return outchannels; }
    size_t  get_num_feedbackchannels() const { return feedbackChannels; }

	
	ff::ff_node* getOriginal(){return this->n;}
	
	static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperOUT*)obj)->serialize(task, id);
	}
};

/*
	Wrapper INOUT class
*/
class WrapperINOUT: public internal_mi_transformer {

private:
    int inchannels;	// number of input channels the wrapped node is supposed to have
	int defaultDestination;
	int myID;
	int remoteFeedbacks;
public:

	WrapperINOUT(ff_node* n, int id, int inchannels=1, int remoteFeedbacks = 0, bool cleanup=false, int defaultDestination = -1): internal_mi_transformer(this, false), inchannels(inchannels), defaultDestination(defaultDestination), myID(id), remoteFeedbacks(remoteFeedbacks){
		this->n       = n;
		this->cleanup = cleanup;
        registerCallback(ff_send_out_to_cbk, this);
	}

	void registerCallback(bool (*cb)(void *,int,unsigned long,unsigned long,void *), void * arg) {
		internal_mi_transformer::registerCallback(cb,arg);
	}

    bool serialize(void* in, int id) {
		if ((void*)in > FF_TAG_MIN) return ff_node::ff_send_out(in);
		
		message_t* msg = new message_t;

		
		msg->feedback = (id < remoteFeedbacks && remoteFeedbacks);

	
		bool datacopied= this->n->serializeF(in, msg->data);
		msg->sender = myID;          // FIX!
		msg->chid   = id;
		if (!msg->feedback) msg->chid -= remoteFeedbacks;
		if (!datacopied)  msg->data.freetaskF = this->n->freetaskF;
		ff_node::ff_send_out(msg);
		if (datacopied) this->n->freetaskF(in);
		return true;
	}

	int svc_init() {
		/*if (this->n->isMultiInput()) { // ??? what??
			ff_minode* mi = reinterpret_cast<ff_minode*>(this->n); // what?????
			mi->set_running(inchannels);
		}*/
		return n->svc_init();
	}

	void svc_end(){this->n->svc_end();}
	
	void * svc(void* in) {
		void* out;
		if (in != nullptr) {
			message_t* msg = (message_t*)in;
			
			// received a logical EOS
			if (msg->data.getLen() == 0){
				this->n->eosnotify(msg->sender); // TODO: msg->sender here is not consistent... always 0
				delete msg;
				return GO_ON;
			}
			
			if (this->n->isMultiInput()) {
				int channelid = msg->sender; 
				ff_minode* mi = reinterpret_cast<ff_minode*>(this->n);
				mi->set_input_channelid(channelid, !msg->feedback);
			}
			bool datacopied=true;
			out = n->svc(this->n->deserializeF(msg->data, datacopied));
			if (!datacopied) msg->data.doNotCleanup();
			delete msg;
		}  else // it can happen if we have a feedback channel
			out = n->svc(nullptr);
        serialize(out, defaultDestination);
        return GO_ON;
	}

	bool init_output_blocking(pthread_mutex_t   *&m,
							  pthread_cond_t    *&c,
							  bool feedback=true) {
        return ff_node::init_output_blocking(m,c,feedback);
    }

    void set_output_blocking(pthread_mutex_t   *&m,
							 pthread_cond_t    *&c,
							 bool canoverwrite=false) {
		ff_node::set_output_blocking(m,c,canoverwrite);
	}

	ff_node* getOriginal(){ return this->n;	}
	
    static inline bool ff_send_out_to_cbk(void* task, int id,
										  unsigned long retry,
										  unsigned long ticks, void* obj) {		
		return ((WrapperINOUT*)obj)->serialize(task, id);
	}
};

#endif
