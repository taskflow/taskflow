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

#ifndef FF_DSENDER_H
#define FF_DSENDER_H

#include <iostream>
#include <map>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_batchbuffer.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netdb.h>
#include <cmath>
#include <thread>

#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>


using namespace ff;
using precomputedRT_t = std::map<std::string, std::pair<std::vector<int>, ChannelType>>;
class ff_dsender: public ff_minode_t<message_t> { 
protected:
    size_t neos=0;
    std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints;
    precomputedRT_t* precomputedRT;
    std::map<std::pair<int, ChannelType>, int> dest2Socket;
    std::vector<int> sockets;
    int last_rr_socket = -1;
    std::map<int, unsigned int> socketsCounters;
    std::map<int, ff_batchBuffer> batchBuffers;
    std::string gName;
    int batchSize;
    int messageOTF;
    int coreid;
    fd_set set, tmpset;
    int fdmax = -1;

    virtual int handshakeHandler(const int sck, ChannelType t){
        size_t sz = htobe64(gName.size());
        struct iovec iov[3];
        iov[0].iov_base = &t;
        iov[0].iov_len = sizeof(ChannelType);
        iov[1].iov_base = &sz;
        iov[1].iov_len = sizeof(sz);
        iov[2].iov_base = (char*)(gName.c_str());
        iov[2].iov_len = gName.size();

        if (writevn(sck, iov, 3) < 0){
            error("Error writing on socket\n");
            return -1;
        }

        return 0;
    }
	
    int create_connect(const ff_endpoint& destination){
        int socketFD;

#ifdef LOCAL
            socketFD = socket(AF_LOCAL, SOCK_STREAM, 0);
            if (socketFD < 0){
                error("\nError creating socket \n");
                return socketFD;
            }
            struct sockaddr_un serv_addr;
            memset(&serv_addr, '0', sizeof(serv_addr));
            serv_addr.sun_family = AF_LOCAL;

            strncpy(serv_addr.sun_path, destination.address.c_str(), destination.address.size()+1);

            if (connect(socketFD, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
                close(socketFD);
                return -1;
            }
#endif

#ifdef REMOTE
            struct addrinfo hints;
            struct addrinfo *result, *rp;

            memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
            hints.ai_socktype = SOCK_STREAM; /* Stream socket */
            hints.ai_flags = 0;
            hints.ai_protocol = IPPROTO_TCP;          /* Allow only TCP */

            // resolve the address 
            if (getaddrinfo(destination.address.c_str() , std::to_string(destination.port).c_str() , &hints, &result) != 0)
                return -1;

            // try to connect to a possible one of the resolution results
            for (rp = result; rp != NULL; rp = rp->ai_next) {
               socketFD = socket(rp->ai_family, rp->ai_socktype,
                            rp->ai_protocol);
               if (socketFD == -1)
                   continue;

               if (connect(socketFD, rp->ai_addr, rp->ai_addrlen) != -1)
                   break;                  /* Success */

               close(socketFD);
           }
		   free(result);
			
           if (rp == NULL)  {          /* No address succeeded */
               return -1;
		   }
#endif

        // receive the reachable destination from this sockets
        return socketFD;
    }

    int tryConnect(const ff_endpoint &destination){
        int fd = -1, retries = 0;

        while((fd = this->create_connect(destination)) < 0 && ++retries < MAX_RETRIES)
            if (retries < AGGRESSIVE_TRESHOLD)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                //std::this_thread::sleep_for(std::chrono::milliseconds((long)std::pow(2, retries - AGGRESSIVE_TRESHOLD)));

        return fd;
    }

    int sendToSck(int sck, message_t* task){
        task->sender = htonl(task->sender);
        task->chid = htonl(task->chid);

        size_t sz = htobe64(task->data.getLen());
        struct iovec iov[4];
        iov[0].iov_base = &task->sender;
        iov[0].iov_len = sizeof(task->sender);
        iov[1].iov_base = &task->chid;
        iov[1].iov_len = sizeof(task->chid);
        iov[2].iov_base = &sz;
        iov[2].iov_len = sizeof(sz);
        iov[3].iov_base = task->data.getPtr();
        iov[3].iov_len = task->data.getLen();

        if (writevn(sck, iov, 4) < 0){
            error("Error writing on socket\n");
            return -1;
        }

        return 0;
    }

     int waitAckFrom(int sck){
        while (socketsCounters[sck] == 0){
            for(auto& [sck_, counter] : socketsCounters){
                int r; ack_t a;
                if ((r = recvnnb(sck_, reinterpret_cast<char*>(&a), sizeof(ack_t))) != sizeof(ack_t)){
                    if (errno == EWOULDBLOCK){
                        assert(r == -1);
                        continue;
                    }
                    perror("recvnnb ack");
                    return -1;
                } else 
                    counter++;
                
            }
			
            if (socketsCounters[sck] == 0){
                tmpset = set;
                if (select(fdmax + 1, &tmpset, NULL, NULL, NULL) == -1){
                    perror("select");
                    return -1;
                }
            }
        }
        return 1;
    }

    int getMostFilledBufferSck(bool feedback){
        int sckMax = 0;
        int sizeMax = 0;
        for(auto& [sck, buffer] : batchBuffers){
            if ((feedback && buffer.ct != ChannelType::FBK) || (!feedback && buffer.ct != ChannelType::FWD)) continue; 
            if (buffer.size > sizeMax) sckMax = sck;
        }
    
        if (sckMax > 0) return sckMax;
        
        do {
        last_rr_socket = (last_rr_socket + 1) % this->sockets.size();
        } while (batchBuffers[sockets[last_rr_socket]].ct != (feedback ? ChannelType::FBK : ChannelType::FWD));
        return sockets[last_rr_socket];
        
    }

    
public:
    ff_dsender(std::pair<ChannelType, ff_endpoint> dest_endpoint, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1): precomputedRT(rt), gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {
        this->dest_endpoints.push_back(std::move(dest_endpoint));
    }

    ff_dsender( std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints_, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1) : dest_endpoints(std::move(dest_endpoints_)), precomputedRT(rt), gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {}

    

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
        
        FD_ZERO(&set);
        FD_ZERO(&tmpset);
		
        //sockets.resize(dest_endpoints.size());
        for(auto& [ct, ep] : this->dest_endpoints){
            int sck = tryConnect(ep);
            if (sck <= 0) return -1;
            sockets.push_back(sck);
            socketsCounters[sck] = messageOTF;
            batchBuffers.emplace(std::piecewise_construct, std::forward_as_tuple(sck), std::forward_as_tuple(this->batchSize, ct, [this, sck](struct iovec* v, int size) -> bool {
                
                if (this->socketsCounters[sck] == 0 && this->waitAckFrom(sck) == -1){
                    error("Errore waiting ack from socket inside the callback\n");
                    return false;
                }

                if (writevn(sck, v, size) < 0){
                    error("Error sending the iovector inside the callback!\n");
                    return false;
                }

                this->socketsCounters[sck]--;

                return true;
            }));

            // compute the routing table!
            for(int dest : precomputedRT->operator[](ep.groupName).first)
                dest2Socket[std::make_pair(dest, ct)] = sck;

            if (handshakeHandler(sck, ct) < 0) {
				error("svc_init ff_dsender failed");
				return -1;
			}

            FD_SET(sck, &set);
            if (sck > fdmax) fdmax = sck;
        }

        // we can erase the list of endpoints
        this->dest_endpoints.clear();

        return 0;
    }

    message_t *svc(message_t* task) {
        int sck;
        //if (task->chid == -1) task->chid = 0;
        if (task->chid != -1)
            sck = dest2Socket[{task->chid, (task->feedback ? ChannelType::FBK : ChannelType::FWD)}];
        else {
            sck = getMostFilledBufferSck(task->feedback); // get the most filled buffer socket or a rr socket
        }

        if (batchBuffers[sck].push(task) == -1) {
			return EOS;
		}

        return this->GO_ON;
    }

    void eosnotify(ssize_t id) {
        for (const auto& sck : sockets)
            batchBuffers[sck].push(new message_t(id, -2));

		if (++neos >= this->get_num_inchannels()) {
			// all input EOS received, now sending the EOS to all connections
            for(const auto& sck : sockets) {
				if (batchBuffers[sck].sendEOS()<0) {
					error("sending EOS to external connections (ff_dsender)\n");
				}										 
				shutdown(sck, SHUT_WR);
			}
		}
    }

	void svc_end() {
		// here we wait all acks from all connections
		size_t totalack = sockets.size()*messageOTF;
		size_t currentack = 0;
		for(const auto& [_, counter] : socketsCounters)
			currentack += counter;

		ack_t a;
		while(currentack<totalack) {
			for(auto scit = socketsCounters.begin(); scit != socketsCounters.end();) {
				auto sck      = scit->first;
				auto& counter = scit->second;
				
				switch(recvnnb(sck, (char*)&a, sizeof(a))) {
				case 0:
				case -1:
					if (errno==EWOULDBLOCK) { ++scit; continue; }
					currentack += (messageOTF-counter);
					socketsCounters.erase(scit++);
					break;
				default: {
					currentack++;
					counter++;
					++scit;
				}
				}
			}
		}
		for(auto& sck : sockets) close(sck);
	}
};


class ff_dsenderH : public ff_dsender {

    std::vector<int> internalSockets;
    int last_rr_socket_Internal = -1;
    int internalMessageOTF;
    bool squareBoxEOS = false;

    /*int getNextReadyInternal(){
        for(size_t i = 0; i < this->internalSockets.size(); i++){
            int actualSocketIndex = (last_rr_socket_Internal + 1 + i) % this->internalSockets.size();
            int sck = internalSockets[actualSocketIndex];
            if (socketsCounters[sck] > 0) {
                last_rr_socket_Internal = actualSocketIndex;
                return sck;
            }
        }

        int sck;
        decltype(internalSockets)::iterator it;

        do {
            sck = waitAckFromAny();   // FIX: error management!
			if (sck < 0) {
				error("waitAckFromAny failed in getNextReadyInternal");
				return -1;
			}
		} while ((it = std::find(internalSockets.begin(), internalSockets.end(), sck)) != internalSockets.end());
        
        last_rr_socket_Internal = it - internalSockets.begin();
        return sck;
    }*/

    int getMostFilledInternalBufferSck(){
         int sckMax = 0;
        int sizeMax = 0;
        for(int sck : internalSockets){
            auto& b = batchBuffers[sck];
            if (b.size > sizeMax) {
                sckMax = sck;
                sizeMax = b.size;
            }
        }
        if (sckMax > 0) return sckMax;

        last_rr_socket_Internal = (last_rr_socket_Internal + 1) % this->internalSockets.size();
        return internalSockets[last_rr_socket_Internal];
    }

public:

    ff_dsenderH(std::pair<ChannelType, ff_endpoint> e, precomputedRT_t* rt, std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsender(e, rt, gName, batchSize, messageOTF, coreid), internalMessageOTF(internalMessageOTF) {} 
    ff_dsenderH(std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints_, precomputedRT_t* rt, std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsender(dest_endpoints_, rt, gName, batchSize, messageOTF, coreid), internalMessageOTF(internalMessageOTF) {}

    int svc_init() {

        if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        FD_ZERO(&set);
        FD_ZERO(&tmpset);
		
        for(const auto& [ct, endpoint] : this->dest_endpoints){
            int sck = tryConnect(endpoint);
            if (sck <= 0) return -1;
            bool isInternal = ct == ChannelType::INT;
            if (isInternal) internalSockets.push_back(sck);
            else sockets.push_back(sck);
            socketsCounters[sck] = isInternal ? internalMessageOTF : messageOTF;
            batchBuffers.emplace(std::piecewise_construct, std::forward_as_tuple(sck), std::forward_as_tuple(this->batchSize, ct, [this, sck](struct iovec* v, int size) -> bool {
                
                if (this->socketsCounters[sck] == 0 && this->waitAckFrom(sck) == -1){
                    error("Errore waiting ack from socket inside the callback\n");
                    return false;
                }

                if (writevn(sck, v, size) < 0){
                    error("Error sending the iovector inside the callback (errno=%d) %s\n", errno);
                    return false;
                }

                this->socketsCounters[sck]--;

                return true;
            })); // change with the correct size

             for(int dest : precomputedRT->operator[](endpoint.groupName).first)
                dest2Socket[std::make_pair(dest, ct)] = sck;

            if (handshakeHandler(sck, ct) < 0) return -1;

            FD_SET(sck, &set);
            if (sck > fdmax) fdmax = sck;
        }

        // we can erase the list of endpoints
        this->dest_endpoints.clear();

        return 0;
    }

    message_t *svc(message_t* task) {
        if (this->get_channel_id() == (ssize_t)(this->get_num_inchannels() - 1)){
            int sck;
            // pick destination from the list of internal connections!
            if (task->chid != -1){ // roundrobin over the destinations
                sck = dest2Socket[{task->chid, ChannelType::INT}];
            } else
                sck = getMostFilledInternalBufferSck();


            if (batchBuffers[sck].push(task) == -1) {
				return EOS;
			}

            return this->GO_ON;
        }

        return ff_dsender::svc(task);
    }

     void eosnotify(ssize_t id) {
         if (id == (ssize_t)(this->get_num_inchannels() - 1)){
            // send the EOS to all the internal connections
            if (squareBoxEOS) return;
            squareBoxEOS = true;
            for(const auto& sck : internalSockets) {
                if (batchBuffers[sck].sendEOS()<0) {
					error("sending EOS to internal connections\n");
				}					
				shutdown(sck, SHUT_WR);
			}
		 }
		 if (++neos >= this->get_num_inchannels()) {
			 // all input EOS received, now sending the EOS to all
			 // others connections
			 for(const auto& sck : sockets) {
				 if (batchBuffers[sck].sendEOS()<0) {
					 error("sending EOS to external connections (ff_dsenderH)\n");
				 }										 
				 shutdown(sck, SHUT_WR);
			 }
		 }

        
	 }

	void svc_end() {
		// here we wait all acks from all connections
		size_t totalack  = internalSockets.size()*internalMessageOTF;
		totalack        += sockets.size()*messageOTF;
		size_t currentack = 0;
		for(const auto& [sck, counter] : socketsCounters) {
			currentack += counter;
		}
		
		ack_t a;
		while(currentack<totalack) {
			for(auto scit = socketsCounters.begin(); scit != socketsCounters.end();) {
				auto sck      = scit->first;
				auto& counter = scit->second;

				switch(recvnnb(sck, (char*)&a, sizeof(a))) {
				case 0:
				case -1: {
					if (errno == EWOULDBLOCK) {
						++scit;
						continue;
					}
					decltype(internalSockets)::iterator it;
					it = std::find(internalSockets.begin(), internalSockets.end(), sck);
					if (it != internalSockets.end())
						currentack += (internalMessageOTF-counter);
					else
						currentack += (messageOTF-counter);
					socketsCounters.erase(scit++);
				} break;
				default: {
					currentack++;
					counter++;
					++scit;					
				}					
				}
			}
		}
		for(const auto& [sck, _] : socketsCounters) close(sck);
	}
	
};


#endif
