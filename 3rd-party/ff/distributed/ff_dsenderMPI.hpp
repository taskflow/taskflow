#ifndef FF_DSENDER_MPI_H
#define FF_DSENDER_MPI_H

#include <iostream>
#include <map>
#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <sys/types.h>
#include <netdb.h>
#include <cmath>
#include <thread>
#include <mpi.h>

#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>

using namespace ff;
using precomputedRT_t = std::map<std::string, std::pair<std::vector<int>, ChannelType>>;
class ff_dsenderMPI: public ff_minode_t<message_t> { 
protected:
    class batchBuffer {
    protected:    
        int rank;
        bool blocked = false;
        size_t size_, actualSize = 0;
        std::vector<char> buffer;
        std::vector<long> headers;
        MPI_Request headersR, datasR;
    public:
        batchBuffer(size_t size_, int rank) : rank(rank), size_(size_){
            headers.reserve(size_*3+1);
        }
        virtual void waitCompletion(){
            if (blocked){
                MPI_Wait(&headersR, MPI_STATUS_IGNORE);
                MPI_Wait(&datasR, MPI_STATUS_IGNORE);
                headers.clear();
                buffer.clear();
                blocked = false;
            }
        }

        virtual size_t size() {return actualSize;}
        virtual int push(message_t* m){
            waitCompletion();
            int idx = 3*actualSize++;
            headers[idx+1] = m->sender;
            headers[idx+2] = m->chid;
            headers[idx+3] = m->data.getLen();

            buffer.insert(buffer.end(), m->data.getPtr(), m->data.getPtr() + m->data.getLen());

            delete m;
            if (actualSize == size_) {
                this->flush();
                return 1;
            }
            return 0;
        }

        virtual void flush(){
            headers[0] = actualSize;
            MPI_Isend(headers.data(), actualSize*3+1, MPI_LONG, rank, DFF_HEADER_TAG, MPI_COMM_WORLD, &headersR);
            MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD, &datasR);
            blocked = true;
            actualSize = 0;
        }

        virtual void pushEOS(){
            int idx = 3*actualSize++;
            headers[idx+1] = 0; headers[idx+2] = 0; headers[idx+3] = 0;

            this->flush();
        }
    };

    class directBatchBuffer : public batchBuffer {
            message_t* currData = NULL;
            long currHeader[4] = {1, 0, 0, 0}; 
        public:
            directBatchBuffer(int rank) : batchBuffer(0, rank){}
            void pushEOS(){
                waitCompletion();
                currHeader[1] = 0; currHeader[2] = 0; currHeader[3] = 0;
                MPI_Send(currHeader, 4, MPI_LONG, this->rank, DFF_HEADER_TAG, MPI_COMM_WORLD);
            }
            void flush() {}
            void waitCompletion(){
                if (blocked){
                    MPI_Wait(&headersR, MPI_STATUS_IGNORE);
                    if (currData->data.getLen() > 0)
                        MPI_Wait(&datasR, MPI_STATUS_IGNORE);
                    if (currData) delete currData;
                    blocked = false;
                }
            }
            int push(message_t* m){
                waitCompletion();
                currHeader[1] = m->sender; currHeader[2] = m->chid; currHeader[3] = m->data.getLen();
                MPI_Isend(currHeader, 4, MPI_LONG, this->rank, DFF_HEADER_TAG, MPI_COMM_WORLD, &this->headersR);
                if (m->data.getLen() > 0)
                    MPI_Isend(m->data.getPtr(), m->data.getLen(), MPI_BYTE, rank, DFF_TASK_TAG, MPI_COMM_WORLD, &datasR);
                currData = m;
                blocked = true;
                return 1;
            }
    };
    size_t neos=0;
    precomputedRT_t* rt;
    int last_rr_rank = 0; //next destiation to send for round robin policy
    std::map<std::pair<int, ChannelType>, int> dest2Rank;
    std::map<int, std::pair<int, std::vector<batchBuffer*>>> buffers;
    std::vector<std::pair<int, ChannelType>> ranks;
    std::vector<std::pair<ChannelType, ff_endpoint>> destRanks;
    std::string gName;
    int batchSize;
    int messageOTF;
	int coreid;

    virtual int handshakeHandler(const int rank, ChannelType ct){
        MPI_Send(gName.c_str(), gName.size(), MPI_BYTE, rank, DFF_GROUP_NAME_TAG, MPI_COMM_WORLD);
        MPI_Send(&ct, sizeof(ChannelType), MPI_BYTE, rank, DFF_CHANNEL_TYPE_TAG, MPI_COMM_WORLD);
        return 0;
    }

    int getMostFilledBufferRank(bool feedback){
        int rankMax = -1;
        size_t sizeMax = 0;
        for(auto& [rank,ct] : ranks){
            if ((feedback && ct != ChannelType::FBK) || (!feedback && ct != ChannelType::FWD)) continue;
            auto& batchBB = buffers[rank];
            size_t sz = batchBB.second[batchBB.first]->size();
            if (sz > sizeMax) {
                rankMax = rank;
                sizeMax = sz;
            }
        }
        if (rankMax >= 0) return rankMax;

        do {
            last_rr_rank = (last_rr_rank + 1) % this->ranks.size();
        } while (this->ranks[last_rr_rank].second != (feedback ? ChannelType::FBK : ChannelType::FWD));
        return this->ranks[last_rr_rank].first; 
    }

public:
    ff_dsenderMPI(std::pair<ChannelType, ff_endpoint> destRank, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1)
		: rt(rt), gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {
        this->destRanks.push_back(std::move(destRank));
    }

    ff_dsenderMPI( std::vector<std::pair<ChannelType, ff_endpoint>> destRanks_, precomputedRT_t* rt, std::string gName = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int coreid=-1)
		: rt(rt), destRanks(std::move(destRanks_)), gName(gName), batchSize(batchSize), messageOTF(messageOTF), coreid(coreid) {}

    int svc_init() {
		if (coreid!=-1)
			ff_mapThreadToCpu(coreid);

        for(auto& [ct, ep]: this->destRanks){
           handshakeHandler(ep.getRank(), ct);
           ranks.push_back({ep.getRank(), ct});
           std::vector<batchBuffer*> appo;
           for(int i = 0; i < messageOTF; i++) appo.push_back(batchSize == 1 ? new directBatchBuffer(ep.getRank()) : new batchBuffer(batchSize, ep.getRank()));
           buffers.emplace(std::make_pair(ep.getRank(), std::make_pair(0, std::move(appo))));

           for(int dest : rt->operator[](ep.groupName).first)
                dest2Rank[std::make_pair(dest, ct)] = ep.getRank();
        }

         this->destRanks.clear();

        return 0;
    }

    void svc_end(){
        for(auto& [rank, bb] : buffers)
            for(auto& b : bb.second) b->waitCompletion();
    }
  
    message_t *svc(message_t* task) {
        int rank;
        if (task->chid != -1)
            rank = dest2Rank[{task->chid, task->feedback ? ChannelType::FBK : ChannelType::FWD}]; 
        else 
            rank = getMostFilledBufferRank(task->feedback);
        
        auto& buffs = buffers[rank];
        assert(buffs.second.size() > 0);
        if (buffs.second[buffs.first]->push(task)) // the push triggered a flush, so we must go ion the next buffer
            buffs.first = (buffs.first + 1) % buffs.second.size(); // increment the used buffer of 1
    
        return this->GO_ON;
    }

     void eosnotify(ssize_t id) {
        for (auto& [rank, _] : ranks){
             auto& buffs = buffers[rank];
            buffs.second[buffs.first]->push(new message_t(id, -2));
        }
        
	    if (++neos >= this->get_num_inchannels())
            for(auto& [rank, ct] : ranks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
            }       
    }
};


class ff_dsenderHMPI : public ff_dsenderMPI {
    std::vector<int> internalRanks;
    int last_rr_rank_Internal = -1;
    int internalMessageOTF;
    bool squareBoxEOS = false;

    int getMostFilledInternalBufferRank(){
        int rankMax = -1;
        size_t sizeMax = 0;
        for(int rank : internalRanks){
            auto& batchBB = buffers[rank];
            size_t sz = batchBB.second[batchBB.first]->size();
            if (sz > sizeMax) {
                rankMax = rank;
                sizeMax = sz;
            }
        }
        if (rankMax >= 0) return rankMax;

        last_rr_rank_Internal = (last_rr_rank_Internal + 1) % this->internalRanks.size();
        return internalRanks[last_rr_rank_Internal];
    }

public:

    ff_dsenderHMPI(std::pair<ChannelType, ff_endpoint> e, precomputedRT_t* rt,  std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMPI(e, rt, gName, batchSize, messageOTF, coreid), internalMessageOTF(internalMessageOTF) {} 
    ff_dsenderHMPI(std::vector<std::pair<ChannelType, ff_endpoint>> dest_endpoints_, precomputedRT_t* rt, std::string gName  = "", int batchSize = DEFAULT_BATCH_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF, int internalMessageOTF = DEFAULT_INTERNALMSG_OTF, int coreid=-1) : ff_dsenderMPI(dest_endpoints_, rt, gName, batchSize, messageOTF, coreid), internalMessageOTF(internalMessageOTF) {}

    int svc_init() {

        if (coreid!=-1)
			ff_mapThreadToCpu(coreid);
		
        for(auto& [ct, endpoint] : this->destRanks){
            int rank = endpoint.getRank();
            bool isInternal = ct == ChannelType::INT;
            if (isInternal) 
                internalRanks.push_back(rank);
            else
                ranks.push_back({rank, ct});

            std::vector<batchBuffer*> appo;
            for(int i = 0; i < (isInternal ? internalMessageOTF : messageOTF); i++) appo.push_back(batchSize == 1 ? new directBatchBuffer(rank) : new batchBuffer(batchSize, rank));
            buffers.emplace(std::make_pair(rank, std::make_pair(0, std::move(appo))));
            
            if (handshakeHandler(rank, ct) < 0) return -1;

            for(int dest : rt->operator[](endpoint.groupName).first)
                dest2Rank[std::make_pair(dest, ct)] = rank;

        }

        this->destRanks.clear();

        return 0;
    }

    message_t *svc(message_t* task) {
        if (this->get_channel_id() == (ssize_t)(this->get_num_inchannels() - 1)){
            int rank;
        
            // pick destination from the list of internal connections!
            if (task->chid != -1){ // roundrobin over the destinations
                rank = dest2Rank[{task->chid, ChannelType::INT}];
            } else
                rank = getMostFilledInternalBufferRank();

            auto& buffs = buffers[rank];
            if (buffs.second[buffs.first]->push(task)) // the push triggered a flush, so we must go ion the next buffer
                buffs.first = (buffs.first + 1) % buffs.second.size(); // increment the used buffer of 1

            return this->GO_ON;
        }
        
        return ff_dsenderMPI::svc(task);
    }

     void eosnotify(ssize_t id) {
         if (id == (ssize_t)(this->get_num_inchannels() - 1)){
            // send the EOS to all the internal connections
            if (squareBoxEOS) return;
            squareBoxEOS = true;
            for(const auto&rank : internalRanks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
            }
		 }
        if (++neos >= this->get_num_inchannels())
            for(auto& [rank, ct] : ranks){
                auto& buffs = buffers[rank];
                buffs.second[buffs.first]->pushEOS();
            }      
	 }
	
};

#endif
