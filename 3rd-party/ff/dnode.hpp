/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \file dnode.hpp
 *  \ingroup building_blocks
 *  
 *  \brief Defines the FastFlow bridge node toward the network (distributed)
 *
 *  It contains the definition of the \p ff_dnode class, which is an
 *  extension of the base class \p ff_node, with features oriented to
 *  distributed systems.
 *
 * \note MarcoA: This documentation has been written by a student
 * and has not been checked, it should not be considered reliable
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

 
#ifndef FF_DNODE_HPP
#define FF_DNODE_HPP

//#include <sys/uio.h>

#if defined(_WIN32)
#include <WinSock2.h>
#endif

#include <stdint.h>
#include <cstdlib>
#include <vector>
#include <ff/platforms/platform.h>
#if !defined(_WIN32)
 #include <sys/uio.h>
#endif
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>
#include <ff/node.hpp>
#include <ff/svector.hpp>


namespace ff {

// the header is msg_t::HEADER_LENGTH bytes
static const char FF_DEOS[]="EOS";

typedef void (*dnode_cbk_t) (void *,void*);


/*!
 *  \class ff_dnode
 *  \ingroup building_blocks
 *
 *  \brief Defines ff::ff_dnode
 *
 *  This class represents the distributed version of \p ff_node. A \p ff_dnode
 *  is actually a \p ff_node with an extra communication channel (<em>external
 *  channel</em>), which connects the edge-node of the graph with one or more
 *  edge-nodes of other FastFlow application graphs running on the same host or
 *  on different host(s).
 *
 *  It is implemented as a template: the template type \p CommImpl refers to
 *  the communication pattern that the programmer wishes to use to connect
 *  different \p ff_dnodes (i.e. unicast, broadcast, scatter, ondemand,
 *  fromAll, fromAny).
 *
 *
 */

template <typename CommImpl>
class ff_dnode: public ff_node {
public:
    /**
     * Implements communication transport layer
     */
    typedef typename CommImpl::TransportImpl::msg_t msg_t;
    
    /**
     * \brief Defines sender and receiver
     */
    enum {SENDER=true, RECEIVER=false};

protected:    
    // this is called once as soon as the message just sent 
    // is no longer in use by the run-time
    static dnode_cbk_t cb;
    
    /**
     * \brief Callback to free the sent message
     *
     * \param data pointer to the data to be freed
     * \param arg arguments to be passed to the callback function
     *
     */
    static inline void freeMsg(void * data, void * arg) {
        if (cb) cb(data,arg);
    }

    /**
     * \brief Frees header
     *
     * It frees the header of the message
     *
     * \param data is pointer to the header
     *
     */
    static inline void freeHdr(void * data, void *) {
        delete static_cast<uint32_t*>(data);
    }

    /**
     * \brief Checks EOS
     *
     * It checks if it is EOS of the message.
     * 
     * \param data is the array of messages
     *
     * \return The EOS of the message
     */
    inline bool isEos(const char data[msg_t::HEADER_LENGTH]) const {
        return !(data[0]-'E' + data[1]-'O' + data[2]-'S' + data[3]-'\0');
    }
    
    /**
     * \brief Constructor
     *
     */
    ff_dnode():ff_node(),skipdnode(true),neos(0) {}
    
    /**
     * \brief Destructor
     *
     */
    virtual ~ff_dnode() {
        com.close();
        delete com.getDescriptor();
    }

    template<typename CI>
    inline bool internal_push(void * ptr, CI& comm) { 
        // gets the peers involved in one single communication
        const int peers=comm.getDescriptor()->getPeers();
        if (ptr == (void*)FF_EOS) {
            //std::cerr << "DNODE prepare to send FF_DEOS to " << peers <<" peers\n";
            for(int i=0;i<peers;++i) {
                msg_t msg; 
                msg.init(FF_DEOS,msg_t::HEADER_LENGTH);
                if (!comm.put(msg,i)) return false;
                //std::cerr << "DNODE sends FF_DEOS to " << i <<"\n";
            }
            return true;
        }
        
        if (CI::MULTIPUT) {
            svector<iovec> v;
            for(int i=0;i<peers;++i) {
                v.clear();
                callbackArg.resize(0);
                prepare(v, ptr, i);
                
                msg_t hdr(new uint32_t(v.size()), msg_t::HEADER_LENGTH, freeHdr);
                comm.putmore(hdr,i);
                callbackArg.resize(v.size());  
                for(size_t j=0;j<v.size()-1;++j) {
                    msg_t msg(v[j].iov_base, v[j].iov_len,freeMsg,callbackArg[j]); 
                    comm.putmore(msg,i);
                }
                msg_t msg(v[v.size()-1].iov_base, v[v.size()-1].iov_len,freeMsg,callbackArg[v.size()-1]);
                if (!comm.put(msg,i)) return false;            
            }
        } else {
            svector<iovec> v;
            callbackArg.resize(0);
            prepare(v, ptr);
                       
            msg_t hdr(new uint32_t(v.size()), msg_t::HEADER_LENGTH, freeHdr);
            comm.putmore(hdr);
            callbackArg.resize(v.size());
            for(size_t j=0;j<v.size()-1;++j) {
                msg_t msg(v[j].iov_base, v[j].iov_len,freeMsg,callbackArg[j]); 
                comm.putmore(msg);
            }
            msg_t msg(v[v.size()-1].iov_base, v[v.size()-1].iov_len,freeMsg,callbackArg[v.size()-1]);
            if (!comm.put(msg)) return false;
        }
        return true;
    }

    /**
     * \brief push ff::ff_dnode -> ff::ff_node (shared-memory channel)
     *
     * It overrides \p ff_node's \p push method 
     *
     * \paarm ptr points to the data
     *
     */
    virtual inline bool push(void * ptr) { 
        if (skipdnode || !P) return ff_node::push(ptr);
        return internal_push(ptr, com);
    }

    /**
     * \brief pop  ff::ff_node -> ff::ff_dnode (shared-memory channel)
     *
     * It overrides \p ff_node 's \p pop method.
     *
     * \param ptr points to the data
     *
     * \return If successful \p true is returned, otherwise \p false.
     */
    virtual inline bool pop(void ** ptr) { 
        if (skipdnode || P) return ff_node::pop(ptr);

        // gets the peers involved in one single communication
        const int sendingPeers=com.getToWait();
#ifdef _WIN32
        svector<msg_t*> ** v = (svector<msg_t*> **) malloc(sendingPeers*sizeof(svector<msg_t*> *));
#else // C99
        svector<msg_t*>* v[sendingPeers];
#endif
        for(int i=0;i<sendingPeers;++i) {
            msg_t hdr;
            int sender=-1;
            if (!com.gethdr(hdr, sender)) {
                error("dnode:pop: ERROR: receiving header from peer");
                // free (v); // Win
                return false;
            }
            if (isEos(static_cast<char *>(hdr.getData()))) {
                //std::cerr << "DNODE gets DEOS\n";
                if (++neos==com.getDescriptor()->getPeers()) {
                    com.done();
                    *ptr = (void*)FF_EOS;
                    neos=0;
                    // free (v) // Win
                    return true;
                }
                if (sendingPeers==1) i=-1; // reset for index
                continue;
            }
            uint32_t len = *static_cast<uint32_t*>(hdr.getData());
            int ventry   = (sendingPeers==1)?0:sender;
            prepare(v[ventry], len, sender);
            assert(v[ventry]->size() == len);
            
            for(size_t j=0;j<len;++j)
                if (!com.get(*(v[ventry]->operator[](j)),sender)) {
                    error("dnode:pop: ERROR: receiving data from peer");
                    // free (v) // Win32
                    return false;
                }
        }
        com.done();
        
        unmarshalling(v, sendingPeers, *ptr);
        // free (v) // Win32
        return true;
    } 
    
    virtual inline bool Push(void *ptr, unsigned long retry=((unsigned long)-1), unsigned long ticks=(ff_node::TICKS2WAIT)) {
        if (skipdnode || !P) return ff_node::Push(ptr,retry,ticks);
        return push(ptr);
    }
    
    virtual inline bool Pop(void **ptr, unsigned long retry=((unsigned long)-1), unsigned long ticks=(ff_node::TICKS2WAIT)) {        
        if (skipdnode || P) return ff_node::Pop(ptr,retry,ticks);
        return pop(ptr);
    }

public:
    /**
     *  \brief Initializes distributed communication channel
     *
     *
     *  \param name is the name of the channel
     *  \param address is the address where to listen or connect to
     *  \param peers is the number of peers
     *  \param transp is the transport layer to be used
     *  \param p is a flag saying whether the current \a dnode is the producer (\p p \p = \p true) 
     *               or the consumer (\p p \p = \p false) w.r.t. the communication pattern used
     *  \param nodeId is the ID of the node
     *  \param cbk is the callback function that will be called once when a message just sent 
     *                 is no longer in use by the run-time
     *
     *  \return The status of \p com.init() as an integer value
     */
    int init(const std::string& name, const std::string& address,
             const int peers, typename CommImpl::TransportImpl* const transp, 
             const bool p, const int nodeId=-1, dnode_cbk_t cbk=0) {
        if (!p && cbk) {
            error("dnode:init: WARNING: callback does not make sense for consumer end-point, ignoring it...\n");
            cbk=NULL;
        }
        skipdnode=false;
        P=p; 
        neos=0;
        ff_dnode::cb=cbk;                               
        if (P) ff_node::create_output_buffer(1,true);            
        else ff_node::create_input_buffer(1);
        com.setDescriptor(new typename CommImpl::descriptor(name,peers,transp,P));
        return com.init(address,nodeId);
    }
        
    /*
     * serialization/deserialization methods
     * The first prepare is used by the Producer (p=true in the init method)
     * whereas the second prepare and the unmarshalling methods are used
     * by the Consumer (p=false in the init method).
     */

    /** 
     *  \brief Prepares output message
     *
     *  It is used to prepare (non contiguous) output messages 
     *
     *  \param v is a vector containing the pool of messages
     *  \param ptr is pointer to the data
     *  \param sender is the sender of the message
     */
    virtual void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,sizeof(void*)};
        v.push_back(iov);
        setCallbackArg(NULL);
    }

    /*
     * COMMENT: 
     * When using ZeroMQ (from zguide.zeromq.org): "There is no way to do
     * zero-copy on receive: ØMQ delivers you a buffer that you can store as
     * long as you wish but it will not write data directly into application
     * buffers."
     */
    
    /** 
     *  \brief Prepare a pool of messages
     *
     *  It is used to give to the run-time a pool of messages on which
     *  input message frames will be received
     *
     *  \param len is the number of input messages expected
     *  \param sender is the sender of the message
     *  \param v vector contains the pool of messages
     */
    virtual void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
        }
        v = v2;
    }
    
    /**
     *  \brief Unmarshalling
     *
     *  It is called once, when all frames composing the message
     *  have been received by the run-time. Within that method, it is possible
     *  to convert or re-arrange all the frames back to their original data or
     *  object layout. 
     *
     *  \param v is vector of messages
     *  \param vlen is the length of the vector
     *  \param task pointer to the task
     */
    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
        task = v[0]->operator[](0)->getData();
        delete v[0];
    }

    /**
     * \brief Sets call back arguments
     *
     * It is used to pass an additional parameter (the 2nd one)
     * to the callback function. Typically it is called in the prepare method
     * of the producer.
     *
     * \param arg points to the arguments to be pussed to the callback
     */
    void setCallbackArg(void* arg) { callbackArg.push_back(arg);}
    
    /** 
     *  \brief Executes the FastFlow dnode
     *
     *  It runs the \p dnode as a stand-alone thread. Typically, it should not
     *  be called by application code unless you want to have just a sequential
     *  \p dnode .
     *
     *  \return The status of the \p run() method.
     */
    int  run(bool=false) { return  ff_node::run(); }

    /**
     * \brief Waits the FastFlow node
     *
     * It waits the thread to finish.
     *
     * \return The status of \p wait() method.
     */
    int  wait() { return ff_node::wait(); }
    
    /**
     * \brief Skips first pop
     *
     * It jumps the first pop from the input queue or from the
     * input external channel. This is typically used in the first stage of a
     * cyclic graph (e.g. the first stage of a torus pipeline).
     *
     * \param sk is boolean value showing whether skifirstpop should be skipped
     * or not
     */
    void skipfirstpop(bool sk)   { ff_node::skipfirstpop(sk); }

protected:
    bool     skipdnode;
    bool     P;   
    int      neos;
    svector<void*> callbackArg;
    CommImpl com;
};
template <typename CommImpl>
dnode_cbk_t ff_dnode<CommImpl>::cb=0;

/*!
 *  \class ff_dinout
 *  \ingroup streaming_network_arbitrary_distributed_memory
 *
 *  \brief A \p ff_dnode with both input and output channels.
 *
 *  A \p ff_dinout is actually a \p ff_dnode with an extra communication
 *  channel (<em>external channel</em>), so that the dinout node is connected
 *  with the "external world" both with input and output channels.
 *
 *  It is implemented as a template class: the template type \p CommImplIn
 *  refers to the input communication pattern, the type \p CommImplOut refers
 *  to the output communication pattern.
 *
 *  This class is defined in \ref dnode.hpp
 *
 */
template <typename CommImplIn, typename CommImplOut>
class ff_dinout: public ff_dnode<CommImplIn> {
protected:

    /**
     * Defines the output communication
     */
    typedef typename CommImplOut::TransportImpl::msg_t msg_t;

    /**
     * \brief Destructor
     *
     * It closes all connections.
     */
    virtual ~ff_dinout() {
        comOut.close();
        delete comOut.getDescriptor();
    }
    
    /** 
     * \brief Pushes the task
     *
     * It overrides \p ff_dnode's \p push method.
     *
     * \param ptr points to the task
     *
     * \return The status of \p internal_push() method.
     */
    virtual inline bool push(void * ptr) { 
        return ff_dnode<CommImplIn>::internal_push(ptr, comOut);
    }
    
public:
    /**
     * \brief Initializes input communication channels
     *
     * It initializes all the input communication channels.
     *
     * \param name is the string to be passed through the communication channel
     * \param address is the address of the destination
     * \param peers are the number of peers involved in the communication
     * \param transp is the transport for the communication
     * \param nodeID is the identifer of the node
     *
     * \return The status of \p com.init() method.
     */
    int initIn(const std::string& name, const std::string& address,
             const int peers, typename CommImplIn::TransportImpl* const transp,
             const int nodeId=-1) {

        ff_dnode<CommImplIn>::skipdnode=false;
        ff_dnode<CommImplIn>::P=ff_dnode<CommImplIn>::RECEIVER;
        ff_dnode<CommImplIn>::ff_node::create_input_buffer(1);
        ff_dnode<CommImplIn>::com.setDescriptor(new typename CommImplIn::descriptor(name,peers,transp,
                                                              ff_dnode<CommImplIn>::RECEIVER));
        return ff_dnode<CommImplIn>::com.init(address,nodeId); 
    }

    /**
     * \brief Initializes the output communicaiton channels
     *
     * It initializes all the output communication channels.
     *
     * \param name is the string to be sent
     * \param address is the address of the destination
     * \param peers are the number of peers involve in the communication
     * \param transp is the transport address
     * \param nodeId is the identifier of the node
     * \param cbk is the callback
     *
     * \return The status of \p comOut.init() method.
     */
    int initOut(const std::string& name, const std::string& address,
                const int peers, typename CommImplOut::TransportImpl* const transp, 
                const int nodeId=-1, dnode_cbk_t cbk=0) {

        ff_dnode<CommImplOut>::skipdnode=false;
        ff_dnode<CommImplOut>::cb=cbk;  
        ff_dnode<CommImplOut>::ff_node::create_output_buffer(1,true);
        comOut.setDescriptor(new typename CommImplOut::descriptor(name,peers,transp,ff_dnode<CommImplOut>::SENDER));
        
        return comOut.init(address,nodeId);
    }

protected:
    CommImplOut comOut;
};

} // namespace ff

#endif /* FF_DNODE_HPP */
