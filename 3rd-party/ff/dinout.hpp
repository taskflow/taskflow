/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! \file dnode.hpp
 *  \brief Contains the definition of the \p ff_dnode class, which is an extension 
 *  of the base class \p ff_node, with features oriented to distributed systems.
 */
/* ***************************************************************************
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */
 
#ifndef FF_DINOUT_HPP
#define FF_DINOUT_HPP

#include <cstdlib>
#include <ff/dnode.hpp>



namespace ff {



/*!
 *  \class ff_dinout
 * \ingroup building_blocks
 *
 *  \brief A \ref ff::ff_dnode serving both Input and Output to the network
 *
 */

template <typename CommImplIn, typename CommImplOut>
class ff_dinout:public dnode<CommImplIn> {
public:
    typedef typename CommImpl::TransportImpl::msg_t msg_t;
    
    enum {SENDER=true, RECEIVER=false};

protected:    
    // this is called once as soon as the message just sent 
    // is no longer in use by the run-time
    static dnode_cbk_t cb;
    
    static inline void freeMsg(void * data, void * arg) {
        if (cb) cb(data,arg);
    }
    static inline void freeHdr(void * data, void *) {
        delete static_cast<uint32_t*>(data);
    }

    inline bool isEos(const char data[msg_t::HEADER_LENGHT]) const {
        return !(data[0]-'E' + data[1]-'O' + data[2]-'S' + data[3]-'\0');
    }
    
    /// Default constructor
    ff_dinout():ff_node(),skipdnode(true) {}
    
    /// Destructor: closes all connections.
    virtual ~ff_dinout() {         
        com.close();
        delete com.getDescriptor();
    }
    
    virtual inline bool push(void * ptr) { 
        if (skipdnode || !P) return ff_node::push(ptr);

        // gets the peers involved in one single communication
        const int peers=com.getDescriptor()->getPeers();
        if (ptr == (void*)FF_EOS) {
            //cerr << "DNODE prepare to send FF_DEOS to " << peers <<" peers\n";
            for(int i=0;i<peers;++i) {
                msg_t msg; 
                msg.init(FF_DEOS,msg_t::HEADER_LENGHT);
                //cerr << "DNODE sends FF_DEOS to " << i <<"\n";
                if (!com.put(msg,i)) return false;
            }
            return true;
        }
        
        if (CommImpl::MULTIPUT) {
            svector<iovec> v;
            for(int i=0;i<peers;++i) {
                v.clear();
                callbackArg.resize(0);
                prepare(v, ptr, i);
                
                msg_t hdr(new uint32_t(v.size()), msg_t::HEADER_LENGHT, freeHdr);
                com.putmore(hdr,i);
                callbackArg.resize(v.size());  
                for(size_t j=0;j<v.size()-1;++j) {
                    msg_t msg(v[j].iov_base, v[j].iov_len,freeMsg,callbackArg[j]); 
                    com.putmore(msg,i);
                }
                msg_t msg(v[v.size()-1].iov_base, v[v.size()-1].iov_len,freeMsg,callbackArg[v.size()-1]);
                if (!com.put(msg,i)) return false;            
            }
        } else {
            svector<iovec> v;
            callbackArg.resize(0);
            prepare(v, ptr);
                       
            msg_t hdr(new uint32_t(v.size()), msg_t::HEADER_LENGHT, freeHdr);
            com.putmore(hdr);
            callbackArg.resize(v.size());
            for(size_t j=0;j<v.size()-1;++j) {
                msg_t msg(v[j].iov_base, v[j].iov_len,freeMsg,callbackArg[j]); 
                com.putmore(msg);
            }
            msg_t msg(v[v.size()-1].iov_base, v[v.size()-1].iov_len,freeMsg,callbackArg[v.size()-1]);
            if (!com.put(msg)) return false;
        }
        return true;
    }
    

    virtual inline bool pop(void ** ptr) { 
        if (skipdnode || P) return ff_node::pop(ptr);

        // gets the peers involved in one single communication
        const int sendingPeers=com.getToWait();

        svector<msg_t*>* v[sendingPeers];
        for(int i=0;i<sendingPeers;++i) {
            msg_t hdr;
            int sender=-1;
            if (!com.gethdr(hdr, sender)) {
                error("dnode:pop: ERROR: receiving header from peer");
                return false;
            }
            if (isEos(static_cast<char *>(hdr.getData()))) {
                //std::cerr << "DNODE gets DEOS\n";
                if (++neos==com.getDescriptor()->getPeers()) {
                    com.done();
                    *ptr = (void*)FF_EOS;
                    neos=0;
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
                    return false;
                }
        }
        com.done();
        
        unmarshalling(v, sendingPeers, *ptr);
        return true;
    } 
    
public:

    int initIn(const std::string& name, const std::string& address,
             const int peers, typename CommImpl::TransportImpl* const transp, 
             const int nodeId=-1) {
        return In.init(name,address,peers,transp,false, nodeId);
    }
        
    // serialization/deserialization methods
    // The first prepare is used by the Producer (p=true in the init method)
    // whereas the second prepare and the unmarshalling methods are used
    // by the Consumer (p=false in the init method).


    virtual void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,sizeof(void*)};
        v.push_back(iov);
        setCallbackArg(NULL);
    }

    // COMMENT: 
    //  When using ZeroMQ (from zguide.zeromq.org):
    //   "There is no way to do zero-copy on receive: ØMQ delivers you a 
    //    buffer that you can store as long as you wish but it will not 
    //    write data directly into application buffers."
    //
    //
    
    /*
     *  Used to give to the run-time a pool of messages on which
     *  input message frames will be received
     *  \param len the number of input messages expected
     *  \param sender the message sender
     *  \param v vector containing the pool of messages
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
    
    /*
     *  This method is called once, when all frames composing the message have been received
     *  by the run-time. Within that method, it is possible to convert or re-arrange
     *  all the frames back to their original data or object layout. 
     */
    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
        task = v[0]->operator[](0)->getData();
        delete v[0];
    }

    /*
     * This methed can be used to pass an additional parameter (the 2nd one) 
     * to the callback function. Typically it is called in the prepare method of the
     * producer.
     */
    void setCallbackArg(void* arg) { callbackArg.push_back(arg);}
    
    /*
     *  Runs the \p dnode as a stand-alone thread.\n
     *  Typically, it should not be called by application code unless you want to have just
     *  a sequential \p dnode
     */
    int  run(bool=false) { return  ff_node::run(); }    

    /// Waits the thread to finish
    int  wait() { return ff_node::wait(); }    
    
    /* jumps the first pop from the input queue or from the input
     *  external channel. This is typically used in the first stage
     *  of a cyclic graph (e.g. the first stage of a torus pipeline)
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

} // namespace ff

#endif /* FF_DINOUT_HPP */
