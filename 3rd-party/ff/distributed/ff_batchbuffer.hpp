#ifndef FF_BATCHBUFFER_H
#define FF_BATCHBUFFER_H
#include "ff_network.hpp"
#include <sys/uio.h>
using namespace ff;

class ff_batchBuffer {	
    std::function<bool(struct iovec*, int)> callback;
    int batchSize;
    struct iovec iov[UIO_MAXIOV];
    std::vector<std::pair<size_t*, message_t*>> toCleanup;
public:
    int size = 0;
    ChannelType ct;
    ff_batchBuffer() {}
    ff_batchBuffer(int _size, ChannelType ct, std::function<bool(struct iovec*, int)> cbk) : callback(cbk), batchSize(_size), ct(ct) {
		if (_size*4+1 > UIO_MAXIOV){
            error("Size too big!\n");
            abort();
        }
        iov[0].iov_base = &(this->size);
        iov[0].iov_len = sizeof(int);
    }

    int push(message_t* m){
        m->sender = htonl(m->sender);
        m->chid = htonl(m->chid);
        size_t* sz = new size_t(htobe64(m->data.getLen()));


        int indexBase = size * 4;
        iov[indexBase+1].iov_base = &m->sender;
        iov[indexBase+1].iov_len = sizeof(int);
        iov[indexBase+2].iov_base = &m->chid;
        iov[indexBase+2].iov_len = sizeof(int);
        iov[indexBase+3].iov_base = sz;
        iov[indexBase+3].iov_len = sizeof(size_t);
        iov[indexBase+4].iov_base = m->data.getPtr();
        iov[indexBase+4].iov_len = m->data.getLen();
    
        toCleanup.push_back(std::make_pair(sz, m));

        if (++size == batchSize)
            return this->flush();
		return 0;
    }

    int sendEOS(){
        if (push(new message_t(0,0))<0) {
			error("pushing EOS");
		}
        return flush();
    }

    int flush(){
		
        if (size == 0) return 0;

        int size_ = size;
        size = htonl(size);
        
        if (!callback(iov, size_*4+1)) {
            error("Callback of the batchbuffer got something wrong!\n");
			return -1;
		}

		while (!toCleanup.empty()){
            delete toCleanup.back().first;
            delete toCleanup.back().second;
            toCleanup.pop_back();
        }

        size = 0;
		return 0;
    }

};

#endif
