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

#ifndef FF_NETWORK
#define FF_NETWORK

#include <sstream>
#include <iostream>
#include <exception>
#include <string>
#include <utility>
#include <sys/socket.h>
#include <sys/un.h>

#include <sys/types.h>
#include <sys/uio.h>
#include <arpa/inet.h>

#define REMOTE

#ifdef __APPLE__
    #include <libkern/OSByteOrder.h>
    #define htobe16(x) OSSwapHostToBigInt16(x)
    #define htole16(x) OSSwapHostToLittleInt16(x)
    #define be16toh(x) OSSwapBigToHostInt16(x)
    #define le16toh(x) OSSwapLittleToHostInt16(x)

    #define htobe32(x) OSSwapHostToBigInt32(x)
    #define htole32(x) OSSwapHostToLittleInt32(x)
    #define be32toh(x) OSSwapBigToHostInt32(x)
    #define le32toh(x) OSSwapLittleToHostInt32(x)

    #define htobe64(x) OSSwapHostToBigInt64(x)
    #define htole64(x) OSSwapHostToLittleInt64(x)
    #define be64toh(x) OSSwapBigToHostInt64(x)
    #define le64toh(x) OSSwapLittleToHostInt64(x)

    #ifndef UIO_MAXIOV
    #define UIO_MAXIOV 1023
    #endif
#endif

enum Proto {TCP , MPI};
enum ChannelType {FWD, INT, FBK};

class dataBuffer: public std::stringbuf {
public:	
    dataBuffer()
        : std::stringbuf(std::ios::in | std::ios::out | std::ios::binary) {
	}
    
    dataBuffer(char p[], size_t len, bool cleanup=false)
        : std::stringbuf(std::ios::in | std::ios::out | std::ios::binary),
		  len(len),cleanup(cleanup) {
        setg(p, p, p + len);
    }

	~dataBuffer() {
		if (cleanup) {
			cleanup = false;
			if (freetaskF) {
				freetaskF(getPtr());
			} else
				delete [] getPtr();
		}		
	}

    void setBuffer(char p[], size_t len, bool cleanup=true){
        setg(p, p, p+len);
        this->len = len;
        this->cleanup = cleanup;
    }

	size_t getLen() const {
		if (len>=0) return len;
		return str().length();
	}
	char* getPtr() const {
		return eback();
	}

	void doNotCleanup(){
		cleanup = false;
	}

	std::function<void(void*)> freetaskF;
	
protected:	
	ssize_t len=-1;
	bool cleanup = false;
};

using ffDbuffer = std::pair<char*, size_t>;

struct message_t {
	message_t(){}
    message_t(int sender, int chid) : sender(sender), chid(chid) {}
	message_t(char *rd, size_t size, bool cleanup=true) : data(rd,size,cleanup){}
	
	int           sender;
	int           chid;
    bool          feedback = false;
	dataBuffer    data;
};

struct ack_t {
    char ack = 'A';
};

struct ff_endpoint {
    ff_endpoint(){}
    ff_endpoint(std::string addr, int port) : address(std::move(addr)), port(port) {}
    ff_endpoint(int rank) : port(rank) {}
    int getRank() const {return port;}
	std::string address, groupName;
	int port;
};


struct FF_Exception: public std::runtime_error {FF_Exception(const char* err) throw() : std::runtime_error(err) {}};

ssize_t readn(int fd, char *ptr, size_t n) {  
    size_t   nleft = n;
    ssize_t  nread;

    while (nleft > 0) {
        if((nread = read(fd, ptr, nleft)) < 0) {
            if (nleft == n) return -1; /* error, return -1 */
            else break; /* error, return amount read so far */
        } else if (nread == 0) break; /* EOF */
        nleft -= nread;
        ptr += nread;
    }
    return(n - nleft); /* return >= 0 */
}

ssize_t readvn(int fd, struct iovec *v, int count){
    ssize_t rread;
    for (int cur = 0;;) {
        rread = readv(fd, v+cur, count-cur);
        if (rread <= 0) return rread; // error or closed connection
        while (cur < count && rread >= (ssize_t)v[cur].iov_len)
            rread -= v[cur++].iov_len;
        if (cur == count) return 1; // success!!
        v[cur].iov_base = (char *)v[cur].iov_base + rread;
        v[cur].iov_len -= rread;
    }
	return -1;
}

ssize_t writen(int fd, const char *ptr, size_t n) {  
    size_t   nleft = n;
    ssize_t  nwritten;
    
    while (nleft > 0) {
        if((nwritten = write(fd, ptr, nleft)) < 0) {
            if (nleft == n) return -1; /* error, return -1 */
            else break; /* error, return amount written so far */
        } else if (nwritten == 0) break; 
        nleft -= nwritten;
        ptr   += nwritten;
    }
    return(n - nleft); /* return >= 0 */
}

ssize_t writevn(int fd, struct iovec *v, int count){
    ssize_t written;
    for (int cur = 0;;) {
        written = writev(fd, v+cur, count-cur);
        if (written < 0) return -1;
        while (cur < count && written >= (ssize_t)v[cur].iov_len)
            written -= v[cur++].iov_len;
        if (cur == count) return 1; // success!!
        v[cur].iov_base = (char *)v[cur].iov_base + written;
        v[cur].iov_len -= written;
    }
	return -1;
}

static inline ssize_t recvnnb(int fd, char *buf, size_t size) {
    size_t left = size;
    int r;
    while(left>0) {
      if ((r=recv(fd ,buf,left,MSG_DONTWAIT)) == -1) {
	    if (errno == EINTR) continue;
	    if (left == size) return -1;
	    break;
      }
      
      if (r == 0) return 0;   // EOF
      left -= r;
      buf  += r;
    }
    return (size-left);
}


/*
    MPI DEFINES 
*/
#ifdef DFF_MPI
    #define DFF_ROUTING_TABLE_REQUEST_TAG 9
    #define DFF_ROUTING_TABLE_TAG 2
    #define DFF_TASK_TAG 3
    #define DFF_HEADER_TAG 4
    #define DFF_ACK_TAG 5
    #define DFF_GROUP_NAME_TAG 6
    #define DFF_CHANNEL_TYPE_TAG 7

    #define DFF_REQUEST_ROUTING_TABLE 10
#endif

#endif
