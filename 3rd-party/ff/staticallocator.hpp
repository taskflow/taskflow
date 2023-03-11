/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

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
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc., 59
 *  Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this file
 *  does not by itself cause the resulting executable to be covered by the GNU
 *  General Public License.  This exception does not however invalidate any
 *  other reasons why the executable file might be covered by the GNU General
 *  Public License.
 *
 * **************************************************************************/

#ifndef FF_STATIC_ALLOCATOR_HPP
#define FF_STATIC_ALLOCATOR_HPP

#include <sys/mman.h>
#include <cstdlib>
#include <cstdio>
#include <ff/config.hpp>
#include <new>

/* Author: Massimo Torquati
 * December 2020
 */


namespace ff {

class StaticAllocator {
public:
	StaticAllocator(const size_t _nslot, const size_t slotsize, const int nchannels=1):
		ssize(slotsize), nchannels(nchannels), cnts(nchannels,0), segment(nullptr) {

        assert(nchannels>0);
        assert(slotsize>0);
        ssize  = slotsize + sizeof(long*);
        
        // rounding up nslot to be multiple of nchannels
        nslot = ((_nslot + nchannels -1) / nchannels) * nchannels;
        slotsxchannel = nslot / nchannels;
    }

	~StaticAllocator() {
		if (segment) munmap(segment, nslot*ssize);
		segment=nullptr;
	}

    int init() {
		void* result = 0;
        result = mmap(NULL, nslot*ssize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (result == MAP_FAILED) return -1;
        segment = (char*)result;
        
        // initialize the "header"
        char* p = segment;
        for(size_t i=0; i<nslot;++i) {
            long** _p = (long**)p;
            _p[0] = (long*)FF_GO_ON;  // this tells me that the slot is free!
            p+=ssize;
        }
        return 0;
    }
    
	template<typename T>
	void alloc(T*& p, int channel=0) {
        assert(channel>=0 && channel<nchannels) ;
        do {
            size_t m  = cnts[channel]++ % slotsxchannel;
            char  *mp = segment + (channel*slotsxchannel + m)*ssize;
            long** _p = (long**)mp;
            if (_p[0] == (long*)FF_GO_ON) {
                _p[0] = (long*)FF_EOS;    // the slot is occupied 
                p = new (mp+sizeof(long*)) T();
                return;
            }
        } while(1);
	}

	template<typename T>
	static inline void dealloc(T* p) {
		p->~T();
        long** _p = (long**)((char*)p-sizeof(long*));
        _p[0] = (long*)FF_GO_ON;  // the slot is free and can be re-used
	}
    
	template<typename T, typename S>
	void realloc(T* in, S*& out) {
        in->~T();
        char* mp = reinterpret_cast<char*>(in);
        out = new (mp) S();
	}
        
private:
	size_t nslot;                // total number of slots in the data segment
    size_t slotsxchannel;        // how many slots for each sub data segment
	size_t ssize;                // size of a data slot (real size + sizeof(long*)) 
    int    nchannels;            // number of sub data segments
    std::vector<size_t> cnts;    // counters
	char *segment;               // data segment
};

};


#endif /* FF_STATIC_ALLOCATOR_HPP */
