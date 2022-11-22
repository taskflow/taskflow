/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file squeue.hpp
 * \ingroup aux_classes
 *
 * \brief Simple yet efficient (not concurrent) unbounded FIFO queue.
 *
 * This queue have to be used by one single thread at a time, or, each method
 * call have to be protected by a mutex lock. For an unbounded
 * producer-consumer queue implementation see \ref ubuffer.hpp
 *
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
 *
 ****************************************************************************
 */
#ifndef FF_SQUEUE_HPP
#define FF_SQUEUE_HPP

#include <stdlib.h>


namespace ff {

/**
 * \class squeue
 * \ingroup aux_classes
 *
 * \brief Unbound sequential FIFO queue
 */
template <typename T>
class squeue {
private:
    struct data_type {
        data_type():h(-1),t(-1),entry(0) {};
        data_type(int h, int t, T * entry):h(h),t(t),entry(entry) {}
        data_type(const data_type & de):h(de.h),t(de.t),entry(de.entry) {}
        int     h;
        int     t;
        T     * entry;
    };

    typedef T          elem_type;
    
protected:
    enum {DATA_CHUNK=1024, SQUEUE_CHUNK=4096};

    inline T * newchunk() {
        T * v =(T *)malloc(chunk*sizeof(T));
        return v;
    }
    

    inline void deletechunk(int idx) { 
        free(data[idx].entry);   
        data[idx].entry = NULL;
    }

public:

    squeue(size_t chunk=SQUEUE_CHUNK):data(0),datacap(DATA_CHUNK),
                                      nelements(0),
                                      head(0),tail(0),chunk(chunk)  {
        data = (data_type *)malloc(datacap*sizeof(data_type));
        data[0] = data_type(0, -1, newchunk());
    }
    

    ~squeue() {
        if (!data) return;
        for(unsigned int i=0;i<=tail;++i)
            if (data[i].entry) free(data[i].entry);
        free(data);
    }
    

    inline void push_back(const elem_type & elem) {
        T * current       = data[tail].entry;
        int current_tail  = data[tail].t++;
        if ((unsigned)current_tail == (chunk-1)) {
            if (tail == (datacap-1)) {
                datacap += DATA_CHUNK;
                data = (data_type *)realloc(data, datacap*sizeof(data_type));
           }

            T * v = newchunk();
            data[++tail] = data_type(0,0,v);
            current = v;
            current_tail=-1;
        }
        current[current_tail+1] = elem;
        ++nelements;
    }


    inline void pop_back() { 
        if (!nelements) return;
        
        T * current        = data[tail].entry;
        int current_tail   = data[tail].t--;
        
        current[current_tail].~T();
                         
        --nelements;
        if (!current_tail && (tail>0)) {
            deletechunk(tail);
            --tail;
            data[tail].t = chunk-1;
        }
        
    }
    
    inline elem_type& back() { 
        if (!nelements) return *(elem_type*)0;

        T * current       = data[tail].entry;
        int current_tail  = data[tail].t;
        return current[current_tail];
    }


    inline void pop_front() { 
        if (!nelements) return;

        T * current      = data[head].entry;
        int current_head = data[head].h++;
        
        current[current_head].~T();

        --nelements;
        if (((unsigned)current_head==(chunk-1)) && (tail>head)) {
            deletechunk(head);
            ++head;
        }        
    }
    

    inline elem_type& front() { 
        if (!nelements) return *(elem_type*)0;

        T * current       = data[head].entry;
        int current_head  = data[head].h;
        return current[current_head];
    }


    inline elem_type& at(size_t idx) {
        if (!nelements || idx > nelements) return *(elem_type*)0;
        
        T * current       = data[head].entry;
        int current_head  = data[head].h+idx;
        return current[current_head];
    } 
    

    inline size_t size() const { return nelements; }
    
    
private:
    data_type    * data;
    size_t         datacap;  
    size_t         nelements;
    unsigned int   head;  
    unsigned int   tail;
    size_t         chunk;    
};


} // namespace ff

#endif /* FF_SQUEUE_HPP */

#if 0

#include <iosfwd>
#include <deque>
#include <squeue.hpp>

int main() {
    std::deque<int> e;

    ffTime(START_TIME);
    for(int i=0;i<200000;++i) e.push_back(i);
    while(e.size()) {
        //std::cout << d.back() << " ";
        e.pop_back();
    }
    ffTime(STOP_TIME);
    std::cerr << "DONE deque, time= " << ffTime(GET_TIME) << " (ms)\n";
    
    
    squeue<int> d;

    ffTime(START_TIME);
    for(int i=0;i<200000;++i) d.push_back(i);
    while(d.size()) {
        //std::cout << d.back() << " ";
        d.pop_back();
    }
    ffTime(STOP_TIME);
    std::cerr << "DONE squeue, time= " << ffTime(GET_TIME) << " (ms)\n";
    
    
    for(int i=0;i<1500;++i) d.push_back(i);    
    for(int i=0;i<500;++i) {
        std::cout << d.back() << " ";
        d.pop_back();
    }    
    while(d.size()) {
        std::cout << d.front() << " ";
        d.pop_front();
    }
    std::cerr << "size= " << d.size() << "\n";    
    return 0;
}

#endif
