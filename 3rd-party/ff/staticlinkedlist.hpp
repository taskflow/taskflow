/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \link
 *  \file staticlinkedlist.hpp
 *  \ingroup aux_classes
 *
 *  \brief Static Linked List. Not currently used.
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

/* *
 *  
 * Static linked list Single-Writer Single-Reader unbounded queue. No lock is
 * needed around pop and push methods.
 *  
 * -- Massimiliano Meneghin: themaxmail@gmail.com 
 */

#ifndef FF_STATICLINKEDLIST_HPP
#define FF_STATICLINKEDLIST_HPP

#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>
#include <assert.h>

#if defined(WHILE)
#define WHILE_VERSION 1
#else
#define WHILE_VERSION 0
#endif

#if defined(NO_PADDING)
#define PADDING_VERSION 0
#else
#define PADDING_VERSION 1
#endif

#if defined(INDEX)
#define POINTERS_VERSION 0
#else
#define POINTERS_VERSION 1
#endif

#if defined(NO_POST_POLLING)
#define POSTPOLLING_VERSION 0
#else
#define POSTPOLLING_VERSION 1
#endif

#if defined(__GNUC__)
#if !defined(likely)
#define likely(x)       __builtin_expect((x),1)
#endif
#if !defined(unlikely)
#define unlikely(x)     __builtin_expect((x),0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif 


#define CAST_TO_UL(X) ((unsigned long)X)
#define CAST_TO_VUL(X) *((volatile unsigned long * )&X)

namespace ff {

class staticlinkedlist {
#if POSTPOLLING_VERSION
    struct Node {
        unsigned long data;
        #if POINTERS_VERSION
      struct Node * previous;
        struct Node * next;
        #else
        unsigned int previous;
        unsigned int next;
        #if PADDING_VERSION
        long paddingindex[1];
        #endif
        #endif
        #if PADDING_VERSION
        long padding[longxCacheLine-3];
        #endif
    };

private:
#if POINTERS_VERSION
    Node *  head;
    long    padding1[longxCacheLine-1];
    Node *  tail;
    long    padding2[longxCacheLine-1];
#else
    int     head;
    int     head_padding;
    long    padding1[longxCacheLine-1];
    int     tail;
    int     tail_previous;
    long    padding2[longxCacheLine-1];
#endif
    /*
      This is a vector of Node elemens.  
      The len is equal to cachesize
     */
    Node * min_cache;
    int min_cache_size;
    void * cache_mem;
public:
    /**
     * TODO
     */
    enum {DEFAULT_CACHE_SIZE=1024};

    staticlinkedlist(int cachesize=DEFAULT_CACHE_SIZE, bool /*fillcache*/=false){
        // avoid unused field warning for padding
        if (longxCacheLine>1)
            padding1[0]=padding2[0];
        // end
        cache_mem =::malloc((sizeof(Node))*(cachesize+5));
        unsigned int CPU_cachesize = longxCacheLine*sizeof(void*);
        if(CAST_TO_UL(cache_mem)%CPU_cachesize){
            min_cache = (Node *)(
                                 ((CAST_TO_UL(cache_mem)
                                   /CPU_cachesize)+1
                                  )*CPU_cachesize
                                 );
        }else{
            min_cache = (Node *)(cache_mem);
        }
        min_cache_size = cachesize;
        int i;
        #if POINTERS_VERSION

        for(i=0; i<cachesize; i++){
            min_cache[i].next = &min_cache[(i+1)%(cachesize)];
            min_cache[i].previous = &min_cache[(cachesize + i-1)%(cachesize)];
            min_cache[i].data = 0;
        }
        head = &min_cache[0];
        tail = &min_cache[1];
        #else
        for(i=0; i<cachesize; i++){
            min_cache[i].next = (i+1)%(cachesize);
            min_cache[i].previous = (cachesize + i-1)%(cachesize);
            min_cache[i].data = 0;
        }
        head = 0;
        tail = 1;
        tail_previous = 0;
#endif
    }
    
    ~staticlinkedlist() {
        free(cache_mem);
    }

#if WHILE_VERSION
    inline bool push(void * const data) {
#if POINTERS_VERSION
        do{}
        while(likely(CAST_TO_VUL(tail->data) != 0));
        WMB();

        tail->previous->data = CAST_TO_UL(data);
        tail = tail->next;
#else
        do{}
        while(likely(CAST_TO_VUL(min_cache[tail].data) != 0));
        WMB();
        min_cache[tail_previous].data = CAST_TO_UL(data);
        tail_previous = tail;
        tail =  min_cache[tail].next;
#endif
        return true;
    }
#else
    inline bool push(void * const data) {
#if POINTERS_VERSION
        if(likely(CAST_TO_VUL(tail->data) == 0)){
            WMB();
            tail->previous->data = CAST_TO_UL(data);
            tail = tail->next;
            return true;
        }
        return false;
#else
        if(likely(CAST_TO_VUL(min_cache[tail].data) == 0)){
            WMB();
            min_cache[tail_previous].data = CAST_TO_UL(data);
            tail_previous = tail;
            tail =  min_cache[tail].next;
            return true;
        }
        return false;
#endif
}
#endif
#if POINTERS_VERSION
    inline bool  pop(void ** data) { 
        if (likely(CAST_TO_VUL(head->data) != 0)) {        
            *data = (void *)head->data;
            head->data = CAST_TO_UL(NULL);
            head = head->next;
            return true;
        }
        return false;
    }    
#else
    inline bool  pop(void ** data) { 
        if (likely(CAST_TO_VUL(min_cache[head].data) != 0)) {        
            *data = (void *)min_cache[head].data;
            min_cache[head].data = CAST_TO_UL(NULL);
            head = min_cache[head].next;
            return true;
        }
        return false;
    }    
#endif
#else //NO POSTPOLLING_VERSION
    
    struct Node {
        unsigned long data;
#if POINTERS_VERSION
        struct Node * next;
#if PADDING_VERSION
        struct Node * previpus_padding;
#endif

#else//POINTERS_VERSION
        unsigned int next;
#if PADDING_VERSION
        unsigned int nextpadding;
        long paddingindex[1];
#endif
#endif//POINTERS_VERSION
#if PADDING_VERSION
        long padding[longxCacheLine-3];
#endif
    };

private:
#if POINTERS_VERSION
    Node *  head;
    long    padding1[longxCacheLine-1];
    Node *  tail;
    long    padding2[longxCacheLine-1];
#else
    int     head;
    int     head_padding;
    long    padding1[longxCacheLine-1];
    int     tail;
    int     tail_previous;
    long    padding2[longxCacheLine-1];
#endif
    /*
      This is a vector of Node elemens.  
      The len is equal to cachesize
     */
    Node * min_cache;
    int min_cache_size;
    void * cache_mem;
public:
    /**
     * TODO
     */
    enum {DEFAULT_CACHE_SIZE=1024};

    staticlinkedlist(int cachesize=DEFAULT_CACHE_SIZE, bool fillcache=false){
        cache_mem =::malloc((sizeof(Node))*(cachesize+5));
        unsigned int CPU_cachesize = longxCacheLine*sizeof(void*);
        if(CAST_TO_UL(cache_mem)%CPU_cachesize){
            min_cache = (Node *)(
                                 ((CAST_TO_UL(cache_mem)
                                   /CPU_cachesize)+1
                                  )*CPU_cachesize
                                 );
        }else{
            min_cache = (Node *)(cache_mem);
        }
        min_cache_size = cachesize;
        int i;
        #if POINTERS_VERSION

        for(i=0; i<cachesize; i++){
            min_cache[i].next = &min_cache[(i+1)%(cachesize)];
            min_cache[i].data = 0;
        }
        head = &min_cache[0];
        tail = &min_cache[0];
        #else
        for(i=0; i<cachesize; i++){
            min_cache[i].next = (i+1)%(cachesize);
            min_cache[i].data = 0;
        }
        head = 0;
        tail = 0;
#endif
    }
    
    ~staticlinkedlist() {
        free(cache_mem);
    }

#if WHILE_VERSION
    inline bool push(void * const data) {
#if POINTERS_VERSION
        do{}
        while(likely(CAST_TO_VUL(tail->data) != 0));
        WMB();

        tail->data = CAST_TO_UL(data);
        tail = tail->next;
#else
        do{}
        while(likely(CAST_TO_VUL(min_cache[tail].data) != 0));
        WMB();
        min_cache[tail].data = CAST_TO_UL(data);
        tail =  min_cache[tail].next;
#endif
        return true;
    }
#else
    inline bool push(void * const data) {
#if POINTERS_VERSION
        if(likely(CAST_TO_VUL(tail->data) == 0)){
            WMB();
            tail->data = CAST_TO_UL(data);
            tail = tail->next;
            return true;
        }
        return false;
#else
        if(likely(CAST_TO_VUL(min_cache[tail].data) == 0)){
            WMB();
            min_cache[tail].data = CAST_TO_UL(data);
            tail =  min_cache[tail].next;
            return true;
        }
        return false;
#endif
}
#endif
#if POINTERS_VERSION
    inline bool  pop(void ** data) { 
        if (likely(CAST_TO_VUL(head->data) != 0)) {        
            *data = (void *)head->data;
            head->data = CAST_TO_UL(NULL);
            head = head->next;
            return true;
        }
        return false;
    }    
#else
    inline bool  pop(void ** data) { 
        if (likely(CAST_TO_VUL(min_cache[head].data) != 0)) {        
            *data = (void *)min_cache[head].data;
            min_cache[head].data = CAST_TO_UL(NULL);
            head = min_cache[head].next;
            return true;
        }
        return false;
    }    
#endif

#endif
};

} // namespace ff

#endif /* FF_STATICLINKEDLIST_HPP */
