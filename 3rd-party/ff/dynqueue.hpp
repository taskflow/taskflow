/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file dynqueue.hpp
 * \ingroup aux_classes
 *
 * \brief Implementation of a dynamic queue. Not currently used.
 *
 * Dynamic (list-based) Single-Writer Single-Reader
 * (or Single-Producer Single-Consumer) unbounded queue.
 *
 * No lock is needed around pop and push methods.
 * See also ubuffer.hpp for a more efficient SPSC unbounded queue.
 *
 * M. Aldinucci, M. Danelutto, P. Kilpatrick, M. Meneghin, and M. Torquati,
 * "An Efficient Unbounded Lock-Free Queue for Multi-core Systems,"
 * in Proc. of 18th Intl. Euro-Par 2012 Parallel Processing, Rhodes Island,
 * Greece, 2012, pp. 662-673. doi:10.1007/978-3-642-32820-6_65
 *
 * \note Not currently used in the FastFlow implementation.
 */

#ifndef FF_DYNQUEUE_HPP
#define FF_DYNQUEUE_HPP

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


#include <stdlib.h>
#include <ff/buffer.hpp>
#include <ff/spin-lock.hpp> // used only for mp_push and mp_pop
#include <ff/sysdep.h>

namespace ff {

#if !defined(_FF_DYNQUEUE_OPTIMIZATION)

class dynqueue {
private:
    struct Node {
        void        * data;
        struct Node * next;
    };

    union {
        Node * volatile   head;
        char padding1[CACHE_LINE_SIZE];
        //long padding1[longxCacheLine-(sizeof(Node *)/sizeof(long))];
    };
    union {
        Node * volatile   tail;
        char padding2[CACHE_LINE_SIZE];
        //long padding2[longxCacheLine-(sizeof(Node*)/sizeof(long))];
    };

    /* ----- two-lock used only in the mp_push and mp_pop methods ------- */
    /*                                                                    */    
    /*  By using the mp_push and mp_pop methods as standard push and pop, */
    /*  the dynqueue algorithm basically implements the well-known        */
    /*  Michael and Scott 2-locks MPMC queue.                             */    
    /*                                                                    */
    /*                                                                    */    
	/*
	union {
        lock_t P_lock;
        char padding3[CACHE_LINE_SIZE];
    };
    union {
        lock_t C_lock;
        char padding4[CACHE_LINE_SIZE];
    };
    */
	ALIGN_TO_PRE(CACHE_LINE_SIZE)
	lock_t P_lock;
	ALIGN_TO_POST(CACHE_LINE_SIZE)

	ALIGN_TO_PRE(CACHE_LINE_SIZE)
		lock_t C_lock;
	ALIGN_TO_POST(CACHE_LINE_SIZE)

    /* -------------------------------------------------------------- */

    // internal cache
    // if mp_push and mp_pop methods are used the cache access is lock protected
#if defined(STRONG_WAIT_FREE)
    Lamport_Buffer     cache;
#else
    SWSR_Ptr_Buffer    cache;
#endif

private:
    inline Node * allocnode() {
        union { Node * n; void * n2; } p;
#if !defined(NO_CACHE)
        if (cache.pop(&p.n2)) return p.n;
#endif
        p.n = (Node *)::malloc(sizeof(Node));
        return p.n;
    }

    inline Node * mp_allocnode() {
        union { Node * n; void * n2; } p;
#if !defined(NO_CACHE)
        spin_lock(P_lock);
        if (cache.pop(&p.n2)) {
            spin_unlock(P_lock);
            return p.n;
        }
        spin_unlock(P_lock);
#endif
        p.n = (Node *)::malloc(sizeof(Node));
        return p.n;
    }

public:
    enum {DEFAULT_CACHE_SIZE=1024};

    dynqueue(int cachesize=DEFAULT_CACHE_SIZE, bool fillcache=false):cache(cachesize) {
        Node * n = (Node *)::malloc(sizeof(Node));
        n->data = NULL; n->next = NULL;
        head=n;
        tail=n;
        cache.init();
        if (fillcache) {
            for(int i=0;i<cachesize;++i) {
                n = (Node *)::malloc(sizeof(Node));
                if (n) cache.push(n);
            }
        }
        init_unlocked(P_lock); 
        init_unlocked(C_lock);
        // Avoid unused private field warning on padding vars
        //(void) padding1; (void) padding2 ; (void) padding3; (void) padding4;
    }

    bool init() { return true;}

    ~dynqueue() {
        union { Node * n; void * n2; } p;
        if (cache.buffersize()>0) while(cache.pop(&p.n2)) free(p.n);
        while(head != tail) {
            p.n = (Node*)head;
            head = head->next;
            free(p.n);
        }
        if (head) free((void*)head);
    }
    
    inline bool push(void * const data) {
        assert(data != NULL);
        Node * n = allocnode();
        n->data = data; n->next = NULL;
        WMB();
        tail->next = n;
        tail       = n;

        return true;
    }

    inline bool  pop(void ** data) {        
        assert(data != NULL);
#if defined(STRONG_WAIT_FREE)
        if (head == tail) return false;
#else
        if (head->next) 
#endif
        {
            Node * n = (Node *)head;
            *data    = (head->next)->data;
            head     = head->next;
#if !defined(NO_CACHE)
            if (!cache.push(n)) ::free(n);
#else
            ::free(n);
#endif      
            return true;
        }
        return false;
    }    


    inline unsigned long length() const { return 0;}

    /*
     * MS 2-lock MPMC algorithm PUSH method 
     */
    inline bool mp_push(void * const data) {
        assert(data != NULL);
        Node* n = mp_allocnode();
        n->data = data; n->next = NULL;
        ff::spin_lock(P_lock);
        tail->next = n;
        tail       = n;
        spin_unlock(P_lock);
        return true;
    }

    /*
     * MS 2-lock MPMC algorithm POP method 
     */
    inline bool  mp_pop(void ** data) {        
        assert(data != NULL);
        spin_lock(C_lock);
        if (head->next) {
            Node * n = (Node *)head;
            *data    = (head->next)->data;
            head     = head->next;
            bool f   = cache.push(n);
            spin_unlock(C_lock);
            if (!f) ::free(n);
            return true;
        }
        spin_unlock(C_lock);
        return false;
    }    
};

#else // _FF_DYNQUEUE_OPTIMIZATION
/* 
 * Experimental code
 */

class dynqueue {
private:
    struct Node {
        void        * data;
        struct Node * next;
    };

    Node * volatile         head;
    volatile unsigned long  pwrite;
    long padding1[longxCacheLine-((sizeof(Node *)+sizeof(unsigned long))/sizeof(long))];
    Node * volatile        tail;
    volatile unsigned long pread;
    long padding2[longxCacheLine-((sizeof(Node*)+sizeof(unsigned long))/sizeof(long))];

    const   size_t cachesize;
    void ** cache;

private:

    inline bool cachepush(void * const data) {
        
        if (!cache[pwrite]) {
            /* Write Memory Barrier: ensure all previous memory write 
             * are visible to the other processors before any later
             * writes are executed.  This is an "expensive" memory fence
             * operation needed in all the architectures with a weak-ordering 
             * memory model where stores can be executed out-or-order 
             * (e.g. Powerpc). This is a no-op on Intel x86/x86-64 CPUs.
             */
            WMB(); 
            cache[pwrite] = data;
            pwrite += (pwrite+1 >= cachesize) ? (1-cachesize): 1;
            return true;
        }
        return false;
    }

    inline bool  cachepop(void ** data) {
        if (!cache[pread]) return false;
        
        *data = cache[pread];
        cache[pread]=NULL;
        pread += (pread+1 >= cachesize) ? (1-cachesize): 1;    
        return true;
    }    
    
public:
    enum {DEFAULT_CACHE_SIZE=1024};

    dynqueue(int cachesize=DEFAULT_CACHE_SIZE, bool fillcache=false):cachesize(cachesize) {
        Node * n = (Node *)::malloc(sizeof(Node));
        n->data = NULL; n->next = NULL;
        head=n;
        tail=n;

        cache=(void**)getAlignedMemory(longxCacheLine*sizeof(long),cachesize*sizeof(void*));
        if (!cache) {
            error("FATAL ERROR: dynqueue no memory available!\n");
            abort();
        }

        if (fillcache) {
            for(int i=0;i<cachesize;++i) {
                n = (Node *)::malloc(sizeof(Node));
                if (n) cachepush(n);
            }
        }
    }

    ~dynqueue() {
        union { Node * n; void * n2; } p;
        while(cachepop(&p.n2)) free(p.n);
        while(head != tail) {
            p.n = (Node*)head;
            head = head->next;
            free(p.n);
        }
        if (head) free((void*)head);
        if (cache) freeAlignedMemory(cache);
    }

    inline bool push(void * const data) {
        assert(data != NULL);

        union { Node * n; void * n2; } p;
        if (!cachepop(&p.n2))
            p.n = (Node *)::malloc(sizeof(Node));
        
        p.n->data = data; p.n->next = NULL;
        WMB();
        tail->next = p.n;
        tail       = p.n;

        return true;
    }

    inline bool  pop(void ** data) {
        assert(data != NULL);
        if (head->next) {
            Node * n = (Node *)head;
            *data    = (head->next)->data;
            head     = head->next;

            if (!cachepush(n)) free(n);
            return true;
        }
        return false;
    }    
};

#endif // _FF_DYNQUEUE_OPTIMIZATION


} // namespace

#endif /* FF_DYNQUEUE_HPP */
