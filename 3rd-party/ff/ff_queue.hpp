/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

#ifndef FF_QUEUE_HPP
#define FF_QUEUE_HPP

/*!
 * \file ff_queue.hpp
 * \ingroup aux_classes
 *
 * \brief Experimental. Not currently used. 
 *
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>
#include <memory>

#ifdef _MSC_VER
#   pragma warning (disable: 4200) // nonstandard extension used : zero-sized array in struct/union
#   define abort() __debugbreak(), (abort)()
#endif

#define INLINE static __inline
#define NOINLINE
#if !defined(CACHE_LINE_SIZE)
#define CACHE_LINE_SIZE 128
#endif


INLINE void* aligned_malloc(size_t sz)
{
    void*               mem;
    if (posix_memalign(&mem, CACHE_LINE_SIZE, sz))
        return 0;
    return mem;
}

INLINE void aligned_free(void* mem)
{
    free(mem);
}


INLINE void atomic_addr_store_release(void* volatile* addr, void* val)
{
    __asm __volatile ("" ::: "memory");
    addr[0] = val;
}


INLINE void* atomic_addr_load_acquire(void* volatile* addr)
{
    void* val;
    val = addr[0];
    __asm __volatile ("" ::: "memory");
    return val;
}

class ff_queue
{
public:

    ff_queue (size_t bucket_size, size_t max_bucket_count)
        : bucket_size_ (bucket_size)
        , max_bucket_count_ (max_bucket_count)
    {
        bucket_count_ = 0;
        bucket_t* bucket = alloc_bucket(bucket_size_);
        head_pos_ = bucket->data;
        tail_pos_ = bucket->data;
        tail_end_ = bucket->data + bucket_size_;
        tail_next_ = 0;
        tail_bucket_ = bucket;
        last_bucket_ = bucket;
        *(void**)head_pos_ = (void*)1;
        pad_[0] = 0;
    }


    ~ff_queue ()
    {
        bucket_t* bucket = last_bucket_;
        while (bucket != 0)
        {
            bucket_t* next_bucket = bucket->next;
            aligned_free(bucket);
            bucket = next_bucket;
        }
    }

    char* enqueue_prepare (size_t sz)
    {
        assert(((uintptr_t)tail_pos_ % sizeof(void*)) == 0);
        size_t msg_size = ((uintptr_t)(sz + sizeof(void*) - 1) & ~(sizeof(void*) - 1)) + sizeof(void*);
        if ((size_t)(tail_end_ - tail_pos_) >= msg_size + sizeof(void*))
        {
            tail_next_ = tail_pos_ + msg_size;
            return tail_pos_ + sizeof(void*);
        }
        else
        {
            return enqueue_prepare_slow(sz);
        }
    }

    void enqueue_commit ()
    {
        *(char* volatile*)tail_next_ = (char*)1;
        atomic_addr_store_release((void* volatile*)tail_pos_, tail_next_);
        tail_pos_ = tail_next_;
    }


    char* dequeue_prepare ()
    {
        assert(((uintptr_t)head_pos_ % sizeof(void*)) == 0);
        void* next = atomic_addr_load_acquire((void* volatile*)head_pos_);
        if (((uintptr_t)next & 1) == 0)
        {
            char* msg = head_pos_ + sizeof(void*);
            return msg;
        }
        else if (((uintptr_t)next & ~1) == 0)
        {
            return 0;
        }
        else
        {
            atomic_addr_store_release((void* volatile*)&head_pos_, (char*)((uintptr_t)next & ~1));
            return dequeue_prepare();
        }
    }

 
    void dequeue_commit ()
    {
        char* next = *(char* volatile*)head_pos_;
        assert(next != 0);
        atomic_addr_store_release((void* volatile*)&head_pos_, next);
    }

private:
    struct bucket_t
    {
        bucket_t*               next;
        size_t                  size;
        char                    data [0];
    };

    char* volatile              head_pos_;

    char                        pad_ [CACHE_LINE_SIZE];

    char*                       tail_pos_;
    char*                       tail_end_;
    char*                       tail_next_;
    bucket_t*                   tail_bucket_;
    bucket_t*                   last_bucket_;
    size_t const                bucket_size_;
    size_t const                max_bucket_count_;
    size_t                      bucket_count_;


    bucket_t* alloc_bucket (size_t sz)
    {
        bucket_t* bucket = (bucket_t*)aligned_malloc(sizeof(bucket_t) + sz);
        if (bucket == 0)
            throw std::bad_alloc();
        bucket->next = 0;
        bucket->size = sz;
        bucket_count_ += 1;
        return bucket;
    }


    NOINLINE
    char* enqueue_prepare_slow (size_t sz)
    {
        size_t bucket_size = bucket_size_;
        if (bucket_size < sz + 2 * sizeof(void*))
            bucket_size = sz + 2 * sizeof(void*);

        bucket_t* bucket = 0;
        char* head_pos = (char*)atomic_addr_load_acquire((void* volatile*)&head_pos_);
        while (head_pos < last_bucket_->data || head_pos >= last_bucket_->data + last_bucket_->size)
        {
            bucket = last_bucket_;
            last_bucket_ = bucket->next;
            bucket->next = 0;
            assert(last_bucket_ != 0);

            if ((bucket->size < bucket_size)
                || (bucket_count_ > max_bucket_count_
                    && (head_pos < last_bucket_->data || head_pos >= last_bucket_->data + last_bucket_->size)))
            {
                aligned_free(bucket);
                bucket = 0;
                continue;
            }
            break;
        }

        if (bucket == 0)
            bucket = alloc_bucket(bucket_size);
        *(void* volatile*)bucket->data = (void*)1;
        atomic_addr_store_release((void* volatile*)tail_pos_, (void*)((uintptr_t)bucket->data | 1));
        tail_pos_ = bucket->data;
        tail_end_ = tail_pos_ + bucket_size;
        tail_bucket_->next = bucket;
        tail_bucket_ = bucket;
        return enqueue_prepare(sz);
    }

    ff_queue (ff_queue const&);
    void operator = (ff_queue const&);
};

#endif /* FF_QUEUE_HPP */
