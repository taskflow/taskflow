/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_malloc_Synchronize_H_
#define __TBB_malloc_Synchronize_H_

#include "tbb/tbb_machine.h"

//! Stripped down version of spin_mutex.
/** Instances of MallocMutex must be declared in memory that is zero-initialized.
    There are no constructors.  This is a feature that lets it be
    used in situations where the mutex might be used while file-scope constructors
    are running.

    There are no methods "acquire" or "release".  The scoped_lock must be used
    in a strict block-scoped locking pattern.  Omitting these methods permitted
    further simplification. */
class MallocMutex : tbb::internal::no_copy {
    __TBB_atomic_flag flag;

public:
    class scoped_lock : tbb::internal::no_copy {
        MallocMutex& mutex;
        bool taken;
    public:
        scoped_lock( MallocMutex& m ) : mutex(m), taken(true) { __TBB_LockByte(m.flag); }
        scoped_lock( MallocMutex& m, bool block, bool *locked ) : mutex(m), taken(false) {
            if (block) {
                __TBB_LockByte(m.flag);
                taken = true;
            } else {
                taken = __TBB_TryLockByte(m.flag);
            }
            if (locked) *locked = taken;
        }
        ~scoped_lock() {
            if (taken) __TBB_UnlockByte(mutex.flag);
        }
    };
    friend class scoped_lock;
};

// TODO: use signed/unsigned in atomics more consistently
inline intptr_t AtomicIncrement( volatile intptr_t& counter ) {
    return __TBB_FetchAndAddW( &counter, 1 )+1;
}

inline uintptr_t AtomicAdd( volatile intptr_t& counter, intptr_t value ) {
    return __TBB_FetchAndAddW( &counter, value );
}

inline intptr_t AtomicCompareExchange( volatile intptr_t& location, intptr_t new_value, intptr_t comparand) {
    return __TBB_CompareAndSwapW( &location, new_value, comparand );
}

inline uintptr_t AtomicFetchStore(volatile void* location, uintptr_t value) {
    return __TBB_FetchAndStoreW(location, value);
}

inline void AtomicOr(volatile void *operand, uintptr_t addend) {
    __TBB_AtomicOR(operand, addend);
}

inline void AtomicAnd(volatile void *operand, uintptr_t addend) {
    __TBB_AtomicAND(operand, addend);
}

inline intptr_t FencedLoad( const volatile intptr_t &location ) {
    return __TBB_load_with_acquire(location);
}

inline void FencedStore( volatile intptr_t &location, intptr_t value ) {
    __TBB_store_with_release(location, value);
}

inline void SpinWaitWhileEq(const volatile intptr_t &location, const intptr_t value) {
    tbb::internal::spin_wait_while_eq(location, value);
}

class AtomicBackoff {
    tbb::internal::atomic_backoff backoff;
public:
    AtomicBackoff() {}
    void pause() { backoff.pause(); }
};

inline void SpinWaitUntilEq(const volatile intptr_t &location, const intptr_t value) {
    tbb::internal::spin_wait_until_eq(location, value);
}

#endif /* __TBB_malloc_Synchronize_H_ */
