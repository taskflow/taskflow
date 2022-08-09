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

#ifndef _TBB_malloc_Customize_H_
#define _TBB_malloc_Customize_H_

// customizing MALLOC_ASSERT macro
#include "tbb/tbb_stddef.h"
#define MALLOC_ASSERT(assertion, message) __TBB_ASSERT(assertion, message)
#define MALLOC_ASSERT_EX(assertion, message) __TBB_ASSERT_EX(assertion, message)

#ifndef MALLOC_DEBUG
#define MALLOC_DEBUG TBB_USE_DEBUG
#endif

#include "Synchronize.h"

#if DO_ITT_NOTIFY
#include "tbb/itt_notify.h"
#define MALLOC_ITT_SYNC_PREPARE(pointer) ITT_NOTIFY(sync_prepare, (pointer))
#define MALLOC_ITT_SYNC_ACQUIRED(pointer) ITT_NOTIFY(sync_acquired, (pointer))
#define MALLOC_ITT_SYNC_RELEASING(pointer) ITT_NOTIFY(sync_releasing, (pointer))
#define MALLOC_ITT_SYNC_CANCEL(pointer) ITT_NOTIFY(sync_cancel, (pointer))
#define MALLOC_ITT_FINI_ITTLIB()        ITT_FINI_ITTLIB()
#else
#define MALLOC_ITT_SYNC_PREPARE(pointer) ((void)0)
#define MALLOC_ITT_SYNC_ACQUIRED(pointer) ((void)0)
#define MALLOC_ITT_SYNC_RELEASING(pointer) ((void)0)
#define MALLOC_ITT_SYNC_CANCEL(pointer) ((void)0)
#define MALLOC_ITT_FINI_ITTLIB()        ((void)0)
#endif

inline intptr_t BitScanRev(uintptr_t x) {
    return !x? -1 : __TBB_Log2(x);
}

template<typename T>
static inline bool isAligned(T* arg, uintptr_t alignment) {
    return tbb::internal::is_aligned(arg,alignment);
}

static inline bool isPowerOfTwo(uintptr_t arg) {
    return tbb::internal::is_power_of_two(arg);
}
static inline bool isPowerOfTwoAtLeast(uintptr_t arg, uintptr_t power2) {
    return arg && tbb::internal::is_power_of_two_at_least(arg,power2);
}

#define MALLOC_STATIC_ASSERT(condition,msg) __TBB_STATIC_ASSERT(condition,msg)

#define USE_DEFAULT_MEMORY_MAPPING 1

// To support malloc replacement
#include "proxy.h"

#if MALLOC_UNIXLIKE_OVERLOAD_ENABLED
#define malloc_proxy __TBB_malloc_proxy
extern "C" void * __TBB_malloc_proxy(size_t)  __attribute__ ((weak));
#elif MALLOC_ZONE_OVERLOAD_ENABLED
// as there is no significant overhead, always suppose that proxy can be present
const bool malloc_proxy = true;
#else
const bool malloc_proxy = false;
#endif

namespace rml {
namespace internal {
    void init_tbbmalloc();
} } // namespaces

#define MALLOC_EXTRA_INITIALIZATION rml::internal::init_tbbmalloc()

// Need these to work regardless of tools support.
namespace tbb {
    namespace internal {

        enum notify_type {prepare=0, cancel, acquired, releasing};

#if TBB_USE_THREADING_TOOLS
        inline void call_itt_notify(notify_type t, void *ptr) {
            switch ( t ) {
            case prepare:
                MALLOC_ITT_SYNC_PREPARE( ptr );
                break;
            case cancel:
                MALLOC_ITT_SYNC_CANCEL( ptr );
                break;
            case acquired:
                MALLOC_ITT_SYNC_ACQUIRED( ptr );
                break;
            case releasing:
                MALLOC_ITT_SYNC_RELEASING( ptr );
                break;
            }
        }
#else
        inline void call_itt_notify(notify_type /*t*/, void * /*ptr*/) {}
#endif // TBB_USE_THREADING_TOOLS

        template <typename T>
        inline void itt_store_word_with_release(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
            call_itt_notify(releasing, &dst);
#endif // TBB_USE_THREADING_TOOLS
            FencedStore(*(intptr_t*)&dst, src);
        }

        template <typename T>
        inline T itt_load_word_with_acquire(T& src) {
            T result = FencedLoad(*(intptr_t*)&src);
#if TBB_USE_THREADING_TOOLS
            call_itt_notify(acquired, &src);
#endif // TBB_USE_THREADING_TOOLS
            return result;

        }
    } // namespace internal
} // namespace tbb

#include "tbb/internal/_aggregator_impl.h"

template <typename OperationType>
struct MallocAggregator {
    typedef tbb::internal::aggregator_generic<OperationType> type;
};

//! aggregated_operation base class
template <typename Derived>
struct MallocAggregatedOperation {
    typedef tbb::internal::aggregated_operation<Derived> type;
};

#endif /* _TBB_malloc_Customize_H_ */
