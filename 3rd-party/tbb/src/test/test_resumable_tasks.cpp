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

#include "tbb/tbb_config.h"
#include "tbb/task_scheduler_observer.h"

#include "harness.h"

#if !__TBB_PREVIEW_RESUMABLE_TASKS
int TestMain() {
    return Harness::Skipped;
}
#else // __TBB_PREVIEW_RESUMABLE_TASKS

#include "tbb/task.h"
#include "tbb/concurrent_queue.h"
#include "tbb/atomic.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"
#include "tbb/tbb_thread.h"

#include <vector>

typedef tbb::enumerable_thread_specific<int, tbb::cache_aligned_allocator<int> > ets_int_t;
const int N = 10;

// External activity used in all tests, which resumes suspended execution point
class AsyncActivity {
public:
    AsyncActivity(int num_) : m_numAsyncThreads(num_) {
        for (int i = 0; i < m_numAsyncThreads ; ++i) {
            m_asyncThreads.push_back( new tbb::tbb_thread(AsyncActivity::asyncLoop, this) );
        }
    }
    ~AsyncActivity() {
        for (int i = 0; i < m_numAsyncThreads; ++i) {
            m_tagQueue.push(NULL);
        }
        for (int i = 0; i < m_numAsyncThreads; ++i) {
            m_asyncThreads[i]->join();
            delete m_asyncThreads[i];
        }
        ASSERT(m_tagQueue.empty(), NULL);
    }
    void submit(void* ctx) {
        m_tagQueue.push(ctx);
    }

private:
    static void asyncLoop(AsyncActivity* async) {
        tbb::task::suspend_point tag;
        async->m_tagQueue.pop(tag);
        while (tag) {
            tbb::task::resume(tag);
            async->m_tagQueue.pop(tag);
        }
    }

    const int m_numAsyncThreads;
    tbb::concurrent_bounded_queue<void*> m_tagQueue;
    std::vector<tbb::tbb_thread*> m_asyncThreads;
};

struct SuspendBody {
    SuspendBody(AsyncActivity& a_) :
        m_asyncActivity(a_) {}
    void operator()(tbb::task::suspend_point tag) {
        m_asyncActivity.submit(tag);
    }

private:
    AsyncActivity& m_asyncActivity;
};

class InnermostArenaBody {
public:
    InnermostArenaBody(AsyncActivity& a_) : m_asyncActivity(a_) {}

    void operator()() {
        InnermostOuterParFor inner_outer_body(m_asyncActivity);
        tbb::parallel_for(0, N, inner_outer_body );
    }

private:
    struct InnermostInnerParFor {
        InnermostInnerParFor(AsyncActivity& a_) : m_asyncActivity(a_) {}
        void operator()(int) const {
            tbb::task::suspend(SuspendBody(m_asyncActivity));
        }
        AsyncActivity& m_asyncActivity;
    };
    struct InnermostOuterParFor {
        InnermostOuterParFor(AsyncActivity& a_) : m_asyncActivity(a_) {}
        void operator()(int) const {
            tbb::task::suspend(SuspendBody(m_asyncActivity));
            InnermostInnerParFor inner_inner_body(m_asyncActivity);
            tbb::parallel_for(0, N, inner_inner_body);
        }
        AsyncActivity& m_asyncActivity;
    };
    AsyncActivity& m_asyncActivity;
};

class OutermostArenaBody {
public:
    OutermostArenaBody(AsyncActivity& a_, tbb::task_arena& o_, tbb::task_arena& i_, tbb::task_arena& id_, ets_int_t& ets_) :
        m_asyncActivity(a_), m_outermostArena(o_), m_innermostArena(i_), m_innermostArenaDefault(id_), m_etsInner(ets_) {}

    void operator()() {
        tbb::parallel_for(0, 32, *this);
    }

    void operator()(int i) const {
        tbb::task::suspend(SuspendBody(m_asyncActivity));

        tbb::task_arena& nested_arena = (i % 3 == 0) ?
            m_outermostArena : (i % 3 == 1 ? m_innermostArena : m_innermostArenaDefault);

        if (i % 3 != 0) {
            // We can only guarantee recall coorectness for "not-same" nested arenas entry
            m_etsInner.local() = i;
        }
        InnermostArenaBody innermost_arena_body(m_asyncActivity);
        nested_arena.execute(innermost_arena_body);
        if (i % 3 != 0) {
            ASSERT(i == m_etsInner.local(), "Original thread wasn't recalled for innermost nested arena.");
        }
    }

private:
    AsyncActivity& m_asyncActivity;
    tbb::task_arena& m_outermostArena;
    tbb::task_arena& m_innermostArena;
    tbb::task_arena& m_innermostArenaDefault;
    ets_int_t& m_etsInner;
};

void TestNestedArena() {
    AsyncActivity asyncActivity(4);

    ets_int_t ets_outer;
    ets_int_t ets_inner;

    tbb::task_arena outermost_arena;
    tbb::task_arena innermost_arena(2,2);
    tbb::task_arena innermost_arena_default;

    outermost_arena.initialize();
    innermost_arena_default.initialize();
    innermost_arena.initialize();

    ets_outer.local() = 42;
    OutermostArenaBody outer_arena_body(asyncActivity, outermost_arena, innermost_arena, innermost_arena_default, ets_inner);
    outermost_arena.execute(outer_arena_body);
    ASSERT(ets_outer.local() == 42, "Original/main thread wasn't recalled.");
}

#if __TBB_CPP11_LAMBDAS_PRESENT

#include <thread>

// External activity used in all tests, which resumes suspended execution point
class EpochAsyncActivity {
public:
    EpochAsyncActivity(int num_, tbb::atomic<int>& e_) : m_numAsyncThreads(num_), m_globalEpoch(e_) {
        for (int i = 0; i < m_numAsyncThreads ; ++i) {
            m_asyncThreads.push_back( new tbb::tbb_thread(EpochAsyncActivity::asyncLoop, this) );
        }
    }
    ~EpochAsyncActivity() {
        for (int i = 0; i < m_numAsyncThreads; ++i) {
            m_ctxQueue.push(NULL);
        }
        for (int i = 0; i < m_numAsyncThreads; ++i) {
            m_asyncThreads[i]->join();
            delete m_asyncThreads[i];
        }
        ASSERT(m_ctxQueue.empty(), NULL);
    }
    void submit(void* ctx) {
        m_ctxQueue.push(ctx);
    }

private:
    static void asyncLoop(EpochAsyncActivity* async) {
        tbb::task::suspend_point ctx;
        async->m_ctxQueue.pop(ctx);
        while (ctx) {
            // Track the global epoch
            async->m_globalEpoch++;
            // Continue execution from suspended ctx
            tbb::task::resume(ctx);
            async->m_ctxQueue.pop(ctx);
        }
    }

    const int m_numAsyncThreads;
    tbb::atomic<int>& m_globalEpoch;
    tbb::concurrent_bounded_queue<void*> m_ctxQueue;
    std::vector<tbb::tbb_thread*> m_asyncThreads;
};

struct EpochSuspendBody {
    EpochSuspendBody(EpochAsyncActivity& a_, tbb::atomic<int>& e_, int& le_) :
        m_asyncActivity(a_), m_globalEpoch(e_), m_localEpoch(le_) {}

    void operator()(tbb::task::suspend_point ctx) {
        m_localEpoch = m_globalEpoch;
        m_asyncActivity.submit(ctx);
    }

private:
    EpochAsyncActivity& m_asyncActivity;
    tbb::atomic<int>& m_globalEpoch;
    int& m_localEpoch;
};

// Simple test for basic resumable tasks functionality
void TestSuspendResume() {
    tbb::atomic<int> global_epoch; global_epoch = 0;
    EpochAsyncActivity async(4, global_epoch);

    tbb::enumerable_thread_specific<int, tbb::cache_aligned_allocator<int>, tbb::ets_suspend_aware> ets_fiber;
    tbb::atomic<int> inner_par_iters, outer_par_iters;
    inner_par_iters = outer_par_iters = 0;

    tbb::parallel_for(0, N, [&](int) {
        for (int i = 0; i < 100; ++i) {
            ets_fiber.local() = i;

            int local_epoch;
            tbb::task::suspend(EpochSuspendBody(async, global_epoch, local_epoch));
            ASSERT(local_epoch < global_epoch, NULL);
            ASSERT(ets_fiber.local() == i, NULL);

            tbb::parallel_for(0, N, [&](int) {
                int local_epoch2;
                tbb::task::suspend(EpochSuspendBody(async, global_epoch, local_epoch2));
                ASSERT(local_epoch2 < global_epoch, NULL);
                ++inner_par_iters;
            });

            ets_fiber.local() = i;
            tbb::task::suspend(EpochSuspendBody(async, global_epoch, local_epoch));
            ASSERT(local_epoch < global_epoch, NULL);
            ASSERT(ets_fiber.local() == i, NULL);
        }
        ++outer_par_iters;
    });
    ASSERT(outer_par_iters == N, NULL);
    ASSERT(inner_par_iters == N*N*100, NULL);
}

// During cleanup master's local task pool may
// e.g. contain proxies of affinitized tasks, but can be recalled
void TestCleanupMaster() {
    AsyncActivity asyncActivity(4);
    tbb::task_group tg;
    tbb::enumerable_thread_specific<int> ets;
    tbb::atomic<int> iter_spawned;
    tbb::atomic<int> iter_executed;

    for (int i = 0; i < 100; i++) {
        ets.local() = i;
        iter_spawned = 0;
        iter_executed = 0;

        NativeParallelFor(N, [&asyncActivity, &tg, &iter_spawned, &iter_executed](int j) {
            tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
            if (tbb::task_scheduler_init::default_num_threads() == 1) {
                init.initialize(2);
            }
            for (int k = 0; k < j*10 + 1; ++k) {
                tg.run([&asyncActivity, j, &iter_executed] {
                    for (volatile int l = 0; l < j*10; ++l) {}
                    tbb::task::suspend(SuspendBody(asyncActivity));
                    iter_executed++;
                });
                iter_spawned++;
            }
        });
        ASSERT(iter_spawned == 460, NULL);
        tg.wait();
        ASSERT(iter_executed == 460, NULL);
        ASSERT(ets.local() == i, NULL);
    }
}

class ParForSuspendBody {
    AsyncActivity& asyncActivity;
    int m_numIters;
public:
    ParForSuspendBody(AsyncActivity& a_, int iters) : asyncActivity(a_), m_numIters(iters) {}
    void operator()(int) const {
        for (volatile int i = 0; i < m_numIters; ++i) {}
        tbb::task::suspend(SuspendBody(asyncActivity));
    }
};

#if __TBB_TASK_PRIORITY
class InnerParFor {
    AsyncActivity& asyncActivity;
public:
    InnerParFor(AsyncActivity& a_) : asyncActivity(a_) {}
    void operator()(int) const {
        tbb::affinity_partitioner ap;
        tbb::task_group_context ctx;
        ctx.set_priority(tbb::priority_high);
        tbb::parallel_for(0, 10, ParForSuspendBody(asyncActivity, 1000), ap, ctx);
    }
};

void TestPriorities() {
    AsyncActivity asyncActivity(4);

    tbb::task_scheduler_init init;
    tbb::affinity_partitioner ap;
    tbb::enumerable_thread_specific<int> ets;
    for (int i = 0; i < 10; ++i) {
        ets.local() = i;
        tbb::parallel_for(0, 10, InnerParFor(asyncActivity), ap);
        ASSERT(ets.local() == i, NULL);
    }
}
#endif

void TestNativeThread() {
    AsyncActivity asyncActivity(4);

    int num_threads = tbb::task_scheduler_init::default_num_threads();
    tbb::task_arena arena(num_threads);
    tbb::task_group tg;
    tbb::atomic<int> iter = 0;
    NativeParallelFor(num_threads / 2, [&arena, &tg, &asyncActivity, &iter](int){
        for (int i = 0; i < 10; i++) {
            arena.execute([&tg, &asyncActivity, &iter]() {
                tg.run([&asyncActivity]() {
                    tbb::task::suspend(SuspendBody(asyncActivity));
                });
                iter++;
            });
        }
    });

    tbb::enumerable_thread_specific<bool> ets;
    ets.local() = true;
    ASSERT(iter == (num_threads / 2) * 10, NULL);
    arena.execute([&tg](){
        tg.wait();
    });
    ASSERT(ets.local() == true, NULL);
}

class ObserverTracker : public tbb::task_scheduler_observer {
    tbb::enumerable_thread_specific<bool> is_in_arena;
public:
    tbb::atomic<int> counter;

    ObserverTracker(tbb::task_arena& a) : tbb::task_scheduler_observer(a) {
        counter = 0;
        observe(true);
    }
    void on_scheduler_entry(bool) __TBB_override {
        bool& l = is_in_arena.local();
        ASSERT(l == false, "The thread must call on_scheduler_entry only one time.");
        l = true;
        ++counter;
    }
    void on_scheduler_exit(bool) __TBB_override {
        bool& l = is_in_arena.local();
        ASSERT(l == true, "The thread must call on_scheduler_entry before calling on_scheduler_exit.");
        l = false;
    }
};

void TestObservers() {
    tbb::task_arena arena;
    ObserverTracker tracker(arena);
    do {
        arena.execute([] {
            tbb::parallel_for(0, 10, [](int) {
                tbb::task::suspend([](tbb::task::suspend_point tag) {
                    tbb::task::resume(tag);
                });
            }, tbb::simple_partitioner());
        });
    } while (tracker.counter < 100);
    tracker.observe(false);
}
#endif

int TestMain() {
    tbb::enumerable_thread_specific<bool> ets;
    ets.local() = true;

    tbb::task_scheduler_init init(max(tbb::task_scheduler_init::default_num_threads(), 16));

    TestNestedArena();
#if __TBB_CPP11_LAMBDAS_PRESENT
    // Using functors would make this test much bigger and with
    // unnecessary complexity, one C++03 TestNestedArena is enough
    TestSuspendResume();
    TestCleanupMaster();
#if __TBB_TASK_PRIORITY
    TestPriorities();
#endif
    TestNativeThread();
    TestObservers();
#endif
    ASSERT(ets.local() == true, NULL);
    return Harness::Done;
}

#endif // !__TBB_PREVIEW_RESUMABLE_TASKS

