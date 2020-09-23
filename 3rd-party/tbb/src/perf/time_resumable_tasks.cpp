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

#define TBB_PREVIEW_RESUMABLE_TASKS 1
#include "tbb/tbb_config.h"

#include "tbb/task.h"
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

#include "tbb/parallel_for.h"

#include <vector>
#include <stack>
#include <functional>
#include <numeric>
#include <algorithm>

/************************************************************************/
/* SETTINGS                                                             */
/************************************************************************/

const int DEF_BENCH_RUNS = 1000;

/************************************************************************/
/* HELPERS                                                              */
/************************************************************************/

#include "harness_perf.h" // harness_perf::median

template<typename T>
T get_median(std::vector<T>& times) {
    return harness_perf::median(times.begin(), times.end());
}

/************************************************************************/
/* SERIAL BENCHMARKS                                                    */
/************************************************************************/

//! Allocate COROUTINES_NUM fibers in a row (suspend) in a recursive manner
//! and then swith back (resume) unwinding the ctx_stack.
void BenchCoroutinesAllocation() {
    tbb::task_scheduler_init init(1);

    const int COROUTINES_NUM = 100;
    std::stack<tbb::task::suspend_point> ctx_stack;
    tbb::task_group tg;

    std::function<void(int)> recursive_f;
    recursive_f = [=, &ctx_stack, &tg, &recursive_f](int i) {
        if (i < COROUTINES_NUM) {
            tg.run([&recursive_f, i]() {
                recursive_f(i + 1);
            });
            tbb::task::suspend([&ctx_stack](tbb::task::suspend_point ctx) {
                ctx_stack.push(ctx);
            });
        }
        if (ctx_stack.size() != 0) {
            tbb::task::suspend_point ctx = ctx_stack.top(); ctx_stack.pop();
            tbb::task::resume(ctx);
        }
    };
    tg.run([=, &recursive_f]() {
        std::vector<double> times;
        for (int i = 0; i < DEF_BENCH_RUNS; i++) {
            tbb::tick_count tick = tbb::tick_count::now();
            recursive_f(1);
            double interval = (tbb::tick_count::now() - tick).seconds() * 1e6;
            times.push_back(interval);
        }
        // COROUTINES_NUM suspend and resume operations in each run
        double median = get_median(times) / double(COROUTINES_NUM);
        printf("Test 1 (Coroutines alloc/dealloc): Median time (microseconds): %.4f\n", median);
    });
    tg.wait();
}

//! Create a task, which suspends and resumes intself, thus reusing once created coroutine
void BenchReusage() {
    tbb::task_scheduler_init init(1);
    tbb::task_group tg;

    std::vector<double> times;
    tg.run([&times]() {
        for (int i = 0; i < DEF_BENCH_RUNS * 10; i++) {
            tbb::tick_count tick = tbb::tick_count::now();
            tbb::task::suspend([](tbb::task::suspend_point ctx) {
                tbb::task::resume(ctx);
            });
            double diff = (tbb::tick_count::now() - tick).seconds() * 1e6;
            times.push_back(diff);
        }
    });
    tg.wait();
    double median = get_median(times);
    printf("Test 2 (Coroutine reusage): Median time (microseconds): %.4f\n", median);
}

//! Create two tasks and switch between them (suspend current and resume previously suspended coroutine)
//! Measure an average time of the context switch
void BenchContextSwitch() {
    tbb::task_scheduler_init init(1);
    tbb::task_group tg;
    const int N = 10000; // number of switches
    const int tasks_num = 2;

    std::vector<double> times;
    for (int i = 0; i < 100; ++i) {
        int switch_counter = N;
        tbb::task::suspend_point current_ctx = NULL;

        tbb::tick_count tick = tbb::tick_count::now();
        for (int j = 0; j < tasks_num; ++j) {
            tg.run([=, &switch_counter, &current_ctx]() {
                while (switch_counter-- > 0) {
                    tbb::task::suspend([=, &switch_counter, &current_ctx](tbb::task::suspend_point ctx) {
                        if (switch_counter == N - 1) {
                            current_ctx = ctx;
                        } else {
                            tbb::task::suspend_point ctx_to_resume = current_ctx;
                            current_ctx = ctx;
                            tbb::task::resume(ctx_to_resume);
                        }
                    });
                }
                if (switch_counter == -1) {
                    tbb::task::resume(current_ctx);
                }
            });
        }
        tg.wait();
        // To get an average context switch time divide the bench time by the number of context switches
        double diff = ((tbb::tick_count::now() - tick).seconds() / double(N)) * 1e6;
        times.push_back(diff);
    }
    printf("Test 3 (Context Switch): Median time (microseconds): %.4f\n", get_median(times));
}

/************************************************************************/
/* PARALLEL BENCHMARKS                                                  */
/************************************************************************/

//! Strong scaling benchmark with predefined number of iterations (N), each parallel_for task
//! suspends and resumes itself with a predefined busy-waiting iterations (work size).
//! Reports 3 numbers: serial, half of the machine, and full available concurrency
template <bool UseResumableTasks>
void ScalabilityBenchmark(const size_t work_size) {
    const int N = 1000;
    const int NUM_THREADS = tbb::task_scheduler_init::default_num_threads();
    const int STEP_RATIO = 2;

    // Count 3 scalability metrics: the serial, half and full machine concurrency
    for (int i = 0; i <= NUM_THREADS; i += (NUM_THREADS / STEP_RATIO)) {
        const int concurrency = (i == 0) ? 1 : i; // just to make step loop nice looking
        tbb::task_scheduler_init init(concurrency);
        std::vector<double> times;
        for (int j = 0; j < 100; j++) {
            tbb::tick_count tick = tbb::tick_count::now();
            tbb::parallel_for(0, N, [&work_size](const int /*j*/) {
                if (UseResumableTasks) {
                    tbb::task::suspend([](tbb::task::suspend_point ctx) {
                        tbb::task::resume(ctx);
                    });
                }
                for (volatile size_t k = 0; k < work_size; ++k);
            }, tbb::simple_partitioner());
            double diff = (tbb::tick_count::now() - tick).seconds() * 1e3;
            times.push_back(diff);
        }
        printf("Test 4 (Scalability): Work Size: %zu, With RT-feature: %s, Concurrency: %d, Time (milliseconds): %.4f\n",
                work_size, (UseResumableTasks ? "true" : "false"), concurrency, get_median(times));
    }
}

/************************************************************************/
/* NATIVE IMPLEMENTATION                                                */
/************************************************************************/

// Dependencies section for co_context.h

#if _WIN32
#include <windows.h> // GetSystemInfo
#else
#include <unistd.h> // sysconf(_SC_PAGESIZE)
#endif

namespace tbb {
namespace internal {
//! System dependent impl
inline size_t GetDefaultSystemPageSize() {
#if _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}
class governor {
    //! Caches the size of OS regular memory page
    static size_t DefaultPageSize;
public:
    //! Staic accessor for OS regular memory page size
    static size_t default_page_size () {
        return DefaultPageSize ? DefaultPageSize : DefaultPageSize = GetDefaultSystemPageSize();
    }
};
size_t governor::DefaultPageSize;
} // namespace internal
} // namespace tbb

// No-op versions of __TBB_ASSERT/EX for co_context.h header
#define __TBB_ASSERT(predicate,comment) ((void)0)
#define __TBB_ASSERT_EX(predicate,comment) ((void)(1 && (predicate)))

// TBB coroutines implementation
// Disable governor header to remove the dependency
#define _TBB_governor_H
#include "../tbb/co_context.h"
using namespace tbb::internal;
#undef _TBB_governor_H

#define HARNESS_CUSTOM_MAIN 1
#include "../test/harness.h" // NativeParallelFor

namespace tbb {
namespace internal {
// Our native coroutine function
#if _WIN32
/* [[noreturn]] */ inline void __stdcall co_local_wait_for_all(void* arg) {
#else
/* [[noreturn]] */ inline void co_local_wait_for_all(void* arg) {
#endif
    coroutine_type next = *static_cast<coroutine_type*>(arg);
    coroutine_type current; current_coroutine(current);
    swap_coroutine(current, next);
}
} // namespace internal
} // namespace tbb

// The same scalability benchmark as for TBB, but written with native OS fibers implementation
void BenchNativeImpl(const size_t work_size) {
    const int N = 1000;
    const int NUM_THREADS = tbb::task_scheduler_init::default_num_threads();
    const int STEP_RATIO = 2;
    const size_t STACK_SIZE = 4 * 1024 * 1024; // Just like default TBB worker thread stack size

    // Count 3 scalability metrics: the serial, half and full machine concurrency
    for (int i = 0; i <= NUM_THREADS; i += (NUM_THREADS / STEP_RATIO)) {
        const int concurrency = (i == 0) ? 1 : i; // just to make step loop nice looking
        const int sub_range = N / concurrency;
        std::vector<double> times;
        for (int r = 0; r < 100; r++) {
            tbb::tick_count tick = tbb::tick_count::now();
            NativeParallelFor(concurrency, [=, &work_size, &sub_range](int /*idx*/) {
                // Each iteration of sub-range emulates a single TBB task
                for (int j = 0; j < sub_range; j++) {
                    coroutine_type co_next;
                    coroutine_type co_current; current_coroutine(co_current);
                    create_coroutine(co_next, STACK_SIZE, &co_current);
                    swap_coroutine(co_current, co_next);

                    // Busy-wait for a while emulating some work
                    for (volatile size_t k = 0; k < work_size; ++k);
                    destroy_coroutine(co_next);
                }
            });
            double diff = (tbb::tick_count::now() - tick).seconds() * 1e3;
            times.push_back(diff);
        }
        printf("Test 5 (Native Implementation): Work size: %zu, Concurrency: %d, Time (milliseconds): %.4f\n",
                work_size, concurrency, get_median(times));
    }
}

/************************************************************************/
/* MAIN DRIVER                                                          */
/************************************************************************/

int main() {
    // Serial microbenchmarks
    BenchCoroutinesAllocation();
    BenchReusage();
    BenchContextSwitch();

    // Scalability benchmarks
    // Big work size + no resumable tasks feature (false)
    ScalabilityBenchmark<false>(100000);
    // Big work size + resumable tasks feature (true)
    ScalabilityBenchmark<true>(100000);
    // Small work size + no resumable tasks feature (false)
    ScalabilityBenchmark<false>(1000);
    // Small work size + resumable tasks feature (true)
    ScalabilityBenchmark<true>(1000);
    // No any work + just resumable tasks feature (true)
    ScalabilityBenchmark<true>(0);

    // Native implementation
    // Big work size
    BenchNativeImpl(100000);
    // Small work size
    BenchNativeImpl(1000);
    // Just coroutines/fibers switching
    BenchNativeImpl(0);

    return 0;
}

