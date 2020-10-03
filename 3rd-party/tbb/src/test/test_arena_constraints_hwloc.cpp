/*
  Copyright (c) 2019-2020 Intel Corporation

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

#define TBB_PREVIEW_NUMA_SUPPORT 1
#define __TBB_EXTRA_DEBUG 1

#ifndef NUMBER_OF_PROCESSORS_GROUPS
#define NUMBER_OF_PROCESSORS_GROUPS 1
#endif

#include "tbb/tbb_config.h"

#include "harness.h"
#include "harness_memory.h"
#include "harness_barrier.h"

#include "tbb/task_arena.h"
#include "tbb/task_scheduler_init.h"

#include <vector>

#if __TBB_CPP11_PRESENT
#include <atomic>
#endif /*__TBB_CPP11_PRESENT*/

#if _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4100 )
#endif
#include <hwloc.h>
#if _MSC_VER
#pragma warning( pop )
#endif

#include "tbb/concurrent_unordered_set.h"
#include "tbb/parallel_for.h"

// Macro to check hwloc interfaces return codes
#define hwloc_assert_ex(command, ...)                                          \
        ASSERT(command(__VA_ARGS__) >= 0, "Error occured inside hwloc call.");

namespace numa_validation {
    namespace {
        class system_info_t {
            hwloc_topology_t topology;

            hwloc_nodeset_t process_node_set;
            hwloc_cpuset_t  process_cpu_set;

            hwloc_cpuset_t buffer_cpu_set;
            hwloc_cpuset_t buffer_node_set;

            // hwloc_cpuset_t, hwloc_nodeset_t (inherited from hwloc_bitmap_t ) is pointers,
            // so we must manage memory allocation and deallocation
            typedef tbb::concurrent_unordered_set<hwloc_bitmap_t> memory_handler_t;
            memory_handler_t memory_handler;

            bool is_initialized;
        public:
            system_info_t() : memory_handler() {
                is_initialized = false;
            }

            void initialize() {
                if (is_initialized) return;

                hwloc_assert_ex(hwloc_topology_init, &topology);
                hwloc_assert_ex(hwloc_topology_load, topology);

                if ( Harness::GetIntEnv("NUMBER_OF_PROCESSORS_GROUPS") > 1 ) {
                    process_cpu_set  = hwloc_bitmap_dup(hwloc_topology_get_complete_cpuset (topology));
                    process_node_set = hwloc_bitmap_dup(hwloc_topology_get_complete_nodeset(topology));
                } else {
                    process_cpu_set  = hwloc_bitmap_alloc();
                    process_node_set = hwloc_bitmap_alloc();

                    hwloc_assert_ex(hwloc_get_cpubind, topology, process_cpu_set, 0);
                    hwloc_cpuset_to_nodeset(topology, process_cpu_set, process_node_set);
                }

                // If system contains no NUMA nodes, HWLOC 1.11 returns an infinitely filled bitmap.
                // hwloc_bitmap_weight() returns negative value for such bitmaps, so we use this check
                // to workaround this case.
                if (hwloc_bitmap_weight(process_node_set) <= 0) {
                    hwloc_bitmap_only(process_node_set, 0);
                }

// Debug macros for test topology parser validation
#if NUMBER_OF_NUMA_NODES
                ASSERT(hwloc_bitmap_weight(process_node_set) == NUMBER_OF_NUMA_NODES,
                    "Manual NUMA nodes count check.");
#endif /*NUMBER_OF_NUMA_NODES*/

                buffer_cpu_set  = hwloc_bitmap_alloc();
                buffer_node_set = hwloc_bitmap_alloc();

                is_initialized = true;
            }

            ~system_info_t() {
                if (is_initialized) {
                    for (memory_handler_t::iterator it = memory_handler.begin();
                        it != memory_handler.end(); it++) {
                        hwloc_bitmap_free(*it);
                    }
                    hwloc_bitmap_free(process_cpu_set);
                    hwloc_bitmap_free(process_node_set);
                    hwloc_bitmap_free(buffer_cpu_set);
                    hwloc_bitmap_free(buffer_node_set);

                    hwloc_topology_destroy(topology);
                }
            }

            hwloc_bitmap_t allocate_empty_affinity_mask() {
                __TBB_ASSERT(is_initialized, "Call of uninitialized system_info");
                hwloc_bitmap_t result = hwloc_bitmap_alloc();
                memory_handler.insert(result);
                return result;
            }

            hwloc_cpuset_t allocate_current_cpu_set() {
                __TBB_ASSERT(is_initialized, "Call of uninitialized system_info");
                hwloc_cpuset_t current_affinity_mask = allocate_empty_affinity_mask();
                hwloc_assert_ex(hwloc_get_cpubind, topology, current_affinity_mask, HWLOC_CPUBIND_THREAD );
                ASSERT(!hwloc_bitmap_iszero(current_affinity_mask), "Empty current affinity mask.");
                return current_affinity_mask;
            }

            hwloc_const_cpuset_t get_process_cpu_set() {
                __TBB_ASSERT(is_initialized, "Call of uninitialized system_info");
                return process_cpu_set;
            }

            hwloc_const_nodeset_t get_process_node_set() {
                __TBB_ASSERT(is_initialized, "Call of uninitialized system_info");
                return process_node_set;
            }

            int numa_node_max_concurrency(int index) {
                __TBB_ASSERT(is_initialized, "Call of uninitialized system_info");
                hwloc_bitmap_only(buffer_node_set, index);
                hwloc_cpuset_from_nodeset(topology, buffer_cpu_set, buffer_node_set);
                hwloc_bitmap_and(buffer_cpu_set, buffer_cpu_set, process_cpu_set);
                ASSERT(hwloc_bitmap_weight(buffer_cpu_set) > 0, "Negative concurrency.");
                return hwloc_bitmap_weight(buffer_cpu_set);
            }
        };

        static system_info_t system_info;
    } /*internal namespace*/

typedef hwloc_bitmap_t affinity_mask;
typedef hwloc_const_bitmap_t const_affinity_mask;

void initialize_system_info() { system_info.initialize(); }

affinity_mask allocate_current_cpu_set() {
    return system_info.allocate_current_cpu_set();
}

bool affinity_masks_isequal(const_affinity_mask first, const_affinity_mask second) {
    return hwloc_bitmap_isequal(first, second) ? true : false;
}

bool affinity_masks_intersects(const_affinity_mask first, const_affinity_mask second) {
    return hwloc_bitmap_intersects(first, second) ? true : false;
}

void validate_topology_information(std::vector<int> numa_indexes) {
    // Generate available numa nodes bitmap
    const_affinity_mask process_node_set = system_info.get_process_node_set();

    // Parse input indexes list to numa nodes bitmap
    affinity_mask merged_input_node_set = system_info.allocate_empty_affinity_mask();
    int whole_system_concurrency = 0;
    for (unsigned i = 0; i < numa_indexes.size(); i++) {
        ASSERT(!hwloc_bitmap_isset(merged_input_node_set, numa_indexes[i]), "Indices are repeated.");
        hwloc_bitmap_set(merged_input_node_set, numa_indexes[i]);

        ASSERT(tbb::info::default_concurrency(numa_indexes[i]) ==
            system_info.numa_node_max_concurrency(numa_indexes[i]),
            "Wrong default concurrency value.");
        whole_system_concurrency += tbb::info::default_concurrency(numa_indexes[i]);
    }

    ASSERT(whole_system_concurrency == tbb::task_scheduler_init::default_num_threads(),
           "Wrong whole system default concurrency level.");
    ASSERT(affinity_masks_isequal(process_node_set, merged_input_node_set),
           "Input array of indices is not equal with proccess numa node set.");
}

} /*namespace numa_validation*/

#if __TBB_CPP11_PRESENT
namespace numa_validation {
    template <typename It>
    typename std::enable_if<std::is_same<typename std::iterator_traits<It>::value_type, affinity_mask>::value, void>::
    type affinity_set_verification(It begin, It end) {
        affinity_mask buffer_mask = system_info.allocate_empty_affinity_mask();
        for (auto it = begin; it != end; it++) {
            ASSERT(!hwloc_bitmap_intersects(buffer_mask, *it),
                   "Bitmaps that are binded to different nodes are intersects.");
            // Add masks to buffer_mask to concatenate process affinity mask
            hwloc_bitmap_or(buffer_mask, buffer_mask,  *it);
        }

        ASSERT(affinity_masks_isequal(system_info.get_process_cpu_set(), buffer_mask),
               "Some cores was not included to bitmaps.");
    }
} /*namespace numa_validation*/

struct execute_wrapper {
    template <typename Callable>
    void emplace_function(tbb::task_arena& ta, Callable functor) {
        ta.execute(functor);
    }
};

struct enqueue_wrapper {
    template <typename Callable>
    void emplace_function(tbb::task_arena& ta, Callable functor) {
        ta.enqueue(functor);
    }
};

template <typename It, typename FuncWrapper>
typename std::enable_if<std::is_same<typename std::iterator_traits<It>::value_type, tbb::task_arena>::value, void>::
type test_numa_binding_impl(It begin, It end, FuncWrapper wrapper) {
    tbb::concurrent_unordered_set<numa_validation::affinity_mask> affinity_masks;
    std::atomic<unsigned> counter(0), expected_count(0);

    auto affinity_mask_checker = [&counter, &affinity_masks]() {
        affinity_masks.insert(numa_validation::allocate_current_cpu_set());
        counter++;
    };

    for (auto it = begin; it != end; it++) {
        expected_count++;
        wrapper.emplace_function(*it, affinity_mask_checker);
    }

    // Wait for all spawned tasks
    while (counter != expected_count) {}
    numa_validation::affinity_set_verification(affinity_masks.begin(),affinity_masks.end());
}

void test_numa_binding(std::vector<int> numa_indexes_vector) {

    std::vector<tbb::task_arena> arenas(numa_indexes_vector.size());

    for(unsigned i = 0; i < numa_indexes_vector.size(); i++) {
        // Bind arenas to numa nodes
        arenas[i].initialize(tbb::task_arena::constraints(numa_indexes_vector[i]));
    }

    test_numa_binding_impl(arenas.begin(), arenas.end(), execute_wrapper());
    test_numa_binding_impl(arenas.begin(), arenas.end(), enqueue_wrapper());
}

void recursive_arena_binding(int*, int, numa_validation::affinity_mask);

void recursive_arena_binding(int* numa_indexes, size_t count,
    std::vector<numa_validation::affinity_mask>& affinity_masks) {
    if (count > 0) {
        tbb::task_arena current_level_arena;
        current_level_arena.initialize(tbb::task_arena::constraints(numa_indexes[count - 1]));
        current_level_arena.execute(
            [&numa_indexes, &count, &affinity_masks]() {
                affinity_masks.push_back(numa_validation::allocate_current_cpu_set());
                recursive_arena_binding(numa_indexes, --count, affinity_masks);
            }
        );
    } else {
        // Validation of assigned affinity masks at the deepest recursion step
        numa_validation::affinity_set_verification(affinity_masks.begin(), affinity_masks.end());
    }

    if (!affinity_masks.empty()) {
        ASSERT(numa_validation::affinity_masks_isequal(affinity_masks.back(),
            numa_validation::allocate_current_cpu_set()),
            "After binding to different NUMA node thread affinity was not returned to previous state.");
        affinity_masks.pop_back();
    }
}

void test_nested_numa_binding(std::vector<int> numa_indexes_vector) {
    std::vector<numa_validation::affinity_mask> affinity_masks;
    recursive_arena_binding(numa_indexes_vector.data(), numa_indexes_vector.size(), affinity_masks);
}

void test_memory_leak(std::vector<int> numa_indexes_vector){
    size_t big_number = 1000;
    size_t current_memory_usage = 0, previous_memory_usage = 0, stability_counter=0;
    for (size_t i = 0; i < big_number; i++) {
        { /* All DTORs must be called before GetMemoryUsage() call*/
            std::vector<tbb::task_arena> arenas(numa_indexes_vector.size());
            std::vector<Harness::SpinBarrier> barriers(numa_indexes_vector.size());

            for(unsigned j = 0; j < numa_indexes_vector.size(); j++) {
                arenas[j].initialize(tbb::task_arena::constraints(numa_indexes_vector[j]));
                barriers[j].initialize(arenas[j].max_concurrency());
                Harness::SpinBarrier& barrier_ref = barriers[j];
                arenas[j].enqueue([&barrier_ref](){
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, tbb::this_task_arena::max_concurrency()),
                        [&barrier_ref](const tbb::blocked_range<size_t>&){
                            barrier_ref.wait();
                        });
                });
            }

            for(unsigned j = 0; j < numa_indexes_vector.size(); j++) {
                arenas[j].debug_wait_until_empty();
            }
        }

        current_memory_usage = GetMemoryUsage();
        stability_counter = current_memory_usage==previous_memory_usage ? stability_counter + 1 : 0;
        // If the amount of used memory has not changed during 10% of executions,
        // then we can assume that the check was successful
        if (stability_counter > big_number / 10) return;
        previous_memory_usage = current_memory_usage;
    }
    ASSERT(false, "Seems like we get memory leak here.");
}

// Check that arena constraints are copied during copy construction
void test_arena_constraints_copying(std::vector<int> numa_indexes) {
    for (auto index: numa_indexes) {
        numa_validation::affinity_mask constructed_mask, copied_mask;

        tbb::task_arena constructed{tbb::task_arena::constraints(index)};
        constructed.execute([&constructed_mask](){
            constructed_mask = numa_validation::allocate_current_cpu_set();
        });

        tbb::task_arena copied(constructed);
        copied.execute([&copied_mask](){
            copied_mask = numa_validation::allocate_current_cpu_set();
        });

        ASSERT(numa_validation::affinity_masks_isequal(constructed_mask, copied_mask),
                    "Affinity mask brokes during copy construction");
    }
}
#endif /*__TBB_CPP11_PRESENT*/

//TODO: Write a test that checks for memory leaks during dynamic link/unlink of TBBbind.
int TestMain() {
#if _WIN32 && !_WIN64
    // HWLOC cannot proceed affinity masks on Windows in 32-bit mode if there are more than 32 logical CPU.
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    if (si.dwNumberOfProcessors > 32) return Harness::Skipped;
#endif // _WIN32 && !_WIN64

    numa_validation::initialize_system_info();

    std::vector<int> numa_indexes = tbb::info::numa_nodes();
    numa_validation::validate_topology_information(numa_indexes);

#if __TBB_CPP11_PRESENT
    test_numa_binding(numa_indexes);
    test_nested_numa_binding(numa_indexes);
    test_memory_leak(numa_indexes);
    test_arena_constraints_copying(numa_indexes);
#endif /*__TBB_CPP11_PRESENT*/

    return Harness::Done;
}
