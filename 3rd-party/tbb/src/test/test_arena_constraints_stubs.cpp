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
#include "tbb/tbb_config.h"

#include "harness.h"

#include "tbb/task_arena.h"
#include "tbb/task_scheduler_init.h"

#include <vector>

void test_stubs(std::vector<int> numa_indexes) {
    ASSERT(numa_indexes.size() == 1, "Number of NUMA nodes must be pinned to 1,"
                                     " if we have no HWLOC on the system.");
    ASSERT(numa_indexes[0] == -1, "Index of NUMA node must be pinned to 0,"
                                 " if we have no HWLOC on the system.");
    ASSERT(tbb::info::default_concurrency(numa_indexes[0]) == tbb::task_scheduler_init::default_num_threads(),
        "Concurrency for NUMA node must be equal to default_num_threads(),"
        " if we have no HWLOC on the system.");
}

int TestMain() {
    std::vector<int> numa_indexes = tbb::info::numa_nodes();
    test_stubs(numa_indexes);

    return Harness::Done;
}
