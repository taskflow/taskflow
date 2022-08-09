/*
    Copyright (c) 2018-2020 Intel Corporation

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

#include "harness_allocator_overload.h"
#include "harness.h"

// Disabling malloc proxy via env variable is available only on Windows for now
#if MALLOC_WINDOWS_OVERLOAD_ENABLED

#define TEST_SYSTEM_COMMAND "test_malloc_overload_disable.exe"

#include "tbb/tbbmalloc_proxy.h"
#include "../tbb/tbb_environment.h"

const size_t SmallObjectSize = 16;
const size_t LargeObjectSize = 2*8*1024;
const size_t HugeObjectSize = 2*1024*1024;

void CheckWindowsProxyDisablingViaMemSize( size_t ObjectSize ) {
    void* ptr = malloc(ObjectSize);
    /*
     * If msize returns 0 - tbbmalloc doesn't contain this object in it`s memory
     * Also msize check that proxy lib is linked
     */
    ASSERT(!__TBB_malloc_safer_msize(ptr,NULL), "Malloc replacement is not deactivated");
    free(ptr);

}

int TestMain() {
    if (!tbb::internal::GetBoolEnvironmentVariable("TBB_MALLOC_DISABLE_REPLACEMENT"))
    {
        Harness::SetEnv("TBB_MALLOC_DISABLE_REPLACEMENT","1");
        if ((system(TEST_SYSTEM_COMMAND)) != 0) {
            REPORT("Test error: unable to run the command: %s", TEST_SYSTEM_COMMAND);
            exit(-1);
        }
        // We must execute exit(0) to avoid duplicate "Done" printing.
        exit(0);
    }
    else
    {
        // Check SMALL objects replacement disable
        CheckWindowsProxyDisablingViaMemSize(SmallObjectSize);
        // Check LARGE objects replacement disable
        CheckWindowsProxyDisablingViaMemSize(LargeObjectSize);
        // Check HUGE objects replacement disable
        CheckWindowsProxyDisablingViaMemSize(HugeObjectSize);
    }
    return Harness::Done;
}
#else // MALLOC_WINDOWS_OVERLOAD_ENABLED
int TestMain() {
    return Harness::Skipped;
}
#endif // MALLOC_WINDOWS_OVERLOAD_ENABLED
