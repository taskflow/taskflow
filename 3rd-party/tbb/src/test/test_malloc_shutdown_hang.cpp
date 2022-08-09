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

#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"

#include <tbb/task.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_scheduler_init.h>

// Lets slow down the main thread on exit
const int MAX_DELAY = 5;
struct GlobalObject {
    ~GlobalObject() {
        Harness::Sleep(rand( ) % MAX_DELAY);
    }
} go;

void allocatorRandomThrashing() {
    const int ARRAY_SIZE = 1000;
    const int MAX_ITER = 10000;
    const int MAX_ALLOC = 10 * 1024 * 1024;

    void *arr[ARRAY_SIZE] = {0};
    for (int i = 0; i < rand() % MAX_ITER; ++i) {
        // Random allocation size for random arrays
        for (int j = 0; j < rand() % ARRAY_SIZE; ++j) {
            arr[j] = scalable_malloc(rand() % MAX_ALLOC);
        }
        // Deallocate everything
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            scalable_free(arr[j]);
            arr[j] = NULL;
        }
    }
}

struct AllocatorThrashTask : tbb::task {
    tbb::task* execute() __TBB_override {
        allocatorRandomThrashing();
        return NULL;
    }
};

void hangOnExitReproducer() {
    const int P = tbb::task_scheduler_init::default_num_threads();
    for (int i = 0; i < P-1; i++) {
        // Enqueue tasks for workers
        tbb::task::enqueue(*new (tbb::task::allocate_root()) AllocatorThrashTask());
    }
}

#if (_WIN32 || _WIN64) && !__TBB_WIN8UI_SUPPORT
#include <process.h> // _spawnl
void processSpawn(const char* self) {
    _spawnl(_P_WAIT, self, self, "1", NULL);
}
#elif __linux__ || __APPLE__
#include <unistd.h> // fork/exec
#include <sys/wait.h> // waitpid
void processSpawn(const char* self) {
    pid_t pid = fork();
    if (pid == -1) {
        REPORT("ERROR: fork failed.\n");
    } else if (pid == 0) { // child
        execl(self, self, "1", NULL);
        REPORT("ERROR: exec never returns\n");
        exit(1);
    } else { // parent
        int status;
        waitpid(pid, &status, 0);
    }
}
#else
void processSpawn(const char* /*self*/) {
    REPORT("Known issue: no support for process spawn on this platform.\n");
    REPORT("done\n");
    exit(0);
}
#endif

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (push)
#pragma warning (disable: 4702)  /* Unreachable code */
#endif

HARNESS_EXPORT
int main(int argc, char* argv[]) {
    ParseCommandLine( argc, argv );

    // Executed from child processes
    if (argc == 2 && strcmp(argv[1],"1") == 0) {
        hangOnExitReproducer();
        return 0;
    }

    // The number of executions is a tradeoff
    // between execution time and NBTS statistics
    const int EXEC_TIMES = 100;
    const char* self = argv[0];
    for (int i = 0; i < EXEC_TIMES; i++) {
        processSpawn(self);
    }

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (pop)
#endif

    REPORT("done\n");
    return 0;
}

