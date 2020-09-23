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

#ifndef GET_DEFAULT_NUM_THREADS_H_
#define GET_DEFAULT_NUM_THREADS_H_

#include "tbb/global_control.h"

namespace utility {
    inline int get_default_num_threads() {
        #if __TBB_MIC_OFFLOAD
            #pragma offload target(mic) out(default_num_threads)
        #endif // __TBB_MIC_OFFLOAD
        static size_t default_num_threads = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
        return static_cast<int>(default_num_threads);
    }
}

#endif /* GET_DEFAULT_NUM_THREADS_H_ */
