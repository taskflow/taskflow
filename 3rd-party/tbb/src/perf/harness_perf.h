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

#ifndef __TBB_HARNESS_PERF
#define __TBB_HARNESS_PERF

#include <iterator>
#include <algorithm>

namespace harness_perf {

template<typename InputIterator>
typename InputIterator::value_type median(InputIterator first, InputIterator last) {
    std::sort(first, last);
    typename InputIterator::difference_type distance = std::distance(first, last);
    std::advance(first, distance / 2 - 1);
    if (distance % 2 == 0)
        return typename InputIterator::value_type((*first + *(++first)) / 2);
    else
        return typename InputIterator::value_type(*first);
}

} // namespace harness_perf

#endif // !__TBB_HARNESS_PERF

