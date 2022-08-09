# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(fibonacci CXX)

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CXX_STANDARD_REQUIRED ON)
endif()

add_executable(fibonacci Fibonacci.cpp)

# find_package will search for available TBBConfig using variables CMAKE_PREFIX_PATH and TBB_DIR.
find_package(TBB REQUIRED tbb)

target_link_libraries(fibonacci
                      ${TBB_IMPORTED_TARGETS}      # Link TBB imported targets to the executable; "TBB::tbb" can be used instead of "${TBB_IMPORTED_TARGETS}".
                      $<$<PLATFORM_ID:Linux>:rt>)  # Link "rt" library on Linux
