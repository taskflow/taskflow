# Copyright (c) 2005-2020 Intel Corporation
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

# detect if a compiler can support C++11
# If CXX0XFLAGS already set, do not change it
ifneq (,$(findstring icc, $(CXX))$(findstring icpc, $(CXX))$(findstring clang++, $(CXX)))
    # every supported icc or clang is OK
    CXX0XFLAGS ?= -std=c++11
else
    ifneq (,$(findstring g++, $(CXX))$(findstring gcc, $(CXX)))
        ifneq (, $(strip $(shell $(CXX) -v 2>&1 | grep "clang-")))
            # This is clang actually,
            # every supported clang is OK
            CXX0XFLAGS ?= -std=c++11
        else
            # support of lambda started GCC 4.5
            ifneq (, $(strip $(shell g++ -dumpfullversion -dumpversion | egrep "^(4\.[5-9]|[5-9]|1[0-9])")))
               CXX0XFLAGS ?= -std=c++11
            endif
        endif
    endif
endif

