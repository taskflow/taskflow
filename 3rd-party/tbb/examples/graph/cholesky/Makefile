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

# GNU Makefile that builds and runs example.
run_cmd=
PROG=cholesky
ARGS=4 2

# The C++ compiler
ifneq (,$(shell which icpc 2>/dev/null))
CXX=icpc
endif # which icpc

ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib  -Wl,-rpath,$(MKLROOT)/lib
# MKL support
override MKL_LIBS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
else
ifeq ($(shell uname), Linux)
LIBS+= -lrt 
endif
# MKL support
override MKL_LIBS += -lmkl_rt
endif

# autodetect C++11 support
include ../../common/examples-common.inc

all:	release test

release: *.cpp
	$(CXX) -O2  -DNDEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb $(MKL_LIBS) $(LIBS) $(CXX0XFLAGS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb_debug $(MKL_LIBS) $(LIBS) $(CXX0XFLAGS)

profile: override CXXFLAGS += -DTBB_USE_THREADING_TOOLS -g
profile: release

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)
