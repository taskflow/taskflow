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

function(tbb_conf_gen_print_help)
    message("Usage: cmake -DINSTALL_DIR=<config_install_dir> -DSYSTEM_NAME=Linux|Darwin|Windows <parameters> -P tbb_config_generator.cmake

Parameters:
  For custom TBB package:
    -DTBB_VERSION_FILE=<tbb_version_file>
    -DTBB_VERSION=<major>.<minor>.<interface> (alternative to TBB_VERSION_FILE)
    -DINC_REL_PATH=<relative_path_to_tbb_headers>
    -DLIB_REL_PATH=<relative_path_to_tbb_libs>
    -DBIN_REL_PATH=<relative_path_to_tbb_dlls> (only for Windows)
  For installed TBB:
    -DINC_PATH=<path_to_installed_tbb_headers>
    -DLIB_PATH=<path_to_installed_tbb_libs>
    -DBIN_PATH=<path_to_installed_tbb_dlls> (only for Windows)
")
endfunction()

if (NOT DEFINED INSTALL_DIR)
    tbb_conf_gen_print_help()
    message(FATAL_ERROR "Required parameter INSTALL_DIR is not defined")
endif()

if (NOT DEFINED SYSTEM_NAME)
    tbb_conf_gen_print_help()
    message(FATAL_ERROR "Required parameter SYSTEM_NAME is not defined")
endif()

foreach (arg TBB_VERSION INC_REL_PATH LIB_REL_PATH BIN_REL_PATH TBB_VERSION_FILE INC_PATH LIB_PATH BIN_PATH)
    set(optional_args ${optional_args} ${arg} ${${arg}})
endforeach()

include(${CMAKE_CURRENT_LIST_DIR}/TBBInstallConfig.cmake)
tbb_install_config(INSTALL_DIR ${INSTALL_DIR} SYSTEM_NAME ${SYSTEM_NAME} ${optional_args})
message(STATUS "TBBConfig files were created in ${INSTALL_DIR}")
