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

include(CMakeParseArguments)

# Save the location of Intel TBB CMake modules here, as it will not be possible to do inside functions,
# see for details: https://cmake.org/cmake/help/latest/variable/CMAKE_CURRENT_LIST_DIR.html
set(_tbb_cmake_module_path ${CMAKE_CURRENT_LIST_DIR})

function(tbb_install_config)
    set(oneValueArgs INSTALL_DIR
                     SYSTEM_NAME
                     LIB_REL_PATH INC_REL_PATH BIN_REL_PATH TBB_VERSION TBB_VERSION_FILE
                     LIB_PATH BIN_PATH INC_PATH)                                      # If TBB is installed on the system

    cmake_parse_arguments(tbb_IC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    get_filename_component(config_install_dir ${tbb_IC_INSTALL_DIR} ABSOLUTE)
    file(MAKE_DIRECTORY ${config_install_dir})

    # --- TBB_LIB_REL_PATH handling ---
    set(TBB_LIB_REL_PATH "../../../lib")

    if (tbb_IC_LIB_REL_PATH)
        file(TO_CMAKE_PATH ${tbb_IC_LIB_REL_PATH} TBB_LIB_REL_PATH)
    endif()

    if (tbb_IC_LIB_PATH)
        get_filename_component(lib_abs_path ${tbb_IC_LIB_PATH} ABSOLUTE)
        file(RELATIVE_PATH TBB_LIB_REL_PATH ${config_install_dir} ${lib_abs_path})
        unset(lib_abs_path)
    endif()
    # ------

    # --- TBB_BIN_REL_PATH handling ---
    set(TBB_BIN_REL_PATH "../../../bin")

    if (tbb_IC_BIN_REL_PATH)
        file(TO_CMAKE_PATH ${tbb_IC_BIN_REL_PATH} TBB_BIN_REL_PATH)
    endif()

    if (tbb_IC_BIN_PATH)
        get_filename_component(bin_abs_path ${tbb_IC_BIN_PATH} ABSOLUTE)
        file(RELATIVE_PATH TBB_BIN_REL_PATH ${config_install_dir} ${bin_abs_path})
        unset(bin_abs_path)
    endif()
    # ------

    # --- TBB_INC_REL_PATH handling ---
    set(TBB_INC_REL_PATH "../../../include")

    if (tbb_IC_INC_REL_PATH)
        file(TO_CMAKE_PATH ${tbb_IC_INC_REL_PATH} TBB_INC_REL_PATH)
    endif()

    if (tbb_IC_INC_PATH)
        get_filename_component(inc_abs_path ${tbb_IC_INC_PATH} ABSOLUTE)
        file(RELATIVE_PATH TBB_INC_REL_PATH ${config_install_dir} ${inc_abs_path})
        unset(inc_abs_path)
    endif()
    # ------

    # --- TBB_VERSION handling ---
    if (tbb_IC_TBB_VERSION)
        set(TBB_VERSION ${tbb_IC_TBB_VERSION})
    else()
        set(tbb_version_file "${config_install_dir}/${TBB_INC_REL_PATH}/tbb/tbb_stddef.h")
        if (tbb_IC_TBB_VERSION_FILE)
            set(tbb_version_file ${tbb_IC_TBB_VERSION_FILE})
        endif()

        file(READ ${tbb_version_file} _tbb_stddef)
        string(REGEX REPLACE ".*#define TBB_VERSION_MAJOR ([0-9]+).*" "\\1" _tbb_ver_major "${_tbb_stddef}")
        string(REGEX REPLACE ".*#define TBB_VERSION_MINOR ([0-9]+).*" "\\1" _tbb_ver_minor "${_tbb_stddef}")
        string(REGEX REPLACE ".*#define TBB_INTERFACE_VERSION ([0-9]+).*" "\\1" TBB_INTERFACE_VERSION "${_tbb_stddef}")
        set(TBB_VERSION "${_tbb_ver_major}.${_tbb_ver_minor}")
    endif()
    # ------

    set(tbb_system_name ${CMAKE_SYSTEM_NAME})
    if (tbb_IC_SYSTEM_NAME)
        set(tbb_system_name ${tbb_IC_SYSTEM_NAME})
    endif()

    if (tbb_system_name STREQUAL "Linux")
        set(TBB_LIB_PREFIX "lib")
        set(TBB_LIB_EXT "so.2")
        set(TBB_IMPLIB_RELEASE "")
        set(TBB_IMPLIB_DEBUG "")
    elseif (tbb_system_name STREQUAL "Darwin")
        set(TBB_LIB_PREFIX "lib")
        set(TBB_LIB_EXT "dylib")
        set(TBB_IMPLIB_RELEASE "")
        set(TBB_IMPLIB_DEBUG "")
    elseif (tbb_system_name STREQUAL "Windows")
        set(TBB_LIB_PREFIX "")
        set(TBB_LIB_EXT "dll")
        # .lib files installed to TBB_LIB_REL_PATH (e.g. <prefix>/lib);
        # .dll files installed to TBB_BIN_REL_PATH (e.g. <prefix>/bin);
        # Expand TBB_LIB_REL_PATH here in IMPORTED_IMPLIB property and
        # redefine it with TBB_BIN_REL_PATH value to properly fill IMPORTED_LOCATION property in TBBConfig.cmake.in template.
        set(TBB_IMPLIB_RELEASE "
                                      IMPORTED_IMPLIB_RELEASE \"\${CMAKE_CURRENT_LIST_DIR}/${TBB_LIB_REL_PATH}/\${_tbb_component}.lib\"")
        set(TBB_IMPLIB_DEBUG "
                                      IMPORTED_IMPLIB_DEBUG \"\${CMAKE_CURRENT_LIST_DIR}/${TBB_LIB_REL_PATH}/\${_tbb_component}_debug.lib\"")
        set(TBB_LIB_REL_PATH ${TBB_BIN_REL_PATH})
    else()
        message(FATAL_ERROR "Unsupported OS name: ${tbb_system_name}")
    endif()

    configure_file(${_tbb_cmake_module_path}/templates/TBBConfig.cmake.in ${config_install_dir}/TBBConfig.cmake @ONLY)
    configure_file(${_tbb_cmake_module_path}/templates/TBBConfigVersion.cmake.in ${config_install_dir}/TBBConfigVersion.cmake @ONLY)
endfunction()
