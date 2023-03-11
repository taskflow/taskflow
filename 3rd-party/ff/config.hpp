/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/* Author: Massimo Torquati
 *
 */

// This file contains some configuration variables. Some of them are
// particularly critical for performance matters, for example:
// FF_MAPPING_STRING, BLOCKING_MODE, TRACE_FASTFLOW, etc. 
//

#ifndef FF_CONFIG_HPP
#define FF_CONFIG_HPP

#include <cstddef> 
#include <climits>
#if defined(TRACE_FASTFLOW)
#include <iostream>
#endif


/*
 * If NO_DEFAULT_MAPPING is not defined (and if FF_MAPPING_STRING is equal 
 * to ""), the FastFlow library pins each spawned thread to a core context 
 * according to a simple policy: thread 0 to core 0, thread 1
 * to core 1, and so on restarting from the beginning if there are more 
 * threads than core contexts (which is usually not a good idea if 
 * performance matters). 
 * Depending on the OS numbering of core contexts, core 0 and core 1 may be
 * "far away" one each other, e.g., they could be on two distinct CPUs,
 * thus not sharing any level of cache.
 * Therefore, to control the mapping of the thread when the OS numbering
 * is a bit mess (e.g. Power8, AMD machines, some Intel Xeon), you have
 * two options: 
 *  1. to use the Mammut library, which provides a layer for discovering 
 *     core contexts
 *  2. to set the FF_MAPPING_STRING in this file. 
 *
 * For case 2, the simplest option is to run the Bash script 
 * 'mapping_string.sh', which returns a suitable string that can be 
 * copy-paste in the FF_MAPPING_STRING preprocessor variable. 
 * Note that, if you wish (and trust it) the script can modify the 
 * config.hpp file for you.  The script also sets the FF_NUM_CORES and 
 * FF_NUM_REAL_CORES variables. 
 * Example: 
 *  > ./mapping_string.sh
 *  > FF_MAPPING_STRING="0,2,1,3"
 *  > FF_NUM_CORES=4
 *  > FF_NUM_REAL_CORES=2
 *  > Do you want that I change the ./config.hpp file for you? (y/N) y
 *  > This is the new FF_MAPPING_STRING variable in the ./config.hpp file:
 *  > #if !defined MAPPING_STRING
 *  > #define FF_MAPPING_STRING "0,2,1,3"
 *  > #else
 *  > ...
 *
 */
/*
 * NOTE: if FF_MAPPING_STRING is "" (default), FastFlow executes a linear
 *       mapping of threads. 
 */
#if !defined MAPPING_STRING
#define FF_MAPPING_STRING ""
#else
#define FF_MAPPING_STRING MAPPING_STRING
#endif
/* 
 * It is the number of the logical cores of the machine.
 * NOTE: if FF_NUM_CORES is -1 (default), FastFlow will use ff_numCores()
 *       (which is a costly function).
 */
#if !defined NUM_CORES
#define FF_NUM_CORES -1
#else
#define FF_NUM_CORES NUM_CORES
#endif
/* 
 * It is the number of the physical cores of the machine.
 * NOTE: if FF_NUM_REAL_CORES is -1 (default), FastFlow will use 
 *       ff_realNumCores() (which is a costly function)
 */
#if !defined NUM_REAL_CORES
#define FF_NUM_REAL_CORES -1
#else
#define FF_NUM_REAL_CORES NUM_REAL_CORES
#endif


#if defined(FF_BOUNDED_BUFFER)
#define FF_FIXED_SIZE true
#else  // NOTE: by default the queues are unbounded!!!!
#define FF_FIXED_SIZE false
#endif

// WARNING: Do not change the following with SWSR_Ptr_Buffer unless
// you know what your are doing....
#define FFBUFFER uSWSR_Ptr_Buffer

/*
 * This is the default buffer capacity and the default difference between the input
 * and output channels capacity.
 * 
 */
#if !defined(DEFAULT_BUFFER_CAPACITY)
#define DEFAULT_BUFFER_CAPACITY              2048
#endif


/* To save energy and improve hyperthreading performance
 * define the following macro
 */
//#define SPIN_USE_PAUSE 1

/* To enable OPENCL support
 *
 */
//#define FF_OPENCL 1 


/* To enable task callbacks
 *
 * If enabled, 2 callbacks are called by the run-time:
 *  - one before receiving the task in input
 *  - one just after having computed the task (before sending it out)
 */
//#define FF_TASK_CALLBACK 1

/*
 ****** DISTRIBUTED SUPPORT PARAMETERS
 */
#define MAX_RETRIES 1500
#define AGGRESSIVE_TRESHOLD 1000

#define MAXBACKLOG 32

/*
 ****** END DISTRIBUTED VERSION PARAMETERS
 */


#if defined(TRACE_FASTFLOW)
#define FFTRACE(x) x
#else
#define FFTRACE(x)
#endif

#if defined(BLOCKING_MODE)
#define FF_RUNTIME_MODE true
#else
#define FF_RUNTIME_MODE false   // by default the run-time is in nonblocking mode
#endif

/* Used in blocking mode to limit the amount of time 
 * before checking again the input/output queue.
 * NOTE: it cannot be greater than 1e+9 (i.e. 1sec)
 */
#define FF_TIMEDWAIT_NS   200000

/*
 * Used in the ordered farm pattern (ff_OFarm). 
 * It is the maximum amount of data elements buffered in the farm's collector
 * to preserve output ordering. In some case such value has to be increased
 * (see set_scheduling_ondemand in ff_ofarm.hpp)
 */
#define DEF_OFARM_ONDEMAND_MEMORY 10000


// If the following is defined, then an initial barrier is executed among all threads
// to ensure that all threads are started. It can be commented out if that condition 
// is not needed. Usually it is useful for debugging purposes.
// #define FF_INITIAL_BARRIER

// Which barrier implementation to use
#if !defined(BARRIER_T)
#define BARRIER_T             spinBarrier
#endif

// maximum number of threads that can be spawned
#if !defined(MAX_NUM_THREADS)
#define MAX_NUM_THREADS       512 
#endif

// maximum number of workers in a farm
#define DEF_MAX_NUM_WORKERS   (MAX_NUM_THREADS-2)

// NOTE: BACKOFF_MIN/MAX are lower and upper bound backoff values.
// Notice that backoff bounds are highly dependent on the system and 
// from the concurrency levels. This values should be carefully tuned
// in order to achieve the maximum performance.
#if !defined(BACKOFF_MIN)
#define BACKOFF_MIN 128
#endif
#if !defined(BACKOFF_MAX)
#define BACKOFF_MAX 1024
#endif

#if !defined(CACHE_LINE_SIZE)
#define CACHE_LINE_SIZE 64
#endif


// TODO:
//#if defined(NO_CMAKE_CONFIG)

// TODO change to __GNUC__ that is portable. GNUC specific code currently works
// on linux only
#if defined(__USE_GNU) //linux
//#if defined(__GNUC__) 
#define HAVE_PTHREAD_SETAFFINITY_NP 1
//#warning "Is GNU compiler"
#endif 

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && (__MAC_OS_X_VERSION_MIN_REQUIRED >= 1050)
#define MAC_OS_X_HAS_AFFINITY 1
#else
#define MAC_OS_X_HAS_AFFINITY 0
#endif
#endif

//#else
// the config.h file will be generated by cmake
//#include <ff/config.h>
//#endif // NO_CMAKE_CONFIG

#if defined(USE_CMAKE_CONFIG) && !defined(NOT_USE_CMAKE_CONFIG)
#include <cmake.modules/ffconfig.h>
#endif

// OpenCL additional code needed to compile kernels
#define FF_OPENCL_DATATYPES_FILE "ff_opencl_datatypes.cl"

// Convenience macros.
#define FF_IGNORE_UNUSED(x) static_cast<void>(x)

#endif /* FF_CONFIG_HPP */
