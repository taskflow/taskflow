/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */

#ifndef FF_PLATFORM_HPP
#define FF_PLATFORM_HPP

#include <ff/platforms/liblfds.h>

// APPLE specific backward compatibility 

// posix_memalign is available on OS X starting with 10.6
#if defined(__APPLE__)
#include <Availability.h>
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
#define __FF_HAS_POSIX_MEMALIGN 1
#else
//#warning "Redefining posix_memalign"
#include <errno.h>
inline static int posix_memalign(void **memptr, size_t alignment, size_t size)
{
    if (memptr && (*memptr = malloc(size))) return 0; 
    else return (ENOMEM);
}
#endif
#endif
 



#if defined(_WIN32)
#pragma unmanaged

#define NOMINMAX

#include <ff/platforms/pthread_minport_windows.h>
#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
//#define CACHE_LINE_SIZE 64
#define __WIN_ALIGNED_16__ __declspec(align(16))

// Thread specific storage
#define __thread __declspec(thread)


// Only x86 and x86_64 are currently supported for Windows OS
INLINE void WMB() {} 
INLINE void PAUSE() {}

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

INLINE static int posix_memalign(void **memptr,size_t alignment, size_t sz)
{
    *memptr =  _aligned_malloc(sz, alignment);
	return(!memptr);
}


INLINE static void posix_memalign_free(void* mem)
{
    _aligned_free(mem);
}

 // Other

#include <string>
typedef unsigned long useconds_t;
//#define strtoll std::stoll
#define strtoll _strtoi64

#define sleep(SECS) Sleep(SECS)

INLINE static int usleep(unsigned long microsecs) {
  if (microsecs > 100000)
    /* At least 100 mS. Typical best resolution is ~ 15ms */
    Sleep (microsecs/ 1000);
  else
    {
      /* Use Sleep for the largest part, and busy-loop for the rest. */
      static double frequency;
      if (frequency == 0)
        {
          LARGE_INTEGER freq;
          if (!QueryPerformanceFrequency (&freq))
            {
              /* Cannot use QueryPerformanceCounter. */
              Sleep (microsecs / 1000);
              return 0;
            }
          frequency = (double) freq.QuadPart / 1000000000.0;
        }
      long long expected_counter_difference = 1000 * microsecs * (long long) frequency;
      int sleep_part = (int) (microsecs) / 1000 - 10;
      LARGE_INTEGER before;
      QueryPerformanceCounter (&before);
      long long expected_counter = before.QuadPart + 
expected_counter_difference;
      if (sleep_part > 0)
        Sleep (sleep_part);
      for (;;)
        {
          LARGE_INTEGER after;
          QueryPerformanceCounter (&after);
          if (after.QuadPart >= expected_counter)
            break;
        }
    }
	return(0);
}


//#define __TICKS2WAIT 1000
#define random rand
#define srandom srand
#define getpid _getpid
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

/*
#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED
struct timeval {
  long tv_sec;
  long tv_usec;
};
#endif 
*/

struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
INLINE static int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }
 
  return 0;
}

//#include <sys/time.h> 
//#include <sys/resource.h> 
//#include <unistd.h>

struct rusage {
    struct timeval ru_utime; /* user time used */
    struct timeval ru_stime; /* system time used */
    long   ru_maxrss;        /* maximum resident set size */
    long   ru_ixrss;         /* integral shared memory size */
    long   ru_idrss;         /* integral unshared data size */
    long   ru_isrss;         /* integral unshared stack size */
    long   ru_minflt;        /* page reclaims */
    long   ru_majflt;        /* page faults */
    long   ru_nswap;         /* swaps */
    long   ru_inblock;       /* block input operations */
    long   ru_oublock;       /* block output operations */
    long   ru_msgsnd;        /* messages sent */
    long   ru_msgrcv;        /* messages received */
    long   ru_nsignals;      /* signals received */
    long   ru_nvcsw;         /* voluntary context switches */
    long   ru_nivcsw;        /* involuntary context switches */
};
 

// sys/uio.h

struct iovec
{
  void*   iov_base;
  size_t  iov_len;
};

//#include "ff/platforms/platform_msvc_windows.h"
//#if (!defined(_FF_SYSTEM_HAVE_WIN_PTHREAD))
//#endif
#include<algorithm>
#elif defined(__GNUC__) && (defined(__linux__) || defined(__APPLE__))
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdlib.h>
inline static void posix_memalign_free(void* mem)
{
    free(mem);
}
//#define __TICKS2WAIT 1000

#else
#   error "unknown platform"
#endif

#endif /* FF_PLATFORM_HPP */


