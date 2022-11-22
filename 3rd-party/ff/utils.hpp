/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file utils.hpp
 * \ingroup aux_classes
 * \brief Utility functions
 *
 */

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
 *
 ****************************************************************************
 */

#ifndef FF_UTILS_HPP
#define FF_UTILS_HPP

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
//#include <unistd.h> // Not availbe on windows - to be managed
#include <iosfwd>
//#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include <ff/config.hpp>
#include <ff/platforms/platform.h>

//#else
//#include <pthread.h>
//#include <sys/time.h>
//#endif

#include <cstring>
#include <string>
#include <fstream>

#include <ff/cycle.h>
#include <ff/spin-lock.hpp>

namespace ff {


enum { START_TIME=0, STOP_TIME=1, GET_TIME=2 };

/* TODO: - nanosleep on Window
 *       - test on Apple
 */
#if defined(__linux__)
/*!!!----Mehdi-- required for DSRIMANAGER NODE----!!*/
static inline void waitCall(double milisec, double sec){
  if(milisec!=0.0 || sec!=0.0){
    struct timespec req;
    req.tv_sec = sec;
    req.tv_nsec = milisec * 1000000L;
    nanosleep(&req, (struct timespec *)NULL);
  }
}

static inline void waitSleep(ticks TICKS2WAIT){
    /*!!!----Mehdi--required to change busy wait with nanosleep ----!!*/ 
    //struct timespac req = {0};
    //req.tv_sec = static_cast<int>((static_cast<double>(TICKS2WAIT))/CLOCKS_PER_SEC);
    //req.tv_nsec =(((static_cast<double>(TICKS2WAIT))/CLOCKS_PER_SEC)-static_cast<int>((static_cast<double>(TICKS2WAIT))/CLOCKS_PER_SEC))*1.0e9;
    //req.tv_nsec =(((static_cast<double>(TICKS2WAIT))/CLOCKS_PER_SEC)-static_cast<int>((static_cast<double>(TICKS2WAIT))/CLOCKS_PER_SEC))*1.0e9;

    /* NOTE: The following implementation is not correct because we don't take into account
     *       the (current) CPU frequency. Anyway, this works well enough for internal FastFlow usage.
     */ 
    struct timespec req = {0, static_cast<long>(TICKS2WAIT)};
    nanosleep(&req, NULL);
}
#endif /* __linux__ */

/* NOTE:  nticks should be something less than 1000000 otherwise 
 *        better to use something else.
 */
static inline ticks ticks_wait(ticks nticks) {
#if defined(__linux__) && defined(FF_ESAVER)
    waitSleep(nticks);
    return 0;
#else
    ticks delta;
    ticks t0 = getticks();
    do { delta = (getticks()) - t0; } while (delta < nticks);
    return delta-nticks;
#endif
}

/* NOTE: Does not make sense to use 'us' grather than or equal to 1000000 */ 
static inline void ff_relax(unsigned long us) {
#if defined(__linux__)
    struct timespec req = {0, static_cast<long>(us*1000L)};
    nanosleep(&req, NULL);
#else
    usleep(us);
#endif
    PAUSE();
}

static inline void error(const char * str, ...) {
    const char err[]="ERROR: ";
    va_list argp;
    char * p=(char *)malloc(strlen(str)+strlen(err)+128);
    if (!p) {
        printf("FATAL ERROR: no enough memory!\n");
        abort();
    }
    strcpy(p,err);
    strcpy(p+strlen(err), str);
    va_start(argp, str);
    vfprintf(stderr, p, argp);
    va_end(argp);
    free(p);
}


/**
 *  Reads memory size data from /proc/<pid>/status 
 *
 */
static inline int memory_Stats(const std::string &status, size_t &vm, size_t &vmp) {
#if defined(__linux__)
    std::string line;
    std::ifstream f;
    f.open(status.c_str());
    if (!f.is_open()) return -1;
    vm = vmp = 0;
    while (!vm || !vmp)	{
        getline(f, line);
        if (line.compare(0,7,"VmPeak:") == 0) {
            /* get rid of " kB" */
            const std::string &s = line.substr(7);
            vmp = std::stol(s.substr(0, s.length()-3));
        } else 
            if (line.compare(0,7,"VmSize:") == 0) {
                /* get rid of " kB"*/
                const std::string &s = line.substr(7);
                vm = std::stol(s.substr(0, s.length()-3));
            }
    }
    f.close();
    return 0;
#else
    error("memory_Stats, not implemented for this platform\n");
    return -1;
#endif
}

static inline unsigned long getusec() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)(tv.tv_sec*1e6+tv.tv_usec);
}

static inline unsigned long getusec(const struct timeval &tv) {
    return  (unsigned long)(tv.tv_sec*1e6+tv.tv_usec);
}

static inline double diffmsec(const struct timeval & a, 
                              const struct timeval & b) {
    long sec  = (a.tv_sec  - b.tv_sec);
    long usec = (a.tv_usec - b.tv_usec);
    
    if(usec < 0) {
        --sec;
        usec += 1000000;
    }
    return ((double)(sec*1000)+ ((double)usec)/1000.0);
}


static inline bool time_compare(struct timeval & a, struct timeval & b) {
    double t1= a.tv_sec*1000 + (double)(a.tv_usec)/1000.0;
    double t2= b.tv_sec*1000 + (double)(b.tv_usec)/1000.0;        
    return (t1<t2);
}

static inline bool time_iszero(const struct timeval & a) {
    if ((a.tv_sec==0) && (a.tv_usec==0)) return true;
    return false;
}


static inline void time_setzero(struct timeval & a) {
    a.tv_sec=0;  
    a.tv_usec=0;
}

static inline bool isPowerOf2(unsigned x) {
	return (x != 0 && (x & (x-1)) == 0);
}

static inline unsigned long nextPowerOf2(unsigned long x) {
    assert(isPowerOf2(x)==false); // x is not a power of two!
    unsigned long p=1;
    while (x>p) p <<= 1;
    return p;
}

static inline void timedwait_timeout(struct timespec&tv) {
    clock_gettime(CLOCK_REALTIME, &tv);
    tv.tv_nsec+=FF_TIMEDWAIT_NS;
    if (tv.tv_nsec>=1e+9) {
        tv.tv_sec+=1;
        tv.tv_nsec-=1e+9;
    }
}

static inline unsigned int nextMultipleOfIf(unsigned int x, unsigned int m) {
    unsigned r = x % m;
    return (r ? (x-r+m):x); 
}


static inline double ffTime(int tag/*, bool lock=false*/) {
    static struct timeval tv_start = {0,0};
    static struct timeval tv_stop  = {0,0};
    // needed to protect gettimeofday
    // if multiple threads call ffTime
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
    static lock_t L;
#else
    static lock_t L = {0};
#endif

    double res=0.0;
    switch(tag) {
    case START_TIME:{
        spin_lock(L);
        gettimeofday(&tv_start,NULL);
        spin_unlock(L);
    } break;
    case STOP_TIME:{
        spin_lock(L);
        gettimeofday(&tv_stop,NULL);
        spin_unlock(L);
        res = diffmsec(tv_stop,tv_start);
    } break;
    case GET_TIME: {        
        res = diffmsec(tv_stop,tv_start);
    } break;
    default:
        res=0;
    }    
    return res;
}

} // namespace ff

#endif /* FF_UTILS_HPP */
