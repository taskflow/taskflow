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

/* **************************************************************************
Ultra-minimal and incomplete compatibility layer for pthreads on Windows platform. Errors are not managed at all. Designed to enable a basic porting of FastFlow on Windows platform, if you plan to extensively use pthread consider to install a full pthread port for windows. Some of the primitives are inspired to Fastflow queue remix by Dmitry Vyukov http://www.1024cores.net/

March 2011 - Ver 0: Basic functional port, tested on Win 7 x64 - Performance not yet extensively tested.

*/

#ifndef FF_MINPORT_WIN_H
#define FF_MINPORT_WIN_H


//#define WIN32_LEAN_AND_MEAN
#pragma once
//#include <WinSock2.h>
#define NOMINMAX
#include <Windows.h>
#include <WinBase.h>
#include <process.h>
#include <intrin.h>
#include <stdio.h>
//#include <stdint.h>
#include <errno.h>
#include <time.h>



#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define RESTRICT __declspec()


// 
#define PTHREAD_CANCEL_ENABLE        0x01  /* Cancel takes place at next cancellation point */
#define PTHREAD_CANCEL_DISABLE       0x00  /* Cancel postponed */
#define PTHREAD_CANCEL_DEFERRED      0x02  /* Cancel waits until cancellation point */

// MA - pthread very partial windows port - for fastflow usage only
typedef HANDLE pthread_t;
typedef struct _opaque_pthread_attr_t { long __sig; } pthread_attr_t;
typedef struct _opaque_pthread_mutexattr_t { long __sig; } pthread_mutexattr_t;
typedef struct _opaque_pthread_condattr_t {long __sig; } pthread_condattr_t;

typedef DWORD pthread_key_t;


// Mutex and cond vars

typedef CRITICAL_SECTION pthread_mutex_t;

INLINE int pthread_create(pthread_t RESTRICT * thread,
		const pthread_attr_t RESTRICT * attr, 
		void *(*start_routine)(void *), 
		void RESTRICT * arg) {
			//std::cerr << "Creating new thread\n";
			thread[0] = (HANDLE)_beginthreadex(0, 0, (unsigned(__stdcall*)(void*))start_routine, arg, 0, 0);
		return(0); 
}


INLINE int pthread_join(pthread_t thread, void **value_ptr) {
	LPDWORD exitcode = 0;
    WaitForSingleObject(thread, INFINITE);
	if (value_ptr)  {
		GetExitCodeThread(thread,exitcode);
		*value_ptr = exitcode;
	}
    CloseHandle(thread);
	return(0); // Errors are not (yet) managed.	 
}

INLINE void pthread_exit(void *value_ptr) {
	if (value_ptr)
		ExitThread(*((DWORD *) value_ptr));
	else 
		ExitThread(0);
}

INLINE int pthread_attr_init(pthread_attr_t *attr) {
	// do nothing currently
	return(0);
}

INLINE int pthread_attr_destroy(pthread_attr_t *attr) {
	// do nothing currently
	return(0);
}

INLINE int pthread_setcancelstate(int state, int *oldstate) {
	// do nothing currently
	return(0);
}

// This requires #define _WIN32_WINNT 0x0403 to be stated as C++ preprocessor option (-D_WIN32_WINNT=0x0403)
// MA: 26/04/14 this seems no longer true

INLINE int pthread_mutex_init(pthread_mutex_t  RESTRICT * mutex,
	const pthread_mutexattr_t RESTRICT  * attr) {
	if (attr) return(EINVAL);
	InitializeCriticalSectionAndSpinCount(mutex, 1500 /*spin count */);	
	return (0); // Errors partially managed
}

INLINE int pthread_mutex_lock(pthread_mutex_t *mutex) {
	EnterCriticalSection(mutex);
	return(0); // Errors not managed
 }

INLINE int pthread_mutex_unlock(pthread_mutex_t *mutex) {
	LeaveCriticalSection(mutex);
	return(0); // Errors not managed
 }

INLINE int pthread_mutex_destroy(pthread_mutex_t *mutex) {
	DeleteCriticalSection(mutex);
	return(0); // Errors not managed
 }
 

INLINE  pthread_t pthread_self(void) {
	return(GetCurrentThread());
}

INLINE int pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
	*key = TlsAlloc();
	return 0;
}

INLINE int pthread_key_delete(pthread_key_t key) {
	TlsFree(key);
	return 0;
}

INLINE  int pthread_setspecific(pthread_key_t key, const void *value) {
	TlsSetValue(key, (LPVOID) value);
	return (0);
}

INLINE void * pthread_getspecific(pthread_key_t key) {
	return(TlsGetValue(key));
}

//#if (_WIN32_WINNT >= _WIN32_WINNT_VISTA)
#ifndef _FF_WIN_XP
typedef CONDITION_VARIABLE pthread_cond_t;
//#include <Windows.h>
#include <WinBase.h>
INLINE int pthread_cond_init(pthread_cond_t  RESTRICT * cond,
    const pthread_condattr_t  RESTRICT * attr) {
	if (attr) return(EINVAL);
	InitializeConditionVariable(cond);	
	return (0);	 // Errors not managed
 }

INLINE int pthread_cond_signal(pthread_cond_t *cond) {
	WakeConditionVariable(cond);
	return(0); // Errors not managed
 }

INLINE int pthread_cond_broadcast(pthread_cond_t *cond) {
	WakeAllConditionVariable(cond);
	return(0);
 }

INLINE int pthread_cond_wait(pthread_cond_t  RESTRICT * cond,
					   pthread_mutex_t  RESTRICT * mutex) {
    SleepConditionVariableCS(cond, mutex, INFINITE);
	//WaitForSingleObject(cond[0], INFINITE); 
	return(0); // Errors not managed
 }

INLINE int pthread_cond_destroy(pthread_cond_t *cond) {
	// Do nothing. Did not find a Windows call .... 
	return (0); // Errors not managed
 }

// Barrier 

/* MA:

Starting from Windows 8 it can be used:

EnterSynchronizationBarrier
DeleteSynchronizationBarrier
Synchronization Barriers

Not supported in Win 7 - here a sketch of the interface
Not really used - the FF code enable an alternative barrier via #ifdef

typedef struct _opaque_pthread_barrier_t {
	pthread_mutex_t bar_lock;
	pthread_cond_t bar_cond;
	unsigned count;
} pthread_barrier_t;

typedef struct _opaque_pthread_barrierattr_t {
	// Not implemented
	char c;
} pthread_barrierattr_t;

INLINE int pthread_barrier_init(pthread_barrier_t RESTRICT * barrier, 
		const pthread_barrierattr_t RESTRICT * attr, unsigned count) {
	barrier->count = count;
	// errors currently not managed;
	return 0;
}

INLINE int pthread_barrier_destroy(pthread_barrier_t *barrier) {
	// errors currently not managed;
	return 0;
}

INLINE int pthread_barrier_wait(pthread_barrier_t *barrier) {
	// errors currently not managed;
	return 0;
}

*/

#else 
// Win XP hasn't Condition variables!
//
// Douglas C. Schmidt and Irfan Pyarali
// Strategies for Implementing POSIX Condition Variables on Win32
// http://www.cs.wustl.edu/~schmidt/win32-cv-1.html

typedef struct
{
  int waiters_count_;
  // Count of the number of waiters.

  CRITICAL_SECTION waiters_count_lock_;
  // Serialize access to <waiters_count_>.

  int release_count_;
  // Number of threads to release via a <pthread_cond_broadcast> or a
  // <pthread_cond_signal>. 
  
  int wait_generation_count_;
  // Keeps track of the current "generation" so that we don't allow
  // one thread to steal all the "releases" from the broadcast.

  HANDLE event_;
  // A manual-reset event that's used to block and release waiting
  // threads. 
} pthread_cond_t;

INLINE int pthread_cond_init(pthread_cond_t  RESTRICT * cv,
    const pthread_condattr_t  RESTRICT * attr) {
  cv->waiters_count_ = 0;
  cv->wait_generation_count_ = 0;
  cv->release_count_ = 0;

  // Create a manual-reset event.
  cv->event_ = CreateEvent (NULL,  // no security
                            TRUE,  // manual-reset
                            FALSE, // non-signaled initially
                            NULL); // unnamed

  pthread_mutex_init(&cv->waiters_count_lock_,NULL);
  return 0;
}

INLINE int pthread_cond_wait(pthread_cond_t  RESTRICT * cv,
					   pthread_mutex_t  RESTRICT * external_mutex) {
  // Avoid race conditions.
  EnterCriticalSection (&cv->waiters_count_lock_);

  // Increment count of waiters.
  cv->waiters_count_++;

  // Store current generation in our activation record.
  int my_generation = cv->wait_generation_count_;

  LeaveCriticalSection (&cv->waiters_count_lock_);
  LeaveCriticalSection (external_mutex);

  for (;;) {
    // Wait until the event is signaled.
    WaitForSingleObject (cv->event_, INFINITE);

    EnterCriticalSection (&cv->waiters_count_lock_);
    // Exit the loop when the <cv->event_> is signaled and
    // there are still waiting threads from this <wait_generation>
    // that haven't been released from this wait yet.
    int wait_done = cv->release_count_ > 0
                    && cv->wait_generation_count_ != my_generation;
    LeaveCriticalSection (&cv->waiters_count_lock_);

    if (wait_done)
      break;
  }

  EnterCriticalSection (external_mutex);
  EnterCriticalSection (&cv->waiters_count_lock_);
  cv->waiters_count_--;
  cv->release_count_--;
  int last_waiter = cv->release_count_ == 0;
  LeaveCriticalSection (&cv->waiters_count_lock_);

  if (last_waiter)
    // We're the last waiter to be notified, so reset the manual event.
    ResetEvent (cv->event_);
  return 0;
}

INLINE int pthread_cond_signal(pthread_cond_t *cv) {
  EnterCriticalSection (&cv->waiters_count_lock_);
  if (cv->waiters_count_ > cv->release_count_) {
    SetEvent (cv->event_); // Signal the manual-reset event.
    cv->release_count_++;
    cv->wait_generation_count_++;
  }
  LeaveCriticalSection (&cv->waiters_count_lock_);
  return 0;
}

INLINE int pthread_cond_broadcast(pthread_cond_t *cv) {
  EnterCriticalSection (&cv->waiters_count_lock_);
  if (cv->waiters_count_ > 0) {  
    SetEvent (cv->event_);
    // Release all the threads in this generation.
    cv->release_count_ = cv->waiters_count_;

    // Start a new generation.
    cv->wait_generation_count_++;
  }
  LeaveCriticalSection (&cv->waiters_count_lock_);
  return 0;
}

INLINE int pthread_cond_destroy(pthread_cond_t *cond) {
	// Do nothing. Did not find a Windows call .... 
	return (0); // Errors not managed
 }

#endif

#endif /* FF_MINPORT_WIN_H */
