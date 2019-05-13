//  - modified the thread monitor from TBB

#pragma once

#include "../predef/compiler.hpp"
#include "../predef/os.hpp"

#if TF_COMP_MSVC || TF_COMP_MSVC_EMULATED 
  #include <windows.h>
#elif TF_COMP_GNUC || TF_COMP_GNUC_EMULATED
  #include <semaphore.h> 
#elif TF_OS_MACOS
  #include <mach/semaphore.h>
  #include <mach/task.h>
  #include <mach/mach_init.h>
  #include <mach/error.h>
#else
#  error "Unsupported compiler"
#endif


#include <cassert>

namespace tf {

#if TF_COMP_MSVC || TF_COMP_MSVC_EMULATED 
struct BinarySemaphore {
  BinarySemaphore() { 
    sem = CreateSemaphore(NULL, 0, 1, NULL);
  }

  ~BinarySemaphore() { 
    CloseHandle(sem); 
  }

  void P() { 
    WaitForSingleObject(sem, INFINITE); 
  }

  void V() { 
    ReleaseSemaphore(sem, 1, NULL);
  }

private:  
  HANDLE sem;
  BinarySemaphore(const BinarySemaphore& other) = delete;
  BinarySemaphore& operator=(const BinarySemaphore& other) = delete;
};

#elif TF_COMP_GNUC || TF_COMP_GNUC_EMULATED 
struct BinarySemaphore {
  BinarySemaphore() {
    int ret = sem_init( &sem, 0, 0 );
    // Return 0 on success; -1 on error
    assert(ret == 0);
  }

  ~BinarySemaphore() {
    int ret = sem_destroy( &sem );
    // Return 0 on success; -1 on error
    assert(ret == 0);
  }

  void P() {
    while( sem_wait( &sem )!=0 );
  }

  void V() { 
    sem_post( &sem ); 
  }
 
private:  
  sem_t sem;
  BinarySemaphore(const BinarySemaphore& other) = delete;
  BinarySemaphore& operator=(const BinarySemaphore& other) = delete;
};

#elif TF_OS_MACOS
struct BinarySemaphore {

  BinarySemaphore() : sem(0) {
    kern_return_t ret = semaphore_create(mach_task_self(), &sem, SYNC_POLICY_FIFO, 0);
  }

  ~BinarySemaphore() {
    kern_return_t ret = semaphore_destroy(mach_task_self(), sem);
  }

  void P() {
    int ret;
    do {
      ret = semaphore_wait( sem );
    } while( ret==KERN_ABORTED );
  }

  void V() { 
    semaphore_signal( sem );  
  }

private:  
  semaphore_t sem;
  BinarySemaphore(const BinarySemaphore& other) = delete;
  BinarySemaphore& operator=(const BinarySemaphore& other) = delete;
};

#else
#  error "Unsupported compiler"
#endif


}  // end of namespace tf. ---------------------------------------------------


