#pragma once

#include "../task.hpp"

/** 
@file critical.hpp
@brief critical include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// CriticalRegion
// ----------------------------------------------------------------------------

/**
@class CriticalRegion

@brief class to create a critical region of limited workers to run tasks
 */
class CriticalRegion {

  public:

    explicit CriticalRegion(int max_workers = 1);
    
    template <typename... Tasks>
    void add(Tasks...tasks);

  private:

    Semaphore _semaphore;
};

inline CriticalRegion::CriticalRegion(int max_workers) : 
  _semaphore {max_workers} {
}

template <typename... Tasks>
void CriticalRegion::add(Tasks... tasks) {
  (tasks.acquire(_semaphore), ...);
  (tasks.release(_semaphore), ...);
}


}  // end of namespace tf. ---------------------------------------------------


