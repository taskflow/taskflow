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
class CriticalRegion : public Semaphore {

  public:
    
    /**
    @brief constructs a critical region of a limited number of workers
    */
    explicit CriticalRegion(int max_workers = 1);
    
    /**
    @brief adds a task into the critical region
    */
    template <typename... Tasks>
    void add(Tasks...tasks);
};

inline CriticalRegion::CriticalRegion(int max_workers) : 
  Semaphore {max_workers} {
}

template <typename... Tasks>
void CriticalRegion::add(Tasks... tasks) {
  (tasks.acquire(*this), ...);
  (tasks.release(*this), ...);
}


}  // end of namespace tf. ---------------------------------------------------


