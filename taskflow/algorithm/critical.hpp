#pragma once

#include "../core/task.hpp"

/**
@file critical.hpp
@brief critical include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// CriticalSection
// ----------------------------------------------------------------------------

/**
@class CriticalSection

@brief class to create a critical region of limited workers to run tasks

tf::CriticalSection is a wrapper over tf::Semaphore and is specialized for
limiting the maximum concurrency over a set of tasks.
A critical section starts with an initial count representing that limit.
When a task is added to the critical section,
the task acquires and releases the semaphore internal to the critical section.
This design avoids explicit call of tf::Task::acquire and tf::Task::release.
The following example creates a critical section of one worker and adds
the five tasks to the critical section.

@code{.cpp}
tf::Executor executor(8);   // create an executor of 8 workers
tf::Taskflow taskflow;

// create a critical section of 1 worker
tf::CriticalSection critical_section(1);

tf::Task A = taskflow.emplace([](){ std::cout << "A" << std::endl; });
tf::Task B = taskflow.emplace([](){ std::cout << "B" << std::endl; });
tf::Task C = taskflow.emplace([](){ std::cout << "C" << std::endl; });
tf::Task D = taskflow.emplace([](){ std::cout << "D" << std::endl; });
tf::Task E = taskflow.emplace([](){ std::cout << "E" << std::endl; });

critical_section.add(A, B, C, D, E);

executor.run(taskflow).wait();
@endcode

*/
class CriticalSection : public Semaphore {

  public:

    /**
    @brief constructs a critical region of a limited number of workers
    */
    explicit CriticalSection(size_t max_workers = 1);

    /**
    @brief adds a task into the critical region
    */
    template <typename... Tasks>
    void add(Tasks...tasks);
};

inline CriticalSection::CriticalSection(size_t max_workers) :
  Semaphore {max_workers} {
}

template <typename... Tasks>
void CriticalSection::add(Tasks... tasks) {
  (tasks.acquire(*this), ...);
  (tasks.release(*this), ...);
}


}  // end of namespace tf. ---------------------------------------------------


