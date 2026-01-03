#pragma once

#include "executor.hpp"

namespace tf {

// ------------------------------------------------------------------------------------------------
// class: TaskGroup
// ------------------------------------------------------------------------------------------------

/**
@class TaskGroup

@brief class to create a task group from a task

A task group executes a group of asynchronous tasks.
It enables asynchronous task spawning, cooperative execution among worker threads,
and naturally supports recursive parallelism.
Due to cooperative execution, a task group can only be created by an executor worker; 
otherwise an exception will be thrown.
The code below demonstrates how to use task groups to implement recursive Fibonacci parallelism.
  
@code{.cpp}
tf::Executor executor;

size_t fibonacci(size_t N) {

  if (N < 2) return N;

  size_t res1, res2;

  // Create a task group from the current executor
  tf::TaskGroup tg = get_executor().task_group();

  // Submit asynchronous tasks to the group
  tg.silent_async([N, &res1](){ res1 = fibonacci(N-1); });
  res2 = fibonacci(N-2);  // compute one branch synchronously

  // Wait for all tasks in the group to complete
  tg.corun();

  return res1 + res2;
}

int main() {
  size_t N = 30, res;
  res = executor.async([](){ return fibonacci(30); }).get();
  std::cout << N << "-th Fibonacci number is " << res << '\n';
  return 0;
}
@endcode

Users must explicitly call tf::TaskGroup::corun() to ensure that all tasks have completed or 
been properly canceled before leaving the scope of a task group.
Failing to do so results in undefined behavior.

@note
To understand how %Taskflow schedules a task group, please refer to @ref TaskGroup.

*/
class TaskGroup {

  friend class Executor;
  
  public:
  
  // ----------------------------------------------------------------------------------------------
  // deleted members
  // ----------------------------------------------------------------------------------------------

  /**
  @brief disabled copy constructor
  */
  TaskGroup(const TaskGroup&) = delete;
  
  /**
  @brief disabled move constructor
  */
  TaskGroup(TaskGroup&&) = delete;

  /**
  @brief disabled copy assignment
  */
  TaskGroup& operator = (TaskGroup&&) = delete;

  /**
  @brief disabled move assignment
  */
  TaskGroup& operator = (const TaskGroup&) = delete;
  
  /**
  @brief obtains the executor that creates this task group

  The running executor of a task group is the executor that creates the task group.

  @code{.cpp}
  executor.silent_async([&](){
    tf::TaskGroup tg = executor.task_group();
    assert(&(tg.executor()) == &executor);
  });
  @endcode
  */
  Executor& executor();
  
  // ----------------------------------------------------------------------------------------------
  // async methods
  // ----------------------------------------------------------------------------------------------

  /**
  @brief runs the given callable asynchronously

  @tparam F callable type
  @param f callable object
    
  This method creates an asynchronous task that executes the given function with the specified arguments.
  Unlike tf::Executor::async, the task created here is parented to the task group,
  where applications can issue tf::TaskGroup::corun to explicitly wait for all 
  asynchronous tasks spawned from the task group to complete.
  For example:

  @code{.cpp}
  executor.silent_async([&](){
    std::atomic<int> counter(0);
    auto tg = executor.task_group();
    auto fu1 = tg.async([&](){ counter++; });
    auto fu2 = tg.async([&](){ counter++; });
    fu1.get();
    fu2.get();
    assert(counter == 2);
    // spawn 100 asynchronous tasks from the task group
    for(int i=0; i<100; i++) {
      tg.silent_async([&](){ counter++; });
    }
    // corun until the 100 asynchronous tasks have completed
    tg.corun();
    assert(counter == 102);
    // do something else afterwards ...
  });
  @endcode
  */
  template <typename F>
  auto async(F&& f);
  
  /**
  @brief runs the given callable asynchronously

  @tparam F callable type
  @tparam P task parameters type

  @param params task parameters
  @param f callable

  Similar to tf::TaskGroup::async, but takes a parameter of type tf::TaskParams to initialize
  the asynchronous task.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    auto future = tg.async("my task", [](){ return 10; });
    assert(future.get() == 10);
  });
  @endcode

  */
  template <typename P, typename F>
  auto async(P&& params, F&& f);
  
  // ----------------------------------------------------------------------------------------------
  // silent async methods
  // ----------------------------------------------------------------------------------------------

  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param f callable

  This function is more efficient than tf::TaskGroup::async and is recommended when the result
  of the asynchronous task does not need to be accessed via a std::future.

  @code{.cpp}
  executor.silent_async([&](){
    std::atomic<int> counter(0);
    auto tg = executor.task_group();
    for(int i=0; i<100; i++) {
      tg.silent_async([&](){ counter++; });
    }
    tg.corun();
    assert(counter == 100);
  });
  @endcode

  */
  template <typename F>
  void silent_async(F&& f);
  
  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param params task parameters
  @param f callable

  Similar to tf::TaskGroup::silent_async, but takes a parameter of type tf::TaskParams to initialize
  the created asynchronous task.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    tg.silent_async("my task", [](){});
    tg.corun();
  });
  @endcode
  */
  template <typename P, typename F>
  void silent_async(P&& params, F&& f);
  
  // ----------------------------------------------------------------------------------------------
  // dependent async methods
  // ----------------------------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam F callable type
  @tparam Tasks tasks of type tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  %Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    tf::AsyncTask A = tg.silent_dependent_async([](){ printf("A\n"); });
    tf::AsyncTask B = tg.silent_dependent_async([](){ printf("B\n"); });
    auto [C, fuC] = tg.dependent_async(
      [](){ 
        printf("C runs after A and B\n"); 
        return 1;
      }, 
      A, B
    );
    fuC.get();  // C finishes, which in turns means both A and B finish,
                // so we don't need tg.corun()
  });
  @endcode
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously
         when the given predecessors finish
  
  @tparam P task parameters type
  @tparam F callable type
  @tparam Tasks tasks of type tf::AsyncTask
  
  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  %Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    tf::AsyncTask A = tg.silent_dependent_async("A", [](){ printf("A\n"); });
    tf::AsyncTask B = tg.silent_dependent_async("B", [](){ printf("B\n"); });
    auto [C, fuC] = tg.dependent_async(
      "C",
      [](){ 
        printf("C runs after A and B\n"); 
        return 1;
      }, 
      A, B
    );
    fuC.get();  // C finishes, which in turns means both A and B finish,
                // so we don't need tg.corun()
  });
  @endcode
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  %Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  executor.silent_async([](){
    auto tg = executor.task_group();
    std::array<tf::AsyncTask, 2> array {
      tg.silent_dependent_async([](){ printf("A\n"); }),
      tg.silent_dependent_async([](){ printf("B\n"); })
    };
    auto [C, fuC] = tg.dependent_async(
      [](){ 
        printf("C runs after A and B\n"); 
        return 1;
      }, 
      array.begin(), array.end()
    );
    fuC.get();  // C finishes, which in turns means both A and B finish,
                // so we don't need tg.corun()
  }); 
  @endcode
  */
  template <typename F, typename I,
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam P task parameters type
  @tparam F callable type
  @tparam I iterator type 
  
  @param params task parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  %Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    std::array<tf::AsyncTask, 2> array {
      tg.silent_dependent_async("A", [](){ printf("A\n"); }),
      tg.silent_dependent_async("B", [](){ printf("B\n"); })
    };
    auto [C, fuC] = tg.dependent_async(
      "C",
      [](){ 
        printf("C runs after A and B\n"); 
        return 1;
      }, 
      array.begin(), array.end()
    );
    fuC.get();  // C finishes, which in turns means both A and B finish,
                // so we don't need tg.corun()
  });
  @endcode
  */
  template <typename P, typename F, typename I,
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, I first, I last);

  
  // ----------------------------------------------------------------------------------------------
  // silent dependent async methods
  // ----------------------------------------------------------------------------------------------

  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish

  @tparam F callable type
  @tparam Tasks tasks of type tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::TaskGroup::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    tf::AsyncTask A = tg.silent_dependent_async([](){ printf("A\n"); });
    tf::AsyncTask B = tg.silent_dependent_async([](){ printf("B\n"); });
    tg.silent_dependent_async([](){ printf("C runs after A and B\n"); }, A, B);
    tg.corun();  // corun until all dependent-async tasks finish
  });
  @endcode
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given predecessors finish
  
  @tparam F callable type
  @tparam Tasks tasks of type tf::AsyncTask

  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::TaskGroup::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    tf::AsyncTask A = tg.silent_dependent_async("A", [](){ printf("A\n"); });
    tf::AsyncTask B = tg.silent_dependent_async("B", [](){ printf("B\n"); });
    tg.silent_dependent_async(
      "C", [](){ printf("C runs after A and B\n"); }, A, B
    );
    tg.corun();  // corun until all dependent-async tasks finish
  }); 
  @endcode
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::TaskGroup::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    std::array<tf::AsyncTask, 2> array {
      tg.silent_dependent_async([](){ printf("A\n"); }),
      tg.silent_dependent_async([](){ printf("B\n"); })
    };
    tg.silent_dependent_async(
      [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
    );
    tg.corun();  // corun until all dependent-async tasks finish
  }); 
  @endcode
  */
  template <typename F, typename I, 
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of predecessors finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param params tasks parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)

  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::TaskGroup::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    std::array<tf::AsyncTask, 2> array {
      tg.silent_dependent_async("A", [](){ printf("A\n"); }),
      tg.silent_dependent_async("B", [](){ printf("B\n"); })
    };
    tg.silent_dependent_async(
      "C", [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
    );
    tg.corun();  // corun until all dependent-async tasks finish
  }); 
  @endcode
  */
  template <typename P, typename F, typename I, 
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, I first, I last);


  // ----------------------------------------------------------------------------------------------
  // cooperative execution methods
  // ----------------------------------------------------------------------------------------------
  
  /**
  @brief corun all tasks spawned by this task group with other workers

  Coruns all tasks spawned by this task group cooperatively with other workers in
  the same executor until all these tasks finish.
  Under cooperative execution, a worker is not preempted. Instead, it continues 
  participating in the work-stealing loop, executing available tasks alongside 
  other workers.  

  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    std::atomic<size_t> counter{0};
    // spawn 100 async tasks and wait
    for(int i=0; i<100; i++) {
      tg.silent_async([&](){ counter++; });
    }
    tg.corun();
    assert(counter == 100);
    
    // spawn another 100 async tasks and wait
    for(int i=0; i<100; i++) {
      tg.silent_async([&](){ counter++; });
    }
    tg.corun();
    assert(counter == 200);
  });
  @endcode

  Note that only the parent worker of this task group (the worker who creates it) 
  can call this corun.
  */
  void corun();
  
  /**
  @brief cancel all tasks in this task group
  
  Marks the task group as cancelled to stop any not-yet-started tasks in the group from running.
  Tasks that are already running will continue to completion, but no new tasks belonging to the 
  task group will be scheduled after cancellation.

  This example below demonstrates how tf::TaskGroup::cancel() prevents pending tasks in a task group from executing,
  while allowing already running tasks to complete cooperatively. 
  The first set of tasks deliberately occupies all but one worker thread, 
  ensuring that subsequently spawned tasks remain pending. 
  After invoking tf::TaskGroup::cancel(), these pending tasks are never scheduled, 
  even after the blocked workers are released. 
  A final call to tf::TaskGroup::corun() synchronizes with all tasks in the group, 
  guaranteeing safe completion and verifying that cancellation successfully suppresses task execution.
  
  @code{.cpp}
  const size_t W = 12;  // must be >1 for this example to work
  tf::Executor executor(W);

  executor.async([&executor, W](){

    auto tg = executor.task_group();

    // deliberately block the other W-1 workers
    std::atomic<size_t> latch(0);
    for(size_t i=0; i<W-1; ++i) {
      tg.async([&](){
        ++latch;
        while(latch != 0);
      });
    }
    
    // wait until the other W-1 workers are blocked
    while(latch != W-1);

    // spawn other tasks which should never run after cancellation
    for(size_t i=0; i<100; ++i) {
      tg.async([&](){ throw std::runtime_error("this should never run"); });
    }
    
    // cancel the task group and unblock the other W-1 workers
    assert(tg.is_cancelled() == false);
    tg.cancel();
    assert(tg.is_cancelled() == true);
    latch = 0;

    tg.corun();
  });
  @endcode

  Note that cancellation is cooperative: tasks should not assume immediate termination.
  Users must still call tf::TaskGroup::corun() to synchronize with all spawned tasks and
  ensure safe completion or cancellation. 
  Failing to do so results in undefined behavior.
  */
  void cancel();
  
  /**
  @brief queries if the task group has been cancelled

  @return `true` if the task group has been marked as cancelled or `false` otherwise

  This method returns `true` if the task group has been marked as cancelled
  via a call to `cancel()`, and `false` otherwise. 

  @code{.cpp}
  executor.async([&](){
    auto tg = executor.task_group();
    assert(tg.is_cancelled() == false);
    tg.cancel(true);
    assert(tg.is_cancelled() == false);
  });
  @endcode
  
  The cancellation state reflects whether the task group is currently in a cancelled state and
  does not imply that all tasks have completed or been synchronized.
  If a task group spawns any task, users must still call `corun()` to synchronize with all spawned tasks 
  and ensure safe completion or cancellation. 
  Failing to do so results in undefined behavior.
  */
  bool is_cancelled();
  
  /**
  @brief queries the number of tasks currently in this task group

  @return the number of tasks currently in this task group

  This method returns the number of tasks that belong to the task group at the
  time of the call. The returned value represents a snapshot and may become
  outdated immediately, as tasks can be concurrently spawned, started, completed,
  or canceled while this method is executing.
  As a result, the value returned by `size()` should be used for informational or diagnostic
  purposes only and must not be relied upon for synchronization or correctness.
  
  @code{.cpp}
  executor.silent_async([&](){
    auto tg = executor.task_group();
    assert(tg.size() == 0);
    for(size_t i=0; i<1000; ++i) {
      tg.silent_async([](){});
    }
    assert(tg.size() >= 0);
    tg.corun();
  });
  @endcode
  */
  size_t size() const;

  private:

  /**
  @private
  */
  explicit TaskGroup(Executor&, Worker&);
  
  /**
  @private
  */
  Executor& _executor;

  /**
  @private
  */
  Worker& _worker;
  
  /**
  @private
  */
  NodeBase _node_base;
};

// constructor
inline TaskGroup::TaskGroup(Executor& executor, Worker& worker) : 
  _executor  {executor}, 
  _worker    {worker}, 
  _node_base {NSTATE::IMPLICITLY_ANCHORED, ESTATE::NONE, nullptr, 0} {
}

// Function: executor
inline Executor& TaskGroup::executor() {
  return _executor;
}

// Function: corun
inline void TaskGroup::corun() {
  {
    ExplicitAnchorGuard anchor(&_node_base);
    _executor._corun_until(_worker, [this] () -> bool {
      return _node_base._join_counter.load(std::memory_order_acquire) == 0;
    });
  }
  _node_base._rethrow_exception();
}

// Function: cancel
inline void TaskGroup::cancel() {
  _node_base._estate.fetch_or(ESTATE::CANCELLED, std::memory_order_relaxed);
}

// Function: is_cancelled
inline bool TaskGroup::is_cancelled() { 
  return _node_base._estate.load(std::memory_order_relaxed) & ESTATE::CANCELLED;
}

// Function: size
inline size_t TaskGroup::size() const {
  return _node_base._join_counter.load(std::memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// TaskGroup::silent_async
// ------------------------------------------------------------------------------------------------

// Function: silent_async
template <typename F>
void TaskGroup::silent_async(F&& f) {
  silent_async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: silent_async
template <typename P, typename F>
void TaskGroup::silent_async(P&& params, F&& f) {
  _node_base._join_counter.fetch_add(1, std::memory_order_relaxed);
  _executor._silent_async(
    std::forward<P>(params), std::forward<F>(f), nullptr, &_node_base
  );
}

// ------------------------------------------------------------------------------------------------
// TaskGroup::async 
// ------------------------------------------------------------------------------------------------

// Function: async
template <typename F>
auto TaskGroup::async(F&& f) {
  return async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: async
template <typename P, typename F>
auto TaskGroup::async(P&& params, F&& f) {
  _node_base._join_counter.fetch_add(1, std::memory_order_relaxed);
  return _executor._async(
    std::forward<P>(params), std::forward<F>(f), nullptr, &_node_base
  );
}

// ------------------------------------------------------------------------------------------------
// silent dependent async
// ------------------------------------------------------------------------------------------------

// Function: silent_dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
tf::AsyncTask TaskGroup::silent_dependent_async(F&& func, Tasks&&... tasks) {
  return silent_dependent_async(
    DefaultTaskParams{}, std::forward<F>(func), std::forward<Tasks>(tasks)...
  );
}

// Function: silent_dependent_async
template <typename P, typename F, typename... Tasks,
  std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
tf::AsyncTask TaskGroup::silent_dependent_async(
  P&& params, F&& func, Tasks&&... tasks 
){
  std::array<AsyncTask, sizeof...(Tasks)> array = { std::forward<Tasks>(tasks)... };
  return silent_dependent_async(
    std::forward<P>(params), std::forward<F>(func), array.begin(), array.end()
  );
}

// Function: silent_dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
tf::AsyncTask TaskGroup::silent_dependent_async(F&& func, I first, I last) {
  return silent_dependent_async(DefaultTaskParams{}, std::forward<F>(func), first, last);
}

// Function: silent_dependent_async
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
tf::AsyncTask TaskGroup::silent_dependent_async(
  P&& params, F&& func, I first, I last
) {
  _node_base._join_counter.fetch_add(1, std::memory_order_relaxed);
  return _executor._silent_dependent_async(
    std::forward<P>(params), std::forward<F>(func), first, last, nullptr, &_node_base
  );
}

// ------------------------------------------------------------------------------------------------
// dependent async
// ------------------------------------------------------------------------------------------------

// Function: dependent_async
template <typename F, typename... Tasks,
  std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
auto TaskGroup::dependent_async(F&& func, Tasks&&... tasks) {
  return dependent_async(DefaultTaskParams{}, std::forward<F>(func), std::forward<Tasks>(tasks)...);
}

// Function: dependent_async
template <typename P, typename F, typename... Tasks,
  std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>*
>
auto TaskGroup::dependent_async(P&& params, F&& func, Tasks&&... tasks) {
  std::array<AsyncTask, sizeof...(Tasks)> array = { std::forward<Tasks>(tasks)... };
  return dependent_async(
    std::forward<P>(params), std::forward<F>(func), array.begin(), array.end()
  );
}

// Function: dependent_async
template <typename F, typename I,
  std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto TaskGroup::dependent_async(F&& func, I first, I last) {
  return dependent_async(DefaultTaskParams{}, std::forward<F>(func), first, last);
}

// Function: dependent_async
template <typename P, typename F, typename I,
  std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>*
>
auto TaskGroup::dependent_async(P&& params, F&& func, I first, I last) {
  _node_base._join_counter.fetch_add(1, std::memory_order_relaxed);
  return _executor._dependent_async(
    std::forward<P>(params), std::forward<F>(func), first, last, nullptr, &_node_base
  );
}

// ----------------------------------------------------------------------------
// Executor Forward Declaration
// ----------------------------------------------------------------------------

// Procedure: task_group
inline TaskGroup Executor::task_group() {
  Worker* w = this_worker();
  if(w == nullptr) {
    TF_THROW("task_group can only created by a worker of the executor");
  }
  return TaskGroup(*this, *w);
}


}  // end of namespace tf -----------------------------------------------------









