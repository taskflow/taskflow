#pragma once

#include "executor.hpp"

namespace tf {

// ------------------------------------------------------------------------------------------------
// class: Runtime
// ------------------------------------------------------------------------------------------------

/**
@class Runtime

@brief class to include a runtime object in a task

A runtime object provides an interface for interacting with the scheduling system from within a task 
(i.e., the parent task of this runtime). 
It enables operations such as spawning asynchronous tasks, executing tasks cooperatively, 
and implementing recursive parallelism. 
The runtime guarantees an implicit join at the end of its scope, 
so all spawned tasks will finish before the parent runtime task continues to its successors.

@code{.cpp}
tf::Executor executor(num_threads);
tf::Taskflow taskflow;
std::atomic<size_t> counter(0);

tf::Task A = taskflow.emplace([&](tf::Runtime& rt){
  // spawn 1000 asynchronous tasks from this runtime task
  for(size_t i=0; i<1000; i++) {
    rt.silent_async([&](){ counter.fetch_add(1, std::memory_order_relaxed); });
  }
  // implicit synchronization at the end of the runtime scope
});
tf::Task B = taskflow.emplace([&](){
  REQUIRE(counter.load(std::memory_order_relaxed) == 1000);
});

A.precede(B);

executor.run(taskflow).wait();
@endcode

A runtime object is associated with the worker and the executor that runs its parent task.

@note
To understand how %Taskflow schedules a runtime task, please refer to @ref RuntimeTasking.

*/
class Runtime {

  friend class Executor;
  friend class FlowBuilder;
  friend class PreemptionGuard;
  friend class Algorithm;
  
  #define TF_RUNTIME_CHECK_CALLER(msg) \
  if(pt::this_worker != &_worker) {    \
    TF_THROW(msg);                     \
  }

  public:
  
  /**
  @brief obtains the running executor

  The running executor of a runtime task is the executor that runs
  the parent taskflow of that runtime task.

  @code{.cpp}
  tf::Executor executor;
  tf::Taskflow taskflow;
  taskflow.emplace([&](tf::Runtime& rt){
    assert(&(rt.executor()) == &executor);
  });
  executor.run(taskflow).wait();
  @endcode
  */
  Executor& executor();
  
  /**
  @brief acquire a reference to the underlying worker
  */
  inline Worker& worker();

  /**
  @brief schedules an active task immediately to the worker's queue

  @param task the given active task to schedule immediately

  This member function immediately schedules an active task to the
  task queue of the associated worker in the runtime task.
  An active task is a task in a running taskflow.
  The task may or may not be running, and scheduling that task
  will immediately put the task into the task queue of the worker
  that is running the runtime task.
  Consider the following example:

  @code{.cpp}
  tf::Task A, B, C, D;
  std::tie(A, B, C, D) = taskflow.emplace(
    [] () { return 0; },
    [&C] (tf::Runtime& rt) {  // C must be captured by reference
      std::cout << "B\n";
      rt.schedule(C);
    },
    [] () { std::cout << "C\n"; },
    [] () { std::cout << "D\n"; }
  );
  A.precede(B, C, D);
  executor.run(taskflow).wait();
  @endcode

  The executor will first run the condition task @c A which returns @c 0
  to inform the scheduler to go to the runtime task @c B.
  During the execution of @c B, it directly schedules task @c C without
  going through the normal taskflow graph scheduling process.
  At this moment, task @c C is active because its parent taskflow is running.
  When the taskflow finishes, we will see both @c B and @c C in the output.

  @attention
  This method can only be called by the parent worker of this runtime,
  or the behavior is undefined.
  */
  void schedule(Task task);
  
  /**
  @brief runs the given callable asynchronously

  @tparam F callable type
  @param f callable object
    
  This method creates an asynchronous task that executes the given function with the specified arguments.
  Unlike tf::Executor::async, the task created here is bound to the runtime object and 
  is implicitly synchronized at the end of the runtime's scope.
  Applications may also call tf::Runtime::corun explicitly to wait for all 
  asynchronous tasks spawned from the runtime to complete.
  For example:

  @code{.cpp}
  std::atomic<int> counter(0);
  taskflow.emplace([&](tf::Runtime& rt){
    auto fu1 = rt.async([&](){ counter++; });
    auto fu2 = rt.async([&](){ counter++; });
    fu1.get();
    fu2.get();
    assert(counter == 2);
    // spawn 100 asynchronous tasks from the worker of the runtime
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    // explicitly wait for the 100 asynchronous tasks to finish
    rt.corun();
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

  Similar to tf::Runtime::async, but takes a parameter of type tf::TaskParams to initialize
  the created asynchronous task.

  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    auto future = rt.async("my task", [](){});
    future.get();
  });
  @endcode

  */
  template <typename P, typename F>
  auto async(P&& params, F&& f);

  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param f callable

  This function is more efficient than tf::Runtime::async and is recommended when the result
  of the asynchronous task does not need to be accessed via a std::future.

  @code{.cpp}
  std::atomic<int> counter(0);
  taskflow.emplace([&](tf::Runtime& rt){
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun();
    assert(counter == 100);
  });
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  void silent_async(F&& f);
  
  /**
  @brief runs the given function asynchronously without returning any future object

  @tparam F callable type
  @param params task parameters
  @param f callable

  Similar to tf::Runtime::silent_async, but takes a parameter of type tf::TaskParams to initialize
  the created asynchronous task.

  @code{.cpp}
  taskflow.emplace([&](tf::Runtime& rt){
    rt.silent_async("my task", [](){});
  });
  @endcode
  */
  template <typename P, typename F>
  void silent_async(P&& params, F&& f);
  
  /**
  @brief corun all tasks spawned by this runtime with other workers

  Coruns all tasks spawned by this runtime cooperatively with other workers 
  until all these tasks finish.
    
  @code{.cpp}
  std::atomic<size_t> counter{0};
  taskflow.emplace([&](tf::Runtime& rt){
    // spawn 100 async tasks and wait
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun();
    assert(counter == 100);
    
    // spawn another 100 async tasks and wait
    for(int i=0; i<100; i++) {
      rt.silent_async([&](){ counter++; });
    }
    rt.corun();
    assert(counter == 200);
  });
  @endcode

  @attention
  This method can only be called by the parent worker of this runtime,
  or the behavior is undefined.
  */
  void corun();

  /**
  @brief equivalent to tf::Runtime::corun - just an alias for legacy purpose
  */
  void corun_all();

  /**
  @brief This method verifies if the task has been cancelled.
  */
  bool is_cancelled();

  private:

  /**
  @private
  */
  explicit Runtime(Executor&, Worker&, Node*);
  
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
  Node* _parent;
};

// constructor
inline Runtime::Runtime(Executor& executor, Worker& worker, Node* parent) :
  _executor {executor},
  _worker   {worker},
  _parent   {parent} {
}

// Function: executor
inline Executor& Runtime::executor() {
  return _executor;
}

// Function: worker
inline Worker& Runtime::worker() {
  return _worker;
}

// Procedure: schedule
inline void Runtime::schedule(Task task) {

  auto node = task._node;
  // need to keep the invariant: when scheduling a task, the task must have
  // zero dependency (join counter is 0)
  // or we can encounter bug when inserting a nested flow (e.g., module task)
  node->_join_counter.store(0, std::memory_order_relaxed);

  auto& j = node->_parent ? node->_parent->_join_counter :
                            node->_topology->_join_counter;
  j.fetch_add(1, std::memory_order_relaxed);
  _executor._schedule(_worker, node);
}

// Function: corun
inline void Runtime::corun() {
  {
    AnchorGuard anchor(_parent);
    _executor._corun_until(_worker, [this] () -> bool {
      return _parent->_join_counter.load(std::memory_order_acquire) == 1;
    });
  }
  _parent->_rethrow_exception();
}

// Function: corun_all
inline void Runtime::corun_all() {
  corun();
}

inline bool Runtime::is_cancelled() { 
  return _parent->_is_cancelled(); 
}

// ------------------------------------
// Runtime::silent_async series
// ------------------------------------

// Function: silent_async
template <typename F>
void Runtime::silent_async(F&& f) {
  silent_async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: silent_async
template <typename P, typename F>
void Runtime::silent_async(P&& params, F&& f) {
  _parent->_join_counter.fetch_add(1, std::memory_order_relaxed);
  _executor._silent_async(
    std::forward<P>(params), std::forward<F>(f), _parent->_topology, _parent
  );
}

// ------------------------------------
// Runtime::async series
// ------------------------------------

// Function: async
template <typename F>
auto Runtime::async(F&& f) {
  return async(DefaultTaskParams{}, std::forward<F>(f));
}

// Function: async
template <typename P, typename F>
auto Runtime::async(P&& params, F&& f) {
  _parent->_join_counter.fetch_add(1, std::memory_order_relaxed);
  return _executor._async(
    std::forward<P>(params), std::forward<F>(f), _parent->_topology, _parent
  );
}

// ----------------------------------------------------------------------------
// Executor Forward Declaration
// ----------------------------------------------------------------------------

// Procedure: _invoke_runtime_task
inline bool Executor::_invoke_runtime_task(Worker& worker, Node* node) {
  return _invoke_runtime_task_impl(
    worker, node, std::get_if<Node::Runtime>(&node->_handle)->work
  );
}

// Function: _invoke_runtime_task_impl
inline bool Executor::_invoke_runtime_task_impl(
  Worker& worker, Node* node, std::function<void(Runtime&)>& work
) {
  // first time
  if((node->_nstate & NSTATE::PREEMPTED) == 0) {

    Runtime rt(*this, worker, node);

    rt._parent->_nstate |= NSTATE::PREEMPTED;
    rt._parent->_join_counter.fetch_add(1, std::memory_order_release);

    _observer_prologue(worker, node);
    TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
      work(rt);
    });
    _observer_epilogue(worker, node);
    
    // I am the last one - no need to preempt this runtime
    if(rt._parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      rt._parent->_nstate &= ~NSTATE::PREEMPTED;
    }
    else {
      return true;
    }
    
    // here, we cannot check the state from node->_nstate due to data race
    // Ex: if preempted, another task may finish real quck and insert this parent task
    // again into the scheduling queue. When running this parent task, it will jump to
    // else branch below and modify tne nstate, thus incuring data race.
    //if(rt._preempted) {
    //  return true;
    //}
  }
  // second time - previously preempted
  else {
    node->_nstate &= ~NSTATE::PREEMPTED;
  }
  return false;
}

// Function: _invoke_runtime_task_impl
inline bool Executor::_invoke_runtime_task_impl(
  Worker& worker, Node* node, std::function<void(Runtime&, bool)>& work
) {
    
  Runtime rt(*this, worker, node);

  // first time
  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    
    rt._parent->_nstate |= NSTATE::PREEMPTED;
    rt._parent->_join_counter.fetch_add(1, std::memory_order_release);

    _observer_prologue(worker, node);
    TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
      work(rt, false);
    });
    _observer_epilogue(worker, node);
    
    // I am the last one - no need to preempt this runtime
    if(rt._parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      rt._parent->_nstate &= ~NSTATE::PREEMPTED;
    }
    else {
      return true;
    }
    
    // here, we cannot check the state from node->_nstate due to data race
    // Ex: if preempted, another task may finish real quck and insert this parent task
    // again into the scheduling queue. When running this parent task, it will jump to
    // else branch below and modify tne nstate, thus incuring data race.
    //if(rt._preempted) {
    //  return true;
    //}
  }
  // second time - previously preempted
  else {
    node->_nstate &= ~NSTATE::PREEMPTED;
  }

  // clean up outstanding work
  work(rt, true);

  return false;
}

// ------------------------------------------------------------------------------------------------
// class: NonpreemptiveRuntime
// ------------------------------------------------------------------------------------------------

/**
@private

@brief currently for internal use only
*/
class NonpreemptiveRuntime {

  friend class Executor;

  public:
  
  /**
  @private
  */
  void schedule(Task task);
  
  private:

  /**
  @private
  */
  explicit NonpreemptiveRuntime(Executor& executor, Worker& worker, Node* parent) :
    _executor {executor}, _worker {worker}, _parent{parent} {
  }
  
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
  Node* _parent;
  
};

// Procedure: schedule
inline void NonpreemptiveRuntime::schedule(Task task) {

  auto node = task._node;
  // need to keep the invariant: when scheduling a task, the task must have
  // zero dependency (join counter is 0)
  // or we can encounter bug when inserting a nested flow (e.g., module task)
  node->_join_counter.store(0, std::memory_order_relaxed);

  auto& j = node->_parent ? node->_parent->_join_counter :
                            node->_topology->_join_counter;
  j.fetch_add(1, std::memory_order_relaxed);
  _executor._schedule(_worker, node);
}

// ----------------------------------------------------------------------------
// Executor Forward Declaration
// ----------------------------------------------------------------------------

// Procedure: _invoke_nonpreemptive_runtime_task
inline void Executor::_invoke_nonpreemptive_runtime_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  tf::NonpreemptiveRuntime nprt(*this, worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    std::get_if<Node::NonpreemptiveRuntime>(&node->_handle)->work(nprt);
  });
  _observer_epilogue(worker, node);
}



}  // end of namespace tf -----------------------------------------------------









