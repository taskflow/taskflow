#pragma once

#include "observer.hpp"
#include "taskflow.hpp"

/**
@file executor.hpp
@brief executor include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------

/** @class Executor

@brief class to create an executor for running a taskflow graph

An executor manages a set of worker threads to run one or multiple taskflows
using an efficient work-stealing scheduling algorithm.

@code{.cpp}
// Declare an executor and a taskflow
tf::Executor executor;
tf::Taskflow taskflow;

// Add three tasks into the taskflow
tf::Task A = taskflow.emplace([] () { std::cout << "This is TaskA\n"; });
tf::Task B = taskflow.emplace([] () { std::cout << "This is TaskB\n"; });
tf::Task C = taskflow.emplace([] () { std::cout << "This is TaskC\n"; });

// Build precedence between tasks
A.precede(B, C);

tf::Future<void> fu = executor.run(taskflow);
fu.wait();                // block until the execution completes

executor.run(taskflow, [](){ std::cout << "end of 1 run"; }).wait();
executor.run_n(taskflow, 4);
executor.wait_for_all();  // block until all associated executions finish
executor.run_n(taskflow, 4, [](){ std::cout << "end of 4 runs"; }).wait();
executor.run_until(taskflow, [cnt=0] () mutable { return ++cnt == 10; });
@endcode

All the @c run methods are @em thread-safe. You can submit multiple
taskflows at the same time to an executor from different threads.
*/
class Executor {

  friend class FlowBuilder;
  friend class Subflow;
  friend class Runtime;

  public:

    /**
    @brief constructs the executor with @c N worker threads


    @param N number of workers (default std::thread::hardware_concurrency)
    @param wix worker interface class to alter worker (thread) behaviors
    
    The constructor spawns @c N worker threads to run tasks in a
    work-stealing loop. The number of workers must be greater than zero
    or an exception will be thrown.
    By default, the number of worker threads is equal to the maximum
    hardware concurrency returned by std::thread::hardware_concurrency.

    Users can alter the worker behavior, such as changing thread affinity,
    via deriving an instance from tf::WorkerInterface.
    */
    explicit Executor(
      size_t N = std::thread::hardware_concurrency(),
      std::shared_ptr<WorkerInterface> wix = nullptr 
    );

    /**
    @brief destructs the executor

    The destructor calls Executor::wait_for_all to wait for all submitted
    taskflows to complete and then notifies all worker threads to stop
    and join these threads.
    */
    ~Executor();

    /**
    @brief runs a taskflow once

    @param taskflow a tf::Taskflow object

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow once and returns a tf::Future
    object that eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run(taskflow);
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    tf::Future<void> run(Taskflow& taskflow);

    /**
    @brief runs a moved taskflow once

    @param taskflow a moved tf::Taskflow object

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow once and returns a tf::Future
    object that eventually holds the result of the execution.
    The executor will take care of the lifetime of the moved taskflow.

    @code{.cpp}
    tf::Future<void> future = executor.run(std::move(taskflow));
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    tf::Future<void> run(Taskflow&& taskflow);

    /**
    @brief runs a taskflow once and invoke a callback upon completion

    @param taskflow a tf::Taskflow object
    @param callable a callable object to be invoked after this run

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow once and invokes the given
    callable when the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run(taskflow, [](){ std::cout << "done"; });
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    template<typename C>
    tf::Future<void> run(Taskflow& taskflow, C&& callable);

    /**
    @brief runs a moved taskflow once and invoke a callback upon completion

    @param taskflow a moved tf::Taskflow object
    @param callable a callable object to be invoked after this run

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow once and invokes the given
    callable when the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.
    The executor will take care of the lifetime of the moved taskflow.

    @code{.cpp}
    tf::Future<void> future = executor.run(
      std::move(taskflow), [](){ std::cout << "done"; }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    template<typename C>
    tf::Future<void> run(Taskflow&& taskflow, C&& callable);

    /**
    @brief runs a taskflow for @c N times

    @param taskflow a tf::Taskflow object
    @param N number of runs

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow @c N times and returns a tf::Future
    object that eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run_n(taskflow, 2);  // run taskflow 2 times
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    tf::Future<void> run_n(Taskflow& taskflow, size_t N);

    /**
    @brief runs a moved taskflow for @c N times

    @param taskflow a moved tf::Taskflow object
    @param N number of runs

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow @c N times and returns a tf::Future
    object that eventually holds the result of the execution.
    The executor will take care of the lifetime of the moved taskflow.

    @code{.cpp}
    tf::Future<void> future = executor.run_n(
      std::move(taskflow), 2    // run the moved taskflow 2 times
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    tf::Future<void> run_n(Taskflow&& taskflow, size_t N);

    /**
    @brief runs a taskflow for @c N times and then invokes a callback

    @param taskflow a tf::Taskflow
    @param N number of runs
    @param callable a callable object to be invoked after this run

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow @c N times and invokes the given
    callable when the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run(
      taskflow, 2, [](){ std::cout << "done"; }  // runs taskflow 2 times and invoke
                                                 // the lambda to print "done"
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    template<typename C>
    tf::Future<void> run_n(Taskflow& taskflow, size_t N, C&& callable);

    /**
    @brief runs a moved taskflow for @c N times and then invokes a callback

    @param taskflow a moved tf::Taskflow
    @param N number of runs
    @param callable a callable object to be invoked after this run

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow @c N times and invokes the given
    callable when the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run_n(
      // run the moved taskflow 2 times and invoke the lambda to print "done"
      std::move(taskflow), 2, [](){ std::cout << "done"; }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    template<typename C>
    tf::Future<void> run_n(Taskflow&& taskflow, size_t N, C&& callable);

    /**
    @brief runs a taskflow multiple times until the predicate becomes true

    @param taskflow a tf::Taskflow
    @param pred a boolean predicate to return @c true for stop

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow multiple times until
    the predicate returns @c true.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run_until(
      taskflow, [](){ return rand()%10 == 0 }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    template<typename P>
    tf::Future<void> run_until(Taskflow& taskflow, P&& pred);

    /**
    @brief runs a moved taskflow and keeps running it
           until the predicate becomes true

    @param taskflow a moved tf::Taskflow object
    @param pred a boolean predicate to return @c true for stop

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow multiple times until
    the predicate returns @c true.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.
    The executor will take care of the lifetime of the moved taskflow.

    @code{.cpp}
    tf::Future<void> future = executor.run_until(
      std::move(taskflow), [](){ return rand()%10 == 0 }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    template<typename P>
    tf::Future<void> run_until(Taskflow&& taskflow, P&& pred);

    /**
    @brief runs a taskflow multiple times until the predicate becomes true and
           then invokes the callback

    @param taskflow a tf::Taskflow
    @param pred a boolean predicate to return @c true for stop
    @param callable a callable object to be invoked after this run completes

    @return a tf::Future that holds the result of the execution

    This member function executes the given taskflow multiple times until
    the predicate returns @c true and then invokes the given callable when
    the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.

    @code{.cpp}
    tf::Future<void> future = executor.run_until(
      taskflow, [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.

    @attention
    The executor does not own the given taskflow. It is your responsibility to
    ensure the taskflow remains alive during its execution.
    */
    template<typename P, typename C>
    tf::Future<void> run_until(Taskflow& taskflow, P&& pred, C&& callable);

    /**
    @brief runs a moved taskflow and keeps running
           it until the predicate becomes true and then invokes the callback

    @param taskflow a moved tf::Taskflow
    @param pred a boolean predicate to return @c true for stop
    @param callable a callable object to be invoked after this run completes

    @return a tf::Future that holds the result of the execution

    This member function executes a moved taskflow multiple times until
    the predicate returns @c true and then invokes the given callable when
    the execution completes.
    This member function returns a tf::Future object that
    eventually holds the result of the execution.
    The executor will take care of the lifetime of the moved taskflow.

    @code{.cpp}
    tf::Future<void> future = executor.run_until(
      std::move(taskflow),
      [](){ return rand()%10 == 0 }, [](){ std::cout << "done"; }
    );
    // do something else
    future.wait();
    @endcode

    This member function is thread-safe.
    */
    template<typename P, typename C>
    tf::Future<void> run_until(Taskflow&& taskflow, P&& pred, C&& callable);
    
    /**
    @brief runs a target graph and waits until it completes using 
           an internal worker of this executor
    
    @tparam T target type which has `tf::Graph& T::graph()` defined
    @param target the target task graph object

    The method runs a target graph which has `tf::Graph& T::graph()` defined 
    and waits until the execution completes.
    Unlike the typical flow of calling `tf::Executor::run` series 
    plus waiting on the result, this method must be called by an internal
    worker of this executor. The caller worker will participate in
    the work-stealing loop of the scheduler, therby avoiding potential
    deadlock caused by blocked waiting.
    
    @code{.cpp}
    tf::Executor executor(2);
    tf::Taskflow taskflow;
    std::array<tf::Taskflow, 1000> others;
    
    std::atomic<size_t> counter{0};
    
    for(size_t n=0; n<1000; n++) {
      for(size_t i=0; i<1000; i++) {
        others[n].emplace([&](){ counter++; });
      }
      taskflow.emplace([&executor, &tf=others[n]](){
        executor.run_and_wait(tf);
        //executor.run(tf).wait();  <- blocking the worker without doing anything
        //                             will introduce deadlock
      });
    }
    executor.run(taskflow).wait();
    @endcode 

    The method is thread-safe as long as the target is not concurrently
    ran by two or more threads.

    @attention
    You must call tf::Executor::run_and_wait from a worker of the calling executor
    or an exception will be thrown.
    */
    template <typename T>
    void run_and_wait(T& target);

    /**
    @brief keeps running the work-stealing loop until the predicate becomes true
    
    @tparam P predicate type
    @param predicate a boolean predicate to indicate when to stop the loop

    The method keeps the caller worker in the work-stealing loop such that it
    does not block (e.g., causing deadlock with other blocking workers) 
    until the stop predicate becomes true.

    @code{.cpp}
    taskflow.emplace([&](){
      std::future<void> fu = std::async([](){ std::sleep(100s); });
      executor.loop_until([](){
        return fu.wait_for(std::chrono::seconds(0)) == future_status::ready;
      });
    });
    @endcode

    @attention
    You must call tf::Executor::loop_until from a worker of the calling executor
    or an exception will be thrown.
    */
    template <typename P>
    void loop_until(P&& predicate);

    /**
    @brief waits for all tasks to complete

    This member function waits until all submitted tasks
    (e.g., taskflows, asynchronous tasks) to finish.

    @code{.cpp}
    executor.run(taskflow1);
    executor.run_n(taskflow2, 10);
    executor.run_n(taskflow3, 100);
    executor.wait_for_all();  // wait until the above submitted taskflows finish
    @endcode
    */
    void wait_for_all();

    /**
    @brief queries the number of worker threads

    Each worker represents one unique thread spawned by an executor
    upon its construction time.

    @code{.cpp}
    tf::Executor executor(4);
    std::cout << executor.num_workers();    // 4
    @endcode
    */
    size_t num_workers() const noexcept;

    /**
    @brief queries the number of running topologies at the time of this call

    When a taskflow is submitted to an executor, a topology is created to store
    runtime metadata of the running taskflow.
    When the execution of the submitted taskflow finishes,
    its corresponding topology will be removed from the executor.

    @code{.cpp}
    executor.run(taskflow);
    std::cout << executor.num_topologies();  // 0 or 1 (taskflow still running)
    @endcode
    */
    size_t num_topologies() const;

    /**
    @brief queries the number of running taskflows with moved ownership

    @code{.cpp}
    executor.run(std::move(taskflow));
    std::cout << executor.num_taskflows();  // 0 or 1 (taskflow still running)
    @endcode
    */
    size_t num_taskflows() const;
  
    /**
    @brief queries the id of the caller thread in this executor

    Each worker has an unique id in the range of @c 0 to @c N-1 associated with
    its parent executor.
    If the caller thread does not belong to the executor, @c -1 is returned.

    @code{.cpp}
    tf::Executor executor(4);   // 4 workers in the executor
    executor.this_worker_id();  // -1 (main thread is not a worker)

    taskflow.emplace([&](){
      std::cout << executor.this_worker_id();  // 0, 1, 2, or 3
    });
    executor.run(taskflow);
    @endcode
    */
    int this_worker_id() const;
  
    /**
    @brief runs a given function asynchronously

    @tparam F callable type
    @tparam ArgsT parameter types

    @param f callable object to call
    @param args parameters to pass to the callable

    @return a tf::Future that will holds the result of the execution

    The method creates an asynchronous task to launch the given
    function on the given arguments.
    Unlike std::async, the return here is a @em tf::Future that holds
    an optional object to the result.
    If the asynchronous task is cancelled before it runs, the return is
    a @c std::nullopt, or the value returned by the callable.

    @code{.cpp}
    tf::Future<std::optional<int>> future = executor.async([](){
      std::cout << "create an asynchronous task and returns 1\n";
      return 1;
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    auto async(F&& f, ArgsT&&... args);

    /**
    @brief runs a given function asynchronously and gives a name to this task

    @tparam F callable type
    @tparam ArgsT parameter types

    @param name name of the asynchronous task
    @param f callable object to call
    @param args parameters to pass to the callable

    @return a tf::Future that will holds the result of the execution

    The method creates a named asynchronous task to launch the given
    function on the given arguments.
    Naming an asynchronous task is primarily used for profiling and visualizing
    the task execution timeline.
    Unlike std::async, the return here is a tf::Future that holds
    an optional object to the result.
    If the asynchronous task is cancelled before it runs, the return is
    a @c std::nullopt, or the value returned by the callable.

    @code{.cpp}
    tf::Future<std::optional<int>> future = executor.named_async("name", [](){
      std::cout << "create an asynchronous task with a name and returns 1\n";
      return 1;
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    auto named_async(const std::string& name, F&& f, ArgsT&&... args);

    /**
    @brief similar to tf::Executor::async but does not return a future object

    This member function is more efficient than tf::Executor::async
    and is encouraged to use when there is no data returned.

    @code{.cpp}
    executor.silent_async([](){
      std::cout << "create an asynchronous task with no return\n";
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    void silent_async(F&& f, ArgsT&&... args);

    /**
    @brief similar to tf::Executor::named_async but does not return a future object

    This member function is more efficient than tf::Executor::named_async
    and is encouraged to use when there is no data returned.

    @code{.cpp}
    executor.named_silent_async("name", [](){
      std::cout << "create an asynchronous task with a name and no return\n";
    });
    @endcode

    This member function is thread-safe.
    */
    template <typename F, typename... ArgsT>
    void named_silent_async(const std::string& name, F&& f, ArgsT&&... args);

    /**
    @brief constructs an observer to inspect the activities of worker threads

    @tparam Observer observer type derived from tf::ObserverInterface
    @tparam ArgsT argument parameter pack

    @param args arguments to forward to the constructor of the observer

    @return a shared pointer to the created observer

    Each executor manages a list of observers with shared ownership with callers.
    For each of these observers, the two member functions,
    tf::ObserverInterface::on_entry and tf::ObserverInterface::on_exit
    will be called before and after the execution of a task.

    This member function is not thread-safe.
    */
    template <typename Observer, typename... ArgsT>
    std::shared_ptr<Observer> make_observer(ArgsT&&... args);

    /**
    @brief removes an observer from the executor

    This member function is not thread-safe.
    */
    template <typename Observer>
    void remove_observer(std::shared_ptr<Observer> observer);

    /**
    @brief queries the number of observers
    */
    size_t num_observers() const noexcept;

  private:
    
    const size_t _MAX_STEALS;

    std::condition_variable _topology_cv;
    std::mutex _taskflow_mutex;
    std::mutex _topology_mutex;
    std::mutex _wsq_mutex;

    size_t _num_topologies {0};
    
    std::unordered_map<std::thread::id, size_t> _wids;
    std::vector<std::thread> _threads;
    std::vector<Worker> _workers;
    std::list<Taskflow> _taskflows;

    Notifier _notifier;

    TaskQueue<Node*> _wsq;

    std::atomic<bool> _done {0};

    std::shared_ptr<WorkerInterface> _worker_interface;
    std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

    Worker* _this_worker();

    bool _wait_for_task(Worker&, Node*&);

    void _observer_prologue(Worker&, Node*);
    void _observer_epilogue(Worker&, Node*);
    void _spawn(size_t);
    void _exploit_task(Worker&, Node*&);
    void _explore_task(Worker&, Node*&);
    void _schedule(Worker&, Node*);
    void _schedule(Node*);
    void _schedule(Worker&, const SmallVector<Node*>&);
    void _schedule(const SmallVector<Node*>&);
    void _set_up_topology(Worker*, Topology*);
    void _tear_down_topology(Worker&, Topology*);
    void _tear_down_async(Node*);
    void _tear_down_invoke(Worker&, Node*);
    void _cancel_invoke(Worker&, Node*);
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
    void _invoke(Worker&, Node*);
    void _invoke_static_task(Worker&, Node*);
    void _invoke_dynamic_task(Worker&, Node*);
    void _consume_graph(Worker&, Node*, Graph&);
    void _detach_dynamic_task(Worker&, Node*, Graph&);
    void _invoke_condition_task(Worker&, Node*, SmallVector<int>&);
    void _invoke_multi_condition_task(Worker&, Node*, SmallVector<int>&);
    void _invoke_module_task(Worker&, Node*);
    void _invoke_async_task(Worker&, Node*);
    void _invoke_silent_async_task(Worker&, Node*);
    void _invoke_cudaflow_task(Worker&, Node*);
    void _invoke_syclflow_task(Worker&, Node*);
    void _invoke_runtime_task(Worker&, Node*);
    
    template <typename P>
    void _loop_until(Worker&, P&&);

    template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr>
    void _invoke_cudaflow_task_entry(Node*, C&&);

    template <typename C, typename Q,
      std::enable_if_t<is_syclflow_task_v<C>, void>* = nullptr
    >
    void _invoke_syclflow_task_entry(Node*, C&&, Q&);
};

// Constructor
inline Executor::Executor(size_t N, std::shared_ptr<WorkerInterface> wix) :
  _MAX_STEALS {((N+1) << 1)},
  _threads    {N},
  _workers    {N},
  _notifier   {N},
  _worker_interface {std::move(wix)} {

  if(N == 0) {
    TF_THROW("no cpu workers to execute taskflows");
  }

  _spawn(N);

  // instantite the default observer if requested
  if(has_env(TF_ENABLE_PROFILER)) {
    TFProfManager::get()._manage(make_observer<TFProfObserver>());
  }
}

// Destructor
inline Executor::~Executor() {

  // wait for all topologies to complete
  wait_for_all();

  // shut down the scheduler
  _done = true;

  _notifier.notify(true);

  for(auto& t : _threads){
    t.join();
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const noexcept {
  return _workers.size();
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
  return _num_topologies;
}

// Function: num_taskflows
inline size_t Executor::num_taskflows() const {
  return _taskflows.size();
}

// Function: _this_worker
inline Worker* Executor::_this_worker() {
  auto itr = _wids.find(std::this_thread::get_id());
  return itr == _wids.end() ? nullptr : &_workers[itr->second];
}

// Function: named_async
template <typename F, typename... ArgsT>
auto Executor::named_async(const std::string& name, F&& f, ArgsT&&... args) {

  _increment_topology();

  using T = std::invoke_result_t<F, ArgsT...>;
  using R = std::conditional_t<std::is_same_v<T, void>, void, std::optional<T>>;

  std::promise<R> p;

  auto tpg = std::make_shared<AsyncTopology>();

  Future<R> fu(p.get_future(), tpg);

  auto node = node_pool.animate(
    std::in_place_type_t<Node::Async>{},
    [p=make_moc(std::move(p)), f=std::forward<F>(f), args...]
    (bool cancel) mutable {
      if constexpr(std::is_same_v<R, void>) {
        if(!cancel) {
          f(args...);
        }
        p.object.set_value();
      }
      else {
        p.object.set_value(cancel ? std::nullopt : std::make_optional(f(args...)));
      }
    },
    std::move(tpg)
  );

  node->_name = name;

  if(auto w = _this_worker(); w) {
    _schedule(*w, node);
  }
  else{
    _schedule(node);
  }

  return fu;
}

// Function: async
template <typename F, typename... ArgsT>
auto Executor::async(F&& f, ArgsT&&... args) {
  return named_async("", std::forward<F>(f), std::forward<ArgsT>(args)...);
}

// Function: named_silent_async
template <typename F, typename... ArgsT>
void Executor::named_silent_async(
  const std::string& name, F&& f, ArgsT&&... args
) {

  _increment_topology();

  Node* node = node_pool.animate(
    std::in_place_type_t<Node::SilentAsync>{},
    [f=std::forward<F>(f), args...] () mutable {
      f(args...);
    }
  );

  node->_name = name;

  if(auto w = _this_worker(); w) {
    _schedule(*w, node);
  }
  else {
    _schedule(node);
  }
}

// Function: silent_async
template <typename F, typename... ArgsT>
void Executor::silent_async(F&& f, ArgsT&&... args) {
  named_silent_async("", std::forward<F>(f), std::forward<ArgsT>(args)...);
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto i = _wids.find(std::this_thread::get_id());
  return i == _wids.end() ? -1 : static_cast<int>(_workers[i->second]._id);
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N) {

  std::mutex mutex;
  std::condition_variable cond;
  size_t n=0;

  for(size_t id=0; id<N; ++id) {

    _workers[id]._id = id;
    _workers[id]._vtm = id;
    _workers[id]._executor = this;
    _workers[id]._waiter = &_notifier._waiters[id];

    _threads[id] = std::thread([this] (
      Worker& w, std::mutex& mutex, std::condition_variable& cond, size_t& n
    ) -> void {
      
      // assign the thread
      w._thread = &_threads[w._id];

      // enables the mapping
      {
        std::scoped_lock lock(mutex);
        _wids[std::this_thread::get_id()] = w._id;
        if(n++; n == num_workers()) {
          cond.notify_one();
        }
      }

      Node* t = nullptr;
      
      // before entering the scheduler (work-stealing loop), 
      // call the user-specified prologue function
      if(_worker_interface) {
        _worker_interface->scheduler_prologue(w);
      }
      
      // must use 1 as condition instead of !done because
      // the previous worker may stop while the following workers
      // are still preparing for entering the scheduling loop
      std::exception_ptr ptr{nullptr};
      try {
        while(1) {

          // execute the tasks.
          _exploit_task(w, t);

          // wait for tasks
          if(_wait_for_task(w, t) == false) {
            break;
          }
        }
      } 
      catch(...) {
        ptr = std::current_exception();
      }
      
      // call the user-specified epilogue function
      if(_worker_interface) {
        _worker_interface->scheduler_epilogue(w, ptr);
      }

    }, std::ref(_workers[id]), std::ref(mutex), std::ref(cond), std::ref(n));
    
    // POSIX-like system can use the following to affine threads to cores 
    //cpu_set_t cpuset;
    //CPU_ZERO(&cpuset);
    //CPU_SET(id, &cpuset);
    //pthread_setaffinity_np(
    //  _threads[id].native_handle(), sizeof(cpu_set_t), &cpuset
    //);
  }

  std::unique_lock<std::mutex> lock(mutex);
  cond.wait(lock, [&](){ return n==N; });
}

// Function: _loop_until
template <typename P>
inline void Executor::_loop_until(Worker& w, P&& stop_predicate) {

  std::uniform_int_distribution<size_t> rdvtm(0, _workers.size()-1);

  exploit:

  while(!stop_predicate()) {

    //exploit:

    if(auto t = w._wsq.pop(); t) {
      _invoke(w, t);
    }
    else {
      size_t num_steals = 0;

      explore:

      t = (w._id == w._vtm) ? _wsq.steal() : _workers[w._vtm]._wsq.steal();

      if(t) {
        _invoke(w, t);
        goto exploit;
      }
      else if(!stop_predicate()) {
        if(num_steals++ > _MAX_STEALS) {
          std::this_thread::yield();
        }
        w._vtm = rdvtm(w._rdgen);
        goto explore;
      }
      else {
        break;
      }
    }
  }
}

// Function: _explore_task
inline void Executor::_explore_task(Worker& w, Node*& t) {

  //assert(_workers[w].wsq.empty());
  //assert(!t);

  size_t num_steals = 0;
  size_t num_yields = 0;

  std::uniform_int_distribution<size_t> rdvtm(0, _workers.size()-1);
  
  // Here, we write do-while to make the worker steal at once
  // from the assigned victim.
  do {
    t = (w._id == w._vtm) ? _wsq.steal() : _workers[w._vtm]._wsq.steal();

    if(t) {
      break;
    }

    if(num_steals++ > _MAX_STEALS) {
      std::this_thread::yield();
      if(num_yields++ > 100) {
        break;
      }
    }

    w._vtm = rdvtm(w._rdgen);
  } while(!_done);

}

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  while(t) {
    _invoke(w, t);
    t = w._wsq.pop();
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& worker, Node*& t) {

  explore_task:

  _explore_task(worker, t);
  
  // The last thief who successfully stole a task will wake up
  // another thief worker to avoid starvation.
  if(t) {
    _notifier.notify(false);
    return true;
  }

  // ---- 2PC guard ----
  _notifier.prepare_wait(worker._waiter);

  if(!_wsq.empty()) {
    _notifier.cancel_wait(worker._waiter);
    worker._vtm = worker._id;
    goto explore_task;
  }
  
  if(_done) {
    _notifier.cancel_wait(worker._waiter);
    _notifier.notify(true);
    return false;
  }
  
  // We need to use index-based scanning to avoid data race
  // with _spawn which may initialize a worker at the same time.
  for(size_t vtm=0; vtm<_workers.size(); vtm++) {
    if(!_workers[vtm]._wsq.empty()) {
      _notifier.cancel_wait(worker._waiter);
      worker._vtm = vtm;
      goto explore_task;
    }
  }
  
  //--_num_thieves;
  //_num_thieves.fetch_sub(1, std::memory_order_release);
  
  /*//if(auto vtm = _find_vtm(me); vtm != _workers.size()) {
  if(!_wsq.empty()) {

    _notifier.cancel_wait(worker._waiter);
    //t = (vtm == me) ? _wsq.steal() : _workers[vtm].wsq.steal();

    t = _wsq.steal();  // must steal here
    if(t) {
      if(_num_thieves.fetch_sub(1) == 1) {
        _notifier.notify(false);
      }
      return true;
    }
    else {
      worker._vtm = worker._id;
      goto explore_task;
    }
  }

  if(_done) {
    _notifier.cancel_wait(worker._waiter);
    _notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1) {
    if(_num_actives) {
      _notifier.cancel_wait(worker._waiter);
      goto wait_for_task;
    }
    // check all queues again
    for(auto& w : _workers) {
      if(!w._wsq.empty()) {
        worker._vtm = w._id;
        _notifier.cancel_wait(worker._waiter);
        goto wait_for_task;
      }
    }
  }*/

  // Now I really need to relinguish my self to others
  _notifier.commit_wait(worker._waiter);

  goto explore_task;
}

// Function: make_observer
template<typename Observer, typename... ArgsT>
std::shared_ptr<Observer> Executor::make_observer(ArgsT&&... args) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  // use a local variable to mimic the constructor
  auto ptr = std::make_shared<Observer>(std::forward<ArgsT>(args)...);

  ptr->set_up(_workers.size());

  _observers.emplace(std::static_pointer_cast<ObserverInterface>(ptr));

  return ptr;
}

// Procedure: remove_observer
template <typename Observer>
void Executor::remove_observer(std::shared_ptr<Observer> ptr) {

  static_assert(
    std::is_base_of_v<ObserverInterface, Observer>,
    "Observer must be derived from ObserverInterface"
  );

  _observers.erase(std::static_pointer_cast<ObserverInterface>(ptr));
}

// Function: num_observers
inline size_t Executor::num_observers() const noexcept {
  return _observers.size();
}

// Procedure: _schedule
inline void Executor::_schedule(Worker& worker, Node* node) {
  
  // We need to fetch p before the release such that the read 
  // operation is synchronized properly with other thread to
  // void data race.
  auto p = node->_priority;

  node->_state.fetch_or(Node::READY, std::memory_order_release);

  // caller is a worker to this pool - starting at v3.5 we do not use
  // any complicated notification mechanism as the experimental result
  // has shown no significant advantage.
  if(worker._executor == this) {
    worker._wsq.push(node, p);
    _notifier.notify(false);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    _wsq.push(node, p);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
inline void Executor::_schedule(Node* node) {
  
  // We need to fetch p before the release such that the read 
  // operation is synchronized properly with other thread to
  // void data race.
  auto p = node->_priority;

  node->_state.fetch_or(Node::READY, std::memory_order_release);

  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    _wsq.push(node, p);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
inline void Executor::_schedule(Worker& worker, const SmallVector<Node*>& nodes) {

  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();

  if(num_nodes == 0) {
    return;
  }

  // caller is a worker to this pool - starting at v3.5 we do not use
  // any complicated notification mechanism as the experimental result
  // has shown no significant advantage.
  if(worker._executor == this) {
    for(size_t i=0; i<num_nodes; ++i) {
      // We need to fetch p before the release such that the read 
      // operation is synchronized properly with other thread to
      // void data race.
      auto p = nodes[i]->_priority;
      nodes[i]->_state.fetch_or(Node::READY, std::memory_order_release);
      worker._wsq.push(nodes[i], p);
      _notifier.notify(false);
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      auto p = nodes[k]->_priority;
      nodes[k]->_state.fetch_or(Node::READY, std::memory_order_release);
      _wsq.push(nodes[k], p);
    }
  }

  _notifier.notify_n(num_nodes);
}

// Procedure: _schedule
inline void Executor::_schedule(const SmallVector<Node*>& nodes) {

  // parent topology may be removed!
  const auto num_nodes = nodes.size();

  if(num_nodes == 0) {
    return;
  }

  // We need to fetch p before the release such that the read 
  // operation is synchronized properly with other thread to
  // void data race.
  {
    std::lock_guard<std::mutex> lock(_wsq_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      auto p = nodes[k]->_priority;
      nodes[k]->_state.fetch_or(Node::READY, std::memory_order_release);
      _wsq.push(nodes[k], p);
    }
  }

  _notifier.notify_n(num_nodes);
}

// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  // synchronize all outstanding memory operations caused by reordering
  while(!(node->_state.load(std::memory_order_acquire) & Node::READY));

  begin_invoke:

  // no need to do other things if the topology is cancelled
  if(node->_is_cancelled()) {
    _cancel_invoke(worker, node);
    return;
  }

  // if acquiring semaphore(s) exists, acquire them first
  if(node->_semaphores && !node->_semaphores->to_acquire.empty()) {
    SmallVector<Node*> nodes;
    if(!node->_acquire_all(nodes)) {
      _schedule(worker, nodes);
      return;
    }
    node->_state.fetch_or(Node::ACQUIRED, std::memory_order_release);
  }

  // condition task
  //int cond = -1;
  SmallVector<int> conds;

  // switch is faster than nested if-else due to jump table
  switch(node->_handle.index()) {
    // static task
    case Node::STATIC:{
      _invoke_static_task(worker, node);
    }
    break;

    // dynamic task
    case Node::DYNAMIC: {
      _invoke_dynamic_task(worker, node);
    }
    break;

    // condition task
    case Node::CONDITION: {
      _invoke_condition_task(worker, node, conds);
    }
    break;

    // multi-condition task
    case Node::MULTI_CONDITION: {
      _invoke_multi_condition_task(worker, node, conds);
    }
    break;

    // module task
    case Node::MODULE: {
      _invoke_module_task(worker, node);
    }
    break;

    // async task
    case Node::ASYNC: {
      _invoke_async_task(worker, node);
      _tear_down_async(node);
      return ;
    }
    break;

    // silent async task
    case Node::SILENT_ASYNC: {
      _invoke_silent_async_task(worker, node);
      _tear_down_async(node);
      return ;
    }
    break;

    // cudaflow task
    case Node::CUDAFLOW: {
      _invoke_cudaflow_task(worker, node);
    }
    break;

    // syclflow task
    case Node::SYCLFLOW: {
      _invoke_syclflow_task(worker, node);
    }
    break;

    // runtime task
    case Node::RUNTIME: {
      _invoke_runtime_task(worker, node);
    }
    break;

    // monostate (placeholder)
    default:
    break;
  }

  // if releasing semaphores exist, release them
  if(node->_semaphores && !node->_semaphores->to_release.empty()) {
    _schedule(worker, node->_release_all());
  }
  
  // Reset the join counter to support the cyclic control flow.
  // + We must do this before scheduling the successors to avoid race
  //   condition on _dependents.
  // + We must use fetch_add instead of direct assigning
  //   because the user-space call on "invoke" may explicitly schedule 
  //   this task again (e.g., pipeline) which can access the join_counter.
  if((node->_state.load(std::memory_order_relaxed) & Node::CONDITIONED)) {
    node->_join_counter.fetch_add(node->num_strong_dependents());
  }
  else {
    node->_join_counter.fetch_add(node->num_dependents());
  }

  // acquire the parent flow counter
  auto& j = (node->_parent) ? node->_parent->_join_counter :
                              node->_topology->_join_counter;

  // Here, we want to cache the latest successor with the highest priority
  Node* cache {nullptr};
  auto max_p = static_cast<unsigned>(TaskPriority::MAX);

  // Invoke the task based on the corresponding type
  switch(node->_handle.index()) {

    // condition and multi-condition tasks
    case Node::CONDITION:
    case Node::MULTI_CONDITION: {
      for(auto cond : conds) {
        if(cond >= 0 && static_cast<size_t>(cond) < node->_successors.size()) {
          auto s = node->_successors[cond];
          // zeroing the join counter for invariant
          s->_join_counter.store(0, std::memory_order_relaxed);
          j.fetch_add(1);
          if(s->_priority <= max_p) {
            if(cache) {
              _schedule(worker, cache);
            }
            cache = s;
            max_p = s->_priority;
          }
          else {
            _schedule(worker, s);
          }
        }
      }
    }
    break;

    // non-condition task
    default: {
      for(size_t i=0; i<node->_successors.size(); ++i) {
        if(auto s = node->_successors[i]; --(s->_join_counter) == 0) {
          j.fetch_add(1);
          if(s->_priority <= max_p) {
            if(cache) {
              _schedule(worker, cache);
            }
            cache = s;
            max_p = s->_priority;
          }
          else {
            _schedule(worker, s);
          }
        }
      }
    }
    break;
  }

  // tear_down the invoke
  _tear_down_invoke(worker, node);

  // perform tail recursion elimination for the right-most child to reduce
  // the number of expensive pop/push operations through the task queue
  if(cache) {
    node = cache;
    //node->_state.fetch_or(Node::READY, std::memory_order_release);
    goto begin_invoke;
  }
}

// Procedure: _tear_down_async
inline void Executor::_tear_down_async(Node* node) {
  if(node->_parent) {
    node->_parent->_join_counter.fetch_sub(1);
  }
  else {
    _decrement_topology_and_notify();
  }
  node_pool.recycle(node);
}

// Proecdure: _tear_down_invoke
inline void Executor::_tear_down_invoke(Worker& worker, Node* node) {
  // we must check parent first before substracting the join counter,
  // or it can introduce data race
  if(node->_parent == nullptr) {
    if(node->_topology->_join_counter.fetch_sub(1) == 1) {
      _tear_down_topology(worker, node->_topology);
    }
  }
  // joined subflow
  else {  
    node->_parent->_join_counter.fetch_sub(1);
  }
}

// Procedure: _cancel_invoke
inline void Executor::_cancel_invoke(Worker& worker, Node* node) {

  switch(node->_handle.index()) {
    // async task needs to carry out the promise
    case Node::ASYNC:
      std::get_if<Node::Async>(&(node->_handle))->work(true);
      _tear_down_async(node);
    break;

    // silent async doesn't need to carry out the promise
    case Node::SILENT_ASYNC:
      _tear_down_async(node);
    break;

    // tear down topology if the node is the last leaf
    default: {
      _tear_down_invoke(worker, node);
    }
    break;
  }
}

// Procedure: _observer_prologue
inline void Executor::_observer_prologue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_entry(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _observer_epilogue
inline void Executor::_observer_epilogue(Worker& worker, Node* node) {
  for(auto& observer : _observers) {
    observer->on_exit(WorkerView(worker), TaskView(*node));
  }
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  std::get_if<Node::Static>(&node->_handle)->work();
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_dynamic_task
inline void Executor::_invoke_dynamic_task(Worker& w, Node* node) {

  _observer_prologue(w, node);

  auto handle = std::get_if<Node::Dynamic>(&node->_handle);

  handle->subgraph._clear();

  Subflow sf(*this, w, node, handle->subgraph);

  handle->work(sf);

  if(sf._joinable) {
    _consume_graph(w, node, handle->subgraph);
  }

  _observer_epilogue(w, node);
}

// Procedure: _detach_dynamic_task
inline void Executor::_detach_dynamic_task(
  Worker& w, Node* p, Graph& g
) {

  // graph is empty and has no async tasks
  if(g.empty() && p->_join_counter == 0) {
    return;
  }

  SmallVector<Node*> src;

  for(auto n : g._nodes) {

    n->_state.store(Node::DETACHED, std::memory_order_relaxed);
    n->_set_up_join_counter();
    n->_topology = p->_topology;
    n->_parent = nullptr;

    if(n->num_dependents() == 0) {
      src.push_back(n);
    }
  }

  {
    std::lock_guard<std::mutex> lock(p->_topology->_taskflow._mutex);
    p->_topology->_taskflow._graph._merge(std::move(g));
  }

  p->_topology->_join_counter.fetch_add(src.size());
  _schedule(w, src);
}

// Procedure: _consume_graph
inline void Executor::_consume_graph(Worker& w, Node* p, Graph& g) {

  // graph is empty and has no async tasks
  if(g.empty() && p->_join_counter == 0) {
    return;
  }

  SmallVector<Node*> src;

  for(auto n : g._nodes) {
    n->_state.store(0, std::memory_order_relaxed);
    n->_set_up_join_counter();
    n->_topology = p->_topology;
    n->_parent = p;
    if(n->num_dependents() == 0) {
      src.push_back(n);
    }
  }
  p->_join_counter.fetch_add(src.size());
  _schedule(w, src);
  _loop_until(w, [p] () -> bool { return p->_join_counter == 0; });
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  conds = { std::get_if<Node::Condition>(&node->_handle)->work() };
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_multi_condition_task
inline void Executor::_invoke_multi_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  conds = std::get_if<Node::MultiCondition>(&node->_handle)->work();
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_cudaflow_task
inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  std::get_if<Node::cudaFlow>(&node->_handle)->work(*this, node);
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_syclflow_task
inline void Executor::_invoke_syclflow_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  std::get_if<Node::syclFlow>(&node->_handle)->work(*this, node);
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_module_task
inline void Executor::_invoke_module_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  _consume_graph(
    w, node, std::get_if<Node::Module>(&node->_handle)->graph
  );
  _observer_epilogue(w, node);
}

// Procedure: _invoke_async_task
inline void Executor::_invoke_async_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  std::get_if<Node::Async>(&node->_handle)->work(false);
  _observer_epilogue(w, node);
}

// Procedure: _invoke_silent_async_task
inline void Executor::_invoke_silent_async_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  std::get_if<Node::SilentAsync>(&node->_handle)->work();
  _observer_epilogue(w, node);
}

// Procedure: _invoke_runtime_task
inline void Executor::_invoke_runtime_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  Runtime rt(*this, w, node);
  std::get_if<Node::Runtime>(&node->_handle)->work(rt);
  _observer_epilogue(w, node);
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
inline tf::Future<void> Executor::run(Taskflow&& f) {
  return run_n(std::move(f), 1, [](){});
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow& f, C&& c) {
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run
template <typename C>
tf::Future<void> Executor::run(Taskflow&& f, C&& c) {
  return run_n(std::move(f), 1, std::forward<C>(c));
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
inline tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat) {
  return run_n(std::move(f), repeat, [](){});
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow& f, size_t repeat, C&& c) {
  return run_until(
    f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_n
template <typename C>
tf::Future<void> Executor::run_n(Taskflow&& f, size_t repeat, C&& c) {
  return run_until(
    std::move(f), [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c)
  );
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: run_until
template<typename P>
tf::Future<void> Executor::run_until(Taskflow&& f, P&& pred) {
  return run_until(std::move(f), std::forward<P>(pred), [](){});
}

// Function: run_until
template <typename P, typename C>
tf::Future<void> Executor::run_until(Taskflow& f, P&& p, C&& c) {

  _increment_topology();

  // Need to check the empty under the lock since dynamic task may
  // define detached blocks that modify the taskflow at the same time
  bool empty;
  {
    std::lock_guard<std::mutex> lock(f._mutex);
    empty = f.empty();
  }

  // No need to create a real topology but returns an dummy future
  if(empty || p()) {
    c();
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return tf::Future<void>(promise.get_future(), std::monostate{});
  }

  // create a topology for this run
  auto t = std::make_shared<Topology>(f, std::forward<P>(p), std::forward<C>(c));

  // need to create future before the topology got torn down quickly
  tf::Future<void> future(t->_promise.get_future(), t);

  // modifying topology needs to be protected under the lock
  {
    std::lock_guard<std::mutex> lock(f._mutex);
    f._topologies.push(t);
    if(f._topologies.size() == 1) {
      _set_up_topology(_this_worker(), t.get());
    }
  }

  return future;
}

// Function: run_until
template <typename P, typename C>
tf::Future<void> Executor::run_until(Taskflow&& f, P&& pred, C&& c) {

  std::list<Taskflow>::iterator itr;

  {
    std::scoped_lock<std::mutex> lock(_taskflow_mutex);
    itr = _taskflows.emplace(_taskflows.end(), std::move(f));
    itr->_satellite = itr;
  }

  return run_until(*itr, std::forward<P>(pred), std::forward<C>(c));
}

// Function: run_and_wait
template <typename T>
void Executor::run_and_wait(T& target) {
  
  auto w = _this_worker();

  if(w == nullptr) {
    TF_THROW("run_and_wait must be called by a worker of the executor");
  }

  Node parent;  // dummy parent
  _consume_graph(*w, &parent, target.graph());
}

// Function: loop_until
template <typename P>
void Executor::loop_until(P&& predicate) {
  
  auto w = _this_worker();

  if(w == nullptr) {
    TF_THROW("loop_until must be called by a worker of the executor");
  }

  _loop_until(*w, std::forward<P>(predicate));
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
}

// Procedure: _decrement_topology_and_notify
inline void Executor::_decrement_topology_and_notify() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  std::lock_guard<std::mutex> lock(_topology_mutex);
  --_num_topologies;
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  std::unique_lock<std::mutex> lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Worker* worker, Topology* tpg) {

  // ---- under taskflow lock ----

  tpg->_sources.clear();
  tpg->_taskflow._graph._clear_detached();

  // scan each node in the graph and build up the links
  for(auto node : tpg->_taskflow._graph._nodes) {

    node->_topology = tpg;
    node->_parent = nullptr;
    node->_state.store(0, std::memory_order_relaxed);

    if(node->num_dependents() == 0) {
      tpg->_sources.push_back(node);
    }

    node->_set_up_join_counter();
  }

  tpg->_join_counter = tpg->_sources.size();

  if(worker) {
    _schedule(*worker, tpg->_sources);
  }
  else {
    _schedule(tpg->_sources);
  }
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Worker& worker, Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_is_cancelled && !tpg->_pred()) {
    //assert(tpg->_join_counter == 0);
    std::lock_guard<std::mutex> lock(f._mutex);
    tpg->_join_counter = tpg->_sources.size();
    _schedule(worker, tpg->_sources);
  }
  // case 2: the final run of this topology
  else {

    // TODO: if the topology is cancelled, need to release all semaphores
    if(tpg->_call != nullptr) {
      tpg->_call();
    }

    // If there is another run (interleave between lock)
    if(std::unique_lock<std::mutex> lock(f._mutex); f._topologies.size()>1) {
      //assert(tpg->_join_counter == 0);

      // Set the promise
      tpg->_promise.set_value();
      f._topologies.pop();
      tpg = f._topologies.front().get();

      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      // set up topology needs to be under the lock or it can
      // introduce memory order error with pop
      _set_up_topology(&worker, tpg);
    }
    else {
      //assert(f._topologies.size() == 1);

      // Need to back up the promise first here becuz taskflow might be
      // destroy soon after calling get
      auto p {std::move(tpg->_promise)};

      // Back up lambda capture in case it has the topology pointer,
      // to avoid it releasing on pop_front ahead of _mutex.unlock &
      // _promise.set_value. Released safely when leaving scope.
      auto c {std::move(tpg->_call)};

      // Get the satellite if any
      auto s {f._satellite};

      // Now we remove the topology from this taskflow
      f._topologies.pop();

      //f._mutex.unlock();
      lock.unlock();

      // We set the promise in the end in case taskflow leaves the scope.
      // After set_value, the caller will return from wait
      p.set_value();

      _decrement_topology_and_notify();

      // remove the taskflow if it is managed by the executor
      // TODO: in the future, we may need to synchronize on wait
      // (which means the following code should the moved before set_value)
      if(s) {
        std::scoped_lock<std::mutex> lock(_taskflow_mutex);
        _taskflows.erase(*s);
      }
    }
  }
}

// ############################################################################
// Forward Declaration: Subflow
// ############################################################################

inline void Subflow::join() {

  // assert(this_worker().worker == &_worker);

  if(!_joinable) {
    TF_THROW("subflow not joinable");
  }

  // only the parent worker can join the subflow
  _executor._consume_graph(_worker, _parent, _graph);
  _joinable = false;
}

inline void Subflow::detach() {

  // assert(this_worker().worker == &_worker);

  if(!_joinable) {
    TF_THROW("subflow already joined or detached");
  }

  // only the parent worker can detach the subflow
  _executor._detach_dynamic_task(_worker, _parent, _graph);
  _joinable = false;
}

// Function: named_async
template <typename F, typename... ArgsT>
auto Subflow::named_async(const std::string& name, F&& f, ArgsT&&... args) {
  return _named_async(
    *_executor._this_worker(), name, std::forward<F>(f), std::forward<ArgsT>(args)...
  );
}

// Function: _named_async
template <typename F, typename... ArgsT>
auto Subflow::_named_async(
  Worker& w,
  const std::string& name,
  F&& f,
  ArgsT&&... args
) {

  _parent->_join_counter.fetch_add(1);

  using T = std::invoke_result_t<F, ArgsT...>;
  using R = std::conditional_t<std::is_same_v<T, void>, void, std::optional<T>>;

  std::promise<R> p;

  auto tpg = std::make_shared<AsyncTopology>();

  Future<R> fu(p.get_future(), tpg);

  auto node = node_pool.animate(
    std::in_place_type_t<Node::Async>{},
    [p=make_moc(std::move(p)), f=std::forward<F>(f), args...]
    (bool cancel) mutable {
      if constexpr(std::is_same_v<R, void>) {
        if(!cancel) {
          f(args...);
        }
        p.object.set_value();
      }
      else {
        p.object.set_value(cancel ? std::nullopt : std::make_optional(f(args...)));
      }
    },
    std::move(tpg)
  );

  node->_name = name;
  node->_topology = _parent->_topology;
  node->_parent = _parent;

  _executor._schedule(w, node);

  return fu;
}

// Function: async
template <typename F, typename... ArgsT>
auto Subflow::async(F&& f, ArgsT&&... args) {
  return named_async("", std::forward<F>(f), std::forward<ArgsT>(args)...);
}

// Function: _named_silent_async
template <typename F, typename... ArgsT>
void Subflow::_named_silent_async(
  Worker& w, const std::string& name, F&& f, ArgsT&&... args
) {

  _parent->_join_counter.fetch_add(1);

  auto node = node_pool.animate(
    std::in_place_type_t<Node::SilentAsync>{},
    [f=std::forward<F>(f), args...] () mutable {
      f(args...);
    }
  );

  node->_name = name;
  node->_topology = _parent->_topology;
  node->_parent = _parent; 

  _executor._schedule(w, node);
}

// Function: silent_async
template <typename F, typename... ArgsT>
void Subflow::named_silent_async(const std::string& name, F&& f, ArgsT&&... args) {
  _named_silent_async(
    *_executor._this_worker(), name, std::forward<F>(f), std::forward<ArgsT>(args)...
  );
}

// Function: named_silent_async
template <typename F, typename... ArgsT>
void Subflow::silent_async(F&& f, ArgsT&&... args) {
  named_silent_async("", std::forward<F>(f), std::forward<ArgsT>(args)...);
}

// ############################################################################
// Forward Declaration: Runtime
// ############################################################################

// Procedure: schedule
inline void Runtime::schedule(Task task) {
  auto node = task._node;
  auto& j = node->_parent ? node->_parent->_join_counter :
                            node->_topology->_join_counter;
  j.fetch_add(1);
  _executor._schedule(_worker, node);
}

// Procedure: emplace
template <typename T>
void Runtime::run_and_wait(T&& target) {

  // dynamic task (subflow)
  if constexpr(is_dynamic_task_v<T>) {
    Graph graph;
    Subflow sf(_executor, _worker, _parent, graph);
    target(sf);
    if(sf._joinable) {
      _executor._consume_graph(_worker, _parent, graph);
    }
  }
  // graph object
  else {
    _executor._consume_graph(_worker, _parent, target.graph());
  }
}

}  // end of namespace tf -----------------------------------------------------






