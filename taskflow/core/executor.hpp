#pragma once

#include "observer.hpp"
#include "taskflow.hpp"
#include "async_task.hpp"

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

  @param N the number of workers (default std::thread::hardware_concurrency)
  
  The constructor spawns @c N worker threads to run tasks in a
  work-stealing loop. The number of workers must be greater than zero
  or an exception will be thrown.
  By default, the number of worker threads is equal to the maximum
  hardware concurrency returned by std::thread::hardware_concurrency.
  */
  explicit Executor(size_t N = std::thread::hardware_concurrency());

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
      executor.corun(tf);
      //executor.run(tf).wait();  <- blocking the worker without doing anything
      //                             will introduce deadlock
    });
  }
  executor.run(taskflow).wait();
  @endcode 

  The method is thread-safe as long as the target is not concurrently
  ran by two or more threads.

  @attention
  You must call tf::Executor::corun from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename T>
  void corun(T& target);

  /**
  @brief keeps running the work-stealing loop until the predicate becomes true
  
  @tparam P predicate type
  @param predicate a boolean predicate to indicate when to stop the loop

  The method keeps the caller worker running in the work-stealing loop
  until the stop predicate becomes true.

  @code{.cpp}
  taskflow.emplace([&](){
    std::future<void> fu = std::async([](){ std::sleep(100s); });
    executor.corun_until([](){
      return fu.wait_for(std::chrono::seconds(0)) == future_status::ready;
    });
  });
  @endcode

  @attention
  You must call tf::Executor::corun_until from a worker of the calling executor
  or an exception will be thrown.
  */
  template <typename P>
  void corun_until(P&& predicate);

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
 
  // --------------------------------------------------------------------------
  // Observer methods
  // --------------------------------------------------------------------------

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

  // --------------------------------------------------------------------------
  // Async Task Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief creates a parameterized asynchronous task to run the given function

  @tparam P task parameter type
  @tparam F callable type

  @param params task parameters
  @param func callable object

  @return a @std_future that will hold the result of the execution
  
  The method creates a parameterized asynchronous task 
  to run the given function and return a @std_future object 
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::future<int> future = executor.async("name", [](){
    std::cout << "create an asynchronous task with a name and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F>
  auto async(P&& params, F&& func);

  /**
  @brief runs a given function asynchronously

  @tparam F callable type

  @param func callable object

  @return a @std_future that will hold the result of the execution

  The method creates an asynchronous task to run the given function
  and return a @std_future object that eventually will hold the result
  of the return value.

  @code{.cpp}
  std::future<int> future = executor.async([](){
    std::cout << "create an asynchronous task and returns 1\n";
    return 1;
  });
  future.get();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  auto async(F&& func);

  /**
  @brief similar to tf::Executor::async but does not return a future object

  @tparam F callable type

  @param params task parameters
  @param func callable object

  The method creates a parameterized asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async("name", [](){
    std::cout << "create an asynchronous task with a name and no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F>
  void silent_async(P&& params, F&& func);
  
  /**
  @brief similar to tf::Executor::async but does not return a future object
  
  @tparam F callable type
  
  @param func callable object

  The method creates an asynchronous task 
  to run the given function without returning any @std_future object.
  This member function is more efficient than tf::Executor::async 
  and is encouraged to use when applications do not need a @std_future to acquire
  the result or synchronize the execution.

  @code{.cpp}
  executor.silent_async([](){
    std::cout << "create an asynchronous task with no return\n";
  });
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F>
  void silent_async(F&& func);

  // --------------------------------------------------------------------------
  // Silent Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given dependents finish

  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  executor.silent_dependent_async([](){ printf("C runs after A and B\n"); }, A, B);
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given dependents finish
  
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, A, B
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of dependents finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename F, typename I, 
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of dependents finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param params tasks parameters
  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)

  @return a tf::AsyncTask handle 
  
  This member function is more efficient than tf::Executor::dependent_async
  and is encouraged to use when you do not want a @std_future to
  acquire the result or synchronize the execution.
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  executor.silent_dependent_async(
    "C", [](){ printf("C runs after A and B\n"); }, array.begin(), array.end()
  );
  executor.wait_for_all();
  @endcode

  This member function is thread-safe.
  */
  template <typename P, typename F, typename I, 
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  tf::AsyncTask silent_dependent_async(P&& params, F&& func, I first, I last);
  
  // --------------------------------------------------------------------------
  // Dependent Async Methods
  // --------------------------------------------------------------------------
  
  /**
  @brief runs the given function asynchronously 
         when the given dependents finish
  
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask

  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async([](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async([](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  fuC.get();  // C finishes, which in turns means both A and B finish
  @endcode

  You can mixed the use of tf::AsyncTask handles 
  returned by Executor::dependent_async and Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, typename... Tasks,
    std::enable_if_t<all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously
         when the given dependents finish
  
  @tparam P task parameters type
  @tparam F callable type
  @tparam Tasks task types convertible to tf::AsyncTask
  
  @param params task parameters
  @param func callable object
  @param tasks asynchronous tasks on which this execution depends
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three named asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  tf::AsyncTask A = executor.silent_dependent_async("A", [](){ printf("A\n"); });
  tf::AsyncTask B = executor.silent_dependent_async("B", [](){ printf("B\n"); });
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    A, B
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mixed the use of tf::AsyncTask handles 
  returned by Executor::dependent_async and Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename P, typename F, typename... Tasks,
    std::enable_if_t<is_task_params_v<P> && all_same_v<AsyncTask, std::decay_t<Tasks>...>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, Tasks&&... tasks);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of dependents finish
  
  @tparam F callable type
  @tparam I iterator type 

  @param func callable object
  @param first iterator to the beginning (inclusive)
  @param last iterator to the end (exclusive)
  
  @return a pair of a tf::AsyncTask handle and 
                    a @std_future that holds the result of the execution
  
  The example below creates three asynchronous tasks, @c A, @c B, and @c C,
  in which task @c C runs after task @c A and task @c B.
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async([](){ printf("A\n"); }),
    executor.silent_dependent_async([](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mixed the use of tf::AsyncTask handles 
  returned by Executor::dependent_async and Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename F, typename I,
    std::enable_if_t<!std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(F&& func, I first, I last);
  
  /**
  @brief runs the given function asynchronously 
         when the given range of dependents finish
  
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
  Task @c C returns a pair of its tf::AsyncTask handle and a std::future<int>
  that eventually will hold the result of the execution.
  Assigned task names will appear in the observers of the executor.

  @code{.cpp}
  std::array<tf::AsyncTask, 2> array {
    executor.silent_dependent_async("A", [](){ printf("A\n"); }),
    executor.silent_dependent_async("B", [](){ printf("B\n"); })
  };
  auto [C, fuC] = executor.dependent_async(
    "C",
    [](){ 
      printf("C runs after A and B\n"); 
      return 1;
    }, 
    array.begin(), array.end()
  );
  assert(fuC.get()==1);  // C finishes, which in turns means both A and B finish
  @endcode

  You can mixed the use of tf::AsyncTask handles 
  returned by Executor::dependent_async and Executor::silent_dependent_async
  when specifying task dependencies.

  This member function is thread-safe.
  */
  template <typename P, typename F, typename I,
    std::enable_if_t<is_task_params_v<P> && !std::is_same_v<std::decay_t<I>, AsyncTask>, void>* = nullptr
  >
  auto dependent_async(P&& params, F&& func, I first, I last);

  private:
    
  const size_t _MAX_STEALS;
  
  std::mutex _wsq_mutex;
  std::mutex _taskflows_mutex;

#ifdef __cpp_lib_atomic_wait
  std::atomic<size_t> _num_topologies {0};
  std::atomic_flag _all_spawned = ATOMIC_FLAG_INIT;
#else
  std::condition_variable _topology_cv;
  std::mutex _topology_mutex;
  size_t _num_topologies {0};
#endif
  
  std::unordered_map<std::thread::id, size_t> _wids;
  std::vector<std::thread> _threads;
  std::vector<Worker> _workers;
  std::list<Taskflow> _taskflows;

  Notifier _notifier;

  TaskQueue<Node*> _wsq;

  std::atomic<bool> _done {0};

  std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

  Worker* _this_worker();
  
  bool _wait_for_task(Worker&, Node*&);
  bool _invoke_module_task_internal(Worker&, Node*);

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
  void _set_up_graph(Graph&, Node*, Topology*, int, SmallVector<Node*>&);
  void _tear_down_topology(Worker&, Topology*);
  void _tear_down_async(Node*);
  void _tear_down_dependent_async(Worker&, Node*);
  void _tear_down_invoke(Worker&, Node*);
  void _increment_topology();
  void _decrement_topology();
  void _invoke(Worker&, Node*);
  void _invoke_static_task(Worker&, Node*);
  void _invoke_subflow_task(Worker&, Node*);
  void _detach_subflow_task(Worker&, Node*, Graph&);
  void _invoke_condition_task(Worker&, Node*, SmallVector<int>&);
  void _invoke_multi_condition_task(Worker&, Node*, SmallVector<int>&);
  void _invoke_module_task(Worker&, Node*);
  void _invoke_async_task(Worker&, Node*);
  void _invoke_dependent_async_task(Worker&, Node*);
  void _process_async_dependent(Node*, tf::AsyncTask&, size_t&);
  void _process_exception(Worker&, Node*);
  void _schedule_async_task(Node*);
  void _corun_graph(Worker&, Node*, Graph&);
  
  template <typename P>
  void _corun_until(Worker&, P&&);
};

// Constructor
inline Executor::Executor(size_t N) :
  _MAX_STEALS {((N+1) << 1)},
  _threads    {N},
  _workers    {N},
  _notifier   {N} {

  if(N == 0) {
    TF_THROW("executor must define at least one worker");
  }

  _spawn(N);

  // initialize the default observer if requested
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
#ifdef __cpp_lib_atomic_wait
  return _num_topologies.load(std::memory_order_relaxed);
#else
  return _num_topologies;
#endif
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

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto i = _wids.find(std::this_thread::get_id());
  return i == _wids.end() ? -1 : static_cast<int>(_workers[i->second]._id);
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N) {

#ifdef __cpp_lib_atomic_wait
#else
  std::mutex mutex;
  std::condition_variable cond;
  size_t n=0;
#endif

  for(size_t id=0; id<N; ++id) {

    _workers[id]._id = id;
    _workers[id]._vtm = id;
    _workers[id]._executor = this;
    _workers[id]._waiter = &_notifier._waiters[id];

    _threads[id] = std::thread([&, &w=_workers[id]] () {

#ifdef __cpp_lib_atomic_wait
      // wait for the caller thread to initialize the ID mapping
      _all_spawned.wait(false, std::memory_order_acquire);
      w._thread = &_threads[w._id];
#else
      // update the ID mapping of this thread
      w._thread = &_threads[w._id];
      {
        std::scoped_lock lock(mutex);
        _wids[std::this_thread::get_id()] = w._id;
        if(n++; n == num_workers()) {
          cond.notify_one();
        }
      }
#endif

      Node* t = nullptr;
      
      while(1) {

        // execute the tasks.
        _exploit_task(w, t);

        // wait for tasks
        if(_wait_for_task(w, t) == false) {
          break;
        }
      }

    });
    
    // POSIX-like system can use the following to affine threads to cores 
    //cpu_set_t cpuset;
    //CPU_ZERO(&cpuset);
    //CPU_SET(id, &cpuset);
    //pthread_setaffinity_np(
    //  _threads[id].native_handle(), sizeof(cpu_set_t), &cpuset
    //);

#ifdef __cpp_lib_atomic_wait
    //_wids[_threads[id].get_id()] = id;
    _wids.emplace(std::piecewise_construct,
      std::forward_as_tuple(_threads[id].get_id()), std::forward_as_tuple(id)
    );
#endif
  }
  
#ifdef __cpp_lib_atomic_wait
  _all_spawned.test_and_set(std::memory_order_release);
  _all_spawned.notify_all();
#else
  std::unique_lock<std::mutex> lock(mutex);
  cond.wait(lock, [&](){ return n==N; });
#endif
}

// Function: _corun_until
template <typename P>
void Executor::_corun_until(Worker& w, P&& stop_predicate) {
  
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
  
  SmallVector<int> conds;

  // no need to do other things if the topology is cancelled
  if(node->_is_cancelled()) {
    _tear_down_invoke(worker, node);
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

  // switch is faster than nested if-else due to jump table
  switch(node->_handle.index()) {
    // static task
    case Node::STATIC:{
      _invoke_static_task(worker, node);
    }
    break;

    // subflow task
    case Node::SUBFLOW: {
      _invoke_subflow_task(worker, node);
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

    // dependent async task
    case Node::DEPENDENT_ASYNC: {
      _invoke_dependent_async_task(worker, node);
      _tear_down_dependent_async(worker, node);
      if(worker._cache) {
        node = worker._cache;
        goto begin_invoke;
      }
      return;
    }
    break;

    // monostate (placeholder)
    default:
    break;
  }

  //invoke_successors:

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
    node->_join_counter.fetch_add(node->num_strong_dependents(), std::memory_order_relaxed);
  }
  else {
    node->_join_counter.fetch_add(node->num_dependents(), std::memory_order_relaxed);
  }

  // acquire the parent flow counter
  auto& j = (node->_parent) ? node->_parent->_join_counter :
                              node->_topology->_join_counter;

  // Here, we want to cache the latest successor with the highest priority
  worker._cache = nullptr;
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
          j.fetch_add(1, std::memory_order_relaxed);
          if(s->_priority <= max_p) {
            if(worker._cache) {
              _schedule(worker, worker._cache);
            }
            worker._cache = s;
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
        //if(auto s = node->_successors[i]; --(s->_join_counter) == 0) {
        if(auto s = node->_successors[i]; 
          s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          j.fetch_add(1, std::memory_order_relaxed);
          if(s->_priority <= max_p) {
            if(worker._cache) {
              _schedule(worker, worker._cache);
            }
            worker._cache = s;
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
  if(worker._cache) {
    node = worker._cache;
    //node->_state.fetch_or(Node::READY, std::memory_order_release);
    goto begin_invoke;
  }
}

// Procedure: _tear_down_invoke
inline void Executor::_tear_down_invoke(Worker& worker, Node* node) {
  // we must check parent first before subtracting the join counter,
  // or it can introduce data race
  if(auto parent = node->_parent; parent == nullptr) {
    if(node->_topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _tear_down_topology(worker, node->_topology);
    }
  }
  // Here we asssume the parent is in a busy loop (e.g., corun) waiting for
  // its join counter to become 0.
  else {
    //parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel);
    parent->_join_counter.fetch_sub(1, std::memory_order_release);
  }
  //// module task
  //else {  
  //  auto id = parent->_handle.index();
  //  if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
  //    if(id == Node::MODULE) {
  //      return parent;
  //    }
  //  }
  //}
  //return nullptr;
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

// Procedure: _process_exception
inline void Executor::_process_exception(Worker&, Node* node) {

  constexpr static auto flag = Topology::EXCEPTION | Topology::CANCELLED;
  
  // if the node has a parent, we store the exception in its parent
  if(auto parent = node->_parent; parent) { 
    if ((parent->_state.fetch_or(Node::EXCEPTION, std::memory_order_relaxed) & Node::EXCEPTION) == 0) {
      parent->_exception_ptr = std::current_exception();
    }
    // TODO if the node has a topology, cancel it to enable early stop
    //if(auto tpg = node->_topology; tpg) {
    //  tpg->_state.fetch_or(Topology::CANCELLED, std::memory_order_relaxed);
    //}
  }
  // multiple tasks may throw, so we only take the first thrown exception
  else if(auto tpg = node->_topology; tpg && 
    ((tpg->_state.fetch_or(flag, std::memory_order_relaxed) & Topology::EXCEPTION) == 0)
  ) {
    tpg->_exception_ptr = std::current_exception();
  }
  // TODO: skip the exception that is not associated with any taskflows
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Static>(&node->_handle)->work;
    switch(work.index()) {
      case 0:
        std::get_if<0>(&work)->operator()();
      break;

      case 1:
        Runtime rt(*this, worker, node);
        std::get_if<1>(&work)->operator()(rt);
        node->_process_exception();
      break;
    }
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_subflow_task
inline void Executor::_invoke_subflow_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(w, node, {
    auto handle = std::get_if<Node::Subflow>(&node->_handle);
    handle->subgraph._clear();
    Subflow sf(*this, w, node, handle->subgraph);
    handle->work(sf);
    if(sf._joinable) {
      _corun_graph(w, node, handle->subgraph);
    }
    node->_process_exception();
  });
  _observer_epilogue(w, node);
}

// Procedure: _detach_subflow_task
inline void Executor::_detach_subflow_task(Worker& w, Node* p, Graph& g) {

  // graph is empty and has no async tasks
  if(g.empty() && p->_join_counter.load(std::memory_order_acquire) == 0) {
    return;
  }

  SmallVector<Node*> src;
  _set_up_graph(g, nullptr, p->_topology, Node::DETACHED, src);

  {
    std::lock_guard<std::mutex> lock(p->_topology->_taskflow._mutex);
    p->_topology->_taskflow._graph._merge(std::move(g));
  }

  p->_topology->_join_counter.fetch_add(src.size(), std::memory_order_relaxed);
  _schedule(w, src);
}

// Procedure: _corun_graph
inline void Executor::_corun_graph(Worker& w, Node* p, Graph& g) {

  // assert(p);

  // graph is empty and has no async tasks (subflow)
  if(g.empty() && p->_join_counter.load(std::memory_order_acquire) == 0) {
    return;
  }

  SmallVector<Node*> src;

  _set_up_graph(g, p, p->_topology, 0, src);
  p->_join_counter.fetch_add(src.size(), std::memory_order_relaxed);
  
  _schedule(w, src);

  _corun_until(w, [p] () -> bool { 
    return p->_join_counter.load(std::memory_order_acquire) == 0; }
  );
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Condition>(&node->_handle)->work;
    switch(work.index()) {
      case 0:
        conds = { std::get_if<0>(&work)->operator()() };
      break;

      case 1:
        Runtime rt(*this, worker, node);
        conds = { std::get_if<1>(&work)->operator()(rt) };
        node->_process_exception();
      break;
    }
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_multi_condition_task
inline void Executor::_invoke_multi_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::MultiCondition>(&node->_handle)->work;
    switch(work.index()) {
      case 0:
        conds = std::get_if<0>(&work)->operator()();
      break;

      case 1:
        Runtime rt(*this, worker, node);
        conds = std::get_if<1>(&work)->operator()(rt);
        node->_process_exception();
      break;
    }
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_module_task
inline void Executor::_invoke_module_task(Worker& w, Node* node) {
  _observer_prologue(w, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(w, node, {
    _corun_graph(w, node, std::get_if<Node::Module>(&node->_handle)->graph);
    node->_process_exception();
  });
  _observer_epilogue(w, node);
}

//// Function: _invoke_module_task_internal
//inline bool Executor::_invoke_module_task_internal(Worker& w, Node* p) {
//  
//  // acquire the underlying graph
//  auto& g = std::get_if<Node::Module>(&p->_handle)->graph;
//
//  // no need to do anything if the graph is empty
//  if(g.empty()) {
//    return false;
//  }
//
//  SmallVector<Node*> src;
//  _set_up_graph(g, p, p->_topology, 0, src);
//  p->_join_counter.fetch_add(src.size(), std::memory_order_relaxed);
//
//  _schedule(w, src);
//  return true;
//}

// Procedure: _invoke_async_task
inline void Executor::_invoke_async_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Async>(&node->_handle)->work;
    switch(work.index()) {
      case 0:
        std::get_if<0>(&work)->operator()();
      break;

      case 1:
        Runtime rt(*this, worker, node);
        std::get_if<1>(&work)->operator()(rt);
      break;
    }
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_dependent_async_task
inline void Executor::_invoke_dependent_async_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::DependentAsync>(&node->_handle)->work;
    switch(work.index()) {
      case 0:
        std::get_if<0>(&work)->operator()();
      break;

      case 1:
        Runtime rt(*this, worker, node);
        std::get_if<1>(&work)->operator()(rt);
      break;
    }
  });
  _observer_epilogue(worker, node);
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

  // Need to check the empty under the lock since subflow task may
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
    _decrement_topology();
    return tf::Future<void>(promise.get_future());
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
    std::scoped_lock<std::mutex> lock(_taskflows_mutex);
    itr = _taskflows.emplace(_taskflows.end(), std::move(f));
    itr->_satellite = itr;
  }

  return run_until(*itr, std::forward<P>(pred), std::forward<C>(c));
}

// Function: corun
template <typename T>
void Executor::corun(T& target) {
  
  auto w = _this_worker();

  if(w == nullptr) {
    TF_THROW("corun must be called by a worker of the executor");
  }

  Node parent;  // auxiliary parent
  _corun_graph(*w, &parent, target.graph());
  parent._process_exception();
}

// Function: corun_until
template <typename P>
void Executor::corun_until(P&& predicate) {
  
  auto w = _this_worker();

  if(w == nullptr) {
    TF_THROW("corun_until must be called by a worker of the executor");
  }

  _corun_until(*w, std::forward<P>(predicate));

  // TODO: exception?
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
#ifdef __cpp_lib_atomic_wait
  _num_topologies.fetch_add(1, std::memory_order_relaxed);
#else
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
#endif
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
#ifdef __cpp_lib_atomic_wait
  if(_num_topologies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    _num_topologies.notify_all();
  }
#else
  std::lock_guard<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
#endif
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
#ifdef __cpp_lib_atomic_wait
  size_t n = _num_topologies.load(std::memory_order_acquire);
  while(n != 0) {
    _num_topologies.wait(n, std::memory_order_acquire);
    n = _num_topologies.load(std::memory_order_acquire);
  }
#else
  std::unique_lock<std::mutex> lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
#endif
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Worker* worker, Topology* tpg) {

  // ---- under taskflow lock ----

  tpg->_sources.clear();
  tpg->_taskflow._graph._clear_detached();
  _set_up_graph(tpg->_taskflow._graph, nullptr, tpg, 0, tpg->_sources);
  tpg->_join_counter.store(tpg->_sources.size(), std::memory_order_relaxed);

  if(worker) {
    _schedule(*worker, tpg->_sources);
  }
  else {
    _schedule(tpg->_sources);
  }
}

// Function: _set_up_graph
inline void Executor::_set_up_graph(
  Graph& g, Node* parent, Topology* tpg, int state, SmallVector<Node*>& src
) {
  for(auto node : g._nodes) {
    node->_topology = tpg;
    node->_parent = parent;
    node->_state.store(state, std::memory_order_relaxed);
    if(node->num_dependents() == 0) {
      src.push_back(node);
    }
    node->_set_up_join_counter();
    node->_exception_ptr = nullptr;
  }
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Worker& worker, Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_exception_ptr && !tpg->cancelled() && !tpg->_pred()) {
    //assert(tpg->_join_counter == 0);
    std::lock_guard<std::mutex> lock(f._mutex);
    tpg->_join_counter.store(tpg->_sources.size(), std::memory_order_relaxed);
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

      auto fetched_tpg {std::move(f._topologies.front())};
      f._topologies.pop();
      auto satellite {f._satellite};

      lock.unlock();
      
      // Soon after we carry out the promise, there is no longer any guarantee
      // for the lifetime of the associated taskflow.
      fetched_tpg->_carry_out_promise();

      _decrement_topology();

      // remove the taskflow if it is managed by the executor
      // TODO: in the future, we may need to synchronize on wait
      // (which means the following code should the moved before set_value)
      if(satellite) {
        std::scoped_lock<std::mutex> satellite_lock(_taskflows_mutex);
        _taskflows.erase(*satellite);
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
  _executor._corun_graph(_worker, _parent, _graph);

  // if any exception is caught from subflow tasks, rethrow it
  _parent->_process_exception();

  _joinable = false;
}

inline void Subflow::detach() {

  // assert(this_worker().worker == &_worker);

  if(!_joinable) {
    TF_THROW("subflow already joined or detached");
  }

  // only the parent worker can detach the subflow
  _executor._detach_subflow_task(_worker, _parent, _graph);
  _joinable = false;
}

// ############################################################################
// Forward Declaration: Runtime
// ############################################################################

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

// Procedure: corun
template <typename T>
void Runtime::corun(T&& target) {
  _executor._corun_graph(_worker, _parent, target.graph());
  _parent->_process_exception();
}

// Procedure: corun_until
template <typename P>
void Runtime::corun_until(P&& predicate) {
  _executor._corun_until(_worker, std::forward<P>(predicate));
  // TODO: exception?
}

// Function: corun_all
inline void Runtime::corun_all() {
  _executor._corun_until(_worker, [this] () -> bool { 
    return _parent->_join_counter.load(std::memory_order_acquire) == 0; 
  });
  _parent->_process_exception();
}

// Destructor
inline Runtime::~Runtime() {
  _executor._corun_until(_worker, [this] () -> bool { 
    return _parent->_join_counter.load(std::memory_order_acquire) == 0; 
  });
}

// ------------------------------------
// Runtime::silent_async series
// ------------------------------------

// Function: _silent_async
template <typename P, typename F>
void Runtime::_silent_async(Worker& w, P&& params, F&& f) {

  _parent->_join_counter.fetch_add(1, std::memory_order_relaxed);

  auto node = node_pool.animate(
    std::forward<P>(params), _parent->_topology, _parent, 0,
    std::in_place_type_t<Node::Async>{}, std::forward<F>(f)
  );

  _executor._schedule(w, node);
}

// Function: silent_async
template <typename F>
void Runtime::silent_async(F&& f) {
  _silent_async(*_executor._this_worker(), DefaultTaskParams{}, std::forward<F>(f));
}

// Function: silent_async
template <typename P, typename F>
void Runtime::silent_async(P&& params, F&& f) {
  _silent_async(*_executor._this_worker(), std::forward<P>(params), std::forward<F>(f));
}

// Function: silent_async_unchecked
template <typename F>
void Runtime::silent_async_unchecked(F&& f) {
  _silent_async(_worker, DefaultTaskParams{}, std::forward<F>(f));
}

// Function: silent_async_unchecked
template <typename P, typename F>
void Runtime::silent_async_unchecked(P&& params, F&& f) {
  _silent_async(_worker, std::forward<P>(params), std::forward<F>(f));
}

// ------------------------------------
// Runtime::async series
// ------------------------------------

// Function: _async
template <typename P, typename F>
auto Runtime::_async(Worker& w, P&& params, F&& f) {

  _parent->_join_counter.fetch_add(1, std::memory_order_relaxed);

  using R = std::invoke_result_t<std::decay_t<F>>;

  std::packaged_task<R()> p(std::forward<F>(f));
  auto fu{p.get_future()};

  auto node = node_pool.animate(
    std::forward<P>(params), _parent->_topology, _parent, 0, 
    std::in_place_type_t<Node::Async>{},
    [p=make_moc(std::move(p))] () mutable { p.object(); }
  );

  _executor._schedule(w, node);

  return fu;
}

// Function: async
template <typename F>
auto Runtime::async(F&& f) {
  return _async(*_executor._this_worker(), DefaultTaskParams{}, std::forward<F>(f));
}

// Function: async
template <typename P, typename F>
auto Runtime::async(P&& params, F&& f) {
  return _async(*_executor._this_worker(), std::forward<P>(params), std::forward<F>(f));
}



}  // end of namespace tf -----------------------------------------------------






