#pragma once

#include "observer.hpp"
#include "taskflow.hpp"
#include "async_task.hpp"
#include "freelist.hpp"

/**
@file executor.hpp
@brief executor include file
*/

namespace tf {

// ----------------------------------------------------------------------------
// Executor Definition
// ----------------------------------------------------------------------------

/** 
@class Executor

@brief class to create an executor 

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
  friend class Algorithm;

  public:

  /**
  @brief constructs the executor with @c N worker threads

  @param N number of workers (default std::thread::hardware_concurrency)
  @param wix interface class instance to configure workers' behaviors

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
  the work-stealing loop of the scheduler, thereby avoiding potential
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
  @brief queries the number of workers that are currently not making any stealing attempts
  */
  size_t num_waiters() const noexcept;
  
  /**
  @brief queries the number of queues used in the work-stealing loop
  */
  size_t num_queues() const noexcept;

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
  @brief queries the id of the caller thread within this executor

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
         when the given predecessors finish

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
         when the given predecessors finish
  
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
         when the given range of predecessors finish
  
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
         when the given range of predecessors finish
  
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
         when the given predecessors finish
  
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
         when the given predecessors finish
  
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
    
  std::mutex _taskflows_mutex;
  
  std::vector<Worker> _workers;
  DefaultNotifier _notifier;

#if __cplusplus >= TF_CPP20
  std::atomic<size_t> _num_topologies {0};
#else
  std::condition_variable _topology_cv;
  std::mutex _topology_mutex;
  size_t _num_topologies {0};
#endif
  
  std::list<Taskflow> _taskflows;

  Freelist<Node*> _buffers;

  std::shared_ptr<WorkerInterface> _worker_interface;
  std::unordered_set<std::shared_ptr<ObserverInterface>> _observers;

  void _observer_prologue(Worker&, Node*);
  void _observer_epilogue(Worker&, Node*);
  void _spawn(size_t);
  void _exploit_task(Worker&, Node*&);
  bool _explore_task(Worker&, Node*&);
  void _schedule(Worker&, Node*);
  void _schedule(Node*);
  void _set_up_topology(Worker*, Topology*);
  void _tear_down_topology(Worker&, Topology*);
  void _tear_down_async(Worker&, Node*, Node*&);
  void _tear_down_dependent_async(Worker&, Node*, Node*&);
  void _tear_down_invoke(Worker&, Node*, Node*&);
  void _increment_topology();
  void _decrement_topology();
  void _invoke(Worker&, Node*);
  void _invoke_static_task(Worker&, Node*);
  void _invoke_condition_task(Worker&, Node*, SmallVector<int>&);
  void _invoke_multi_condition_task(Worker&, Node*, SmallVector<int>&);
  void _process_async_dependent(Node*, tf::AsyncTask&, size_t&);
  void _process_exception(Worker&, Node*);
  void _schedule_async_task(Node*);
  void _update_cache(Worker&, Node*&, Node*);

  bool _wait_for_task(Worker&, Node*&);
  bool _invoke_subflow_task(Worker&, Node*);
  bool _invoke_module_task(Worker&, Node*);
  bool _invoke_module_task_impl(Worker&, Node*, Graph&);
  bool _invoke_async_task(Worker&, Node*);
  bool _invoke_dependent_async_task(Worker&, Node*);
  bool _invoke_runtime_task(Worker&, Node*);
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&)>&);
  bool _invoke_runtime_task_impl(Worker&, Node*, std::function<void(Runtime&, bool)>&);

  template <typename I>
  I _set_up_graph(I, I, Topology*, Node*);
  
  template <typename P>
  void _corun_until(Worker&, P&&);
  
  template <typename I>
  void _corun_graph(Worker&, Node*, I, I);

  template <typename I>
  void _schedule(Worker&, I, I);

  template <typename I>
  void _schedule(I, I);

  template <typename I>
  void _schedule_graph_with_parent(Worker&, I, I, Node*);

  template <typename P, typename F>
  auto _async(P&&, F&&, Topology*, Node*);

  template <typename P, typename F>
  void _silent_async(P&&, F&&, Topology*, Node*);

};

#ifndef DOXYGEN_GENERATING_OUTPUT

// Constructor
inline Executor::Executor(size_t N, std::shared_ptr<WorkerInterface> wix) :
  _workers  (N),
  _notifier (N),
  _buffers  (N),
  _worker_interface(std::move(wix)) {

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
  for(size_t i=0; i<_workers.size(); ++i) {
  #if __cplusplus >= TF_CPP20
    _workers[i]._done.test_and_set(std::memory_order_relaxed);
  #else
    _workers[i]._done.store(true, std::memory_order_relaxed);
  #endif
  }

  _notifier.notify_all();

  for(auto& w : _workers) {
    w._thread.join();
  }
}

// Function: num_workers
inline size_t Executor::num_workers() const noexcept {
  return _workers.size();
}

// Function: num_waiters
inline size_t Executor::num_waiters() const noexcept {
#if __cplusplus >= TF_CPP20
  return _notifier.num_waiters();
#else
  // Unfortunately, nonblocking notifier does not have an easy way to return
  // the number of workers that are not making stealing attempts.
  return 0;
#endif
}

// Function: num_queues
inline size_t Executor::num_queues() const noexcept {
  return _workers.size() + _buffers.size();
}

// Function: num_topologies
inline size_t Executor::num_topologies() const {
#if __cplusplus >= TF_CPP20
  return _num_topologies.load(std::memory_order_relaxed);
#else
  return _num_topologies;
#endif
}

// Function: num_taskflows
inline size_t Executor::num_taskflows() const {
  return _taskflows.size();
}

// Function: this_worker_id
inline int Executor::this_worker_id() const {
  auto w = pt::this_worker;
  return (w && w->_executor == this) ? static_cast<int>(w->_id) : -1;
}

// Procedure: _spawn
inline void Executor::_spawn(size_t N) {

  for(size_t id=0; id<N; ++id) {

    _workers[id]._id = id;
    _workers[id]._vtm = id;
    _workers[id]._executor = this;
    _workers[id]._waiter = &_notifier._waiters[id];
    _workers[id]._thread = std::thread([&, &w=_workers[id]] () {

      pt::this_worker = &w;

      // initialize the random engine and seed for work-stealing loop
      w._rdgen.seed(static_cast<std::default_random_engine::result_type>(
        std::hash<std::thread::id>()(std::this_thread::get_id()))
      );

      // before entering the work-stealing loop, call the scheduler prologue
      if(_worker_interface) {
        _worker_interface->scheduler_prologue(w);
      }

      Node* t = nullptr;
      std::exception_ptr ptr = nullptr;

      // must use 1 as condition instead of !done because
      // the previous worker may stop while the following workers
      // are still preparing for entering the scheduling loop
      try {

        // worker loop
        while(1) {

          // drain out the local queue
          _exploit_task(w, t);

          // steal and wait for tasks
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

    });
  } 
}

// Function: _corun_until
template <typename P>
void Executor::_corun_until(Worker& w, P&& stop_predicate) {

  const size_t MAX_STEALS = ((num_queues() + 1) << 1);
    
  std::uniform_int_distribution<size_t> udist(0, num_queues()-1);
  
  exploit:

  while(!stop_predicate()) {

    if(auto t = w._wsq.pop(); t) {
      _invoke(w, t);
    }
    else {
      size_t num_steals = 0;
      size_t vtm = w._vtm;

      explore:
      
      //auto vtm = udist(w._rdgen);

      t = (vtm < _workers.size()) ? _workers[vtm]._wsq.steal() : 
                                    _buffers.steal(vtm - _workers.size());

      if(t) {
        _invoke(w, t);
        w._vtm = vtm;
        goto exploit;
      }
      else if(!stop_predicate()) {
        if(++num_steals > MAX_STEALS) {
          std::this_thread::yield();
        }
        vtm = udist(w._rdgen);
        goto explore;
      }
      else {
        break;
      }
    }
  }
}

// Function: _explore_task
inline bool Executor::_explore_task(Worker& w, Node*& t) {

  //assert(!t);
  
  const size_t MAX_STEALS = ((num_queues() + 1) << 1);
  std::uniform_int_distribution<size_t> udist(0, num_queues()-1);

  size_t num_steals = 0;
  size_t vtm = w._vtm;

  // Make the worker steal immediately from the assigned victim.
  while(true) {
    
    // Randomely generate a next victim.
    //vtm = udist(w._rdgen); //w._rdvtm();

    // If the worker's victim thread is within the worker pool, steal from the worker's queue.
    // Otherwise, steal from the buffer, adjusting the victim index based on the worker pool size.
    t = (vtm < _workers.size())
      ? _workers[vtm]._wsq.steal()
      : _buffers.steal(vtm - _workers.size());

    if(t) {
      w._vtm = vtm;
      break;
    }

    // Increment the steal count, and if it exceeds MAX_STEALS, yield the thread.
    // If the number of *consecutive* empty steals reaches MAX_STEALS, exit the loop.
    if (++num_steals > MAX_STEALS) {
      std::this_thread::yield();
      if(num_steals > 100 + MAX_STEALS) {
        break;
      }
    }

  #if __cplusplus >= TF_CPP20
    if(w._done.test(std::memory_order_relaxed)) {
  #else
    if(w._done.load(std::memory_order_relaxed)) {
  #endif
      return false;
    } 

    vtm = udist(w._rdgen); //w._rdvtm();
  } 
  return true;
}

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& w, Node*& t) {
  while(t) {
    _invoke(w, t);
    t = w._wsq.pop();
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& w, Node*& t) {

  explore_task:

  if(_explore_task(w, t) == false) {
    return false;
  }
  
  // Go exploit the task if we successfully steal one.
  if(t) {
    return true;
  }

  // Entering the 2PC guard as all queues should be empty after many stealing attempts.
  _notifier.prepare_wait(w._waiter);
  
  // Condition #1: buffers should be empty
  for(size_t vtm=0; vtm<_buffers.size(); ++vtm) {
    if(!_buffers._buckets[vtm].queue.empty()) {
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm + _workers.size();
      goto explore_task;
    }
  }
  
  // Condition #2: worker queues should be empty
  // Note: We need to use index-based looping to avoid data race with _spawan
  // which initializes other worker data structure at the same time
  for(size_t vtm=0; vtm<w._id; ++vtm) {
    if(!_workers[vtm]._wsq.empty()) {
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm;
      goto explore_task;
    }
  }
  
  // due to the property of the work-stealing queue, we don't need to check
  // the queue of this worker
  for(size_t vtm=w._id+1; vtm<_workers.size(); vtm++) {
    if(!_workers[vtm]._wsq.empty()) {
      _notifier.cancel_wait(w._waiter);
      w._vtm = vtm;
      goto explore_task;
    }
  }
  
  // Condition #3: worker should be alive
#if __cplusplus >= TF_CPP20
  if(w._done.test(std::memory_order_relaxed)) {
#else
  if(w._done.load(std::memory_order_relaxed)) {
#endif
    _notifier.cancel_wait(w._waiter);
    return false;
  }
  
  // Now I really need to relinquish myself to others.
  _notifier.commit_wait(w._waiter);
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
  
  // caller is a worker of this executor - starting at v3.5 we do not use
  // any complicated notification mechanism as the experimental result
  // has shown no significant advantage.
  if(worker._executor == this) {
    worker._wsq.push(node, [&](){ _buffers.push(node); });
    _notifier.notify_one();
    return;
  }
  
  // caller is not a worker of this executor - go through the centralized queue
  _buffers.push(node);
  _notifier.notify_one();
}

// Procedure: _schedule
inline void Executor::_schedule(Node* node) {
  _buffers.push(node);
  _notifier.notify_one();
}

// Procedure: _schedule
template <typename I>
void Executor::_schedule(Worker& worker, I first, I last) {

  size_t num_nodes = last - first;
  
  if(num_nodes == 0) {
    return;
  }
  
  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  if(worker._executor == this) {
    for(size_t i=0; i<num_nodes; i++) {
      auto node = detail::get_node_ptr(first[i]);
      worker._wsq.push(node, [&](){ _buffers.push(node); });
      _notifier.notify_one();
    }
    return;
  }
  
  // caller is not a worker of this executor - go through the centralized queue
  for(size_t i=0; i<num_nodes; i++) {
    _buffers.push(detail::get_node_ptr(first[i]));
  }
  _notifier.notify_n(num_nodes);
}

// Procedure: _schedule
template <typename I>
inline void Executor::_schedule(I first, I last) {
  
  size_t num_nodes = last - first;

  if(num_nodes == 0) {
    return;
  }

  // NOTE: We cannot use first/last in the for-loop (e.g., for(; first != last; ++first)).
  // This is because when a node v is inserted into the queue, v can run and finish 
  // immediately. If v is the last node in the graph, it will tear down the parent task vector
  // which cause the last ++first to fail. This problem is specific to MSVC which has a stricter
  // iterator implementation in std::vector than GCC/Clang.
  for(size_t i=0; i<num_nodes; i++) {
    _buffers.push(detail::get_node_ptr(first[i]));
  }
  _notifier.notify_n(num_nodes);
}
  
template <typename I>
void Executor::_schedule_graph_with_parent(Worker& worker, I beg, I end, Node* parent) {
  auto send = _set_up_graph(beg, end, parent->_topology, parent);
  parent->_join_counter.fetch_add(send - beg, std::memory_order_relaxed);
  _schedule(worker, beg, send);
}

TF_FORCE_INLINE void Executor::_update_cache(Worker& worker, Node*& cache, Node* node) {
  if(cache) {
    _schedule(worker, cache);
  }
  cache = node;
}
  
// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  #define TF_INVOKE_CONTINUATION()  \
  if (cache) {                      \
    node = cache;                   \
    goto begin_invoke;              \
  }

  begin_invoke:

  Node* cache {nullptr};
  
  // if this is the second invoke due to preemption, directly jump to invoke task
  if(node->_nstate & NSTATE::PREEMPTED) {
    goto invoke_task;
  }

  // if the work has been cancelled, there is no need to continue
  if(node->_is_cancelled()) {
    _tear_down_invoke(worker, node, cache);
    TF_INVOKE_CONTINUATION();
    return;
  }

  // if acquiring semaphore(s) exists, acquire them first
  if(node->_semaphores && !node->_semaphores->to_acquire.empty()) {
    SmallVector<Node*> waiters;
    if(!node->_acquire_all(waiters)) {
      _schedule(worker, waiters.begin(), waiters.end());
      return;
    }
  }
  
  invoke_task:
  
  SmallVector<int> conds;

  // switch is faster than nested if-else due to jump table
  switch(node->_handle.index()) {
    // static task
    case Node::STATIC:{
      _invoke_static_task(worker, node);
    }
    break;
    
    // runtime task
    case Node::RUNTIME:{
      if(_invoke_runtime_task(worker, node)) {
        return;
      }
    }
    break;

    // subflow task
    case Node::SUBFLOW: {
      if(_invoke_subflow_task(worker, node)) {
        return;
      }
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
      if(_invoke_module_task(worker, node)) {
        return;
      }
    }
    break;

    // async task
    case Node::ASYNC: {
      if(_invoke_async_task(worker, node)) {
        return;
      }
      _tear_down_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();
      return;
    }
    break;

    // dependent async task
    case Node::DEPENDENT_ASYNC: {
      if(_invoke_dependent_async_task(worker, node)) {
        return;
      }
      _tear_down_dependent_async(worker, node, cache);
      TF_INVOKE_CONTINUATION();
      return;
    }
    break;

    // monostate (placeholder)
    default:
    break;
  }

  // if releasing semaphores exist, release them
  if(node->_semaphores && !node->_semaphores->to_release.empty()) {
    SmallVector<Node*> waiters;
    node->_release_all(waiters);
    _schedule(worker, waiters.begin(), waiters.end());
  }

  // Reset the join counter with strong dependencies to support cycles.
  // + We must do this before scheduling the successors to avoid race
  //   condition on _predecessors.
  // + We must use fetch_add instead of direct assigning
  //   because the user-space call on "invoke" may explicitly schedule 
  //   this task again (e.g., pipeline) which can access the join_counter.
  node->_join_counter.fetch_add(
    node->num_predecessors() - (node->_nstate & ~NSTATE::MASK), std::memory_order_relaxed
  );

  // acquire the parent flow counter
  auto& join_counter = (node->_parent) ? node->_parent->_join_counter :
                       node->_topology->_join_counter;

  // Invoke the task based on the corresponding type
  switch(node->_handle.index()) {

    // condition and multi-condition tasks
    case Node::CONDITION:
    case Node::MULTI_CONDITION: {
      for(auto cond : conds) {
        if(cond >= 0 && static_cast<size_t>(cond) < node->_num_successors) {
          auto s = node->_edges[cond]; 
          // zeroing the join counter for invariant
          s->_join_counter.store(0, std::memory_order_relaxed);
          join_counter.fetch_add(1, std::memory_order_relaxed);
          _update_cache(worker, cache, s);
        }
      }
    }
    break;

    // non-condition task
    default: {
      for(size_t i=0; i<node->_num_successors; ++i) {
        if(auto s = node->_edges[i]; s->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          join_counter.fetch_add(1, std::memory_order_relaxed);
          _update_cache(worker, cache, s);
        }
      }
    }
    break;
  }
  
  // clean up the node after execution
  _tear_down_invoke(worker, node, cache);
  TF_INVOKE_CONTINUATION();
}

// Procedure: _tear_down_invoke
inline void Executor::_tear_down_invoke(Worker& worker, Node* node, Node*& cache) {
  
  // we must check parent first before subtracting the join counter,
  // or it can introduce data race
  if(auto parent = node->_parent; parent == nullptr) {
    if(node->_topology->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      _tear_down_topology(worker, node->_topology);
    }
  }
  else {  
    // needs to fetch every data before join counter becomes zero at which
    // the node may be deleted
    auto state = parent->_nstate;
    if(parent->_join_counter.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if(state & NSTATE::PREEMPTED) {
        _update_cache(worker, cache, parent);
      }
    }
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

// Procedure: _process_exception
inline void Executor::_process_exception(Worker&, Node* node) {

  constexpr static auto flag = ESTATE::EXCEPTION | ESTATE::CANCELLED;

  // find the anchor and mark the entire path with exception so recursive
  // or nested tasks can be cancelled properly
  // since exception can come from asynchronous task (with runtime), the node
  // itself can be anchored
  auto anchor = node;
  while(anchor && (anchor->_estate.load(std::memory_order_relaxed) & ESTATE::ANCHORED) == 0) {
    anchor->_estate.fetch_or(flag, std::memory_order_relaxed);
    anchor = anchor->_parent;
  }

  // the exception occurs under a blocking call (e.g., corun, join)
  if(anchor) {
    // multiple tasks may throw, and we only take the first thrown exception
    if((anchor->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::EXCEPTION) == 0) {
      anchor->_exception_ptr = std::current_exception();
      return;
    }
  }
  // otherwise, we simply store the exception in the topology and cancel it
  else if(auto tpg = node->_topology; tpg) {
    // multiple tasks may throw, and we only take the first thrown exception
    if((tpg->_estate.fetch_or(flag, std::memory_order_relaxed) & ESTATE::EXCEPTION) == 0) {
      tpg->_exception_ptr = std::current_exception();
      return;
    }
  }
  
  // for now, we simply store the exception in this node; this can happen in an 
  // execution that does not have any external control to capture the exception,
  // such as silent async task
  node->_exception_ptr = std::current_exception();
}

// Procedure: _invoke_static_task
inline void Executor::_invoke_static_task(Worker& worker, Node* node) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    std::get_if<Node::Static>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_subflow_task
inline bool Executor::_invoke_subflow_task(Worker& worker, Node* node) {
    
  auto& h = *std::get_if<Node::Subflow>(&node->_handle);
  auto& g = h.subgraph;

  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    
    // set up the subflow
    Subflow sf(*this, worker, node, g);

    // invoke the subflow callable
    _observer_prologue(worker, node);
    TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
      h.work(sf);
    });
    _observer_epilogue(worker, node);
    
    // spawn the subflow if it is joinable and its graph is non-empty
    // implicit join is faster than Subflow::join as it does not involve corun
    if(sf.joinable() && g.size()) {

      // signal the executor to preempt this node
      node->_nstate |= NSTATE::PREEMPTED;

      // set up and schedule the graph
      _schedule_graph_with_parent(worker, g.begin(), g.end(), node);
      return true;
    }
  }
  else {
    node->_nstate &= ~NSTATE::PREEMPTED;
  }

  // the subflow has finished or joined
  if((node->_nstate & NSTATE::RETAIN_SUBFLOW) == 0) {
    g.clear();
  }

  return false;
}

// Procedure: _invoke_condition_task
inline void Executor::_invoke_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    auto& work = std::get_if<Node::Condition>(&node->_handle)->work;
    conds = { work() };
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_multi_condition_task
inline void Executor::_invoke_multi_condition_task(
  Worker& worker, Node* node, SmallVector<int>& conds
) {
  _observer_prologue(worker, node);
  TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
    conds = std::get_if<Node::MultiCondition>(&node->_handle)->work();
  });
  _observer_epilogue(worker, node);
}

// Procedure: _invoke_module_task
inline bool Executor::_invoke_module_task(Worker& w, Node* node) {
  return _invoke_module_task_impl(w, node, std::get_if<Node::Module>(&node->_handle)->graph);  
}

// Procedure: _invoke_module_task_impl
inline bool Executor::_invoke_module_task_impl(Worker& w, Node* node, Graph& graph) {

  // No need to do anything for empty graph
  if(graph.empty()) {
    return false;
  }

  // first entry - not spawned yet
  if((node->_nstate & NSTATE::PREEMPTED) == 0) {
    // signal the executor to preempt this node
    node->_nstate |= NSTATE::PREEMPTED;
    _schedule_graph_with_parent(w, graph.begin(), graph.end(), node);
    return true;
  }

  // second entry - already spawned
  node->_nstate &= ~NSTATE::PREEMPTED;

  return false;
}


// Procedure: _invoke_async_task
inline bool Executor::_invoke_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::Async>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&)
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;
    
    // void(Runtime&, bool)
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }

  return false;
}

// Procedure: _invoke_dependent_async_task
inline bool Executor::_invoke_dependent_async_task(Worker& worker, Node* node) {
  auto& work = std::get_if<Node::DependentAsync>(&node->_handle)->work;
  switch(work.index()) {
    // void()
    case 0:
      _observer_prologue(worker, node);
      TF_EXECUTOR_EXCEPTION_HANDLER(worker, node, {
        std::get_if<0>(&work)->operator()();
      });
      _observer_epilogue(worker, node);
    break;
    
    // void(Runtime&) - silent async
    case 1:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<1>(&work))) {
        return true;
      }
    break;

    // void(Runtime&, bool) - async
    case 2:
      if(_invoke_runtime_task_impl(worker, node, *std::get_if<2>(&work))) {
        return true;
      }
    break;
  }
  return false;
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

  //// Need to check the empty under the lock since subflow task may
  //// define detached blocks that modify the taskflow at the same time
  //bool empty;
  //{
  //  std::lock_guard<std::mutex> lock(f._mutex);
  //  empty = f.empty();
  //}

  // No need to create a real topology but returns an dummy future
  if(f.empty() || p()) {
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
      _set_up_topology(pt::this_worker, t.get());
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

  static_assert(has_graph_v<T>, "target must define a member function 'Graph& graph()'");
  
  if(pt::this_worker == nullptr || pt::this_worker->_executor != this) {
    TF_THROW("corun must be called by a worker of the executor");
  }

  Node anchor;
  _corun_graph(*pt::this_worker, &anchor, target.graph().begin(), target.graph().end());
}

// Function: corun_until
template <typename P>
void Executor::corun_until(P&& predicate) {
  
  if(pt::this_worker == nullptr || pt::this_worker->_executor != this) {
    TF_THROW("corun_until must be called by a worker of the executor");
  }

  _corun_until(*pt::this_worker, std::forward<P>(predicate));
}

// Procedure: _corun_graph
template <typename I>
void Executor::_corun_graph(Worker& w, Node* p, I first, I last) {

  // empty graph
  if(first == last) {
    return;
  }
  
  // anchor this parent as the blocking point
  {
    AnchorGuard anchor(p);
    _schedule_graph_with_parent(w, first, last, p);
    _corun_until(w, [p] () -> bool { 
      return p->_join_counter.load(std::memory_order_acquire) == 0; }
    );
  }

  // rethrow the exception to the blocker
  p->_rethrow_exception();
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
#if __cplusplus >= TF_CPP20
  _num_topologies.fetch_add(1, std::memory_order_relaxed);
#else
  std::lock_guard<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
#endif
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
#if __cplusplus >= TF_CPP20
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
#if __cplusplus >= TF_CPP20
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
inline void Executor::_set_up_topology(Worker* w, Topology* tpg) {

  // ---- under taskflow lock ----
  auto& g = tpg->_taskflow._graph;
  
  auto send = _set_up_graph(g.begin(), g.end(), tpg, nullptr);
  tpg->_join_counter.store(send - g.begin(), std::memory_order_relaxed);

  w ? _schedule(*w, g.begin(), send) : _schedule(g.begin(), send);
}

// Function: _set_up_graph
template <typename I>
I Executor::_set_up_graph(I first, I last, Topology* tpg, Node* parent) {

  auto send = first;
  for(; first != last; ++first) {

    auto node = first->get();
    node->_topology = tpg;
    node->_parent = parent;
    node->_nstate = NSTATE::NONE;
    node->_estate.store(ESTATE::NONE, std::memory_order_relaxed);
    node->_set_up_join_counter();
    node->_exception_ptr = nullptr;

    // move source to the first partition
    // root, root, root, v1, v2, v3, v4, ...
    if(node->num_predecessors() == 0) {
      std::iter_swap(send++, first);
    }
  }
  return send;
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Worker& worker, Topology* tpg) {

  auto &f = tpg->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!tpg->_exception_ptr && !tpg->cancelled() && !tpg->_pred()) {
    //assert(tpg->_join_counter == 0);
    std::lock_guard<std::mutex> lock(f._mutex);
    _set_up_topology(&worker, tpg);
  }
  // case 2: the final run of this topology
  else {

    // invoke the callback after each run
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

      // decrement the topology
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

  if(!joinable()) {
    TF_THROW("subflow already joined");
  }
    
  _executor._corun_graph(_worker, _parent, _graph.begin(), _graph.end());
  
  // join here since corun graph may throw exception
  _parent->_nstate |= NSTATE::JOINED_SUBFLOW;
}

#endif




}  // end of namespace tf -----------------------------------------------------






