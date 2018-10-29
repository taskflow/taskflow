# A1: Thread Management and Task Execution

We discuss in this tutorial the thread management and task execution schemes
of Cpp-Taskflow.
We first define the *thread* and *ownership* in Cpp-Taskflow.
Then we introduce the *executor* and demonstrate 
how to customize it for your taskflow object.

+ [Master, Workers, and Executor](#master-workers-and-executor)
+ [Share an Executor among Taskflow Objects](#share-an-executor-among-taskflow-objects)
+ [Customize Your Executor Interface](#customize-your-executor-interface)
+ [Thread Safety](#thread-safety)
+ [Example 1: Impact of Over-subscription](#example-1-impact-of-over-subscription)

# Master, Workers, and Executor

Cpp-Taskflow defines a strict relationship between the *master* and *workers*.
Master is the thread that creates the taskflow object
and 
workers are threads that invoke the callable objects of tasks.
Each taskflow object owns an *executor* instance that implements the execution of a task, 
for example, by a thread in a shared pool.
By default, Cpp-Taskflow uses [std::thread::hardware_concurrency][std::thread::hardware_concurrency]
to decide the number of worker threads
and [std::thread::id][std::thread::id] to identify the ownership between 
the master and workers.

```cpp
std::cout << std::thread::hardware_concurrency() << std::endl;  // 8, for example 
std::cout << std::thread::get_id() << std::endl;                // master thread id

tf::Taskflow tf1;     // create a taskflow object with the default number of workers
tf::Taskflow tf2{4};  // create a taskflow object with four workers
```

In the above example,
the master thread owns both taskflow objects.
The first taskflow object `tf1` creates eight 
(default by `std::thread::hardware_concurrency`) 
worker threads and 
the second taskflow object `tf2` creates four worker threads.
Including the master thread, there will be a total of 1 + 8 + 4 = 13 threads running in this program.
If you create a taskflow with zero workers, the master will carry out all the tasks by itself.
That is, using one worker and zero worker are conceptually equivalent to each other
since they both end up using one thread to run all tasks (see the snippet below).

```cpp
tf::Taskflow tf1(0);    // one master, zero worker (master to run tasks)
tf::Taskflow tf2(1);    // one master, one worker (one thread to run tasks)
```

In general, the master thread is exposed to users at programming time (main thread),
while the worker threads are transparently maintained by the taskflow object.
Each taskflow object owns an *executor* managed by [std::shared_ptr][std::shared_ptr].
The default executor implements a thread pool coupled with a speculative execution strategy
to efficiently carry out tasks.
The `tf::Taskflow` class defines a member type `Executor` as an alias of the associated executor type.
Users can acquire an ownership of the executor from a taskflow object
through the method `share_executor`.

```cpp
tf::Taskflow tf;
std::shared_ptr<tf::Taskflow::Executor> ptr = tf.share_executor();  // share the executor
assert(ptr.use_count() == 2);  // the executor is owned by ptr and tf
```

The *shared* property allows users to create their own resource manager and construct
a taskflow object on top.
The executor has only one constructor that takes an unsigned integer indicating the number
of worker threads to spawn.

```cpp
auto ptr = std::make_shared<tf::Taskflow::Executor>(4);  // create an executor of 4 workers
tf::Taskflow tf(ptr);          // create a taskflow object on top of the executor
assert(ptr.use_count() == 2);  // the executor is owned by ptr and tf
```


# Share an Executor among Taskflow Objects

It is sometime useful to share one executor among multiple taskflow objects
in order to avoid the thread *over-subscription* problem.
In the case of over-subscription, 
the number of threads running in a program exceeds the number of available logical cores,
resulting in additional and unnecessary context switches.
Context switch has nonzero cost and is especially costly when it crosses cores.
The following example mimics the over-subscription problem
through a creation of 100 taskflow objects each with 
its own executor of four threads, assuming only four logical cores present in the machine.


```cpp
// create 100 taskflow objects each with its own executor
std::list<tf::Taskflow> tfs;

for(size_t i=0; i<100; ++i) {
  auto& tf = tfs.emplace_back(4);  // create a taskflow object with four threads
  assert(tf.share_executor().use_count() == 2);  // by the taskflow and the caller
}
// a total of 1 + 4*100 = 401 threads running in this program
```

Over-subscription can cause significant performance issues,
especially in *compute-intensive* applications.
To avoid this, you can create a taskflow object from a given executor
such that multiple taskflow objects run on the same set of threads.
The following example demonstrates how to use Cpp-Taskflow's default executor
to create a shared thread pool for 100 taskflow objects,
assuming only four logical cores present in the machine.

```cpp
// create 100 taskflow objects on top of the same executor
std::list<tf::Taskflow> tfs;

auto executor = std::make_shared<tf::Taskflow::Executor>(4);

for(size_t i=0; i<100; ++i) {
  assert(executor.use_count() == i + 1);  // by the executor and each taskflow
  tfs.emplace_back(executor);  // create a taskflow object from the executor
}
// a total of 1 + 4 = 5 threads running in this program
```

# Customize Your Executor Interface

Cpp-Taskflow permits users to define their own executor interface and
integrate it into the taskflow object being built.
In most cases, the executor is implemented as a thread pool to run given tasks.
Your executor class must obey the following concepts in order to work with Cpp-Taskflow:

```cpp
template <typename C>
class MyExecutor {               // closure type C, callable on operator ()
  
  public:
    
  MyExecutor(unsigned);          // constructor on a number of worker threads (might be zero)

  size_t num_workers() const;    // return the number of worker threads (might be zero)

  template <typename... ArgsT>
  void emplace(ArgsT&&...);      // arguments to construct the closure C
};

using MyTaskflow = tf::BasicTaskflow<MyExecutor>;
```

The executor class template with one parameter on the task type.
The task type can be a generic polymorphic function wrapper like `std::function<void()>`
or a callable class with fixed memory layout.
It is completely up to users to define how to invoke the task.
Your executor class must meet the following concepts:

+ a constructor on a given number of worker threads
+ a constant method `num_workers` to return the number of worker threads
+ a method `emplace` to dispatch a task to thread with arguments to forward to the constructor of the task

Cpp-Taskflow imposes little requirements on the executor class.
Each taskflow object has its own internal data structure to keep track of the lifetime
and execution status of a task.
The executor only needs to guarantee a thread to run the task given by the method `emplace`.
We recommend users to read our built-in executor implementation
[SimpleThreadpool](../../taskflow/threadpool/simple_threadpool.hpp)
for more details.


# Thread Safety

The Taskflow object is ***NOT*** thread-safe.
Touching a taskflow object from multiple threads can result in *undefined behavior*.
The thread safety has nothing to do with the master nor the workers.
It is completely safe to access the taskflow object as long as only one thread presents at a time.
However, we strongly recommend users to acknowledge the definition of the master and the workers,
and separate the program control flow accordingly.
Having a clear thread ownership
can greatly reduce the chance of buggy implementations
and undefined behaviors.

# Example 1: Impact of Over-subscription

The example below shows a portion of the code of [executor.cpp](../../example/executor.cpp)
to demonstrate the impact of thread over-subscription.
The workload is a task dependency graph of four tasks doing compute-intensive 
matrix multiplication.
We benchmarked the performance between the two implementations
with and without sharing an executor.


```cpp
 1: void create_task_dependency_graph(tf::Taskflow& tf) {
 2:   for(size_t i=0; i<MAX_COUNT; ++i) {
 3:     auto [A, B, C, D] = tf.silent_emplace(
 4:       [&] () { matrix_multiplication(); },
 5:       [&] () { matrix_multiplication(); },
 6:       [&] () { matrix_multiplication(); },
 7:       [&] () { matrix_multiplication(); }
 8:     );
 9:     A.precede(B);
10:     A.precede(C);
11:     C.precede(D);
12:     B.precede(D);
13:   }
14: } 
15: 
16: // create multiple executors each with its own executor
17: auto unique_executor() {
18: 
19:   auto beg = std::chrono::high_resolution_clock::now();
20:     
21:   std::list<tf::Taskflow> tfs;
22: 
23:   for(size_t i=0; i<MAX_TASKFLOW; ++i) {
24:     auto& tf = tfs.emplace_back(MAX_THREAD);
25:     create_task_dependency_graph(tf);
26:     assert(tf.share_executor().use_count() == 2);
27:   }
28:     
29:   std::vector<std::shared_future<void>> futures;
30:   for(auto& tf : tfs) {
31:     futures.emplace_back(tf.dispatch());
32:   }   
33:     
34:   for(auto& fu : futures) {
35:     fu.get();
36:   } 
37:     
38:   auto end = std::chrono::high_resolution_clock::now();
39:   return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
40: }
41:
42: // create multiple executors on top of the same executor
43: auto shared_executor() {
44:   
45:   auto beg = std::chrono::high_resolution_clock::now();
46:   
47:   std::list<tf::Taskflow> tfs;
48:   
49:   auto executor = std::make_shared<tf::Taskflow::Executor>(MAX_THREAD);
50:     
51:   for(size_t i=0; i<MAX_TASKFLOW; ++i) {
52:     assert(executor.use_count() == i + 1);
53:     auto& tf = tfs.emplace_back(executor);
54:     create_task_dependency_graph(tf);
55:   }
56:     
57:   std::vector<std::shared_future<void>> futures;
58:   for(auto& tf : tfs) {
59:     futures.emplace_back(tf.dispatch());
60:   } 
61:   
62:   for(auto& fu : futures) {
63:     fu.get();
64:   }
65:  
66:   auto end = std::chrono::high_resolution_clock::now();
67:   return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
68: }
```

Debrief:

+ Line 1-14 creates a task dependency graph composed of compute-intensive matrix operations
+ Line 16-40 creates multiple independent taskflow objects without sharing the running threads
+ Line 42-68 creates multiple taskflow objects from the same executor to run on the same set of threads

Running the program on different number of taskflow objects gives
the following runtime values:

```bash
# taskflows shared (ms) unique (ms)
          1         120         114
          2         225         229
          4         451         452
          8         908         904
         16        1791        1837
         32        3581        3782
         64        7183        7636
        128       14341       15482
```

As we increase the number of taskflow objects, the implementation without sharing the executor
encounters more context switches among threads.
This overhead reflected on the slower runtime (15482 vs 14341 on 128 taskflow objects).


* * *

[std::thread::hardware_concurrency]: https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency
[std::thread::id]: https://en.cppreference.com/w/cpp/thread/thread/id
[std::shared_ptr]: https://en.cppreference.com/w/cpp/memory/shared_ptr


