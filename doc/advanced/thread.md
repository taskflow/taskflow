# A1: Thread Management and Task Execution

We discuss in this tutorial the thread management and task execution schemes
of Cpp-Taskflow.
We first define the *thread* and *ownership* in Cpp-Taskflow.
Then we introduce the *executor* and demonstrate 
how to customize it for your taskflow object.

+ [Master, Workers, and Executor](#threads-workers-and-executor)
+ [Share an Executor among Taskflow Objects](#share-the-executor-among-taskflow-objects)
+ [Thread Safety](#thread-safety)

# Master, Workers, and Executor

Cpp-Taskflow defines a strict relationship between a *master* and *workers*.
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
That is, using one worker and zero worker are conceptually similar to each other
since they both end up using one thread to run all tasks (see the snippet below).

```cpp
tf::Taskflow tf1(0);    // one master, zero worker (master to run tasks)
tf::Taskflow tf2(1);    // one master, one worker (one thread to run tasks)
```

In most cases, the master thread is exposed to users at programming time,
while the worker threads are transparently maintained by the taskflow object.
The only exception happens when users declare a taskflow object 
with a customized executor interface
that implements a different thread management scheme.

Each taskflow object owns an *executor* managed by [std::shared_ptr][std::shared_ptr].
The default executor implements a thread pool coupled with a speculative execution strategy
to efficiently carry out tasks.
The `tf::Taskflow` class defines a member type `Executor` as an alias of the associated executor type.
We allow users to acquire an ownership of the executor of a taskflow object
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

```cpp
// create 100 taskflow objects each with its own executor
std::list<tf::Taskflow> tfs;

for(size_t i=0; i<100; ++i) {
  auto& tf = tfs.emplace_back(4);  // create a taskflow object with four threads
  assert(tf.share_executor().use_count() == 2);  // by the taskflow and the caller
}
// a total of 1 + 4*100 = 401 threads running in this program
```

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


# Thread Safety

The Taskflow object is ***NOT*** thread-safe.
Touching a taskflow object from multiple threads can result in *undefined behavior*.
The thread safety has nothing to do with the master nor the workers.
It is completely safe to access the taskflow object as long as only one thread presents at a time.
We recommend users to acknowledge the concept of the master and the workers,
and separate the program control flow accordingly.
Having a clear thread ownership
can greatly reduce the chance of buggy implementations
and undefined behaviors.


* * *

[std::thread::hardware_concurrency]: https://en.cppreference.com/w/cpp/thread/thread/hardware_concurrency
[std::thread::id]: https://en.cppreference.com/w/cpp/thread/thread/id
[std::shared_ptr]: https://en.cppreference.com/w/cpp/memory/shared_ptr


