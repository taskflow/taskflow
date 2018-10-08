# Execute a Task Dependency Graph

After you create a task dependency graph,
you need to dispatch it for execution.
In this tutorial, we will show you how to execute a 
task dependency graph.

+ [Graph and Topology](#Graph-and-Topology)
+ [Blocking Execution](#Blocking-Execution)
+ [Non-blocking Execution](#Non-blocking-Execution)
+ [Wait on Topologies](#Wait-on-Topologies)
+ [Example 1: Multiple Dispatches](#Example-1-Multiple-Dispatches)

# Graph and Topology

Each taskflow object has exactly one graph at a time to represent
task dependencies constructed so far.
The graph exists until users dispatch it for execution.
We call a dispatched graph a *topology*.
Each taskflow object has a list of topologies to keep track of the 
execution status of dispatched graphs.
All tasks are executed in a shared thread pool.

# Blocking Execution

One way to dispatch the present task dependency graph
is to use the method `wait_for_all`.
Calling `wait_for_all` will dispatch the present graph to a shared thread storage
and block the program until all tasks finish.

```cpp
1: tf::Taskflow tf(4);
2:
3: auto A = tf.silent_emplace([] () { std::cout << "TaskA\n"; });
4: auto B = tf.silent_emplace([] () { std::cout << "TaskB\n"; });
5: A.precede(B);
6:
7: tf.wait_for_all();
```

When `wait_for_all` returns, all tasks including previously dispatched ones
are guaranteed to finish.
All topologies will be cleaned up as well.

# Non-blocking Execution

Another way to dispatch the present task dependency graph
is to use the method `dispatch` or `silent_dispatch`.
These two methods will dispatch the present graph to threads
and returns immediately without blocking the program.
Non-blocking methods allows the program to perform other computations
that can overlap with the execution of topologies.

```cpp
 1: tf::Taskflow tf(4);
 2:
 3: auto A = tf.silent_emplace([] () { std::cout << "TaskA\n"; });
 4: auto B = tf.silent_emplace([] () { std::cout << "TaskB\n"; });
 5: A.precede(B);
 6:
 7: auto F = tf.dispatch();
 8: // do some computation to overlap the execution of tasks A and B
 9: // ...
10: F.get();
```

Debrief:

+ Line 1-5 creates a graph with two tasks and one dependency
+ Line 7 dispatches this graph to a topology 
  and obtains a `std::future` to access its execution status
+ Line 8-9 performs some computations to overlap the execution of this topology
+ Line 10 blocks the program until this topology finishes

If you do not care the status of a dispatched graph, 
use the method `silent_dispatch`.
This method does not return anything.


```cpp
1: tf::Taskflow tf(4);
2:
3: auto A = tf.silent_emplace([] () { std::cout << "TaskA\n"; });
4: auto B = tf.silent_emplace([] () { std::cout << "TaskB\n"; });
5: A.precede(B);
6:
7: auto F = tf.dispatch();
8: // do some computation to overlap the execution of tasks A and B
9: // ...
```

# Wait on Topologies

When you call `dispatch` or `silent_dispatch`,
the taskflow object will dispatch the present graph to threads
and maintain a list of data structures called *topology* 
to store the execution status.
These topologies are not cleaned up automatically on completion.


```cpp
 1: tf::Taskflow tf(4);
 2: 
 3: auto A = tf.silent_emplace([] () { std::cout << "TaskA\n"; });
 4: auto B = tf.silent_emplace([] () { std::cout << "TaskB\n"; });
 5: A.precede(B);
 6:
 7: auto F = tf.dispatch();    // dispatch the present graph
 8:
 9: auto C = tf.silent_emplace([] () { std::cout << "TaskC\n"; });
10:
11: tf.silent_dispatch();      // dispatch the present graph
12:
13: tf.wait_for_topologies();  // block until the two graphs finish
14: 
15: assert(F.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
```

Debrief
+ Line 1 creates a taskflow object with four worker threads
+ Line 3-5 creates a dependency graph of two tasks and one dependency
+ Line 7 dispatches the present graph to threads and obtains a future object
  to access the execution status
+ Line 9 starts with a new dependency graph with one task
+ Line 11 dispatches the present graph to threads
+ Line 13 blocks the program until all running topologies finish

It's clear now Line 9 overlaps the execution of the first graph.
After Line 11, there are two topologies running inside the taskflow object.
Calling the method `wait_for_topologies` blocks the
program until both complete.

# Example 1: Multiple Dispatches

The example below demonstrates how to create multiple task dependency graphs and dispatch each of them asynchronously.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: std::atomic<int> counter {0};
 4:
 5: void create_graph(tf::Taskflow& tf) {
 6:   auto [A, B] = tf.silent_emplace(
 7:     [&] () { counter.fetch_add(1, std::memory_order_relaxed); },
 8:     [&] () { counter.fetch_add(1, std::memory_order_relaxed); }
 9:   );
10: }
11:
12: void multiple_dispatches() {
13:   tf::Taskflow tf(4);
14:   for(int i=0; i<10; ++i) {
15:     std::cout << "dispatch iteration " << i << std::endl;
16:     create_graph(tf);
17:     tf.silent_dispatch();
18:   }
19: }
20:
21: int main() {
22:
23:   multiple_dispatches();
24:   assert(counter == 20);
25:
26:   return 0;
27: }
```

Debrief:
+ Line 3 declares a global atomic variable initialized to zero
+ Line 5-10 defines a function that takes a taskflow object and creates two tasks to increment the counter
+ Line 12-19 defines a function that iteratively creates a task dependency graph and dispatches it asynchronously
+ Line 23 starts the procedure of multiple dispatches

Notice in Line 24 the counter ends up with 20. 
The destructor of a taskflow object will not leave until all running
topologies finish.



