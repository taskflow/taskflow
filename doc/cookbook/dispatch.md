# Execute a Task Dependency Graph

After you create a task dependency graph,
you need to dispatch it to threads for execution
In this tutorial, we will show you how to execute a 
task dependency graph.

+ [Graph and Topology](#Graph-and-Topology)
+ [Blocking Execution](#Blocking-Execution)
+ [Non-blocking Execution](#Non-blocking-Execution)
+ [Wait on Topologies](#Wait-on-Topologies)
+ [Example 1: Multiple Dispatches](#Example-1-Multiple-Dispatches)
+ [Example 2: Connect Two Dependency Graphs](#Example-2-Connect-Two-Dependency-Graphs)

# Graph and Topology

Each taskflow object has exactly one graph at a time that represents
the tasks and the dependencies constructed so far.
The graph exists until users dispatch it for execution.
In Cpp-Taskflow, we call a dispatched graph a *topology*.
A topology is a data structure that wraps up a dispatched graph 
and stores a few metadata obtained at runtime.
Each taskflow object has a list of topologies to keep track of the 
execution status of dispatched graphs.
Users can retrieve this information later on for graph inspection and debugging.

All tasks are executed in a shared thread storage coupled with a
*task scheduler* to decide which thread runs which task.
Cpp-Taskflow provides two ways to dispatch a task dependency graph,
*blocking* and *non-blocking*.

# Blocking Execution

One way to dispatch the present task dependency graph
is to use the method `wait_for_all`.
Calling `wait_for_all` dispatches the graph to threads and blocks the program flow
until all tasks finish.

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
These two methods both dispatch the present graph to threads
and return immediately without blocking the program flow.
Non-blocking methods allow the program to perform other computations
that can overlap the graph execution.

```cpp
 1: tf::Taskflow tf(4);
 2:
 3: auto A = tf.silent_emplace([] () { std::cout << "Task A\n"; });
 4: auto B = tf.silent_emplace([] () { std::cout << "Task B\n"; });
 5: A.precede(B);
 6:
 7: auto F = tf.dispatch();
 8: // do some computation to overlap the execution of tasks A and B
 9: // ...
10: F.get();
```

Debrief:

+ Line 1-5 creates a graph with two tasks and one dependency
+ Line 7 dispatches this graph 
  and obtains a `std::future` object to access its execution status
+ Line 8-9 performs some computations to overlap the execution of task A and task B
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

Unlike `wait_for_all`, calling `dispatch` or `silent_dispatch`
will not clean up the topologies upon completion.
This allows users to dump the graph structure, in particular, created from dynamic tasking.
However, it may be necessary at some points of the program
to synchronize with the previously dispatched graphs.
Cpp-Taskflow provides a method `wait_for_topologies` for this purpose.


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
+ Line 7 dispatches this graph to threads and obtains a future object for users
  to access the execution status
+ Line 9 starts with a new dependency graph with one task
+ Line 11 dispatches the graph to threads
+ Line 13 blocks the program until both graphs finish

It is clear now Line 9 overlaps the execution of the first graph.
After Line 11, there are two topologies in the taskflow object.
Calling the method `wait_for_topologies` blocks the
program until both graph complete.

# Example 1: Multiple Dispatches

The example below demonstrates how to create multiple task dependency graphs and 
dispatch each of them asynchronously.

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

Notice in Line 24 the counter ends up being 20. 
By default, destructing a taskflow object will wait on all topologies to finish.

# Example 2: Connect Two Dependency Graphs

The example demonstrates how to use the `std::future` to explicitly impose a dependency 
link on two dispatched graphs.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items;   // uninitialized
 8:   int sum;                  // uninitialized
 9:
10:   // the first dependency graph
11:   // task C to resize the item vector
12:   auto A = tf.silent_emplace([&] () { items.resize(1024); });
13:  
14:   // task B to initialize the item vector
15:   auto B = tf.silent_emplace([&] () { std::iota(items.begin(), items.end(), 0); });
16:
17:   // A must run before B
18:   A.precede(B);
19:   
20:   // dispatch the graph asynchronously and obtain the future to access its status
21:   auto fu1 = tf.dispatch();
22:
23:   // the second dependency graph
24:   // task C to overlap the exeuction of the first graph
25:   auto C = tf.silent_emplace([&] () {
26:     sum = 0;  // in practice, this can be some expensive initializations
27:   });
28:
29:   // task D can't start until the first graph completes
30:   auto D = tf.silent_emplace([&] () {
31:     fu1.get();
32:     for(auto item : items) {
33:       sum += item;
34:     }
35:   });
36: 
37:   C.precede(D);
38: 
39:   auto fu2 = tf.dispatch();
40: 
41:   // wait on the second dependency graph to finish
42:   fu2.get();
43: 
44:   assert(sum == (0 + 1023) * 1024 / 2);
45: 
46:   return 0;
47: }
```

Debrief:
+ Line 5 creates a taskflow object with four worker threads
+ Line 7-8 creates a vector of integer items and an integer variable to store the summation value
+ Line 10-21 creates a dependency graph that resizes the vector and fills it with sequentially increasing values starting with zero
+ Line 23-39 creates another dependency graph that sums up the values in the vector
+ Line 25-27 creates a task that initializes the variable `sum` to zero, and overlaps its execution with the first dependency graph
+ Line 30-35 creates a task that blocks until the first dependency graph completes and then sums up all integer values in the properly initialized vector
+ Line 42 blocks until the second dependency graph finishes
+ Line 44 puts an assertion guard on the final summation value

By the time the second dependency graph finishes, 
the first dependency graph must have already finished due to Line 31.
The result of the variable `sum` ends up being the summation over 
the integer sequence `[0, 1, 2, ..., 1024)`.


