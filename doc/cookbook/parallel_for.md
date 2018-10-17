# Parallelize a For Loop

In this tutorial, we are going to demonstrate how to use Cpp-Taskflow
to run a for loop in parallel.

+ [Range-based For Loop](#Range-based-For-Loop)
+ [Index-based For Loop](#Index-based-For-Loop)
+ [Example 1: Parallel Map](#Example-1-Parallel-Map)
+ [Example 2: Pipeline a Parallel For](#Example-2-Pipeline-a-Parallel-For)

# Range-based For Loop

Cpp-Taskflow has a STL-style method `parallel_for` 
that takes a range of items and applies a callable to each of the item in parallel.
The method constructs a sub-graph representing this workload
and returns a task pair as two synchronization points to this task pattern.

```cpp
 1:   tf::Taskflow tf(4);
 2:
 3:   std::vector<int> items {1, 2, 3, 4, 5, 6, 7, 8};
 4:
 5:   auto [S, T] = tf.parallel_for(items.begin(), items.end(), [] (int item) {
 6:     std::cout << std::this_thread::get_id() << " runs " << item << std::endl;
 7:   });
 8:
 9:   S.work([] () { std::cout << "S\n"; }).name("S");
10:   T.work([] () { std::cout << "T\n"; }).name("T");
11:
12:   std::cout << tf.dump();
13:
14:   tf.wait_for_all();
```

The above code generates the following task dependency graph. 
The label 0x56\* represents an internal task node to execute the callable object.
By default, Cpp-Taskflow evenly partitions and distributes the workload 
to all threads.
In our example of eight tasks and four workers, each internal node is responsible for two items.

![](parallel_for1.png)

Debrief:
+ Line 1 creates a taskflow object of four worker threads
+ Line 3 creates a vector container of eight items
+ Line 5-7 creates a parallel execution graph using the method `parallel_for`
+ Line 9-10 names the synchronization tasks `S` and `T`
+ Line 12 dumps the graph to a dot format which can be visualized through [GraphvizOnline][GraphVizOnline]
+ Line 14 dispatches the graph to execution and blocks until the graph finishes

Here is one possible output of this program:

```bash
S
139931471636224 runs 1
139931471636224 runs 2
139931480028928 runs 7
139931480028928 runs 8
139931496814336 runs 3
139931496814336 runs 4
139931488421632 runs 5
139931488421632 runs 6
T
```

## Partition the Workload Explicitly

By default, Cpp-Taskflow partitions the workload evenly across the workers.
In some cases, it is useful to disable this feature and apply user-specified partition.
The method `parallel_for` has an overload that takes an extra unsigned integer as 
the number of items in each partition.

```cpp
auto [S, T] = tf.parallel_for(items.begin(), items.end(), [] (int item) {
  std::cout << std::this_thread::get_id() << " runs " << item << std::endl;
}, 1);
```

The above example will force each partition to run exactly one item.
This can be useful when you have unbalanced workload
and would like to enable more efficient parallelization.

![](parallel_for2.png)

## Construct the Graph Explicitly

You can explicitly construct a dependency graph that represents a parallel execution 
of a for loop.
using only the basic methods `silent_emplace` and `precede`.


```cpp
auto S = tf.silent_emplace([] () {}).name("S");
auto T = tf.silent_emplace([] () {}).name("T");

for(auto item : items) {
  auto task = tf.silent_emplace([item] () {
    std::cout << std::this_thread::get_id() << " runs " << item << std::endl;
  });
  S.precede(task);
  task.precede(T);
}
```

# Index-based For Loop

To parallelize a for loop based on index, you can use the capture feature of C++ lambda.

```cpp
 1: auto S = tf.silent_emplace([] () {}).name("S");
 2: auto T = tf.silent_emplace([] () {}).name("T");
 3: 
 4: for(int i=0; i<8; ++i) {
 5:   auto task = tf.silent_emplace([i] () {
 6:     std::cout << std::this_thread::get_id() << " runs " << i << std::endl;
 7:   }); 
 8:   S.precede(task);
 9:   task.precede(T);
10: }
```

Debrief:
+ Line 1-2 creates two tasks, source and target, as the synchronization points
+ Line 4-10 creates eight parallel tasks that print out the executing thread's ID 
  and the iteration index
+ Line 8-9 adds one dependency link from the source to each task and
  one dependency link from each task to the target

---

# Example 1: Parallel Map

This example demonstrates how to use the method `parallel_for`
to create a parallel map pattern.
The map operator modifies each item in the container to one if it was an odd number,
or zero if it was an even number.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items{1, 2, 3, 4, 5, 6, 7, 8};
 8:
 9:   tf.parallel_for(items.begin(), items.end(), [] (int& item) {
10:     item = (item & 1) ? 1 : 0;
11:   });
12:
13:   tf.wait_for_all();
14:
15:   for(auto item : items) {
16:     std::cout << item << " ";
17:   }
18:
19:   return 0;
20: }
```

The program outputs the following:

```bash
1 0 1 0 1 0 1 0 
```

# Example 2: Pipeline a Parallel For

This example demonstrates how to pipeline a parallel-for workload
with other tasks.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<int> items(1024);
 8:   std::atomic<int> sum {0};
 9:
10:   auto T1 = tf.silent_emplace([&] () {  // create a modifier task
11:     for(auto& item : items) {
12:       item = 1;
13:     }
14:   }).name("Create Items");
15:
16:   auto [S, T] = tf.parallel_for(items.begin(), items.end(), [&] (int item) {
17:     sum.fetch_add(item, std::memory_order_relaxed);
18:   }, 128);
19:
20:   auto T2 = tf.silent_emplace([&] () {  // create a output task
21:     std::cout << "sum is: " << sum << std::endl;
22:   }).name("Print Sum");
23:
24:   T1.precede(S);  // modifier precedes parallel-for
25:   T.precede(T2);  // parallel-for precedes the output task
26:
27:   tf.wait_for_all();
28:
29:   return 0;
30: }
```

![](parallel_for_example2.png)

The output of this programs is:

```bash
sum is: 1024
```

Debrief:
+ Line 5 creates a taskflow object with four worker threads
+ Line 7 creates a vector of 1024 uninitialized integers
+ Line 8 creates an atomic integer variable
+ Line 10-14 creates a task that captures the vector to initialize all items to one
+ Line 16-18 sums up all items with each thread running on a partition of 128 items (total 1024/128=8 partitions)
+ Line 20-22 creates a task that outputs the summation value
+ Line 24-25 pipelines the parallel-for workload with the two tasks
+ Line 27 dispatches the graph to threads and blocks until the execution completes
 

 
 * * *
 
 [GraphViz]:              https://www.graphviz.org/
 [GraphVizOnline]:        https://dreampuf.github.io/GraphvizOnline/
 
 
 
 
 
 
 
 
 
 
 
 
 
