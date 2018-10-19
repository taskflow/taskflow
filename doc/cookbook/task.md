# Understand the Task

In this tutorial, we are going to demonstrate the basic construct of 
a task dependency graph - *Task*.

+ [Create a Task](#create-a-task)
+ [Access the Result of a Task](#access-the-result-of-a-task)
+ [Create Multiple Tasks at One Time](#create-multiple-tasks-at-one-time)
+ [Lifetime of a Task](#lifetime-of-a-task)
+ [Example 1: Create Multiple Dependency Graphs](#example-1-create-multiple-dependency-graphs)
+ [Example 2: Modify Task Attributes](#example-2-modify-task-attributes)

# What is a Task?

A task in Cpp-Taskflow is a callable object for which the operation [std::invoke][std::invoke] is applicable.
It can be either 
a functor, a lambda expression, a bind expression, or a class objects with `operator()` overloaded.

Cpp-Taskflow provides three methods, `placeholder`, `silent_emplace`, and `emplace`
to create a task.

```cpp
 1: auto A = tf.placeholder();           // create an empty task
 2: auto B = tf.silent_emplace([](){});  // create a task on top of a callable
 3: auto [C, FuC] = tf.emplace([](){ return 1; });  // create a accessible task on top of a callable
```

Debrief:
+ Line 1 creates an empty task 
+ Line 2 creates a task from a given callable object and returns a task handle
+ Line 3 creates a task from a given callable object and returns, in addition to a task handle,
  a [std::future][std::future] object for the program
  to access the return value when the task finishes

Each time you create a task, including an empty one, 
the taskflow object adds a node to the present graph
and returns a *task handle* of type `tf::Task`.
A task handle is a lightweight object
that wraps up a particular node in a graph
and provides a set of methods for you to assign different attributes to the task
such as adding dependencies, naming, and assigning a new work.


```cpp
 1: auto A = tf.silent_emplace([] () { std::cout << "create a task A\n"; });
 2: auto B = tf.silent_emplace([] () { std::cout << "create a task B\n"; });
 3:
 4: A.name("TaskA");
 5: A.work([] () { std::cout << "reassign A to a new task\n"; });
 6: A.precede(B);
 7:
 8: std::cout << A.name() << std::endl;            // print "TaskA"
 9: std::cout << A.num_successors() << std::endl;  // 1
10: std::cout << A.num_dependents() << std::endl;  // 0
11: 
12: std::cout << B.num_successors() << std::endl;  // 0
13: std::cout << B.num_dependents() << std::endl;  // 1
```

Debrief:
+ Line 1-2 creates two tasks A and B
+ Line 4-5 assigns names and work to task A, and add a precedence link to task B
+ Line 6 adds a dependency link from A to B
+ Line 8-13 dumps the task attributes 

Cpp-Taskflow uses the general-purpose polymorphic function wrapper
[std::function][std::function] to store and invoke any callable target in a task.
You will have to follow its contract to create a task.
For instance, the callable object must be copy constructible.

# Access the Result of a Task

Unlike `silent_emplace`, the method `emplace` returns a pair of a task handle 
and a `std::future` object to provide a mechanism to access the result when 
the associated task finishes.
This is particularly useful when you would like to pass data between tasks.
  
```cpp 
1: tf::Taskflow tf(4);
2: auto [A, FuA] = tf.emplace([](){ return 1; });
3: tf.wait_for_all();
4: std::cout << FuA.get() << std::endl;   // 1
```

You should be aware that every time you add a task or a dependency,
it creates only a node or an edge to the present graph.
The execution does not start until you dispatch the graph.
For example, the following code will block and never finish:

```cpp 
1: tf::Taskflow tf(4);
2: auto [A, FuA] = tf.emplace([](){ return 1; });
3: std::cout << FuA.get() << std::endl;   // block
4: tf.wait_for_all();                     // never enter this line
```

# Create Multiple Tasks at One Time

Cpp-Taskflow uses C++ structured binding coupled with tuple
to make the creation of tasks simple.
Both `silent_emplace` and `emplace` can accept arbitrary numbers of callable objects
and create multiple tasks at one time.

```cpp
1: auto [A, B, C] = tf.silent_emplace(  // create three tasks in one call
2:   [](){ std::cout << "Task A\n"; },
3:   [](){ std::cout << "Task B\n"; },
4:   [](){ std::cout << "Task C\n"; }
5: );
```

# Lifetime of A Task

A task lives with its graph, and is not destroyed until its parent graph
gets cleaned up.
A task belongs to only a graph at a time.
The lifetime of a task mostly refers to the user-given callable object,
including captured values.
As long as the graph is alive,
all the associated tasks remain their existence.


---

# Example 1: Create Multiple Dependency Graphs

The example below demonstrates how to reuse task handles to create two 
task dependency graphs.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4: 
 5:   tf::Taskflow tf(4);
 6:
 7:   // create a task dependency graph
 8:   std::array<tf::Task, 4> tasks {
 9:     tf.silent_emplace([] () { std::cout << "Task A\n"; }),
10:     tf.silent_emplace([] () { std::cout << "Task B\n"; }),
11:     tf.silent_emplace([] () { std::cout << "Task C\n"; }),
12:     tf.silent_emplace([] () { std::cout << "Task D\n"; })
13:   };
14:
15:   tasks[0].precede(tasks[1]); 
16:   tasks[0].precede(tasks[2]);
17:   tasks[1].precede(tasks[3]);
18:   tasks[2].precede(tasks[3]);
19:
20:   tf.wait_for_all();
21:
22:   // create another task dependency graph
23:   tasks = {
24:     tf.silent_emplace([] () { std::cout << "New Task A\n"; }),
25:     tf.silent_emplace([] () { std::cout << "New Task B\n"; }),
26:     tf.silent_emplace([] () { std::cout << "New Task C\n"; }),
27:     tf.silent_emplace([] () { std::cout << "New Task D\n"; })
28:   };
29:
30:   tasks[3].precede(tasks[2]);
31:   tasks[2].precede(tasks[1]);
32:   tasks[1].precede(tasks[0]);
33:
34:   tf.wait_for_all();
35:
36:   return 0;
37: }
```

Debrief:
+ Line 5 creates a taskflow object of four worker threads
+ Line 8 creates a task array to store four task handles
+ Line 9-12 creates four tasks
+ Line 15-18 adds four task dependency links
+ Line 20 dispatches the graph and blocks until it completes
+ Line 23-28 creates four new tasks and reassigns the task array to these four tasks
+ Line 30-32 adds a linear dependency to these four tasks
+ Line 34 dispatches the graph and blocks until it completes

Notice that trying to modify a task in a dispatched graph 
results in undefined behavior.
For examples, starting from Line 21, you should not modify any tasks 
but assign them to new targets (Line 23-28).

# Example 2: Modify Task Attributes

This example demonstrates how to modify a task's attributes using methods defined in
the task handler.

```cpp
 1: #include <taskflow/taskflow.hpp>
 2:
 3: int main() {
 4:
 5:   tf::Taskflow tf(4);
 6:
 7:   std::vector<tf::Task> tasks = { 
 8:     tf.placeholder(),         // create a task with no work
 9:     tf.placeholder()          // create a task with no work
10:   };
11:
12:   tasks[0].name("This is Task 0");
13:   tasks[1].name("This is Task 1");
14:   tasks[0].precede(tasks[1]);
15:
16:   for(auto task : tasks) {    // print out each task's attributes
17:     std::cout << task.name() << ": "
18:               << "num_dependents=" << task.num_dependents() << ", "
19:               << "num_successors=" << task.num_successors() << '\n';
20:   }
21:
22:   tf.dump(std::cout);         // dump the taskflow graph
23:
24:   tasks[0].work([](){ std::cout << "got a new work!\n"; });
25:   tasks[1].work([](){ std::cout << "got a new work!\n"; });
26:
27:   tf.wait_for_all();
28:
29:   return 0;
30: }
```

The output of this program looks like the following:

```bash
This is Task 0: num_dependents=0, num_successors=1
This is Task 1: num_dependents=1, num_successors=0
digraph Taskflow {
"This is Task 1";
"This is Task 0";
"This is Task 0" -> "This is Task 1";
}
got a new work!
got a new work!
```

Debrief:
+ Line 5 creates a taskflow object of four worker threads
+ Line 7-10 creates two tasks with empty target and stores the corresponding task handles
  in a vector
+ Line 12-13 names the two tasks with human-readable strings 
+ Line 14 adds a dependency link from the first task to the second task
+ Line 16-20 prints out the name of each task, the number of dependents, 
  and the number of successors
+ Line 22 dumps the task dependency graph to a [GraphViz][GraphViz] format (dot)
+ Line 24-25 assigns a new target to each task
+ Line 27 dispatches the graph and blocks until the execution finishes

You can change the name and work of a task at anytime before dispatching the graph.
The later assignment overwrites the previous values.
Only the latest information will be used.

* * *

[GraphViz]:              https://www.graphviz.org/
[GraphVizOnline]:        https://dreampuf.github.io/GraphvizOnline/
[std::function]:         https://en.cppreference.com/w/cpp/utility/functional/function
[std::invoke]:           https://en.cppreference.com/w/cpp/utility/functional/invoke
[std::future]:           https://en.cppreference.com/w/cpp/thread/future

