# Understand the Task Construct

In this tutorial, we are going to demonstrate the basic construct of 
a task dependency graph - *Task*.

+ [Create a Task](#Create-a-Task)
+ [Obtain a Result of a Task](#Obtain-a-Result-of-a-Task)
+ [Create Multiple Tasks at One Time](#Create-Multiple-Tasks-at-One-Time)

# Create a Task

Cpp-Taskflow provides three methods, `placeholder`, `silent_emplace`, and `emplace`
to create a task.

```cpp
 1: auto A = tf.placeholder();           // create an empty task
 2: auto B = tf.silent_emplace([](){});  // create a task on top of a callable
 3: auto [C, FuC] = tf.emplace([](){});  // create a accessible task on top of a callable
```

Debrief:
+ Line 1 creates an empty task that will execute nothing
+ Line 2 creates a task to execute a given callable 
+ Line 3 creates a task to execute a given callable with result accessible through a 
  `std::future` object


Each time you creates a task, the taskflow object will create a new node binding to 
the given callable and return to you a `tf::Task` handle that wraps around the internal node.
A task handler is a lightweight object that provides a set of methods
for you to assign different attributes such as
adding dependencies, naming, and assigning work.

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
+ Line 4-5 assigns attributes to task A
+ Line 6 adds a dependency link from A to B
+ Line 8-13 dumps the task attributes 

Cpp-Taskflow uses `std::function` to store the callable of each task.
You will have to follow the contract defined by [std::function][std::function].
For instance, the callable must be copy constructible.

# Obtain the Result of a Task

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
one node or one edge will be created and included to the present task dependency graph.
The execution does not start until you dispatch the graph.
For example, the following code will block and never finish:

```cpp 
1: tf::Taskflow tf(4);
2: auto [A, FuA] = tf.emplace([](){ return 1; });
3: std::cout << FuA.get() << std::endl;   // block
4: tf.wait_for_all();                     // never enter this line
```

# Create Multiple Tasks at One Time

Cpp-Taskflow leverages the power of modern C++ to make its API easier to use.
You can create a batch of tasks within a single call
and apply structured binding to capture the return values.

```cpp
1: auto [A, B, C] = tf.silent_emplace(  // create three tasks in one call
2:   [](){ std::cout << "Task A\n"; },
3:   [](){ std::cout << "Task B\n"; },
4:   [](){ std::cout << "Task C\n"; }
5: );
```

* * *

[GraphViz]:              https://www.graphviz.org/
[GraphVizOnline]:        https://dreampuf.github.io/GraphvizOnline/
[std::function]:         https://en.cppreference.com/w/cpp/utility/functional/function



