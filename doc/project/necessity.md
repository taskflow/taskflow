# Necessity of a New Tasking Library

We discuss in this page why we decide to develop a new tasking library,
rather than sticking with existing frameworks.
We will show the pros and cons of existing tools and compare 
their programming models with Cpp-Taskflow. 

+ [OpenMP Task Dependency Clause](#openmp-task-dependency-clause)
+ [Intel TBB Flow Graph](#intel-tbb-flow-graph)
+ [Summary](#summary)

# OpenMP Task Dependency Clause

In 2015, [OpenMP 4.5][OpenMP 4.5] introduced 
the task group and dependency clause `depend(type : list)`.
While being daunting at first glance,
they are powerful in extracting task dependencies from a `#pragma omp` block
to form a *static dependency graph*.
However, this paradigm enforces us to
have a good understanding about the graph structure and topological constraints.
Users need to explicitly annotate the dependency constraints on the code
through compiler directives `#pragma omp task depend(...)`.
Let's take a look at the following task dependency graph of seven tasks and eight dependencies.

![](static_programmability.png)

The following block shows the implementation of this task dependency graph
using OpenMP task constructs and dependency clauses,
and compares it with Cpp-Taskflow.

<table>
<tr>
  <th>Cpp-Taskflow</th>
  <th>OpenMP</th>
</tr>

<tr>
<td>
<pre>
tf::Taskflow tf;

auto [a0, a1, a2, a3, b0, b1, b2]
  = tf.silent_emplace(
  [] () { std::cout << "a0\n"; },
  [] () { std::cout << "a1\n"; },
  [] () { std::cout << "a2\n"; },
  [] () { std::cout << "a3\n"; },
  [] () { std::cout << "b0\n"; },
  [] () { std::cout << "b1\n"; },
  [] () { std::cout << "b2\n"; },
);

a0.precede(a1);
a1.precede(a2)
a1.precede(b2);
a2.precede(a3);
  
b0.precede(b1);
b1.precede(a2);
b1.precede(b2);
b2.precede(a3);

tf.wait_for_all();
</pre>
</td>

<td>
<pre>
#pragma omp parallel
{
#pragma omp single
{
  int a0_a1, a1_a2;
  int b0_b1, b1_b2;
  int a1_b2, b1_a2;
  int a2_a3, b2_a3;
 
  #pragma omp task depend(out:a0_a1)
  std::cout << "a0\n";
 
  #pragma omp task depend(out:b0_b1)
  std::cout << "b0\n";
 
  #pragma omp task depend(in:a0_a1) depend(out:a1_a2, a1_b2)
  std::cout << "a1\n";
 
  #pragma omp task depend(in:b0_b1) depend(out:b1_b2, b1_a2)
  std::cout << "b1\n";
 
  #pragma omp task depend(in:a1_a2, b1_a2) depend(out:a2_a3)
  std::cout << "a2\n";
 
  #pragma omp task depend(in:a1_b2, b1_b2) depend(out:b2_a3)
  std::cout << "b2\n";
 
  #pragma omp task depend(in:a2_a3, b2_a3)
  std::cout << "a3\n";

}  // end of omp single
}  // end of omp parallel
</pre>
</td>
</tr>

</table>

OpenMP requires programmers to explicitly specify the dependency clause 
for both sides of a constraint.
These must be done at compile time, making the graph description inflexible and not general.
Also, it is users' responsibility to identify a correct topological order
to describe each task such that it is consistent with the sequential program flow.
For example, the `#pragma task` block for `a1` cannot go above `a0`.
Otherwise, the program can produce unexpected results
when the compiler disables OpenMP to fall back to sequential execution.


# Intel TBB Flow Graph

In 2007, Intel released the [Threading Building Blocks (TBB)][Intel TBB] library
that supports loop-level parallelism and task-based programming.
The TBB task model is object-oriented.
It supports a variety of methods to create a task dependency graph
and schedule the task execution.
Because of various supports,
the TBB graph description language is very complex.
It often results in a lot of source lines of code (SLOC) 
that are difficult to read and debug.

![](static_programmability.png)

The following block shows the implementation of the task dependency graph
(see above) using Intel TBB's [Flow Graph][TBB Flow Graph] class,
and compares it with Cpp-Taskflow.

<table>
<tr>
  <th>Cpp-Taskflow</th>
  <th>Intel TBB</th>
</tr>

<tr>
<td>
<pre>
tf::Taskflow tf;

auto [a0, a1, a2, a3, b0, b1, b2]
  = tf.silent_emplace(
  [] () { std::cout << "a0\n"; },
  [] () { std::cout << "a1\n"; },
  [] () { std::cout << "a2\n"; },
  [] () { std::cout << "a3\n"; },
  [] () { std::cout << "b0\n"; },
  [] () { std::cout << "b1\n"; },
  [] () { std::cout << "b2\n"; },
);

a0.precede(a1);
a1.precede(a2)
a1.precede(b2);
a2.precede(a3);
  
b0.precede(b1);
b1.precede(a2);
b1.precede(b2);
b2.precede(a3);

tf.wait_for_all();
</pre>
</td>

<td>
<pre>
using namespace tbb;
using namespace tbb::flow;

int n = task_scheduler_init::default_num_threads();
task_scheduler_init init(n);

graph g;

continue_node<continue_msg> a0(g, [] (const continue_msg &) { 
  std::cout << "a0\n";
});

continue_node<continue_msg> a1(g, [] (const continue_msg &) { 
  std::cout << "a1\n"; 
});

continue_node<continue_msg> a2(g, [] (const continue_msg &) { 
  std::cout << "a2\n"; 
});

continue_node<continue_msg> a3(g, [] (const continue_msg &) { 
  std::cout << "a3\n"; 
});

continue_node<continue_msg> b0(g, [] (const continue_msg &) { 
  std::cout << "b0\n"; 
});

continue_node<continue_msg> b1(g, [] (const continue_msg &) { 
  std::cout << "b1\n"; 
});

continue_node<continue_msg> b2(g, [] (const continue_msg &) { 
  std::cout << "b2\n"; 
});

make_edge(a0, a1);
make_edge(a1, a2);
make_edge(a1, b2);
make_edge(a2, a3);

make_edge(b0, b1);
make_edge(b1, b2);
make_edge(b1, a2);
make_edge(b2, a3);

a0.try_put(continue_msg());
b0.try_put(continue_msg());

g.wait_for_all();
</pre>
</td>
</tr>

</table>

From our perspective, the TBB-based implementation is quite verbose.
Programmers need to understand the template class `continue_node`
and the role of the message class `continue_msg` before starting with 
a simple task dependency graph.
Also we need to explicitly tell TBB the *source* tasks
and call the method `try_put` to either enable a nominal message
or an actual data input to run the graph.
All these add up to extra programming effort and increase the possibility
of writing buggy code.

# Summary

The library programmability is the main incentive for us to develop a new tasking library.
Many existing libraries require users to write a lot of redundant or boilerplate code to implement
even a simple parallel task decomposition strategy.
This not only imposes burden on developers 
but increases the chance of buggy implementations.
Due to the long history of parallel programming,
the design of many existing C/C++-based libraries stayed rooted 
in old-fashioned standards,
not allowing developers to utilize modern C++17/20 idioms 
to enhance functionality and performance that were previously not possible.
Our focus is to leverage the power of modern C++
to enable efficient implementations of
both loop-based and general task-based parallel algorithms.


* * *


[OpenMP 4.5]:        https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf
[Intel TBB]:         https://www.threadingbuildingblocks.org/
[TBB Flow Graph]:    https://www.threadingbuildingblocks.org/tutorial-intel-tbb-flow-graph
