# Level Graph Traversal

We demonstrate in this page how to implement a level graph traversal
using Cpp-Taskflow. We define a `NxM` level graph as a directed graph which contains `N` levels where each level has `M` nodes. Each node in level `i` has `1-4` outgoing edges to random nodes in level `i+1`.  
An example of a `3x4` level graph is shown below. The tuple `(x, y)` represents the node with the level index `x` and level specific node index `y`.

![](levelgraph_sample.png)

The goal is to traverse all the nodes in the graph following the edge dependencies. A node can be traversed only if its parents were traversed before. Note that order of traversal can be asynchronous to level index since a node in a higher level with less constraints could be traversed before a node in a lower level with more constraints. For instance, node (2,3) could be traversed before node (1.0) given the graph structure above.

We compare with three parallel implementations with Cpp-Taskflow, OpenMP,
and Intel Thread Building Blocks (TBB).

+ [Cpp-Taskflow](#cpp-taskflow)
+ [OpenMP](#openmp)
+ [Intel Thread Building Blocks (TBB)](#intel-thread-building-blocks)
+ [Debrief](#debrief)

# Cpp-Taskflow

The code block below demonstrates the graph traversal routine using Cpp-Taskflow given a graph class instance.
We delegate node to a `tf::Task` and use the `precede` function to specify
the dependency between tasks. The `tf.wait_for_all()` blocks until all tasks finish.

```cpp
tf::Taskflow tf(4);

for(size_t i=0; i<graph.length(); i++){
  Node& n = graph.node_at(graph.level()-1, i);
  n._task = tf.silent_emplace([&](){ n.mark(); });
}

for(int l=graph.level()-2; l>=0 ; l--){
  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(l, i);
    n._task = tf.silent_emplace([&](){ n.mark();});
    for(size_t k=0; k<n._out_edges.size(); k++){
      n._task.precede(graph.node_at(l+1, n._out_edges[k])._task);
    }
  }
}
tf.wait_for_all();
```

The code of this implementation can be found at [taskflow.cpp](taskflow.cpp).


# OpenMP

The code block below demonstrates a snippet of graph traversal routine using OpenMP task dependency clause.
Each block is delegated to a OpenMP task.
we have created an additional dependency vector `out` for each node
and use it to explicitly specify both input and output task dependencies. The task scheduling in openMP is dependent on the number of inner and outer edges. Only a part of the implementation is listed given too many repetitive swtich cases. For the full program, please refer to the source code provided below.

```cpp
#pragma omp parallel
{
  #pragma omp single
  {
    for(size_t l=0; l<graph.level(); l++){
      for(int i=0; i<graph.length(); i++){
        Node& n = graph.node_at(l, i);
        size_t out_edge_num = n._out_edges.size();
        size_t in_edge_num = n._in_edges.size();

        switch(in_edge_num){

          case(0):{

            switch(out_edge_num){

              case(1):{
                int* out0 = n.edge_ptr(0);
                #pragma omp task depend(out: out0[0]) shared(n)
                { n.mark(); }
                break;
              }
        //.....................................
        //...........16 switch cases...........
        //.....................................
      }
    }
  }
}
```

The code of this implementation can be found at [omp.cpp](omp.cpp).

# Intel Thread Building Blocks

The code block below demonstrates the graph traversal routine using Intel TBB's flow graph.
We build a dependency graph using the `continue_node` type in TBB flow graph. The `make_edge` function specifies the dependency between two nodes
and calling `wait_for_all` waits until all computations complete.

```cpp
using namespace tbb;
using namespace tbb::flow;
tbb::task_scheduler_init init(std::thread::hardware_concurrency());

tbb::flow::graph G;

for(size_t i=0; i<graph.length(); i++){
  Node& n = graph.node_at(graph.level()-1, i);
  n.tbb_node = std::make_unique<continue_node<continue_msg>>(G, [&](const continue_msg&){ n.mark(); });
}

for(int l=graph.level()-2; l>=0 ; l--){
  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(l, i);
    n.tbb_node = std::make_unique<continue_node<continue_msg>>(G, [&](const continue_msg&){ n.mark(); });
    for(size_t k=0; k<n._out_edges.size(); k++){
      make_edge(*n.tbb_node, *(graph.node_at(l+1, n._out_edges[k]).tbb_node));
    }
  }
}

auto source = std::make_unique<continue_node<continue_msg>>(G, [](const continue_msg&){});
for(int l=0; l>=0 ; l--){
  for(size_t i=0; i<graph.length(); i++){
    Node& n = graph.node_at(l, i);
    make_edge(*source, *(n.tbb_node));
  }
}

source->try_put(continue_msg());
G.wait_for_all();
graph.reset_tbb_node();
```

The code of this implementation can be found at [tbb.cpp](tbb.cpp).

# Debrief

We evaluate our implementations on a
Linux Ubuntu machine of 4 Intel CPUs 3.2GHz and 24 GB memory.

## Software Cost

We use the famous Linux tool [SLOCCount][SLOCCount] to measure the software cost of
each implementation.
The cost model is based on the *constructive cost model* (COCOMO).
In the table below, **SLOC** denotes souce lines of code,
**Dev Effort** denotes development effort estimate (person-months),
**Sched Estimate** denotes schedule estimate (years),
**Developers** denotes the estimate number of developers,
**Dev Cost** denotes total estimated cost to develop.
All quantities are better with fewer values.

| Task Model   | SLOC | Dev Effort | Sched Estimate | Developers | Dev Cost |
| ------------ | ---- | ---------- | -------------- | ---------- | -------- |
| Cpp-Taskflow | 38   | 0.01       | 0.08           | 0.08       | $872     |
| OpenMP 4.5   | 224  | 0.04       | 0.16           | 0.26       | $5,616   |
| Intel TBB    | 55   | 0.01       | 0.09           | 0.10       | $1,285   |

For three different parallel implementations,
Cpp-Taskflow is the most cost-efficient compared with OpenMP and TBB.
Our task model facilitates the implementations of parallel decomposition strategies
and algorithms.

## Performance

We alter the graph level and length
to compare the performance between the Cpp-Taskflow, OpenMP, and Intel TBB.
The dimensions (levelxlength) of graph are constantly increasing from 1x1 to 200x200.

![](result_200by200.png)

By the figure shown above,
both Cpp-Taskflow and TBB outperforms OpenMP by a certain degree.
The performance measures between Cpp-Taskflow and TBB are similar. Cpp-Taskslow has a slight advantage. We believe the slowness of OpenMP attributes to the static property of its dependency clauses. In OpenMP,
We need to explicitly specify all possible dependency constraints
in order to cover all conditions.
This gives rise to a number of if-else and switch statements
that can potentially slow down the runtime.


* * *

[GraphvizOnline]:        https://dreampuf.github.io/GraphvizOnline/
[Intel Developer Zone]:  https://software.intel.com/en-us/blogs/2011/09/09/implementing-a-wave-front-computation-using-the-intel-threading-building-blocks-flow-graph
[SLOCCount]:             https://dwheeler.com/sloccount/
