# Wavefront

This page we compare the three implementations of wavefront computing pattern using OpenMP, Intel-TBB and Cpp-Taskflow.
The wavefront computing pattern is from the blog in [Intel Developer Zone]. 

![](wavefront.png)

As shown in the figure, we partition a 2D matrix into a set of identical square sub-matrices (blocks). 
Each submatrix is mapped to a task that performs a linear scan through each element and 
apply some arithmetic calculation. The wavefront propagates the block dependency diagonally 
from the top-left submatrix to the bottom-right submatrix. Each block precedes two blocks, one to the
right and another below. The blocks with the same color can run concurrently.


+ [OpenMP](#OpenMP)
+ [Intel-TBB](#TBB)
+ [Cpp-Taskflow](#Cpp-Taskflow)

# OpenMP 

```cpp
// MB, NB: number of blocks in the two dimensions. B: dimension of a block
// matrix: the given 2D matrix 
// D: dependency matrix 
1:  void wavefront(size_t MB, size_t NB, size_t B, double** matrix, int** D){
2:    omp_set_num_threads(std::thread::hardware_concurrency());
3:    #pragma omp parallel
4:    {
5:      #pragma omp single
6:      {
7:        for(int i=0; i<MB; i++){
8:          for(int j=0; j<NB; j++) {
9:            if(i > 0 && j > 0){
10:             #pragma omp task depend(in:D[i-1][j], D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
11:             block_computation(matrix, B, i, j);
12:           }
13:           // Top left corner
14:           else if(i == 0 && j == 0){
15:             #pragma omp task depend(out:D[i][j]) firstprivate(i, j)
16:             block_computation(matrix, B, i, j);
17:           }
18:           // Top edge  
19:           else if(j+1 <= NB && i == 0 && j > 0){
20:             #pragma omp task depend(in:D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
21:             block_computation(matrix, B, i, j);
22:           }
23:           // Left edge
24:           else if(i+1 <= MB && i > 0 && j == 0){
25:             #pragma omp task depend(in:D[i-1][j]) depend(out:D[i][j]) firstprivate(i, j)
26:             block_computation(matrix, B, i, j);
27:           }
28:           // Bottom right corner
29:           else{
30:             #pragma omp task depend(in:D[i-1][j] ,D[i][j-1]) firstprivate(i, j)
31:             block_computation(matrix, B, i, j);
32:           }
33:         } // End of inner loop
34:       }  // End of outer loop
35:     } // End of omp single 
36:   } // End of omp parallel 
37: }
```

This function shows the wavefront computing implemented using OpenMP. Each
block is delegated to a OpenMP task. For each task we need to explicitly specify both the
input and output depedency and an additional depedency matrix `D` is
created for this purpose.


# Intel-TBB

```cpp 
1:  using namespace tbb;
2:  using namespace tbb::flow;
3:  
4:  // MB, NB: number of blocks in the two dimensions. B: dimension of a block
5:  // matrix: the given 2D matrix   
6:  // nodes: the nodes in flow graph
7:  // G: Intel-TBB flow graph
8:  void wavefront(size_t MB, size_t NB, size_t B, double** matrix, continue_node<continue_msg> ***nodes, Graph& G){ 
9:   for(int i=MB; --i>=0;) { 
10:     for(int j=NB; --j>=0;) {
11:       node[i][j] = new tbb::flow::continue_node<tbb::flow::continue_msg>(G,
12:           [=](const continue_msg&) {
13:            block_computation(matrix, i, j); 
14:       });
15:       if(i+1 < MB) {
16:          make_edge(*node[i][j], *node[i+1][j]);
17:       }
18:       if(j+1 < NB) {
19:          make_edge(*node[i][j], *node[i][j+1]);
20:       } 
21:     } // End of inner loop
22:   } // End of outer loop
23:  
24:   nodes[0][0]->try_put(continue_msg());
25:   G.wait_for_all();
26: }
```

This function shows the wavefront computing implemented using Intel-TBB flow graph. We 
build a depedency graph using the `continue_node` type in TBB flow graph and delegate 
each block to a node. The `make_edge` function specifies the depedency between two nodes 
and calling `wait_for_all` to wait until all computations complete.

# Cpp-Taskflow

```cpp 
1:  // MB, NB: number of blocks in the two dimensions. B: dimension of a block
2:  // matrix: the given 2D matrix   
3:  // tasks: the placeholders for tasks in Taskflow
4:  // tf: Taskflow object
5:  void wavefront(size_t MB, size_t NB, size_t B, double** matrix, std::vector<std::vector<tf::Task>>& tasks, tf::Taskflow& tf){ 
6:  for(int i=MB; --i>=0;) { 
7:     for(int j=NB; --j>=0;) { 
8:       task[i][j].work([=]() {
9:           block_computation(matrix, B, i, j); 
10:        }
11:      );  
12:      if(j+1 < NB) {
13:        task[i][j].precede(task[i][j+1]);
14:      }
15:      if(i+1 < MB) {
16:        task[i][j].precede(task[i+1][j]);
17:      }
18:    } // End of inner loop
19:  } // End of outer loop
20:
21:  tf.wait_for_all();
22: }
```
This function shows the wavefront computing implemented using Cpp-Taskflow. We
delegate each block to a `tf::Task` and use the `precede` function to specify
the dependency between tasks. The `tf.wait_for_all()` blocks until all tasks
are executed.


* * *

[GraphvizOnline]:        https://dreampuf.github.io/GraphvizOnline/
[Intel Developer Zone]:  https://software.intel.com/en-us/blogs/2011/09/09/implementing-a-wave-front-computation-using-the-intel-threading-building-blocks-flow-graph
