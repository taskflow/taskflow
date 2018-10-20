# Wavefront Computation

We demonstrate in this page how to implement a wavefront computation workload 
using Cpp-Taskflow. 
An example of the wavefront pattern is shown below.

![](wavefront.png)

We partition a given 2D matrix into a set of identical square blocks (sub-matrices).
Each block is mapped to a task that performs a linear scan through each element
coupled with a fixed set of arithmetic operations.
The wavefront propagates the dependency constraints diagonally 
from the top-left block to the bottom-right block. 
In other word, each block precedes two blocks, one to the
right and another below. 
The blocks with the same color can run concurrently.

We consider a baseline sequential program and 
compare with three parallel implementations with Cpp-Taskflow, OpenMP, 
and Intel Thread Building Blocks (TBB).

+ [Sequential Program](#sequential-program)
+ [Cpp-Taskflow](#cpp-taskflow)
+ [OpenMP](#openmp)
+ [Intel Thread Building Blocks (TBB)](#intel-thread-building-blocks)
+ [Debrief](#debrief)

# Sequential Program

The sequential program simply iterates the block row by row,
naturally satisfying the dependency constraints of wavefront computing.
The procedure `block_computation` takes an index pair indicating the location
of the block,
and performs a linear scan coupled with fixed arithmetic operations 
over each element in the block.


```cpp
1: // MB, NB: number of blocks in the two dimensions.
2: for( int i=0; i<MB; ++i){
3:   for( int j=0; j<NB; ++j) {
4:     block_computation(i, j);
5:   }
6: }
```

The code of this implementation can be found at [seq.cpp](seq.cpp).

# Cpp-Taskflow

The code block below demonstrates the core wavefront routine using Cpp-Taskflow. 
We delegate each block to a `tf::Task` and use the `precede` function to specify
the dependency between tasks. The `tf.wait_for_all()` blocks until all tasks finish.

```cpp 
 1: tf::Taskflow tf;
 2:
 3: std::vector<std::vector<tf::Task>> node(MB);
 4:
 5: for(auto &n : node){
 6:   for(size_t i=0; i<NB; i++){
 7:     n.emplace_back(tf.placeholder());
 8:   }
 9: }
10:  
11:  matrix[M-1][N-1] = 0;
12:  for( int i=MB; --i>=0; ) {
13:    for( int j=NB; --j>=0; ) {
14:      node[i][j].work([i=i, j=j, &]() {
15:        block_computation(i, j);
16:      });
17:      if(j+1 < NB) node[i][j].precede(node[i][j+1]);
18:      if(i+1 < MB) node[i][j].precede(node[i+1][j]);
19:    }
20:  }
21:
22:  tf.wait_for_all();
```

The code of this implementation can be found at [taskflow.cpp](taskflow.cpp).


# OpenMP 

The code block below demonstrates the core wavefront routine using OpenMP task dependency clause. 
Each block is delegated to a OpenMP task. 
Since the OpenMP task dependency clause is *static*,
we need to create an additional dependency matrix `D` 
and use it to explicitly specify both input and output task dependencies.

```cpp
 1: // set up the dependency matrix
 2: D = new int *[MB];
 3: for(int i=0; i<MB; ++i) D[i] = new int [NB];
 4: for(int i=0; i<MB; ++i) {
 5:   for(int j=0; j<NB; ++j) {
 6:     D[i][j] = 0;
 7:   }
 8: }
 9: 
10:  omp_set_num_threads(std::thread::hardware_concurrency());
11:
12:  #pragma omp parallel
13:  {
14:    #pragma omp single
15:    {
16:       matrix[M-1][N-1] = 0;
17:       for( int k=1; k <= 2*MB-1; k++) {
18:         int i, j;
19:         if(k <= MB){
20:           i = k-1;
21:           j = 0;
22:         }
23:         else {
24:           //assume matrix is square
25:           i = MB-1;
26:           j = k-MB;
27:         }       
28:         
29:         for(; (k <= MB && i>=0) || (k > MB && j <= NB-1) ; i--, j++){
30:
31:           if(i > 0 && j > 0){
32:             #pragma omp task depend(in:D[i-1][j], D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
33:               block_computation(i, j); 
34:           }
35:           //top left corner
36:           else if(i == 0 && j == 0){
37:             #pragma omp task depend(out:D[i][j]) firstprivate(i, j)
38:               block_computation(i, j); 
39:           } 
40:           //top edge  
41:           else if(j+1 <= NB && i == 0 && j > 0){
42:             #pragma omp task depend(in:D[i][j-1]) depend(out:D[i][j]) firstprivate(i, j)
43:               block_computation(i, j); 
44:           }
45:           //left edge
46:           else if(i+1 <= MB && i > 0 && j == 0){
47:             #pragma omp task depend(in:D[i-1][j]) depend(out:D[i][j]) firstprivate(i, j)
48:               block_computation(i, j); 
49:           }
50:           //bottom right corner
51:           else if(i == MB-1 && j == NB-1){
52:             #pragma omp task depend(in:D[i-1][j] ,D[i][j-1]) firstprivate(i, j)
53:               block_computation(i, j); 
54:           }
55:           else{
56:             assert(false);
57:           }
58:         }
59:       }
60:    }
61:  }
62:  
63:  for ( int i = 0; i < MB; ++i ) delete [] D[i];
64:  delete [] D;
```

The code of this implementation can be found at [omp.cpp](omp.cpp).

# Intel Thread Building Blocks

The code block below demonstrates the core wavefront routine using Intel TBB's flow graph.
We build a dependency graph using the `continue_node` type in TBB flow graph and delegate 
each block to a node. The `make_edge` function specifies the dependency between two nodes 
and calling `wait_for_all` waits until all computations complete.

```cpp 
 1: using namespace tbb;
 2: using namespace tbb::flow;
 3: continue_node<continue_msg> ***node = new continue_node<continue_msg> **[MB];
 4:
 5: for ( int i = 0; i < MB; ++i ) {
 6:  node[i] = new continue_node<continue_msg> *[NB];
 7: };
 8:
 9: graph g;
10:
11: matrix[M-1][N-1] = 0;
12: 
13: for( int i=MB; --i>=0; ) {
14:   for( int j=NB; --j>=0; ) {
15:     node[i][j] = new continue_node<continue_msg>( 
16:       g,
17:       [=]( const continue_msg& ) {
18:          block_computation(i, j);
19:       }
20:     );
21:     if(i+1 < MB) make_edge(*node[i][j], *node[i+1][j]);
22:     if(j+1 < NB) make_edge(*node[i][j], *node[i][j+1]);
23:   }
24: }
25:
26: node[0][0]->try_put(continue_msg());
27: g.wait_for_all();
28:
29: for(int i=0; i<MB; ++i) {
30:   for(int j=0; j<NB; ++j) {
31:     delete node[i][j];
32:   }
33: }
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
| Sequential   | 63   | 0.01       | 0.10           | 0.11       | $1,482   |
| Cpp-Taskflow | 69   | 0.01       | 0.10           | 0.12       | $1,631   |
| OpenMP 4.5   | 113  | 0.02       | 0.12           | 0.17       | $2,738   |
| Intel TBB    | 84   | 0.01       | 0.11           | 0.14       | $2,005   |

The sequential program obviously has the least software cost.
In terms of parallel implementations,
Cpp-Taskflow is the most cost-efficient compared with OpenMP and TBB,
and has very close cost margin to the sequential program.
Our task model facilitates the implementations of parallel decomposition strategies
and algorithms.

## Performance

We alter the matrix size and the block size
to compare the performance between the sequential program, Cpp-Taskflow, OpenMP, and Intel TBB.
For the first
experiment, we fix the block size to 100x100 and test four matrix sizes:
10Kx10K, 20Kx20K, 30Kx30K and 40Kx40K. 
For the second experiment, we fix the
matrix size to 40Kx40K and test four block sizes: 20x20, 40x40, 80x80 and 160x160.

![](experiment.png)

The left figure shows the first experiment under different matrix sizes and
the right figure shows the second experiment under different block sizes.
In general,
both Cpp-Taskflow and TBB outperforms OpenMP.
The performance margin between TBB and Cpp-Taskflow is within 1\%.
We believe the slowness of OpenMP attributes to the static property of its dependency clauses.
We need to explicitly specify all possible dependency constraints
in order to cover all conditions.
This gives rise to a number of if-else and switch statements
that can potentially slow down the runtime.


* * *

[GraphvizOnline]:        https://dreampuf.github.io/GraphvizOnline/
[Intel Developer Zone]:  https://software.intel.com/en-us/blogs/2011/09/09/implementing-a-wave-front-computation-using-the-intel-threading-building-blocks-flow-graph
[SLOCCount]:             https://dwheeler.com/sloccount/

