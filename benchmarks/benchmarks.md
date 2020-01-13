# Benchmarks

This folder contains a set of benchmarks to evaluate and compare the performance 
of Cpp-Taskflow with the following task programming frameworks:

  + [OpenMP Task Dependency Clause][OpenMP Tasking]
  + [Intel Threading Building Blocks (TBB) FlowGraph][TBB FlowGraph]

To compile the benchmark sources, 
enable the option `TF_BUILD_BENCHMARKS` in cmake build:

```bash
~$ mkdir build        # create a build folder under cpp-taskflow/
~$ cd build
~$ cmake ../ -DTF_BUILD_BENCHMARKS=ON
~$ make 
```

After you successfully compile all benchmark sources,
executables will be available in the respective folder of an application.
Currently, we provide the following applications:

  + [Graph Traveral](./graph_traversal): traverses a direct acyclic graph
  + [Wavefront](./wavefront): propagates computations in a two-dimensional (2D) grid
  + [Linear Chain](./linear_chain): computes a linear chain of tasks
  + [Binary Tree](./binary_tree): traverse a complete binary tree
  + [Matrix Multiplication](./matrix_multiplication): multiplies two matrices
  + [MNIST](./mnist): trains a neural network-based image classfier on the MNIST dataset

We have provided a python wrapper [regression.py](./regression.py) to help
configure the benchmark of each application,
including thread count, rounds to average, tasking methods, and plot.

```bash
usage: regression.py [-h] -b
                     {wavefront,graph_traversal,binary_tree,linear_chain,matrix_multiplication,mnist}
                     [{wavefront,graph_traversal,binary_tree,linear_chain,matrix_multiplication,mnist} ...]
                     [-m {tf,tbb,omp} [{tf,tbb,omp} ...]] -t THREADS
                     [THREADS ...] [-r NUM_ROUNDS] [-p PLOT]
```

For example, the following command benchmarks 
Cpp-Taskflow (tf), OpenMP (omp), and TBB (tbb)
on graph traversal, wavefront, and linear chain applications
across 1, 4, 8, and 16 threads,
with data collected in an average of ten runs. 
Results are illustrated in a plot and saved to `result.png`.

```bash
~$ python regression.py -m tf omp tbb \ 
                        -b graph_traversal wavefront linear_chain \
                        -t 1 4 8 16 \
                        -r 10 \
                        -plot true
```


---


[OpenMP Tasking]:        https://www.openmp.org/spec-html/5.0/openmpsu99.html 
[TBB FlowGraph]:         https://www.threadingbuildingblocks.org/tutorial-intel-tbb-flow-graph
