var Examples =
[
    [ "Task Graph", "Examples.html#ExamplesTaskGraph", null ],
    [ "Async Tasking", "Examples.html#ExamplesAsyncTasking", null ],
    [ "Parallel Algorithms", "Examples.html#ExamplesParallelAlgorithms", null ],
    [ "GPU Task Graph", "Examples.html#ExamplesGPUTaskGraph", null ],
    [ "Wavefront Parallelism", "wavefront.html", [
      [ "Problem Formulation", "wavefront.html#WavefrontComputingFormulation", null ],
      [ "Building the Task Graph", "wavefront.html#WavefrontTaskGraph", null ],
      [ "Applications", "wavefront.html#WavefrontApplications", null ]
    ] ],
    [ "Graph Traversal", "graphtraversal.html", [
      [ "Problem Formulation", "graphtraversal.html#GraphTraversalProblemFormulation", null ],
      [ "Graph Representation", "graphtraversal.html#GraphTraversalGraphRepresentation", null ],
      [ "Parallel Static Traversal", "graphtraversal.html#GraphTraversalStaticTraversal", null ]
    ] ],
    [ "Nondeterministic Control Flow", "ExamplesNondeterministicControlFlow.html", [
      [ "Problem Formulation", "ExamplesNondeterministicControlFlow.html#CoinFlippingProblemFormulation", null ],
      [ "Implementation with Conditional Tasking", "ExamplesNondeterministicControlFlow.html#CoinFlippingBinary", null ],
      [ "Extension to Ternary Coins", "ExamplesNondeterministicControlFlow.html#CoinFlippingTernary", null ]
    ] ],
    [ "Blocked Cholesky Factorization", "ExamplesCholesky.html", [
      [ "Motivation", "ExamplesCholesky.html#CholeskyMotivation", null ],
      [ "The Blocked Algorithm", "ExamplesCholesky.html#CholeskyAlgorithm", null ],
      [ "Task Graph Structure", "ExamplesCholesky.html#CholeskyTaskGraph", null ],
      [ "Implementation", "ExamplesCholesky.html#CholeskyImplementation", null ],
      [ "Design Points", "ExamplesCholesky.html#CholeskyDesignPoints", null ]
    ] ],
    [ "Critical Path Scheduling", "ExamplesPERT.html", [
      [ "What is a PERT Chart?", "ExamplesPERT.html#PERTIntroduction", null ],
      [ "A Concrete Project", "ExamplesPERT.html#PERTProblem", null ],
      [ "Finding the Critical Path", "ExamplesPERT.html#PERTCriticalPath", null ],
      [ "Implementation", "ExamplesPERT.html#PERTImplementation", null ],
      [ "Design Points", "ExamplesPERT.html#PERTDesignPoints", null ]
    ] ],
    [ "Incremental Build Graph", "ExamplesMakeGraph.html", [
      [ "What is an Incremental Build?", "ExamplesMakeGraph.html#MakeGraphIntroduction", null ],
      [ "A Concrete Build Graph", "ExamplesMakeGraph.html#MakeGraphProblem", null ],
      [ "Using Condition Tasks for Staleness", "ExamplesMakeGraph.html#MakeGraphConditionTask", null ],
      [ "The Task Race at the Join Point", "ExamplesMakeGraph.html#MakeGraphPitfall", null ],
      [ "The Auxiliary Join Task Pattern", "ExamplesMakeGraph.html#MakeGraphJoinTask", null ],
      [ "Implementation", "ExamplesMakeGraph.html#MakeGraphImplementation", null ],
      [ "Design Points", "ExamplesMakeGraph.html#MakeGraphDesignPoints", null ]
    ] ],
    [ "Speculative Execution", "ExamplesSpeculativeExecution.html", [
      [ "Motivation", "ExamplesSpeculativeExecution.html#SpecMotivation", null ],
      [ "The Problem: Bloom Filter Lookup", "ExamplesSpeculativeExecution.html#SpecProblem", null ],
      [ "Per-Item Graph Structure", "ExamplesSpeculativeExecution.html#SpecSingleItem", null ],
      [ "Batch Graph: N Keys in Parallel", "ExamplesSpeculativeExecution.html#SpecBatchGraph", null ],
      [ "Implementation", "ExamplesSpeculativeExecution.html#SpecImplementation", null ],
      [ "Design Points", "ExamplesSpeculativeExecution.html#SpecDesignPoints", null ]
    ] ],
    [ "Fibonacci Number", "ExamplesFibonacciNumber.html", [
      [ "Problem Formulation", "ExamplesFibonacciNumber.html#FibonacciNumberProblem", null ],
      [ "Recursive Fibonacci with Runtime Tasking", "ExamplesFibonacciNumber.html#RecursiveFibonacciParallelismUsingRuntimeTasking", [
        [ "Tail Recursion Optimisation", "ExamplesFibonacciNumber.html#TailRecursionOptimization", null ],
        [ "Benchmarking", "ExamplesFibonacciNumber.html#FibonacciNumberBenchmarking", null ]
      ] ],
      [ "Recursive Fibonacci with Task Group", "ExamplesFibonacciNumber.html#RecursiveFibonacciParallelismUsingTaskGroup", null ]
    ] ],
    [ "Async Producer-Consumer Pipeline", "ExamplesAsyncProducerConsumer.html", [
      [ "Problem Formulation", "ExamplesAsyncProducerConsumer.html#AsyncProducerConsumerProblem", null ],
      [ "Implementation", "ExamplesAsyncProducerConsumer.html#AsyncProducerConsumerImplementation", null ],
      [ "Conditional Consumption with Cooperative Execution", "ExamplesAsyncProducerConsumer.html#AsyncProducerConsumerIsDone", null ]
    ] ],
    [ "Divide-and-Conquer Parallelism", "ExamplesDivideAndConquer.html", [
      [ "What is Divide-and-Conquer?", "ExamplesDivideAndConquer.html#DivideAndConquerWhatIs", null ],
      [ "Merge Sort Example", "ExamplesDivideAndConquer.html#DivideAndConquerMergeSort", null ],
      [ "Parallel Merge Sort with TaskGroup", "ExamplesDivideAndConquer.html#DivideAndConquerParallelMergeSort", null ],
      [ "General Template", "ExamplesDivideAndConquer.html#DivideAndConquerGeneralTemplate", null ],
      [ "Benchmarking", "ExamplesDivideAndConquer.html#DivideAndConquerBenchmarking", null ]
    ] ],
    [ "Dynamic Dependency Graph", "ExamplesDynamicDependencyGraph.html", [
      [ "Problem Formulation", "ExamplesDynamicDependencyGraph.html#DynamicDependencyGraphProblem", null ],
      [ "Why Static Construction Fails", "ExamplesDynamicDependencyGraph.html#DynamicDependencyGraphStaticFails", null ],
      [ "Implementation", "ExamplesDynamicDependencyGraph.html#DynamicDependencyGraphImplementation", null ],
      [ "Branching Based on Runtime Conditions", "ExamplesDynamicDependencyGraph.html#DynamicDependencyGraphBranchingTopology", null ]
    ] ],
    [ "Travelling Salesman Problem", "ExamplesTSP.html", [
      [ "What is the Travelling Salesman Problem?", "ExamplesTSP.html#TSPIntroduction", null ],
      [ "Concrete Walkthrough", "ExamplesTSP.html#TSPWalkthrough", null ],
      [ "Implementation", "ExamplesTSP.html#TSPImplementation", null ],
      [ "Design Points", "ExamplesTSP.html#TSPDesignPoints", null ]
    ] ],
    [ "Graph Coloring", "ExamplesGraphColoring.html", [
      [ "What is Graph Coloring?", "ExamplesGraphColoring.html#GCIntroduction", null ],
      [ "Concrete Walkthrough", "ExamplesGraphColoring.html#GCWalkthrough", null ],
      [ "Implementation", "ExamplesGraphColoring.html#GCImplementation", null ],
      [ "Design Points", "ExamplesGraphColoring.html#GCDesignPoints", null ]
    ] ],
    [ "Parallel Breadth-First Search", "ExamplesBFS.html", [
      [ "What is BFS and Why Parallelize It?", "ExamplesBFS.html#BFSIntroduction", null ],
      [ "Implementation", "ExamplesBFS.html#BFSImplementation", null ],
      [ "Encoding the Loop as a Condition Task", "ExamplesBFS.html#BFSConditionTask", null ]
    ] ],
    [ "Matrix Multiplication", "matrix_multiplication.html", [
      [ "Problem Formulation", "matrix_multiplication.html#MatrixMultiplicationProblem", null ],
      [ "Parallel Decomposition", "matrix_multiplication.html#MatrixMultiplicationParallelPattern", null ],
      [ "Benchmarking", "matrix_multiplication.html#MatrixMultiplicationBenchmarking", null ]
    ] ],
    [ "k-means Clustering", "kmeans.html", [
      [ "Problem Formulation", "kmeans.html#KMeansProblemFormulation", null ],
      [ "Parallel k-means on CPU", "kmeans.html#ParallelKMeansUsingCPUs", null ],
      [ "Benchmarking", "kmeans.html#KMeansBenchmarking", null ]
    ] ],
    [ "3D Stencil Computation", "ExamplesStencil3D.html", [
      [ "Problem Formulation", "ExamplesStencil3D.html#Stencil3DProblem", null ],
      [ "Parallel Implementation", "ExamplesStencil3D.html#Stencil3DImplementation", null ],
      [ "Multiple Sweeps", "ExamplesStencil3D.html#Stencil3DMultipleSweeps", null ],
      [ "Iteration Control with a Condition Task", "ExamplesStencil3D.html#Stencil3DConditionTask", null ],
      [ "Design Points", "ExamplesStencil3D.html#Stencil3DDesignPoints", null ]
    ] ],
    [ "Sparse Matrix-Vector Multiplication", "ExamplesSpMV.html", [
      [ "What is Sparse Matrix-Vector Multiplication?", "ExamplesSpMV.html#SpMVIntroduction", null ],
      [ "The CSR Storage Format", "ExamplesSpMV.html#SpMVCSR", null ],
      [ "Why Partitioner Choice Matters", "ExamplesSpMV.html#SpMVWhyPartitionerMatters", null ],
      [ "Implementation", "ExamplesSpMV.html#SpMVImplementation", null ],
      [ "Design Points", "ExamplesSpMV.html#SpMVDesignPoints", null ]
    ] ],
    [ "2D Image Convolution", "ExamplesConv2D.html", [
      [ "What is 2D Convolution?", "ExamplesConv2D.html#Conv2DIntroduction", null ],
      [ "Concrete Walkthrough", "ExamplesConv2D.html#Conv2DWalkthrough", null ],
      [ "Why it Maps Perfectly to IndexRange?", "ExamplesConv2D.html#Conv2DParallelism", null ],
      [ "Implementation", "ExamplesConv2D.html#Conv2DImplementation", null ],
      [ "Design Points", "ExamplesConv2D.html#Conv2DDesignPoints", null ]
    ] ],
    [ "Text Processing Pipeline", "TextProcessingPipeline.html", [
      [ "Problem Formulation", "TextProcessingPipeline.html#FormulateTheTextProcessingPipelineProblem", null ],
      [ "Creating the Pipeline", "TextProcessingPipeline.html#CreateAParallelTextPipeline", [
        [ "Data Buffer", "TextProcessingPipeline.html#TextPipelineBuffer", null ],
        [ "Sample Output", "TextProcessingPipeline.html#TextPipelineOutput", null ]
      ] ]
    ] ],
    [ "Graph Processing Pipeline", "GraphProcessingPipeline.html", [
      [ "Problem Formulation", "GraphProcessingPipeline.html#FormulateTheGraphProcessingPipelineProblem", null ],
      [ "Implementation", "GraphProcessingPipeline.html#CreateAGraphProcessingPipeline", [
        [ "Topological Order", "GraphProcessingPipeline.html#GraphPipelineTopologicalOrder", null ],
        [ "Sample Output", "GraphProcessingPipeline.html#GraphPipelineOutput", null ]
      ] ],
      [ "Reference", "GraphProcessingPipeline.html#GraphPipelineReference", null ]
    ] ],
    [ "Taskflow Processing Pipeline", "TaskflowProcessingPipeline.html", [
      [ "Problem Formulation", "TaskflowProcessingPipeline.html#FormulateTheTaskflowProcessingPipelineProblem", null ],
      [ "Implementation", "TaskflowProcessingPipeline.html#CreateATaskflowProcessingPipeline", [
        [ "Why corun Instead of run", "TaskflowProcessingPipeline.html#TaskflowPipelineCorun", null ],
        [ "Taskflow Storage", "TaskflowProcessingPipeline.html#TaskflowPipelineStorage", null ],
        [ "Task Graph", "TaskflowProcessingPipeline.html#TaskflowPipelineGraph", null ]
      ] ]
    ] ],
    [ "Matrix Multiplication with CUDA GPU", "MatrixMultiplicationWithCUDAGPU.html", [
      [ "CUDA Kernel", "MatrixMultiplicationWithCUDAGPU.html#GPUAcceleratedMatrixMultiplication", null ],
      [ "CUDA Graph Task", "MatrixMultiplicationWithCUDAGPU.html#DefineACUDAGraphForMatrixMultiplication", null ],
      [ "Benchmarking", "MatrixMultiplicationWithCUDAGPU.html#MatrixMultiplicationcudaFlowBenchmarking", null ]
    ] ],
    [ "k-means Clustering with CUDA GPU", "KMeansWithCUDAGPU.html", [
      [ "CUDA Kernels", "KMeansWithCUDAGPU.html#DefineTheKMeansKernels", null ],
      [ "CUDA Graph Task", "KMeansWithCUDAGPU.html#DefineTheKMeansCUDAGraph", null ],
      [ "Benchmarking", "KMeansWithCUDAGPU.html#KMeansWithGPUBenchmarking", null ]
    ] ]
];