var Examples =
[
    [ "Task Graph", "Examples.html#ExamplesTaskGraph", null ],
    [ "Parallel Algorithms", "Examples.html#ExamplesParallelAlgorithms", null ],
    [ "GPU Task Graph", "Examples.html#ExamplesGPUTaskGraph", null ],
    [ "Async Tasking", "Examples.html#ExamplesAsyncTasking", null ],
    [ "Wavefront Parallelism", "wavefront.html", [
      [ "Problem Formulation", "wavefront.html#WavefrontComputingFormulation", null ],
      [ "Building the Task Graph", "wavefront.html#WavefrontTaskGraph", null ],
      [ "Applications", "wavefront.html#WavefrontApplications", null ]
    ] ],
    [ "Nondeterministic Control Flow", "ExamplesNondeterministicControlFlow.html", [
      [ "Problem Formulation", "ExamplesNondeterministicControlFlow.html#CoinFlippingProblemFormulation", null ],
      [ "Implementation with Conditional Tasking", "ExamplesNondeterministicControlFlow.html#CoinFlippingBinary", null ],
      [ "Extension to Ternary Coins", "ExamplesNondeterministicControlFlow.html#CoinFlippingTernary", null ]
    ] ],
    [ "Graph Traversal", "graphtraversal.html", [
      [ "Problem Formulation", "graphtraversal.html#GraphTraversalProblemFormulation", null ],
      [ "Graph Representation", "graphtraversal.html#GraphTraversalGraphRepresentation", null ],
      [ "Parallel Static Traversal", "graphtraversal.html#GraphTraversalStaticTraversal", null ]
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
    ] ]
];