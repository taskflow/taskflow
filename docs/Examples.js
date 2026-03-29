var Examples =
[
    [ "Wavefront Parallelism", "wavefront.html", [
      [ "Problem Formulation", "wavefront.html#WavefrontComputingFormulation", null ],
      [ "Wavefront Task Graph", "wavefront.html#WavefrontTaskGraph", null ]
    ] ],
    [ "Fibonacci Number", "ExamplesFibonacciNumber.html", [
      [ "Problem Formulation", "ExamplesFibonacciNumber.html#FibonacciNumberProblem", null ],
      [ "Recursive Fibonacci Parallelism using Runtime Tasking", "ExamplesFibonacciNumber.html#RecursiveFibonacciParallelismUsingRuntimeTasking", [
        [ "Tail Recursion Optimization", "ExamplesFibonacciNumber.html#TailRecursionOptimization", null ],
        [ "Benchmarking", "ExamplesFibonacciNumber.html#FibonacciNumberBenchmarking", null ]
      ] ],
      [ "Recursive Fibonacci Parallelism using Task Group", "ExamplesFibonacciNumber.html#RecursiveFibonacciParallelismUsingTaskGroup", null ]
    ] ],
    [ "Nondeterministic Control Flow", "ExamplesNondeterministicControlFlow.html", [
      [ "Problem Formulation", "ExamplesNondeterministicControlFlow.html#CoinFlippingProblemFormulation", null ],
      [ "Implementation using Conditional Tasking", "ExamplesNondeterministicControlFlow.html#CoinFlippingProbabilistic", null ],
      [ "Ternary Coins", "ExamplesNondeterministicControlFlow.html#CoinFlippingTernaryCoins", null ]
    ] ],
    [ "Graph Traversal", "graphtraversal.html", [
      [ "Problem Formulation", "graphtraversal.html#GraphTraversalProblemFormulation", null ],
      [ "Graph Representation", "graphtraversal.html#GraphTraversalGraphRepresentation", null ],
      [ "Static Traversal", "graphtraversal.html#GraphTraversalStaticTraversal", null ]
    ] ],
    [ "Matrix Multiplication", "matrix_multiplication.html", [
      [ "Problem Formulation", "matrix_multiplication.html#MatrixMultiplicationProblem", null ],
      [ "Parallel Patterns", "matrix_multiplication.html#MatrixMultiplicationParallelPattern", null ],
      [ "Benchmarking", "matrix_multiplication.html#MatrixMultiplicationBenchmarking", null ]
    ] ],
    [ "Matrix Multiplication with CUDA GPU", "MatrixMultiplicationWithCUDAGPU.html", [
      [ "Define a Matrix Multiplication Kernel", "MatrixMultiplicationWithCUDAGPU.html#GPUAcceleratedMatrixMultiplication", null ],
      [ "Define a CUDA Graph for Matrix Multiplication", "MatrixMultiplicationWithCUDAGPU.html#DefineACUDAGraphForMatrixMultiplication", null ],
      [ "Benchmarking", "MatrixMultiplicationWithCUDAGPU.html#MatrixMultiplicationcudaFlowBenchmarking", null ]
    ] ],
    [ "k-means Clustering", "kmeans.html", [
      [ "Problem Formulation", "kmeans.html#KMeansProblemFormulation", null ],
      [ "Parallel k-means using CPUs", "kmeans.html#ParallelKMeansUsingCPUs", null ],
      [ "Benchmarking", "kmeans.html#KMeansBenchmarking", null ]
    ] ],
    [ "k-means Clustering with CUDA GPU", "KMeansWithCUDAGPU.html", [
      [ "Define the k-means Kernels", "KMeansWithCUDAGPU.html#DefineTheKMeansKernels", null ],
      [ "Define the k-means CUDA Graph", "KMeansWithCUDAGPU.html#DefineTheKMeansCUDAGraph", null ],
      [ "Benchmarking", "KMeansWithCUDAGPU.html#KMeansWithGPUBenchmarking", null ]
    ] ],
    [ "Text Processing Pipeline", "TextProcessingPipeline.html", [
      [ "Formulate the Text Processing Pipeline Problem", "TextProcessingPipeline.html#FormulateTheTextProcessingPipelineProblem", null ],
      [ "Create a Text Processing Pipeline", "TextProcessingPipeline.html#CreateAParallelTextPipeline", [
        [ "Define the Data Buffer", "TextProcessingPipeline.html#TextPipelineDefineTheDataBuffer", null ],
        [ "Define the Pipes", "TextProcessingPipeline.html#TextPipelineDefineThePipes", null ],
        [ "Define the Task Graph", "TextProcessingPipeline.html#TextPipelineDefineTheTaskGraph", null ],
        [ "Submit the Task Graph", "TextProcessingPipeline.html#TextPipelineSubmitTheTaskGraph", null ]
      ] ]
    ] ],
    [ "Graph Processing Pipeline", "GraphProcessingPipeline.html", [
      [ "Formulate the Graph Processing Pipeline Problem", "GraphProcessingPipeline.html#FormulateTheGraphProcessingPipelineProblem", null ],
      [ "Create a Graph Processing Pipeline", "GraphProcessingPipeline.html#CreateAGraphProcessingPipeline", [
        [ "Find a Topological Order of the Graph", "GraphProcessingPipeline.html#GraphPipelineFindATopologicalOrderOfTheGraph", null ],
        [ "Define the Stage Function", "GraphProcessingPipeline.html#GraphPipelineDefineTheStageFunction", null ],
        [ "Define the Pipes", "GraphProcessingPipeline.html#GraphPipelineDefineThePipes", null ],
        [ "Define the Task Graph", "GraphProcessingPipeline.html#GraphPipelineDefineTheTaskGraph", null ],
        [ "Submit the Task Graph", "GraphProcessingPipeline.html#GraphPipelineSubmitTheTaskGraph", null ]
      ] ],
      [ "Reference", "GraphProcessingPipeline.html#GraphPipelineReference", null ]
    ] ],
    [ "Taskflow Processing Pipeline", "TaskflowProcessingPipeline.html", [
      [ "Formulate the Taskflow Processing Pipeline Problem", "TaskflowProcessingPipeline.html#FormulateTheTaskflowProcessingPipelineProblem", null ],
      [ "Create a Taskflow Processing Pipeline", "TaskflowProcessingPipeline.html#CreateATaskflowProcessingPipeline", [
        [ "Define Taskflows", "TaskflowProcessingPipeline.html#TaskflowPipelineDefineTaskflows", null ],
        [ "Define the Pipes", "TaskflowProcessingPipeline.html#TaskflowPipelineDefineThePipes", null ],
        [ "Define the Task Graph", "TaskflowProcessingPipeline.html#TaskflowPipelineDefineTheTaskGraph", null ],
        [ "Submit the Task Graph", "TaskflowProcessingPipeline.html#TaskflowPipelineSubmitTheTaskGraph", null ]
      ] ]
    ] ]
];