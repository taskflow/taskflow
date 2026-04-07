var Algorithms =
[
    [ "Partitioning Algorithm", "PartitioningAlgorithm.html", [
      [ "Define a Partitioner for Parallel Algorithms", "PartitioningAlgorithm.html#DefineAPartitionerForParallelAlgorithms", null ],
      [ "Define a Static Partitioner", "PartitioningAlgorithm.html#DefineAStaticPartitioner", null ],
      [ "Define a Dynamic Partitioner", "PartitioningAlgorithm.html#DefineADynamicPartitioner", null ],
      [ "Define a Guided Partitioner", "PartitioningAlgorithm.html#DefineAGuidedPartitioner", null ],
      [ "Define a Closure Wrapper for a Partitioner", "PartitioningAlgorithm.html#DefineAClosureWrapperForAPartitioner", null ]
    ] ],
    [ "Parallel Iterations", "ParallelIterations.html", [
      [ "Include the Header", "ParallelIterations.html#ParallelIterationsIncludeTheHeader", null ],
      [ "Create an Index-based Parallel-Iteration Task", "ParallelIterations.html#ParallelIterationsIndexBased", [
        [ "Capture Indices by Reference", "ParallelIterations.html#ParallelForEachCaptureIndicesByReference", null ]
      ] ],
      [ "Create an IndexRange-based Parallel-Iteration Task", "ParallelIterations.html#ParallelIterationsIndexRangeBased", [
        [ "1D IndexRange", "ParallelIterations.html#ParallelIterationsIndexRange1D", null ],
        [ "Multi-dimensional IndexRange", "ParallelIterations.html#ParallelIterationsIndexRangeMD", null ],
        [ "Capture Range by Reference", "ParallelIterations.html#ParallelIterationsIndexRangeByReference", null ]
      ] ],
      [ "Create an Iterator-based Parallel-Iteration Task", "ParallelIterations.html#ParallelIterationsIteratorBased", [
        [ "Capture Iterators by Reference", "ParallelIterations.html#ParallelForEachCaptureIteratorsByReference", null ]
      ] ],
      [ "Configure a Partitioner", "ParallelIterations.html#ParallelIterationsConfigureAPartitioner", null ]
    ] ],
    [ "Parallel Transforms", "ParallelTransforms.html", [
      [ "Include the Header", "ParallelTransforms.html#ParallelTransformsInclude", null ],
      [ "Create a Unary Parallel-Transform Task", "ParallelTransforms.html#ParallelTransformsOverARange", null ],
      [ "Capture Iterators by Reference", "ParallelTransforms.html#ParallelTransformsCaptureIteratorsByReference", null ],
      [ "Create a Binary Parallel-Transform Task", "ParallelTransforms.html#ParallelBinaryTransformsOverARange", null ],
      [ "Configure a Partitioner", "ParallelTransforms.html#ParallelTransformsCfigureAPartitioner", null ]
    ] ],
    [ "Parallel Reduction", "ParallelReduction.html", [
      [ "Include the Header", "ParallelReduction.html#ParallelReductionInclude", null ],
      [ "Create a Parallel-Reduction Task", "ParallelReduction.html#A2ParallelReduction", null ],
      [ "Capture Iterators by Reference", "ParallelReduction.html#ParallelReductionCaptureIteratorsByReference", null ],
      [ "Create a Parallel-Transform-Reduction Task", "ParallelReduction.html#A2ParallelTransformationReduction", null ],
      [ "Create a Reduce-by-Index Task", "ParallelReduction.html#ParallelReductionCreateAReduceByIndexTask", null ],
      [ "Configure a Partitioner", "ParallelReduction.html#ParallelReductionConfigureAPartitioner", null ]
    ] ],
    [ "Parallel Sort", "ParallelSort.html", [
      [ "Include the Header", "ParallelSort.html#ParallelSortInclude", null ],
      [ "Sort a Range of Items", "ParallelSort.html#SortARangeOfItems", null ],
      [ "Sort a Range of Items with a Custom Comparator", "ParallelSort.html#SortARangeOfItemsWithACustomComparator", null ],
      [ "Enable Stateful Data Passing", "ParallelSort.html#ParallelSortEnableStatefulDataPassing", null ]
    ] ],
    [ "Parallel Scan", "ParallelScan.html", [
      [ "Include the Header", "ParallelScan.html#ParallelScanInclude", null ],
      [ "What is a Scan Operation?", "ParallelScan.html#WhatIsAScanOperation", null ],
      [ "Create a Parallel Inclusive Scan Task", "ParallelScan.html#CreateAParallelInclusiveScanTask", null ],
      [ "Create a Parallel Transform-Inclusive Scan Task", "ParallelScan.html#CreateAParallelTransformInclusiveScanTask", null ],
      [ "Create a Parallel Exclusive Scan Task", "ParallelScan.html#CreateAParallelExclusiveScanTask", null ],
      [ "Create a Parallel Transform-Exclusive Scan Task", "ParallelScan.html#CreateAParallelTransformExclusiveScanTask", null ]
    ] ],
    [ "Parallel Find", "ParallelFind.html", [
      [ "Include the Header", "ParallelFind.html#ParallelFindIncludeTheHeader", null ],
      [ "What is a Find Algorithm?", "ParallelFind.html#WhatIsAFindAlgorithm", null ],
      [ "Create a Parallel Find-If Task", "ParallelFind.html#CreateAParallelFindIfTask", null ],
      [ "Capture Iterators by Reference", "ParallelFind.html#ParallelFindCaptureIteratorsByReference", null ],
      [ "Create a Parallel Find-If-Not Task", "ParallelFind.html#CreateAParallelFindIfNotTask", null ],
      [ "Find the Smallest and the Largest Elements", "ParallelFind.html#ParallelFindMinMaxElement", null ],
      [ "Configure a Partitioner", "ParallelFind.html#ParallelFindConfigureAPartitioner", null ]
    ] ],
    [ "Module Algorithm", "ModuleAlgorithm.html", [
      [ "Include the Header", "ModuleAlgorithm.html#ModuleAlgorithmInclude", null ],
      [ "What is a Module Task", "ModuleAlgorithm.html#WhatIsAModuleTask", null ],
      [ "Create a Module Task over a Custom Graph", "ModuleAlgorithm.html#CreateAModuleTaskOverACustomGraph", null ]
    ] ],
    [ "Task-parallel Pipeline", "TaskParallelPipeline.html", [
      [ "Include the Header", "TaskParallelPipeline.html#TaskParallelPipelineIncludeHeaderFile", null ],
      [ "Understand the Pipeline Scheduling Framework", "TaskParallelPipeline.html#UnderstandPipelineScheduling", null ],
      [ "Create a Task-parallel Pipeline Module Task", "TaskParallelPipeline.html#CreateATaskParallelPipelineModuleTask", null ],
      [ "Connect Pipeline with Other Tasks", "TaskParallelPipeline.html#ConnectWithTasks", [
        [ "Example 1: Iterate a Pipeline", "TaskParallelPipeline.html#IterateAPipeline", null ],
        [ "Example 2: Concatenate Two Pipelines", "TaskParallelPipeline.html#ConcatenateTwoPipelines", null ],
        [ "Example 3: Define Multiple Parallel Pipelines", "TaskParallelPipeline.html#DefineMultipleTaskParallelPipelines", null ]
      ] ],
      [ "Reset a Pipeline", "TaskParallelPipeline.html#ResetPipeline", null ]
    ] ],
    [ "Scalable Task-parallel Pipeline", "ScalableTaskParallelPipeline.html", [
      [ "Include the Header", "ScalableTaskParallelPipeline.html#IncludeTheScalablePipelineHeader", null ],
      [ "Create a Scalable Pipeline Module Task", "ScalableTaskParallelPipeline.html#CreateAScalablePipelineModuleTask", null ],
      [ "Reset a Placeholder Scalable Pipeline", "ScalableTaskParallelPipeline.html#ResetAPlaceholderScalablePipeline", null ],
      [ "Use Other Iterator Types", "ScalableTaskParallelPipeline.html#ScalablePipelineUseOtherIteratorTypes", null ]
    ] ],
    [ "Data-parallel Pipeline", "DataParallelPipeline.html", [
      [ "Include the Header", "DataParallelPipeline.html#ParallelDataPipelineIncludeHeaderFile", null ],
      [ "Create a Data Pipeline Module Task", "DataParallelPipeline.html#CreateADataPipelineModuleTask", null ],
      [ "Understand Internal Data Storage", "DataParallelPipeline.html#UnderstandInternalDataStorage", null ]
    ] ]
];