var Cookbook =
[
    [ "Project Motivation", "ProjectMotivation.html", [
      [ "The Parallel Programming Challenge", "ProjectMotivation.html#TheParallelProgrammingChallenge", null ],
      [ "Heterogeneous Computing", "ProjectMotivation.html#HeterogeneousComputing", null ],
      [ "From Loops to Task Graphs", "ProjectMotivation.html#LoopVsTaskParallelism", null ],
      [ "Task-based Parallelism", "ProjectMotivation.html#TaskBasedParallelism", null ],
      [ "What Taskflow Does Differently", "ProjectMotivation.html#TheProjectMantra", null ]
    ] ],
    [ "Static Tasking", "StaticTasking.html", [
      [ "Create a Task Dependency Graph", "StaticTasking.html#CreateATaskDependencyGraph", null ],
      [ "Visualize a Task Dependency Graph", "StaticTasking.html#VisualizeATaskDependencyGraph", null ],
      [ "Traverse Adjacent Tasks", "StaticTasking.html#TraverseAdjacentTasks", null ],
      [ "Attach User Data to a Task", "StaticTasking.html#AttachUserDataToATask", null ],
      [ "Understand the Lifetime of a Task", "StaticTasking.html#UnderstandTheLifetimeOfATask", null ],
      [ "Move a Taskflow", "StaticTasking.html#MoveATaskflow", null ]
    ] ],
    [ "Executor", "ExecuteTaskflow.html", [
      [ "Create an Executor", "ExecuteTaskflow.html#CreateAnExecutor", null ],
      [ "Understand Work Stealing in Executor", "ExecuteTaskflow.html#UnderstandWorkStealingInExecutor", null ],
      [ "Execute a Taskflow", "ExecuteTaskflow.html#ExecuteATaskflow", null ],
      [ "Understand the Execution Order", "ExecuteTaskflow.html#UnderstandTheExecutionOrder", null ],
      [ "Understand the Ownership", "ExecuteTaskflow.html#UnderstandTheOwnership", null ],
      [ "Execute a Taskflow with Transferred Ownership", "ExecuteTaskflow.html#ExecuteATaskflowWithTransferredOwnership", null ],
      [ "Execute a Taskflow from an Internal Worker Cooperatively", "ExecuteTaskflow.html#ExecuteATaskflowFromAnInternalWorker", null ],
      [ "Thread Safety of Executor", "ExecuteTaskflow.html#ThreadSafetyOfExecution", null ],
      [ "Query the Worker ID", "ExecuteTaskflow.html#QueryTheWorkerID", null ],
      [ "Observe Thread Activities", "ExecuteTaskflow.html#ObserveThreadActivities", null ],
      [ "Modify Worker Property", "ExecuteTaskflow.html#ModifyWorkerProperty", null ]
    ] ],
    [ "Subflow Tasking", "SubflowTasking.html", [
      [ "Create a Subflow", "SubflowTasking.html#CreateASubflow", null ],
      [ "Retain a Subflow", "SubflowTasking.html#RetainASubflow", null ],
      [ "Join a Subflow Explicitly", "SubflowTasking.html#JoinASubflow", null ],
      [ "Create a Nested Subflow", "SubflowTasking.html#CreateANestedSubflow", null ]
    ] ],
    [ "Conditional Tasking", "ConditionalTasking.html", [
      [ "Create a Condition Task", "ConditionalTasking.html#CreateAConditionTask", null ],
      [ "Understand our Task-level Scheduling", "ConditionalTasking.html#TaskSchedulingPolicy", [
        [ "Example", "ConditionalTasking.html#TaskLevelSchedulingExample", null ]
      ] ],
      [ "Avoid Common Pitfalls", "ConditionalTasking.html#AvoidCommonPitfalls", null ],
      [ "Implement Control-flow Graphs", "ConditionalTasking.html#ImplementControlFlowGraphs", [
        [ "Implement If-Else Control Flow", "ConditionalTasking.html#ImplementIfElseControlFlow", null ],
        [ "Implement Switch Control Flow", "ConditionalTasking.html#ImplementSwitchControlFlow", null ],
        [ "Implement Do-While-Loop Control Flow", "ConditionalTasking.html#ImplementDoWhileLoopControlFlow", null ],
        [ "Implement While-Loop Control Flow", "ConditionalTasking.html#ImplementWhileLoopControlFlow", null ]
      ] ],
      [ "Create a Multi-condition Task", "ConditionalTasking.html#CreateAMultiConditionTask", null ]
    ] ],
    [ "Composable Tasking", "ComposableTasking.html", [
      [ "Compose a Taskflow", "ComposableTasking.html#ComposeATaskflow", null ],
      [ "Create a Module Task from a Taskflow", "ComposableTasking.html#CreateAModuleTaskFromATaskflow", null ],
      [ "Create a Custom Composable Graph", "ComposableTasking.html#CreateACustomComposableGraph", null ],
      [ "Create an Adopted Module Task", "ComposableTasking.html#CreateAnAdoptedModuleTask", null ]
    ] ],
    [ "Asynchronous Tasking", "AsyncTasking.html", [
      [ "Launch Asynchronous Tasks from an Executor", "AsyncTasking.html#LaunchAsynchronousTasksFromAnExecutor", null ],
      [ "Launch Asynchronous Tasks from a Runtime", "AsyncTasking.html#LaunchAsynchronousTasksFromARuntime", null ],
      [ "Launch Asynchronous Tasks Recursively from a Runtime", "AsyncTasking.html#LaunchAsynchronousTasksRecursivelyFromARuntime", null ]
    ] ],
    [ "Asynchronous Tasking with Dependencies", "DependentAsyncTasking.html", [
      [ "Create a Dynamic Task Graph", "DependentAsyncTasking.html#CreateADynamicTaskGraph", null ],
      [ "Specify a Range of Dependent Async Tasks", "DependentAsyncTasking.html#SpecifyARagneOfDependentAsyncTasks", null ],
      [ "Understand the Lifetime of a Dependent-async Task", "DependentAsyncTasking.html#UnderstandTheLifeTimeOfADependentAsyncTask", null ],
      [ "Create a Dynamic Task Graph by Multiple Threads", "DependentAsyncTasking.html#CreateADynamicTaskGraphByMultipleThreads", null ],
      [ "Query the Completion Status of Dependent Async Tasks", "DependentAsyncTasking.html#QueryTheComppletionStatusOfDependentAsyncTasks", null ]
    ] ],
    [ "Runtime Tasking", "RuntimeTasking.html", [
      [ "Create a Runtime Task", "RuntimeTasking.html#CreateARuntimeTask", null ],
      [ "Create Asynchronous Tasks from a Runtime Task", "RuntimeTasking.html#CreateAsynchronousTasksFromARuntimeTask", null ],
      [ "Synchronize with Cooperative Execution", "RuntimeTasking.html#SynchronizeWithCooperativeExecution", null ],
      [ "Create Recursive Asynchronous Tasks from a Runtime Task", "RuntimeTasking.html#CreateRecursiveAsynchronousTasksFromARuntimeTask", null ]
    ] ],
    [ "Task Group", "TaskGroup.html", [
      [ "Create a Task Group", "TaskGroup.html#CreateATaskGroup", null ],
      [ "Submit Asynchronous Tasks with Cooperative Execution", "TaskGroup.html#SubmitAsynchronousTasksWithCooperativeExecution", null ],
      [ "Cancel a Task Group", "TaskGroup.html#CancelATaskGroup", null ],
      [ "Implement Recursive Task Parallelism", "TaskGroup.html#ImplementRecursiveTaskParallelismUsingTaskGroup", null ]
    ] ],
    [ "Exception Handling", "ExceptionHandling.html", [
      [ "Why Parallel Exception Handling is Hard", "ExceptionHandling.html#ExceptionHandlingWhyHard", null ],
      [ "How Taskflow Handles Exceptions", "ExceptionHandling.html#ExceptionHandlingLogic", [
        [ "Scenario 1: Synchronous Propagation", "ExceptionHandling.html#Scenario1SynchronousExceptionPropagation", null ],
        [ "Scenario 2: Asynchronous Propagation", "ExceptionHandling.html#Scenario2AsynchronousExceptionPropagation", null ],
        [ "Scenario 3: Contextual Propagation", "ExceptionHandling.html#Scenario3ContextualExceptionPropagation", null ],
        [ "Algorithm Flow", "ExceptionHandling.html#ExceptionHandlingAlgorithmFlow", null ]
      ] ],
      [ "Catch an Exception from a Running Taskflow", "ExceptionHandling.html#ExceptionHandlingRunningTaskflow", null ],
      [ "Catch an Exception from a Subflow", "ExceptionHandling.html#ExceptionHandlingSubflow", null ],
      [ "Catch an Exception from an Async Task", "ExceptionHandling.html#ExceptionHandlingAsyncTask", null ],
      [ "Catch an Exception from a Corun Loop", "ExceptionHandling.html#ExceptionHandlingCorun", null ],
      [ "Retrieve the Exception Pointer of a Task", "ExceptionHandling.html#RetrieveTheExceptionPointerOfATask", null ],
      [ "Disable Exception Handling at Compile Time", "ExceptionHandling.html#DisableExceptionHandling", null ]
    ] ],
    [ "Limit the Maximum Concurrency", "LimitTheMaximumConcurrency.html", [
      [ "Define a Semaphore", "LimitTheMaximumConcurrency.html#DefineASemaphore", null ],
      [ "Use Semaphores Across Different Tasks", "LimitTheMaximumConcurrency.html#UseSemaphoresAcrossDifferentTasks", null ],
      [ "Define a Conflict Graph", "LimitTheMaximumConcurrency.html#DefineAConflictGraph", null ],
      [ "Reset a Semaphore", "LimitTheMaximumConcurrency.html#ResetASemaphore", null ],
      [ "Understand the Limitation of Semaphores", "LimitTheMaximumConcurrency.html#UnderstandTheLimitationOfSemaphores", null ]
    ] ],
    [ "Request Cancellation", "RequestCancellation.html", [
      [ "Cancel a Running Taskflow", "RequestCancellation.html#CancelARunningTaskflow", null ],
      [ "Understand the Limitations of Cancellation", "RequestCancellation.html#UnderstandTheLimitationsOfCancellation", null ]
    ] ],
    [ "GPU Tasking", "GPUTasking.html", [
      [ "Include the Header", "GPUTasking.html#GPUTaskingIncludeTheHeader", null ],
      [ "What is a CUDA Graph?", "GPUTasking.html#WhatIsACudaGraph", null ],
      [ "Create a CUDA Graph", "GPUTasking.html#CreateACUDAGraph", null ],
      [ "Compile a CUDA Graph Program", "GPUTasking.html#CompileACUDAGraphProgram", null ],
      [ "Run a CUDA Graph on Specific GPU", "GPUTasking.html#RunACUDAGraphOnASpecificGPU", null ],
      [ "Create Memory Operation Tasks", "GPUTasking.html#GPUMemoryOperations", null ],
      [ "Run a CUDA Graph", "GPUTasking.html#RunACUDAGraph", null ],
      [ "Update an Executable CUDA Graph", "GPUTasking.html#UpdateAnExecutableCUDAGraph", null ],
      [ "Integrate a CUDA Graph into Taskflow", "GPUTasking.html#IntegrateACUDAGraphIntoTaskflow", null ]
    ] ],
    [ "Profile Taskflow Programs", "Profiler.html", [
      [ "Enable Taskflow Profiler", "Profiler.html#ProfilerEnableTFProf", null ],
      [ "Enable Taskflow Profiler on a HTTP Server", "Profiler.html#ProfilerEnableTFProfServer", null ],
      [ "Display Profile Summary", "Profiler.html#ProfilerDisplayProfileSummary", null ]
    ] ]
];